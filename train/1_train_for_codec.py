import sys
sys.path.append("../")
import os
import random
import glob
import PIL
from PIL import Image
from tqdm.auto import tqdm
import argparse

import torch.utils.checkpoint
from torch.utils.data import Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from diffusers import AutoencoderKL
import lpips

from networks.models import *
from utils import torch_to_pil, RGB_HSV
from train.noise_layers.noiser import Noiser
import torchsummary

WINDOW_SIZE = 32
KERNEL = torch.ones((1, 1, WINDOW_SIZE, WINDOW_SIZE), dtype=torch.float32) / (WINDOW_SIZE**2)

def PRVL_loss(img1, img2):
    global KERNEL
    diff = torch.abs(img1 - img2)
    diff_combined = torch.mean(diff, dim=1, keepdim=True)
    if KERNEL.device != diff_combined.device:
        KERNEL = KERNEL.to(diff_combined.device)
    diff_sum = F.conv2d(diff_combined, KERNEL, padding=WINDOW_SIZE//2).squeeze(0) # [1, 513, 513]
    max_diff = torch.max(diff_sum)
    return max_diff

def base_augment(image):
    if random.random() > 0.5:
        image = torch.flip(image, dims=[-1])
    image = torch.rot90(image, k=random.randint(0, 3), dims=[-2, -1])
    return image

def hsv_purple_suppression_loss(rgb_image):
    color_space_converter = RGB_HSV()
    hsv_image = color_space_converter.rgb_to_hsv(rgb_image)
    h_channel, s_channel, _ = hsv_image[:, 0:1], hsv_image[:, 1:2], hsv_image[:, 2:3]

    lower = 0.7
    upper = 0.85
    purple_mask = ((h_channel >= lower) & (h_channel <= upper)).float()
    purple_saturation = purple_mask * s_channel
    loss = torch.mean(purple_saturation)

    return loss

class traindataset(Dataset):
    def __init__(self, root, random_aug=True):
        self.root = root
        self.image_files = glob.glob(root + "/*.png") + glob.glob(root + "/*.jpg")
        self.random_aug = random_aug

    def __len__(self):
        return len(self.image_files)

    def process(self, image):
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = image.resize((512, 512), resample=PIL.Image.Resampling.BICUBIC)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = torch.from_numpy(image).permute(2, 0, 1)
        if self.random_aug and random.random() > 0.5:
            image = base_augment(image)
        return image

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(img_name)
        image = self.process(image)
        return image


def main(args):

    train_loader = torch.utils.data.DataLoader(
        traindataset(args.dataset, args.random_aug),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    vae = vae.cuda()
    # Freeze vae and unet
    vae.requires_grad_(False)

    def decode_latents(latents):
        # latents = 1 / vae.config.scaling_factor * latents
        image = vae.decode(latents).sample
        return image

    sec_encoder = SecretEncoder(args.bit_num).cuda()
    sec_decoder = SecretDecoder(output_size=args.bit_num).cuda()
    # get params size
    torchsummary.summary(sec_decoder, (3, 512, 512))
    
    loss_fn_alex = lpips.LPIPS(net='vgg').cuda()
    loss_fn_alex.requires_grad_(False)

    noise_config = ['Identity','Jpeg','CropandResize','GaussianBlur','GaussianNoise','ColorJitter']
    posibilities = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    noiser = Noiser(noise_config, posibilities, device='cuda')

    current_epoch = 0
    if args.resume_from_ckpt is not None:
        models = torch.load(args.resume_from_ckpt)
        sec_encoder.load_state_dict(models['sec_encoder'])
        sec_decoder.load_state_dict(models['sec_decoder'])
        current_epoch = int(args.resume_from_ckpt.split('_')[-1].split('.')[0])

    optimizer = optim.AdamW([
        {'params': sec_encoder.parameters()},
        {'params': sec_decoder.parameters()}
    ], lr=args.lr, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.8)

    writer = SummaryWriter(args.output_dir + '/logs')

    def gen_combined_latents(latents, wm_latent, scale=1.0):
        cornerfy_aug = random.choice([True, False, False, False]) # 1/4 chance to cornerfy_aug

        height, width = wm_latent.shape[2], wm_latent.shape[3]
        height_scale, width_scale = (random.uniform(1.0, 2.0), random.uniform(1.0, 2.0)) if cornerfy_aug else (1.0, 1.0)
        if cornerfy_aug:
            wm_template = F.interpolate(torch.zeros_like(latents), scale_factor=(height_scale, width_scale), mode='bilinear') if cornerfy_aug else latents
            wm_template[:,:,:height//2, :width//2] =  wm_latent[:,:,:height//2, :width//2]
            wm_template[:,:,:height//2, -width//2:] = wm_latent[:,:,:height//2, -width//2:]
            wm_template[:,:,-height//2:,:width//2] =  wm_latent[:,:,-height//2:,:width//2]
            wm_template[:,:,-height//2:,-width//2:] = wm_latent[:,:,-height//2:,-width//2:]
            wm_template = F.interpolate(wm_template, size=(height, width), mode='bilinear')
        else:
            wm_template = wm_latent

        watermarked_latents = latents + wm_template * scale
        return watermarked_latents

    all_iter = args.epochs * len(train_loader)

    pbar = tqdm(total=all_iter)
    iterations = 0
    warmup = args.warmup
    fixinit = args.fixinit

    zero_batch = torch.zeros(args.batch_size, 3, 512, 512).cuda()
    for epoch in range(current_epoch, current_epoch + args.epochs):
        sec_encoder.train()
        sec_decoder.train()

        msgloss_10buffer = []
        for batch_idx, oimage in enumerate(train_loader):
            if fixinit:
                # use zero image when start training
                oimage = zero_batch

            if epoch > 35:
                use_zero_msg = random.random() < 0.2
            else:
                use_zero_msg = False

            optimizer.zero_grad()
            oimage = oimage.cuda()
            latents = vae.encode(oimage).latent_dist.sample().detach()

            msg = torch.randint(0, 2, (args.batch_size, args.bit_num)).cuda()
            _, wm_latent = sec_encoder(latents, msg.float())

            if use_zero_msg:
                msg = torch.zeros((args.batch_size, args.bit_num), dtype=torch.long).cuda()
                with torch.no_grad():
                    watermarked_latents = latents
            else:
                msg = torch.randint(0, 2, (args.batch_size, args.bit_num)).cuda()
                _, wm_latent = sec_encoder(latents, msg.float())
                watermarked_latents = gen_combined_latents(latents, wm_latent, scale=0.03 if warmup else 1.0)

            clean_image = decode_latents(latents).detach()
            watermarked_image = decode_latents(watermarked_latents)

            if epoch > 10:
                residual_image = (watermarked_image - clean_image).clamp(-1, 1)
                residual_image = (residual_image + 1) / 2
                PRTS_loss_ = hsv_purple_suppression_loss(residual_image)

            lpips_loss = loss_fn_alex(clean_image, watermarked_image).mean()
            prvl_loss = PRVL_loss(clean_image, watermarked_image)

            if epoch > 12:
                watermarked_image = noiser([watermarked_image,None],[0.4, 0.1, 0.2, 0.05, 0.1, 0.15])[0]
            else:
                watermarked_image = noiser([watermarked_image,None],[0.6, 0., 0.4, 0., 0., 0.])[0]

            reveal_output = sec_decoder(watermarked_image)

            # Create labels tensor
            labels = F.one_hot(msg, num_classes=2).float()
            # change to BCE loss
            msgloss = F.binary_cross_entropy_with_logits(reveal_output, labels.cuda())

            if len(msgloss_10buffer) == 10:
                msgloss_10buffer.pop(0)
            msgloss_10buffer.append(msgloss.item())

            # when the message loss is less than 0.1 for 10 consecutive batches, we consider the model is warmed up
            if len(msgloss_10buffer) == 10 and sum(msgloss_10buffer) / 10 < 0.1:
                warmup = False
                fixinit = False

            if warmup:
                loss = msgloss
            else:
                if epoch > 30:
                    loss = lpips_loss * 5 + msgloss * 1.0 + prvl_loss * 1.5 + PRTS_loss_ * 50
                elif epoch > 10:
                    loss = lpips_loss * 5 + msgloss * 1.0 + prvl_loss * 1.5
                elif epoch > 6:
                    loss = lpips_loss + msgloss
                else:
                    loss = msgloss

            # with torch.autograd.detect_anomaly():
            loss.backward()
            optimizer.step()

            pbar.update(1)
            iterations += 1
            writer.add_scalar('Loss/train', loss, iterations)
            writer.add_scalar('Loss/lpips_loss', lpips_loss, iterations)
            writer.add_scalar('Loss/prvl_loss', prvl_loss, iterations)
            if epoch > 10:
                writer.add_scalar('Loss/saturation_loss', PRTS_loss_, iterations)
            writer.add_scalar('Loss/msgloss', msgloss, iterations)

        # watermarked_image_pil = torch_to_pil(watermarked_image)[0]
        # watermarked_image_pil.save(f"{args.output_dir}/log_images/watermarked_{epoch}_{batch_idx}.png")
        if epoch > 10:
            residual_image_PIL = torch_to_pil(residual_image)[0]
            residual_image_PIL.save(f"{args.output_dir}/log_images/residual_{epoch}.png")

        sec_encoder.eval()
        sec_decoder.eval()
        with torch.no_grad():
            msg_val = torch.randint(0, 2, (args.batch_size, args.bit_num)).cuda()
            watermarked_latent, _ = sec_encoder(latents, msg_val.float())
            watermarked = decode_latents(watermarked_latent)
            decoded_msg = sec_decoder(watermarked)
            decoded_msg = torch.argmax(decoded_msg, dim=2)
            acc = 1 - torch.abs(decoded_msg - msg_val).sum().float() / (args.bit_num * args.batch_size)
            print(f"Epoch {epoch}: acc {acc}")

            clean_output = sec_decoder(decode_latents(latents))
            clean_decoded = torch.argmax(clean_output, dim=2)
            clean_acc = (clean_decoded == 0).all(dim=1).float().mean()

            print(f"Step:{iterations}: Normal Acc {acc}, Clean Acc {clean_acc}")
            writer.add_scalar('Accuracy/Normal', acc, iterations)
            writer.add_scalar('Accuracy/Clean', clean_acc, iterations)

        scheduler.step()

        if epoch > 35:
            torch.save({
                'sec_decoder': sec_decoder.state_dict(),
                'sec_encoder': sec_encoder.state_dict(),
            }, f"{args.output_dir}/checkpoints/state_dict_{epoch}.pth")

    writer.close()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--pretrained_model_name_or_path', type=str, default='runwayml/stable-diffusion-v1-5')
    argparser.add_argument('--epochs', type=int, default=50)
    argparser.add_argument('--batch_size', type=int, default=1)
    argparser.add_argument('--bit_num', type=int, default=48)
    argparser.add_argument('--resume_from_ckpt', type=str, default=None)
    argparser.add_argument('--dataset', type=str, default='../dataset/COCO2017test')
    argparser.add_argument('--output_dir', default='../output/codec_test')
    argparser.add_argument('--warmup', default=True)
    argparser.add_argument('--fixinit', default=True)
    argparser.add_argument('--random_aug', default=True)
    argparser.add_argument('--lr', type=float, default=0.001)
    args = argparser.parse_args()

    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
        os.makedirs(args.output_dir + '/logs')
        os.makedirs(f'{args.output_dir}/checkpoints')
        os.makedirs(f'{args.output_dir}/log_images')

    main(args)
