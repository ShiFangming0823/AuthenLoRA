import argparse
import importlib
import io
import os
import random
import sys

from pytorch_fid import fid_score
from safetensors import safe_open

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from math import comb

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from torchvision import transforms
from tqdm import tqdm

from library import train_util
from library.lpw_stable_diffusion import StableDiffusionLongPromptWeightingPipeline

from library.train_util import get_my_scheduler

from networks.models import SecretDecoder, MapperNet
from train.utils import load_target_model, log_print

import kornia as K
import torchvision.transforms as T

weight_dtype = torch.float16
device = torch.device("cuda:0")
pipe = None

def load_quantized_model(model_path):
    tensors = {}
    metadata = {}

    with safe_open(model_path, framework="pt", device="cpu") as f:
        # Load the metadata
        if hasattr(f, "metadata"):
            metadata = f.metadata()

        for key in f.keys():
            tensor = f.get_tensor(key)

            # Check scale_key existed (whether scaled)
            scale_key = f"{key}.quant_scale"
            if scale_key in metadata:
                # This is a scaled int tensor
                scale = float(metadata[scale_key])
                # Unscale to fp16
                tensors[key] = tensor.to(torch.float16) / scale
                print(f"De-quantized '{key}' using scale {scale}")
            else:
                tensors[key] = tensor

    return tensors, metadata

def get_threshold(k, fpr):
    tau = 0
    while calculate_fpr(tau, k) > fpr:
        tau += 1
    return tau

def calculate_fpr(tau, k):
    sum_combinations = sum(comb(k, i) for i in range(tau + 1, k + 1))
    fpr = 1 / (2 ** k) * sum_combinations
    return fpr

def PIL2tensor(image):
    if not image.mode == "RGB":
        image = image.convert("RGB")
    image = image.resize((512, 512), resample=Image.Resampling.BICUBIC)
    image = np.array(image).astype(np.uint8)
    image = (image / 127.5 - 1.0).astype(np.float32)
    image = torch.from_numpy(image).permute(2, 0, 1)
    return image

def tensor2PIL(images):
    images = images.detach().cpu().float()
    if images.ndim == 3:
        images = images[None, ...]
    images = images.permute(0, 2, 3, 1)
    images = (images + 1.0) * 127.5  # [-1,1] -> [0, 255]
    images = images.clamp(0, 255).to(torch.uint8)
    images = images.numpy()
    if images.shape[-1] == 1:
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]
    return pil_images

def SDEdit(image, version):
    global pipe
    if pipe is None:
        if version == 1:
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', safety_checker=None).to('cuda')
        elif version == 2:
            pipe= StableDiffusionImg2ImgPipeline.from_pretrained('stabilityai/stable-diffusion-2-1', safety_checker=None).to('cuda')
    images = pipe(prompt="masterpiece", image=image, strength=0.1 if version == 1 else 0.2, num_inference_steps=10, guidance_scale=7.5, output_type='pt').images
    return images.squeeze(0)

def distortion_unit(encoded_image, type):
    if type == 'color_jitter':
        encoded_image_01 = (encoded_image + 1.0) / 2.0
        distorted_image = K.augmentation.ColorJiggle(brightness=(0.9, 1.1), contrast=(0.9, 1.1), saturation=(0.9, 1.1), hue=(-0.1, 0.1), p=1)(encoded_image_01)
        distorted_image = distorted_image * 2.0 - 1.0
    elif type == 'crop':
        distorted_image = T.RandomCrop(size=(args.crop_resolution, args.crop_resolution))(encoded_image).unsqueeze(0)
    elif type == 'blur':
        distorted_image = K.augmentation.RandomGaussianBlur((3, 3), (4.0, 4.0), p=1.)(encoded_image)
    elif type == 'noise':
        encoded_image_01 = (encoded_image + 1.0) / 2.0
        distorted_image = K.augmentation.RandomGaussianNoise(mean=0.0, std=0.1, p=1)(encoded_image_01)
        distorted_image = distorted_image * 2.0 - 1.0
    elif type == 'jpeg_compress':
        encoded_image_uint8 = ((encoded_image + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
        out = T.ToPILImage()(encoded_image_uint8.squeeze(0))
        buffer = io.BytesIO()
        out.save(buffer, format='JPEG', quality=50)
        buffer.seek(0)
        pil_image = Image.open(buffer)
        distorted_image = T.ToTensor()(pil_image)
        distorted_image = distorted_image.unsqueeze(0)
        distorted_image = distorted_image * 2.0 - 1.0
    elif type == 'rotation':
        distorted_image = K.augmentation.RandomRotation(degrees=(15, 15), p=1)(encoded_image)
    elif type == 'sharpness':
        encoded_image_01 = (encoded_image + 1.0) / 2.0
        distorted_image = K.augmentation.RandomSharpness(sharpness=10., p=1)(encoded_image_01)
        distorted_image = distorted_image * 2.0 - 1.0
    elif type == 'SDEdit':
        encoded_image_01 = (encoded_image + 1.0) / 2.0
        distorted_image = SDEdit(encoded_image_01, version=1).unsqueeze(0)
        distorted_image = distorted_image * 2.0 - 1.0
    elif type == 'SDEdit2':
        encoded_image_01 = (encoded_image + 1.0) / 2.0
        distorted_image = SDEdit(encoded_image_01, version=2).unsqueeze(0)
        distorted_image = distorted_image * 2.0 - 1.0
    else:
        raise ValueError(f'Wrong distortion type in add_distortion().')
    return distorted_image


def eval(args):
    ###############################################################
    #                           Prepare                           #
    ###############################################################
    results = {
        'clean': {},
        'distortions': {}
    }

    if args.generate_image:
        tokenizer = train_util.load_tokenizer(args)
        text_encoder, vae, unet, _ = load_target_model(args.pretrained_model_name_or_path, weight_dtype, device)

        mapper = MapperNet(input_size=args.msg_bits, output_size=args.lora_rank)

        # Disable grads
        unet.requires_grad_(False)
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        mapper.requires_grad_(False)

        unet.to(device, dtype=weight_dtype)
        vae.to(device, dtype=weight_dtype)
        text_encoder.to(device, dtype=weight_dtype)
        mapper.to(device)

        unet.eval()
        vae.eval()
        text_encoder.eval()
        mapper.eval()

        # Load pretrained weights
        lora_module = "networks.lycoris.kohya"
        net_kwargs = {}
        net_kwargs['preset'] = args.finetune_scope
        if args.lora_type == 'lora':
            net_kwargs['algo'] = 'lora'
        elif args.lora_type == 'locon':
            net_kwargs['algo'] = 'locon'
        elif args.lora_type == 'loha':
            net_kwargs['algo'] = 'loha'
        elif args.lora_type == 'lokr':
            net_kwargs['bypass_mode'] = False
            net_kwargs['algo'] = 'lokr'
            net_kwargs['factor'] = 2
        elif args.lora_type == 'dylora':
            net_kwargs['algo'] = 'dylora'

        # Randomly generate secret info
        secret_info_tensor = torch.randint(0, 2, (1, args.msg_bits)).to(device)
        secret_info_flatten = secret_info_tensor.flatten()
        secret_info_str = ''.join(str(bit.item()) for bit in secret_info_flatten)
        log_print(f"Secret Info: {secret_info_str}")

        # Initialize LoRA module
        network_module = importlib.import_module(lora_module)
        network = network_module.create_network(multiplier=1.0, network_dim=args.lora_rank, network_alpha=args.lora_alpha, vae=vae, text_encoder=text_encoder, unet=unet, neuron_dropout=None, **net_kwargs)
        network.apply_to(text_encoder, unet, apply_text_encoder=True, apply_unet=True)
        network.to(device, dtype=weight_dtype)
        network.eval()
        log_print(network.load_weights(args.LoRA_weight_path))
        log_print(f"The LoRA weight file has been loaded: {args.LoRA_weight_path}")

        # Load pretrained mapper weight
        parent_dir = os.path.dirname(args.LoRA_weight_path)
        if args.mapper_weight_path is None:
            lora_weight_filename = os.path.basename(args.LoRA_weight_path)
            lora_weight_basename = os.path.splitext(lora_weight_filename)[0]
            mapper_weight_filename = f"{lora_weight_basename}.pth"
            mapper_weight_path = os.path.join(parent_dir, mapper_weight_filename)
        else:
            mapper_weight_path = args.mapper_weight_path

        if not os.path.exists(mapper_weight_path):
            raise ValueError(f"The mapper weight file does not exist: {mapper_weight_path}")
        else:
            mapper_weight = torch.load(mapper_weight_path)
            mapper.load_state_dict(mapper_weight)
            log_print(f"The mapper weight file has been loaded: {mapper_weight_path}")

        # Merge secret info with LoRA
        secret_emb = mapper(secret_info_tensor)
        diag_secret = torch.diag_embed(secret_emb)[0].to(weight_dtype)
        network.apply_mapper(diag_secret)
        network.set_te_output()
        network.set_lora_output()
        log_print("The secret_info has been merged with LoRA. In this script, the secret LoRA will not be saved.")

        # Load the prompt file for validate
        if args.validate_prompt_file is not None:
            try:
                with open(args.validate_prompt_file, "r", encoding='utf-8') as file:
                    prompt_lines = file.read().splitlines()
                if not prompt_lines:
                    raise ValueError(f"The prompt file {args.validate_prompt_file} is empty.")
            except:
                raise ValueError(f"The prompt file {args.validate_prompt_file} is invalid.")
        elif args.validate_default_prompt is not None:
            pass
        else:
            raise ValueError(f"The prompt for validation must be provided.")

    ###############################################################
    #                            Eval                             #
    ###############################################################
    # Set seed
    if args.seed is None:
        args.seed = random.randint(0, 2 ** 32)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    if args.generate_image:
        # Generate Images
        log_print("********** Generate Images **********")
        log_print(f"*** Total Image Num : {len(prompt_lines) * args.validate_image_num}")
        log_print(f"*** Secret Info : {secret_info_str}")

        save_path = f"{parent_dir}/sample_{secret_info_str}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        with torch.no_grad():
            scheduler = get_my_scheduler(sample_sampler=args.sample_sampler, v_parameterization=False, )
            pipeline = StableDiffusionLongPromptWeightingPipeline(text_encoder=text_encoder, vae=vae, unet=unet, tokenizer=tokenizer, scheduler=scheduler, safety_checker=None, feature_extractor=None, requires_safety_checker=False, clip_skip=args.clip_skip)
            pipeline.to(device)
            for prompt_index in range(len(prompt_lines)):
                print(f"Current Prompt Line: {prompt_index + 1}. Total Image Index: {len(prompt_lines)}")
                latents = pipeline(prompt=f"{args.validate_prompt_prefix}, {prompt_lines[prompt_index]}", height=512, width=512, num_inference_steps=args.steps, guidance_scale=args.CFG, negative_prompt=args.validate_negative_prompt, num_images_per_prompt=args.validate_image_num)
                images = pipeline.latents_to_image(latents)
                for image_num in range(args.validate_image_num):
                    image = images[image_num]
                    image.save(f"{save_path}/{prompt_index:03d}_{image_num + 1:02d}.png")

    # Eval Bit Accuracy & TPR
    if args.eval_clean_bitacc_tpr:
        log_print("********** Evaluate Clean Bit Accuracy & TPR **********")

        if args.generate_image:
            if save_path is not None:
                img_path = save_path
                img_list = os.listdir(save_path)
        elif args.validate_image_path is not None:
            img_path = args.validate_image_path
            img_list = os.listdir(args.validate_image_path)
            secret_info_str = os.path.basename(args.validate_image_path)[-int(args.msg_bits):]

        else:
            raise ValueError(f"The validation image path must be provided.")

        sec_decoder = SecretDecoder(output_size=args.msg_bits)
        codec_weight = torch.load(args.codec_weight_path)
        sec_decoder.load_state_dict(codec_weight['sec_decoder'])

        sec_decoder.requires_grad_(False)
        sec_decoder.to(device)
        sec_decoder.eval()

        log_print(f"Secret Info: {secret_info_str}")

        tau = get_threshold(args.msg_bits, args.tpr_threshold) / args.msg_bits
        accuracy = []
        TP = 0
        FN = 0

        with torch.no_grad():
            for img in tqdm(img_list):
                image = Image.open(os.path.join(img_path, img))
                image = PIL2tensor(image)
                image = image.to(device)
                image = image.unsqueeze(0)

                msg = sec_decoder(image)
                msg = torch.argmax(msg, dim=-1)
                msg = ''.join(map(str, msg.tolist()[0]))

                assert secret_info_str is not None, "The key must be specified"
                assert len(secret_info_str) == len(msg), "The key and msg must have the same length"
                acc = sum([1 for i in range(len(msg)) if msg[i] == secret_info_str[i]]) / len(msg)

                accuracy.append(acc)

                if acc >= tau:
                    TP += 1
                else:
                    FN += 1

            mean_bitacc = np.mean(accuracy)
            print("Mean bitacc:", mean_bitacc)
            results['clean']['mean_bitacc'] = mean_bitacc

            print(TP)
            print(FN)
            TPR = TP / (TP + FN)
            print("TPR:", TPR)
            results['clean']['TPR'] = TPR

    if args.eval_distortion_bitacc_tpr:
        log_print("********** Evaluate Distortion Bit Accuracy & TPR **********")

        if args.generate_image:
            if save_path is not None:
                img_path = save_path
                img_list = os.listdir(save_path)
        elif args.validate_image_path is not None:
            img_path = args.validate_image_path
            img_list = os.listdir(args.validate_image_path)
            secret_info_str = os.path.basename(args.validate_image_path)[-int(args.msg_bits):]
        else:
            raise ValueError(f"The validation image path must be provided.")

        log_print(f"Secret Info: {secret_info_str}")

        tau = get_threshold(args.msg_bits, args.tpr_threshold) / args.msg_bits

        with torch.no_grad():
            for distortion_type in args.distortion_types:
                accuracy = []
                TP = 0
                FN = 0

                log_print(f"Distortion Type: {distortion_type}")
                for img in tqdm(img_list):
                    clean_image = Image.open(os.path.join(img_path, img))
                    clean_image_tensor = PIL2tensor(clean_image)
                    distortion_image_tensor = distortion_unit(clean_image_tensor, distortion_type).to(device)
                    distortion_image_PIL = tensor2PIL(distortion_image_tensor)[0]
                    if args.save_result:
                        if not os.path.exists(os.path.join(f"{img_path}", distortion_type)):
                            os.makedirs(os.path.join(f"{img_path}", distortion_type))
                        distortion_image_PIL.save(f"{img_path}/{distortion_type}/{img}")

                    msg = sec_decoder(distortion_image_tensor)
                    msg = torch.argmax(msg, dim=-1)
                    msg = ''.join(map(str, msg.tolist()[0]))

                    assert secret_info_str is not None, "The key must be specified"
                    assert len(secret_info_str) == len(msg), "The key and msg must have the same length"
                    acc = sum([1 for i in range(len(msg)) if msg[i] == secret_info_str[i]]) / len(msg)

                    accuracy.append(acc)
                    if acc >= tau:
                        TP += 1
                    else:
                        FN += 1

                mean_bitacc = np.mean(accuracy)
                print("Mean bitacc:", mean_bitacc)

                print(TP)
                print(FN)
                TPR = TP / (TP + FN)
                print("TPR:", TPR)

                results['distortions'][distortion_type] = {
                    'mean_bitacc': mean_bitacc,
                    'TPR': TPR
                }

                del accuracy, mean_bitacc, TP, FN, TPR

    if args.eval_composite_distortion_bitacc_tpr:
        log_print("\n=== Evaluate Composite Distortion Bit Accuracy & TPR ===")
        if args.generate_image:
            if save_path is not None:
                img_path = save_path
                img_list = os.listdir(save_path)
        elif args.validate_image_path is not None:
            img_path = args.validate_image_path
            img_list = os.listdir(args.validate_image_path)
            secret_info_str = os.path.basename(args.validate_image_path)[-int(args.msg_bits):]
        else:
            raise ValueError(f"The validation image path must be provided.")

        log_print(f"Secret Info: {secret_info_str}")

        tau = get_threshold(args.msg_bits, args.tpr_threshold) / args.msg_bits

        # Parse the comma-separated string of distortions into a list
        composite_distortions = [d.strip() for d in args.composite_distortion_types.split(',')]
        log_print(f"Composite Distortion Sequence: {' -> '.join(composite_distortions)}")

        accuracy = []
        TP = 0
        FN = 0

        with torch.no_grad():
            for img in tqdm(img_list):
                clean_image = Image.open(os.path.join(img_path, img))
                # Start with the clean image tensor
                distorted_image_tensor = PIL2tensor(clean_image)

                # Sequentially apply each distortion in the list
                for distortion_type in composite_distortions:
                    # The output of one distortion becomes the input for the next
                    distorted_image_tensor = distortion_unit(distorted_image_tensor, distortion_type)

                # Move the final distorted tensor to the correct device
                distorted_image_tensor = distorted_image_tensor.to(device)

                if args.save_result:
                    composite_name = "_".join(composite_distortions)
                    save_dir = os.path.join(img_path, composite_name)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    distortion_image_PIL = tensor2PIL(distorted_image_tensor)[0]
                    distortion_image_PIL.save(os.path.join(save_dir, img))

                # The following decoding logic is identical to the single distortion block
                msg = sec_decoder(distorted_image_tensor)
                msg = torch.argmax(msg, dim=-1)
                msg = ''.join(map(str, msg.tolist()[0]))

                assert secret_info_str is not None, "The key must be specified"
                assert len(secret_info_str) == len(msg), "The key and msg must have the same length"
                acc = sum([1 for i in range(len(msg)) if msg[i] == secret_info_str[i]]) / len(msg)

                accuracy.append(acc)
                if acc >= tau:
                    TP += 1
                else:
                    FN += 1

        mean_bitacc = np.mean(accuracy)
        print("Mean bitacc:", mean_bitacc)
        TPR = TP / (TP + FN)
        print("TPR:", TPR)

        # Store the results for final printing
        composite_key = ' -> '.join(composite_distortions)
        results['composite_distortions'][composite_key] = {'mean_bitacc': mean_bitacc, 'TPR': TPR}

    if args.eval_clean_bitacc_tpr or args.eval_distortion_bitacc_tpr:
        print("\n=== Final Evaluation Results ===")

        # Clean Results
        if args.eval_clean_bitacc_tpr:
            print("\n[Clean Images]")
            print(f"Mean Bit Accuracy: {results['clean'].get('mean_bitacc', 'N/A'):.4f}")
            print(f"TPR: {results['clean'].get('TPR', 'N/A'):.4f}")

        # Distortion Results
        if args.eval_distortion_bitacc_tpr:
            print("\n[Distortion Types]")
            for dist_type, metrics in results['distortions'].items():
                print(f"\n{dist_type.upper():<15}")
                print(f"  Mean Bit Accuracy: {metrics['mean_bitacc']:.4f}")
                if metrics['TPR'] is not None:
                    print(f"  TPR: {metrics['TPR']:.4f}")

        # Composite Distortion Results
        if args.eval_composite_distortion_bitacc_tpr:
            print("\n[Composite Distortion Types]")
            for composite_key, metrics in results['composite_distortions'].items():
                print(f"\n{composite_key.upper()}")
                print(f"  Mean Bit Accuracy: {metrics['mean_bitacc']:.4f}")
                if metrics['TPR'] is not None:
                    print(f"  TPR: {metrics['TPR']:.4f}")

    if args.eval_FID:
        log_print("********** Evaluate FID **********")

        if args.generate_image:
            if save_path is not None:
                img_path = save_path
        elif args.validate_image_path is not None:
            img_path = args.validate_image_path
        else:
            raise ValueError(f"The validation image path must be provided.")

        fid = fid_score.calculate_fid_given_paths([args.clean_image_folder, img_path], batch_size=50, device=device, dims=2048)

        print(f"FID: {fid}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Eval the performance of LoRA")

    # Models
    parser.add_argument("--msg_bits", type=int, default=48)
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="../model/base/anyloraCheckpoint_bakedvaeBlessedFp16.safetensors")
    parser.add_argument("--codec_weight_path", type=str, default="../model/codec_48bits.pth")
    parser.add_argument("--LoRA_weight_path", type=str, default=r"../../models/Extra/Anime_Character_rank160_fp16_48bits.safetensors")
    parser.add_argument("--mapper_weight_path", type=str, default=None)
    parser.add_argument("--lora_rank", type=int, default=160)
    parser.add_argument("--lora_alpha", type=int, default=80)
    parser.add_argument("--finetune_scope", type=str, default="full-lin", choices=["full", "full-lin", "attn-mlp", "attn-only"])
    parser.add_argument("--lora_type", type=str, default="locon", choices=["standard", "locon", "loha", "lokr"])
    parser.add_argument("--seed", type=int, default=42)

    # Generation Images
    parser.add_argument("--generate_image", type=bool, default=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--validate_prompt_file", type=str, default='../validate_prompt/character_prompt.txt')
    parser.add_argument("--validate_image_num", type=int, default=5)
    parser.add_argument("--validate_prompt_prefix", type=str, default='')
    parser.add_argument("--validate_negative_prompt", type=str, default='lowres, bad anatomy, bad hands, text, error, missing fingers, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, cropped, blurry, artist name, extra hands, extra legs')

    parser.add_argument("--sample_sampler", type=str, default='ddim')
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--CFG", type=float, default=7)
    parser.add_argument("--tokenizer_cache_dir", type=str, default=None)
    parser.add_argument("--v2", action="store_true")
    parser.add_argument("--clip_skip", type=int, default=None)

    # Eval Clean BitAcc & TPR
    parser.add_argument("--eval_clean_bitacc_tpr", type=bool, default=True)
    parser.add_argument("--validate_image_path", type=str, default=r"")
    parser.add_argument("--tpr_threshold", type=float, default=1e-6)

    # Eval Distortion BitAcc & TPR
    parser.add_argument("--eval_distortion_bitacc_tpr", type=bool, default=True)
    parser.add_argument("--distortion_types", type=str, default=['blur', 'color_jitter', 'crop', 'jpeg_compress', 'noise', 'sharpness', 'SDEdit', 'SDEdit2'])
    parser.add_argument("--crop_resolution", type=int, default=460)
    parser.add_argument("--save_result", type=bool, default=True)

    # Eval Composite Distortion BitAcc & TPR
    parser.add_argument("--eval_composite_distortion_bitacc_tpr", type=bool, default=True)
    parser.add_argument("--composite_distortion_types", type=str, default='blur, jpeg_compress, noise')

    # Eval FID
    parser.add_argument("--eval_FID", type=bool, default=False)
    parser.add_argument("--clean_image_folder", type=str, default=r"")

    args = parser.parse_args()

    eval(args)
