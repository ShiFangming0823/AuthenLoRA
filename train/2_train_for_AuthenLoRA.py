import argparse
import importlib
import math
import os
import sys

import lpips

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from datetime import datetime
import random
from multiprocessing import Value

import torch
from diffusers import DDPMScheduler
from torchvision import transforms

from library.lpw_stable_diffusion import StableDiffusionLongPromptWeightingPipeline

from library.train_util import get_my_scheduler
from torch.utils.tensorboard import SummaryWriter

from library.custom_train_functions import prepare_scheduler_for_custom_training
from library.config_util import BlueprintGenerator, ConfigSanitizer
from torch.cuda.amp import autocast, GradScaler

from library import train_util, config_util
from tqdm import tqdm

from networks.models import SecretEncoder, MapperNet, SecretDecoder
from train import utils
from train.utils import log_print, TeeOutput, pyramid_noise_like, load_target_model, get_cosine_schedule_with_warmup_lr_end

weight_dtype = torch.float16
device = torch.device("cuda:0")
start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
output_name = ''
log_path = ''


def train(args):
    ###############################################################
    #                           Prepare                           #
    ###############################################################
    ### Check the path and load LoRA module ###
    global output_name, log_path
    tb_name = f"{args.output_name}_{start_time}"
    tb_path = os.path.join(os.path.join(args.output_dir, "logs"), tb_name)
    writer = SummaryWriter(log_dir=tb_path, comment=tb_name)

    lora_rank = args.mapper_rank
    lora_alpha = args.lora_alpha
    log_print(f"The rank of new LoRA is {lora_rank}, the alpha is {lora_alpha}")

    ### Prepare the training pipeline ###
    log_print("Prepare the training pipeline")
    tokenizer = train_util.load_tokenizer(args)

    text_encoder, vae, unet, _ = load_target_model(args.pretrained_model_name_or_path, weight_dtype, device)

    # Prepare the Encoder, Decoder and Mapper network
    sec_encoder = SecretEncoder(secret_len=args.msg_bits, resolution=64)
    sec_decoder = SecretDecoder(output_size=args.msg_bits)
    mapper = MapperNet(input_size=args.msg_bits, output_size=args.mapper_rank)

    pretrain_dict = torch.load(args.codec_weight_path)
    sec_encoder.load_state_dict(pretrain_dict['sec_encoder'])
    sec_decoder.load_state_dict(pretrain_dict['sec_decoder'])

    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    sec_encoder.requires_grad_(False)
    sec_decoder.requires_grad_(False)
    mapper.requires_grad_(True)

    unet.to(device, dtype=weight_dtype)
    vae.to(device, dtype=torch.float16)
    text_encoder.to(device, dtype=weight_dtype)
    sec_encoder.to(device)
    sec_decoder.to(device)
    mapper.to(device)

    # Create LoRA Network and load pretrained weights
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
        net_kwargs['algo'] = 'lokr'
        net_kwargs['factor'] = 6
    elif args.lora_type == 'dylora':
        net_kwargs['algo'] = 'dylora'
    else:
        lora_module = "networks.lora_standard"
        net_kwargs = {}

    network_module = importlib.import_module(lora_module)

    network = network_module.create_network(multiplier=1.0, network_dim=lora_rank, network_alpha=lora_alpha, vae=vae, text_encoder=text_encoder, unet=unet, neuron_dropout=None, **net_kwargs)
    network.apply_to(text_encoder, unet, apply_text_encoder=True, apply_unet=True)
    network.to(device)
    network.train()

    # Determine the train strategy
    if args.is_resume_from_checkpoint:
        mapper.load_state_dict(torch.load(args.resume_from_mapper))

        if network.load_weights(args.resume_from_LoRA):
            log_print(f"The saved LoRA weights {args.resume_from_LoRA} is loaded.")

    # Prepare the optimizer
    # Train text_encoder + unet
    lora_te_params = network.prepare_optimizer_params(args.LoRA_learning_rate, args.LoRA_learning_rate, args.LoRA_learning_rate)[0]['params']
    lora_unet_params = network.prepare_optimizer_params(args.LoRA_learning_rate, args.LoRA_learning_rate, args.LoRA_learning_rate)[1]['params']

    optimizer = torch.optim.AdamW([{'params': lora_te_params, 'lr': args.LoRA_learning_rate}, {'params': lora_unet_params, 'lr': args.LoRA_learning_rate}, {'params': mapper.parameters(), 'lr': args.mapper_learning_rate}, ], betas=(
        args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay, eps=args.adam_epsilon, )
    log_print("Optimizer is created")

    # Prepare the dataset
    blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, False, True))

    user_config = {
        "general":{
            "enable_bucket": True
        },
        "datasets":[
            {
                "resolution": int(args.resolution),
                "batch_size": int(args.train_batch_size),
                "subsets":[
                    {
                        "image_dir": str(args.dataset_dir),
                        "caption_extension": '.txt',
                        "num_repeats": 5
                    }
                ]
            }
        ]
    }
    blueprint = blueprint_generator.generate(user_config, args, tokenizer=tokenizer)
    train_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    current_epoch = Value("i", 0)
    current_step = Value("i", 0)
    ds_for_collator = train_dataset_group if args.dataloader_num_workers == 0 else None
    collator = train_util.collator_class(current_epoch, current_step, ds_for_collator)
    n_workers = min(args.dataloader_num_workers, os.cpu_count())
    train_dataloader = torch.utils.data.DataLoader(train_dataset_group, batch_size=args.train_batch_size, shuffle=True, collate_fn=collator, num_workers=n_workers)

    noise_scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False)
    prepare_scheduler_for_custom_training(noise_scheduler, device)

    unet.eval()
    vae.eval()
    text_encoder.eval()
    sec_encoder.eval()
    sec_decoder.eval()


    if len(train_dataset_group) == 0:
        log_print("No data found in the dataset. Please check your dataset config.")
        return

    # Training step and epoch calculation
    if args.max_train_steps is not None and args.max_train_epochs is not None:
        estimated_epochs = math.ceil(args.max_train_steps / len(train_dataloader))
        total_train_epochs = min(estimated_epochs, args.max_train_epochs)
        total_train_steps = total_train_epochs * len(train_dataloader)

    elif args.max_train_epochs is not None:
        total_train_epochs = args.max_train_epochs
        total_train_steps = total_train_epochs * len(train_dataloader)

    elif args.max_train_steps is not None:
        total_train_steps = args.max_train_steps
        total_train_epochs = math.ceil(total_train_steps / len(train_dataloader))

    else:
        raise ValueError("Either `max_train_steps` or `train_epochs` must be provided.")

    lr_scheduler = get_cosine_schedule_with_warmup_lr_end(optimizer=optimizer, num_warmup_steps=args.lr_warmup_steps, num_training_steps=total_train_steps, lr_end=args.lr_end, )

    # Cache the train dataset latent to accelerate the training process
    if args.cache_latent:
        with torch.no_grad():
            train_dataset_group.cache_latents(vae, args.train_batch_size, args.cache_latents_to_disk)
    del train_dataset_group
    torch.cuda.empty_cache()

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
    #                            Train                            #
    ###############################################################
    scaler = GradScaler()
    train_stage = False
    # Set seed
    if args.seed is None:
        args.seed = random.randint(0, 2 ** 32)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    # Save the arguments and configs
    config_file_name = f"config_{args.output_name}_{start_time}.log"
    config_file_path = os.path.join(log_path, config_file_name)
    with open(config_file_path, 'w') as f:
        for key, value in vars(args).items():
            f.write(f"{key}:{value}\n")

    log_print("********** Running Training **********")
    log_print(f"*** Image Num / batch_size : {len(train_dataloader)}")
    log_print(f"*** Total Train Steps : {total_train_steps}")
    log_print(f"*** Total Train Epochs : {total_train_epochs}")

    global_step = args.start_step
    progress_bar = tqdm(range(total_train_steps), initial=global_step, smoothing=0, desc="Steps")
    style_save_delay_counter = 3

    for epoch in range(total_train_epochs):
        current_epoch.value = epoch + 1
        for step, batch in enumerate(train_dataloader):
            mapper.train()
            network.train()
            current_step.value = global_step
            # Check the latent cache.
            if "latents" in batch and batch["latents"] is not None:
                latents = batch["latents"].to(device)
            else:
                with torch.no_grad():
                    latents = vae.encode(batch["images"].to(device, dtype=weight_dtype)).latent_dist.sample()
                    if torch.any(torch.isnan(latents)):
                        log_print("NaN found in latents, replacing with zeros")
                        latents = torch.nan_to_num(latents, 0, out=latents)
            latents = latents * 0.18215

            ################## forward ####################
            # Randomly generate secret bits and map it to diag_matrix
            with autocast():
                secret_info = torch.randint(0, 2, (args.train_batch_size, args.msg_bits)).to(device)

                secret_emb = mapper(secret_info)
                diag_secret = torch.diag_embed(secret_emb)[0]

                network.apply_mapper(diag_secret)
                network.set_te_output()

                # Get the text embedding for conditioning
                with torch.no_grad():
                    input_ids = batch["input_ids"].to(device)
                    text_encoder_conds = train_util.get_hidden_states(args, input_ids, tokenizer, text_encoder, weight_dtype)

                    _, wm_latent = sec_encoder(latents, secret_info.float())
                    wm_latent = wm_latent * 0.18215

                # Generate initial noise
                noise = pyramid_noise_like(torch.randn_like(latents, device=device), latents.device, args.multires_noise_iterations, args.multires_noise_discount)
                bsz, _, _, _ = latents.shape

                # Sample a random timestep
                timesteps = torch.randint(0, 1000, (bsz,), device=device).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps).to(device=device)
                noisy_wm_latents = noise_scheduler.add_noise(latents + wm_latent, noise, timesteps).to(device=device)

                network.set_original_output()
                clean_pred = unet(noisy_latents, timesteps, text_encoder_conds).sample.detach()

                # Predict the noise residual
                network.set_lora_output()
                finale_pred = unet(noisy_wm_latents, timesteps, text_encoder_conds).sample

                style_loss = torch.nn.functional.mse_loss(finale_pred.float(), noise.float(), reduction="mean")
                # real_style_loss = style_ratio_scheduler(global_step, global_step,0,300000,1) * style_loss
                real_style_loss = style_loss

                watermark_loss = torch.nn.functional.mse_loss(finale_pred.float(), clean_pred.float(), reduction="mean")

                if args.train_mode == 'switch':
                    if global_step > 99999:
                        if train_stage == False:
                            loss = watermark_loss
                        elif train_stage == True:
                            loss = style_loss
                    else:
                        loss = watermark_loss

                    if global_step % 5 == 0:
                        train_stage = not train_stage

                elif args.train_mode == 'both':
                    if style_save_delay_counter <= 0:
                        loss = watermark_loss + real_style_loss
                    else:
                        loss = watermark_loss

            # with torch.autograd.detect_anomaly():
            scaler.scale(loss).backward()

            writer.add_scalar("lr", optimizer.param_groups[0]['lr'], global_step)
            writer.add_scalar("Calculated Style Loss", style_loss, global_step)
            writer.add_scalar("Real Style Loss", real_style_loss, global_step)
            writer.add_scalar("Watermark Loss", watermark_loss, global_step)

            # Backpropagate, update weights
            scaler.step(optimizer)

            lr_scheduler.step()

            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            progress_bar.update(1)
            global_step += 1

            # Validate the Accuracy
            if global_step % args.validate_every_n_steps == 0:
                mapper.eval()
                network.eval()

                with torch.no_grad():
                    scheduler = get_my_scheduler(sample_sampler=args.sample_sampler, v_parameterization=False, )
                    pipeline = StableDiffusionLongPromptWeightingPipeline(text_encoder=text_encoder, vae=vae, unet=unet, tokenizer=tokenizer, scheduler=scheduler, safety_checker=None, feature_extractor=None, requires_safety_checker=False, clip_skip=args.clip_skip, )
                    pipeline.to(device)
                    valid_accuracy_total = []
                    for _ in range(args.validate_image_num):
                        with autocast():
                            eval_secret_info = torch.randint(0, 2, (1, args.msg_bits)).to(device)
                            secret_emb = mapper(eval_secret_info)
                            diag_secret = torch.diag_embed(secret_emb)[0]
                            network.apply_mapper(diag_secret)

                            latents = pipeline(prompt=f"{args.validate_prompt_prefix}, {random.choice(prompt_lines)}", height=512, width=512, num_inference_steps=20, guidance_scale=7, negative_prompt=args.validate_negative_prompt, )

                            latents = latents.to(torch.float16)

                        image = pipeline.latents_to_image(latents)[0]

                        tensor_image = transforms.ToTensor()(image).to(device, dtype=torch.float16).unsqueeze(0)

                        decoded_info = sec_decoder(tensor_image.float())
                        decoded_msg = torch.argmax(decoded_info, dim=-1)
                        res = eval_secret_info.repeat(decoded_msg.shape[0], 1) - decoded_msg
                        valid_accuracy_ = (res == 0).float().mean()
                        valid_accuracy_total.append(valid_accuracy_)

                    valid_accuracy = sum(valid_accuracy_total) / len(valid_accuracy_total)
                    writer.add_scalar("Validation Accuracy", valid_accuracy, global_step)
                    writer.add_image("Validation Image", torch.squeeze(tensor_image, dim=0), global_step)
                    print(f"Valid Accuracy: {valid_accuracy}, Steps: {global_step}, Timesteps: {timesteps}")

                    if valid_accuracy > 0.75:
                        style_save_delay_counter -= 1

            if (style_save_delay_counter <= 0 and global_step % args.save_every_n_steps == 0 and global_step > 50000) or args.save_now == True:
                model_save_dir = os.path.join(args.output_dir, f"checkpoint_{start_time}")
                if os.path.exists(model_save_dir) and os.path.isdir(model_save_dir):
                    pass
                else:
                    os.mkdir(model_save_dir)
                step_str = f"{global_step:06d}"
                LoRA_file = os.path.join(model_save_dir, f"{args.output_name}_{step_str}.safetensors")
                mapper_file = os.path.join(model_save_dir, f"{args.output_name}_{step_str}.pth")
                network.save_weights(LoRA_file, torch.float16, {})
                torch.save(mapper.state_dict(), mapper_file)

                args.save_now = False

            if global_step == total_train_steps:
                torch.save(mapper.state_dict(), f"{args.output_dir}/mapper.pth")
                break

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a mapper which can add watermark to a existed and well-trained LoRA module")
    utils.add_extra_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_optimizer_arguments(parser)

    # Performance
    parser.add_argument("--mapper_rank", type=int, default=160)
    parser.add_argument("--lora_alpha", type=int, default=80)
    parser.add_argument("--msg_bits", type=int, default=48)
    parser.add_argument("--finetune_scope", type=str, default="full-lin", choices=["full", "full-lin", "attn-mlp", "attn-only"])

    # Prepare
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="../model/base/anyloraCheckpoint_bakedvaeBlessedFp16.safetensors")
    parser.add_argument("--codec_weight_path", type=str, default="../model/codec_48bits.pth")
    parser.add_argument("--dataset_dir", type=str, default='../dataset/Anime_Character')
    parser.add_argument("--cache_latent", type=bool, default=True)
    parser.add_argument("--cache_latents_to_disk", type=bool, default=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--caption_extension", type=str, default=".txt")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16", "fp32"])

    # LoRA structure
    parser.add_argument("--lora_type", type=str, default="locon", choices=["standard", "locon", "loha", "lokr", "dylora", "lora"])

    # Resume
    parser.add_argument("--is_resume_from_checkpoint", type=bool, default=False)
    parser.add_argument("--resume_from_LoRA", type=str, default='')
    parser.add_argument("--resume_from_mapper", type=str, default='')

    # Train
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optimizer_type", type=str, default="AdamW")
    parser.add_argument("--LoRA_learning_rate", type=float, default=1e-4)
    parser.add_argument("--mapper_learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_train_steps", type=int, default=150000)
    parser.add_argument("--max_train_epochs", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--train_mode", type=str, default='both', choices=['both', 'switch'])
    parser.add_argument("--multires_noise_discount", type=float, default=0.3)
    parser.add_argument("--multires_noise_iterations", type=int, default=6)
    parser.add_argument("--clip_skip", type=int, default=None)

    ### lr scheduler
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_restarts")
    parser.add_argument("--lr_scheduler_num_cycles", type=int, default=3)
    parser.add_argument("--lr_scheduler_type", type=str, default="")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--lr_end", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_power", type=int, default=1)

    # Validate
    parser.add_argument("--validate_every_n_steps", type=int, default=1000)
    parser.add_argument("--validate_prompt_file", type=str, default='../validate_prompt/character_prompt.txt')
    parser.add_argument("--validate_image_num", type=int, default=4)
    parser.add_argument("--validate_prompt_prefix", type=str, default='')
    parser.add_argument("--validate_negative_prompt", type=str, default='lowres, bad anatomy, bad hands, text, error, missing fingers, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, cropped, blurry, artist name, extra hands, extra legs')
    parser.add_argument("--sample_sampler", type=str, default='ddim')

    # Save
    parser.add_argument("--log_with", type=str, default='tensorboard')
    parser.add_argument("--logging_dir", type=str, default='logs')
    parser.add_argument("--save_now", type=bool, default=False)
    parser.add_argument("--output_name", type=str, default='Release_Test')
    parser.add_argument("--output_dir", type=str, default="../output/Release_Test")
    parser.add_argument("--save_every_n_steps", type=int, default=10000)

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    log_path = os.path.join(args.output_dir, "logs")
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    log_file_name = f"log_{start_time}.log"
    log_file_path = os.path.join(log_path, log_file_name)

    tee = TeeOutput(log_file_path)
    sys.stdout = tee
    try:
        train(args)
    finally:
        sys.stdout = tee.stdout
        tee.close()
