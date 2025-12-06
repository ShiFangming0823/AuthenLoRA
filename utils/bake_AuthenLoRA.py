import argparse
import importlib
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
from networks.models import MapperNet
from train.utils import load_target_model, log_print

calculate_dtype = torch.float16
save_dtype = torch.float16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def apply_mapper_result(mapper_path, secret_info, input_size, output_size):
    mapper = MapperNet(input_size=input_size, output_size=output_size)
    mapper.load_state_dict(torch.load(mapper_path))
    secret_emb = mapper(secret_info)
    diag_secret = torch.diag_embed(secret_emb)[0]

    return diag_secret

def process_model(args):
    if args.secret_info is None:
        secret_info = torch.randint(0, 2, (1, args.msg_bits))
        flatten = secret_info.flatten()
        str_secret = ''.join(str(bit.item()) for bit in flatten)
    else:
        str_secret = args.secret_info
        bit_list = [int(bit) for bit in str_secret]
        secret_info = torch.tensor(bit_list, dtype=calculate_dtype).view(1, args.msg_bits)

    diag_secret = apply_mapper_result(args.mapper_path, secret_info, input_size=args.msg_bits, output_size=args.mapper_rank).to(calculate_dtype)
    print(str_secret)
    print(diag_secret)

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

    text_encoder, vae, unet, _ = load_target_model(args.pretrained_model_name_or_path, calculate_dtype, device)

    network_module = importlib.import_module(lora_module)
    network = network_module.create_network(multiplier=1.0, network_dim=args.mapper_rank, network_alpha=args.lora_alpha, vae=vae, text_encoder=text_encoder, unet=unet, neuron_dropout=None, **net_kwargs)
    network.apply_to(text_encoder, unet, apply_text_encoder=True, apply_unet=True)
    network.to(device, dtype=calculate_dtype)
    network.eval()
    log_print(network.load_weights(args.LoRA_path))
    log_print(f"The LoRA weight file has been loaded: {args.LoRA_path}")

    network.burn_mapper(diag_secret)

    # Save the result
    dir_path = os.path.dirname(args.LoRA_path)
    file_name = os.path.basename(args.LoRA_path)
    file_name_without_ext, ext = os.path.splitext(file_name)

    save_dir = os.path.join(dir_path, str_secret)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{file_name_without_ext}_{str_secret}{ext}")

    network.save_weights(save_path, save_dtype, {})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="../model/base/anyloraCheckpoint_bakedvaeBlessedFp16.safetensors")
    parser.add_argument("--tokenizer_cache_dir", type=str, default=None)
    parser.add_argument("--v2", action="store_true")
    parser.add_argument("--clip_skip", type=int, default=None)
    parser.add_argument("--mapper_rank", type=int, default=160)
    parser.add_argument("--lora_alpha", type=int, default=80)
    parser.add_argument("--msg_bits", type=int, default=48)
    parser.add_argument("--finetune_scope", type=str, default="full-lin", choices=["full", "full-lin", "attn-mlp", "attn-only"])
    parser.add_argument("--LoRA_path", type=str, default=r"../../models/Extra/Anime_Character_rank160_fp16_48bits.safetensors")
    parser.add_argument("--mapper_path", type=str, default=r"../../models/Extra/Anime_Character_rank160_fp16_48bits.pth")
    parser.add_argument("--lora_type", type=str, default="lora", choices=['lora', 'locon', 'loha', 'lokr'])
    parser.add_argument("--secret_info", type=str, default=None)
 

    args = parser.parse_args()

    process_model(args)