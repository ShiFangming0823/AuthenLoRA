import argparse
import inspect
import math
import os
import sys
from datetime import datetime
import random

import numpy as np
import torch
from PIL import Image
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from library.device_utils import clean_memory_on_device

from library.train_util import _load_target_model

"""
Pytorch implementation of RGB convert to HSV, and HSV convert to RGB,
RGB or HSV's shape: (B * C * H * W)
RGB or HSV's range: [0, 1)
"""
import torch
from torch import nn


class RGB_HSV(nn.Module):
    def __init__(self, eps=1e-8):
        super(RGB_HSV, self).__init__()
        self.eps = eps

    def rgb_to_hsv(self, img):
        hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(img.device)

        hue[img[:, 2] == img.max(1)[0]] = 4.0 + ((img[:, 0] - img[:, 1]) / (img.max(1)[0] - img.min(1)[0] + self.eps))[img[:, 2] == img.max(1)[0]]
        hue[img[:, 1] == img.max(1)[0]] = 2.0 + ((img[:, 2] - img[:, 0]) / (img.max(1)[0] - img.min(1)[0] + self.eps))[img[:, 1] == img.max(1)[0]]
        hue[img[:, 0] == img.max(1)[0]] = (0.0 + ((img[:, 1] - img[:, 2]) / (img.max(1)[0] - img.min(1)[0] + self.eps))[img[:, 0] == img.max(1)[0]]) % 6

        hue[img.min(1)[0] == img.max(1)[0]] = 0.0
        hue = hue / 6

        saturation = (img.max(1)[0] - img.min(1)[0]) / (img.max(1)[0] + self.eps)
        saturation[img.max(1)[0] == 0] = 0

        value = img.max(1)[0]

        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)
        hsv = torch.cat([hue, saturation, value], dim=1)
        return hsv

    def hsv_to_rgb(self, hsv):
        h, s, v = hsv[:, 0, :, :], hsv[:, 1, :, :], hsv[:, 2, :, :]
        # 对出界值的处理
        h = h % 1
        s = torch.clamp(s, 0, 1)
        v = torch.clamp(v, 0, 1)

        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)

        hi = torch.floor(h * 6)
        f = h * 6 - hi
        p = v * (1 - s)
        q = v * (1 - (f * s))
        t = v * (1 - ((1 - f) * s))

        hi0 = hi == 0
        hi1 = hi == 1
        hi2 = hi == 2
        hi3 = hi == 3
        hi4 = hi == 4
        hi5 = hi == 5

        r[hi0] = v[hi0]
        g[hi0] = t[hi0]
        b[hi0] = p[hi0]

        r[hi1] = q[hi1]
        g[hi1] = v[hi1]
        b[hi1] = p[hi1]

        r[hi2] = p[hi2]
        g[hi2] = v[hi2]
        b[hi2] = t[hi2]

        r[hi3] = p[hi3]
        g[hi3] = q[hi3]
        b[hi3] = v[hi3]

        r[hi4] = t[hi4]
        g[hi4] = p[hi4]
        b[hi4] = v[hi4]

        r[hi5] = v[hi5]
        g[hi5] = p[hi5]
        b[hi5] = q[hi5]

        r = r.unsqueeze(1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)
        rgb = torch.cat([r, g, b], dim=1)
        return rgb

def log_print(*args, **kwargs):
    now = datetime.now()
    formatted_datetime = now.strftime("%Y.%m.%d %H:%M:%S")
    timestamp = now.strftime(formatted_datetime)

    caller_frame = inspect.currentframe().f_back
    script_name = os.path.basename(caller_frame.f_code.co_filename)
    line_number = caller_frame.f_lineno

    script_info = f"{script_name}:{line_number}"
    formatted_script_info = f"{script_info:<25}"

    print(f"[{timestamp}] {formatted_script_info}", *args, **kwargs)


def load_target_model(name_or_path, weight_dtype, device):
    text_encoder, vae, unet, load_stable_diffusion_format = _load_target_model(name_or_path, weight_dtype, device, )
    clean_memory_on_device(device)
    return text_encoder, vae, unet, load_stable_diffusion_format


def generate_train_sequence(input_str, total_train_steps):
    items = input_str.split(', ')

    result = []

    for item in items[:-2]:
        key, value = item.split(':')
        result.extend([key] * int(value))

    last_two = items[-2:]
    while len(result) < total_train_steps:
        for item in last_two:
            key, value = item.split(':')
            result.extend([key] * int(value))

    result = result[:total_train_steps]

    return result


def cosine_scheduler(global_step):
    if global_step < 1000:
        return 1.0
    elif global_step > 150000:
        return 0.5
    else:
        cos_value = math.cos(math.pi * (global_step - 1000) / 149000)
        return 0.25 * cos_value + 0.75


def sine_scheduler(global_step):
    if global_step < 1000:
        return 0.0
    elif global_step > 150000:
        return 10.0
    else:
        sin_value = math.sin((math.pi / 2) * (global_step - 1000) / 149000)
        return 10.0 * sin_value

def style_ratio_scheduler(global_step, start_step, start_ratio, end_step, end_ratio):
    if global_step < start_step:
        return start_ratio
    elif global_step > end_step:
        return end_ratio
    else:
        sin_value = math.sin((math.pi / 2) * (global_step - start_ratio) / (end_step - start_step))
        return start_ratio + (end_ratio - start_ratio) * sin_value


def pyramid_noise_like(noise, device, iterations, discount):
    b, c, w, h = noise.shape  # EDIT: w and h get over-written, rename for a different variant!
    u = torch.nn.Upsample(size=(w, h), mode="bilinear").to(device)
    for i in range(iterations):
        r = random.random() * 2 + 2  # Rather than always going 2x,
        wn, hn = max(1, int(w / (r ** i))), max(1, int(h / (r ** i)))
        noise += u(torch.randn(b, c, wn, hn).to(device)) * discount ** i
        if wn == 1 or hn == 1:
            break  # Lowest resolution is 1x1
    return noise / noise.std()

def get_cosine_schedule_with_warmup_lr_end(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1, lr_end=0.0
):

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(lr_end, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def torch_to_pil(images):
    images = images.detach().cpu().float()
    if images.ndim == 3:
        images = images[None, ...]
    images = images.permute(0, 2, 3, 1)
    images = (images + 1) * 0.5
    images = (images * 255).clamp(0, 255).detach().cpu().numpy().astype(np.uint8)
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def differentiable_max(image, alpha):
    weights = torch.softmax(alpha * image, dim=1)
    return (weights * image).sum(dim=1, keepdim=True)  # 平滑最大值


def add_extra_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--log_prefix", type=str, default=None)
    parser.add_argument("--tokenizer_cache_dir", type=str, default=None)
    parser.add_argument("--torch_compile", action="store_true")
    parser.add_argument("--ddp_timeout", type=int, default=None)
    parser.add_argument("--ddp_gradient_as_bucket_view", action="store_true")
    parser.add_argument("--ddp_static_graph", action="store_true")
    parser.add_argument("--deepspeed", type=bool, default=False)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--v2", action="store_true")
    parser.add_argument("--vae", type=str, default=None)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--max_token_length", type=int, default=None)
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "quad", "warmup10", "warmup50", "const", "jsd"])
    parser.add_argument("--beta_start", type=float, default=0.0001)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--num_diffusion_timesteps", type=int, default=1000)


class TeeOutput:
    def __init__(self, filename, mode="w"):
        self.file = open(filename, mode)
        self.stdout = sys.stdout

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()
