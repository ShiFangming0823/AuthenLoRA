### Train

#### Stage 1: Pre-training for the Watermark Pattern

You can download datasets from [Hugging Face](https://huggingface.co/datasets/shifangming0823/AuthenLoRA_Dataset): **COCO2017test.zip**

The unzipped dataset directory structure should be as follows:

```
/COCO2017test
└── train
    ├── 000000000071.jpg ... (10000 images)
    └── metadata.jsonl
```

Then run the following command to start the training process for watermark encoder and decoder:

```bash
python train_1_encoder&decoder.py
	--pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
	--epochs 50 \
    --batch_size 2 \
    --bit_num 48 \
	--dataset /path/to/dataset_dir \
	--output_dir /path/to/save_dir 
```

After training, you will obtain these dirs and files:

```
/save_dir
├── checkpoints
|	└── state_dict_{epoch}_{step}.pth 	// codec model weight
├── logs 								// You can locate and start Tensorboard in this dir to monitor the training process.
└── log_images
	└── residual_{validate_step}.png	// You can view the residual images to check the training process.
```

#### Stage 2: Stylization and Watermark Embedding

We provide four stylized datasets on [Hugging Face](https://huggingface.co/datasets/shifangming0823/AuthenLoRA_Dataset): Anime_Character.zip, Realistic_Human.zip, PixelArt.zip, and Lolita.zip. 

You may select any one for experimentation. Alternatively, you can build and utilize your own stylized datasets following the same directory structure.

The dataset directory structure should be as follows:

```
/stylized_dataset
├── 00001.png
├── 00001.txt
├── 00002.png
└── 00002.txt...
```

To achieve optimal generation quality, we recommend utilizing base models that align with the specific stylistic attributes of the training datasets.

| Dataset File Name | Base Model File Name | Download Link |
| --- | --- | --- |
| `Anime_Character.zip` | `anyloraCheckpoint_bakedvaeBlessedFp16.safetensors` | [Civitai Link](https://civitai.com/models/23900?modelVersionId=95489) |
| `Realistic_Human.zip` | `v1-5-pruned-emaonly.safetensors` | [HuggingFace Link](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main) |
| `PixelArt.zip` | `anyloraCheckpoint_bakedvaeBlessedFp16.safetensors` | [Civitai Link](https://civitai.com/models/23900?modelVersionId=95489) |
| `Lolita.zip` | `majicmixRealistic_v7.safetensors` | [Civitai Link](https://civitai.com/models/43331/majicmix-realistic) |

**Then run the following command to start the training process for LoRA and mapper:**

```bash
python train/2_train_for_AuthenLoRA.py \
	--pretrained_model_name_or_path="/path/to/base_model.safetensors" \
	--codec_weight_path="/path/to/codec_weight.pth" \
	--dataset_dir="/path/to/dataset_dir" \
	--train_batch_size=1 \
	--validate_prompt_file="/path/to/prompt.txt" \
	--validate_prompt_prefix="The prefix of your stylized dataset" \
	--validate_negative_prompt="lowres, bad anatomy, bad hands, text, error, missing fingers, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, cropped, blurry, artist name, extra hands, extra legs" \
	--output_name="save_name" \
	--output_dir="/path/to/save_dir"
```

You can also append the following parameters to the running command as needed. Parameters not listed can generally be left at their default values.

| Parameter                   | Type | Default  | Description                                                  |
| --------------------------- | ---- | -------- | ------------------------------------------------------------ |
| `mapper_rank`               | int  | 160      | It controls the rank of LoRA and the final file size. Experiments in the paper demonstrate that it can be minimized to as low as 64 while maintaining the BitAcc up to 95%. |
| `lora_alpha`                | int  | 80       | It controls the strength of LoRA together with `mapper_rank`. It is recommended to set it to half of `mapper_rank`. |
| `msg_bits`                  | int  | 48       | Larger values can carry more watermark information in bits, but require additional training of the codec. We have provided pre-trained 80-bit and 100-bit codecs on Hugging Face for test. |
| `finetune_scope`            | str  | full-lin | choices: `full` (Best performance, largest size), `full-lin` (Default), `attn-mlp` (Default for normal LoRA), `attn-only` (Failed to embed watermark) |
| `is_resume_from_checkpoint` | bool | False    | When you need to resume training from a specific step, set this parameter to True. Simultaneously, provide the safetensors file and pth file in both `resume_from_LoRA` and `resume_from_mapper`. Additionally, set `start_step` to the step number where training was interrupted. |
| `lora_type`                 | str  | 'lora'   | choices: `lora`, `locon`, `loha`, `lokr`                     |
| `max_train_steps`           | int  | 150000   | In most cases, AuthenLoRA converges between 75,000 and 100,000 steps. |
| `validate_prompt_file`      | str  |          | Includes a series of prompts for generating evaluation images. |

The ``validate_prompt_prefix`` and in our provided dataset is shown in the table.

| Dataset File Name   | `validate_prompt_prefix`                                     |
| ------------------- | ------------------------------------------------------------ |
| Anime_Character.zip | kanadehisaishi, kanade hisaishi, short hair, bangs, black hair, red eyes, bow, hair bow, red bow, skirt, shirt, school uniform, white shirt, short sleeves, pleated skirt, serafuku, sailor collar, blue skirt, neckerchief, blue sailor collar, green neckerchief, kitauji high school uniform, masterpiece, best quality, high resolution, unity 8k wallpaper, illustration, beautiful detailed eyes, extremely detailed face, perfect lighting, extremely detailed CG, perfect hands, perfect anatomy |
| Realistic_Human.zip | RyanGosling, A stunning closeup intricate full colour portrait of (man:1), epic character composition, sharp focus, natural lighting, subsurface scattering, solo, close-up, portrait |
| PixelArt.zip        | pixel, pixel art, toon \(style\)                             |
| Lolita.zip          | lo dress, layered dress, long dress, lace-trimmed dress, frills, puffy sleeves, wide sleeves, bow, jewelry, long sleeves, hat, (pink dress:1.2), best quality, masterpiece, photorealistic, solo, full body, standing, long black straight hair, blunt bangs, looking at viewer, smile, detailed_background |

If you use your own stylized dataset, ensure that the ``validate_prompt_prefix`` matches the prefix of the image prompts in your dataset. This ensures LoRA is correctly invoked during image generation.

**After training**, you will obtain these dirs and files:

```
/save_dir
├── checkpoint_DATE_TIME
|	├── {output_name}_{save_steps}.safetensors 	// LoRA model weight
|	└── {output_name}_{save_steps}.pth 			// mapper model weight
└── logs 										// You can locate and start Tensorboard in this dir to monitor the training process, including loss curve, BitAcc curve and visualization of generated stylized images. This dir also contains with the config file and terminal output log file.

```

You can:

* use trained LoRA weight and mapper weight to generate stylized and watermarked images using [our provided code](Generation_Evaluation.md). 

* use mapper to [embed the watermark information into the LoRA module](Merging.md), and then use the watermarked LoRA to generate images using the official Stable Diffusion implementation.
