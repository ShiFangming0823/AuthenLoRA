## Generation and Evaluation

To generate stylized and watermarked images using well-trained AuthenLoRA and evaluate their BitAcc and TPR in both clean and adversarial circumstances, you can run the following command. 

The final results will be summarized after all tests are completed.

```bash
python utils/Eval.py
	--msg_bits=48 \
	--pretrained_model_name_or_path="/path/to/base_model.safetensors" \
	--codec_weight_path="/path/to/codec_weight.pth" \
	--LoRA_weight_path="/path/to/LoRA_weight.safetensors" \
	--lora_rank=160 \
	--lora_alpha=80 \
	--generate_image True \
	--validate_prompt_file="/path/to/prompt.txt" \
	--validate_image_num=5 \
	--validate_prompt_prefix="The prefix of LoRA" \
	--validate_negative_prompt='lowres, bad anatomy, bad hands, text, error, missing fingers, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, cropped, blurry, artist name, extra hands, extra legs' \
	--eval_clean_bitacc_tpr=True --tpr_threshold=1e-6 \
	--eval_distortion_bitacc_tpr=True \
	--distortion_types ['blur', 'color_jitter', 'crop', 'jpeg_compress', 'noise', 'sharpness', 'SDEdit', 'SDEdit2'] \
```

**General**

| Parameter            | Type | Default  | Description                                                  |
| -------------------- | ---- | -------- | ------------------------------------------------------------ |
| `msg_bits`           | int  | 48       | The `msg_bits` should match the bit num of the LoRA you set during training. |
| `mapper_weight_path` | str  | None     | If set to None, the program will automatically search for the .pth file with the same name under the same path as the `LoRA_weight_path`. |
| `lora_rank`          | int  | 160      | The `lora_rank` should match the rank of the LoRA you set during training. |
| `lora_alpha`         | int  | 80       | The `lora_alpha` should match the alpha of the LoRA you set during training. |
| `finetune_scope`     | str  | full-lin | The `finetune_scope` should match the finetuning scope of the LoRA you set during training. |
| `lora_type`          | str  | lora     | The `lora_type` should match the type of the LoRA you set during training. |

**Image Generation**

| Parameter                | Type | Default | Description                                                  |
| ------------------------ | ---- | ------- | ------------------------------------------------------------ |
| `generate_image`         | bool | True    | If set to True, the script will generate images using LoRA and a mapper. If set to False, the image generation process will be skipped. |
| `validate_prompt_file`   | str  |         | Includes a series of prompts for generating evaluation images. |
| `validate_image_num`     | int  | 5       | Control how many images each prompt generates.               |
| `validate_prompt_prefix` | str  |         | Refer to [Train.md](Train.md). This parameter should match the same setting during training. |
| `sample_sampler`         | str  | ddim    |                                                              |
| `steps`                  | int  | 20      | Denoising steps                                              |
| `CFG`                    | int  | 7       | Classifier-Free Guidance Scale                               |

The ``validate_prompt_prefix`` and in our provided trained AuthenLoRA model is shown in the table.

| Model File Name   | `validate_prompt_prefix`                                     |
| ------------------- | ------------------------------------------------------------ |
| Anime_Character | kanadehisaishi, kanade hisaishi, short hair, bangs, black hair, red eyes, bow, hair bow, red bow, skirt, shirt, school uniform, white shirt, short sleeves, pleated skirt, serafuku, sailor collar, blue skirt, neckerchief, blue sailor collar, green neckerchief, kitauji high school uniform, masterpiece, best quality, high resolution, unity 8k wallpaper, illustration, beautiful detailed eyes, extremely detailed face, perfect lighting, extremely detailed CG, perfect hands, perfect anatomy |
| Realistic_Human | RyanGosling, A stunning closeup intricate full colour portrait of (man:1), epic character composition, sharp focus, natural lighting, subsurface scattering, solo, close-up, portrait |
| PixelArt        | pixel, pixel art, toon \(style\)                             |
| Lolita          | lo dress, layered dress, long dress, lace-trimmed dress, frills, puffy sleeves, wide sleeves, bow, jewelry, long sleeves, hat, (pink dress:1.2), best quality, masterpiece, photorealistic, solo, full body, standing, long black straight hair, blunt bangs, looking at viewer, smile, detailed_background |

**Clean Image Evaluation**

| Parameter                    | Type  | Default | Description                                                  |
| ---------------------------- | ----- | ------- | ------------------------------------------------------------ |
| `eval_distortion_bitacc_tpr` | bool  | True    | If set to True, the script will evaluate the BitAcc and TPR of generated images. If set to False, this process will be skipped. |
| `validate_image_path`        | str   |         | If `generate_image` is True, the script will automatically evaluate the generated image. This parameter can be left empty. If `generate_image` is False or you wish to evaluate an image that has already been generated, specify the path to the folder containing the image to be evaluated here. **Note that the folder name must end with the binary sequence of the ground truth secret information.** |
| `tpr_threshold`              | float | 1e-6    | Used to calculate the True Positive Rate (TPR) based on the False Positive Rate (FPR) |

**Distortion & Composite Distortion Image Evaluation**

| Parameter                              | Type | Default                      | Description                                                  |
| -------------------------------------- | ---- | ---------------------------- | ------------------------------------------------------------ |
| `eval_distortion_bitacc_tpr`           | bool | True                         | If set to True, the script will evaluate the BitAcc and TPR of generated images after distortion processing. If set to False, this process will be skipped. |
| `distortion_types`                     | list |                              | We provide a range of distortion processing operations, including `['blur', 'color_jitter', 'crop', 'jpeg_compress', 'noise', 'sharpness', 'SDEdit', 'SDEdit2']`. If you don't want to evaluate any of them, simply remove them from the list. |
| `crop_resolution`                      | int  | 460                          | Used to control the intensity of random cropping.            |
| `save_result`                          | bool | False                        | If set to True, the processed image will be saved.           |
| `eval_composite_distortion_bitacc_tpr` | bool | False                        | If set to True, the script will evaluate the BitAcc and TPR of generated images after composite distortion processing. If set to False, this process will be skipped. |
| `composite_distortion_types`           | str  | 'blur, jpeg_compress, noise' |                                                              |

**Evaluate FID Score**

| Parameter            | Type | Default | Description                                                  |
| -------------------- | ---- | ------- | ------------------------------------------------------------ |
| `eval_FID`           | bool | False   | If set to True, the script will evaluate the FID of generated images with images from `clean_image_folder`. If set to False, this process will be skipped. |
| `clean_image_folder` | str  |         | Path of the image being calculated FID.                      |

After each evaluation, the terminal will print the current evaluation type and result. 

After all evaluations are complete, the terminal will print all results.

