### Merging Watermark Information into LoRA

In Generation and Evaluation, watermark information is randomly sampled and embedded into LoRA's B matrix through a loosely coupled approach. We provide scripts here to **bake random or specified secret information into LoRA**, enabling other Stable Diffusion implementations to utilize the watermarked LoRA.

In this script, we need well-trained LoRA weights (.safetensors file) and the corresponding mapper weights (.pth files).

```bash
python utils/bake_AuthenLoRA.py \
	--pretrained_model_name_or_path="/path/to/base_model.safetensors" \
	--LoRA_path="/path/to/LoRA.safetensors" \
	--mapper_path="/path/to/mapper.pth" \
	--mapper_rank=160 \
	--lora_alpha=80 \
	--msg_bits=48 \
	--secret_info=None
```

| Parameter        | Type | Default  | Description                                                  |
| ---------------- | ---- | -------- | ------------------------------------------------------------ |
| `mapper_rank`    | int  | 160      | The `mapper_rank` should match the rank of the LoRA you set during training. |
| `lora_alpha`     | int  | 80       | The `lora_alpha` should match the alpha of the LoRA you set during training. |
| `msg_bits`       | int  | 48       | The `msg_bits` should match the bit num of the LoRA you set during training. |
| `finetune_scope` | str  | full-lin | The `finetune_scope` should match the finetuning scope of the LoRA you set during training. |
| `lora_type`      | str  | lora     | The `lora_type` should match the type of the LoRA you set during training. LoHa and LoKr has not been tested in this script. |
| `secret_info`    | str  | None     | If `secret_info` is set to None, the embedded 48-bit watermark information will be randomly generated. If you wish to specify the watermark information, assign this variable a 48-bit binary sequence at runtime. |

After the script runs, it will generate a folder named `{secret_info}` in the specified LoRA weights directory. This folder contains an AuthenLoRA file with embedded watermark information.

```
/LoRA_path
├── {secret_info}
|	├── {LoRA_name}.safetensors 	// Baked AuthenLoRA file
```

