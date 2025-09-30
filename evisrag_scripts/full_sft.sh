#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path xxx/Qwen2.5-VL-7B-Instruct \
    --preprocessing_num_workers 16 \
    --finetuning_type full \
    --freeze_vision_tower true \
    --template qwen2_vl \
    --flash_attn fa2 \
    --bf16 true \
    --dataset_dir data/Train \
    --dataset evidencecot \
    --val_size 0.0 \
    --image_max_pixels 3920000 \
    --cutoff_len 32000 \
    --learning_rate 5e-7 \
    --num_train_epochs 1.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 200 \
    --packing False \
    --output_dir xxx/Qwen7B-SFT \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --deepspeed config/ds_z3_config.json \
    --report_to tensorboard > sft.log 2>&1