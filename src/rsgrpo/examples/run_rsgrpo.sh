#!/bin/bash
# REMINDER: this script uses test data split and should ONLY be used for debugging. DO NOT use for training.

set -x

#sleep 6h

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0

MODEL_PATH=../models/Qwen2.5-VL-7B-SFT  # replace it with your local file path
# MODEL_PATH=../LLaMA-Factory-main/saves/evidencecot/Qwen2.5-VL-7B-Instruct/checkpoint-1500

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files="./data/newnewtrain.jsonl" \
    data.val_files="./data/test.jsonl" \
    data.rollout_batch_size=32 \
    worker.rollout.enable_chunked_prefill=false \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.limit_images=5 \
    trainer.experiment_name=sleuth5 \
    trainer.n_gpus_per_node=8
