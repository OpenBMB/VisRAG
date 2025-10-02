#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0

MODEL_PATH=xxx/Qwen7B-SFT  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files="../../data/Train/evidencecot_GRPO.jsonl" \
    data.val_files="../../data/Train/test_GRPO.jsonl" \
    data.rollout_batch_size=32 \
    worker.rollout.enable_chunked_prefill=false \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.limit_images=5 \
    trainer.experiment_name=EVisRAG7B \
    trainer.n_gpus_per_node=8
