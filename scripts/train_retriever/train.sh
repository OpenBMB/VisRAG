# on each node, the script will only run once.
export CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7
MAX_SEQ_LEN=$1
PER_DEV_BATCH_SIZE=$2
GPUS_PER_NODE=$3
SOFTMAX_TEMPERATURE=$4
EPOCH=$5
QUERY_INSTRUCTION=$6 # bool
CORPUS_INSTRUCTION=$7 # bool
DEEPSPEED=$8
LR=$9
MAPPING=${10} # stream data
POOLING=${11}
ATTENTION=${12} 
NPASSAGE=${13}
GRADCACHE=${14}
GRADCACHE_MICRO=${15}
PASSAGE_STOP_GRAD=${16} # by default it is false
MODEL_PATH=${17}
DATASET_PATH=${18}

WORLD_SIZE=1
RANK=0
MASTER_ENDPOINT=localhost
MASTER_PORT=23456

TIMESTR=$(date "+%Y-%m-%d-%H%M%S")
IDENTITY="train-$TIMESTR-model-data-lr-$LR-softm_temp-$SOFTMAX_TEMPERATURE-bsz$PER_DEV_BATCH_SIZE-ngpus$GPUS_PER_NODE-nnodes$WORLD_SIZE-inbatch-$IN_BATCH-nepoch-$EPOCH-pooling-$POOLING-attention-$ATTENTION-qinstruct-$QUERY_INSTRUCTION-cinstruct-$CORPUS_INSTRUCTION-gradcache-$GRADCACHE-passage-stopgrad-$PASSAGE_STOP_GRAD-npassage-$NPASSAGE"
CHECKPOINT_DIR=/home/tangchaoyue/checkpoints
LOG_DIR=/home/tangchaoyue/tensorboard
IN_BATCH=true
LORA=false
LORA_R=32
MAX_Q_LEN=$MAX_SEQ_LEN
MAX_P_LEN=$MAX_SEQ_LEN


if [[ $MODEL_PATH != *"SigLIP"* ]]; then
    attn_implementation='sdpa'
else
    attn_implementation='eager'
fi

echo attn_implementation: $attn_implementation


TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun \
    --nnodes=$WORLD_SIZE \
    --node_rank=$RANK \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=$MASTER_ENDPOINT \
    --master_port=$MASTER_PORT \
    src/openmatch/driver/train.py \
    --overwrite_output_dir \
    --output_dir "$CHECKPOINT_DIR/$IDENTITY" \
    --model_name_or_path $MODEL_PATH \
    --do_train  \
    --save_steps 500  \
    --train_dir $DATASET_PATH \
    --per_device_train_batch_size $PER_DEV_BATCH_SIZE  \
    --train_n_passages $NPASSAGE  \
    --learning_rate $LR  \
    --q_max_len $MAX_Q_LEN  \
    --p_max_len $MAX_P_LEN  \
    --num_train_epochs $EPOCH  \
    --logging_dir "$LOG_DIR/$IDENTITY" \
    --negatives_x_device \
    --softmax_temperature $SOFTMAX_TEMPERATURE \
    --logging_steps 1 \
    --inbatch_loss $IN_BATCH \
    --lora $LORA \
    --lora_r $LORA_R \
    --gradient_checkpointing true \
    --dataloader_num_workers 1 \
    --save_safetensors true \
    --query_instruction $QUERY_INSTRUCTION \
    --corpus_instruction $CORPUS_INSTRUCTION \
    --use_mapping_dataset $MAPPING \
    --normalize true \
    --pooling $POOLING \
    --attention $ATTENTION \
    --grad_cache_enable $GRADCACHE \
    --grad_cache_micro_batch_size $GRADCACHE_MICRO \
    --passage_stop_grad $PASSAGE_STOP_GRAD \
    --dataloader_drop_last true \
    --attn_implementation $attn_implementation \
    --dtype bfloat16 \
    --from_hf_repo \
    $( [[ $DEEPSPEED != 'false' ]] && echo "--deepspeed $DEEPSPEED" ) \
    
