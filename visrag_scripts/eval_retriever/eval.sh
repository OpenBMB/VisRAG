# on each node, the script will only run once.
MAX_Q_LEN=$1
MAX_P_LEN=$2
PER_DEV_BATCH_SIZE=$3
GPUS_PER_NODE=${4}
POOLING=${5}
ATTENTION=${6}
SUB_DATASET=${7}
MODEL_PATH=${8}

if [[ $MODEL_PATH != *"SigLIP"* ]]; then
    attn_implementation='sdpa'
else
    attn_implementation='eager'
fi

WORLD_SIZE=1
RANK=0
MASTER_ENDPOINT=localhost
MASTER_PORT=23456

CHECKPOINT_DIR="/data/checkpoints"
TIMESTR=$(date "+%Y-%m-%d-%H%M%S")
IDENTITY="eval-$TIMESTR-maxq-$MAX_Q_LEN-maxp-$MAX_P_LEN-bsz-$PER_DEV_BATCH_SIZE-pooling-$POOLING-attention-$ATTENTION-gpus-per-node-$GPUS_PER_NODE"
RESULT_DIR="$CHECKPOINT_DIR/$IDENTITY"

# use IFS to split the string into array
IFS=',' read -r -a SUB_DATASET_LIST <<< "$SUB_DATASET"

for SUB_DATASET in "${SUB_DATASET_LIST[@]}"
do
    echo "Evaluating: $SUB_DATASET"
    
    THIS_RESULT_DIR="$RESULT_DIR/$SUB_DATASET"
    echo "This dataset result dir: $THIS_RESULT_DIR"

    CORPUS_PATH="openbmb/VisRAG-Ret-Test-${SUB_DATASET}"
    QUERY_PATH="openbmb/VisRAG-Ret-Test-${SUB_DATASET}"
    QRELS_PATH="openbmb/VisRAG-Ret-Test-${SUB_DATASET}"

    echo "CORPUS_PATH: $CORPUS_PATH" 
    echo "QUERY_PATH: $QUERY_PATH" 
    echo "QRELS_PATH: $QRELS_PATH" 

    QUERY_TEMPLATE="Represent this query for retrieving relevant documents: <query>"
    CORPUS_TEMPLATE="<text>"
    
    TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun \
        --nnodes=$WORLD_SIZE \
        --node_rank=$RANK \
        --nproc_per_node=$GPUS_PER_NODE \
        --master_addr=$MASTER_ENDPOINT \
        --master_port=$MASTER_PORT \
        src/openmatch/driver/eval.py \
        --qrels_path $QRELS_PATH \
        --query_path $QUERY_PATH \
        --corpus_path $CORPUS_PATH \
        --model_name_or_path "$MODEL_PATH" \
        --output_dir "$THIS_RESULT_DIR" \
        --query_template "$QUERY_TEMPLATE" \
        --doc_template "$CORPUS_TEMPLATE" \
        --q_max_len $MAX_Q_LEN \
        --p_max_len $MAX_P_LEN  \
        --per_device_eval_batch_size $PER_DEV_BATCH_SIZE \
        --dataloader_num_workers 1 \
        --dtype float16 \
        --use_gpu \
        --overwrite_output_dir false \
        --max_inmem_docs 1000000 \
        --normalize true \
        --pooling "$POOLING" \
        --attention "$ATTENTION" \
        --attn_implementation $attn_implementation \
        --phase "encode" \
        --from_hf_repo \

    TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun \
        --nnodes=$WORLD_SIZE \
        --node_rank=$RANK \
        --nproc_per_node=$GPUS_PER_NODE \
        --master_addr=$MASTER_ENDPOINT \
        --master_port=$MASTER_PORT \
        src/openmatch/driver/eval.py \
        --model_name_or_path "$MODEL_PATH" \
        --qrels_path $QRELS_PATH \
        --query_path $QUERY_PATH \
        --corpus_path $CORPUS_PATH \
        --output_dir "$THIS_RESULT_DIR" \
        --use_gpu \
        --phase "retrieve" \
        --retrieve_depth 10 \
        --from_hf_repo \

done



