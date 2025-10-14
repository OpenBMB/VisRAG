
# Benchmark: ChartVQA InfoVQA SlideVQA DocVQA ViDoSeek
# vanilla vlm: qwen3b qwen7b, qwen32b mimo
# vlrm: Vision-R1 Ocean_R1 ThinkLite MM-Eureka OpenVLThinker
# vrag: r1router mmsearch vragrl EVisRAG7B EVisRAG3B
# method: baseline, CCOT, COCOT, DDCOT, evidence_prompt_grpo

python src/evisrag/predict.py \
    --benchmark DocVQA \
    --model EVisRAG7B \
    --method evidence_prompt_grpo \
    --idx 0 \
    --temperature 0.0 \
    --topk 3