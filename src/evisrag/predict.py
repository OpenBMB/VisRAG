import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import sys
import json
from tqdm import tqdm
import re
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import argparse
from prompt import (
    evidence_promot_grpo,
    evidence_promot_oneshot,
    baseline, COCOT, CCOT, DDCOT)

"""
Benchmark: ChartVQA InfoVQA SlideVQA DocVQA ViDoSeek

Models:
# vanilla vlm: qwen3b qwen7b, qwen32b mimo
# vlrm: Vision-R1 Ocean_R1 ThinkLite MM-Eureka OpenVLThinker
# vrag: r1router mmsearch vragrl EVisRAG3B EVisRAG7B
# method: baseline, CCOT, COCOT, DDCOT, v3_old, v3_prompt evidence_prompt_grpo

python evisrag_scripts/predict.py --benchmark DocVQA --model EVisRAG7B --method evidence_prompt_grpo --idx 0 --temperature 0.0 --topk 3
"""

parser = argparse.ArgumentParser()
parser.add_argument("--benchmark", type=str, default="ChartQA")  # benchmark
parser.add_argument("--model", type=str, default="qwen7b")  # model
parser.add_argument("--method", type=str, default="baseline_multiimg") # prompt
parser.add_argument("--idx", type=int, default=-1)
parser.add_argument("--temperature", type=float, default=0.1)
parser.add_argument("--topk", type=int, default=3)

args = parser.parse_args()


if __name__ == "__main__":
    if args.model == "qwen7b":
        model_path = "xxx/Qwen2.5-VL-7B-Instruct"
    elif args.model == "mimo":
        model_path = "xxx/MiMo-VL-7B-RL"
    elif args.model == "qwen3b":
        model_path = "xxx/Qwen2.5-VL-3B-Instruct"
    elif args.model == "qwen32b":
        model_path = "xxx/Qwen2.5-VL-32B-Instruct"
    elif args.model == "qwen72b":
        model_path = "xxx/Qwen2.5-VL-72B-Instruct"
    elif args.model == "r1router":
        model_path = "xxx/public/R1-Router"
    elif args.model == "mmsearch":
        model_path = "xxx/MMSearch-R1"
    elif args.model == "vragrl":
        model_path = "xxx/VRAG-RL"
    elif args.model == "Ocean_R1":
        model_path = "xxx/Ocean_R1_7B_Instruct"
    elif args.model == "OpenVLThinker":
        model_path = "xxx/OpenVLThinker-7B"
    elif args.model == "Vision-R1":
        model_path = "xxx/Qwen2.5-VL-7B-Instruct-Vision-R1"
    elif args.model == "ThinkLite":
        model_path = "xxx/ThinkLite-VL-7B"
    elif args.model == "MM-Eureka":
        model_path = "xxx/MM-Eureka-Qwen-7B"
    elif args.model == "EVisRAG3B":
        model_path = "xxx/EVisRAG-3B"
    elif args.model == "EVisRAG7B":
        model_path = "xxx/EVisRAG-7B"

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, padding_side='left')

    msgs, results = [], []
    data_path = "./data"
    with open(f'{data_path}/EVisRAG-Test-{args.benchmark}/top3_test.jsonl', 'r') as fin:
        for line in fin:
            item = json.loads(line.strip())
            qid, imgs, query = item["qid"], item["image"][:args.topk], item["query"]

            imgs = [f'{data_path}/EVisRAG-Test-{args.benchmark}/{p}' for p in imgs]
            results.append({
                "qid": qid,
                "imgs": imgs,
            })

            if args.method == "baseline":
                input_prompt = baseline(query, args.model)
            elif args.method == "COCOT":
                input_prompt = COCOT(query)
            elif args.method == "CCOT":
                input_prompt = CCOT(query)
            elif args.method == "DDCOT":
                input_prompt = DDCOT(query)
            elif args.method == "evidence_prompt_notrain":
                input_prompt = evidence_promot_oneshot(query)
            elif args.method == "evidence_prompt_grpo":
                input_prompt = evidence_promot_grpo(query)

            content = [{"type": "text", "text": input_prompt}]
            for imgP in imgs:
                content.append({
                    "type": "image",
                    "image": imgP
                })

            msgs.append([{
                        "role": "user",
                        "content": content,
                    }])

    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        dtype="bfloat16",
        limit_mm_per_prompt={"image":5, "video":0},
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        repetition_penalty=1.05,
        max_tokens=2048,
    )

    pred_path = f'./preds/{args.benchmark}'
    os.makedirs(pred_path, exist_ok=True)

    batch_size = 1
    for i in tqdm(range(0, len(msgs), batch_size)):
        batch_msg = msgs[i:i+batch_size]
        
        batch_input = []
        for msg in batch_msg:
            prompt = processor.apply_chat_template(
                msg,
                tokenize=False,
                add_generation_prompt=True,
            )

            image_inputs, _ = process_vision_info(msg)

            batch_input.append({
                "prompt": prompt,
                "multi_modal_data": {"image": image_inputs},
            })

        output_texts = llm.generate(batch_input,
            sampling_params=sampling_params,
        )

        with open(f'{pred_path}/{args.model}_{args.method}_{args.idx}-{args.topk}.jsonl', 'a') as fout:
            for item, pred, msg in zip(results[i:i+batch_size], output_texts, batch_msg):
                pred_text = pred.outputs[0].text
                
                formattag = None
                if "evidence" in args.method:
                    formattag = "evidence"
                elif args.model == "mimo":
                    formattag = "mimo"

                item["pred"] = pred_text
                obj_str = json.dumps(item)
                fout.write(obj_str + '\n')