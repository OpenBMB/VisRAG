import os
import re
import sys
import json
import numpy as np
import pandas as pd
from collections import Counter
import string
import time, sys
from tqdm import tqdm
from collections import defaultdict
import csv
import argparse
# from torchmetrics.text.bert import BERTScore
# from transformers import AutoTokenizer

"""
# ChartVQA InfoVQA SlideVQA DocVQA ViDoSeek

# baselines: Vision-R1 Ocean-R1 ThinkLite MM-Eureka OpenVLThinker
# qwen7b_top3 qwen7b_baseline evidence_cot evidence_grpo middle evidence_json evidence_cotv2 evidence_cotv2_json middle2hard

v3_prompt bestModel
baselineGRPO_v3_prompt_100
# qatype: 0-可以回答 1-无法回答 2-整体 qwen7b_v2_prompt_0 qwen7b_v3_prompt_111
python src/eval.py --benchmark DocVQA --model_tag qwen7b_baseline_0-3
"""


parser = argparse.ArgumentParser()
parser.add_argument("--benchmark", type=str, default="ChartQA")  # benchmark 选择
parser.add_argument("--model_tag", type=str, default="MM-Eureka")  # model 选择

args = parser.parse_args()

# 计算BERTScore
# bertscore = BERTScore(model_name_or_path="../models/roberta-large", device="cuda", truncation=True)

def normalize_answer_qa(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.strip().split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))



def evaluate_predictions(output_info, labeled_answer, mode='qa'):
    final_metric = {"is_valid_answer": False, "acc": 0, "em": 0, "f1": 0, 'math_equal': 0, 'bertsc': 0, 'hallucination': 0}

    pred_answer = output_info
    if mode == 'qa':
        normalized_pred_answer = normalize_answer_qa(pred_answer)
        for answer in labeled_answer:
            normalized_ground_truth = normalize_answer_qa(answer)
            em = int(normalized_pred_answer == normalized_ground_truth)
            # acc1 = int(normalized_ground_truth in normalized_pred_answer) # 顺序也一样有些问题

            normalized_gt_set = set(normalized_ground_truth.split())
            normalized_pred_set = set(normalized_pred_answer.split())
            acc = int(normalized_gt_set.issubset(normalized_pred_set))
            # if acc == 1:
            #     evidence = normalized_pred_answer.split('<evidence>')[1].split('</evidence>')[0].strip()
            #     for i in range(3):
            #         front = f"[{i+1}]:"
            #         back = f"[{i+2}]"
            #         evi = evidence.split(front)[1].split(back)[0].strip()
            #         if evi != "no relevant information":
            #             print()

            bertsc = 0
            # bertsc = bertscore([pred_answer], [normalized_ground_truth])['f1'].item()

            prediction_tokens = normalized_pred_answer.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)

            hallucination = 1
            if normalized_ground_truth == "no relevant information" or normalized_pred_answer == "no relevant information": 
                if normalized_ground_truth != normalized_pred_answer:
                    hallucination = 0

            f1 = (2 * precision * recall) / (precision + recall + 1e-7)
            for k in ["em", "acc", "f1", "bertsc", "hallucination"]:
                final_metric[k] = max(eval(k), final_metric[k])

    # with open('./evals/test.csv', 'a') as fout:
    #     fout.write('\t'.join([pred_answer, answer] + [f"{k}:{v}" for k, v in final_metric.items()]) + '\n')
    return final_metric, pred_answer


trn_set = set()
def run_evaluation(output_list, labeled_answer_list, is_sufficient_list):
    # Existing evaluation for other datasets
    unsuff_em = []
    issuff_em, issuff_acc, issuff_f1 = [], [], []
    global_em, global_acc, global_f1 = [], [], []
    # avg_em, avg_acc, avg_f1, avg_hallucination, avg_bertscore = [], [], [], [], []


    i = 0
    for output_info, labeled_answer, is_sufficient in tqdm(zip(output_list, labeled_answer_list, is_sufficient_list)):
        
        if type(output_info) == str:
            if output_info.find("<answer>") != -1:
                output_info = output_info.split("<answer>")[1].split('</answer>')[0]
            output_text = output_info
        else:
            output_text = output_info.outputs[0].text
        metric, pred_answer = evaluate_predictions(output_text, labeled_answer)

        if metric['acc'] == 1.0:
            trn_set.add(i)
        i += 1

        # Compute overall metrics
        global_em.append(metric['em'])
        global_acc.append(metric['acc'])
        global_f1.append(metric['f1'])
        if is_sufficient:
            issuff_em.append(metric['em'])
            issuff_acc.append(metric['acc'])
            issuff_f1.append(metric['f1'])
        else:
            unsuff_em.append(metric['em'])

    overall_results = {
        'global_em': np.mean(global_em) if len(global_em) > 0 else 0.0,
        'global_acc': np.mean(global_acc) if len(global_acc) > 0 else 0.0,
        'global_f1': np.mean(global_f1) if len(global_f1) > 0 else 0.0,
        'issuff_em': np.mean(issuff_em) if len(issuff_em) > 0 else 0.0,
        'issuff_acc': np.mean(issuff_acc) if len(issuff_acc) > 0 else 0.0,
        'issuff_f1': np.mean(issuff_f1) if len(issuff_f1) > 0 else 0.0,
        'unsuff_em': np.mean(unsuff_em) if len(unsuff_em) > 0 else 0.0,
        'cnt_global': len(global_em),
        'cnt_issuff': len(issuff_em),
        'cnt_unsuff': len(unsuff_em)
    }

    print(overall_results)
    return overall_results



def get_file_paths(root_dir, file_type):
    fpaths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(file_type):
                fpaths.append(os.path.join(dirpath, filename))
    return fpaths



qid2info = {}
data_path = './data'
with open(f'{data_path}/EVisRAG-Test-{args.benchmark}/top3_test.jsonl', 'r') as fin:
    for line in fin:
        item = json.loads(line.strip())
        qid, ans, is_sufficient = item["qid"], item["answer"], item["is_sufficient"]
        qid2info[qid] = (ans, is_sufficient)


preds_list, labeled_answer_list, is_sufficient_list = [], [], []
with open(f'./preds/{args.benchmark}/{args.model_tag}.jsonl', 'r') as fin:
    for line in fin:
        item = json.loads(line.strip())
        qid, pred = item["qid"], item["pred"]
        if qid not in qid2info:
            print(f"{qid} not have information!!!")
            sys.exit(1)

        ans, is_sufficient = qid2info[qid]
        if not is_sufficient: # 拒绝回答
            ans = ["no relevant information", "insufficient to answer", "insufficient to answer the question"]
        preds_list.append(pred)
        labeled_answer_list.append(ans)
        is_sufficient_list.append(is_sufficient)

overall_results = run_evaluation(preds_list, labeled_answer_list, is_sufficient_list)