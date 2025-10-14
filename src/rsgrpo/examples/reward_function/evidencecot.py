# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Any, Dict, List
import string

# Constants for normalization
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]

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

def f1_match(response: str, ground_truth: str) -> float:
    pred = normalize_answer_qa(response)
    gt = normalize_answer_qa(ground_truth)

    pred_tokens = set(pred.split())  # 按词拆分（示例）
    true_tokens = set(gt.split())
    
    # 计算重叠
    tp = len(pred_tokens & true_tokens)
    fp = len(pred_tokens - true_tokens)
    fn = len(true_tokens - pred_tokens)

    # 处理分母为0的情况
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1

def em_match(response: str, ground_truth: str) -> float:
    if normalize_answer_qa(ground_truth) in normalize_answer_qa(response):
        return 1.0
    else:
        return 0.0

def accuracy_reward(response: str, ground_truth: str) -> float:
    if response.find('<answer>') == -1 or response.find('</answer>') == -1:
        return 0.0
    response = normalize_answer_qa(response.split('<answer>')[1].split('</answer>')[0])
    ground_truth = normalize_answer_qa(ground_truth.split('<answer>')[1].split('</answer>')[0])
    
    return f1_match(response, ground_truth)
    # words = ground_truth.split()
    # if len(words) >= 5:
    #     return f1_match(response, ground_truth)
    # else:
    #     return em_match(response, ground_truth)
        
# print(accuracy_reward("<answer>11%</answer>", "<answer>11</answer>"))

def evidence_reward(response: str, ground_truth: str) -> float:
    if response.find('<evidence>') == -1 or response.find('</evidence>') == -1:
        return 0.0

    evidence_score = 0.0
    obs = response.split('<evidence>')[1].split('</evidence>')[0]
    gt_obs = ground_truth.split('<evidence>')[1].split('</evidence>')[0]

    full = 0
    for i in range(6):
        if f"[{i+1}]:" not in gt_obs:
            if full > 0.0:
                evidence_score /= full
            break

        gold = gt_obs.split(f"[{i+1}]:")[1].split(f"[{i+2}]")[0].strip()
        n_gold = len(gold.split())

        score = 3.0 if gold != "no relevant information" else 1.0
        full += score

        if f"[{i+1}]:" not in obs:
            continue

        gen = obs.split(f"[{i+1}]:")[1].split(f"[{i+2}]")[0].strip()
        if n_gold >= 5:
            evidence_score += f1_match(gen, gold) * score
        else:
            evidence_score += em_match(gen, gold) * score

    return evidence_score

def format_reward(response: str, ground_truth: str) -> float:
    # 限定结构和顺序
    pattern = r"<observe>.*?</observe>\s*<evidence>.*?</evidence>\s*<think>.*?</think>\s*<answer>.*?</answer>"
    # 先整体匹配结构
    if not re.fullmatch(pattern, response, flags=re.DOTALL):
        return 0.0
    # 检查每个标签是否只出现一次
    if len(re.findall(r"<observe>", response)) != 1: return 0.0
    if len(re.findall(r"</observe>", response)) != 1: return 0.0
    if len(re.findall(r"<evidence>", response)) != 1: return 0.0
    if len(re.findall(r"</evidence>", response)) != 1: return 0.0
    if len(re.findall(r"<think>", response)) != 1: return 0.0
    if len(re.findall(r"</think>", response)) != 1: return 0.0
    if len(re.findall(r"<answer>", response)) != 1: return 0.0
    if len(re.findall(r"</answer>", response)) != 1: return 0.0
    
    # obs = response.split('<evidence>')[1].split('</evidence>')[0]
    # gt_obs = ground_truth.split('<evidence>')[1].split('</evidence>')[0]
    # for i in range(5):
    #     if f"[{i+1}]:" in gt_obs and f"[{i+1}]:" not in obs:
    #         return 0.0
    #     if f"[{i+1}]:" not in gt_obs and f"[{i+1}]:" in obs: 
    #         return 0.0
    
    return 1.0
    


def soft_overlong_punishment(response_length: int, max_response_length: int, overlong_buffer_length: int, min_response_length: int):
    if response_length < min_response_length:
        return -1.0
    
    expected_len = max_response_length - overlong_buffer_length
    if response_length <= expected_len:
        return 0.0
    elif response_length <= max_response_length:
        return (expected_len - response_length) / overlong_buffer_length
    else:
        return -1.0


def isObserve_punishment(response: str) -> float:
    if response.find('<observe>') == -1 or response.find('</observe>') == -1:
        return 0.0
    observe = response.split('<observe>')[1].split('</observe>')[0]
    for i in range(5):
        if f"[{i+1}]" in observe:
            return -1.0
    return 0.0

def isThink_punishment(response: str) -> float:
    if response.find('<think>') == -1 or response.find('</think>') == -1:
        return 0.0
    if response.find('<answer>') == -1 or response.find('</answer>') == -1:
        return 0.0
    think = response.split('<think>')[1].split('</think>')[0]
    answer = response.split('<answer>')[1].split('</answer>')[0]
    if think == answer:
        return -1.0

    return 0.0



def remove_evidence(
    reward_inputs: List[Dict[str, Any]]
) -> List[Dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for dapo reward function.")

    scores = []
    for reward_input in reward_inputs:
        response = reward_input["response"]  # The longest answer in MATH-500 has 159 characters
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        overlong_score = soft_overlong_punishment(
            reward_input["response_length"], 1536, 512
        )
        format_score = format_reward(response, reward_input["ground_truth"])
        # evidence_score = evidence_reward(response, reward_input["ground_truth"])
        isObserve_score = isObserve_punishment(response)
        isThink_score = isThink_punishment(response)
        # "overall": accuracy_score + overlong_score + format_score + isObserve_score + isThink_score,
        scores.append(
            {
                "overall": format_score,
                "accuracy": accuracy_score,
                "format": format_score,
                "overlong": overlong_score,
                "isObserve": isObserve_score,
                "isThink": isThink_score
            }
        )

    return scores



def sum_all(
    reward_inputs: List[Dict[str, Any]]
) -> List[Dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for dapo reward function.")

    scores = []
    for reward_input in reward_inputs:
        response = reward_input["response"]  # The longest answer in MATH-500 has 159 characters
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        overlong_score = soft_overlong_punishment(
            reward_input["response_length"], 1536, 512, 200
        )
        format_score = format_reward(response, reward_input["ground_truth"])
        evidence_score = evidence_reward(response, reward_input["ground_truth"])
        isObserve_score = isObserve_punishment(response)
        isThink_score = isThink_punishment(response)
        scores.append(
            {
                "overall": 3*accuracy_score + 3*evidence_score + overlong_score + format_score + isObserve_score + isThink_score,
                "accuracy": 3*accuracy_score,
                "evidence": 3*evidence_score,
                "format": format_score,
                "overlong": overlong_score,
                "isObserve": isObserve_score,
                "isThink": isThink_score
                # "accuracy_normalized": 0.5 * (accuracy_score + 1.0),
            }
        )

    return scores
