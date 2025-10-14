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

import importlib.util
import os
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, TypedDict

import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from .config import RewardConfig


class RewardInput(TypedDict):
    response: str
    response_length: int
    ground_truth: str


class RewardScore(TypedDict):
    overall: float
    format: Optional[float]
    accuracy: Optional[float]


SequentialRewardFunction = Callable[[RewardInput], RewardScore]

BatchRewardFunction = Callable[[List[RewardInput]], List[RewardScore]]


class FunctionRewardManager(ABC):
    """Reward manager for rule-based reward."""

    def __init__(self, config: RewardConfig, tokenizer: PreTrainedTokenizer):
        if config.reward_function is None:
            raise ValueError("Reward function is not provided.")

        if not os.path.exists(config.reward_function):
            raise FileNotFoundError(f"Reward function file {config.reward_function} not found.")

        spec = importlib.util.spec_from_file_location("custom_reward_fn", config.reward_function)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_reward_fn"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load reward function: {e}")

        if not hasattr(module, config.reward_function_name):
            raise AttributeError(f"Module {module} does not have function {config.reward_function_name}.")

        reward_fn = getattr(module, config.reward_function_name)
        print(f"Using reward function `{config.reward_function_name}` from `{config.reward_function}`.")
        self.reward_fn = partial(reward_fn, **config.reward_function_kwargs)
        self.config = config
        self.tokenizer = tokenizer

    @abstractmethod
    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        """Compute reward for a batch of data."""
        ...


class SequentialFunctionRewardManager(FunctionRewardManager):
    reward_fn: SequentialRewardFunction

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)
        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            score = self.reward_fn(
                {
                    "response": response_str,
                    "response_length": cur_response_length,
                    "ground_truth": data.non_tensor_batch["ground_truth"][i],
                }
            )
            reward_tensor[i, cur_response_length - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics


class BatchFunctionRewardManager(FunctionRewardManager):
    reward_fn: BatchRewardFunction

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        """
        return reward_tensor(bs, n_rewards), reward_masks(bs, n_rewards, seq_len)
        """
        reward_fn_list = ["accuracy", "evidence", "format", "overlong", "isObserve", "isThink"]
        # reward_fn_list = ["format"]
        n_rewards = len(reward_fn_list)
        
        reward_mask_tokens = []
        for rfc in reward_fn_list:
            if rfc == "accuracy":
                reward_mask_tokens.append(("<think>", "end"))
            elif rfc == "evidence":
                reward_mask_tokens.append(("start", "<think>"))
            elif rfc == "format":
                reward_mask_tokens.append(("start", "end"))
            elif rfc == "overlong":
                reward_mask_tokens.append(("start", "end"))
            elif rfc == "isObserve":
                reward_mask_tokens.append(("start", "<evidence>"))
            elif rfc == "isThink":
                reward_mask_tokens.append(("<think>", "<answer>"))

        def find_first_subsequence(a: torch.Tensor, sub: torch.Tensor):
            """
            在a（一维tensor）中滑窗查找sub，返回第一个完全匹配的起始位置（没有则返回a.shape[0]）
            """
            n = a.shape[0]
            m = sub.shape[0]
            if m > n:
                return -1
            # 滑窗比较
            for i in range(n - m + 1):
                if torch.equal(a[i:i + m], sub):
                    return i
            return -1
        
        
        reward_inputs = []
        response_ids = data.batch["responses"]
        bts, seq_len = response_ids.shape

        response_length = torch.sum(data.batch["response_mask"], dim=-1)

        reward_masks = data.batch["response_mask"]
        reward_masks = reward_masks.unsqueeze(1).repeat(1, n_rewards, 1)
        st_idxs = torch.full((bts,), seq_len, dtype=torch.long, device=response_ids.device)  # 默认没找到
        ed_idxs = torch.full((bts,), seq_len, dtype=torch.long, device=response_ids.device)  # 默认没找到

        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            
            for i_reward, (st_token, ed_token) in enumerate(reward_mask_tokens):
                st_mask, ed_mask = None, None
                if st_token != "start":
                    st_token_ids = self.tokenizer.encode(st_token, add_special_tokens=False)
                    st_token_ids = torch.tensor(st_token_ids, device=response_ids.device)

                    idx = find_first_subsequence(response_ids[i], st_token_ids)
                    if idx == -1: idx = 0
                    st_idxs[i] = idx

                    sequence_indices = torch.arange(seq_len, device=response_ids.device).expand(bts, seq_len)
                    st_mask = (sequence_indices >= st_idxs.unsqueeze(1)).int()
                
                if ed_token != "end":
                    ed_token_ids = self.tokenizer.encode(ed_token, add_special_tokens=False)
                    ed_token_ids = torch.tensor(ed_token_ids, device=response_ids.device)

                    idx = find_first_subsequence(response_ids[i], ed_token_ids)
                    if idx == -1: idx = seq_len
                    ed_idxs[i] = idx

                    sequence_indices = torch.arange(seq_len, device=response_ids.device).expand(bts, seq_len)
                    ed_mask = (sequence_indices < ed_idxs.unsqueeze(1)).int()

                if st_mask is not None:
                    reward_masks[:,i_reward,:] *= st_mask
                if ed_mask is not None:
                    reward_masks[:,i_reward,:] *= ed_mask

            reward_inputs.append(
                {
                    "response": response_str,
                    "response_length": cur_response_length,
                    "ground_truth": data.non_tensor_batch["ground_truth"][i],
                }
            )

        scores = self.reward_fn(reward_inputs)
        reward_tensor = torch.zeros((bts, n_rewards), dtype=torch.float32) # B, n
        
        reward_metrics = defaultdict(list)
        for i, score in enumerate(scores):
            for j, rfn in enumerate(reward_fn_list):
                # cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
                reward_tensor[i, j] = score[rfn]
    
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_masks, reward_metrics
