# Adapted from Tevatron (https://github.com/texttron/tevatron)

import os
from functools import lru_cache
from typing import Callable, Dict, List, Union

import datasets
from datasets import Dataset, load_dataset
from torch.utils.data import Dataset, IterableDataset
from transformers import AutoProcessor, PreTrainedTokenizer, ProcessorMixin
from transformers.trainer_pt_utils import IterableDatasetShard

from ..arguments import DataArguments
from ..utils import fill_template, find_all_markers

import base64
from PIL import Image
from io import BytesIO

import logging

logger = logging.getLogger(__name__)


def get_idx(obj):
    example_id = obj.get("_id", None)
    if example_id is None:
        example_id = obj.get("id", None)
    if example_id is None:
        example_id = obj.get("text_id", None)
    if example_id is None:
        example_id = obj.get("sample_id", None)
    if example_id is None:
        example_id = obj.get("filename", None) # for multimodal dataset
    if example_id is None:
        example_id = obj.get("corpus-id", None)
    if example_id is None:
        example_id = obj.get("query-id", None)
    if example_id is None:
        raise ValueError("No id field found in data, tried `_id`, `id`, `text_id`")
    example_id = str(example_id) if example_id is not None else None
    return example_id


def convert_base64string_to_image(base64_string: str):
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    image = image.convert("RGB")
    return image

def convert_raw_bytes_to_image(raw_bytes: bytes):
    image = Image.open(BytesIO(raw_bytes))
    image = image.convert("RGB")
    return image


class InferenceDataset:
    def __init__(
        self,
        data_files: Union[str, List[str]],
        data: List[Dict] = None, # in case you want to use a list of dictionaries instead of a file path
        max_len: int = 128,
        template: str = None,
        tokenizer: PreTrainedTokenizer = None,
        processor: ProcessorMixin = None,
        full_tokenization: bool = True,
        mode: str = "processed",
        batch_size: int = 1,
        num_processes: int = 1,
        process_index: int = 0,
        filter_fn: Callable = lambda x: True,
        cache_dir: str = None,
        content: str = None
    ):
        self.cache_dir = cache_dir
        self.data_files = data_files
        self.data = data
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_len = max_len
        self.template = template
        self.full_tokenization = full_tokenization
        self.dataset_repo = [
            'openbmb/VisRAG-Ret-Test-ArxivQA',
            'openbmb/VisRAG-Ret-Test-ChartQA',
            'openbmb/VisRAG-Ret-Test-MP-DocVQA',
            'openbmb/VisRAG-Ret-Test-InfoVQA',
            'openbmb/VisRAG-Ret-Test-PlotQA',
            'openbmb/VisRAG-Ret-Test-SlideVQA'
        ]                     
        self.content = content
        
        modes = [
            "raw", # for inference is fine
            "dict_processed", 
            "processed",
            "multimodal" # this is multimodal minicpmv style inference dataset
        ]
        
        if mode not in modes:
            raise ValueError(f"mode must be one of {modes}")

        self.mode = mode
        self.batch_size = batch_size
        self.num_processes = num_processes # this is for shard 
        self.process_index = process_index # this is for shard
        self.filter_fn = filter_fn
        self._prepare_data()

        if self.template is None:
            self.all_markers = None
        else:
            self.all_markers = (
                find_all_markers(self.template)
            )

    def _prepare_data(self):
        raise NotImplementedError

    @classmethod
    def load(
        cls,
        data_args: DataArguments = None,
        data: List[Dict] = None,
        data_files: Union[str, List[str]] = None,
        max_len: int = 128,
        template: str = None,
        # column_names: str = None,
        tokenizer: PreTrainedTokenizer = None,
        processor: ProcessorMixin = None,
        # is_query: bool = False,
        full_tokenization: bool = True,
        mode: str = "processed",
        stream: bool = True,
        batch_size: int = 1,
        num_processes: int = 1,
        process_index: int = 0,
        filter_fn: Callable = lambda x: True,
        cache_dir: str = None,
        content: str = None
    ):
        if data is not None:
            return StreamInMemoryDataset(
                tokenizer=tokenizer,
                processor=processor,
                data_files=data_files,
                max_len=max_len,
                template=template,
                full_tokenization=full_tokenization,
                mode=mode,
                batch_size=batch_size,
                num_processes=num_processes,
                process_index=process_index,
                filter_fn=filter_fn,
                cache_dir=cache_dir,
            )
        else:
            pass # load from file
            
        if data_files is not None:
            data_files = [data_files] if isinstance(data_files, str) else data_files
        else:
            raise ValueError("no data_files provided")
        
        ext = os.path.splitext(data_files[0])[1]
        
        ext_to_cls = {
            ".parquet": StreamParquetDataset if stream else MappingParquetDataset,
            ".tsv": StreamTsvDataset if stream else MappingTsvDataset,
            ".txt": StreamTsvDataset if stream else MappingTsvDataset,
        }
        
        cls_ = ext_to_cls.get(ext, None) if ext != "" else StreamImageDataset

        if (data_args.from_hf_repo):
            cls_ = StreamParquetDataset
        
        if cls_ is None:
            raise ValueError("Unsupported dataset file extension {}".format(ext))
        return cls_(
            tokenizer=tokenizer,
            processor=processor,
            data_files=data_files,
            max_len=max_len,
            template=template,
            full_tokenization=full_tokenization,
            mode=mode,
            batch_size=batch_size,
            num_processes=num_processes, # this is for StreamingInferenceDataset Sharding
            process_index=process_index, # this is for StreamingInferenceDataset Sharding
            filter_fn=filter_fn,
            cache_dir=cache_dir,
            content=content
        )

    def _tokenize(self, example: str):
        return self.tokenizer(
            example,
            add_special_tokens=self.full_tokenization,
            padding="max_length" if self.full_tokenization else False,
            truncation=True,
            max_length=self.max_len,
            return_attention_mask=self.full_tokenization,
            return_token_type_ids=False,
        )

    def process_one(self, example):
        if self.mode == "raw":
            return example
        
        elif self.mode == "dict_processed":
            example_id = get_idx(example)
            tokenized = {}
            for marker in self.all_markers:
                tokenized[marker] = (
                    dict(self._tokenize(example[marker]))
                    if (marker in example and example[marker] is not None)
                    else None
                )
            return {"text_id": example_id, **tokenized}
        
        elif self.mode == "multimodal":
            # this is for multimodal
            
            example_id = get_idx(example)
            
            text = fill_template(
                self.template, example, self.all_markers, allow_not_found=True
            )
            
            image = None
            
            # new version
            if "image" in example:
                if isinstance(example["image"], Image.Image): 
                    image = example["image"].convert("RGB")
                else:
                    image = convert_raw_bytes_to_image(example["image"]['image-bytes'])
            
            
            # bug fix
            if image is None:
                if text == "":
                    text = "empty document"
            
            output_dict = {
                "id": example_id,
                "text": text,
                "image": image,
            }

            return output_dict
            
        
        else:
            example_id = get_idx(example)
            full_text = fill_template(
                self.template, example, self.all_markers, allow_not_found=True
            )
            
            tokenized = self._tokenize(full_text)
            return {"text_id": example_id, **tokenized}


class StreamInferenceDataset(IterableDataset):
    def __iter__(self):

        real_batch_size = self.batch_size * self.num_processes
        process_slice = range(
            self.process_index * self.batch_size, (self.process_index + 1) * self.batch_size
        )

        current_batch = []
        for element in self.dataset:
            current_batch.append(element)
            if len(current_batch) == real_batch_size:
                for i in process_slice:
                    yield self.process_one(current_batch[i])
                current_batch = []

        if len(current_batch) > 0:
            for i in process_slice:
                if i < len(current_batch):
                    yield self.process_one(current_batch[i])


class MappingInferenceDataset(Dataset):
    @lru_cache(maxsize=None)
    def __getitem__(self, index):
        return self.process_one(self.dataset[index])

    def get_raw(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class StreamParquetDataset(StreamInferenceDataset, InferenceDataset):
    def _prepare_data(self):
        
        if (self.data_files[0] in self.dataset_repo):
            logger.info(f"Loading dataset from HuggingFace repo.")
            self.dataset = load_dataset(
                self.data_files[0], self.content, streaming=True, cache_dir=self.cache_dir
            )["train"]
            logger.info(f"Dataset loaded from HuggingFace repo.")
        else:
            logger.info(f"Loading dataset from local file.")
            self.dataset = load_dataset(
                "parquet", data_files=self.data_files, streaming=True, cache_dir=self.cache_dir
            )["train"]
            logger.info(f"Dataset loaded from local file.")
        
        if (self.filter_fn is not None):
            self.dataset = self.dataset.filter(self.filter_fn)
            print("filtered dataset ok...")
        
        sample = list(self.dataset.take(1))[0]
        
        self.all_columns = sample.keys()


class MappingParquetDataset(MappingInferenceDataset, InferenceDataset):
    def _prepare_data(self):
        
        if self.filter_fn is None:
            hf_dataset = load_dataset(
                "parquet", data_files=self.data_files, streaming=True, cache_dir=self.cache_dir
            )["train"].filter(self.filter_fn)
        else:
            self.dataset = load_dataset(
                "parquet", data_files=self.data_files, streaming=True, cache_dir=self.cache_dir
            )["train"].filter(self.filter_fn)
        
        sample = list(hf_dataset.take(1))[0]
        self.all_columns = sample.keys()
        self.dataset = {}
        for item in hf_dataset:
            self.dataset[get_idx(item)] = item


class StreamTsvDataset(StreamInferenceDataset, InferenceDataset):
    def _prepare_data(self):
        if self.all_columns is not None:
            self.all_columns = self.all_columns.split(",")
        self.dataset = load_dataset(
            "csv",
            data_files=self.data_files,
            streaming=True,
            delimiter="\t",
            cache_dir=self.cache_dir,
        )["train"].filter(self.filter_fn)


class MappingTsvDataset(MappingInferenceDataset, InferenceDataset):
    def _prepare_data(self):
        if self.all_columns is not None:
            self.all_columns = self.all_columns.split(",")
        hf_dataset = load_dataset(
            "csv",
            data_files=self.data_files,
            streaming=True,
            delimiter="\t",
            cache_dir=self.cache_dir,
        )["train"].filter(self.filter_fn)
        self.dataset = {}
        for item in hf_dataset:
            self.dataset[get_idx(item)] = item


class StreamImageDataset(StreamInferenceDataset, InferenceDataset):
    def _prepare_data(self):
        self.is_image = True
        self.dataset = load_dataset(
            self.data_files[0],
            split="train",
            streaming=True,
        )
        self.dataset = self.dataset.cast_column("image", datasets.Image(decode=False))


class StreamInMemoryDataset(StreamInferenceDataset, InferenceDataset):
    def _prepare_data(self):
        self.dataset = Dataset.from_list(self.data).filter(self.filter_fn)
        sample = self.dataset[0]
        self.all_columns = sample.keys()

