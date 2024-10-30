# Adapted from Tevatron (https://github.com/texttron/tevatron)

import glob
import logging
import os
import random
from typing import List
import json
from datasets import load_dataset
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizer
from ..arguments import DataArguments
from ..trainer import DRTrainer
import base64
from PIL import Image
from io import BytesIO

logger = logging.getLogger(__name__)


class TrainDatasetBase:
    """
    Abstract base class for all train datasets in Openmatch.\n
    This implants arguments and data preparation, but should be mostly used for identifying an OpenMatch Train Dataset.\n
    All future dataset ABCs would subclass this and `(Iterable)Dataset`.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_args: DataArguments,
        trainer: DRTrainer = None,
        is_eval: bool = False,
        shuffle_seed: int = None,
        cache_dir: str = None
    ) -> None:
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.q_max_len = data_args.q_max_len
        self.p_max_len = data_args.p_max_len
        self.trainer = trainer
        self.is_eval = is_eval
        self.from_hf_repo = data_args.from_hf_repo
        self.dataset_repo = ['openbmb/VisRAG-Ret-Train-In-domain-data', 'openbmb/VisRAG-Ret-Train-Synthetic-data']
        self._prepare_data(data_args, shuffle_seed, cache_dir)

    def _prepare_data(self, data_args, shuffle_seed, cache_dir):
        if not self.is_eval:
            self.data_files = (
                [data_args.train_path]
                if data_args.train_dir is None
                else [data_args.train_dir] if self.from_hf_repo else glob.glob(os.path.join(data_args.train_dir, "*.parquet")) 
            )

            if not self.from_hf_repo:
                if len(self.data_files) == 0:
                    raise FileNotFoundError(f"Cannot find any parquet files in {data_args.train_dir}")
        else:
            self.data_files = [data_args.eval_path]

    def get_process_fn(self, epoch, hashed_seed):
        raise NotImplementedError


class StreamTrainDatasetMixin(IterableDataset):
    def _prepare_data(self, data_args, shuffle_seed, cache_dir):
        super()._prepare_data(data_args, shuffle_seed, cache_dir)
        if self.from_hf_repo:
            logger.info("Loading dataset from HuggingFace repo.")
            self.dataset = load_dataset(
                self.data_files[0], streaming=True, cache_dir=cache_dir
            )["train"]
            logger.info(f"Dataset loaded from HuggingFace repo.")
        else:
            logger.info("load dataset from local files...")
            self.dataset = load_dataset(
                "parquet", data_files=self.data_files, streaming=True, cache_dir=cache_dir
            )["train"]
            logger.info(f"loaded dataset from local files.")

        sample = list(self.dataset.take(1))[0]
        self.all_columns = sample.keys()

    def __len__(self):
        if self.data_args.train_dir is not None:
            if (self.from_hf_repo):
                logger.info("Loading dataset length from HuggingFace repo.")
                return self.dataset.info.splits['train'].num_examples
                logger.info(f"Dataset length loaded from HuggingFace repo.")
            else:
                logger.info("load dataset length from metadata...")
                metadata_path = os.path.join(self.data_args.train_dir, "metadata.json")
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.loads(f.read())
                        length = int(metadata["length"])
                        logger.info(f"loaded metadata, using length = {length}")
                        return length
                except Exception as e:
                    logger.warning(e)
                    logger.info(f"failed to load metadata, maybe the metadata file is missing or 'length' field is not found.")
                    raise Exception("Please provide the metadata file with 'length' field or use huggingface datasets.")

    def __iter__(self):
        return iter(self.dataset.map(self.get_process_fn(0, None), remove_columns=self.all_columns))


class MappingTrainDatasetMixin(Dataset):
    def _prepare_data(self, data_args, shuffle_seed, cache_dir):
        super()._prepare_data(data_args, shuffle_seed, cache_dir)
        self.dataset = load_dataset(
            "json", data_files=self.data_files, streaming=False, cache_dir=cache_dir
        )["train"]
        
        sample = self.dataset[0]
        self.all_columns = sample.keys()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        group = self.dataset[index]
        return self.get_process_fn(0, None)(group)



# For multimodal Dense Retrieval Model
class MMDRTrainDataset(TrainDatasetBase):
    def convert_base64string_to_image(self, base64_string):
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))
        image = image.convert("RGB")
        return image
    
    def get_process_fn(self, epoch, hashed_seed):
        def process_fn(example):
            # here we don't do tokenization.
            
            # example format:
            # {
            #     "query": string,
            #     "image":{
            #         "bytes": bytes,
            #         'path': string
            #     }
            # }

            query = example["query"]
            
            if "image" in example:
                if isinstance(example['image'], Image.Image):
                    pos = example["image"].convert("RGB")
                else:
                    pos = Image.open(BytesIO(example['image']['bytes'])).convert("RGB")
                passages = [{'text': '', 'image': pos, 'instruction': ''}]
            else:
                pos = example['text']
                passages = [{'text': pos, 'image': None, 'instruction': ''}]
            
            query = 'Represent this query for retrieving relevant documents: ' + query
                   
            query_ = [{'text': query, 'image': None, 'instruction': ''}]
            
            return {"query_": query_, "passages": passages}

        return process_fn

class StreamMMDRTrainDataset(StreamTrainDatasetMixin, MMDRTrainDataset):
    pass

class MappingMMDRTrainDataset(MappingTrainDatasetMixin, MMDRTrainDataset):
    pass


