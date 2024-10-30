# Adapted from Tevatron (https://github.com/texttron/tevatron)

import logging
import os
import sys
import json

from transformers import AutoTokenizer, HfArgumentParser, set_seed

from openmatch.arguments import DataArguments
from openmatch.arguments import DRTrainingArguments as TrainingArguments
from openmatch.arguments import ModelArguments
from openmatch.dataset import MappingMMDRTrainDataset, StreamMMDRTrainDataset, MMQPCollator
from openmatch.modeling import DRModel
from openmatch.trainer import DRTrainer as Trainer

logger = logging.getLogger(__name__)



def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments
        print(model_args, data_args, training_args)

    if (model_args.dtype == 'float16'):
        training_args.fp16 = True
        training_args.fp16_full_eval = True
    elif (model_args.dtype == 'bfloat16'):
        training_args.bf16 = True
        training_args.bf16_full_eval = True

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    print(f"training_args.local_rank = {training_args.local_rank}")
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        True if model_args.dtype == 'float16' or model_args.dtype == 'bfloat16' else False
    )
    
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    config_json = json.load(open(os.path.join(model_args.model_name_or_path, 'config.json')))

    assert "_name_or_path" in config_json or "model_name_or_path" in config_json, "building model will need to determine the modeling file, please make sure _name_or_path or model_name_or_path is in the config.json"
    if "_name_or_path" in config_json:
        name = config_json["_name_or_path"]
    else:
        name = config_json["model_name_or_path"]
    
    if "siglip" in name or "SigLIP" in name:
        from openmatch.modeling.modeling_siglip.tokenization_siglip import SiglipTokenizer as tokenizer_cls
    elif "CPM-2B" in name:
        from transformers import AutoTokenizer as tokenizer_cls
    elif "MiniCPM-V-2" in name or "VisRAG" in name:
        from openmatch.modeling.modeling_minicpmv.modeling_minicpmv import LlamaTokenizerWrapper as tokenizer_cls
    else:
        tokenizer_cls = AutoTokenizer

    # ----- tokenizer -----
    tokenizer = tokenizer_cls.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False, 
    )
    
    # ----- model ----
    model = DRModel.build(
        model_args=model_args,
        data_args=data_args,
        train_args=training_args,
        cache_dir=model_args.cache_dir
    )
        
    train_dataset_cls = (
        MappingMMDRTrainDataset if training_args.use_mapping_dataset else StreamMMDRTrainDataset
    )
    
    train_dataset = train_dataset_cls(
        tokenizer,
        data_args,
        cache_dir=data_args.data_cache_dir or model_args.cache_dir,
    )
    
    logger.info(f"DataArgs: {data_args}")
    
    eval_dataset = (
        train_dataset_cls(
            tokenizer,
            data_args,
            is_eval=True,
            cache_dir=data_args.data_cache_dir or model_args.cache_dir,
        )
        if data_args.eval_path is not None
        else None
    )

    data_collator = MMQPCollator()
    
    trainer_cls = Trainer
    
    trainer = trainer_cls(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        model_name_or_path=model_args.model_name_or_path,
    )
    train_dataset.trainer = trainer

    trainer.train()
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
