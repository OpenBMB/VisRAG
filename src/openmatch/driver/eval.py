import logging
import os
import sys
import glob
import csv
import json

import torch
import pytrec_eval

# from transformers import AutoConfig, AutoTokenizer, HfArgumentParser

from transformers import HfArgumentParser

from openmatch.arguments import DataArguments
from openmatch.arguments import InferenceArguments as EncodingArguments
from openmatch.arguments import ModelArguments
from openmatch.dataset import InferenceDataset
from openmatch.modeling import DRModelForInference

from openmatch.inference import distributed_parallel_embedding_inference
from openmatch.retriever import distributed_parallel_retrieve
from openmatch.utils import save_as_trec, load_from_trec, eval_mrr, get_qrels_from_hf_repo

logger = logging.getLogger(__name__)

def load_beir_qrels(qrels_file):
    qrels = {}
    with open(qrels_file) as f:
        tsvreader = csv.DictReader(f, delimiter="\t")
        for row in tsvreader:
            qid = row["query-id"]
            pid = row["corpus-id"]
            rel = int(row["score"])
            if qid in qrels:
                qrels[qid][pid] = rel
            else:
                qrels[qid] = {pid: rel}
    return qrels

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, EncodingArguments))
    
    model_args, data_args, encoding_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    encoding_args: EncodingArguments

    if (model_args.dtype == 'bfloat16'):
        raise NotImplementedError("bfloat16 is not supported yet, because it is not supported by numpy.")

    if os.path.exists(encoding_args.output_dir) and os.listdir(encoding_args.output_dir):
        if not encoding_args.overwrite_output_dir:
            logger.warning(
                f"Output directory ({encoding_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            )
        else:
            # remove all files in the output directory
            if encoding_args.local_process_index == 0:
                for file in os.listdir(encoding_args.output_dir):
                    os.remove(os.path.join(encoding_args.output_dir, file))

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if encoding_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed inference: %s, 16-bits inference: %s",
        encoding_args.local_rank,
        encoding_args.device,
        encoding_args.n_gpu,
        bool(encoding_args.local_rank != -1),
        encoding_args.fp16,
    )
    logger.info("Encoding parameters %s", encoding_args)
    logger.info("Model parameters %s", model_args)
    
    config_json = json.load(open(os.path.join(model_args.model_name_or_path, 'config.json')))
    
    if "MiniCPM-V-2.0" in config_json["_name_or_path"]:
        from openmatch.modeling.modeling_minicpmv.modeling_minicpmv import LlamaTokenizerWrapper as tokenizer_cls
    elif "CPM-2B" in config_json["_name_or_path"]:
        from transformers import AutoTokenizer as tokenizer_cls
    elif "siglip" in config_json['_name_or_path'] or "SigLIP" in config_json['_name_or_path']:
        from openmatch.modeling.modeling_siglip.tokenization_siglip import SiglipTokenizer as tokenizer_cls
    else:
        raise NotImplementedError("your model config arch is not supported")
    
    tokenizer = tokenizer_cls.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False
    )
    
    if encoding_args.phase in ["encode_corpus", "encode_query", "encode"]:
        logger.info("loading model for embedding inference")
        model = DRModelForInference.build(
            model_args=model_args,
            cache_dir=model_args.cache_dir,
        )
        model.to(encoding_args.device)
        model.eval()
    else:
        logging.info("No need to load model for retrieval")
        model = None

    # encode query
    if encoding_args.phase in ["encode_query", "encode"]:
                   
        logger.info(f"Loading queries")
        
        if (data_args.from_hf_repo):
            query_file = [data_args.query_path]
        else:
            if os.path.exists(data_args.query_path):
                query_file = [data_args.query_path]
            else:
                raise ValueError(f"--query_path {data_args.query_path} does not exist, please check.")
        
        query_dataset = InferenceDataset.load(
            tokenizer=tokenizer,
            data_args=data_args,
            data_files=query_file,
            full_tokenization=False,
            mode="multimodal",
            template=data_args.query_template,
            stream=True,
            batch_size=encoding_args.per_device_eval_batch_size,
            num_processes=encoding_args.world_size,
            process_index=encoding_args.process_index,
            filter_fn=None,
            cache_dir=data_args.data_cache_dir,
            content='queries'
        )
    
    
        logger.info("Encoding query")
        distributed_parallel_embedding_inference(
            dataset=query_dataset,
            model=model,
            args=encoding_args,
            dataset_type="query",
            split_save=False,
            model_additional_args = { # for VisRAG_Ret only
                "tokenizer": tokenizer,
                "max_inp_length": data_args.p_max_len
            }
        )
    
    # encode corpus
    if encoding_args.phase in ["encode_corpus", "encode"]:
        logger.info("Loading corpus dataset")

        if (data_args.from_hf_repo):
            corpus_file = [data_args.corpus_path]
        else:
            if not os.path.exists(data_args.corpus_path):
                raise ValueError(f"--corpus_path {data_args.corpus_path} does not exist, please check.")
            
            if os.path.isdir(data_args.corpus_path): # if it is a directory
                corpus_file = [data_args.corpus_path]
                logger.info(f"corpus_dir: {data_args.corpus_path}")
            else: # it is a file
                assert data_args.corpus_path.endswith('.parquet'), f"corpus file {data_args.corpus_path} should be a .parquet file."
                corpus_file = [data_args.corpus_path]
        
        corpus_dataset = InferenceDataset.load(
            tokenizer=tokenizer,
            data_args=data_args,
            data_files=corpus_file,
            full_tokenization=True,
            mode="multimodal",
            template=data_args.doc_template,
            stream=True,
            batch_size=encoding_args.per_device_eval_batch_size,
            num_processes=encoding_args.world_size,
            process_index=encoding_args.process_index,
            cache_dir=data_args.data_cache_dir,
            filter_fn=None,
            content='corpus'
        )
        
        logger.info("Encoding corpus")
        
        distributed_parallel_embedding_inference(
            dataset=corpus_dataset,
            model=model,
            args=encoding_args,
            dataset_type="corpus",
            split_save=True,
            model_additional_args = { # for VisRAG_Ret only
                "tokenizer": tokenizer,
                "max_inp_length": data_args.p_max_len
            }
        )
    
    # retrieve
    if encoding_args.phase == "retrieve":
        
        if data_args.from_hf_repo:
            logger.info(f"Loading qrels from HuggingFace repo.")
            qrels = get_qrels_from_hf_repo(data_args.qrels_path)
            logger.info(f"qrels load finished.")
        else:
            if os.path.exists(data_args.qrels_path): 
                qrels_path = data_args.qrels_path
                logger.info(f"Loading qrels from local file.")
                qrels = load_beir_qrels(qrels_path)
                logger.info(f"qrels load finished.")
            else:
                raise ValueError(f"--qrels_path {data_args.qrels_path} does not exist, can't proceed, please check.")
        
        logger.info("Retrieving")
        run = distributed_parallel_retrieve(args=encoding_args, topk=encoding_args.retrieve_depth)
        
        # save trec file
        if encoding_args.trec_save_path is None:
            encoding_args.trec_save_path = os.path.join(encoding_args.output_dir, f"test.{encoding_args.process_index}.trec")
        
        save_as_trec(run, encoding_args.trec_save_path)

        if encoding_args.world_size > 1:
            torch.distributed.barrier()
        
        # collect trec file and compute metric for rank = 0
        if encoding_args.process_index == 0: 
            # use glob library to to list all trec files from encoding_args.output_dir:
            partitions = glob.glob(os.path.join(encoding_args.output_dir, "test.*.trec"))
            logger.info(f"trec files: {partitions}")
            run = {}
            for part in partitions:
                print("loading", part)
                run.update(load_from_trec(part))
            
            
        
            evaluator = pytrec_eval.RelevanceEvaluator(
                qrels, {"ndcg_cut.10", "recall.10"})
            eval_results = evaluator.evaluate(run)

            def print_line(measure, scope, value):
                print("{:25s}{:8s}{:.4f}".format(measure, scope, value))
                with open(
                    os.path.join(encoding_args.output_dir, "test_result.log"), "w", encoding="utf-8"
                ) as fw:
                    fw.write("{:25s}{:8s}{:.4f}\n".format(measure, scope, value))
            
            for query_id, query_measures in sorted(eval_results.items()):
                for measure, value in sorted(query_measures.items()):
                    pass

            for measure in sorted(query_measures.keys()):
                print_line(
                    measure,
                    "all",
                    pytrec_eval.compute_aggregated_measure(
                        measure, [query_measures[measure] for query_measures in eval_results.values()]
                    ),
                )
                            
            mrr_at_10 = eval_mrr(qrels, run, 10)['all']
            print(f'MRR@10: {mrr_at_10}')
        
        if encoding_args.world_size > 1:
            torch.distributed.barrier()


if __name__ == "__main__":
    main()
