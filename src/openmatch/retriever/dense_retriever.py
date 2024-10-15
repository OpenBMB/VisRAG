import gc
import glob
import logging
import os
import pickle
import numpy as np
import torch
from ..arguments import InferenceArguments as EncodingArguments

logger = logging.getLogger(__name__)


def _retrieve_one_shard(
    corpus_shard_path: str,
    encoded_queries_tensor: torch.Tensor,
    topk: int,
    device: str,
):
    
    with open(corpus_shard_path, "rb") as f:
        data = pickle.load(f)
    encoded_corpus = data[0]
    corpus_lookup_indices = data[1]

    encoded_corpus_tensor = torch.tensor(encoded_corpus, device=device)
    
    # compute the inner product of the query and corpus
    scores = torch.matmul(encoded_queries_tensor, encoded_corpus_tensor.T)

    topk_scores, topk_indices = torch.topk(scores, topk, dim=1)

    del encoded_corpus, encoded_corpus_tensor, scores
    gc.collect()
    return topk_scores.clone(), topk_indices.clone(), corpus_lookup_indices # return the cloned tensor, then the inner tensor will be destroyed


def distributed_parallel_retrieve(
    args: EncodingArguments,
    topk: int,
):
    with torch.no_grad():
        
        # step1: this process only load its own sharded queriess
        encoded_queries = []
        query_lookup = []
        
        # use glob to list all the partitions like embeddings.query.{process_index}: (belong to this process)
        query_all_partitions = glob.glob(
            os.path.join(args.output_dir, f"embeddings.query.rank.{args.process_index}*")
        )

        logger.info(f"query_all_partitions = {query_all_partitions}")
        
        for part in query_all_partitions:
            with open(part, "rb") as f:
                data = pickle.load(f)
            query_lookup_indices = data[1]
            if len(query_lookup_indices) == 0:
                continue
            encoded_queries.append(data[0])
            query_lookup.extend(query_lookup_indices)
        encoded_queries_all = np.concatenate(encoded_queries)
        
        encoded_queries_tensor = torch.tensor(encoded_queries_all, device=args.device)
        
        # step2: iterate corpus partitions
        corpus_all_partitions = glob.glob(
            os.path.join(args.output_dir, "embeddings.corpus.rank.*")
        )
        if len(corpus_all_partitions) == 0:
            raise ValueError("No pre-computed document embeddings found")

        logger.info(f"corpus_all_partitions = {corpus_all_partitions}")
        
        cur_result = {}
        for qid in query_lookup:
            cur_result[qid] = {}
        
        for i, part in enumerate(corpus_all_partitions):
            topk_scores, topk_indices, corpus_lookup_indices = _retrieve_one_shard(
                corpus_shard_path=part, 
                encoded_queries_tensor=encoded_queries_tensor, 
                topk=topk,
                device=args.device
            )
            
            # update the cur_result by topk_scores and topk_indices:
            for q in range(topk_scores.shape[0]):
                qid = query_lookup[q]
                for idx, score in zip(topk_indices[q], topk_scores[q]):
                    idx = corpus_lookup_indices[idx.item()]
                    cur_result[qid][idx] = score.item()
            
            del topk_scores, topk_indices, corpus_lookup_indices
            gc.collect()
        
        return cur_result
    

def distributed_parallel_self_retrieve(
    args: EncodingArguments,
    topk: int,
):
    with torch.no_grad():
        
        # step1: this process only load its own sharded queriess
        encoded_queries = []
        query_lookup = []
        
        # use glob to list all the partitions like embeddings.query.{process_index}: (belong to this process)
        query_all_partitions = glob.glob(
            os.path.join(args.output_dir, f"embeddings.query.rank.{args.process_index}*")
        )
        
        logger.info(f"query_all_partitions = {query_all_partitions}")
        
        for part in query_all_partitions:
            with open(part, "rb") as f:
                data = pickle.load(f)
            query_lookup_indices = data[1]
            if len(query_lookup_indices) == 0:
                continue
            encoded_queries.append(data[0])
            query_lookup.extend(query_lookup_indices)
        
        encoded_queries_all = np.concatenate(encoded_queries)

        encoded_queries_tensor = torch.tensor(encoded_queries_all, device=args.device)
    
        # step2: iterate corpus partitions
        corpus_all_partitions = glob.glob(
            os.path.join(args.output_dir, "embeddings.query.rank.*")
        )
        if len(corpus_all_partitions) == 0:
            raise ValueError("No pre-computed query embeddings found")

        logger.info(f"corpus(in fact query itself)_all_partitions = {corpus_all_partitions}")
        
        cur_result = {}
        for qid in query_lookup:
            cur_result[qid] = {}
        
        for i, part in enumerate(corpus_all_partitions):
            topk_scores, topk_indices, corpus_lookup_indices = _retrieve_one_shard(
                corpus_shard_path=part, 
                encoded_queries_tensor=encoded_queries_tensor, 
                topk=topk,
                device=args.device
            )
            
            # update the cur_result by topk_scores and topk_indices:
            for q in range(topk_scores.shape[0]):
                qid = query_lookup[q]
                for idx, score in zip(topk_indices[q], topk_scores[q]):
                    idx = corpus_lookup_indices[idx.item()]
                    cur_result[qid][idx] = score.item()
            
            del topk_scores, topk_indices, corpus_lookup_indices
            gc.collect()
            
        return cur_result
    
