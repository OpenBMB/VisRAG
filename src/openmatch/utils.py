# Adapted from Tevatron (https://github.com/texttron/tevatron)

import csv
import json
import pathlib
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List

import datasets
import torch
from transformers import PreTrainedTokenizer

try:
    raise ModuleNotFoundError
except ModuleNotFoundError:
    _opendelta_available = False


@dataclass
class SimpleTrainPreProcessor:
    query_file: str
    collection_file: str
    tokenizer: PreTrainedTokenizer

    doc_max_len: int = 128
    query_max_len: int = 32
    columns = ["text_id", "title", "text"]
    title_field = "title"
    text_field = "text"
    query_field = "text"
    doc_template: str = None
    query_template: str = None
    allow_not_found: bool = False

    def __post_init__(self):
        self.queries = self.read_queries(self.query_file)
        self.collection = datasets.load_dataset(
            "csv",
            data_files=self.collection_file,
            column_names=self.columns,
            delimiter="\t",
        )["train"]

    @staticmethod
    def read_queries(queries):
        qmap = {}
        with open(queries) as f:
            for l in f:
                qid, qry = l.strip().split("\t")
                qmap[qid] = qry
        return qmap

    @staticmethod
    def read_qrel(relevance_file):
        qrel = {}
        with open(relevance_file, encoding="utf8") as f:
            tsvreader = csv.reader(f, delimiter="\t")
            for [topicid, _, docid, rel] in tsvreader:
                assert rel == "1"
                if topicid in qrel:
                    qrel[topicid].append(docid)
                else:
                    qrel[topicid] = [docid]
        return qrel

    def get_query(self, q):
        if self.query_template is None:
            query = self.queries[q]
        else:
            query = fill_template(
                self.query_template,
                data={self.query_field: self.queries[q]},
                allow_not_found=self.allow_not_found,
            )

        return query

    def get_passage(self, p):
        entry = self.collection[int(p)] if p != "None" else self.collection[0]
        title = entry[self.title_field]
        title = "" if title is None else title
        body = entry[self.text_field]
        if self.doc_template is None:
            content = title + self.tokenizer.sep_token + body
        else:
            content = fill_template(
                self.doc_template, data=entry, allow_not_found=self.allow_not_found
            )

        return content

    def process_one(self, train):
        q, pp, nn = train
        assert len(nn) == 1, "negative should be 1"
        
        train_example = {
            "query": self.get_query(q),
            "pos": [self.get_passage(p) for p in pp],
            "neg": [self.get_passage(n) for n in nn],
        }

        return json.dumps(train_example)


@dataclass
class SimpleCollectionPreProcessor:
    tokenizer: PreTrainedTokenizer
    separator: str = "\t"
    max_length: int = 128

    def process_line(self, line: str):
        xx = line.strip().split(self.separator)
        text_id, text = xx[0], xx[1:]
        text_encoded = self.tokenizer.encode(
            self.tokenizer.sep_token.join(text),
            add_special_tokens=False,
            max_length=self.max_length,
            truncation=True,
        )
        encoded = {"text_id": text_id, "text": text_encoded}
        return json.dumps(encoded)


def save_as_trec(
    rank_result: Dict[str, Dict[str, Dict[str, Any]]], output_path: str, run_id: str = "OpenMatch"
):
    """
    Save the rank result as TREC format:
    <query_id> Q0 <doc_id> <rank> <score> <run_id>
    """
    pathlib.Path("/".join(output_path.split("/")[:-1])).mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for qid in rank_result:
            # sort the results by score
            sorted_results = sorted(
                rank_result[qid].items(), key=lambda x: x[1], reverse=True
            )
            for i, (doc_id, score) in enumerate(sorted_results):
                f.write("{}\tQ0\t{}\t{}\t{}\t{}\n".format(qid, doc_id, i + 1, score, run_id))


def load_from_trec(input_path: str, as_list: bool = False, max_len_per_q: int = None):
    """
    Load the rank result from TREC format:
    <query_id> Q0 <doc_id> <rank> <score> <run_id> or
    <query_id> <doc_id> <score>
    """
    rank_result = {}
    cnt = 0
    with open(input_path, "r") as f:
        for line in f:
            content = line.strip().split('\t')
            if len(content) == 6:
                qid, _, doc_id, _, score, _ = content
            elif len(content) == 3:
                qid, doc_id, score = content
            else:
                print("will raise error", cnt)
                print("line", line)
                raise ValueError("Invalid run format")
            if not as_list:
                if qid not in rank_result:
                    rank_result[qid] = {}
                    cnt = 0
                if max_len_per_q is None or cnt < max_len_per_q:
                    rank_result[qid][doc_id] = float(score)
            else:
                if qid not in rank_result:
                    rank_result[qid] = []
                    cnt = 0
                if max_len_per_q is None or cnt < max_len_per_q:
                    rank_result[qid].append((doc_id, float(score)))
            cnt += 1
    return rank_result


def load_positives(relevance_file, threshold=1):
    qrel = {}
    with open(relevance_file, encoding="utf8") as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [topicid, _, docid, rel] in tsvreader:
            rel = int(rel)
            if rel >= threshold:
                if topicid in qrel:
                    qrel[topicid].append(docid)
                else:
                    qrel[topicid] = [docid]
    return qrel


def load_beir_positives(qrels_file, threshold=1):
    qrels = {}
    with open(qrels_file) as f:
        tsvreader = csv.DictReader(f, delimiter="\t")
        for row in tsvreader:
            qid = row["query-id"]
            pid = row["corpus-id"]
            rel = int(row["score"])
            if rel >= threshold:
                if qid in qrels:
                    qrels[qid].append(pid)
                else:
                    qrels[qid] = [pid]
    return qrels


def find_all_markers(template: str):
    """
    Find all markers' names (quoted in "<>") in a template.
    """
    markers = []
    start = 0
    while True:
        start = template.find("<", start)
        if start == -1:
            break
        end = template.find(">", start)
        if end == -1:
            break
        markers.append(template[start + 1 : end])
        start = end + 1
    return markers


def fill_template(
    template: str, data: Dict, markers: List[str] = None, allow_not_found: bool = False
):
    """
    Fill a template with data.
    """
    if markers is None:
        markers = find_all_markers(template)
    for marker in markers:
        marker_hierarchy = marker.split(".")
        found = True
        content = data
        for marker_level in marker_hierarchy:
            content = content.get(marker_level, None)
            if content is None:
                found = False
                break
        if not found:
            if allow_not_found:
                warnings.warn(
                    "Marker '{}' not found in data. Replacing it with an empty string.".format(
                        marker
                    ),
                    RuntimeWarning,
                )
                content = ""
            else:
                raise ValueError("Cannot find the marker '{}' in the data".format(marker))
        template = template.replace("<{}>".format(marker), str(content))
    return template


def merge_retrieval_results_by_score(results: List[Dict[str, Dict[str, float]]], topk: int = 100):
    """
    Merge retrieval results from multiple partitions of document embeddings and keep topk.
    """
    merged_results = {}
    for result in results:
        for qid in result:
            if qid not in merged_results:
                merged_results[qid] = {}
            for doc_id in result[qid]:
                if doc_id not in merged_results[qid]:
                    merged_results[qid][doc_id] = result[qid][doc_id]
    for qid in merged_results:
        merged_results[qid] = {
            k: v
            for k, v in sorted(merged_results[qid].items(), key=lambda x: x[1], reverse=True)[:topk]
        }
    return merged_results


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )

def eval_mrr(qrel, run, cutoff=None):
    """
    Compute MRR@cutoff manually.
    """
    mrr = 0.0
    num_ranked_q = 0
    results = {}
    for qid in qrel:
        if qid not in run:
            continue
        num_ranked_q += 1
        docid_and_score = [(docid, score) for docid, score in run[qid].items()]
        docid_and_score.sort(key=lambda x: x[1], reverse=True)
        for i, (docid, _) in enumerate(docid_and_score):
            rr = 0.0
            if cutoff is None or i < cutoff:
                if docid in qrel[qid] and qrel[qid][docid] > 0:
                    rr = 1.0 / (i + 1)
                    break
        results[qid] = rr
        mrr += rr
    mrr /= num_ranked_q
    results["all"] = mrr
    return results

def get_qrels_from_hf_repo(dataset_name):
    """
    Get qrels from Hugging Face datasets repository.
    """
    dataset = datasets.load_dataset(dataset_name, 'qrels')['train']
    qrels = {}
    for i in range(len(dataset)):
        qid = dataset[i]["query-id"]
        docid = dataset[i]["corpus-id"]
        rel = dataset[i]["score"]
        if qid in qrels:
            qrels[qid][docid] = rel
        else:
            qrels[qid] = {docid: rel}
    return qrels
