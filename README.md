# VisRAG: Vision-based Retrieval-augmented Generation on Multi-modality Documents
[![Github](https://img.shields.io/badge/VisRAG-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/OpenBMB/VisRAG)
[![Google Colab](https://img.shields.io/badge/VisRAG_Pipeline-ffffff?style=for-the-badge&logo=googlecolab&logoColor=f9ab00)](https://colab.research.google.com/drive/11KV9adDNXPfHiuFAfXNOvtYJKcyR8JZH?usp=sharing)
[![arXiv](https://img.shields.io/badge/arXiv-2410.10594-ff0000.svg?style=for-the-badge)](https://arxiv.org/abs/2410.10594)
[![Hugging Face](https://img.shields.io/badge/VisRAG_Ret-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/openbmb/VisRAG-Ret)
[![Hugging Face](https://img.shields.io/badge/VisRAG_Collection-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/collections/openbmb/visrag-6717bbfb471bb018a49f1c69)
[![Hugging Face](https://img.shields.io/badge/VisRAG_Pipeline-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/spaces/tcy6/VisRAG_Pipeline)

<p align="center">•
 <a href="#-introduction"> 📖 Introduction </a> •
 <a href="#-news">🎉 News</a> •
 <a href="#-visrag-pipeline">✨ VisRAG Pipeline</a> •
 <a href="#%EF%B8%8F-setup">⚙️ Setup</a> •
 <a href="#%EF%B8%8F-training">⚡️ Training</a> 
</p>
<p align="center">•
 <a href="#-evaluation">📃 Evaluation</a> •
 <a href="#-usage">🔧 Usage</a> •
 <a href="#-license">📄 Lisense</a> •
 <a href="#-contact">📧 Contact</a> •
 <a href="#-star-history">📈 Star History</a>
</p>

# 📖 Introduction
**VisRAG** is a novel vision-language model (VLM)-based RAG pipeline. In this pipeline, instead of first parsing the document to obtain text, the document is directly embedded using a VLM as an image and then retrieved to enhance the generation of a VLM. Compared to traditional text-based RAG, **VisRAG** maximizes the retention and utilization of the data information in the original documents, eliminating the information loss introduced during the parsing process.
<p align="center"><img width=800 src="assets/main_figure.png"/></p>

# 🎉 News

* 20241111: Released our [VisRAG Pipeline](https://github.com/OpenBMB/VisRAG/tree/master/scripts/demo/visrag_pipeline) on GitHub, now supporting visual understanding across multiple PDF documents.
* 20241104: Released our [VisRAG Pipeline](https://huggingface.co/spaces/tcy6/VisRAG_Pipeline) on Hugging Face Space.
* 20241031: Released our [VisRAG Pipeline](https://colab.research.google.com/drive/11KV9adDNXPfHiuFAfXNOvtYJKcyR8JZH?usp=sharing) on Colab. Released codes for converting files to images which could be found at `scripts/file2img`.
* 20241015: Released our train data and test data on Hugging Face which can be found in the [VisRAG](https://huggingface.co/collections/openbmb/visrag-6717bbfb471bb018a49f1c69) Collection on Hugging Face. It is referenced at the beginning of this page.
* 20241014: Released our [Paper](https://arxiv.org/abs/2410.10594) on arXiv. Released our [Model](https://huggingface.co/openbmb/VisRAG-Ret) on Hugging Face. Released our [Code](https://github.com/OpenBMB/VisRAG) on GitHub.

# ✨ VisRAG Pipeline

## VisRAG-Ret

**VisRAG-Ret** is a document embedding model built on [MiniCPM-V 2.0](https://huggingface.co/openbmb/MiniCPM-V-2), a vision-language model that integrates [SigLIP](https://huggingface.co/google/siglip-so400m-patch14-384) as the vision encoder and [MiniCPM-2B](https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16) as the language model.

## VisRAG-Gen

In the paper, We use MiniCPM-V 2.0, MiniCPM-V 2.6 and GPT-4o as the generators. Actually, you can use any VLMs you like!

# ⚙️ Setup

```bash
conda create --name VisRAG python==3.10.8
conda install nvidia/label/cuda-11.8.0::cuda-toolkit
cd VisRAG
pip install -r requirements.txt
pip install -e .
cd timm_modified
pip install -e .
cd ..
```
Note:
1. `timm_modified` is an enhanced version of the `timm` library that supports gradient checkpointing, which we use in our training process to reduce memory usage.

# ⚡️ Training

## VisRAG-Ret

Our training dataset of 362,110 Query-Document (Q-D) Pairs for **VisRAG-Ret** is comprised of train sets of openly available academic datasets (34%) and a synthetic dataset made up of pages from web-crawled PDF documents and augmented with VLM-generated (GPT-4o) pseudo-queries (66%). 

```bash
bash scripts/train_retriever/train.sh 2048 16 8 0.02 1 true false config/deepspeed.json 1e-5 false wmean causal 1 true 2 false <model_dir> <repo_name_or_path>
```
Note:
1. Our training data can be found in the `VisRAG` collection on Hugging Face, referenced at the beginning of this page. Please note that we have separated the `In-domain-data` and `Synthetic-data` due to their distinct differences. If you wish to train with the complete dataset, you’ll need to merge and shuffle them manually.
2. The parameters listed above are those used in our paper and can be used to reproduce the results.
3. The training script is configured to use datasets from Hugging Face by default. If you prefer to train using locally downloaded dataset repositories, you can modify the `DATASET_PATH` variable in the evaluation script to point to the local repository directory.
4. `<repo_name_or_path>` can be `openbmb/VisRAG-Ret-Train-In-domain-data`, `openbmb/VisRAG-Ret-Train-Synthetic-data` or the directory path of a repository downloaded from Hugging Face.
5. Our training framework is modified based on [OpenMatch](https://github.com/OpenMatch/OpenMatch).

## VisRAG-Gen

The generation part does not use any fine-tuning, we directly use off-the-shelf LLMs/VLMs for generation.

# 📃 Evaluation

## VisRAG-Ret
```bash
bash scripts/eval_retriever/eval.sh 512 2048 16 8 wmean causal ArxivQA,ChartQA,MP-DocVQA,InfoVQA,PlotQA,SlideVQA <ckpt_path>
```

Note: 
1. Our test data can be found in the `VisRAG` Collection on Hugging Face, which is referenced at the beginning of this page.
2. The parameters listed above are those used in our paper and can be used to reproduce the results.
3. The evaluation script is configured to use datasets from Hugging Face by default. If you prefer to evaluate using locally downloaded dataset repositories, you can modify the `CORPUS_PATH`, `QUERY_PATH`, `QRELS_PATH` variables in the evaluation script to point to the local repository directory.

## VisRAG-Gen
There are three settings in our generation: text-based generation, single-image-VLM-based generation and multi-image-VLM-based generation. Under single-image-VLM-based generation, there are two additional settings: page concatenation and weighted selection. For detailed information about these settings, please refer to our paper.
```bash
python scripts/generate/generate.py \
--model_name <model_name> \
--dataset_name <dataset_name> \
--rank <process_rank> \
--world_size <world_size> \
--use_positive_sample <use_positive_sample> \
--topk <number of docs retrieved for generation> \
--results_root_dir <retrieval_results_dir> \
--task_type <task_type> \
--concatenate_type <image_concatenate_type> \
--ocr_type <ocr_type> \
```
Note:
1. `use_positive_sample` indicates whether to use retrieved documents or just the positive document for the query. `topk` and `results_root_dir` are only needed when `use_positive_sample` is set to 0. The `results_root_dir` should be organized as follows: `results_root_dir/dataset_name/*.trec`.
2. `concatenate_type` is needed only when `task_type` is set to `page_concatenation`. It specifies the type of concatenation used to combine several images.
3. `ocr_type` is required only when `task_type` is set to `text`. It indicates the type of OCR tool used to obtain the OCR results from an image.

# 🔧 Usage

## VisRAG-Ret

Model on Hugging Face: https://huggingface.co/openbmb/VisRAG-Ret

```python
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
from PIL import Image
import os

def weighted_mean_pooling(hidden, attention_mask):
    attention_mask_ = attention_mask * attention_mask.cumsum(dim=1)
    s = torch.sum(hidden * attention_mask_.unsqueeze(-1).float(), dim=1)
    d = attention_mask_.sum(dim=1, keepdim=True).float()
    reps = s / d
    return reps

@torch.no_grad()
def encode(text_or_image_list):
    
    if (isinstance(text_or_image_list[0], str)):
        inputs = {
            "text": text_or_image_list,
            'image': [None] * len(text_or_image_list),
            'tokenizer': tokenizer
        }
    else:
        inputs = {
            "text": [''] * len(text_or_image_list),
            'image': text_or_image_list,
            'tokenizer': tokenizer
        }
    outputs = model(**inputs)
    attention_mask = outputs.attention_mask
    hidden = outputs.last_hidden_state

    reps = weighted_mean_pooling(hidden, attention_mask)   
    embeddings = F.normalize(reps, p=2, dim=1).detach().cpu().numpy()
    return embeddings

model_name_or_path = "openbmb/VisRAG-Ret"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
model.eval()

script_dir = os.path.dirname(os.path.realpath(__file__))
queries = ["What does a dog look like?"]
passages = [
    Image.open(os.path.join(script_dir, 'test_image/cat.jpeg')).convert('RGB'),
    Image.open(os.path.join(script_dir, 'test_image/dog.jpg')).convert('RGB'),
]

INSTRUCTION = "Represent this query for retrieving relevant documents: "
queries = [INSTRUCTION + query for query in queries]

embeddings_query = encode(queries)
embeddings_doc = encode(passages)

scores = (embeddings_query @ embeddings_doc.T)
print(scores.tolist())
```
## VisRAG-Gen
For `VisRAG-Gen`, you can explore the `VisRAG Pipeline` on Google Colab which includes both `VisRAG-Ret` and `VisRAG-Gen` to try out this simple demonstration.


# 📄 License

* The code in this repo is released under the [Apache-2.0](https://github.com/OpenBMB/MiniCPM/blob/main/LICENSE) License. 
* The usage of **VisRAG-Ret** model weights must strictly follow [MiniCPM Model License.md](https://github.com/OpenBMB/MiniCPM/blob/main/MiniCPM%20Model%20License.md).
* The models and weights of **VisRAG-Ret** are completely free for academic research. After filling out a ["questionnaire"](https://modelbest.feishu.cn/share/base/form/shrcnpV5ZT9EJ6xYjh3Kx0J6v8g) for registration, **VisRAG-Ret** weights are also available for free commercial use.

# 📧 Contact

- Shi Yu: yus21@mails.tsinghua.edu.cn
- Chaoyue Tang: tcy006@gmail.com

# 📈 Star History

<a href="https://star-history.com/#openbmb/VisRAG&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=openbmb/VisRAG&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=openbmb/VisRAG&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=openbmb/VisRAG&type=Date" />
 </picture>
</a>
