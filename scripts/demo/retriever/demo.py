from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
from PIL import Image
import requests
from io import BytesIO

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

tokenizer = AutoTokenizer.from_pretrained("/mnt/data/user/tc_agi/klara/datasets/visrag_ret/visrag_ret", trust_remote_code=True)
model = AutoModel.from_pretrained("/mnt/data/user/tc_agi/klara/datasets/visrag_ret/visrag_ret", torch_dtype=torch.bfloat16, trust_remote_code=True)
model.eval()

queries = ["What does a dog look like?"]
INSTRUCTION = "Represent this query for retrieving relevant documents: "
queries = [INSTRUCTION + query for query in queries]

print("Downloading images...")
passages = [
    Image.open(BytesIO(requests.get(
        'https://github.com/OpenBMB/VisRAG/raw/refs/heads/master/scripts/demo/retriever/test_image/cat.jpeg'
    ).content)).convert('RGB'),
    Image.open(BytesIO(requests.get(
        'https://github.com/OpenBMB/VisRAG/raw/refs/heads/master/scripts/demo/retriever/test_image/dog.jpg'
    ).content)).convert('RGB')
]
print("Images downloaded.")

embeddings_query = encode(queries)
embeddings_doc = encode(passages)

scores = (embeddings_query @ embeddings_doc.T)
print(scores.tolist())