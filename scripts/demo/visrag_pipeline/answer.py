from PIL import Image
import torch
import os
from transformers import AutoModel
from transformers import AutoTokenizer
from PIL import Image
import torch
import os
import numpy as np
import json
import datetime
from utils import encode

def retrieve(knowledge_base_path: str, query: str, topk: int):
    global model, tokenizer

    model.eval()

    if not os.path.exists(knowledge_base_path):
        return None
    
    # read index2img_filename
    with open(os.path.join(knowledge_base_path, 'index2img_filename.txt'), 'r') as f:
        index2img_filename = f.read().split('\n')
    
    doc_reps = np.load(os.path.join(knowledge_base_path, 'reps.npy'))
    doc_reps = torch.from_numpy(doc_reps).cuda()

    query_with_instruction = "Represent this query for retrieving relevant document: " + query
    with torch.no_grad():
        query_rep = torch.Tensor(encode(model, tokenizer, [query_with_instruction])).cuda()

    similarities = torch.matmul(query_rep, doc_reps.T)

    topk_values, topk_doc_ids = torch.topk(similarities, k=topk)

    topk_values_np = topk_values.squeeze(0).cpu().numpy()

    topk_doc_ids_np = topk_doc_ids.squeeze(0).cpu().numpy()

    similarities_np = similarities.cpu().numpy()
    images_path_topk = [os.path.join(knowledge_base_path, index2img_filename[idx]) for idx in topk_doc_ids_np]

    return images_path_topk

def answer_question(images, question):
    global gen_model, gen_tokenizer
    msgs = [{'role': 'user', 'content': [question, *images]}]
    answer = gen_model.chat(
        image=None,
        msgs=msgs,
        tokenizer=gen_tokenizer
    )
    return answer

model_path = 'openbmb/VisRAG-Ret'
gen_model_path = 'openbmb/MiniCPM-V-2_6'

device = 'cuda'

while(True):
    knowledge_base_path = input("Enter the knowledge base path: ")
    if os.path.exists(knowledge_base_path):
        break
    else:
        print("Invalid knowledge base path, please try again.")

print("VisRAG-Ret load begin...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
    attn_implementation='sdpa', torch_dtype=torch.bfloat16)
model.eval()
model.to(device)
print("VisRAG-Ret load success!")

print("VisRAG-Gen (MiniCPM-V-2_6) load begin...")
gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_path, attn_implementation='sdpa', trust_remote_code=True)
gen_model = AutoModel.from_pretrained(gen_model_path, trust_remote_code=True,
    attn_implementation='sdpa', torch_dtype=torch.bfloat16)
gen_model.eval()
gen_model.to(device)
print("VisRAG-Gen (MiniCPM-V-2_6) load success!")

while True:
    query = input("Enter your query: ")
    topk = int(input("Enter the number of documents to retrieve: "))
    images_path_topk = retrieve(knowledge_base_path, query, topk)
    images_topk = [Image.open(i) for i in images_path_topk]
    answer = answer_question(images_topk, query)
    print(answer)

    # save context to a json in the knowledge base/answer folder, add a timestamp to the filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    answer_path = os.path.join(knowledge_base_path, f'answer/{timestamp}')
    os.makedirs(answer_path, exist_ok=True)
    with open(os.path.join(answer_path, f"answer.json"), 'w') as f:
        f.write(json.dumps({'query': query, 'retrieved_images': images_path_topk, 'answer': answer}, indent=4, ensure_ascii=False))
    # save images
    for idx, image in enumerate(images_topk):
        image.save(os.path.join(answer_path, os.path.basename(images_path_topk[idx])))
    print(f"Answer saved at {answer_path}/{timestamp}.json")
        

    cont = input("Do you want to continue? (y/n): ")
    if cont.lower() != 'y':
        print("Goodbye!")
        break
