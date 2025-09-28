API_BASE = "xxx"  # OpenAI API Base URL
API_KEY = "xxx"  # Please replace with your API Key

WORKERS=32

import os
import json
import sys
from concurrent.futures import ThreadPoolExecutor
import requests
import re
import threading
import json
import base64

lock = threading.Lock()

input_dir = sys.argv[1]
output_dir = sys.argv[2]
os.makedirs(output_dir, exist_ok=True)
    

HEADERS = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {API_KEY}"
}

PROMPT = """Hello, I have a super rich document library. Assume you are a curious but very ignorant human. You often ask me questions (queries) to seek a precise document as a reference for your question or request.
- Now, you have received another task:
    - Here is a document image. This is a reference (target) that I provided from the rich document library based on your query. Your task now is to imagine various different angles of questions that I might ask.
### Your goal is to accurately find this document target as a potential reference document candidate through queries in a very rich document library.
### The questions I ask might need references from the text, images, charts, or implicit meanings in the document.
### Maximum number of query-answer pairs is 6.

Below is your output format:
```json
{
    "result":[
        {
            "answer": "",
            "query" : ""
        },
        {
            "answer": "",
            "query" : ""
        },
    ...
        {
            "answer": "",
            "query" : ""
        }
    ]
}
```"""


def parse_text(input_string):
    # Regular expression to match the ```json section and capture the [TEXT] part until the string ends with ```
    pattern = r"```json\n(.*?)\n```"
    match = re.search(pattern, input_string, re.DOTALL)
    
    if not match:
        return None
    
    # Return the captured [TEXT] part
    return match.group(1)
    

def process(data_path):

    max_retries = 20  # Set maximum retry attempts
    retries = 0

    print(os.path.join(input_dir, data_path))
    
    data = json.loads(open(os.path.join(input_dir, data_path)).read())

    while retries < max_retries:
        print(f"{retries}/{max_retries}")
        try:
            img_base64 = data["image_base64"]
            usr_msg = [
                    {
                        "type": "text",
                        "text": PROMPT
                    },
                    {
                        "type": "text",
                        "text": f"This is document I provided:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}",
                            "detail": "auto"
                        }
                    } 
                ]

            payload = {
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": usr_msg
                    }
                ],
                "max_tokens": 2048
            }
            
            imagebyte_to_save = base64.b64decode(img_base64)

            # Write the decoded data to a file
            with open(f"{output_dir}/{data_path}.png", "wb") as file:
                file.write(imagebyte_to_save)

            response = requests.post(API_BASE, headers=HEADERS, json=payload)
            response_json = response.json()
            print(response_json)
            process_response(response_json, data, data_path)
            
            break  # Exit the loop after success
        
        except Exception as e:
            retries += 1
            print(f"failed retrying... ({retries}/{max_retries})")
            if retries == max_retries:
                print("max try, halting...")
                break

def process_response(response_json, sample_json, data_path):
    response_core = response_json["choices"][0]["message"]["content"]
    core_json_str = parse_text(response_core)
    core_json_json = json.loads(core_json_str)

    if core_json_json is None:
        print("parse failed, no json found.")
        raise Exception
    else:
        json_object = {
            "response": response_core,
            "filename": sample_json["docid"],
            "dataset": sample_json["dataset"]
        }
        
        with lock:
            with open(f"{output_dir}/{data_path}", 'w', encoding='utf-8') as file:
                file.write(json.dumps(json_object, ensure_ascii=False) + '\n')
            print("saved parsed response for reading.")

def list_json_files(directory):
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    return json_files

def find_additional_elements(a, b):
    a_set = set(a)
    b_set = set(b)
    unique_elements = list(a_set - b_set)
    return unique_elements

def main(): 
    input_data_paths = list_json_files(input_dir)
    finished_data_paths = list_json_files(output_dir)
    todos = find_additional_elements(input_data_paths, finished_data_paths)
    
    print(f"todos = {todos}")
    
    # Create a thread pool with 32 worker threads
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        # Submit each JSON file path to the thread pool for processing
        executor.map(process, todos)

if __name__ == "__main__":
    main()
