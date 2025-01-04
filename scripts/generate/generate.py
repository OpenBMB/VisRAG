import json
import os
import argparse
import glob
from openmatch.utils import load_from_trec
import torch
from PIL import Image
import base64
from io import BytesIO
from transformers import AutoTokenizer as Tokenizer_class
from openmatch.generation_utils import get_flatten_table, preprocess_text, is_numeric_data, is_within_5_percent, horizontal_concat, vertical_concat
from openai import OpenAI
from datasets import load_dataset


def images_to_base64_list(image_list):
    base64_list = []
    for img in image_list:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        base64_list.append(img_base64)
    return base64_list


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, choices=['MiniCPM', 'MiniCPMV2.0', 'MiniCPMV2.6', 'gpt4o'])
    parser.add_argument('--dataset_name', type=str, choices=['ArxivQA', 'ChartQA', 'PlotQA', 'MP-DocVQA', 'SlideVQA', 'InfoVQA'], required=True)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--world_size', type=int, required=True)

    parser.add_argument('--use_positive_sample')
    parser.add_argument('--topk', type=int)
    parser.add_argument('--results_root_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    
    parser.add_argument('--task_type', type=str, required=True, choices=['text', 'page_concatenation', 'weighted_selection', 'multi_image'])
    parser.add_argument('--concatenate_type', type=str, choices=['horizontal', 'vertical'])
    parser.add_argument('--openai_api_key', type=str, help='api key for open_ai, required only if --model_name == gpt4o')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    check_args(args)

    model_name = args.model_name
    dataset_name = args.dataset_name

    if (dataset_name == 'ArxivQA'):
        max_new_tokens = 2
    else:
        max_new_tokens = 20
    
    if (not args.use_positive_sample):
        run = get_run(args, dataset_name)
        
    task_type = args.task_type
    # Load corpus
    corpus = {} # Build a dict that maps docid to OCR results or image
    if (task_type == 'text'):
        """
        Please write the logic for reading OCR results and building a dict that maps docid to OCR results yourself,
        as different people might have different ways of organizing those results.
        """
        raise Exception("Please write the logic yourself.")
    else:
        """
        Please write either a HF dataset name or a path to the dataset here.
        For example,
        dataset_name_or_path = 'openbmb/VisRAG-Ret-Test-ArxivQA'
        dataset_name_or_path = '/path/to/VisRAG-Ret-Test-ArxivQA'
        """
        dataset_name_or_path = f"openbmb/VisRAG-Ret-Test-{dataset_name}"
        corpus_ds = load_dataset(dataset_name_or_path, name="corpus", split="train")
        print(f"We defaultly load the dataset (corpus) from HF, if you want to load the dataset from local, please modify the dataset_name_or_path in the script.")
        for i in range(len(corpus_ds)):
            corpus_id = corpus_ds[i]['corpus-id']
            image = corpus_ds[i]['image'].convert('RGB')
            corpus[corpus_id] = image  

    # Load queries
    dataset_name_or_path = f"openbmb/VisRAG-Ret-Test-{dataset_name}"
    queries = load_dataset(dataset_name_or_path, name="queries", split="train")
    print(f"We defaultly load the dataset (queries) from HF, if you want to load the dataset from local, please modify the dataset_name_or_path in the script.")


    #加载模型
    if (task_type == 'weighted_selection'):
        if (model_name == 'MiniCPMV2.0'):
            from openmatch.modeling.weighted_selection.MiniCPMV20.modeling_minicpmv import MiniCPMV as ModelForCausalLM_class
    else:
        if (model_name == 'gpt4o'):
            client = OpenAI(api_key=args.openai_api_key)
        else:
            from transformers import AutoModel as Model_class
            from transformers import AutoModelForCausalLM as ModelForCausalLM_class
        
    if (model_name == 'MiniCPM'):
        model_name_or_path = None # Write your model path here
        if (model_name_or_path == None):
            raise Exception("model_name_or_path is None! Please write your model path.")
        tokenizer = Tokenizer_class.from_pretrained(model_name_or_path)
        model = ModelForCausalLM_class.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, trust_remote_code=True)

    elif (model_name == 'MiniCPMV2.0'):
        model_name_or_path = None # Write your model path here
        if (model_name_or_path == None):
            raise Exception("model_name_or_path is None! Please write your model path.")
        tokenizer = Tokenizer_class.from_pretrained(model_name_or_path, trust_remote_code=True)
        model = ModelForCausalLM_class.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
        model = model.to(device='cuda', dtype=torch.bfloat16)
        model.eval()

    elif (model_name == 'MiniCPMV2.6'):
        model_name_or_path = None # Write your model path here
        if (model_name_or_path == None):
            raise Exception("model_name_or_path is None! Please write your model path.")
        model = Model_class.from_pretrained(model_name_or_path, trust_remote_code=True,
            attn_implementation='sdpa', torch_dtype=torch.bfloat16)
        model = model.eval().cuda()
        tokenizer = Tokenizer_class.from_pretrained(model_name_or_path, trust_remote_code=True)

    if (model_name != 'gpt4o'):
        model.to(args.rank)
    
    history_datas = []
    correct = 0
    total_num = 0
    for cnt, example in enumerate(queries):
        if (cnt % args.world_size != args.rank):
            continue
        history_data = {}
        query = example['query']
        history_data['query'] = query
        qid = example['query-id']
        history_data['qid'] = qid
        answer = example['answer']
        history_data['original_answer'] = answer
        if (answer == None):
            raise Exception("answer is None!")
        if (args.use_positive_sample):
            if (dataset_name == 'SlideVQA'):
                # due to the special format of SlideVQA, we need to split the qid to get the ground truth docid
                docid = qid.split('query_number')[0]
                docid = docid.split('tcy6')
            else:
                docid = [qid[:-1 - len(qid.split('-')[-1])]]

            if (task_type == 'weighted_selection'):
                doc_scores = len(docid) * [1/len(docid)]
        else:
            # get top-k docid
            docid = []
            doc_scores = []
            doc_cnt = 0
            for key, value in sorted(run[qid].items(), key=lambda item: item[1], reverse=True):
                if (doc_cnt < args.topk):
                    docid.append(key)
                    doc_scores.append(value)
                    doc_cnt += 1
                else:
                    break
            if (len(docid) < args.topk):
                raise Exception("len(docid) < topk!")
        history_data['docid'] = docid
        if (task_type == 'text'):
            if (dataset_name == 'ChartQA'):
                table_dir = None # Write your table path here
                if (table_dir == None):
                    raise Exception("""table_dir is None! Please download the table data from https://huggingface.co/datasets/ahmed-masry/ChartQA/tree/main""")
                csv_file_path = [os.path.join(table_dir, f"{docid_item.split('.')[0]}.csv") for docid_item in docid] # get table 
                doc_list = [get_flatten_table(csv_file_path_item) for csv_file_path_item in csv_file_path]
                doc = '\n'.join(doc_list)
                input = f"Image:{doc}\nAnswer the question using a single word or phrase.\nQuestion:{query}\nAnswer:"
            elif (dataset_name == 'ArxivQA'):
                prompt = ''
                doc_list = [corpus[docid_item] for docid_item in docid]
                doc = '\n'.join(doc_list)
                options = example['options']
                options_prompt = 'Options:\n'
                # if A, B, C, D is not at the beginning
                flag = 0
                for i, option in enumerate(options):
                    if not option.startswith(f"{chr(65 + i)}"):
                        flag = 1
                        break
                if flag:
                    # pre-process
                    for i, option in enumerate(options):
                        options[i] = f"{chr(65 + i)}. {option.strip()}"
                for item in options:
                    options_prompt += f'{item}\n'
                prompt += f'Hint: {doc}\n'
                prompt += f'Question: {query}\n'
                prompt += options_prompt
                prompt += '''Answer directly with the letter of the correct option as the first character.'''
                input = prompt
            elif (dataset_name == 'PlotQA'):
                doc_list = [corpus[docid_item] for docid_item in docid]
                doc = '\n'.join(doc_list)
                input = f"Image:{doc}\nAnswer the question using a single word or phrase.\nQuestion:{query}\nAnswer:"
            elif (dataset_name == 'MP-DocVQA'):
                doc_list = [corpus[docid_item] for docid_item in docid]
                doc = '\n'.join(doc_list)
                input = f"Image:{doc}\nAnswer the question using a single word or phrase.\nQuestion:{query}\nAnswer:"
            elif (dataset_name == 'SlideVQA'):
                doc_list = [corpus[docid_item] for docid_item in docid]
                doc = '\n'.join(doc_list)
                input = f"Image:{doc}\nAnswer the question using a single word or phrase.\nQuestion:{query}\nAnswer:"
            elif (dataset_name == 'InfoVQA'):
                doc_list = [corpus[docid_item] for docid_item in docid]
                doc = '\n'.join(doc_list)
                input = f"Image:{doc}\nAnswer the question using a single word or phrase.\nQuestion:{query}\nAnswer:"
            
            history_data['prompt'] = input
            
            if (model_name == 'MiniCPM'):
                responds, history = model.chat(tokenizer, input, temperature=0.8, top_p=0.8, max_new_tokens=max_new_tokens)
            elif (model_name == 'gpt4o'):
                max_retries = 10
                retries = 0
                while retries < max_retries:
                    try:
                        response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": f"{input}"
                                        }
                                    ],
                                }
                            ],
                            max_tokens=max_new_tokens,
                        )
                        responds = response.choices[0].message.content
                        break
                    except Exception as e:
                        retries += 1
                        print(f"retry times: {retries}/{max_retries}")
                        if retries >= max_retries:
                            print("Unable to call the API, skipping this call.")
                            responds = None
                if (retries >= max_retries):
                    continue
                    
        else:
            image_list = [corpus[docid_item] for docid_item in docid]
            
            if (task_type == 'page_concatenation'):
                if (args.concatenate_type == 'horizontal'):
                    image_list = [horizontal_concat(image_list)]
                elif (args.concatenate_type == 'vertical'):
                    image_list = [vertical_concat(image_list)]
                
            if (dataset_name == 'ChartQA'):
                input = [{'role': 'user', 'content': f"Answer the question using a single word or phrase.\nQuestion:{query}\nAnswer:"}]
            elif (dataset_name == 'ArxivQA'):
                prompt = ''
                options = example['options']
                options_prompt = 'Options:\n'
                # if A, B, C, D is not at the beginning
                flag = 0
                for i, option in enumerate(options):
                    if not option.startswith(f"{chr(65 + i)}"):
                        flag = 1
                        break
                if flag:
                    # pre-process
                    for i, option in enumerate(options):
                        options[i] = f"{chr(65 + i)}. {option.strip()}"
                for item in options:
                    options_prompt += f'{item}\n'
                prompt += f'Question: {query}\n'
                prompt += options_prompt
                prompt += '''Answer directly with the letter of the correct option as the first character.'''
                input = [{'role': 'user', 'content': prompt}]
            elif (dataset_name == 'PlotQA'):
                input = [{'role': 'user', 'content': f"Answer the question using a single word or phrase.\nQuestion:{query}\nAnswer:"}]
            elif (dataset_name == 'MP-DocVQA'):
                input = [{'role': 'user', 'content': f"Answer the question using a single word or phrase.\nQuestion:{query}\nAnswer:"}]
            elif (dataset_name == 'SlideVQA'):
                input = [{'role': 'user', 'content': f"Answer the question using a single word or phrase.\nQuestion:{query}\nAnswer:"}]
            elif (dataset_name == 'InfoVQA'):
                input = [{'role': 'user', 'content': f"Answer the question using a single word or phrase.\nQuestion:{query}\nAnswer:"}]
            
            history_data['prompt'] = input[0]['content']

            if (task_type == 'page_concatenation'):
                if (model_name == 'MiniCPMV2.0'):
                    responds, context, _ = model.chat(
                        image=image_list[0], # image_list only has one element
                        msgs=input,
                        context=None,
                        tokenizer=tokenizer,
                        sampling=False,
                        max_new_tokens=max_new_tokens
                    )
            elif (task_type == 'multi_image'):
                if (model_name == 'MiniCPMV2.6'):
                    input = [{'role': 'user', 'content': image_list + [input[0]['content']]}]
                    responds = model.chat(
                        image=None,
                        msgs=input,
                        tokenizer=tokenizer,
                        sampling=False,
                        max_new_tokens=max_new_tokens
                    )
                elif (model_name == 'gpt4o'):
                    max_retries = 10
                    retries = 0
                    while retries < max_retries:
                        try:
                            messages = [
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": f"{input[0]['content']}"
                                        }
                                    ],
                                }
                            ]
                            for base64_string_item in images_to_base64_list(image_list):
                                messages[0]["content"].append({
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{base64_string_item}"}
                                })

                            response = client.chat.completions.create(
                                model="gpt-4o",
                                messages=messages,
                                max_tokens=max_new_tokens,
                            )
                            responds = response.choices[0].message.content
                            break
                        except Exception as e:
                            retries += 1
                            print(f"retry times: {retries}/{max_retries}")
                            if retries >= max_retries:
                                print("Unable to call the API, skipping this call.")
                                responds = None
                    if (retries >= max_retries):
                        continue
            elif (task_type == 'weighted_selection'):
                if (model_name == 'MiniCPMV2.0'):
                    responds = model.weighted_selection(
                    image_list=image_list,
                    msgs=input,
                    doc_scores=doc_scores,
                    context=None,
                    tokenizer=tokenizer,
                    sampling=False,
                    max_new_tokens=max_new_tokens
                    )
            
        total_num += 1
        
        responds_backup = responds
            
        #pre-process
        if (dataset_name == 'ChartQA'):
            responds = preprocess_text(responds)
            answer = preprocess_text(answer)
            if ('%' in responds and '%' not in answer):
                responds = responds.replace('%', '')
            if ('%' not in responds and '%' in answer):
                answer = answer.replace('%', '')
            print(f"query: {query}")
            print(f"responds:{responds}")
            print(f"answer:{answer}")
            if (responds == answer):
                correct += 1
            elif(is_numeric_data(responds) and is_numeric_data(answer) and answer != '0' and is_within_5_percent(responds, answer)):
                correct += 1
        elif (dataset_name == 'ArxivQA'):
            responds = responds[0].upper()
            answer = answer[0].upper()
            print(f"query: {query}")
            print(f"responds:{responds}")
            print(f"answer:{answer}")
            if (responds == answer):
                correct += 1
        elif (dataset_name == 'PlotQA'):
            responds = preprocess_text(responds)
            is_str = 1
            if (type(answer) != str):
                is_str = 0
                answer = str(answer)
            answer = preprocess_text(answer)
            if ('%' in responds and '%' not in answer):
                responds = responds.replace('%', '')
            if ('%' not in responds and '%' in answer):
                answer = answer.replace('%', '')
            print(f"query: {query}")
            print(f"responds:{responds}")
            print(f"answer:{answer}")
            if (responds == answer):
                correct += 1
            elif(is_numeric_data(responds) and (not is_str) and float(answer) != 0.0 and is_within_5_percent(responds, answer)):
                correct += 1
        elif (dataset_name == 'MP-DocVQA'):
            responds = preprocess_text(responds)
            if (not isinstance(answer, list)):
                answer = [answer]
            for i, answer_item in enumerate(answer):
                answer[i] = preprocess_text(answer_item)
            if ('%' in responds and '%' not in answer[0]):
                responds = responds.replace('%', '')
            if ('%' not in responds and '%' in answer[0]):
                answer = [answer_item.replace('%', '') for answer_item in answer]
            print(f"query: {query}")
            print(f"responds:{responds}")
            print(f"answer:{answer}")
            for answer_item in answer:
                if (responds == answer_item):
                    correct += 1
                    break
        elif (dataset_name == 'SlideVQA'):
            responds = preprocess_text(responds)
            answer = preprocess_text(answer)
            if ('%' in responds and '%' not in answer):
                responds = responds.replace('%', '')
            if ('%' not in responds and '%' in answer):
                answer = answer.replace('%', '')
            print(f"query: {query}")
            print(f"responds:{responds}")
            print(f"answer:{answer}")
            if (responds == answer):
                correct += 1
        elif (dataset_name == 'InfoVQA'):
            responds = preprocess_text(responds)
            if (not isinstance(answer, list)):
                answer = [answer]
            for i, answer_item in enumerate(answer):
                answer[i] = preprocess_text(answer_item)
            if ('%' in responds and '%' not in answer[0]):
                responds = responds.replace('%', '')
            if ('%' not in responds and '%' in answer[0]):
                answer = [answer_item.replace('%', '') for answer_item in answer]
            print(f"query: {query}")
            print(f"responds:{responds}")
            print(f"answer:{answer}")
            for answer_item in answer:
                if (responds == answer_item):
                    correct += 1
                    break
        
        history_data['preprocessed_responds'] = responds
        history_data['preprocessed_answer'] = answer
        history_data['original_responds'] = responds_backup
        

        # calculate accuracy
        print(f"{dataset_name}:{total_num}_Accuracy:{float(correct) / total_num}")
        print('---------------')
            
        history_datas.append(json.dumps(history_data))
    
    prefix, results_output_dir = make_prefix_output_dir(args.output_dir, model_name, args.use_positive_sample, args.results_root_dir, dataset_name, task_type, args.topk)
    write_results(results_output_dir, prefix, correct, total_num)   
    write_history(results_output_dir, prefix, history_datas)


def make_prefix_output_dir(output_dir, model_name, use_positive_sample, results_root_dir, dataset_name, task_type, topk):
    prefix = model_name
    results_output_dir = os.path.join(output_dir, prefix)
    prefix += '_'
    if (use_positive_sample):
        results_output_dir = os.path.join(output_dir, 'upper_bound')
        prefix += 'upper_bound'
    else:
        results_output_dir = os.path.join(output_dir, os.path.basename(results_root_dir))
        prefix += str(os.path.basename(results_root_dir))

    if (use_positive_sample):
        prefix = f"{prefix}_{dataset_name}_oracle"
    else:
        prefix = f"{prefix}_{dataset_name}_{task_type}_top{topk}"
    os.makedirs(results_output_dir, exist_ok=True)


def write_results(output_dir, prefix, correct, total_num):
    result_path = os.path.join(output_dir, f"{prefix}_result.jsonl")
    print(f"writing to {result_path}")
    with open(result_path, 'w') as file:
        acc = float(correct) / total_num
        data = {'Accuracy':acc}
        file.write(json.dumps(data)+'\n')


def write_history(output_dir, prefix, history_datas):
    history_path = os.path.join(output_dir, f"{prefix}_history.jsonl")
    print(f"writing to {history_path}")
    with open(history_path, 'w') as file:
        for history_data in history_datas:
            file.write(history_data + '\n')


def get_run(args, dataset_name):
    if (args.topk == None):
        raise Exception("topk is None!")
    if (args.results_root_dir == None):
        raise Exception("results_root_dir is None!")
    
    results_dir = os.path.join(args.results_root_dir, dataset_name)

    # Load trec files which is generated after retrieval evaluation
    partitions = glob.glob(os.path.join(results_dir, "test.*.trec"))
    run = {}
    for part in partitions:
        print("loading", part)
        run.update(load_from_trec(part))
    
    return run


def check_args(args):
    if args.output_dir == None:
        raise Exception("output_dir is None! Please write your output path.")
    if args.task_type == 'page_concatenation':
        if args.concatenate_type == None:
            raise Exception("concatenate_type is None!")
        if (args.concatenate_type not in ['horizontal', 'vertical']):
                raise Exception("concatenate_type error!")
    if (args.model_name == 'gpt4o'):
        if (args.openai_api_key == None):
            raise Exception("Model is gpt4o but openai_api_key is None! Please write your OpenAI API key in openai_api_key argument.")

if __name__ == '__main__':
    main()
