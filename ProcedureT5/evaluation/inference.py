import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import torch.multiprocessing as mp
import torch
import os
import json
import textdistance
from analysis import original_bleu

def predict_batch(model, tokenizer, dataset, batch_size, gpu_id=0):
    torch.cuda.set_device(gpu_id)
    model = model.to(f'cuda:{gpu_id}')  # Move the model to the specific GPU

    results = []
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i+batch_size]
        batch_sources = [data['source'] for data in batch]
        batch_encoding = tokenizer(
            batch_sources,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        input_ids = batch_encoding['input_ids'].to(f'cuda:{gpu_id}')
        attention_mask = batch_encoding['attention_mask'].to(f'cuda:{gpu_id}')

        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=512,
            num_beams=2,
            repetition_penalty=1,
            length_penalty=1,
            early_stopping=True,
        )
        pred_list = [tokenizer.decode(pred, skip_special_tokens=True) for pred in generated_ids]
        for j in range(len(batch)):
            dataset[i+j]['pred'] = pred_list[j]
    return dataset

def predict_mGPU(dataset, model, tokenizer, batch_size, gpu_ids):
    chunk_size = len(dataset) // len(gpu_ids)
    dataset_chunks = [dataset[i:i+chunk_size] for i in range(0, len(gpu_ids)-1, chunk_size)]
    dataset_chunks.append(dataset[(len(gpu_ids)-1)*chunk_size:])
    # Create a multiprocessing Pool with one process per GPU
    with mp.Pool(processes=len(gpu_ids)) as pool:
        results = pool.starmap(predict_batch, [(model, tokenizer, chunk, batch_size, gpu_id) for chunk, gpu_id in zip(dataset_chunks, gpu_ids)])

    # Flatten the list of results
    results = [item for sublist in results for item in sublist]
    return results

def dataset_inference(dataset, model_path, tokenizer_path, batch_size=8, multi_gpu=False):
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
    if multi_gpu:
        gpu_ids = [i for i in range(torch.cuda.device_count())]
        dataset = predict_mGPU(dataset, model, tokenizer, batch_size, gpu_ids)
    else:
        dataset = predict_batch(model, tokenizer, dataset, batch_size)
    # dataset = predict_batch(model, tokenizer, dataset, batch_size)
    return dataset

def evaluation(test_set_dict, model_path, tokenizer_path, batch_size=32, ckpt_name='', multi_gpu=True, aug_test=False):
    model_dir = os.path.dirname(model_path)
    test_dir_path = os.path.join(model_dir, 'test_results_'+ckpt_name)
    os.makedirs(test_dir_path, exist_ok=True)
    if aug_test:
        file_name = 'augmented_test_results.jsonl'
        test_set = test_set_dict['augmented']
    else:
        file_name = 'original_test_results.jsonl'
        test_set = test_set_dict['original']
    if os.path.exists(os.path.join(test_dir_path, file_name)):
        print('Test results already exist. Loading from file...')
        test_results = []
        with open(os.path.join(test_dir_path, file_name), 'r') as f:
            for line in f:
                test_results.append(json.loads(line))
        return test_results
    else:
        test_results = dataset_inference(test_set, model_path, tokenizer_path, batch_size=batch_size, multi_gpu=multi_gpu)
        for i in range(len(test_results)):
            test_results[i]['sim'] = textdistance.levenshtein.normalized_similarity(test_results[i]['target'], test_results[i]['pred'])
        with open(os.path.join(test_dir_path, file_name), 'w') as f:
            for item in test_results:
                f.write(json.dumps(item) + '\n')
        return test_results
    
def metric_printer(model_name, print_aug_test=False):
    test_dir_list = os.listdir(f"../../results/{model_name}")
    test_dir_list = [item for item in test_dir_list if 'test_results' in item]
    test_path_list = [os.path.join(f"../../results/{model_name}", item) for item in test_dir_list]
    for test_path in test_path_list:
        print(f"Test Path: {test_path}")
        results_list = []
        if not print_aug_test:
            with open(os.path.join(test_path, 'original_test_results.jsonl')) as f:
                original_test_results = [json.loads(line) for line in f]
                results_list.append(original_test_results)
        else:
            if os.path.exists(os.path.join(test_path, 'augmented_test_results.jsonl')):
                with open(os.path.join(test_path, 'augmented_test_results.jsonl')) as f:
                    augmented_test_results = [json.loads(line) for line in f]
                    results_list.append(augmented_test_results)
            else:
                print("Augmented Test Results Not Found")
        for results in results_list:
            avg_sim = round(np.mean([item['sim'] for item in results]),4)
            acc_100 = round(len([item for item in results if item['sim'] == 1]) / len(results), 4)
            acc_90 = round(len([item for item in results if item['sim'] >= 0.9]) / len(results), 4)
            acc_75 = round(len([item for item in results if item['sim'] >= 0.75]) / len(results), 4)
            acc_50 = round(len([item for item in results if item['sim'] >= 0.5]) / len(results), 4)
            pred_list = [item['pred'] for item in results]
            target_list = [item['target'] for item in results]
            bleu = round(original_bleu(pred_list, target_list), 4)
            print(f"Model: {model_name}, Bleu: {bleu}, Ave Accuracy:{avg_sim}, Accuracy 100: {acc_100}, Accuracy 90: {acc_90}, Accuracy 75: {acc_75}, Accuracy 50: {acc_50}")