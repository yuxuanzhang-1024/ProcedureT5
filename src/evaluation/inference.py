import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import torch.multiprocessing as mp
import torch

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