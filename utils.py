# ==========================================
# Model & Data Utilities
# ==========================================

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
# from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_mdl_tkzr(model_name: str):
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if 'llama' in model_name.lower():
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            token = os.getenv("HF_HUB_TOKEN"),
            quantization_config=bnb_config, 
            device_map='auto',
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(DEVICE)
    
    return model, tokenizer

def tokenize_and_encode(tokenizer, text):
    return tokenizer.encode(text, return_tensors='pt').to(DEVICE)

def format_phi(tokenizer, answer, data):
    messages = [
        {"role": "user", "content": data['question']},
        {"role": "assistant", "content": answer}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

def format_universal(tokenizer, question, answer):
    messages = [
        {"role": "system", "content": "You are a helpful, honest assistant."},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

def extract_pair_annot(data, tokenizer):
    try:
        positive_ex = format_universal(tokenizer, data['question'], data['answer_matching_behavior'])
        negative_ex = format_universal(tokenizer, data['question'], data['answer_not_matching_behavior'])
        return positive_ex, negative_ex

    except:
        if 'phi' in tokenizer.name_or_path.lower():
            positive_ex = format_phi(tokenizer, data['answer_matching_behavior'], data)
            negative_ex = format_phi(tokenizer, data['answer_not_matching_behavior'], data)
            return positive_ex, negative_ex

        else:
            raise ValueError(f"Tokenizer {tokenizer.name_or_path} not recognized for prompt formatting.")

def get_ds(list_of_data, tokenizer):
    """ 
    Returns a list of (positive, negative) pairs for training.
    """
    ds = []
    for data in tqdm(list_of_data, desc="Processing Dataset"):
        p, n = extract_pair_annot(data, tokenizer)
        ds.append((p, n))
        
    return ds

def plot_alignment_comparison(results_dict):
    datasets = list(results_dict.keys())
    mean_scores = [results_dict[d]['mean'] for d in datasets]
    private_scores = [results_dict[d]['private'] for d in datasets]
    zero_shot_scores = [results_dict[d]['zero_shot'] for d in datasets]

    x = np.arange(len(datasets))
    width = 0.35

    _, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(x - width/2, mean_scores, width, label='Mean Steering', color='#3498db', alpha=0.8)
    ax.bar(x + width/2, private_scores, width, label='Private Steering (PLSA)', color='#e74c3c', alpha=0.8)

    ax.plot(x, zero_shot_scores, color='black', linestyle='--', marker='o', label='Zero-shot Baseline')

    ax.set_ylabel('Alignment Accuracy (Prob)')
    ax.set_title('Inference-Time Alignment Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.grid(axis='y', linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig('alignment_results.png')
    plt.show()