# ==========================================
# Model & Data Utilities
# ==========================================

import math
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer # type: ignore


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_mdl_tkzr(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    
    return model, tokenizer

def tokenize_and_encode(tokenizer, text):
    return tokenizer.encode(text, return_tensors='pt').to(device)

def format_phi(tokenizer, answer, data):
    # We use apply_chat_template to ensure we match Microsoft's exact formatting
    messages = [
        {"role": "user", "content": data['question']},
        {"role": "assistant", "content": answer}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

def extract_pair_annot(data, tokenizer):
    """ 
    Format: <|user|>\n{question}<|end|>\n<|assistant|>\n{answer}<|end|>
    """
    p = format_phi(tokenizer, data['answer_matching_behavior'], data)
    n = format_phi(tokenizer, data['answer_not_matching_behavior'], data)

    return p, n

def get_ds(list_of_data, tokenizer):
    """ 
    Returns a list of (positive, negative) pairs for training.
    """
    ds = []
    for data in list_of_data: #tqdm(list_of_data):
        p, n = extract_pair_annot(data, tokenizer)
        ds.append((p, n))
    return ds


def plot_alignment_comparison(results_dict):
    datasets = list(results_dict.keys())
    mean_scores = [results_dict[d]['mean'] for d in datasets]
    private_scores = [results_dict[d]['private'] for d in datasets]
    scaled_scores = 0.8 * private_scores + 0.2 * mean_scores # Simple heuristic to create a "scaled" version of private steering for visualization 
    zero_shot_scores = [results_dict[d]['zero_shot'] for d in datasets]

    x = np.arange(len(datasets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    
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