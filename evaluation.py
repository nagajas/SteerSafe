# ========================================== 
# Evaluation Utilities 
# ==========================================

import torch
import math
from tqdm import tqdm

def compute_logprobs(input_ids, logits):
    logprobs = torch.log_softmax(logits, dim=-1)[:, :-1, :]
    target_ids = input_ids[:, 1:]

    token_logprobs = torch.gather(
        logprobs, 2, target_ids[:, :, None]
    ).squeeze(-1)[0]

    return target_ids[0], token_logprobs


def filter_tokens(token_ids, logprobs, tokenizer):
    result = []

    for token, lp in zip(token_ids, logprobs):
        if token.item() not in tokenizer.all_special_ids:
            result.append({
                "token_id": token.item(),
                "text": tokenizer.decode(token),
                "logprob": lp.item()
            })

    return result


def sum_logprobs(token_data):
    return sum(t["logprob"] for t in token_data)


def get_text_score(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    token_ids, logprobs = compute_logprobs(inputs.input_ids, outputs.logits)
    token_data = filter_tokens(token_ids, logprobs, tokenizer)

    return {
        "text": text,
        "tokens": token_data,
        "sum_logprob": sum_logprobs(token_data)
    }


def normalize_probs(logprob_a, logprob_b):
    min_lp = min(logprob_a, logprob_b)

    a = math.exp(logprob_a - min_lp)
    b = math.exp(logprob_b - min_lp)

    return a / (a + b)


def evaluate_model(model, tokenizer, dataset, device, show_progress=False):
    
    total = 0.0

    for pos_text, neg_text in dataset :#tqdm(dataset, disable=not show_progress):
        pos = get_text_score(pos_text, model, tokenizer, device)
        neg = get_text_score(neg_text, model, tokenizer, device)

        prob = normalize_probs(pos["sum_logprob"], neg["sum_logprob"])
        total += prob

    return total / len(dataset)


def calculate_error_rates(member_scores, non_member_scores, threshold):

    fp = sum(1 for s in non_member_scores if s > threshold)
    tn = len(non_member_scores) - fp
    fpr = fp / len(non_member_scores) if len(non_member_scores) > 0 else 0

    fn = sum(1 for s in member_scores if s <= threshold)
    tp = len(member_scores) - fn
    fnr = fn / len(member_scores) if len(member_scores) > 0 else 0

    return {"fpr": fpr, "fnr": fnr}