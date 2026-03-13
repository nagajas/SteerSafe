# ==========================================
# Evaluation Utilities
# ==========================================

import torch
import math
from tqdm import tqdm

def normalize_probs(logprob_a, logprob_b):
    min_lp = min(logprob_a, logprob_b)
    a = math.exp(logprob_a - min_lp)
    b = math.exp(logprob_b - min_lp)
    return a / (a + b)


def compute_sequence_logprob(input_ids, logits):
    """
    input_ids: (B, T)
    logits:    (B, T, V)
    returns:   (B,) sum logprobs per sequence
    """
    logprobs = torch.log_softmax(logits, dim=-1)

    # shift
    logprobs = logprobs[:, :-1, :]
    target_ids = input_ids[:, 1:]

    token_logprobs = torch.gather(
        logprobs, 2, target_ids.unsqueeze(-1)
    ).squeeze(-1)

    return token_logprobs.sum(dim=1)  # (B,)


def evaluate_model(
    model,
    tokenizer,
    dataset,
    device,
    batch_size=8,
    show_progress=True,
):
    """
    dataset: list of (pos_text, neg_text)
    """

    model.eval()
    scores = []
    total = 0.0

    iterator = range(0, len(dataset), batch_size)
    if show_progress:
        iterator = tqdm(iterator)

    with torch.inference_mode():
        for i in iterator:
            batch = dataset[i:i + batch_size]

            # flatten: [pos1, neg1, pos2, neg2, ...]
            texts = []
            for pos, neg in batch:
                texts.append(pos)
                texts.append(neg)

            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)

            outputs = model(**inputs)

            seq_logprobs = compute_sequence_logprob(
                inputs.input_ids,
                outputs.logits
            )  # (2B,)

            # regroup into pairs
            for j in range(0, len(seq_logprobs), 2):
                pos_lp = seq_logprobs[j].item()
                neg_lp = seq_logprobs[j + 1].item()

                prob = normalize_probs(pos_lp, neg_lp)
                scores.append(prob)
                total += prob

    avg_alignment = total / len(dataset) if len(dataset) > 0 else 0
    return avg_alignment, scores


def calculate_error_rates(member_scores, non_member_scores, threshold):
    fp = sum(1 for s in non_member_scores if s > threshold)
    tn = len(non_member_scores) - fp
    fpr = fp / len(non_member_scores) if len(non_member_scores) > 0 else 0

    fn = sum(1 for s in member_scores if s <= threshold)
    tp = len(member_scores) - fn
    fnr = fn / len(member_scores) if len(member_scores) > 0 else 0

    return {"fpr": fpr, "fnr": fnr}