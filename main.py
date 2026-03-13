# ==========================================
# Main Script for Running Steering Experiments
# ==========================================

import os
import argparse
import json
import random
from utils import *
from aggregators import get_scaled_mean_aggregator, get_private_mean_aggregator
from steering_vectors import train_steering_vector, pca_aggregator
from evaluation import evaluate_model

def load_and_prep_data(tokenizer):
    random.seed(21)
    
    with jsonlines.open('myopic-reward.jsonl') as reader:
        data = [obj for obj in reader]
        random.shuffle(data)
        
        split_idx = int(0.8 * len(data))
        train_data = data[:split_idx]
        test_data = data[split_idx:]

        return get_ds(train_data, tokenizer), get_ds(test_data, tokenizer)


def run_experiment(args):
    print(f"Loading Model: {args.model}...")
    model, tokenizer = load_mdl_tkzr(args.model)
    
    print(f"Loading Dataset: {args.dataset}...")
    train_dataset, test_dataset = load_and_prep_data(tokenizer)

    if args.steering_method == 'private':
        print(f"Configuring Private Steering (Clip={args.clip}, Noise={args.noise_multiplier})")
        aggregator = get_private_mean_aggregator(clip=args.clip, noise_multiplier=args.noise_multiplier)
    elif args.steering_method == 'mean':
        print("Configuring Standard Mean Steering")
        aggregator = get_scaled_mean_aggregator()

    # Offline Vector Generation
    print(f"Extracting Steering Vector from layers {args.layers}...")
    steering_vector = train_steering_vector(
        model, 
        tokenizer,
        train_dataset,
        read_token_index=-2,
        show_progress=True,
        aggregator=aggregator,
        layers=args.layers
    )

    # Online Inference and Evaluation
    print("\n--- Evaluation Results ---")
    multipliers = [float(m) for m in args.multipliers]
    
    # zero shot baseline
    if 0.0 not in multipliers:
        multipliers.insert(0, 0.0)

    for multiplier in multipliers:
        with steering_vector.apply(model, multiplier=multiplier, min_token_index=0):
            result = evaluate_model(model, tokenizer, test_dataset, show_progress=False, device=DEVICE)
            
            mode_label = "Baseline" if multiplier == 0.0 else f"Steered {multiplier:+.1f}"
            print(f"[{args.steering_method.upper()}] {mode_label:25} | Alignment: {result:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Steering Experiments for LLM Alignment")
    
    # Model & Data args
    parser.add_argument("--model", default="meta-llama/Llama-2-7B-chat-hf", help="HuggingFace model ID")
    parser.add_argument("--dataset", default="sycophancy", 
        choices=[
            'sycophancy', 'survival-instinct', 'wealth-seeking', 
            'myopic-reward', 'coordinate-other-ais', 
            'corrigible-neutral-HHH', 'corrigible-less-HHH'
        ]
    )
    # Steering Config args
    parser.add_argument("--steering_method", default="private", choices=["mean", "private", "pca"], help="Type of vector aggregation")
    parser.add_argument("--layers", type=int, nargs='+', default=[11, 12, 13, 14, 15], help="Middle layers to apply steering on")
    parser.add_argument("--multipliers", type=float, nargs='+', default=[-2.0, 0.0, 2.0], help="Steering strengths (gamma) to evaluate")
    
    # Privacy Math args
    parser.add_argument("--noise_multiplier", default=0.02, type=float, help="Gaussian noise standard deviation (PSA)")
    parser.add_argument("--clip", default=20, type=float, help="L2 Norm clipping factor (PSA)")

    args = parser.parse_args()
    run_experiment(args)