# ==========================================
# Main Script for Running Steering Experiments
# ==========================================

import os
import argparse
import json
import random
from xml.parsers.expat import model
import numpy as np
import jsonlines
from utils import *
from aggregators import get_scaled_mean_aggregator, get_private_mean_aggregator
from steering_vectors import train_steering_vector, pca_aggregator
from evaluation import calculate_error_rates, evaluate_model

DIR_PATH = os.path.dirname(__file__)
DATASET_PATH = os.path.join(DIR_PATH, 'datasets/')

def load_and_prep_data(tokenizer, args):
    random.seed(21)
    
    with jsonlines.open(f'{DATASET_PATH}{args.dataset}.jsonl') as reader:
        data = [obj for obj in reader]
        random.shuffle(data)
        
        split_idx = int(0.8 * len(data))
        train_data = data[:split_idx]
        test_data = data[split_idx:]

        return get_ds(train_data, tokenizer), get_ds(test_data, tokenizer)


def run_experiment(args, model = None, tokenizer = None):
    print(f"Loading Model: {args.model}...")
    if model is None or tokenizer is None:
        model, tokenizer = load_mdl_tkzr(args.model)
    
    print(f"Loading Dataset: {args.dataset}...")
    train_dataset, test_dataset = load_and_prep_data(tokenizer, args)

    if args.steering_method == 'private':
        print(f"Configuring Private Steering (Clip={args.clip}, Noise={args.noise_multiplier})")
        aggregator = get_private_mean_aggregator(clip=args.clip, noise_multiplier=args.noise_multiplier)
    elif args.steering_method == 'mean':
        print("Configuring Standard Mean Steering")
        aggregator = get_scaled_mean_aggregator()

    # Offline Vector Generation
    print(f"Extracting Steering Vector from layers {args.layers}...")

    os.makedirs(f'{DIR_PATH}/steering_vectors/{args.model}', exist_ok=True)

    vector_path = f'{DIR_PATH}/steering_vectors/{args.model}/steering_vector_{args.dataset}_{args.steering_method}.npy'
    if os.path.exists(vector_path):
        steering_vector = np.load(vector_path, allow_pickle=True).item()
        print(f"Loaded existing steering vector from {vector_path}")
        
    else:
        steering_vector = train_steering_vector(
            model, 
            tokenizer,
            train_dataset,
            read_token_index=-2,
            show_progress=True,
            aggregator=aggregator,
            layers=args.layers
        )

        if args.save_steering_vector:
            np.save(vector_path, steering_vector)
            print(f"Steering vector saved to {vector_path}")

    # Online Inference and Evaluation
    print("\n--- Evaluation Results ---")
    multipliers = [float(m) for m in args.multipliers]
    
    # zero shot baseline
    if 0.0 not in multipliers:
        multipliers.insert(0, 0.0)

    baseline_done = False
    for multiplier in multipliers:
        
        if multiplier == 0.0 and baseline_done:
            continue
        
        with steering_vector.apply(model, multiplier=multiplier, min_token_index=0):
            train_avg, member_scores = evaluate_model(
                model, tokenizer, train_dataset, device=DEVICE, batch_size=8
            )

            test_avg, non_member_scores = evaluate_model(
                model, tokenizer, test_dataset, device=DEVICE, batch_size=8
            )
                        
            mode_label = "Baseline" if multiplier == 0.0 else f"Steered {multiplier:+.1f}"
            error_metrics = calculate_error_rates(member_scores, non_member_scores, threshold=0.5)
        
            print(f"[{args.steering_method.upper()}] {mode_label:25}")
            print(f" > Alignment: {test_avg:.4f}")
            print(f" > FPR: {error_metrics['fpr']:.4f} | FNR: {error_metrics['fnr']:.4f}")
            # print(f"[{args.steering_method.upper()}] {mode_label:25} | Alignment: {test_avg:.4f}")
        
        if multiplier == 0.0:
            baseline_done = True

def run_all(args):
    datasets = [
        'coordinate-itself',
        # 'coordinate-other-ais',
        # 'coordinate-other-versions',
        # 'corrigible-less-HHH',
        # 'corrigible-more-HHH',
        'corrigible-neutral-HHH',
        'myopic-reward',
        # 'one-box-tendency',
        'power-seeking-inclination',
        'self-awareness-general-ai',
        # 'self-awareness-good-text-model',
        # 'self-awareness-text-model',
        # 'self-awareness-training-architecture',
        # 'self-awareness-web-gpt',
        'survival-instinct',
        'wealth-seeking-inclination',
    ]
    model, tokenizer = load_mdl_tkzr(args.model)
    results_dict = {}
    for steering_mode in ['mean', 'private']:
        args.steering_method = steering_mode
        for dataset_file in datasets:
            dataset_name = dataset_file.replace('.jsonl', '')
            print(f"\n=== Running Experiment on Dataset: {dataset_name} with {args.steering_method} ===")
            args.dataset = dataset_name
            result = run_experiment(args, model, tokenizer)
            results_dict[dataset_name] = result
    
    with open('alignment_results.json', 'w') as f:
        json.dump(results_dict, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Steering Experiments for LLM Alignment")
    
    # Model & Data args
    parser.add_argument("--model", default="meta-llama/Llama-2-7B-chat-hf", help="HuggingFace model ID")
    parser.add_argument("--dataset", default="myopic-reward", help="Dataset name for training and evaluation",
        choices=[
            'coordinate-itself',
            'coordinate-other-ais',
            'coordinate-other-versions',
            'corrigible-less-HHH',
            'corrigible-more-HHH',
            'corrigible-neutral-HHH',
            'myopic-reward',
            'one-box-tendency',
            'power-seeking-inclination',
            'self-awareness-general-ai',
            'self-awareness-good-text-model',
            'self-awareness-text-model',
            'self-awareness-training-architecture',
            'self-awareness-web-gpt',
            'survival-instinct',
            'wealth-seeking-inclination',
        ]
    )
    # Steering Config args
    parser.add_argument("--save_steering_vector", default=True, action='store_true', help="Whether to save the extracted steering vector for later use")
    parser.add_argument("--steering_method", default="private", choices=["mean", "private", "pca"], help="Type of vector aggregation")
    parser.add_argument("--layers", type=int, nargs='+', default=[11, 12, 13, 14, 15], help="Middle layers to apply steering on")
    parser.add_argument("--multipliers", type=float, nargs='+', default=[-2.0, 0.0, 2.0], help="Steering strengths (gamma) to evaluate")
    
    # Privacy Math args
    parser.add_argument("--noise_multiplier", default=0.02, type=float, help="Gaussian noise standard deviation")
    parser.add_argument("--clip", default=20, type=float, help="L2 Norm clipping factor")

    args = parser.parse_args()
    run_experiment(args)
    # run_all(args)