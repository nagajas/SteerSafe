# SteerSafe
This repository implements a framework for aligning Large Language Models (LLMs) at inference-time using Differentially Private (DP) steering vectors. The goal is to steer model behavior (e.g., reducing sycophancy or hallucination) while ensuring that the demonstration data used to create those steering vectors remains private.

## Prerequisites
- Python 3.10+
- PyTorch
- transformers, accelerate, bitsandbytes, steering-vectors

## Usage

```
python main.py \
    --model "meta-llama/Llama-2-7B-chat-hf" \
    --dataset "sycophancy" \
    --steering_method "private" \
    --noise_multiplier 0.02 \
    --clip 20
```

## Evaluation Metrics
- Alignment Accuracy: Measured via contrastive log-probabilities between behavior-matching and non-matching responses.
- Privacy Audit: * FPR (False Positive Rate); FNR (False Negative Rate)

