# ==========================================
# Steering Aggregators
# ==========================================

import torch
import math
from dataclasses import dataclass
from typing import Iterable, Callable
from utils import DEVICE

def get_scaled_mean_aggregator():
    """Returns an aggregator for standard, non-private mean steering."""
    def scaled_mean(vp, vn):
        diff = vp - vn
        scale = torch.max(torch.norm(diff, dim=1))
        diff /= scale
        return torch.mean(diff, dim=0)
    return scaled_mean

def get_private_mean_aggregator(clip, noise_multiplier):
    """
    Aggregator for private steering.
    clip + noise_multiplier
    """
    def priv_mean(vp, vn):
        diff = vp - vn
        norms = torch.norm(diff, dim=1)
        
        # 1. DP Clipping: bound the sensitivity of any single demonstration
        scale_factors = torch.clamp(clip / norms, max=1.0).view(-1, 1)
        diff = diff * scale_factors
        diff /= clip 
        
        # 2. Mean Calculation
        mu = torch.mean(diff, dim=0)
        
        # 3. DP Noise Injection
        noise = torch.normal(0, 1.0, size=mu.shape).to(DEVICE)
        return mu + (noise_multiplier * noise)
        
    return priv_mean