from __future__ import annotations
from typing import Optional
import torch
from torch import Tensor

def apply_d2d_variation(weights: Tensor, sigma: float, generator: Optional[torch.Generator] = None) -> Tensor:
    """
    Apply device-to-device conductance variation (mismatch).
    G_noisy = G * (1 + N(0, sigma))
    """
    if sigma <= 0.0:
        return weights
        
    noise = torch.randn(weights.shape, dtype=weights.dtype, device=weights.device, generator=generator)
    return weights * (1.0 + noise * sigma)

def apply_stuck_at_faults(
    weights: Tensor,
    saf_rate: float,
    g_min: float = -1.0,
    g_max: float = 1.0,
    generator: Optional[torch.Generator] = None
) -> Tensor:
    """
    Apply stuck-at-faults (SAF). A percentage of cells are stuck in HRS (g_min) or LRS (g_max).
    saf_rate: probability of a cell being stuck.
    """
    if saf_rate <= 0.0:
        return weights
        
    noisy_weights = weights.clone()
    
    # Generate random probabilities for SAF
    prob = torch.rand(weights.shape, dtype=weights.dtype, device=weights.device, generator=generator)
    
    # 50% stuck at min (HRS), 50% stuck at max (LRS)
    stuck_mask = prob < saf_rate
    stuck_high_mask = stuck_mask & (torch.rand(weights.shape, dtype=weights.dtype, device=weights.device, generator=generator) < 0.5)
    stuck_low_mask = stuck_mask & (~stuck_high_mask)
    
    noisy_weights[stuck_high_mask] = g_max
    noisy_weights[stuck_low_mask] = g_min
    
    return noisy_weights
