from __future__ import annotations
from typing import Optional
import torch
from torch import Tensor

def apply_read_noise(output: Tensor, sigma: float, generator: Optional[torch.Generator] = None) -> Tensor:
    """
    Apply additive read noise to the output of a matrix-vector multiplication.
    O_noisy = O + N(0, sigma * avg_output_magnitude)
    """
    if sigma <= 0.0:
        return output
        
    ref_magnitude = output.abs().mean().clamp(min=1e-5)
    noise = torch.randn(output.shape, dtype=output.dtype, device=output.device, generator=generator)
    return output + noise * (sigma * ref_magnitude)

def apply_programming_noise(weights: Tensor, sigma: float, generator: Optional[torch.Generator] = None) -> Tensor:
    """
    Apply multiplicative programming noise to weights (applied once at deploy time).
    W_noisy = W * (1 + N(0, sigma))
    """
    if sigma <= 0.0:
        return weights
        
    noise = torch.randn(weights.shape, dtype=weights.dtype, device=weights.device, generator=generator)
    return weights * (1.0 + noise * sigma)

def apply_drift(
    weights: Tensor,
    nu: float,
    nu_std: float,
    t_inference: float,
    t_0: float = 1.0,
    generator: Optional[torch.Generator] = None
) -> Tensor:
    """
    Apply PCM-style power-law temporal drift.
    G(t) = G_0 * (t / t_0)^(-nu)
    nu is sampled per-element from N(nu, nu_std), clamped to >= 0.0.
    """
    if nu <= 0.0 or t_inference <= t_0:
        return weights
        
    # Sample nu values per-element
    nu_vals = torch.randn(weights.shape, dtype=weights.dtype, device=weights.device, generator=generator)
    nu_vals = (nu_vals * nu_std + nu).clamp(min=0.0)
    
    # Compute drift factor
    drift_factor = (t_inference / t_0) ** (-nu_vals)
    return weights * drift_factor
