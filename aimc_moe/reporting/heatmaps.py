from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn
from torch import Tensor

from aimc_moe.ir.graph import AIMCGraph
from aimc_moe.noise.profiles import NoiseProfile, IDEAL
from aimc_moe.training.noisy_forward import wrap_model_noisy, unwrap_model
from aimc_moe.training.hooks import SensitivityMonitor

def generate_sensitivity_data(
    graph: AIMCGraph,
    noise_profile: NoiseProfile,
    model: nn.Module,
    sample_input: Tensor,
    n_samples: int = 5
) -> Dict[str, float]:
    """
    Compute layer sensitivity under noise.
    
    Runs 1 clean forward pass to establish a baseline, then n_samples noisy
    forward passes to record output variance. Sensitivity is calculated as 1/SNR.
    """
    model.eval()
    
    # 1. Establish clean baseline
    # Wrap model with IDEAL profile (no noise)
    wrap_model_noisy(model, IDEAL)
    monitor = SensitivityMonitor(model)
    
    # Clean forward pass
    with torch.no_grad():
        _ = model(sample_input)
        
    # 2. Run noisy forward passes
    # Re-wrap model with target noise profile
    unwrap_model(model)
    wrap_model_noisy(model, noise_profile)
    
    # Keep the same monitor hooks, but update the underlying layers
    # To be safe and clean, let's remove the old monitor and build a new one.
    monitor.remove()
    monitor = SensitivityMonitor(model)
    
    # Clean outputs are still saved in our logic? No, removing the monitor cleared hooks.
    # Let's write this properly:
    # We want a single monitor to capture both clean and noisy.
    # Since NoisyLinear layers remain the same, we can just update their noise profile!
    
    unwrap_model(model)
    wrap_model_noisy(model, noise_profile)
    # Get all NoisyLinear modules and make them IDEAL for 1 pass
    noisy_layers = []
    for m in model.modules():
        if m.__class__.__name__ == "NoisyLinear":
            noisy_layers.append(m)
            
    # Set to ideal for clean pass
    for layer in noisy_layers:
        layer.noise_profile = IDEAL
        
    # Setup monitor
    monitor = SensitivityMonitor(model)
    
    with torch.no_grad():
        _ = model(sample_input)
        
    # Set back to target noise profile for noisy passes
    for layer in noisy_layers:
        layer.noise_profile = noise_profile
        
    # Run n_samples noisy passes
    with torch.no_grad():
        for _ in range(n_samples):
            # Seed update happens automatically inside forward due to torch.Generator
            _ = model(sample_input)
            
    # 3. Compute metrics
    sensitivity = monitor.compute_sensitivity()
    
    # Cleanup
    monitor.remove()
    unwrap_model(model)
    
    # Map from NoisyLinear module path names to the corresponding graph op IDs.
    # The NoisyLinear names are like 'layers.0.attn.q_proj'
    # While graph op ids are like 'layers.0.attn.q_proj' as well.
    # So they match directly!
    return sensitivity
