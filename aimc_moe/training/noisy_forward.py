from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor

from aimc_moe.noise.profiles import NoiseProfile
from aimc_moe.noise.static import apply_d2d_variation, apply_stuck_at_faults
from aimc_moe.noise.dynamic import apply_read_noise, apply_programming_noise
from aimc_moe.noise.nonlinearity import apply_iv_nonlinearity, quantize_adc, quantize_dac
from aimc_moe.noise.data_level import apply_data_level_noise

class NoisyLinear(nn.Module):
    """
    Drop-in replacement wrapper for nn.Linear that simulates analog computing noise
    and non-idealities in the forward pass. Gradients flow through to the original weights.
    """
    def __init__(self, original_linear: nn.Linear, noise_profile: NoiseProfile):
        super().__init__()
        self.original_linear = original_linear
        self.noise_profile = noise_profile
        self.generator = torch.Generator(device=original_linear.weight.device)
        
    def set_seed(self, seed: int) -> None:
        self.generator.manual_seed(seed)
        
    def forward(self, x: Tensor) -> Tensor:
        if self.noise_profile.is_ideal:
            return self.original_linear(x)
            
        w = self.original_linear.weight
        b = self.original_linear.bias
        
        # 1. Apply static weight variations & programming noise
        # Using data-level mapping (FAME S-M-R style) or simple element-wise noise
        # For fast training, we can do element-wise.
        w_noisy = apply_programming_noise(w, self.noise_profile.sigma_prog, generator=self.generator)
        w_noisy = apply_d2d_variation(w_noisy, self.noise_profile.sigma_d2d, generator=self.generator)
        
        # Apply stuck-at faults (if any)
        # Weight range is normalized dynamically for SAF
        if self.noise_profile.saf_rate > 0.0:
            w_max = w.abs().max().clamp(min=1e-5).item()
            w_noisy = apply_stuck_at_faults(w_noisy, self.noise_profile.saf_rate, 
                                            g_min=-w_max, g_max=w_max, generator=self.generator)
            
        # Apply analytical IR drop (position-dependent systematic degradation)
        if self.noise_profile.ir_drop_alpha > 0.0 or self.noise_profile.ir_drop_beta > 0.0:
            from aimc_moe.noise.nonlinearity import apply_analytical_ir_drop
            w_noisy = apply_analytical_ir_drop(
                w_noisy,
                wire_resistance=self.noise_profile.wire_resistance,
                alpha=self.noise_profile.ir_drop_alpha,
                beta=self.noise_profile.ir_drop_beta
            )
            
        # Bias noise
        b_noisy = b
        if b is not None and self.noise_profile.sigma_prog > 0.0:
            b_noisy = apply_programming_noise(b, self.noise_profile.sigma_prog, generator=self.generator)
            b_noisy = apply_d2d_variation(b_noisy, self.noise_profile.sigma_d2d, generator=self.generator)
            
        # 2. Input Quantization (DAC)
        x_dac = quantize_dac(x, self.noise_profile.dac_bits)
        
        # 3. Core computation (analog MVM)
        out = nn.functional.linear(x_dac, w_noisy, b_noisy)
        
        # 4. I-V Curve Nonlinearity
        if self.noise_profile.nonlinearity_coeff > 0.0:
            out = apply_iv_nonlinearity(out, w_noisy, x_dac, self.noise_profile.nonlinearity_coeff)
            
        # 5. Read noise (thermal/dynamic)
        out = apply_read_noise(out, self.noise_profile.sigma_read, generator=self.generator)
        
        # 6. Output Quantization (ADC)
        out_adc = quantize_adc(out, self.noise_profile.adc_bits)
        
        return out_adc

def wrap_model_noisy(model: nn.Module, noise_profile: NoiseProfile) -> nn.Module:
    """
    Recursively replaces all nn.Linear layers in a model with NoisyLinear wrappers.
    Modifies the model in-place.
    """
    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            # Do not wrap output heads or routers if we want to keep them digital
            # (Usually, output classification heads and router softmax run in digital domain).
            if "router" in name or "output" in name or "gate" in name:
                # Keep original
                continue
            setattr(model, name, NoisyLinear(child, noise_profile))
        else:
            wrap_model_noisy(child, noise_profile)
    return model

def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively removes NoisyLinear wrappers, restoring the original nn.Linear layers.
    Modifies the model in-place.
    """
    for name, child in model.named_children():
        if isinstance(child, NoisyLinear):
            setattr(model, name, child.original_linear)
        else:
            unwrap_model(child)
    return model
