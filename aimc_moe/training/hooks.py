from __future__ import annotations
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torch import Tensor

from aimc_moe.training.noisy_forward import NoisyLinear

class SensitivityMonitor:
    """
    Monitor layer activations to compute layer-wise noise sensitivity.
    Registers forward hooks to compare the output of a layer under noise vs clean.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks = []
        self.clean_outputs: Dict[str, Tensor] = {}
        self.noisy_outputs: Dict[str, List[Tensor]] = {}
        self._register_hooks()
        
    def _register_hooks(self) -> None:
        for name, module in self.model.named_modules():
            if isinstance(module, NoisyLinear):
                # Register hook
                self.hooks.append(
                    module.register_forward_hook(self._make_hook(name))
                )
                
    def _make_hook(self, name: str):
        def hook_fn(module, inputs, output):
            if module.noise_profile.is_ideal:
                # Save clean output
                self.clean_outputs[name] = output.detach().cpu()
            else:
                # Save noisy output
                self.noisy_outputs.setdefault(name, []).append(output.detach().cpu())
        return hook_fn
        
    def clear(self) -> None:
        self.clean_outputs.clear()
        self.noisy_outputs.clear()
        
    def remove(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        
    def compute_sensitivity(self) -> Dict[str, float]:
        """
        Compute noise sensitivity per layer.
        Sensitivity = 1 / SNR, where SNR = Mean(Clean^2) / Mean((Clean - Noisy)^2)
        Higher value means the layer output is more degraded by noise.
        """
        sensitivity = {}
        for name, clean in self.clean_outputs.items():
            noisy_list = self.noisy_outputs.get(name, [])
            if not noisy_list:
                continue
                
            # Check if shapes match
            all_shapes_match = all(n.shape == clean.shape for n in noisy_list)
            
            if all_shapes_match:
                # Average noisy outputs if multiple samples taken
                noisy = torch.stack(noisy_list).mean(dim=0)
                signal_power = (clean ** 2).mean().item()
                noise_power = ((clean - noisy) ** 2).mean().item()
                
                if noise_power == 0:
                    sens = 0.0
                else:
                    snr = signal_power / noise_power
                    sens = 1.0 / snr if snr > 0 else float("inf")
            else:
                # Shape-independent distribution discrepancy metric (Mean & Std deviation)
                clean_mean = clean.mean().item()
                clean_std = clean.std().clamp(min=1e-5).item()
                
                discrepancies = []
                for n in noisy_list:
                    n_mean = n.mean().item()
                    n_std = n.std().item()
                    # Normalized discrepancy
                    disc = abs(clean_mean - n_mean) / clean_std + abs(clean_std - n_std) / clean_std
                    discrepancies.append(disc)
                sens = sum(discrepancies) / len(discrepancies)
                
            sensitivity[name] = sens
            
        return sensitivity
