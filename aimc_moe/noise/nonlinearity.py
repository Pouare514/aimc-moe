from __future__ import annotations
import torch
from torch import Tensor

def apply_iv_nonlinearity(output: Tensor, weights: Tensor, inputs: Tensor, coeff: float) -> Tensor:
    """
    Model I-V curve nonlinearity using second-order Taylor expansion:
    I = G * V + coeff * G^2 * V^2 (scaled)
    
    In matrix form:
    O_noisy = inputs @ weights.T + coeff * (inputs^2) @ (weights^2).T
    """
    if coeff <= 0.0:
        return output
        
    orig_shape = output.shape
    
    # Flatten inputs and output if 3D
    flat_inputs = inputs.reshape(-1, inputs.shape[-1])
    
    # Quadratic correction term
    correction = (flat_inputs ** 2) @ (weights ** 2).t()
    
    # Reshape correction back to original output shape
    correction = correction.reshape(orig_shape)
    
    # Scale correction to keep magnitudes stable
    correction_scale = output.abs().mean().clamp(min=1e-5) / correction.abs().mean().clamp(min=1e-5)
    
    return output + coeff * correction * correction_scale

def _quantize(tensor: Tensor, bits: int, symmetric: bool = True) -> Tensor:
    """Helper to quantize tensor to given number of bits."""
    if bits >= 16 or bits <= 0:
        return tensor
        
    # Find scale
    max_val = tensor.abs().max().clamp(min=1e-5)
    
    if symmetric:
        # Scale to [-1, 1]
        scaled = tensor / max_val
        # Quantize to [-(2^(B-1)-1), 2^(B-1)-1]
        levels = 2 ** (bits - 1) - 1
        quantized = torch.round(scaled * levels) / levels
        return quantized * max_val
    else:
        # Scale to [0, 1]
        min_val = tensor.min()
        scaled = (tensor - min_val) / (max_val - min_val).clamp(min=1e-5)
        levels = 2 ** bits - 1
        quantized = torch.round(scaled * levels) / levels
        return quantized * (max_val - min_val) + min_val

def quantize_adc(tensor: Tensor, bits: int, symmetric: bool = True) -> Tensor:
    """Simulate ADC conversion on matrix-vector output."""
    return _quantize(tensor, bits, symmetric)

def quantize_dac(tensor: Tensor, bits: int, symmetric: bool = True) -> Tensor:
    """Simulate DAC conversion on matrix inputs."""
    return _quantize(tensor, bits, symmetric)

def apply_analytical_ir_drop(
    tile_weights: Tensor,
    wire_resistance: float = 0.5,
    alpha: float = 0.0001,
    beta: float = 0.00001
) -> Tensor:
    """
    Apply systematic position-dependent weight degradation to represent IR drop:
    G_effective = G * (1 - alpha * (i + j) - beta * (i^2 + j^2) * wire_resistance)
    
    Where i is the row index and j is the column index on the crossbar tile.
    """
    if alpha <= 0.0 and beta <= 0.0:
        return tile_weights
        
    rows, cols = tile_weights.shape
    
    i_indices = torch.arange(rows, dtype=tile_weights.dtype, device=tile_weights.device)
    j_indices = torch.arange(cols, dtype=tile_weights.dtype, device=tile_weights.device)
    
    I, J = torch.meshgrid(i_indices, j_indices, indexing='ij')
    
    drop_factor = 1.0 - alpha * (I + J) - beta * (I**2 + J**2) * wire_resistance
    # Keep drop_factor within safe bounds [0, 1]
    drop_factor = drop_factor.clamp(min=0.0, max=1.0)
    
    return tile_weights * drop_factor

