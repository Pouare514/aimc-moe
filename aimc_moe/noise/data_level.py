from __future__ import annotations
from typing import Tuple, List, Optional
import torch
from torch import Tensor
from aimc_moe.noise.profiles import NoiseProfile
from aimc_moe.noise.static import apply_d2d_variation, apply_stuck_at_faults
from aimc_moe.noise.dynamic import apply_programming_noise

from aimc_moe.noise.nonlinearity import apply_analytical_ir_drop

def fame_split(weights: Tensor, crossbar_rows: int, crossbar_cols: int) -> List[List[Tensor]]:
    """
    Split the weight matrix (out_features, in_features) into crossbar-sized tiles.
    Because CIM maps input features to rows and output features to columns, we first
    transpose the weight matrix to (in_features, out_features) before splitting.
    
    Returns a 2D list of tiles.
    """
    # PyTorch linear weights are (out_features, in_features)
    in_feats, out_feats = weights.shape[1], weights.shape[0]
    
    transposed = weights.t()
    
    # Split along rows (in_features)
    row_splits = torch.split(transposed, crossbar_rows, dim=0)
    
    tiles = []
    for row_split in row_splits:
        col_splits = torch.split(row_split, crossbar_cols, dim=1)
        tiles.append(list(col_splits))
        
    return tiles

def fame_map(tile: Tensor, noise_profile: NoiseProfile, generator: Optional[torch.Generator] = None) -> Tensor:
    """
    Apply physical non-idealities (D2D variation, SAFs, programming noise) 
    directly to a single crossbar weight tile.
    """
    if noise_profile.is_ideal:
        return tile
        
    # Scale tile to physical conductance range for noise application (e.g. [-1, 1])
    max_val = tile.abs().max().clamp(min=1e-5)
    scaled_tile = tile / max_val
    
    # Apply programming noise (once per deployment)
    noisy_tile = apply_programming_noise(scaled_tile, noise_profile.sigma_prog, generator=generator)
    
    # Apply static device-to-device variation
    noisy_tile = apply_d2d_variation(noisy_tile, noise_profile.sigma_d2d, generator=generator)
    
    # Apply stuck-at faults (SAF)
    noisy_tile = apply_stuck_at_faults(noisy_tile, noise_profile.saf_rate, g_min=-1.0, g_max=1.0, generator=generator)
    
    # Apply analytical IR drop (position-dependent systematic degradation)
    if noise_profile.ir_drop_alpha > 0.0 or noise_profile.ir_drop_beta > 0.0:
        noisy_tile = apply_analytical_ir_drop(
            noisy_tile,
            wire_resistance=noise_profile.wire_resistance,
            alpha=noise_profile.ir_drop_alpha,
            beta=noise_profile.ir_drop_beta
        )
        
    # Scale back to weight domain
    return noisy_tile * max_val

def fame_reorganize(tiles: List[List[Tensor]]) -> Tensor:
    """
    Reassemble the 2D grid of weight tiles back into a single weight matrix.
    Transposes the result back to PyTorch's (out_features, in_features) format.
    """
    row_tensors = []
    for col_list in tiles:
        row_tensors.append(torch.cat(col_list, dim=1))
        
    reassembled = torch.cat(row_tensors, dim=0)
    return reassembled.t()

def apply_data_level_noise(
    weights: Tensor,
    noise_profile: NoiseProfile,
    crossbar_rows: int = 256,
    crossbar_cols: int = 256,
    generator: Optional[torch.Generator] = None
) -> Tensor:
    """
    FAME S-M-R pipeline:
    1. Split: Partition weight matrix into crossbar tiles.
    2. Map: Apply physical non-idealities to each tile.
    3. Reorganize: Reassemble tiles back into a single weight matrix.
    """
    if noise_profile.is_ideal:
        return weights
        
    tiles = fame_split(weights, crossbar_rows, crossbar_cols)
    
    # Apply noise to each tile independently
    for r in range(len(tiles)):
        for c in range(len(tiles[r])):
            tiles[r][c] = fame_map(tiles[r][c], noise_profile, generator=generator)
            
    return fame_reorganize(tiles)
