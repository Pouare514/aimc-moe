from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from typing import Dict, Tuple, Optional

from aimc_moe.backend.fabric import Fabric3D
from aimc_moe.backend.mapper import MappingResult

@dataclass
class ThermalConfig:
    r_horizontal: float = 10.0          # Horizontal thermal resistance between adjacent tiles (°C/W)
    r_vertical: float = 15.0            # Vertical thermal resistance between stacked tiers (°C/W)
    r_convective: float = 2.0           # Resistance from Tier 0 to ambient cooling sink (°C/W)
    t_ambient: float = 25.0             # Ambient cooling temperature (°C)

@dataclass
class ThermalResult:
    tile_temperatures: Dict[str, float]  # tile_id -> temperature (°C)
    max_temperature: float
    min_temperature: float
    average_temperature: float
    is_thermal_unsafe: bool

def solve_steady_state_temperatures(
    fabric: Fabric3D,
    mapping: MappingResult,
    config: Optional[ThermalConfig] = None,
    ops_per_second: float = 1e9,         # Throughput in operations/second (e.g. 1 GOPS)
    thermal_threshold: float = 85.0
) -> ThermalResult:
    """
    Solve the steady-state thermal model G * T = B.
    
    1. Grid tiles are mapped to linear matrix indexes:
       idx = tier * (rows * cols) + row * cols + col
    2. Vertical connections exist between (t, r, c) and (t+1, r, c).
    3. Horizontal connections exist on the same tier to north, south, east, west.
    4. Convective ambient link connects Tier 0 to t_ambient.
    5. Power is calculated based on tile utilization and energy per MAC.
    """
    if config is None:
        config = ThermalConfig()
        
    rows = fabric.mesh_rows
    cols = fabric.mesh_cols
    tiers = fabric.n_tiers
    n_nodes = rows * cols * tiers
    
    # Pre-calculate node indices
    def get_node_idx(t: int, r: int, c: int) -> int:
        return t * (rows * cols) + r * cols + c
        
    def get_coords(idx: int) -> Tuple[int, int, int]:
        t = idx // (rows * cols)
        rem = idx % (rows * cols)
        r = rem // cols
        c = rem % cols
        return t, r, c
        
    # Build Conductance matrix G (size N x N) and boundary vector B (size N)
    G = np.zeros((n_nodes, n_nodes))
    B = np.zeros(n_nodes)
    
    # Calculate Power vector P (in Watts)
    # Total active parameters / weights in the model
    # Each mapped block consumes power proportional to its utilization
    tech = fabric.crossbar_config.technology
    # Energy per operation (in Joules)
    e_op_j = tech.energy_per_mac * 1e-12  # pJ -> J
    
    # Power for each tile
    P = np.zeros(n_nodes)
    for coord, tile in fabric.tiles.items():
        idx = get_node_idx(*coord)
        # Power = utilization * ops_per_second * energy_per_mac (Watts)
        P[idx] = tile.utilization * ops_per_second * e_op_j
        
    # Connect nodes
    for idx in range(n_nodes):
        t, r, c = get_coords(idx)
        
        # Conduction paths (conductance = 1/R)
        cond_sum = 0.0
        
        # Convective link to ambient heatsink (only on Tier 0 - bottom tier close to cooler)
        if t == 0:
            cond_c = 1.0 / config.r_convective
            G[idx, idx] += cond_c
            B[idx] += cond_c * config.t_ambient
            
        # Horizontal neighbors (North, South, East, West)
        neighbors_h = []
        if r > 0: neighbors_h.append(get_node_idx(t, r - 1, c))
        if r < rows - 1: neighbors_h.append(get_node_idx(t, r + 1, c))
        if c > 0: neighbors_h.append(get_node_idx(t, r, c - 1))
        if c < cols - 1: neighbors_h.append(get_node_idx(t, r, c + 1))
        
        cond_h = 1.0 / config.r_horizontal
        for neigh in neighbors_h:
            G[idx, neigh] -= cond_h
            G[idx, idx] += cond_h
            
        # Vertical neighbors (Above, Below)
        neighbors_v = []
        if t > 0: neighbors_v.append(get_node_idx(t - 1, r, c))
        if t < tiers - 1: neighbors_v.append(get_node_idx(t + 1, r, c))
        
        cond_v = 1.0 / config.r_vertical
        for neigh in neighbors_v:
            G[idx, neigh] -= cond_v
            G[idx, idx] += cond_v
            
        # Add generated heat to right hand side boundary vector
        B[idx] += P[idx]
        
    # Solve steady state temperatures: T = G^-1 * B
    try:
        T = np.linalg.solve(G, B)
    except np.linalg.LinAlgError:
        # Fallback if singular matrix (should not happen for physical conductive system)
        T = np.ones(n_nodes) * config.t_ambient
        
    # Map back to tile_ids
    tile_temperatures = {}
    for idx, temp in enumerate(T):
        t, r, c = get_coords(idx)
        tile = fabric.get_tile(t, r, c)
        tile_temperatures[tile.tile_id] = float(temp)
        
    max_temp = float(np.max(T))
    min_temp = float(np.min(T))
    avg_temp = float(np.mean(T))
    is_unsafe = max_temp > thermal_threshold
    
    return ThermalResult(
        tile_temperatures=tile_temperatures,
        max_temperature=max_temp,
        min_temperature=min_temp,
        average_temperature=avg_temp,
        is_thermal_unsafe=is_unsafe
    )
