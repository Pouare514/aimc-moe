from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
from aimc_moe.backend.crossbar import CrossbarConfig
from aimc_moe.backend.fabric import Fabric3D
from aimc_moe.backend.mapper import MappingResult

@dataclass
class AreaResult:
    total_area_mm2: float          # Total 2D footprint projection area
    active_area_mm2: float         # 3D surface area of only tiles containing mapped weights
    per_tier_area_mm2: Dict[int, float]
    tiles_used: int
    tiles_total: int
    utilization_pct: float

def compute_tile_area(config: CrossbarConfig) -> float:
    """
    Compute area of a single crossbar tile in mm².
    """
    return config.tile_area_mm2

def compute_total_area(fabric: Fabric3D, mapping: MappingResult) -> AreaResult:
    """
    Compute total area and tier breakdowns.
    """
    single_tile_area = fabric.crossbar_config.tile_area_mm2
    
    # 2D projection footprint (mesh_rows * mesh_cols)
    total_area_mm2 = fabric.mesh_rows * fabric.mesh_cols * single_tile_area
    
    # Find all tiles containing mapped operations
    used_tile_coords = set()
    for assign in mapping.assignments:
        used_tile_coords.add(assign.coordinates)
        
    tiles_used = len(used_tile_coords)
    tiles_total = fabric.total_tiles()
    
    active_area_mm2 = tiles_used * single_tile_area
    
    # Per tier area breakdown
    per_tier_area = {}
    for coord in used_tile_coords:
        tier = coord[0]
        per_tier_area[tier] = per_tier_area.get(tier, 0.0) + single_tile_area
        
    # Set default 0.0 for unused tiers in breakdown
    for tier in range(fabric.n_tiers):
        per_tier_area.setdefault(tier, 0.0)
        
    utilization_pct = (tiles_used / tiles_total * 100.0) if tiles_total > 0 else 0.0
    
    return AreaResult(
        total_area_mm2=total_area_mm2,
        active_area_mm2=active_area_mm2,
        per_tier_area_mm2=per_tier_area,
        tiles_used=tiles_used,
        tiles_total=tiles_total,
        utilization_pct=utilization_pct
    )
