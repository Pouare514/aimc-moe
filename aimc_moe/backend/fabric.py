from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Any
from aimc_moe.backend.crossbar import CrossbarTile, CrossbarConfig
from aimc_moe.backend.technology import get_technology

@dataclass
class Fabric3D:
    mesh_rows: int = 8
    mesh_cols: int = 8
    n_tiers: int = 1
    crossbar_config: CrossbarConfig = field(default_factory=CrossbarConfig)
    one_tier_at_a_time: bool = False
    
    tiles: Dict[Tuple[int, int, int], CrossbarTile] = field(default_factory=dict)
    # Key: (tier, row, col)
    
    def __post_init__(self):
        if not self.tiles:
            for tier in range(self.n_tiers):
                for row in range(self.mesh_rows):
                    for col in range(self.mesh_cols):
                        tile_id = f"tile_t{tier}_r{row}_c{col}"
                        self.tiles[(tier, row, col)] = CrossbarTile(
                            tile_id=tile_id,
                            config=self.crossbar_config,
                            tier=tier,
                            position=(row, col)
                        )
                        
    def total_tiles(self) -> int:
        return len(self.tiles)
        
    def get_tile(self, tier: int, row: int, col: int) -> CrossbarTile:
        tile = self.tiles.get((tier, row, col))
        if tile is None:
            raise KeyError(f"No tile at (tier={tier}, row={row}, col={col}).")
        return tile
        
    def available_tiles(self, tier: Optional[int] = None) -> List[CrossbarTile]:
        """Returns all tiles, optionally filtered by tier, that have remaining capacity."""
        avail = []
        for coord, tile in self.tiles.items():
            if tier is not None and coord[0] != tier:
                continue
            if tile.utilization < 1.0:
                avail.append(tile)
        return avail
        
    def tiles_on_tier(self, tier: int) -> List[CrossbarTile]:
        return [tile for coord, tile in self.tiles.items() if coord[0] == tier]
        
    def total_area_mm2(self) -> float:
        # Footprint is 2D projection (mesh_rows * mesh_cols)
        # 3D stack doesn't increase 2D area (footprint) but increases active area
        # We report footprint area.
        single_area = self.crossbar_config.tile_area_mm2
        return self.mesh_rows * self.mesh_cols * single_area
        
    def total_capacity(self) -> int:
        return self.total_tiles() * self.crossbar_config.capacity
        
    def summary(self) -> str:
        lines = [
            "=================== Fabric3D Summary ===================",
            f"Dimensions: {self.mesh_rows} rows x {self.mesh_cols} cols x {self.n_tiers} tiers",
            f"Total Tiles: {self.total_tiles()}",
            f"Crossbar Size: {self.crossbar_config.rows}x{self.crossbar_config.cols}",
            f"Device Tech: {self.crossbar_config.technology.name} ({self.crossbar_config.technology.device_type})",
            f"Footprint Area: {self.total_area_mm2():.4f} mm²",
            f"Total Weight Capacity: {self.total_capacity():,} weights",
            f"One-Tier-At-A-Time Constraint: {self.one_tier_at_a_time}",
            "========================================================"
        ]
        return "\n".join(lines)
        
    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> Fabric3D:
        rows = config_dict.get("mesh_rows", 8)
        cols = config_dict.get("mesh_cols", 8)
        tiers = config_dict.get("n_tiers", 1)
        one_tier = config_dict.get("one_tier_at_a_time", False)
        
        # Crossbar config
        xb_dict = config_dict.get("crossbar", {})
        xb_rows = xb_dict.get("rows", 256)
        xb_cols = xb_dict.get("cols", 256)
        
        # Technology config
        tech_name = xb_dict.get("technology", "rram_hfo2")
        tech = get_technology(tech_name)
        
        xb_config = CrossbarConfig(rows=xb_rows, cols=xb_cols, technology=tech)
        return cls(mesh_rows=rows, mesh_cols=cols, n_tiers=tiers, 
                   crossbar_config=xb_config, one_tier_at_a_time=one_tier)
