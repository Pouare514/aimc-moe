from __future__ import annotations
import math
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
from aimc_moe.ir.ops import AIMCOp, OpType
from aimc_moe.ir.graph import AIMCGraph
from aimc_moe.backend.fabric import Fabric3D
from aimc_moe.backend.crossbar import CrossbarTile

class MappingStrategy(Enum):
    SEQUENTIAL = "sequential"
    EXPERT_COLOCATE = "expert_colocate"
    THERMAL_AWARE = "thermal_aware"

@dataclass
class TileAssignment:
    op_id: str
    tile_id: str
    block_row: int      # row-block index in partitioned matrix
    block_col: int      # col-block index
    rows_used: int
    cols_used: int
    coordinates: Tuple[int, int, int]  # (tier, row, col)
    assigned_range: Tuple[int, int, int, int]  # (row_start, row_end, col_start, col_end) on tile

@dataclass
class MappingResult:
    assignments: List[TileAssignment] = field(default_factory=list)
    total_tiles_used: int = 0
    total_crossbars_needed: int = 0
    average_utilization: float = 0.0
    unmapped_ops: List[str] = field(default_factory=list)
    
    _op_assignments: Dict[str, List[TileAssignment]] = field(init=False, default_factory=dict, repr=False)
    _tile_assignments: Dict[str, List[TileAssignment]] = field(init=False, default_factory=dict, repr=False)
    
    def __post_init__(self):
        self._op_assignments = {}
        self._tile_assignments = {}
        for assign in self.assignments:
            self._op_assignments.setdefault(assign.op_id, []).append(assign)
            self._tile_assignments.setdefault(assign.tile_id, []).append(assign)
            
    def get_op_tiles(self, op_id: str) -> List[TileAssignment]:
        return self._op_assignments.get(op_id, [])
        
    def get_tile_ops(self, tile_id: str) -> List[TileAssignment]:
        return self._tile_assignments.get(tile_id, [])
        
    def summary(self) -> str:
        lines = [
            "=================== Mapping Result ===================",
            f"Crossbar Tiles Used: {self.total_tiles_used}",
            f"Total Crossbars Needed: {self.total_crossbars_needed}",
            f"Average Tile Utilization: {self.average_utilization:.2%}",
            f"Unmapped Operations: {len(self.unmapped_ops)}"
        ]
        if self.unmapped_ops:
            lines.append("  Unmapped IDs:")
            for op_id in self.unmapped_ops[:5]:
                lines.append(f"    - {op_id}")
            if len(self.unmapped_ops) > 5:
                lines.append(f"    - ... and {len(self.unmapped_ops) - 5} more")
        lines.append("======================================================")
        return "\n".join(lines)

def compute_tiling(weight_shape: Tuple[int, ...], crossbar_rows: int, crossbar_cols: int) -> List[Tuple[int, int, int, int, int, int]]:
    """
    Decompose a weight matrix of shape (in_features, out_features) into blocks of size <= (crossbar_rows, crossbar_cols).
    Returns a list of: (block_row_idx, block_col_idx, row_start, row_end, col_start, col_end)
    """
    if len(weight_shape) < 2:
        # For 1D weights (e.g. bias), skip tiling or treat as 1xN
        in_feats = 1
        out_feats = weight_shape[0]
    else:
        # Standard linear weights: (out_features, in_features) in PyTorch linear layers,
        # but in CIM, we typically map input features to rows and output features to columns.
        # So in_features represents the input dimension, out_features the output dimension.
        # Let's map PyTorch linear weight W which is (out_features, in_features).
        # W^T is (in_features, out_features).
        out_feats, in_feats = weight_shape[0], weight_shape[1]
        
    n_row_blocks = math.ceil(in_feats / crossbar_rows)
    n_col_blocks = math.ceil(out_feats / crossbar_cols)
    
    tiling = []
    for r_idx in range(n_row_blocks):
        r_start = r_idx * crossbar_rows
        r_end = min(r_start + crossbar_rows, in_feats)
        rows_used = r_end - r_start
        
        for c_idx in range(n_col_blocks):
            c_start = c_idx * crossbar_cols
            c_end = min(c_start + crossbar_cols, out_feats)
            cols_used = c_end - c_start
            
            tiling.append((r_idx, c_idx, r_start, r_end, c_start, c_end))
            
    return tiling

def map_graph_to_fabric(
    graph: AIMCGraph,
    fabric: Fabric3D,
    strategy: MappingStrategy = MappingStrategy.SEQUENTIAL
) -> MappingResult:
    """
    Main mapping function that places ops onto physical fabric tiles.
    """
    mappable_ops = graph.get_mappable_ops()
    xb_rows = fabric.crossbar_config.rows
    xb_cols = fabric.crossbar_config.cols
    
    assignments = []
    unmapped_ops = []
    
    # Reset fabric tile cursors
    for tile in fabric.tiles.values():
        tile.assigned_ops.clear()
        tile._current_row_ptr = 0
        tile._current_col_ptr = 0
        
    total_crossbars_needed = 0
    
    if strategy == MappingStrategy.SEQUENTIAL:
        # Standard topological mapping
        ordered_ops = graph.topological_order()
        mappable_ordered = [op for op in ordered_ops if op.is_mappable]
        
        tile_list = list(fabric.tiles.values())
        tile_idx = 0
        
        for op in mappable_ordered:
            tiling = compute_tiling(op.shape, xb_rows, xb_cols)
            total_crossbars_needed += len(tiling)
            
            op_assignments = []
            success = True
            
            for r_idx, c_idx, r_start, r_end, c_start, c_end in tiling:
                rows_needed = r_end - r_start
                cols_needed = c_end - c_start
                
                # Find a tile with space
                placed = False
                attempts = 0
                while not placed and attempts < len(tile_list):
                    current_tile = tile_list[tile_idx]
                    if current_tile.can_fit(rows_needed, cols_needed):
                        rng = current_tile.assign(op.id, rows_needed, cols_needed)
                        assign = TileAssignment(
                            op_id=op.id,
                            tile_id=current_tile.tile_id,
                            block_row=r_idx,
                            block_col=c_idx,
                            rows_used=rows_needed,
                            cols_used=cols_needed,
                            coordinates=(current_tile.tier, current_tile.position[0], current_tile.position[1]),
                            assigned_range=rng
                        )
                        op_assignments.append(assign)
                        placed = True
                    else:
                        # Move to next tile
                        tile_idx = (tile_idx + 1) % len(tile_list)
                        attempts += 1
                        
                if not placed:
                    unmapped_ops.append(op.id)
                    success = False
                    break
                    
            if success:
                assignments.extend(op_assignments)
                
    elif strategy == MappingStrategy.EXPERT_COLOCATE:
        # Group experts of same layer together on the same tier if possible
        # Tiers are allocated to layer-specific experts
        ordered_ops = graph.topological_order()
        mappable_ordered = [op for op in ordered_ops if op.is_mappable]
        
        # Split into shared layers (LINEAR but not MOE_EXPERT) and expert layers (MOE_EXPERT)
        shared_ops = [op for op in mappable_ordered if op.op_type != OpType.MOE_EXPERT]
        expert_ops = [op for op in mappable_ordered if op.op_type == OpType.MOE_EXPERT]
        
        # Group experts by (layer_id, expert_id)
        # We want to dedicate specific tiers or tiles to experts
        # Place shared ops first starting from tier 0
        tile_list = list(fabric.tiles.values())
        tile_idx = 0
        
        for op in shared_ops:
            tiling = compute_tiling(op.shape, xb_rows, xb_cols)
            total_crossbars_needed += len(tiling)
            op_assignments = []
            success = True
            
            for r_idx, c_idx, r_start, r_end, c_start, c_end in tiling:
                rows_needed = r_end - r_start
                cols_needed = c_end - c_start
                placed = False
                attempts = 0
                while not placed and attempts < len(tile_list):
                    current_tile = tile_list[tile_idx]
                    if current_tile.can_fit(rows_needed, cols_needed):
                        rng = current_tile.assign(op.id, rows_needed, cols_needed)
                        assign = TileAssignment(
                            op_id=op.id,
                            tile_id=current_tile.tile_id,
                            block_row=r_idx,
                            block_col=c_idx,
                            rows_used=rows_needed,
                            cols_used=cols_needed,
                            coordinates=(current_tile.tier, current_tile.position[0], current_tile.position[1]),
                            assigned_range=rng
                        )
                        op_assignments.append(assign)
                        placed = True
                    else:
                        tile_idx = (tile_idx + 1) % len(tile_list)
                        attempts += 1
                if not placed:
                    unmapped_ops.append(op.id)
                    success = False
                    break
            if success:
                assignments.extend(op_assignments)
                
        # Now place expert layers. We try to map each expert_id to a specific tier
        # if we have multiple tiers. If n_tiers == 1, we just do sequential.
        if fabric.n_tiers > 1:
            # Let's map experts to tiers. Let tier_idx = (expert_id % (n_tiers - 1)) + 1
            # Tier 0 is saved for shared ops.
            for op in expert_ops:
                tiling = compute_tiling(op.shape, xb_rows, xb_cols)
                total_crossbars_needed += len(tiling)
                
                exp_id = op.expert_id if op.expert_id is not None else 0
                target_tier = (exp_id % (fabric.n_tiers - 1)) + 1
                
                tier_tiles = fabric.tiles_on_tier(target_tier)
                tier_tile_idx = 0
                
                op_assignments = []
                success = True
                
                for r_idx, c_idx, r_start, r_end, c_start, c_end in tiling:
                    rows_needed = r_end - r_start
                    cols_needed = c_end - c_start
                    placed = False
                    attempts = 0
                    while not placed and attempts < len(tier_tiles):
                        current_tile = tier_tiles[tier_tile_idx]
                        if current_tile.can_fit(rows_needed, cols_needed):
                            rng = current_tile.assign(op.id, rows_needed, cols_needed)
                            assign = TileAssignment(
                                op_id=op.id,
                                tile_id=current_tile.tile_id,
                                block_row=r_idx,
                                block_col=c_idx,
                                rows_used=rows_needed,
                                cols_used=cols_needed,
                                coordinates=(current_tile.tier, current_tile.position[0], current_tile.position[1]),
                                assigned_range=rng
                            )
                            op_assignments.append(assign)
                            placed = True
                        else:
                            tier_tile_idx = (tier_tile_idx + 1) % len(tier_tiles)
                            attempts += 1
                    if not placed:
                        # Fallback to any tier
                        all_tiles = list(fabric.tiles.values())
                        all_tile_idx = 0
                        placed_fb = False
                        attempts_fb = 0
                        while not placed_fb and attempts_fb < len(all_tiles):
                            current_tile = all_tiles[all_tile_idx]
                            if current_tile.can_fit(rows_needed, cols_needed):
                                rng = current_tile.assign(op.id, rows_needed, cols_needed)
                                assign = TileAssignment(
                                    op_id=op.id,
                                    tile_id=current_tile.tile_id,
                                    block_row=r_idx,
                                    block_col=c_idx,
                                    rows_used=rows_needed,
                                    cols_used=cols_needed,
                                    coordinates=(current_tile.tier, current_tile.position[0], current_tile.position[1]),
                                    assigned_range=rng
                                )
                                op_assignments.append(assign)
                                placed_fb = True
                            else:
                                all_tile_idx = (all_tile_idx + 1) % len(all_tiles)
                                attempts_fb += 1
                        if not placed_fb:
                            unmapped_ops.append(op.id)
                            success = False
                            break
                if success:
                    assignments.extend(op_assignments)
        else:
            # Fall back to sequential
            for op in expert_ops:
                tiling = compute_tiling(op.shape, xb_rows, xb_cols)
                total_crossbars_needed += len(tiling)
                op_assignments = []
                success = True
                for r_idx, c_idx, r_start, r_end, c_start, c_end in tiling:
                    rows_needed = r_end - r_start
                    cols_needed = c_end - c_start
                    placed = False
                    attempts = 0
                    while not placed and attempts < len(tile_list):
                        current_tile = tile_list[tile_idx]
                        if current_tile.can_fit(rows_needed, cols_needed):
                            rng = current_tile.assign(op.id, rows_needed, cols_needed)
                            assign = TileAssignment(
                                op_id=op.id,
                                tile_id=current_tile.tile_id,
                                block_row=r_idx,
                                block_col=c_idx,
                                rows_used=rows_needed,
                                cols_used=cols_needed,
                                coordinates=(current_tile.tier, current_tile.position[0], current_tile.position[1]),
                                assigned_range=rng
                            )
                            op_assignments.append(assign)
                            placed = True
                        else:
                            tile_idx = (tile_idx + 1) % len(tile_list)
                            attempts += 1
                    if not placed:
                        unmapped_ops.append(op.id)
                        success = False
                        break
                if success:
                    assignments.extend(op_assignments)
                    
    elif strategy == MappingStrategy.THERMAL_AWARE:
        # Place shared layers on tier 0 (closest to heat sink/cooling).
        # Experts (highly dynamic, higher power density) are mapped on higher tiers (tier 1, 2, ...).
        # In a thermal model, inner tiers get hotter, so we distribute the thermal load evenly.
        # Place shared ops on tier 0.
        tier0_tiles = fabric.tiles_on_tier(0)
        tier0_idx = 0
        
        ordered_ops = graph.topological_order()
        mappable_ordered = [op for op in ordered_ops if op.is_mappable]
        
        shared_ops = [op for op in mappable_ordered if op.op_type != OpType.MOE_EXPERT]
        expert_ops = [op for op in mappable_ordered if op.op_type == OpType.MOE_EXPERT]
        
        for op in shared_ops:
            tiling = compute_tiling(op.shape, xb_rows, xb_cols)
            total_crossbars_needed += len(tiling)
            op_assignments = []
            success = True
            
            for r_idx, c_idx, r_start, r_end, c_start, c_end in tiling:
                rows_needed = r_end - r_start
                cols_needed = c_end - c_start
                placed = False
                attempts = 0
                while not placed and attempts < len(tier0_tiles):
                    current_tile = tier0_tiles[tier0_idx]
                    if current_tile.can_fit(rows_needed, cols_needed):
                        rng = current_tile.assign(op.id, rows_needed, cols_needed)
                        assign = TileAssignment(
                            op_id=op.id,
                            tile_id=current_tile.tile_id,
                            block_row=r_idx,
                            block_col=c_idx,
                            rows_used=rows_needed,
                            cols_used=cols_needed,
                            coordinates=(current_tile.tier, current_tile.position[0], current_tile.position[1]),
                            assigned_range=rng
                        )
                        op_assignments.append(assign)
                        placed = True
                    else:
                        tier0_idx = (tier0_idx + 1) % len(tier0_tiles)
                        attempts += 1
                if not placed:
                    # Fallback to any other tier
                    all_tiles = list(fabric.tiles.values())
                    all_tile_idx = 0
                    placed_fb = False
                    attempts_fb = 0
                    while not placed_fb and attempts_fb < len(all_tiles):
                        current_tile = all_tiles[all_tile_idx]
                        if current_tile.can_fit(rows_needed, cols_needed):
                            rng = current_tile.assign(op.id, rows_needed, cols_needed)
                            assign = TileAssignment(
                                op_id=op.id,
                                tile_id=current_tile.tile_id,
                                block_row=r_idx,
                                block_col=c_idx,
                                rows_used=rows_needed,
                                cols_used=cols_needed,
                                coordinates=(current_tile.tier, current_tile.position[0], current_tile.position[1]),
                                assigned_range=rng
                            )
                            op_assignments.append(assign)
                            placed_fb = True
                        else:
                            all_tile_idx = (all_tile_idx + 1) % len(all_tiles)
                            attempts_fb += 1
                    if not placed_fb:
                        unmapped_ops.append(op.id)
                        success = False
                        break
            if success:
                assignments.extend(op_assignments)
                
        # Place experts. We distribute them across tiers >= 1 to balance thermal density.
        other_tiers = list(range(1, fabric.n_tiers)) if fabric.n_tiers > 1 else [0]
        
        # Sort experts: let's group by expert_id. Alternate expert placement.
        expert_tiles_pool = []
        for t in other_tiers:
            expert_tiles_pool.extend(fabric.tiles_on_tier(t))
        pool_idx = 0
        
        for op in expert_ops:
            tiling = compute_tiling(op.shape, xb_rows, xb_cols)
            total_crossbars_needed += len(tiling)
            op_assignments = []
            success = True
            
            for r_idx, c_idx, r_start, r_end, c_start, c_end in tiling:
                rows_needed = r_end - r_start
                cols_needed = c_end - c_start
                placed = False
                attempts = 0
                while not placed and attempts < len(expert_tiles_pool):
                    current_tile = expert_tiles_pool[pool_idx]
                    if current_tile.can_fit(rows_needed, cols_needed):
                        rng = current_tile.assign(op.id, rows_needed, cols_needed)
                        assign = TileAssignment(
                            op_id=op.id,
                            tile_id=current_tile.tile_id,
                            block_row=r_idx,
                            block_col=c_idx,
                            rows_used=rows_needed,
                            cols_used=cols_needed,
                            coordinates=(current_tile.tier, current_tile.position[0], current_tile.position[1]),
                            assigned_range=rng
                        )
                        op_assignments.append(assign)
                        placed = True
                    else:
                        pool_idx = (pool_idx + 1) % len(expert_tiles_pool)
                        attempts += 1
                if not placed:
                    # Fallback to any tile
                    all_tiles = list(fabric.tiles.values())
                    all_tile_idx = 0
                    placed_fb = False
                    attempts_fb = 0
                    while not placed_fb and attempts_fb < len(all_tiles):
                        current_tile = all_tiles[all_tile_idx]
                        if current_tile.can_fit(rows_needed, cols_needed):
                            rng = current_tile.assign(op.id, rows_needed, cols_needed)
                            assign = TileAssignment(
                                op_id=op.id,
                                tile_id=current_tile.tile_id,
                                block_row=r_idx,
                                block_col=c_idx,
                                rows_used=rows_needed,
                                cols_used=cols_needed,
                                coordinates=(current_tile.tier, current_tile.position[0], current_tile.position[1]),
                                assigned_range=rng
                            )
                            op_assignments.append(assign)
                            placed_fb = True
                        else:
                            all_tile_idx = (all_tile_idx + 1) % len(all_tiles)
                            attempts_fb += 1
                    if not placed_fb:
                        unmapped_ops.append(op.id)
                        success = False
                        break
            if success:
                assignments.extend(op_assignments)
                
    # Calculate global metrics
    used_tile_ids = {a.tile_id for a in assignments}
    total_tiles_used = len(used_tile_ids)
    
    utilizations = [fabric.tiles[coord].utilization for coord in fabric.tiles if fabric.tiles[coord].utilization > 0.0]
    average_utilization = sum(utilizations) / len(utilizations) if utilizations else 0.0
    
    return MappingResult(
        assignments=assignments,
        total_tiles_used=total_tiles_used,
        total_crossbars_needed=total_crossbars_needed,
        average_utilization=average_utilization,
        unmapped_ops=unmapped_ops
    )
