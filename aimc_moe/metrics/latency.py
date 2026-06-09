from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, Optional
from aimc_moe.ir.ops import AIMCOp, OpType
from aimc_moe.ir.graph import AIMCGraph
from aimc_moe.backend.fabric import Fabric3D
from aimc_moe.backend.mapper import MappingResult
from aimc_moe.backend.interconnect import InterconnectModel

@dataclass
class LatencyResult:
    per_token_ns: float
    per_sequence_ns: float
    per_op_ns: Dict[str, float]  # op_id -> latency
    breakdown: Dict[str, float]  # 'dac', 'crossbar', 'adc', 'activation', 'interconnect', 'control'
    seq_len: int
    batch_size: int

def compute_op_latency(
    op: AIMCOp,
    mapping: MappingResult,
    fabric: Fabric3D,
    interconnect: Optional[InterconnectModel] = None
) -> float:
    """
    Compute latency for a single mappable op in ns.
    
    T_op = dac_bits × (T_DAC + T_crossbar + T_ADC) × max_blocks_on_tile + T_activation + T_control
    
    Plus interconnect routing latency if tiled across multiple tiles.
    """
    op_assigns = mapping.get_op_tiles(op.id)
    if not op_assigns:
        return 0.0
        
    tech = fabric.crossbar_config.technology
    
    # Count how many blocks of this op are mapped to each tile
    tile_counts = {}
    for assign in op_assigns:
        tile_counts[assign.tile_id] = tile_counts.get(assign.tile_id, 0) + 1
        
    max_blocks = max(tile_counts.values()) if tile_counts else 0
    
    # Base computation cycles (bit-serial input processing)
    comp_latency = tech.dac_bits * max_blocks * tech.cycle_time_ns
    overhead = tech.t_activation + tech.t_control
    
    base_latency = comp_latency + overhead
    
    # Interconnect overhead (NoC hops between involved tiles)
    if interconnect is not None and len(tile_counts) > 1:
        tiles_list = []
        for coord, tile in fabric.tiles.items():
            if tile.tile_id in tile_counts:
                tiles_list.append(tile)
                
        max_interconnect_lat = 0.0
        for i in range(len(tiles_list)):
            for j in range(i + 1, len(tiles_list)):
                lat = interconnect.latency_between(tiles_list[i], tiles_list[j])
                max_interconnect_lat = max(max_interconnect_lat, lat)
                
        base_latency += max_interconnect_lat
        
    return base_latency

def compute_total_latency(
    graph: AIMCGraph,
    mapping: MappingResult,
    fabric: Fabric3D,
    interconnect: Optional[InterconnectModel] = None,
    seq_len: Optional[int] = None,
    batch_size: Optional[int] = None
) -> LatencyResult:
    """
    Compute total model latency assuming sequential execution of operations.
    For MoE layers, only the active experts contribute to latency (approximated).
    """
    if seq_len is None:
        seq_len = graph.batch_config.get("seq_len", 2048)
    if batch_size is None:
        batch_size = graph.batch_config.get("batch_size", 1)
        
    per_op_ns = {}
    total_ns = 0.0
    
    dac_total = 0.0
    cb_total = 0.0
    adc_total = 0.0
    act_total = 0.0
    interconnect_total = 0.0
    control_total = 0.0
    
    tech = fabric.crossbar_config.technology
    
    # Compute latency for each operation in topological order
    for op in graph.topological_order():
        if op.is_mappable:
            # Latency for this op
            op_lat = compute_op_latency(op, mapping, fabric, interconnect)
            
            # For MoE experts, only active experts run per token.
            # In our top-1 routing design, only 1 expert is active per token.
            # Thus, we scale down the expert op latency by the active fraction.
            if op.op_type == OpType.MOE_EXPERT:
                num_experts = graph.model_config.get("num_experts") or 8
                top_k = graph.model_config.get("top_k") or 1
                activation_factor = top_k / num_experts
                op_lat *= activation_factor
                
            per_op_ns[op.id] = op_lat
            total_ns += op_lat
            
            # Breakdown accounting
            op_assigns = mapping.get_op_tiles(op.id)
            if op_assigns:
                tile_counts = {}
                for assign in op_assigns:
                    tile_counts[assign.tile_id] = tile_counts.get(assign.tile_id, 0) + 1
                max_blocks = max(tile_counts.values()) if tile_counts else 0
                
                # Expert scale factor
                num_experts = graph.model_config.get("num_experts") or 8
                top_k = graph.model_config.get("top_k") or 1
                scale = (top_k / num_experts) if op.op_type == OpType.MOE_EXPERT else 1.0
                
                dac_total += tech.dac_bits * max_blocks * tech.t_dac * scale
                cb_total += tech.dac_bits * max_blocks * tech.t_crossbar * scale
                adc_total += tech.dac_bits * max_blocks * tech.t_adc * scale
                act_total += tech.t_activation * scale
                control_total += tech.t_control * scale
                
                if len(tile_counts) > 1 and interconnect is not None:
                    # Difference between total and core
                    interconnect_total += (op_lat - (tech.dac_bits * max_blocks * tech.cycle_time_ns + tech.t_activation + tech.t_control) * scale)
        else:
            # Non-mappable ops (norms, activations, embedding, residual)
            # Add simple digital CMOS delay
            digital_delay = 5.0  # 5ns digital execution delay
            per_op_ns[op.id] = digital_delay
            total_ns += digital_delay
            act_total += digital_delay
            
    # Autoregressive generation latency:
    # 1. Prompt phase (prefill): 1 token generation over seq_len inputs (parallel)
    # 2. Decode phase: seq_len - 1 sequential single-token steps
    # We estimate decode-phase token latency = total_ns (per_token_ns).
    # Total sequence latency = per_token_ns * seq_len
    per_token_ns = total_ns
    per_sequence_ns = per_token_ns * seq_len
    
    return LatencyResult(
        per_token_ns=per_token_ns,
        per_sequence_ns=per_sequence_ns,
        per_op_ns=per_op_ns,
        breakdown={
            "dac": dac_total,
            "crossbar": cb_total,
            "adc": adc_total,
            "activation": act_total,
            "interconnect": interconnect_total,
            "control": control_total
        },
        seq_len=seq_len,
        batch_size=batch_size
    )
