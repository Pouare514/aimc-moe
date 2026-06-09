from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
from aimc_moe.ir.ops import AIMCOp, OpType
from aimc_moe.ir.graph import AIMCGraph
from aimc_moe.backend.fabric import Fabric3D
from aimc_moe.backend.mapper import MappingResult
from aimc_moe.backend.interconnect import InterconnectModel

@dataclass
class EnergyResult:
    per_token_pj: float
    per_sequence_pj: float
    per_op_pj: Dict[str, float]
    breakdown: Dict[str, float]  # 'mac', 'adc', 'dac', 'interconnect'
    seq_len: int
    batch_size: int
    
    @property
    def per_token_uj(self) -> float:
        return self.per_token_pj / 1e6
        
    @property
    def per_sequence_uj(self) -> float:
        return self.per_sequence_pj / 1e6

def compute_op_energy(
    op: AIMCOp,
    mapping: MappingResult,
    fabric: Fabric3D,
    interconnect: Optional[InterconnectModel] = None
) -> float:
    """
    Compute energy in pJ consumed by a single linear operation.
    """
    op_assigns = mapping.get_op_tiles(op.id)
    if not op_assigns:
        return 0.0
        
    tech = fabric.crossbar_config.technology
    xb_rows = fabric.crossbar_config.rows
    
    # 1. Core crossbar compute energy
    in_feats = op.shape[1] if len(op.shape) >= 2 else 1
    out_feats = op.shape[0] if len(op.shape) >= 2 else op.shape[0]
    
    mac_count = in_feats * out_feats
    e_mac = mac_count * tech.energy_per_mac
    
    # 2. Peripheral ADCs and DACs energy
    # We need one DAC input per active row of the crossbars used
    num_dac_conversions = in_feats
    e_dac = num_dac_conversions * tech.energy_dac * tech.dac_bits
    
    # One ADC conversion per col-block per column
    # For each column-block, we perform an MVM on a tile, which triggers an ADC per output column
    n_row_blocks = len({assign.block_row for assign in op_assigns})
    num_adc_conversions = out_feats * n_row_blocks
    e_adc = num_adc_conversions * tech.energy_adc * tech.adc_bits
    
    # 3. Interconnect energy
    e_interconnect = 0.0
    if interconnect is not None and len(op_assigns) > 1:
        # Hops from the first tile to subsequent tiles
        first_tile = fabric.get_tile(*op_assigns[0].coordinates)
        for assign in op_assigns[1:]:
            target_tile = fabric.get_tile(*assign.coordinates)
            e_interconnect += interconnect.energy_between(first_tile, target_tile, tech)
            
    return e_mac + e_dac + e_adc + e_interconnect

def compute_total_energy(
    graph: AIMCGraph,
    mapping: MappingResult,
    fabric: Fabric3D,
    interconnect: Optional[InterconnectModel] = None,
    seq_len: Optional[int] = None,
    batch_size: Optional[int] = None
) -> EnergyResult:
    """
    Compute total model energy for processing a sequence of tokens.
    For MoE layers, only active experts consume dynamic energy.
    """
    if seq_len is None:
        seq_len = graph.batch_config.get("seq_len", 2048)
    if batch_size is None:
        batch_size = graph.batch_config.get("batch_size", 1)
        
    per_op_pj = {}
    total_pj = 0.0
    
    mac_total = 0.0
    adc_total = 0.0
    dac_total = 0.0
    interconnect_total = 0.0
    
    tech = fabric.crossbar_config.technology
    xb_rows = fabric.crossbar_config.rows
    
    for op in graph.topological_order():
        if op.is_mappable:
            op_energy = compute_op_energy(op, mapping, fabric, interconnect)
            
            # Scale energy if it's a MoE expert (sparsity)
            scale = 1.0
            if op.op_type == OpType.MOE_EXPERT:
                num_experts = graph.model_config.get("num_experts") or 8
                top_k = graph.model_config.get("top_k") or 1
                scale = top_k / num_experts
                op_energy *= scale
                
            per_op_pj[op.id] = op_energy
            total_pj += op_energy
            
            # Breakdown accounting
            in_feats = op.shape[1] if len(op.shape) >= 2 else 1
            out_feats = op.shape[0] if len(op.shape) >= 2 else op.shape[0]
            
            op_assigns = mapping.get_op_tiles(op.id)
            n_row_blocks = len({assign.block_row for assign in op_assigns}) if op_assigns else 0
            
            mac_total += (in_feats * out_feats) * tech.energy_per_mac * scale
            dac_total += in_feats * tech.energy_dac * tech.dac_bits * scale
            adc_total += (out_feats * n_row_blocks) * tech.energy_adc * tech.adc_bits * scale
            
            if len(op_assigns) > 1 and interconnect is not None:
                first_tile = fabric.get_tile(*op_assigns[0].coordinates)
                for assign in op_assigns[1:]:
                    target_tile = fabric.get_tile(*assign.coordinates)
                    interconnect_total += interconnect.energy_between(first_tile, target_tile, tech) * scale
        else:
            # Digital CMOS energy (e.g. activations, LayerNorm in digital domain)
            # Standard CMOS energy factor (e.g. 0.01 pJ per parameter)
            digital_energy = op.num_params * 0.01
            per_op_pj[op.id] = digital_energy
            total_pj += digital_energy
            mac_total += digital_energy
            
    # Autoregressive generation consumes energy for each token
    # Total sequence energy = energy per token * seq_len
    per_token_pj = total_pj
    per_sequence_pj = per_token_pj * seq_len
    
    return EnergyResult(
        per_token_pj=per_token_pj,
        per_sequence_pj=per_sequence_pj,
        per_op_pj=per_op_pj,
        breakdown={
            "mac": mac_total,
            "adc": adc_total,
            "dac": dac_total,
            "interconnect": interconnect_total
        },
        seq_len=seq_len,
        batch_size=batch_size
    )
