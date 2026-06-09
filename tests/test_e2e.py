from __future__ import annotations
import math
import torch
import torch.nn as nn
import pytest

from aimc_moe.models.transformer import MiniTransformer
from aimc_moe.models.moe_transformer import MoETransformer
from aimc_moe.frontend.tracer import trace_model
from aimc_moe.frontend.moe_tracer import trace_moe_model
from aimc_moe.backend.technology import get_technology, IDEAL
from aimc_moe.backend.crossbar import CrossbarConfig
from aimc_moe.backend.fabric import Fabric3D
from aimc_moe.backend.mapper import map_graph_to_fabric, MappingStrategy
from aimc_moe.metrics.latency import compute_total_latency
from aimc_moe.metrics.energy import compute_total_energy
from aimc_moe.metrics.area import compute_total_area
from aimc_moe.noise.profiles import get_noise_profile
from aimc_moe.training.noisy_forward import wrap_model_noisy, unwrap_model

def test_ir_and_tracing():
    model = MiniTransformer(
        vocab_size=100,
        d_model=64,
        n_heads=2,
        n_layers=1,
        d_ff=128,
        max_seq_len=64
    )
    model.eval()
    
    # Trace model
    graph = trace_model(model, seq_len=64, batch_size=1)
    
    assert len(graph.ops) > 0
    assert len(graph.edges) > 0
    
    # Test topological sorting
    order = graph.topological_order()
    assert len(order) == len(graph.ops)
    
    # Assert mappable ops are nn.Linear weight projections
    mappables = graph.get_mappable_ops()
    assert len(mappables) > 0
    for op in mappables:
        assert op.weights is not None
        assert len(op.shape) == 2

def test_mapping_completeness():
    model = MiniTransformer(vocab_size=100, d_model=64, n_heads=2, n_layers=1, d_ff=128, max_seq_len=64)
    model.eval()
    graph = trace_model(model, seq_len=64, batch_size=1)
    
    tech = get_technology("rram_hfo2")
    fabric = Fabric3D(mesh_rows=4, mesh_cols=4, n_tiers=1, crossbar_config=CrossbarConfig(rows=256, cols=256, technology=tech))
    
    mapping = map_graph_to_fabric(graph, fabric, MappingStrategy.SEQUENTIAL)
    
    assert len(mapping.unmapped_ops) == 0
    assert mapping.total_tiles_used > 0
    assert len(mapping.assignments) == mapping.total_crossbars_needed

def test_moe_tracing_and_colocation():
    model = MoETransformer(
        vocab_size=100,
        d_model=64,
        n_heads=2,
        n_layers=1,
        d_ff=128,
        num_experts=4,
        top_k=1,
        max_seq_len=64
    )
    model.eval()
    
    graph = trace_moe_model(model, seq_len=64, batch_size=1)
    assert len(graph.get_expert_ops()) > 0
    
    tech = get_technology("pcm_gst")
    fabric = Fabric3D(mesh_rows=8, mesh_cols=8, n_tiers=3, crossbar_config=CrossbarConfig(rows=128, cols=128, technology=tech))
    
    mapping = map_graph_to_fabric(graph, fabric, MappingStrategy.EXPERT_COLOCATE)
    
    assert len(mapping.unmapped_ops) == 0
    assert mapping.total_tiles_used > 0

def test_ideal_noise_equivalence():
    """
    Under the IDEAL noise profile, the output of the model wrapped with NoisyLinear
    must match the clean floating point model within a very close tolerance.
    """
    torch.manual_seed(42)
    model = MiniTransformer(vocab_size=100, d_model=64, n_heads=2, n_layers=1, d_ff=128, max_seq_len=64)
    model.eval()
    
    inputs = torch.randint(0, 100, (1, 32))
    
    # 1. Clean forward pass
    with torch.no_grad():
        clean_out = model(inputs)
        
    # 2. Ideal noise forward pass
    ideal_noise = get_noise_profile("ideal")
    wrap_model_noisy(model, ideal_noise)
    with torch.no_grad():
        noisy_out = model(inputs)
    unwrap_model(model)
    
    # Assert values are identical or extremely close (floating point tolerance)
    assert torch.allclose(clean_out, noisy_out, atol=1e-5)

def test_metrics_sanity():
    model = MiniTransformer(vocab_size=100, d_model=64, n_heads=2, n_layers=1, d_ff=128, max_seq_len=64)
    model.eval()
    graph = trace_model(model, seq_len=64, batch_size=1)
    
    tech = get_technology("rram_hfo2")
    fabric = Fabric3D(mesh_rows=4, mesh_cols=4, n_tiers=1, crossbar_config=CrossbarConfig(rows=256, cols=256, technology=tech))
    mapping = map_graph_to_fabric(graph, fabric, MappingStrategy.SEQUENTIAL)
    
    lat = compute_total_latency(graph, mapping, fabric)
    energy = compute_total_energy(graph, mapping, fabric)
    area = compute_total_area(fabric, mapping)
    
    # Assert latency boundaries are within typical physically realistic ranges
    assert lat.per_token_ns > 0.0
    assert lat.per_sequence_ns == lat.per_token_ns * 64
    
    # Assert energy boundaries
    assert energy.per_token_pj > 0.0
    assert energy.per_sequence_pj == energy.per_token_pj * 64
    
    # Area must match tile size
    assert area.total_area_mm2 > 0.0
    assert area.utilization_pct > 0.0
