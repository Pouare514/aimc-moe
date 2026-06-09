from __future__ import annotations
import math
import torch
import torch.nn as nn
import pytest

from aimc_moe.frontend.adapters import trace_hf_model
from aimc_moe.backend.technology import get_technology
from aimc_moe.backend.crossbar import CrossbarConfig
from aimc_moe.backend.fabric import Fabric3D
from aimc_moe.backend.mapper import map_graph_to_fabric, MappingStrategy
from aimc_moe.metrics.thermal import ThermalConfig, solve_steady_state_temperatures
from aimc_moe.noise.nonlinearity import apply_analytical_ir_drop
from aimc_moe.ir.ops import OpType

# Mock configuration and model classes matching HuggingFace structure
class MockConfig:
    def __init__(self, hidden_size=64, num_local_experts=4, num_experts_per_tok=1):
        self.hidden_size = hidden_size
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok

class MockQwen2MoeAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        return self.o_proj(context)

class MockQwen2MoeExpert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff)
        self.up_proj = nn.Linear(d_model, d_ff)
        self.down_proj = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

class MockQwen2MoeMLP(nn.Module):
    def __init__(self, d_model, d_ff, num_experts):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([
            MockQwen2MoeExpert(d_model, d_ff)
            for _ in range(num_experts)
        ])

    def forward(self, x):
        orig_shape = x.shape
        flat_x = x.view(-1, orig_shape[-1])
        gate_logits = self.gate(flat_x)
        probs = torch.softmax(gate_logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, k=1, dim=-1)
        
        out = torch.zeros_like(flat_x)
        for i, expert in enumerate(self.experts):
            mask = (top_indices[:, 0] == i)
            if mask.any():
                expert_out = expert(flat_x[mask])
                expert_out = expert_out * probs[mask, i:i+1]
                out[mask] = expert_out
        return out.view(*orig_shape)

class MockQwen2MoeDecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_experts):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(d_model)
        self.self_attn = MockQwen2MoeAttention(d_model)
        self.post_attention_layernorm = nn.LayerNorm(d_model)
        self.mlp = MockQwen2MoeMLP(d_model, d_ff, num_experts)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

class MockQwen2MoeModel(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, num_experts, n_layers):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            MockQwen2MoeDecoderLayer(d_model, d_ff, num_experts)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.embed_tokens(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class MockQwen2MoeForCausalLM(nn.Module):
    def __init__(self, vocab_size=100, d_model=64, d_ff=128, num_experts=4, n_layers=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.config = MockConfig(hidden_size=d_model, num_local_experts=num_experts, num_experts_per_tok=1)
        self.model = MockQwen2MoeModel(vocab_size, d_model, d_ff, num_experts, n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        x = self.model(x)
        return self.lm_head(x)


def test_trace_hf_model():
    model = MockQwen2MoeForCausalLM()
    model.eval()
    
    graph = trace_hf_model(model, seq_len=64, batch_size=1, model_name="Qwen2Moe-Test")
    
    assert graph.model_config.get("model_name") == "Qwen2Moe-Test"
    assert graph.model_config.get("d_model") == 64
    assert graph.model_config.get("num_experts") == 4
    assert graph.model_config.get("top_k") == 1
    
    # Assert presence of specific op types
    ops_types = [op.op_type for op in graph.ops]
    assert OpType.EMBEDDING in ops_types
    assert OpType.MOE_ROUTER in ops_types
    assert OpType.MOE_EXPERT in ops_types
    assert OpType.LINEAR in ops_types
    assert OpType.OUTPUT in ops_types
    
    # Verify topological sorting order is valid
    order = graph.topological_order()
    assert len(order) == len(graph.ops)


def test_solve_steady_state_temperatures():
    # Setup fabric and technology
    tech = get_technology("pcm_gst")
    xb_config = CrossbarConfig(rows=256, cols=256, technology=tech)
    
    fabric = Fabric3D(
        mesh_rows=2,
        mesh_cols=2,
        n_tiers=2,
        crossbar_config=xb_config
    )
    
    # Trace a simple mock layer and get a dummy mapping
    model = MockQwen2MoeForCausalLM(n_layers=1, num_experts=2)
    model.eval()
    graph = trace_hf_model(model, seq_len=64, batch_size=1)
    mapping = map_graph_to_fabric(graph, fabric, strategy=MappingStrategy.SEQUENTIAL)
    
    # Configure thermal options
    thermal_config = ThermalConfig(
        r_horizontal=10.0,
        r_vertical=15.0,
        r_convective=2.0,
        t_ambient=25.0
    )
    
    # 1. Idle run: zero operations/second -> temperatures should remain at ambient
    res_idle = solve_steady_state_temperatures(
        fabric, mapping, config=thermal_config, ops_per_second=0.0
    )
    for temp in res_idle.tile_temperatures.values():
        assert math.isclose(temp, 25.0, abs_tol=1e-5)
        
    # 2. Active run: high ops/second to generate heat
    res_active = solve_steady_state_temperatures(
        fabric, mapping, config=thermal_config, ops_per_second=1e13
    )
    
    # Active tiles must have higher temperatures than idle tiles
    active_temps = []
    idle_temps = []
    
    for coord, tile in fabric.tiles.items():
        temp = res_active.tile_temperatures[tile.tile_id]
        if tile.utilization > 0.0:
            active_temps.append(temp)
        else:
            idle_temps.append(temp)
            
    # There must be some heating
    assert res_active.max_temperature > 25.0
    
    # Active tiles should be warmer than purely idle/ambient tiles if any conduction is considered
    if active_temps and idle_temps:
        assert max(active_temps) >= min(idle_temps)

    # 3. Check thermal threshold warning
    res_unsafe = solve_steady_state_temperatures(
        fabric, mapping, config=thermal_config, ops_per_second=1e13, thermal_threshold=25.01
    )
    assert res_unsafe.is_thermal_unsafe is True


def test_apply_analytical_ir_drop():
    # Create a dummy tile weight matrix (e.g. all ones)
    weights = torch.ones((256, 256))
    
    # 1. Zero drop check: should return exact match
    no_drop = apply_analytical_ir_drop(weights, wire_resistance=0.5, alpha=0.0, beta=0.0)
    assert torch.allclose(weights, no_drop)
    
    # 2. Realistic drop check: drop should increase with indices
    drop_weights = apply_analytical_ir_drop(weights, wire_resistance=0.5, alpha=0.0001, beta=0.00001)
    
    # Driver end (0, 0) should have minimal or no drop
    assert math.isclose(drop_weights[0, 0].item(), 1.0, abs_tol=1e-5)
    
    # Further ends should show drop
    assert drop_weights[100, 100].item() < 1.0
    assert drop_weights[255, 255].item() < drop_weights[100, 100].item()
    
    # Values must remain bounded in [0.0, 1.0]
    assert (drop_weights >= 0.0).all()
    assert (drop_weights <= 1.0).all()
