from __future__ import annotations
import os
import torch
import torch.nn as nn

from aimc_moe.frontend.adapters import trace_hf_model
from aimc_moe.backend.technology import get_technology
from aimc_moe.backend.crossbar import CrossbarConfig
from aimc_moe.backend.fabric import Fabric3D
from aimc_moe.backend.mapper import map_graph_to_fabric, MappingStrategy
from aimc_moe.backend.interconnect import InterconnectModel
from aimc_moe.metrics.latency import compute_total_latency
from aimc_moe.metrics.energy import compute_total_energy
from aimc_moe.metrics.area import compute_total_area
from aimc_moe.metrics.thermal import ThermalConfig, solve_steady_state_temperatures
from aimc_moe.noise.profiles import get_noise_profile
from aimc_moe.training.noisy_forward import wrap_model_noisy, unwrap_model
from aimc_moe.reporting.report import AIMCReport
from aimc_moe.reporting.plots import plot_thermal_heatmap, plot_accuracy_comparison

# 1. Define a Mock Qwen2MoeForCausalLM model to match HuggingFace class structure
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
    def __init__(self, vocab_size=1000, d_model=128, d_ff=256, num_experts=4, n_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.model = MockQwen2MoeModel(vocab_size, d_model, d_ff, num_experts, n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        x = self.model(x)
        return self.lm_head(x)

def run_huggingface_thermal_example():
    print("--- Starting HuggingFace Qwen2Moe 3D Thermal & IR Drop Simulation ---")
    
    # 1. Instantiate Model
    model = MockQwen2MoeForCausalLM()
    model.eval()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Mock Qwen2MoeForCausalLM initialized: Parameters: {total_params:,}")
    
    # 2. Trace using HF Adapter
    print("Parsing HuggingFace weight layout naming conventions...")
    graph = trace_hf_model(model, seq_len=512, batch_size=1, model_name="Qwen2Moe-A2.7B-Mock")
    print(graph.summary())
    
    # 3. Create Stack Fabric (8x8 tiles, 4 tiers, PCM GST)
    print("Configuring Virtual 3D Stack Fabric...")
    tech = get_technology("pcm_gst")
    xb_config = CrossbarConfig(rows=256, cols=256, technology=tech)
    
    fabric = Fabric3D(
        mesh_rows=8,
        mesh_cols=8,
        n_tiers=4,
        crossbar_config=xb_config
    )
    print(fabric.summary())
    
    # 4. Map Model (EXPERT_COLOCATE)
    print("Mapping model layers to fabric tiles...")
    mapping = map_graph_to_fabric(graph, fabric, strategy=MappingStrategy.EXPERT_COLOCATE)
    print(mapping.summary())
    
    # 5. Solve Steady-State Thermal Mesh (RC network)
    print("Solving 3D steady-state thermal resistor-capacitor mesh...")
    thermal_config = ThermalConfig(
        r_horizontal=12.0,
        r_vertical=18.0,
        r_convective=1.5,
        t_ambient=25.0
    )
    # Target execution rate: 2 GOPS (2e9 MACs/sec)
    thermal_res = solve_steady_state_temperatures(
        fabric, mapping, config=thermal_config, ops_per_second=2e9, thermal_threshold=80.0
    )
    
    print("\n--- Thermal Simulation Results ---")
    print(f"Max Temperature:  {thermal_res.max_temperature:.2f}°C")
    print(f"Min Temperature:  {thermal_res.min_temperature:.2f}°C")
    print(f"Average Temp:     {thermal_res.average_temperature:.2f}°C")
    print(f"Thermal Alarm:    {'WARNING - UNSAFE' if thermal_res.is_thermal_unsafe else 'SAFE'}")
    
    # 6. Apply Analytical position-dependent IR Drop
    print("\nSimulating accuracy under realistic RRAM noise + analytical IR Drop...")
    inputs = torch.randint(0, 1000, (1, 256))
    
    # Clean output
    with torch.no_grad():
        clean_logits = model(inputs)
        
    rram_noise_with_ir = get_noise_profile("realistic_rram")
    # Wrap model with IR drop and mismatch
    wrap_model_noisy(model, rram_noise_with_ir)
    with torch.no_grad():
        noisy_logits = model(inputs)
    unwrap_model(model)
    
    clean_loss = nn.functional.cross_entropy(clean_logits.view(-1, 1000), inputs.view(-1)).item()
    noisy_loss = nn.functional.cross_entropy(noisy_logits.view(-1, 1000), inputs.view(-1)).item()
    
    # 7. Physical Performance
    interconnect = InterconnectModel()
    lat_res = compute_total_latency(graph, mapping, fabric, interconnect)
    energy_res = compute_total_energy(graph, mapping, fabric, interconnect)
    area_res = compute_total_area(fabric, mapping)
    
    # 8. Report & Visualize
    report = AIMCReport(
        model_name="Qwen2Moe-A2.7B-Mock",
        total_params=total_params,
        fabric_summary=fabric.summary(),
        technology_name=tech.name,
        noise_profile_name=rram_noise_with_ir.name,
        latency=lat_res,
        energy=energy_res,
        area=area_res,
        baseline_metric=clean_loss,
        baseline_metric_name="cross-entropy loss",
        noisy_metric=noisy_loss
    )
    
    report.print_summary()
    
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/hf_thermal_report.md", "w") as f:
        f.write(report.to_markdown())
        
    os.makedirs("plots", exist_ok=True)
    plot_thermal_heatmap(fabric, thermal_res, "plots/hf_thermal_heatmap.png")
    plot_accuracy_comparison([report], "plots/hf_accuracy_comparison.png")
    print("Finetuning simulation finished successfully!")

if __name__ == "__main__":
    run_huggingface_thermal_example()
