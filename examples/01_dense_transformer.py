from __future__ import annotations
import os
import torch
import torch.nn as nn

from aimc_moe.models.transformer import MiniTransformer
from aimc_moe.frontend.tracer import trace_model
from aimc_moe.backend.technology import get_technology
from aimc_moe.backend.crossbar import CrossbarConfig
from aimc_moe.backend.fabric import Fabric3D
from aimc_moe.backend.mapper import map_graph_to_fabric, MappingStrategy
from aimc_moe.backend.interconnect import InterconnectModel
from aimc_moe.metrics.latency import compute_total_latency
from aimc_moe.metrics.energy import compute_total_energy
from aimc_moe.metrics.area import compute_total_area
from aimc_moe.noise.profiles import get_noise_profile
from aimc_moe.training.noisy_forward import wrap_model_noisy, unwrap_model
from aimc_moe.reporting.report import AIMCReport
from aimc_moe.reporting.plots import (
    plot_latency_breakdown,
    plot_energy_breakdown,
    plot_per_layer_latency,
    plot_per_layer_energy,
    plot_accuracy_comparison,
    plot_sensitivity_heatmap
)
from aimc_moe.reporting.heatmaps import generate_sensitivity_data

def run_dense_example():
    print("--- Starting Dense Transformer AIMC Compilation & Simulation ---")
    
    # 1. Initialize Model (~10M parameters)
    model = MiniTransformer(
        vocab_size=1000,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=512,
        max_seq_len=512
    )
    model.eval()
    total_params = model.count_parameters()
    print(f"Model initialized: MiniTransformer, Parameters: {total_params:,}")
    
    # 2. Trace to IR
    print("Tracing model to AIMCGraph IR...")
    graph = trace_model(model, seq_len=512, batch_size=1, model_name="MiniDenseTransformer")
    print(graph.summary())
    
    # 3. Create Hardware Fabric Layout (2D Mesh, 1 tier, 65nm RRAM)
    print("Configuring Virtual 2D AIMC Fabric...")
    tech = get_technology("rram_hfo2")
    xb_config = CrossbarConfig(rows=256, cols=256, technology=tech)
    
    # Create an 8x8 grid of crossbar tiles
    fabric = Fabric3D(
        mesh_rows=8,
        mesh_cols=8,
        n_tiers=1,
        crossbar_config=xb_config
    )
    print(fabric.summary())
    
    # 4. Map Model weights to Fabric
    print("Mapping weights to fabric tiles...")
    mapping = map_graph_to_fabric(graph, fabric, strategy=MappingStrategy.SEQUENTIAL)
    print(mapping.summary())
    
    # 5. Estimate Performance Metrics
    print("Estimating latency, energy, and physical area...")
    interconnect = InterconnectModel(hop_latency_ns=2.0, vertical_hop_latency_ns=5.0, bandwidth_gbps=256.0)
    
    lat_res = compute_total_latency(graph, mapping, fabric, interconnect)
    energy_res = compute_total_energy(graph, mapping, fabric, interconnect)
    area_res = compute_total_area(fabric, mapping)
    
    # 6. Noise & Accuracy Simulation
    print("Simulating accuracy degradation under analog noise...")
    # Generate mock inputs (seq_len=512, batch=1)
    inputs = torch.randint(0, 1000, (1, 512))
    
    # Clean output
    with torch.no_grad():
        clean_logits = model(inputs)
        
    # Noisy output
    rram_noise = get_noise_profile("realistic_rram")
    wrap_model_noisy(model, rram_noise)
    with torch.no_grad():
        noisy_logits = model(inputs)
    unwrap_model(model)
    
    # Compute relative degradation (mock loss)
    clean_loss = nn.functional.cross_entropy(clean_logits.view(-1, 1000), inputs.view(-1)).item()
    noisy_loss = nn.functional.cross_entropy(noisy_logits.view(-1, 1000), inputs.view(-1)).item()
    
    # Measure Layer-wise Sensitivity
    print("Measuring layer-wise noise sensitivity...")
    sensitivity = generate_sensitivity_data(graph, rram_noise, model, inputs, n_samples=3)
    
    # 7. Generate Report
    print("Generating simulation report...")
    report = AIMCReport(
        model_name="MiniDenseTransformer",
        total_params=total_params,
        fabric_summary=fabric.summary(),
        technology_name=tech.name,
        noise_profile_name=rram_noise.name,
        latency=lat_res,
        energy=energy_res,
        area=area_res,
        baseline_metric=clean_loss,
        baseline_metric_name="cross-entropy loss",
        noisy_metric=noisy_loss,
        per_layer_sensitivity=sensitivity
    )
    
    report.print_summary()
    
    # Save Report Outputs
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/dense_report.json", "w") as f:
        f.write(report.to_json())
        
    with open("outputs/dense_report.md", "w") as f:
        f.write(report.to_markdown())
        
    print("Saved reports to outputs/dense_report.json and outputs/dense_report.md")
    
    # Generate Charts
    print("Generating visual plots...")
    os.makedirs("plots", exist_ok=True)
    plot_latency_breakdown(report, "plots/dense_latency_breakdown.png")
    plot_energy_breakdown(report, "plots/dense_energy_breakdown.png")
    plot_per_layer_latency(report, "plots/dense_per_layer_latency.png")
    plot_per_layer_energy(report, "plots/dense_per_layer_energy.png")
    plot_sensitivity_heatmap(report, "plots/dense_sensitivity_heatmap.png")
    print("Saved charts to plots/ directory.")
    
if __name__ == "__main__":
    run_dense_example()
