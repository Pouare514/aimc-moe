from __future__ import annotations
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from aimc_moe.models.moe_transformer import MoETransformer
from aimc_moe.frontend.moe_tracer import trace_moe_model
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
from aimc_moe.training.finetune import aimc_finetune
from aimc_moe.reporting.report import AIMCReport
from aimc_moe.reporting.plots import plot_accuracy_comparison, plot_sensitivity_heatmap
from aimc_moe.reporting.heatmaps import generate_sensitivity_data

def run_moe_example():
    print("--- Starting MoE Transformer 3D AIMC Compilation & Finetuning ---")
    
    # 1. Initialize MoE Model
    # 4 experts, top-1 routing, 2 layers, d_model=128
    model = MoETransformer(
        vocab_size=1000,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=512,
        num_experts=4,
        top_k=1,
        max_seq_len=256
    )
    model.eval()
    total_params = model.count_parameters()
    print(f"Model initialized: MoETransformer, Parameters: {total_params:,}")
    
    # 2. Trace MoE Model
    print("Tracing MoE model using hybrid module tracer...")
    graph = trace_moe_model(model, seq_len=256, batch_size=2, model_name="MoETransformer")
    print(graph.summary())
    
    # 3. Configure Virtual 3D Fabric (8x8 tiles, 4 tiers, PCM technology)
    print("Configuring Virtual 3D Stack Fabric...")
    tech = get_technology("pcm_gst")
    xb_config = CrossbarConfig(rows=256, cols=256, technology=tech)
    
    fabric = Fabric3D(
        mesh_rows=8,
        mesh_cols=8,
        n_tiers=4,  # 4 layers stacked vertically
        crossbar_config=xb_config
    )
    print(fabric.summary())
    
    # 4. Map MoE using EXPERT_COLOCATE strategy
    print("Mapping experts and shared layers using EXPERT_COLOCATE strategy...")
    mapping = map_graph_to_fabric(graph, fabric, strategy=MappingStrategy.EXPERT_COLOCATE)
    print(mapping.summary())
    
    # 5. Estimate Performance Metrics
    print("Estimating performance metrics on 3D stack...")
    interconnect = InterconnectModel(hop_latency_ns=2.0, vertical_hop_latency_ns=5.0, bandwidth_gbps=256.0)
    
    lat_res = compute_total_latency(graph, mapping, fabric, interconnect)
    energy_res = compute_total_energy(graph, mapping, fabric, interconnect)
    area_res = compute_total_area(fabric, mapping)
    
    # 6. Baseline & Noisy Evaluation
    inputs = torch.randint(0, 1000, (2, 256))
    
    # Clean baseline
    with torch.no_grad():
        clean_logits, _ = model(inputs)
        
    pcm_noise = get_noise_profile("realistic_pcm")
    
    # Noisy pass
    wrap_model_noisy(model, pcm_noise)
    with torch.no_grad():
        noisy_logits, _ = model(inputs)
    unwrap_model(model)
    
    clean_loss = nn.functional.cross_entropy(clean_logits.view(-1, 1000), inputs.view(-1)).item()
    noisy_loss = nn.functional.cross_entropy(noisy_logits.view(-1, 1000), inputs.view(-1)).item()
    
    print(f"Baseline Loss: {clean_loss:.4f}")
    print(f"Noisy Loss (Pre-Finetuning): {noisy_loss:.4f}")
    
    # 7. Noise-Aware Finetuning
    print("Preparing toy dataset for Noise-Aware Finetuning...")
    # Create a small dataset (10 batches of batch_size=2)
    x_train = torch.randint(0, 1000, (20, 256))
    y_train = torch.randint(0, 1000, (20, 256))
    
    dataset = TensorDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    print("Starting training loop with noise injection...")
    # Run 10 finetuning steps
    ft_res = aimc_finetune(
        model=model,
        noise_profile=pcm_noise,
        dataloader=dataloader,
        steps=10,
        lr=5e-5,
        device="cpu"
    )
    print(f"Finetuning completed. Elapsed time: {ft_res['elapsed_seconds']:.2f}s")
    
    # Evaluate post-finetuning accuracy
    wrap_model_noisy(model, pcm_noise)
    with torch.no_grad():
        ft_logits, _ = model(inputs)
    unwrap_model(model)
    
    ft_loss = nn.functional.cross_entropy(ft_logits.view(-1, 1000), inputs.view(-1)).item()
    print(f"Noisy Loss (Post-Finetuning): {ft_loss:.4f}")
    
    # Measure sensitivity
    print("Measuring layer sensitivity...")
    sensitivity = generate_sensitivity_data(graph, pcm_noise, model, inputs, n_samples=2)
    
    # 8. Report & Save
    report = AIMCReport(
        model_name="MiniMoETransformer",
        total_params=total_params,
        fabric_summary=fabric.summary(),
        technology_name=tech.name,
        noise_profile_name=pcm_noise.name,
        latency=lat_res,
        energy=energy_res,
        area=area_res,
        baseline_metric=clean_loss,
        baseline_metric_name="cross-entropy loss",
        noisy_metric=noisy_loss,
        finetuned_metric=ft_loss,
        per_layer_sensitivity=sensitivity
    )
    
    report.print_summary()
    
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/moe_report.json", "w") as f:
        f.write(report.to_json())
        
    with open("outputs/moe_report.md", "w") as f:
        f.write(report.to_markdown())
        
    os.makedirs("plots", exist_ok=True)
    plot_accuracy_comparison([report], "plots/moe_accuracy_comparison.png")
    plot_sensitivity_heatmap(report, "plots/moe_sensitivity_heatmap.png")
    print("Finetuning simulation finished successfully!")

if __name__ == "__main__":
    run_moe_example()
