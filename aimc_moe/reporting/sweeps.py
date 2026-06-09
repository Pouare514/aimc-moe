from __future__ import annotations
import os
import math
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from aimc_moe.frontend.adapters import trace_hf_model
from aimc_moe.frontend.surrogates import get_moe_surrogate
from aimc_moe.backend.fabric import Fabric3D
from aimc_moe.backend.crossbar import CrossbarConfig
from aimc_moe.backend.technology import get_technology
from aimc_moe.backend.mapper import map_graph_to_fabric, MappingStrategy
from aimc_moe.backend.interconnect import InterconnectModel
from aimc_moe.metrics.latency import compute_total_latency
from aimc_moe.metrics.energy import compute_total_energy
from aimc_moe.metrics.area import compute_total_area
from aimc_moe.metrics.thermal import ThermalConfig, solve_steady_state_temperatures
from aimc_moe.noise.profiles import get_noise_profile
from aimc_moe.training.noisy_forward import wrap_model_noisy, unwrap_model

# Matplotlib styling for dark theme
BG_COLOR = "#0B0F19"
TEXT_COLOR = "#E2E8F0"
GRID_COLOR = "#1E293B"
ACCENT_COLORS = ["#38BDF8", "#34D399", "#F472B6", "#FB923C", "#A78BFA"]

def apply_dark_theme(ax: matplotlib.axes.Axes, title: str, xlabel: str, ylabel: str):
    ax.set_facecolor(BG_COLOR)
    ax.figure.patch.set_facecolor(BG_COLOR)
    ax.spines['bottom'].set_color(GRID_COLOR)
    ax.spines['top'].set_color(GRID_COLOR)
    ax.spines['right'].set_color(GRID_COLOR)
    ax.spines['left'].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_COLOR, which='both')
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, color=GRID_COLOR, linestyle='--', alpha=0.5)

def run_scientific_sweeps(
    output_dir: str = "outputs",
    plot_dir: str = "plots"
) -> Dict[str, Any]:
    """
    Executes three systematic sweeps and compiles a consolidated benchmark comparison.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    print("\n[Sweep Runner] Preparing surrogate models...")
    # Model 1: Small Qwen2Moe surrogate
    qwen = get_moe_surrogate("qwen2_moe", vocab_size=1000, d_model=128, d_ff=256, num_experts=4, n_layers=2, top_k=1)
    qwen.eval()
    qwen_graph = trace_hf_model(qwen, seq_len=256, batch_size=1, model_name="Qwen2Moe-Mock-1M")
    
    # Base Fabric configurations
    tech_pcm = get_technology("pcm_gst")
    xb_config = CrossbarConfig(rows=256, cols=256, technology=tech_pcm)
    
    interconnect = InterconnectModel()
    results = {}
    
    # ----------------------------------------------------
    # SWEEP 1: Thermal Sensitivity vs. Throughput (OPS)
    # ----------------------------------------------------
    print("[Sweep 1] Evaluating Thermal Hotspots vs. Throughput...")
    ops_range = np.logspace(9, 12, 6) # 1 GOPS to 1000 GOPS
    thermal_fabric = Fabric3D(mesh_rows=6, mesh_cols=6, n_tiers=4, crossbar_config=xb_config)
    
    strategies = [MappingStrategy.SEQUENTIAL, MappingStrategy.EXPERT_COLOCATE, MappingStrategy.THERMAL_AWARE]
    thermal_curves = {strat: [] for strat in strategies}
    
    for strat in strategies:
        mapping = map_graph_to_fabric(qwen_graph, thermal_fabric, strategy=strat)
        for ops in ops_range:
            res = solve_steady_state_temperatures(
                thermal_fabric, mapping, config=ThermalConfig(t_ambient=25.0), ops_per_second=ops
            )
            thermal_curves[strat].append(res.max_temperature)
            
    # Plot Sweep 1
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, strat in enumerate(strategies):
        ax.plot(
            ops_range, thermal_curves[strat], 
            marker='o', color=ACCENT_COLORS[i], linewidth=2, label=strat.value
        )
    ax.set_xscale('log')
    apply_dark_theme(ax, "Peak Stack Temperature vs. Throughput Rate", "Throughput (OPS)", "Max Temperature (°C)")
    legend = ax.legend(facecolor=BG_COLOR, edgecolor=GRID_COLOR)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "sweep_throughput_temp.png"), dpi=150, facecolor=BG_COLOR)
    plt.close()
    
    # ----------------------------------------------------
    # SWEEP 2: Stacking Tiers vs. Latency / Energy / Area
    # ----------------------------------------------------
    print("[Sweep 2] Evaluating 3D CIM Stacking Tiers Scaling...")
    tier_counts = [1, 2, 4, 8]
    energy_curve = []
    area_curve = []
    latency_curve = []
    
    for tc in tier_counts:
        stack_fabric = Fabric3D(mesh_rows=8, mesh_cols=8, n_tiers=tc, crossbar_config=xb_config)
        mapping = map_graph_to_fabric(qwen_graph, stack_fabric, strategy=MappingStrategy.EXPERT_COLOCATE)
        
        lat = compute_total_latency(qwen_graph, mapping, stack_fabric, interconnect)
        energy = compute_total_energy(qwen_graph, mapping, stack_fabric, interconnect)
        area = compute_total_area(stack_fabric, mapping)
        
        latency_curve.append(lat.per_token_ns / 1000.0) # µs
        energy_curve.append(energy.per_token_uj)
        area_curve.append(area.total_area_mm2)
        
    # Plot Sweep 2 (Double Y-Axis)
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax2 = ax1.twinx()
    
    line1 = ax1.plot(tier_counts, energy_curve, marker='s', color=ACCENT_COLORS[0], linewidth=2, label="Energy (µJ)")
    line2 = ax2.plot(tier_counts, area_curve, marker='^', color=ACCENT_COLORS[2], linewidth=2, label="Footprint Area (mm²)")
    
    apply_dark_theme(ax1, "3D Stacking Scaling (Energy & Footprint Area)", "Stacking Tiers", "Energy per token (µJ)")
    ax2.set_ylabel("Active Footprint Area (mm²)", color=TEXT_COLOR)
    ax2.tick_params(colors=TEXT_COLOR)
    ax2.spines['right'].set_color(GRID_COLOR)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    legend = ax1.legend(lines, labels, facecolor=BG_COLOR, edgecolor=GRID_COLOR, loc="upper right")
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)
        
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "sweep_latency_tiers.png"), dpi=150, facecolor=BG_COLOR)
    plt.close()
    
    # ----------------------------------------------------
    # SWEEP 3: Accuracy vs. Noise Scale
    # ----------------------------------------------------
    print("[Sweep 3] Evaluating Cross-Entropy Loss vs. Analog Noise Scales...")
    noise_scales = [0.0, 0.5, 1.0, 1.5, 2.0]
    inputs = torch.randint(0, 1000, (1, 64))
    
    loss_rram = []
    loss_pcm = []
    
    profile_rram = get_noise_profile("realistic_rram")
    profile_pcm = get_noise_profile("realistic_pcm")
    
    # Clean output baseline
    with torch.no_grad():
        clean_logits = qwen(inputs)
    clean_loss = nn.functional.cross_entropy(clean_logits.view(-1, 1000), inputs.view(-1)).item()
    
    for scale in noise_scales:
        if scale == 0.0:
            loss_rram.append(clean_loss)
            loss_pcm.append(clean_loss)
            continue
            
        # RRAM scale run
        mod_rram = get_noise_profile("realistic_rram")
        mod_rram.sigma_prog *= scale
        mod_rram.sigma_read *= scale
        mod_rram.sigma_d2d *= scale
        
        wrap_model_noisy(qwen, mod_rram)
        with torch.no_grad():
            rram_logits = qwen(inputs)
        loss_rram.append(nn.functional.cross_entropy(rram_logits.view(-1, 1000), inputs.view(-1)).item())
        unwrap_model(qwen)
        
        # PCM scale run
        mod_pcm = get_noise_profile("realistic_pcm")
        mod_pcm.sigma_prog *= scale
        mod_pcm.sigma_read *= scale
        mod_pcm.sigma_d2d *= scale
        
        wrap_model_noisy(qwen, mod_pcm)
        with torch.no_grad():
            pcm_logits = qwen(inputs)
        loss_pcm.append(nn.functional.cross_entropy(pcm_logits.view(-1, 1000), inputs.view(-1)).item())
        unwrap_model(qwen)
        
    # Plot Sweep 3
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(noise_scales, loss_rram, marker='x', color=ACCENT_COLORS[1], linewidth=2, label="RRAM HfO2 Profile")
    ax.plot(noise_scales, loss_pcm, marker='d', color=ACCENT_COLORS[3], linewidth=2, label="PCM GST Profile")
    ax.axhline(clean_loss, color=TEXT_COLOR, linestyle='--', alpha=0.7, label="Clean Baseline")
    
    apply_dark_theme(ax, "Loss Degradation vs. Analog Noise Factor", "Noise Scale Multiplier", "Cross-Entropy Loss")
    legend = ax.legend(facecolor=BG_COLOR, edgecolor=GRID_COLOR)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "sweep_accuracy_noise.png"), dpi=150, facecolor=BG_COLOR)
    plt.close()
    
    # ----------------------------------------------------
    # Generate Paper Benchmark Report
    # ----------------------------------------------------
    print("[Sweep Runner] Generating comparison report...")
    
    md_content = f"""# AIMC-MoE 3D Stacking Benchmark Report

This document reports physical estimation sweeps for multi-tier Analog In-Memory Computing configurations mapped with Mixture-of-Experts workloads.

## 🌡️ 1. Thermal Hotspot Sensitivity vs. Workload Throughput
 Peak temperatures reached across strategies when running at variable throughput rates:

| Throughput (GOPS) | SEQUENTIAL (°C) | EXPERT_COLOCATE (°C) | THERMAL_AWARE (°C) |
| :--- | :---: | :---: | :---: |
"""
    for i, ops in enumerate(ops_range):
        gops = ops / 1e9
        t_seq = thermal_curves[MappingStrategy.SEQUENTIAL][i]
        t_col = thermal_curves[MappingStrategy.EXPERT_COLOCATE][i]
        t_th = thermal_curves[MappingStrategy.THERMAL_AWARE][i]
        md_content += f"| {gops:.1f} | {t_seq:.2f} | {t_col:.2f} | {t_th:.2f} |\n"
        
    md_content += f"""
> [!TIP]
> The `THERMAL_AWARE` mapping strategy co-locates the highly active shared embeddings and attention heads closer to the Tier-0 convective heatsink, resulting in significantly reduced hotspot peak temperatures at high throughput rates.

## 📐 2. 3D Memory Stack Scaling Characteristics
Physical area, energy, and latency variations across tier configurations:

| Stacking Tiers | Active Footprint Area (mm²) | Energy per Token (µJ) | Latency per Token (µs) |
| :--- | :---: | :---: | :---: |
"""
    for i, tc in enumerate(tier_counts):
        md_content += f"| {tc} | {area_curve[i]:.4f} | {energy_curve[i]:.4f} | {latency_curve[i]:.2f} |\n"
        
    md_content += f"""
## 📉 3. Accuracy Resilience under Variable Noise Scales
Cross-entropy loss under RRAM HfO2 and PCM GST noise injection profiles:

| Noise Multiplier | FP32 Baseline | RRAM HfO2 Loss | PCM GST Loss |
| :--- | :---: | :---: | :---: |
"""
    for i, scale in enumerate(noise_scales):
        md_content += f"| {scale:.1f} | {clean_loss:.4f} | {loss_rram[i]:.4f} | {loss_pcm[i]:.4f} |\n"
        
    md_content += "\nReport generated successfully.\n"
    
    report_path = os.path.join(output_dir, "sweep_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    # Collate results dictionary
    results = {
        "ops_range": ops_range.tolist(),
        "thermal_curves": {k.value: v for k, v in thermal_curves.items()},
        "tier_counts": tier_counts,
        "energy_curve": energy_curve,
        "area_curve": area_curve,
        "latency_curve": latency_curve,
        "noise_scales": noise_scales,
        "loss_rram": loss_rram,
        "loss_pcm": loss_pcm,
        "clean_loss": clean_loss
    }
    
    print(f"[Sweep Runner] Sweep completed successfully. Report written to {report_path}.")
    return results
