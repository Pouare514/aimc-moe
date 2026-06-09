from __future__ import annotations
import os
from typing import List, Optional
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt

from aimc_moe.reporting.report import AIMCReport

# Sleek, premium dark theme constants
BG_COLOR = "#0B0F19"
AXIS_COLOR = "#E2E8F0"
GRID_COLOR = "#1E293B"
TEXT_COLOR = "#F8FAFC"
ACCENT_COLORS = ["#22D3EE", "#A78BFA", "#F43F5E", "#F59E0B", "#10B981", "#EC4899"]

def _apply_premium_dark_theme(ax, fig):
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.spines['bottom'].set_color(GRID_COLOR)
    ax.spines['top'].set_color(GRID_COLOR)
    ax.spines['right'].set_color(GRID_COLOR)
    ax.spines['left'].set_color(GRID_COLOR)
    ax.xaxis.label.set_color(AXIS_COLOR)
    ax.yaxis.label.set_color(AXIS_COLOR)
    ax.tick_params(colors=AXIS_COLOR, which='both')
    ax.title.set_color(TEXT_COLOR)
    ax.grid(True, color=GRID_COLOR, linestyle="--", alpha=0.5)

def plot_latency_breakdown(report: AIMCReport, save_path: Optional[str] = None) -> None:
    if report.latency is None:
        return
        
    breakdown = report.latency.breakdown
    labels = list(breakdown.keys())
    values = [breakdown[k] for k in labels]
    
    # Filter out 0 values
    filtered = [(l, v) for l, v in zip(labels, values) if v > 0]
    if not filtered:
        return
    labels, values = zip(*filtered)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    _apply_premium_dark_theme(ax, fig)
    
    bars = ax.bar(labels, values, color=ACCENT_COLORS[:len(labels)], width=0.6)
    
    # Add labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.2f} ns",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', color=TEXT_COLOR, fontsize=9)
                    
    ax.set_title(f"Latency Breakdown (Per Token): {report.model_name}")
    ax.set_ylabel("Latency (ns)")
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, facecolor=BG_COLOR, dpi=150)
    plt.close()

def plot_energy_breakdown(report: AIMCReport, save_path: Optional[str] = None) -> None:
    if report.energy is None:
        return
        
    breakdown = report.energy.breakdown
    labels = list(breakdown.keys())
    values = [breakdown[k] / 1e6 for k in labels]  # pJ to µJ
    
    filtered = [(l, v) for l, v in zip(labels, values) if v > 0]
    if not filtered:
        return
    labels, values = zip(*filtered)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    _apply_premium_dark_theme(ax, fig)
    
    bars = ax.bar(labels, values, color=ACCENT_COLORS[:len(labels)], width=0.6)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.4f} µJ",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', color=TEXT_COLOR, fontsize=9)
                    
    ax.set_title(f"Energy Breakdown (Per Token): {report.model_name}")
    ax.set_ylabel("Energy (µJ)")
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, facecolor=BG_COLOR, dpi=150)
    plt.close()

def plot_per_layer_latency(report: AIMCReport, save_path: Optional[str] = None) -> None:
    if report.latency is None:
        return
        
    per_op = report.latency.per_op_ns
    # Sort and take top 15 most expensive ops for readability
    sorted_ops = sorted(per_op.items(), key=lambda x: x[1], reverse=True)[:15]
    if not sorted_ops:
        return
        
    labels, values = zip(*sorted_ops)
    # Convert to µs
    values = [v / 1000.0 for v in values]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    _apply_premium_dark_theme(ax, fig)
    
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color="#22D3EE", align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()  # top-down
    
    ax.set_xlabel("Latency (µs)")
    ax.set_title(f"Top 15 Layer Latency (Per Token): {report.model_name}")
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, facecolor=BG_COLOR, dpi=150)
    plt.close()

def plot_per_layer_energy(report: AIMCReport, save_path: Optional[str] = None) -> None:
    if report.energy is None:
        return
        
    per_op = report.energy.per_op_pj
    sorted_ops = sorted(per_op.items(), key=lambda x: x[1], reverse=True)[:15]
    if not sorted_ops:
        return
        
    labels, values = zip(*sorted_ops)
    # Convert to µJ
    values = [v / 1e6 for v in values]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    _apply_premium_dark_theme(ax, fig)
    
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color="#A78BFA", align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    
    ax.set_xlabel("Energy (µJ)")
    ax.set_title(f"Top 15 Layer Energy (Per Token): {report.model_name}")
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, facecolor=BG_COLOR, dpi=150)
    plt.close()

def plot_accuracy_comparison(reports: List[AIMCReport], save_path: Optional[str] = None) -> None:
    if not reports:
        return
        
    names = [r.noise_profile_name for r in reports]
    baselines = [r.baseline_metric for r in reports]
    noisies = [r.noisy_metric for r in reports]
    finetuneds = [r.finetuned_metric for r in reports]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    _apply_premium_dark_theme(ax, fig)
    
    x = np.arange(len(names))
    width = 0.25
    
    # We plot baseline, noisy, and finetuned (if available)
    rects1 = ax.bar(x - width, baselines, width, label='FP32 Clean', color="#10B981")
    rects2 = ax.bar(x, noisies, width, label='AIMC Noisy', color="#F43F5E")
    
    # Only plot finetuned if we have at least one valid measurement
    if any(ft is not None for ft in finetuneds):
        # Fallback to noisy value if finetuned not measured
        ft_plot = [ft if ft is not None else n for ft, n in zip(finetuneds, noisies)]
        rects3 = ax.bar(x + width, ft_plot, width, label='AIMC Finetuned', color="#22D3EE")
        
    ax.set_ylabel(reports[0].baseline_metric_name.capitalize())
    ax.set_title(f"Accuracy/Loss Comparison across Noise Profiles")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    
    # Legend
    legend = ax.legend(facecolor=BG_COLOR, edgecolor=GRID_COLOR)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)
        
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, facecolor=BG_COLOR, dpi=150)
    plt.close()

def plot_sensitivity_heatmap(report: AIMCReport, save_path: Optional[str] = None) -> None:
    sens = report.per_layer_sensitivity
    if not sens:
        return
        
    # Heatmap of layer sensitivity
    labels, values = zip(*sorted(sens.items(), key=lambda x: x[0]))
    
    # Reshape to a 2D array for plotting (approximate grid)
    n = len(labels)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    
    grid = np.zeros((rows, cols))
    for idx, val in enumerate(values):
        r = idx // cols
        c = idx % cols
        grid[r, c] = val
        
    fig, ax = plt.subplots(figsize=(10, 8))
    _apply_premium_dark_theme(ax, fig)
    
    im = ax.imshow(grid, cmap="plasma", aspect="auto")
    
    # Label each cell
    for idx, (label, val) in enumerate(zip(labels, values)):
        r = idx // cols
        c = idx % cols
        ax.text(c, r, f"{label[:12]}\n{val:.4f}", ha="center", va="center", 
                color="white" if val < 0.5 * np.max(values) else "black", fontsize=8)
                
    # Heatbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_tick_params(color=AXIS_COLOR)
    cbar.ax.yaxis.label.set_color(AXIS_COLOR)
    cbar.set_label("Sensitivity Score (1 / SNR)", color=AXIS_COLOR)
    
    ax.set_title(f"Layer Sensitivity Heatmap (1/SNR): {report.model_name}")
    ax.axis('off')  # Hide grid axes
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, facecolor=BG_COLOR, dpi=150)
    plt.close()

def plot_thermal_heatmap(
    fabric: Fabric3D,
    thermal_res: ThermalResult,
    save_path: Optional[str] = None
) -> None:
    """
    Generate subplots showing the temperature distribution across tiles
    for each tier of the 3D stacked chip.
    """
    rows = fabric.mesh_rows
    cols = fabric.mesh_cols
    tiers = fabric.n_tiers
    
    fig, axes = plt.subplots(1, tiers, figsize=(4.5 * tiers, 4), squeeze=False)
    
    for t in range(tiers):
        ax = axes[0, t]
        _apply_premium_dark_theme(ax, fig)
        
        # Build 2D grid of temperatures
        grid = np.zeros((rows, cols))
        for r in range(rows):
            for c in range(cols):
                tile = fabric.get_tile(t, r, c)
                grid[r, c] = thermal_res.tile_temperatures.get(tile.tile_id, 25.0)
                
        # Draw heatmap
        im = ax.imshow(grid, cmap="plasma", vmin=thermal_res.min_temperature, vmax=thermal_res.max_temperature)
        
        # Add values on cells
        for r in range(rows):
            for c in range(cols):
                val = grid[r, c]
                ax.text(c, r, f"{val:.1f}°", ha="center", va="center",
                        color="white" if val < (0.5 * (thermal_res.max_temperature + thermal_res.min_temperature)) else "black",
                        fontsize=7)
                
        ax.set_title(f"Tier {t}", color=TEXT_COLOR)
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))
        
    # Add shared colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.yaxis.set_tick_params(color=AXIS_COLOR)
    cbar.ax.yaxis.label.set_color(AXIS_COLOR)
    cbar.set_label("Temperature (°C)", color=AXIS_COLOR)
    
    fig.suptitle(f"3D Fabric Steady-State Temperature (Max: {thermal_res.max_temperature:.1f}°C)", color=TEXT_COLOR, y=0.98)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, facecolor=BG_COLOR, dpi=150, bbox_inches="tight")
    plt.close()

