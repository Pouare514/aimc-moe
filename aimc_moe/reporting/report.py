from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
from aimc_moe.metrics.latency import LatencyResult
from aimc_moe.metrics.energy import EnergyResult
from aimc_moe.metrics.area import AreaResult

@dataclass
class AIMCReport:
    model_name: str
    total_params: int
    
    # Hardware Configuration
    fabric_summary: str
    technology_name: str
    noise_profile_name: str
    
    # Simulation Results
    latency: Optional[LatencyResult] = None
    energy: Optional[EnergyResult] = None
    area: Optional[AreaResult] = None
    
    # Accuracy Metrics
    baseline_metric: Optional[float] = None       # e.g., clean loss or perplexity
    baseline_metric_name: str = "loss"
    noisy_metric: Optional[float] = None          # under noise
    finetuned_metric: Optional[float] = None      # post finetuning
    
    # Per-layer metrics
    per_layer_sensitivity: Dict[str, float] = field(default_factory=dict)
    
    @property
    def degradation_pct(self) -> Optional[float]:
        """Percent change from baseline to noisy. (e.g. increase in loss or decrease in accuracy)"""
        if self.baseline_metric is None or self.noisy_metric is None or self.baseline_metric == 0:
            return None
        return ((self.noisy_metric - self.baseline_metric) / self.baseline_metric) * 100.0
        
    @property
    def recovery_pct(self) -> Optional[float]:
        """Percent recovery of accuracy loss after noise-aware finetuning."""
        if (self.baseline_metric is None or self.noisy_metric is None or 
            self.finetuned_metric is None or self.noisy_metric == self.baseline_metric):
            return None
        total_loss = self.noisy_metric - self.baseline_metric
        recovered = self.noisy_metric - self.finetuned_metric
        return (recovered / total_loss) * 100.0
        
    def to_dict(self) -> Dict[str, Any]:
        # Helper to convert dataclasses within report to nested dicts
        d = asdict(self)
        return d
        
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
        
    def to_markdown(self) -> str:
        lines = [
            f"# AIMC MoE Toolchain Simulation Report: {self.model_name}",
            "",
            "## 1. Executive Summary",
            f"- **Model Name**: {self.model_name}",
            f"- **Total Parameters**: {self.total_params:,}",
            f"- **Hardware Backend**: {self.technology_name}",
            f"- **Noise Profile**: {self.noise_profile_name}",
            ""
        ]
        
        # Accuracy section
        if self.baseline_metric is not None:
            lines.extend([
                "## 2. Accuracy & Degradation Analysis",
                "| Configuration | Value | Status |",
                "| --- | --- | --- |",
                f"| FP32 Baseline | {self.baseline_metric:.4f} | Ideal |",
            ])
            if self.noisy_metric is not None:
                deg_str = f"+{self.degradation_pct:.2f}%" if self.degradation_pct >= 0 else f"{self.degradation_pct:.2f}%"
                lines.append(f"| AIMC (Noisy) | {self.noisy_metric:.4f} | Degradation: {deg_str} |")
            if self.finetuned_metric is not None:
                lines.append(f"| AIMC (Finetuned) | {self.finetuned_metric:.4f} | Recovery: {self.recovery_pct:.2f}% |")
            lines.append("")
            
        # Hardware Metrics
        lines.append("## 3. Hardware Performance Estimation")
        if self.latency is not None:
            lines.extend([
                "### Latency",
                f"- **Per Token (decode step)**: {self.latency.per_token_ns / 1000.0:.3f} µs ({self.latency.per_token_ns:.2f} ns)",
                f"- **Per Sequence ({self.latency.seq_len} tokens)**: {self.latency.per_sequence_ns / 1e6:.3f} ms",
                ""
            ])
        if self.energy is not None:
            lines.extend([
                "### Energy",
                f"- **Per Token**: {self.energy.per_token_uj:.4f} µJ ({self.energy.per_token_pj:,.2f} pJ)",
                f"- **Per Sequence ({self.energy.seq_len} tokens)**: {self.energy.per_sequence_uj:.4f} mJ",
                ""
            ])
        if self.area is not None:
            lines.extend([
                "### Physical Area (Layout)",
                f"- **2D Silicon Footprint**: {self.area.total_area_mm2:.4f} mm²",
                f"- **Active Tier Surface Area**: {self.area.active_area_mm2:.4f} mm²",
                f"- **Crossbar Tile Utilization**: {self.area.tiles_used} / {self.area.tiles_total} ({self.area.utilization_pct:.2f}%)",
                ""
            ])
            
        lines.extend([
            "## 4. Hardware Fabric Layout Details",
            "```",
            self.fabric_summary,
            "```"
        ])
        return "\n".join(lines)
        
    def print_summary(self) -> None:
        print("=" * 60)
        print(f"            AIMC-MOE SIMULATION SUMMARY REPORT")
        print("=" * 60)
        print(f"Model: {self.model_name} ({self.total_params:,} parameters)")
        print(f"Tech:  {self.technology_name} | Noise: {self.noise_profile_name}")
        print("-" * 60)
        
        # Accuracy
        if self.baseline_metric is not None:
            print(f"Accuracy ({self.baseline_metric_name}):")
            print(f"  - FP32 Baseline: {self.baseline_metric:.4f}")
            if self.noisy_metric is not None:
                print(f"  - AIMC Simulated: {self.noisy_metric:.4f} (deg: {self.degradation_pct:.2f}%)")
            if self.finetuned_metric is not None:
                print(f"  - Post-Finetuning: {self.finetuned_metric:.4f} (recovery: {self.recovery_pct:.2f}%)")
            print("-" * 60)
            
        # Hardware
        print("Hardware Metrics (Estimated):")
        if self.latency is not None:
            print(f"  - Latency per token:  {self.latency.per_token_ns / 1000.0:.3f} µs")
            print(f"  - Latency per seq:    {self.latency.per_sequence_ns / 1e6:.3f} ms")
        if self.energy is not None:
            print(f"  - Energy per token:   {self.energy.per_token_uj:.4f} µJ")
            print(f"  - Energy per seq:     {self.energy.per_sequence_uj:.4f} mJ")
        if self.area is not None:
            print(f"  - Footprint Area:     {self.area.total_area_mm2:.4f} mm²")
            print(f"  - Crossbar Tiles:     {self.area.tiles_used} / {self.area.tiles_total} ({self.area.utilization_pct:.2f}%)")
        print("=" * 60)
