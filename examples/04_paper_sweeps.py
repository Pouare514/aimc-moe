from __future__ import annotations
import os
import sys

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aimc_moe.reporting.sweeps import run_scientific_sweeps

def main():
    print("================================================================")
    # Scientific Title
    print(" AIMC-MoE: Systematic 3D Stacked CIM & MoE Profiling Benchmarks")
    print("================================================================")
    print("[*] Running multi-variable hardware scaling & noise sweeps...")
    print("[*] Target Models: Surrogate MoE Transformer Configurations")
    print("[*] Output Folder: outputs/")
    print("[*] Plots Folder:  plots/")
    
    # Run sweeps
    results = run_scientific_sweeps(output_dir="outputs", plot_dir="plots")
    
    print("\n[+] Benchmark Suite Execution Completed Successfully!")
    print("----------------------------------------------------------------")
    print("Generated Outputs:")
    print("  - [Markdown Report]  outputs/sweep_report.md")
    print("  - [Thermal Plot]      plots/sweep_throughput_temp.png")
    print("  - [Scaling Plot]      plots/sweep_latency_tiers.png")
    print("  - [Accuracy Plot]     plots/sweep_accuracy_noise.png")
    print("================================================================")

if __name__ == "__main__":
    main()
