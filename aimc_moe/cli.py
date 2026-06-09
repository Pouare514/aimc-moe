from __future__ import annotations
import argparse
import sys
import os
import torch
import torch.nn as nn
import json

from aimc_moe.frontend.adapters import trace_hf_model
from aimc_moe.frontend.surrogates import get_moe_surrogate
from aimc_moe.backend.configs import load_fabric_from_yaml
from aimc_moe.backend.mapper import map_graph_to_fabric, MappingStrategy
from aimc_moe.backend.interconnect import InterconnectModel
from aimc_moe.metrics.latency import compute_total_latency
from aimc_moe.metrics.energy import compute_total_energy
from aimc_moe.metrics.area import compute_total_area
from aimc_moe.metrics.thermal import ThermalConfig, solve_steady_state_temperatures
from aimc_moe.noise.profiles import get_noise_profile
from aimc_moe.training.noisy_forward import wrap_model_noisy, unwrap_model
from aimc_moe.reporting.report import AIMCReport
from aimc_moe.reporting.plots import plot_thermal_heatmap
from aimc_moe.reporting.sweeps import run_scientific_sweeps

def add_model_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("Model Architecture Configuration")
    group.add_argument("--model-type", default="qwen2_moe", choices=["qwen2_moe", "deepseek_moe", "olmoe"], help="Architecture style to surrogate")
    group.add_argument("--d-model", type=int, default=128, help="Hidden embedding dimension size")
    group.add_argument("--d-ff", type=int, default=256, help="Intermediate FFN projection size")
    group.add_argument("--num-experts", type=int, default=4, help="Total local experts in MoE layers")
    group.add_argument("--n-layers", type=int, default=2, help="Number of decoder blocks")
    group.add_argument("--top-k", type=int, default=1, help="Active experts routed per token")
    group.add_argument("--seq-len", type=int, default=256, help="Active sequence length context")
    group.add_argument("--batch-size", type=int, default=1, help="Batch size context")

def parse_mapping_strategy(strategy_str: str) -> MappingStrategy:
    s = strategy_str.lower().strip()
    if s == "sequential":
        return MappingStrategy.SEQUENTIAL
    elif s == "expert-colocate":
        return MappingStrategy.EXPERT_COLOCATE
    elif s == "thermal-aware":
        return MappingStrategy.THERMAL_AWARE
    return MappingStrategy.EXPERT_COLOCATE

def main():
    parser = argparse.ArgumentParser(
        description="aimc-moe-sim: Compilation and simulation toolchain for 3D Stacked Analog MoE CIM."
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommand to execute")
    
    # 1. TRACE Command
    parser_trace = subparsers.add_parser("trace", help="Trace a model structure and compile to AIMCGraph IR")
    add_model_args(parser_trace)
    parser_trace.add_argument("--save-graph", help="Optional path to write serialized AIMCGraph JSON file")
    
    # 2. MAP Command
    parser_map = subparsers.add_parser("map", help="Compile a model and map its weights onto a target YAML fabric")
    add_model_args(parser_map)
    parser_map.add_argument("--fabric", required=True, help="Path to fabric YAML config file")
    parser_map.add_argument("--strategy", default="expert-colocate", choices=["sequential", "expert-colocate", "thermal-aware"], help="Layout strategy")
    
    # 3. SIMULATE Command
    parser_sim = subparsers.add_parser("simulate", help="Compile, map, and run full performance + thermal + noise simulations")
    add_model_args(parser_sim)
    parser_sim.add_argument("--fabric", required=True, help="Path to fabric YAML config file")
    parser_sim.add_argument("--strategy", default="expert-colocate", choices=["sequential", "expert-colocate", "thermal-aware"], help="Layout strategy")
    parser_sim.add_argument("--noise-profile", default="realistic_pcm", help="Noise profile preset name")
    parser_sim.add_argument("--ops-per-second", type=float, default=2e9, help="OPS throughput target rate for thermal solving")
    parser_sim.add_argument("--output", default="outputs/cli_report.md", help="Path to write markdown simulation report")
    parser_sim.add_argument("--plot-thermal", help="Optional path to save tiered thermal heatmap plot")
    
    # 4. SWEEP Command
    parser_sweep = subparsers.add_parser("sweep", help="Run multi-variable scientific sweeps and output comparative charts")
    parser_sweep.add_argument("--output-dir", default="outputs", help="Output directory for markdown tables")
    parser_sweep.add_argument("--plot-dir", default="plots", help="Output directory for curves")
    
    args = parser.parse_args()
    
    if args.command == "sweep":
        print(f"[*] Running scientific sweep suite. Outputs: {args.output_dir}, plots: {args.plot-dir if hasattr(args, 'plot-dir') else args.plot_dir}")
        run_scientific_sweeps(output_dir=args.output_dir, plot_dir=args.plot_dir)
        sys.exit(0)
        
    # Build surrogate model for trace, map, and simulate
    print(f"[*] Instantiating {args.model_type} surrogate model ({args.d_model} d_model, {args.num_experts} experts)...")
    model = get_moe_surrogate(
        model_type=args.model_type,
        vocab_size=1000,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_experts=args.num_experts,
        n_layers=args.n_layers,
        top_k=args.top_k
    )
    model.eval()
    
    print("[*] Tracing model topology to AIMCGraph IR...")
    graph = trace_hf_model(
        model,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        model_name=f"{args.model_type}-surrogate"
    )
    
    if args.command == "trace":
        print(graph.summary())
        if args.save_graph:
            os.makedirs(os.path.dirname(args.save_graph) or ".", exist_ok=True)
            with open(args.save_graph, "w", encoding="utf-8") as f:
                json.dump(graph.to_dict(), f, indent=2)
            print(f"[*] Wrote graph JSON structure to {args.save_graph}")
            
    elif args.command in ("map", "simulate"):
        print(f"[*] Loading fabric description from {args.fabric}...")
        fabric = load_fabric_from_yaml(args.fabric)
        print(fabric.summary())
        
        strategy = parse_mapping_strategy(args.strategy)
        print(f"[*] Mapping IR nodes using strategy: {strategy.value}...")
        mapping = map_graph_to_fabric(graph, fabric, strategy=strategy)
        print(mapping.summary())
        
        if args.command == "simulate":
            print(f"[*] Simulating performance using technology preset and interconnect routing...")
            interconnect = InterconnectModel()
            lat = compute_total_latency(graph, mapping, fabric, interconnect)
            energy = compute_total_energy(graph, mapping, fabric, interconnect)
            area = compute_total_area(fabric, mapping)
            
            print(f"[*] Solving steady-state thermal RC mesh at {args.ops_per_second:.2e} OPS...")
            thermal_res = solve_steady_state_temperatures(
                fabric, mapping, ops_per_second=args.ops_per_second
            )
            print(f"    Peak Temperature: {thermal_res.max_temperature:.2f}°C")
            
            print(f"[*] Simulating forward accuracy degradation under {args.noise_profile} noise...")
            noise = get_noise_profile(args.noise_profile)
            inputs = torch.randint(0, 1000, (1, args.seq_len))
            with torch.no_grad():
                clean_logits = model(inputs)
                
            wrap_model_noisy(model, noise)
            with torch.no_grad():
                noisy_logits = model(inputs)
            unwrap_model(model)
            
            clean_loss = nn.functional.cross_entropy(clean_logits.view(-1, 1000), inputs.view(-1)).item()
            noisy_loss = nn.functional.cross_entropy(noisy_logits.view(-1, 1000), inputs.view(-1)).item()
            
            # Formulate full report
            report = AIMCReport(
                model_name=f"{args.model_type}-surrogate",
                total_params=graph.total_params(),
                fabric_summary=fabric.summary(),
                technology_name=fabric.crossbar_config.technology.name,
                noise_profile_name=noise.name,
                latency=lat,
                energy=energy,
                area=area,
                baseline_metric=clean_loss,
                baseline_metric_name="cross-entropy loss",
                noisy_metric=noisy_loss
            )
            
            report.print_summary()
            
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(report.to_markdown())
            print(f"[*] Saved simulation report to {args.output}")
            
            if args.plot_thermal:
                os.makedirs(os.path.dirname(args.plot_thermal) or ".", exist_ok=True)
                plot_thermal_heatmap(fabric, thermal_res, save_path=args.plot_thermal)
                print(f"[*] Saved thermal plot to {args.plot_thermal}")

if __name__ == "__main__":
    main()
