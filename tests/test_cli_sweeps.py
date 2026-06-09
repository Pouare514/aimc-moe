from __future__ import annotations
import os
import sys
import pytest
import torch
import yaml
from unittest.mock import patch

from aimc_moe.frontend.surrogates import get_moe_surrogate
from aimc_moe.frontend.adapters import trace_hf_model
from aimc_moe.backend.configs import load_fabric_from_yaml
from aimc_moe.cli import main
from aimc_moe.reporting.sweeps import run_scientific_sweeps


def test_surrogate_models():
    """Verify that surrogate architectures generate correctly and are traceable."""
    model_types = ["qwen2_moe", "deepseek_moe", "olmoe"]
    
    for mtype in model_types:
        model = get_moe_surrogate(
            model_type=mtype,
            vocab_size=100,
            d_model=64,
            d_ff=128,
            num_experts=2,
            n_layers=1,
            top_k=1
        )
        
        assert isinstance(model, torch.nn.Module)
        
        # Test trace compatibility
        graph = trace_hf_model(model, seq_len=32, batch_size=1)
        assert graph.total_params() > 0
        assert len(graph.get_expert_ops()) > 0


def test_yaml_config_presets(tmp_path):
    """Verify loading pre-packaged YAML config files and custom configurations."""
    # 1. Custom temp YAML test
    yaml_content = {
        "mesh_rows": 4,
        "mesh_cols": 4,
        "n_tiers": 2,
        "crossbar": {
            "rows": 128,
            "cols": 128,
            "technology": "rram_hfo2"
        }
    }
    
    config_file = tmp_path / "test_fabric.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f)
        
    fabric = load_fabric_from_yaml(str(config_file))
    assert fabric.mesh_rows == 4
    assert fabric.mesh_cols == 4
    assert fabric.n_tiers == 2
    assert fabric.crossbar_config.rows == 128
    assert fabric.crossbar_config.cols == 128
    assert fabric.crossbar_config.technology.name == "RRAM_HfO2"
    
    # 2. Package preset test
    preset_path = "configs/fabrics/rram_edge_2d.yaml"
    if os.path.exists(preset_path):
        preset_fabric = load_fabric_from_yaml(preset_path)
        assert preset_fabric.n_tiers == 1
        assert preset_fabric.crossbar_config.technology.name == "RRAM_HfO2"


def test_cli_subcommands(tmp_path):
    """Verify CLI main entrypoint executions for trace, map, and simulate commands."""
    os.makedirs(str(tmp_path / "outputs"), exist_ok=True)
    os.makedirs(str(tmp_path / "plots"), exist_ok=True)
    
    # Setup a dummy fabric configuration
    yaml_content = {
        "mesh_rows": 4,
        "mesh_cols": 4,
        "n_tiers": 2,
        "crossbar": {
            "rows": 128,
            "cols": 128,
            "technology": "pcm_gst"
        }
    }
    config_file = tmp_path / "test_fabric.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f)
        
    output_report = tmp_path / "outputs" / "test_report.md"
    plot_thermal = tmp_path / "plots" / "test_thermal.png"
    save_graph = tmp_path / "outputs" / "test_graph.json"
    
    # 1. Test cli 'trace'
    test_args_trace = [
        "aimc-moe-sim", "trace",
        "--model-type", "olmoe",
        "--n-layers", "1",
        "--num-experts", "2",
        "--save-graph", str(save_graph)
    ]
    with patch("sys.argv", test_args_trace):
        main()
    assert os.path.exists(save_graph)
    
    # 2. Test cli 'map'
    test_args_map = [
        "aimc-moe-sim", "map",
        "--model-type", "qwen2_moe",
        "--n-layers", "1",
        "--num-experts", "2",
        "--fabric", str(config_file),
        "--strategy", "thermal-aware"
    ]
    with patch("sys.argv", test_args_map):
        main()
        
    # 3. Test cli 'simulate'
    test_args_sim = [
        "aimc-moe-sim", "simulate",
        "--model-type", "deepseek_moe",
        "--n-layers", "1",
        "--num-experts", "2",
        "--fabric", str(config_file),
        "--strategy", "expert-colocate",
        "--output", str(output_report),
        "--plot-thermal", str(plot_thermal)
    ]
    with patch("sys.argv", test_args_sim):
        main()
    assert os.path.exists(output_report)
    assert os.path.exists(plot_thermal)


def test_cli_sweep_command(tmp_path):
    """Verify CLI 'sweep' command integration."""
    out_dir = tmp_path / "outputs_sweep"
    plt_dir = tmp_path / "plots_sweep"
    
    test_args = [
        "aimc-moe-sim", "sweep",
        "--output-dir", str(out_dir),
        "--plot-dir", str(plt_dir)
    ]
    
    with patch("sys.argv", test_args):
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 0
        
    assert os.path.exists(os.path.join(out_dir, "sweep_report.md"))
    assert os.path.exists(os.path.join(plt_dir, "sweep_throughput_temp.png"))
    assert os.path.exists(os.path.join(plt_dir, "sweep_latency_tiers.png"))
    assert os.path.exists(os.path.join(plt_dir, "sweep_accuracy_noise.png"))
