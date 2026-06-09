from __future__ import annotations
import os
import yaml
from typing import Dict, Any

from aimc_moe.backend.technology import get_technology
from aimc_moe.backend.crossbar import CrossbarConfig
from aimc_moe.backend.fabric import Fabric3D

def load_fabric_from_yaml(yaml_path: str) -> Fabric3D:
    """
    Parse a fabric description from a YAML file.
    
    YAML Format:
    ------------
    mesh_rows: 8
    mesh_cols: 8
    n_tiers: 4
    crossbar:
      rows: 256
      cols: 256
      technology: pcm_gst  # or rram_hfo2, ideal
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
    with open(yaml_path, 'r') as f:
        config_data = yaml.safe_load(f)
        
    mesh_rows = config_data.get("mesh_rows", 8)
    mesh_cols = config_data.get("mesh_cols", 8)
    n_tiers = config_data.get("n_tiers", 1)
    
    xb_data = config_data.get("crossbar", {})
    xb_rows = xb_data.get("rows", 256)
    xb_cols = xb_data.get("cols", 256)
    
    tech_name = xb_data.get("technology", "pcm_gst")
    tech = get_technology(tech_name)
    
    xb_config = CrossbarConfig(
        rows=xb_rows,
        cols=xb_cols,
        technology=tech
    )
    
    return Fabric3D(
        mesh_rows=mesh_rows,
        mesh_cols=mesh_cols,
        n_tiers=n_tiers,
        crossbar_config=xb_config
    )
