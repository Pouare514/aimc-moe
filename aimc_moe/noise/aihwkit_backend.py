from __future__ import annotations
import logging
from typing import Optional
import torch.nn as nn

try:
    from aihwkit.nn import AnalogLinear
    from aihwkit.nn.conversion import convert_to_analog
    from aihwkit.simulator.configs import InferenceRPUConfig
    from aihwkit.inference import PCMLikeNoiseModel, GlobalDriftCompensation
    HAS_AIHWKIT = True
except ImportError:
    HAS_AIHWKIT = False

logger = logging.getLogger(__name__)

def check_aihwkit_available() -> bool:
    return HAS_AIHWKIT

def get_aihwkit_rpu_config(g_max: float = 25.0, w_noise: float = 0.02, out_noise: float = 0.04) -> Optional[InferenceRPUConfig]:
    """
    Generate an IBM AIHWKIT InferenceRPUConfig with PCMLikeNoiseModel.
    Returns None if aihwkit is not installed.
    """
    if not HAS_AIHWKIT:
        logger.warning("aihwkit is not installed. Returning None for RPU config.")
        return None
        
    config = InferenceRPUConfig()
    config.noise_model = PCMLikeNoiseModel(g_max=g_max)
    config.drift_compensation = GlobalDriftCompensation()
    config.forward.w_noise = w_noise
    config.forward.out_noise = out_noise
    return config

def convert_linear_to_analog(layer: nn.Linear, rpu_config: Optional[InferenceRPUConfig] = None) -> nn.Module:
    """
    Convert a PyTorch nn.Linear layer into an AIHWKIT AnalogLinear layer.
    """
    if not HAS_AIHWKIT:
        raise ImportError(
            "IBM AIHWKIT is required for this action. "
            "Note that AIHWKIT is officially supported on Linux/WSL. "
            "Install it via: pip install aihwkit"
        )
        
    if rpu_config is None:
        rpu_config = get_aihwkit_rpu_config()
        
    return AnalogLinear.from_digital(layer, rpu_config=rpu_config)

def convert_model_to_analog(model: nn.Module, rpu_config: Optional[InferenceRPUConfig] = None) -> nn.Module:
    """
    Convert an entire digital PyTorch model to its analog AIHWKIT equivalent.
    """
    if not HAS_AIHWKIT:
        raise ImportError("IBM AIHWKIT is required for this action.")
        
    if rpu_config is None:
        rpu_config = get_aihwkit_rpu_config()
        
    return convert_to_analog(model, rpu_config)
