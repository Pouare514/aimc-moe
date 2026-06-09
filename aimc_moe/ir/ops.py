from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, Any
import torch

class OpType(Enum):
    LINEAR = "linear"
    ATTENTION = "attention"
    MOE_EXPERT = "moe_expert"
    MOE_ROUTER = "moe_router"
    ACTIVATION = "activation"
    NORM = "normalization"
    EMBEDDING = "embedding"
    RESIDUAL = "residual"
    SOFTMAX = "softmax"
    OUTPUT = "output"

@dataclass
class AIMCOp:
    id: str                                    # Unique ID like "layer0.ffn.linear1"
    op_type: OpType
    shape: Tuple[int, ...]                     # (in_features, out_features) for linear
    weights: Optional[torch.Tensor] = None
    bias: Optional[torch.Tensor] = None
    expert_id: Optional[int] = None
    layer_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def num_params(self) -> int:
        count = 0
        if self.weights is not None:
            count += self.weights.numel()
        if self.bias is not None:
            count += self.bias.numel()
        return count
    
    @property
    def is_mappable(self) -> bool:
        """Whether this op has weights that can be mapped to crossbars."""
        return self.op_type in (OpType.LINEAR, OpType.MOE_EXPERT) and self.weights is not None
    
    def __repr__(self) -> str:
        return f"AIMCOp(id={self.id!r}, type={self.op_type.value}, shape={self.shape}, params={self.num_params})"
