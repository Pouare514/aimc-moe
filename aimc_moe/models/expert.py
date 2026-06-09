from __future__ import annotations
import torch.nn as nn
from torch import Tensor

class ExpertMLP(nn.Module):
    """
    Single MoE expert: two-layer MLP with activation.
    
    This acts as the contract class. MoE experts must subclass this
    or be instances of it for the MoE tracer to identify them.
    """
    def __init__(self, d_model: int, d_ff: int, expert_id: int, 
                 activation: str = "gelu", dropout: float = 0.0):
        super().__init__()
        self.expert_id = expert_id
        self.gate_proj = nn.Linear(d_model, d_ff)   # mapping upwards
        self.down_proj = nn.Linear(d_ff, d_model)    # mapping back down
        
        if activation.lower() == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.down_proj(self.activation(self.gate_proj(x))))
