from __future__ import annotations
from typing import Tuple, Optional
import torch
import torch.nn as nn
from torch import Tensor

class TopKRouter(nn.Module):
    """Top-K expert routing with load balancing."""
    def __init__(self, d_model: int, num_experts: int, top_k: int = 1,
                 noise_std: float = 0.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Returns (dispatch_weights, expert_indices, aux_loss).
        dispatch_weights: (batch, seq, top_k) - normalized weights
        expert_indices: (batch, seq, top_k) - selected expert ids
        aux_loss: scalar load balancing loss
        """
        # x is (batch, seq, d_model)
        batch_size, seq_len, d_model = x.shape
        flat_x = x.view(-1, d_model)
        
        logits = self.gate(flat_x)
        
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise
            
        # Top-K selection
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        
        # Softmax over top-K
        dispatch_weights = torch.softmax(top_k_logits, dim=-1)
        
        # Compute load balancing loss
        # We need density and fraction of selection
        # Gating probabilities (full softmax)
        gate_probs = torch.softmax(logits, dim=-1)
        
        # Fraction of tokens dispatched to each expert
        # For each token, did it select expert i?
        selection = torch.zeros(flat_x.shape[0], self.num_experts, device=x.device)
        selection.scatter_(1, top_k_indices, 1.0)
        f_i = selection.mean(dim=0)  # average routing frequency
        
        # Gating probability sum per expert
        P_i = gate_probs.mean(dim=0)
        
        # GShard auxiliary loss: num_experts * sum(f_i * P_i)
        aux_loss = self.num_experts * torch.sum(f_i * P_i)
        
        # Reshape to match batch/sequence dims
        dispatch_weights = dispatch_weights.view(batch_size, seq_len, self.top_k)
        top_k_indices = top_k_indices.view(batch_size, seq_len, self.top_k)
        
        return dispatch_weights, top_k_indices, aux_loss
