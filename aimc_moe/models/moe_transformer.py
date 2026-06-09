from __future__ import annotations
from typing import Tuple, Optional, Dict, Any
import torch
import torch.nn as nn
from torch import Tensor

from aimc_moe.models.expert import ExpertMLP
from aimc_moe.models.router import TopKRouter
from aimc_moe.models.transformer import MultiHeadAttention

class MoEBlock(nn.Module):
    """MoE feed-forward block: router + list of ExpertMLP."""
    def __init__(self, d_model: int, d_ff: int, num_experts: int = 8,
                 top_k: int = 1, dropout: float = 0.0):
        super().__init__()
        self.router = TopKRouter(d_model, num_experts, top_k)
        self.experts = nn.ModuleList([
            ExpertMLP(d_model, d_ff, expert_id=i, dropout=dropout)
            for i in range(num_experts)
        ])
        self.top_k = top_k
        self.num_experts = num_experts
        
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Returns (output, aux_loss).
        """
        batch_size, seq_len, d_model = x.shape
        flat_x = x.view(-1, d_model)  # (N, D)
        
        # Route tokens
        weights, indices, aux_loss = self.router(x)
        flat_weights = weights.view(-1, self.top_k)  # (N, K)
        flat_indices = indices.view(-1, self.top_k)  # (N, K)
        
        out = torch.zeros_like(flat_x)
        
        # Route tokens to active experts
        for exp_idx in range(self.num_experts):
            mask = (flat_indices == exp_idx)
            if not mask.any():
                continue
                
            token_ids, slot_ids = torch.where(mask)
            exp_inputs = flat_x[token_ids]
            exp_outputs = self.experts[exp_idx](exp_inputs)
            
            exp_weights = flat_weights[token_ids, slot_ids].unsqueeze(-1)
            out.index_add_(0, token_ids, exp_outputs * exp_weights)
            
        return out.view(batch_size, seq_len, d_model), aux_loss

class MoETransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, 
                 num_experts: int = 8, top_k: int = 1, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = MoEBlock(d_model, d_ff, num_experts, top_k, dropout)
        
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        x = x + self.attn(self.norm1(x), mask)
        ffn_out, aux_loss = self.ffn(self.norm2(x))
        x = x + ffn_out
        return x, aux_loss

class MoETransformer(nn.Module):
    def __init__(self, vocab_size: int = 10000, d_model: int = 256, n_heads: int = 4,
                 n_layers: int = 4, d_ff: int = 1024, num_experts: int = 8, top_k: int = 1,
                 max_seq_len: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k = top_k
        self.max_seq_len = max_seq_len
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.layers = nn.ModuleList([
            MoETransformerBlock(d_model, n_heads, d_ff, num_experts, top_k, dropout)
            for _ in range(n_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        batch_size, seq_len = x.shape
        assert seq_len <= self.max_seq_len, f"Sequence length {seq_len} exceeds max {self.max_seq_len}"
        
        pos = torch.arange(0, seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
        
        x = self.token_embedding(x) + self.pos_embedding(pos)
        x = self.dropout(x)
        
        total_aux_loss = torch.tensor(0.0, device=x.device)
        for layer in self.layers:
            x, aux_loss = layer(x, mask)
            total_aux_loss = total_aux_loss + aux_loss
            
        x = self.final_norm(x)
        logits = self.output_head(x)
        return logits, total_aux_loss
        
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> MoETransformer:
        return cls(
            vocab_size=config.get("vocab_size", 10000),
            d_model=config.get("d_model", 256),
            n_heads=config.get("n_heads", 4),
            n_layers=config.get("n_layers", 4),
            d_ff=config.get("d_ff", 1024),
            num_experts=config.get("num_experts", 8),
            top_k=config.get("top_k", 1),
            max_seq_len=config.get("max_seq_len", 2048),
            dropout=config.get("dropout", 0.0)
        )
