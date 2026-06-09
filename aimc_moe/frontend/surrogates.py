from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional

class SurrogateConfig:
    """Mock config matching HuggingFace structure."""
    def __init__(self, hidden_size: int, num_local_experts: int, num_experts_per_tok: int):
        self.hidden_size = hidden_size
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok

# Generic surrogate components that use HF module naming structures
class SurrogateAttention(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        return self.o_proj(context)

class SurrogateExpert(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff)
        self.up_proj = nn.Linear(d_model, d_ff)
        self.down_proj = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

class SurrogateMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_experts: int, top_k: int):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([
            SurrogateExpert(d_model, d_ff)
            for _ in range(num_experts)
        ])
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        flat_x = x.view(-1, orig_shape[-1])
        gate_logits = self.gate(flat_x)
        probs = torch.softmax(gate_logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, k=self.top_k, dim=-1)
        
        out = torch.zeros_like(flat_x)
        for i, expert in enumerate(self.experts):
            mask = (top_indices == i).any(dim=-1)
            if mask.any():
                expert_out = expert(flat_x[mask])
                # Scaling output by the first gating weight for simplicity in mock
                expert_out = expert_out * probs[mask, i:i+1]
                out[mask] += expert_out
        return out.view(*orig_shape)

class SurrogateDecoderLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_experts: int, top_k: int):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(d_model)
        self.self_attn = SurrogateAttention(d_model)
        self.post_attention_layernorm = nn.LayerNorm(d_model)
        self.mlp = SurrogateMLP(d_model, d_ff, num_experts, top_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

class SurrogateModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, d_ff: int, num_experts: int, n_layers: int, top_k: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            SurrogateDecoderLayer(d_model, d_ff, num_experts, top_k)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

# Model Wrapper classes to present the right class names for reporting
class Qwen2MoeForCausalLM(nn.Module):
    def __init__(self, config: SurrogateConfig, vocab_size: int, d_ff: int, n_layers: int):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.d_model = config.hidden_size
        self.model = SurrogateModel(vocab_size, config.hidden_size, d_ff, config.num_local_experts, n_layers, config.num_experts_per_tok)
        self.lm_head = nn.Linear(config.hidden_size, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.model(x))

class DeepSeekMoEForCausalLM(nn.Module):
    def __init__(self, config: SurrogateConfig, vocab_size: int, d_ff: int, n_layers: int):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.d_model = config.hidden_size
        self.model = SurrogateModel(vocab_size, config.hidden_size, d_ff, config.num_local_experts, n_layers, config.num_experts_per_tok)
        self.lm_head = nn.Linear(config.hidden_size, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.model(x))

class OLMoEForCausalLM(nn.Module):
    def __init__(self, config: SurrogateConfig, vocab_size: int, d_ff: int, n_layers: int):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.d_model = config.hidden_size
        self.model = SurrogateModel(vocab_size, config.hidden_size, d_ff, config.num_local_experts, n_layers, config.num_experts_per_tok)
        self.lm_head = nn.Linear(config.hidden_size, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.model(x))


def get_moe_surrogate(
    model_type: str,
    vocab_size: int = 1000,
    d_model: int = 128,
    d_ff: int = 256,
    num_experts: int = 4,
    n_layers: int = 2,
    top_k: int = 1
) -> nn.Module:
    """
    Build a surrogate model structure that behaves like the specified HF architecture class.
    Supported model types: 'qwen2_moe', 'deepseek_moe', 'olmoe'.
    """
    config = SurrogateConfig(
        hidden_size=d_model,
        num_local_experts=num_experts,
        num_experts_per_tok=top_k
    )
    
    m_type = model_type.lower().strip()
    if m_type == "qwen2_moe":
        return Qwen2MoeForCausalLM(config, vocab_size, d_ff, n_layers)
    elif m_type == "deepseek_moe":
        return DeepSeekMoEForCausalLM(config, vocab_size, d_ff, n_layers)
    elif m_type == "olmoe":
        return OLMoEForCausalLM(config, vocab_size, d_ff, n_layers)
    else:
        # Fallback to standard Qwen2MoE style
        return Qwen2MoeForCausalLM(config, vocab_size, d_ff, n_layers)
