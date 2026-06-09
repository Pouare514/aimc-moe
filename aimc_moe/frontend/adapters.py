from __future__ import annotations
import re
from typing import Dict, Any, Tuple, List, Optional
import torch
import torch.nn as nn

from aimc_moe.ir.ops import AIMCOp, OpType
from aimc_moe.ir.graph import AIMCGraph

def _infer_layer_idx(name: str) -> Optional[int]:
    match = re.search(r'layers\.(\d+)', name)
    if match:
        return int(match.group(1))
    return None

def trace_hf_model(
    model: nn.Module,
    seq_len: int = 2048,
    batch_size: int = 1,
    model_name: Optional[str] = None
) -> AIMCGraph:
    """
    Ingest a HuggingFace MoE or dense transformer model (e.g. Qwen2Moe, Llama) 
    and compile it into a structured AIMCGraph.
    
    Tries to map standard layer names:
    - self_attn.q_proj, k_proj, v_proj, o_proj -> attention linear projections
    - mlp.gate / mlp.router -> MoE router
    - mlp.experts.X.gate_proj, up_proj, down_proj -> MoE expert projection layers
    - embed_tokens -> Embedding
    - lm_head -> Output projection head
    """
    ops: List[AIMCOp] = []
    edges: List[Tuple[str, str]] = []
    
    if model_name is None:
        model_name = model.__class__.__name__
        
    # 1. Identify input and embedding
    embeddings = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            op_id = f"embedding.{name}"
            op = AIMCOp(
                id=op_id,
                op_type=OpType.EMBEDDING,
                shape=module.weight.shape,
                weights=module.weight.data.clone(),
                metadata={"class": module.__class__.__name__}
            )
            ops.append(op)
            embeddings.append(op_id)
            
    input_id = "input"
    ops.insert(0, AIMCOp(id=input_id, op_type=OpType.ACTIVATION, shape=(batch_size, seq_len)))
    for emb_id in embeddings:
        edges.append((input_id, emb_id))
        
    last_op_id = embeddings[-1] if embeddings else input_id
    
    # 2. Extract block layers
    # In standard transformers (Qwen/Llama), blocks are under model.layers.X
    block_pattern = re.compile(r'(?:model\.)?layers\.(\d+)$')
    blocks = []
    for name, module in model.named_modules():
        if block_pattern.search(name):
            blocks.append((name, module))
            
    blocks.sort(key=lambda item: int(block_pattern.search(item[0]).group(1)))
    
    for block_name, block in blocks:
        layer_idx = _infer_layer_idx(block_name)
        
        # Look for Layernorms
        input_ln = None
        post_attn_ln = None
        
        for name, m in block.named_modules():
            if "input_layernorm" in name or "input_ln" in name:
                input_ln = m
            elif "post_attention_layernorm" in name or "post_attn_ln" in name:
                post_attn_ln = m
                
        # Input LayerNorm
        if input_ln is not None:
            op_id = f"{block_name}.input_layernorm"
            ops.append(AIMCOp(
                id=op_id, op_type=OpType.NORM, shape=input_ln.weight.shape,
                weights=input_ln.weight.data.clone() if hasattr(input_ln, "weight") and input_ln.weight is not None else None,
                bias=input_ln.bias.data.clone() if hasattr(input_ln, "bias") and input_ln.bias is not None else None,
                layer_id=layer_idx
            ))
            edges.append((last_op_id, op_id))
            last_op_id = op_id
            
        # Attention Projections
        q_proj = None
        k_proj = None
        v_proj = None
        o_proj = None
        
        for name, m in block.named_modules():
            if isinstance(m, nn.Linear):
                if "q_proj" in name:
                    q_proj = m
                elif "k_proj" in name:
                    k_proj = m
                elif "v_proj" in name:
                    v_proj = m
                elif "o_proj" in name:
                    o_proj = m
                    
        if q_proj is not None and k_proj is not None and v_proj is not None and o_proj is not None:
            attn_id = f"{block_name}.self_attn"
            q_id = f"{attn_id}.q_proj"
            k_id = f"{attn_id}.k_proj"
            v_id = f"{attn_id}.v_proj"
            o_id = f"{attn_id}.o_proj"
            
            ops.extend([
                AIMCOp(id=q_id, op_type=OpType.LINEAR, shape=q_proj.weight.shape, weights=q_proj.weight.data.clone(), layer_id=layer_idx),
                AIMCOp(id=k_id, op_type=OpType.LINEAR, shape=k_proj.weight.shape, weights=k_proj.weight.data.clone(), layer_id=layer_idx),
                AIMCOp(id=v_id, op_type=OpType.LINEAR, shape=v_proj.weight.shape, weights=v_proj.weight.data.clone(), layer_id=layer_idx),
                AIMCOp(id=o_id, op_type=OpType.LINEAR, shape=o_proj.weight.shape, weights=o_proj.weight.data.clone(), layer_id=layer_idx)
            ])
            
            edges.append((last_op_id, q_id))
            edges.append((last_op_id, k_id))
            edges.append((last_op_id, v_id))
            edges.append((q_id, o_id))
            edges.append((k_id, o_id))
            edges.append((v_id, o_id))
            last_op_id = o_id
            
        # Post-Attention LayerNorm
        if post_attn_ln is not None:
            op_id = f"{block_name}.post_attention_layernorm"
            ops.append(AIMCOp(
                id=op_id, op_type=OpType.NORM, shape=post_attn_ln.weight.shape,
                weights=post_attn_ln.weight.data.clone() if hasattr(post_attn_ln, "weight") and post_attn_ln.weight is not None else None,
                bias=post_attn_ln.bias.data.clone() if hasattr(post_attn_ln, "bias") and post_attn_ln.bias is not None else None,
                layer_id=layer_idx
            ))
            edges.append((last_op_id, op_id))
            last_op_id = op_id
            
        # MLP / FFN / MoE Block detection
        # If it contains "experts" submodules, it's a Mixture of Experts
        experts = []
        router = None
        for name, m in block.named_modules():
            if "mlp.gate" in name or "mlp.router" in name:
                router = m
            # Check for expert projection layers
            if "mlp.experts" in name and isinstance(m, nn.Linear):
                # E.g. "mlp.experts.0.gate_proj"
                match_exp = re.search(r'experts\.(\d+)', name)
                if match_exp:
                    exp_idx = int(match_exp.group(1))
                    experts.append((name, m, exp_idx))
                    
        if router is not None and len(experts) > 0:
            moe_id = f"{block_name}.mlp"
            router_id = f"{moe_id}.gate"
            
            ops.append(AIMCOp(
                id=router_id,
                op_type=OpType.MOE_ROUTER,
                shape=router.weight.shape,
                weights=router.weight.data.clone(),
                layer_id=layer_idx
            ))
            edges.append((last_op_id, router_id))
            
            # Group experts
            expert_out_ids = []
            for exp_name, exp_module, exp_idx in experts:
                # E.g. "model.layers.0.mlp.experts.0.gate_proj" -> id will be "mlp.expert0.gate_proj"
                proj_name = exp_name.split(".")[-1] # "gate_proj", "up_proj", "down_proj"
                op_id = f"{moe_id}.expert{exp_idx}.{proj_name}"
                
                ops.append(AIMCOp(
                    id=op_id,
                    op_type=OpType.MOE_EXPERT,
                    shape=exp_module.weight.shape,
                    weights=exp_module.weight.data.clone(),
                    bias=exp_module.bias.data.clone() if exp_module.bias is not None else None,
                    layer_id=layer_idx,
                    expert_id=exp_idx
                ))
                
                # Connect router to gate/up projection
                if proj_name in ("gate_proj", "up_proj"):
                    edges.append((last_op_id, op_id))
                    edges.append((router_id, op_id))
                elif proj_name == "down_proj":
                    # gate/up feed into down_proj
                    gate_id = f"{moe_id}.expert{exp_idx}.gate_proj"
                    up_id = f"{moe_id}.expert{exp_idx}.up_proj"
                    edges.append((gate_id, op_id))
                    edges.append((up_id, op_id))
                    expert_out_ids.append(op_id)
                    
            agg_id = f"{moe_id}.aggregation"
            ops.append(AIMCOp(id=agg_id, op_type=OpType.ACTIVATION, shape=()))
            for out_id in expert_out_ids:
                edges.append((out_id, agg_id))
                
            last_op_id = agg_id
            
        else:
            # Standard dense FeedForward (gate_proj, up_proj, down_proj in SwiGLU, or gate/down in standard FFN)
            lin1 = None
            lin2 = None
            
            for name, m in block.named_modules():
                if isinstance(m, nn.Linear) and "mlp" in name:
                    if "linear1" in name or "gate_proj" in name:
                        lin1 = m
                    elif "linear2" in name or "down_proj" in name:
                        lin2 = m
                        
            if lin1 is not None and lin2 is not None:
                ffn_id = f"{block_name}.mlp"
                l1_id = f"{ffn_id}.linear1"
                l2_id = f"{ffn_id}.linear2"
                
                ops.extend([
                    AIMCOp(id=l1_id, op_type=OpType.LINEAR, shape=lin1.weight.shape, weights=lin1.weight.data.clone(), layer_id=layer_idx),
                    AIMCOp(id=l2_id, op_type=OpType.LINEAR, shape=lin2.weight.shape, weights=lin2.weight.data.clone(), layer_id=layer_idx)
                ])
                edges.append((last_op_id, l1_id))
                edges.append((l1_id, l2_id))
                last_op_id = l2_id
                
    # 3. Final LayerNorm & LM Head
    final_ln = None
    lm_head = None
    
    for name, module in model.named_modules():
        if "norm" in name and name != "layers" and name not in [f"layers.{i}" for i in range(len(blocks))]:
            final_ln = module
        if "lm_head" in name:
            lm_head = module
            
    if final_ln is not None:
        op_id = "final_norm"
        ops.append(AIMCOp(
            id=op_id, op_type=OpType.NORM, shape=final_ln.weight.shape if hasattr(final_ln, "weight") else (),
            weights=final_ln.weight.data.clone() if hasattr(final_ln, "weight") and final_ln.weight is not None else None,
            bias=final_ln.bias.data.clone() if hasattr(final_ln, "bias") and final_ln.bias is not None else None
        ))
        edges.append((last_op_id, op_id))
        last_op_id = op_id
        
    if lm_head is not None:
        op_id = "lm_head"
        ops.append(AIMCOp(
            id=op_id, op_type=OpType.LINEAR, shape=lm_head.weight.shape,
            weights=lm_head.weight.data.clone(),
            bias=lm_head.bias.data.clone() if lm_head.bias is not None else None
        ))
        edges.append((last_op_id, op_id))
        last_op_id = op_id
        
    # Output node
    ops.append(AIMCOp(id="output", op_type=OpType.OUTPUT, shape=(batch_size, seq_len)))
    edges.append((last_op_id, "output"))
    
    return AIMCGraph(
        ops=ops,
        edges=edges,
        model_config={
            "model_name": model_name,
            "d_model": getattr(model.config, "hidden_size", None) if hasattr(model, "config") else None,
            "num_experts": getattr(model.config, "num_local_experts", None) if hasattr(model, "config") else None,
            "top_k": getattr(model.config, "num_experts_per_tok", None) if hasattr(model, "config") else None,
        },
        batch_config={"seq_len": seq_len, "batch_size": batch_size}
    )
