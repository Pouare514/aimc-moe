from __future__ import annotations
import re
from typing import List, Tuple, Dict, Any, Optional
import torch
import torch.nn as nn

from aimc_moe.ir.ops import AIMCOp, OpType
from aimc_moe.ir.graph import AIMCGraph
from aimc_moe.models.expert import ExpertMLP
from aimc_moe.models.router import TopKRouter
from aimc_moe.models.moe_transformer import MoEBlock

def _extract_layer_id(name: str) -> Optional[int]:
    match = re.search(r'layers\.(\d+)', name)
    if match:
        return int(match.group(1))
    return None

def trace_moe_model(
    model: nn.Module,
    seq_len: int = 2048,
    batch_size: int = 1,
    model_name: str = "moe_model"
) -> AIMCGraph:
    """
    Hybrid module-walking tracer for MoE models.
    
    Traverses the model's named module hierarchy to extract parameters and shapes.
    Builds sequential edges based on the execution order of standard Transformer layers.
    """
    ops: List[AIMCOp] = []
    edges: List[Tuple[str, str]] = []
    
    # 1. Identify input/embeddings
    # We look for token embedding and positional embedding
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
            
    # Connect embeddings to input
    input_id = "input"
    ops.insert(0, AIMCOp(id=input_id, op_type=OpType.ACTIVATION, shape=(batch_size, seq_len)))
    for emb_id in embeddings:
        edges.append((input_id, emb_id))
        
    last_op_id = embeddings[-1] if embeddings else input_id
    
    # 2. Iterate through layers
    # We look specifically for Transformer block layers
    # To be general, let's sort layers by their names to maintain topological execution order
    block_pattern = re.compile(r'layers\.(\d+)$')
    blocks = []
    for name, module in model.named_modules():
        if block_pattern.search(name):
            blocks.append((name, module))
            
    # Sort blocks by layer index
    blocks.sort(key=lambda item: int(block_pattern.search(item[0]).group(1)))
    
    for block_name, block in blocks:
        layer_idx = _extract_layer_id(block_name)
        
        # Inside each block, we have norm1, attn, norm2, ffn (MoEBlock)
        # Let's inspect submodules
        norm1 = getattr(block, "norm1", None)
        attn = getattr(block, "attn", None)
        norm2 = getattr(block, "norm2", None)
        ffn = getattr(block, "ffn", None)
        
        # Add norm1
        if norm1 is not None:
            op_id = f"{block_name}.norm1"
            ops.append(AIMCOp(
                id=op_id, op_type=OpType.NORM, shape=norm1.normalized_shape,
                weights=norm1.weight.data.clone() if norm1.elementwise_affine else None,
                bias=norm1.bias.data.clone() if norm1.elementwise_affine else None,
                layer_id=layer_idx
            ))
            edges.append((last_op_id, op_id))
            last_op_id = op_id
            
        # Add attention projection operations
        # MultiHeadAttention has q_proj, k_proj, v_proj, out_proj
        if attn is not None:
            attn_id = f"{block_name}.attn"
            # We map q, k, v, out projections
            q_id = f"{attn_id}.q_proj"
            k_id = f"{attn_id}.k_proj"
            v_id = f"{attn_id}.v_proj"
            out_id = f"{attn_id}.out_proj"
            
            # Create ops
            q_op = AIMCOp(id=q_id, op_type=OpType.LINEAR, shape=attn.q_proj.weight.shape,
                          weights=attn.q_proj.weight.data.clone(),
                          bias=attn.q_proj.bias.data.clone() if attn.q_proj.bias is not None else None,
                          layer_id=layer_idx)
            k_op = AIMCOp(id=k_id, op_type=OpType.LINEAR, shape=attn.k_proj.weight.shape,
                          weights=attn.k_proj.weight.data.clone(),
                          bias=attn.k_proj.bias.data.clone() if attn.k_proj.bias is not None else None,
                          layer_id=layer_idx)
            v_op = AIMCOp(id=v_id, op_type=OpType.LINEAR, shape=attn.v_proj.weight.shape,
                          weights=attn.v_proj.weight.data.clone(),
                          bias=attn.v_proj.bias.data.clone() if attn.v_proj.bias is not None else None,
                          layer_id=layer_idx)
            out_op = AIMCOp(id=out_id, op_type=OpType.LINEAR, shape=attn.out_proj.weight.shape,
                            weights=attn.out_proj.weight.data.clone(),
                            bias=attn.out_proj.bias.data.clone() if attn.out_proj.bias is not None else None,
                            layer_id=layer_idx)
            
            ops.extend([q_op, k_op, v_op, out_op])
            
            # Connect. q, k, v are computed in parallel
            edges.append((last_op_id, q_id))
            edges.append((last_op_id, k_id))
            edges.append((last_op_id, v_id))
            
            # Attn output connects to out_proj
            edges.append((q_id, out_id))
            edges.append((k_id, out_id))
            edges.append((v_id, out_id))
            
            last_op_id = out_id
            
        # Add norm2
        if norm2 is not None:
            op_id = f"{block_name}.norm2"
            ops.append(AIMCOp(
                id=op_id, op_type=OpType.NORM, shape=norm2.normalized_shape,
                weights=norm2.weight.data.clone() if norm2.elementwise_affine else None,
                bias=norm2.bias.data.clone() if norm2.elementwise_affine else None,
                layer_id=layer_idx
            ))
            edges.append((last_op_id, op_id))
            last_op_id = op_id
            
        # Add FFN / MoEBlock
        if ffn is not None and isinstance(ffn, MoEBlock):
            # Trace the MoE Block
            moe_id = f"{block_name}.ffn"
            
            # Add Router
            router_id = f"{moe_id}.router"
            ops.append(AIMCOp(
                id=router_id,
                op_type=OpType.MOE_ROUTER,
                shape=ffn.router.gate.weight.shape,
                weights=ffn.router.gate.weight.data.clone(),
                layer_id=layer_idx,
                metadata={"num_experts": ffn.num_experts, "top_k": ffn.top_k}
            ))
            edges.append((last_op_id, router_id))
            
            # Add Experts
            expert_ops_ids = []
            for exp_idx, expert in enumerate(ffn.experts):
                exp_id = f"{moe_id}.expert{exp_idx}"
                
                # Each expert has gate_proj and down_proj
                gate_proj_id = f"{exp_id}.gate_proj"
                down_proj_id = f"{exp_id}.down_proj"
                
                gate_op = AIMCOp(
                    id=gate_proj_id,
                    op_type=OpType.MOE_EXPERT,
                    shape=expert.gate_proj.weight.shape,
                    weights=expert.gate_proj.weight.data.clone(),
                    bias=expert.gate_proj.bias.data.clone() if expert.gate_proj.bias is not None else None,
                    layer_id=layer_idx,
                    expert_id=exp_idx
                )
                
                down_op = AIMCOp(
                    id=down_proj_id,
                    op_type=OpType.MOE_EXPERT,
                    shape=expert.down_proj.weight.shape,
                    weights=expert.down_proj.weight.data.clone(),
                    bias=expert.down_proj.bias.data.clone() if expert.down_proj.bias is not None else None,
                    layer_id=layer_idx,
                    expert_id=exp_idx
                )
                
                ops.extend([gate_op, down_op])
                
                # Router selects expert, input connects to gate_proj
                edges.append((last_op_id, gate_proj_id))
                edges.append((router_id, gate_proj_id)) # gating signal
                edges.append((gate_proj_id, down_proj_id))
                expert_ops_ids.append(down_proj_id)
                
            # Aggregate outputs
            agg_id = f"{moe_id}.aggregation"
            ops.append(AIMCOp(id=agg_id, op_type=OpType.ACTIVATION, shape=()))
            for exp_out_id in expert_ops_ids:
                edges.append((exp_out_id, agg_id))
                
            last_op_id = agg_id
            
    # 3. Final norm and output head
    final_norm = getattr(model, "final_norm", None)
    output_head = getattr(model, "output_head", None)
    
    if final_norm is not None:
        op_id = "final_norm"
        ops.append(AIMCOp(
            id=op_id, op_type=OpType.NORM, shape=final_norm.normalized_shape,
            weights=final_norm.weight.data.clone() if final_norm.elementwise_affine else None,
            bias=final_norm.bias.data.clone() if final_norm.elementwise_affine else None
        ))
        edges.append((last_op_id, op_id))
        last_op_id = op_id
        
    if output_head is not None:
        op_id = "output_head"
        ops.append(AIMCOp(
            id=op_id, op_type=OpType.LINEAR, shape=output_head.weight.shape,
            weights=output_head.weight.data.clone(),
            bias=output_head.bias.data.clone() if output_head.bias is not None else None
        ))
        edges.append((last_op_id, op_id))
        last_op_id = op_id
        
    # Output node
    ops.append(AIMCOp(id="output", op_type=OpType.OUTPUT, shape=(batch_size, seq_len, model.vocab_size)))
    edges.append((last_op_id, "output"))
    
    model_config = {
        "model_name": model_name,
        "d_model": getattr(model, "d_model", None),
        "n_heads": getattr(model, "n_heads", None),
        "n_layers": getattr(model, "n_layers", None),
        "d_ff": getattr(model, "d_ff", None),
        "num_experts": getattr(model, "num_experts", None),
        "top_k": getattr(model, "top_k", None),
        "vocab_size": getattr(model, "vocab_size", None)
    }
    
    return AIMCGraph(
        ops=ops,
        edges=edges,
        model_config=model_config,
        batch_config={"seq_len": seq_len, "batch_size": batch_size}
    )
