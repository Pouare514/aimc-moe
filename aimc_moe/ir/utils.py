from __future__ import annotations
import json
import os
from typing import Dict, Any
import torch
from aimc_moe.ir.graph import AIMCGraph
from aimc_moe.ir.ops import AIMCOp, OpType

def save_graph(graph: AIMCGraph, path: str) -> None:
    """
    Save graph metadata to JSON and its weights to a separate PyTorch file (.pt).
    """
    base_path, _ = os.path.splitext(path)
    json_path = base_path + ".json"
    weights_path = base_path + "_weights.pt"
    
    # Extract weights and save them
    weights_dict = {}
    for op in graph.ops:
        if op.weights is not None:
            weights_dict[f"{op.id}.weights"] = op.weights
        if op.bias is not None:
            weights_dict[f"{op.id}.bias"] = op.bias
            
    if weights_dict:
        torch.save(weights_dict, weights_path)
        
    # Serialize structure
    graph_dict = graph.to_dict()
    graph_dict["weights_file"] = os.path.basename(weights_path) if weights_dict else None
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(graph_dict, f, indent=2)

def load_graph(path: str) -> AIMCGraph:
    """
    Load graph metadata from JSON and its weights from the PyTorch file if present.
    """
    base_path, _ = os.path.splitext(path)
    json_path = base_path + ".json"
    weights_path = base_path + "_weights.pt"
    
    with open(json_path, "r", encoding="utf-8") as f:
        graph_dict = json.load(f)
        
    weights_dict = {}
    if os.path.exists(weights_path):
        weights_dict = torch.load(weights_path)
        
    ops = []
    for op_data in graph_dict["ops"]:
        op_id = op_data["id"]
        weights = weights_dict.get(f"{op_id}.weights")
        bias = weights_dict.get(f"{op_id}.bias")
        
        op = AIMCOp(
            id=op_id,
            op_type=OpType(op_data["op_type"]),
            shape=tuple(op_data["shape"]),
            weights=weights,
            bias=bias,
            expert_id=op_data.get("expert_id"),
            layer_id=op_data.get("layer_id"),
            metadata=op_data.get("metadata", {})
        )
        ops.append(op)
        
    graph = AIMCGraph(
        ops=ops,
        edges=[(edge[0], edge[1]) for edge in graph_dict["edges"]],
        model_config=graph_dict.get("model_config", {}),
        batch_config=graph_dict.get("batch_config", {"seq_len": 2048, "batch_size": 1})
    )
    return graph

def print_graph_table(graph: AIMCGraph) -> None:
    """
    Print a neat text-based table summarizing the graph operations.
    """
    headers = ["ID", "Type", "Shape", "Parameters", "Mappable", "Layer ID", "Expert ID"]
    col_widths = [30, 15, 15, 12, 10, 10, 10]
    
    # Print header
    header_str = " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
    print(header_str)
    print("-" * len(header_str))
    
    for op in graph.topological_order():
        shape_str = str(op.shape)
        params_str = f"{op.num_params:,}" if op.num_params > 0 else "0"
        mappable_str = "Yes" if op.is_mappable else "No"
        layer_str = str(op.layer_id) if op.layer_id is not None else "-"
        expert_str = str(op.expert_id) if op.expert_id is not None else "-"
        
        row_fields = [
            op.id[:30],
            op.op_type.value,
            shape_str,
            params_str,
            mappable_str,
            layer_str,
            expert_str
        ]
        row_str = " | ".join(f"{str(field):<{w}}" for field, w in zip(row_fields, col_widths))
        print(row_str)
