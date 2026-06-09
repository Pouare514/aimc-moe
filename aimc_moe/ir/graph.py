from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
from aimc_moe.ir.ops import AIMCOp, OpType

@dataclass
class AIMCGraph:
    ops: List[AIMCOp] = field(default_factory=list)
    edges: List[Tuple[str, str]] = field(default_factory=list)  # (src_id, dst_id)
    model_config: Dict[str, Any] = field(default_factory=dict)
    batch_config: Dict[str, Any] = field(default_factory=lambda: {"seq_len": 2048, "batch_size": 1})
    
    _op_dict: Dict[str, AIMCOp] = field(init=False, default_factory=dict, repr=False)
    
    def __post_init__(self):
        self._op_dict = {op.id: op for op in self.ops}
        
    def add_op(self, op: AIMCOp) -> None:
        if op.id in self._op_dict:
            raise ValueError(f"Operation with id '{op.id}' already exists in the graph.")
        self.ops.append(op)
        self._op_dict[op.id] = op
        
    def add_edge(self, src_id: str, dst_id: str) -> None:
        self.edges.append((src_id, dst_id))
        
    def get_op(self, op_id: str) -> Optional[AIMCOp]:
        return self._op_dict.get(op_id)
        
    def get_ops_by_type(self, op_type: OpType) -> List[AIMCOp]:
        return [op for op in self.ops if op.op_type == op_type]
        
    def get_linear_ops(self) -> List[AIMCOp]:
        return self.get_ops_by_type(OpType.LINEAR)
        
    def get_expert_ops(self, layer_id: Optional[int] = None) -> List[AIMCOp]:
        expert_ops = self.get_ops_by_type(OpType.MOE_EXPERT)
        if layer_id is not None:
            expert_ops = [op for op in expert_ops if op.layer_id == layer_id]
        return expert_ops
        
    def get_mappable_ops(self) -> List[AIMCOp]:
        return [op for op in self.ops if op.is_mappable]
        
    def get_successors(self, op_id: str) -> List[str]:
        return [dst for src, dst in self.edges if src == op_id]
        
    def get_predecessors(self, op_id: str) -> List[str]:
        return [src for src, dst in self.edges if dst == op_id]
        
    def topological_order(self) -> List[AIMCOp]:
        # Kahn's algorithm
        in_degree = {op.id: 0 for op in self.ops}
        for src, dst in self.edges:
            if dst in in_degree:
                in_degree[dst] += 1
                
        # Find start nodes
        queue = [op.id for op in self.ops if in_degree[op.id] == 0]
        # To maintain deterministic topological order, we sort queue
        queue.sort()
        
        order = []
        visited_count = 0
        
        while queue:
            node_id = queue.pop(0)
            order.append(self._op_dict[node_id])
            visited_count += 1
            
            successors = self.get_successors(node_id)
            for succ in sorted(successors):
                if succ in in_degree:
                    in_degree[succ] -= 1
                    if in_degree[succ] == 0:
                        queue.append(succ)
                        
        if visited_count != len(self.ops):
            # Graph has a cycle, fallback to returning nodes in insertion order
            return list(self.ops)
            
        return order
        
    def total_params(self) -> int:
        return sum(op.num_params for op in self.ops)
        
    def total_mappable_params(self) -> int:
        return sum(op.num_params for op in self.ops if op.is_mappable)
        
    def summary(self) -> str:
        lines = [
            "=================== AIMC Graph Summary ===================",
            f"Total Operations: {len(self.ops)}",
            f"Total Edges: {len(self.edges)}",
            f"Total Parameters: {self.total_params():,}",
            f"Mappable Parameters: {self.total_mappable_params():,}",
            "Operations by Type:"
        ]
        type_counts = {}
        for op in self.ops:
            type_counts[op.op_type.value] = type_counts.get(op.op_type.value, 0) + 1
        for otype, count in sorted(type_counts.items()):
            lines.append(f"  - {otype}: {count}")
        lines.append("==========================================================")
        return "\n".join(lines)
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ops": [
                {
                    "id": op.id,
                    "op_type": op.op_type.value,
                    "shape": list(op.shape),
                    "has_weights": op.weights is not None,
                    "has_bias": op.bias is not None,
                    "expert_id": op.expert_id,
                    "layer_id": op.layer_id,
                    "metadata": op.metadata
                }
                for op in self.ops
            ],
            "edges": self.edges,
            "model_config": self.model_config,
            "batch_config": self.batch_config
        }
        
    def validate(self) -> List[str]:
        errors = []
        op_ids = set()
        for op in self.ops:
            if op.id in op_ids:
                errors.append(f"Duplicate operation ID found: '{op.id}'")
            op_ids.add(op.id)
            
        for src, dst in self.edges:
            if src not in op_ids:
                errors.append(f"Edge source not found in operations: '{src}'")
            if dst not in op_ids:
                errors.append(f"Edge destination not found in operations: '{dst}'")
                
        # Cycle detection
        in_degree = {op.id: 0 for op in self.ops}
        for src, dst in self.edges:
            if dst in in_degree:
                in_degree[dst] += 1
        queue = [node_id for node_id, deg in in_degree.items() if deg == 0]
        visited = 0
        while queue:
            node_id = queue.pop(0)
            visited += 1
            for succ in self.get_successors(node_id):
                if succ in in_degree:
                    in_degree[succ] -= 1
                    if in_degree[succ] == 0:
                        queue.append(succ)
        if visited != len(self.ops):
            errors.append("Cycle detected in the graph.")
            
        return errors
