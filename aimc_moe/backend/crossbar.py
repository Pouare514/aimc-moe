from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, List
from aimc_moe.backend.technology import TechnologyProfile, RRAM_HfO2

@dataclass
class CrossbarConfig:
    rows: int = 256
    cols: int = 256
    technology: TechnologyProfile = field(default_factory=lambda: RRAM_HfO2)
    
    @property
    def capacity(self) -> int:
        return self.rows * self.cols
        
    @property
    def tile_area_mm2(self) -> float:
        # core area + peripheral overhead
        core = self.rows * self.cols * self.technology.cell_area_um2
        return core * self.technology.peripheral_overhead / 1e6  # um² → mm²

@dataclass
class CrossbarTile:
    tile_id: str
    config: CrossbarConfig
    tier: int = 0
    position: Tuple[int, int] = (0, 0)  # (row, col) in the 2D mesh on this tier
    assigned_ops: List[Tuple[str, Tuple[int, int, int, int]]] = field(default_factory=list)
    # Each assignment is: (op_id, (row_start, row_end, col_start, col_end))
    
    # Track current allocation cursor to fill tile sequentially (greedy)
    _current_row_ptr: int = field(init=False, default=0)
    _current_col_ptr: int = field(init=False, default=0)
    
    @property
    def utilization(self) -> float:
        total_cells = self.config.rows * self.config.cols
        if total_cells == 0:
            return 0.0
        used_cells = sum((r_end - r_start) * (c_end - c_start) 
                         for _, (r_start, r_end, c_start, c_end) in self.assigned_ops)
        return float(used_cells) / total_cells
        
    @property
    def remaining_capacity(self) -> Tuple[int, int]:
        return (self.config.rows - self._current_row_ptr, self.config.cols - self._current_col_ptr)
        
    def can_fit(self, rows_needed: int, cols_needed: int) -> bool:
        # Simplistic mapping: check if remaining rows and columns can fit the block
        # Or more simply, check if there's enough capacity in the current cursor position
        return (self._current_row_ptr + rows_needed <= self.config.rows and
                self._current_col_ptr + cols_needed <= self.config.cols)
                
    def assign(self, op_id: str, rows: int, cols: int) -> Tuple[int, int, int, int]:
        if not self.can_fit(rows, cols):
            raise ValueError(f"Not enough space in tile {self.tile_id} for ({rows}x{cols}) block.")
            
        r_start = self._current_row_ptr
        r_end = r_start + rows
        c_start = self._current_col_ptr
        c_end = c_start + cols
        
        assignment = (r_start, r_end, c_start, c_end)
        self.assigned_ops.append((op_id, assignment))
        
        # Advance pointers: we allocate column blocks first, then wrap around row-wise
        # This is a basic placement scheme.
        self._current_col_ptr += cols
        if self._current_col_ptr >= self.config.cols:
            self._current_col_ptr = 0
            self._current_row_ptr += rows
            
        return assignment
