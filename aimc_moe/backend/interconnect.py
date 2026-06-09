from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
from aimc_moe.backend.crossbar import CrossbarTile
from aimc_moe.backend.technology import TechnologyProfile

@dataclass
class InterconnectModel:
    hop_latency_ns: float = 2.0
    vertical_hop_latency_ns: float = 5.0
    bandwidth_gbps: float = 256.0
    
    def hops_between(self, tile_a: CrossbarTile, tile_b: CrossbarTile) -> Tuple[int, int]:
        """Returns (horizontal_hops, vertical_hops) using Manhattan distance in 3D space."""
        h_hops = abs(tile_a.position[0] - tile_b.position[0]) + abs(tile_a.position[1] - tile_b.position[1])
        v_hops = abs(tile_a.tier - tile_b.tier)
        return h_hops, v_hops
        
    def latency_between(self, tile_a: CrossbarTile, tile_b: CrossbarTile) -> float:
        """Total link latency in ns between two tiles."""
        h_hops, v_hops = self.hops_between(tile_a, tile_b)
        return h_hops * self.hop_latency_ns + v_hops * self.vertical_hop_latency_ns
        
    def data_transfer_latency_ns(self, payload_bytes: int) -> float:
        """Time to transfer a data payload over the network in ns."""
        if payload_bytes <= 0:
            return 0.0
        bits = payload_bytes * 8
        seconds = bits / (self.bandwidth_gbps * 1e9)
        return seconds * 1e9
        
    def energy_between(self, tile_a: CrossbarTile, tile_b: CrossbarTile, tech: TechnologyProfile) -> float:
        """Energy in pJ for sending a standard signal between two tiles."""
        h_hops, v_hops = self.hops_between(tile_a, tile_b)
        total_hops = h_hops + v_hops
        return total_hops * tech.energy_interconnect_per_hop
