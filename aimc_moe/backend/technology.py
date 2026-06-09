from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

@dataclass
class TechnologyProfile:
    name: str
    device_type: str  # "RRAM" or "PCM"
    
    # Conductance (µS)
    g_min: float       # HRS
    g_max: float       # LRS
    on_off_ratio: float
    
    # Precision
    n_analog_levels: int  # 16-32 RRAM, 16-64 PCM
    adc_bits: int         # 4-8
    dac_bits: int         # 1-8
    
    # Area
    cell_area_factor: float    # k in k*F^2 (4 for 1R, 6-10 for 1T1R)
    feature_size_nm: float     # tech node
    peripheral_overhead: float # 1.5-2.0x
    
    # Timing (ns)
    t_dac: float = 10.0
    t_crossbar: float = 1.0
    t_adc: float = 35.0
    t_activation: float = 5.0
    t_control: float = 10.0
    
    # Energy (pJ)
    energy_per_mac: float = 0.189
    energy_adc: float = 0.5
    energy_dac: float = 0.1
    energy_interconnect_per_hop: float = 0.5
    
    @property
    def cycle_time_ns(self) -> float:
        return self.t_dac + self.t_crossbar + self.t_adc
    
    @property
    def cell_area_um2(self) -> float:
        f_um = self.feature_size_nm / 1000.0
        return self.cell_area_factor * f_um * f_um

# Preset constants
RRAM_HfO2 = TechnologyProfile(
    name="RRAM_HfO2",
    device_type="RRAM",
    g_min=0.1,
    g_max=100.0,
    on_off_ratio=1000.0,
    n_analog_levels=32,
    adc_bits=6,
    dac_bits=4,
    cell_area_factor=4.0,
    feature_size_nm=65.0,
    peripheral_overhead=1.8,
    energy_per_mac=0.189
)

PCM_GST = TechnologyProfile(
    name="PCM_GST",
    device_type="PCM",
    g_min=0.1,
    g_max=50.0,
    on_off_ratio=500.0,
    n_analog_levels=16,
    adc_bits=8,
    dac_bits=4,
    cell_area_factor=6.0,
    feature_size_nm=65.0,
    peripheral_overhead=2.0,
    energy_per_mac=1.0,
    t_adc=40.0
)

IDEAL = TechnologyProfile(
    name="IDEAL",
    device_type="IDEAL",
    g_min=0.0,
    g_max=1.0,
    on_off_ratio=float("inf"),
    n_analog_levels=1000,
    adc_bits=32,
    dac_bits=32,
    cell_area_factor=0.0,
    feature_size_nm=0.0,
    peripheral_overhead=1.0,
    t_dac=0.0,
    t_crossbar=0.0,
    t_adc=0.0,
    t_activation=0.0,
    t_control=0.0,
    energy_per_mac=0.0,
    energy_adc=0.0,
    energy_dac=0.0,
    energy_interconnect_per_hop=0.0
)

TECHNOLOGY_PRESETS: Dict[str, TechnologyProfile] = {
    "rram_hfo2": RRAM_HfO2,
    "pcm_gst": PCM_GST,
    "ideal": IDEAL
}

def get_technology(name: str) -> TechnologyProfile:
    tech = TECHNOLOGY_PRESETS.get(name.lower())
    if tech is None:
        raise ValueError(f"Unknown technology profile: '{name}'. Available presets: {list(TECHNOLOGY_PRESETS.keys())}")
    return tech
