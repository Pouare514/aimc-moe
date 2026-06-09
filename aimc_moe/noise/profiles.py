from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

@dataclass
class NoiseProfile:
    name: str = "custom"
    
    # Static variations (device-to-device)
    sigma_d2d: float = 0.0         # D2D variation (σ/µ ratio, e.g., 0.05–0.30)
    saf_rate: float = 0.0          # Stuck-at-fault probability (e.g., 0.001–0.01)
    
    # Dynamic / temporal noise
    sigma_read: float = 0.0        # Read noise std dev (relative)
    sigma_prog: float = 0.0       # Programming noise std dev (relative)
    
    # Drift (PCM-specific)
    drift_nu: float = 0.0          # Drift coefficient ν (0.05–0.1 typical)
    drift_nu_std: float = 0.0      # Variability in ν
    drift_t0: float = 1.0          # Reference time (seconds)
    
    # I-V nonlinearity
    nonlinearity_coeff: float = 0.0  # Curvature parameter (0 = linear)
    
    # ADC/DAC quantization
    adc_bits: int = 32             # Effective ADC precision (32 = ideal)
    dac_bits: int = 32             # Effective DAC precision (32 = ideal)
    
    # Position-dependent IR drop
    ir_drop_alpha: float = 0.0     # first-order drop coefficient
    ir_drop_beta: float = 0.0      # second-order drop coefficient
    wire_resistance: float = 0.5   # wire resistance per segment (Ohms)
    
    @property
    def is_ideal(self) -> bool:
        return (self.sigma_d2d == 0.0 and self.saf_rate == 0.0 and
                self.sigma_read == 0.0 and self.sigma_prog == 0.0 and
                self.drift_nu == 0.0 and self.nonlinearity_coeff == 0.0 and
                self.ir_drop_alpha == 0.0 and self.ir_drop_beta == 0.0 and
                self.adc_bits >= 16 and self.dac_bits >= 16)

# Presets
IDEAL = NoiseProfile(name="ideal")

REALISTIC_RRAM = NoiseProfile(
    name="realistic_rram",
    sigma_d2d=0.10,
    saf_rate=0.001,
    sigma_read=0.02,
    sigma_prog=0.05,
    nonlinearity_coeff=0.1,
    adc_bits=6,
    dac_bits=4,
    ir_drop_alpha=0.0002,
    ir_drop_beta=0.00003,
    wire_resistance=0.8
)

REALISTIC_PCM = NoiseProfile(
    name="realistic_pcm",
    sigma_d2d=0.08,
    saf_rate=0.0005,
    sigma_read=0.03,
    sigma_prog=0.04,
    drift_nu=0.06,
    drift_nu_std=0.02,
    drift_t0=1.0,
    adc_bits=8,
    dac_bits=4
)

WORST_CASE = NoiseProfile(
    name="worst_case",
    sigma_d2d=0.30,
    saf_rate=0.01,
    sigma_read=0.05,
    sigma_prog=0.10,
    drift_nu=0.10,
    drift_nu_std=0.05,
    nonlinearity_coeff=0.3,
    adc_bits=4,
    dac_bits=2
)

NOISE_PRESETS: Dict[str, NoiseProfile] = {
    "ideal": IDEAL,
    "realistic_rram": REALISTIC_RRAM,
    "realistic_pcm": REALISTIC_PCM,
    "worst_case": WORST_CASE
}

def get_noise_profile(name: str) -> NoiseProfile:
    profile = NOISE_PRESETS.get(name.lower())
    if profile is None:
        raise ValueError(f"Unknown noise profile: '{name}'. Available presets: {list(NOISE_PRESETS.keys())}")
    return profile
