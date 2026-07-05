"""Trebuchet physics simulation, optimization, and visualization toolkit."""

from trebuchet_sim.config import TrebuchetParams
from trebuchet_sim.physics import SimulationResult, TrebuchetSimulator, simulate_trebuchet

__all__ = [
    "TrebuchetParams",
    "SimulationResult",
    "TrebuchetSimulator",
    "simulate_trebuchet",
]
