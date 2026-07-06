"""Metric <-> Imperial conversions for the dashboard's inputs and results tables.

Physics, optimization, the CLI, and saved defaults always stay in canonical SI
(m, kg, rad) - these conversions exist only at the web UI's display/input
boundary (see web/app.py); nothing else in the package imports this module.
"""

from typing import Optional, Tuple

METERS_PER_FOOT = 0.3048
KG_PER_LB = 0.45359237


def meters_to_feet_inches(value_m: float) -> Tuple[int, float]:
    """Split a length into whole feet and remaining inches (0 <= inches < 12)."""
    total_inches = value_m / METERS_PER_FOOT * 12
    feet = int(total_inches // 12)
    inches = total_inches - feet * 12
    return feet, inches


def feet_inches_to_meters(feet: Optional[float], inches: Optional[float]) -> float:
    """Combine feet and inches (either may be None, treated as 0) into meters."""
    return ((feet or 0) * 12 + (inches or 0)) / 12 * METERS_PER_FOOT


def kg_to_lb(value_kg: float) -> float:
    return value_kg / KG_PER_LB


def lb_to_kg(value_lb: float) -> float:
    return value_lb * KG_PER_LB


def mps_to_fps(value_mps: float) -> float:
    return value_mps / METERS_PER_FOOT


def fps_to_mps(value_fps: float) -> float:
    return value_fps * METERS_PER_FOOT
