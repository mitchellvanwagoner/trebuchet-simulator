"""Trebuchet configuration parameters and physical constants."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

G = 9.81                        # gravity (m/s^2)
RHO_AIR = 1.225                 # air density at sea level (kg/m^3)
ARM_CROSS_SECTION_WIDTH = 0.05  # arm cross-section width, used for mass and drag (m)

# Canonical defaults for the five optimizable parameters, shared by the CLI,
# the web UI, and the tests so they can't drift apart.
DEFAULT_OPTIMIZABLE_PARAMS = {
    "counter_weight_mass": 16.865,  # kg
    "pulley_radius": 0.121,         # m
    "arm_length": 0.813,            # m
    "string_length": 0.669,         # m
    "release_angle": -4.877,        # radians
}


@dataclass
class TrebuchetParams:
    """Trebuchet configuration parameters."""

    counter_weight_mass: float                       # kg
    pulley_radius: float                              # m
    arm_length: float                                 # m
    string_length: float                              # m
    release_angle: float                              # radians

    pivot_height: float = 1.0                          # m (height of the arm pivot above the ground)
    pulley_density: float = 1250                      # kg/m^3
    arm_density: float = 530                          # kg/m^3
    counter_weight_density: float = 7850               # kg/m^3, steel - sizes the counterweight's cube for ground collision
    projectile_mass: float = 0.25                     # kg (apple)
    projectile_radius: float = 0.04                   # m
    initial_arm_angle: float = np.pi / 4               # 45 degree start position
    arm_drag_coefficient: float = 1.05
    projectile_drag_coefficient: float = 0.47
    joint_friction_coefficient: float = 0.01          # N*m*s/rad (viscous damper at pivot)
    counter_weight_rope_length: Optional[float] = None  # m; rope from the pivot axle to the counterweight at
                                                         # t=0. None defaults to 2x pulley radius (one wrap).

    # Thickness of the pulley disc used for its mass: 1 inch, matching the
    # plywood stock of the physical build (see CAD/).
    PULLEY_THICKNESS = 0.0254  # m

    @property
    def weight_height(self) -> float:
        """Counterweight release height; the weight hangs from the pivot axle."""
        return self.pivot_height

    @property
    def initial_cw_rope_length(self) -> float:
        """Rope length from the pivot axle to the counterweight at t=0.

        Defaults to twice the pulley radius (a single wrap) when
        `counter_weight_rope_length` isn't explicitly set.
        """
        if self.counter_weight_rope_length is not None:
            return self.counter_weight_rope_length
        return 2 * self.pulley_radius

    @property
    def pulley_mass(self) -> float:
        """Mass of a solid disc: density * (circle area) * thickness."""
        return self.pulley_density * np.pi * self.pulley_radius**2 * self.PULLEY_THICKNESS

    @property
    def arm_mass(self) -> float:
        """Mass of a square-section beam: density * length * width^2."""
        return self.arm_density * self.arm_length * ARM_CROSS_SECTION_WIDTH**2

    @property
    def counter_weight_size(self) -> float:
        """Side length of the counterweight, modeled as a solid cube: (mass/density)^(1/3).

        Used so the aftermath ground-collision check (and the 3D render) land the
        counterweight's bottom face on the ground rather than its center of mass.
        """
        return (self.counter_weight_mass / self.counter_weight_density) ** (1 / 3)

    @property
    def projectile_area(self) -> float:
        return np.pi * self.projectile_radius**2

    @property
    def moi_pulley(self) -> float:
        return 0.5 * self.pulley_mass * self.pulley_radius**2

    @property
    def moi_arm(self) -> float:
        return (1 / 3) * self.arm_mass * self.arm_length**2

    @property
    def total_mass(self) -> float:
        return self.counter_weight_mass + self.pulley_mass + self.arm_mass + self.projectile_mass

    @property
    def string_arm_ratio(self) -> float:
        """String to arm length ratio."""
        return self.string_length / self.arm_length

    @property
    def arm_string_clearance(self) -> float:
        """Ground clearance when arm + sling hang straight down, minus a 0.1 m
        safety margin. Negative means the projectile could strike the ground
        during the swing."""
        return self.pivot_height - (self.arm_length + self.string_length + 0.1)
