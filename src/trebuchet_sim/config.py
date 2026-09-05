"""Trebuchet configuration parameters and physical constants."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

G = 9.81                        # gravity (m/s^2)
RHO_AIR = 1.225                 # air density at sea level (kg/m^3)
ARM_CROSS_SECTION_WIDTH = 0.05  # arm cross-section width, used for mass and drag (m)

# Canonical defaults for the five optimizable parameters, shared by the CLI,
# the web UI, and the tests so they can't drift apart. Optimizer output for the
# 30 m target with the rope-slack penalty active: the sling stays taut for the
# whole launch (min tension ~5 N), so the rigid-link model - and the ~91%
# efficiency it reports - is physically valid for this set.
DEFAULT_OPTIMIZABLE_PARAMS = {
    "counter_weight_mass": 54.989,  # kg
    "pulley_radius": 0.0173,        # m
    "arm_length": 0.406,            # m
    "string_length": 0.235,         # m
    "release_angle": -4.228,        # radians
}


DEFAULT_INITIAL_ARM_ANGLE = {}  # populated below, once MachineType exists


class MachineType(str, Enum):
    """Which counterweight linkage the machine uses.

    The two share one set of equations of motion; they differ only in how the
    counterweight couples to the arm angle (see TrebuchetSimulator.__init__):

    PULLEY      - the counterweight hangs from a rope over a pulley on the pivot
                  axle, so it travels straight down at `pulley_radius` metres per
                  radian of arm rotation. Its lever arm is constant.
    TRADITIONAL - the counterweight is bolted to the arm itself, `length_counterweight`
                  from the pivot on the far side from the sling, so it swings on
                  a circle with the arm. Its lever arm varies with cos(theta).

    A str Enum so it round-trips through the saved-defaults JSON unchanged.
    """

    PULLEY = "pulley"
    TRADITIONAL = "traditional"


# Canonical defaults for the traditional machine. The arm starts cocked at -135
# degrees (long arm down and forward, counterweight raised behind the pivot) with
# the sling hanging below the tip, and the counterweight rides the arm rather than
# a pulley - so `length_counterweight` replaces `pulley_radius` as the linkage
# parameter. Geometry is chosen so the projectile starts at ground level: the tip
# sits at pivot_height - arm_length*sin(45 deg) and the sling hangs string_length
# below it.
# Cocked positions. The pulley machine starts with the arm raised at 45 degrees;
# the traditional one starts at -135 degrees, long arm down and forward with the
# counterweight raised behind the pivot, which is the mirror image about the
# vertical - both then rotate in the same (decreasing-theta) direction.
DEFAULT_INITIAL_ARM_ANGLE.update({
    MachineType.PULLEY: np.pi / 4,
    MachineType.TRADITIONAL: -3 * np.pi / 4,
})


DEFAULT_TRADITIONAL_PARAMS = {
    "counter_weight_mass": 50.0,     # kg
    "length_counterweight": 0.35,    # m
    "arm_length": 1.8,               # m
    "string_length": 1.35,           # m
    "release_angle": -4.94,          # radians; swept for maximum range on this geometry
}

# Fixed (never-optimized) fields whose defaults also differ per machine. The pivot
# is tall enough that the cocked arm tip sits one sling length above the ground, so
# the projectile starts resting on it; the rope length is the pin-to-weight link.
DEFAULT_TRADITIONAL_FIXED = {
    "pivot_height": 2.6,                 # m
    "counter_weight_rope_length": 0.5,   # m
}


# The one design variable that has no counterpart on the other machine: the pulley's
# radius sets how far the weight falls per radian of arm rotation, while the traditional
# machine's short arm sets how far the weight sits from the pivot. Both are "the size of
# the counterweight linkage", both are worth optimizing, and neither means anything on
# the other machine - so the search space swaps one for the other (see
# optimization.param_names).
LINKAGE_PARAM = {
    MachineType.PULLEY: "pulley_radius",
    MachineType.TRADITIONAL: "length_counterweight",
}

# Starting point for each machine's design variables, keyed the way the CLI and the web
# UI both want them. Selecting a machine loads its set: the two are different enough
# that carrying numbers across (a 0.4 m arm onto a 50 kg counterweight, say) would
# simulate a machine nobody asked for.
DEFAULT_MACHINE_PARAMS = {
    MachineType.PULLEY: DEFAULT_OPTIMIZABLE_PARAMS,
    MachineType.TRADITIONAL: DEFAULT_TRADITIONAL_PARAMS,
}

# Per-machine overrides for the never-optimized fields. Only the differences are listed;
# anything absent falls back to the TrebuchetParams default, so there is one place to
# change a value that both machines share.
DEFAULT_MACHINE_FIXED = {
    MachineType.PULLEY: {},
    MachineType.TRADITIONAL: DEFAULT_TRADITIONAL_FIXED,
}


@dataclass
class TrebuchetParams:
    """Trebuchet configuration parameters."""

    counter_weight_mass: float                       # kg
    arm_length: float                                 # m
    string_length: float                              # m
    release_angle: float                              # radians

    machine: MachineType = MachineType.PULLEY
    pulley_radius: float = DEFAULT_OPTIMIZABLE_PARAMS["pulley_radius"]      # m; PULLEY only
    length_counterweight: float = 0.35                                       # m; TRADITIONAL only

    pivot_height: float = 1.0                          # m (height of the arm pivot above the ground)
    pulley_density: float = 1250                      # kg/m^3
    arm_density: float = 530                          # kg/m^3
    counter_weight_density: float = 7850               # kg/m^3, steel - sizes the counterweight's cube for ground collision
    projectile_mass: float = 0.25                     # kg (apple)
    projectile_radius: float = 0.04                   # m
    initial_arm_angle: Optional[float] = None          # radians; None -> the machine's default
                                                       # (see DEFAULT_INITIAL_ARM_ANGLE)
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

    def __post_init__(self) -> None:
        # Accept a plain string (e.g. from saved defaults JSON) as the machine.
        self.machine = MachineType(self.machine)
        if self.initial_arm_angle is None:
            self.initial_arm_angle = DEFAULT_INITIAL_ARM_ANGLE[self.machine]

    @property
    def has_pulley(self) -> bool:
        return self.machine is MachineType.PULLEY

    @property
    def arm_back_length(self) -> float:
        """How far the arm extends behind the pivot, opposite the sling.

        Zero on the pulley machine (the arm is a single beam from the pivot to
        the sling); `length_counterweight` on the traditional one, where the
        counterweight is bolted to the arm's short end.
        """
        return 0.0 if self.has_pulley else self.length_counterweight

    @property
    def arm_total_length(self) -> float:
        """Full beam length, both sides of the pivot."""
        return self.arm_length + self.arm_back_length

    @property
    def counter_weight_lever(self) -> float:
        """Radius at which the counterweight's mass acts about the pivot.

        The pulley converts arm rotation into vertical weight travel at
        `pulley_radius` metres per radian, so that is its effective lever;
        on the traditional machine the weight simply rides the arm.
        """
        return self.pulley_radius if self.has_pulley else self.length_counterweight

    @property
    def pulley_mass(self) -> float:
        """Mass of a solid disc: density * (circle area) * thickness. Zero when
        the machine has no pulley."""
        if not self.has_pulley:
            return 0.0
        return self.pulley_density * np.pi * self.pulley_radius**2 * self.PULLEY_THICKNESS

    @property
    def arm_mass(self) -> float:
        """Mass of a square-section beam spanning both sides of the pivot."""
        return self.arm_density * self.arm_total_length * ARM_CROSS_SECTION_WIDTH**2

    @property
    def arm_cm_offset(self) -> float:
        """Signed distance from the pivot to the arm's centre of mass, along the
        arm's long (sling) direction.

        A uniform beam running from -b to +a about the pivot balances at
        (a^2 - b^2) / 2(a + b), which factors to (a - b) / 2 - and with b = 0 that
        is exactly the half-length the single-sided pulley arm uses. The factored
        form matters: it reduces to the original expression bit-for-bit, so adding
        the back section changes nothing about the pulley machine's numbers.
        """
        a, b = self.arm_length, self.arm_back_length
        return (a - b) / 2

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
        """Beam inertia about the pivot, integrating both sides.

        For a uniform rod from -b to +a this is m (a^3 + b^3) / 3(a + b); the sum of
        cubes factors, leaving m (a^2 - ab + b^2) / 3, which collapses to the
        familiar m*a^2/3 exactly when the arm has no back section.
        """
        a, b = self.arm_length, self.arm_back_length
        return (1 / 3) * self.arm_mass * (a**2 - a * b + b**2)

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
