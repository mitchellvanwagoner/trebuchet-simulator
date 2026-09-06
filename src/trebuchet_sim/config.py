"""Trebuchet configuration parameters and physical constants."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

G = 9.81                        # gravity (m/s^2)
RHO_AIR = 1.225                 # air density at sea level (kg/m^3)
ARM_CROSS_SECTION_WIDTH = 0.05  # arm cross-section width, used for mass and drag (m)

# How hard the sling has to stay loaded before a launch counts as "not snappy", in
# multiples of the projectile's own weight. It is a design margin, not a physical
# threshold: a sling only actually goes slack at zero tension, but a launch that runs
# a hair above zero is one gust - or one build tolerance - away from detaching, and
# every such solution jerks. Measured across randomized designs that stay taut, the
# time spent below one projectile weight predicts whether a +-10% parameter change
# tips the design into snapping (r = 0.81); the compression impulse alone, which is
# zero until tension actually goes negative, predicts almost nothing (r = 0.23)
# because it is flat across every design that has not already failed.
SLING_TENSION_FLOOR = 1.0

# Canonical defaults for the five optimizable parameters, shared by the CLI,
# the web UI, and the tests so they can't drift apart. Optimizer output for the
# 30 m target with the rope-slack penalty active: the sling stays taut for the
# whole launch and the projectile never touches the ground, so nothing here is
# resting on a regime the model handles but a builder would not want.
#
# Re-swept when `pivot_height` rose to 2.5 m. The pulley machine's equations of motion
# do not contain the pivot height, so the previous set still launched identically - it
# just released 1.5 m higher and landed 31.37 m out, overshooting the target it was named
# for by 4.6%. The re-swept set costs 0.5 points of efficiency (90.4% against 90.9%) and
# buys back the 1.37 m, which is the trade the shipped weights ask for: distance 10
# against efficiency 5.
DEFAULT_OPTIMIZABLE_PARAMS = {
    "counter_weight_mass": 46.111,  # kg
    "pulley_radius": 0.0197,        # m
    "arm_length": 0.4322,           # m
    "string_length": 0.2502,        # m
    "release_angle": -4.3237,       # radians
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
# degrees (long arm down and forward, counterweight raised behind the pivot) and the
# counterweight rides the arm rather than a pulley - so `length_counterweight` replaces
# `pulley_radius` as the linkage parameter. Geometry is chosen so the tip stands one sling
# length above the ground, which is what lets the machine be loaded the way a real one is:
# the projectile lies on the ground at the far end of a sling stretched back behind the
# pivot (see physics.ground_start_state), rather than dangling from the tip - which on this
# geometry put it 23 mm underground.
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
    # Swept for maximum range on this geometry, which means the pose above: laid back
    # along the ground rather than hanging. The two differ by 10.5 degrees of initial sling
    # lean and the sweep by 4.4 degrees of arm - -4.94 was the answer for the hanging pose,
    # and reads 63.9 m on the real one against this angle's 70.2 m.
    "release_angle": -5.016,         # radians
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

    # Tall enough to swing every arm in PARAM_BOUNDS, which is the point of the number:
    # a beam longer than its pivot is tall reaches the ground partway round and the launch
    # ends there, so a shorter default would put most of the search range out of reach (at
    # the old 1 m it put all but the bottom 40% of it there). The traditional machine
    # overrides it upward again for its own geometry - see DEFAULT_TRADITIONAL_FIXED.
    pivot_height: float = 2.5                          # m (height of the arm pivot above the ground)
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
        """Length of the link the counterweight hangs on at t=0.

        Explicit when `counter_weight_rope_length` is set; otherwise each machine falls
        back to its own linkage. The pulley machine hangs its weight from a rope over the
        axle, so one wrap of the pulley is the natural length. The traditional machine has
        no pulley at all, and used to inherit that same expression anyway - which meant its
        counterweight swung on a length derived from `pulley_radius`, a parameter it does
        not use, whose default is whatever the *other* machine's search last landed on. Its
        own short arm is the length that sizes its counterweight linkage, so that is what
        it falls back to.
        """
        if self.counter_weight_rope_length is not None:
            return self.counter_weight_rope_length
        return 2 * self.pulley_radius if self.has_pulley else self.length_counterweight

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
