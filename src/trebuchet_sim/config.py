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

# How high the traditional machine's cocked arm tip sits above the ground, in metres.
# That machine is loaded by walking the long arm down until it is all but touching, laying
# the sling out along the ground behind it and setting the stone at the far end - so the
# cocked arm angle is not a free choice, it is whatever puts the tip here (see
# resolve_initial_arm_angle). Small and positive: exactly zero would have the beam's own
# ground event firing at t=0.
TRADITIONAL_START_CLEARANCE = 0.05  # m

# Rotation speed below which the pivot's Coulomb friction is smoothed out, rad/s. Dry
# friction is a sign function of the velocity, which is a discontinuity an adaptive
# integrator has to chase; regularizing it as tau * w / sqrt(w^2 + eps^2) is smooth,
# reaches full magnitude within a few eps, and is exactly zero at rest - so a machine at
# t = 0 is not handed a friction torque with an arbitrary sign. What it does not model is
# stiction: a machine whose driving torque is below the friction torque will creep here
# where a real one would sit still.
PIVOT_FRICTION_SMOOTHING = 1e-3  # rad/s

# "Resolve the cocked arm angle from the geometry" as a number, for the Numba engine, which
# has no None to pass down. Deliberately a huge finite value rather than NaN: fastsim's
# kernels are compiled with fastmath=True, which lets LLVM assume no NaNs and fold an
# isnan() test to False, so a NaN sentinel silently arrives as a NaN arm angle.
AUTO_INITIAL_ARM_ANGLE = 1e30

# Canonical defaults for the five optimizable parameters, shared by the CLI,
# the web UI, and the tests so they can't drift apart. Optimizer output for the
# 30 m target on the shipped weights and the shipped seed - `trebuchet optimize
# --target-distance 30` reproduces every digit shown, though only to the six decimals
# shown: the objective is compiled fastmath and parallel, and a freshly compiled kernel
# and one reloaded from Numba's cache differ in the last bit or two, which differential
# evolution carries into about the seventh significant figure of its answer (the pulley
# machine happens to come out bit-identical either way; the traditional one moves by
# ~1e-7 relative, on the same design). So the sling stays taut for the whole launch and
# the projectile never touches the ground. Both engines are asked, not just the one the
# search runs on: a design the optimizer likes and physics.py then reports differently is
# no use as a default whatever it scores (see DEFAULT_TRADITIONAL_PARAMS for the draw that
# made this a rule).
#
# Re-swept whenever the physics under it moves, which it has five times: when
# `pivot_height` rose to 2.5 m, when the arm's drag torque was corrected, for two modelling
# changes and two engine fixes at once, and now for the beam becoming a surface the stone
# can hit.
#
# That last one is why this set looks nothing like the one before it, and why its
# efficiency reads 74.8% against that set's 90.1%. The 90.1% machine was a 0.37 m arm on a
# 0.33 m sling that swung its stone straight through its own beam - and was credited with
# the throw. Nothing stopped it, so nothing charged it, and the search had been free to
# buy range with a collision the model could not see. With the beam solid it throws 16.4 m
# against its own 30 m target. The re-swept machine is three times the arm at 1.05 m and
# keeps its stone clear the whole way: no beam contact, no ground contact, and its 74.8%
# is what a pulley machine of this size is actually worth. See the beam-contact regimes in
# physics.py and the step scan both engines needed to see them at all.
#
# `release_angle` stopped being the arm's angle in the world and became the release pin's
# angle, measured from the arm to the sling. That is what a pin can actually sense, and it
# is why this set exists rather than the previous one: under the old convention the
# shipped design reached its release arm-angle on the sling's *third* pass, so a real pin
# set for it would have fired on the first, at 11.35 m/s pointing backwards against the
# 17.6 m/s the model claimed. This machine releases at -67.2 degrees of pin angle, on the
# first crossing, and is therefore buildable.
#
# And the pivot's dry bearing friction is now modelled (see
# TrebuchetParams.bearing_friction_coefficient). It costs the same at every speed, so it
# is charged per radian the arm turns, which is the thing a short-armed machine has most
# of. It is one of the reasons the short arm stopped winning once the beam was solid.
#
# Its counterweight rope still pushes, but barely: -0.77 N for an impulse of
# 0.0018 N*s, an order of magnitude under the previous set's 0.018 N*s and two under the
# 0.05 N*s the CLI and dashboard warn at. That is real rather than noise - this machine's
# weight hangs on a rope over the axle and a rope cannot push - and it is what
# `slack_penalty_weight` is there to keep small. (The traditional machine has no
# equivalent: its weight is pinned to the arm through a rigid strut, so compression there
# is a member load rather than a fault. See physics._cw_link_tension.)
#
# These five numbers are defined as what `trebuchet optimize --target-distance 30` returns
# on the *shipped* weights, so tuning a weight to tidy them up is a calibration decision
# about every search anyone runs rather than about this dict, and it belongs to whoever is
# prepared to re-check the calibration in optimization.py against it.
#
# The seed is still shared with the traditional machine - one seed has to serve both - and
# still matters, though the landscape it is picking a basin out of has changed shape now
# that the beam is solid. See OptimizationConfig.seed.
DEFAULT_OPTIMIZABLE_PARAMS = {
    "counter_weight_mass": 59.982328,  # kg
    "pulley_radius": 0.049736,         # m
    "arm_length": 1.047940,            # m
    "string_length": 0.822843,         # m
    "release_angle": -1.173475,        # radians (pin angle, sling measured from the arm)
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


# Cocked positions, for the machine whose arm angle is a free choice. The pulley machine
# starts with the arm raised at 45 degrees. The traditional machine's entry is the pose it
# falls back to when its own geometry cannot be solved - see resolve_initial_arm_angle,
# which is what actually decides that machine's cocked angle.
DEFAULT_INITIAL_ARM_ANGLE.update({
    MachineType.PULLEY: np.pi / 4,
    MachineType.TRADITIONAL: -np.pi / 2,
})


def resolve_initial_arm_angle(machine, arm_length: float, pivot_height: float) -> float:
    """The arm angle a machine is cocked at, when the caller has not pinned one.

    The pulley machine's is a constant: its arm is short, it stands under a tall pivot,
    and where it starts is a free choice.

    The traditional machine's is not a free choice at all - it is geometry. That machine
    is loaded by walking the long arm down until the tip is all but on the ground, laying
    the sling out along the ground behind it and setting the stone at the far end. So the
    cocked angle is whatever puts the tip `TRADITIONAL_START_CLEARANCE` above the ground,
    and it moves with the arm length and the pivot height rather than being carried as a
    separate number that has to be re-chosen every time either of those changes:

        l_a * sin(theta) + h_T = clearance   ->   theta = -pi - asin((clearance - h_T)/l_a)

    of the two arcsine branches, the one in (-pi, -pi/2) - long arm down and *behind* the
    pivot, counterweight raised in front of it, so the throw sweeps up and over toward +x.

    An arm shorter than the pivot is tall cannot reach that low however far it is walked
    down; the closest it gets is straight down, at -pi/2, and the clamp below returns
    exactly that. Such a machine does not launch, and that is the honest answer rather
    than a defect: at -pi/2 every gravity torque about the pivot carries a cos(theta) and
    is identically zero, so the machine is parked on its own balance point. The geometry
    is saying the pivot is too tall for the arm.
    """
    if MachineType(machine) is MachineType.PULLEY:
        return float(DEFAULT_INITIAL_ARM_ANGLE[MachineType.PULLEY])
    if arm_length <= 0.0:
        # No arm, no geometry to solve: fall back to the clamp's own answer rather than
        # dividing by zero. Nothing launches either way.
        return float(DEFAULT_INITIAL_ARM_ANGLE[MachineType.TRADITIONAL])
    ratio = (TRADITIONAL_START_CLEARANCE - pivot_height) / arm_length
    return float(-np.pi - np.arcsin(max(-1.0, min(1.0, ratio))))


# Canonical defaults for the traditional machine, derived exactly the way the pulley
# machine's are: optimizer output for the 30 m target on the shipped weights and seed,
# with the fixed geometry below held. The counterweight rides the arm rather than a pulley,
# so `length_counterweight` replaces `pulley_radius` as the linkage parameter, and the arm
# is cocked at -140 degrees - not chosen, but wherever walking the long arm down to the
# ground puts it (resolve_initial_arm_angle).
#
# The search is barely seed-sensitive: 47 of 48 seeds land within 0.005 of the same score,
# on the same 1.49 m arm at the same 13 degree pin. The 48th is the reason the defaults are
# now checked against *both* engines. It scores -145 against this set's +7.2 - by far the
# best number any seed returns - on a design fastsim throws 30 m and physics.py throws
# 0.0 m. Neither engine is misbehaving: the sling's tension dips to -0.024 N for 7 ms, the
# fast engine resolves the dip and lets the sling go slack, and scipy's event test, which
# only looks at the ends of its accepted steps, straddles the whole excursion with one 14 ms
# step and sails past it. That one branch moves the release 0.7 s and turns a 30 m throw
# into a stone lobbed backwards. A design whose distance depends on which engine you ask is
# not a default, whatever it scores.
#
# The pivot is low because the cocked angle is no longer a number anyone chose - it is
# whatever walks the long arm down to the ground, and an arm shorter than the pivot is tall
# cannot get there. So this machine loads the way a real one does: the beam down, the sling
# stretched to a stone lying on the ground, and the arm rotating away from it.
#
# Which way it is stretched is the other thing that moved this set. The stone is laid out
# *downrange* of the cocked tip now - ground_start_state takes the +x root of the loading
# circle - and this machine's tip sweeps back and up, away from it, so the sling is
# carrying load before anything moves. The launch opens in TAUT_GROUND rather than
# SLACK_GROUND: it drags the stone back along the ground for 43% of the run, then whips it
# forward, which is the stroke a real trebuchet makes. It never goes slack at all -
# `string_slack_fraction` and `sling_tension_deficit` are both exactly zero, where the
# far-side pose spent a third of every launch taking up slack and was charged for it.
#
# 93.1% efficiency at 17.9 kg all in, against the 73.7% of the set before the loading root
# moved and the beam became solid. The stone never touches the beam and never lands, the
# sling is under load throughout, and the counterweight link bottoms out at +61.9 N -
# always in tension, which is incidental rather than designed for: nothing charges it if it
# pushes, because a pinned strut carries compression by design (see
# physics._cw_link_tension).
DEFAULT_TRADITIONAL_PARAMS = {
    "counter_weight_mass": 15.882893,   # kg
    "length_counterweight": 0.290764,   # m
    "arm_length": 1.078622,             # m
    "string_length": 1.023034,          # m
    "release_angle": 0.232733,          # radians (pin angle, sling measured from the arm)
}

# Fixed (never-optimized) fields whose defaults also differ per machine: the pivot the
# machine stands on, and the pin-to-weight link the counterweight hangs from.
#
# The pivot came down from 2.6 m when the cocked angle stopped being a free number and
# became geometry (see resolve_initial_arm_angle). This machine is loaded with its long
# arm walked down to the ground, and an arm shorter than its pivot is tall cannot get
# there - at 2.6 m nothing inside the arm's own search range could, so every design would
# have been cocked straight down on its balance point and none would have launched. At
# 1.0 m an arm of 0.95 m or more reaches, which is most of the range and is also what a
# traditional trebuchet looks like: the beam is longer than the frame is tall.
#
# The tall-pivot argument that raised it in the first place does not apply here any more.
# It was there to stop the beam digging itself in, but a machine cocked at the ground
# rotates *away* from it - the first ground crossing after the cocked angle is more than
# 250 degrees of rotation later on the shipped design - so the beam has the whole throw
# to work in.
DEFAULT_TRADITIONAL_FIXED = {
    "pivot_height": 1.0,                 # m
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
    # The angle the release pin is set at, measured from the arm's own direction to the
    # sling: beta = alpha - theta, in radians. Zero is a sling lying along the arm and
    # pointing straight out past the tip; positive is a sling trailing behind the tip.
    #
    # It is deliberately not the arm's angle in the world frame, which is what this used
    # to be. A trebuchet releases when the ring on the sling's free cord slides off a pin
    # fixed to the arm tip, and that pin turns with the arm - so what it can sense is the
    # sling's direction *relative to the arm*, and nothing on the machine can sense where
    # the arm is pointing in the world. The distinction is not academic: the shipped
    # pulley design under the old convention reached its release arm-angle on the third
    # pass, having already gone by twice, so a real pin set for it would have fired at
    # 11.35 m/s pointing backwards instead of 17.6 m/s downrange.
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
    # The arm's angle in the world, radians, measured the ordinary way: zero points from
    # the pivot straight out along +x - horizontal, level with the pivot, and in the
    # direction the machine throws - and positive turns counter-clockwise, raising the
    # tip. Both machines use it, and both throw toward +x, so zero means the same thing on
    # each: the long arm held out level, aimed downrange. The tip is at
    # (l_a*cos(theta), l_a*sin(theta) + pivot_height), which is where that convention
    # comes from and the only place it is expressed.
    #
    # Not to be confused with `release_angle`, which since it became the pin angle is
    # measured from the *arm* to the sling rather than in the world at all.
    #
    # None means "resolve it per machine" - a constant for the pulley machine, geometry
    # for the traditional one (see resolve_initial_arm_angle).
    initial_arm_angle: Optional[float] = None          # radians; None -> the machine's default
                                                       # (see DEFAULT_INITIAL_ARM_ANGLE)
    arm_drag_coefficient: float = 1.05
    projectile_drag_coefficient: float = 0.47
    joint_friction_coefficient: float = 0.01          # N*m*s/rad (viscous damper at pivot)
    # Dry friction in the main bearing, which is the dominant real loss and the only one
    # that scales with how far the arm turns rather than with how fast. A plain bearing
    # carrying the machine's weight resists with mu * N * r_shaft whatever the speed, so a
    # design that buys range by spinning the arm most of a turn pays for every radian of
    # it. 0.2 is steel on dry hardwood; a greased bronze bush is nearer 0.1, a rolling
    # bearing 0.005.
    bearing_friction_coefficient: float = 0.2         # dimensionless
    pivot_shaft_radius: float = 0.0125                # m (25 mm shaft)
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
            self.initial_arm_angle = resolve_initial_arm_angle(
                self.machine, self.arm_length, self.pivot_height
            )

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
    def pivot_friction_torque(self) -> float:
        """Magnitude of the pivot's Coulomb friction torque, N*m.

        mu * N * r_shaft, with N the static bearing load - everything the pivot carries.
        Both linkages hang their counterweight off the pivot (over the axle on one, on the
        arm's short end on the other), so `total_mass` is that load on either machine.

        The dynamic reaction at the pivot is larger than the static one while the machine
        is accelerating, and it would make the friction depend on the accelerations it is
        itself changing - an implicit equation for one term of a model whose other terms
        are idealized far harder than this. The static load is the standard reading.
        """
        return self.bearing_friction_coefficient * self.pivot_shaft_radius * self.total_mass * G

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
