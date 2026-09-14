"""Trebuchet parameter optimization via differential evolution.

Pure computation only - no printing or interactive prompts. See cli.py for the
command-line presentation layer built on top of this module.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import differential_evolution

from trebuchet_sim.config import (
    AUTO_INITIAL_ARM_ANGLE,
    LINKAGE_PARAM,
    MAX_PHYSICAL_EFFICIENCY,
    MachineType,
    TrebuchetParams,
)
from trebuchet_sim.physics import SimulationResult, simulate_trebuchet

try:
    from trebuchet_sim import fastsim

    _FASTSIM_AVAILABLE = True
except Exception:
    _FASTSIM_AVAILABLE = False

# The cost a design scores when it cannot be simulated at all - no release, a geometry the
# model refuses, or a sling longer than its arm. Both engines return exactly this (see
# fastsim.INVALID_COST), so a search whose best score reaches it found nothing at all.
INVALID_COST = 1e6

PARAM_NAMES = ["counter_weight_mass", "pulley_radius", "arm_length", "string_length", "release_angle"]


def param_names(machine: MachineType = MachineType.PULLEY) -> List[str]:
    """The five design variables for one machine, in display order.

    Four are shared; the linkage slot holds whichever parameter sizes that machine's
    counterweight coupling (config.LINKAGE_PARAM). The list is always five long, so the
    search space has the same shape either way - only its second entry changes.

    PARAM_NAMES stays the pulley list: it is what the module-level constants
    (PARAM_BOUNDS keys, the fastsim call signature) are written against.
    """
    return [LINKAGE_PARAM[MachineType(machine)] if name == "pulley_radius" else name for name in PARAM_NAMES]


# Non-optimizable TrebuchetParams fields the fast engine needs but that never appear
# in PARAM_NAMES; a run either uses the dataclass default or an OptimizationConfig
# fixed_params override.
_FASTSIM_FIXED_FIELDS = [
    "pivot_height", "pulley_density", "arm_density", "projectile_mass", "projectile_radius",
    "initial_arm_angle", "arm_drag_coefficient", "projectile_drag_coefficient", "joint_friction_coefficient",
    # Dry friction at the main pivot: a torque that costs the same at every speed, so it
    # is charged per radian the arm turns rather than per unit of anything the other loss
    # terms measure. Both fields together give its magnitude.
    "bearing_friction_coefficient", "pivot_shaft_radius",
    # Sets how fast a pinned counterweight swings, so it is part of the traditional
    # machine's equations of motion rather than just its rendering.
    "counter_weight_rope_length",
]

# Called once per DE generation with (generation, score, params, result); result is None if the
# generation's best-so-far params fail to simulate. Score is the raw objective value (lower is better).
ProgressCallback = Callable[[int, float, TrebuchetParams, Optional[SimulationResult]], None]

PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "counter_weight_mass": (5.0, 60.0),    # kg
    "pulley_radius": (0.01, 1.0),          # m
    # Capped below the shorter of the two machines' default pivot heights (2.5 m on the
    # pulley machine, 2.6 m on the traditional one), so a search run with the shipped
    # defaults can never return a beam that digs itself into the ground - see
    # physics._first_arm_ground_angle, which ends such a launch with no throw at all. The
    # floor is the traditional machine's own 1.8 m default arm, which has to stay inside
    # its own search range. PARAM_LIMITS still reaches 10 m for anyone who raises the pivot
    # to match, but there is not much waiting up there: holding this cap and sweeping the
    # pivot, a pulley machine measures 93.8 / 93.5 / 105.0 / 105.9 / 106.1 / 107.6 m at
    # 1.0 / 1.5 / 2.0 / 2.5 / 3.0 / 4.0 m, and the winning arm stops growing at about
    # 1.65 m - past a 2 m pivot the ground has stopped deciding the design at all.
    "arm_length": (0.1, 2.0),              # m
    # Shared by both machines, and for the pulley machine it is the binding limit twice
    # over: the objective refuses a sling past 0.95 of that machine's arm, so with the arm
    # at 2.0 m nothing above 1.9 m is ever scored there, and a bound of 2.5 m was search
    # space that only ever produced invalid designs.
    #
    # For the traditional machine this range is the *only* limit. That machine lays its
    # sling out from the tip rather than tucking it along the arm, and a real one is often
    # slung as long as its long arm or longer, so nothing couples the two lengths - a
    # caller who wants a longer sling than this raises the range (param_bounds reaches
    # PARAM_LIMITS' 10 m) rather than lengthening the arm to earn it.
    "string_length": (0.1, 2.0),           # m
    # The release pin's angle, measured from the arm to the sling (see
    # config.TrebuchetParams.release_angle), not the arm's angle in the world - so the
    # range is about where a pin can usefully sit rather than about how far the arm turns.
    # The upper end is under the tightest cocked sling-to-arm angle either machine starts
    # at, since a pin above where the sling already is cannot catch it on the way down
    # without a whole extra turn; the lower end is past full extension, where a sling that
    # has swung out in front of the arm is thrown flattest.
    "release_angle": (np.radians(-90), np.radians(150)),  # rad
    # Traditional machine only, in the linkage slot where pulley_radius sits otherwise.
    # Capped well under the arm-length bound: a short arm approaching the long one is a
    # balanced beam that throws nothing.
    "length_counterweight": (0.05, 1.0),   # m
}

# How far a caller may move a search range with OptimizationConfig.param_bounds, which is
# wider than the defaults above: PARAM_BOUNDS is where the search starts looking, while
# these are the values the model can still be trusted at. The lower ends are what stops a
# zero or negative length reaching the equations of motion; the upper ends are loose
# enough for a genuinely large machine. The arm can sweep a full turn before releasing.
PARAM_LIMITS: Dict[str, Tuple[float, float]] = {
    "counter_weight_mass": (0.1, 1000.0),           # kg
    "pulley_radius": (0.001, 2.0),                  # m
    "arm_length": (0.05, 10.0),                     # m
    "string_length": (0.05, 10.0),                  # m
    "release_angle": (np.radians(-180), np.radians(180)),  # rad
    "length_counterweight": (0.01, 5.0),            # m
}


@dataclass
class OptimizationConfig:
    """Optimization objective weights, search bounds, and parameter locks."""

    target_distance: float = 30.0
    # These two are one setting in two parts: their *ratio* is the exchange rate the
    # search trades on, in efficiency points per 1% of target distance. The absolute
    # scale only decides how much the mass term and the two penalties below can move
    # things, so at a fixed ratio the numbers here barely change the answer.
    #
    # 10:5 = 2.0 is measured. The benchmark is 60 problem/seed pairs over both machines
    # with and without locks, each aimed at 40/70/95% of the range that machine can
    # honestly reach - "honestly" meaning with the snap penalty active, since a machine
    # is allowed to throw much further by wrecking its own sling (see
    # snap_penalty_weight) and targets drawn from that would be unreachable by anything
    # worth building. Against those, the share landing within 2% of the target is 37% at
    # a distance weight of 1, 88% at 5, and 100% at 10. The old 1.0 - an exchange rate of
    # 0.2, where the search would miss by 5% to buy a single point of efficiency - missed
    # by a median 17.9%, which is to say it mostly ignored the target it was given.
    # Past 10 there is nothing left to buy: everything is already being hit.
    efficiency_weight: float = 5.0
    distance_weight: float = 10.0
    mass_weight: float = 0.15
    # Cost per N*s of the counterweight rope's "compression impulse", on the pulley
    # machine. That machine hangs its weight on a rope over the axle, which the model
    # holds rigid; a rope pulls and never pushes, so a run where the model needs it to
    # push is unphysical from that moment on, and penalizing the impulse keeps the search
    # out of there. It is a feasibility term, not a design preference: there is no version
    # of the machine the answer describes.
    #
    # The traditional machine is charged nothing, because it has no rope. Its weight is
    # pinned to the arm's short end through a rigid link, and a pinned two-force member is
    # a strut - it carries compression as readily as tension, and does so on the stroke
    # where the weight is swung rather than dropped. `cw_rope_compression_impulse` is
    # therefore identically zero for it and this weight has nothing to multiply.
    #
    # This has moved twice. It was pulley-only, then charged on both, on the argument that
    # a pin holds the link's *end* and the weight hangs a further
    # `counter_weight_rope_length` below it - true of a rope hanger, but that machine is
    # pinned right through. Charging it cost real designs: the traditional winners it
    # forbade, down to -1212 N, are strut loads to size for rather than runs to discard.
    slack_penalty_weight: float = 200.0
    # Cost per unit of `sling_tension_deficit` - the share of the launch the sling spent
    # below config.SLING_TENSION_FLOOR projectile weights, weighted by how far below
    # (see physics._tension_metrics). This is the term that makes a *jerky* design cost
    # more than a smooth one. The slack penalty above cannot: a compression impulse is
    # identically zero until the rope has already gone slack, so across designs that
    # still hold together it is flat, and differential evolution has nothing to descend.
    # The deficit is graded all the way down, so the search feels the cliff coming.
    #
    # This is the term that decides how much range the design is allowed to buy with the
    # sling's own health, and on the pulley machine that is a real trade rather than a
    # rounding: left unpenalized, a plain pulley machine reached 182 m with the sling limp
    # for over half the launch, against 106 m with it loaded throughout - a 42% range
    # premium for a machine that beats itself up, and 39-60% across the pulley benchmark.
    # (Those figures, and the calibration below, were measured before the arm's drag
    # torque was corrected and the defaults re-derived; the trade they describe is the
    # same, and every winner in the post-fix sweep still scores a zero deficit.)
    # The traditional machine pays nothing for the same promise (0-2%), because its
    # geometry keeps the sling loaded anyway.
    #
    # Two measurements set the value. Robustness: over 24 randomized problems, scoring
    # each winner by how many +-10% one-parameter perturbations tip it into snapping,
    # 14.2% of perturbations for the unpenalized objective falls to 10.4 / 7.5 / 5.4 /
    # 1.3% at 100 / 200 / 300 / 600, with mean efficiency flat throughout. Then, against
    # the reachable-target benchmark above at a distance weight of 10, 300 leaves a mean
    # tension deficit of 0.0082 and 2.0% fragility with one winner in 60 actually
    # snapping, while 1000 leaves 0.0004 and 0.5% with none - for 0.008 of mean
    # efficiency. The margin is small because the distance term is no longer pulling
    # against a broken engine; it is consistent, and it is nearly free.
    #
    # (Before fastsim modelled the sling as a rope, this weight was also load-bearing in
    # a way it is not now: it was what kept the search inside the region where that
    # engine was faithful. It no longer has that job - the engines agree everywhere - so
    # what is left is the design preference above.)
    snap_penalty_weight: float = 1000.0
    # Cost per joule the launch destroys in a discontinuity: `sling_snap_energy` (a rope
    # that let go and came back) plus `projectile_ground_energy` (a stone that hit the
    # dirt mid-throw). Those are the two jerks that have already happened, and until this
    # term existed the objective only saw them through `efficiency`, which prices a lost
    # joule exactly like a joule spent on air drag. It is not the same thing: a launch
    # that throws its stone into the ground and snatches it back out is a machine that
    # shakes itself apart, not merely a lossy one.
    #
    # The two terms above cannot cover it. `sling_tension_deficit` grades how close the
    # *sling* runs to slack and says nothing about the ground, and the counterweight
    # impulse is a different link again. With the shipped pivot heights nothing lands here
    # anyway, so this changes no default answer; it bites where the geometry is forced -
    # a pulley machine on a 0.6 m pivot aiming at 45 m won by driving the projectile into
    # the ground and paying 17.2 J for it, which the old objective accepted because the
    # throw still reached the target.
    #
    # Priced in joules, the way the counterweight impulse is priced in N*s. Measured over
    # 30 problems spanning both machines - the shipped geometry, pivots forced down to
    # 0.6/0.9/1.2 m, and locked linkages - the joules the winners throw away in total run
    # 47.1 / 9.9 / 1.3 / 0.3 at a weight of 0 / 5 / 20 / 100. 20 removes 97% of it. It
    # costs nothing to buy: mean efficiency over those problems is 0.801 unpenalized
    # against 0.818 at 20, the median distance miss stays at 0.002% of target, and over
    # the shipped-geometry problems alone mean efficiency is 0.863 at every weight
    # including 0 - the term is identically zero for a design that never lands or snaps,
    # so a clean winner's score is untouched.
    jerk_penalty_weight: float = 20.0
    # Which counterweight linkage to design for. It decides the search space (see
    # param_names) rather than being searched itself, so it is a field of its own
    # instead of a fixed_params entry.
    machine: MachineType = MachineType.PULLEY
    locked_params: Dict[str, float] = field(default_factory=dict)
    # Per-parameter search ranges, overriding PARAM_BOUNDS for the names given. Narrowing
    # one steers the search at a region worth exploring (and shrinks the space it has to
    # cover); widening one reaches machines the defaults exclude. A locked parameter is
    # not searched at all, so a range given for one is simply unused.
    param_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    fixed_params: Dict[str, float] = field(default_factory=dict)
    # 33 rather than the old 572956 because the pulley machine's landscape has one narrow
    # basin that is much better than the broad one around it, and most seeds miss it. Over
    # 48 seeds at the 30 m target, 40 converge on a 1.0 m arm releasing at a -69 degree pin
    # for 73.5% efficiency and score about -235; the 8 that find the narrow one get a
    # 0.37 m arm at a -4.2 degree pin, 90.1% efficiency and about -282.7. This is not quite
    # the best of those (-282.636 against -282.833) and is chosen because one seed has to
    # serve both machines: the traditional search is nearly seed-independent, but not
    # perfectly, and this seed is within 0.002 of its best where the pulley machine's own
    # best seed is 0.36 adrift of it.
    seed: int = 33
    max_iterations: int = 1000
    # scipy multiplies popsize by the number of free params, so 40 means a
    # population of ~200 individuals for the 5-parameter search space.
    population_size: int = 40
    absolute_tolerance: float = 0.001
    workers: int = -1                     # -1 = one process per CPU core; ignored when the fast engine runs
    # How many threads the fast engine scores a population across. `workers` above is the
    # scipy fallback's *process* count and does nothing when the fast engine runs, because
    # that engine is one call per generation rather than one per design: it spreads the
    # population over numba's own thread pool inside `evaluate_population`'s prange. So the
    # two are separate settings for separate engines rather than one knob spelled twice.
    #
    # None means "leave numba where it is", which is one thread per core. An explicit count
    # is clamped to `fastsim.thread_ceiling()` - a process cannot be given more threads
    # than it was imported with, so going above the core count needs the
    # TREBUCHET_NUM_THREADS environment variable instead (see fastsim._seed_thread_ceiling).
    threads: Optional[int] = None
    display_progress: bool = False        # scipy's per-iteration convergence printout
    use_fast_engine: bool = True          # Numba-vectorized objective when available; falls back to scipy otherwise

    def __post_init__(self):
        # Accept a plain string, so a machine read back from saved JSON works unchanged.
        self.machine = MachineType(self.machine)

        if self.threads is not None:
            self.threads = int(self.threads)
            if self.threads < 1:
                raise ValueError(f"threads must be at least 1, got {self.threads}.")

        names = self.param_names
        unknown = set(self.locked_params) - set(names)
        if unknown:
            # Naming the machine matters here: the linkage parameter is the one that
            # differs, so "pulley_radius is unknown" is otherwise a baffling message.
            raise ValueError(
                f"Unknown parameter(s) for the {self.machine.value} machine: {unknown}. Available: {names}"
            )

        if "machine" in self.fixed_params:
            raise ValueError("Set the machine with OptimizationConfig(machine=...), not via fixed_params.")

        unknown_ranges = set(self.param_bounds) - set(names)
        if unknown_ranges:
            raise ValueError(
                f"No such parameter(s) to range on the {self.machine.value} machine: {unknown_ranges}. "
                f"Available: {names}"
            )
        for name, span in self.param_bounds.items():
            lo, hi = (float(value) for value in span)
            if not (np.isfinite(lo) and np.isfinite(hi)):
                raise ValueError(f"Range for {name} must be finite, got ({lo}, {hi}).")
            if lo >= hi:
                raise ValueError(f"Range for {name} must have min < max, got ({lo}, {hi}).")
            limit_lo, limit_hi = PARAM_LIMITS[name]
            if lo < limit_lo or hi > limit_hi:
                raise ValueError(
                    f"Range for {name} must lie within {(limit_lo, limit_hi)} (see PARAM_LIMITS), "
                    f"got ({lo}, {hi})."
                )
            self.param_bounds[name] = (lo, hi)

        valid_fields = {f.name for f in fields(TrebuchetParams)}
        unknown_fixed = set(self.fixed_params) - valid_fields
        if unknown_fixed:
            raise ValueError(f"Unknown fixed parameter(s): {unknown_fixed}. Valid fields: {sorted(valid_fields)}")

        # Search-space params must go through locked_params; a fixed_params entry for one
        # would be silently overwritten by optimizer values in build_params.
        overlap = set(self.fixed_params) & set(names)
        if overlap:
            raise ValueError(f"Parameter(s) {overlap} are optimizable; use locked_params to pin them, not fixed_params.")

    @property
    def param_names(self) -> List[str]:
        """This machine's five design variables (see the module-level param_names)."""
        return param_names(self.machine)

    @property
    def free_params(self) -> List[str]:
        return [name for name in self.param_names if name not in self.locked_params]

    def bounds_for(self, name: str) -> Tuple[float, float]:
        """The search range for one parameter: the caller's override, else the default."""
        return tuple(self.param_bounds.get(name, PARAM_BOUNDS[name]))

    @property
    def bounds(self) -> List[Tuple[float, float]]:
        return [self.bounds_for(name) for name in self.free_params]

    def build_params(self, free_values: Sequence[float]) -> TrebuchetParams:
        """Combine fixed, locked, and optimized values into a full TrebuchetParams.

        fixed_params (e.g. pivot height, initial arm angle, projectile mass/radius) are
        never part of the search space - they're merged in first so the optimizer can't
        touch them, regardless of what free_params/locked_params contain.
        """
        values = dict(self.fixed_params)
        values.update(self.locked_params)
        values.update(zip(self.free_params, free_values))
        return TrebuchetParams(machine=self.machine, **values)


def _objective(free_values: Sequence[float], config: OptimizationConfig) -> float:
    """Differential-evolution objective: minimize weighted (-efficiency, distance error, mass)."""
    params = config.build_params(free_values)

    # Pulley machine only, and it is a statement about that machine's cocked pose rather
    # than about slings. It tucks its sling alongside the arm to load (see
    # simulate_fast's alpha0), so a sling approaching the arm's own length has nowhere to
    # lie and the pose stops describing anything buildable.
    #
    # The traditional machine loads the other way - the sling laid out from the tip, to a
    # stone on the ground or hanging - and a real one is commonly slung as long as its
    # long arm or longer. Nothing about that pose degenerates, so its sling is bounded by
    # the search range alone (PARAM_BOUNDS / param_bounds) and by nothing here.
    if params.has_pulley and params.string_length > 0.95 * params.arm_length:
        return INVALID_COST

    try:
        # Looser tolerance and no dense interpolants: the objective only reads
        # distance/efficiency, and at rtol=1e-6 the distance shifts by ~0.02%
        # vs the 1e-8 display runs - far below what the weights can resolve.
        result = simulate_trebuchet(params, rtol=1e-6, dense_output=False)
    except Exception:
        return 1e6

    # Mirrored from fastsim._score: an efficiency above MAX_PHYSICAL_EFFICIENCY means the
    # potential energy the launch spent came out as very nearly nothing, so the ratio is a
    # broken measurement. The objective maximizes efficiency, so it has to be refused
    # rather than merely reported.
    if result.distance <= 0 or result.efficiency <= 0 or result.efficiency > MAX_PHYSICAL_EFFICIENCY:
        return INVALID_COST

    efficiency_cost = -result.efficiency * 100
    distance_cost = abs(result.distance - config.target_distance) / config.target_distance * 100
    mass_cost = (params.total_mass / 30.0) * 100
    # Only the counterweight's link is still a rigid one, so it is the only connector
    # with a compression impulse to charge (the key this used to read for the sling,
    # `string_compression_impulse`, is not in the metrics any more - both engines let the
    # sling go slack for real instead). Both machines report one: the pulley machine's
    # rope over the axle, and the link a pinned weight hangs from.
    slack_cost = config.slack_penalty_weight * result.metrics.get("cw_rope_compression_impulse", 0.0)
    # The sling's own loss does land in `efficiency` once it detaches, but only once:
    # this is what keeps the search off the cliff edge rather than merely off the bottom,
    # since by the time efficiency has noticed, the run has already lost the energy.
    # Both engines charge for it, computed the same way from the same floor.
    snap_cost = config.snap_penalty_weight * result.metrics.get("sling_tension_deficit", 0.0)
    # What the launch actually threw away in its discontinuities. Both engines report the
    # same two numbers; the fast one used to compute them and discard them here.
    jerk_cost = config.jerk_penalty_weight * (
        result.metrics.get("sling_snap_energy", 0.0)
        + result.metrics.get("projectile_ground_energy", 0.0)
    )

    return (
        config.efficiency_weight * efficiency_cost
        + config.distance_weight * distance_cost
        + config.mass_weight * mass_cost
        + slack_cost
        + snap_cost
        + jerk_cost
    )


def _fastsim_fixed_scalar(config: "OptimizationConfig", name: str) -> float:
    """A fixed (non-optimizable) TrebuchetParams field as a plain float: the config
    override if present, otherwise the dataclass default (matches build_params).

    Two fields have no usable dataclass default, and both mean "resolve it downstream"
    rather than "use this number". `initial_arm_angle` resolves per machine in
    __post_init__, and on the traditional machine it resolves from the *arm length* - which
    differs per individual, so it cannot be a scalar here at all;
    config.AUTO_INITIAL_ARM_ANGLE is the sentinel that tells fastsim to derive it the same
    way. `counter_weight_rope_length` defaults to None,
    meaning "one wrap of the pulley"; numba has no None either, so it is passed as 0.0 and
    fastsim applies the same fallback.

    An explicit None *in* fixed_params means what an absent key means - "resolve it
    downstream" - so it falls through to the sentinel above rather than to a number. It
    used to be coerced to 0.0, which is right for the rope, whose sentinel is 0.0 anyway,
    and silently wrong for the angle: 0.0 is a perfectly good arm angle, so a machine
    cocked flat along the ground went to the search with nothing to complain about. The
    dashboard passes exactly that on the traditional machine, whose cocked angle has no
    box because it follows the arm length, and the two halves of a run then disagreed
    about which machine they were: the fast objective scored every candidate at 0 degrees
    while build_params went on resolving the same design to -140.5, so the reported result
    - and the design DE returned - came from a machine that was never scored. On the
    shipped traditional defaults that is a cost of 1069 against the 7.2 the CLI's own dict
    gets, which is not a tuning difference but a different machine.
    """
    default = getattr(TrebuchetParams, name)
    if default is None:
        if name == "initial_arm_angle":
            default = AUTO_INITIAL_ARM_ANGLE
        elif name == "counter_weight_rope_length":
            default = 0.0
    value = config.fixed_params.get(name, default)
    if value is None:
        value = default
    return float(value)


def _objective_vectorized(x: np.ndarray, config: "OptimizationConfig") -> np.ndarray:
    """Batch objective for scipy's vectorized differential_evolution, powered by the
    Numba fast engine. `x` has shape (n_free, S); returns costs of shape (S,)."""
    s = x.shape[1]
    values = {}
    for name in config.param_names:
        if name in config.locked_params:
            values[name] = np.full(s, config.locked_params[name], dtype=np.float64)
        else:
            idx = config.free_params.index(name)
            values[name] = np.ascontiguousarray(x[idx], dtype=np.float64)

    # fastsim takes both linkage parameters as arrays and reads the one its machine uses.
    # Only this machine's is in the search space, so the other is filled with the value a
    # TrebuchetParams would have carried - unused by the dynamics, but it still has to be
    # a well-typed array of the right length.
    unused_linkage = LINKAGE_PARAM[
        MachineType.TRADITIONAL if config.machine is MachineType.PULLEY else MachineType.PULLEY
    ]
    values[unused_linkage] = np.full(s, _fastsim_fixed_scalar(config, unused_linkage), dtype=np.float64)

    fixed = {name: _fastsim_fixed_scalar(config, name) for name in _FASTSIM_FIXED_FIELDS}

    return fastsim.evaluate_population(
        values["counter_weight_mass"], values["pulley_radius"], values["length_counterweight"],
        values["arm_length"], values["string_length"], values["release_angle"],
        fixed["counter_weight_rope_length"],
        fixed["pivot_height"], fixed["pulley_density"], fixed["arm_density"],
        fixed["projectile_mass"], fixed["projectile_radius"], fixed["initial_arm_angle"],
        fixed["arm_drag_coefficient"], fixed["projectile_drag_coefficient"], fixed["joint_friction_coefficient"],
        fixed["bearing_friction_coefficient"], fixed["pivot_shaft_radius"],
        config.machine is MachineType.PULLEY,
        config.target_distance, config.efficiency_weight, config.distance_weight, config.mass_weight,
        config.slack_penalty_weight, config.snap_penalty_weight, config.jerk_penalty_weight,
    )


def _no_valid_design_message(config: OptimizationConfig) -> str:
    """Why a search came back with nothing, in terms of the settings to change.

    Worth saying out loud rather than just reporting a failure, because a search that
    scores every design INVALID_COST has a perfectly flat landscape, converges on it, and
    returns whichever design the population happened to be standing on - which reads
    downstream as a result rather than as the absence of one. The settings that can empty
    a search space are quoted back because the fix is always to move one of them, and the
    run itself is the only thing that knows which were set.
    """
    locked = ", ".join(f"{name}={value:g}" for name, value in sorted(config.locked_params.items()))
    narrowed = ", ".join(sorted(config.param_bounds))
    pivot = float(config.fixed_params.get("pivot_height", TrebuchetParams.pivot_height))

    reason = (
        f"\nNo design in the search space produced a throw. Its fixed geometry stands on a "
        f"{pivot:g} m pivot"
    )
    reason += f", with {locked} locked" if locked else ", with nothing locked"
    reason += f" and the range narrowed on {narrowed}." if narrowed else " and default search ranges."
    reason += (
        "\nA locked parameter that rules out the rest of the space is the usual cause, followed by "
        "ranges narrowed to a region where nothing releases, and then a pivot height the arm cannot "
        "work against. Widen whichever of those this run set."
    )
    return f"The optimizer found no valid design for the {config.machine.value} machine.{reason}"


def thread_count(requested: Optional[int] = None) -> int:
    """How many threads the fast engine would actually use for `requested`.

    `None` reports the count numba is already set to. A number is clamped to
    `fastsim.thread_ceiling()`, since a process cannot be given more threads than it was
    imported with - ask for more than the machine was started with and this is what says
    so, rather than the count silently not taking. Raising the ceiling is
    TREBUCHET_NUM_THREADS' job (see fastsim._seed_thread_ceiling).

    Returns 0 when the fast engine is unavailable, there being no thread pool to report.
    """
    if not _FASTSIM_AVAILABLE:
        return 0
    if requested is None:
        return int(fastsim.get_num_threads())
    return max(1, min(int(requested), fastsim.thread_ceiling()))


@contextmanager
def _numba_threads(requested: Optional[int]):
    """Run the block with numba's thread count set to `requested`, then put it back.

    Restored on the way out because the count is process-wide: a dashboard that optimizes
    twice, or a test that pins one thread, would otherwise leave every later run on
    whatever the last one asked for.
    """
    if requested is None or not _FASTSIM_AVAILABLE:
        yield
        return
    previous = fastsim.get_num_threads()
    fastsim.set_num_threads(thread_count(requested))
    try:
        yield
    finally:
        fastsim.set_num_threads(previous)


def optimize_trebuchet(
    config: Optional[OptimizationConfig] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> Tuple[TrebuchetParams, SimulationResult, object]:
    """Optimize trebuchet parameters for maximum efficiency at a target distance.

    Returns (optimal_params, sim_result, scipy_optimize_result).
    """
    config = config or OptimizationConfig()

    if not config.free_params:
        raise ValueError("All parameters are locked; nothing to optimize.")

    def _report_generation(intermediate_result):
        # Param name must be exactly `intermediate_result` - scipy inspects the callback's
        # signature to decide whether to pass the new-style OptimizeResult or legacy (xk, convergence).
        params = config.build_params(intermediate_result.x)
        try:
            # The same settings _objective scores at, rather than the display defaults
            # (rtol=1e-8 with a dense interpolant). This runs once per generation while
            # the search is waiting on it, and a progress row only reads distance,
            # efficiency and the metrics dict - nothing that needs an interpolant, and
            # nothing the last two digits of rtol can move. Matching the objective also
            # makes the row honest: the distance shown is the one the generation was
            # actually scored on, not a more accurate number DE never saw.
            result = simulate_trebuchet(params, rtol=1e-6, dense_output=False)
        except Exception:
            result = None
        progress_callback(intermediate_result.nit, intermediate_result.fun, params, result)

    # fastsim models both linkages, so the machine no longer decides the engine.
    use_fast = config.use_fast_engine and _FASTSIM_AVAILABLE
    if use_fast:
        # The whole search runs inside one thread-count setting rather than one per
        # generation: the count is process-wide state, and DE calls the objective
        # hundreds of times.
        with _numba_threads(config.threads):
            de_result = differential_evolution(
                partial(_objective_vectorized, config=config),
                config.bounds,
                seed=config.seed,
                maxiter=config.max_iterations,
                popsize=config.population_size,
                atol=config.absolute_tolerance,
                vectorized=True,
                updating="deferred",
                disp=config.display_progress,
                callback=_report_generation if progress_callback is not None else None,
            )
    else:
        de_result = differential_evolution(
            partial(_objective, config=config),
            config.bounds,
            seed=config.seed,
            maxiter=config.max_iterations,
            popsize=config.population_size,
            atol=config.absolute_tolerance,
            workers=config.workers,
            disp=config.display_progress,
            callback=_report_generation if progress_callback is not None else None,
        )

    if de_result.fun >= INVALID_COST:
        # Every design the search touched scored INVALID_COST, so `de_result.x` is not a
        # winner - it is wherever the population happened to be standing when a completely
        # flat landscape met the convergence test. Simulating it and returning it anyway
        # is what produced a results panel of blanks and a log of empty rows: the caller
        # had no way to tell "here is the best machine" from "there was no machine".
        raise ValueError(_no_valid_design_message(config))

    optimal_params = config.build_params(de_result.x)
    sim_result = simulate_trebuchet(optimal_params, track_energy=True, simulate_aftermath=True)

    return optimal_params, sim_result, de_result
