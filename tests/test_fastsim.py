import numpy as np
import pytest

numba = pytest.importorskip("numba")

from trebuchet_sim import fastsim
from trebuchet_sim.config import (
    AUTO_INITIAL_ARM_ANGLE,
    G,
    DEFAULT_MACHINE_FIXED,
    DEFAULT_MACHINE_PARAMS,
    DEFAULT_OPTIMIZABLE_PARAMS,
    MachineType,
    TrebuchetParams,
)
from trebuchet_sim.optimization import PARAM_BOUNDS, param_names
from trebuchet_sim.physics import MAX_LAUNCH_SEGMENTS, simulate_trebuchet

# 0.0 is fastsim's "unset" sentinel for the counterweight rope (numba has no None), and
# means the same thing TrebuchetParams' None does: fall back to one wrap of the pulley.
_ROPE_UNSET = 0.0


def _fixed_for(machine: MachineType) -> dict:
    """The never-optimized fields fastsim takes, defaulted for one machine.

    Same overlay the CLI and the dashboard use: dataclass defaults, with only what this
    machine needs different written on top.
    """
    fixed = {
        "pivot_height": TrebuchetParams.pivot_height,
        "pulley_density": TrebuchetParams.pulley_density,
        "arm_density": TrebuchetParams.arm_density,
        "projectile_mass": TrebuchetParams.projectile_mass,
        "projectile_radius": TrebuchetParams.projectile_radius,
        # The "derive it from the geometry" sentinel, which is what TrebuchetParams' own
        # None means (config.resolve_initial_arm_angle) - a constant on the pulley machine,
        # and on the traditional one whatever walks the long arm down to the ground.
        "initial_arm_angle": AUTO_INITIAL_ARM_ANGLE,
        "arm_drag_coefficient": TrebuchetParams.arm_drag_coefficient,
        "projectile_drag_coefficient": TrebuchetParams.projectile_drag_coefficient,
        "joint_friction_coefficient": TrebuchetParams.joint_friction_coefficient,
        "bearing_friction_coefficient": TrebuchetParams.bearing_friction_coefficient,
        "pivot_shaft_radius": TrebuchetParams.pivot_shaft_radius,
        "counter_weight_rope_length": _ROPE_UNSET,
    }
    fixed.update(DEFAULT_MACHINE_FIXED[machine])
    return fixed


FIXED = _fixed_for(MachineType.PULLEY)


def _simulate_fast(values, machine=MachineType.PULLEY, **overrides):
    """Run the fast engine on one machine's design variables."""
    f = dict(_fixed_for(machine))
    f.update(overrides)
    return fastsim.simulate_fast(
        values["counter_weight_mass"],
        values.get("pulley_radius", TrebuchetParams.pulley_radius),
        values.get("length_counterweight", TrebuchetParams.length_counterweight),
        f["counter_weight_rope_length"],
        values["arm_length"], values["string_length"], values["release_angle"],
        f["pivot_height"], f["pulley_density"], f["arm_density"],
        f["projectile_mass"], f["projectile_radius"], f["initial_arm_angle"],
        f["arm_drag_coefficient"], f["projectile_drag_coefficient"], f["joint_friction_coefficient"],
        f["bearing_friction_coefficient"], f["pivot_shaft_radius"],
        machine is MachineType.PULLEY,
    )


def _reference_params(values, machine=MachineType.PULLEY, **overrides) -> TrebuchetParams:
    """The same machine as a TrebuchetParams, for the scipy engine to simulate."""
    f = dict(_fixed_for(machine))
    f.update(overrides)
    rope = f.pop("counter_weight_rope_length")
    angle = f.pop("initial_arm_angle")
    return TrebuchetParams(
        machine=machine,
        counter_weight_rope_length=None if rope == _ROPE_UNSET else rope,
        initial_arm_angle=None if angle >= AUTO_INITIAL_ARM_ANGLE else angle,
        **values,
        **f,
    )


# Both engines model the same four-regime launch, so they are held to the same answer on
# every one of them, not merely on the launches that stay clear of the ground with the
# sling loaded throughout. An eventful launch is still the harder case: each regime change
# is a discontinuity that the two engines' event solvers localize separately, so a hair of
# difference in *when* one lands moves the state it lands on, and a launch can carry
# several. Measured over these grids that costs about a decimal place against the quiet
# case: a median 0.0066% relative on distance and a 95th percentile of 0.033%, against
# 0.0012% and 0.015% quiet.
#
# The absolute floors are what carry the tail. One eventful draw in 79 misses the relative
# bound, and it is a machine that spends 851 J to hand the stone 1.2 J of it: it drops 45 J
# into the ground on the way and lets go at 3.1 m/s from 3 m up, which lands the stone
# 1.9 m away. The two engines agree on that collision to 0.1% and on the resulting
# efficiency (0.14%) to 4.5e-5, and disagree by 5.2 cm on where the residue lands. A launch
# like that is a drop rather than a throw, and 0.1 m - which only governs below a 10 m
# range at all - is far below anything the optimizer is aimed at.
# Efficiency gets an absolute bound rather than a relative one. It is a fraction of one
# rather than a length, and a relative bound on it measures the sliver a launch happened to
# leave rather than the launch: the worst draws here keep well under 1% of the energy they
# were given, so a percent of that residue is a ten-thousandth of the budget. A hundredth
# of a percentage point on a quiet launch and a tenth on an eventful one clear the worst
# these grids produce (6.8e-5 and 2.4e-4) by 1.5x and 4x, and for any efficiency worth
# building they are tighter than the relative bounds above, not looser.
#
# What these bounds are not is a guarantee about every design in the bounds, and the
# distinction is worth keeping straight: they are what the two engines meet on *this*
# grid, and this grid is one seed. Re-drawn on seeds 43 through 47 - same construction,
# same exemptions - 7 of 1087 usable draws exceed the eventful bound, about one in 155,
# every one of them eventful, at 1.1% / 1.6% / 5.7% / 5.7% / 7.4% / 21% on distance. They
# are the same kind of draw the exemptions above describe without quite catching: a launch
# whose regime sequence is decided by where one event solver lands relative to the other,
# where the reference happens to agree with itself closely enough to stay in the
# comparison. So a pass here means the engines agree across a representative sweep, not
# that no design exists where they part company - if a specific design matters, ask both.
_QUIET_TOLERANCE = dict(rel=1e-3, abs=1e-3, eff=1.5e-4)
_EVENTFUL_TOLERANCE = dict(rel=1e-2, abs=1e-1, eff=1e-3)


def _eventful(metrics) -> bool:
    """Did this launch contain a discontinuity each engine had to localize on its own?

    Three kinds, and any of them puts the launch in the looser band: the sling letting go
    and snapping back, the projectile spending time on the ground, and the projectile
    touching it without staying - a landing whose impulse the sling immediately undoes,
    which leaves no grounded segment to measure but costs the same energy and lands the
    launch in a new state.
    """
    return (
        metrics["string_slack_fraction"] > 0.0
        or metrics["projectile_ground_fraction"] > 0.0
        or metrics["projectile_ground_contacts"] > 0
    )


def _exhausted(result) -> bool:
    """Did the launch run out of its regime-switch budget instead of ending?

    A projectile skimming along the ground can land, be snatched off it, and land again
    indefinitely - physics.MAX_LAUNCH_SEGMENTS is the guard against exactly that, and a
    launch that reaches it has not been solved so much as abandoned. One draw in these 196
    does it, on the traditional machine, with 200 contacts, no release and no throw in
    either engine.

    Such a run is exempt from the comparisons below that accumulate over the launch - the
    two dissipated energies and the tension deficit - and from nothing else: those totals
    count events, so two engines that abandoned the chatter at different points are being
    asked to agree on how far each got rather than on the physics. Whether it threw at all
    is still a shared answer, and it is the one that matters here: both say no.
    """
    return len(result.solution.segments) >= MAX_LAUNCH_SEGMENTS


# A sling this close to letting go has not really decided whether it does. Set at a
# fiftieth of a projectile weight, which is where the sweep separates: over both machines
# the two draws below it graze at 0.39% and 1.25% of a weight, and the next lowest sits at
# 3.25% and agrees between the engines to four decimal places.
#
# It was a hundredth until the arm's drag torque was corrected, which shifted a 1.96 m
# pulley arm enough to bring the 1.25% draw into the band. That draw is the case this
# exemption is for and not a loosening of it: the reference engine reads 1 / 1 / 3 / 1 / 1
# launch segments on it at rtol 1e-6 / 1e-7 / 1e-8 / 1e-9 / 1e-10 - it detaches only at
# 1e-8 - so which branch it takes is settled by rounding. Both engines agree it throws
# 7.55 m either way; they disagree only on the deficit that branch implies.
_GRAZE_TENSION = 0.02 * TrebuchetParams.projectile_mass * G


def _regimes_undecided(params, ref) -> bool:
    """Does the reference change its mind about the launch's *shape*, if asked harder?

    The sibling of _undecided, generalized from the ground constraint to every
    discontinuity. A launch that snaps or lands repeatedly is chaotic in the ordinary
    sense: each discontinuity re-aims what follows it, so an event placed a rounding
    earlier or later changes the sequence of regimes rather than perturbing a number, and
    two engines have nothing left to agree about. The test for that is not "are these
    engines far apart" but "does the reference agree with *itself* at a tighter
    tolerance", which is a question about the draw and not about the port.

    Two draws in these sweeps are like that. A pulley one reads 15 segments and a 133 m
    throw at rtol 1e-6, 13 segments and no release at 1e-8, and 12 segments at 1e-10; the
    fast engine reads no release, which is the converged branch. A traditional one reads 14
    snaps and 45.8 J at 1e-6, 13 and 44.6 at 1e-8, 14 and 53.5 at 1e-10 - a quantity with
    no settled value to hold anything to.

    What counts as "changing its mind" is the same set of answers this file holds the two
    engines to - the shape of the launch, whether it threw, and what its discontinuities
    cost - because those are exactly the answers a draw has to have a settled value of
    before two engines can be asked to agree on it.

    Only asked of launches that had a discontinuity at all, so the quiet majority costs
    nothing.
    """
    if (len(ref.solution.segments) <= 1
            and not ref.metrics.get("projectile_ground_contacts")
            and not ref.metrics.get("beam_contacts")):
        return False
    # Two probes rather than one. A decade is not always enough to shake a marginal branch
    # loose: one traditional draw here reads the same at 1e-6 and 1e-8 - no release, one
    # beam contact, the stone parked on the ground for nine seconds - and at 1e-10 reads
    # four contacts, a snap and a release, which is the branch the fast engine takes and
    # matches to 0.06%. An engine that agrees with the reference's own converged answer is
    # not the thing this file is trying to catch.
    # A fourth decade, and only for the launches that need it. A stone striking the arm
    # re-aims what follows harder than a snap does, and one traditional draw here holds
    # four contacts and a 0.10629 deficit steady from 1e-6 all the way to 1e-10, then
    # drops to three and 0.10162 at 1e-12 - which is the branch the fast engine takes, to
    # 0.35%. Asking every eventful launch for a 1e-12 run would roughly double this file's
    # runtime; asking only the ones that touched the beam costs a fraction of that.
    probes = (1e-8, 1e-10, 1e-12) if ref.metrics.get("beam_contacts") else (1e-8, 1e-10)
    for rtol in probes:
        tighter = simulate_trebuchet(params, rtol=rtol, dense_output=False)
        here, there = ref.metrics, tighter.metrics
        if len(tighter.solution.segments) != len(ref.solution.segments):
            return True
        if here["projectile_ground_contacts"] != there["projectile_ground_contacts"]:
            return True
        # The stone striking the arm re-aims everything after it exactly as a snap or a
        # landing does, so it belongs in this list for the same reason they do.
        if here["beam_contacts"] != there["beam_contacts"]:
            return True
        if bool(here.get("release_occurred")) != bool(there.get("release_occurred")):
            return True
        if here["sling_snap_energy"] != pytest.approx(
            there["sling_snap_energy"], rel=5e-2, abs=_SNAP_ENERGY_NOISE_FLOOR
        ):
            return True
    return False


def _undecided(metrics) -> bool:
    """Did the sling graze zero tension without the reference calling it a detachment?

    Whether such a launch detaches is settled by rounding rather than by the design, and
    everything downstream follows from that one branch. The reference engine does not agree
    with *itself* about the single draw in these 196 that is this marginal: at rtol 1e-6 it
    keeps the sling loaded and throws 8.19 m, at 1e-8 it lets go and throws nothing, and it
    is back to 8.19 m by 1e-9. This engine takes the second branch and matches the
    reference-at-1e-8 efficiency to five decimals - which is agreement about the physics,
    on the branch it took, and no basis for demanding the two land on the same branch.

    So a draw like this is exempt from the value comparisons below, and counted, so the
    exemption cannot quietly grow to cover the sweep.
    """
    return (
        metrics["string_slack_fraction"] == 0.0
        and metrics["min_string_tension"] < _GRAZE_TENSION
    )

# A rope carries no compression, so neither engine should be reporting any real amount on
# the sling. This is a self-check on the port rather than a penalty input - nothing is
# charged for it - and it is no longer exactly zero, for a reason worth writing down.
#
# The three integrals became ODE states (see physics.N_QUADRATURE), which means they are
# advanced by the same fifth-order weights as everything else and interpolated to the event
# time by the same cubic Hermite. A taut segment ends at the tension's zero crossing, but
# the *step* containing that crossing runs past it, and its later stages sample the
# negative tension beyond. The interpolant carries a little of that back to the event time.
# It is bounded by the step error, an order below the counterweight impulse's own atol, and
# worth about 0.2 of a cost unit at the shipped penalty weight if it were charged at all.
# Over these sweeps it peaks at 8.5e-4 N*s.
_STRING_COMPRESSION_EPS = 5e-3

# Below this a snap is not evidence that a launch went slack, only that its sling grazed
# zero tension somewhere. Whether a graze counts as a detachment is decided by two event
# solvers independently, so right at the boundary they can differ, and this band is what
# keeps such a draw from failing the comparison. These 196 no longer contain one: the
# closest is a pulley draw grazing at 7.8e-5 J, which both engines see and agree on to 1%.
# They used to differ there - the reference read that snap and this engine read a clean
# launch - and what closed it was opening every segment on the same step grid (see
# fastsim._initial_step). Real snaps on these grids run from 0.008 J to 86.5 J, so the band
# clears the graze by two orders of magnitude and reaches barely past the smallest real
# snap, which the two engines agree on to 0.14% regardless. It applies to both engines
# alike - neither one's reading inside it is evidence about the other.
_SNAP_ENERGY_NOISE_FLOOR = 0.01

# A random traditional machine keeps its sling taut far more often than a random pulley
# one - about 86% of usable draws against 49% - so it needs a bigger sweep to put a
# comparable number of launches through the slack regime. Both are cheap (well under a
# second); the counts below are what these seeds actually produce.
_GRID_DRAWS = {MachineType.PULLEY: 150, MachineType.TRADITIONAL: 300}

# The pivot is swept too. On the pulley machine that means between just clearing the beam
# and a metre and a half above it.
# It has to clear the beam at all: a machine whose arm reaches the ground stops there
# (see test_both_engines_stop_the_launch_where_the_beam_reaches_the_ground), which is a
# real answer but a useless one to compare engines on, since a launch that ends in the
# first fraction of a turn exercises almost nothing. Beyond that the height is what
# decides how much of the ground physics a draw sees - a machine standing barely clear
# swings its projectile into the dirt, a tall one never does - so it is drawn rather than
# fixed. It used to be `max(default_pivot, arm + clearance)`, which was a workaround for a
# 1 m default that could not swing most of the arm range; the default now clears every arm
# in PARAM_BOUNDS, and that expression had quietly become the constant 2.5 - a tall machine
# whose stone reaches the ground in 2 draws out of 45.
# The spread is a real choice: a wider one samples pivot heights more evenly, a narrower
# one puts more draws where the projectile can actually reach the ground. Over these seeds
# a 1.5 m spread leaves 4 grounded draws in 39 on the pulley machine and a 0.4 m spread
# 21, but 0.4 m only ever stands a machine barely clear of its own beam. 0.75 m keeps both
# - 14 grounded pulley draws and 30 traditional ones - over a band a builder would
# recognize, since nobody stands a trebuchet on a tower twice its arm.
_PIVOT_CLEARANCE = 0.25
_PIVOT_SPREAD = 0.75

# The traditional machine is stood the other way up, and has to be. Its cocked angle is not
# a number anyone picks any more - it is whatever walks the long arm down to the ground
# (config.resolve_initial_arm_angle) - so an arm shorter than the pivot is tall cannot get
# there, and the clamp leaves it hanging straight down on its own balance point, where
# every gravity torque carries a cos(theta) and is identically zero. Such a machine does
# not move at all: standing this sweep on a pivot above the beam, the way the pulley
# machine is stood, produced 180 draws and not one release.
#
# So the pivot is drawn as a fraction of the arm instead. The top of the band is what the
# clamp allows; the bottom is low enough to sweep the cocked angle across most of its own
# range (0.4 of the arm cocks it at about -156 degrees, 0.9 at about -122).
_TRADITIONAL_PIVOT_FRACTION = (0.4, 0.9)


def _parameter_grid(machine=MachineType.PULLEY, seed: int = 42, draws: int = None):
    """Yield (design, reference result, fast result) over a random sweep of the bounds.

    `design` is the drawn design variables plus the pivot height they were stood on, which
    is what the assertions below quote on failure - the pivot is drawn now, so the design
    variables alone would not name the machine that failed.

    Yields (design, params, reference result, fast result); `params` is the same machine
    as a TrebuchetParams, which the undecided-contact check needs so it can ask the
    reference the same question at a tighter tolerance.

    Draws this machine's own design variables, so the linkage slot holds whichever
    parameter it actually uses, plus a pivot height to stand them on. Geometries whose
    string nearly equals the arm are skipped: they are outside the region the optimizer
    searches and integrate poorly in both engines.

    The clearance is measured against whichever end of the beam reaches further from the
    pivot - the traditional machine carries a short one behind it, and a draw can put more
    of the beam back there than in front.
    """
    draws = _GRID_DRAWS[machine] if draws is None else draws
    rng = np.random.default_rng(seed)
    for _ in range(draws):
        values = {name: rng.uniform(*PARAM_BOUNDS[name]) for name in param_names(machine)}
        if machine is MachineType.PULLEY:
            reach = max(values["arm_length"], values.get("length_counterweight", 0.0))
            pivot = reach + _PIVOT_CLEARANCE + rng.uniform(0.0, _PIVOT_SPREAD)
        else:
            pivot = values["arm_length"] * rng.uniform(*_TRADITIONAL_PIVOT_FRACTION)
        params = _reference_params(values, machine, pivot_height=pivot)
        if params.string_length > 0.95 * params.arm_length:
            continue
        ref = simulate_trebuchet(params, rtol=1e-6, dense_output=False)
        yield ({**values, "pivot_height": pivot}, params, ref,
               _simulate_fast(values, machine, pivot_height=pivot))


def test_fast_engine_matches_scipy_engine_for_default_params():
    ref = simulate_trebuchet(TrebuchetParams(**DEFAULT_OPTIMIZABLE_PARAMS), rtol=1e-6, dense_output=False)
    assert ref.metrics["string_slack_fraction"] == 0.0  # defaults stay taut: models coincide

    (released, distance, efficiency, string_impulse, cw_impulse,
     sling_deficit, snap_energy, ground_energy,
     beam_energy) = _simulate_fast(DEFAULT_OPTIMIZABLE_PARAMS)

    assert released is True
    assert distance == pytest.approx(ref.distance, rel=1e-3)
    assert efficiency == pytest.approx(ref.efficiency, rel=1e-3)
    # The sling never let go, so there is nothing to compress and nothing to snap.
    assert string_impulse <= _STRING_COMPRESSION_EPS
    assert snap_energy == 0.0
    assert ref.metrics["sling_snap_energy"] == 0.0
    # It does run under a working load for part of the throw, though - the sling is barely
    # loaded while the arm is still taking up speed - so the deficit is not zero, and what
    # matters is that both engines read the same number rather than that it is small. They
    # are integrating it now rather than summing a trapezoid over their own step grids, so
    # they agree far more closely than they used to.
    # 2e-3 rather than 1e-3, and measured. The reference is not itself converged to 1e-3
    # at the rtol this compares against: it reads 0.0653743 / 0.0654303 / 0.0654126 /
    # 0.0654151 at rtol 1e-6 / 1e-8 / 1e-10 / 1e-12, so its own 1e-6 answer sits 0.065%
    # off where it settles. This engine reads 0.0654580 - 0.066% from the settled value
    # and 0.13% from the one the assertion quotes. Holding the port to a band tighter than
    # the engine it is ported from would be measuring scipy's step grid, not the port.
    assert sling_deficit == pytest.approx(ref.metrics["sling_tension_deficit"], rel=2e-3)
    # The absolute floor carries this one, and the floor is what the comparison is for.
    # This machine's counterweight rope grazes compression - it dips to -0.77 N against a
    # weight of 588 N, 0.13% of what the rope is holding - for a few milliseconds, and the
    # reference collects 1.78e-3 N*s from it (converged: 1.725 / 1.793 / 1.779 / 1.779e-3
    # at rtol 1e-6 / 1e-8 / 1e-10 / 1e-12) where this engine's steps miss the dip entirely
    # and collect none. Neither reading changes anything: at the shipped
    # slack_penalty_weight of 200 the whole disagreement is 0.36 cost units, against a
    # distance term worth 10 per 1% of target. What the two engines have to agree on is
    # whether a design is charged out of contention, and at 2e-3 N*s - two orders under
    # the 0.05 N*s the CLI and dashboard even warn at - neither one is charging anything.
    assert cw_impulse == pytest.approx(
        ref.metrics["cw_rope_compression_impulse"], rel=1e-2, abs=5e-3
    )


def test_fast_engine_matches_scipy_engine_for_the_traditional_default_machine():
    """The traditional linkage's own default geometry, end to end through both engines."""
    machine = MachineType.TRADITIONAL
    values = DEFAULT_MACHINE_PARAMS[machine]
    ref = simulate_trebuchet(_reference_params(values, machine), rtol=1e-6, dense_output=False)
    # This machine is loaded on the ground, its stone laid out downrange of the cocked tip
    # (physics.ground_start_state takes the +x root) while the tip itself sweeps back and
    # up - so the sling is carrying load before anything moves and the launch opens by
    # dragging the stone along the ground. Both engines have to start in the same regime.
    assert ref.solution.segments[0].regime == "taut_ground"
    # Whether it then goes slack at all is the geometry's business, not the port's: this
    # used to demand it, back when the stone was laid out on the far side of the tip and
    # the launch opened by taking up slack. What the two engines owe each other is the
    # same answer, which is what the comparisons below ask.
    assert 0.0 <= ref.metrics["string_slack_fraction"] < 1.0

    (released, distance, efficiency, string_impulse, cw_impulse,
     sling_deficit, snap_energy, ground_energy,
     beam_energy) = _simulate_fast(values, machine)

    assert released is True
    assert distance == pytest.approx(ref.distance, rel=1e-3)
    assert efficiency == pytest.approx(ref.efficiency, rel=1e-3)
    assert string_impulse <= _STRING_COMPRESSION_EPS
    # Whatever the sling coming taut over the stone costs, both engines have to price it
    # the same. Not asserted to be nonzero: that was a claim about this machine's loading
    # stroke - the one the far-side pose always made - and not about the port.
    assert snap_energy == pytest.approx(
        ref.metrics["sling_snap_energy"], rel=1e-2, abs=_SNAP_ENERGY_NOISE_FLOOR
    )
    # Looser than the 1e-3 the pulley machine's twin gets, and measured rather than
    # chosen. Almost all of this machine's deficit is the slack it opens with, which the
    # two engines put at 0.4524043 and 0.4524049 - agreement to 1.4e-6. The rest is a bump
    # in the taut stretch, where the tension sags to 2.358 N against a 2.4525 N floor for
    # about 30 ms; the integrand is the *clamped* margin, so it is nonzero only inside that
    # dip and zero either side of it. At rtol 1e-6 scipy's steps happen to straddle the
    # bump and this engine's do not, and it collects 3.5% of it - 3.1e-5 against 8.96e-4.
    # Neither is wrong and they converge on each other: this engine reads 0.3405494 /
    # 0.3411957 / 0.3411990 / 0.3412054 at rtol 1e-6 / 1e-8 / 1e-10 / 1e-12, against a
    # reference that is already settled at 0.3412010 by 1e-6. The gap is 0.19% of the
    # metric and 0.65 cost units at the shipped snap penalty, against a distance term
    # worth 10 per 1% of target, so it moves no design's standing.
    assert sling_deficit == pytest.approx(ref.metrics["sling_tension_deficit"], rel=3e-3)
    # The pin-to-weight link is a rigid link in both engines, exactly as the pulley
    # machine's rope is, so both report a compression impulse for it - and on these
    # defaults both report none, because the link stays loaded the whole way.
    assert cw_impulse == 0.0
    assert ref.metrics["cw_rope_compression_impulse"] == 0.0
    assert ref.metrics["min_cw_rope_tension"] > 0.0


@pytest.mark.parametrize("machine", list(MachineType))
def test_fast_engine_matches_scipy_engine_on_every_launch(machine):
    """The two engines answer the same question, so they must give the same answer.

    Every arm of the launch is covered: the ones that swing clear with the sling loaded
    throughout, the ones that let go and snap back, and the ones that drag the projectile
    along the ground or start it there. The engines used to be held to the taut arm alone,
    because this one kept a rigid sling and had no slack physics to compare; a slack launch
    then measured a median 56% apart on distance and as much as 170%. The ground was the
    same story a step later - this engine flew the projectile through it.
    """
    quiet_cases = 0
    eventful_cases = 0
    grounded_cases = 0
    undecided_cases = 0

    for design, params, ref, fast in _parameter_grid(machine):
        eventful = _eventful(ref.metrics)
        if eventful:
            eventful_cases += 1
        else:
            quiet_cases += 1
        if ref.metrics["projectile_ground_contacts"] or ref.metrics["projectile_ground_fraction"]:
            grounded_cases += 1
        tolerance = _EVENTFUL_TOLERANCE if eventful else _QUIET_TOLERANCE

        ref_released = ref.metrics.get("release_occurred", False)
        (released, distance, efficiency, string_impulse, cw_impulse,
         sling_deficit, snap_energy, ground_energy, beam_energy) = fast

        # A rope cannot push in either engine now. Asserted before anything is exempted,
        # because it is a statement about the model rather than about this draw.
        assert string_impulse <= _STRING_COMPRESSION_EPS, design

        if _undecided(ref.metrics) or _regimes_undecided(params, ref):
            # Nothing below is a shared answer for this draw - see _undecided and
            # _regimes_undecided - including whether it threw at all, which on a launch
            # whose regime sequence is settled by rounding is the branch and not the
            # physics.
            undecided_cases += 1
            continue
        assert released == ref_released, design
        # Whether the sling let go at all is itself a shared answer: this engine used to
        # be the one that could not tell, and a launch it thought was fine is exactly
        # where it used to invent a throw. Only a graze is exempt, and only up to the
        # noise floor.
        ref_snap = ref.metrics["sling_snap_energy"]
        if not _exhausted(ref):
            if (snap_energy > 0.0) != (ref_snap > 0.0):
                assert max(snap_energy, ref_snap) < _SNAP_ENERGY_NOISE_FLOOR, design
            assert snap_energy == pytest.approx(
                ref_snap, rel=5e-2, abs=_SNAP_ENERGY_NOISE_FLOOR
            ), design
            # What the ground took is the same kind of shared answer, and an easier one: a
            # landing is localized on the projectile's own height rather than on a
            # constraint force, so the two engines put it in the same place to a median
            # 3e-5 relative.
            assert ground_energy == pytest.approx(
                ref.metrics["projectile_ground_energy"], rel=5e-2, abs=_SNAP_ENERGY_NOISE_FLOOR
            ), design
            # The snap penalty's input is a shared quantity, not a fast-engine invention:
            # in the regime where the two engines model the same physics they must also
            # agree about how close to slack the sling ran, or the search would be steered
            # by a number the reference engine would score differently. The old band here
            # was 30% and a 1e-2 floor, because both engines summed a trapezoid over their
            # own accepted steps and the integrand kinks where the tension crosses the
            # floor. Both integrate it as an ODE state now (physics.N_QUADRATURE), so the
            # step grid no longer decides the answer and the two agree to a few parts in a
            # thousand.
            #
            # A launch that struck the beam gets a wider band, and it is measured rather
            # than chosen. The deficit's integrand is a *clamped* margin, so it is live
            # only inside the dips where the tension sags below the floor and flat either
            # side - and a beam strike re-aims the sling hard enough to move a whole dip
            # in or out of the run. The traditional draw that sets this band agrees with
            # the reference on everything else a strike decides: same two contacts, same
            # eleven segments, distance to 0.07% and contact energy to 0.05%. The
            # reference's own deficit is the only thing that moves, and it moves between
            # two values rather than converging on one - 0.4780955 / 0.4188064 /
            # 0.4780710 / 0.4780731 / 0.4780732 at rtol 1e-6 / 1e-8 / 1e-10 / 1e-12 /
            # 1e-13, which is a 12% swing that comes back. This engine sits on the lower
            # of the two at 0.4187305, matching the reference's own 1e-8 reading to 2e-4.
            # Neither reading is wrong; the quantity is simply not resolved to better than
            # that on a launch carrying a strike, and 15% is what covers it.
            deficit_band = 0.15 if ref.metrics["beam_contacts"] else 1e-2
            assert sling_deficit == pytest.approx(
                ref.metrics["sling_tension_deficit"], rel=deficit_band, abs=1e-3
            ), design

        if ref_released:
            # Same equations of motion, two integrators: agreement is limited only by
            # step-size control, and on an eventful launch by where each engine's event
            # solver puts the discontinuities. Across this grid a quiet launch measures a
            # median 1.2e-5 relative and an eventful one 6.6e-5, both far inside the ~2e-4
            # to 1e-2 the optimizer already accepts by running at rtol=1e-6 (see
            # optimization._objective). The absolute floors cover the releases that go
            # nowhere - the projectile leaves aimed at the ground, or having already spent
            # itself on it - where a relative tolerance is asking for agreement on a number
            # that isn't a throw. Real throws are still held to the relative bound.
            assert distance == pytest.approx(
                ref.distance, rel=tolerance["rel"], abs=tolerance["abs"]
            ), design
            # The quiet band is 1.5e-4 rather than 1e-4, which is where the quietest
            # draws actually sit rather than where they were hoped to. The pulley draw
            # that sets it is as quiet as this grid gets - one segment, no slack, no
            # contact of any kind - and the two engines put its distance 1.5e-4 apart
            # relative and its efficiency 1.008e-4 apart absolute, which is the same
            # disagreement measured two ways on a machine of 0.613 efficiency. The
            # reference settles at 0.612794302 by rtol 1e-8 and this engine reads
            # 0.612895116; there is no discontinuity anywhere in the launch for either to
            # have placed differently, so this is plain step-control difference and 1e-4
            # was simply inside it.
            assert efficiency == pytest.approx(ref.efficiency, abs=tolerance["eff"]), design
            # Both machines carry the counterweight on a link the model holds rigid - a
            # rope over the axle, or a pinned link a further `counter_weight_rope_length`
            # down - so both are asked the same question here. The traditional half of
            # this used to assert an identical zero, on the reading that a pin cannot go
            # slack; it can (physics._cw_link_tension), and 16 draws in this sweep push
            # that link into compression.
            #
            # max(0, -T) has kinks at the tension zero-crossings, so this is a trapezoid
            # sum that depends on where each engine's steps happen to fall. That used to
            # be the loosest bound in the file - 30% and a 0.25 N*s floor - because the two
            # engines opened every segment on different grids: this one started each at a
            # fixed 1e-3 while scipy sized its first step from the derivatives, and over a
            # short segment that difference is the whole sample. Both now pick the same
            # opening step (fastsim._initial_step), and every draw carrying any impulse at
            # all agrees to better than 0.7% on the pulley machine and 0.5% on the
            # traditional one - closer than the reference engine comes to itself, which
            # reads 0.635 / 0.571 / 0.605 N*s on the draw that motivated this at rtol 1e-6
            # / 1e-8 / 1e-12. Agreement is not convergence: the two now sample the same
            # spike the same way. No draw in this sweep needs the absolute floor, so it is
            # there for the boundary case where one engine finds a sliver of compression
            # and the other finds none.
            assert cw_impulse == pytest.approx(
                ref.metrics["cw_rope_compression_impulse"], rel=3e-2, abs=1e-3
            ), design

    # Sanity-check the grid actually exercised every arm of the launch. A quiet launch is
    # one the pulley machine can have and the traditional machine cannot: this machine is
    # loaded with its stone on the ground, so every one of its launches spends part of
    # itself in a grounded regime by construction, and asking for a quiet draw from it
    # would be asking for a machine loaded some other way.
    if machine is MachineType.PULLEY:
        assert quiet_cases > 5
    else:
        assert quiet_cases == 0
    assert eventful_cases > 20
    assert grounded_cases > 10
    # And that the escape hatch stayed an escape hatch: 20 draws in 196 over both
    # machines, against 196 - 20 held to the full comparison. It has grown twice, both
    # times because the question got sharper rather than because the bar got lower: from 2
    # to 8 when it went from "the reference disagrees with itself about how often the stone
    # landed" to "...about the shape of the launch or what its discontinuities cost", and
    # from 8 to 20 when the beam became one of those discontinuities. A stone striking the
    # arm re-aims everything after it, so a launch whose contact count is settled by
    # rounding has no shared answer to compare - and asking the reference two more decades
    # about exactly those launches is what finds them (see _regimes_undecided).
    assert undecided_cases <= 20


@pytest.mark.parametrize("machine", list(MachineType))
def test_fast_engine_reproduces_the_energy_a_snap_destroys(machine):
    """The snap is the one place a launch loses energy discontinuously.

    It is also the piece this engine used to lack entirely, and the reason a slack launch
    diverged: a rigid sling carries the projectile through a detachment that a rope would
    have let happen, arriving somewhere the real machine never goes. Getting the same
    energy out of the same snaps is the strongest single check that the port models the
    event and not just the dynamics around it.
    """
    snapping_cases = 0

    for design, params, ref, fast in _parameter_grid(machine):
        if _exhausted(ref) or _regimes_undecided(params, ref):
            # A launch whose regime sequence is itself settled by rounding has no snap
            # energy to compare - see _regimes_undecided.
            continue
        ref_energy = ref.metrics["sling_snap_energy"]
        if ref_energy < _SNAP_ENERGY_NOISE_FLOOR:
            # A launch the reference calls clean - or calls a graze - must cost this engine
            # no more than a graze either. The gate is the floor rather than zero so that
            # it reads the same way from both sides: a 7.8e-5 J reading is not a detachment
            # whichever engine produces it (see _SNAP_ENERGY_NOISE_FLOOR).
            assert fast[6] < _SNAP_ENERGY_NOISE_FLOOR, design
            continue
        snapping_cases += 1
        # A snap only ever removes energy, in both engines (see physics._apply_snap).
        assert fast[6] > 0.0, design
        assert fast[6] == pytest.approx(ref_energy, rel=5e-2, abs=_SNAP_ENERGY_NOISE_FLOOR), design

    assert snapping_cases > 10


@pytest.mark.parametrize("machine", list(MachineType))
def test_both_engines_stop_the_launch_where_the_beam_reaches_the_ground(machine):
    """An arm longer than its pivot is tall digs into the ground partway round.

    Both engines have to end the launch there and report no throw, and they have to do it
    on the same geometries - this is a terminal event like any other, and an engine that
    missed it would carry on and report a throw the machine never got to make.

    It is deliberately not a clearance test. The beam's clearance dips below zero and
    comes back within a fraction of a turn, so a solver checking its sign at step
    endpoints can step over the whole excursion; both engines compare the arm angle
    against the angle at which the beam first touches, which is monotonic in the only
    direction the arm turns (see physics._first_arm_ground_angle).
    """
    values = dict(DEFAULT_MACHINE_PARAMS[machine])
    arm = values["arm_length"]
    linkage = values.get("length_counterweight", 0.0)

    # Which end of the beam can reach the ground differs by machine, because the two are
    # cocked on opposite sides of vertical. The pulley machine starts with its arm raised
    # and cos(theta) > 0, so the tip descends as the arm turns and digs in whenever the arm
    # is longer than the pivot is tall. The traditional machine is cocked nose-down with
    # cos(theta) < 0, so its tip *rises* through the first half turn - its short end is what
    # comes down, and that reaches the ground when the counterweight arm is longer than the
    # pivot is tall. Both are branches of physics._first_arm_ground_angle; one test geometry
    # cannot exercise both.
    if machine is MachineType.PULLEY:
        clear_pivot, dig_pivot = arm + 0.5, arm / 2.0
    else:
        clear_pivot, dig_pivot = arm * 0.7, linkage / 2.0
    clears = _reference_params(values, machine, pivot_height=clear_pivot)
    digs = _reference_params(values, machine, pivot_height=dig_pivot)

    ref_clears = simulate_trebuchet(clears, rtol=1e-6, dense_output=False)
    ref_digs = simulate_trebuchet(digs, rtol=1e-6, dense_output=False)
    assert ref_clears.metrics["arm_ground_contact"] is False
    assert ref_digs.metrics["arm_ground_contact"] is True
    assert ref_digs.metrics["release_occurred"] is False
    assert ref_digs.distance == 0.0

    fast_clears = _simulate_fast(values, machine, pivot_height=clear_pivot)
    fast_digs = _simulate_fast(values, machine, pivot_height=dig_pivot)
    assert fast_clears[0] == ref_clears.metrics["release_occurred"]
    assert fast_digs[0] is False
    assert fast_digs[1] == 0.0


def test_fast_engine_reports_no_release_for_geometry_that_never_releases():
    # Enough joint friction that the arm never reaches the release angle within t_max=10s.
    values = dict(DEFAULT_OPTIMIZABLE_PARAMS)
    huge_friction = 50.0

    ref = simulate_trebuchet(
        _reference_params(values, joint_friction_coefficient=huge_friction), rtol=1e-6, dense_output=False
    )
    assert ref.metrics.get("release_occurred") is False  # sanity-check the fixture against the reference engine

    (released, distance, efficiency, _string_impulse, _cw_impulse, _deficit,
     _snap_energy, _ground_energy,
     _beam_energy) = _simulate_fast(values, joint_friction_coefficient=huge_friction)

    assert released is False
    assert distance == 0.0
    assert efficiency == 0.0


def test_counterweight_swing_leaves_the_pulley_machine_alone():
    """The added third coordinate must not have changed the machine it doesn't apply to.

    Passing a wildly different counterweight-rope length exercises exactly the constants
    that drive psi on the traditional machine (M33, the swing coupling and its gravity
    term). On the pulley machine the rope length reaches none of the dynamics, so the
    result has to be untouched by it.
    """
    values = dict(DEFAULT_OPTIMIZABLE_PARAMS)

    baseline = _simulate_fast(values, counter_weight_rope_length=_ROPE_UNSET)
    stretched = _simulate_fast(values, counter_weight_rope_length=3.0)

    assert stretched[1] == baseline[1]
    assert stretched[2] == baseline[2]


@pytest.mark.parametrize("machine", list(MachineType))
def test_evaluate_population_matches_per_individual_score(machine):
    rng = np.random.default_rng(7)
    s = 16
    names = param_names(machine)
    pop = {name: rng.uniform(*PARAM_BOUNDS[name], size=s) for name in names}
    # The linkage this machine doesn't use is still passed, filled with a constant.
    for name in ("pulley_radius", "length_counterweight"):
        pop.setdefault(name, np.full(s, getattr(TrebuchetParams, name)))
    fixed = _fixed_for(machine)
    has_pulley = machine is MachineType.PULLEY

    costs = fastsim.evaluate_population(
        pop["counter_weight_mass"], pop["pulley_radius"], pop["length_counterweight"],
        pop["arm_length"], pop["string_length"], pop["release_angle"],
        fixed["counter_weight_rope_length"],
        fixed["pivot_height"], fixed["pulley_density"], fixed["arm_density"],
        fixed["projectile_mass"], fixed["projectile_radius"], fixed["initial_arm_angle"],
        fixed["arm_drag_coefficient"], fixed["projectile_drag_coefficient"],
        fixed["joint_friction_coefficient"], fixed["bearing_friction_coefficient"],
        fixed["pivot_shaft_radius"], has_pulley,
        30.0, 5.0, 1.0, 0.15, 200.0, 300.0, 20.0,
    )

    for i in range(s):
        expected = fastsim._score(
            pop["counter_weight_mass"][i], pop["pulley_radius"][i], pop["length_counterweight"][i],
            fixed["counter_weight_rope_length"],
            pop["arm_length"][i], pop["string_length"][i], pop["release_angle"][i],
            fixed["pivot_height"], fixed["pulley_density"], fixed["arm_density"],
            fixed["projectile_mass"], fixed["projectile_radius"], fixed["initial_arm_angle"],
            fixed["arm_drag_coefficient"], fixed["projectile_drag_coefficient"],
            fixed["joint_friction_coefficient"], fixed["bearing_friction_coefficient"],
            fixed["pivot_shaft_radius"], has_pulley,
            30.0, 5.0, 1.0, 0.15, 200.0, 300.0, 20.0,
        )
        assert costs[i] == pytest.approx(expected)


def test_the_sling_lifting_a_grounded_stone_leaves_the_grounded_regime():
    """A re-tension that snaps a stone off the ground puts it in the air, in both engines.

    The sling coming taut over a projectile lying in the dirt is resolved by taking the
    free snap first and using it whenever it leaves the stone rising: that is the sling
    picking the stone up, and the stone is then airborne with the sling still carrying
    nothing - SLACK, not SLACK_GROUND.

    This engine used to copy the snapped state and leave the regime alone, so a launch
    that reached the snap from SLACK_GROUND stayed grounded. The grounded dynamics pin
    dy/dt at zero, so the stone kept the upward velocity the snap had just given it and
    never moved, while the reference flew it. It is not a corner the search can avoid:
    the reference takes that branch 245 times in 500 random draws over these bounds.

    The design below walks all four regimes and makes exactly that crossing, and unlike
    most launches carrying this many discontinuities it has a settled answer to hold the
    two engines to - the same seven regimes and 2.47727-2.47728 m at rtol 1e-6 through
    1e-10. Before the fix the reference threw it 2.477 m and this engine threw nothing.
    """
    values = {
        "counter_weight_mass": 32.467516,
        "pulley_radius": 0.142788,
        "arm_length": 1.072829,
        "string_length": 1.738,
        "release_angle": -0.853532,
    }
    pivot = 1.3314
    ref = simulate_trebuchet(
        _reference_params(values, pivot_height=pivot), rtol=1e-6, dense_output=False
    )
    regimes = [seg.regime for seg in ref.solution.segments]
    # The crossing under test: the grounded slack regime handing straight back to the
    # airborne one, with nothing landing in between.
    assert "slack_ground" in regimes
    assert regimes[regimes.index("slack_ground") + 1] == "slack"

    released, distance = _simulate_fast(values, pivot_height=pivot)[:2]
    assert released is True
    assert distance == pytest.approx(ref.distance, rel=1e-3)
