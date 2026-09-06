import numpy as np
import pytest

numba = pytest.importorskip("numba")

from trebuchet_sim import fastsim
from trebuchet_sim.config import (
    DEFAULT_INITIAL_ARM_ANGLE,
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
        # Read from the machine's start-angle table rather than the dataclass default,
        # which is None until __post_init__ resolves it per machine.
        "initial_arm_angle": float(DEFAULT_INITIAL_ARM_ANGLE[machine]),
        "arm_drag_coefficient": TrebuchetParams.arm_drag_coefficient,
        "projectile_drag_coefficient": TrebuchetParams.projectile_drag_coefficient,
        "joint_friction_coefficient": TrebuchetParams.joint_friction_coefficient,
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
        machine is MachineType.PULLEY,
    )


def _reference_params(values, machine=MachineType.PULLEY, **overrides) -> TrebuchetParams:
    """The same machine as a TrebuchetParams, for the scipy engine to simulate."""
    f = dict(_fixed_for(machine))
    f.update(overrides)
    rope = f.pop("counter_weight_rope_length")
    return TrebuchetParams(
        machine=machine,
        counter_weight_rope_length=None if rope == _ROPE_UNSET else rope,
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
_QUIET_TOLERANCE = dict(rel=1e-3, abs=1e-3, eff=1e-4)
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
# hundredth of a projectile weight, which is where the sweep separates cleanly: the one
# draw below it grazes at 0.39% of a weight, and the next lowest sits at 3.25% and agrees
# between the engines to four decimal places.
_GRAZE_TENSION = 0.01 * TrebuchetParams.projectile_mass * G


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

# A rope carries no compression, so neither engine should be reporting any on the sling.
# This is a self-check on the port rather than a tolerance: it measures 0 exactly on most
# draws and never worse than machine epsilon on the rest.
_STRING_COMPRESSION_EPS = 1e-12

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

# The pivot is swept too, between just clearing the beam and a metre and a half above it.
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


def _parameter_grid(machine=MachineType.PULLEY, seed: int = 42, draws: int = None):
    """Yield (design, reference result, fast result) over a random sweep of the bounds.

    `design` is the drawn design variables plus the pivot height they were stood on, which
    is what the assertions below quote on failure - the pivot is drawn now, so the design
    variables alone would not name the machine that failed.

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
        reach = max(values["arm_length"], values.get("length_counterweight", 0.0))
        pivot = reach + _PIVOT_CLEARANCE + rng.uniform(0.0, _PIVOT_SPREAD)
        params = _reference_params(values, machine, pivot_height=pivot)
        if params.string_length > 0.95 * params.arm_length:
            continue
        ref = simulate_trebuchet(params, rtol=1e-6, dense_output=False)
        yield ({**values, "pivot_height": pivot}, ref,
               _simulate_fast(values, machine, pivot_height=pivot))


def test_fast_engine_matches_scipy_engine_for_default_params():
    ref = simulate_trebuchet(TrebuchetParams(**DEFAULT_OPTIMIZABLE_PARAMS), rtol=1e-6, dense_output=False)
    assert ref.metrics["string_slack_fraction"] == 0.0  # defaults stay taut: models coincide

    (released, distance, efficiency, string_impulse, cw_impulse,
     sling_deficit, snap_energy, ground_energy) = _simulate_fast(DEFAULT_OPTIMIZABLE_PARAMS)

    assert released is True
    assert distance == pytest.approx(ref.distance, rel=1e-3)
    assert efficiency == pytest.approx(ref.efficiency, rel=1e-3)
    # The sling never let go, so there is nothing to compress and nothing to snap.
    assert string_impulse <= _STRING_COMPRESSION_EPS
    assert snap_energy == 0.0
    assert ref.metrics["sling_snap_energy"] == 0.0
    # The defaults keep real tension margin, so they owe nothing on the snap penalty
    # either - and both engines have to agree on that, not just on the distance.
    assert sling_deficit == 0.0
    assert ref.metrics["sling_tension_deficit"] == 0.0
    # The cw impulse comes from trapezoid sums over each engine's own accepted steps,
    # so it agrees only to the ~0.05 N*s noise floor (see the parameter-grid test).
    assert cw_impulse == pytest.approx(ref.metrics["cw_rope_compression_impulse"], rel=3e-1, abs=5e-2)


def test_fast_engine_matches_scipy_engine_for_the_traditional_default_machine():
    """The traditional linkage's own default geometry, end to end through both engines."""
    machine = MachineType.TRADITIONAL
    values = DEFAULT_MACHINE_PARAMS[machine]
    ref = simulate_trebuchet(_reference_params(values, machine), rtol=1e-6, dense_output=False)
    assert ref.metrics["string_slack_fraction"] == 0.0

    (released, distance, efficiency, string_impulse, cw_impulse,
     sling_deficit, snap_energy, ground_energy) = _simulate_fast(values, machine)

    assert released is True
    assert distance == pytest.approx(ref.distance, rel=1e-3)
    assert efficiency == pytest.approx(ref.efficiency, rel=1e-3)
    assert string_impulse <= _STRING_COMPRESSION_EPS
    assert snap_energy == 0.0
    assert sling_deficit == 0.0
    assert ref.metrics["sling_tension_deficit"] == 0.0
    # A pinned link is rigid by construction, so there is no rope tension to report and
    # the counterweight impulse is identically zero rather than merely small.
    assert cw_impulse == 0.0


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

    for design, ref, fast in _parameter_grid(machine):
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
         sling_deficit, snap_energy, ground_energy) = fast

        assert released == ref_released, design
        # A rope cannot push in either engine now.
        assert string_impulse <= _STRING_COMPRESSION_EPS, design

        if _undecided(ref.metrics):
            # Nothing below is a shared answer for this draw - see _undecided. Whether it
            # threw at all still is, and is asserted above.
            undecided_cases += 1
            continue
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
            # by a number the reference engine would score differently. Same trapezoid
            # caveat as the counterweight impulse below - the integrand kinks where the
            # tension crosses the floor, so two adaptive step grids sum it slightly
            # differently. Most designs sit at exactly zero on both engines, which is the
            # agreement the penalty most needs; the 1e-2 absolute floor covers the
            # boundary band where one engine reads a sliver of slack before a release the
            # other takes straight out of the taut regime.
            assert sling_deficit == pytest.approx(
                ref.metrics["sling_tension_deficit"], rel=3e-1, abs=1e-2
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
            assert efficiency == pytest.approx(ref.efficiency, abs=tolerance["eff"]), design
            if machine is MachineType.PULLEY:
                # max(0, -T) has kinks at the tension zero-crossings, so this is a
                # trapezoid sum that depends on where each engine's steps happen to fall.
                # That used to be the loosest bound in the file - 30% and a 0.25 N*s floor
                # - because the two engines opened every segment on different grids: this
                # one started each at a fixed 1e-3 while scipy sized its first step from
                # the derivatives, and over a short segment that difference is the whole
                # sample. Both now pick the same opening step (fastsim._initial_step), and
                # all eleven draws carrying any impulse at all agree to better than 0.7%,
                # from 0.038 N*s to 66.9 - closer than the reference engine comes to
                # itself, which reads 0.635 / 0.571 / 0.605 N*s on the draw that motivated
                # this at rtol 1e-6 / 1e-8 / 1e-12. Agreement is not convergence: the two
                # now sample the same spike the same way. No draw in this sweep needs the
                # absolute floor - every one of the eleven is carried by the relative bound
                # - so it is there for the boundary case where one engine finds a sliver of
                # compression and the other finds none.
                assert cw_impulse == pytest.approx(
                    ref.metrics["cw_rope_compression_impulse"], rel=3e-2, abs=1e-3
                ), design
            else:
                assert cw_impulse == 0.0, design  # no counterweight rope to go slack

    # Sanity-check the grid actually exercised every arm of the launch.
    assert quiet_cases > 5
    assert eventful_cases > 20
    assert grounded_cases > 10
    # And that the escape hatch stayed an escape hatch: one draw in 196 over both machines.
    assert undecided_cases <= 2


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

    for design, ref, fast in _parameter_grid(machine):
        if _exhausted(ref):
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

    # Tall enough to swing, then short enough that it cannot.
    clears = _reference_params(values, machine, pivot_height=arm + linkage + 0.5)
    digs = _reference_params(values, machine, pivot_height=arm / 2.0)

    ref_clears = simulate_trebuchet(clears, rtol=1e-6, dense_output=False)
    ref_digs = simulate_trebuchet(digs, rtol=1e-6, dense_output=False)
    assert ref_clears.metrics["arm_ground_contact"] is False
    assert ref_digs.metrics["arm_ground_contact"] is True
    assert ref_digs.metrics["release_occurred"] is False
    assert ref_digs.distance == 0.0

    fast_clears = _simulate_fast(values, machine, pivot_height=arm + linkage + 0.5)
    fast_digs = _simulate_fast(values, machine, pivot_height=arm / 2.0)
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
     _snap_energy, _ground_energy) = _simulate_fast(values, joint_friction_coefficient=huge_friction)

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
        fixed["joint_friction_coefficient"], has_pulley,
        30.0, 5.0, 1.0, 0.15, 200.0, 300.0,
    )

    for i in range(s):
        expected = fastsim._score(
            pop["counter_weight_mass"][i], pop["pulley_radius"][i], pop["length_counterweight"][i],
            fixed["counter_weight_rope_length"],
            pop["arm_length"][i], pop["string_length"][i], pop["release_angle"][i],
            fixed["pivot_height"], fixed["pulley_density"], fixed["arm_density"],
            fixed["projectile_mass"], fixed["projectile_radius"], fixed["initial_arm_angle"],
            fixed["arm_drag_coefficient"], fixed["projectile_drag_coefficient"],
            fixed["joint_friction_coefficient"], has_pulley,
            30.0, 5.0, 1.0, 0.15, 200.0, 300.0,
        )
        assert costs[i] == pytest.approx(expected)
