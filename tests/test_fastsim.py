import numpy as np
import pytest

numba = pytest.importorskip("numba")

from trebuchet_sim import fastsim
from trebuchet_sim.config import (
    DEFAULT_INITIAL_ARM_ANGLE,
    DEFAULT_MACHINE_FIXED,
    DEFAULT_MACHINE_PARAMS,
    DEFAULT_OPTIMIZABLE_PARAMS,
    MachineType,
    TrebuchetParams,
)
from trebuchet_sim.optimization import PARAM_BOUNDS, param_names
from trebuchet_sim.physics import simulate_trebuchet

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


# Both engines model the sling as a rope, so they are held to the same answer on every
# launch, not merely on the ones that stay taut. A launch that lets go is still the
# harder case: it turns on two discontinuities - the tension zero-crossing and the
# re-tension snap - that each engine's own event solver localizes separately, so a hair
# of difference in *when* a snap lands moves the state it lands on. Measured over these
# grids that costs about a decimal place against the taut case (a median 0.008%
# relative on distance and a worst case of 0.35%, against 0.002% and 0.03% taut), which
# is what the two bounds below are.
_TAUT_TOLERANCE = dict(rel=1e-3, abs=1e-3)
_SLACK_TOLERANCE = dict(rel=1e-2, abs=1e-2)

# A rope carries no compression, so neither engine should be reporting any on the sling.
# This is a self-check on the port rather than a tolerance: it measures 0 exactly on most
# draws and never worse than machine epsilon on the rest.
_STRING_COMPRESSION_EPS = 1e-12

# Below this a snap is not evidence that a launch went slack, only that its sling grazed
# zero tension somewhere. Whether a graze counts as a detachment is decided by two event
# solvers independently, so right at the boundary they can differ - one draw in these 188
# does, on the traditional machine, where the reference bottoms out at 0.04 N of tension
# and reports a clean launch while this engine reads a 0.001 J snap. It costs nothing in
# the answer: the two still land 1 mm apart on a 60.6 m throw. Real snaps on these grids
# start at 0.006 J and run to 70 J.
_SNAP_ENERGY_NOISE_FLOOR = 0.01

# A random traditional machine keeps its sling taut far more often than a random pulley
# one - about 86% of usable draws against 49% - so it needs a bigger sweep to put a
# comparable number of launches through the slack regime. Both are cheap (well under a
# second); the counts below are what these seeds actually produce.
_GRID_DRAWS = {MachineType.PULLEY: 100, MachineType.TRADITIONAL: 300}


def _parameter_grid(machine=MachineType.PULLEY, seed: int = 42, draws: int = None):
    """Yield (values, reference result, fast result) over a random sweep of the bounds.

    Draws this machine's own design variables, so the linkage slot holds whichever
    parameter it actually uses. Geometries whose string nearly equals the arm are
    skipped: they are outside the region the optimizer searches and integrate poorly in
    both engines.
    """
    draws = _GRID_DRAWS[machine] if draws is None else draws
    rng = np.random.default_rng(seed)
    for _ in range(draws):
        values = {name: rng.uniform(*PARAM_BOUNDS[name]) for name in param_names(machine)}
        params = _reference_params(values, machine)
        if params.string_length > 0.95 * params.arm_length:
            continue
        ref = simulate_trebuchet(params, rtol=1e-6, dense_output=False)
        yield values, ref, _simulate_fast(values, machine)


def test_fast_engine_matches_scipy_engine_for_default_params():
    ref = simulate_trebuchet(TrebuchetParams(**DEFAULT_OPTIMIZABLE_PARAMS), rtol=1e-6, dense_output=False)
    assert ref.metrics["string_slack_fraction"] == 0.0  # defaults stay taut: models coincide

    (released, distance, efficiency, string_impulse, cw_impulse,
     sling_deficit, snap_energy) = _simulate_fast(DEFAULT_OPTIMIZABLE_PARAMS)

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
     sling_deficit, snap_energy) = _simulate_fast(values, machine)

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

    Both arms of the launch are covered: the ones whose sling stays taut throughout and
    the ones that let go, detach and snap back. The engines used to be held to the taut
    arm alone, because this one kept a rigid sling and had no slack physics to compare;
    a slack launch then measured a median 56% apart on distance and as much as 170%.
    """
    taut_cases = 0
    slack_cases = 0

    for values, ref, fast in _parameter_grid(machine):
        went_slack = ref.metrics["string_slack_fraction"] > 0.0
        if went_slack:
            slack_cases += 1
        else:
            taut_cases += 1
        tolerance = _SLACK_TOLERANCE if went_slack else _TAUT_TOLERANCE

        ref_released = ref.metrics.get("release_occurred", False)
        (released, distance, efficiency, string_impulse, cw_impulse,
         sling_deficit, snap_energy) = fast

        assert released == ref_released, values
        # A rope cannot push in either engine now.
        assert string_impulse <= _STRING_COMPRESSION_EPS, values
        # Whether the sling let go at all is itself a shared answer: this engine used to
        # be the one that could not tell, and a launch it thought was fine is exactly
        # where it used to invent a throw. Only a graze is exempt, and only up to the
        # noise floor.
        ref_snap = ref.metrics["sling_snap_energy"]
        if (snap_energy > 0.0) != (ref_snap > 0.0):
            assert max(snap_energy, ref_snap) < _SNAP_ENERGY_NOISE_FLOOR, values
        assert snap_energy == pytest.approx(ref_snap, rel=5e-2, abs=_SNAP_ENERGY_NOISE_FLOOR), values
        # The snap penalty's input is a shared quantity, not a fast-engine invention:
        # in the regime where the two engines model the same physics they must also
        # agree about how close to slack the sling ran, or the search would be steered
        # by a number the reference engine would score differently. Same trapezoid
        # caveat as the counterweight impulse below - the integrand kinks where the
        # tension crosses the floor, so two adaptive step grids sum it slightly
        # differently: a median 1.3% relative across this sweep on the pulley machine
        # and 0.2% on the traditional one. Most designs sit at exactly zero on both
        # engines (96 of the 106 taut traditional draws), which is the agreement the
        # penalty most needs. The 1e-2 absolute floor covers the same boundary band
        # _IMPULSE_NOISE_FLOOR does - one traditional draw releases straight out of the
        # taut regime here while the fast engine reads a sliver of slack first, and
        # they land 9.2e-3 apart on a launch that is 1-2% marginal either way.
        assert sling_deficit == pytest.approx(
            ref.metrics["sling_tension_deficit"], rel=3e-1, abs=1e-2
        ), values

        if ref_released:
            # Same equations of motion, two integrators: agreement is limited only by
            # step-size control, and on a slack launch by where each engine's event
            # solver puts the snap. Across this grid a taut launch measures a median 6e-6
            # relative on the pulley machine and 2e-5 on the traditional one; a slack one
            # a median 8e-5, worst case 3.5e-3 - both inside the ~2e-4 to 1e-2 the
            # optimizer already accepts by running at rtol=1e-6 (see
            # optimization._objective). The absolute floors cover the releases that go
            # nowhere - the projectile leaves aimed at the ground and lands a few
            # millimetres away, or not at all - where a relative tolerance is asking for
            # agreement on a number that isn't a throw. A millimetre is far below any
            # range worth optimizing for, and real throws are still held to the relative
            # bound.
            assert distance == pytest.approx(ref.distance, **tolerance), values
            assert efficiency == pytest.approx(
                ref.efficiency, rel=tolerance["rel"], abs=1e-9
            ), values
            if machine is MachineType.PULLEY:
                # max(0, -T) has kinks at the tension zero-crossings, so trapezoid sums
                # over the two engines' different step grids agree only loosely when the
                # impulse itself is small; genuinely jerky launches measure 1-7 N*s, so
                # sub-0.05 N*s disagreement is noise the penalty weight can't resolve.
                assert cw_impulse == pytest.approx(
                    ref.metrics["cw_rope_compression_impulse"], rel=3e-1, abs=5e-2
                ), values
            else:
                assert cw_impulse == 0.0, values  # no counterweight rope to go slack

    # Sanity-check the grid actually exercised both arms of the launch.
    assert taut_cases > 20
    assert slack_cases > 20


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

    for values, ref, fast in _parameter_grid(machine):
        ref_energy = ref.metrics["sling_snap_energy"]
        if ref_energy <= 0.0:
            # A launch the reference calls clean must cost this engine nothing either,
            # bar a graze at the boundary (see _SNAP_ENERGY_NOISE_FLOOR).
            assert fast[6] < _SNAP_ENERGY_NOISE_FLOOR, values
            continue
        snapping_cases += 1
        # A snap only ever removes energy, in both engines (see physics._apply_snap).
        assert fast[6] > 0.0, values
        assert fast[6] == pytest.approx(ref_energy, rel=5e-2, abs=_SNAP_ENERGY_NOISE_FLOOR), values

    assert snapping_cases > 10


def test_fast_engine_reports_no_release_for_geometry_that_never_releases():
    # Enough joint friction that the arm never reaches the release angle within t_max=10s.
    values = dict(DEFAULT_OPTIMIZABLE_PARAMS)
    huge_friction = 50.0

    ref = simulate_trebuchet(
        _reference_params(values, joint_friction_coefficient=huge_friction), rtol=1e-6, dense_output=False
    )
    assert ref.metrics.get("release_occurred") is False  # sanity-check the fixture against the reference engine

    (released, distance, efficiency, _string_impulse, _cw_impulse, _deficit,
     _snap_energy) = _simulate_fast(values, joint_friction_coefficient=huge_friction)

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
