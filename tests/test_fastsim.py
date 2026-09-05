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


# The two engines model the sling differently on purpose (see fastsim's module
# docstring): this one keeps the rigid link and reports the compression impulse the
# optimizer penalizes, while physics.py lets the sling go slack and snap back. They
# are therefore only required to agree on always-taut launches - which is exactly the
# regime the slack penalty drives the search into. `string_slack_fraction == 0` in the
# reference result is what marks a launch as staying in that shared regime.


# Below this the string impulse is not evidence of anything: it is the same 0.05 N*s
# floor the optimizer treats as noise. The two engines part company at the instant the
# rigid-link tension first crosses zero, and when that crossing falls in the last
# moments before release the reference can release straight out of the taut regime -
# reporting no slack at all - while this engine still integrates a sliver of negative
# tension. One draw in 131 lands in that band on the traditional machine (measuring
# 0.007 N*s); genuinely slack launches measure upwards of 0.014.
_IMPULSE_NOISE_FLOOR = 0.05

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

    released, distance, efficiency, string_impulse, cw_impulse = _simulate_fast(DEFAULT_OPTIMIZABLE_PARAMS)

    assert released is True
    assert distance == pytest.approx(ref.distance, rel=1e-3)
    assert efficiency == pytest.approx(ref.efficiency, rel=1e-3)
    # Rigid-link tension never went negative either, so there is no compression to report.
    assert string_impulse == 0.0
    # The cw impulse comes from trapezoid sums over each engine's own accepted steps,
    # so it agrees only to the ~0.05 N*s noise floor (see the parameter-grid test).
    assert cw_impulse == pytest.approx(ref.metrics["cw_rope_compression_impulse"], rel=3e-1, abs=5e-2)


def test_fast_engine_matches_scipy_engine_for_the_traditional_default_machine():
    """The traditional linkage's own default geometry, end to end through both engines."""
    machine = MachineType.TRADITIONAL
    values = DEFAULT_MACHINE_PARAMS[machine]
    ref = simulate_trebuchet(_reference_params(values, machine), rtol=1e-6, dense_output=False)
    assert ref.metrics["string_slack_fraction"] == 0.0

    released, distance, efficiency, string_impulse, cw_impulse = _simulate_fast(values, machine)

    assert released is True
    assert distance == pytest.approx(ref.distance, rel=1e-3)
    assert efficiency == pytest.approx(ref.efficiency, rel=1e-3)
    assert string_impulse == 0.0
    # A pinned link is rigid by construction, so there is no rope tension to report and
    # the counterweight impulse is identically zero rather than merely small.
    assert cw_impulse == 0.0


@pytest.mark.parametrize("machine", list(MachineType))
def test_fast_engine_matches_scipy_engine_on_always_taut_launches(machine):
    """Where both engines model the same physics, they must agree closely."""
    taut_cases = 0

    for values, ref, fast in _parameter_grid(machine):
        if ref.metrics["string_slack_fraction"] != 0.0:
            continue  # covered by the divergence test below
        taut_cases += 1

        ref_released = ref.metrics.get("release_occurred", False)
        released, distance, efficiency, string_impulse, cw_impulse = fast

        assert released == ref_released, values
        # No slack in the reference means the rigid model barely had to push either.
        assert string_impulse < _IMPULSE_NOISE_FLOOR, values

        if ref_released:
            # Same equations of motion, two integrators: agreement is limited only by
            # step-size control. Across this grid it measures a median 6e-6 relative on
            # the pulley machine and 2e-5 on the traditional one, worst case 3e-4 for any
            # throw of a metre or more - well inside the ~2e-4 the optimizer already
            # accepts by running at rtol=1e-6 (see optimization._objective). The
            # absolute floors cover the releases that go nowhere - the projectile leaves
            # aimed at the ground and lands a few millimetres away, or not at all - where
            # a relative tolerance is asking for agreement on a number that isn't a
            # throw. A millimetre is far below any range worth optimizing for, and real
            # throws are still held to the relative bound.
            assert distance == pytest.approx(ref.distance, rel=1e-3, abs=1e-3), values
            assert efficiency == pytest.approx(ref.efficiency, rel=1e-3, abs=1e-9), values
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

    assert taut_cases > 20  # sanity check the grid actually exercised the shared regime


@pytest.mark.parametrize("machine", list(MachineType))
def test_fast_engine_compression_impulse_predicts_where_the_sling_goes_slack(machine):
    """The rigid engine's string impulse is the signal that flags the divergence.

    Both engines integrate identical dynamics up to the instant the rigid-link string
    tension first crosses zero. physics.py switches to a slack regime there; fastsim
    carries on and accumulates that negative tension into `string_impulse`. So the two
    have to agree about whether a launch left the shared regime, even though everything
    downstream of that instant - distance, efficiency, and even whether a release happens
    at all - is then free to diverge. Only the boundary band is exempt, and only in one
    direction: see _IMPULSE_NOISE_FLOOR.
    """
    slack_cases = 0

    for values, ref, fast in _parameter_grid(machine):
        string_impulse = fast[3]
        went_slack = ref.metrics["string_slack_fraction"] > 0

        if went_slack:
            slack_cases += 1
            assert string_impulse > 0, values
        else:
            assert string_impulse < _IMPULSE_NOISE_FLOOR, values

    assert slack_cases > 20  # sanity check the grid actually exercised the slack regime


def test_fast_engine_reports_no_release_for_geometry_that_never_releases():
    # Enough joint friction that the arm never reaches the release angle within t_max=10s.
    values = dict(DEFAULT_OPTIMIZABLE_PARAMS)
    huge_friction = 50.0

    ref = simulate_trebuchet(
        _reference_params(values, joint_friction_coefficient=huge_friction), rtol=1e-6, dense_output=False
    )
    assert ref.metrics.get("release_occurred") is False  # sanity-check the fixture against the reference engine

    released, distance, efficiency, _string_impulse, _cw_impulse = _simulate_fast(
        values, joint_friction_coefficient=huge_friction
    )

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
        30.0, 5.0, 1.0, 0.15, 200.0,
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
            30.0, 5.0, 1.0, 0.15, 200.0,
        )
        assert costs[i] == pytest.approx(expected)
