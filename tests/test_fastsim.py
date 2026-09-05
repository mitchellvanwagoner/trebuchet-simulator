import numpy as np
import pytest

numba = pytest.importorskip("numba")

from trebuchet_sim import fastsim
from trebuchet_sim.config import (
    DEFAULT_INITIAL_ARM_ANGLE,
    DEFAULT_OPTIMIZABLE_PARAMS,
    MachineType,
    TrebuchetParams,
)
from trebuchet_sim.optimization import PARAM_BOUNDS, PARAM_NAMES
from trebuchet_sim.physics import simulate_trebuchet

FIXED = {
    "pivot_height": TrebuchetParams.pivot_height,
    "pulley_density": TrebuchetParams.pulley_density,
    "arm_density": TrebuchetParams.arm_density,
    "projectile_mass": TrebuchetParams.projectile_mass,
    "projectile_radius": TrebuchetParams.projectile_radius,
    # Read from the machine's start-angle table rather than the dataclass default,
    # which is None until __post_init__ resolves it per machine.
    "initial_arm_angle": float(DEFAULT_INITIAL_ARM_ANGLE[MachineType.PULLEY]),
    "arm_drag_coefficient": TrebuchetParams.arm_drag_coefficient,
    "projectile_drag_coefficient": TrebuchetParams.projectile_drag_coefficient,
    "joint_friction_coefficient": TrebuchetParams.joint_friction_coefficient,
}


def _simulate_fast(values):
    return fastsim.simulate_fast(
        values["counter_weight_mass"], values["pulley_radius"], values["arm_length"],
        values["string_length"], values["release_angle"],
        FIXED["pivot_height"], FIXED["pulley_density"], FIXED["arm_density"],
        FIXED["projectile_mass"], FIXED["projectile_radius"], FIXED["initial_arm_angle"],
        FIXED["arm_drag_coefficient"], FIXED["projectile_drag_coefficient"], FIXED["joint_friction_coefficient"],
    )


# The two engines model the sling differently on purpose (see fastsim's module
# docstring): this one keeps the rigid link and reports the compression impulse the
# optimizer penalizes, while physics.py lets the sling go slack and snap back. They
# are therefore only required to agree on always-taut launches - which is exactly the
# regime the slack penalty drives the search into. `string_slack_fraction == 0` in the
# reference result is what marks a launch as staying in that shared regime.


def _parameter_grid(seed: int = 42, draws: int = 100):
    """Yield (values, reference result, fast result) over a random sweep of the bounds.

    Geometries whose string nearly equals the arm are skipped: they are outside the
    region the optimizer searches and integrate poorly in both engines.
    """
    rng = np.random.default_rng(seed)
    for _ in range(draws):
        values = {name: rng.uniform(*PARAM_BOUNDS[name]) for name in PARAM_NAMES}
        params = TrebuchetParams(**values)
        if params.string_length > 0.95 * params.arm_length:
            continue
        ref = simulate_trebuchet(params, rtol=1e-6, dense_output=False)
        yield values, ref, _simulate_fast(values)


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


def test_fast_engine_matches_scipy_engine_on_always_taut_launches():
    """Where both engines model the same physics, they must agree closely."""
    taut_cases = 0

    for values, ref, fast in _parameter_grid():
        if ref.metrics["string_slack_fraction"] != 0.0:
            continue  # covered by the divergence test below
        taut_cases += 1

        ref_released = ref.metrics.get("release_occurred", False)
        released, distance, efficiency, string_impulse, cw_impulse = fast

        assert released == ref_released, values
        # No slack in the reference means the rigid model never needed to push either.
        assert string_impulse == 0.0, values

        if ref_released:
            # Same equations of motion, two integrators: agreement is limited only by
            # step-size control, which measures ~4e-5 relative across this grid. The
            # absolute floor covers the handful of releases that go nowhere (the
            # projectile leaves aimed at the ground and both engines report a range of
            # exactly 0.0), where a bare relative tolerance would demand bit equality.
            assert distance == pytest.approx(ref.distance, rel=1e-3, abs=1e-6), values
            assert efficiency == pytest.approx(ref.efficiency, rel=1e-3, abs=1e-9), values
            # max(0, -T) has kinks at the tension zero-crossings, so trapezoid sums
            # over the two engines' different step grids agree only loosely when the
            # impulse itself is small; genuinely jerky launches measure 1-7 N*s, so
            # sub-0.05 N*s disagreement is noise the penalty weight can't resolve.
            assert cw_impulse == pytest.approx(
                ref.metrics["cw_rope_compression_impulse"], rel=3e-1, abs=5e-2
            ), values

    assert taut_cases > 20  # sanity check the grid actually exercised the shared regime


def test_fast_engine_compression_impulse_predicts_where_the_sling_goes_slack():
    """The rigid engine's string impulse is the signal that flags the divergence.

    Both engines integrate identical dynamics up to the instant the rigid-link string
    tension first crosses zero. physics.py switches to a slack regime there; fastsim
    carries on and accumulates that negative tension into `string_impulse`. So a
    nonzero impulse and a nonzero slack fraction have to appear together, even though
    everything downstream of that instant - distance, efficiency, and even whether a
    release happens at all - is then free to diverge.
    """
    slack_cases = 0

    for values, ref, fast in _parameter_grid():
        string_impulse = fast[3]
        went_slack = ref.metrics["string_slack_fraction"] > 0

        assert (string_impulse > 0) == went_slack, values
        if went_slack:
            slack_cases += 1

    assert slack_cases > 20  # sanity check the grid actually exercised the slack regime


def test_fast_engine_reports_no_release_for_geometry_that_never_releases():
    # Enough joint friction that the arm never reaches the release angle within t_max=10s.
    values = dict(DEFAULT_OPTIMIZABLE_PARAMS)
    huge_friction = 50.0

    ref = simulate_trebuchet(
        TrebuchetParams(**values, joint_friction_coefficient=huge_friction), rtol=1e-6, dense_output=False
    )
    assert ref.metrics.get("release_occurred") is False  # sanity-check the fixture against the reference engine

    released, distance, efficiency, _string_impulse, _cw_impulse = fastsim.simulate_fast(
        values["counter_weight_mass"], values["pulley_radius"], values["arm_length"],
        values["string_length"], values["release_angle"],
        FIXED["pivot_height"], FIXED["pulley_density"], FIXED["arm_density"],
        FIXED["projectile_mass"], FIXED["projectile_radius"], FIXED["initial_arm_angle"],
        FIXED["arm_drag_coefficient"], FIXED["projectile_drag_coefficient"], huge_friction,
    )

    assert released is False
    assert distance == 0.0
    assert efficiency == 0.0


def test_evaluate_population_matches_per_individual_score():
    rng = np.random.default_rng(7)
    s = 16
    pop = {name: rng.uniform(*PARAM_BOUNDS[name], size=s) for name in PARAM_NAMES}

    costs = fastsim.evaluate_population(
        pop["counter_weight_mass"], pop["pulley_radius"], pop["arm_length"], pop["string_length"],
        pop["release_angle"], FIXED["pivot_height"], FIXED["pulley_density"], FIXED["arm_density"],
        FIXED["projectile_mass"], FIXED["projectile_radius"], FIXED["initial_arm_angle"],
        FIXED["arm_drag_coefficient"], FIXED["projectile_drag_coefficient"], FIXED["joint_friction_coefficient"],
        30.0, 5.0, 1.0, 0.15, 200.0,
    )

    for i in range(s):
        expected = fastsim._score(
            pop["counter_weight_mass"][i], pop["pulley_radius"][i], pop["arm_length"][i],
            pop["string_length"][i], pop["release_angle"][i],
            FIXED["pivot_height"], FIXED["pulley_density"], FIXED["arm_density"],
            FIXED["projectile_mass"], FIXED["projectile_radius"], FIXED["initial_arm_angle"],
            FIXED["arm_drag_coefficient"], FIXED["projectile_drag_coefficient"], FIXED["joint_friction_coefficient"],
            30.0, 5.0, 1.0, 0.15, 200.0,
        )
        assert costs[i] == pytest.approx(expected)
