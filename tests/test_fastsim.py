import numpy as np
import pytest

numba = pytest.importorskip("numba")

from trebuchet_sim import fastsim
from trebuchet_sim.config import DEFAULT_OPTIMIZABLE_PARAMS, TrebuchetParams
from trebuchet_sim.optimization import PARAM_BOUNDS, PARAM_NAMES
from trebuchet_sim.physics import simulate_trebuchet

FIXED = {
    "pivot_height": TrebuchetParams.pivot_height,
    "pulley_density": TrebuchetParams.pulley_density,
    "arm_density": TrebuchetParams.arm_density,
    "projectile_mass": TrebuchetParams.projectile_mass,
    "projectile_radius": TrebuchetParams.projectile_radius,
    "initial_arm_angle": float(TrebuchetParams.initial_arm_angle),
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


def test_fast_engine_matches_scipy_engine_for_default_params():
    ref = simulate_trebuchet(TrebuchetParams(**DEFAULT_OPTIMIZABLE_PARAMS), rtol=1e-6, dense_output=False)

    released, distance, efficiency = _simulate_fast(DEFAULT_OPTIMIZABLE_PARAMS)

    assert released is True
    assert distance == pytest.approx(ref.distance, rel=1e-3)
    assert efficiency == pytest.approx(ref.efficiency, rel=1e-3)


def test_fast_engine_matches_scipy_engine_across_random_parameter_grid():
    rng = np.random.default_rng(42)
    tested = 0

    for _ in range(100):
        values = {name: rng.uniform(*PARAM_BOUNDS[name]) for name in PARAM_NAMES}
        params = TrebuchetParams(**values)
        if params.string_length > 0.95 * params.arm_length:
            continue

        ref = simulate_trebuchet(params, rtol=1e-6, dense_output=False)
        ref_released = ref.metrics.get("release_occurred", False)
        released, distance, efficiency = _simulate_fast(values)

        assert released == ref_released, values

        if ref_released:
            tested += 1
            assert distance == pytest.approx(ref.distance, rel=1e-2), values
            assert efficiency == pytest.approx(ref.efficiency, rel=1e-2), values

    assert tested > 50  # sanity check the grid actually exercised the release path


def test_fast_engine_reports_no_release_for_geometry_that_never_releases():
    # Enough joint friction that the arm never reaches the release angle within t_max=10s.
    values = dict(DEFAULT_OPTIMIZABLE_PARAMS)
    huge_friction = 50.0

    ref = simulate_trebuchet(
        TrebuchetParams(**values, joint_friction_coefficient=huge_friction), rtol=1e-6, dense_output=False
    )
    assert ref.metrics.get("release_occurred") is False  # sanity-check the fixture against the reference engine

    released, distance, efficiency = fastsim.simulate_fast(
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
        30.0, 5.0, 1.0, 0.15,
    )

    for i in range(s):
        expected = fastsim._score(
            pop["counter_weight_mass"][i], pop["pulley_radius"][i], pop["arm_length"][i],
            pop["string_length"][i], pop["release_angle"][i],
            FIXED["pivot_height"], FIXED["pulley_density"], FIXED["arm_density"],
            FIXED["projectile_mass"], FIXED["projectile_radius"], FIXED["initial_arm_angle"],
            FIXED["arm_drag_coefficient"], FIXED["projectile_drag_coefficient"], FIXED["joint_friction_coefficient"],
            30.0, 5.0, 1.0, 0.15,
        )
        assert costs[i] == pytest.approx(expected)
