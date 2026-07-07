import pytest

from trebuchet_sim.optimization import PARAM_NAMES, OptimizationConfig, _objective, optimize_trebuchet

pytest.importorskip("numba")


def _quick_config(**overrides) -> OptimizationConfig:
    # Small population/iteration budget: this test checks wiring and result shape,
    # not convergence quality (see test_fastsim.py for numerical agreement checks).
    defaults = dict(population_size=6, max_iterations=15)
    defaults.update(overrides)
    return OptimizationConfig(**defaults)


def test_fast_engine_optimize_returns_valid_result():
    params, result, de_result = optimize_trebuchet(_quick_config())

    assert result.distance > 0
    assert 0 < result.efficiency < 1
    assert de_result.nfev > 0


def test_scipy_fallback_engine_still_works():
    params, result, de_result = optimize_trebuchet(_quick_config(use_fast_engine=False, workers=1))

    assert result.distance > 0
    assert 0 < result.efficiency < 1


def test_objective_penalizes_slack_sling_solutions():
    # A jerky parameter set (the pre-slack-penalty optimizer defaults) holds the sling
    # in compression, so the slack penalty must raise its cost by weight * impulse.
    jerky = {
        "counter_weight_mass": 16.865,
        "pulley_radius": 0.121,
        "arm_length": 0.813,
        "string_length": 0.669,
        "release_angle": -4.877,
    }
    free_values = [jerky[name] for name in PARAM_NAMES]

    base = _objective(free_values, OptimizationConfig(slack_penalty_weight=0.0))
    penalized = _objective(free_values, OptimizationConfig(slack_penalty_weight=200.0))

    assert penalized > base + 100.0  # this set has ~1.2 N*s of compression impulse


def test_locked_params_are_respected_by_fast_engine():
    locked_mass = 15.0
    config = _quick_config(locked_params={"counter_weight_mass": locked_mass})

    params, result, de_result = optimize_trebuchet(config)

    assert params.counter_weight_mass == locked_mass
