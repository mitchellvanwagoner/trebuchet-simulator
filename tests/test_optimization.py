import pytest

from trebuchet_sim.optimization import OptimizationConfig, optimize_trebuchet

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


def test_locked_params_are_respected_by_fast_engine():
    locked_mass = 15.0
    config = _quick_config(locked_params={"counter_weight_mass": locked_mass})

    params, result, de_result = optimize_trebuchet(config)

    assert params.counter_weight_mass == locked_mass
