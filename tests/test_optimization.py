import pytest

from trebuchet_sim.config import MachineType
from trebuchet_sim.optimization import (
    PARAM_BOUNDS,
    PARAM_LIMITS,
    PARAM_NAMES,
    OptimizationConfig,
    _objective,
    optimize_trebuchet,
)

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


def test_default_bounds_lie_inside_the_hard_limits():
    """PARAM_BOUNDS is where the search starts; PARAM_LIMITS is how far it may be moved.

    A default outside its own limit would make that parameter impossible to range
    without also widening it, and would reject a config that only restated the default.
    """
    for name, (low, high) in PARAM_BOUNDS.items():
        limit_low, limit_high = PARAM_LIMITS[name]
        assert limit_low <= low < high <= limit_high, name


def test_param_bounds_override_the_default_search_range():
    config = OptimizationConfig(param_bounds={"arm_length": (0.3, 0.8)})

    assert config.bounds_for("arm_length") == (0.3, 0.8)
    assert config.bounds_for("string_length") == PARAM_BOUNDS["string_length"]
    # bounds follows free_params order, so the optimizer sees the override in place.
    assert config.bounds[config.free_params.index("arm_length")] == (0.3, 0.8)


@pytest.mark.parametrize(
    "bad, message",
    [
        ({"arm_length": (0.8, 0.3)}, "min < max"),
        ({"arm_length": (0.0, 0.8)}, "PARAM_LIMITS"),
        ({"arm_length": (0.3, 99.0)}, "PARAM_LIMITS"),
        ({"arm_length": (float("nan"), 0.8)}, "finite"),
        ({"not_a_param": (0.3, 0.8)}, "No such parameter"),
        ({"length_counterweight": (0.1, 0.5)}, "No such parameter"),  # wrong machine
    ],
)
def test_invalid_ranges_are_rejected(bad, message):
    with pytest.raises(ValueError, match=message):
        OptimizationConfig(param_bounds=bad)


def test_traditional_machine_ranges_its_own_linkage_parameter():
    config = OptimizationConfig(
        machine=MachineType.TRADITIONAL, param_bounds={"length_counterweight": (0.1, 0.5)}
    )

    assert config.bounds_for("length_counterweight") == (0.1, 0.5)
    with pytest.raises(ValueError, match="No such parameter"):
        OptimizationConfig(machine=MachineType.TRADITIONAL, param_bounds={"pulley_radius": (0.1, 0.5)})


def test_optimizer_searches_only_inside_a_narrowed_range():
    """The whole point: a narrowed range has to actually confine the result.

    The unconstrained optimum for this target puts the arm near 0.42 m, well below the
    window here, so a range that was accepted but never applied would show up as an arm
    length outside it.
    """
    low, high = 0.9, 1.4
    config = _quick_config(param_bounds={"arm_length": (low, high)})

    params, result, _de_result = optimize_trebuchet(config)

    assert low <= params.arm_length <= high
    assert result.distance > 0


def test_a_locked_parameter_ignores_its_range():
    """Locking wins: the parameter is pinned, so its range is simply unused.

    Accepted rather than rejected, because the dashboard carries a range for every
    parameter whether or not it happens to be locked.
    """
    config = _quick_config(
        locked_params={"arm_length": 0.45}, param_bounds={"arm_length": (0.9, 1.4)}
    )

    assert "arm_length" not in config.free_params
    assert all(name != "arm_length" for name in config.free_params)

    params, _result, _de_result = optimize_trebuchet(config)

    assert params.arm_length == 0.45
