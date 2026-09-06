import pytest

import numpy as np

from trebuchet_sim.config import MachineType, TrebuchetParams
from trebuchet_sim.optimization import (
    PARAM_BOUNDS,
    PARAM_LIMITS,
    PARAM_NAMES,
    OptimizationConfig,
    _objective,
    _objective_vectorized,
    optimize_trebuchet,
)
from trebuchet_sim.physics import simulate_trebuchet

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


# Copied from tests/test_physics.py: a sling that never detaches but runs on almost no
# load. Every signal the objective had before this one - slack fraction, snap count,
# snap energy, both compression impulses - reads exactly zero for it.
MARGINAL_SLING_PARAMS = {
    "counter_weight_mass": 21.396,
    "pulley_radius": 0.0229,
    "arm_length": 0.9,
    "string_length": 0.416,
    "release_angle": -4.064,
}


def test_objective_penalizes_a_jerky_sling_that_never_actually_goes_slack():
    """The gap the snap penalty fills: a design one nudge away from snapping.

    The slack penalty cannot price this. It is paid on compression impulse, which is
    zero until the rope has already let go, so between two designs that both hold
    together it is the same number - and a search whose objective is flat across the
    approach to a cliff will happily park on the edge of it.
    """
    free_values = [MARGINAL_SLING_PARAMS[name] for name in PARAM_NAMES]
    metrics = simulate_trebuchet(TrebuchetParams(**MARGINAL_SLING_PARAMS)).metrics
    assert metrics["cw_rope_compression_impulse"] == 0.0  # nothing for the slack penalty to charge
    assert metrics["sling_snap_count"] == 0

    unpenalized = _objective(free_values, OptimizationConfig(snap_penalty_weight=0.0))
    penalized = _objective(free_values, OptimizationConfig(snap_penalty_weight=300.0))

    # The charge is the deficit at the objective's own rtol=1e-6, which sums the
    # trapezoid over a slightly coarser step grid than the rtol=1e-8 metrics above.
    assert penalized - unpenalized == pytest.approx(300.0 * metrics["sling_tension_deficit"], rel=0.1)
    assert penalized > unpenalized + 25.0


def test_both_engines_charge_the_same_snap_penalty():
    """Whichever engine scores it, a jerky design has to cost about the same.

    They compute the deficit from their own step grids and, past a detachment, from
    different sling models entirely - so this holds where it matters, on a design that
    stays attached and that both engines therefore agree about.
    """
    free_values = [MARGINAL_SLING_PARAMS[name] for name in PARAM_NAMES]
    population = np.array([[value] for value in free_values])

    scipy_costs, fast_costs = [], []
    for weight in (0.0, 300.0):
        config = OptimizationConfig(snap_penalty_weight=weight)
        scipy_costs.append(_objective(free_values, config))
        fast_costs.append(float(_objective_vectorized(population, config)[0]))

    scipy_charge = scipy_costs[1] - scipy_costs[0]
    fast_charge = fast_costs[1] - fast_costs[0]
    assert scipy_charge > 25.0
    assert fast_charge == pytest.approx(scipy_charge, rel=0.1)


def test_snap_penalty_weight_of_zero_leaves_the_objective_alone():
    """The knob has to be a knob: at 0 the score is exactly what it was without it."""
    free_values = [MARGINAL_SLING_PARAMS[name] for name in PARAM_NAMES]
    population = np.array([[value] for value in free_values])
    config = OptimizationConfig(snap_penalty_weight=0.0)

    # Same point, both engines, no penalty: the two objectives are one formula.
    assert float(_objective_vectorized(population, config)[0]) == pytest.approx(
        _objective(free_values, config), rel=1e-3
    )


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
