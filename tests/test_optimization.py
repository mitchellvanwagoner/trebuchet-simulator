import pytest

import numpy as np

from trebuchet_sim.config import (
    AUTO_INITIAL_ARM_ANGLE,
    DEFAULT_MACHINE_FIXED,
    DEFAULT_MACHINE_PARAMS,
    DEFAULT_OPTIMIZABLE_PARAMS,
    MachineType,
    TrebuchetParams,
)
from trebuchet_sim.optimization import (
    INVALID_COST,
    PARAM_BOUNDS,
    PARAM_LIMITS,
    PARAM_NAMES,
    OptimizationConfig,
    _fastsim_fixed_scalar,
    _objective,
    _objective_vectorized,
    optimize_trebuchet,
    thread_count,
)
from trebuchet_sim.physics import simulate_trebuchet

pytest.importorskip("numba")


# A pulley machine that both lets its sling go and drops the stone on the ground before
# throwing it 58 m - the two dissipative discontinuities the jerk penalty prices, in one
# launch. It spends 5.5 J on snaps and 70.9 J on the ground, and never touches the beam,
# which keeps it a fixture about the two energies it is here for.
#
# The second design to sit here. The first was chosen before the beam was a surface the
# stone could hit, and once it was, that launch caught the arm on the way down instead of
# the ground - `projectile_ground_energy` went to exactly zero and it had nothing left to
# price. Drawn from a 500-draw sweep of PARAM_BOUNDS (seed 909), and picked over a larger
# jerk that both engines do *not* agree on: that one carries a beam strike, and the two
# score it 10.5% apart at a jerk weight of 250, which is a fact about a launch whose
# contact placement is marginal rather than about the penalty this is testing.
JERK_PARAMS = {
    "counter_weight_mass": 17.588115307836468,
    "pulley_radius": 0.36383226126553714,
    "arm_length": 1.6449421511814302,
    "string_length": 0.6627046710227973,
    "release_angle": -0.15179840800539646,
    "pivot_height": 2.1470008614689604,
}


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
    # A pulley machine that drives its counterweight rope deep into compression - the rope
    # would have to push, which no rope does - so the slack penalty must raise its cost by
    # weight * impulse. It reaches -336 N for 11.9 N*s while still throwing 14 m, so it is
    # a design the search could plausibly wander into rather than a wreck.
    #
    # The second design to sit here. The first was the pre-slack-penalty optimizer
    # defaults, which under the solid beam ride the arm from the cocked pose onward, throw
    # nothing, and report exactly zero impulse - nothing left to penalize. Drawn from the
    # same 500-draw sweep JERK_PARAMS comes from.
    jerky = {
        "counter_weight_mass": 33.552160,
        "pulley_radius": 0.268780,
        "arm_length": 1.819110,
        "string_length": 1.078258,
        "release_angle": -1.120253,
    }
    fixed = {"pivot_height": 2.635905}
    free_values = [jerky[name] for name in PARAM_NAMES]

    base = _objective(free_values, OptimizationConfig(slack_penalty_weight=0.0, fixed_params=dict(fixed)))
    penalized = _objective(
        free_values, OptimizationConfig(slack_penalty_weight=200.0, fixed_params=dict(fixed))
    )

    # It scores at all, so the gap between the two is the penalty and nothing else - and
    # the gap is exactly weight * impulse, which is the whole claim.
    assert base < 1e6
    impulse = simulate_trebuchet(
        TrebuchetParams(**jerky, **fixed), rtol=1e-6, dense_output=False
    ).metrics["cw_rope_compression_impulse"]
    assert impulse > 1.0
    assert penalized - base == pytest.approx(200.0 * impulse, rel=1e-6)


# Copied from tests/test_physics.py: a sling that never detaches but runs on almost no
# load. Every signal the objective had before this one - slack fraction, snap count,
# snap energy, both compression impulses - reads exactly zero for it. The raised pivot
# is part of the fixture; see the copy in test_physics.py for why.
MARGINAL_SLING_PARAMS = {
    "counter_weight_mass": 59.251943,
    "pulley_radius": 0.053070,
    "arm_length": 0.639161,
    "string_length": 0.560699,
    "release_angle": -0.101006,
    "pivot_height": 2.5,
}

# The five design variables and the geometry they were measured on, in the two shapes
# the objective wants them: the search space is five names wide, so the pivot height has
# to reach the objective as fixed geometry or the run is a different machine.
MARGINAL_FREE_VALUES = [MARGINAL_SLING_PARAMS[name] for name in PARAM_NAMES]
MARGINAL_FIXED = {"pivot_height": MARGINAL_SLING_PARAMS["pivot_height"]}


def _marginal_config(**overrides) -> OptimizationConfig:
    return OptimizationConfig(fixed_params=dict(MARGINAL_FIXED), **overrides)


def test_objective_penalizes_a_jerky_sling_that_never_actually_goes_slack():
    """The gap the snap penalty fills: a design one nudge away from snapping.

    The slack penalty cannot price this. It is paid on compression impulse, which is
    zero until the rope has already let go, so between two designs that both hold
    together it is the same number - and a search whose objective is flat across the
    approach to a cliff will happily park on the edge of it.
    """
    metrics = simulate_trebuchet(TrebuchetParams(**MARGINAL_SLING_PARAMS)).metrics
    assert metrics["cw_rope_compression_impulse"] == 0.0  # nothing for the slack penalty to charge
    assert metrics["sling_snap_count"] == 0

    unpenalized = _objective(MARGINAL_FREE_VALUES, _marginal_config(snap_penalty_weight=0.0))
    penalized = _objective(MARGINAL_FREE_VALUES, _marginal_config(snap_penalty_weight=300.0))

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
    population = np.array([[value] for value in MARGINAL_FREE_VALUES])

    scipy_costs, fast_costs = [], []
    for weight in (0.0, 300.0):
        config = _marginal_config(snap_penalty_weight=weight)
        scipy_costs.append(_objective(MARGINAL_FREE_VALUES, config))
        fast_costs.append(float(_objective_vectorized(population, config)[0]))

    scipy_charge = scipy_costs[1] - scipy_costs[0]
    fast_charge = fast_costs[1] - fast_costs[0]
    assert scipy_charge > 25.0
    assert fast_charge == pytest.approx(scipy_charge, rel=0.1)


def test_snap_penalty_weight_of_zero_leaves_the_objective_alone():
    """The knob has to be a knob: at 0 the score is exactly what it was without it."""
    population = np.array([[value] for value in MARGINAL_FREE_VALUES])
    config = _marginal_config(snap_penalty_weight=0.0)

    # Same point, both engines, no penalty: the two objectives are one formula.
    assert float(_objective_vectorized(population, config)[0]) == pytest.approx(
        _objective(MARGINAL_FREE_VALUES, config), rel=1e-3
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


def test_the_objective_charges_for_the_energy_a_launch_throws_away():
    """Snaps and ground impacts are jerks, not merely losses.

    Efficiency already notices the joules - it divides by the potential energy spent - but
    it prices a joule lost to a stone hitting the dirt exactly like a joule lost to air
    drag. This term is what makes the first one worse, and it is identically zero for a
    design that does neither, so a clean winner's score is untouched by it.
    """
    from trebuchet_sim.optimization import _objective

    clean = OptimizationConfig(jerk_penalty_weight=0.0)
    charged = OptimizationConfig(jerk_penalty_weight=250.0)
    values = [DEFAULT_OPTIMIZABLE_PARAMS[name] for name in clean.free_params]

    # The shipped machine neither snaps nor lands, so the weight cannot move its score.
    default_result = simulate_trebuchet(TrebuchetParams(**DEFAULT_OPTIMIZABLE_PARAMS))
    assert default_result.metrics["sling_snap_energy"] == 0.0
    assert default_result.metrics["projectile_ground_energy"] == 0.0
    assert _objective(values, charged) == pytest.approx(_objective(values, clean))

    # One that does both costs strictly more, by exactly the weight times the joules.
    jerky_free = OptimizationConfig(
        jerk_penalty_weight=0.0, fixed_params={"pivot_height": JERK_PARAMS["pivot_height"]}
    )
    jerky_charged = OptimizationConfig(
        jerk_penalty_weight=250.0, fixed_params={"pivot_height": JERK_PARAMS["pivot_height"]}
    )
    jerky = [JERK_PARAMS[name] for name in jerky_free.free_params]
    metrics = simulate_trebuchet(TrebuchetParams(**JERK_PARAMS)).metrics
    thrown_away = metrics["sling_snap_energy"] + metrics["projectile_ground_energy"]
    assert metrics["sling_snap_energy"] > 0.0
    assert metrics["projectile_ground_energy"] > 0.0
    assert _objective(jerky, jerky_charged) - _objective(jerky, jerky_free) == pytest.approx(
        250.0 * thrown_away, rel=1e-3
    )


def test_both_engines_charge_the_same_jerk_cost():
    """The fast engine has always measured these two energies and discarded them here."""
    for weight in (0.0, 250.0):
        config = OptimizationConfig(
            jerk_penalty_weight=weight,
            fixed_params={"pivot_height": JERK_PARAMS["pivot_height"]},
        )
        values = np.array(
            [[JERK_PARAMS[name]] for name in config.free_params], dtype=np.float64
        )
        assert _objective_vectorized(values, config)[0] == pytest.approx(
            _objective([JERK_PARAMS[name] for name in config.free_params], config), rel=1e-3
        )


def test_a_none_fixed_param_means_the_same_as_leaving_it_out():
    """`fixed_params={"initial_arm_angle": None}` must not become a real arm angle.

    The dashboard has no cocked-angle box on the traditional machine - that angle follows
    the arm length, which is what the search varies - so it passes None, meaning what an
    absent key means: resolve it per individual. It used to be coerced to 0.0, which is
    harmless for the counterweight rope, whose "unset" sentinel is 0.0 anyway, and silent
    for the angle, because 0.0 radians is a perfectly good arm cocked flat along the
    ground. The two halves of a run then described different machines: the fast objective
    scored every candidate at 0 degrees while build_params resolved the same design to
    -140.5, so the design DE returned and the result reported beside it came from a
    machine nothing had scored. On these defaults that is a cost of about 1069 against
    about 7.2.
    """
    machine = MachineType.TRADITIONAL
    fixed = dict(DEFAULT_MACHINE_FIXED[machine])
    absent = OptimizationConfig(machine=machine, fixed_params=dict(fixed))
    explicit = OptimizationConfig(
        machine=machine, fixed_params=dict(fixed, initial_arm_angle=None)
    )

    assert _fastsim_fixed_scalar(explicit, "initial_arm_angle") == AUTO_INITIAL_ARM_ANGLE
    assert _fastsim_fixed_scalar(explicit, "initial_arm_angle") == _fastsim_fixed_scalar(
        absent, "initial_arm_angle"
    )
    # And the sentinel has to survive all the way to a score, not just to the scalar: the
    # angle it stands for is the one the reference resolves geometrically, so the two
    # configs must price the same design identically.
    values = np.array(
        [[DEFAULT_MACHINE_PARAMS[machine][name]] for name in absent.free_params],
        dtype=np.float64,
    )
    assert _objective_vectorized(values, explicit)[0] == pytest.approx(
        _objective_vectorized(values, absent)[0], rel=1e-9
    )
    # The angle both of them mean, for good measure: walked down to the ground, not flat.
    assert explicit.build_params(values[:, 0]).initial_arm_angle == pytest.approx(
        absent.build_params(values[:, 0]).initial_arm_angle
    )
    assert explicit.build_params(values[:, 0]).initial_arm_angle < -np.pi / 2


def test_a_search_with_nothing_to_find_says_so_instead_of_returning_a_blank():
    """A search space with no scoreable design in it has to be reported as such.

    Every design scoring INVALID_COST leaves differential evolution a perfectly flat
    landscape: it converges on that flatness and `de_result.x` is then wherever the
    population happened to be standing, not a winner. Returning it anyway is what reached
    the dashboard as a panel of blank metrics and a log of empty rows, with no way to tell
    "here is the best machine" from "there was no machine".

    A pulley machine's sling longer than 0.95 of its arm is the one rule either engine
    refuses outright, before any integration, so locking both lengths there empties the
    space exactly rather than nearly - and it is an ordinary thing to type by accident.
    The rule is that machine's alone: it cocks with the sling tucked along the arm, where
    the traditional machine lays its sling out from the tip and is bounded only by the
    search range (see optimization._objective). The message names the locks because they
    are what has to move.
    """
    config = OptimizationConfig(
        machine=MachineType.PULLEY,
        locked_params={"arm_length": 0.5, "string_length": 0.49},
    )
    # The premise: the space really is unscoreable, not merely hard.
    rng = np.random.default_rng(3)
    low = np.array([b[0] for b in config.bounds])[:, None]
    high = np.array([b[1] for b in config.bounds])[:, None]
    sample = low + (high - low) * rng.random((len(config.bounds), 400))
    assert (_objective_vectorized(sample, config) >= INVALID_COST).all()

    with pytest.raises(ValueError) as excinfo:
        optimize_trebuchet(config)
    message = str(excinfo.value)
    assert "no valid design" in message
    assert "pulley" in message
    assert "arm_length=0.5" in message and "string_length=0.49" in message


def test_thread_count_is_clamped_to_what_the_process_was_started_with():
    """numba fixes the ceiling when it is imported, so asking for more has to report the
    number actually used rather than silently not taking (see fastsim._seed_thread_ceiling).
    """
    from trebuchet_sim import fastsim

    ceiling = fastsim.thread_ceiling()
    assert thread_count(1) == 1
    assert thread_count(ceiling) == ceiling
    assert thread_count(ceiling + 1000) == ceiling
    assert thread_count(0) == 1  # a nonsense count still has to name a real thread
    assert thread_count(None) == fastsim.get_num_threads()


def test_the_thread_count_cannot_move_a_score():
    """evaluate_population gives each individual its own slot and never reduces across
    them, so the number of threads is a throughput setting and nothing else. Asserted
    bit-exactly rather than approximately: a thread count that moved a score by even an
    ulp would make an optimizer run depend on the machine it was run on.
    """
    from trebuchet_sim import fastsim

    config = OptimizationConfig()
    rng = np.random.default_rng(17)
    low = np.array([b[0] for b in config.bounds])[:, None]
    high = np.array([b[1] for b in config.bounds])[:, None]
    sample = low + (high - low) * rng.random((len(config.bounds), 500))

    previous = fastsim.get_num_threads()
    try:
        fastsim.set_num_threads(1)
        expected = _objective_vectorized(sample, config)
        for count in (2, 3, fastsim.thread_ceiling()):
            fastsim.set_num_threads(thread_count(count))
            assert np.array_equal(_objective_vectorized(sample, config), expected)
    finally:
        fastsim.set_num_threads(previous)


def test_optimizing_puts_the_thread_count_back():
    """The count is process-wide, so a run that pinned it would otherwise decide how every
    later run in the same process - a dashboard session, a test module - was scored.
    """
    from trebuchet_sim import fastsim

    before = fastsim.get_num_threads()
    optimize_trebuchet(OptimizationConfig(threads=1, max_iterations=2))
    assert fastsim.get_num_threads() == before


def test_a_thread_count_below_one_is_refused():
    with pytest.raises(ValueError, match="threads must be at least 1"):
        OptimizationConfig(threads=0)
