import math

import pytest

from trebuchet_sim.config import DEFAULT_OPTIMIZABLE_PARAMS, TrebuchetParams
from trebuchet_sim.physics import simulate_trebuchet


def default_params(**overrides) -> TrebuchetParams:
    values = dict(DEFAULT_OPTIMIZABLE_PARAMS)
    values.update(overrides)
    return TrebuchetParams(**values)


def test_simulate_produces_a_release_and_positive_range():
    result = simulate_trebuchet(default_params())

    assert result.metrics["release_occurred"] is True
    assert result.distance > 0
    assert 0 < result.efficiency < 1


def test_energy_history_is_populated_when_tracking_enabled():
    result = simulate_trebuchet(default_params(), track_energy=True)

    assert result.energy_history
    assert result.energy_history[0]["total"] == math.fsum(
        [result.energy_history[0]["kinetic"], result.energy_history[0]["potential"]]
    )


def test_energy_history_is_none_when_tracking_disabled():
    result = simulate_trebuchet(default_params(), track_energy=False)

    assert result.energy_history is None


def test_string_length_ratio_derived_property():
    params = default_params(arm_length=1.0, string_length=0.5)

    assert params.string_arm_ratio == 0.5


def test_release_velocity_is_true_speed_not_speed_squared():
    result = simulate_trebuchet(default_params())

    vx, vy = result.metrics["release_velocity_components"]
    expected_speed = math.hypot(vx, vy)

    assert math.isclose(result.metrics["release_velocity"], expected_speed, rel_tol=1e-9)
    # A sanity bound: true release speed should be well under 100 m/s for these defaults,
    # whereas the old (buggy) v^2 value would be in the thousands.
    assert result.metrics["release_velocity"] < 100.0


def test_moi_pulley_scales_with_pulley_radius():
    small = default_params(pulley_radius=0.05)
    large = default_params(pulley_radius=0.2)

    assert large.moi_pulley > small.moi_pulley
    assert small.moi_pulley == 0.5 * small.pulley_mass * small.pulley_radius**2


def test_counter_weight_size_scales_with_mass_not_density():
    heavier = default_params(counter_weight_mass=30.0)
    lighter = default_params(counter_weight_mass=10.0)

    assert heavier.counter_weight_size > lighter.counter_weight_size
    assert heavier.counter_weight_size == pytest.approx(
        (heavier.counter_weight_mass / heavier.counter_weight_density) ** (1 / 3)
    )


def test_initial_cw_rope_length_defaults_to_twice_pulley_radius():
    params = default_params()

    assert params.counter_weight_rope_length is None
    assert params.initial_cw_rope_length == pytest.approx(2 * params.pulley_radius)


def test_initial_cw_rope_length_uses_explicit_override():
    params = default_params(counter_weight_rope_length=0.5)

    assert params.initial_cw_rope_length == 0.5


def test_cw_rope_length_shifts_initial_counterweight_height_without_changing_dynamics():
    from trebuchet_sim.physics import TrebuchetSimulator

    default = default_params()
    custom = default_params(counter_weight_rope_length=0.5)

    y0 = TrebuchetSimulator(default).initial_state()
    (_x_default, y_default), _ = TrebuchetSimulator(default).weight_position_velocity(y0)
    (_x_custom, y_custom), _ = TrebuchetSimulator(custom).weight_position_velocity(y0)

    assert y_default == pytest.approx(default.weight_height - default.initial_cw_rope_length)
    assert y_custom == pytest.approx(custom.weight_height - custom.initial_cw_rope_length)
    assert y_default != pytest.approx(y_custom)

    # Rope length only shifts the counterweight's rendered/ground-collision reference
    # height; the launch dynamics never read it directly, so the flight is unaffected.
    result_default = simulate_trebuchet(default)
    result_custom = simulate_trebuchet(custom)
    assert result_default.distance == pytest.approx(result_custom.distance)
    assert result_default.efficiency == pytest.approx(result_custom.efficiency)
