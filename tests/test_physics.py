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


# A parameter set (found by random search, seed 3 over PARAM_BOUNDS) whose launch keeps
# both the sling and the counterweight rope taut throughout - the rigid-link model is
# physically valid for it, so its compression impulses must be exactly zero.
ALWAYS_TAUT_PARAMS = {
    "counter_weight_mass": 31.382616,
    "pulley_radius": 0.154871,
    "arm_length": 1.776223,
    "string_length": 0.800749,
    "release_angle": -3.388988,
}


def test_tension_metrics_are_zero_impulse_for_an_always_taut_launch():
    result = simulate_trebuchet(TrebuchetParams(**ALWAYS_TAUT_PARAMS))

    assert result.metrics["release_occurred"] is True
    assert result.metrics["min_string_tension"] > 0
    assert result.metrics["min_cw_rope_tension"] > 0
    assert result.metrics["string_compression_impulse"] == 0.0
    assert result.metrics["cw_rope_compression_impulse"] == 0.0
    assert result.metrics["string_slack_fraction"] == 0.0


# A jerky parameter set (the pre-slack-penalty optimizer defaults): the rigid-link
# model holds the sling in compression for ~half the launch, which a rope cannot do.
JERKY_SLING_PARAMS = {
    "counter_weight_mass": 16.865,
    "pulley_radius": 0.121,
    "arm_length": 0.813,
    "string_length": 0.669,
    "release_angle": -4.877,
}


def test_tension_metrics_flag_a_slack_sling_launch():
    result = simulate_trebuchet(TrebuchetParams(**JERKY_SLING_PARAMS))

    assert result.metrics["min_string_tension"] < 0
    assert result.metrics["string_compression_impulse"] > 0.1
    assert 0 < result.metrics["string_slack_fraction"] < 1


def test_constraint_tensions_satisfy_newtons_law_for_the_projectile():
    import numpy as np

    from trebuchet_sim.config import G, RHO_AIR
    from trebuchet_sim.physics import TrebuchetSimulator

    # The string is a two-force member: m_p * a must equal gravity + drag plus a force
    # of magnitude -T purely along the string. Reconstruct that force from the solved
    # generalized accelerations and check it against constraint_tensions at several
    # points of a real launch (one that includes a slack/compression phase).
    params = TrebuchetParams(**JERKY_SLING_PARAMS)
    sim = TrebuchetSimulator(params)
    sol = simulate_trebuchet(params).solution
    m_p, l_a, l_s = params.projectile_mass, params.arm_length, params.string_length
    drag_k = 0.5 * RHO_AIR * params.projectile_drag_coefficient * params.projectile_area

    for t in np.linspace(0.0, sol.t[-1], 9):
        y = sol.sol(float(t))
        theta, theta_dot, alpha, alpha_dot = (float(v) for v in y)
        _, theta_ddot, _, alpha_ddot = sim.trebuchet_dynamics(float(t), y)

        ax = -l_a * (theta_ddot * math.sin(theta) + theta_dot**2 * math.cos(theta)) - l_s * (
            alpha_ddot * math.sin(alpha) + alpha_dot**2 * math.cos(alpha)
        )
        ay = l_a * (theta_ddot * math.cos(theta) - theta_dot**2 * math.sin(theta)) + l_s * (
            alpha_ddot * math.cos(alpha) - alpha_dot**2 * math.sin(alpha)
        )

        _, (vx, vy) = sim.projectile_position_velocity(y)
        speed = math.hypot(vx, vy)
        string_fx = m_p * ax - (-drag_k * speed * vx)
        string_fy = m_p * ay - (-drag_k * speed * vy - m_p * G)

        expected_tension = -(string_fx * math.cos(alpha) + string_fy * math.sin(alpha))
        tangential = -string_fx * math.sin(alpha) + string_fy * math.cos(alpha)

        string_tension, cw_tension = sim.constraint_tensions(float(t), y)

        scale = max(1.0, abs(expected_tension))
        assert abs(tangential) < 1e-8 * scale  # string force is purely radial
        assert string_tension == pytest.approx(expected_tension, rel=1e-9, abs=1e-9)
        assert cw_tension == pytest.approx(
            params.counter_weight_mass * (G + params.pulley_radius * theta_ddot)
        )


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
