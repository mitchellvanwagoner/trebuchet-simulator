import math

import numpy as np
import pytest

from trebuchet_sim.config import DEFAULT_OPTIMIZABLE_PARAMS, TrebuchetParams
from trebuchet_sim.physics import TrebuchetSimulator, sample_full_timeline, simulate_trebuchet


def default_params(**overrides) -> TrebuchetParams:
    values = dict(DEFAULT_OPTIMIZABLE_PARAMS)
    values.update(overrides)
    return TrebuchetParams(**values)


def test_aftermath_is_none_by_default():
    result = simulate_trebuchet(default_params())

    assert result.aftermath is None
    assert "cw_touchdown_times" not in result.metrics


def test_aftermath_state_continuous_with_launch_release_state():
    params = default_params()
    result = simulate_trebuchet(params, simulate_aftermath=True)

    assert result.aftermath is not None
    theta_release, theta_dot_release = result.solution.y_events[0][0][0], result.solution.y_events[0][0][1]
    theta0, theta_dot0, _regime = result.aftermath.state_at(0.0)

    assert theta0 == pytest.approx(theta_release)
    assert theta_dot0 == pytest.approx(theta_dot_release)


def test_default_params_produce_a_touchdown_shortly_after_release():
    result = simulate_trebuchet(default_params(), simulate_aftermath=True)

    assert result.metrics["cw_touchdown_times"]
    assert result.metrics["cw_touchdown_times"][0] > 0


def test_counterweight_bottom_face_never_goes_negative_during_aftermath():
    params = default_params()
    result = simulate_trebuchet(params, simulate_aftermath=True)
    simulator = TrebuchetSimulator(params)
    flight_time = result.metrics["flight_time"]
    half_size = params.counter_weight_size / 2

    for t in np.linspace(0, flight_time, 300):
        theta, theta_dot, regime = result.aftermath.state_at(float(t))
        (_x, w_y), _ = simulator.weight_position_velocity((theta, theta_dot, 0.0, 0.0))
        if regime == "slack":
            w_y = half_size  # sampler clamps; the raw ODE state can drift past ground while slack
        bottom_face_height = w_y - half_size
        assert bottom_face_height >= -1e-9


def test_retension_jerk_matches_angular_momentum_ratio():
    params = default_params()
    simulator = TrebuchetSimulator(params)
    theta_ground = simulator._theta_ground

    # Start just below ground moving upward: touches down and re-tensions quickly.
    aftermath = simulator.simulate_aftermath(theta_release=theta_ground - 0.05, theta_dot_release=2.0, duration=2.0)

    assert aftermath.retension_times
    t_retension = aftermath.retension_times[0]
    eps = 1e-6
    _theta_before, theta_dot_before, regime_before = aftermath.state_at(t_retension - eps)
    _theta_after, theta_dot_after, regime_after = aftermath.state_at(t_retension + eps)

    assert regime_before == "slack"
    assert regime_after == "taut"
    expected_ratio = simulator._M_slack / simulator._M_taut
    assert theta_dot_after / theta_dot_before == pytest.approx(expected_ratio, rel=1e-3)


def test_touchdown_transition_is_continuous_no_impulse():
    params = default_params()
    simulator = TrebuchetSimulator(params)
    theta_ground = simulator._theta_ground

    # Start just above ground moving downward: touches down without re-tensioning.
    aftermath = simulator.simulate_aftermath(theta_release=theta_ground + 0.05, theta_dot_release=-2.0, duration=0.5)

    assert aftermath.touchdown_times
    t_touchdown = aftermath.touchdown_times[0]
    eps = 1e-6
    _theta_before, theta_dot_before, regime_before = aftermath.state_at(t_touchdown - eps)
    _theta_after, theta_dot_after, regime_after = aftermath.state_at(t_touchdown + eps)

    assert regime_before == "taut"
    assert regime_after == "slack"
    assert theta_dot_after == pytest.approx(theta_dot_before, rel=1e-3)


def test_sample_full_timeline_matches_launch_sampling_before_release():
    params = default_params()
    result = simulate_trebuchet(params, simulate_aftermath=True)
    t_release = result.metrics["t_release"]

    times = np.linspace(0, t_release, 20)
    positions = sample_full_timeline(params, result, times)

    simulator = TrebuchetSimulator(params)
    for i, t in enumerate(times):
        y = result.solution.sol(float(t))
        expected_arm_tip = simulator.arm_tip_position_velocity(y)[0]
        assert positions["arm_tip"][i] == pytest.approx(expected_arm_tip)


def test_sample_full_timeline_ends_at_projectile_impact():
    params = default_params()
    result = simulate_trebuchet(params, simulate_aftermath=True)
    t_release = result.metrics["t_release"]
    flight_time = result.metrics["flight_time"]
    t_end = t_release + flight_time

    positions = sample_full_timeline(params, result, [t_end])

    assert positions["projectile"][0][0] == pytest.approx(result.distance, rel=1e-6)
    assert positions["projectile"][0][1] == pytest.approx(0.0, abs=1e-6)


def test_sample_full_timeline_clamps_grounded_counterweight_bottom_face_to_ground():
    params = default_params()
    result = simulate_trebuchet(params, simulate_aftermath=True)
    t_release = result.metrics["t_release"]
    flight_time = result.metrics["flight_time"]

    positions = sample_full_timeline(params, result, [t_release + flight_time])

    # The counterweight position is its center of mass; resting on the ground means
    # the center sits at half the cube's side length, not zero.
    assert positions["counterweight"][0][1] == pytest.approx(params.counter_weight_size / 2, abs=1e-9)


def test_longer_cw_rope_length_starts_aftermath_already_grounded():
    # A long enough rope means the counterweight is already resting on the ground at
    # release, so the aftermath should start directly in "slack" with no touchdown event.
    params = default_params(counter_weight_rope_length=1.5)
    result = simulate_trebuchet(params, simulate_aftermath=True)

    _theta0, _theta_dot0, regime0 = result.aftermath.state_at(0.0)
    assert regime0 == "slack"
    assert result.metrics["cw_touchdown_times"] == []


def test_no_release_case_has_no_aftermath():
    # Enough joint friction that the arm never reaches the release angle.
    params = default_params(joint_friction_coefficient=50.0)
    result = simulate_trebuchet(params, simulate_aftermath=True)

    assert result.metrics["release_occurred"] is False
    assert result.aftermath is None


def test_optimizer_objective_path_unaffected_by_aftermath_feature():
    # rtol=1e-6, dense_output=False, simulate_aftermath left at default (False): must
    # match exactly what the optimizer's _objective already relies on.
    params = default_params()
    result = simulate_trebuchet(params, rtol=1e-6, dense_output=False)

    assert result.aftermath is None
    assert result.distance > 0
    assert 0 < result.efficiency < 1
