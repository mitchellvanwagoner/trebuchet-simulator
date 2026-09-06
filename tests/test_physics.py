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
#
# It states its own pivot height rather than leaning on the default. It was added when the
# default was 1 m and this 1.78 m arm dug itself into the ground partway round, ending the
# launch; the default has since risen to 2.5 m and would clear it anyway, but a fixture
# about rope tension should not quietly become a fixture about something else the next
# time a default moves.
ALWAYS_TAUT_PARAMS = {
    "counter_weight_mass": 31.382616,
    "pulley_radius": 0.154871,
    "arm_length": 1.776223,
    "string_length": 0.800749,
    "release_angle": -3.388988,
    "pivot_height": 2.5,
}


def test_tension_metrics_report_no_slack_for_an_always_taut_launch():
    result = simulate_trebuchet(TrebuchetParams(**ALWAYS_TAUT_PARAMS))
    sol = result.solution

    assert result.metrics["release_occurred"] is True
    assert result.metrics["min_string_tension"] > 0
    assert result.metrics["min_cw_rope_tension"] > 0
    assert result.metrics["cw_rope_compression_impulse"] == 0.0
    # Nothing ever detached, so there is no slack time and no re-tension snap.
    assert result.metrics["string_slack_fraction"] == 0.0
    assert result.metrics["sling_snap_count"] == 0
    assert result.metrics["sling_snap_energy"] == 0.0
    # A launch that never goes slack is a single uninterrupted taut segment - the
    # regime never switches, so the rigid-link model is exact here.
    assert [seg.regime for seg in sol.segments] == ["taut"]
    assert sol.slack_time == 0
    # This set was picked for taut ropes rather than for margin, and it does graze the
    # floor - but only just, below the 0.01 the UI reports at and the two engines can
    # tell apart. The shipped defaults keep enough margin to sit at exactly zero.
    assert result.metrics["sling_tension_deficit"] < 0.01
    assert simulate_trebuchet(default_params()).metrics["sling_tension_deficit"] == 0.0
    # Standing clear of the ground is part of what makes this launch uneventful.
    assert result.metrics["arm_ground_contact"] is False


# A sling that holds on by its fingernails: it never detaches, so every metric that
# only fires on detachment reads exactly as clean as the always-taut set above, yet
# 4 of 10 one-parameter +-10% perturbations of it do snap. The gap between those two
# facts is what sling_tension_deficit exists to close.
#
# Like the always-taut set it states its own pivot height, but for the projectile rather
# than the beam: this design swings the stone down to 1.234 m below the pivot, so under
# the 1 m default in force when this was written it ploughed into the ground, landed,
# dragged and detached - a launch about the ground rather than about a marginal sling. A
# pulley machine's equations of motion do not contain the pivot height at all (it only
# shifts where everything is drawn), so any height that clears reproduces exactly the
# launch this was picked for, to the last digit of tension. 1.5 m does, with 0.27 m to
# spare, and pins the fixture where a later default cannot move it.
MARGINAL_SLING_PARAMS = {
    "counter_weight_mass": 21.396,
    "pulley_radius": 0.0229,
    "arm_length": 0.9,
    "string_length": 0.416,
    "release_angle": -4.064,
    "pivot_height": 1.5,
}


def test_a_marginal_sling_is_graded_before_it_ever_goes_slack():
    metrics = simulate_trebuchet(TrebuchetParams(**MARGINAL_SLING_PARAMS)).metrics

    # Every pre-existing sling signal says this launch is fine...
    assert metrics["min_string_tension"] > 0
    assert metrics["string_slack_fraction"] == 0.0
    assert metrics["sling_snap_count"] == 0
    assert metrics["sling_snap_energy"] == 0.0

    # ...but it spent a tenth of the launch with barely any load in the rope. Only a
    # graded measure can say so, which is why the optimizer steers by this one and not
    # by the snap energy: the snap energy of a launch that never snaps is zero however
    # close it came (see optimization.snap_penalty_weight).
    assert metrics["sling_tension_deficit"] > 0.1
    assert metrics["min_string_tension"] < 0.15  # newtons, on a 1.5 N projectile


def test_sling_tension_deficit_is_a_share_of_the_launch_that_slack_time_bounds():
    """It is a fraction, and detached time is the part of it that is already known.

    While the sling is slack it carries nothing at all, so every slack second counts
    in full - the deficit can never come out under the slack fraction, and the extra
    is precisely the marginal-but-still-attached time that nothing else measures.
    """
    for values in (DEFAULT_OPTIMIZABLE_PARAMS, MARGINAL_SLING_PARAMS, JERKY_SLING_PARAMS):
        metrics = simulate_trebuchet(TrebuchetParams(**values)).metrics
        deficit = metrics["sling_tension_deficit"]

        assert 0.0 <= deficit <= 1.0, values
        assert deficit >= metrics["string_slack_fraction"] - 1e-12, values


# A jerky parameter set (the pre-slack-penalty optimizer defaults). Under the old
# rigid-link model the sling was held in compression for much of the launch, which a
# rope cannot do; the sling is now modelled as a real rope, so the same parameters
# instead produce detach/re-tension cycles.
JERKY_SLING_PARAMS = {
    "counter_weight_mass": 16.865,
    "pulley_radius": 0.121,
    "arm_length": 0.813,
    "string_length": 0.669,
    "release_angle": -4.877,
}


def test_slack_sling_launch_detaches_and_snaps_instead_of_pushing():
    result = simulate_trebuchet(TrebuchetParams(**JERKY_SLING_PARAMS))
    sol = result.solution
    metrics = result.metrics

    # The point of the rope model: the sling never carries compression. The rigid-link
    # model used to report a large negative minimum here; now the solver switches to a
    # slack regime at the zero crossing instead, so the minimum can only touch zero.
    assert metrics["min_string_tension"] >= -1e-9

    # It pays for that with detached flight and an inelastic re-tension snap.
    assert 0 < metrics["string_slack_fraction"] < 1
    assert metrics["sling_snap_count"] >= 1
    assert metrics["sling_snap_energy"] > 0

    # Metrics are just summaries of the stitched solution, so they must agree with it.
    assert metrics["sling_snap_count"] == len(sol.snap_times)
    assert metrics["sling_snap_energy"] == pytest.approx(sum(sol.snap_energy_losses))
    assert metrics["string_slack_fraction"] == pytest.approx(sol.slack_time / sol.t_end)

    regimes = [seg.regime for seg in sol.segments]
    assert "slack" in regimes
    # Segments are a stitched alternation - two consecutive segments of the same
    # regime would mean a spurious switch that changed nothing.
    assert all(a != b for a, b in zip(regimes, regimes[1:]))
    assert sol.slack_time == pytest.approx(
        sum(seg.t1 - seg.t0 for seg in sol.segments if seg.regime == "slack")
    )

    # The counterweight rope is still a rigid link, so it keeps the feasibility-style
    # compression impulse - this fixture is unphysical there even though the sling is
    # now handled properly.
    assert metrics["cw_rope_compression_impulse"] > 0.1


def test_sling_snap_only_ever_removes_energy():
    """The re-tension snap is inelastic: it must dissipate, never act as a spring."""
    import numpy as np

    result = simulate_trebuchet(TrebuchetParams(**JERKY_SLING_PARAMS), track_energy=True)
    sol = result.solution
    assert sol.snap_times, "fixture is expected to snap at least once"

    totals = np.array([entry["total"] for entry in result.energy_history])
    times = np.array([entry["time"] for entry in result.energy_history])

    # Total energy is monotonically non-increasing across the whole launch: drag,
    # joint friction and the snaps all remove energy and nothing adds any. The
    # tolerance is scaled to the energy in the machine (~1e-7 J here) purely to absorb
    # float noise - a snap behaving like a spring would show up as a jump of order the
    # snap loss itself, ~1 J.
    assert np.max(np.diff(totals)) <= 1e-9 * totals[0]

    # Across a tight window around the snap, the drop accounts for the reported loss.
    # It is slightly larger because drag and friction keep acting during the window.
    t_snap, loss = sol.snap_times[0], sol.snap_energy_losses[0]
    window = 0.002
    before = totals[times < t_snap - window]
    after = totals[times > t_snap + window]
    assert len(before) and len(after)
    drop = before[-1] - after[0]
    assert drop >= loss
    assert drop == pytest.approx(loss, rel=0.15)


def test_constraint_tensions_satisfy_newtons_law_for_the_projectile():
    import numpy as np

    from trebuchet_sim.config import G, RHO_AIR
    from trebuchet_sim.physics import TrebuchetSimulator

    # The string is a two-force member: m_p * a must equal gravity + drag plus a force
    # of magnitude -T purely along the string. Reconstruct that force from the solved
    # generalized accelerations and check it against constraint_tensions at several
    # points of a real launch (one that includes slack phases).
    #
    # Sampled per taut segment rather than across the whole launch: constraint_tensions
    # and trebuchet_dynamics both take the taut state [theta, theta_dot, alpha,
    # alpha_dot, psi, psi_dot], and while the sling is slack the solution carries
    # [theta, theta_dot, px, py, pvx, pvy, psi, psi_dot] instead - there is no string
    # force to check there, because the rope is carrying nothing.
    params = TrebuchetParams(**JERKY_SLING_PARAMS)
    sim = TrebuchetSimulator(params)
    sol = simulate_trebuchet(params).solution
    m_p, l_a, l_s = params.projectile_mass, params.arm_length, params.string_length
    drag_k = 0.5 * RHO_AIR * params.projectile_drag_coefficient * params.projectile_area

    taut_segments = [seg for seg in sol.segments if seg.regime == "taut"]
    assert taut_segments, "fixture is expected to spend part of the launch taut"
    sampled = 0

    for seg in taut_segments:
        for t in np.linspace(seg.t0, seg.t1, 5):
            y = seg.sol.sol(float(t))
            sampled += 1
            theta, theta_dot, alpha, alpha_dot = (float(v) for v in y[:4])
            derivs = sim.trebuchet_dynamics(float(t), y)
            theta_ddot, alpha_ddot = derivs[1], derivs[3]

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
            # A taut segment runs from one tension zero-crossing to the next, so the
            # rope is pulling (or exactly slack at the boundary) everywhere inside it.
            assert string_tension >= -1e-9 * scale

    assert sampled >= 8  # the fixture has several taut stretches; keep coverage real


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
