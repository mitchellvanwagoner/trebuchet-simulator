"""Traditional (arm-mounted counterweight) machine.

The two machines share one set of equations of motion, so these tests come in pairs:
what the traditional linkage must do, and what adding it must *not* have changed about
the pulley machine.
"""

import numpy as np
import pytest

from trebuchet_sim.config import (
    G,
    DEFAULT_INITIAL_ARM_ANGLE,
    DEFAULT_OPTIMIZABLE_PARAMS,
    DEFAULT_TRADITIONAL_FIXED,
    DEFAULT_TRADITIONAL_PARAMS,
    MachineType,
    TrebuchetParams,
)
from trebuchet_sim.physics import TrebuchetSimulator, simulate_trebuchet


def traditional_params(**overrides) -> TrebuchetParams:
    values = dict(DEFAULT_TRADITIONAL_PARAMS)
    values.update(DEFAULT_TRADITIONAL_FIXED)
    values.update(overrides)
    return TrebuchetParams(machine=MachineType.TRADITIONAL, **values)


def test_machine_type_round_trips_through_a_plain_string():
    """Saved defaults are JSON, so the enum has to survive being written as its value."""
    params = TrebuchetParams(machine="traditional", **DEFAULT_TRADITIONAL_PARAMS)

    assert params.machine is MachineType.TRADITIONAL
    assert not params.has_pulley


def test_initial_arm_angle_resolves_per_machine():
    """None is not a usable start angle - __post_init__ fills in the machine's own."""
    assert TrebuchetParams(**DEFAULT_OPTIMIZABLE_PARAMS).initial_arm_angle == (
        DEFAULT_INITIAL_ARM_ANGLE[MachineType.PULLEY]
    )
    assert traditional_params().initial_arm_angle == DEFAULT_INITIAL_ARM_ANGLE[MachineType.TRADITIONAL]


def test_traditional_machine_releases_and_throws():
    result = simulate_trebuchet(traditional_params())

    assert result.metrics["release_occurred"] is True
    assert result.distance > 0
    assert 0 < result.efficiency < 1
    # The defaults are optimizer output for the 30 m target, the same as the pulley
    # machine's: landing this far off it would mean the linkage torque has the wrong sign
    # or scale rather than that the search had a bad day.
    assert 25 < result.distance < 36


def test_traditional_launch_conserves_energy_with_dissipation_switched_off():
    """The sharpest check on the added counterweight-swing terms.

    Drag, joint friction and sling snaps are the only ways this model can lose energy.
    With all three off, any drift in the total is an error in the Lagrangian itself -
    a wrong sign on the psi coupling would show up here long before it moved the range
    enough to look suspicious.
    """
    params = traditional_params(
        arm_drag_coefficient=0.0, projectile_drag_coefficient=0.0, joint_friction_coefficient=0.0
    )
    result = simulate_trebuchet(params, track_energy=True)

    assert result.metrics["string_slack_fraction"] == 0.0  # no snap losses either
    totals = np.array([entry["total"] for entry in result.energy_history])
    assert np.max(np.abs(totals - totals[0])) < 1e-6 * abs(totals[0])


def test_counterweight_swing_is_inert_on_the_pulley_machine():
    """psi exists in both state vectors but the pulley linkage gives it no freedom.

    This is what lets one 3x3 solve serve both machines: on the pulley machine the
    third coordinate has to stay exactly where it started, contributing nothing.
    """
    simulator = TrebuchetSimulator(TrebuchetParams(**DEFAULT_OPTIMIZABLE_PARAMS))
    state = simulator.initial_state()
    state[1], state[3] = -1.5, 0.4  # arm and sling moving, so the couplings are live

    derivatives = simulator.trebuchet_dynamics(0.0, state)

    assert derivatives[4] == 0.0  # psi_dot
    assert derivatives[5] == 0.0  # psi_ddot


def test_arm_geometry_reduces_to_the_single_sided_beam_without_a_back_section():
    """The two-sided beam formulas must collapse onto the originals for the pulley arm."""
    params = TrebuchetParams(**DEFAULT_OPTIMIZABLE_PARAMS)
    from trebuchet_sim.config import ARM_CROSS_SECTION_WIDTH

    assert params.arm_back_length == 0.0
    assert params.arm_cm_offset == params.arm_length / 2
    assert params.arm_mass == params.arm_density * params.arm_length * ARM_CROSS_SECTION_WIDTH**2
    assert params.moi_arm == (1 / 3) * params.arm_mass * params.arm_length**2


def test_traditional_arm_carries_mass_behind_the_pivot():
    params = traditional_params()

    a, b = params.arm_length, params.length_counterweight

    assert params.arm_back_length == b
    assert params.pulley_mass == 0.0  # no pulley to weigh
    assert params.arm_total_length == a + b
    # The back section pulls the balance point in toward the pivot.
    assert params.arm_cm_offset == pytest.approx((a - b) / 2)
    assert params.arm_cm_offset < a / 2
    # Inertia integrates both sides: m(a^3 + b^3)/3(a + b), i.e. m(a^2 - ab + b^2)/3.
    assert params.moi_arm == pytest.approx(params.arm_mass * (a**3 + b**3) / (3 * (a + b)))
    assert params.counter_weight_lever == b  # the weight rides the arm, no pulley radius


def test_pinned_counterweight_hangs_from_the_arm_not_the_axle():
    """Its pin rides the arm's short end, so it orbits the pivot instead of dropping
    vertically the way the pulley machine's weight does."""
    params = traditional_params()
    simulator = TrebuchetSimulator(params)
    theta = params.initial_arm_angle

    pin_x, pin_y = simulator.counterweight_pin_position(theta)

    assert pin_x == pytest.approx(-params.length_counterweight * np.cos(theta))
    assert pin_y == pytest.approx(params.pivot_height - params.length_counterweight * np.sin(theta))
    # Cocked nose-down, the weight starts above the pivot, ready to fall.
    assert pin_y > params.pivot_height


def test_the_pinned_counterweight_link_is_graded_like_the_pulley_machine_s_rope():
    """A pin holds the link's end, not its length.

    These metrics were reported for the pulley machine alone, on the reading that a pinned
    counterweight has no rope to go slack. It has: the weight hangs a further
    `counter_weight_rope_length` below that pin, on a link this model holds rigid exactly
    as it holds the pulley machine's rope rigid, and a rigid link pushes where a rope would
    let go. Nothing measured that, so the optimizer designed straight into it - every
    traditional winner across a target sweep came back with the link in compression, down
    to -1212 N, while the pulley machine's equivalent was penalized to zero.

    The shipped defaults keep it loaded the whole way, which is the point of measuring it.
    """
    result = simulate_trebuchet(traditional_params())

    assert result.metrics["min_cw_rope_tension"] > 0.0
    assert result.metrics["cw_rope_compression_impulse"] == 0.0
    assert result.metrics["min_string_tension"] > 0.0


def test_traditional_aftermath_runs_as_one_regime_and_stops_at_the_ground():
    result = simulate_trebuchet(traditional_params(), simulate_aftermath=True)

    assert result.aftermath is not None
    assert [seg.regime for seg in result.aftermath.segments] == ["taut"]
    assert result.aftermath.retension_times == []  # nothing to re-tension

    simulator = TrebuchetSimulator(result_params := traditional_params())
    for t in np.linspace(0, result.aftermath.segments[-1].t1, 40):
        theta, theta_dot, _regime = result.aftermath.state_at(float(t))
        psi, psi_dot = result.aftermath.swing_at(float(t))
        (_wx, wy), _v = simulator.weight_position_velocity((theta, theta_dot, 0.0, 0.0, psi, psi_dot))
        # The arm stops when the weight's bottom face lands; it never sinks through.
        assert wy >= result_params.counter_weight_size / 2 - 1e-6


def test_animation_timeline_carries_the_pin_the_counterweight_hangs_from():
    """Both animation frontends draw the linkage from `cw_pin`.

    On the pulley machine the weight hangs from the axle, so the pin never moves; on the
    traditional one the pin rides the arm's short end and swings with it. A pin that sat
    still on the traditional machine would draw the counterweight hanging off the pivot -
    which is exactly the pulley machine, and what both renderers used to show.
    """
    pytest.importorskip("streamlit")
    from trebuchet_sim.web.animation3d import _build_timeline

    for machine, params in (
        (MachineType.PULLEY, TrebuchetParams(**DEFAULT_OPTIMIZABLE_PARAMS)),
        (MachineType.TRADITIONAL, traditional_params()),
    ):
        result = simulate_trebuchet(params, simulate_aftermath=True)
        timeline = _build_timeline(params, result)

        assert timeline["geometry"]["has_pulley"] is (machine is MachineType.PULLEY)
        pins = [frame["cw_pin"] for frame in timeline["launch_frames"]]
        assert all(len(pin) == 2 for pin in pins)
        assert timeline["aftermath_frames"], "aftermath frames drive the post-release pose"
        assert all("cw_pin" in frame for frame in timeline["aftermath_frames"])

        moved = max(abs(pin[0] - pins[0][0]) + abs(pin[1] - pins[0][1]) for pin in pins)
        if machine is MachineType.PULLEY:
            assert pins[0] == [0.0, params.pivot_height]  # the axle
            assert moved == 0.0
        else:
            # Swept through a good fraction of a circle of radius length_counterweight.
            assert moved > params.length_counterweight


def test_traditional_scene_drops_the_pulley_and_draws_the_linkage():
    """The embedded Three.js page builds a different machine per linkage."""
    pytest.importorskip("streamlit")
    from trebuchet_sim.web.animation3d import build_trebuchet_3d_html

    def scene_for(params):
        return build_trebuchet_3d_html(params, simulate_trebuchet(params, simulate_aftermath=True))

    traditional = scene_for(traditional_params())
    assert '"has_pulley": false' in traditional
    # The meshes are created conditionally, so their presence is decided by this flag at
    # runtime; what the payload must carry is the pin track they are drawn along.
    assert '"cw_pin"' in traditional
    assert "backArmMesh" in traditional and "cwLinkMesh" in traditional

    pulley = scene_for(TrebuchetParams(**DEFAULT_OPTIMIZABLE_PARAMS))
    assert '"has_pulley": true' in pulley


def _objective_chosen_for(monkeypatch, machine) -> str:
    """Name of the objective `optimize_trebuchet` hands to differential_evolution.

    differential_evolution is stubbed out rather than run: the question here is purely
    which engine the machine routes to, and answering it by inspecting an actual search's
    output would mean waiting for one to converge.
    """
    from trebuchet_sim import optimization

    captured = {}

    class _StubResult:
        # One value per free parameter, in the order build_params will read them.
        x = [0.0] * len(optimization.param_names(machine))

    def stub(func, bounds, **kwargs):
        captured["objective"] = func.func.__name__  # func is a functools.partial
        captured["vectorized"] = kwargs.get("vectorized", False)
        return _StubResult()

    monkeypatch.setattr(optimization, "differential_evolution", stub)
    monkeypatch.setattr(optimization, "simulate_trebuchet", lambda *a, **k: None)
    optimization.optimize_trebuchet(optimization.OptimizationConfig(machine=machine))
    return captured


@pytest.mark.parametrize("machine", list(MachineType))
def test_optimizer_uses_the_fast_engine_for_either_machine(machine, monkeypatch):
    """fastsim models both linkages, so neither is stuck on the slower scipy objective.

    The traditional machine used to be routed away from the fast engine, because scoring
    it there would have simulated a pulley machine and reported a traditional simulation
    of the winner. Now that fastsim carries the counterweight-swing coordinate too, the
    machine no longer decides the engine.
    """
    pytest.importorskip("numba")

    captured = _objective_chosen_for(monkeypatch, machine)

    assert captured["objective"] == "_objective_vectorized"
    assert captured["vectorized"] is True


def test_an_unset_counterweight_link_falls_back_to_the_machines_own_linkage():
    """The traditional machine has no pulley, so no pulley parameter may size its link.

    It used to inherit the pulley machine's fallback - twice `pulley_radius`, whose default
    is whatever the *other* machine's search last landed on - which made a traditional
    machine's counterweight swing (and so its whole launch) move whenever the pulley
    defaults were re-tuned. That is not a hypothetical: re-sweeping the pulley defaults
    silently changed two fixtures built this way.
    """
    values = dict(DEFAULT_TRADITIONAL_PARAMS)
    params = TrebuchetParams(machine=MachineType.TRADITIONAL, **values)

    assert params.counter_weight_rope_length is None  # nothing explicit to fall back from
    assert params.initial_cw_rope_length == params.length_counterweight
    # And it does not move when the pulley machine's design variables do.
    shifted = TrebuchetParams(
        machine=MachineType.TRADITIONAL, pulley_radius=TrebuchetParams.pulley_radius * 3, **values
    )
    assert shifted.initial_cw_rope_length == params.initial_cw_rope_length
    assert simulate_trebuchet(shifted).distance == simulate_trebuchet(params).distance

    # The pulley machine keeps its own fallback: one wrap of the rope over the axle.
    pulley = TrebuchetParams(**DEFAULT_OPTIMIZABLE_PARAMS)
    assert pulley.initial_cw_rope_length == 2 * pulley.pulley_radius


def test_both_engines_agree_on_the_fallback_link_length():
    """A fallback that only one engine applies would be a silent fork in the physics."""
    pytest.importorskip("numba")
    from trebuchet_sim import fastsim

    values = dict(DEFAULT_TRADITIONAL_PARAMS)
    params = TrebuchetParams(machine=MachineType.TRADITIONAL, **values)
    reference = simulate_trebuchet(params, rtol=1e-6, dense_output=False)

    fast = fastsim.simulate_fast(
        values["counter_weight_mass"], TrebuchetParams.pulley_radius,
        values["length_counterweight"],
        0.0,  # fastsim's "unset" sentinel: numba has no None
        values["arm_length"], values["string_length"], values["release_angle"],
        params.pivot_height, params.pulley_density, params.arm_density,
        params.projectile_mass, params.projectile_radius, params.initial_arm_angle,
        params.arm_drag_coefficient, params.projectile_drag_coefficient,
        params.joint_friction_coefficient, False,
    )

    assert fast[0] is reference.metrics["release_occurred"]
    assert fast[1] == pytest.approx(reference.distance, rel=1e-3)


def test_the_counterweight_link_tension_matches_the_measured_acceleration():
    """The link tension is a constraint force, so it has to agree with the motion.

    The weight rides a massless link pinned to the arm - a two-force member, so the whole
    of its tension acts along the link. `_cw_link_tension` writes that out in closed form;
    this measures the weight's acceleration off the solved launch by central difference and
    projects Newton onto the same line, which is the same statement arrived at without the
    algebra. Anchored at rest, too, where a hanging weight must read exactly its own weight.
    """
    params = traditional_params()
    simulator = TrebuchetSimulator(params)
    result = simulator.simulate()
    sol = result.solution
    m, l_w = params.counter_weight_mass, params.initial_cw_rope_length

    at_rest = simulator._cw_link_tension(
        params.initial_arm_angle, 0.0, 0.0, -np.pi / 2, 0.0
    )
    assert at_rest == pytest.approx(m * G)

    def weight_velocity(t):
        theta, theta_dot, _pos, _vel, psi, psi_dot = sol.full_state(t)
        return np.array(
            simulator.weight_position_velocity((theta, theta_dot, 0.0, 0.0, psi, psi_dot))[1]
        )

    # Differencing the analytic velocity once rather than the position twice: scipy's
    # dense output is a 4th-order interpolant, so its second derivative is the noisiest
    # thing in reach and dividing that noise by h^2 swamps the comparison whatever the
    # solver's rtol. One difference of a velocity the model states in closed form leaves
    # the interpolant as the only error left, which is what the 0.2% bound is - the
    # formula itself is pinned exactly by the at-rest anchor above.
    h = 1e-5
    checked = 0
    for t in np.linspace(4 * h, sol.t_end - 4 * h, 40):
        t = float(t)
        y = sol._y_at(t)[1]
        theta, theta_dot, psi, psi_dot = y[0], y[1], y[4], y[5]
        closed_form = simulator._cw_link_tension(
            theta, theta_dot, simulator.trebuchet_dynamics(t, y)[1], psi, psi_dot
        )
        accel = (weight_velocity(t + h) - weight_velocity(t - h)) / (2 * h)
        link = np.array([np.cos(psi), np.sin(psi)])  # pin -> weight
        # m a = -T * link + m g, projected on the link.
        measured = m * (np.array([0.0, -G]) @ link - accel @ link)
        assert closed_form == pytest.approx(measured, rel=2e-3, abs=1e-2)
        checked += 1

    assert checked == 40
    assert l_w > 0  # there is a link to have a tension at all


def test_a_whipping_counterweight_pushes_its_link_and_is_reported_for_it():
    """The failure mode the pulley machine has always been graded on, on this linkage.

    A short lever swinging a weight on a long link whips it past the pin, and the link has
    to push to keep up - which a rope cannot do. It is the region the optimizer used to
    walk straight into, so the metrics have to see it.
    """
    params = traditional_params(
        counter_weight_mass=50.0, length_counterweight=0.35, arm_length=1.8,
        string_length=1.35, release_angle=-5.016,
    )
    result = simulate_trebuchet(params)

    assert result.metrics["release_occurred"] is True
    assert result.metrics["min_cw_rope_tension"] < -100.0
    assert result.metrics["cw_rope_compression_impulse"] > 1.0
    # And nothing else about the launch complains: the sling is loaded the whole way and
    # the stone never touches the ground, so this is the only term that can catch it.
    assert result.metrics["sling_tension_deficit"] == 0.0
    assert result.metrics["projectile_ground_contacts"] == 0


def test_the_aftermath_carries_the_counterweight_swing_across_release():
    """psi is a live coordinate on this machine, and the launch leaves it mid-swing.

    The post-release machine used never to be handed it, so it restarted the weight hanging
    straight down at rest - a 323 mm jump in the counterweight's position at the instant the
    projectile left, in every animation and in the dynamics that follow it.
    """
    params = traditional_params()
    simulator = TrebuchetSimulator(params)
    result = simulator.simulate(simulate_aftermath=True)

    psi_release, psi_dot_release = result.solution.release_swing_state
    psi_after, psi_dot_after = result.aftermath.swing_at(0.0)

    assert psi_after == pytest.approx(psi_release)
    assert psi_dot_after == pytest.approx(psi_dot_release)
    # The pose it left is not the resting one, so this is a real continuity check rather
    # than two ways of spelling the same number.
    assert psi_release != pytest.approx(-np.pi / 2)

    theta, theta_dot = result.solution.release_machine_state
    before = simulator.weight_position_velocity(
        (theta, theta_dot, 0.0, 0.0, psi_release, psi_dot_release)
    )[0]
    after = simulator.weight_position_velocity(
        (theta, theta_dot, 0.0, 0.0, psi_after, psi_dot_after)
    )[0]
    assert np.hypot(before[0] - after[0], before[1] - after[1]) == pytest.approx(0.0, abs=1e-9)
