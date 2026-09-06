"""Traditional (arm-mounted counterweight) machine.

The two machines share one set of equations of motion, so these tests come in pairs:
what the traditional linkage must do, and what adding it must *not* have changed about
the pulley machine.
"""

import numpy as np
import pytest

from trebuchet_sim.config import (
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
    # A 50 kg weight on a 1.8 m arm is a real machine's worth of energy, not a toy:
    # a range this far off would mean the linkage torque has the wrong sign or scale.
    assert 40 < result.distance < 150


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


def test_counterweight_rope_metrics_are_omitted_for_a_pinned_weight():
    """A pinned link is rigid by construction - there is no rope that could go slack,
    so reporting a tension or a compression impulse for one would be meaningless."""
    result = simulate_trebuchet(traditional_params())

    assert "min_cw_rope_tension" not in result.metrics
    assert "cw_rope_compression_impulse" not in result.metrics
    assert "min_string_tension" in result.metrics  # the sling is still a real rope


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
