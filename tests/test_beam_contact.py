"""The projectile's clearance from the beam that is throwing it.

The ground is a static surface and the sling is a rope between two points; the beam is
neither. It is a segment that rotates about the pivot, so "how close is the stone to the
arm" is only a fixed question in the beam's own frame - which is what
`beam_contact_geometry` puts it in, and what these tests pin down.

This is the measurement layer. Nothing here stops the two interpenetrating yet; the
metric exists so that a design which needs them to is visible rather than silently
scored as buildable.
"""

import math

import numpy as np
import pytest

from trebuchet_sim.config import (
    DEFAULT_MACHINE_FIXED,
    DEFAULT_MACHINE_PARAMS,
    MachineType,
    TrebuchetParams,
)
from trebuchet_sim.physics import TrebuchetSimulator, simulate_trebuchet


def default_params(machine: MachineType) -> TrebuchetParams:
    return TrebuchetParams(
        machine=machine, **DEFAULT_MACHINE_PARAMS[machine], **DEFAULT_MACHINE_FIXED[machine]
    )


def _sim(machine=MachineType.TRADITIONAL) -> TrebuchetSimulator:
    return TrebuchetSimulator(default_params(machine))


@pytest.mark.parametrize("theta_deg", [0.0, 37.0, 90.0, -140.5, 200.0, -359.0])
def test_the_beam_frame_is_orthonormal_and_turns_with_the_arm(theta_deg):
    """e along the beam, n a quarter turn ahead of it, both unit, at every angle."""
    sim = _sim()
    ex, ey, nx, ny = sim.beam_frame(math.radians(theta_deg))

    assert math.hypot(ex, ey) == pytest.approx(1.0)
    assert math.hypot(nx, ny) == pytest.approx(1.0)
    assert ex * nx + ey * ny == pytest.approx(0.0, abs=1e-12)
    # Right-handed: e x n = +1, so n is e rotated counter-clockwise and never clockwise.
    assert ex * ny - ey * nx == pytest.approx(1.0)
    # e points where the arm tip is, which is what ties this frame to the rest of the model.
    tip = sim.arm_tip_position_velocity(
        (math.radians(theta_deg), 0.0, 0.0, 0.0, 0.0, 0.0)
    )[0]
    reach = sim.params.arm_length
    assert tip[0] == pytest.approx(reach * ex)
    assert tip[1] - sim.params.pivot_height == pytest.approx(reach * ey)


@pytest.mark.parametrize("theta_deg", [0.0, 55.0, 143.0, -90.0])
def test_clearance_is_measured_in_the_rotating_frame_not_the_world(theta_deg):
    """The same stone-and-beam arrangement reads the same however the pair is turned.

    This is the property the whole contact question rests on: the beam is a surface that
    moves, so a clearance that changed when the machine rotated would be measuring the
    world rather than the gap.
    """
    sim = _sim()
    r_p = sim.params.projectile_radius
    h = sim.params.pivot_height
    theta = math.radians(theta_deg)
    ex, ey, nx, ny = sim.beam_frame(theta)

    # Place the stone at a fixed (s, d) in the beam's frame and rotate the pair together.
    s_want, d_want = 0.4 * sim.params.arm_length, 1.7 * r_p
    px = s_want * ex + d_want * nx
    py = h + s_want * ey + d_want * ny

    s, d, gap, on_span = sim.beam_contact_geometry(theta, px, py)
    assert s == pytest.approx(s_want)
    assert d == pytest.approx(d_want)
    assert gap == pytest.approx(d_want - r_p)
    assert on_span


def test_a_stone_touching_the_beam_reads_exactly_zero_clearance():
    sim = _sim()
    r_p = sim.params.projectile_radius
    theta = math.radians(20.0)
    ex, ey, nx, ny = sim.beam_frame(theta)
    s_want = 0.5 * sim.params.arm_length

    for side in (+1.0, -1.0):
        px = s_want * ex + side * r_p * nx
        py = sim.params.pivot_height + s_want * ey + side * r_p * ny
        _s, d, gap, on_span = sim.beam_contact_geometry(theta, px, py)
        assert gap == pytest.approx(0.0, abs=1e-12)
        assert on_span
        # Which side it rests on is the sign of d, and both sides are legal: the stone
        # can be swung past the beam either way.
        assert math.copysign(1.0, d) == side


def test_a_stone_at_the_pivot_is_a_full_radius_inside_the_beam():
    """The deepest interpenetration the geometry can report on the span, and the sign
    convention that makes 'negative means overlapping' mean something."""
    sim = _sim()
    s, d, gap, on_span = sim.beam_contact_geometry(0.3, 0.0, sim.params.pivot_height)
    assert (s, d) == pytest.approx((0.0, 0.0))
    assert gap == pytest.approx(-sim.params.projectile_radius)
    assert on_span


def test_past_the_end_of_the_beam_the_gap_is_measured_to_the_corner():
    """Off the span the nearest feature is an end, not the centreline.

    Measured to the line, a stone level with the beam but a metre past its tip would
    read as deeply embedded in it. The beam is a segment, and it stops.
    """
    sim = _sim()
    r_p = sim.params.projectile_radius
    theta = 0.0
    beyond = sim.params.arm_length + 0.5

    s, d, gap, on_span = sim.beam_contact_geometry(theta, beyond, sim.params.pivot_height)
    assert not on_span
    assert s == pytest.approx(beyond)
    assert d == pytest.approx(0.0)
    assert gap == pytest.approx(0.5 - r_p)

    # And behind the pivot, where only the traditional machine has beam at all.
    back = sim.params.arm_back_length
    assert back > 0.0
    s_b, _d_b, gap_b, on_span_b = sim.beam_contact_geometry(
        theta, -(back + 0.5), sim.params.pivot_height
    )
    assert not on_span_b
    assert s_b == pytest.approx(-(back + 0.5))
    assert gap_b == pytest.approx(0.5 - r_p)


def test_the_pulley_machine_has_no_beam_behind_its_pivot():
    """Its arm is single-sided, so anything behind the pivot is clear air."""
    sim = _sim(MachineType.PULLEY)
    assert sim.params.arm_back_length == 0.0

    behind = -0.5
    _s, _d, gap, on_span = sim.beam_contact_geometry(0.0, behind, sim.params.pivot_height)
    assert not on_span
    assert gap == pytest.approx(0.5 - sim.params.projectile_radius)


@pytest.mark.parametrize("machine", list(MachineType))
def test_every_launch_reports_its_tightest_approach_to_the_beam(machine):
    result = simulate_trebuchet(default_params(machine))

    assert "min_beam_clearance" in result.metrics
    clearance = result.metrics["min_beam_clearance"]
    assert math.isfinite(clearance)
    # A launch cannot be further from the beam than the machine is big.
    params = default_params(machine)
    assert clearance < params.arm_length + params.string_length


def test_the_shipped_pulley_design_puts_its_stone_through_its_own_beam():
    """The reason this measurement exists, pinned so it cannot quietly change.

    The default pulley machine - optimizer output, and what the dashboard opens on -
    drives the stone about 2 mm into the arm partway through the throw. The model has no
    contact there, so nothing charged it and nothing reported it; the search was free to
    return a design that cannot be built. The traditional machine's own defaults stay
    well clear, which is why this asks them separately.
    """
    pulley = simulate_trebuchet(default_params(MachineType.PULLEY))
    traditional = simulate_trebuchet(default_params(MachineType.TRADITIONAL))

    assert pulley.metrics["min_beam_clearance"] < 0.0
    assert pulley.metrics["min_beam_clearance"] == pytest.approx(-0.002, abs=0.0015)
    assert traditional.metrics["min_beam_clearance"] > 0.1


def test_interpenetration_is_common_across_the_search_space():
    """Not a corner case: the optimizer searches a space where most designs do it.

    Sampled rather than exhaustive, and asserted loosely - the point is the order of
    magnitude, which is what decides whether beam contact is worth modelling at all.
    """
    from trebuchet_sim.optimization import PARAM_BOUNDS, param_names

    rng = np.random.default_rng(4242)
    machine = MachineType.PULLEY
    overlapping = checked = 0
    for _ in range(60):
        values = {name: rng.uniform(*PARAM_BOUNDS[name]) for name in param_names(machine)}
        pivot = max(TrebuchetParams.pivot_height, values["arm_length"] + 0.25)
        params = TrebuchetParams(machine=machine, pivot_height=pivot, **values)
        if params.string_length > 0.95 * params.arm_length:
            continue
        result = simulate_trebuchet(params, rtol=1e-6, dense_output=False)
        if "error" in result.metrics:
            continue
        checked += 1
        if result.metrics["min_beam_clearance"] < 0.0:
            overlapping += 1

    assert checked > 20
    assert overlapping > checked // 4
