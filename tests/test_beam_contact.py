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


def test_the_shipped_pulley_design_now_rides_its_beam_instead_of_passing_through():
    """The design that motivated all of this, and what contact did to it.

    The default pulley machine used to drive its stone about 2 mm into its own arm
    partway through the throw, and nothing reported or charged it. It no longer can.

    What it does instead is start *touching*: initial_state offsets the hanging sling by
    arcsin(r_p / l_s), which is exactly the angle that lays the stone against the beam,
    so the cocked pose is tangent by construction rather than by accident. The launch
    therefore opens in TAUT_BEAM, rides the arm for the first stretch, and only then
    flies - which is why `beam_contacts` is zero here and not one. A strike and a start
    in contact are different things, the same way a machine loaded on the ground has a
    ground *fraction* but no ground *contacts*.

    It costs this design most of its range, because it was buying that range partly by
    passing through the arm: 30.0 m before, about 16 m now, at the same efficiency and
    the same release pin angle - the throw is aimed differently, not weakened. That is
    the re-derivation this change forces on the shipped defaults.
    """
    result = simulate_trebuchet(default_params(MachineType.PULLEY))

    assert result.metrics["min_beam_clearance"] >= -1e-9
    assert result.solution.segments[0].regime == "taut_beam"
    assert result.metrics["beam_contacts"] == 0
    assert result.metrics["release_occurred"] is True


def test_contact_removes_most_but_not_yet_all_interpenetration():
    """Where the constraint stands, measured rather than claimed.

    Before contact existed, 61% of pulley draws inside PARAM_BOUNDS put the stone into
    the arm, a median 25 mm into a 40 mm ball. With it, the median case is gone - what
    remains reads at the tangency rounding - but the worst still reaches a few
    millimetres, because entering contact is detected by a sign change in the clearance
    at the ends of accepted steps and a shallow dip can begin and end inside one. That is
    the same failure `physics._first_arm_ground_angle` avoids by testing an angle instead
    of a clearance, and the same one `fastsim.RETENSION_CIRCLE_SLOP` guards by rejecting
    an over-long step outright.

    Asserted as a bound rather than an equality so it records the state honestly: the
    typical draw must be clean, and the worst is allowed to be a few millimetres until
    the crossing test is made robust.
    """
    from trebuchet_sim.optimization import PARAM_BOUNDS, param_names

    rng = np.random.default_rng(4242)
    machine = MachineType.PULLEY
    worst = 0.0
    penetrations = []
    checked = 0
    for _ in range(50):
        values = {name: rng.uniform(*PARAM_BOUNDS[name]) for name in param_names(machine)}
        pivot = max(TrebuchetParams.pivot_height, values["arm_length"] + 0.25)
        params = TrebuchetParams(machine=machine, pivot_height=pivot, **values)
        if params.string_length > 0.95 * params.arm_length:
            continue
        result = simulate_trebuchet(params, rtol=1e-6, dense_output=False)
        if "error" in result.metrics:
            continue
        checked += 1
        clearance = result.metrics["min_beam_clearance"]
        worst = min(worst, clearance)
        if clearance < -1e-9:
            penetrations.append(clearance)

    assert checked > 15
    penetrations.sort()
    # The median draw no longer ends up inside the beam to any depth that matters.
    if penetrations:
        median = penetrations[len(penetrations) // 2]
        assert median > -1e-3, f"median penetration {median * 1000:.2f} mm"
    # And nothing is anywhere near the 25-37 mm the model used to allow.
    assert worst > -0.010, f"worst penetration {worst * 1000:.2f} mm"


@pytest.mark.parametrize("machine", list(MachineType))
def test_a_launch_that_never_nears_the_beam_is_untouched_by_any_of_this(machine):
    """The traditional machine keeps its stone 280 mm clear, so contact must be inert
    for it - no regimes entered, no energy charged, and the throw it had before."""
    result = simulate_trebuchet(default_params(MachineType.TRADITIONAL))

    assert result.metrics["min_beam_clearance"] > 0.1
    assert result.metrics["beam_contacts"] == 0
    assert result.metrics["beam_contact_energy"] == 0.0
    assert not any(
        seg.regime in ("taut_beam", "slack_beam") for seg in result.solution.segments
    )
    assert result.distance == pytest.approx(30.0, abs=0.05)
