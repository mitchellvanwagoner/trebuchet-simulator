import math

from trebuchet_sim.trajectory import integrate_ballistic_trajectory


def test_projectile_launched_horizontally_lands_downrange():
    trajectory = integrate_ballistic_trajectory(
        x0=0.0, y0=5.0, vx0=10.0, vy0=0.0,
        mass=0.25, drag_coefficient=0.47, area=math.pi * 0.04**2,
    )

    assert trajectory.impact_x > 0
    assert trajectory.flight_time > 0


def test_position_at_zero_matches_initial_conditions():
    trajectory = integrate_ballistic_trajectory(
        x0=1.0, y0=2.0, vx0=3.0, vy0=4.0,
        mass=0.25, drag_coefficient=0.47, area=math.pi * 0.04**2,
    )

    x, y = trajectory.position_at(0.0)
    assert x == 1.0
    assert y == 2.0


def test_velocity_starts_at_the_release_and_ends_at_rest():
    """The energy plot reads the stone's speed through the flight (see
    physics.post_release_energy_at), and a landed stone has given the ground the lot."""
    trajectory = integrate_ballistic_trajectory(
        x0=1.0, y0=2.0, vx0=3.0, vy0=4.0,
        mass=0.25, drag_coefficient=0.47, area=math.pi * 0.04**2,
    )

    assert trajectory.velocity_at(0.0) == (3.0, 4.0)
    assert trajectory.velocity_at(trajectory.flight_time + 0.5) == (0.0, 0.0)
    # Drag is the only horizontal force, so the downrange speed only ever falls.
    assert trajectory.velocity_at(0.1)[0] < 3.0


def test_drag_reduces_range_versus_vacuum_trajectory():
    with_drag = integrate_ballistic_trajectory(
        x0=0.0, y0=5.0, vx0=20.0, vy0=0.0,
        mass=0.05, drag_coefficient=1.5, area=0.01,
    )
    without_drag = integrate_ballistic_trajectory(
        x0=0.0, y0=5.0, vx0=20.0, vy0=0.0,
        mass=0.05, drag_coefficient=0.0, area=0.01,
    )

    assert with_drag.impact_x < without_drag.impact_x
