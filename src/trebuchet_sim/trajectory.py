"""Ballistic trajectory integration with quadratic air drag.

Shared by the simulation engine (which only needs the final impact distance)
and the visualization module (which needs the full flight path for animation).
"""

import math
from dataclasses import dataclass
from typing import Tuple

from scipy.integrate import solve_ivp

from trebuchet_sim.config import G, RHO_AIR


@dataclass
class BallisticTrajectory:
    """Result of integrating a projectile's flight after release."""

    solution: object  # scipy solve_ivp dense-output solution; None for a flight that never began
    flight_time: float
    impact_x: float
    ground_level: float = 0.0  # height of the projectile's centre when it is resting on the ground

    def position_at(self, t: float) -> Tuple[float, float]:
        """Position (x, y) at time t since release, clamped to ground level."""
        if self.solution is None:
            return self.impact_x, self.ground_level
        if t <= self.solution.t[-1]:
            x, y, _vx, _vy = self.solution.sol(t)
        else:
            x, y, _vx, _vy = self.solution.y[:, -1]
        return x, max(self.ground_level, y)


def integrate_ballistic_trajectory(
    x0: float,
    y0: float,
    vx0: float,
    vy0: float,
    mass: float,
    drag_coefficient: float,
    area: float,
    t_max: float = 60.0,
    rtol: float = 1e-6,
    dense_output: bool = True,
    ground_level: float = 0.0,
) -> BallisticTrajectory:
    """Integrate projectile motion under gravity and quadratic air drag until ground impact.

    `dense_output=False` skips building the per-step interpolant: callers that only
    read `impact_x`/`flight_time` (e.g. the optimizer objective) can opt out, since
    `position_at()` requires the dense solution and isn't usable in that case.

    `ground_level` is the height of the projectile's *centre* when it is resting on the
    ground - its own radius, for a sphere. The launch phase holds a grounded projectile on
    that same line, so a stone that leaves the sling low lands where it would have lain
    rather than sinking one radius further in.
    """

    drag_scale = 0.5 * RHO_AIR * drag_coefficient * area

    # Released at or below the ground: there is no flight to integrate, and asking for one
    # would leave the impact event with no downward crossing to find and run out the clock.
    if y0 <= ground_level:
        return BallisticTrajectory(solution=None, flight_time=0.0, impact_x=max(0.0, x0),
                                   ground_level=ground_level)

    def dynamics(t, state):
        _x, _y, vx, vy = state
        speed = math.hypot(vx, vy)  # math over np: scalar hot path, called every ODE step
        drag_accel = -drag_scale * speed / mass if speed > 1e-12 else 0.0
        return [vx, vy, drag_accel * vx, -G + drag_accel * vy]

    def ground_impact(t, state):
        return state[1] - ground_level

    ground_impact.terminal = True
    ground_impact.direction = -1

    sol = solve_ivp(
        dynamics,
        (0, t_max),
        [x0, y0, vx0, vy0],
        events=ground_impact,
        dense_output=dense_output,
        rtol=rtol,
    )

    if sol.t_events[0].size > 0:
        flight_time = sol.t_events[0][0]
        impact_x = sol.y_events[0][0][0]
    else:
        flight_time = sol.t[-1]
        impact_x = sol.y[0, -1]

    return BallisticTrajectory(solution=sol, flight_time=flight_time, impact_x=max(0.0, impact_x),
                               ground_level=ground_level)
