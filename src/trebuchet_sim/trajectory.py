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

    solution: object  # scipy solve_ivp dense-output solution
    flight_time: float
    impact_x: float

    def position_at(self, t: float) -> Tuple[float, float]:
        """Position (x, y) at time t since release, clamped to ground level."""
        if t <= self.solution.t[-1]:
            x, y, _vx, _vy = self.solution.sol(t)
        else:
            x, y, _vx, _vy = self.solution.y[:, -1]
        return x, max(0.0, y)


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
) -> BallisticTrajectory:
    """Integrate projectile motion under gravity and quadratic air drag until ground impact.

    `dense_output=False` skips building the per-step interpolant: callers that only
    read `impact_x`/`flight_time` (e.g. the optimizer objective) can opt out, since
    `position_at()` requires the dense solution and isn't usable in that case.
    """

    drag_scale = 0.5 * RHO_AIR * drag_coefficient * area

    def dynamics(t, state):
        _x, _y, vx, vy = state
        speed = math.hypot(vx, vy)  # math over np: scalar hot path, called every ODE step
        drag_accel = -drag_scale * speed / mass if speed > 1e-12 else 0.0
        return [vx, vy, drag_accel * vx, -G + drag_accel * vy]

    def ground_impact(t, state):
        return state[1]

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

    return BallisticTrajectory(solution=sol, flight_time=flight_time, impact_x=max(0.0, impact_x))
