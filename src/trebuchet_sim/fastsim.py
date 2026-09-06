"""Numba-JIT fast physics engine, used only by the optimizer's objective function.

`physics.py` (scipy `solve_ivp`, dense output, full metrics/energy tracking) remains
the reference implementation and the engine behind every animation and displayed
result. This module re-implements just the launch dynamics (`trebuchet_dynamics` in
physics.py) and the ballistic flight (`trajectory.py`) as scalar, JIT-compiled code so
`optimization.py` can evaluate an entire differential-evolution population per call
instead of one simulation per Python-level objective call.

Because numba's nopython mode can't call TrebuchetParams properties or scipy's
solve_ivp, the machine-constant formulas and the Dormand-Prince (RK45) integrator are
duplicated here in scalar form. Keep this numerically consistent with physics.py and
trajectory.py whenever the equations of motion change - `tests/test_fastsim.py` checks
agreement against the scipy engine across the parameter space.

Both counterweight linkages are supported (`has_pulley`, mirroring config.MachineType).
As in physics.py, one set of equations covers them: the state carries the counterweight's
swing psi, which is live on the traditional machine and inert on the pulley one, where
M13 = 0 and M33 = 1 collapse the 3x3 solve back to a two-coordinate one. psi is carried
through the tableau and the error norm on both machines, exactly as it is under scipy, so
the two engines take the same step path rather than merely solving the same equations.

One deliberate divergence: physics.py simulates the sling going slack (free-flight
projectile + inelastic re-tension snaps), while this engine keeps the rigid-sling
model and instead reports the string/cw-rope compression impulses that the objective
penalizes (`slack_penalty_weight`). The two models coincide exactly on trajectories
whose ropes stay taut - which is precisely where the penalty drives the optimizer, so
the search result is always in the regime where this engine is faithful.

`sling_deficit` is what keeps it there. A compression impulse is zero right up to the
instant the sling detaches, so on its own it gives the search nothing to descend until
the design has already failed; the deficit measures how close to slack the sling ran
(physics._tension_metrics computes the identical quantity from the reference engine's
own steps) and so charges a jerky design before it snaps.

The integrator mirrors scipy's RK45: same Dormand-Prince tableau, same step-size
control. Events (release angle, ground impact) are localized with a cubic Hermite
interpolant built from the bracketing step's endpoint states/derivatives, refined by
bisection - cheaper than scipy's dense-output event solver and accurate enough for the
optimizer's tolerance (see the `rtol=1e-6` note in `optimization._objective`).
"""

import numpy as np
from numba import njit, prange

from trebuchet_sim.config import ARM_CROSS_SECTION_WIDTH, G, RHO_AIR, SLING_TENSION_FLOOR

PULLEY_THICKNESS = 0.0254  # m; matches TrebuchetParams.PULLEY_THICKNESS

INVALID_COST = 1e6

# Dormand-Prince RK45 tableau (identical to scipy.integrate._ivp.rk.RK45)
A21 = 1 / 5
A31, A32 = 3 / 40, 9 / 40
A41, A42, A43 = 44 / 45, -56 / 15, 32 / 9
A51, A52, A53, A54 = 19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729
A61, A62, A63, A64, A65 = 9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656

B1, B3, B4, B5, B6 = 35 / 384, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84

# Error coefficients: 5th-order minus embedded 4th-order weights (B2 = B7 = 0)
E1 = B1 - 5179 / 57600
E3 = B3 - 7571 / 16695
E4 = B4 - 393 / 640
E5 = B5 - (-92097 / 339200)
E6 = B6 - 187 / 2100
E7 = -1 / 40

SAFETY = 0.9
MIN_FACTOR = 0.2
MAX_FACTOR = 10.0
ERROR_EXPONENT = -1.0 / 5.0
MAX_STEPS = 20000


@njit(cache=True, fastmath=True, inline="always")
def _hermite(y0, y1, f0, f1, h, s):
    """Cubic Hermite interpolant at fraction s in [0, 1] of a step of size h."""
    h00 = 2 * s**3 - 3 * s**2 + 1
    h10 = s**3 - 2 * s**2 + s
    h01 = -2 * s**3 + 3 * s**2
    h11 = s**3 - s**2
    return y0 * h00 + h * f0 * h10 + y1 * h01 + h * f1 * h11


# Index map for the `c` constants tuple threaded through the launch integrator. Packing
# them into one tuple rather than passing ~20 scalars keeps the RK45 stage calls below
# readable; numba unpacks a homogeneous tuple at compile time, so it costs nothing.
#   0 l_a                 6 cw_swing_coupling    12 arm_gravity_k
#   1 l_s                 7 arm_drag_k           13 proj_gravity_theta_k
#   2 M11                 8 proj_drag_k          14 proj_gravity_alpha_k
#   3 M22                 9 cw_torque_const      15 joint_friction
#   4 M33                10 cw_torque_cos
#   5 coupling           11 cw_swing_gravity_k


@njit(cache=True, fastmath=True, inline="always")
def _trebuchet_dynamics(theta, theta_dot, alpha, alpha_dot, psi, psi_dot, c):
    """Scalar port of physics.TrebuchetSimulator.trebuchet_dynamics.

    Six states, like the reference engine: psi is the counterweight's swing about its
    pin, live on the traditional machine and inert on the pulley one, where M13 = 0 and
    M33 = 1 collapse the 3x3 solve back to the original two-coordinate one exactly.
    """
    (l_a, l_s, M11, M22, M33, coupling, cw_swing_coupling, arm_drag_k, proj_drag_k,
     cw_torque_const, cw_torque_cos, cw_swing_gravity_k, arm_gravity_k,
     proj_gravity_theta_k, proj_gravity_alpha_k, joint_friction) = c

    sin_t, cos_t = np.sin(theta), np.cos(theta)
    sin_a, cos_a = np.sin(alpha), np.cos(alpha)
    sin_p, cos_p = np.sin(psi), np.cos(psi)
    sin_at = sin_a * cos_t - cos_a * sin_t
    cos_at = cos_a * cos_t + sin_a * sin_t
    sin_pt = sin_p * cos_t - cos_p * sin_t
    cos_pt = cos_p * cos_t + sin_p * sin_t

    p_vx = -l_a * theta_dot * sin_t - l_s * alpha_dot * sin_a
    p_vy = l_a * theta_dot * cos_t + l_s * alpha_dot * cos_a
    proj_speed = np.sqrt(p_vx * p_vx + p_vy * p_vy)

    M12 = coupling * cos_at
    M13 = -cw_swing_coupling * cos_pt

    arm_drag_torque = -np.copysign(arm_drag_k * theta_dot * theta_dot, theta_dot)

    drag_scale = proj_drag_k * proj_speed
    drag_fx, drag_fy = -drag_scale * p_vx, -drag_scale * p_vy
    Q_theta_drag = drag_fx * (-l_a * sin_t) + drag_fy * (l_a * cos_t)
    Q_alpha_drag = drag_fx * (-l_s * sin_a) + drag_fy * (l_s * cos_a)

    Q_theta = (
        coupling * sin_at * alpha_dot * alpha_dot
        - cw_swing_coupling * sin_pt * psi_dot * psi_dot
        + cw_torque_const + cw_torque_cos * cos_t
        - arm_gravity_k * cos_t
        - proj_gravity_theta_k * cos_t
        + arm_drag_torque
        + Q_theta_drag
        - joint_friction * theta_dot
    )
    Q_alpha = (
        -coupling * sin_at * theta_dot * theta_dot
        - proj_gravity_alpha_k * cos_a
        + Q_alpha_drag
    )
    Q_psi = cw_swing_coupling * sin_pt * theta_dot * theta_dot - cw_swing_gravity_k * cos_p

    # Symmetric 3x3 with M23 = 0, solved via its adjugate (same form as physics.py).
    A, B, C, D, E = M11, M12, M13, M22, M33
    det = A * D * E - B * B * E - C * C * D
    if abs(det) < 1e-12:
        return theta_dot, 0.0, alpha_dot, 0.0, psi_dot, 0.0

    theta_ddot = (D * E * Q_theta - B * E * Q_alpha - C * D * Q_psi) / det
    alpha_ddot = (-B * E * Q_theta + (A * E - C * C) * Q_alpha + B * C * Q_psi) / det
    psi_ddot = (-C * D * Q_theta + B * C * Q_alpha + (A * D - B * B) * Q_psi) / det
    return theta_dot, theta_ddot, alpha_dot, alpha_ddot, psi_dot, psi_ddot


@njit(cache=True, fastmath=True, inline="always")
def _tensions(theta, theta_dot, alpha, alpha_dot, theta_ddot, l_a, l_s, proj_drag_k,
              projectile_mass, counter_weight_mass, pulley_radius, has_pulley):
    """Scalar port of physics.TrebuchetSimulator.constraint_tensions (theta_ddot passed in).

    The counterweight rope only exists on the pulley machine; a pinned link is rigid by
    construction and can't go slack, so 0.0 is returned for it - which contributes
    nothing to the compression impulse the caller accumulates.
    """
    sin_t, cos_t = np.sin(theta), np.cos(theta)
    sin_a, cos_a = np.sin(alpha), np.cos(alpha)
    sin_at = sin_a * cos_t - cos_a * sin_t
    cos_at = cos_a * cos_t + sin_a * sin_t

    p_vx = -l_a * theta_dot * sin_t - l_s * alpha_dot * sin_a
    p_vy = l_a * theta_dot * cos_t + l_s * alpha_dot * cos_a
    drag_scale = proj_drag_k * np.sqrt(p_vx * p_vx + p_vy * p_vy)

    radial_acc = l_a * theta_ddot * sin_at - l_a * theta_dot**2 * cos_at - l_s * alpha_dot**2
    string_tension = (
        -projectile_mass * G * sin_a
        - drag_scale * (p_vx * cos_a + p_vy * sin_a)
        - projectile_mass * radial_acc
    )
    cw_tension = counter_weight_mass * (G + pulley_radius * theta_ddot) if has_pulley else 0.0
    return string_tension, cw_tension


@njit(cache=True, fastmath=True)
def _integrate_launch(theta0, alpha0, psi0, c, release_angle, t_max, rtol, atol,
                      projectile_mass, counter_weight_mass, pulley_radius, has_pulley,
                      tension_floor):
    """Integrate launch dynamics until the release-angle event fires.

    Returns (released, t, theta, theta_dot, alpha, alpha_dot, psi, psi_dot,
    string_impulse, cw_impulse, sling_deficit): the state at release (or the final
    state at t_max if the arm never reached the release angle) plus the string/cw-rope
    compression impulses and the integral of (tension_floor - clamp(T_string, 0,
    tension_floor)) dt (trapezoid rule over accepted steps, mirroring
    physics.TrebuchetSimulator._tension_metrics).

    Clamping at zero is what keeps the two penalties from measuring the same thing: a
    rope that has gone slack is exactly as limp at -400 N of rigid-link compression as
    at 0, and how deep into the impossible the rigid model went is what string_impulse
    is for. Without the clamp the deficit would grow without bound down there and stop
    being the [0, 1] share of the launch that physics.py reports.

    The six components run through the Dormand-Prince tableau unrolled, matching the
    reference engine's state layout - including on the pulley machine, where psi stays
    inert but still counts toward the error norm exactly as it does under scipy.
    """
    l_a, l_s = c[0], c[1]
    proj_drag_k = c[8]

    t = 0.0
    theta, theta_dot, alpha, alpha_dot, psi, psi_dot = theta0, 0.0, alpha0, 0.0, psi0, 0.0
    f0_1, f0_2, f0_3, f0_4, f0_5, f0_6 = _trebuchet_dynamics(
        theta, theta_dot, alpha, alpha_dot, psi, psi_dot, c
    )

    h = 1e-3
    g_prev = theta - release_angle

    string_impulse = 0.0
    cw_impulse = 0.0
    sling_deficit = 0.0
    string_T_prev, cw_T_prev = _tensions(
        theta, theta_dot, alpha, alpha_dot, f0_2, l_a, l_s, proj_drag_k,
        projectile_mass, counter_weight_mass, pulley_radius, has_pulley,
    )

    for _ in range(MAX_STEPS):
        if t >= t_max:
            return (False, t, theta, theta_dot, alpha, alpha_dot, psi, psi_dot,
                    string_impulse, cw_impulse, sling_deficit)
        if t + h > t_max:
            h = t_max - t

        y2_1 = theta + h * A21 * f0_1
        y2_2 = theta_dot + h * A21 * f0_2
        y2_3 = alpha + h * A21 * f0_3
        y2_4 = alpha_dot + h * A21 * f0_4
        y2_5 = psi + h * A21 * f0_5
        y2_6 = psi_dot + h * A21 * f0_6
        k2_1, k2_2, k2_3, k2_4, k2_5, k2_6 = _trebuchet_dynamics(y2_1, y2_2, y2_3, y2_4, y2_5, y2_6, c)

        y3_1 = theta + h * (A31 * f0_1 + A32 * k2_1)
        y3_2 = theta_dot + h * (A31 * f0_2 + A32 * k2_2)
        y3_3 = alpha + h * (A31 * f0_3 + A32 * k2_3)
        y3_4 = alpha_dot + h * (A31 * f0_4 + A32 * k2_4)
        y3_5 = psi + h * (A31 * f0_5 + A32 * k2_5)
        y3_6 = psi_dot + h * (A31 * f0_6 + A32 * k2_6)
        k3_1, k3_2, k3_3, k3_4, k3_5, k3_6 = _trebuchet_dynamics(y3_1, y3_2, y3_3, y3_4, y3_5, y3_6, c)

        y4_1 = theta + h * (A41 * f0_1 + A42 * k2_1 + A43 * k3_1)
        y4_2 = theta_dot + h * (A41 * f0_2 + A42 * k2_2 + A43 * k3_2)
        y4_3 = alpha + h * (A41 * f0_3 + A42 * k2_3 + A43 * k3_3)
        y4_4 = alpha_dot + h * (A41 * f0_4 + A42 * k2_4 + A43 * k3_4)
        y4_5 = psi + h * (A41 * f0_5 + A42 * k2_5 + A43 * k3_5)
        y4_6 = psi_dot + h * (A41 * f0_6 + A42 * k2_6 + A43 * k3_6)
        k4_1, k4_2, k4_3, k4_4, k4_5, k4_6 = _trebuchet_dynamics(y4_1, y4_2, y4_3, y4_4, y4_5, y4_6, c)

        y5_1 = theta + h * (A51 * f0_1 + A52 * k2_1 + A53 * k3_1 + A54 * k4_1)
        y5_2 = theta_dot + h * (A51 * f0_2 + A52 * k2_2 + A53 * k3_2 + A54 * k4_2)
        y5_3 = alpha + h * (A51 * f0_3 + A52 * k2_3 + A53 * k3_3 + A54 * k4_3)
        y5_4 = alpha_dot + h * (A51 * f0_4 + A52 * k2_4 + A53 * k3_4 + A54 * k4_4)
        y5_5 = psi + h * (A51 * f0_5 + A52 * k2_5 + A53 * k3_5 + A54 * k4_5)
        y5_6 = psi_dot + h * (A51 * f0_6 + A52 * k2_6 + A53 * k3_6 + A54 * k4_6)
        k5_1, k5_2, k5_3, k5_4, k5_5, k5_6 = _trebuchet_dynamics(y5_1, y5_2, y5_3, y5_4, y5_5, y5_6, c)

        y6_1 = theta + h * (A61 * f0_1 + A62 * k2_1 + A63 * k3_1 + A64 * k4_1 + A65 * k5_1)
        y6_2 = theta_dot + h * (A61 * f0_2 + A62 * k2_2 + A63 * k3_2 + A64 * k4_2 + A65 * k5_2)
        y6_3 = alpha + h * (A61 * f0_3 + A62 * k2_3 + A63 * k3_3 + A64 * k4_3 + A65 * k5_3)
        y6_4 = alpha_dot + h * (A61 * f0_4 + A62 * k2_4 + A63 * k3_4 + A64 * k4_4 + A65 * k5_4)
        y6_5 = psi + h * (A61 * f0_5 + A62 * k2_5 + A63 * k3_5 + A64 * k4_5 + A65 * k5_5)
        y6_6 = psi_dot + h * (A61 * f0_6 + A62 * k2_6 + A63 * k3_6 + A64 * k4_6 + A65 * k5_6)
        k6_1, k6_2, k6_3, k6_4, k6_5, k6_6 = _trebuchet_dynamics(y6_1, y6_2, y6_3, y6_4, y6_5, y6_6, c)

        yn_1 = theta + h * (B1 * f0_1 + B3 * k3_1 + B4 * k4_1 + B5 * k5_1 + B6 * k6_1)
        yn_2 = theta_dot + h * (B1 * f0_2 + B3 * k3_2 + B4 * k4_2 + B5 * k5_2 + B6 * k6_2)
        yn_3 = alpha + h * (B1 * f0_3 + B3 * k3_3 + B4 * k4_3 + B5 * k5_3 + B6 * k6_3)
        yn_4 = alpha_dot + h * (B1 * f0_4 + B3 * k3_4 + B4 * k4_4 + B5 * k5_4 + B6 * k6_4)
        yn_5 = psi + h * (B1 * f0_5 + B3 * k3_5 + B4 * k4_5 + B5 * k5_5 + B6 * k6_5)
        yn_6 = psi_dot + h * (B1 * f0_6 + B3 * k3_6 + B4 * k4_6 + B5 * k5_6 + B6 * k6_6)
        k7_1, k7_2, k7_3, k7_4, k7_5, k7_6 = _trebuchet_dynamics(yn_1, yn_2, yn_3, yn_4, yn_5, yn_6, c)

        err_1 = h * (E1 * f0_1 + E3 * k3_1 + E4 * k4_1 + E5 * k5_1 + E6 * k6_1 + E7 * k7_1)
        err_2 = h * (E1 * f0_2 + E3 * k3_2 + E4 * k4_2 + E5 * k5_2 + E6 * k6_2 + E7 * k7_2)
        err_3 = h * (E1 * f0_3 + E3 * k3_3 + E4 * k4_3 + E5 * k5_3 + E6 * k6_3 + E7 * k7_3)
        err_4 = h * (E1 * f0_4 + E3 * k3_4 + E4 * k4_4 + E5 * k5_4 + E6 * k6_4 + E7 * k7_4)
        err_5 = h * (E1 * f0_5 + E3 * k3_5 + E4 * k4_5 + E5 * k5_5 + E6 * k6_5 + E7 * k7_5)
        err_6 = h * (E1 * f0_6 + E3 * k3_6 + E4 * k4_6 + E5 * k5_6 + E6 * k6_6 + E7 * k7_6)

        scale_1 = atol + rtol * max(abs(theta), abs(yn_1))
        scale_2 = atol + rtol * max(abs(theta_dot), abs(yn_2))
        scale_3 = atol + rtol * max(abs(alpha), abs(yn_3))
        scale_4 = atol + rtol * max(abs(alpha_dot), abs(yn_4))
        scale_5 = atol + rtol * max(abs(psi), abs(yn_5))
        scale_6 = atol + rtol * max(abs(psi_dot), abs(yn_6))

        err_norm = np.sqrt(
            ((err_1 / scale_1) ** 2 + (err_2 / scale_2) ** 2 + (err_3 / scale_3) ** 2
             + (err_4 / scale_4) ** 2 + (err_5 / scale_5) ** 2 + (err_6 / scale_6) ** 2) / 6.0
        )

        if err_norm <= 1.0:
            g_new = yn_1 - release_angle
            if g_prev > 0.0 and g_new <= 0.0:
                lo, hi = 0.0, 1.0
                for _ in range(50):
                    mid = 0.5 * (lo + hi)
                    g_mid = _hermite(theta, yn_1, f0_1, k7_1, h, mid) - release_angle
                    if g_mid > 0.0:
                        lo = mid
                    else:
                        hi = mid
                s = 0.5 * (lo + hi)
                theta_r = _hermite(theta, yn_1, f0_1, k7_1, h, s)
                theta_dot_r = _hermite(theta_dot, yn_2, f0_2, k7_2, h, s)
                alpha_r = _hermite(alpha, yn_3, f0_3, k7_3, h, s)
                alpha_dot_r = _hermite(alpha_dot, yn_4, f0_4, k7_4, h, s)
                psi_r = _hermite(psi, yn_5, f0_5, k7_5, h, s)
                psi_dot_r = _hermite(psi_dot, yn_6, f0_6, k7_6, h, s)
                _, theta_ddot_r, _, _, _, _ = _trebuchet_dynamics(
                    theta_r, theta_dot_r, alpha_r, alpha_dot_r, psi_r, psi_dot_r, c
                )
                string_T_r, cw_T_r = _tensions(
                    theta_r, theta_dot_r, alpha_r, alpha_dot_r, theta_ddot_r, l_a, l_s,
                    proj_drag_k, projectile_mass, counter_weight_mass, pulley_radius, has_pulley,
                )
                string_impulse += 0.5 * (max(0.0, -string_T_prev) + max(0.0, -string_T_r)) * h * s
                cw_impulse += 0.5 * (max(0.0, -cw_T_prev) + max(0.0, -cw_T_r)) * h * s
                sling_deficit += 0.5 * (
                    tension_floor - min(tension_floor, max(0.0, string_T_prev))
                    + tension_floor - min(tension_floor, max(0.0, string_T_r))
                ) * h * s
                return (True, t + h * s, theta_r, theta_dot_r, alpha_r, alpha_dot_r,
                        psi_r, psi_dot_r, string_impulse, cw_impulse, sling_deficit)

            string_T_new, cw_T_new = _tensions(
                yn_1, yn_2, yn_3, yn_4, k7_2, l_a, l_s, proj_drag_k,
                projectile_mass, counter_weight_mass, pulley_radius, has_pulley,
            )
            string_impulse += 0.5 * (max(0.0, -string_T_prev) + max(0.0, -string_T_new)) * h
            cw_impulse += 0.5 * (max(0.0, -cw_T_prev) + max(0.0, -cw_T_new)) * h
            sling_deficit += 0.5 * (
                tension_floor - min(tension_floor, max(0.0, string_T_prev))
                + tension_floor - min(tension_floor, max(0.0, string_T_new))
            ) * h
            string_T_prev, cw_T_prev = string_T_new, cw_T_new

            t = t + h
            theta, theta_dot, alpha, alpha_dot, psi, psi_dot = yn_1, yn_2, yn_3, yn_4, yn_5, yn_6
            f0_1, f0_2, f0_3, f0_4, f0_5, f0_6 = k7_1, k7_2, k7_3, k7_4, k7_5, k7_6
            g_prev = g_new

            factor = MAX_FACTOR if err_norm == 0.0 else min(MAX_FACTOR, SAFETY * err_norm**ERROR_EXPONENT)
            h = h * factor
        else:
            factor = max(MIN_FACTOR, SAFETY * err_norm**ERROR_EXPONENT)
            h = h * factor

    return (False, t, theta, theta_dot, alpha, alpha_dot, psi, psi_dot,
            string_impulse, cw_impulse, sling_deficit)


@njit(cache=True, fastmath=True, inline="always")
def _ballistic_dynamics(vx, vy, drag_scale, mass):
    speed = np.sqrt(vx * vx + vy * vy)
    drag_accel = -drag_scale * speed / mass if speed > 1e-12 else 0.0
    return vx, vy, drag_accel * vx, -G + drag_accel * vy


@njit(cache=True, fastmath=True)
def _integrate_ballistic(x0, y0, vx0, vy0, mass, drag_coefficient, area, t_max, rtol, atol):
    """Integrate ballistic flight (with quadratic drag) until ground impact; returns impact_x."""
    if y0 <= 0.0:
        return x0

    drag_scale = 0.5 * RHO_AIR * drag_coefficient * area

    t = 0.0
    x, y, vx, vy = x0, y0, vx0, vy0
    f0_1, f0_2, f0_3, f0_4 = _ballistic_dynamics(vx, vy, drag_scale, mass)

    h = 1e-2
    g_prev = y

    for _ in range(MAX_STEPS):
        if t >= t_max:
            return x
        if t + h > t_max:
            h = t_max - t

        y2_1 = x + h * A21 * f0_1
        y2_2 = y + h * A21 * f0_2
        y2_3 = vx + h * A21 * f0_3
        y2_4 = vy + h * A21 * f0_4
        k2_1, k2_2, k2_3, k2_4 = _ballistic_dynamics(y2_3, y2_4, drag_scale, mass)

        y3_1 = x + h * (A31 * f0_1 + A32 * k2_1)
        y3_2 = y + h * (A31 * f0_2 + A32 * k2_2)
        y3_3 = vx + h * (A31 * f0_3 + A32 * k2_3)
        y3_4 = vy + h * (A31 * f0_4 + A32 * k2_4)
        k3_1, k3_2, k3_3, k3_4 = _ballistic_dynamics(y3_3, y3_4, drag_scale, mass)

        y4_1 = x + h * (A41 * f0_1 + A42 * k2_1 + A43 * k3_1)
        y4_2 = y + h * (A41 * f0_2 + A42 * k2_2 + A43 * k3_2)
        y4_3 = vx + h * (A41 * f0_3 + A42 * k2_3 + A43 * k3_3)
        y4_4 = vy + h * (A41 * f0_4 + A42 * k2_4 + A43 * k3_4)
        k4_1, k4_2, k4_3, k4_4 = _ballistic_dynamics(y4_3, y4_4, drag_scale, mass)

        y5_1 = x + h * (A51 * f0_1 + A52 * k2_1 + A53 * k3_1 + A54 * k4_1)
        y5_2 = y + h * (A51 * f0_2 + A52 * k2_2 + A53 * k3_2 + A54 * k4_2)
        y5_3 = vx + h * (A51 * f0_3 + A52 * k2_3 + A53 * k3_3 + A54 * k4_3)
        y5_4 = vy + h * (A51 * f0_4 + A52 * k2_4 + A53 * k3_4 + A54 * k4_4)
        k5_1, k5_2, k5_3, k5_4 = _ballistic_dynamics(y5_3, y5_4, drag_scale, mass)

        y6_1 = x + h * (A61 * f0_1 + A62 * k2_1 + A63 * k3_1 + A64 * k4_1 + A65 * k5_1)
        y6_2 = y + h * (A61 * f0_2 + A62 * k2_2 + A63 * k3_2 + A64 * k4_2 + A65 * k5_2)
        y6_3 = vx + h * (A61 * f0_3 + A62 * k2_3 + A63 * k3_3 + A64 * k4_3 + A65 * k5_3)
        y6_4 = vy + h * (A61 * f0_4 + A62 * k2_4 + A63 * k3_4 + A64 * k4_4 + A65 * k5_4)
        k6_1, k6_2, k6_3, k6_4 = _ballistic_dynamics(y6_3, y6_4, drag_scale, mass)

        yn_1 = x + h * (B1 * f0_1 + B3 * k3_1 + B4 * k4_1 + B5 * k5_1 + B6 * k6_1)
        yn_2 = y + h * (B1 * f0_2 + B3 * k3_2 + B4 * k4_2 + B5 * k5_2 + B6 * k6_2)
        yn_3 = vx + h * (B1 * f0_3 + B3 * k3_3 + B4 * k4_3 + B5 * k5_3 + B6 * k6_3)
        yn_4 = vy + h * (B1 * f0_4 + B3 * k3_4 + B4 * k4_4 + B5 * k5_4 + B6 * k6_4)
        k7_1, k7_2, k7_3, k7_4 = _ballistic_dynamics(yn_3, yn_4, drag_scale, mass)

        err_1 = h * (E1 * f0_1 + E3 * k3_1 + E4 * k4_1 + E5 * k5_1 + E6 * k6_1 + E7 * k7_1)
        err_2 = h * (E1 * f0_2 + E3 * k3_2 + E4 * k4_2 + E5 * k5_2 + E6 * k6_2 + E7 * k7_2)
        err_3 = h * (E1 * f0_3 + E3 * k3_3 + E4 * k4_3 + E5 * k5_3 + E6 * k6_3 + E7 * k7_3)
        err_4 = h * (E1 * f0_4 + E3 * k3_4 + E4 * k4_4 + E5 * k5_4 + E6 * k6_4 + E7 * k7_4)

        scale_1 = atol + rtol * max(abs(x), abs(yn_1))
        scale_2 = atol + rtol * max(abs(y), abs(yn_2))
        scale_3 = atol + rtol * max(abs(vx), abs(yn_3))
        scale_4 = atol + rtol * max(abs(vy), abs(yn_4))

        err_norm = np.sqrt(
            ((err_1 / scale_1) ** 2 + (err_2 / scale_2) ** 2 + (err_3 / scale_3) ** 2 + (err_4 / scale_4) ** 2) / 4.0
        )

        if err_norm <= 1.0:
            g_new = yn_2
            if g_prev > 0.0 and g_new <= 0.0:
                lo, hi = 0.0, 1.0
                for _ in range(50):
                    mid = 0.5 * (lo + hi)
                    g_mid = _hermite(y, yn_2, f0_2, k7_2, h, mid)
                    if g_mid > 0.0:
                        lo = mid
                    else:
                        hi = mid
                s = 0.5 * (lo + hi)
                return _hermite(x, yn_1, f0_1, k7_1, h, s)

            t = t + h
            x, y, vx, vy = yn_1, yn_2, yn_3, yn_4
            f0_1, f0_2, f0_3, f0_4 = k7_1, k7_2, k7_3, k7_4
            g_prev = g_new

            factor = MAX_FACTOR if err_norm == 0.0 else min(MAX_FACTOR, SAFETY * err_norm**ERROR_EXPONENT)
            h = h * factor
        else:
            factor = max(MIN_FACTOR, SAFETY * err_norm**ERROR_EXPONENT)
            h = h * factor

    return x


@njit(cache=True, fastmath=True, inline="always")
def _machine_constants(counter_weight_mass, pulley_radius, length_counterweight,
                        counter_weight_rope_length, arm_length, string_length,
                        pulley_density, arm_density, projectile_mass, projectile_radius,
                        arm_drag_coefficient, projectile_drag_coefficient,
                        joint_friction_coefficient, has_pulley):
    """Scalar port of the machine constants folded in TrebuchetSimulator.__init__.

    Returns (c, extras): `c` is the dynamics tuple documented above; `extras` carries
    what simulate_fast needs for geometry and the energy bookkeeping.

    The counterweight enters in exactly two places - an inertia about the pivot and a
    gravity torque - which is what lets one set of equations serve both linkages. On the
    traditional machine the beam also carries mass behind the pivot, so arm_mass,
    arm_cm_offset and moi_arm all integrate both sides (see TrebuchetParams).
    """
    arm_back_length = 0.0 if has_pulley else length_counterweight
    arm_total_length = arm_length + arm_back_length
    arm_mass = arm_density * arm_total_length * ARM_CROSS_SECTION_WIDTH**2
    # (a - b)/2 and m(a^2 - ab + b^2)/3; both collapse to the single-sided beam at b = 0.
    arm_cm_offset = (arm_length - arm_back_length) / 2.0
    moi_arm = (1.0 / 3.0) * arm_mass * (
        arm_length * arm_length - arm_length * arm_back_length + arm_back_length * arm_back_length
    )

    pulley_mass = pulley_density * np.pi * pulley_radius**2 * PULLEY_THICKNESS if has_pulley else 0.0
    moi_pulley = 0.5 * pulley_mass * pulley_radius**2
    projectile_area = np.pi * projectile_radius**2

    # TrebuchetParams.initial_cw_rope_length: the explicit length when one is given,
    # otherwise one wrap of the pulley. A non-positive value is the "not set" sentinel,
    # since numba has no None to pass down.
    rope_length = counter_weight_rope_length if counter_weight_rope_length > 0.0 else 2.0 * pulley_radius

    cw_lever = pulley_radius if has_pulley else length_counterweight
    M11 = counter_weight_mass * cw_lever**2 + moi_pulley + moi_arm + projectile_mass * arm_length**2
    M22 = projectile_mass * string_length**2
    coupling = projectile_mass * arm_length * string_length

    if has_pulley:
        # The weight descends r_pul metres per radian whatever the arm angle, so its
        # torque is constant and it has no swing of its own; M33 = 1 keeps the 3x3
        # solve non-singular while leaving psi inert.
        l_w = 0.0
        cw_torque_const = -(counter_weight_mass * pulley_radius * G)
        cw_torque_cos = 0.0
        cw_swing_coupling = 0.0
        cw_swing_gravity_k = 0.0
        M33 = 1.0
    else:
        l_w = rope_length
        cw_torque_const = 0.0
        cw_torque_cos = counter_weight_mass * G * length_counterweight
        cw_swing_coupling = counter_weight_mass * length_counterweight * l_w
        cw_swing_gravity_k = counter_weight_mass * G * l_w
        M33 = counter_weight_mass * l_w * l_w

    arm_drag_k = (1.0 / 6.0) * ARM_CROSS_SECTION_WIDTH * arm_drag_coefficient * RHO_AIR * arm_length**3
    proj_drag_k = 0.5 * RHO_AIR * projectile_drag_coefficient * projectile_area
    arm_gravity_k = arm_cm_offset * G * arm_mass
    proj_gravity_theta_k = projectile_mass * G * arm_length
    proj_gravity_alpha_k = projectile_mass * string_length * G

    c = (
        arm_length, string_length, M11, M22, M33, coupling, cw_swing_coupling,
        arm_drag_k, proj_drag_k, cw_torque_const, cw_torque_cos, cw_swing_gravity_k,
        arm_gravity_k, proj_gravity_theta_k, proj_gravity_alpha_k, joint_friction_coefficient,
    )
    return c, (arm_mass, pulley_mass, projectile_area, arm_cm_offset, l_w)


@njit(cache=True, fastmath=True)
def simulate_fast(counter_weight_mass, pulley_radius, length_counterweight,
                   counter_weight_rope_length, arm_length, string_length, release_angle,
                   pivot_height, pulley_density, arm_density, projectile_mass, projectile_radius,
                   initial_arm_angle, arm_drag_coefficient, projectile_drag_coefficient,
                   joint_friction_coefficient, has_pulley):
    """Scalar port of simulate_trebuchet's rtol=1e-6/dense_output=False path.

    Returns (released, distance, efficiency, string_impulse, cw_impulse,
    sling_deficit). The two impulses are the rope compression impulses (N*s, see
    physics._tension_metrics) used by the objective's slack penalty; sling_deficit is
    the dimensionless share of the launch the sling spent under-loaded, weighted by how
    far under (physics.py reports the same number as `sling_tension_deficit`), used by
    its snap penalty. (False, 0.0, 0.0, 0.0, 0.0, 0.0) if release never occurs or the
    geometry/result is invalid (mirrors physics.py's degenerate cases).

    `has_pulley` selects the linkage; the machine's own linkage parameter is read and the
    other one ignored, exactly as TrebuchetParams does. A non-positive
    counter_weight_rope_length means "unset" (numba has no None), so it falls back to one
    wrap of the pulley.
    """
    c, extras = _machine_constants(
        counter_weight_mass, pulley_radius, length_counterweight, counter_weight_rope_length,
        arm_length, string_length, pulley_density, arm_density, projectile_mass, projectile_radius,
        arm_drag_coefficient, projectile_drag_coefficient, joint_friction_coefficient, has_pulley,
    )
    arm_mass, _pulley_mass, projectile_area, arm_cm_offset, l_w = extras

    # Cocked pose, mirroring physics.TrebuchetSimulator.initial_state.
    theta0 = initial_arm_angle
    psi_rest = -np.pi / 2.0
    if has_pulley:
        # Sling tucked alongside the arm, angled just far enough off it to clear.
        arcsin_arg = projectile_radius / string_length
        if arcsin_arg > 1.0 or arcsin_arg < -1.0:
            return False, 0.0, 0.0, 0.0, 0.0, 0.0
        alpha0 = theta0 + np.pi - np.arcsin(arcsin_arg)
        psi0 = 0.0
    else:
        # Sling and counterweight link both hang straight down from their attachments.
        alpha0 = psi_rest
        psi0 = psi_rest

    # Absolute floor in newtons; the deficit is normalized by it below, so the metric
    # means the same thing on a 0.15 kg stone as on a 20 kg one.
    tension_floor = SLING_TENSION_FLOOR * projectile_mass * G

    (released, t_rel, theta_r, theta_dot_r, alpha_r, alpha_dot_r, psi_r, _psi_dot_r,
     string_impulse, cw_impulse, sling_deficit) = _integrate_launch(
        theta0, alpha0, psi0, c, release_angle, 10.0, 1e-6, 1e-6,
        projectile_mass, counter_weight_mass, pulley_radius, has_pulley, tension_floor,
    )
    # Same guard physics._tension_metrics uses: no clock or no projectile weight means
    # there is no share of the launch to report.
    if t_rel > 0.0 and tension_floor > 0.0:
        sling_deficit = sling_deficit / (tension_floor * t_rel)
    else:
        sling_deficit = 0.0
    if not released:
        return False, 0.0, 0.0, string_impulse, cw_impulse, sling_deficit

    x0 = arm_length * np.cos(theta_r) + string_length * np.cos(alpha_r)
    y0 = arm_length * np.sin(theta_r) + string_length * np.sin(alpha_r) + pivot_height
    vx0 = -arm_length * theta_dot_r * np.sin(theta_r) - string_length * alpha_dot_r * np.sin(alpha_r)
    vy0 = arm_length * theta_dot_r * np.cos(theta_r) + string_length * alpha_dot_r * np.cos(alpha_r)

    if np.isnan(x0) or np.isnan(y0) or np.isnan(vx0) or np.isnan(vy0):
        return False, 0.0, 0.0, string_impulse, cw_impulse, sling_deficit

    proj_speed2 = vx0 * vx0 + vy0 * vy0
    proj_KE = 0.5 * projectile_mass * proj_speed2

    distance = 0.0
    if y0 >= 0.0:
        distance = _integrate_ballistic(
            x0, y0, vx0, vy0, projectile_mass, projectile_drag_coefficient, projectile_area, 60.0, 1e-6, 1e-6
        )
        distance = max(0.0, distance)

    if has_pulley:
        # The weight descends r_pul per radian, so the drop is exactly linear.
        height_dropped = pulley_radius * (initial_arm_angle - theta_r)
    else:
        # The pinned weight follows the pin around the pivot and swings on top of that,
        # so measure its height at both ends of the launch (physics._release_result does
        # the same through weight_position_velocity).
        start_y = pivot_height - length_counterweight * np.sin(theta0) + l_w * np.sin(psi_rest)
        end_y = pivot_height - length_counterweight * np.sin(theta_r) + l_w * np.sin(psi_r)
        height_dropped = start_y - end_y
    counterweight_PE_spent = counter_weight_mass * G * height_dropped

    arm_height_change = (np.sin(initial_arm_angle) - np.sin(theta_r)) * arm_cm_offset
    arm_PE_spent = arm_height_change * arm_mass * G

    start_y = arm_length * np.sin(theta0) + string_length * np.sin(alpha0) + pivot_height
    projectile_PE_spent = (start_y - y0) * projectile_mass * G

    total_PE_spent = counterweight_PE_spent + arm_PE_spent + projectile_PE_spent
    efficiency = proj_KE / total_PE_spent if total_PE_spent > 0.0 else 0.0
    efficiency = max(0.0, efficiency)

    return True, distance, efficiency, string_impulse, cw_impulse, sling_deficit


@njit(cache=True, fastmath=True)
def _score(counter_weight_mass, pulley_radius, length_counterweight, counter_weight_rope_length,
           arm_length, string_length, release_angle,
           pivot_height, pulley_density, arm_density, projectile_mass, projectile_radius,
           initial_arm_angle, arm_drag_coefficient, projectile_drag_coefficient,
           joint_friction_coefficient, has_pulley,
           target_distance, efficiency_weight, distance_weight, mass_weight,
           slack_penalty_weight, snap_penalty_weight):
    """Scalar port of optimization._objective's cost formula for one individual."""
    if string_length > 0.95 * arm_length:
        return INVALID_COST

    released, distance, efficiency, string_impulse, cw_impulse, sling_deficit = simulate_fast(
        counter_weight_mass, pulley_radius, length_counterweight, counter_weight_rope_length,
        arm_length, string_length, release_angle,
        pivot_height, pulley_density, arm_density, projectile_mass, projectile_radius,
        initial_arm_angle, arm_drag_coefficient, projectile_drag_coefficient,
        joint_friction_coefficient, has_pulley,
    )
    if not released or distance <= 0.0 or efficiency <= 0.0:
        return INVALID_COST

    # TrebuchetParams.total_mass: no pulley to weigh on the traditional machine, and its
    # beam spans both sides of the pivot.
    pulley_mass = pulley_density * np.pi * pulley_radius**2 * PULLEY_THICKNESS if has_pulley else 0.0
    arm_total_length = arm_length if has_pulley else arm_length + length_counterweight
    arm_mass = arm_density * arm_total_length * ARM_CROSS_SECTION_WIDTH**2
    total_mass = counter_weight_mass + pulley_mass + arm_mass + projectile_mass

    efficiency_cost = -efficiency * 100.0
    distance_cost = abs(distance - target_distance) / target_distance * 100.0
    mass_cost = (total_mass / 30.0) * 100.0
    # Two terms, two jobs, and they are not redundant even though max(0, floor - T)
    # contains max(0, -T). The impulses are absolute (N*s) and unbounded, and mark a run
    # this engine can no longer model at all - past the first negative tension the rigid
    # link is pushing where a rope would have let go, so the trajectory is fiction; they
    # have to stay big enough to be a wall. The deficit is normalized to [0, 1], which is
    # what makes it comparable across projectile sizes and therefore usable as a gradient,
    # but that same bound is why it cannot double as the wall.
    slack_cost = slack_penalty_weight * (string_impulse + cw_impulse)
    snap_cost = snap_penalty_weight * sling_deficit

    return (
        efficiency_weight * efficiency_cost + distance_weight * distance_cost
        + mass_weight * mass_cost + slack_cost + snap_cost
    )


@njit(cache=True, fastmath=True, parallel=True)
def evaluate_population(counter_weight_mass, pulley_radius, length_counterweight,
                         arm_length, string_length, release_angle,
                         counter_weight_rope_length,
                         pivot_height, pulley_density, arm_density, projectile_mass, projectile_radius,
                         initial_arm_angle, arm_drag_coefficient, projectile_drag_coefficient,
                         joint_friction_coefficient, has_pulley, target_distance, efficiency_weight,
                         distance_weight, mass_weight, slack_penalty_weight, snap_penalty_weight):
    """Cost for an entire DE population in one call.

    The six per-individual args are arrays of shape (S,); everything else is a scalar
    shared across individuals. Both linkage parameters are passed as arrays even though
    only one is searched - the machine the population is being scored for reads its own
    and ignores the other, so the caller fills the unused one with a constant.
    """
    n = counter_weight_mass.shape[0]
    costs = np.empty(n, dtype=np.float64)
    for i in prange(n):
        costs[i] = _score(
            counter_weight_mass[i], pulley_radius[i], length_counterweight[i],
            counter_weight_rope_length,
            arm_length[i], string_length[i], release_angle[i],
            pivot_height, pulley_density, arm_density, projectile_mass, projectile_radius,
            initial_arm_angle, arm_drag_coefficient, projectile_drag_coefficient,
            joint_friction_coefficient, has_pulley,
            target_distance, efficiency_weight, distance_weight, mass_weight,
            slack_penalty_weight, snap_penalty_weight,
        )
    return costs
