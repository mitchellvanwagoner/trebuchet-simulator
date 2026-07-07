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

One deliberate divergence: physics.py simulates the sling going slack (free-flight
projectile + inelastic re-tension snaps), while this engine keeps the rigid-sling
model and instead reports the string/cw-rope compression impulses that the objective
penalizes (`slack_penalty_weight`). The two models coincide exactly on trajectories
whose ropes stay taut - which is precisely where the penalty drives the optimizer, so
the search result is always in the regime where this engine is faithful.

The integrator mirrors scipy's RK45: same Dormand-Prince tableau, same step-size
control. Events (release angle, ground impact) are localized with a cubic Hermite
interpolant built from the bracketing step's endpoint states/derivatives, refined by
bisection - cheaper than scipy's dense-output event solver and accurate enough for the
optimizer's tolerance (see the `rtol=1e-6` note in `optimization._objective`).
"""

import numpy as np
from numba import njit, prange

from trebuchet_sim.config import ARM_CROSS_SECTION_WIDTH, G, RHO_AIR

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


@njit(cache=True, fastmath=True, inline="always")
def _trebuchet_dynamics(theta, theta_dot, alpha, alpha_dot, l_a, l_s, M11, M22, coupling,
                         arm_drag_k, proj_drag_k, cw_gravity_torque, arm_gravity_k,
                         proj_gravity_theta_k, proj_gravity_alpha_k, joint_friction):
    """Scalar port of physics.TrebuchetSimulator.trebuchet_dynamics."""
    sin_t, cos_t = np.sin(theta), np.cos(theta)
    sin_a, cos_a = np.sin(alpha), np.cos(alpha)
    sin_at = sin_a * cos_t - cos_a * sin_t
    cos_at = cos_a * cos_t + sin_a * sin_t

    p_vx = -l_a * theta_dot * sin_t - l_s * alpha_dot * sin_a
    p_vy = l_a * theta_dot * cos_t + l_s * alpha_dot * cos_a
    proj_speed = np.sqrt(p_vx * p_vx + p_vy * p_vy)

    M12 = coupling * cos_at

    arm_drag_torque = -np.copysign(arm_drag_k * theta_dot * theta_dot, theta_dot)

    drag_scale = proj_drag_k * proj_speed
    drag_fx, drag_fy = -drag_scale * p_vx, -drag_scale * p_vy
    Q_theta_drag = drag_fx * (-l_a * sin_t) + drag_fy * (l_a * cos_t)
    Q_alpha_drag = drag_fx * (-l_s * sin_a) + drag_fy * (l_s * cos_a)

    Q_theta = (
        coupling * sin_at * alpha_dot * alpha_dot
        - cw_gravity_torque
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

    det = M11 * M22 - M12 * M12
    if abs(det) < 1e-12:
        return theta_dot, 0.0, alpha_dot, 0.0

    theta_ddot = (Q_theta * M22 - Q_alpha * M12) / det
    alpha_ddot = (Q_alpha * M11 - Q_theta * M12) / det
    return theta_dot, theta_ddot, alpha_dot, alpha_ddot


@njit(cache=True, fastmath=True, inline="always")
def _tensions(theta, theta_dot, alpha, alpha_dot, theta_ddot, l_a, l_s, proj_drag_k,
              projectile_mass, counter_weight_mass, pulley_radius):
    """Scalar port of physics.TrebuchetSimulator.constraint_tensions (theta_ddot passed in)."""
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
    cw_tension = counter_weight_mass * (G + pulley_radius * theta_ddot)
    return string_tension, cw_tension


@njit(cache=True, fastmath=True)
def _integrate_launch(theta0, alpha0, l_a, l_s, M11, M22, coupling, arm_drag_k, proj_drag_k,
                       cw_gravity_torque, arm_gravity_k, proj_gravity_theta_k,
                       proj_gravity_alpha_k, joint_friction, release_angle, t_max, rtol, atol,
                       projectile_mass, counter_weight_mass, pulley_radius):
    """Integrate launch dynamics until the release-angle event fires.

    Returns (released, t, theta, theta_dot, alpha, alpha_dot, string_impulse, cw_impulse):
    the state at release (or the final state at t_max if the arm never reached the
    release angle) plus the string/cw-rope compression impulses (trapezoid rule over
    accepted steps, mirroring physics.TrebuchetSimulator._tension_metrics).
    """
    t = 0.0
    theta, theta_dot, alpha, alpha_dot = theta0, 0.0, alpha0, 0.0
    f0_1, f0_2, f0_3, f0_4 = _trebuchet_dynamics(
        theta, theta_dot, alpha, alpha_dot, l_a, l_s, M11, M22, coupling, arm_drag_k,
        proj_drag_k, cw_gravity_torque, arm_gravity_k, proj_gravity_theta_k,
        proj_gravity_alpha_k, joint_friction,
    )

    h = 1e-3
    g_prev = theta - release_angle

    string_impulse = 0.0
    cw_impulse = 0.0
    string_T_prev, cw_T_prev = _tensions(
        theta, theta_dot, alpha, alpha_dot, f0_2, l_a, l_s, proj_drag_k,
        projectile_mass, counter_weight_mass, pulley_radius,
    )

    for _ in range(MAX_STEPS):
        if t >= t_max:
            return False, t, theta, theta_dot, alpha, alpha_dot, string_impulse, cw_impulse
        if t + h > t_max:
            h = t_max - t

        y2_1 = theta + h * A21 * f0_1
        y2_2 = theta_dot + h * A21 * f0_2
        y2_3 = alpha + h * A21 * f0_3
        y2_4 = alpha_dot + h * A21 * f0_4
        k2_1, k2_2, k2_3, k2_4 = _trebuchet_dynamics(
            y2_1, y2_2, y2_3, y2_4, l_a, l_s, M11, M22, coupling, arm_drag_k, proj_drag_k,
            cw_gravity_torque, arm_gravity_k, proj_gravity_theta_k, proj_gravity_alpha_k, joint_friction,
        )

        y3_1 = theta + h * (A31 * f0_1 + A32 * k2_1)
        y3_2 = theta_dot + h * (A31 * f0_2 + A32 * k2_2)
        y3_3 = alpha + h * (A31 * f0_3 + A32 * k2_3)
        y3_4 = alpha_dot + h * (A31 * f0_4 + A32 * k2_4)
        k3_1, k3_2, k3_3, k3_4 = _trebuchet_dynamics(
            y3_1, y3_2, y3_3, y3_4, l_a, l_s, M11, M22, coupling, arm_drag_k, proj_drag_k,
            cw_gravity_torque, arm_gravity_k, proj_gravity_theta_k, proj_gravity_alpha_k, joint_friction,
        )

        y4_1 = theta + h * (A41 * f0_1 + A42 * k2_1 + A43 * k3_1)
        y4_2 = theta_dot + h * (A41 * f0_2 + A42 * k2_2 + A43 * k3_2)
        y4_3 = alpha + h * (A41 * f0_3 + A42 * k2_3 + A43 * k3_3)
        y4_4 = alpha_dot + h * (A41 * f0_4 + A42 * k2_4 + A43 * k3_4)
        k4_1, k4_2, k4_3, k4_4 = _trebuchet_dynamics(
            y4_1, y4_2, y4_3, y4_4, l_a, l_s, M11, M22, coupling, arm_drag_k, proj_drag_k,
            cw_gravity_torque, arm_gravity_k, proj_gravity_theta_k, proj_gravity_alpha_k, joint_friction,
        )

        y5_1 = theta + h * (A51 * f0_1 + A52 * k2_1 + A53 * k3_1 + A54 * k4_1)
        y5_2 = theta_dot + h * (A51 * f0_2 + A52 * k2_2 + A53 * k3_2 + A54 * k4_2)
        y5_3 = alpha + h * (A51 * f0_3 + A52 * k2_3 + A53 * k3_3 + A54 * k4_3)
        y5_4 = alpha_dot + h * (A51 * f0_4 + A52 * k2_4 + A53 * k3_4 + A54 * k4_4)
        k5_1, k5_2, k5_3, k5_4 = _trebuchet_dynamics(
            y5_1, y5_2, y5_3, y5_4, l_a, l_s, M11, M22, coupling, arm_drag_k, proj_drag_k,
            cw_gravity_torque, arm_gravity_k, proj_gravity_theta_k, proj_gravity_alpha_k, joint_friction,
        )

        y6_1 = theta + h * (A61 * f0_1 + A62 * k2_1 + A63 * k3_1 + A64 * k4_1 + A65 * k5_1)
        y6_2 = theta_dot + h * (A61 * f0_2 + A62 * k2_2 + A63 * k3_2 + A64 * k4_2 + A65 * k5_2)
        y6_3 = alpha + h * (A61 * f0_3 + A62 * k2_3 + A63 * k3_3 + A64 * k4_3 + A65 * k5_3)
        y6_4 = alpha_dot + h * (A61 * f0_4 + A62 * k2_4 + A63 * k3_4 + A64 * k4_4 + A65 * k5_4)
        k6_1, k6_2, k6_3, k6_4 = _trebuchet_dynamics(
            y6_1, y6_2, y6_3, y6_4, l_a, l_s, M11, M22, coupling, arm_drag_k, proj_drag_k,
            cw_gravity_torque, arm_gravity_k, proj_gravity_theta_k, proj_gravity_alpha_k, joint_friction,
        )

        yn_1 = theta + h * (B1 * f0_1 + B3 * k3_1 + B4 * k4_1 + B5 * k5_1 + B6 * k6_1)
        yn_2 = theta_dot + h * (B1 * f0_2 + B3 * k3_2 + B4 * k4_2 + B5 * k5_2 + B6 * k6_2)
        yn_3 = alpha + h * (B1 * f0_3 + B3 * k3_3 + B4 * k4_3 + B5 * k5_3 + B6 * k6_3)
        yn_4 = alpha_dot + h * (B1 * f0_4 + B3 * k3_4 + B4 * k4_4 + B5 * k5_4 + B6 * k6_4)
        k7_1, k7_2, k7_3, k7_4 = _trebuchet_dynamics(
            yn_1, yn_2, yn_3, yn_4, l_a, l_s, M11, M22, coupling, arm_drag_k, proj_drag_k,
            cw_gravity_torque, arm_gravity_k, proj_gravity_theta_k, proj_gravity_alpha_k, joint_friction,
        )

        err_1 = h * (E1 * f0_1 + E3 * k3_1 + E4 * k4_1 + E5 * k5_1 + E6 * k6_1 + E7 * k7_1)
        err_2 = h * (E1 * f0_2 + E3 * k3_2 + E4 * k4_2 + E5 * k5_2 + E6 * k6_2 + E7 * k7_2)
        err_3 = h * (E1 * f0_3 + E3 * k3_3 + E4 * k4_3 + E5 * k5_3 + E6 * k6_3 + E7 * k7_3)
        err_4 = h * (E1 * f0_4 + E3 * k3_4 + E4 * k4_4 + E5 * k5_4 + E6 * k6_4 + E7 * k7_4)

        scale_1 = atol + rtol * max(abs(theta), abs(yn_1))
        scale_2 = atol + rtol * max(abs(theta_dot), abs(yn_2))
        scale_3 = atol + rtol * max(abs(alpha), abs(yn_3))
        scale_4 = atol + rtol * max(abs(alpha_dot), abs(yn_4))

        err_norm = np.sqrt(
            ((err_1 / scale_1) ** 2 + (err_2 / scale_2) ** 2 + (err_3 / scale_3) ** 2 + (err_4 / scale_4) ** 2) / 4.0
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
                _, theta_ddot_r, _, _ = _trebuchet_dynamics(
                    theta_r, theta_dot_r, alpha_r, alpha_dot_r, l_a, l_s, M11, M22, coupling,
                    arm_drag_k, proj_drag_k, cw_gravity_torque, arm_gravity_k,
                    proj_gravity_theta_k, proj_gravity_alpha_k, joint_friction,
                )
                string_T_r, cw_T_r = _tensions(
                    theta_r, theta_dot_r, alpha_r, alpha_dot_r, theta_ddot_r, l_a, l_s,
                    proj_drag_k, projectile_mass, counter_weight_mass, pulley_radius,
                )
                string_impulse += 0.5 * (max(0.0, -string_T_prev) + max(0.0, -string_T_r)) * h * s
                cw_impulse += 0.5 * (max(0.0, -cw_T_prev) + max(0.0, -cw_T_r)) * h * s
                return True, t + h * s, theta_r, theta_dot_r, alpha_r, alpha_dot_r, string_impulse, cw_impulse

            string_T_new, cw_T_new = _tensions(
                yn_1, yn_2, yn_3, yn_4, k7_2, l_a, l_s, proj_drag_k,
                projectile_mass, counter_weight_mass, pulley_radius,
            )
            string_impulse += 0.5 * (max(0.0, -string_T_prev) + max(0.0, -string_T_new)) * h
            cw_impulse += 0.5 * (max(0.0, -cw_T_prev) + max(0.0, -cw_T_new)) * h
            string_T_prev, cw_T_prev = string_T_new, cw_T_new

            t = t + h
            theta, theta_dot, alpha, alpha_dot = yn_1, yn_2, yn_3, yn_4
            f0_1, f0_2, f0_3, f0_4 = k7_1, k7_2, k7_3, k7_4
            g_prev = g_new

            factor = MAX_FACTOR if err_norm == 0.0 else min(MAX_FACTOR, SAFETY * err_norm**ERROR_EXPONENT)
            h = h * factor
        else:
            factor = max(MIN_FACTOR, SAFETY * err_norm**ERROR_EXPONENT)
            h = h * factor

    return False, t, theta, theta_dot, alpha, alpha_dot, string_impulse, cw_impulse


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
def _machine_constants(counter_weight_mass, pulley_radius, arm_length, string_length,
                        pulley_density, arm_density, projectile_mass, projectile_radius,
                        arm_drag_coefficient, projectile_drag_coefficient):
    """Scalar port of the machine constants folded in TrebuchetSimulator.__init__."""
    pulley_mass = pulley_density * np.pi * pulley_radius**2 * PULLEY_THICKNESS
    arm_mass = arm_density * arm_length * ARM_CROSS_SECTION_WIDTH**2
    moi_pulley = 0.5 * pulley_mass * pulley_radius**2
    moi_arm = (1.0 / 3.0) * arm_mass * arm_length**2
    projectile_area = np.pi * projectile_radius**2

    M11 = counter_weight_mass * pulley_radius**2 + moi_pulley + moi_arm + projectile_mass * arm_length**2
    M22 = projectile_mass * string_length**2
    coupling = projectile_mass * arm_length * string_length
    arm_drag_k = (1.0 / 6.0) * ARM_CROSS_SECTION_WIDTH * arm_drag_coefficient * RHO_AIR * arm_length**3
    proj_drag_k = 0.5 * RHO_AIR * projectile_drag_coefficient * projectile_area
    cw_gravity_torque = counter_weight_mass * pulley_radius * G
    arm_gravity_k = arm_length / 2.0 * G * arm_mass
    proj_gravity_theta_k = projectile_mass * G * arm_length
    proj_gravity_alpha_k = projectile_mass * string_length * G

    return (
        M11, M22, coupling, arm_drag_k, proj_drag_k, cw_gravity_torque,
        arm_gravity_k, proj_gravity_theta_k, proj_gravity_alpha_k, arm_mass, pulley_mass, projectile_area,
    )


@njit(cache=True, fastmath=True)
def simulate_fast(counter_weight_mass, pulley_radius, arm_length, string_length, release_angle,
                   pivot_height, pulley_density, arm_density, projectile_mass, projectile_radius,
                   initial_arm_angle, arm_drag_coefficient, projectile_drag_coefficient,
                   joint_friction_coefficient):
    """Scalar port of simulate_trebuchet's rtol=1e-6/dense_output=False path.

    Returns (released, distance, efficiency, string_impulse, cw_impulse); the last two
    are the rope compression impulses (N*s, see physics._tension_metrics) used by the
    objective's slack penalty. (False, 0.0, 0.0, 0.0, 0.0) if release never occurs or
    the geometry/result is invalid (mirrors physics.py's degenerate cases).
    """
    arcsin_arg = projectile_radius / string_length
    if arcsin_arg > 1.0 or arcsin_arg < -1.0:
        return False, 0.0, 0.0, 0.0, 0.0

    (M11, M22, coupling, arm_drag_k, proj_drag_k, cw_gravity_torque, arm_gravity_k,
     proj_gravity_theta_k, proj_gravity_alpha_k, arm_mass, _pulley_mass, projectile_area) = _machine_constants(
        counter_weight_mass, pulley_radius, arm_length, string_length, pulley_density,
        arm_density, projectile_mass, projectile_radius, arm_drag_coefficient, projectile_drag_coefficient,
    )

    theta0 = initial_arm_angle
    alpha0 = theta0 + np.pi - np.arcsin(arcsin_arg)

    released, _t_rel, theta_r, theta_dot_r, alpha_r, alpha_dot_r, string_impulse, cw_impulse = _integrate_launch(
        theta0, alpha0, arm_length, string_length, M11, M22, coupling, arm_drag_k, proj_drag_k,
        cw_gravity_torque, arm_gravity_k, proj_gravity_theta_k, proj_gravity_alpha_k,
        joint_friction_coefficient, release_angle, 10.0, 1e-6, 1e-6,
        projectile_mass, counter_weight_mass, pulley_radius,
    )
    if not released:
        return False, 0.0, 0.0, string_impulse, cw_impulse

    x0 = arm_length * np.cos(theta_r) + string_length * np.cos(alpha_r)
    y0 = arm_length * np.sin(theta_r) + string_length * np.sin(alpha_r) + pivot_height
    vx0 = -arm_length * theta_dot_r * np.sin(theta_r) - string_length * alpha_dot_r * np.sin(alpha_r)
    vy0 = arm_length * theta_dot_r * np.cos(theta_r) + string_length * alpha_dot_r * np.cos(alpha_r)

    if np.isnan(x0) or np.isnan(y0) or np.isnan(vx0) or np.isnan(vy0):
        return False, 0.0, 0.0, string_impulse, cw_impulse

    proj_speed2 = vx0 * vx0 + vy0 * vy0
    proj_KE = 0.5 * projectile_mass * proj_speed2

    distance = 0.0
    if y0 >= 0.0:
        distance = _integrate_ballistic(
            x0, y0, vx0, vy0, projectile_mass, projectile_drag_coefficient, projectile_area, 60.0, 1e-6, 1e-6
        )
        distance = max(0.0, distance)

    arm_angle_rotated = initial_arm_angle - theta_r
    height_dropped = pulley_radius * arm_angle_rotated
    counterweight_PE_spent = counter_weight_mass * G * height_dropped

    arm_height_change = (np.sin(initial_arm_angle) - np.sin(theta_r)) * arm_length / 2.0
    arm_PE_spent = arm_height_change * arm_mass * G

    start_y = arm_length * np.sin(theta0) + string_length * np.sin(alpha0) + pivot_height
    projectile_PE_spent = (start_y - y0) * projectile_mass * G

    total_PE_spent = counterweight_PE_spent + arm_PE_spent + projectile_PE_spent
    efficiency = proj_KE / total_PE_spent if total_PE_spent > 0.0 else 0.0
    efficiency = max(0.0, efficiency)

    return True, distance, efficiency, string_impulse, cw_impulse


@njit(cache=True, fastmath=True)
def _score(counter_weight_mass, pulley_radius, arm_length, string_length, release_angle,
           pivot_height, pulley_density, arm_density, projectile_mass, projectile_radius,
           initial_arm_angle, arm_drag_coefficient, projectile_drag_coefficient,
           joint_friction_coefficient, target_distance, efficiency_weight, distance_weight, mass_weight,
           slack_penalty_weight):
    """Scalar port of optimization._objective's cost formula for one individual."""
    if string_length > 0.95 * arm_length:
        return INVALID_COST

    released, distance, efficiency, string_impulse, cw_impulse = simulate_fast(
        counter_weight_mass, pulley_radius, arm_length, string_length, release_angle,
        pivot_height, pulley_density, arm_density, projectile_mass, projectile_radius,
        initial_arm_angle, arm_drag_coefficient, projectile_drag_coefficient, joint_friction_coefficient,
    )
    if not released or distance <= 0.0 or efficiency <= 0.0:
        return INVALID_COST

    pulley_mass = pulley_density * np.pi * pulley_radius**2 * PULLEY_THICKNESS
    arm_mass = arm_density * arm_length * ARM_CROSS_SECTION_WIDTH**2
    total_mass = counter_weight_mass + pulley_mass + arm_mass + projectile_mass

    efficiency_cost = -efficiency * 100.0
    distance_cost = abs(distance - target_distance) / target_distance * 100.0
    mass_cost = (total_mass / 30.0) * 100.0
    slack_cost = slack_penalty_weight * (string_impulse + cw_impulse)

    return (
        efficiency_weight * efficiency_cost + distance_weight * distance_cost
        + mass_weight * mass_cost + slack_cost
    )


@njit(cache=True, fastmath=True, parallel=True)
def evaluate_population(counter_weight_mass, pulley_radius, arm_length, string_length, release_angle,
                         pivot_height, pulley_density, arm_density, projectile_mass, projectile_radius,
                         initial_arm_angle, arm_drag_coefficient, projectile_drag_coefficient,
                         joint_friction_coefficient, target_distance, efficiency_weight,
                         distance_weight, mass_weight, slack_penalty_weight):
    """Cost for an entire DE population in one call. The five optimizable-param args
    are arrays of shape (S,); everything else is a scalar shared across individuals."""
    n = counter_weight_mass.shape[0]
    costs = np.empty(n, dtype=np.float64)
    for i in prange(n):
        costs[i] = _score(
            counter_weight_mass[i], pulley_radius[i], arm_length[i], string_length[i], release_angle[i],
            pivot_height, pulley_density, arm_density, projectile_mass, projectile_radius,
            initial_arm_angle, arm_drag_coefficient, projectile_drag_coefficient, joint_friction_coefficient,
            target_distance, efficiency_weight, distance_weight, mass_weight, slack_penalty_weight,
        )
    return costs
