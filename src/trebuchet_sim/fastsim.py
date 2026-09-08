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

The sling is modelled here exactly as physics.py models it: a rope that can pull but
never push. A launch is a stitched alternation of taut and slack segments, and the
re-tension between them is an inelastic snap that destroys energy (_apply_snap). This
engine used to keep a rigid sling instead and lean on the objective's penalties to steer
the search away from where that lie mattered, which left the two engines reporting
distances a median 56% apart on the pulley machine once a sling let go - far enough that
a search pushed hard on distance could win on a throw only the rigid model believed in.
Both engines now run the same four-state machine, so they agree wherever they are asked
the same question.

The compression impulses and `sling_deficit` remain, no longer as a stand-in for the
missing physics but as what they say they are: the impulse marks a rigid link (the
counterweight rope) being pushed, and the deficit grades how close to slack a sling ran,
which is the continuous signal the optimizer steers by (see optimization.py).

The integrator mirrors scipy's RK45: same Dormand-Prince tableau, same step-size
control. Events (release angle, ground impact) are localized with a cubic Hermite
interpolant built from the bracketing step's endpoint states/derivatives, refined by
bisection - cheaper than scipy's dense-output event solver and accurate enough for the
optimizer's tolerance (see the `rtol=1e-6` note in `optimization._objective`).
"""

import numpy as np
from numba import njit, prange

from trebuchet_sim.config import (
    ARM_CROSS_SECTION_WIDTH,
    AUTO_INITIAL_ARM_ANGLE,
    DEFAULT_INITIAL_ARM_ANGLE,
    G,
    PIVOT_FRICTION_SMOOTHING,
    RHO_AIR,
    SLING_TENSION_FLOOR,
    TRADITIONAL_START_CLEARANCE,
    MachineType,
)

# Local names so the JIT closes over plain floats rather than reaching into a module.
_FRICTION_EPS = PIVOT_FRICTION_SMOOTHING
_START_CLEARANCE = TRADITIONAL_START_CLEARANCE
_PULLEY_COCKED_ANGLE = float(DEFAULT_INITIAL_ARM_ANGLE[MachineType.PULLEY])
_AUTO_ANGLE = AUTO_INITIAL_ARM_ANGLE

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

# Why a launch segment ended, mirroring the terminal events physics.py arms on each
# regime. Running out of clock, reaching the release angle and the beam striking the
# ground can end any of the four; the rest are a regime's own exits, and the outer loop
# reads them to decide which regime comes next.
_SEG_TMAX = 0
_SEG_RELEASE = 1
_SEG_ARM_GROUND = 2
_SEG_SLING_SLACK = 3   # string tension crossed zero: the sling lets go
_SEG_RETENSION = 4     # tip-to-projectile distance grew back to the sling length
_SEG_LANDING = 5       # the projectile reached the ground
_SEG_LIFTOFF = 6       # the ground's normal force crossed zero: the sling has it again

# The four states a launch can be in, mirroring physics.py's TAUT / SLACK / TAUT_GROUND /
# SLACK_GROUND. The sling is either carrying load or not, and the projectile is either on
# the ground or off it, and those are independent: a sling can go slack over a projectile
# already lying in the dirt, and a taut one can drag a projectile along it. _TAUT is the
# only one on the six-component [theta, theta_dot, alpha, alpha_dot, psi, psi_dot] layout;
# the other three carry the projectile explicitly as
# [theta, theta_dot, px, py, pvx, pvy, psi, psi_dot], the grounded pair holding py and pvy
# at zero.
_TAUT = 0
_SLACK = 1
_TAUT_GROUND = 2
_SLACK_GROUND = 3

# Cap on taut/slack regime switches, matching physics.MAX_LAUNCH_SEGMENTS. Each snap
# destroys energy so the switching always dies out; the cap only guards numerical
# chatter right at a regime boundary.
MAX_LAUNCH_SEGMENTS = 200

# Three running integrals ride at the end of every state vector here as they do in
# physics.py (see physics.N_QUADRATURE for why they are states and not a quadrature summed
# afterwards): the sling's compression impulse, the counterweight link's, and the sling's
# tension deficit. The taut layout is 6 physical components plus these; the eight-component
# regimes are 8 plus these. They count toward the error norm on both engines, so the two
# still open and walk every segment on the same step grid.
N_QUADRATURE = 3
QUADRATURE_ATOL = 1e-7  # physics.QUADRATURE_ATOL

SAFETY = 0.9
MIN_FACTOR = 0.2
MAX_FACTOR = 10.0
ERROR_EXPONENT = -1.0 / 5.0
MAX_STEPS = 20000

# How many times one step may be shrunk for leaving the projectile outside the sling
# circle before the stepper gives up and calls the sling taut where it stands (see
# _integrate_eight_segment). Reaching it needs a projectile that leaves the circle and
# never comes back, which the regime cannot produce - a slack segment opens with zero
# radial rate and inward acceleration - so this is a termination guard rather than a path
# the physics takes. Twenty halvings shrink a step by a factor of a million.
MAX_RETENSION_SHRINKS = 20

# How far outside the sling circle a segment may open and still count as opening *on* it,
# as a fraction of the sling length. Only a segment that opens on the circle can have its
# steps rejected for crossing it (see _integrate_eight_segment), and the two cases are
# eleven orders of magnitude apart, so this separates them with room to spare: a slack
# segment entered from the taut regime opens at the rounding of the sling length, measured
# at 2.8e-17 m, while one entered from TAUT_GROUND opens whereever that regime's own
# constraint drift left it - 3.8e-6 m outside on the draw that motivated this. A state
# already outside is not a step that escaped and must not be treated as one; the reference
# engine's event, which arms on the way back down, simply carries it.
RETENSION_CIRCLE_SLOP = 1e-12


@njit(cache=True, fastmath=True, inline="always")
def _probe_step(d0, d1, interval_length):
    """The trial step scipy takes before it can measure how fast the derivative changes.

    First half of scipy.integrate._ivp.common.select_initial_step: a step scaled by the
    state against its own derivative, both measured in units of the error tolerance, and
    a small fixed step when either is too near zero to divide by.
    """
    h0 = 1e-6 if (d0 < 1e-5 or d1 < 1e-5) else 0.01 * d0 / d1
    return min(h0, interval_length)


@njit(cache=True, fastmath=True, inline="always")
def _initial_step(h0, d1, d2, interval_length):
    """Second half of select_initial_step, once the probe has produced d2.

    `d2` is how much the derivative moved across the probe, per unit time; the step is
    chosen so the local error over it sits at the tolerance, for an error estimator of
    order 4 (hence the fifth root, matching ERROR_EXPONENT).
    """
    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** 0.2
    return min(100.0 * h0, min(h1, interval_length))


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
#   4 M33                10 cw_torque_cos          16 M_taut
#                                                  17 arm_back_length
#   5 coupling           11 cw_swing_gravity_k
#
# The next three are geometry the impulses need rather than the equations of motion: an
# impulse has to be priced in energy, which means knowing what the machine's own kinetic
# energy is (see _launch_kinetic_energy).
#  18 l_w (pin-to-weight link)   19 moi_machine (arm + pulley)   20 cw_lever
#
# And two more: the pivot's dry-friction torque, which is a term of the equations of
# motion, and the height a resting projectile's centre sits at, which every ground test
# in this engine measures against.
#  21 pivot_friction (N*m)       22 ground_y (m)
#
# And the five the constraint tensions need, so the dynamics can work them out inline for
# the running integrals rather than being handed them:
#  23 projectile_mass  24 counter_weight_mass  25 pulley_radius
#  26 has_pulley (1.0 / 0.0 - the tuple is floats)   27 tension_floor (N)


@njit(cache=True, fastmath=True, inline="always")
def _quadrature_rates(string_T, cw_T, tension_floor, grades_cw_link):
    """Derivatives of the three running integrals; physics._quadrature_rates.

    `grades_cw_link` is the pulley machine's has_pulley: only that machine carries its
    weight on a rope, and only a rope can be faulted for pushing. The traditional
    machine's weight is pinned to the arm on a rigid link, which is a strut and carries
    compression by design, so its rate is identically zero.
    """
    loaded = string_T if string_T > 0.0 else 0.0
    cw_rate = 0.0
    if grades_cw_link and cw_T < 0.0:
        cw_rate = -cw_T
    return (
        -string_T if string_T < 0.0 else 0.0,
        cw_rate,
        tension_floor - (tension_floor if loaded > tension_floor else loaded),
    )


@njit(cache=True, fastmath=True, inline="always")
def _trebuchet_dynamics(theta, theta_dot, alpha, alpha_dot, psi, psi_dot, c):
    """Scalar port of physics.TrebuchetSimulator.trebuchet_dynamics.

    Six physical states, like the reference engine - psi is the counterweight's swing
    about its pin, live on the traditional machine and inert on the pulley one, where
    M13 = 0 and M33 = 1 collapse the 3x3 solve back to the original two-coordinate one
    exactly - and then the three running integrals' derivatives, which need the constraint
    tensions and so are worked out here where the trig for them is already in hand.

    Nine values out, six in: the integrals never feed back into the dynamics, so the RK
    stages below pass only the physical components and pick the rates up from the return.
    """
    (l_a, l_s, M11, M22, M33, coupling, cw_swing_coupling, arm_drag_k, proj_drag_k,
     cw_torque_const, cw_torque_cos, cw_swing_gravity_k, arm_gravity_k,
     proj_gravity_theta_k, proj_gravity_alpha_k, joint_friction, _M_taut, _back,
     _l_w, _moi, _lever, pivot_friction, _ground_y,
     projectile_mass, counter_weight_mass, pulley_radius, has_pulley_f, tension_floor) = c

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
    # Dry bearing friction: constant magnitude, opposing rotation, smoothed through zero
    # (see physics.TrebuchetSimulator.__init__ and config.PIVOT_FRICTION_SMOOTHING).
    bearing_torque = -pivot_friction * theta_dot / np.sqrt(
        theta_dot * theta_dot + _FRICTION_EPS * _FRICTION_EPS
    )

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
        + bearing_torque
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
        return theta_dot, 0.0, alpha_dot, 0.0, psi_dot, 0.0, 0.0, 0.0, 0.0

    theta_ddot = (D * E * Q_theta - B * E * Q_alpha - C * D * Q_psi) / det
    alpha_ddot = (-B * E * Q_theta + (A * E - C * C) * Q_alpha + B * C * Q_psi) / det
    psi_ddot = (-C * D * Q_theta + B * C * Q_alpha + (A * D - B * B) * Q_psi) / det

    radial_acc = l_a * theta_ddot * sin_at - l_a * theta_dot * theta_dot * cos_at \
        - l_s * alpha_dot * alpha_dot
    string_T = (
        -projectile_mass * G * sin_a
        - drag_scale * (p_vx * cos_a + p_vy * sin_a)
        - projectile_mass * radial_acc
    )
    cw_T = _cw_link_tension(theta, theta_dot, theta_ddot, psi, psi_dot, c,
                            counter_weight_mass, pulley_radius, has_pulley_f > 0.5)
    q1, q2, q3 = _quadrature_rates(string_T, cw_T, tension_floor, has_pulley_f > 0.5)
    return theta_dot, theta_ddot, alpha_dot, alpha_ddot, psi_dot, psi_ddot, q1, q2, q3


@njit(cache=True, fastmath=True)
def _first_arm_ground_angle(l_a, l_back, h_T, initial_arm_angle):
    """The largest arm angle below the cocked one at which the beam touches the ground.

    Scalar port of physics.TrebuchetSimulator._first_arm_ground_angle. The test has to be
    on the angle rather than on the beam's clearance: clearance dips below zero and comes
    back within a fraction of a turn, and a solver that checks its sign at step endpoints
    can step over the whole excursion - which it does, during a slack stretch, where the
    machine is a smooth pendulum and RK45 stretches its step past the width of the dip.
    An arm that turns one way either reaches this angle or does not.

    Returns -inf when neither end of the beam can reach the ground, which leaves the
    event unarmed.
    """
    best = -np.inf
    for k in range(2):
        radius = l_a if k == 0 else l_back
        sign = -1.0 if k == 0 else 1.0
        if radius < h_T or radius <= 0.0:
            continue
        ratio = sign * h_T / radius
        if ratio > 1.0:
            ratio = 1.0
        elif ratio < -1.0:
            ratio = -1.0
        base = np.arcsin(ratio)
        for j in range(2):
            root = base if j == 0 else np.pi - base
            turns = np.ceil((root - initial_arm_angle) / (2.0 * np.pi))
            candidate = root - 2.0 * np.pi * turns
            if candidate > best:
                best = candidate
    return best


@njit(cache=True, fastmath=True, inline="always")
def _machine_only_accelerations(theta, theta_dot, psi, psi_dot, c):
    """(theta_ddot, psi_ddot) for the machine carrying no projectile.

    Scalar port of physics.TrebuchetSimulator._machine_only_accelerations: the arm plus
    its counterweight swinging as their own body, which is what the launch runs on while
    the sling is slack. On the pulley machine M13 = 0 and M33 = 1 reduce it to
    Q_theta / M_taut exactly, as they do there.

    The forces come from _machine_only_forces, which the grounded regimes need raw; this
    is only the solve on top of them. It is inlined, so the arithmetic is unchanged.
    """
    M33, M_taut = c[4], c[16]
    Q_theta, Q_psi, M13 = _machine_only_forces(theta, theta_dot, psi, psi_dot, c)
    det = M_taut * M33 - M13 * M13
    if abs(det) < 1e-12:
        return 0.0, 0.0
    return ((M33 * Q_theta - M13 * Q_psi) / det,
            (M_taut * Q_psi - M13 * Q_theta) / det)


@njit(cache=True, fastmath=True, inline="always")
def _machine_only_forces(theta, theta_dot, psi, psi_dot, c):
    """(Q_theta, Q_psi, M13) for the machine carrying no projectile.

    Scalar port of physics.TrebuchetSimulator._machine_only_forces: the generalized
    forces and the one off-diagonal inertia term, before anything solves them for
    accelerations. The grounded regimes need them raw, because there the sling tension is
    an unknown solved alongside the accelerations rather than something the machine can be
    integrated without (see _grounded_taut_solve).

    Spelled out rather than shared with _machine_only_accelerations below, which computes
    the same three quantities and then solves them - the same split physics.py makes, and
    keeping it means the two engines' hot paths stay line-for-line comparable.
    """
    (_l_a, _l_s, _M11, _M22, _M33, _coupling, cw_swing_coupling, arm_drag_k, _proj_drag_k,
     cw_torque_const, cw_torque_cos, cw_swing_gravity_k, arm_gravity_k,
     _pgt, _pga, joint_friction, _M_taut, _back, _l_w, _moi, _lever,
     pivot_friction, _ground_y, _mp, _mcw, _rp, _hp, _floor) = c

    sin_t, cos_t = np.sin(theta), np.cos(theta)
    sin_p, cos_p = np.sin(psi), np.cos(psi)
    sin_pt = sin_p * cos_t - cos_p * sin_t
    cos_pt = cos_p * cos_t + sin_p * sin_t

    arm_drag_torque = -np.copysign(arm_drag_k * theta_dot * theta_dot, theta_dot)
    bearing_torque = -pivot_friction * theta_dot / np.sqrt(
        theta_dot * theta_dot + _FRICTION_EPS * _FRICTION_EPS
    )
    Q_theta = (
        -cw_swing_coupling * sin_pt * psi_dot * psi_dot
        + cw_torque_const + cw_torque_cos * cos_t
        - arm_gravity_k * cos_t
        + arm_drag_torque
        - joint_friction * theta_dot
        + bearing_torque
    )
    Q_psi = cw_swing_coupling * sin_pt * theta_dot * theta_dot - cw_swing_gravity_k * cos_p
    return Q_theta, Q_psi, -cw_swing_coupling * cos_pt


@njit(cache=True, fastmath=True, inline="always")
def _grounded_geometry(theta, px, l_a, h_T, ground_y):
    """Sling geometry for a projectile lying on the ground under an arm at `theta`.

    Scalar port of physics.TrebuchetSimulator._grounded_geometry. Returns
    (ex, ey, a, b, dist): the unit vector from the arm tip to the projectile, the tip's
    velocity and acceleration directions projected onto it, and the actual separation -
    the measured one rather than the sling length, so the geometry stays honest about the
    integrator's drift.
    """
    sin_t, cos_t = np.sin(theta), np.cos(theta)
    tip_x, tip_y = l_a * cos_t, l_a * sin_t + h_T
    dx, dy = px - tip_x, ground_y - tip_y
    dist = np.sqrt(dx * dx + dy * dy)
    if dist < 1e-12:
        dist = 1e-12
    ex, ey = dx / dist, dy / dist
    a = -l_a * sin_t * ex + l_a * cos_t * ey
    b = -l_a * cos_t * ex - l_a * sin_t * ey
    return ex, ey, a, b, dist


@njit(cache=True, fastmath=True, inline="always")
def _grounded_taut_solve(y, c, projectile_mass, h_T):
    """(theta_ddot, psi_ddot, ax, tension, normal) with the sling taut over the ground.

    Two constraints act on the projectile at once - the sling holds its distance from the
    arm tip, the ground holds its height - which between them leave the machine's own
    coordinates as the only freedom. Rather than eliminate the projectile symbolically,
    solve for the sling tension alongside the accelerations: the projectile's vertical
    equation then just reports the normal force, and both constraint forces come out where
    this regime's exit events can read them. Unknowns are theta_ddot, psi_ddot and the
    tension; psi_ddot follows from the machine's second equation, leaving a 2x2:

        M_eff * theta_ddot - a * T = Q_eff          (machine, swing eliminated)
        a * theta_ddot + (ex^2/m) * T = R           (sling length, twice differentiated)

    physics.py splits this across `grounded_forces` and `_grounded_taut_dynamics`, which
    solve the same 2x2 for different halves of the answer; one solve here returns both,
    which is the same arithmetic in the same order.

    The projectile slides without friction, so the only horizontal force on it besides the
    sling is aerodynamic drag.
    """
    l_a, M33, proj_drag_k, M_taut = c[0], c[4], c[8], c[16]
    theta, theta_dot, px, pvx, psi, psi_dot = y[0], y[1], y[2], y[4], y[6], y[7]

    ex, ey, a, b, dist = _grounded_geometry(theta, px, l_a, h_T, c[22])
    Q_theta, Q_psi, M13 = _machine_only_forces(theta, theta_dot, psi, psi_dot, c)
    M_eff = M_taut - M13 * M13 / M33
    Q_eff = Q_theta - M13 * Q_psi / M33

    drag_x = -proj_drag_k * abs(pvx) * pvx
    # Relative velocity of projectile and tip, squared - the centripetal term the rigid
    # sling needs to keep its length.
    rel_x = pvx + theta_dot * l_a * np.sin(theta)
    rel_y = -theta_dot * l_a * np.cos(theta)
    w = rel_x * rel_x + rel_y * rel_y

    R = drag_x * ex / projectile_mass - theta_dot * theta_dot * b + w / dist
    det = M_eff * (ex * ex / projectile_mass) + a * a
    if abs(det) < 1e-15:
        return 0.0, 0.0, 0.0, 0.0, projectile_mass * G

    theta_ddot = (Q_eff * (ex * ex / projectile_mass) + a * R) / det
    tension = (M_eff * R - a * Q_eff) / det
    psi_ddot = (Q_psi - M13 * theta_ddot) / M33
    ax = (-tension * ex + drag_x) / projectile_mass
    return theta_ddot, psi_ddot, ax, tension, tension * ey + projectile_mass * G


@njit(cache=True, fastmath=True, inline="always")
def _grounded_taut_derivs(y, c, projectile_mass, h_T, out):
    """Sling taut, projectile sliding on the ground; the ground pins py and pvy at zero."""
    theta_ddot, psi_ddot, ax, tension, _normal = _grounded_taut_solve(y, c, projectile_mass, h_T)
    out[0] = y[1]
    out[1] = theta_ddot
    out[2] = y[4]
    out[3] = 0.0
    out[4] = ax
    out[5] = 0.0
    out[6] = y[7]
    out[7] = psi_ddot
    cw_T = _cw_link_tension(y[0], y[1], theta_ddot, y[6], y[7], c, c[24], c[25], c[26] > 0.5)
    out[8], out[9], out[10] = _quadrature_rates(tension, cw_T, c[27], c[26] > 0.5)


@njit(cache=True, fastmath=True, inline="always")
def _grounded_slack_derivs(y, c, projectile_mass, out):
    """Sling slack and the projectile lying on the ground: the two are uncoupled.

    Scalar port of physics.TrebuchetSimulator._grounded_slack_dynamics. The machine swings
    as its own body and the projectile keeps whatever speed it had along the ground,
    shedding it only to air drag - the ground is frictionless by choice, so nothing else
    slows it.
    """
    theta_ddot, psi_ddot = _machine_only_accelerations(y[0], y[1], y[6], y[7], c)
    pvx = y[4]
    out[0] = y[1]
    out[1] = theta_ddot
    out[2] = pvx
    out[3] = 0.0
    out[4] = -c[8] * abs(pvx) * pvx / projectile_mass
    out[5] = 0.0
    out[6] = y[7]
    out[7] = psi_ddot
    cw_T = _cw_link_tension(y[0], y[1], theta_ddot, y[6], y[7], c, c[24], c[25], c[26] > 0.5)
    out[8], out[9], out[10] = _quadrature_rates(0.0, cw_T, c[27], c[26] > 0.5)


@njit(cache=True, fastmath=True, inline="always")
def _slack_derivs(y, c, projectile_mass, out):
    """Launch dynamics while the sling is slack, written into `out`.

    Scalar port of physics.TrebuchetSimulator._launch_slack_dynamics, on that engine's
    slack state layout: [theta, theta_dot, px, py, pvx, pvy, psi, psi_dot] - the machine
    running as its own body, and the projectile in free flight under the same drag law
    trajectory.py uses.
    """
    theta_ddot, psi_ddot = _machine_only_accelerations(y[0], y[1], y[6], y[7], c)
    pvx, pvy = y[4], y[5]
    speed = np.sqrt(pvx * pvx + pvy * pvy)
    drag_accel = -c[8] * speed / projectile_mass if speed > 1e-12 else 0.0
    out[0] = y[1]
    out[1] = theta_ddot
    out[2] = pvx
    out[3] = pvy
    out[4] = drag_accel * pvx
    out[5] = -G + drag_accel * pvy
    out[6] = y[7]
    out[7] = psi_ddot
    cw_T = _cw_link_tension(y[0], y[1], theta_ddot, y[6], y[7], c, c[24], c[25], c[26] > 0.5)
    out[8], out[9], out[10] = _quadrature_rates(0.0, cw_T, c[27], c[26] > 0.5)


@njit(cache=True, fastmath=True, inline="always")
def _derivs8(y, c, projectile_mass, h_T, regime, out):
    """Whichever of the three eight-component regimes' dynamics `regime` names.

    A branch per right-hand-side call rather than three copies of the tableau. The taut
    regime is the hot path a converged design spends its whole launch in and keeps its own
    unrolled stepper; these three share one (see _integrate_eight_segment).
    """
    if regime == _SLACK:
        _slack_derivs(y, c, projectile_mass, out)
    elif regime == _TAUT_GROUND:
        _grounded_taut_derivs(y, c, projectile_mass, h_T, out)
    else:
        _grounded_slack_derivs(y, c, projectile_mass, out)


@njit(cache=True, fastmath=True, inline="always")
def _slack_state_from_taut(theta, theta_dot, alpha, alpha_dot, psi, psi_dot, l_a, l_s, h_T, out):
    """Map a taut state into the slack layout: the projectile cut loose where it stands."""
    sin_t, cos_t = np.sin(theta), np.cos(theta)
    sin_a, cos_a = np.sin(alpha), np.cos(alpha)
    out[0] = theta
    out[1] = theta_dot
    out[2] = l_a * cos_t + l_s * cos_a
    out[3] = l_a * sin_t + l_s * sin_a + h_T
    out[4] = -l_a * theta_dot * sin_t - l_s * alpha_dot * sin_a
    out[5] = l_a * theta_dot * cos_t + l_s * alpha_dot * cos_a
    out[6] = psi
    out[7] = psi_dot
    for i in range(N_QUADRATURE):
        out[8 + i] = 0.0


@njit(cache=True, fastmath=True)
def _apply_snap(y, c, projectile_mass, h_T, taut_out, slack_out):
    """Inelastic re-tension snap, at the instant the string comes taut again.

    Scalar port of physics.TrebuchetSimulator._apply_snap. An impulse along the string
    removes exactly the radial separation velocity: it conserves momentum and can only
    ever destroy energy, so the string never acts as a spring. Writes the post-snap
    physics into both layouts - the caller picks a regime by the resulting tension - and
    returns the energy destroyed.
    """
    l_a, l_s, M33, cw_swing_coupling, M_taut = c[0], c[1], c[4], c[6], c[16]
    theta, theta_dot = y[0], y[1]
    px, py, pvx, pvy = y[2], y[3], y[4], y[5]
    psi, psi_dot = y[6], y[7]

    sin_t, cos_t = np.sin(theta), np.cos(theta)
    tip_x, tip_y = l_a * cos_t, l_a * sin_t + h_T
    dx, dy = px - tip_x, py - tip_y
    dist = np.sqrt(dx * dx + dy * dy)
    if dist < 1e-12:
        dist = 1e-12
    ex, ey = dx / dist, dy / dist
    tvx, tvy = -l_a * sin_t, l_a * cos_t

    g_dot = (pvx - theta_dot * tvx) * ex + (pvy - theta_dot * tvy) * ey
    t_dot_e = tvx * ex + tvy * ey

    # The sling pulls on theta only, but the traditional machine couples theta to the
    # counterweight swing inertially, so the arm resists with M_taut - M13^2/M33 and the
    # weight takes its share of the jerk. M13 = 0 on the pulley machine leaves
    # M_eff = M_taut and psi untouched.
    cos_p, sin_p = np.cos(psi), np.sin(psi)
    cos_pt = cos_p * cos_t + sin_p * sin_t
    M13 = -cw_swing_coupling * cos_pt
    M_eff = M_taut - M13 * M13 / M33

    energy_lost = 0.0
    if g_dot > 0.0:
        P = g_dot / (1.0 / projectile_mass + t_dot_e * t_dot_e / M_eff)
        theta_dot += P * t_dot_e / M_eff
        psi_dot -= M13 * P * t_dot_e / (M33 * M_eff)
        pvx -= P / projectile_mass * ex
        pvy -= P / projectile_mass * ey
        energy_lost = 0.5 * P * g_dot

    alpha = np.arctan2(dy, dx)
    v_tip_x, v_tip_y = theta_dot * tvx, theta_dot * tvy
    alpha_dot = ((pvx - v_tip_x) * -np.sin(alpha) + (pvy - v_tip_y) * np.cos(alpha)) / l_s

    taut_out[0] = theta
    taut_out[1] = theta_dot
    taut_out[2] = alpha
    taut_out[3] = alpha_dot
    taut_out[4] = psi
    taut_out[5] = psi_dot
    for i in range(N_QUADRATURE):
        taut_out[6 + i] = 0.0
    # Put the projectile exactly on the string circle, so a slack segment that continues
    # from here starts at separation == l_s rather than integration error above it.
    slack_out[0] = theta
    slack_out[1] = theta_dot
    slack_out[2] = tip_x + l_s * ex
    slack_out[3] = tip_y + l_s * ey
    slack_out[4] = pvx
    slack_out[5] = pvy
    slack_out[6] = psi
    slack_out[7] = psi_dot
    for i in range(N_QUADRATURE):
        slack_out[8 + i] = 0.0
    return energy_lost


@njit(cache=True, fastmath=True, inline="always")
def _launch_kinetic_energy(theta, theta_dot, pvx, pvy, psi, psi_dot, c,
                           projectile_mass, counter_weight_mass, has_pulley):
    """Kinetic energy of machine plus projectile, for pricing an impulse.

    Scalar port of physics.TrebuchetSimulator._launch_kinetic_energy, whose counterweight
    velocity comes from weight_position_velocity: straight down at cw_lever metres per
    radian on the pulley machine, and on the traditional one the pin's motion about the
    pivot plus the link's own swing about the pin.
    """
    l_w, moi_machine, cw_lever = c[18], c[19], c[20]
    if has_pulley:
        cw_vx, cw_vy = 0.0, cw_lever * theta_dot
    else:
        cw_vx = cw_lever * theta_dot * np.sin(theta) - l_w * psi_dot * np.sin(psi)
        cw_vy = -cw_lever * theta_dot * np.cos(theta) + l_w * psi_dot * np.cos(psi)
    return (
        0.5 * projectile_mass * (pvx * pvx + pvy * pvy)
        + 0.5 * moi_machine * theta_dot * theta_dot
        + 0.5 * counter_weight_mass * (cw_vx * cw_vx + cw_vy * cw_vy)
    )


@njit(cache=True, fastmath=True)
def _apply_ground_impulse(y, c, projectile_mass, counter_weight_mass, has_pulley, h_T,
                          radial_target, out):
    """Impulse a projectile onto the ground, through the sling and the ground at once.

    Scalar port of physics.TrebuchetSimulator._apply_ground_impulse. Two impulses act
    together: the ground's, straight up, and the sling's, along its own line. Two
    conditions pin them - the projectile ends with no vertical speed, and the sling ends at
    `radial_target` separation rate (0 both for a landing that must stay landed and for a
    re-tension, which is the same statement). Solving them together is what keeps the arm's
    share of the jerk honest: a landing on a taut sling is felt by the arm, and dropping
    that term would quietly create energy in the machine.

    Writes the grounded state into `out` and returns the kinetic energy destroyed, always
    >= 0 - both impulses are dissipative.
    """
    l_a, M33, cw_swing_coupling, M_taut = c[0], c[4], c[6], c[16]
    theta, theta_dot = y[0], y[1]
    px, pvx, pvy = y[2], y[4], y[5]
    psi, psi_dot = y[6], y[7]

    ex, ey, a, _b, _dist = _grounded_geometry(theta, px, l_a, h_T, c[22])
    sin_t, cos_t = np.sin(theta), np.cos(theta)
    cos_p, sin_p = np.cos(psi), np.sin(psi)
    M13 = -cw_swing_coupling * (cos_p * cos_t + sin_p * sin_t)
    M_eff = M_taut - M13 * M13 / M33

    before = _launch_kinetic_energy(theta, theta_dot, pvx, pvy, psi, psi_dot, c,
                                    projectile_mass, counter_weight_mass, has_pulley)

    # g' = radial_target and vy' = 0, solved for the sling impulse; the ground's follows
    # from it. Positive pulls the projectile toward the tip, as in _apply_snap.
    g0 = (pvx - theta_dot * (-l_a * sin_t)) * ex + (pvy - theta_dot * (l_a * cos_t)) * ey
    denom = (1.0 - ey * ey) / projectile_mass + a * a / M_eff
    impulse = 0.0
    if abs(denom) >= 1e-15:
        impulse = (g0 - radial_target - pvy * ey) / denom

    pvx -= impulse * ex / projectile_mass
    pvy -= impulse * ey / projectile_mass
    theta_dot += impulse * a / M_eff
    psi_dot -= M13 * impulse * a / (M33 * M_eff)
    # Whatever vertical speed is left is the ground's to absorb; it takes the momentum away
    # rather than passing it back through the machine.
    pvy = 0.0

    after = _launch_kinetic_energy(theta, theta_dot, pvx, pvy, psi, psi_dot, c,
                                   projectile_mass, counter_weight_mass, has_pulley)

    out[0] = theta
    out[1] = theta_dot
    out[2] = px
    out[3] = c[22]
    out[4] = pvx
    out[5] = 0.0
    out[6] = psi
    out[7] = psi_dot
    for i in range(N_QUADRATURE):
        out[8 + i] = 0.0
    return max(0.0, before - after)


@njit(cache=True, fastmath=True, inline="always")
def _taut_state_from_grounded(y, c, h_T, out):
    """Lift a grounded projectile into the taut six-component layout.

    The sling angle and its rate are read off the tip-to-projectile line, which is defined
    whether or not the sling is at full stretch (physics._effective_string_state).
    """
    l_a = c[0]
    theta, theta_dot = y[0], y[1]
    sin_t, cos_t = np.sin(theta), np.cos(theta)
    tip_x, tip_y = l_a * cos_t, l_a * sin_t + h_T
    dx, dy = y[2] - tip_x, y[3] - tip_y
    dist = np.sqrt(dx * dx + dy * dy)
    alpha = np.arctan2(dy, dx)
    v_tip_x, v_tip_y = -l_a * theta_dot * sin_t, l_a * theta_dot * cos_t
    alpha_dot = 0.0
    if dist > 1e-12:
        alpha_dot = ((y[4] - v_tip_x) * -np.sin(alpha) + (y[5] - v_tip_y) * np.cos(alpha)) / dist
    out[0] = theta
    out[1] = theta_dot
    out[2] = alpha
    out[3] = alpha_dot
    out[4] = y[6]
    out[5] = y[7]
    for i in range(N_QUADRATURE):
        out[6 + i] = 0.0


@njit(cache=True, fastmath=True, inline="always")
def _cw_link_tension(theta, theta_dot, theta_ddot, psi, psi_dot, c,
                     counter_weight_mass, pulley_radius, has_pulley):
    """Scalar port of physics.TrebuchetSimulator._cw_link_tension.

    Both linkages hold the counterweight on a link this model keeps rigid, so on both a
    negative value marks the run unphysical from there on and the objective charges for
    the impulse it accumulates.
    """
    if has_pulley:
        return counter_weight_mass * (G + pulley_radius * theta_ddot)
    l_w, l_cw = c[18], c[20]
    sin_pt = np.sin(psi - theta)
    cos_pt = np.cos(psi - theta)
    return counter_weight_mass * (
        l_w * psi_dot * psi_dot
        + l_cw * (theta_ddot * sin_pt - theta_dot * theta_dot * cos_pt)
        - G * np.sin(psi)
    )


@njit(cache=True, fastmath=True, inline="always")
def _tensions(theta, theta_dot, alpha, alpha_dot, psi, psi_dot, theta_ddot, c,
              projectile_mass, counter_weight_mass, pulley_radius, has_pulley):
    """Scalar port of physics.TrebuchetSimulator.constraint_tensions (theta_ddot passed in)."""
    l_a, l_s, proj_drag_k = c[0], c[1], c[8]
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
    cw_tension = _cw_link_tension(theta, theta_dot, theta_ddot, psi, psi_dot, c,
                                  counter_weight_mass, pulley_radius, has_pulley)
    return string_tension, cw_tension


@njit(cache=True, fastmath=True)
def _integrate_taut_segment(t0, theta, theta_dot, alpha, alpha_dot, psi, psi_dot,
                            c, release_angle, t_max, rtol, atol,
                            projectile_mass, counter_weight_mass, pulley_radius, has_pulley,
                            h_T, initial_arm_angle, out):
    """Integrate one taut-sling stretch, until release, slack onset, or t_max.

    Mirrors the taut arm of physics.TrebuchetSimulator._integrate_launch, whose taut
    segment carries four terminal events: the sling reaching the release pin's angle
    (measured from the arm, not the world - see config.TrebuchetParams.release_angle), the
    string tension crossing zero downward (a rope would go slack and the projectile fly
    free), the projectile reaching the ground, and the beam reaching it. Returns
    (status, t) with status _SEG_TMAX /
    _SEG_RELEASE / _SEG_SLING_SLACK / _SEG_LANDING / _SEG_ARM_GROUND, and writes the
    nine-component taut state at that instant into `out`.

    The impulses and the deficit are the last three of those nine: running integrals
    carried through the tableau alongside the dynamics rather than summed over accepted
    steps afterwards, exactly as physics.py carries them (see physics.N_QUADRATURE for why
    a trapezoid over the solver's own grid was not good enough). They start each segment at
    zero, so the launch's totals are the sums the caller accumulates.

    Nine components through the Dormand-Prince tableau, matching the reference engine's
    layout - including on the pulley machine, where psi stays inert but still counts toward
    the error norm exactly as it does under scipy. Only the six physical ones are carried
    as stage *values*: the integrals never feed back into the dynamics, so each stage call
    passes six and takes nine rates back.
    """
    l_a, l_s, ground_y = c[0], c[1], c[22]
    theta_arm_ground = _first_arm_ground_angle(l_a, c[17], h_T, initial_arm_angle)

    # The pin's angle, re-anchored to the copy sitting just below where the sling is now -
    # physics.TrebuchetSimulator._release_target, and see it for why an event on an angle
    # has to be an event on a number first.
    span = 2.0 * np.pi
    release_target = release_angle + span * (np.ceil(((alpha - theta) - release_angle) / span) - 1.0)

    t = t0
    # The three integrals, at zero where every segment starts them.
    q1 = 0.0
    q2 = 0.0
    q3 = 0.0
    f0_1, f0_2, f0_3, f0_4, f0_5, f0_6, f0_7, f0_8, f0_9 = _trebuchet_dynamics(
        theta, theta_dot, alpha, alpha_dot, psi, psi_dot, c
    )

    # Same opening step scipy would take. Every segment restarts the integrator, and a
    # fixed guess put the two engines on visibly different grids over the short ones -
    # which is where a metric summed over accepted steps, like the counterweight rope's
    # compression impulse, is most sensitive to where the steps fall.
    s0_1 = atol + rtol * abs(theta)
    s0_2 = atol + rtol * abs(theta_dot)
    s0_3 = atol + rtol * abs(alpha)
    s0_4 = atol + rtol * abs(alpha_dot)
    s0_5 = atol + rtol * abs(psi)
    s0_6 = atol + rtol * abs(psi_dot)
    # The integrals are at zero, so their scale is their own atol alone.
    s0_7 = QUADRATURE_ATOL
    s0_8 = QUADRATURE_ATOL
    s0_9 = QUADRATURE_ATOL
    d0 = np.sqrt(((theta / s0_1) ** 2 + (theta_dot / s0_2) ** 2 + (alpha / s0_3) ** 2
                  + (alpha_dot / s0_4) ** 2 + (psi / s0_5) ** 2 + (psi_dot / s0_6) ** 2) / 9.0)
    d1 = np.sqrt(((f0_1 / s0_1) ** 2 + (f0_2 / s0_2) ** 2 + (f0_3 / s0_3) ** 2
                  + (f0_4 / s0_4) ** 2 + (f0_5 / s0_5) ** 2 + (f0_6 / s0_6) ** 2
                  + (f0_7 / s0_7) ** 2 + (f0_8 / s0_8) ** 2 + (f0_9 / s0_9) ** 2) / 9.0)
    h0 = _probe_step(d0, d1, t_max - t0)
    p1_1, p1_2, p1_3, p1_4, p1_5, p1_6, p1_7, p1_8, p1_9 = _trebuchet_dynamics(
        theta + h0 * f0_1, theta_dot + h0 * f0_2, alpha + h0 * f0_3,
        alpha_dot + h0 * f0_4, psi + h0 * f0_5, psi_dot + h0 * f0_6, c
    )
    d2 = np.sqrt((((p1_1 - f0_1) / s0_1) ** 2 + ((p1_2 - f0_2) / s0_2) ** 2
                  + ((p1_3 - f0_3) / s0_3) ** 2 + ((p1_4 - f0_4) / s0_4) ** 2
                  + ((p1_5 - f0_5) / s0_5) ** 2 + ((p1_6 - f0_6) / s0_6) ** 2
                  + ((p1_7 - f0_7) / s0_7) ** 2 + ((p1_8 - f0_8) / s0_8) ** 2
                  + ((p1_9 - f0_9) / s0_9) ** 2) / 9.0) / h0
    h = _initial_step(h0, d1, d2, t_max - t0)

    g_prev = (alpha - theta) - release_target
    gg_prev = theta - theta_arm_ground
    gl_prev = l_a * np.sin(theta) + l_s * np.sin(alpha) + h_T - ground_y
    # Not accumulated any more - the integrals do that - but still the event function for
    # the sling letting go, so it is carried step to step to bracket the zero crossing.
    string_T_prev, _cw_T_prev = _tensions(
        theta, theta_dot, alpha, alpha_dot, psi, psi_dot, f0_2, c,
        projectile_mass, counter_weight_mass, pulley_radius, has_pulley,
    )

    for _ in range(MAX_STEPS):
        if t >= t_max:
            out[0] = theta; out[1] = theta_dot; out[2] = alpha
            out[3] = alpha_dot; out[4] = psi; out[5] = psi_dot
            out[6] = q1; out[7] = q2; out[8] = q3
            return _SEG_TMAX, t
        if t + h > t_max:
            h = t_max - t

        y2_1 = theta + h * A21 * f0_1
        y2_2 = theta_dot + h * A21 * f0_2
        y2_3 = alpha + h * A21 * f0_3
        y2_4 = alpha_dot + h * A21 * f0_4
        y2_5 = psi + h * A21 * f0_5
        y2_6 = psi_dot + h * A21 * f0_6
        k2_1, k2_2, k2_3, k2_4, k2_5, k2_6, k2_7, k2_8, k2_9 = _trebuchet_dynamics(y2_1, y2_2, y2_3, y2_4, y2_5, y2_6, c)

        y3_1 = theta + h * (A31 * f0_1 + A32 * k2_1)
        y3_2 = theta_dot + h * (A31 * f0_2 + A32 * k2_2)
        y3_3 = alpha + h * (A31 * f0_3 + A32 * k2_3)
        y3_4 = alpha_dot + h * (A31 * f0_4 + A32 * k2_4)
        y3_5 = psi + h * (A31 * f0_5 + A32 * k2_5)
        y3_6 = psi_dot + h * (A31 * f0_6 + A32 * k2_6)
        k3_1, k3_2, k3_3, k3_4, k3_5, k3_6, k3_7, k3_8, k3_9 = _trebuchet_dynamics(y3_1, y3_2, y3_3, y3_4, y3_5, y3_6, c)

        y4_1 = theta + h * (A41 * f0_1 + A42 * k2_1 + A43 * k3_1)
        y4_2 = theta_dot + h * (A41 * f0_2 + A42 * k2_2 + A43 * k3_2)
        y4_3 = alpha + h * (A41 * f0_3 + A42 * k2_3 + A43 * k3_3)
        y4_4 = alpha_dot + h * (A41 * f0_4 + A42 * k2_4 + A43 * k3_4)
        y4_5 = psi + h * (A41 * f0_5 + A42 * k2_5 + A43 * k3_5)
        y4_6 = psi_dot + h * (A41 * f0_6 + A42 * k2_6 + A43 * k3_6)
        k4_1, k4_2, k4_3, k4_4, k4_5, k4_6, k4_7, k4_8, k4_9 = _trebuchet_dynamics(y4_1, y4_2, y4_3, y4_4, y4_5, y4_6, c)

        y5_1 = theta + h * (A51 * f0_1 + A52 * k2_1 + A53 * k3_1 + A54 * k4_1)
        y5_2 = theta_dot + h * (A51 * f0_2 + A52 * k2_2 + A53 * k3_2 + A54 * k4_2)
        y5_3 = alpha + h * (A51 * f0_3 + A52 * k2_3 + A53 * k3_3 + A54 * k4_3)
        y5_4 = alpha_dot + h * (A51 * f0_4 + A52 * k2_4 + A53 * k3_4 + A54 * k4_4)
        y5_5 = psi + h * (A51 * f0_5 + A52 * k2_5 + A53 * k3_5 + A54 * k4_5)
        y5_6 = psi_dot + h * (A51 * f0_6 + A52 * k2_6 + A53 * k3_6 + A54 * k4_6)
        k5_1, k5_2, k5_3, k5_4, k5_5, k5_6, k5_7, k5_8, k5_9 = _trebuchet_dynamics(y5_1, y5_2, y5_3, y5_4, y5_5, y5_6, c)

        y6_1 = theta + h * (A61 * f0_1 + A62 * k2_1 + A63 * k3_1 + A64 * k4_1 + A65 * k5_1)
        y6_2 = theta_dot + h * (A61 * f0_2 + A62 * k2_2 + A63 * k3_2 + A64 * k4_2 + A65 * k5_2)
        y6_3 = alpha + h * (A61 * f0_3 + A62 * k2_3 + A63 * k3_3 + A64 * k4_3 + A65 * k5_3)
        y6_4 = alpha_dot + h * (A61 * f0_4 + A62 * k2_4 + A63 * k3_4 + A64 * k4_4 + A65 * k5_4)
        y6_5 = psi + h * (A61 * f0_5 + A62 * k2_5 + A63 * k3_5 + A64 * k4_5 + A65 * k5_5)
        y6_6 = psi_dot + h * (A61 * f0_6 + A62 * k2_6 + A63 * k3_6 + A64 * k4_6 + A65 * k5_6)
        k6_1, k6_2, k6_3, k6_4, k6_5, k6_6, k6_7, k6_8, k6_9 = _trebuchet_dynamics(y6_1, y6_2, y6_3, y6_4, y6_5, y6_6, c)

        yn_1 = theta + h * (B1 * f0_1 + B3 * k3_1 + B4 * k4_1 + B5 * k5_1 + B6 * k6_1)
        yn_2 = theta_dot + h * (B1 * f0_2 + B3 * k3_2 + B4 * k4_2 + B5 * k5_2 + B6 * k6_2)
        yn_3 = alpha + h * (B1 * f0_3 + B3 * k3_3 + B4 * k4_3 + B5 * k5_3 + B6 * k6_3)
        yn_4 = alpha_dot + h * (B1 * f0_4 + B3 * k3_4 + B4 * k4_4 + B5 * k5_4 + B6 * k6_4)
        yn_5 = psi + h * (B1 * f0_5 + B3 * k3_5 + B4 * k4_5 + B5 * k5_5 + B6 * k6_5)
        yn_6 = psi_dot + h * (B1 * f0_6 + B3 * k3_6 + B4 * k4_6 + B5 * k5_6 + B6 * k6_6)
        yn_7 = q1 + h * (B1 * f0_7 + B3 * k3_7 + B4 * k4_7 + B5 * k5_7 + B6 * k6_7)
        yn_8 = q2 + h * (B1 * f0_8 + B3 * k3_8 + B4 * k4_8 + B5 * k5_8 + B6 * k6_8)
        yn_9 = q3 + h * (B1 * f0_9 + B3 * k3_9 + B4 * k4_9 + B5 * k5_9 + B6 * k6_9)
        k7_1, k7_2, k7_3, k7_4, k7_5, k7_6, k7_7, k7_8, k7_9 = _trebuchet_dynamics(yn_1, yn_2, yn_3, yn_4, yn_5, yn_6, c)

        err_1 = h * (E1 * f0_1 + E3 * k3_1 + E4 * k4_1 + E5 * k5_1 + E6 * k6_1 + E7 * k7_1)
        err_2 = h * (E1 * f0_2 + E3 * k3_2 + E4 * k4_2 + E5 * k5_2 + E6 * k6_2 + E7 * k7_2)
        err_3 = h * (E1 * f0_3 + E3 * k3_3 + E4 * k4_3 + E5 * k5_3 + E6 * k6_3 + E7 * k7_3)
        err_4 = h * (E1 * f0_4 + E3 * k3_4 + E4 * k4_4 + E5 * k5_4 + E6 * k6_4 + E7 * k7_4)
        err_5 = h * (E1 * f0_5 + E3 * k3_5 + E4 * k4_5 + E5 * k5_5 + E6 * k6_5 + E7 * k7_5)
        err_6 = h * (E1 * f0_6 + E3 * k3_6 + E4 * k4_6 + E5 * k5_6 + E6 * k6_6 + E7 * k7_6)
        err_7 = h * (E1 * f0_7 + E3 * k3_7 + E4 * k4_7 + E5 * k5_7 + E6 * k6_7 + E7 * k7_7)
        err_8 = h * (E1 * f0_8 + E3 * k3_8 + E4 * k4_8 + E5 * k5_8 + E6 * k6_8 + E7 * k7_8)
        err_9 = h * (E1 * f0_9 + E3 * k3_9 + E4 * k4_9 + E5 * k5_9 + E6 * k6_9 + E7 * k7_9)

        scale_1 = atol + rtol * max(abs(theta), abs(yn_1))
        scale_2 = atol + rtol * max(abs(theta_dot), abs(yn_2))
        scale_3 = atol + rtol * max(abs(alpha), abs(yn_3))
        scale_4 = atol + rtol * max(abs(alpha_dot), abs(yn_4))
        scale_5 = atol + rtol * max(abs(psi), abs(yn_5))
        scale_6 = atol + rtol * max(abs(psi_dot), abs(yn_6))
        scale_7 = QUADRATURE_ATOL + rtol * max(abs(q1), abs(yn_7))
        scale_8 = QUADRATURE_ATOL + rtol * max(abs(q2), abs(yn_8))
        scale_9 = QUADRATURE_ATOL + rtol * max(abs(q3), abs(yn_9))

        err_norm = np.sqrt(
            ((err_1 / scale_1) ** 2 + (err_2 / scale_2) ** 2 + (err_3 / scale_3) ** 2
             + (err_4 / scale_4) ** 2 + (err_5 / scale_5) ** 2 + (err_6 / scale_6) ** 2
             + (err_7 / scale_7) ** 2 + (err_8 / scale_8) ** 2 + (err_9 / scale_9) ** 2) / 9.0
        )

        if err_norm <= 1.0:
            string_T_new, _cw_T_new = _tensions(
                yn_1, yn_2, yn_3, yn_4, yn_5, yn_6, k7_2, c,
                projectile_mass, counter_weight_mass, pulley_radius, has_pulley,
            )

            # Two terminal events share this step. scipy stops at whichever comes first,
            # so localize both and take the earlier fraction - a step that reaches the
            # release angle and lets the string go slack has to resolve the same way in
            # both engines or the launches part company over a rounding.
            g_new = (yn_3 - yn_1) - release_target
            s_release = 2.0
            if g_prev > 0.0 and g_new <= 0.0:
                lo, hi = 0.0, 1.0
                for _ in range(50):
                    mid = 0.5 * (lo + hi)
                    g_mid = (_hermite(alpha, yn_3, f0_3, k7_3, h, mid)
                             - _hermite(theta, yn_1, f0_1, k7_1, h, mid)) - release_target
                    if g_mid > 0.0:
                        lo = mid
                    else:
                        hi = mid
                s_release = 0.5 * (lo + hi)

            # The beam reaching the ground is a third terminal event on this step,
            # localized on the arm angle exactly as the release is.
            gg_new = yn_1 - theta_arm_ground
            s_ground = 2.0
            if gg_prev > 0.0 and gg_new <= 0.0:
                lo, hi = 0.0, 1.0
                for _ in range(50):
                    mid = 0.5 * (lo + hi)
                    if _hermite(theta, yn_1, f0_1, k7_1, h, mid) - theta_arm_ground > 0.0:
                        lo = mid
                    else:
                        hi = mid
                s_ground = 0.5 * (lo + hi)

            s_switch = 2.0
            if string_T_prev > 0.0 and string_T_new <= 0.0:
                lo, hi = 0.0, 1.0
                for _ in range(50):
                    mid = 0.5 * (lo + hi)
                    th = _hermite(theta, yn_1, f0_1, k7_1, h, mid)
                    th_d = _hermite(theta_dot, yn_2, f0_2, k7_2, h, mid)
                    al = _hermite(alpha, yn_3, f0_3, k7_3, h, mid)
                    al_d = _hermite(alpha_dot, yn_4, f0_4, k7_4, h, mid)
                    ps = _hermite(psi, yn_5, f0_5, k7_5, h, mid)
                    ps_d = _hermite(psi_dot, yn_6, f0_6, k7_6, h, mid)
                    th_dd = _trebuchet_dynamics(th, th_d, al, al_d, ps, ps_d, c)[1]
                    T_mid, _ = _tensions(th, th_d, al, al_d, ps, ps_d, th_dd, c,
                                         projectile_mass, counter_weight_mass, pulley_radius,
                                         has_pulley)
                    if T_mid > 0.0:
                        lo = mid
                    else:
                        hi = mid
                s_switch = 0.5 * (lo + hi)

            # The projectile reaching the ground, still on a loaded sling: the fourth
            # terminal event, on its height rather than on any angle.
            #
            # Armed on a strict inequality, where scipy's direction=-1 would also accept a
            # segment starting at exactly zero. That case is a taut segment entered from
            # the ground, and _settle_grounded only sends one there when the normal force
            # has gone negative - the sling is lifting the projectile, so the height is
            # rising and neither engine fires. Accepting it here would instead let a
            # segment that begins on the line resolve the event at s = 0 and return a
            # zero-length segment, which is what the strictness is buying.
            gl_new = l_a * np.sin(yn_1) + l_s * np.sin(yn_3) + h_T - ground_y
            s_land = 2.0
            if gl_prev > 0.0 and gl_new <= 0.0:
                lo, hi = 0.0, 1.0
                for _ in range(50):
                    mid = 0.5 * (lo + hi)
                    th = _hermite(theta, yn_1, f0_1, k7_1, h, mid)
                    al = _hermite(alpha, yn_3, f0_3, k7_3, h, mid)
                    if l_a * np.sin(th) + l_s * np.sin(al) + h_T - ground_y > 0.0:
                        lo = mid
                    else:
                        hi = mid
                s_land = 0.5 * (lo + hi)

            # scipy stops at whichever event comes first and breaks a tie on the order it
            # was given them, so the comparisons below run in that same order.
            s = min(min(s_release, s_switch), min(s_land, s_ground))
            if s <= 1.0:
                if s_release <= s:
                    status = _SEG_RELEASE
                elif s_switch <= s:
                    status = _SEG_SLING_SLACK
                elif s_land <= s:
                    status = _SEG_LANDING
                else:
                    status = _SEG_ARM_GROUND
                theta_r = _hermite(theta, yn_1, f0_1, k7_1, h, s)
                theta_dot_r = _hermite(theta_dot, yn_2, f0_2, k7_2, h, s)
                alpha_r = _hermite(alpha, yn_3, f0_3, k7_3, h, s)
                alpha_dot_r = _hermite(alpha_dot, yn_4, f0_4, k7_4, h, s)
                psi_r = _hermite(psi, yn_5, f0_5, k7_5, h, s)
                psi_dot_r = _hermite(psi_dot, yn_6, f0_6, k7_6, h, s)
                out[0] = theta_r; out[1] = theta_dot_r; out[2] = alpha_r
                out[3] = alpha_dot_r; out[4] = psi_r; out[5] = psi_dot_r
                out[6] = _hermite(q1, yn_7, f0_7, k7_7, h, s)
                out[7] = _hermite(q2, yn_8, f0_8, k7_8, h, s)
                out[8] = _hermite(q3, yn_9, f0_9, k7_9, h, s)
                return status, t + h * s

            t = t + h
            string_T_prev = string_T_new
            theta, theta_dot, alpha, alpha_dot, psi, psi_dot = yn_1, yn_2, yn_3, yn_4, yn_5, yn_6
            q1, q2, q3 = yn_7, yn_8, yn_9
            f0_1, f0_2, f0_3, f0_4, f0_5, f0_6 = k7_1, k7_2, k7_3, k7_4, k7_5, k7_6
            f0_7, f0_8, f0_9 = k7_7, k7_8, k7_9
            g_prev = g_new
            gg_prev = gg_new
            gl_prev = gl_new

            factor = MAX_FACTOR if err_norm == 0.0 else min(MAX_FACTOR, SAFETY * err_norm**ERROR_EXPONENT)
            h = h * factor
        else:
            factor = max(MIN_FACTOR, SAFETY * err_norm**ERROR_EXPONENT)
            h = h * factor

    out[0] = theta; out[1] = theta_dot; out[2] = alpha
    out[3] = alpha_dot; out[4] = psi; out[5] = psi_dot
    out[6] = q1; out[7] = q2; out[8] = q3
    return _SEG_TMAX, t


@njit(cache=True, fastmath=True, inline="always")
def _tip_separation(y, l_a, l_s, h_T):
    """Tip-to-projectile distance minus the string length, on the eight-component layout.

    Negative while the projectile hangs inside the string circle; the crossing back up
    through zero is the re-tension event. Serves the grounded slack regime too - there
    y[3] is pinned at the resting projectile's own centre height, one radius up, which is
    the same statement physics.py's `ground_retension_event` makes by measuring from that
    line to the tip.
    """
    tip_x = l_a * np.cos(y[0])
    tip_y = l_a * np.sin(y[0]) + h_T
    dx, dy = y[2] - tip_x, y[3] - tip_y
    return np.sqrt(dx * dx + dy * dy) - l_s


@njit(cache=True, fastmath=True, inline="always")
def _eight_event(y, c, projectile_mass, h_T, regime, which):
    """One of a regime's two transition events, signed so every crossing is downward.

    Which pair `which` selects depends on the regime, mirroring the per-regime event lists
    in physics.TrebuchetSimulator._integrate_launch:

      _SLACK         1 re-tension (separation reaches the sling)   2 the projectile lands
      _SLACK_GROUND  1 re-tension over a grounded projectile       - (nothing left to land)
      _TAUT_GROUND   1 lift-off (the normal force lets go)         2 the sling lets go

    Re-tension is the one event that fires on the way *up*, so it is returned negated and
    the caller can test every event the same way.
    """
    if regime == _TAUT_GROUND:
        _theta_ddot, _psi_ddot, _ax, tension, normal = _grounded_taut_solve(
            y, c, projectile_mass, h_T
        )
        return normal if which == 1 else tension
    if which == 1:
        return -_tip_separation(y, c[0], c[1], h_T)
    return y[3] - c[22]


@njit(cache=True, fastmath=True, inline="always")
def _eight_tensions(y, c, projectile_mass, counter_weight_mass, pulley_radius, has_pulley,
                    h_T, regime):
    """(sling tension, counterweight-rope tension) on the eight-component layout.

    A sling under load reports its tension whether or not the projectile it is pulling
    happens to be resting on the ground; a slack one carries nothing either way. The
    counterweight link is read from whatever angular acceleration the regime produced -
    exactly the split physics._tension_metrics makes.
    """
    if regime == _TAUT_GROUND:
        theta_ddot, _psi_ddot, _ax, string_T, _normal = _grounded_taut_solve(
            y, c, projectile_mass, h_T
        )
    else:
        theta_ddot, _psi_ddot = _machine_only_accelerations(y[0], y[1], y[6], y[7], c)
        string_T = 0.0
    cw_T = _cw_link_tension(y[0], y[1], theta_ddot, y[6], y[7], c,
                            counter_weight_mass, pulley_radius, has_pulley)
    return string_T, cw_T


@njit(cache=True, fastmath=True)
def _integrate_eight_segment(t0, y, c, regime, t_max, rtol, atol,
                             projectile_mass, initial_arm_angle, h_T, out):
    """Integrate one stretch of whichever eight-component regime `regime` names.

    The three non-taut arms of physics.TrebuchetSimulator._integrate_launch share this
    stepper: the projectile is carried explicitly as
    [theta, theta_dot, px, py, pvx, pvy, psi, psi_dot] in all of them, and they differ only
    in their right-hand side (_derivs8) and in which two transition events can end them
    (_eight_event). Every one of them can also end with the beam striking the ground, but
    none of them can end at the release, which is armed in the taut regime alone: the pin
    lets go of a sling it is holding under load and swinging free, and none of these three
    is that. Returns (status, t), writing the eleven-component state at that instant into
    `out`.

    Eleven components - eight physical plus the three running integrals - on arrays rather
    than the taut segment's unrolled scalars. The taut path is where a converged design
    spends its whole launch and is the optimizer's hot loop, so it keeps the unrolled form;
    these three are the exception, and one readable array-based stepper here beats three
    more copies of the tableau. Everything counts toward the error norm, including the two
    components the grounded regimes pin at zero, so the step path matches what scipy takes
    over the same state vector.
    """
    l_a = c[0]
    theta_arm_ground = _first_arm_ground_angle(l_a, c[17], h_T, initial_arm_angle)
    has_ev2 = regime != _SLACK_GROUND
    n = 8 + N_QUADRATURE
    k1 = np.empty(n); k2 = np.empty(n); k3 = np.empty(n); k4 = np.empty(n)
    k5 = np.empty(n); k6 = np.empty(n); k7 = np.empty(n)
    stage = np.empty(n); yn = np.empty(n)

    t = t0
    _derivs8(y, c, projectile_mass, h_T, regime, k1)

    # Same opening step scipy would take - see the note in _integrate_taut_segment. `stage`
    # and `k2` are borrowed as scratch for the probe; both are overwritten on the first
    # tableau stage below before anything reads them.
    d0 = 0.0
    d1 = 0.0
    for i in range(n):
        scale = (atol if i < 8 else QUADRATURE_ATOL) + rtol * abs(y[i])
        stage[i] = scale
        d0 += (y[i] / scale) ** 2
        d1 += (k1[i] / scale) ** 2
    d0 = np.sqrt(d0 / n)
    d1 = np.sqrt(d1 / n)
    h0 = _probe_step(d0, d1, t_max - t0)
    for i in range(n):
        yn[i] = y[i] + h0 * k1[i]
    _derivs8(yn, c, projectile_mass, h_T, regime, k2)
    d2 = 0.0
    for i in range(n):
        d2 += ((k2[i] - k1[i]) / stage[i]) ** 2
    d2 = np.sqrt(d2 / n) / h0
    h = _initial_step(h0, d1, d2, t_max - t0)

    gg_prev = y[0] - theta_arm_ground
    e1_prev = _eight_event(y, c, projectile_mass, h_T, regime, 1)
    e2_prev = _eight_event(y, c, projectile_mass, h_T, regime, 2) if has_ev2 else 1.0
    retension_shrinks = 0
    # Whether this segment opened on the sling circle - see RETENSION_CIRCLE_SLOP and the
    # step rejection below.
    opened_on_circle = e1_prev > -RETENSION_CIRCLE_SLOP * c[1]

    for _ in range(MAX_STEPS):
        if t >= t_max:
            for i in range(n):
                out[i] = y[i]
            return _SEG_TMAX, t
        if t + h > t_max:
            h = t_max - t

        for i in range(n):
            stage[i] = y[i] + h * A21 * k1[i]
        _derivs8(stage, c, projectile_mass, h_T, regime, k2)
        for i in range(n):
            stage[i] = y[i] + h * (A31 * k1[i] + A32 * k2[i])
        _derivs8(stage, c, projectile_mass, h_T, regime, k3)
        for i in range(n):
            stage[i] = y[i] + h * (A41 * k1[i] + A42 * k2[i] + A43 * k3[i])
        _derivs8(stage, c, projectile_mass, h_T, regime, k4)
        for i in range(n):
            stage[i] = y[i] + h * (A51 * k1[i] + A52 * k2[i] + A53 * k3[i] + A54 * k4[i])
        _derivs8(stage, c, projectile_mass, h_T, regime, k5)
        for i in range(n):
            stage[i] = y[i] + h * (A61 * k1[i] + A62 * k2[i] + A63 * k3[i] + A64 * k4[i]
                                   + A65 * k5[i])
        _derivs8(stage, c, projectile_mass, h_T, regime, k6)
        for i in range(n):
            yn[i] = y[i] + h * (B1 * k1[i] + B3 * k3[i] + B4 * k4[i] + B5 * k5[i] + B6 * k6[i])
        _derivs8(yn, c, projectile_mass, h_T, regime, k7)

        err_sq = 0.0
        for i in range(n):
            err = h * (E1 * k1[i] + E3 * k3[i] + E4 * k4[i] + E5 * k5[i] + E6 * k6[i]
                       + E7 * k7[i])
            scale = (atol if i < 8 else QUADRATURE_ATOL) + rtol * max(abs(y[i]), abs(yn[i]))
            err_sq += (err / scale) ** 2
        err_norm = np.sqrt(err_sq / n)

        if err_norm <= 1.0:
            e1_new = _eight_event(yn, c, projectile_mass, h_T, regime, 1)
            # The sling is a rope, so while it is slack the projectile lies *inside* the
            # circle the rope sweeps: e1 = l_s - separation is non-negative for the whole
            # segment, and the re-tension is its return to zero. Every slack segment opens
            # exactly on that surface with e1 == 0 - which leaves the ordinary "was
            # positive, is now negative" test nothing to arm on. A first step long enough
            # to span the whole excursion therefore ends below zero and is accepted, and
            # the projectile is left outside a rope that cannot stretch.
            #
            # It is not a hypothetical: a traditional draw whose tension grazes zero at
            # -0.013 N leaves the circle by 1.3 micrometres for 13 milliseconds, and the
            # slack regime - a smooth pendulum plus a free-flying stone - justifies a step
            # a hundred times that. The stone escaped, flew for a second, and landed 1.79 m
            # outside the sling; the launch that should have thrown 1.4 m threw nothing.
            #
            # So a step that ends outside is simply too long, and is rejected the way a
            # failed error test is. The shrinking is self-limiting: the excursion has
            # finite width, and once one step lands inside it e1_prev is positive and the
            # ordinary test takes over.
            #
            # Only for a segment that opened *on* the circle, though. One entered from
            # TAUT_GROUND opens whereever that regime's constraint drift left it, which is
            # microns outside rather than at the rounding of zero, and that is not a step
            # that escaped - it is a state that started there. Rejecting for it shrinks
            # twenty times, gives up, and returns a zero-length segment, which is a stall
            # rather than an answer (see RETENSION_CIRCLE_SLOP).
            if opened_on_circle and e1_new < 0.0 and e1_prev <= 0.0:
                if retension_shrinks < MAX_RETENSION_SHRINKS:
                    retension_shrinks += 1
                    h = h * 0.5
                    continue
                # Unreachable on the physics (see MAX_RETENSION_SHRINKS): a projectile
                # that leaves and never returns. Call the sling taut where it stands and
                # let the caller's snap sort it out, rather than accepting the escape.
                for i in range(n):
                    out[i] = y[i]
                return _SEG_RETENSION, t

            # Up to three terminal events share this step (the taut segment has four; the
            # release is not one of ours).
            gg_new = yn[0] - theta_arm_ground
            s_ground = 2.0
            if gg_prev > 0.0 and gg_new <= 0.0:
                lo, hi = 0.0, 1.0
                for _ in range(50):
                    mid = 0.5 * (lo + hi)
                    if _hermite(y[0], yn[0], k1[0], k7[0], h, mid) - theta_arm_ground > 0.0:
                        lo = mid
                    else:
                        hi = mid
                s_ground = 0.5 * (lo + hi)

            s_ev1 = 2.0
            if e1_prev > 0.0 and e1_new <= 0.0:
                lo, hi = 0.0, 1.0
                for _ in range(50):
                    mid = 0.5 * (lo + hi)
                    for i in range(n):
                        stage[i] = _hermite(y[i], yn[i], k1[i], k7[i], h, mid)
                    if _eight_event(stage, c, projectile_mass, h_T, regime, 1) > 0.0:
                        lo = mid
                    else:
                        hi = mid
                s_ev1 = 0.5 * (lo + hi)

            e2_new = _eight_event(yn, c, projectile_mass, h_T, regime, 2) if has_ev2 else 1.0
            s_ev2 = 2.0
            if has_ev2 and e2_prev > 0.0 and e2_new <= 0.0:
                lo, hi = 0.0, 1.0
                for _ in range(50):
                    mid = 0.5 * (lo + hi)
                    for i in range(n):
                        stage[i] = _hermite(y[i], yn[i], k1[i], k7[i], h, mid)
                    if _eight_event(stage, c, projectile_mass, h_T, regime, 2) > 0.0:
                        lo = mid
                    else:
                        hi = mid
                s_ev2 = 0.5 * (lo + hi)

            s = min(min(s_ev1, s_ev2), s_ground)
            if s <= 1.0:
                # Same tie-break order the reference engine's event list gives: the
                # regime's own two, then the beam.
                if s_ev1 <= s:
                    status = _SEG_LIFTOFF if regime == _TAUT_GROUND else _SEG_RETENSION
                elif s_ev2 <= s:
                    status = _SEG_SLING_SLACK if regime == _TAUT_GROUND else _SEG_LANDING
                else:
                    status = _SEG_ARM_GROUND
                for i in range(n):
                    out[i] = _hermite(y[i], yn[i], k1[i], k7[i], h, s)
                return status, t + h * s

            t = t + h
            for i in range(n):
                y[i] = yn[i]
                k1[i] = k7[i]
            gg_prev = gg_new
            e1_prev = e1_new
            e2_prev = e2_new

            factor = MAX_FACTOR if err_norm == 0.0 else min(MAX_FACTOR, SAFETY * err_norm**ERROR_EXPONENT)
            h = h * factor
        else:
            factor = max(MIN_FACTOR, SAFETY * err_norm**ERROR_EXPONENT)
            h = h * factor

    for i in range(n):
        out[i] = y[i]
    return _SEG_TMAX, t



@njit(cache=True, fastmath=True, inline="always")
def _ground_start_state(theta0, psi_rest, l_a, l_s, h_T, ground_y, out):
    """The cocked pose with the projectile lying on the ground, or False if it cannot be.

    Scalar port of physics.TrebuchetSimulator.ground_start_state. A trebuchet is loaded by
    laying the projectile out behind the machine with the sling stretched along the ground,
    as far back from the pivot as the sling reaches - not by dangling it in mid-air, and
    certainly not by burying it, which is where the hanging pose put it. So the projectile
    starts on the ground at the far end of the sling, on the side the tip leans towards,
    which is the side away from the throw.

    False when the sling cannot reach the ground from the cocked tip - a pulley machine's
    tip stands a metre up with a 24 cm sling - in which case the sling really does hang and
    the six-component cocked pose describes it.
    """
    tip_x = l_a * np.cos(theta0)
    tip_y = l_a * np.sin(theta0) + h_T
    # Measured to the resting stone's centre, which sits one radius up.
    drop = tip_y - ground_y
    if drop < 0.0 or drop > l_s:
        return False
    reach = np.sqrt(max(0.0, l_s * l_s - drop * drop))
    if tip_x != 0.0:
        px = tip_x + np.copysign(reach, tip_x)
    else:
        px = tip_x - reach
    out[0] = theta0
    out[1] = 0.0
    out[2] = px
    out[3] = ground_y
    out[4] = 0.0
    out[5] = 0.0
    out[6] = psi_rest
    out[7] = 0.0
    for i in range(N_QUADRATURE):
        out[8 + i] = 0.0
    return True


@njit(cache=True, fastmath=True, inline="always")
def _settle_grounded(y, c, projectile_mass, h_T, taut_out):
    """Which grounded regime a just-landed or just-snapped projectile belongs in.

    Scalar port of physics.TrebuchetSimulator._settle_grounded. Both constraint forces are
    one-sided, so a state can arrive already violating one of them; deciding here rather
    than letting the regime's own event fire at t0 avoids a zero-length segment. Writes the
    six-component taut state into `taut_out` when the answer is that the sling has already
    picked the projectile up.
    """
    _theta_ddot, _psi_ddot, _ax, tension, normal = _grounded_taut_solve(
        y, c, projectile_mass, h_T
    )
    if tension < 0.0:
        return _SLACK_GROUND
    if normal >= 0.0:
        return _TAUT_GROUND
    _taut_state_from_grounded(y, c, h_T, taut_out)
    return _TAUT


@njit(cache=True, fastmath=True)
def _integrate_launch(theta0, alpha0, psi0, c, release_angle, t_max, rtol, atol,
                      projectile_mass, counter_weight_mass, pulley_radius, has_pulley,
                      tension_floor, h_T):
    """Integrate the launch through all four regimes until release, ground, or t_max.

    The outer loop of physics.TrebuchetSimulator._integrate_launch. The sling is either
    loaded or not and the projectile is either on the ground or off it, independently, so a
    launch is a stitched walk through _TAUT / _SLACK / _TAUT_GROUND / _SLACK_GROUND. Every
    regime can end at the release angle or with the beam striking the ground - the latter
    for good, since a machine that has dug itself in has nothing left to throw - and beyond
    that each has its own exits:

      _TAUT          tension crosses zero (the rope lets go) -> _SLACK
                     the projectile reaches the ground       -> grounded, via an impulse
      _SLACK         separation grows back to the sling      -> snap, then taut or slack
                     the projectile reaches the ground       -> _SLACK_GROUND
      _TAUT_GROUND   normal force crosses zero (lift-off)    -> _TAUT
                     tension crosses zero                    -> _SLACK_GROUND
      _SLACK_GROUND  separation grows back to the sling      -> snap, then grounded again

    Every crossing into the ground costs energy - the ground takes the projectile's
    downward momentum - which is accumulated the same way a snap's loss is, and reported
    separately from it.

    The three compression/deficit integrals are read off the end of each segment's state
    vector and summed here (see N_QUADRATURE): every segment restarts them at zero, so no
    interval ever spans a snap, and this is exactly what physics._tension_metrics does with
    the reference engine's segments.

    Returns (released, t, theta, theta_dot, psi, px, py, pvx, pvy, string_impulse,
    cw_impulse, sling_deficit, snap_energy, ground_energy, arm_ground, start_py). The
    projectile is reported as position and velocity rather than a sling angle, because a
    release out of a slack stretch has no sling angle to report - which is exactly how
    physics.LaunchSolution hands it over. `start_py` is where the projectile actually
    started, which the caller needs for the potential energy the launch spent on it.
    """
    l_a, l_s, ground_y = c[0], c[1], c[22]
    # Six/eight physical components plus the three integrals; the state maps between
    # regimes write zeros into the tail, which is the restart.
    taut = np.empty(6 + N_QUADRATURE)
    slack = np.empty(8 + N_QUADRATURE)
    snap_taut = np.empty(6 + N_QUADRATURE)
    snap_slack = np.empty(8 + N_QUADRATURE)
    landed = np.empty(8 + N_QUADRATURE)
    seg_out = np.empty(8 + N_QUADRATURE)

    theta, theta_dot = theta0, 0.0
    alpha, alpha_dot = alpha0, 0.0
    psi, psi_dot = psi0, 0.0

    string_impulse = 0.0
    cw_impulse = 0.0
    sling_deficit = 0.0
    snap_energy = 0.0
    ground_energy = 0.0
    arm_ground = False
    t = 0.0

    # Which regime the launch starts in, decided the way physics._initial_launch_regime
    # decides it: a projectile the sling can reach the ground with starts lying on it and
    # the constraint forces at that pose pick the regime; one it cannot starts hanging, and
    # the tension the taut model would need there picks between taut and slack.
    psi_rest = -np.pi / 2.0
    regime = _TAUT
    if _ground_start_state(theta0, psi_rest, l_a, l_s, h_T, ground_y, slack):
        regime = _settle_grounded(slack, c, projectile_mass, h_T, taut)
        if regime == _TAUT:
            theta, theta_dot = taut[0], taut[1]
            alpha, alpha_dot = taut[2], taut[3]
            psi, psi_dot = taut[4], taut[5]
        start_py = ground_y
    else:
        theta_ddot0 = _trebuchet_dynamics(
            theta, theta_dot, alpha, alpha_dot, psi, psi_dot, c
        )[1]
        string_T0, _ = _tensions(theta, theta_dot, alpha, alpha_dot, psi, psi_dot,
                                 theta_ddot0, c, projectile_mass, counter_weight_mass,
                                 pulley_radius, has_pulley)
        start_py = l_a * np.sin(theta0) + l_s * np.sin(alpha0) + h_T
        if string_T0 < 0.0:
            _slack_state_from_taut(theta, theta_dot, alpha, alpha_dot, psi, psi_dot,
                                   l_a, l_s, h_T, slack)
            regime = _SLACK

    for _ in range(MAX_LAUNCH_SEGMENTS):
        if t >= t_max:
            break

        if regime == _TAUT:
            status, t = _integrate_taut_segment(
                t, theta, theta_dot, alpha, alpha_dot, psi, psi_dot, c, release_angle,
                t_max, rtol, atol, projectile_mass, counter_weight_mass, pulley_radius,
                has_pulley, h_T, theta0, taut,
            )
            # Clamped as physics._tension_metrics clamps them: non-negative integrands,
            # but a fifth-order rule over a kink can undershoot, and a compression impulse
            # that came out negative would pay the design back.
            string_impulse += max(0.0, taut[6])
            cw_impulse += max(0.0, taut[7])
            sling_deficit += max(0.0, taut[8])
            theta, theta_dot = taut[0], taut[1]
            alpha, alpha_dot = taut[2], taut[3]
            psi, psi_dot = taut[4], taut[5]

            if status == _SEG_RELEASE:
                sin_t, cos_t = np.sin(theta), np.cos(theta)
                sin_a, cos_a = np.sin(alpha), np.cos(alpha)
                return (True, t, theta, theta_dot, psi,
                        l_a * cos_t + l_s * cos_a,
                        l_a * sin_t + l_s * sin_a + h_T,
                        -l_a * theta_dot * sin_t - l_s * alpha_dot * sin_a,
                        l_a * theta_dot * cos_t + l_s * alpha_dot * cos_a,
                        string_impulse, cw_impulse, sling_deficit, snap_energy,
                        ground_energy, arm_ground, start_py)
            if status == _SEG_ARM_GROUND:
                arm_ground = True
                break
            if status == _SEG_TMAX:
                break

            _slack_state_from_taut(theta, theta_dot, alpha, alpha_dot, psi, psi_dot,
                                   l_a, l_s, h_T, slack)
            if status == _SEG_SLING_SLACK:
                regime = _SLACK
                continue
            # _SEG_LANDING: the projectile reached the ground on a still-loaded sling, so
            # the landing is felt by the arm as well as by the ground.
            ground_energy += _apply_ground_impulse(
                slack, c, projectile_mass, counter_weight_mass, has_pulley, h_T, 0.0, landed
            )
            regime = _settle_grounded(landed, c, projectile_mass, h_T, taut)
            if regime == _TAUT:
                theta, theta_dot = taut[0], taut[1]
                alpha, alpha_dot = taut[2], taut[3]
                psi, psi_dot = taut[4], taut[5]
            else:
                for i in range(8 + N_QUADRATURE):
                    slack[i] = landed[i]
            continue

        status, t = _integrate_eight_segment(
            t, slack, c, regime, t_max, rtol, atol, projectile_mass, theta0, h_T, seg_out,
        )
        for i in range(8 + N_QUADRATURE):
            slack[i] = seg_out[i]
        string_impulse += max(0.0, slack[8])
        cw_impulse += max(0.0, slack[9])
        sling_deficit += max(0.0, slack[10])
        theta, theta_dot, psi, psi_dot = slack[0], slack[1], slack[6], slack[7]

        if status == _SEG_ARM_GROUND:
            arm_ground = True
            break
        if status == _SEG_TMAX:
            break

        if status == _SEG_LIFTOFF:
            # The sling has taken the projectile's weight off the ground.
            _taut_state_from_grounded(slack, c, h_T, taut)
            theta, theta_dot = taut[0], taut[1]
            alpha, alpha_dot = taut[2], taut[3]
            psi, psi_dot = taut[4], taut[5]
            regime = _TAUT
            continue
        if status == _SEG_SLING_SLACK:
            # From _TAUT_GROUND; the physical state carries over unchanged, the integrals
            # restart with the segment.
            for i in range(N_QUADRATURE):
                slack[8 + i] = 0.0
            regime = _SLACK_GROUND
            continue
        if status == _SEG_LANDING:
            # From _SLACK: nothing is transmitted to the machine - the sling is carrying
            # nothing - so the whole of the downward kinetic energy is simply gone.
            ground_energy += 0.5 * projectile_mass * slack[5] * slack[5]
            slack[3] = ground_y
            slack[5] = 0.0
            for i in range(N_QUADRATURE):
                slack[8 + i] = 0.0
            regime = _SLACK_GROUND
            continue

        # _SEG_RETENSION, from one of the two slack regimes. The free snap is tried first
        # in both, because the ground's own impulse is one-sided: it can push a stone up
        # and never pull one down, and over a stone lying on the ground the sling always
        # pulls upward (the tip is above it), so solving the two together under "the stone
        # stays down" answers with the ground holding it - see
        # physics.TrebuchetSimulator._launch_transition. The stone lifting off is the
        # sling picking it up, which is the loading stroke of a machine loaded in the dirt.
        snap_e = _apply_snap(slack, c, projectile_mass, h_T, snap_taut, snap_slack)
        if regime == _SLACK_GROUND and snap_slack[5] < 0.0:
            # A sling pulling downward - nearly horizontal, under a tip almost on the
            # ground - does drive the stone in, and there the ground's impulse is real.
            snap_energy += _apply_ground_impulse(
                slack, c, projectile_mass, counter_weight_mass, has_pulley, h_T, 0.0, landed
            )
            regime = _settle_grounded(landed, c, projectile_mass, h_T, taut)
            if regime == _TAUT:
                theta, theta_dot = taut[0], taut[1]
                alpha, alpha_dot = taut[2], taut[3]
                psi, psi_dot = taut[4], taut[5]
            else:
                for i in range(8 + N_QUADRATURE):
                    slack[i] = landed[i]
            continue

        snap_energy += snap_e
        theta, theta_dot = snap_taut[0], snap_taut[1]
        alpha, alpha_dot = snap_taut[2], snap_taut[3]
        psi, psi_dot = snap_taut[4], snap_taut[5]
        theta_ddot_s = _trebuchet_dynamics(
            theta, theta_dot, alpha, alpha_dot, psi, psi_dot, c
        )[1]
        string_T_s, _ = _tensions(theta, theta_dot, alpha, alpha_dot, psi, psi_dot,
                                  theta_ddot_s, c, projectile_mass, counter_weight_mass,
                                  pulley_radius, has_pulley)
        # A tiny positive threshold, not zero: at exactly zero tension the next taut
        # segment would trip its own slack event at t0 and return a zero-length segment.
        if string_T_s > 1e-9:
            regime = _TAUT
        else:
            for i in range(8 + N_QUADRATURE):
                slack[i] = snap_slack[i]
            # Airborne, whichever regime the snap was reached from, which is why the
            # regime is assigned here rather than left to carry over. Reaching this from
            # _SLACK_GROUND means the sling picked a grounded stone up and did not keep
            # hold of it: the snap left the stone rising (the branch above already sent
            # the downward case to the ground's own impulse), so it is off the ground even
            # though the sling is carrying nothing. Carrying _SLACK_GROUND over instead
            # glued it there - the grounded dynamics pin dy/dt at zero, so the stone kept
            # the upward velocity the snap had just given it and never moved - while
            # physics._launch_transition returned SLACK and flew it. Not a corner: that
            # branch is taken 245 times in 500 random draws, and it was the largest single
            # source of the two engines disagreeing on a launch that touched the ground.
            regime = _SLACK

    # No release: hand back the machine state reached, with the projectile wherever the
    # last regime left it.
    if regime == _TAUT:
        sin_t, cos_t = np.sin(theta), np.cos(theta)
        sin_a, cos_a = np.sin(alpha), np.cos(alpha)
        return (False, t, theta, theta_dot, psi,
                l_a * cos_t + l_s * cos_a, l_a * sin_t + l_s * sin_a + h_T,
                -l_a * theta_dot * sin_t - l_s * alpha_dot * sin_a,
                l_a * theta_dot * cos_t + l_s * alpha_dot * cos_a,
                string_impulse, cw_impulse, sling_deficit, snap_energy, ground_energy,
                arm_ground, start_py)
    return (False, t, slack[0], slack[1], slack[6], slack[2], slack[3], slack[4], slack[5],
            string_impulse, cw_impulse, sling_deficit, snap_energy, ground_energy,
            arm_ground, start_py)


@njit(cache=True, fastmath=True, inline="always")
def _ballistic_dynamics(vx, vy, drag_scale, mass):
    speed = np.sqrt(vx * vx + vy * vy)
    drag_accel = -drag_scale * speed / mass if speed > 1e-12 else 0.0
    return vx, vy, drag_accel * vx, -G + drag_accel * vy


@njit(cache=True, fastmath=True)
def _integrate_ballistic(x0, y0, vx0, vy0, mass, drag_coefficient, area, t_max, rtol, atol,
                         ground_y):
    """Integrate ballistic flight (with quadratic drag) until ground impact; returns impact_x.

    Only the velocity components are carried through the intermediate RK stages: drag
    depends on speed alone, so the stage positions would be computed and then never read.
    The final position still comes from the full fifth-order combination below, and the
    error norm still spans all four components, so the step path is unchanged - numba's
    LLVM was already dropping the dead stores, and removing them measures the same.
    """
    if y0 <= ground_y:
        return x0

    drag_scale = 0.5 * RHO_AIR * drag_coefficient * area

    t = 0.0
    x, y, vx, vy = x0, y0, vx0, vy0
    f0_1, f0_2, f0_3, f0_4 = _ballistic_dynamics(vx, vy, drag_scale, mass)

    h = 1e-2
    # A resting sphere's centre sits at its own radius, which is the line the launch phase
    # holds a grounded projectile on and so the line the flight has to end at too.
    g_prev = y - ground_y

    for _ in range(MAX_STEPS):
        if t >= t_max:
            return x
        if t + h > t_max:
            h = t_max - t

        y2_3 = vx + h * A21 * f0_3
        y2_4 = vy + h * A21 * f0_4
        k2_1, k2_2, k2_3, k2_4 = _ballistic_dynamics(y2_3, y2_4, drag_scale, mass)

        y3_3 = vx + h * (A31 * f0_3 + A32 * k2_3)
        y3_4 = vy + h * (A31 * f0_4 + A32 * k2_4)
        k3_1, k3_2, k3_3, k3_4 = _ballistic_dynamics(y3_3, y3_4, drag_scale, mass)

        y4_3 = vx + h * (A41 * f0_3 + A42 * k2_3 + A43 * k3_3)
        y4_4 = vy + h * (A41 * f0_4 + A42 * k2_4 + A43 * k3_4)
        k4_1, k4_2, k4_3, k4_4 = _ballistic_dynamics(y4_3, y4_4, drag_scale, mass)

        y5_3 = vx + h * (A51 * f0_3 + A52 * k2_3 + A53 * k3_3 + A54 * k4_3)
        y5_4 = vy + h * (A51 * f0_4 + A52 * k2_4 + A53 * k3_4 + A54 * k4_4)
        k5_1, k5_2, k5_3, k5_4 = _ballistic_dynamics(y5_3, y5_4, drag_scale, mass)

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
            g_new = yn_2 - ground_y
            if g_prev > 0.0 and g_new <= 0.0:
                lo, hi = 0.0, 1.0
                for _ in range(50):
                    mid = 0.5 * (lo + hi)
                    g_mid = _hermite(y, yn_2, f0_2, k7_2, h, mid) - ground_y
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
                        joint_friction_coefficient, bearing_friction_coefficient,
                        pivot_shaft_radius, has_pulley):
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
    # otherwise the machine's own linkage - one wrap of the pulley, or the traditional
    # machine's short arm, which has no pulley radius to wrap. A non-positive value is the
    # "not set" sentinel, since numba has no None to pass down.
    if counter_weight_rope_length > 0.0:
        rope_length = counter_weight_rope_length
    elif has_pulley:
        rope_length = 2.0 * pulley_radius
    else:
        rope_length = length_counterweight

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

    # Drag torque along the beam: 1/8 rho Cd w L^4 per side (see
    # physics.TrebuchetSimulator.__init__ for why the moment arm belongs in there).
    arm_drag_k = (1.0 / 8.0) * ARM_CROSS_SECTION_WIDTH * arm_drag_coefficient * RHO_AIR * (
        arm_length**4 + arm_back_length**4
    )
    proj_drag_k = 0.5 * RHO_AIR * projectile_drag_coefficient * projectile_area
    arm_gravity_k = arm_cm_offset * G * arm_mass
    proj_gravity_theta_k = projectile_mass * G * arm_length
    proj_gravity_alpha_k = projectile_mass * string_length * G

    # physics.TrebuchetSimulator._M_taut: the machine's inertia about theta carrying no
    # projectile, which is what the arm swings on once the sling has let go.
    M_taut = M11 - projectile_mass * arm_length**2

    # TrebuchetParams.pivot_friction_torque: mu * r_shaft * (static bearing load), the
    # load being everything the pivot carries on either linkage.
    total_mass = counter_weight_mass + pulley_mass + arm_mass + projectile_mass
    pivot_friction = bearing_friction_coefficient * pivot_shaft_radius * total_mass * G

    c = (
        arm_length, string_length, M11, M22, M33, coupling, cw_swing_coupling,
        arm_drag_k, proj_drag_k, cw_torque_const, cw_torque_cos, cw_swing_gravity_k,
        arm_gravity_k, proj_gravity_theta_k, proj_gravity_alpha_k, joint_friction_coefficient,
        M_taut, arm_back_length, l_w, moi_arm + moi_pulley, cw_lever,
        pivot_friction, projectile_radius,
        projectile_mass, counter_weight_mass, pulley_radius,
        1.0 if has_pulley else 0.0,
        SLING_TENSION_FLOOR * projectile_mass * G,
    )
    return c, (arm_mass, pulley_mass, projectile_area, arm_cm_offset, l_w)


@njit(cache=True, fastmath=True)
def simulate_fast(counter_weight_mass, pulley_radius, length_counterweight,
                   counter_weight_rope_length, arm_length, string_length, release_angle,
                   pivot_height, pulley_density, arm_density, projectile_mass, projectile_radius,
                   initial_arm_angle, arm_drag_coefficient, projectile_drag_coefficient,
                   joint_friction_coefficient, bearing_friction_coefficient, pivot_shaft_radius,
                   has_pulley):
    """Scalar port of simulate_trebuchet's rtol=1e-6/dense_output=False path.

    Returns (released, distance, efficiency, string_impulse, cw_impulse, sling_deficit,
    snap_energy, ground_energy). `cw_impulse` is the counterweight rope's compression
    impulse (N*s, see
    physics._tension_metrics), which the objective's slack penalty charges; `sling_deficit`
    is the dimensionless share of the launch the sling spent under-loaded, weighted by how
    far under, which its snap penalty charges (physics.py reports the same number as
    `sling_tension_deficit`); `snap_energy` is the kinetic energy destroyed by re-tension
    snaps (`sling_snap_energy` there) and `ground_energy` what the ground took from the
    projectile on the way in (`projectile_ground_energy`). `string_impulse` is a self-check
    rather than a
    penalty input: the sling is a rope in this engine too, so a taut stretch ends at the
    tension zero-crossing and this should come back at the event solver's own error.
    (False, 0.0, 0.0, ...) if release never occurs or the geometry/result is invalid
    (mirrors physics.py's degenerate cases).

    `has_pulley` selects the linkage; the machine's own linkage parameter is read and the
    other one ignored, exactly as TrebuchetParams does. A non-positive
    counter_weight_rope_length means "unset" (numba has no None), so it falls back to one
    wrap of the pulley.
    """
    c, extras = _machine_constants(
        counter_weight_mass, pulley_radius, length_counterweight, counter_weight_rope_length,
        arm_length, string_length, pulley_density, arm_density, projectile_mass, projectile_radius,
        arm_drag_coefficient, projectile_drag_coefficient, joint_friction_coefficient,
        bearing_friction_coefficient, pivot_shaft_radius, has_pulley,
    )
    arm_mass, _pulley_mass, projectile_area, arm_cm_offset, l_w = extras

    # Cocked pose, mirroring physics.TrebuchetSimulator.initial_state.
    # config.AUTO_INITIAL_ARM_ANGLE is the "not pinned by the caller" sentinel (numba has
    # no None, and fastmath rules out NaN), and the angle is then resolved the way
    # config.resolve_initial_arm_angle resolves it: a constant on the pulley machine, and
    # on the traditional one whatever walks the long arm down to the ground.
    theta0 = initial_arm_angle
    if initial_arm_angle >= _AUTO_ANGLE:
        if has_pulley:
            theta0 = _PULLEY_COCKED_ANGLE
        else:
            ratio = (_START_CLEARANCE - pivot_height) / arm_length
            if ratio > 1.0:
                ratio = 1.0
            elif ratio < -1.0:
                ratio = -1.0
            theta0 = -np.pi - np.arcsin(ratio)
    psi_rest = -np.pi / 2.0
    if has_pulley:
        # Sling tucked alongside the arm, angled just far enough off it to clear.
        arcsin_arg = projectile_radius / string_length
        if arcsin_arg > 1.0 or arcsin_arg < -1.0:
            return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        alpha0 = theta0 + np.pi - np.arcsin(arcsin_arg)
        psi0 = 0.0
    else:
        # Sling and counterweight link both hang straight down from their attachments.
        alpha0 = psi_rest
        psi0 = psi_rest

    # Absolute floor in newtons; the deficit is normalized by it below, so the metric
    # means the same thing on a 0.15 kg stone as on a 20 kg one.
    tension_floor = SLING_TENSION_FLOOR * projectile_mass * G

    (released, t_rel, theta_r, theta_dot_r, psi_r, x0, y0, vx0, vy0,
     string_impulse, cw_impulse, sling_deficit, snap_energy, ground_energy, _arm_ground,
     start_py) = _integrate_launch(
        theta0, alpha0, psi0, c, release_angle, 10.0, 1e-6, 1e-6,
        projectile_mass, counter_weight_mass, pulley_radius, has_pulley, tension_floor,
        pivot_height,
    )
    # Same guard physics._tension_metrics uses: no clock or no projectile weight means
    # there is no share of the launch to report.
    if t_rel > 0.0 and tension_floor > 0.0:
        sling_deficit = sling_deficit / (tension_floor * t_rel)
    else:
        sling_deficit = 0.0
    if not released:
        return (False, 0.0, 0.0, string_impulse, cw_impulse, sling_deficit, snap_energy,
                ground_energy)

    if np.isnan(x0) or np.isnan(y0) or np.isnan(vx0) or np.isnan(vy0):
        return (False, 0.0, 0.0, string_impulse, cw_impulse, sling_deficit, snap_energy,
                ground_energy)

    proj_speed2 = vx0 * vx0 + vy0 * vy0
    proj_KE = 0.5 * projectile_mass * proj_speed2

    distance = 0.0
    if y0 >= 0.0:
        distance = _integrate_ballistic(
            x0, y0, vx0, vy0, projectile_mass, projectile_drag_coefficient, projectile_area,
            60.0, 1e-6, 1e-6, projectile_radius,
        )
        distance = max(0.0, distance)

    if has_pulley:
        # The weight descends r_pul per radian, so the drop is exactly linear.
        height_dropped = pulley_radius * (theta0 - theta_r)
    else:
        # The pinned weight follows the pin around the pivot and swings on top of that,
        # so measure its height at both ends of the launch (physics._release_result does
        # the same through weight_position_velocity).
        start_y = pivot_height - length_counterweight * np.sin(theta0) + l_w * np.sin(psi_rest)
        end_y = pivot_height - length_counterweight * np.sin(theta_r) + l_w * np.sin(psi_r)
        height_dropped = start_y - end_y
    counterweight_PE_spent = counter_weight_mass * G * height_dropped

    arm_height_change = (np.sin(theta0) - np.sin(theta_r)) * arm_cm_offset
    arm_PE_spent = arm_height_change * arm_mass * G

    # Where the projectile actually started, which is not the hanging pose whenever the
    # sling could reach the ground and the machine was loaded on it (see
    # physics.ground_start_state).
    projectile_PE_spent = (start_py - y0) * projectile_mass * G

    total_PE_spent = counterweight_PE_spent + arm_PE_spent + projectile_PE_spent
    efficiency = proj_KE / total_PE_spent if total_PE_spent > 0.0 else 0.0
    efficiency = max(0.0, efficiency)

    return (True, distance, efficiency, string_impulse, cw_impulse, sling_deficit,
            snap_energy, ground_energy)


# Deliberately not cache=True, and neither is evaluate_population below. Both call
# simulate_fast, and a cached _score reloaded from disk does not link against the
# simulate_fast in this process: it answers from a stale copy, and the very first thing it
# does with the answer is branch on it. Reloaded from cache it scored a design that throws
# nothing - the reference and a freshly compiled simulate_fast both put it at 0.0 m - as
# -144.9 instead of INVALID_COST, and differential evolution duly returned it as the winner
# of its seed. The first run after clearing the cache was right and every run afterwards
# was wrong, which is the tell: it is the reload, not the source. Compiling simulate_fast
# first in the same process also fixed it, which is the same tell from the other side.
#
# So the objective is the one part of the chain that is compiled in-process every time.
# That is a few seconds once per run against an optimizer that then evaluates hundreds of
# thousands of designs, and it is the difference between a search that scores what it
# thinks it is scoring and one that does not. Everything below simulate_fast keeps its
# cache - the fault is in reloading a *caller* of a cached function, not in the cache
# itself.
@njit(fastmath=True)
def _score(counter_weight_mass, pulley_radius, length_counterweight, counter_weight_rope_length,
           arm_length, string_length, release_angle,
           pivot_height, pulley_density, arm_density, projectile_mass, projectile_radius,
           initial_arm_angle, arm_drag_coefficient, projectile_drag_coefficient,
           joint_friction_coefficient, bearing_friction_coefficient, pivot_shaft_radius,
           has_pulley,
           target_distance, efficiency_weight, distance_weight, mass_weight,
           slack_penalty_weight, snap_penalty_weight, jerk_penalty_weight):
    """Scalar port of optimization._objective's cost formula for one individual."""
    if string_length > 0.95 * arm_length:
        return INVALID_COST

    (released, distance, efficiency, _string_impulse, cw_impulse, sling_deficit,
     snap_energy, ground_energy) = simulate_fast(
        counter_weight_mass, pulley_radius, length_counterweight, counter_weight_rope_length,
        arm_length, string_length, release_angle,
        pivot_height, pulley_density, arm_density, projectile_mass, projectile_radius,
        initial_arm_angle, arm_drag_coefficient, projectile_drag_coefficient,
        joint_friction_coefficient, bearing_friction_coefficient, pivot_shaft_radius,
        has_pulley,
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
    # Only the counterweight rope is charged a compression impulse, exactly as in
    # optimization._objective. The sling used to be charged one here too, because this
    # engine held it rigid and the impulse was the only sign it had gone somewhere the
    # model could not follow; it is a rope in both engines now, so a taut stretch simply
    # ends at the zero crossing and there is no sling compression left to bill.
    slack_cost = slack_penalty_weight * cw_impulse
    snap_cost = snap_penalty_weight * sling_deficit
    # The joules the launch destroyed in its discontinuities - a sling that let go and
    # came back, and a stone that reached the ground - which this engine has always
    # measured and, until the objective had a term for them, discarded.
    jerk_cost = jerk_penalty_weight * (snap_energy + ground_energy)

    return (
        efficiency_weight * efficiency_cost + distance_weight * distance_cost
        + mass_weight * mass_cost + slack_cost + snap_cost + jerk_cost
    )


@njit(fastmath=True, parallel=True)  # not cached - see _score above
def evaluate_population(counter_weight_mass, pulley_radius, length_counterweight,
                         arm_length, string_length, release_angle,
                         counter_weight_rope_length,
                         pivot_height, pulley_density, arm_density, projectile_mass, projectile_radius,
                         initial_arm_angle, arm_drag_coefficient, projectile_drag_coefficient,
                         joint_friction_coefficient, bearing_friction_coefficient,
                         pivot_shaft_radius, has_pulley, target_distance, efficiency_weight,
                         distance_weight, mass_weight, slack_penalty_weight, snap_penalty_weight,
                         jerk_penalty_weight):
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
            joint_friction_coefficient, bearing_friction_coefficient, pivot_shaft_radius,
            has_pulley,
            target_distance, efficiency_weight, distance_weight, mass_weight,
            slack_penalty_weight, snap_penalty_weight, jerk_penalty_weight,
        )
    return costs
