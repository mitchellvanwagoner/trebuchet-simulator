"""Trebuchet physics simulation engine: Euler-Lagrange dynamics and ODE integration."""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from trebuchet_sim.config import ARM_CROSS_SECTION_WIDTH, G, RHO_AIR, TrebuchetParams
from trebuchet_sim.trajectory import BallisticTrajectory, integrate_ballistic_trajectory

# Launch state, taut sling: theta (arm), alpha (sling), psi (counterweight swing).
# psi is inert on the pulley machine, where the weight hangs from a rope over the
# axle instead of a pin on the arm - see TrebuchetSimulator.__init__.
State = Tuple[float, float, float, float, float, float]
PositionVelocity = Tuple[Tuple[float, float], Tuple[float, float]]

# Evenly spaced samples taken from the dense ODE solution for energy tracking.
ENERGY_SAMPLES = 400

# Cap on taut/slack regime switches during the launch phase. Each sling snap
# destroys energy, so the switching always dies out; the cap only guards against
# numerical chatter right at a regime boundary.
MAX_LAUNCH_SEGMENTS = 200


@dataclass
class AftermathSegment:
    """One constant-regime stretch of the post-release machine dynamics."""

    sol: object  # scipy solve_ivp dense-output solution, state = [theta, theta_dot], local clock (0 = segment start)
    t0: float    # segment start, on the aftermath's own clock (0 = release)
    t1: float    # segment end, same clock
    regime: str  # "taut" (rope loaded, counterweight airborne) or "slack" (counterweight resting on the ground)


@dataclass
class AftermathResult:
    """Post-release single-pendulum dynamics of the arm/pulley/counterweight.

    The sling and projectile are gone the instant the projectile releases, so the
    machine collapses to a single DOF (theta). It's integrated independently of the
    ballistic flight, on its own clock starting at 0 = release, for a duration the
    caller supplies (typically the flight time, so the two can be stitched together
    for animation - see sample_full_timeline).
    """

    segments: List[AftermathSegment]
    touchdown_times: List[float]   # counterweight-hits-ground events, aftermath-clock seconds
    retension_times: List[float]   # rope-snaps-taut-again events, aftermath-clock seconds

    def _raw_state_at(self, t: float):
        """(state vector, regime) at aftermath-clock time t; clamped to the ends."""
        if not self.segments:
            raise ValueError("AftermathResult has no segments")
        if t <= self.segments[0].t0:
            return self.segments[0].sol.y[:, 0], self.segments[0].regime
        for seg in self.segments:
            if t <= seg.t1:
                return seg.sol.sol(t - seg.t0), seg.regime
        last = self.segments[-1]
        return last.sol.y[:, -1], last.regime

    def state_at(self, t: float) -> Tuple[float, float, str]:
        """(theta, theta_dot, regime) at aftermath-clock time t; clamped to the timeline's ends."""
        y, regime = self._raw_state_at(t)
        return float(y[0]), float(y[1]), regime

    def swing_at(self, t: float) -> Tuple[float, float]:
        """(psi, psi_dot) - the counterweight's swing about its pin at time t.

        Inert on the pulley machine, whose weight has no freedom of its own.
        """
        y, _regime = self._raw_state_at(t)
        return float(y[2]), float(y[3])


@dataclass
class LaunchSegment:
    """One constant-regime stretch of the launch dynamics, on the absolute launch clock.

    State layout depends on the regime:
      "taut":  y = [theta, theta_dot, alpha, alpha_dot]        (sling rigid, projectile on the circle)
      "slack": y = [theta, theta_dot, px, py, pvx, pvy]        (projectile in free flight with drag)
    """

    sol: object  # scipy solve_ivp dense-output solution over (t0, t1), absolute time
    t0: float
    t1: float
    regime: str  # "taut" or "slack"


class LaunchSolution:
    """Launch-phase solution stitched across taut/slack sling regimes.

    The sling is a rope: it can pull but never push. Whenever the taut (rigid-sling)
    dynamics would need negative string tension, the projectile detaches and flies
    free while the arm/pulley/counterweight continue as a single-DOF machine; when
    the tip-to-projectile distance grows back to the string length, an inelastic
    snap impulse restores the constraint, destroying the kinetic energy of the
    radial separation (recorded in `snap_energy_losses` - the "jerk" loss a rigid
    model never sees). The snap conserves momentum and only ever removes energy, so
    the string can't act as a spring.

    Replaces the raw solve_ivp object as `SimulationResult.solution`: consumers use
    the regime-aware accessors below instead of indexing a state vector whose
    meaning depends on the regime.
    """

    def __init__(self, simulator: "TrebuchetSimulator"):
        self._sim = simulator
        self.segments: List[LaunchSegment] = []
        self.release_occurred: bool = False
        self.t_release: Optional[float] = None
        self.release_machine_state: Optional[Tuple[float, float]] = None  # theta, theta_dot
        self.release_projectile_state: Optional[PositionVelocity] = None
        self.release_swing_state: Optional[Tuple[float, float]] = None  # psi, psi_dot
        self.snap_times: List[float] = []
        self.snap_energy_losses: List[float] = []

    @property
    def t_end(self) -> float:
        return self.segments[-1].t1

    @property
    def slack_time(self) -> float:
        return sum(seg.t1 - seg.t0 for seg in self.segments if seg.regime == "slack")

    def _segment_at(self, t: float) -> LaunchSegment:
        for seg in self.segments:
            if t <= seg.t1:
                return seg
        return self.segments[-1]

    def _y_at(self, t: float):
        seg = self._segment_at(t)
        if seg.sol.sol is None:
            # No dense interpolants (dense_output=False callers, e.g. the optimizer
            # objective): snap to the nearest accepted step - those callers only ever
            # read segment-boundary states.
            idx = int(np.argmin(np.abs(seg.sol.t - t)))
            return seg, seg.sol.y[:, idx]
        t_clamped = min(max(t, seg.t0), seg.t1)
        return seg, seg.sol.sol(t_clamped)

    def machine_state(self, t: float) -> Tuple[float, float]:
        """(theta, theta_dot) at time t; both regimes carry them as the first two states."""
        _seg, y = self._y_at(t)
        return float(y[0]), float(y[1])

    def projectile_state(self, t: float) -> PositionVelocity:
        """Projectile (position, velocity) at time t, regardless of sling regime."""
        seg, y = self._y_at(t)
        if seg.regime == "taut":
            return self._sim.projectile_position_velocity(y)
        return (float(y[2]), float(y[3])), (float(y[4]), float(y[5]))

    def swing_state(self, t: float) -> Tuple[float, float]:
        """(psi, psi_dot) - the counterweight's swing about its pin at time t.

        psi sits at the end of both state layouts, so its index depends on the
        regime: 4 while the sling is taut, 6 once the projectile has been cut
        loose and the state grew four projectile components.
        """
        seg, y = self._y_at(t)
        base = 4 if seg.regime == "taut" else 6
        return float(y[base]), float(y[base + 1])


@dataclass
class SimulationResult:
    """Complete simulation results."""

    distance: float
    efficiency: float
    metrics: Dict
    solution: object  # LaunchSolution (stitched taut/slack launch dynamics)
    energy_history: Optional[List[Dict]] = None
    trajectory: Optional[BallisticTrajectory] = None  # post-release flight, when a release occurred
    aftermath: Optional[AftermathResult] = None        # post-release machine dynamics, when requested


def sample_component_positions(params: TrebuchetParams, sol: LaunchSolution, times) -> Dict[str, List[Tuple[float, float]]]:
    """Sample projectile, arm-tip, and counterweight positions from a solved launch at each time.

    Shared by the matplotlib and web-3D animations so both render from identical state.
    The projectile comes from the regime-aware accessor, so slack phases (projectile
    detached, inside the string circle) render exactly where the physics puts it.
    """
    simulator = TrebuchetSimulator(params)
    positions = {"projectile": [], "arm_tip": [], "counterweight": [], "cw_pin": []}
    for t in times:
        theta, theta_dot = sol.machine_state(float(t))
        psi, psi_dot = sol.swing_state(float(t))
        machine_state = (theta, theta_dot, 0.0, 0.0, psi, psi_dot)
        positions["counterweight"].append(simulator.weight_position_velocity(machine_state)[0])
        positions["cw_pin"].append(simulator.counterweight_pin_position(theta))
        positions["arm_tip"].append(simulator.arm_tip_position_velocity(machine_state)[0])
        positions["projectile"].append(sol.projectile_state(float(t))[0])
    return positions


def sample_full_timeline(
    params: TrebuchetParams, result: SimulationResult, times
) -> Dict[str, List[Tuple[float, float]]]:
    """Sample projectile, arm-tip, and counterweight positions across the full timeline:
    launch dynamics before release, then the independently-integrated aftermath (arm/
    pulley/counterweight) stitched with the ballistic trajectory (projectile) after.

    `times` are absolute times from t=0 (launch start). The timeline naturally ends when
    the projectile lands (t_release + trajectory.flight_time) - both the aftermath and the
    trajectory were run for exactly that duration. Requires `result.aftermath` (i.e. the
    result came from `simulate_trebuchet(..., simulate_aftermath=True)`); without it, the
    post-release machine holds its release pose as a fallback.

    Shared by the matplotlib and web-3D animations so both render from identical state.
    """
    simulator = TrebuchetSimulator(params)
    sol = result.solution
    release_occurred = bool(result.metrics.get("release_occurred", False))
    t_release = result.metrics.get("t_release") if release_occurred else None

    positions = {"projectile": [], "arm_tip": [], "counterweight": [], "cw_pin": []}

    for t in times:
        if not release_occurred or t <= t_release:
            t_clamped = float(min(t, sol.t_end))
            theta, theta_dot = sol.machine_state(t_clamped)
            psi, psi_dot = sol.swing_state(t_clamped)
            machine_state = (theta, theta_dot, 0.0, 0.0, psi, psi_dot)
            positions["counterweight"].append(simulator.weight_position_velocity(machine_state)[0])
            positions["cw_pin"].append(simulator.counterweight_pin_position(theta))
            positions["arm_tip"].append(simulator.arm_tip_position_velocity(machine_state)[0])
            positions["projectile"].append(sol.projectile_state(t_clamped)[0])
            continue

        t_local = float(t) - t_release

        if result.aftermath is not None:
            theta, theta_dot, regime = result.aftermath.state_at(t_local)
            psi, psi_dot = result.aftermath.swing_at(t_local)
            machine_state = (theta, theta_dot, 0.0, 0.0, psi, psi_dot)
            arm_tip = simulator.arm_tip_position_velocity(machine_state)[0]
            cw_pos = (
                (params.pulley_radius, params.counter_weight_size / 2)  # box resting on the ground, bottom at y=0
                if regime == "slack"
                else simulator.weight_position_velocity(machine_state)[0]
            )
        else:
            # No aftermath computed: hold the release pose.
            theta_r, theta_dot_r = sol.release_machine_state
            psi_r, psi_dot_r = sol.release_swing_state or (0.0, 0.0)
            theta = theta_r
            release_state = (theta_r, theta_dot_r, 0.0, 0.0, psi_r, psi_dot_r)
            arm_tip = simulator.arm_tip_position_velocity(release_state)[0]
            cw_pos = simulator.weight_position_velocity(release_state)[0]

        positions["arm_tip"].append(arm_tip)
        positions["counterweight"].append(cw_pos)
        positions["cw_pin"].append(simulator.counterweight_pin_position(theta))
        positions["projectile"].append(
            result.trajectory.position_at(t_local) if result.trajectory is not None else arm_tip
        )

    return positions


class TrebuchetSimulator:
    """Euler-Lagrange trebuchet physics simulator."""

    def __init__(self, params: TrebuchetParams, track_energy: bool = False):
        self.params = params
        self.track_energy = track_energy
        self.energy_history: List[Dict] = []

        # Constant-fold everything the dynamics RHS needs: it runs a few hundred
        # times per simulation, and the dataclass properties (moi_arm, projectile_area,
        # ...) would otherwise be recomputed on every call. Only M12 of the inertia
        # matrix depends on the state; M11 and M22 are constants of the machine.
        p = params
        m_p, l_a, l_s = p.projectile_mass, p.arm_length, p.string_length
        self._l_a, self._l_s = l_a, l_s
        self._m_p = m_p
        self._h_T = p.pivot_height
        # The counterweight's contribution is the only thing that differs between
        # the two machines (see config.MachineType), and it enters in exactly two
        # places: an inertia about the pivot, and a gravity torque.
        self._cw_inertia = p.counter_weight_mass * p.counter_weight_lever**2
        self._M11 = self._cw_inertia + p.moi_pulley + p.moi_arm + m_p * l_a**2
        self._M22 = m_p * l_s**2
        self._coupling = m_p * l_a * l_s  # projectile inertial coupling between theta and alpha
        self._arm_drag_k = 1 / 6 * ARM_CROSS_SECTION_WIDTH * p.arm_drag_coefficient * RHO_AIR * l_a**3
        self._proj_drag_k = 0.5 * RHO_AIR * p.projectile_drag_coefficient * p.projectile_area
        # Counterweight gravity torque, written once for both linkages as
        #     Q_cw(theta) = cw_torque_const + cw_torque_cos * cos(theta)
        # Pulley: the weight descends r_pul metres per radian of rotation whatever
        # the arm angle, so its torque is the constant -m*g*r_pul. Traditional: the
        # weight rides the arm's far end at height h_T - l_cw*sin(theta), giving
        # -dU/dtheta = +m*g*l_cw*cos(theta). Keeping both in one expression means
        # the hot ODE path needs no branch.
        # On the traditional machine the weight hangs from a pin on the arm and can
        # swing relative to it, so it carries its own coordinate psi. Those terms are
        # all zero on the pulley machine, where the rope-over-axle linkage leaves the
        # weight no freedom of its own; M33 = 1 there keeps the 3x3 solve below
        # non-singular while making psi inert (see the reduction note there).
        m_cw = p.counter_weight_mass
        self._l_w = 0.0 if p.has_pulley else p.initial_cw_rope_length
        self._psi_rest = -math.pi / 2  # link hanging straight down
        if p.has_pulley:
            # Factor order matches the original m*r*G exactly: reassociating a
            # product can shift the last bit, and that is enough for the ODE to
            # diverge in the 6th digit over a launch.
            self._cw_torque_const = -(m_cw * p.pulley_radius * G)
            self._cw_torque_cos = 0.0
            self._cw_swing_coupling = 0.0
            self._cw_swing_gravity_k = 0.0
            self._M33 = 1.0
        else:
            self._cw_torque_const = 0.0
            self._cw_torque_cos = m_cw * G * p.length_counterweight
            self._cw_swing_coupling = m_cw * p.length_counterweight * self._l_w
            self._cw_swing_gravity_k = m_cw * G * self._l_w
            self._M33 = m_cw * self._l_w**2
        self._arm_gravity_k = p.arm_cm_offset * G * p.arm_mass
        self._proj_gravity_theta_k = m_p * G * l_a
        self._proj_gravity_alpha_k = m_p * l_s * G
        self._joint_friction = p.joint_friction_coefficient

        # Post-release aftermath: sling and projectile are gone, so the machine collapses
        # to a single DOF (theta). "Taut" reuses the launch-phase counterweight/pulley/arm
        # inertia and gravity torque with the projectile terms dropped; "slack" additionally
        # drops the counterweight (it's resting on the ground). theta_ground solves
        # weight_position_velocity's w_y = counter_weight_size/2 for theta - i.e. the
        # counterweight's bottom face (not its center of mass) touches the ground.
        self._M_taut = self._cw_inertia + p.moi_pulley + p.moi_arm
        self._M_slack = p.moi_pulley + p.moi_arm
        self._cw_half_size = p.counter_weight_size / 2
        # Arm angle at which the counterweight's bottom face reaches the ground.
        # Only the pulley machine needs it: there the weight descends linearly with
        # theta and the rope goes slack on touchdown. The traditional machine's
        # weight is pinned to the arm, so ground contact is a hard stop instead
        # (see _simulate_aftermath_pinned).
        self._theta_ground = (
            p.initial_arm_angle
            + (self._cw_half_size - p.weight_height + p.initial_cw_rope_length) / p.pulley_radius
            if p.has_pulley
            else -math.inf
        )

    def weight_position_velocity(self, y: State) -> PositionVelocity:
        """Counterweight position and velocity from the state vector."""
        theta, theta_dot = y[0], y[1]
        p = self.params

        if p.has_pulley:
            r_pul, h_w, theta_i = p.pulley_radius, p.weight_height, p.initial_arm_angle
            rope_length = p.initial_cw_rope_length
            # The weight hangs `rope_length` below the axle at t=0 (theta=theta_i)
            # and descends by r_pul for every radian the pulley/arm rotates.
            w_y = (h_w - rope_length) + r_pul * (theta - theta_i)
            return (r_pul, w_y), (0.0, r_pul * theta_dot)

        # Traditional: the weight hangs on a pin carried by the arm's short end,
        # so its motion is the pin's plus the link's own swing about it.
        psi, psi_dot = (y[4], y[5]) if len(y) > 5 else (self._psi_rest, 0.0)
        l_cw, l_w, h_T = p.length_counterweight, self._l_w, p.pivot_height
        pin_x, pin_y = -l_cw * np.cos(theta), h_T - l_cw * np.sin(theta)
        pin_vx, pin_vy = l_cw * theta_dot * np.sin(theta), -l_cw * theta_dot * np.cos(theta)
        w_x = pin_x + l_w * np.cos(psi)
        w_y = pin_y + l_w * np.sin(psi)
        w_vx = pin_vx - l_w * psi_dot * np.sin(psi)
        w_vy = pin_vy + l_w * psi_dot * np.cos(psi)
        return (w_x, w_y), (w_vx, w_vy)

    def counterweight_pin_position(self, theta: float) -> Tuple[float, float]:
        """Where the counterweight's pin sits, for rendering the link."""
        p = self.params
        if p.has_pulley:
            return 0.0, p.pivot_height
        l_cw = p.length_counterweight
        return -l_cw * math.cos(theta), p.pivot_height - l_cw * math.sin(theta)

    def arm_position_velocity(self, y: State) -> PositionVelocity:
        """Arm center-of-mass position and velocity."""
        theta, theta_dot = y[0], y[1]
        # Offset, not half-length: on the traditional machine the beam extends
        # behind the pivot too, so its balance point is not at arm_length/2.
        h_T, r_cm = self.params.pivot_height, self.params.arm_cm_offset

        a_x = r_cm * np.cos(theta)
        a_y = r_cm * np.sin(theta) + h_T
        a_vx = -r_cm * theta_dot * np.sin(theta)
        a_vy = r_cm * theta_dot * np.cos(theta)

        return (a_x, a_y), (a_vx, a_vy)

    def arm_tip_position_velocity(self, y: State) -> PositionVelocity:
        """Arm tip position and velocity."""
        theta, theta_dot = y[0], y[1]
        h_T, l_a = self.params.pivot_height, self.params.arm_length

        a_x = l_a * np.cos(theta)
        a_y = l_a * np.sin(theta) + h_T
        a_vx = -l_a * theta_dot * np.sin(theta)
        a_vy = l_a * theta_dot * np.cos(theta)

        return (a_x, a_y), (a_vx, a_vy)

    def projectile_position_velocity(self, y: State) -> PositionVelocity:
        """Projectile position and velocity (arm tip plus string extension)."""
        theta, theta_dot, alpha, alpha_dot = y[0], y[1], y[2], y[3]
        l_a, l_s, h_T = self.params.arm_length, self.params.string_length, self.params.pivot_height

        pos_x = l_a * np.cos(theta) + l_s * np.cos(alpha)
        pos_y = l_a * np.sin(theta) + l_s * np.sin(alpha) + h_T
        vel_x = -l_a * theta_dot * np.sin(theta) - l_s * alpha_dot * np.sin(alpha)
        vel_y = l_a * theta_dot * np.cos(theta) + l_s * alpha_dot * np.cos(alpha)

        return (pos_x, pos_y), (vel_x, vel_y)

    def calculate_system_energy(self, y: State, t: float) -> Dict[str, float]:
        """Kinetic/potential energy breakdown at a taut launch state."""
        proj_pos, proj_vel = self.projectile_position_velocity(y)
        return self._system_energy(t, y[0], y[1], proj_pos, proj_vel, y[4], y[5])

    def launch_energy_at(self, launch: LaunchSolution, t: float) -> Dict[str, float]:
        """Energy breakdown at any point of a stitched launch (taut or slack).

        Across a snap the total drops discontinuously by the recorded snap loss -
        that dissipation is the point of the slack-sling model.
        """
        theta, theta_dot = launch.machine_state(t)
        proj_pos, proj_vel = launch.projectile_state(t)
        psi, psi_dot = launch.swing_state(t)
        return self._system_energy(t, theta, theta_dot, proj_pos, proj_vel, psi, psi_dot)

    def _system_energy(
        self, t: float, theta: float, theta_dot: float, proj_pos, proj_vel,
        psi: float = 0.0, psi_dot: float = 0.0,
    ) -> Dict[str, float]:
        machine_state = (theta, theta_dot, 0.0, 0.0, psi, psi_dot)
        arm_cm_pos, _ = self.arm_position_velocity(machine_state)
        cw_pos, cw_vel = self.weight_position_velocity(machine_state)

        proj_ke = 0.5 * self.params.projectile_mass * (proj_vel[0] ** 2 + proj_vel[1] ** 2)
        arm_ke = 0.5 * self.params.moi_arm * theta_dot**2
        pulley_ke = 0.5 * self.params.moi_pulley * theta_dot**2
        # Both components: the pulley machine's weight only moves vertically (vx is
        # exactly 0 there, so this is unchanged for it), but a pinned weight swings.
        cw_ke = 0.5 * self.params.counter_weight_mass * (cw_vel[0] ** 2 + cw_vel[1] ** 2)
        total_ke = proj_ke + arm_ke + pulley_ke + cw_ke

        proj_pe = self.params.projectile_mass * G * proj_pos[1]
        arm_pe = self.params.arm_mass * G * arm_cm_pos[1]
        cw_pe = self.params.counter_weight_mass * G * cw_pos[1]
        total_pe = proj_pe + arm_pe + cw_pe

        return {
            "time": t,
            "kinetic": total_ke,
            "potential": total_pe,
            "total": total_ke + total_pe,
            "proj_ke": proj_ke,
            "arm_ke": arm_ke,
            "cw_ke": cw_ke,
            "pulley_ke": pulley_ke,
            "proj_pe": proj_pe,
            "arm_pe": arm_pe,
            "cw_pe": cw_pe,
        }

    def _check_energy_conservation(self, max_percent_change: float = 5.0) -> Dict:
        """Flag time steps where total system energy jumps by more than a threshold."""
        violations, energy_increases, energy_decreases = [], [], []

        if len(self.energy_history) < 2:
            return {
                "violations": violations,
                "energy_increases": energy_increases,
                "energy_decreases": energy_decreases,
                "overall_trend": "insufficient_data",
            }

        total_energy_change = 0.0
        for prev, curr in zip(self.energy_history, self.energy_history[1:]):
            prev_energy, curr_energy = prev["total"], curr["total"]
            if abs(prev_energy) < 1e-12:
                continue

            energy_change = curr_energy - prev_energy
            percent_change = (energy_change / prev_energy) * 100
            abs_percent_change = abs(percent_change)
            total_energy_change += energy_change

            if abs_percent_change > max_percent_change:
                violation = {
                    "time": curr["time"],
                    "prev_energy": prev_energy,
                    "curr_energy": curr_energy,
                    "energy_change": energy_change,
                    "percent_change": percent_change,
                    "abs_percent_change": abs_percent_change,
                    "direction": "increase" if energy_change > 0 else "decrease",
                }
                violations.append(violation)
                (energy_increases if energy_change > 0 else energy_decreases).append(violation)

        if abs(total_energy_change) < 1e-6:
            overall_trend = "constant"
        else:
            overall_trend = "increasing" if total_energy_change > 0 else "decreasing"

        return {
            "violations": violations,
            "energy_increases": energy_increases,
            "energy_decreases": energy_decreases,
            "overall_trend": overall_trend,
            "total_energy_change": total_energy_change,
            "initial_energy": self.energy_history[0]["total"],
            "final_energy": self.energy_history[-1]["total"],
        }

    def trebuchet_dynamics(self, t: float, y: State) -> List[float]:
        """Euler-Lagrange equations of motion: state = [theta, theta_dot, alpha, alpha_dot].

        This is the ODE hot path (a few hundred calls per simulation, hundreds of
        thousands per optimizer run), so it uses `math` scalar functions and the
        machine constants folded in __init__ rather than numpy scalars and the
        dataclass properties.
        """
        theta, theta_dot, alpha, alpha_dot, psi, psi_dot = y
        l_a, l_s = self._l_a, self._l_s

        sin_t, cos_t = math.sin(theta), math.cos(theta)
        sin_a, cos_a = math.sin(alpha), math.cos(alpha)
        sin_p, cos_p = math.sin(psi), math.cos(psi)
        # Angle-difference identities: sin/cos(alpha - theta) and sin/cos(psi - theta)
        # from what we already have
        sin_at = sin_a * cos_t - cos_a * sin_t
        cos_at = cos_a * cos_t + sin_a * sin_t
        sin_pt = sin_p * cos_t - cos_p * sin_t
        cos_pt = cos_p * cos_t + sin_p * sin_t

        # Projectile velocity (arm tip plus string extension)
        p_vx = -l_a * theta_dot * sin_t - l_s * alpha_dot * sin_a
        p_vy = l_a * theta_dot * cos_t + l_s * alpha_dot * cos_a
        proj_speed = math.sqrt(p_vx * p_vx + p_vy * p_vy)

        # Inertia matrix: M11/M22/M33 are constants of the machine; the off-diagonal
        # couplings vary with the angle between the members. M23 is identically zero -
        # the sling and the counterweight link never touch each other's coordinates.
        M12 = self._coupling * cos_at
        M13 = -self._cw_swing_coupling * cos_pt

        # Arm aero drag torque always opposes the arm's angular velocity, whichever way it spins
        arm_drag_torque = -math.copysign(self._arm_drag_k * theta_dot * theta_dot, theta_dot)

        # Projectile drag force vector (opposes its velocity). The projectile position depends
        # on both coordinates, so the drag contributes a generalized force on each:
        #   Q_q = F . d(pos)/d(q), with d(pos)/d(theta) = (-l_a*sin(theta), l_a*cos(theta))
        #   and d(pos)/d(alpha) = (-l_s*sin(alpha), l_s*cos(alpha))
        drag_scale = self._proj_drag_k * proj_speed
        drag_fx, drag_fy = -drag_scale * p_vx, -drag_scale * p_vy
        Q_theta_drag = drag_fx * (-l_a * sin_t) + drag_fy * (l_a * cos_t)
        Q_alpha_drag = drag_fx * (-l_s * sin_a) + drag_fy * (l_s * cos_a)

        # Generalized forces on theta (arm angle)
        Q_theta = (
            self._coupling * sin_at * alpha_dot**2                       # projectile inertial coupling
            - self._cw_swing_coupling * sin_pt * psi_dot**2              # counterweight swing coupling
            + self._cw_torque_const + self._cw_torque_cos * cos_t        # counterweight gravity (via arm)
            - self._arm_gravity_k * cos_t                                # arm gravity
            - self._proj_gravity_theta_k * cos_t                         # projectile gravity (via arm)
            + arm_drag_torque                                            # arm drag
            + Q_theta_drag                                               # projectile drag (via arm)
            - self._joint_friction * theta_dot                           # joint friction
        )

        # Generalized forces on alpha (string angle)
        Q_alpha = (
            -self._coupling * sin_at * theta_dot**2                      # projectile inertial coupling
            - self._proj_gravity_alpha_k * cos_a                         # projectile gravity
            + Q_alpha_drag                                               # projectile drag
        )

        # Generalized force on psi (counterweight swing about its pin)
        Q_psi = (
            self._cw_swing_coupling * sin_pt * theta_dot**2              # arm inertial coupling
            - self._cw_swing_gravity_k * cos_p                           # weight gravity
        )

        # Solve M q_ddot = Q for the symmetric 3x3 with M23 = 0, via its adjugate.
        # On the pulley machine M13 = 0 and M33 = 1, so every expression below
        # collapses to the original two-coordinate solve exactly - multiplying by
        # 1.0 and subtracting 0.0 are exact in IEEE754, so each call returns the same
        # accelerations it did before the third coordinate existed.
        #
        # The integrated *trajectory* still shifts very slightly (~3e-7 relative on the
        # default machine). psi is inert there, but RK45 measures its error over the
        # whole state vector, so two always-zero components dilute the norm and change
        # the accepted step sizes - 73 steps where the 4-state system took 75. It is a
        # different path to the same solution, orders of magnitude below the model's
        # own accuracy, not a change in the physics.
        A, B, C, D, E = self._M11, M12, M13, self._M22, self._M33
        det = A * D * E - B * B * E - C * C * D
        if abs(det) < 1e-12:
            return [theta_dot, 0.0, alpha_dot, 0.0, psi_dot, 0.0]

        theta_ddot = (D * E * Q_theta - B * E * Q_alpha - C * D * Q_psi) / det
        alpha_ddot = (-B * E * Q_theta + (A * E - C * C) * Q_alpha + B * C * Q_psi) / det
        psi_ddot = (-C * D * Q_theta + B * C * Q_alpha + (A * D - B * B) * Q_psi) / det

        return [theta_dot, theta_ddot, alpha_dot, alpha_ddot, psi_dot, psi_ddot]

    def constraint_tensions(self, t: float, y: State) -> Tuple[float, float]:
        """(string tension, counterweight rope tension) at the current state, in newtons.

        The Lagrangian model treats both connectors as rigid links, which can push as
        well as pull. A real sling/rope can only pull: wherever a tension goes negative
        the real machine's rope would go slack, the projectile (or counterweight) would
        fly free, and the eventual re-tensioning snap would dissipate energy the rigid
        model never sees. Negative tension therefore marks the solution as unphysical
        from that moment on.

        Only theta_ddot is needed: the string tension comes from the projectile's
        radial (along-string) acceleration, and alpha_ddot only contributes
        tangentially; the counterweight's acceleration is r_pul * theta_ddot.
        """
        theta, theta_dot, alpha, alpha_dot = y[0], y[1], y[2], y[3]
        theta_ddot = self.trebuchet_dynamics(t, y)[1]
        l_a, l_s = self._l_a, self._l_s
        m_p = self.params.projectile_mass

        sin_t, cos_t = math.sin(theta), math.cos(theta)
        sin_a, cos_a = math.sin(alpha), math.cos(alpha)
        sin_at = sin_a * cos_t - cos_a * sin_t
        cos_at = cos_a * cos_t + sin_a * sin_t

        p_vx = -l_a * theta_dot * sin_t - l_s * alpha_dot * sin_a
        p_vy = l_a * theta_dot * cos_t + l_s * alpha_dot * cos_a
        drag_scale = self._proj_drag_k * math.sqrt(p_vx * p_vx + p_vy * p_vy)

        # Newton for the projectile projected on the string direction e_r = (cos a, sin a):
        # m_p * (a . e_r) = -T + (gravity + drag) . e_r
        radial_acc = l_a * theta_ddot * sin_at - l_a * theta_dot**2 * cos_at - l_s * alpha_dot**2
        string_tension = (
            -m_p * G * sin_a
            - drag_scale * (p_vx * cos_a + p_vy * sin_a)
            - m_p * radial_acc
        )

        # Pulley machine only: the weight moves vertically with a_y = r_pul *
        # theta_ddot and the rope pulls up, so the rope can be checked for going
        # slack. The traditional machine hangs its weight on a pinned link, a rigid
        # two-force member by construction - no rope, so no tension to report.
        if not self.params.has_pulley:
            return string_tension, math.nan
        cw_tension = self.params.counter_weight_mass * (G + self.params.pulley_radius * theta_ddot)

        return string_tension, cw_tension

    def _tension_metrics(self, launch: LaunchSolution) -> Dict:
        """Rope diagnostics sampled at the solver's accepted steps of each segment.

        The sling is handled physically (slack regime + snap losses), so its metrics
        report what actually happened: `sling_snap_energy` is the kinetic energy
        destroyed by re-tension snaps and `string_slack_fraction` the time share the
        projectile flew detached. The counterweight rope is still a rigid link, so it
        keeps the feasibility-style `cw_rope_compression_impulse` (integral of
        max(0, -T) dt, N*s): nonzero means the arm out-accelerated the falling
        counterweight and that part of the solution isn't physical.
        """
        string_T_min = math.inf
        cw_T_min = math.inf
        cw_impulse = 0.0
        m_cw, r_pul = self.params.counter_weight_mass, self.params.pulley_radius
        # Counterweight-rope diagnostics only mean something on the pulley machine
        # (see constraint_tensions); a pinned link cannot go slack.
        track_cw = self.params.has_pulley

        for seg in launch.segments:
            ts = seg.sol.t
            prev_deficit = None
            for i in range(len(ts)):
                t_i, y_i = float(ts[i]), seg.sol.y[:, i]
                if seg.regime == "taut":
                    string_T, cw_T = self.constraint_tensions(t_i, y_i)
                else:
                    string_T = 0.0  # slack rope carries nothing
                    theta_ddot = self._launch_slack_dynamics(t_i, y_i)[1]
                    cw_T = m_cw * (G + r_pul * theta_ddot) if track_cw else math.nan
                string_T_min = min(string_T_min, string_T)
                if not track_cw:
                    continue
                cw_T_min = min(cw_T_min, cw_T)
                deficit = max(0.0, -cw_T)
                if prev_deficit is not None:
                    cw_impulse += 0.5 * (prev_deficit + deficit) * (t_i - float(ts[i - 1]))
                prev_deficit = deficit

        duration = launch.t_end
        metrics = {
            "min_string_tension": float(string_T_min),
            "string_slack_fraction": float(launch.slack_time / duration) if duration > 0 else 0.0,
            "sling_snap_count": len(launch.snap_times),
            "sling_snap_energy": float(sum(launch.snap_energy_losses)),
        }
        if track_cw:
            metrics["min_cw_rope_tension"] = float(cw_T_min)
            metrics["cw_rope_compression_impulse"] = float(cw_impulse)
        return metrics

    def _machine_only_accelerations(self, theta, theta_dot, psi, psi_dot):
        """(theta_ddot, psi_ddot) for the machine carrying no projectile.

        Shared by the slack-sling launch regime and the post-release aftermath: both
        are the same body - arm, plus either a pulley-hung or a pinned counterweight -
        swinging on its own. Two coordinates on the traditional machine; on the pulley
        machine M13 = 0 and M33 = 1 reduce it to Q_theta / M_taut exactly.
        """
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        cos_p, sin_p = math.cos(psi), math.sin(psi)
        sin_pt = sin_p * cos_t - cos_p * sin_t
        cos_pt = cos_p * cos_t + sin_p * sin_t

        arm_drag_torque = -math.copysign(self._arm_drag_k * theta_dot * theta_dot, theta_dot)
        Q_theta = (
            -self._cw_swing_coupling * sin_pt * psi_dot**2
            + self._cw_torque_const + self._cw_torque_cos * cos_t
            - self._arm_gravity_k * cos_t
            + arm_drag_torque
            - self._joint_friction * theta_dot
        )
        Q_psi = self._cw_swing_coupling * sin_pt * theta_dot**2 - self._cw_swing_gravity_k * cos_p

        M13 = -self._cw_swing_coupling * cos_pt
        det = self._M_taut * self._M33 - M13 * M13
        if abs(det) < 1e-12:
            return 0.0, 0.0
        return (
            (self._M33 * Q_theta - M13 * Q_psi) / det,
            (self._M_taut * Q_psi - M13 * Q_theta) / det,
        )

    def _launch_slack_dynamics(self, t: float, y) -> List[float]:
        """Launch dynamics while the sling is slack.

        y = [theta, theta_dot, px, py, pvx, pvy, psi, psi_dot]. The machine runs as
        the same body as the taut aftermath (projectile terms gone); the projectile
        is in free flight with quadratic drag (same force law as trajectory.py).
        """
        theta, theta_dot, _px, _py, pvx, pvy, psi, psi_dot = y
        theta_ddot, psi_ddot = self._machine_only_accelerations(theta, theta_dot, psi, psi_dot)
        speed = math.hypot(pvx, pvy)
        drag_accel = -self._proj_drag_k * speed / self._m_p if speed > 1e-12 else 0.0
        return [
            theta_dot, theta_ddot, pvx, pvy,
            drag_accel * pvx, -G + drag_accel * pvy,
            psi_dot, psi_ddot,
        ]

    def _slack_state_from_taut(self, y_taut) -> List[float]:
        """Map a taut state to the slack state vector (projectile cut loose in place)."""
        pos, vel = self.projectile_position_velocity(y_taut)
        return [
            float(y_taut[0]), float(y_taut[1]),
            float(pos[0]), float(pos[1]), float(vel[0]), float(vel[1]),
            float(y_taut[4]), float(y_taut[5]),
        ]

    def _apply_snap(self, y_slack) -> Tuple[List[float], List[float], float]:
        """Inelastic re-tension snap at the moment the string comes taut again.

        An impulse P along the string (pulling projectile and arm tip toward each
        other) removes exactly the radial separation velocity - the momentum-conserving,
        energy-destroying "jerk". Returns (taut_state, slack_state, energy_lost): the
        same post-snap physics expressed in both state layouts, so the caller can pick
        a regime by checking the post-snap string tension. The energy lost is
        0.5 * P * g_dot, always >= 0: the snap can only ever dissipate.
        """
        theta, theta_dot, px, py, pvx, pvy, psi, psi_dot = (float(v) for v in y_slack)
        l_a, l_s, m_p = self._l_a, self._l_s, self._m_p

        sin_t, cos_t = math.sin(theta), math.cos(theta)
        tip_x, tip_y = l_a * cos_t, l_a * sin_t + self._h_T
        dx, dy = px - tip_x, py - tip_y
        dist = math.hypot(dx, dy)
        ex, ey = dx / dist, dy / dist              # unit vector tip -> projectile
        tvx, tvy = -l_a * sin_t, l_a * cos_t       # d(tip)/d(theta)

        g_dot = (pvx - theta_dot * tvx) * ex + (pvy - theta_dot * tvy) * ey  # radial separation speed
        t_dot_e = tvx * ex + tvy * ey

        # The sling pulls on theta only, but on the traditional machine theta is
        # inertially coupled to the counterweight's swing, so the arm resists the
        # impulse with an effective inertia M_taut - M13^2/M33 rather than M_taut,
        # and the weight picks up its share of the jerk. On the pulley machine
        # M13 = 0, leaving M_eff = M_taut and the weight untouched, exactly as before.
        cos_p, sin_p = math.cos(psi), math.sin(psi)
        cos_pt = cos_p * cos_t + sin_p * sin_t
        M13 = -self._cw_swing_coupling * cos_pt
        M_eff = self._M_taut - M13 * M13 / self._M33

        energy_lost = 0.0
        if g_dot > 0.0:
            P = g_dot / (1.0 / m_p + t_dot_e * t_dot_e / M_eff)
            theta_dot += P * t_dot_e / M_eff
            psi_dot -= M13 * P * t_dot_e / (self._M33 * M_eff)
            pvx -= P / m_p * ex
            pvy -= P / m_p * ey
            energy_lost = 0.5 * P * g_dot

        alpha = math.atan2(dy, dx)
        v_tip_x, v_tip_y = theta_dot * tvx, theta_dot * tvy
        alpha_dot = ((pvx - v_tip_x) * -math.sin(alpha) + (pvy - v_tip_y) * math.cos(alpha)) / l_s

        taut_state = [theta, theta_dot, alpha, alpha_dot, psi, psi_dot]
        # Snap the projectile exactly onto the string circle so a continued slack
        # segment starts with separation == l_s rather than integration-error above it.
        slack_state = [
            theta, theta_dot, tip_x + l_s * ex, tip_y + l_s * ey, pvx, pvy, psi, psi_dot,
        ]
        return taut_state, slack_state, energy_lost

    def _integrate_launch(self, t_max: float, rtol: float, dense_output: bool) -> LaunchSolution:
        """Integrate the launch through taut/slack sling regimes until release or t_max.

        Terminal events per regime: taut ends when the arm reaches the release angle
        or the string tension crosses zero downward (rope can't push -> slack); slack
        ends at release or when the tip-to-projectile distance grows back to the
        string length (inelastic snap, see _apply_snap). After each snap the post-snap
        tension decides whether the string stays taut or immediately goes slack again.
        """
        release_angle = self.params.release_angle
        launch = LaunchSolution(self)

        def release_event(t, y):
            return y[0] - release_angle

        release_event.terminal = True
        release_event.direction = -1

        def slack_event(t, y):
            return self.constraint_tensions(t, y)[0]

        slack_event.terminal = True
        slack_event.direction = -1

        def retension_event(t, y):
            theta = y[0]
            tip_x = self._l_a * math.cos(theta)
            tip_y = self._l_a * math.sin(theta) + self._h_T
            return math.hypot(y[2] - tip_x, y[3] - tip_y) - self._l_s

        retension_event.terminal = True
        retension_event.direction = 1

        t = 0.0
        y = self.initial_state()
        regime = "taut" if self.constraint_tensions(0.0, y)[0] >= 0.0 else "slack"
        if regime == "slack":
            y = self._slack_state_from_taut(y)

        for _ in range(MAX_LAUNCH_SEGMENTS):
            if t >= t_max:
                break

            if regime == "taut":
                sol = solve_ivp(
                    self.trebuchet_dynamics, (t, t_max), y,
                    events=[release_event, slack_event], dense_output=dense_output, rtol=rtol,
                )
            else:
                sol = solve_ivp(
                    self._launch_slack_dynamics, (t, t_max), y,
                    events=[release_event, retension_event], dense_output=dense_output, rtol=rtol,
                )
            launch.segments.append(LaunchSegment(sol=sol, t0=t, t1=float(sol.t[-1]), regime=regime))

            if sol.t_events[0].size > 0:  # release
                y_release = sol.y_events[0][0]
                launch.release_occurred = True
                launch.t_release = float(sol.t_events[0][0])
                launch.release_machine_state = (float(y_release[0]), float(y_release[1]))
                swing_base = 4 if regime == "taut" else 6
                launch.release_swing_state = (
                    float(y_release[swing_base]), float(y_release[swing_base + 1])
                )
                if regime == "taut":
                    launch.release_projectile_state = self.projectile_position_velocity(y_release)
                else:
                    launch.release_projectile_state = (
                        (float(y_release[2]), float(y_release[3])),
                        (float(y_release[4]), float(y_release[5])),
                    )
                break

            if sol.t_events[1].size > 0:  # regime switch
                t = float(sol.t_events[1][0])
                y_event = sol.y_events[1][0]
                if regime == "taut":
                    y = self._slack_state_from_taut(y_event)
                    regime = "slack"
                else:
                    taut_state, slack_state, energy_lost = self._apply_snap(y_event)
                    launch.snap_times.append(t)
                    launch.snap_energy_losses.append(energy_lost)
                    # Tiny positive threshold: at exactly zero tension scipy would
                    # re-fire the slack event at t0 as a zero-length segment.
                    if self.constraint_tensions(t, taut_state)[0] > 1e-9:
                        y, regime = taut_state, "taut"
                    else:
                        y = slack_state
                continue

            break  # no event: integrated to t_max without a release

        return launch

    def _aftermath_dynamics_taut(self, t: float, y) -> List[float]:
        """Single-DOF dynamics with the counterweight coupled through the taut rope."""
        theta, theta_dot, psi, psi_dot = y
        theta_ddot, psi_ddot = self._machine_only_accelerations(theta, theta_dot, psi, psi_dot)
        return [theta_dot, theta_ddot, psi_dot, psi_ddot]

    def _aftermath_dynamics_slack(self, t: float, y) -> List[float]:
        """Dynamics with the counterweight resting on the ground (rope slack).

        Pulley machine only - psi is inert there, so it is carried unchanged.
        """
        theta, theta_dot = y[0], y[1]
        cos_t = math.cos(theta)
        arm_drag_torque = -math.copysign(self._arm_drag_k * theta_dot * theta_dot, theta_dot)
        Q_theta = -self._arm_gravity_k * cos_t + arm_drag_torque - self._joint_friction * theta_dot
        return [theta_dot, Q_theta / self._M_slack, 0.0, 0.0]

    def _simulate_aftermath_pinned(
        self, theta_release: float, theta_dot_release: float, duration: float, rtol: float = 1e-8,
    ) -> AftermathResult:
        """Post-release dynamics for a counterweight pinned to the arm.

        Nothing can go slack here - the weight is part of the machine - so there is
        one regime for the whole window rather than the pulley machine's taut/slack
        alternation. The only discontinuity is the weight striking the ground, which
        stops the arm rather than letting it swing on through the floor; after that
        the pose is held (state_at clamps past the final segment).
        """
        half = self._cw_half_size

        def ground_event(t, y):
            (_wx, wy), _vel = self.weight_position_velocity((y[0], y[1], 0.0, 0.0, y[2], y[3]))
            return wy - half

        ground_event.terminal = True
        ground_event.direction = -1

        y0 = [theta_release, theta_dot_release, self._psi_rest, 0.0]
        # Only arm the event if the weight starts clear of the ground; otherwise the
        # solver would trip it immediately at t=0 and return a zero-length segment.
        events = ground_event if ground_event(0.0, y0) > 0 else None
        sol = solve_ivp(
            self._aftermath_dynamics_taut, (0, duration), y0,
            events=events, dense_output=True, rtol=rtol,
        )
        touchdowns = []
        if events is not None and sol.t_events[0].size:
            touchdowns.append(float(sol.t[-1]))
        return AftermathResult(
            segments=[AftermathSegment(sol=sol, t0=0.0, t1=float(sol.t[-1]), regime="taut")],
            touchdown_times=touchdowns,
            retension_times=[],
        )

    def simulate_aftermath(
        self, theta_release: float, theta_dot_release: float, duration: float,
        rtol: float = 1e-8, max_segments: int = 500,
    ) -> AftermathResult:
        """Post-release single-pendulum dynamics of arm+pulley+counterweight, run
        independently of the ballistic flight for `duration` seconds on its own clock
        (0 = release).

        The counterweight alternates between "taut" (rope loaded, airborne) and "slack"
        (resting on the ground) regimes. Touchdown (taut -> slack) is continuous: the
        rope can only pull, so the ground absorbs the counterweight's momentum without
        transmitting an impulse back through the rope to the arm. Re-tensioning
        (slack -> taut, when the arm swings back and pulls the rope taut again) is an
        inelastic angular-momentum-conserving jerk: theta_dot jumps by M_slack/M_taut.
        This jerk plus joint friction/drag is what eventually brings the system to rest.

        Known simplification: rope tension is never checked for going negative within
        the taut regime (the arm out-accelerating the falling counterweight) - the same
        assumption the launch phase already makes.
        """
        if not self.params.has_pulley:
            return self._simulate_aftermath_pinned(
                theta_release, theta_dot_release, duration, rtol=rtol
            )

        theta_ground = self._theta_ground

        def touchdown_event(t, y):
            return y[0] - theta_ground

        touchdown_event.terminal = True
        touchdown_event.direction = -1

        def retension_event(t, y):
            return y[0] - theta_ground

        retension_event.terminal = True
        retension_event.direction = 1

        segments: List[AftermathSegment] = []
        touchdown_times: List[float] = []
        retension_times: List[float] = []

        t = 0.0
        theta, theta_dot = theta_release, theta_dot_release
        regime = "taut" if theta > theta_ground else "slack"

        for _ in range(max_segments):
            remaining = duration - t
            if remaining <= 0:
                break

            if regime == "taut":
                sol = solve_ivp(
                    self._aftermath_dynamics_taut, (0, remaining), [theta, theta_dot, 0.0, 0.0],
                    events=touchdown_event, dense_output=True, rtol=rtol,
                )
            else:
                sol = solve_ivp(
                    self._aftermath_dynamics_slack, (0, remaining), [theta, theta_dot, 0.0, 0.0],
                    events=retension_event, dense_output=True, rtol=rtol,
                )

            t_end = float(sol.t[-1])
            segments.append(AftermathSegment(sol=sol, t0=t, t1=t + t_end, regime=regime))
            t += t_end

            if sol.t_events[0].size == 0:
                break  # ran out of duration without crossing the ground line again

            theta, theta_dot = sol.y_events[0][0][0], sol.y_events[0][0][1]
            # The event leaves theta exactly on the ground line, where scipy sees
            # g(t0) == 0 as a sign change in both directions and fires the next
            # segment's terminal event again at t0 - an endless chatter of
            # zero-length segments. Nudge theta strictly into the new regime in the
            # direction of travel (1e-9 rad is far below any physical resolution).
            theta = theta_ground + math.copysign(1e-9, theta_dot)

            if regime == "taut":
                touchdown_times.append(t)
                regime = "slack"
            else:
                retension_times.append(t)
                theta_dot = theta_dot * (self._M_slack / self._M_taut)
                regime = "taut"

        return AftermathResult(segments=segments, touchdown_times=touchdown_times, retension_times=retension_times)

    def initial_state(self) -> List[float]:
        """Launch-ready state: arm cocked at the initial angle, everything at rest.

        Pulley machine: the projectile is tucked alongside the arm, the sling
        angled just far enough off the arm line to clear it.

        Traditional machine: the arm is cocked nose-down and the sling hangs
        straight below the tip, putting the projectile on the ground in front of
        the machine; the counterweight likewise hangs straight down from its pin.
        (The projectile is not ground-constrained once the launch starts - the
        model has never had a contact phase - so choose a geometry where the tip
        sits about a sling length above the ground.)
        """
        p = self.params
        theta_i = float(p.initial_arm_angle)
        if p.has_pulley:
            alpha_i = theta_i + np.pi - np.arcsin(p.projectile_radius / p.string_length)
            return [theta_i, 0.0, float(alpha_i), 0.0, 0.0, 0.0]
        return [theta_i, 0.0, self._psi_rest, 0.0, self._psi_rest, 0.0]

    def simulate(
        self, t_max: float = 10.0, rtol: float = 1e-8, dense_output: bool = True, simulate_aftermath: bool = False,
    ) -> SimulationResult:
        """Run the full trebuchet simulation from launch through projectile release.

        `dense_output=False` skips building the per-step interpolants; the release
        metrics only need the event state, so callers that never sample the solution
        over time (e.g. the optimizer objective) can opt out. Energy tracking samples
        the dense solution, so it forces interpolants on regardless.

        `simulate_aftermath=True` additionally integrates the post-release machine
        dynamics (see simulate_aftermath()) for the duration of the ballistic flight,
        so animations can show the arm/counterweight settling instead of freezing at
        release. It's opt-in: the optimizer objective and default callers never pay for it.
        """
        self.energy_history = []

        launch = self._integrate_launch(t_max, rtol=rtol, dense_output=dense_output or self.track_energy)

        if self.track_energy:
            # Sample from the accepted dense solution rather than inside the RHS:
            # solve_ivp evaluates the RHS at trial points (including rejected steps),
            # which would leave the history unordered and non-physical.
            for t in np.linspace(0.0, launch.t_end, ENERGY_SAMPLES):
                self.energy_history.append(self.launch_energy_at(launch, float(t)))

        if not launch.release_occurred:
            return self._no_release_result(launch, t_max)
        return self._release_result(
            launch, dense_output=dense_output or self.track_energy, simulate_aftermath=simulate_aftermath
        )

    def _energy_metrics(self) -> Dict:
        """Energy-conservation diagnostics, if energy tracking was enabled."""
        if not (self.track_energy and len(self.energy_history) > 1):
            return {}

        analysis = self._check_energy_conservation()
        initial_energy, final_energy = self.energy_history[0], self.energy_history[-1]
        return {
            "energy_analysis": analysis,
            "energy_violations": analysis["violations"],
            "energy_conserved": len(analysis["violations"]) == 0,
            "initial_energy_details": {
                "kinetic": initial_energy["kinetic"],
                "potential": initial_energy["potential"],
                "total": initial_energy["total"],
            },
            "final_energy_details": {
                "kinetic": final_energy["kinetic"],
                "potential": final_energy["potential"],
                "total": final_energy["total"],
            },
        }

    def _effective_string_state(self, theta, theta_dot, proj_pos, proj_vel) -> Tuple[float, float]:
        """(alpha, alpha_dot) of the tip-to-projectile line, defined in both regimes.

        Matches the taut coordinates exactly when the string is taut; during slack it
        describes the line to the free-flying projectile (separation may be < l_s).
        """
        sin_t, cos_t = math.sin(theta), math.cos(theta)
        tip_x, tip_y = self._l_a * cos_t, self._l_a * sin_t + self._h_T
        dx, dy = proj_pos[0] - tip_x, proj_pos[1] - tip_y
        dist = math.hypot(dx, dy)
        alpha = math.atan2(dy, dx)
        v_tip_x, v_tip_y = -self._l_a * theta_dot * sin_t, self._l_a * theta_dot * cos_t
        alpha_dot = (
            ((proj_vel[0] - v_tip_x) * -math.sin(alpha) + (proj_vel[1] - v_tip_y) * math.cos(alpha)) / dist
            if dist > 1e-12
            else 0.0
        )
        return alpha, alpha_dot

    def _no_release_result(self, launch: LaunchSolution, t_max: float) -> SimulationResult:
        """Result when the arm never reaches the release angle within t_max."""
        theta, theta_dot = launch.machine_state(launch.t_end)
        final_pos, final_vel = launch.projectile_state(launch.t_end)
        alpha, alpha_dot = self._effective_string_state(theta, theta_dot, final_pos, final_vel)

        metrics = {
            "simulation_time": t_max,
            "final_arm_angle_deg": theta * 180 / np.pi,
            "final_string_angle_deg": alpha * 180 / np.pi,
            "arm_angular_velocity": theta_dot,
            "string_angular_velocity": alpha_dot,
            "total_rotation_deg": (self.params.initial_arm_angle - theta) * 180 / np.pi,
            "final_projectile_pos": final_pos,
            "final_projectile_vel": final_vel,
            "release_occurred": False,
            **self._tension_metrics(launch),
            **self._energy_metrics(),
        }

        return SimulationResult(
            distance=0.0,
            efficiency=0.0,
            metrics=metrics,
            solution=launch,
            energy_history=self.energy_history if self.track_energy else None,
        )

    def _release_result(
        self, launch: LaunchSolution, dense_output: bool = True, simulate_aftermath: bool = False
    ) -> SimulationResult:
        """Result when the projectile reaches release angle: compute flight distance and efficiency."""
        t_release = launch.t_release
        theta_release, theta_dot_release = launch.release_machine_state

        start_pos, _ = self.projectile_position_velocity(self.initial_state())
        release_pos, release_vel = launch.release_projectile_state
        x0, y0_height = release_pos
        vx0, vy0 = release_vel

        if np.isnan(vx0) or np.isnan(vy0) or np.isnan(x0) or np.isnan(y0_height):
            return SimulationResult(0.0, 0.0, {"error": "Invalid position/velocity at release"}, launch)

        proj_speed2 = vx0**2 + vy0**2
        proj_KE_before = 0.5 * self.params.projectile_mass * proj_speed2
        release_velocity = np.sqrt(proj_speed2)

        distance = 0.0
        flight_time = 0.0
        trajectory = None
        if y0_height >= 0:
            trajectory = integrate_ballistic_trajectory(
                x0,
                y0_height,
                vx0,
                vy0,
                self.params.projectile_mass,
                self.params.projectile_drag_coefficient,
                self.params.projectile_area,
                dense_output=dense_output,
            )
            distance = trajectory.impact_x
            flight_time = trajectory.flight_time

        aftermath = None
        if simulate_aftermath and trajectory is not None:
            # Integrated independently of the ballistic flight above (no shared state,
            # no shared dynamics) - only the stopping duration is passed in, so the two
            # can be stitched together for animation and both end when the projectile lands.
            aftermath = self.simulate_aftermath(theta_release, theta_dot_release, flight_time)

        arm_angle_rotated = self.params.initial_arm_angle - theta_release
        if self.params.has_pulley:
            # The weight descends r_pul per radian, so the drop is exactly linear.
            height_dropped = self.params.pulley_radius * arm_angle_rotated
        else:
            # The pinned weight follows the pin around the pivot and swings on top of
            # that, so measure its height directly at both ends of the launch.
            psi_0, psi_dot_0 = self._psi_rest, 0.0
            psi_r, psi_dot_r = launch.release_swing_state or (self._psi_rest, 0.0)
            start = self.weight_position_velocity(
                (self.params.initial_arm_angle, 0.0, 0.0, 0.0, psi_0, psi_dot_0)
            )[0][1]
            end = self.weight_position_velocity(
                (theta_release, theta_dot_release, 0.0, 0.0, psi_r, psi_dot_r)
            )[0][1]
            height_dropped = start - end
        counterweight_PE_spent = self.params.counter_weight_mass * G * height_dropped

        # The beam's centre of mass, not its half-length: on the traditional machine the
        # arm extends behind the pivot as well, which pulls the balance point in to
        # (a - b)/2. Dividing by two is exact, so the pulley machine (b = 0) is unchanged.
        arm_height_change = (
            (np.sin(self.params.initial_arm_angle) - np.sin(theta_release)) * self.params.arm_cm_offset
        )
        arm_PE_spent = arm_height_change * self.params.arm_mass * G

        projectile_height_change = start_pos[1] - release_pos[1]
        projectile_PE_spent = projectile_height_change * self.params.projectile_mass * G

        total_PE_spent = counterweight_PE_spent + arm_PE_spent + projectile_PE_spent
        efficiency = proj_KE_before / total_PE_spent if total_PE_spent > 0 else 0.0

        metrics = {
            "release_velocity": release_velocity,
            "release_velocity_components": (vx0, vy0),
            "release_height": y0_height,
            "release_angle_deg": theta_release * 180 / np.pi,
            "string_arm_ratio": self.params.string_arm_ratio,
            "arm_string_clearance": self.params.arm_string_clearance,
            "pe_spent": counterweight_PE_spent,
            "ke_projectile": proj_KE_before,
            "total_pe_spent": total_PE_spent,
            "arm_pe_spent": arm_PE_spent,
            "projectile_pe_spent": projectile_PE_spent,
            "t_release": t_release,
            "flight_time": flight_time,
            "arm_rotation_deg": arm_angle_rotated * 180 / np.pi,
            "release_occurred": True,
            **self._tension_metrics(launch),
            **self._energy_metrics(),
        }
        if aftermath is not None:
            metrics["cw_touchdown_times"] = aftermath.touchdown_times
            metrics["cw_retension_times"] = aftermath.retension_times

        return SimulationResult(
            distance=distance,
            efficiency=max(0.0, efficiency),
            metrics=metrics,
            solution=launch,
            energy_history=self.energy_history if self.track_energy else None,
            trajectory=trajectory,
            aftermath=aftermath,
        )


def simulate_trebuchet(
    params: TrebuchetParams,
    t_max: float = 10.0,
    track_energy: bool = False,
    rtol: float = 1e-8,
    dense_output: bool = True,
    simulate_aftermath: bool = False,
) -> SimulationResult:
    """Convenience function for a single simulation run."""
    return TrebuchetSimulator(params, track_energy=track_energy).simulate(
        t_max, rtol=rtol, dense_output=dense_output, simulate_aftermath=simulate_aftermath
    )
