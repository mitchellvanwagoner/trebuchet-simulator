#!/usr/bin/env python3
"""
Trebuchet Physics Simulation Engine
Core dynamics, ODE solving, and force data storage
"""

from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

# GPU acceleration - test CuPy and use as numpy replacement if available
try:
    import cupy as cp
    # Test basic GPU operation
    test_array = cp.array([1.0, 2.0, 3.0])
    test_result = cp.sin(test_array)
    # If we get here, CuPy works
    import cupy as np
    GPU_AVAILABLE = True
    print("GPU acceleration enabled with CuPy")
except (ImportError, Exception):
    import numpy as np
    GPU_AVAILABLE = False
    print("GPU acceleration not available, using CPU (install cupy for GPU support)")

# Constants
g = 9.81                                             # gravity (m/s^2)
rho_air = 1.225                                      # air density at sea level (kg/m³)

def to_cpu(array):
    """Convert GPU array to CPU array if needed for scipy compatibility"""
    if GPU_AVAILABLE and hasattr(array, 'get'):
        return array.get()
    return array

@dataclass
class TrebuchetParams:
    """Trebuchet configuration parameters"""
    counter_weight_mass: float                       # kg
    pulley_radius: float                             # m
    arm_length: float                                # m
    string_length: float                             # m
    release_angle: float                             # radians

    # Fixed parameters
    trebuchet_height: float = 1.0                    # m
    weight_height: float = trebuchet_height          # m
    pulley_density: float = 1250                     # kg/m³
    arm_density: float = 530                         # kg/m³
    projectile_mass: float = 0.25                    # kg (apple)
    projectile_radius: float = 0.04                  # m
    initial_arm_angle: float = np.pi/4               # 45° start position
    arm_drag_coefficient: float = 1.05               # N⋅m⋅s/rad
    projectile_drag_coefficient: float = 0.47        # N⋅m⋅s/rad
    joint_friction_coefficient: float = 0.1          # N⋅m⋅s/rad (friction at pivot)

    @property
    def pulley_mass(self):
        return self.pulley_density * np.pi * self.pulley_radius**2 * 0.1

    @property
    def arm_mass(self):
        return self.arm_density * self.arm_length * 0.05**2
    
    @property
    def projectile_area(self):
        return np.pi * self.projectile_radius**2

    @property
    def moi_pulley(self):
        return 0.5 * self.pulley_mass * self.pulley_radius**2

    @property
    def moi_arm(self):
        return (1/3) * self.arm_mass * self.arm_length**2

    @property
    def total_mass(self):
        return self.counter_weight_mass + self.pulley_mass + self.arm_mass + self.projectile_mass

    @property
    def string_arm_ratio(self):
        """String to arm length ratio"""
        return self.string_length / self.arm_length
    
    @property
    def arm_string_clearance(self):
        """Arm to string length"""
        return self.trebuchet_height - (self.arm_length + self.string_length + 0.1)

@dataclass
class ForceData:
    """Storage for force analysis at a single time point"""
    time: float
    state: Dict[str, float]  # theta, theta_dot, alpha, alpha_dot
    torques: Dict[str, float]  # individual torque components
    totals: Dict[str, float]   # Q_theta, Q_alpha totals
    inertia: Dict[str, float]  # M11, M12, M21, M22
    positions: Dict[str, Tuple[float, float]]  # component positions
    energies: Dict[str, float]  # potential, kinetic, total, time

@dataclass
class SimulationResult:
    """Complete simulation results"""
    distance: float
    efficiency: float
    metrics: Dict
    solution: object  # scipy solve_ivp solution object
    force_history: Optional[List[ForceData]] = None
    energy_history: Optional[List[Dict]] = None

class TrebuchetSimulator:
    """Main trebuchet physics simulator with GPU acceleration support"""

    def __init__(self, params: TrebuchetParams, store_forces: bool = False, track_energy: bool = False):
        self.params = params
        self.store_forces = store_forces
        self.track_energy = track_energy
        self.force_history: List[ForceData] = []
        self.energy_history: List[Dict] = []

    def weight_position_velocity(self, y):
        """Calculate weight position and velocity from state vector"""
        theta, theta_dot, alpha, alpha_dot = y
        r_pul, h_w, theta_i = self.params.pulley_radius, self.params.weight_height, self.params.initial_arm_angle

        w_y = (h_w - r_pul * (2 + theta_i )) + r_pul * theta

        w_vy = r_pul * theta_dot

        return (r_pul, w_y), (0, w_vy)
    
    def arm_position_velocity(self, y):
        """Calculate arm cm position and velocity"""
        theta, theta_dot, alpha, alpha_dot = y
        h_T, l_a = self.params.trebuchet_height, self.params.arm_length
        
        a_x = l_a/2 * np.cos(theta)
        a_y = l_a/2 * np.sin(theta) + h_T
        
        a_vx = -l_a/2 * theta_dot * np.sin(theta)
        a_vy = l_a/2 * theta_dot * np.cos(theta)
        
        return (a_x, a_y), (a_vx, a_vy)
    
    def arm_tip_position_velocity(self, y):
        """Calculate arm tip position and velocity"""
        theta, theta_dot, alpha, alpha_dot = y
        h_T, l_a = self.params.trebuchet_height, self.params.arm_length
        
        a_x = l_a * np.cos(theta)
        a_y = l_a * np.sin(theta) + h_T
        
        a_vx = -l_a * theta_dot * np.sin(theta)
        a_vy = l_a * theta_dot * np.cos(theta)
        
        return (a_x, a_y), (a_vx, a_vy)
    
    def projectile_position_velocity(self, y):
        """Calculate projectile position and velocity from state vector"""
        theta, theta_dot, alpha, alpha_dot = y
        l_a, l_s, h_T = self.params.arm_length, self.params.string_length, self.params.trebuchet_height

        # Position: arm tip - string extension
        pos_x = l_a * np.cos(theta) + l_s * np.cos(alpha)
        pos_y = l_a * np.sin(theta) + l_s * np.sin(alpha) + h_T

        # Velocity: derivatives of position
        vel_x = -l_a * theta_dot * np.sin(theta) - l_s * alpha_dot * np.sin(alpha)
        vel_y = l_a * theta_dot * np.cos(theta) + l_s * alpha_dot * np.cos(alpha)

        return (pos_x, pos_y), (vel_x, vel_y)
    
    
    def calculate_system_energy(self, y, t):
        """Calculate total system energy at current state"""
        theta, theta_dot, alpha, alpha_dot = y

        # Get positions and velocities for all components
        proj_pos, proj_vel = self.projectile_position_velocity(y)
        arm_cm_pos, arm_cm_vel = self.arm_position_velocity(y)
        arm_tip_pos, arm_tip_vel = self.arm_tip_position_velocity(y)
        cw_pos, cw_vel = self.weight_position_velocity(y)

        # Calculate kinetic energies
        proj_ke = 0.5 * self.params.projectile_mass * (proj_vel[0]**2 + proj_vel[1]**2)
        arm_ke = 0.5 * self.params.moi_arm * theta_dot**2
        pulley_ke = 0.5 * self.params.moi_pulley * theta_dot**2
        cw_ke = 0.5 * self.params.counter_weight_mass * cw_vel[1]**2

        total_ke = proj_ke + arm_ke + pulley_ke + cw_ke 

        # Calculate potential energies (relative to ground at y=0)
        proj_pe = self.params.projectile_mass * g * proj_pos[1]
        arm_pe = self.params.arm_mass * g * arm_cm_pos[1]
        cw_pe = self.params.counter_weight_mass * g * cw_pos[1]

        total_pe = proj_pe + arm_pe + cw_pe

        # Total system energy
        total_energy = total_ke + total_pe

        return {
            'time': t,
            'kinetic': total_ke,
            'potential': total_pe,
            'total': total_energy,
            'proj_ke': proj_ke,
            'arm_ke': arm_ke,
            'cw_ke': cw_ke,
            'pulley_ke': pulley_ke,
            'proj_pe': proj_pe,
            'arm_pe': arm_pe,
            'cw_pe': cw_pe
        }

    def _check_energy_conservation(self, max_percent_change: float = 5.0):
        """Check energy conservation throughout simulation"""
        violations = []
        energy_increases = []
        energy_decreases = []

        if len(self.energy_history) < 2:
            return {
                'violations': violations,
                'energy_increases': energy_increases,
                'energy_decreases': energy_decreases,
                'overall_trend': 'insufficient_data'
            }

        total_energy_change = 0
        for i in range(1, len(self.energy_history)):
            prev_energy = self.energy_history[i-1]['total']
            curr_energy = self.energy_history[i]['total']

            # Avoid division by zero
            if abs(prev_energy) < 1e-12:
                continue

            energy_change = curr_energy - prev_energy
            percent_change = (energy_change / prev_energy) * 100
            abs_percent_change = abs(percent_change)

            total_energy_change += energy_change

            if abs_percent_change > max_percent_change:
                violation_data = {
                    'time': self.energy_history[i]['time'],
                    'prev_energy': prev_energy,
                    'curr_energy': curr_energy,
                    'energy_change': energy_change,
                    'percent_change': percent_change,
                    'abs_percent_change': abs_percent_change,
                    'direction': 'increase' if energy_change > 0 else 'decrease'
                }

                violations.append(violation_data)

                if energy_change > 0:
                    energy_increases.append(violation_data)
                else:
                    energy_decreases.append(violation_data)

        # Determine overall energy trend
        if abs(total_energy_change) < 1e-6:
            overall_trend = 'constant'
        elif total_energy_change > 0:
            overall_trend = 'increasing'
        else:
            overall_trend = 'decreasing'

        return {
            'violations': violations,
            'energy_increases': energy_increases,
            'energy_decreases': energy_decreases,
            'overall_trend': overall_trend,
            'total_energy_change': total_energy_change,
            'initial_energy': self.energy_history[0]['total'] if self.energy_history else 0,
            'final_energy': self.energy_history[-1]['total'] if self.energy_history else 0
        }

    def trebuchet_dynamics(self, t, y):
        """Euler-Lagrange dynamics: [theta, theta_dot, alpha, alpha_dot]"""
        theta, theta_dot, alpha, alpha_dot = y

        # Extract parameters
        m_w, r_pul, l_a, l_s = (self.params.counter_weight_mass, self.params.pulley_radius,
                               self.params.arm_length, self.params.string_length)
        m_p, m_a, moi_a, moi_p = (self.params.projectile_mass, self.params.arm_mass,
                                 self.params.moi_arm, self.params.moi_pulley)
        cd_a, cd_p, A_p = (self.params.arm_drag_coefficient, self.params.projectile_drag_coefficient, self.params.projectile_area)
        joint_friction = self.params.joint_friction_coefficient
        
        # Calculate current velocity of projectile
        _, p_v = self.projectile_position_velocity(y)
        proj_v2 = p_v[0]**2 + p_v[1]**2


        # Inertia matrix coefficients
        M11 = m_w * r_pul**2 + moi_p + moi_a + m_p * l_a**2
        M12 = m_p * l_a * l_s * np.cos(alpha - theta)
        M21 = m_p * l_a * l_s * np.cos(alpha - theta)
        M22 = m_p * l_s**2

        # Force components impact on theta
        F11 = + m_p * l_a * l_s * np.sin(alpha - theta) * alpha_dot**2      # Projectile force contribution to theta
        F12 = - m_w * r_pul * g                                             # Pulley gravity contribution to theta
        F13 = - l_a/2 * g * np.cos(theta) * m_a                             # Arm gravity contribution to theta
        F14 = - m_p * g * l_a * np.cos(theta)                               # Projectile gravity contribution to theta
        F15 = + 1/6 * 0.05 * cd_a * rho_air * l_a**3 * theta_dot**2         # Arm drag contribution to theta
        F16 = - joint_friction * theta_dot                                  # Joint friction contribution to theta

        # Force components impact on alpha
        F21 = - m_p * l_a * l_s * np.sin(alpha - theta) * theta_dot**2      # Projectile force contribution to alpha
        F22 = - m_p * l_s * g * np.cos(alpha)                               # Projectile gravity contribution to alpha
        F23 = + 1/2 * rho_air * cd_p * A_p * proj_v2                        # Projectile drag contribution to alpha

        # Total generalized forces
        Q_theta = F11 + F12 + F13 + F14 + F15 + F16
        Q_alpha = F21 + F22 + F23

        # Solve system: M * [theta_ddot, alpha_ddot] = [Q_theta, Q_alpha]
        det = M11 * M22 - M12 * M21
        if abs(det) < 1e-12:
            return [theta_dot, 0, alpha_dot, 0]

        theta_ddot = (Q_theta * M22 - Q_alpha * M12) / det
        alpha_ddot = (Q_alpha * M11 - Q_theta * M21) / det

        # Track energy if requested
        if self.track_energy:
            energy_data = self.calculate_system_energy(y, t)
            self.energy_history.append(energy_data)

        # Convert to CPU arrays for scipy compatibility
        result = [to_cpu(theta_dot), to_cpu(theta_ddot), to_cpu(alpha_dot), to_cpu(alpha_ddot)]
        return result

    def _calculate_ballistic_range_with_drag(self, x0: float, y0: float, vx0: float, vy0: float) -> float:
        """Calculate ballistic range including air resistance"""
        from scipy.integrate import solve_ivp

        def ballistic_dynamics(t, state):
            """Ballistic trajectory with air resistance: [x, y, vx, vy]"""
            x, y, vx, vy = state

            # Current speed and drag coefficient
            v_mag = np.sqrt(vx**2 + vy**2)

            # Air drag parameters
            drag_coeff = 0.5 * rho_air * self.params.projectile_drag_coefficient * self.params.projectile_area

            # Drag force components (opposite to velocity direction)
            if v_mag > 1e-12:  # Avoid division by zero
                drag_x = -drag_coeff * v_mag * vx / self.params.projectile_mass
                drag_y = -drag_coeff * v_mag * vy / self.params.projectile_mass
            else:
                drag_x = drag_y = 0.0

            # Equations of motion
            dxdt = vx
            dydt = vy
            dvxdt = drag_x
            dvydt = -g + drag_y

            return [dxdt, dydt, dvxdt, dvydt]

        def ground_impact(t, state):
            """Event function for ground impact"""
            return state[1]  # y-coordinate

        ground_impact.terminal = True
        ground_impact.direction = -1

        # Initial conditions: [x, y, vx, vy]
        y0_state = [x0, y0, vx0, vy0]

        # Solve ballistic trajectory until ground impact
        try:
            sol = solve_ivp(ballistic_dynamics, (0, 60), y0_state,
                           events=ground_impact, dense_output=True, rtol=1e-6)

            if sol.t_events[0].size > 0:
                # Ground impact occurred
                final_state = sol.y_events[0][0]
                return max(0.0, final_state[0])  # Final x position
            else:
                # No ground impact within time limit
                final_state = sol.y[:, -1]
                return max(0.0, final_state[0])

        except Exception:
            # Fallback to simple ballistic equation if integration fails
            discriminant = vy0**2 + 2*g*y0
            if discriminant >= 0:
                t_flight = (vy0 + np.sqrt(discriminant)) / g
                return max(0.0, x0 + vx0 * t_flight)
            return 0.0

    def release_condition(self, t, y):
        """Event function for projectile release"""
        return y[0] - self.params.release_angle

    def simulate(self, t_max: float = 10.0) -> SimulationResult:
        """Run complete trebuchet simulation"""
        theta_i = self.params.initial_arm_angle
        
        # Clear force and energy history
        self.force_history = []
        self.energy_history = []

        # Calculate initial alpha
        alpha_i = theta_i + np.pi - np.arcsin(self.params.projectile_radius / self.params.string_length)
        
        # Set initial conditions
        y0 = [theta_i, 0.0, alpha_i, 0.0]

        # Create a standalone event function with proper attributes
        def release_event(t, y):
            return y[0] - self.params.release_angle

        release_event.terminal = True
        release_event.direction = -1

        # Integrate until release OR end time
        sol = solve_ivp(self.trebuchet_dynamics, (0, t_max), y0,
                       events=release_event, dense_output=True, rtol=1e-8)

        # Check if release occurred
        release_occurred = sol.t_events[0].size > 0

        if not release_occurred:
            # No release detected - return simulation end state
            final_state = sol.y[:, -1]
            final_pos, final_vel = self.projectile_position_velocity(final_state)

            metrics = {
                'simulation_time': t_max,
                'final_arm_angle_deg': final_state[0] * 180 / np.pi,
                'final_string_angle_deg': final_state[2] * 180 / np.pi,
                'arm_angular_velocity': final_state[1],
                'string_angular_velocity': final_state[3],
                'total_rotation_deg': (self.params.initial_arm_angle - final_state[0]) * 180 / np.pi,
                'final_projectile_pos': final_pos,
                'final_projectile_vel': final_vel,
                'release_occurred': False
            }

            # Check energy conservation if tracking is enabled
            if self.track_energy and len(self.energy_history) > 1:
                energy_analysis = self._check_energy_conservation()
                metrics['energy_analysis'] = energy_analysis
                metrics['energy_violations'] = energy_analysis['violations']
                metrics['energy_conserved'] = len(energy_analysis['violations']) == 0

                # Store detailed initial and final energy breakdown
                initial_energy = self.energy_history[0]
                final_energy = self.energy_history[-1]
                metrics['initial_energy_details'] = {
                    'kinetic': initial_energy['kinetic'],
                    'potential': initial_energy['potential'],
                    'total': initial_energy['total']
                }
                metrics['final_energy_details'] = {
                    'kinetic': final_energy['kinetic'],
                    'potential': final_energy['potential'],
                    'total': final_energy['total']
                }

            return SimulationResult(
                distance=0.0,
                efficiency=0.0,
                metrics=metrics,
                solution=sol,
                force_history=self.force_history if self.store_forces else None,
                energy_history=self.energy_history if self.track_energy else None
            )

        # Calculate release state and projectile trajectory
        t_release = sol.t_events[0][0]
        y_release = sol.y_events[0][0]

        start_pos, _ = self.projectile_position_velocity(y0)
        release_pos, release_vel = self.projectile_position_velocity(y_release)
        x0, y0 = release_pos
        vx0, vy0 = release_vel

        # Validate release state
        if np.isnan(vx0) or np.isnan(vy0) or np.isnan(x0) or np.isnan(y0):
            metrics = {'error': 'Invalid position/velocity at release'}
            return SimulationResult(0.0, 0.0, metrics, sol)

        # Calculate kinetic energy at release
        proj_speed_before = vx0**2 + vy0**2
        proj_KE_before = 0.5 * self.params.projectile_mass * proj_speed_before

        # Ballistic trajectory to ground with air resistance
        distance = 0.0
        if y0 >= 0 and not (np.isnan(vx0) or np.isnan(vy0)):
            distance = self._calculate_ballistic_range_with_drag(x0, y0, vx0, vy0)

        # Calculate efficiency: KE_projectile / PE_total_spent
        arm_angle_rotated = self.params.initial_arm_angle - y_release[0]
        height_dropped = self.params.pulley_radius * arm_angle_rotated
        counterweight_PE_spent = self.params.counter_weight_mass * g * height_dropped

        arm_height_change = ((np.sin(self.params.initial_arm_angle) - np.sin(y_release[0])) *
                           self.params.arm_length/2)
        arm_PE_spent = arm_height_change * self.params.arm_mass * g

        projectile_height_change = start_pos[1] - release_pos[1]
        projectile_PE_spent = projectile_height_change * self.params.projectile_mass * g

        total_PE_spent = counterweight_PE_spent + arm_PE_spent + projectile_PE_spent
        efficiency = proj_KE_before / total_PE_spent if total_PE_spent > 0 else 0.0

        # Compile detailed metrics
        metrics = {
            'release_velocity': proj_speed_before,
            'release_height': y0,
            'release_angle_deg': y_release[0] * 180 / np.pi,
            'string_arm_ratio': self.params.string_arm_ratio,
            'arm_string_clearance': self.params.arm_string_clearance,
            'pe_spent': counterweight_PE_spent,
            'ke_projectile': proj_KE_before,
            'total_pe_spent': total_PE_spent,
            'arm_pe_spent': arm_PE_spent,
            'projectile_pe_spent': projectile_PE_spent,
            't_release': t_release,
            'arm_rotation_deg': arm_angle_rotated * 180 / np.pi,
            'release_occurred': True
        }

        # Check energy conservation if tracking is enabled
        if self.track_energy and len(self.energy_history) > 1:
            energy_analysis = self._check_energy_conservation()
            metrics['energy_analysis'] = energy_analysis
            metrics['energy_violations'] = energy_analysis['violations']
            metrics['energy_conserved'] = len(energy_analysis['violations']) == 0

            # Store detailed initial and final energy breakdown
            initial_energy = self.energy_history[0]
            final_energy = self.energy_history[-1]
            metrics['initial_energy_details'] = {
                'kinetic': initial_energy['kinetic'],
                'potential': initial_energy['potential'],
                'total': initial_energy['total']
            }
            metrics['final_energy_details'] = {
                'kinetic': final_energy['kinetic'],
                'potential': final_energy['potential'],
                'total': final_energy['total']
            }

        return SimulationResult(
            distance=distance,
            efficiency=max(0.0, efficiency),
            metrics=metrics,
            solution=sol,
            force_history=self.force_history if self.store_forces else None,
            energy_history=self.energy_history if self.track_energy else None
        )

def simulate_trebuchet(params: TrebuchetParams, t_max: float = 10.0,
                      store_forces: bool = False, track_energy: bool = False) -> SimulationResult:
    """Convenience function for single simulation"""
    simulator = TrebuchetSimulator(params, store_forces=store_forces, track_energy=track_energy)
    return simulator.simulate(t_max)


def print_simulation_results(params: TrebuchetParams, result: SimulationResult):
    """Print detailed simulation results"""
    print(f"\n=== TREBUCHET SIMULATION RESULTS ===")
    print(f"Parameters:")
    print(f"  Counterweight: {params.counter_weight_mass:.1f} kg")
    print(f"  Pulley radius: {params.pulley_radius:.3f} m")
    print(f"  Arm length: {params.arm_length:.3f} m")
    print(f"  String length: {params.string_length:.3f} m")
    print(f"  Release angle: {params.release_angle*180/np.pi:.1f}°")
    print(f"  String/Arm ratio: {params.string_arm_ratio:.3f}")
    print(f"  Total mass: {params.total_mass:.1f} kg")

    # Check if release occurred
    if result.metrics.get('release_occurred', True):
        print(f"\nResults:")
        print(f"  Range: {result.distance:.2f} m")
        print(f"  Efficiency: {result.efficiency:.4f} ({result.efficiency*100:.2f}%)")
        print(f"  Release velocity: {result.metrics['release_velocity']:.2f} m/s")
        print(f"  Release height: {result.metrics['release_height']:.2f} m")
        print(f"  Release time: {result.metrics['t_release']:.3f} s")
        print(f"  Arm rotation: {result.metrics['arm_rotation_deg']:.1f}°")

        print(f"\nEnergy Analysis:")
        print(f"  Projectile KE: {result.metrics['ke_projectile']:.1f} J")
        print(f"  Counterweight PE: {result.metrics['pe_spent']:.1f} J")
        print(f"  Arm PE: {result.metrics['arm_pe_spent']:.1f} J")
        print(f"  Projectile PE: {result.metrics['projectile_pe_spent']:.1f} J")
        print(f"  Total PE spent: {result.metrics['total_pe_spent']:.1f} J")

        # Print initial and release energy if energy tracking was enabled
        if result.energy_history:
            initial_energy = result.energy_history[0]
            print(f"\nSystem Energy Tracking:")
            print(f"  Initial total energy: {initial_energy['total']:.1f} J")

            # Find energy at release time
            if 't_release' in result.metrics:
                release_time = result.metrics['t_release']
                # Find the energy entry closest to release time
                release_energy = None
                min_time_diff = float('inf')
                for energy_entry in result.energy_history:
                    time_diff = abs(energy_entry['time'] - release_time)
                    if time_diff < min_time_diff:
                        min_time_diff = time_diff
                        release_energy = energy_entry

                if release_energy:
                    print(f"  Energy at release: {release_energy['total']:.1f} J")
                    energy_change = release_energy['total'] - initial_energy['total']
                    energy_change_percent = (energy_change / initial_energy['total']) * 100
                    print(f"  Energy change: {energy_change:+.1f} J ({energy_change_percent:+.2f}%)")
                else:
                    print(f"  Energy at release: Not found")

        # Print initial and final system energies if energy tracking was enabled
        if 'initial_energy_details' in result.metrics and 'final_energy_details' in result.metrics:
            initial = result.metrics['initial_energy_details']
            final = result.metrics['final_energy_details']
            print(f"\nSystem Energy Summary:")
            print(f"  Initial: KE = {initial['kinetic']:.1f} J, PE = {initial['potential']:.1f} J, Total = {initial['total']:.1f} J")
            print(f"  Final:   KE = {final['kinetic']:.1f} J, PE = {final['potential']:.1f} J, Total = {final['total']:.1f} J")
    else:
        print(f"\nResults:")
        print(f"  **NO RELEASE OCCURRED** - Simulation ran for full duration")
        print(f"  Simulation time: {result.metrics['simulation_time']:.1f} s")
        print(f"  Final arm angle: {result.metrics['final_arm_angle_deg']:.1f}°")
        print(f"  Final string angle: {result.metrics['final_string_angle_deg']:.1f}°")
        print(f"  Total arm rotation: {result.metrics['total_rotation_deg']:.1f}°")
        print(f"  Final arm angular velocity: {result.metrics['arm_angular_velocity']:.3f} rad/s")
        print(f"  Final string angular velocity: {result.metrics['string_angular_velocity']:.3f} rad/s")

        final_pos = result.metrics['final_projectile_pos']
        final_vel = result.metrics['final_projectile_vel']
        print(f"  Final projectile position: ({final_pos[0]:.2f}, {final_pos[1]:.2f}) m")
        print(f"  Final projectile velocity: ({final_vel[0]:.2f}, {final_vel[1]:.2f}) m/s")

        # Print initial and final system energies if energy tracking was enabled
        if 'initial_energy_details' in result.metrics and 'final_energy_details' in result.metrics:
            initial = result.metrics['initial_energy_details']
            final = result.metrics['final_energy_details']
            print(f"\nSystem Energy Summary:")
            print(f"  Initial: KE = {initial['kinetic']:.1f} J, PE = {initial['potential']:.1f} J, Total = {initial['total']:.1f} J")
            print(f"  Final:   KE = {final['kinetic']:.1f} J, PE = {final['potential']:.1f} J, Total = {final['total']:.1f} J")

    # Print energy conservation results if available
    if 'energy_analysis' in result.metrics:
        analysis = result.metrics['energy_analysis']
        print(f"\nEnergy Conservation Analysis:")

        # Overall trend analysis
        trend = analysis['overall_trend']
        print(f"  Overall energy trend: {trend}")

        if 'total_energy_change' in analysis:
            total_change = analysis['total_energy_change']
            initial_energy = analysis['initial_energy']
            final_energy = analysis['final_energy']
            percent_change = (total_change / initial_energy * 100) if initial_energy != 0 else 0

            print(f"  Initial energy: {initial_energy:.1f} J")
            print(f"  Final energy: {final_energy:.1f} J")
            print(f"  Total energy change: {total_change:.1f} J ({percent_change:+.2f}%)")

            if trend == 'decreasing':
                print(f"  [OK] Energy decrease expected (drag forces dissipating energy)")
            elif trend == 'increasing':
                print(f"  [WARNING] Energy increase detected (potential physics issue!)")

        # Violation analysis
        violations = analysis['violations']
        increases = analysis['energy_increases']
        decreases = analysis['energy_decreases']

        print(f"  Total violations (>1%): {len(violations)}")
        print(f"    Energy increases: {len(increases)} {'[!]' if len(increases) > 0 else '[OK]'}")
        print(f"    Energy decreases: {len(decreases)} {'[OK]' if len(decreases) > 0 else ''}")

        if violations:
            max_violation = max(violations, key=lambda x: x['abs_percent_change'])
            direction = "UP" if max_violation['direction'] == 'increase' else "DOWN"
            print(f"  Largest violation: {max_violation['percent_change']:+.2f}% {direction} at t={max_violation['time']:.4f}s")

        if increases:
            print(f"  [WARNING]: {len(increases)} energy increases detected - check physics!")

    if result.force_history:
        print(f"\nForce Analysis:")
        print(f"  Force data points collected: {len(result.force_history)}")
        print(f"  Force storage available for animation")

    # Energy plotting is now available in animation.py
    if result.energy_history:
        print(f"\nEnergy history available - use animation.plot_energy_history() to visualize")