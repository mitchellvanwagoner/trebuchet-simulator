#!/usr/bin/env python3
"""
Advanced Trebuchet Physics Simulator and Optimizer
Optimizes for energy transfer efficiency while targeting 30m range
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from dataclasses import dataclass

# Constants
g = 9.81  # gravity (m/s^2)
TARGET_DISTANCE = 30.0  # target range (m)

# OPTIMIZATION TUNING PARAMETERS (easily adjustable)
EFFICIENCY_WEIGHT = 1.0      # Primary: maximize energy transfer efficiency
DISTANCE_WEIGHT = 1.0        # Secondary: hit target distance
MASS_WEIGHT = 1.0           # Tertiary: minimize total mass
STRING_UTILIZATION_WEIGHT = 0.5  # Encourage proper string utilization

seed= 42
maxiter= 500
popsize= 40
atol= 0.0001
workers= 16

# Parameter bounds: [mass, pulley_radius, arm_length, string_length, release_angle]
PARAM_BOUNDS = [
    (1.0, 50.0),                     # counterweight mass (kg)
    (0.1, 1.0),                      # pulley radius (m)
    (0.1, 2.5),                      # arm length (m)
    (0.1, 2.5),                      # string length (m) - wider range for proper trebuchet
    (-np.pi*1.65, -np.pi*1.25)       # release angle (rad)
]

@dataclass
class TrebuchetParams:
    """Trebuchet configuration parameters"""
    counter_weight_mass: float  # kg
    pulley_radius: float       # m
    arm_length: float          # m
    string_length: float       # m
    release_angle: float       # radians

    # Fixed parameters
    pulley_density: float = 1250     # kg/m³
    arm_density: float = 530         # kg/m³
    projectile_mass: float = 0.25    # kg (apple)
    projectile_radius: float = 0.04  # m
    initial_arm_angle: float = np.pi/4  # 45° start position

    @property
    def pulley_mass(self):
        return self.pulley_density * np.pi * self.pulley_radius**2 * 0.1

    @property
    def arm_mass(self):
        return self.arm_density * self.arm_length * 0.05**2

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
        """String to arm length ratio - optimal trebuchet should utilize longer strings"""
        return self.string_length / self.arm_length

def trebuchet_dynamics(t, y, params):
    """Euler-Lagrange dynamics: [theta, theta_dot, alpha, alpha_dot]"""
    theta, theta_dot, alpha, alpha_dot = y

    # Extract parameters
    m_w, r_pul, l_a, l_s = params.counter_weight_mass, params.pulley_radius, params.arm_length, params.string_length
    m_p, m_a, moi_a = params.projectile_mass, params.arm_mass, params.moi_arm

    # Inertia matrix coefficients from your new equations
    # Equation 1: theta_ddot coefficient
    M11 = m_w * r_pul**2 + moi_a + m_p * l_a**2 + m_p * l_s**2 + m_p * l_a * l_s * np.sin(alpha)
    # Equation 1: alpha_ddot coefficient (note the sign correction)
    M12 = -0.5 * m_p * l_s**2
    
    # Equation 2: theta_ddot coefficient
    M21 = -0.5 * m_p * l_s**2 - 0.5 * m_p * l_a * l_s * np.sin(alpha)
    # Equation 2: alpha_ddot coefficient
    M22 = m_p * l_s**2

    # Right-hand side forces from your equations
    # Equation 1 RHS (counterweight should pull clockwise, hence negative)
    Q_theta = (-m_w * g * r_pul + m_a * g * (l_a/2) * np.cos(theta) +
               m_p * g * l_a * np.cos(theta) - m_p * g * l_s * np.cos(theta - alpha))

    # Equation 2 RHS
    Q_alpha = (-0.5 * m_p * l_a * l_s * theta_dot**2 * np.sin(alpha) +
               0.5 * m_p * l_a * l_s * theta_dot * np.sin(alpha) + m_p * g * l_s * np.cos(theta - alpha))

    # Solve system: M * [theta_ddot, alpha_ddot] = [Q_theta, Q_alpha]
    # Using your matrix structure: [[M11, M12], [M21, M22]]
    det = M11 * M22 - M12 * M21
    if abs(det) < 1e-12:
        return [theta_dot, 0, alpha_dot, 0]

    theta_ddot = (Q_theta * M22 - Q_alpha * M12) / det
    alpha_ddot = (M21 * Q_theta - M11 * Q_alpha) / det

    return [theta_dot, theta_ddot, alpha_dot, alpha_ddot]


def projectile_position_velocity(y, params):
    """Calculate projectile position and velocity"""
    theta, theta_dot, alpha, alpha_dot = y
    L_a, L_s = params.arm_length, params.string_length

    # Position: arm tip - string extension
    pos_x = L_a * np.cos(theta) - L_s * np.cos(theta - alpha)
    pos_y = L_a * np.sin(theta) - L_s * np.sin(theta - alpha)

    # Velocity: derivatives of position
    vel_x = -L_a * theta_dot * np.sin(theta) + L_s * (theta_dot - alpha_dot) * np.sin(theta - alpha)
    vel_y = L_a * theta_dot * np.cos(theta) - L_s * (theta_dot - alpha_dot) * np.cos(theta - alpha)

    return (pos_x, pos_y), (vel_x, vel_y)

def release_condition(t, y, params):
    """Event function for projectile release"""
    return y[0] - params.release_angle

release_condition.terminal = True
release_condition.direction = -1

def simulate_trebuchet(params):
    """Simulate trebuchet and return distance, efficiency, and detailed metrics"""
    # Initial conditions: projectile starts folded back on arm
    initial_alpha = np.arcsin(params.projectile_radius / params.string_length)
    y0 = [params.initial_arm_angle, 0.0, initial_alpha, 0.0]

    # Integrate until release
    sol = solve_ivp(trebuchet_dynamics, (0, 10), y0, args=(params,),
                   events=release_condition, dense_output=True, rtol=1e-8)

    if not sol.t_events[0].size:
        return 0.0, 0.0, {}  # No release detected

    # Calculate release state and projectile trajectory
    t_release = sol.t_events[0][0]
    y_release = sol.y_events[0][0]

    start_pos, start_vel = projectile_position_velocity(y0, params)
    release_pos, release_vel = projectile_position_velocity(y_release, params)
    x0, y0 = release_pos
    vx0, vy0 = release_vel

    # Debug check for valid values
    if np.isnan(vx0) or np.isnan(vy0) or np.isnan(x0) or np.isnan(y0):
        return 0.0, 0.0, {'error': 'Invalid position/velocity at release'}

    # Check for energy conservation at release
    proj_speed_before = np.sqrt(vx0**2 + vy0**2)
    proj_KE_before = 0.5 * params.projectile_mass * proj_speed_before**2

    # Ballistic trajectory to ground
    if y0 >= 0 and not (np.isnan(vx0) or np.isnan(vy0)):
        discriminant = vy0**2 + 2*g*y0
        if discriminant >= 0:
            t_flight = (vy0 + np.sqrt(discriminant)) / g
            distance = x0 + vx0 * t_flight
        else:
            distance = 0.0
    else:
        distance = 0.0

    # Calculate efficiency: KE_projectile / PE_counterweight_spent    
    # Counterweight PE Calc
    arm_angle_rotated = params.initial_arm_angle - y_release[0]  # Total rotation
    height_dropped = params.pulley_radius * arm_angle_rotated
    counterweight_PE_spent = params.counter_weight_mass * g * height_dropped
    
    # Arm PE Calc
    arm_height_change = (np.sin(params.initial_arm_angle) - np.sin(y_release[0])) * params.arm_length/2
    arm_PE_spent = arm_height_change * params.arm_mass * g
    
    # Projectile PE Calc
    projectile_height_change = release_pos[1] - start_pos[1]
    projectile_PE_spent = projectile_height_change * params.projectile_mass * g
    
    # Total PE Spent
    totalt_PE_spent = counterweight_PE_spent + arm_PE_spent + projectile_PE_spent

    efficiency = proj_KE_before / totalt_PE_spent if totalt_PE_spent > 0 else 0.0

    # Additional metrics for analysis
    metrics = {
        'release_velocity': proj_speed_before,
        'release_height': y0,
        'release_angle_deg': y_release[0] * 180 / np.pi,
        'string_arm_ratio': params.string_arm_ratio,
        'pe_spent': counterweight_PE_spent,
        'ke_projectile': proj_KE_before
    }

    return max(0.0, distance), max(0.0, efficiency), metrics

def calculate_string_utilization_factor(params):
    """
    Calculate how well the trebuchet utilizes string length for mechanical advantage
    Higher values indicate better trebuchet design vs catapult design
    """
    ratio = params.string_arm_ratio

    # Optimal trebuchet typically has string/arm ratio between 0.4-0.8
    # Penalize both too short (catapult-like) and too long (inefficient)
    if ratio < 0.3:
        return 0.1  # Very catapult-like
    elif ratio < 0.4:
        return 0.5  # Borderline
    elif ratio <= 0.8:
        return 1.0  # Good trebuchet range
    elif ratio <= 1.0:
        return 0.8  # Getting too long
    else:
        return 0.4  # Too long, losing efficiency

def efficiency_objective(param_vector):
    """
    Normalized multi-objective optimization function
    """
    # Parameter bounds check
    for i, (val, (min_val, max_val)) in enumerate(zip(param_vector, PARAM_BOUNDS)):
        if not (min_val <= val <= max_val):
            return 1e6

    # Create parameters
    params = TrebuchetParams(*param_vector)

    # String length constraint: <= 0.95 * arm_length
    if params.string_length > 0.95 * params.arm_length:
        return 1e6

    try:
        distance, efficiency, metrics = simulate_trebuchet(params)

        if distance <= 0 or efficiency <= 0:
            return 1e6

        # NORMALIZED COST COMPONENTS

        # 1. Efficiency component (primary objective - maximize)
        # Normalize: typical efficiency range 0.1-0.3, target ~0.2
        efficiency_normalized = efficiency / 0.2
        efficiency_cost = -efficiency_normalized  # Negative to maximize

        # 2. Distance component (secondary objective - hit target)
        # Normalize: error relative to target distance
        distance_error = abs(distance - TARGET_DISTANCE) / TARGET_DISTANCE
        distance_cost = distance_error**2

        # 3. Mass component (tertiary objective - minimize weight)
        # Normalize: typical mass range 20-100kg, target ~50kg
        mass_normalized = params.total_mass / 50.0
        mass_cost = mass_normalized

        # 4. String utilization component (encourage proper trebuchet design)
        string_util_factor = calculate_string_utilization_factor(params)
        string_cost = -(string_util_factor - 1.0)  # Reward good utilization

        # Combine all components with tunable weights
        total_cost = (EFFICIENCY_WEIGHT * efficiency_cost +
                     DISTANCE_WEIGHT * distance_cost +
                     MASS_WEIGHT * mass_cost +
                     STRING_UTILIZATION_WEIGHT * string_cost)

        return total_cost

    except Exception as e:
        print(f"Optimization failed: {e}")
        return 1e6

def optimize_trebuchet():
    """Optimize trebuchet for efficiency while targeting 30m range"""
    print(f"Optimizing for {TARGET_DISTANCE}m range with efficiency priority")

    # Optimize using differential evolution
    result = differential_evolution(
        efficiency_objective,
        PARAM_BOUNDS,
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        atol=atol,
        workers=workers,
        disp=True
    )

    # Extract results
    optimal_params = TrebuchetParams(*result.x)
    distance, efficiency, metrics = simulate_trebuchet(optimal_params)

    return result, optimal_params, distance, efficiency, metrics

def print_results(result, params, distance, efficiency, metrics):
    """Print essential system properties"""
    print(f"\nRange: {distance:.2f}m | Target: {TARGET_DISTANCE}m | Error: {abs(distance - TARGET_DISTANCE):.2f}m")
    print(f"Efficiency: {efficiency:.4f} ({efficiency*100:.2f}%) | Mass: {params.total_mass:.1f}kg")
    print(f"Counterweight: {params.counter_weight_mass:.1f}kg | Arm: {params.arm_length:.3f}m | String: {params.string_length:.3f}m")
    print(f"String/Arm ratio: {params.string_arm_ratio:.3f} | Release: {params.release_angle*180/np.pi:.1f}°")
    if 'release_velocity' in metrics:
        print(f"Release velocity: {metrics['release_velocity']:.2f}m/s | Energy: {metrics['ke_projectile']:.1f}J from {metrics['pe_spent']:.1f}J")
    else:
        print(f"Simulation error: {metrics.get('error', 'Unknown error')}")

    return params

def create_dual_visualization(params):
    """Create dual view: full trajectory + trebuchet close-up"""

    # Animation timing constants
    fps = 30  # Simplified framerate
    dt_anim = 1.0 / fps

    # Simulate for plotting
    initial_alpha = np.arcsin(params.projectile_radius / params.string_length)
    y0 = [params.initial_arm_angle, 0.0, initial_alpha, 0.0]

    sol = solve_ivp(trebuchet_dynamics, (0, 10), y0, args=(params,),
                   events=release_condition, dense_output=True, rtol=1e-8)

    if not sol.t_events[0].size:
        print("Cannot plot - no release detected")
        return

    from matplotlib.animation import FuncAnimation
    import matplotlib.patches as patches

    # Calculate release data for ballistic trajectory
    t_release = sol.t_events[0][0]
    y_release = sol.y_events[0][0]
    release_pos, release_vel = projectile_position_velocity(y_release, params)
    x0, y0 = release_pos
    vx0, vy0 = release_vel

    # Calculate smooth ballistic trajectory at 60 fps with physics timing
    if y0 >= 0 and not (np.isnan(vx0) or np.isnan(vy0)):
        discriminant = vy0**2 + 2*g*y0
        if discriminant >= 0:
            t_flight = (vy0 + np.sqrt(discriminant)) / g
            # Create ballistic time points
            t_ballistic = np.arange(0, t_flight, dt_anim)
            if t_ballistic[-1] < t_flight:
                t_ballistic = np.append(t_ballistic, t_flight)

            ballistic_positions = []
            ballistic_times = []  # Store corresponding physics times
            for i, t in enumerate(t_ballistic):
                x_ball = x0 + vx0 * t
                y_ball = y0 + vy0 * t - 0.5 * g * t**2
                ballistic_positions.append((x_ball, max(0, y_ball)))
                ballistic_times.append(t_release + t)  # Absolute time from start

            ballistic_positions = np.array(ballistic_positions)
            ballistic_times = np.array(ballistic_times)
        else:
            ballistic_positions = np.array([release_pos])
            ballistic_times = np.array([t_release])
            t_flight = 0
    else:
        ballistic_positions = np.array([release_pos])
        ballistic_times = np.array([t_release])
        t_flight = 0

    # Create time points for animation
    t_anim_trebuchet = np.arange(0, t_release, dt_anim)
    if t_anim_trebuchet[-1] < t_release:
        t_anim_trebuchet = np.append(t_anim_trebuchet, t_release)

    # Use dense output for smooth interpolation at 60 fps
    projectile_positions = []
    arm_positions = []
    weight_positions = []
    initial_weight_height = 2.5

    for t in t_anim_trebuchet:
        # Interpolate solution at this time point
        y_interp = sol.sol(t)

        proj_pos, _ = projectile_position_velocity(y_interp, params)
        projectile_positions.append(proj_pos)

        theta = y_interp[0]
        arm_end_x = params.arm_length * np.cos(theta)
        arm_end_y = params.arm_length * np.sin(theta)
        arm_positions.append((arm_end_x, arm_end_y))

        rope_unwound = params.pulley_radius * (params.initial_arm_angle - theta)
        weight_x = -1.2
        weight_y = initial_weight_height - rope_unwound
        weight_positions.append((weight_x, weight_y))

    projectile_positions = np.array(projectile_positions)
    arm_positions = np.array(arm_positions)
    weight_positions = np.array(weight_positions)
    n_trebuchet_frames = len(t_anim_trebuchet)

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle('Trebuchet Physics Simulation', fontsize=14)

    # Full trajectory view (left, larger)
    ax1 = plt.subplot(1, 2, 1)

    # Trebuchet close-up (right, smaller)
    ax2 = plt.subplot(1, 2, 2)

    # FULL TRAJECTORY VIEW SETUP
    all_x_full = np.concatenate([projectile_positions[:, 0], ballistic_positions[:, 0]])
    all_y_full = np.concatenate([projectile_positions[:, 1], ballistic_positions[:, 1]])

    margin_full = 2.0
    ax1.set_xlim(min(all_x_full) - margin_full, max(all_x_full) + margin_full)
    ax1.set_ylim(-1, max(all_y_full) + margin_full)

    # Plot full trajectories
    ax1.plot(projectile_positions[:, 0], projectile_positions[:, 1], 'b--', alpha=0.4, linewidth=2, label='Launch path')
    ax1.plot(ballistic_positions[:, 0], ballistic_positions[:, 1], 'g--', alpha=0.6, linewidth=2, label='Ballistic trajectory')

    # Mark key points
    final_distance = ballistic_positions[-1, 0]
    ax1.plot(0, 0, 'ko', markersize=8, label='Pivot')
    ax1.plot(final_distance, 0, 'ro', markersize=10, label=f'Impact ({final_distance:.1f}m)')
    ax1.axhline(y=0, color='brown', linestyle='-', linewidth=3, label='Ground')

    ax1.set_xlabel('Horizontal Distance (m)', fontsize=12)
    ax1.set_ylabel('Height (m)', fontsize=12)
    ax1.set_title(f'Full Trajectory View\nRange: {final_distance:.1f}m, Efficiency: {simulate_trebuchet(params)[1]:.3f}', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # TREBUCHET CLOSE-UP SETUP
    trebuchet_margin = 0.5
    ax2.set_xlim(-2, max(arm_positions[:, 0]) + trebuchet_margin)
    ax2.set_ylim(-2, max(max(arm_positions[:, 1]), max(weight_positions[:, 1])) + trebuchet_margin)

    # Initialize animated elements for close-up
    arm_line, = ax2.plot([], [], 'k-', linewidth=8, label='Arm')
    string_line, = ax2.plot([], [], 'r-', linewidth=4, label='String')
    rope_line, = ax2.plot([], [], 'brown', linewidth=4, label='Rope')

    pulley_circle = patches.Circle((0, 0), params.pulley_radius, fill=False, edgecolor='brown', linewidth=6)
    weight_rect = patches.Rectangle((0, 0), 0.4, 0.5, facecolor='gray', edgecolor='black', linewidth=3)
    projectile_circle = patches.Circle((0, 0), params.projectile_radius*3, facecolor='red', edgecolor='darkred', linewidth=3)

    ax2.add_patch(pulley_circle)
    ax2.add_patch(weight_rect)
    ax2.add_patch(projectile_circle)

    ax2.axhline(y=0, color='brown', linestyle='-', linewidth=4, label='Ground')
    ax2.plot(0, 0, 'ko', markersize=12, label='Pivot')

    ax2.set_xlabel('Position (m)', fontsize=12)
    ax2.set_ylabel('Height (m)', fontsize=12)
    ax2.set_title('Trebuchet Close-up View', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    # Animation setup
    total_frames = n_trebuchet_frames + len(ballistic_positions)

    def animate(frame):
        if frame < n_trebuchet_frames:
            # Trebuchet motion phase
            i = frame
            current_time = t_anim_trebuchet[i]

            # Get current theta from interpolated solution
            y_interp = sol.sol(current_time)
            current_theta = y_interp[0]

            # Update close-up view
            arm_line.set_data([0, arm_positions[i, 0]], [0, arm_positions[i, 1]])
            string_line.set_data([arm_positions[i, 0], projectile_positions[i, 0]],
                               [arm_positions[i, 1], projectile_positions[i, 1]])

            pulley_edge_x = -params.pulley_radius
            rope_line.set_data([pulley_edge_x, weight_positions[i, 0]],
                              [0, weight_positions[i, 1]])

            weight_rect.set_xy((weight_positions[i, 0] - 0.2, weight_positions[i, 1] - 0.25))
            projectile_circle.center = (projectile_positions[i, 0], projectile_positions[i, 1])

            # Update titles
            current_angle = current_theta * 180 / np.pi
            ax2.set_title(f'Trebuchet Close-up\nTime: {current_time:.3f}s, Arm: {current_angle:.1f}deg', fontsize=12)

        else:
            # Ballistic phase
            ballistic_frame = frame - n_trebuchet_frames

            if ballistic_frame < len(ballistic_positions):
                # Update full trajectory view with moving projectile
                proj_x, proj_y = ballistic_positions[ballistic_frame]
                current_ballistic_time = ballistic_times[ballistic_frame]

                # Clear and redraw trajectory points on full view
                ax1.clear()
                ax1.plot(projectile_positions[:, 0], projectile_positions[:, 1], 'b--', alpha=0.4, linewidth=2, label='Launch path')
                ax1.plot(ballistic_positions[:ballistic_frame+1, 0], ballistic_positions[:ballistic_frame+1, 1], 'g-', linewidth=3, label='Flight path')
                ax1.plot(ballistic_positions[ballistic_frame+1:, 0], ballistic_positions[ballistic_frame+1:, 1], 'g--', alpha=0.3, linewidth=2)

                # Mark current projectile position
                ax1.plot(proj_x, proj_y, 'ro', markersize=12, label='Projectile')
                ax1.plot(0, 0, 'ko', markersize=8, label='Pivot')
                ax1.plot(final_distance, 0, 'ro', markersize=8, alpha=0.5, label=f'Target ({final_distance:.1f}m)')
                ax1.axhline(y=0, color='brown', linestyle='-', linewidth=3)

                ax1.set_xlim(min(all_x_full) - margin_full, max(all_x_full) + margin_full)
                ax1.set_ylim(-1, max(all_y_full) + margin_full)
                ax1.set_xlabel('Horizontal Distance (m)', fontsize=12)
                ax1.set_ylabel('Height (m)', fontsize=12)
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                # Update title with time info
                flight_duration = current_ballistic_time - t_release
                ax1.set_title(f'Ballistic Flight\nTime: {current_ballistic_time:.3f}s, Flight: {flight_duration:.3f}s, Height: {proj_y:.1f}m, Distance: {proj_x:.1f}m', fontsize=12)

            # Keep trebuchet close-up showing final position during ballistic phase
            if n_trebuchet_frames > 0:
                final_trebuchet_frame = n_trebuchet_frames - 1
                final_time = t_anim_trebuchet[final_trebuchet_frame]
                y_final = sol.sol(final_time)
                final_theta = y_final[0] * 180 / np.pi
                ax2.set_title(f'Trebuchet (Final Position)\nRelease: {t_release:.3f}s, Final Arm: {final_theta:.1f}deg', fontsize=12)

        return arm_line, string_line, rope_line, weight_rect, projectile_circle

    # Create standard animation
    interval_ms = 1000 / fps
    anim = FuncAnimation(fig, animate, frames=total_frames, interval=interval_ms, blit=False, repeat=True)

    plt.tight_layout()
    plt.show()

    return anim

def main():
    """Main execution function"""
    try:
        # Optimize trebuchet
        result, params, distance, efficiency, metrics = optimize_trebuchet()

        # Print results
        optimal_params = print_results(result, params, distance, efficiency, metrics)

        # Dual visualization
        try:
            create_dual_visualization(optimal_params)
        except Exception as e:
            print(f"Visualization failed: {e}")

        print(f"\nOptimization complete.")

    except Exception as e:
        print(f"Optimization failed: {e}")

if __name__ == "__main__":
    main()