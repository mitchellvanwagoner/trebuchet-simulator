#!/usr/bin/env python3
"""
Compact Trebuchet Physics Simulator and Optimizer
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

# Parameter Constraints
bounds = [
    (1.0, 30.0),                     # mass (kg)
    (0.1, 0.5),                      # pulley radius (m)
    (0.1, 2.0),                      # arm length (m)
    (0.5, 1.0),                      # string length (m)
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
        return self.pulley_density * np.pi * self.pulley_radius**2 * 0.05

    @property
    def arm_mass(self):
        return self.arm_density * self.arm_length * 0.05*0.1

    @property
    def moi_pulley(self):
        return 0.5 * self.pulley_mass * self.pulley_radius**2

    @property
    def moi_arm(self):
        return (1/3) * self.arm_mass * self.arm_length**2

    @property
    def total_mass(self):
        return self.counter_weight_mass + self.pulley_mass + self.arm_mass + self.projectile_mass

def trebuchet_dynamics(t, y, params):
    """Euler-Lagrange dynamics: [theta, theta_dot, alpha, alpha_dot]"""
    theta, theta_dot, alpha, alpha_dot = y

    # Extract parameters
    m_cw, r_p, L_a, L_s = params.counter_weight_mass, params.pulley_radius, params.arm_length, params.string_length
    m_p, m_arm, I_p, I_a = params.projectile_mass, params.arm_mass, params.moi_pulley, params.moi_arm

    # Inertia matrix coefficients
    M11 = m_p * L_a**2 + I_a + m_cw * r_p**2 + I_p + m_p * L_s**2 - 2*m_p*L_a*L_s*np.cos(alpha)
    M12 = -m_p * L_s**2 + m_p*L_a*L_s*np.cos(alpha)
    M22 = m_p * L_s**2

    # Generalized forces
    Q_theta = (-m_cw * g * r_p - m_arm * g * (L_a/2) * np.cos(theta) -
               m_p * g * L_a * np.cos(theta) + m_p * g * L_s * np.cos(theta - alpha) +
               m_p * L_a * L_s * alpha_dot * theta_dot * np.sin(alpha))

    Q_alpha = (-m_p * g * L_s * np.cos(theta - alpha) -
               m_p * L_a * L_s * theta_dot**2 * np.sin(alpha))

    # Solve system: M * [theta_ddot, alpha_ddot] = [Q_theta, Q_alpha]
    det = M11 * M22 - M12**2
    if abs(det) < 1e-12:
        return [theta_dot, 0, alpha_dot, 0]

    theta_ddot = (Q_theta * M22 - Q_alpha * M12) / det
    alpha_ddot = (M11 * Q_alpha - M12 * Q_theta) / det

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
    return y[0] - params.release_angle  # theta reaches release angle

release_condition.terminal = True
release_condition.direction = -1

def simulate_trebuchet(params):
    """Simulate trebuchet and return distance, efficiency"""
    # Initial conditions: projectile starts folded back on arm
    initial_alpha = np.arcsin(params.projectile_radius / params.string_length)
    y0 = [params.initial_arm_angle, 0.0, initial_alpha, 0.0]

    # Integrate until release
    sol = solve_ivp(trebuchet_dynamics, (0, 10), y0, args=(params,),
                   events=release_condition, dense_output=True, rtol=1e-8)

    if not sol.t_events[0].size:
        return 0.0, 0.0  # No release detected

    # Calculate release state and projectile trajectory
    t_release = sol.t_events[0][0]
    y_release = sol.y_events[0][0]

    release_pos, release_vel = projectile_position_velocity(y_release, params)
    proj_x0, proj_y0 = release_pos
    proj_vx0, proj_vy0 = release_vel

    # Ballistic trajectory to ground
    if proj_y0 >= 0 and not (np.isnan(proj_vx0) or np.isnan(proj_vy0)):
        discriminant = proj_vy0**2 + 2*g*proj_y0
        if discriminant >= 0:
            t_flight = (proj_vy0 + np.sqrt(discriminant)) / g
            distance = proj_x0 + proj_vx0 * t_flight
        else:
            distance = 0.0
    else:
        distance = 0.0

    # Calculate efficiency: KE_projectile / PE_counterweight_spent
    projectile_KE = 0.5 * params.projectile_mass * (proj_vx0**2 + proj_vy0**2)
    height_dropped = params.pulley_radius * (params.initial_arm_angle - y_release[0])
    counterweight_PE_spent = params.counter_weight_mass * g * height_dropped

    efficiency = projectile_KE / counterweight_PE_spent if counterweight_PE_spent > 0 else 0.0

    return max(0.0, distance), max(0.0, efficiency)

def efficiency_objective(param_vector):
    """
    Optimization objective: maximize efficiency while targeting 30m range
    Primary: efficiency maximization
    Secondary: distance target (soft constraint)
    """
    
    for i, (val, (min_val, max_val)) in enumerate(zip(param_vector, bounds)):
        if not (min_val <= val <= max_val):
            return 1e6

    # Create parameters
    params = TrebuchetParams(*param_vector)

    # String length constraint: <= 0.95 * arm_length
    if params.string_length > 0.95 * params.arm_length:
        return 1e6

    try:
        distance, efficiency = simulate_trebuchet(params)

        if distance <= 0 or efficiency <= 0:
            return 1e6

        # Primary objective: maximize efficiency (minimize negative efficiency)
        efficiency_cost = -efficiency

        # Secondary objective: soft penalty for missing distance target
        distance_penalty = max(0, TARGET_DISTANCE - distance)**2 * 0.00048

        # Light penalty for excessive mass (encourage elegant designs)
        mass_penalty = params.total_mass * 0.008

        return efficiency_cost + distance_penalty + mass_penalty

    except:
        return 1e6

def optimize_trebuchet():
    """Optimize trebuchet for efficiency while targeting 30m range"""
    print("Optimizing Trebuchet for Energy Transfer Efficiency")
    print("=" * 55)
    print(f"Target: {TARGET_DISTANCE}m range with maximum efficiency")

    # Optimize using differential evolution
    result = differential_evolution(
        efficiency_objective,
        bounds,
        seed=42,
        maxiter=1000,
        popsize=30,
        atol=0.00001,
        workers=16,
        disp=True
    )

    # Extract results
    optimal_params = TrebuchetParams(*result.x)
    distance, efficiency = simulate_trebuchet(optimal_params)

    return result, optimal_params, distance, efficiency

def print_results(result, params, distance, efficiency):
    """Print formatted optimization results"""
    print(f"\n{'='*60}")
    print("OPTIMIZATION RESULTS")
    print(f"{'='*60}")
    print(f"Success: {result.success}")
    print(f"Optimization cost: {result.fun:.6f}")
    print(f"Function evaluations: {result.nfev}")

    print(f"\nPERFORMANCE METRICS:")
    print(f"Range achieved: {distance:.2f}m (target: {TARGET_DISTANCE}m)")
    print(f"Energy efficiency: {efficiency:.4f} ({efficiency*100:.2f}%)")
    print(f"Distance error: {abs(distance - TARGET_DISTANCE):.2f}m")
    print(f"Total mass: {params.total_mass:.1f}kg")
    print(f"Range per kg: {distance/params.total_mass:.3f} m/kg")

    print(f"\nOPTIMAL PARAMETERS:")
    print(f"Counterweight mass: {params.counter_weight_mass:.1f}kg")
    print(f"Pulley radius: {params.pulley_radius:.3f}m")
    print(f"Arm length: {params.arm_length:.3f}m")
    print(f"String length: {params.string_length:.3f}m")
    print(f"String/arm ratio: {params.string_length/params.arm_length:.3f}")
    print(f"Release angle: {params.release_angle*180/np.pi:.1f}°")

    print(f"\nCONSTRAINT CHECK:")
    constraint_ok = params.string_length <= 0.95 * params.arm_length
    print(f"String constraint satisfied: {constraint_ok}")

    if efficiency > 0.18:
        print(f"\n[*] EXCELLENT EFFICIENCY ACHIEVED!")
    elif distance >= TARGET_DISTANCE * 0.9:
        print(f"\n[+] GOOD PERFORMANCE - Close to target range")

    return params

def animate_trebuchet_operation(params):
    """Create animated visualization of the optimal trebuchet operation"""
    print(f"\nGenerating animated trebuchet operation...")

    # Simulate for animation
    initial_alpha = np.arcsin(params.projectile_radius / params.string_length)
    y0 = [params.initial_arm_angle, 0.0, initial_alpha, 0.0]

    sol = solve_ivp(trebuchet_dynamics, (0, 10), y0, args=(params,),
                   events=release_condition, dense_output=True, rtol=1e-8)

    if not sol.t_events[0].size:
        print("Cannot animate - no release detected")
        return

    from matplotlib.animation import FuncAnimation
    import matplotlib.patches as patches

    # Prepare animation data
    t_release = sol.t_events[0][0]
    y_release = sol.y_events[0][0]

    # Calculate release position and velocity
    release_pos, release_vel = projectile_position_velocity(y_release, params)
    x0, y0 = release_pos
    vx0, vy0 = release_vel

    # Calculate ballistic trajectory after release
    if y0 >= 0 and not (np.isnan(vx0) or np.isnan(vy0)):
        discriminant = vy0**2 + 2*g*y0
        if discriminant >= 0:
            t_flight = (vy0 + np.sqrt(discriminant)) / g

            # Ballistic trajectory points
            n_ballistic_frames = 30
            t_ballistic = np.linspace(0, t_flight, n_ballistic_frames)
            ballistic_positions = []
            for t in t_ballistic:
                x_ball = x0 + vx0 * t
                y_ball = y0 + vy0 * t - 0.5 * g * t**2
                ballistic_positions.append((x_ball, max(0, y_ball)))  # Don't go below ground
            ballistic_positions = np.array(ballistic_positions)
        else:
            ballistic_positions = np.array([release_pos])
    else:
        ballistic_positions = np.array([release_pos])

    # Trebuchet motion frames (before release)
    n_trebuchet_frames = min(40, len(sol.t))
    frame_indices = np.linspace(0, len(sol.t)-1, n_trebuchet_frames, dtype=int)

    # Calculate trebuchet positions for animation
    projectile_positions = []
    arm_positions = []
    weight_positions = []
    initial_weight_height = 2.0  # Starting height

    for i in frame_indices:
        # Projectile position
        proj_pos, _ = projectile_position_velocity(sol.y[:, i], params)
        projectile_positions.append(proj_pos)

        # Arm endpoint position
        theta = sol.y[0, i]
        arm_end_x = params.arm_length * np.cos(theta)
        arm_end_y = params.arm_length * np.sin(theta)
        arm_positions.append((arm_end_x, arm_end_y))

        # Counterweight position (falls as rope unwinds)
        rope_unwound = params.pulley_radius * (params.initial_arm_angle - theta)
        weight_x = -1.0  # Fixed horizontal position
        weight_y = initial_weight_height - rope_unwound
        weight_positions.append((weight_x, weight_y))

    projectile_positions = np.array(projectile_positions)
    arm_positions = np.array(arm_positions)
    weight_positions = np.array(weight_positions)

    # Combine trebuchet motion with ballistic trajectory
    total_frames = n_trebuchet_frames + len(ballistic_positions)

    # Set up figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Animation subplot - include ballistic trajectory in bounds
    all_x = np.concatenate([projectile_positions[:, 0], arm_positions[:, 0],
                           weight_positions[:, 0], ballistic_positions[:, 0]])
    all_y = np.concatenate([projectile_positions[:, 1], arm_positions[:, 1],
                           weight_positions[:, 1], ballistic_positions[:, 1]])

    margin = 1.0
    ax1.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax1.set_ylim(-0.5, max(all_y) + margin)  # Show ground level

    # Initialize plot elements
    arm_line, = ax1.plot([], [], 'k-', linewidth=6, label='Arm')
    string_line, = ax1.plot([], [], 'r-', linewidth=3, label='String')
    rope_line, = ax1.plot([], [], 'brown', linewidth=3, label='Rope')
    pulley_circle = patches.Circle((0, 0), params.pulley_radius, fill=False, edgecolor='brown', linewidth=4)
    weight_rect = patches.Rectangle((0, 0), 0.3, 0.4, facecolor='gray', edgecolor='black', linewidth=2)
    projectile_circle = patches.Circle((0, 0), params.projectile_radius, facecolor='red', edgecolor='darkred', linewidth=2)

    # Add full trajectory paths
    ax1.plot(projectile_positions[:, 0], projectile_positions[:, 1], 'b--', alpha=0.4, linewidth=2, label='Trebuchet path')
    ax1.plot(ballistic_positions[:, 0], ballistic_positions[:, 1], 'g--', alpha=0.6, linewidth=2, label='Ballistic trajectory')

    # Add trajectory trace that will be drawn during animation
    trajectory_trace, = ax1.plot([], [], 'orange', linewidth=3, alpha=0.8, label='Live trajectory')

    ax1.add_patch(pulley_circle)
    ax1.add_patch(weight_rect)
    ax1.add_patch(projectile_circle)

    # Static elements
    ax1.axhline(y=0, color='brown', linestyle='-', linewidth=3, label='Ground')
    ax1.plot(0, 0, 'ko', markersize=8, label='Pivot')

    # Mark impact point
    final_distance = ballistic_positions[-1, 0]
    ax1.plot(final_distance, 0, 'ro', markersize=10, label=f'Impact ({final_distance:.1f}m)')

    ax1.set_xlabel('Horizontal Distance (m)', fontsize=12)
    ax1.set_ylabel('Height (m)', fontsize=12)
    ax1.set_title('Animated Trebuchet Operation\n(Efficiency-Optimized Design)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Performance metrics subplot
    distance, efficiency = simulate_trebuchet(params)
    metrics = [
        "OPTIMAL TREBUCHET DESIGN",
        "",
        f"Energy Efficiency: {efficiency:.4f} ({efficiency*100:.2f}%)",
        f"Range Achieved: {distance:.1f}m",
        f"Target Distance: {TARGET_DISTANCE}m",
        f"Accuracy: {(distance/TARGET_DISTANCE)*100:.1f}%",
        "",
        "PARAMETERS:",
        f"Counterweight: {params.counter_weight_mass:.1f}kg",
        f"Pulley Radius: {params.pulley_radius:.3f}m",
        f"Arm Length: {params.arm_length:.3f}m",
        f"String Length: {params.string_length:.3f}m",
        f"String/Arm Ratio: {params.string_length/params.arm_length:.3f}",
        f"Release Angle: {params.release_angle*180/np.pi:.1f}°",
        "",
        f"Total Mass: {params.total_mass:.1f}kg",
        f"Range per kg: {distance/params.total_mass:.3f} m/kg"
    ]

    for i, metric in enumerate(metrics):
        if metric == "OPTIMAL TREBUCHET DESIGN":
            ax2.text(0.05, 0.95 - i*0.05, metric, fontsize=16, fontweight='bold', transform=ax2.transAxes)
        elif metric == "PARAMETERS:":
            ax2.text(0.05, 0.95 - i*0.05, metric, fontsize=14, fontweight='bold', transform=ax2.transAxes)
        elif metric == "":
            continue
        else:
            ax2.text(0.05, 0.95 - i*0.05, metric, fontsize=12, transform=ax2.transAxes)

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')

    # Animation function
    def animate(frame):
        if frame < n_trebuchet_frames:
            # Trebuchet motion phase
            i = frame
            theta = sol.y[0, frame_indices[i]]

            # Update arm (from pivot to tip)
            arm_line.set_data([0, arm_positions[i, 0]], [0, arm_positions[i, 1]])

            # Update string (from arm tip to projectile)
            string_line.set_data([arm_positions[i, 0], projectile_positions[i, 0]],
                               [arm_positions[i, 1], projectile_positions[i, 1]])

            # Update rope (from pulley edge to counterweight)
            pulley_edge_x = -params.pulley_radius
            pulley_edge_y = 0
            rope_line.set_data([pulley_edge_x, weight_positions[i, 0]],
                              [pulley_edge_y, weight_positions[i, 1]])

            # Update counterweight (centered on position)
            weight_rect.set_xy((weight_positions[i, 0] - 0.15, weight_positions[i, 1] - 0.2))

            # Update projectile
            projectile_circle.center = (projectile_positions[i, 0], projectile_positions[i, 1])

            # Update trajectory trace up to current position
            if i > 0:
                trace_x = projectile_positions[:i+1, 0]
                trace_y = projectile_positions[:i+1, 1]
                trajectory_trace.set_data(trace_x, trace_y)

            # Update title with time and angle
            current_time = sol.t[frame_indices[i]]
            current_angle = theta * 180 / np.pi
            ax1.set_title(f'Trebuchet Launch Phase\nTime: {current_time:.2f}s, Arm Angle: {current_angle:.1f}°',
                         fontsize=14, fontweight='bold')

        else:
            # Ballistic flight phase
            ballistic_frame = frame - n_trebuchet_frames

            # Hide trebuchet components during flight
            arm_line.set_data([], [])
            string_line.set_data([], [])

            # Keep final trebuchet position
            final_theta = sol.y[0, frame_indices[-1]]
            final_arm_x = params.arm_length * np.cos(final_theta)
            final_arm_y = params.arm_length * np.sin(final_theta)
            arm_line.set_data([0, final_arm_x], [0, final_arm_y])

            # Update projectile position in ballistic flight
            if ballistic_frame < len(ballistic_positions):
                projectile_circle.center = (ballistic_positions[ballistic_frame, 0],
                                           ballistic_positions[ballistic_frame, 1])

                # Update trajectory trace to include ballistic path
                all_trace_x = np.concatenate([projectile_positions[:, 0],
                                            ballistic_positions[:ballistic_frame+1, 0]])
                all_trace_y = np.concatenate([projectile_positions[:, 1],
                                            ballistic_positions[:ballistic_frame+1, 1]])
                trajectory_trace.set_data(all_trace_x, all_trace_y)

                # Calculate flight time and velocity
                flight_time = ballistic_frame * t_flight / len(ballistic_positions)
                current_height = ballistic_positions[ballistic_frame, 1]
                current_distance = ballistic_positions[ballistic_frame, 0]

                ax1.set_title(f'Ballistic Flight Phase\nTime: {flight_time:.2f}s, Height: {current_height:.1f}m, Distance: {current_distance:.1f}m',
                             fontsize=14, fontweight='bold')

        return arm_line, string_line, rope_line, weight_rect, projectile_circle, trajectory_trace

    # Create and run animation
    anim = FuncAnimation(fig, animate, frames=total_frames, interval=120, blit=False, repeat=True)

    plt.tight_layout()
    plt.show()

    return anim

def plot_trebuchet_performance(params):
    """Static performance plot (backup if animation fails)"""
    print(f"\nGenerating performance visualization...")

    # Simulate for plotting
    initial_alpha = np.arcsin(params.projectile_radius / params.string_length)
    y0 = [params.initial_arm_angle, 0.0, initial_alpha, 0.0]

    sol = solve_ivp(trebuchet_dynamics, (0, 10), y0, args=(params,),
                   events=release_condition, dense_output=True, rtol=1e-8)

    if not sol.t_events[0].size:
        print("Cannot plot - no release detected")
        return

    # Calculate trajectory
    t_points = sol.t
    positions = []

    for i, t in enumerate(t_points):
        pos, _ = projectile_position_velocity(sol.y[:, i], params)
        positions.append(pos)

    positions = np.array(positions)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Trajectory plot
    ax1.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, label='Projectile path')
    ax1.plot(positions[0, 0], positions[0, 1], 'go', markersize=8, label='Start')
    ax1.plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=8, label='Release')
    ax1.axhline(y=0, color='brown', linestyle='-', linewidth=2, label='Ground')
    ax1.set_xlabel('Horizontal Distance (m)')
    ax1.set_ylabel('Height (m)')
    ax1.set_title(f'Trebuchet Trajectory\nRange: {simulate_trebuchet(params)[0]:.1f}m')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Performance metrics
    distance, efficiency = simulate_trebuchet(params)
    metrics = [
        f"Range: {distance:.1f}m",
        f"Efficiency: {efficiency:.3f}",
        f"Mass: {params.total_mass:.1f}kg",
        f"Range/kg: {distance/params.total_mass:.2f}",
        f"CW: {params.counter_weight_mass:.1f}kg",
        f"Arm: {params.arm_length:.2f}m",
        f"String: {params.string_length:.2f}m",
        f"Release: {params.release_angle*180/np.pi:.0f}°"
    ]

    ax2.text(0.1, 0.9, "PERFORMANCE METRICS", fontsize=14, fontweight='bold', transform=ax2.transAxes)
    for i, metric in enumerate(metrics):
        ax2.text(0.1, 0.8 - i*0.08, metric, fontsize=11, transform=ax2.transAxes)

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

def main():
    """Main execution function"""
    try:
        # Optimize trebuchet
        result, params, distance, efficiency = optimize_trebuchet()

        # Print results
        optimal_params = print_results(result, params, distance, efficiency)

        # Animated visualization
        try:
            animate_trebuchet_operation(optimal_params)
        except Exception as e:
            print(f"Animation failed ({e}), showing static plot instead...")
            plot_trebuchet_performance(optimal_params)

        print(f"\n{'='*60}")
        print("SIMULATION COMPLETE")
        print(f"{'='*60}")

    except Exception as e:
        print(f"Optimization failed: {e}")

if __name__ == "__main__":
    main()