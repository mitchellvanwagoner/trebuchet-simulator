#!/usr/bin/env python3
"""
Trebuchet Animation and Visualization
Clean refactored animation system with dual views and force visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from typing import Optional, List, Tuple
from simulation import TrebuchetParams, SimulationResult, ForceData, g

def create_animation(params: TrebuchetParams, result: SimulationResult,
                    show_forces: bool = False) -> FuncAnimation:
    """Create trebuchet animation with dual views and optional force arrows"""

    # Extract simulation data
    sol = result.solution
    force_data = result.force_history if show_forces else None

    # Determine simulation end point
    if sol.t_events[0].size > 0:
        t_release = sol.t_events[0][0]
        y_release = sol.y_events[0][0]
        release_occurred = True
    else:
        t_release = result.metrics.get('simulation_time', 10.0)
        release_occurred = False

    # Create time array for animation
    # Target 30 FPS for smooth playback
    target_fps = 30
    dt = 1.0 / target_fps  # Real-time step
    t_anim = np.arange(0, t_release, dt)
    if t_anim[-1] < t_release:
        t_anim = np.append(t_anim, t_release)
    frame_interval = dt * 1000  # Convert to milliseconds for matplotlib

    # Calculate all state data for animation
    state_data = _calculate_animation_data(params, sol, t_anim, force_data, show_forces)

    # Calculate ballistic trajectory if release occurred
    ballistic_data = None
    ballistic_frame_interval = frame_interval  # Match trebuchet FPS
    if release_occurred:
        ballistic_data = _calculate_ballistic_trajectory(params, sol, y_release, t_release, dt)

    # Create figure and animation
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle('Trebuchet Physics Simulation', fontsize=14)

    # Create subplots
    ax_trajectory = plt.subplot(1, 2, 1)  # Full trajectory view
    ax_system = plt.subplot(1, 2, 2)      # System close-up view

    # Setup views
    _setup_trajectory_view(ax_trajectory, state_data, ballistic_data, result, params)
    _setup_system_view(ax_system, state_data, params)

    # Animation function
    def animate(frame):
        return _animate_frame(frame, ax_trajectory, ax_system, state_data, ballistic_data,
                            show_forces, params, t_release)

    # Create animation
    total_frames = len(t_anim)
    if ballistic_data:
        total_frames += len(ballistic_data['positions'])

    anim = FuncAnimation(fig, animate, frames=total_frames, interval=frame_interval,
                        blit=False, repeat=True)

    plt.tight_layout()
    return anim

def _calculate_animation_data(params: TrebuchetParams, sol, t_anim: np.ndarray,
                            force_data: Optional[List[ForceData]], show_forces: bool) -> dict:
    """Calculate all animation data from simulation results"""

    # Initialize arrays
    positions = {
        'projectile': [],
        'arm_tip': [],
        'counterweight': []
    }

    forces = [] if show_forces and force_data else None

    # Create simulator instance to access position calculation methods
    from simulation import TrebuchetSimulator
    simulator = TrebuchetSimulator(params)

    # Calculate positions for each time point
    for i, t in enumerate(t_anim):
        # Get state from ODE solution
        y = sol.sol(t)

        # Calculate weight position
        w_pos, _ = simulator.weight_position_velocity(y)
        positions['counterweight'].append(w_pos)

        # Calculate arm tip position (end of arm)
        arm_tip_pos, _ = simulator.arm_tip_position_velocity(y)
        positions['arm_tip'].append(arm_tip_pos)

        # Calculate projectile position
        proj_pos, _ = simulator.projectile_position_velocity(y)
        positions['projectile'].append(proj_pos)

        # Add synchronized force data
        if forces is not None and i < len(force_data):
            forces.append(force_data[i])
        elif forces is not None:
            forces.append(None)

    # Convert to numpy arrays
    for key in positions:
        positions[key] = np.array(positions[key])

    return {
        'times': t_anim,
        'positions': positions,
        'forces': forces
    }

def _calculate_ballistic_trajectory(params: TrebuchetParams, sol, y_release: np.ndarray,
                                  t_release: float, dt: float) -> dict:
    """Calculate ballistic trajectory after release with air resistance"""
    from scipy.integrate import solve_ivp

    # Create simulator instance to access position calculation methods
    from simulation import TrebuchetSimulator, rho_air
    simulator = TrebuchetSimulator(params)

    # Use simulation functions for consistent position and velocity calculations
    release_pos, release_vel = simulator.projectile_position_velocity(y_release)
    x0, y0 = release_pos
    vx0, vy0 = release_vel

    if y0 < 0 or np.isnan(vx0) or np.isnan(vy0):
        return None

    def ballistic_dynamics(t, state):
        """Ballistic trajectory with air resistance: [x, y, vx, vy]"""
        x, y, vx, vy = state

        # Current speed and drag coefficient
        v_mag = np.sqrt(vx**2 + vy**2)

        # Air drag parameters
        drag_coeff = 0.5 * rho_air * params.projectile_drag_coefficient * params.projectile_area

        # Drag force components (opposite to velocity direction)
        if v_mag > 1e-12:  # Avoid division by zero
            drag_x = -drag_coeff * v_mag * vx / params.projectile_mass
            drag_y = -drag_coeff * v_mag * vy / params.projectile_mass
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

    try:
        # Solve ballistic trajectory until ground impact
        sol_ballistic = solve_ivp(ballistic_dynamics, (0, 60), y0_state,
                                 events=ground_impact, dense_output=True, rtol=1e-6)

        if sol_ballistic.t_events[0].size > 0:
            # Ground impact occurred
            t_flight = sol_ballistic.t_events[0][0]
        else:
            # Use full simulation time if no ground impact
            t_flight = sol_ballistic.t[-1]

        # Create time array for animation
        t_ballistic = np.arange(0, t_flight, dt)
        if len(t_ballistic) == 0 or t_ballistic[-1] < t_flight:
            t_ballistic = np.append(t_ballistic, t_flight)

        # Calculate positions using the solved trajectory
        positions = []
        times = []
        for t in t_ballistic:
            if t <= sol_ballistic.t[-1]:
                state = sol_ballistic.sol(t)
                x, y = state[0], max(0, state[1])
            else:
                # Extrapolate if needed
                final_state = sol_ballistic.y[:, -1]
                x, y = final_state[0], max(0, final_state[1])

            positions.append((x, y))
            times.append(t_release + t)

        return {
            'positions': np.array(positions),
            'times': np.array(times),
            'release_pos': (x0, y0),
            'final_distance': positions[-1][0]
        }

    except Exception:
        # Fallback to simple ballistic trajectory if integration fails
        if vy0**2 + 2*g*y0 >= 0:
            t_flight = (vy0 + np.sqrt(vy0**2 + 2*g*y0)) / g

            # Create time array for ballistic phase
            t_ballistic = np.arange(0, t_flight, dt)
            if t_ballistic[-1] < t_flight:
                t_ballistic = np.append(t_ballistic, t_flight)

            # Calculate positions with simple ballistic equations
            positions = []
            times = []
            for t in t_ballistic:
                x = x0 + vx0 * t
                y = max(0, y0 + vy0 * t - 0.5 * g * t**2)
                positions.append((x, y))
                times.append(t_release + t)

            return {
                'positions': np.array(positions),
                'times': np.array(times),
                'release_pos': (x0, y0),
                'final_distance': positions[-1][0]
            }

    return None

def _setup_trajectory_view(ax, state_data: dict, ballistic_data: Optional[dict], result: SimulationResult, params: Optional[TrebuchetParams] = None):
    """Setup the full trajectory view with proper scaling"""

    # Get all trajectory points
    proj_positions = state_data['positions']['projectile']
    all_x = list(proj_positions[:, 0])
    all_y = list(proj_positions[:, 1])

    if ballistic_data:
        all_x.extend(ballistic_data['positions'][:, 0])
        all_y.extend(ballistic_data['positions'][:, 1])
        final_distance = ballistic_data['final_distance']
        title_suffix = f"Range: {final_distance:.1f}m, Efficiency: {result.efficiency:.3f}"
    else:
        final_distance = proj_positions[-1, 0]
        title_suffix = f"No Release - Final: ({final_distance:.1f}m, {proj_positions[-1, 1]:.1f}m)"

    # Set bounds with margin
    margin = 2.0
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    # Plot static elements
    ax.plot(proj_positions[:, 0], proj_positions[:, 1], 'b--', alpha=0.4, linewidth=2, label='Launch path')
    if ballistic_data:
        ax.plot(ballistic_data['positions'][:, 0], ballistic_data['positions'][:, 1],
               'g--', alpha=0.6, linewidth=2, label='Ballistic trajectory')

    if params:
        ax.plot(0, params.trebuchet_height, 'ko', markersize=8, label='Pivot')
    else:
        ax.plot(0, 5.0, 'ko', markersize=8, label='Pivot')  # Default trebuchet height
    ax.axhline(y=0, color='brown', linestyle='-', linewidth=3, label='Ground')

    ax.set_xlabel('Horizontal Distance (m)', fontsize=12)
    ax.set_ylabel('Height (m)', fontsize=12)
    ax.set_title(f'Full Trajectory View\n{title_suffix}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

def _setup_system_view(ax, state_data: dict, params: TrebuchetParams):
    """Setup the system close-up view with proper scaling"""

    # Get all system positions for bounds
    all_positions = []
    for key in state_data['positions']:
        all_positions.extend(state_data['positions'][key])

    all_x = [pos[0] for pos in all_positions]
    all_y = [pos[1] for pos in all_positions]

    # Set bounds with margin
    margin = 0.5
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    # Static elements
    ax.plot(0, params.trebuchet_height, 'ko', markersize=12, label='Pivot')
    ax.axhline(y=0, color='brown', linestyle='-', linewidth=4, label='Ground')

    ax.set_xlabel('Position (m)', fontsize=12)
    ax.set_ylabel('Height (m)', fontsize=12)
    ax.set_title('System Close-up View', fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

def _animate_frame(frame: int, ax_trajectory, ax_system, state_data: dict,
                  ballistic_data: Optional[dict], show_forces: bool, params: TrebuchetParams, t_release: float):
    """Animate a single frame"""

    n_system_frames = len(state_data['times'])

    if frame < n_system_frames:
        # System motion phase
        _animate_system_frame(frame, ax_system, state_data, show_forces, params)
        _update_trajectory_frame(frame, ax_trajectory, state_data)

    elif ballistic_data and frame < n_system_frames + len(ballistic_data['positions']):
        # Ballistic phase
        ballistic_frame = frame - n_system_frames
        _animate_ballistic_frame(ballistic_frame, ax_trajectory, ax_system, state_data, ballistic_data, t_release, params)

def _animate_system_frame(frame: int, ax, state_data: dict, show_forces: bool, params: TrebuchetParams):
    """Animate the system close-up view"""

    # Clear axis for clean redraw
    ax.clear()
    _setup_system_view(ax, state_data, params)

    # Get current positions
    proj_pos = state_data['positions']['projectile'][frame]
    arm_tip_pos = state_data['positions']['arm_tip'][frame]
    cw_pos = state_data['positions']['counterweight'][frame]

    # Draw trebuchet components (all positions include trebuchet height offset)
    pivot_x, pivot_y = 0, params.trebuchet_height
    ax.plot([pivot_x, arm_tip_pos[0]], [pivot_y, arm_tip_pos[1]], 'k-', linewidth=8, label='Arm')
    ax.plot([arm_tip_pos[0], proj_pos[0]], [arm_tip_pos[1], proj_pos[1]], 'r-', linewidth=4, label='String')
    ax.plot(proj_pos[0], proj_pos[1], 'ro', markersize=12, label='Projectile')
    ax.plot(cw_pos[0], cw_pos[1], 's', color='gray', markersize=12, label='Counterweight')

    # Add pulley
    pulley_circle = patches.Circle((pivot_x, pivot_y), params.pulley_radius, fill=False, edgecolor='brown', linewidth=6)
    ax.add_patch(pulley_circle)

    # Draw force arrows if requested
    if show_forces and state_data['forces'] and frame < len(state_data['forces']) and state_data['forces'][frame]:
        # Calculate arm center of mass position for force arrows only
        theta = state_data['times'][frame]  # Need to get current theta from simulation
        # For now, use arm tip position as approximation or skip arm forces
        _draw_force_arrows(ax, state_data['forces'][frame], {
            'counterweight': cw_pos,
            'projectile': proj_pos
        })

    # Update title with current time
    current_time = state_data['times'][frame]
    ax.set_title(f'System Close-up\nTime: {current_time:.3f}s', fontsize=12)

def _update_trajectory_frame(frame: int, ax, state_data: dict):
    """Update trajectory view during system motion"""
    # For now, just maintain the static view
    # Could add progressive trajectory drawing here
    pass

def _animate_ballistic_frame(frame: int, ax_trajectory, ax_system, state_data: dict,
                           ballistic_data: dict, t_release: float, params: TrebuchetParams):
    """Animate the ballistic phase"""

    # Clear and redraw trajectory view
    ax_trajectory.clear()
    # Create a minimal result object for setup
    dummy_result = type('DummyResult', (), {'efficiency': 0.0})()
    _setup_trajectory_view(ax_trajectory, state_data, ballistic_data, dummy_result, params)

    # Draw trajectory up to current point
    proj_positions = state_data['positions']['projectile']
    ax_trajectory.plot(proj_positions[:, 0], proj_positions[:, 1], 'b-', linewidth=2, label='Launch path')
    ax_trajectory.plot(ballistic_data['positions'][:frame+1, 0],
                      ballistic_data['positions'][:frame+1, 1], 'g-', linewidth=3, label='Flight path')

    # Show current projectile position
    current_pos = ballistic_data['positions'][frame]
    ax_trajectory.plot(current_pos[0], current_pos[1], 'ro', markersize=12, label='Projectile')

    # Update trajectory title
    current_time = ballistic_data['times'][frame]
    flight_time = current_time - t_release
    ax_trajectory.set_title(f'Ballistic Flight\nTime: {current_time:.3f}s, Flight: {flight_time:.3f}s', fontsize=12)

    # Keep system view showing final position
    ax_system.set_title(f'System (Final Position)\nRelease: {t_release:.3f}s', fontsize=12)

def _draw_force_arrows(ax, force_data: ForceData, positions: dict, scale_factor: float = 0.05):
    """Draw force arrows at element positions with legend"""

    arrows_drawn = []

    # Counterweight forces (blue)
    cw_torque = force_data.torques['counterweight_gravity']
    if abs(cw_torque) > 1e-6:
        arrow_length = cw_torque * scale_factor
        cw_pos = positions['counterweight']
        ax.arrow(cw_pos[0], cw_pos[1], 0, arrow_length,
                head_width=0.08, head_length=0.08,
                fc='blue', ec='blue', alpha=0.9, linewidth=2)
        arrows_drawn.append(('blue', 'Counterweight Gravity'))

    # Arm gravity forces (green)
    arm_torque = force_data.torques['arm_gravity']
    if abs(arm_torque) > 1e-6:
        arrow_length = arm_torque * scale_factor
        arm_pos = positions['arm_cm']
        ax.arrow(arm_pos[0], arm_pos[1], 0, arrow_length,
                head_width=0.08, head_length=0.08,
                fc='green', ec='green', alpha=0.9, linewidth=2)
        arrows_drawn.append(('green', 'Arm Gravity'))

    # Projectile forces (red and purple)
    proj_pos = positions['projectile']
    offset = 0.08

    # Gravity force (red)
    gravity_torque = force_data.torques['projectile_gravity']
    if abs(gravity_torque) > 1e-6:
        arrow_length = gravity_torque * scale_factor
        ax.arrow(proj_pos[0] - offset, proj_pos[1], 0, arrow_length,
                head_width=0.06, head_length=0.06,
                fc='red', ec='red', alpha=0.9, linewidth=2)
        arrows_drawn.append(('red', 'Projectile Gravity'))

    # Centrifugal force (purple)
    centrifugal_torque = force_data.torques['centrifugal']
    if abs(centrifugal_torque) > 1e-6:
        arrow_length = centrifugal_torque * scale_factor
        ax.arrow(proj_pos[0] + offset, proj_pos[1], 0, arrow_length,
                head_width=0.06, head_length=0.06,
                fc='purple', ec='purple', alpha=0.9, linewidth=2)
        arrows_drawn.append(('purple', 'Centrifugal Force'))

    # Create legend for visible force arrows
    if arrows_drawn:
        _add_force_legend(ax, arrows_drawn)

def _add_force_legend(ax, arrows_drawn):
    """Add force legend to the plot"""
    import matplotlib.patches as mpatches

    legend_elements = []
    for color, label in arrows_drawn:
        patch = mpatches.Patch(color=color, alpha=0.9, label=label)
        legend_elements.append(patch)

    # Add legend in upper right corner
    force_legend = ax.legend(handles=legend_elements, loc='upper right',
                           title='Force Arrows', fontsize=10, title_fontsize=11,
                           framealpha=0.9, fancybox=True, shadow=True)

def show_animation(anim: FuncAnimation):
    """Display animation and block until closed"""
    plt.show(block=True)

def save_animation_gif(anim: FuncAnimation, filename: str = "trebuchet_animation.gif",
                      fps: int = 30, optimize: bool = True):
    """
    Save animation as GIF file

    Args:
        anim: The FuncAnimation object to save
        filename: Output filename (should end with .gif)
        fps: Frames per second for the GIF
        optimize: Whether to optimize the GIF file size
    """
    print(f"Saving animation as GIF: {filename}")
    print(f"This may take a moment...")

    try:
        # Use pillow writer for better GIF compression
        from matplotlib.animation import PillowWriter
        writer = PillowWriter(fps=fps)
        anim.save(filename, writer=writer)
        print(f"✓ Animation saved successfully as {filename}")

    except ImportError:
        print("Pillow not available, using default GIF writer...")
        try:
            anim.save(filename, writer='pillow', fps=fps)
            print(f"✓ Animation saved successfully as {filename}")
        except Exception as e:
            print(f"✗ Failed to save animation: {e}")
            print("Note: You may need to install pillow: pip install pillow")
    except Exception as e:
        print(f"✗ Failed to save animation: {e}")

def create_and_save_gif(params: TrebuchetParams, result: SimulationResult,
                       filename: str = "trebuchet_animation.gif",
                       show_forces: bool = False,
                       fps: int = 30) -> None:
    """
    Convenience function to create animation and save as GIF

    Args:
        params: Trebuchet parameters
        result: Simulation result
        filename: Output GIF filename
        show_forces: Whether to show force arrows
        fps: Frames per second for the GIF
    """
    print("Creating trebuchet animation for GIF export...")

    # Create animation
    anim = create_animation(params, result, show_forces=show_forces)

    # Save as GIF
    save_animation_gif(anim, filename, fps=fps)

    return anim

def plot_energy_history(result: SimulationResult, params: TrebuchetParams):
    """Plot kinetic and potential energy over time with improved two-panel layout"""
    # Check if energy history is available
    if not result.energy_history:
        print("Energy tracking was not enabled. Cannot plot energy history.")
        print("To enable energy plotting, run simulation with track_energy=True")
        return

    # Extract data for plotting
    times = [entry['time'] for entry in result.energy_history]

    # Individual kinetic energy components
    proj_ke = [entry['proj_ke'] for entry in result.energy_history]
    arm_ke = [entry['arm_ke'] for entry in result.energy_history]
    cw_ke = [entry['cw_ke'] for entry in result.energy_history]
    pulley_ke = [entry['pulley_ke'] for entry in result.energy_history]

    # Individual potential energy components
    proj_pe = [entry['proj_pe'] for entry in result.energy_history]
    arm_pe = [entry['arm_pe'] for entry in result.energy_history]
    cw_pe = [entry['cw_pe'] for entry in result.energy_history]

    # Total energies
    total_energies = [entry['total'] for entry in result.energy_history]

    # Create figure with two subplots (top and bottom)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[1, 1])
    fig.suptitle('Trebuchet Energy Components Over Time', fontsize=16, fontweight='bold')

    # Top plot: Total Energy and Counterweight PE (dominant energies)
    ax1.plot(times, total_energies, 'k-', label='Total Energy', linewidth=3, alpha=0.9)
    ax1.plot(times, cw_pe, 'g--', label='Counterweight PE', linewidth=2.5, alpha=0.8)

    ax1.set_ylabel('Energy (J)', fontsize=12)
    ax1.set_title('Total Energy and Counterweight Potential Energy', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Bottom plot: All other energy components (for better detail)
    ax2.plot(times, proj_ke, 'r-', label='Projectile KE', linewidth=2)
    ax2.plot(times, arm_ke, 'b-', label='Arm KE', linewidth=2)
    ax2.plot(times, cw_ke, 'g-', label='Counterweight KE', linewidth=2)
    ax2.plot(times, pulley_ke, 'm-', label='Pulley KE', linewidth=2)
    ax2.plot(times, proj_pe, 'r--', label='Projectile PE', linewidth=2, alpha=0.8)
    ax2.plot(times, arm_pe, 'b--', label='Arm PE', linewidth=2, alpha=0.8)

    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Energy (J)', fontsize=12)
    ax2.set_title('Kinetic and Potential Energy Components (Detail View)', fontsize=14)
    ax2.legend(fontsize=10, ncol=2)
    ax2.grid(True, alpha=0.3)

    # Mark release point on both plots if it occurred
    if result.metrics.get('release_occurred', True) and 't_release' in result.metrics:
        release_time = result.metrics['t_release']

        # Add release line to both plots
        ax1.axvline(x=release_time, color='orange', linestyle=':', linewidth=2, alpha=0.8)
        ax2.axvline(x=release_time, color='orange', linestyle=':', linewidth=2, alpha=0.8)

        # Add release annotation to top plot
        max_energy_top = max(total_energies)
        ax1.annotate(f'Release\n(t={release_time:.3f}s)',
                    xy=(release_time, max_energy_top * 0.9),
                    xytext=(release_time + 0.1, max_energy_top * 0.9),
                    arrowprops=dict(arrowstyle='->', color='orange', alpha=0.8),
                    fontsize=10, color='orange', fontweight='bold')

        # Add release annotation to bottom plot
        max_energy_bottom = max(max(proj_ke + arm_ke + cw_ke + pulley_ke),
                               max(proj_pe + arm_pe))
        if max_energy_bottom > 0:  # Only add annotation if there's measurable energy
            ax2.annotate(f'Release',
                        xy=(release_time, max_energy_bottom * 0.9),
                        xytext=(release_time + 0.1, max_energy_bottom * 0.9),
                        arrowprops=dict(arrowstyle='->', color='orange', alpha=0.8),
                        fontsize=9, color='orange', fontweight='bold')

    plt.tight_layout()
    plt.show()