#!/usr/bin/env python3
"""
Trebuchet Parameter Optimization
Simple optimization function for trebuchet parameters using differential evolution
"""

import numpy as np
from scipy.optimize import differential_evolution
from simulation import TrebuchetParams, simulate_trebuchet
from animation import create_animation, show_animation, plot_energy_history, save_animation_gif
import matplotlib.pyplot as plt

# Global optimization variables
_target_distance = 30.0
_efficiency_weight = 5.0
_distance_weight = 1.0
_mass_weight = 0.15

_seed=78235
_maxiter=1000
_popsize=40
_atol=0.001
_workers=16

# Parameter locking configuration - set to None to optimize, or specify value to lock
counter_weight_mass = None
pulley_radius = 0.12065
arm_length = 0.8128
string_length = None
release_angle = None

# Parameter bounds: [mass, pulley_radius, arm_length, string_length, release_angle]
_bounds = [
    (8.0, 18.0),                     # counterweight mass (kg)
    (0.01, 1.0),                     # pulley radius (m)
    (0.1, 2.5),                      # arm length (m)
    (0.1, 2.5),                      # string length (m)
    (-np.pi*1.75, -np.pi*1.15),       # release angle (rad)
]

# Parameter names for reference
_param_names = [
    'counter_weight_mass',
    'pulley_radius',
    'arm_length',
    'string_length',
    'release_angle'
]

# Global parameter locking variables
_locked_params = {}  # Dict mapping parameter index to locked value
_optimization_bounds = []  # Bounds for only unlocked parameters
_param_mapping = []  # Maps optimization vector indices to full parameter indices

def _setup_parameter_locking(locked_params):
    """Setup global variables for parameter locking"""
    global _locked_params, _optimization_bounds, _param_mapping

    _locked_params = locked_params.copy()
    _optimization_bounds = []
    _param_mapping = []

    # Create bounds and mapping for only unlocked parameters
    for i in range(len(_bounds)):
        if i not in _locked_params:
            _optimization_bounds.append(_bounds[i])
            _param_mapping.append(i)

    print(f"Locked parameters: {len(_locked_params)}")
    for param_idx, value in _locked_params.items():
        print(f"  {_param_names[param_idx]}: {value}")
    print(f"Optimizing over {len(_param_mapping)} parameters: {[_param_names[i] for i in _param_mapping]}")

def _reconstruct_full_params(optimization_vector):
    """Reconstruct full parameter vector from optimization vector and locked params"""
    full_params = [0.0] * len(_bounds)

    # Fill in locked parameters
    for param_idx, value in _locked_params.items():
        full_params[param_idx] = value

    # Fill in optimization parameters
    for opt_idx, param_idx in enumerate(_param_mapping):
        full_params[param_idx] = optimization_vector[opt_idx]

    return full_params

def _objective_function(optimization_vector):
    """Multi-objective optimization function (global for multiprocessing)"""
    # Safety check for parameter mapping
    if len(_param_mapping) != len(optimization_vector):
        return 1e6

    # Reconstruct full parameter vector
    param_vector = _reconstruct_full_params(optimization_vector)

    # Parameter bounds check (only check unlocked parameters)
    for i, val in enumerate(optimization_vector):
        if i >= len(_param_mapping):
            return 1e6
        param_idx = _param_mapping[i]
        min_val, max_val = _bounds[param_idx]
        if not (min_val <= val <= max_val):
            return 1e6

    # Create parameters
    params = TrebuchetParams(*param_vector)

    # String length constraint: <= 0.95 * arm_length
    if params.string_length > 0.95 * params.arm_length:
        return 1e6

    try:
        result = simulate_trebuchet(params)

        if result.distance <= 0 or result.efficiency <= 0:
            return 1e6

        # Cost components
        efficiency_cost = -result.efficiency * 100  # Negative to maximize
        distance_error = abs(result.distance - _target_distance) / _target_distance
        distance_cost = distance_error * 100
        mass_cost = (params.total_mass / 30.0) * 100

        # Weighted combination
        total_cost = (_efficiency_weight * efficiency_cost +
                     _distance_weight * distance_cost +
                     _mass_weight * mass_cost)

        return total_cost

    except Exception:
        return 1e6

def _ask_yes_no(question: str) -> bool:
    """Ask user a yes/no question and return True for yes, False for no"""
    while True:
        try:
            response = input(f"{question} (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            else:
                print("Please enter 'y' for yes or 'n' for no.")
        except (EOFError, KeyboardInterrupt):
            print("\nOperation cancelled.")
            return False

def _save_energy_plot(result, params):
    """Save the energy plot as an image file"""
    try:
        # Generate a descriptive filename
        filename = f"energy_plot_mass{params.counter_weight_mass:.0f}kg_range{result.distance:.0f}m.png"

        # Create the plot again but save instead of show
        if not result.energy_history:
            print("No energy history available to save.")
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

        # Create figure with two subplots (same as in animation.py)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[1, 1])
        fig.suptitle('Trebuchet Energy Components Over Time', fontsize=16, fontweight='bold')

        # Top plot: Total Energy and Counterweight PE
        ax1.plot(times, total_energies, 'k-', label='Total Energy', linewidth=3, alpha=0.9)
        ax1.plot(times, cw_pe, 'g--', label='Counterweight PE', linewidth=2.5, alpha=0.8)
        ax1.set_ylabel('Energy (J)', fontsize=12)
        ax1.set_title('Total Energy and Counterweight Potential Energy', fontsize=14)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Bottom plot: All other energy components
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

        # Mark release point if it occurred
        if result.metrics.get('release_occurred', True) and 't_release' in result.metrics:
            release_time = result.metrics['t_release']
            ax1.axvline(x=release_time, color='orange', linestyle=':', linewidth=2, alpha=0.8)
            ax2.axvline(x=release_time, color='orange', linestyle=':', linewidth=2, alpha=0.8)

            # Add annotations
            max_energy_top = max(total_energies)
            ax1.annotate(f'Release\n(t={release_time:.3f}s)',
                        xy=(release_time, max_energy_top * 0.9),
                        xytext=(release_time + 0.1, max_energy_top * 0.9),
                        arrowprops=dict(arrowstyle='->', color='orange', alpha=0.8),
                        fontsize=10, color='orange', fontweight='bold')

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        print(f"Energy plot saved as: {filename}")

    except Exception as e:
        print(f"Failed to save energy plot: {e}")

def optimize_trebuchet(show_animation: bool = True, locked_params: dict = None) -> TrebuchetParams:
    """
    Optimize trebuchet parameters for maximum efficiency and target distance

    Args:
        show_animation: Whether to show animation of optimal design
        locked_params: Dict of parameter names to locked values, e.g.:
                      {'counter_weight_mass': 50.0, 'arm_length': 2.0}
                      Available parameters: counter_weight_mass, pulley_radius,
                      arm_length, string_length, release_angle

    Returns:
        Optimized TrebuchetParams object
    """

    # Set global variables for multiprocessing
    global _target_distance, _efficiency_weight, _distance_weight, _mass_weight, _bounds

    # Setup parameter locking - use global variables if no locked_params provided
    if locked_params is None:
        # Use global parameter locking variables
        global counter_weight_mass, pulley_radius, arm_length, string_length, release_angle
        locked_params = {}

        if counter_weight_mass is not None:
            locked_params['counter_weight_mass'] = counter_weight_mass
        if pulley_radius is not None:
            locked_params['pulley_radius'] = pulley_radius
        if arm_length is not None:
            locked_params['arm_length'] = arm_length
        if string_length is not None:
            locked_params['string_length'] = string_length
        if release_angle is not None:
            locked_params['release_angle'] = release_angle

    # Convert parameter names to indices
    locked_param_indices = {}
    for param_name, value in locked_params.items():
        if param_name not in _param_names:
            raise ValueError(f"Unknown parameter: {param_name}. Available: {_param_names}")
        param_idx = _param_names.index(param_name)
        locked_param_indices[param_idx] = value

    _setup_parameter_locking(locked_param_indices)

    print(f"Optimizing trebuchet for {_target_distance}m target distance...")
    print(f"Weights - Efficiency: {_efficiency_weight}, Distance: {_distance_weight}, Mass: {_mass_weight}")

    # Check if we have any parameters to optimize
    if not _optimization_bounds:
        print("All parameters are locked! Cannot optimize.")
        # Return trebuchet with locked parameters
        full_params = [0.0] * len(_bounds)
        for param_idx, value in _locked_params.items():
            full_params[param_idx] = value
        return TrebuchetParams(*full_params)

    # Run optimization (disable multiprocessing for parameter locking to avoid global variable issues)
    workers_to_use = 1 if _locked_params else _workers
    result = differential_evolution(
        _objective_function,
        _optimization_bounds,  # Use optimization bounds instead of full bounds
        seed=_seed,
        maxiter=_maxiter,
        popsize=_popsize,
        atol=_atol,
        workers=workers_to_use,
        disp=True
    )

    # Get optimal parameters
    optimal_param_vector = _reconstruct_full_params(result.x)
    optimal_params = TrebuchetParams(*optimal_param_vector)
    sim_result = simulate_trebuchet(optimal_params, track_energy=True)

    # Print results
    print(f"\n" + "="*60)
    print(f"OPTIMIZATION RESULTS")
    print(f"="*60)

    print(f"\nOptimization Status:")
    print(f"  Success: {result.success}")
    print(f"  Function evaluations: {result.nfev}")
    print(f"  Final cost: {result.fun:.6f}")

    print(f"\nOptimal Parameters:")
    print(f"  Counterweight mass: {optimal_params.counter_weight_mass:.3f} kg")
    print(f"  Pulley radius: {optimal_params.pulley_radius:.3f} m")
    print(f"  Arm length: {optimal_params.arm_length:.3f} m")
    print(f"  String length: {optimal_params.string_length:.3f} m")
    print(f"  Release angle: {optimal_params.release_angle:.3f} rad ({optimal_params.release_angle*180/np.pi:.1f}°)")
    print(f"  String/Arm ratio: {optimal_params.string_arm_ratio:.3f}")
    print(f"  Total mass: {optimal_params.total_mass:.1f} kg")

    print(f"\nPerformance:")
    print(f"  Range: {sim_result.distance:.2f} m (target: {_target_distance:.0f} m)")
    print(f"  Range error: {abs(sim_result.distance - _target_distance):.2f} m ({abs(sim_result.distance - _target_distance)/_target_distance*100:.1f}%)")
    print(f"  Efficiency: {sim_result.efficiency:.4f} ({sim_result.efficiency*100:.2f}%)")

    if sim_result.metrics.get('release_occurred', True):
        print(f"  Release velocity: {sim_result.metrics['release_velocity']:.2f} m/s")
        print(f"  Release time: {sim_result.metrics['t_release']:.3f} s")

        print(f"\nEnergy Analysis:")
        print(f"  Projectile KE: {sim_result.metrics['ke_projectile']:.1f} J")
        print(f"  Total PE spent: {sim_result.metrics['total_pe_spent']:.1f} J")
        print(f"  Energy efficiency: {sim_result.metrics['ke_projectile']/sim_result.metrics['total_pe_spent']*100:.1f}%")

        # Print initial and release energy if energy tracking was enabled
        if sim_result.energy_history:
            initial_energy = sim_result.energy_history[0]
            print(f"\nSystem Energy Tracking:")
            print(f"  Initial total energy: {initial_energy['total']:.1f} J")

            # Find energy at release time
            if 't_release' in sim_result.metrics:
                release_time = sim_result.metrics['t_release']
                # Find the energy entry closest to release time
                release_energy = None
                min_time_diff = float('inf')
                for energy_entry in sim_result.energy_history:
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

    print(f"\nCopy-Paste Parameters:")
    print(f"counter_weight_mass = {optimal_params.counter_weight_mass:.3f}")
    print(f"pulley_radius = {optimal_params.pulley_radius:.3f}")
    print(f"arm_length = {optimal_params.arm_length:.3f}")
    print(f"string_length = {optimal_params.string_length:.3f}")
    print(f"release_angle = {optimal_params.release_angle:.3f}")

    # Show energy plot
    if sim_result.energy_history:
        print("\nDisplaying energy components over time...")
        plot_energy_history(sim_result, optimal_params)

    # Show animation
    if show_animation:
        print("\nCreating animation of optimal design...")
        try:
            from animation import show_animation as display_animation
            anim = create_animation(optimal_params, sim_result, show_forces=False)
            display_animation(anim)

            # Ask to save both energy plot and animation after animation is done
            if sim_result.energy_history:
                save_energy = _ask_yes_no("Would you like to save the energy plot as an image?")
                save_animation = _ask_yes_no("Would you like to save the animation as a GIF?")

                if save_energy:
                    _save_energy_plot(sim_result, optimal_params)

                if save_animation:
                    # Generate descriptive filename
                    gif_filename = f"animation_mass{optimal_params.counter_weight_mass:.0f}kg_range{sim_result.distance:.0f}m.gif"
                    print(f"Saving animation as: {gif_filename}")
                    save_animation_gif(anim, gif_filename, fps=60)
            else:
                # Only ask about animation if no energy history
                if _ask_yes_no("Would you like to save the animation as a GIF?"):
                    # Generate descriptive filename
                    gif_filename = f"animation_mass{optimal_params.counter_weight_mass:.0f}kg_range{sim_result.distance:.0f}m.gif"
                    print(f"Saving animation as: {gif_filename}")
                    save_animation_gif(anim, gif_filename, fps=60)

        except Exception as e:
            print(f"Animation failed: {e}")

    return optimal_params

def optimize_with_locked_params(**kwargs) -> TrebuchetParams:
    """
    Convenience function to optimize with locked parameters using keyword arguments

    Example:
        optimize_with_locked_params(counter_weight_mass=50.0, arm_length=2.0)

    Available parameters:
        counter_weight_mass, pulley_radius, arm_length, string_length, release_angle
    """
    return optimize_trebuchet(locked_params=kwargs)

def main():
    """Main function with example usage"""
    print("=== TREBUCHET OPTIMIZATION ===")
    print("\nTo lock parameters, edit the global variables at the top of this file:")
    print("counter_weight_mass, pulley_radius, arm_length, string_length, release_angle")
    print("Set to None to optimize, or to a specific value to lock.\n")

    optimal_params = optimize_trebuchet()

if __name__ == "__main__":
    main()