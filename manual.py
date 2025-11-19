#!/usr/bin/env python3
"""
Manual Trebuchet Testing
Set parameters at top and run simulation with animation
"""

import numpy as np
import matplotlib.pyplot as plt
from simulation import TrebuchetParams, simulate_trebuchet, print_simulation_results
from animation import create_animation, show_animation, plot_energy_history, save_animation_gif

# EDIT THESE PARAMETERS
counter_weight_mass = 16.865
pulley_radius = 0.121
arm_length = 0.813
string_length = 0.669
release_angle = -4.877
show_forces = False  # Set to True to show force arrows

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

def main():
    """Run simulation with specified parameters"""
    # Create parameters object
    params = TrebuchetParams(
        counter_weight_mass=counter_weight_mass,
        pulley_radius=pulley_radius,
        arm_length=arm_length,
        string_length=string_length,
        release_angle=release_angle
    )

    # Run simulation with energy tracking enabled
    result = simulate_trebuchet(params, store_forces=show_forces, track_energy=True)

    # Print results
    print_simulation_results(params, result)

    # Show energy plot if available
    if result.energy_history:
        print("\nDisplaying energy components over time...")
        plot_energy_history(result, params)

    # Create and show animation
    if 'error' not in result.metrics:
        print("\nCreating animation...")
        anim = create_animation(params, result, show_forces=show_forces)
        show_animation(anim)

        # Ask to save both energy plot and animation after animation is done
        if result.energy_history:
            save_energy = _ask_yes_no("Would you like to save the energy plot as an image?")
            save_animation = _ask_yes_no("Would you like to save the animation as a GIF?")

            if save_energy:
                _save_energy_plot(result, params)

            if save_animation:
                # Generate descriptive filename
                gif_filename = f"animation_mass{params.counter_weight_mass:.0f}kg_range{result.distance:.0f}m.gif"
                print(f"Saving animation as: {gif_filename}")
                save_animation_gif(anim, gif_filename, fps=30)
        else:
            # Only ask about animation if no energy history
            if _ask_yes_no("Would you like to save the animation as a GIF?"):
                # Generate descriptive filename
                gif_filename = f"animation_mass{params.counter_weight_mass:.0f}kg_range{result.distance:.0f}m.gif"
                print(f"Saving animation as: {gif_filename}")
                save_animation_gif(anim, gif_filename, fps=30)

    else:
        print(f"ERROR: {result.metrics['error']}")

if __name__ == "__main__":
    main()