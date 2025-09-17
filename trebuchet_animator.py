#!/usr/bin/env python3
"""
Trebuchet Animation Script - Standalone visualization
"""

import numpy as np
from trebuchet_simulator import TrebuchetParams, simulate_trebuchet, create_visualization, print_simulation_results

def animate_trebuchet(params):
    """Run simulation and show animation"""
    print("Running simulation...")
    distance, efficiency, metrics = simulate_trebuchet(params)

    # Print results first
    print_simulation_results(params, distance, efficiency, metrics)

    # Check for errors
    if 'error' in metrics:
        print(f"\nERROR: {metrics['error']}")
        print("Cannot create animation - simulation failed")
        return

    print("\nCreating animation...")
    try:
        anim = create_visualization(params)
        print("Animation complete!")
        return anim
    except Exception as e:
        print(f"Animation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Demo animation with default parameters"""
    print("=== TREBUCHET ANIMATION DEMO ===")

    # Demo parameters (optimized from previous runs)
    params = TrebuchetParams(
        counter_weight_mass=1.999,
        pulley_radius=0.103,
        arm_length=0.781,
        string_length=0.742,
        release_angle=-4.472
    )

    print("Demo Parameters:")
    print(f"  Counterweight: {params.counter_weight_mass} kg")
    print(f"  Pulley radius: {params.pulley_radius} m")
    print(f"  Arm length: {params.arm_length} m")
    print(f"  String length: {params.string_length} m")
    print(f"  Release angle: {params.release_angle:.3f} rad ({params.release_angle*180/np.pi:.1f}°)")

    # Run animation
    animate_trebuchet(params)

if __name__ == "__main__":
    main()