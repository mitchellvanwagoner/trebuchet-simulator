#!/usr/bin/env python3
"""
Manual Trebuchet Testing - Set parameters manually and test
"""

import numpy as np
from trebuchet_simulator import TrebuchetParams, simulate_trebuchet, create_visualization, print_simulation_results

def animate_trebuchet(params):
    """Run simulation and show animation with error handling"""
    print("\nRunning simulation...")
    distance, efficiency, metrics = simulate_trebuchet(params)

    # Print results first
    print_simulation_results(params, distance, efficiency, metrics)

    # Check for errors
    if 'error' in metrics:
        print(f"\nERROR: {metrics['error']}")
        print("Cannot create animation - simulation failed")
        return None

    print("\nCreating animation...")
    try:
        anim = create_visualization(params)
        print("Animation complete! Close the window to continue.")
        return anim
    except Exception as e:
        print(f"Animation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_trebuchet():
    """Test trebuchet with manually set parameters"""

    # MANUAL PARAMETERS - Edit these values to test different configurations
    counter_weight_mass = 9.530
    pulley_radius = 0.0502
    arm_length = 0.811
    string_length = 0.74
    release_angle = -4.015  # -230.0°

    # Create parameters object
    params = TrebuchetParams(
        counter_weight_mass=counter_weight_mass,
        pulley_radius=pulley_radius,
        arm_length=arm_length,
        string_length=string_length,
        release_angle=release_angle
    )

    print("Testing trebuchet with manual parameters...")
    print(f"Parameters:")
    print(f"  Counterweight: {counter_weight_mass} kg")
    print(f"  Pulley radius: {pulley_radius} m")
    print(f"  Arm length: {arm_length} m")
    print(f"  String length: {string_length} m")
    print(f"  Release angle: {release_angle*180/np.pi:.1f}°")

    # Run simulation
    distance, efficiency, metrics = simulate_trebuchet(params)

    # Print detailed results
    print_simulation_results(params, distance, efficiency, metrics)

    # Check for errors
    if 'error' in metrics:
        print(f"\nERROR: {metrics['error']}")
        return

    animate_trebuchet(params)

def test_parameter_sweep():
    """Test a range of parameters to understand behavior"""
    print("Running parameter sweep...")

    base_params = {
        'counter_weight_mass': 20.0,
        'pulley_radius': 0.3,
        'arm_length': 1.5,
        'string_length': 1.0,
        'release_angle': -225 * np.pi / 180
    }

    print(f"\nBase parameters: {base_params}")
    print(f"\nRelease Angle Sweep:")
    print(f"{'Angle (deg)':<12} {'Range (m)':<12} {'Efficiency':<12} {'Status'}")
    print("-" * 50)

    # Test different release angles
    for angle_deg in [-90, -135, -180, -225, -270, -315]:
        test_params = TrebuchetParams(
            counter_weight_mass=base_params['counter_weight_mass'],
            pulley_radius=base_params['pulley_radius'],
            arm_length=base_params['arm_length'],
            string_length=base_params['string_length'],
            release_angle=angle_deg * np.pi / 180
        )

        distance, efficiency, metrics = simulate_trebuchet(test_params)

        if 'error' in metrics:
            status = metrics['error']
        else:
            status = "OK"

        print(f"{angle_deg:<12} {distance:<12.1f} {efficiency:<12.3f} {status}")

def test_realistic_trebuchet():
    """Test with realistic trebuchet parameters"""
    print("Testing realistic trebuchet design...")

    # Based on historical trebuchet proportions
    params = TrebuchetParams(
        counter_weight_mass=25.0,     # 25 kg counterweight
        pulley_radius=0.2,            # 20 cm pulley
        arm_length=2.0,               # 2 meter arm
        string_length=1.5,            # 1.5 meter string (0.75 ratio)
        release_angle=-225 * np.pi / 180  # Release at 225° (-45° from horizontal)
    )

    distance, efficiency, metrics = simulate_trebuchet(params)
    print_simulation_results(params, distance, efficiency, metrics)

    if 'error' not in metrics:
        response = input("\nShow animation? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            animate_trebuchet(params)

def animate_only():
    """Just run animation with current manual parameters"""
    # Use the same parameters as in test_trebuchet()
    counter_weight_mass = 1.999
    pulley_radius = 0.103
    arm_length = 0.781
    string_length = 0.742
    release_angle = -4.472

    params = TrebuchetParams(
        counter_weight_mass=counter_weight_mass,
        pulley_radius=pulley_radius,
        arm_length=arm_length,
        string_length=string_length,
        release_angle=release_angle
    )

    print("Running animation with current manual parameters...")
    animate_trebuchet(params)

def main():
    """Main menu for manual testing"""
    print("=== TREBUCHET MANUAL TESTING ===")
    print("1. Test with manual parameters (edit the code)")
    print("2. Run parameter sweep")
    print("3. Test realistic trebuchet")
    print("4. Animation only (current manual parameters)")
    print("5. Exit")

    while True:
        choice = input("\nEnter choice (1-5): ").strip()

        if choice == '1':
            test_trebuchet()
        elif choice == '2':
            test_parameter_sweep()
        elif choice == '3':
            test_realistic_trebuchet()
        elif choice == '4':
            animate_only()
        elif choice == '5':
            break
        else:
            print("Invalid choice. Please enter 1-5.")

    print("Goodbye!")

if __name__ == "__main__":
    test_trebuchet()