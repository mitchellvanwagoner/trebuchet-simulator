#!/usr/bin/env python3
"""
Trebuchet Optimizer - Finds optimal parameters for range and efficiency
"""

import numpy as np
from scipy.optimize import differential_evolution
from trebuchet_simulator import TrebuchetParams, simulate_trebuchet, create_visualization

# Target distance
TARGET_DISTANCE = 30.0  # meters

# OPTIMIZATION TUNING PARAMETERS
EFFICIENCY_WEIGHT = 1.00         # Primary: maximize energy transfer efficiency
DISTANCE_WEIGHT = 1.0           # Secondary: hit target distance
MASS_WEIGHT = 1.0               # Tertiary: minimize total mass

# Optimization settings
SEED = 8888
MAX_ITERATIONS = 1000
POPULATION_SIZE = 40
TOLERANCE = 0.001
WORKERS = 16

# Parameter bounds: [mass, pulley_radius, arm_length, string_length, release_angle]
PARAM_BOUNDS = [
    (1.0, 50.0),                     # counterweight mass (kg)
    (0.1, 1.0),                      # pulley radius (m)
    (0.1, 2.5),                      # arm length (m)
    (0.1, 2.5),                      # string length (m)
    (-np.pi*1.65, -np.pi*1.25)       # release angle (rad)
]


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
        efficiency_cost = - efficiency  # Negative to maximize

        # 2. Distance component
        # Normalize: error relative to target distance
        distance_error = abs(distance - TARGET_DISTANCE) / TARGET_DISTANCE
        distance_cost = (distance_error*100)

        # 3. Mass component (tertiary objective - minimize weight)
        # Normalize: typical mass range 20-100kg, target ~50kg
        mass_normalized = params.total_mass / 40.0
        mass_cost = mass_normalized*100

        # Combine all components with tunable weights
        total_cost = (EFFICIENCY_WEIGHT * efficiency_cost +
                     DISTANCE_WEIGHT * distance_cost +
                     MASS_WEIGHT * mass_cost )

        return total_cost

    except Exception as e:
        print(f"Optimization failed: {e}")
        return 1e6

def optimize_trebuchet():
    """Optimize trebuchet for efficiency while targeting range"""
    print(f"Optimizing for {TARGET_DISTANCE}m range with efficiency priority")
    print(f"Optimization settings:")
    print(f"  Max iterations: {MAX_ITERATIONS}")
    print(f"  Population size: {POPULATION_SIZE}")
    print(f"  Workers: {WORKERS}")
    print(f"  Seed: {SEED}")
    print(f"\nObjective weights:")
    print(f"  Efficiency: {EFFICIENCY_WEIGHT}")
    print(f"  Distance: {DISTANCE_WEIGHT}")
    print(f"  Mass: {MASS_WEIGHT}")

    # Optimize using differential evolution
    result = differential_evolution(
        efficiency_objective,
        PARAM_BOUNDS,
        seed=SEED,
        maxiter=MAX_ITERATIONS,
        popsize=POPULATION_SIZE,
        atol=TOLERANCE,
        workers=WORKERS,
        disp=True
    )

    # Extract results
    optimal_params = TrebuchetParams(*result.x)
    distance, efficiency, metrics = simulate_trebuchet(optimal_params)

    return result, optimal_params, distance, efficiency, metrics

def print_optimization_results(result, params, distance, efficiency, metrics):
    """Print detailed optimization results"""
    print(f"\n" + "="*60)
    print(f"OPTIMIZATION RESULTS")
    print(f"="*60)

    print(f"\nOptimization Status:")
    print(f"  Success: {result.success}")
    print(f"  Function evaluations: {result.nfev}")
    print(f"  Final objective value: {result.fun:.6f}")
    if hasattr(result, 'message'):
        print(f"  Message: {result.message}")

    print(f"\nOPTIMAL PARAMETERS:")
    print(f"  Counterweight mass: {params.counter_weight_mass:.3f} kg")
    print(f"  Pulley radius: {params.pulley_radius:.3f} m")
    print(f"  Arm length: {params.arm_length:.3f} m")
    print(f"  String length: {params.string_length:.3f} m")
    print(f"  Release angle: {params.release_angle:.3f} rad ({params.release_angle*180/np.pi:.1f}°)")

    print(f"\nDERIVED PROPERTIES:")
    print(f"  String/Arm ratio: {params.string_arm_ratio:.3f}")
    print(f"  Arm mass: {params.arm_mass:.1f} kg")
    print(f"  Pulley mass: {params.pulley_mass:.1f} kg")
    print(f"  Total mass: {params.total_mass:.1f} kg")

    print(f"\nPERFORMANCE RESULTS:")
    print(f"  Range: {distance:.2f} m (target: {TARGET_DISTANCE:.0f} m)")
    print(f"  Range error: {abs(distance - TARGET_DISTANCE):.2f} m ({abs(distance - TARGET_DISTANCE)/TARGET_DISTANCE*100:.1f}%)")
    print(f"  Efficiency: {efficiency:.4f} ({efficiency*100:.2f}%)")
    print(f"  Release velocity: {metrics['release_velocity']:.2f} m/s")
    print(f"  Release height: {metrics['release_height']:.2f} m")
    print(f"  Release time: {metrics['t_release']:.3f} s")
    print(f"  Arm rotation: {metrics['arm_rotation_deg']:.1f}°")

    print(f"\nENERGY ANALYSIS:")
    print(f"  Projectile KE: {metrics['ke_projectile']:.1f} J")
    print(f"  Counterweight PE: {metrics['pe_spent']:.1f} J")
    print(f"  Arm PE: {metrics['arm_pe_spent']:.1f} J")
    print(f"  Projectile PE: {metrics['projectile_pe_spent']:.1f} J")
    print(f"  Total PE spent: {metrics['total_pe_spent']:.1f} J")
    print(f"  Energy transfer efficiency: {metrics['ke_projectile']/metrics['total_pe_spent']*100:.1f}%")

    print(f"\nCOPY-PASTE PARAMETERS FOR MANUAL TESTING:")
    print(f"counter_weight_mass = {params.counter_weight_mass:.3f}")
    print(f"pulley_radius = {params.pulley_radius:.3f}")
    print(f"arm_length = {params.arm_length:.3f}")
    print(f"string_length = {params.string_length:.3f}")
    print(f"release_angle = {params.release_angle:.3f}  # {params.release_angle*180/np.pi:.1f}°")

def animate_optimal_design(params):
    """Show animation of the optimal trebuchet design"""
    print("\nCreating animation of optimal design...")
    try:
        anim = create_visualization(params)
        return anim
    except Exception as e:
        print(f"Animation failed: {e}")
        return None

def main():
    """Main optimization execution"""
    try:
        # Run optimization
        result, params, distance, efficiency, metrics = optimize_trebuchet()

        # Print detailed results
        print_optimization_results(result, params, distance, efficiency, metrics)

        animate_optimal_design(params)

    except Exception as e:
        print(f"Optimization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()