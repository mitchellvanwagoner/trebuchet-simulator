"""Command-line interface for trebuchet simulation and optimization.

Examples:
    trebuchet simulate --arm-length 0.813 --animate
    trebuchet optimize --target-distance 30 --lock counter_weight_mass=14 --save-gif best.gif
"""

import argparse
import math
import sys
from pathlib import Path

from trebuchet_sim.config import DEFAULT_OPTIMIZABLE_PARAMS, TrebuchetParams
from trebuchet_sim.optimization import PARAM_NAMES, OptimizationConfig, optimize_trebuchet
from trebuchet_sim.physics import SimulationResult, simulate_trebuchet
from trebuchet_sim.visualization import (
    create_animation,
    plot_energy_history,
    save_animation_gif,
    save_energy_plot,
    show_animation,
)

OUTPUT_DIR = Path("outputs")


def print_simulation_results(params: TrebuchetParams, result: SimulationResult) -> None:
    """Print a summary of a simulation run to stdout."""
    print("\n=== TREBUCHET SIMULATION RESULTS ===")
    print("Parameters:")
    print(f"  Counterweight: {params.counter_weight_mass:.1f} kg")
    print(f"  Pulley radius: {params.pulley_radius:.3f} m")
    print(f"  Arm length: {params.arm_length:.3f} m")
    print(f"  String length: {params.string_length:.3f} m")
    print(f"  Release angle: {math.degrees(params.release_angle):.1f} deg")
    print(f"  String/Arm ratio: {params.string_arm_ratio:.3f}")
    print(f"  Total mass: {params.total_mass:.1f} kg")

    if "error" in result.metrics:
        print(f"\nERROR: {result.metrics['error']}")
        return

    if not result.metrics.get("release_occurred", True):
        print("\nResults:")
        print("  **NO RELEASE OCCURRED** - simulation ran for the full duration")
        print(f"  Simulation time: {result.metrics['simulation_time']:.1f} s")
        print(f"  Final arm angle: {result.metrics['final_arm_angle_deg']:.1f} deg")
        print(f"  Total arm rotation: {result.metrics['total_rotation_deg']:.1f} deg")
        return

    print("\nResults:")
    print(f"  Range: {result.distance:.2f} m")
    print(f"  Efficiency: {result.efficiency:.4f} ({result.efficiency * 100:.2f}%)")
    print(f"  Release velocity: {result.metrics['release_velocity']:.2f} m/s")
    print(f"  Release height: {result.metrics['release_height']:.2f} m")
    print(f"  Release time: {result.metrics['t_release']:.3f} s")
    print(f"  Arm rotation: {result.metrics['arm_rotation_deg']:.1f} deg")

    print("\nEnergy Analysis:")
    print(f"  Projectile KE: {result.metrics['ke_projectile']:.1f} J")
    print(f"  Counterweight PE: {result.metrics['pe_spent']:.1f} J")
    print(f"  Total PE spent: {result.metrics['total_pe_spent']:.1f} J")

    if "min_string_tension" in result.metrics:
        print(f"  Min sling tension: {result.metrics['min_string_tension']:.1f} N")
        snap_energy = result.metrics.get("sling_snap_energy", 0.0)
        if result.metrics.get("string_slack_fraction", 0.0) > 1e-3 or snap_energy > 1e-3:
            print(
                f"  [WARNING] Sling goes slack for {result.metrics['string_slack_fraction'] * 100:.0f}% "
                f"of the launch and snaps taut {result.metrics.get('sling_snap_count', 0)} time(s), "
                f"dissipating {snap_energy:.1f} J."
            )
        # 0.05 N*s is the integration-noise floor for the rigid-link counterweight rope.
        if result.metrics.get("cw_rope_compression_impulse", 0.0) > 0.05:
            print(
                f"  [WARNING] Counterweight rope goes slack (min tension "
                f"{result.metrics['min_cw_rope_tension']:.1f} N) - results are not physical."
            )

    if "energy_analysis" in result.metrics:
        analysis = result.metrics["energy_analysis"]
        print("\nEnergy Conservation:")
        print(f"  Overall trend: {analysis['overall_trend']}")
        print(f"  Violations (>5% step change): {len(analysis['violations'])}")
        if analysis["energy_increases"]:
            print(f"  [WARNING] {len(analysis['energy_increases'])} energy increases detected - check physics!")


def _parse_lock(spec: str) -> tuple:
    name, _, value = spec.partition("=")
    if not value:
        raise argparse.ArgumentTypeError(f"Expected NAME=VALUE, got {spec!r}")
    if name not in PARAM_NAMES:
        raise argparse.ArgumentTypeError(f"Unknown parameter {name!r}. Available: {', '.join(PARAM_NAMES)}")
    return name, float(value)


def _add_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--animate", action="store_true", help="Show an interactive animation window")
    parser.add_argument("--energy-plot", action="store_true", help="Show an interactive energy plot window")
    parser.add_argument("--save-gif", metavar="FILE", help="Save the animation as a GIF under outputs/")
    parser.add_argument("--save-energy-plot", metavar="FILE", help="Save the energy plot as a PNG under outputs/")
    parser.add_argument("--gif-fps", type=int, default=30, help="Frames per second for saved GIFs (default: 30)")


def _handle_outputs(args: argparse.Namespace, params: TrebuchetParams, result: SimulationResult) -> None:
    if args.energy_plot or args.save_energy_plot:
        if not result.energy_history:
            print("No energy history recorded; nothing to plot.")
        else:
            if args.save_energy_plot:
                OUTPUT_DIR.mkdir(exist_ok=True)
                save_energy_plot(result, str(OUTPUT_DIR / args.save_energy_plot))
            if args.energy_plot:
                plot_energy_history(result)

    if args.animate or args.save_gif:
        anim = create_animation(params, result)
        if args.save_gif:
            OUTPUT_DIR.mkdir(exist_ok=True)
            save_animation_gif(anim, str(OUTPUT_DIR / args.save_gif), fps=args.gif_fps)
        if args.animate:
            show_animation(anim)


def cmd_simulate(args: argparse.Namespace) -> int:
    params = TrebuchetParams(
        counter_weight_mass=args.counterweight_mass,
        pulley_radius=args.pulley_radius,
        arm_length=args.arm_length,
        string_length=args.string_length,
        release_angle=args.release_angle,
    )
    result = simulate_trebuchet(params, track_energy=True, simulate_aftermath=True)
    print_simulation_results(params, result)

    if "error" in result.metrics:
        return 1

    _handle_outputs(args, params, result)
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    config = OptimizationConfig(
        target_distance=args.target_distance,
        efficiency_weight=args.efficiency_weight,
        distance_weight=args.distance_weight,
        mass_weight=args.mass_weight,
        locked_params=dict(args.lock or []),
        display_progress=True,
    )

    print(f"Optimizing for {config.target_distance:.0f}m target distance...")
    print(f"Free parameters: {', '.join(config.free_params)}")

    optimal_params, sim_result, de_result = optimize_trebuchet(config)

    print(f"\nOptimization {'succeeded' if de_result.success else 'FAILED'} after {de_result.nfev} evaluations")
    print_simulation_results(optimal_params, sim_result)

    print("\nCopy-paste parameters:")
    for name in PARAM_NAMES:
        print(f"  {name} = {getattr(optimal_params, name):.3f}")

    _handle_outputs(args, optimal_params, sim_result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trebuchet", description="Trebuchet physics simulation toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    defaults = DEFAULT_OPTIMIZABLE_PARAMS
    sim_parser = subparsers.add_parser("simulate", help="Run a single simulation with fixed parameters")
    sim_parser.add_argument(
        "--counterweight-mass", type=float, default=defaults["counter_weight_mass"],
        help=f"kg (default: {defaults['counter_weight_mass']})",
    )
    sim_parser.add_argument(
        "--pulley-radius", type=float, default=defaults["pulley_radius"],
        help=f"m (default: {defaults['pulley_radius']})",
    )
    sim_parser.add_argument(
        "--arm-length", type=float, default=defaults["arm_length"],
        help=f"m (default: {defaults['arm_length']})",
    )
    sim_parser.add_argument(
        "--string-length", type=float, default=defaults["string_length"],
        help=f"m (default: {defaults['string_length']})",
    )
    sim_parser.add_argument(
        "--release-angle", type=float, default=defaults["release_angle"],
        help=f"radians (default: {defaults['release_angle']})",
    )
    _add_output_args(sim_parser)
    sim_parser.set_defaults(func=cmd_simulate)

    opt_parser = subparsers.add_parser("optimize", help="Search for optimal parameters via differential evolution")
    opt_parser.add_argument("--target-distance", type=float, default=30.0, help="m (default: 30.0)")
    opt_parser.add_argument("--efficiency-weight", type=float, default=5.0)
    opt_parser.add_argument("--distance-weight", type=float, default=1.0)
    opt_parser.add_argument("--mass-weight", type=float, default=0.15)
    opt_parser.add_argument(
        "--lock",
        type=_parse_lock,
        action="append",
        metavar="NAME=VALUE",
        help=f"Lock a parameter to a fixed value. Repeatable. Available: {', '.join(PARAM_NAMES)}",
    )
    _add_output_args(opt_parser)
    opt_parser.set_defaults(func=cmd_optimize)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
