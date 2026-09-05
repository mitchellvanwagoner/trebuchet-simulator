"""Command-line interface for trebuchet simulation and optimization.

Examples:
    trebuchet simulate --arm-length 0.813 --animate
    trebuchet simulate --machine traditional --counterweight-mass 80
    trebuchet optimize --target-distance 30 --lock counter_weight_mass=14 --save-gif best.gif
    trebuchet optimize --machine traditional --target-distance 60 --lock length_counterweight=0.4

Every numeric default depends on the machine (see config.DEFAULT_MACHINE_PARAMS), so
the argument parser leaves them unset and cmd_simulate fills them in once --machine is
known. Angles are radians throughout, matching TrebuchetParams.
"""

import argparse
import math
import sys
from pathlib import Path

from trebuchet_sim.config import (
    DEFAULT_INITIAL_ARM_ANGLE,
    DEFAULT_MACHINE_FIXED,
    DEFAULT_MACHINE_PARAMS,
    LINKAGE_PARAM,
    MachineType,
    TrebuchetParams,
)
from trebuchet_sim.optimization import OptimizationConfig, optimize_trebuchet, param_names
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
    print(f"  Machine: {params.machine.value}")
    print(f"  Counterweight: {params.counter_weight_mass:.1f} kg")
    # The linkage parameter is the one that differs between the machines; printing the
    # other machine's would be reporting a number this run never used.
    if params.has_pulley:
        print(f"  Pulley radius: {params.pulley_radius:.3f} m")
    else:
        print(f"  CW arm length: {params.length_counterweight:.3f} m")
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
    """Parse NAME=VALUE. Which names are legal is checked later, in cmd_optimize.

    argparse runs this during parsing, before --machine has been read, and the legal
    names depend on the machine: the linkage parameter is pulley_radius on one and
    length_counterweight on the other.
    """
    name, _, value = spec.partition("=")
    if not value:
        raise argparse.ArgumentTypeError(f"Expected NAME=VALUE, got {spec!r}")
    return name, float(value)


def _machine_defaults(machine: MachineType) -> dict:
    """Every value a `simulate` run needs, defaulted for one machine.

    Design variables come from that machine's table; the fixed fields fall back to the
    TrebuchetParams defaults wherever the machine doesn't override them, so a value the
    two machines share is still written down only once.
    """
    fixed = DEFAULT_MACHINE_FIXED[machine]
    values = dict(DEFAULT_MACHINE_PARAMS[machine])
    values["pivot_height"] = fixed.get("pivot_height", TrebuchetParams.pivot_height)
    values["counter_weight_rope_length"] = fixed.get("counter_weight_rope_length")
    values["initial_arm_angle"] = DEFAULT_INITIAL_ARM_ANGLE[machine]
    return values


def _default_note(name: str, unit: str) -> str:
    """Help text spelling out both machines' defaults for one argument.

    The parser cannot carry a real default for these - it does not know the machine yet
    - so the help has to say what each machine will fall back to.
    """
    parts = []
    for machine in MachineType:
        value = _machine_defaults(machine).get(name)
        parts.append(f"{value:g} {machine.value}" if value is not None else f"auto ({machine.value})")
    return f"{unit} (default: {', '.join(parts)})"


def _add_machine_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--machine",
        choices=[m.value for m in MachineType],
        default=MachineType.PULLEY.value,
        help="Counterweight linkage: 'pulley' hangs the weight from a rope over the pivot "
             "axle, 'traditional' bolts it to the arm's short end (default: pulley)",
    )


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


def _resolve_linkage(machine: MachineType, args: argparse.Namespace) -> float:
    """The linkage argument that belongs to this machine, defaulted if not given.

    Both flags always exist, so passing the other machine's is a plain mistake worth
    naming rather than silently ignoring - it would otherwise look like the value had
    been applied.
    """
    wanted = LINKAGE_PARAM[machine]
    unused = "length_counterweight" if wanted == "pulley_radius" else "pulley_radius"
    if getattr(args, unused) is not None:
        raise SystemExit(
            f"--{unused.replace('_', '-')} does not apply to the {machine.value} machine; "
            f"use --{wanted.replace('_', '-')}."
        )
    given = getattr(args, wanted)
    return _machine_defaults(machine)[wanted] if given is None else given


def cmd_simulate(args: argparse.Namespace) -> int:
    machine = MachineType(args.machine)
    defaults = _machine_defaults(machine)

    def value(name):
        """A command-line value, or this machine's default when the flag was omitted."""
        given = getattr(args, name)
        return defaults[name] if given is None else given

    params = TrebuchetParams(
        machine=machine,
        counter_weight_mass=value("counter_weight_mass"),
        arm_length=value("arm_length"),
        string_length=value("string_length"),
        release_angle=value("release_angle"),
        pivot_height=value("pivot_height"),
        initial_arm_angle=value("initial_arm_angle"),
        counter_weight_rope_length=value("counter_weight_rope_length"),
        **{LINKAGE_PARAM[machine]: _resolve_linkage(machine, args)},
    )
    result = simulate_trebuchet(params, track_energy=True, simulate_aftermath=True)
    print_simulation_results(params, result)

    if "error" in result.metrics:
        return 1

    _handle_outputs(args, params, result)
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    machine = MachineType(args.machine)
    locked = dict(args.lock or [])
    names = param_names(machine)
    unknown = set(locked) - set(names)
    if unknown:
        # Caught here rather than in _parse_lock, which runs before --machine is known.
        raise SystemExit(
            f"Cannot lock {', '.join(sorted(unknown))} on the {machine.value} machine. "
            f"Available: {', '.join(names)}"
        )

    # The machine's own fixed geometry, so `optimize --machine traditional` starts from a
    # buildable machine instead of the pulley machine's 1 m pivot.
    defaults = _machine_defaults(machine)
    fixed = {
        "pivot_height": defaults["pivot_height"],
        "initial_arm_angle": defaults["initial_arm_angle"],
    }
    if defaults["counter_weight_rope_length"] is not None:
        fixed["counter_weight_rope_length"] = defaults["counter_weight_rope_length"]

    config = OptimizationConfig(
        machine=machine,
        target_distance=args.target_distance,
        efficiency_weight=args.efficiency_weight,
        distance_weight=args.distance_weight,
        mass_weight=args.mass_weight,
        locked_params=locked,
        fixed_params=fixed,
        display_progress=True,
    )

    print(f"Optimizing a {machine.value} machine for {config.target_distance:.0f}m target distance...")
    print(f"Free parameters: {', '.join(config.free_params)}")

    optimal_params, sim_result, de_result = optimize_trebuchet(config)

    print(f"\nOptimization {'succeeded' if de_result.success else 'FAILED'} after {de_result.nfev} evaluations")
    print_simulation_results(optimal_params, sim_result)

    print("\nCopy-paste parameters:")
    for name in names:
        print(f"  {name} = {getattr(optimal_params, name):.3f}")

    _handle_outputs(args, optimal_params, sim_result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trebuchet", description="Trebuchet physics simulation toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sim_parser = subparsers.add_parser("simulate", help="Run a single simulation with fixed parameters")
    _add_machine_arg(sim_parser)
    # Defaults stay None so cmd_simulate can fill in the selected machine's values; see
    # the module docstring.
    sim_parser.add_argument(
        "--counterweight-mass", dest="counter_weight_mass", type=float,
        help=_default_note("counter_weight_mass", "kg"),
    )
    sim_parser.add_argument(
        "--pulley-radius", type=float,
        help="m, pulley machine only - how far the weight falls per radian of arm "
             f"rotation (default: {DEFAULT_MACHINE_PARAMS[MachineType.PULLEY]['pulley_radius']:g})",
    )
    sim_parser.add_argument(
        "--length-counterweight", type=float,
        help="m, traditional machine only - how far the weight sits behind the pivot "
             f"(default: {DEFAULT_MACHINE_PARAMS[MachineType.TRADITIONAL]['length_counterweight']:g})",
    )
    sim_parser.add_argument("--arm-length", type=float, help=_default_note("arm_length", "m"))
    sim_parser.add_argument("--string-length", type=float, help=_default_note("string_length", "m"))
    sim_parser.add_argument("--release-angle", type=float, help=_default_note("release_angle", "radians"))
    sim_parser.add_argument("--pivot-height", type=float, help=_default_note("pivot_height", "m"))
    sim_parser.add_argument(
        "--initial-arm-angle", type=float, help=_default_note("initial_arm_angle", "radians"),
    )
    sim_parser.add_argument(
        "--counter-weight-rope-length", type=float,
        help=_default_note("counter_weight_rope_length", "m") + "; auto = twice the pulley radius",
    )
    _add_output_args(sim_parser)
    sim_parser.set_defaults(func=cmd_simulate)

    opt_parser = subparsers.add_parser("optimize", help="Search for optimal parameters via differential evolution")
    _add_machine_arg(opt_parser)
    opt_parser.add_argument("--target-distance", type=float, default=30.0, help="m (default: 30.0)")
    opt_parser.add_argument("--efficiency-weight", type=float, default=5.0)
    opt_parser.add_argument("--distance-weight", type=float, default=1.0)
    opt_parser.add_argument("--mass-weight", type=float, default=0.15)
    opt_parser.add_argument(
        "--lock",
        type=_parse_lock,
        action="append",
        metavar="NAME=VALUE",
        help="Lock a parameter to a fixed value. Repeatable. Available: "
             + ", ".join(sorted(set(param_names(MachineType.PULLEY)) | set(param_names(MachineType.TRADITIONAL))))
             + " (the linkage parameter depends on --machine)",
    )
    _add_output_args(opt_parser)
    opt_parser.set_defaults(func=cmd_optimize)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
