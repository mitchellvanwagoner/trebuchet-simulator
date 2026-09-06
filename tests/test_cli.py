"""The command-line surface itself: does the parser build, and does it still describe
the same defaults the rest of the code runs on?

Nothing here simulates anything - physics coverage lives in the other modules. These
guard the parts of the CLI that are only exercised by actually invoking it, which no
other test does.
"""

import argparse

import pytest

from trebuchet_sim.cli import build_parser
from trebuchet_sim.config import MachineType
from trebuchet_sim.optimization import OptimizationConfig


def test_parser_builds_and_every_help_string_renders():
    """argparse %-formats help text, so a bare "%" in one is a crash, not a typo.

    It is raised while *building* the parser, which is the first thing main() does - so
    a single stray percent sign takes down every invocation of every subcommand,
    including `simulate` and `--help`. Formatting the help is what forces the expansion.
    """
    parser = build_parser()
    parser.format_help()

    subparsers = [
        action for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    ]
    assert subparsers, "the CLI is expected to have subcommands"
    names = sorted(subparsers[0].choices)
    assert names == ["optimize", "simulate"]
    for name in names:
        subparsers[0].choices[name].format_help()


@pytest.mark.parametrize(
    "flag,field",
    [
        ("--target-distance", "target_distance"),
        ("--efficiency-weight", "efficiency_weight"),
        ("--distance-weight", "distance_weight"),
        ("--mass-weight", "mass_weight"),
        ("--snap-penalty-weight", "snap_penalty_weight"),
    ],
)
def test_objective_flags_default_to_the_dataclass(flag, field):
    """The CLI must not carry its own copy of a weight.

    Three places choose these numbers - the dataclass, this parser and the dashboard -
    and only the dataclass should decide. They drifted once already.
    """
    args = build_parser().parse_args(["optimize"])
    assert getattr(args, flag.lstrip("-").replace("-", "_")) == getattr(OptimizationConfig, field)


def test_optimize_defaults_are_a_usable_config():
    """Whatever the parser hands back has to be something OptimizationConfig accepts."""
    args = build_parser().parse_args(["optimize"])
    config = OptimizationConfig(
        machine=MachineType(args.machine),
        target_distance=args.target_distance,
        efficiency_weight=args.efficiency_weight,
        distance_weight=args.distance_weight,
        mass_weight=args.mass_weight,
        snap_penalty_weight=args.snap_penalty_weight,
    )
    assert config.free_params  # nothing locked by default, so there is something to search
