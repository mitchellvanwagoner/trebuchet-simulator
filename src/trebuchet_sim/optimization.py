"""Trebuchet parameter optimization via differential evolution.

Pure computation only - no printing or interactive prompts. See cli.py for the
command-line presentation layer built on top of this module.
"""

from dataclasses import dataclass, field, fields
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import differential_evolution

from trebuchet_sim.config import (
    DEFAULT_INITIAL_ARM_ANGLE,
    LINKAGE_PARAM,
    MachineType,
    TrebuchetParams,
)
from trebuchet_sim.physics import SimulationResult, simulate_trebuchet

try:
    from trebuchet_sim import fastsim

    _FASTSIM_AVAILABLE = True
except Exception:
    _FASTSIM_AVAILABLE = False

PARAM_NAMES = ["counter_weight_mass", "pulley_radius", "arm_length", "string_length", "release_angle"]


def param_names(machine: MachineType = MachineType.PULLEY) -> List[str]:
    """The five design variables for one machine, in display order.

    Four are shared; the linkage slot holds whichever parameter sizes that machine's
    counterweight coupling (config.LINKAGE_PARAM). The list is always five long, so the
    search space has the same shape either way - only its second entry changes.

    PARAM_NAMES stays the pulley list: it is what the module-level constants
    (PARAM_BOUNDS keys, the fastsim call signature) are written against.
    """
    return [LINKAGE_PARAM[MachineType(machine)] if name == "pulley_radius" else name for name in PARAM_NAMES]


# Non-optimizable TrebuchetParams fields the fast engine needs but that never appear
# in PARAM_NAMES; a run either uses the dataclass default or an OptimizationConfig
# fixed_params override.
_FASTSIM_FIXED_FIELDS = [
    "pivot_height", "pulley_density", "arm_density", "projectile_mass", "projectile_radius",
    "initial_arm_angle", "arm_drag_coefficient", "projectile_drag_coefficient", "joint_friction_coefficient",
    # Sets how fast a pinned counterweight swings, so it is part of the traditional
    # machine's equations of motion rather than just its rendering.
    "counter_weight_rope_length",
]

# Called once per DE generation with (generation, score, params, result); result is None if the
# generation's best-so-far params fail to simulate. Score is the raw objective value (lower is better).
ProgressCallback = Callable[[int, float, TrebuchetParams, Optional[SimulationResult]], None]

PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "counter_weight_mass": (5.0, 60.0),    # kg
    "pulley_radius": (0.01, 1.0),          # m
    "arm_length": (0.1, 2.5),              # m
    "string_length": (0.1, 2.5),           # m
    "release_angle": (np.radians(-290), np.radians(-180)),  # rad (-290 to -180 deg)
    # Traditional machine only, in the linkage slot where pulley_radius sits otherwise.
    # Capped well under the arm-length bound: a short arm approaching the long one is a
    # balanced beam that throws nothing.
    "length_counterweight": (0.05, 1.0),   # m
}


@dataclass
class OptimizationConfig:
    """Optimization objective weights, search bounds, and parameter locks."""

    target_distance: float = 30.0
    efficiency_weight: float = 5.0
    distance_weight: float = 1.0
    mass_weight: float = 0.15
    # Cost per N*s of rope "compression impulse". The fast engine keeps the rigid-link
    # sling model, which can push where a real rope would go slack and lose energy in
    # re-tensioning snaps; penalizing the impulse steers the search to always-taut
    # solutions, where the rigid model matches physics.py's slack/snap model exactly.
    # (The scipy fallback objective simulates the snaps for real, so there the sling
    # loss shows up directly in efficiency and only the cw-rope impulse is penalized.)
    slack_penalty_weight: float = 200.0
    # Which counterweight linkage to design for. It decides the search space (see
    # param_names) rather than being searched itself, so it is a field of its own
    # instead of a fixed_params entry.
    machine: MachineType = MachineType.PULLEY
    locked_params: Dict[str, float] = field(default_factory=dict)
    fixed_params: Dict[str, float] = field(default_factory=dict)
    seed: int = 572956
    max_iterations: int = 1000
    # scipy multiplies popsize by the number of free params, so 40 means a
    # population of ~200 individuals for the 5-parameter search space.
    population_size: int = 40
    absolute_tolerance: float = 0.001
    workers: int = -1                     # -1 = one process per CPU core; ignored when the fast engine runs
    display_progress: bool = False        # scipy's per-iteration convergence printout
    use_fast_engine: bool = True          # Numba-vectorized objective when available; falls back to scipy otherwise

    def __post_init__(self):
        # Accept a plain string, so a machine read back from saved JSON works unchanged.
        self.machine = MachineType(self.machine)

        names = self.param_names
        unknown = set(self.locked_params) - set(names)
        if unknown:
            # Naming the machine matters here: the linkage parameter is the one that
            # differs, so "pulley_radius is unknown" is otherwise a baffling message.
            raise ValueError(
                f"Unknown parameter(s) for the {self.machine.value} machine: {unknown}. Available: {names}"
            )

        if "machine" in self.fixed_params:
            raise ValueError("Set the machine with OptimizationConfig(machine=...), not via fixed_params.")

        valid_fields = {f.name for f in fields(TrebuchetParams)}
        unknown_fixed = set(self.fixed_params) - valid_fields
        if unknown_fixed:
            raise ValueError(f"Unknown fixed parameter(s): {unknown_fixed}. Valid fields: {sorted(valid_fields)}")

        # Search-space params must go through locked_params; a fixed_params entry for one
        # would be silently overwritten by optimizer values in build_params.
        overlap = set(self.fixed_params) & set(names)
        if overlap:
            raise ValueError(f"Parameter(s) {overlap} are optimizable; use locked_params to pin them, not fixed_params.")

    @property
    def param_names(self) -> List[str]:
        """This machine's five design variables (see the module-level param_names)."""
        return param_names(self.machine)

    @property
    def free_params(self) -> List[str]:
        return [name for name in self.param_names if name not in self.locked_params]

    @property
    def bounds(self) -> List[Tuple[float, float]]:
        return [PARAM_BOUNDS[name] for name in self.free_params]

    def build_params(self, free_values: Sequence[float]) -> TrebuchetParams:
        """Combine fixed, locked, and optimized values into a full TrebuchetParams.

        fixed_params (e.g. pivot height, initial arm angle, projectile mass/radius) are
        never part of the search space - they're merged in first so the optimizer can't
        touch them, regardless of what free_params/locked_params contain.
        """
        values = dict(self.fixed_params)
        values.update(self.locked_params)
        values.update(zip(self.free_params, free_values))
        return TrebuchetParams(machine=self.machine, **values)


def _objective(free_values: Sequence[float], config: OptimizationConfig) -> float:
    """Differential-evolution objective: minimize weighted (-efficiency, distance error, mass)."""
    params = config.build_params(free_values)

    if params.string_length > 0.95 * params.arm_length:
        return 1e6

    try:
        # Looser tolerance and no dense interpolants: the objective only reads
        # distance/efficiency, and at rtol=1e-6 the distance shifts by ~0.02%
        # vs the 1e-8 display runs - far below what the weights can resolve.
        result = simulate_trebuchet(params, rtol=1e-6, dense_output=False)
    except Exception:
        return 1e6

    if result.distance <= 0 or result.efficiency <= 0:
        return 1e6

    efficiency_cost = -result.efficiency * 100
    distance_cost = abs(result.distance - config.target_distance) / config.target_distance * 100
    mass_cost = (params.total_mass / 30.0) * 100
    # Only the counterweight rope is still a rigid link here. The sling's slack phases
    # and re-tension snaps are simulated for real by physics.py, so that loss already
    # shows up in `efficiency` and needs no separate penalty (the key this used to read,
    # `string_compression_impulse`, is not in the metrics any more). A traditional
    # machine has no counterweight rope either, leaving this term zero for it.
    slack_cost = config.slack_penalty_weight * result.metrics.get("cw_rope_compression_impulse", 0.0)

    return (
        config.efficiency_weight * efficiency_cost
        + config.distance_weight * distance_cost
        + config.mass_weight * mass_cost
        + slack_cost
    )


def _fastsim_fixed_scalar(config: "OptimizationConfig", name: str) -> float:
    """A fixed (non-optimizable) TrebuchetParams field as a plain float: the config
    override if present, otherwise the dataclass default (matches build_params).

    Two fields have no usable dataclass default. `initial_arm_angle` resolves per machine
    in __post_init__, so the machine's cocked angle stands in. `counter_weight_rope_length`
    defaults to None, meaning "one wrap of the pulley"; numba has no None, so it is passed
    as 0.0 and fastsim applies the same fallback.
    """
    default = getattr(TrebuchetParams, name)
    if default is None:
        if name == "initial_arm_angle":
            default = DEFAULT_INITIAL_ARM_ANGLE[config.machine]
        elif name == "counter_weight_rope_length":
            default = 0.0
    value = config.fixed_params.get(name, default)
    return float(0.0 if value is None else value)


def _objective_vectorized(x: np.ndarray, config: "OptimizationConfig") -> np.ndarray:
    """Batch objective for scipy's vectorized differential_evolution, powered by the
    Numba fast engine. `x` has shape (n_free, S); returns costs of shape (S,)."""
    s = x.shape[1]
    values = {}
    for name in config.param_names:
        if name in config.locked_params:
            values[name] = np.full(s, config.locked_params[name], dtype=np.float64)
        else:
            idx = config.free_params.index(name)
            values[name] = np.ascontiguousarray(x[idx], dtype=np.float64)

    # fastsim takes both linkage parameters as arrays and reads the one its machine uses.
    # Only this machine's is in the search space, so the other is filled with the value a
    # TrebuchetParams would have carried - unused by the dynamics, but it still has to be
    # a well-typed array of the right length.
    unused_linkage = LINKAGE_PARAM[
        MachineType.TRADITIONAL if config.machine is MachineType.PULLEY else MachineType.PULLEY
    ]
    values[unused_linkage] = np.full(s, _fastsim_fixed_scalar(config, unused_linkage), dtype=np.float64)

    fixed = {name: _fastsim_fixed_scalar(config, name) for name in _FASTSIM_FIXED_FIELDS}

    return fastsim.evaluate_population(
        values["counter_weight_mass"], values["pulley_radius"], values["length_counterweight"],
        values["arm_length"], values["string_length"], values["release_angle"],
        fixed["counter_weight_rope_length"],
        fixed["pivot_height"], fixed["pulley_density"], fixed["arm_density"],
        fixed["projectile_mass"], fixed["projectile_radius"], fixed["initial_arm_angle"],
        fixed["arm_drag_coefficient"], fixed["projectile_drag_coefficient"], fixed["joint_friction_coefficient"],
        config.machine is MachineType.PULLEY,
        config.target_distance, config.efficiency_weight, config.distance_weight, config.mass_weight,
        config.slack_penalty_weight,
    )


def optimize_trebuchet(
    config: Optional[OptimizationConfig] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> Tuple[TrebuchetParams, SimulationResult, object]:
    """Optimize trebuchet parameters for maximum efficiency at a target distance.

    Returns (optimal_params, sim_result, scipy_optimize_result).
    """
    config = config or OptimizationConfig()

    if not config.free_params:
        raise ValueError("All parameters are locked; nothing to optimize.")

    def _report_generation(intermediate_result):
        # Param name must be exactly `intermediate_result` - scipy inspects the callback's
        # signature to decide whether to pass the new-style OptimizeResult or legacy (xk, convergence).
        params = config.build_params(intermediate_result.x)
        try:
            result = simulate_trebuchet(params)
        except Exception:
            result = None
        progress_callback(intermediate_result.nit, intermediate_result.fun, params, result)

    # fastsim models both linkages, so the machine no longer decides the engine.
    use_fast = config.use_fast_engine and _FASTSIM_AVAILABLE
    if use_fast:
        de_result = differential_evolution(
            partial(_objective_vectorized, config=config),
            config.bounds,
            seed=config.seed,
            maxiter=config.max_iterations,
            popsize=config.population_size,
            atol=config.absolute_tolerance,
            vectorized=True,
            updating="deferred",
            disp=config.display_progress,
            callback=_report_generation if progress_callback is not None else None,
        )
    else:
        de_result = differential_evolution(
            partial(_objective, config=config),
            config.bounds,
            seed=config.seed,
            maxiter=config.max_iterations,
            popsize=config.population_size,
            atol=config.absolute_tolerance,
            workers=config.workers,
            disp=config.display_progress,
            callback=_report_generation if progress_callback is not None else None,
        )

    optimal_params = config.build_params(de_result.x)
    sim_result = simulate_trebuchet(optimal_params, track_energy=True, simulate_aftermath=True)

    return optimal_params, sim_result, de_result
