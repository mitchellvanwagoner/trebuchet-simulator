# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup (Windows venv lives in .venv/)
.venv\Scripts\activate
pip install -e ".[dev]"

# Run all tests / a single test
pytest
pytest tests/test_physics.py -k test_name

# Launch the Streamlit web UI (opens browser at localhost:8501).
# Rebuilds .venv automatically if it is missing or stale - a venv records
# absolute paths, so moving the project folder breaks it until this repairs it.
python run.py

# CLI simulation and optimization
trebuchet simulate --arm-length 0.813 --animate
trebuchet optimize --target-distance 30 --lock counter_weight_mass=14
```

There is no linter configured. Saved CLI outputs (GIFs, plots) land in `outputs/` (git-ignored).

## Architecture

Python package under `src/trebuchet_sim/` (src-layout, installed editable). The core dataflow:

`config.TrebuchetParams` → `physics.TrebuchetSimulator.simulate()` → `SimulationResult` → consumed by visualization, web UI, CLI, and optimizer.

- **physics.py** — Euler-Lagrange dynamics of the launch phase (state = `[theta, theta_dot, alpha, alpha_dot, psi, psi_dot]`: arm angle, sling angle, and the counterweight's swing about its pin), integrated with `scipy.solve_ivp` until a release-angle event fires. Produces `SimulationResult` with `distance`, `efficiency`, a `metrics` dict, the `LaunchSolution`, optional energy history, and the post-release `trajectory`. The sling is modeled as a real rope: when the rigid-link tension would go negative the projectile detaches (`LaunchSolution` stitches taut/slack segments) and the re-tension snap dissipates energy, reported as `string_slack_fraction` / `sling_snap_count` / `sling_snap_energy`. The counterweight rope is *still* a rigid link, so it keeps the feasibility-style `min_cw_rope_tension` / `cw_rope_compression_impulse` (∫max(0,−T)dt, N·s) metrics — nonzero impulse means the arm out-accelerates the falling counterweight and that part of the run is unphysical (web UI and CLI warn above the 0.05 N·s noise floor). Both metrics are pulley-machine only; a pinned counterweight has no rope to go slack.
- **trajectory.py** — ballistic flight after release with quadratic air drag. Shared so the sim engine, matplotlib animation, and web animation all render the identical flight.
- **`sample_component_positions()`** in physics.py samples projectile/arm-tip/counterweight positions from a solved run; both animation frontends (matplotlib and Three.js) build frames from it so they can never disagree with the physics.
- **optimization.py** — differential evolution over the five optimizable params (`PARAM_NAMES` / `PARAM_BOUNDS`). The objective penalizes counterweight-rope compression impulse (`slack_penalty_weight`, mirrored in both engines) so the search only returns solutions where that rigid link stays in tension. Three parameter tiers matter throughout the codebase: *free* (optimizer searches), *locked* (user pins an optimizable param), *fixed* (`fixed_params` — never optimizable, e.g. pivot height, projectile mass). `fastsim.py` models the pulley machine only, so a traditional machine falls back to the scipy objective. Pure computation, no printing; cli.py is the presentation layer.
- **config.py** — `TrebuchetParams` dataclass; derived quantities (masses, moments of inertia, areas) are properties. `DEFAULT_OPTIMIZABLE_PARAMS` is the single source of defaults shared by CLI, web UI, and tests. `MachineType` picks the counterweight linkage: `PULLEY` (weight on a rope over the pivot axle, constant lever `pulley_radius`, single-sided arm) or `TRADITIONAL` (weight pinned to the arm's short end `length_counterweight` behind the pivot, so it swings on its own coordinate and the beam has mass on both sides — see `arm_cm_offset` / `moi_arm`). One set of equations of motion covers both; `has_pulley` branches the handful of places they differ. `initial_arm_angle` defaults to `None` and `__post_init__` resolves it per machine from `DEFAULT_INITIAL_ARM_ANGLE`.
- **web/app.py** — Streamlit dashboard (3 columns: parameter inputs / animation + energy plot / results). Streamlit reruns the whole script on every widget interaction, so expensive outputs (animation HTML, energy figure) are pre-rendered once in `_store_result()` and cached in `st.session_state`. Results render as custom HTML (`_metric_grid` / `_spec_list` / the hero block), not `st.table`. Unit switching: length inputs swap between separate metric and ft/in widgets so they reseed themselves, but mass shares one box across both systems and must be converted in place by `_sync_mass_units()` before any input renders.
- **web/theme.py** — the design system, and the single source of truth for the dashboard's look: palette constants, `streamlit_theme_env()` (Streamlit resolves its theme at server start, before the app script runs, so it can only arrive via env vars — applied by both `web/launcher.py` and `run.py`, which is why the container and a local checkout match without a `.streamlit/config.toml`, a path that is git-ignored), and `DASHBOARD_CSS`. The layout is a fixed-viewport flex column: the page is pinned to `100vh` and the panels that should absorb slack are marked `flex: 1 1 0`, replacing the previous `calc(100vh - Npx)` magic numbers that had to be re-measured by hand and had already drifted far enough to push the header off-screen. Imports nothing, so `run.py` can read the palette before the package is installed.
- **web/animation3d.py** — no physics: samples the solved result into a JSON timeline (`_build_timeline`) and embeds a self-contained Three.js HTML page that plays it back with scrub/speed controls. The HTML template is a Python string with `__TIMELINE_JSON__` / `__HEIGHT__` / `__THREE_JS__` / `__ORBIT_JS__` placeholders plus the palette placeholders in `_THEME_SUBSTITUTIONS` (so the scene shares `web/theme.py`'s colors); Three.js r128 is vendored in `web/static/` and inlined so the animation works offline (r128 pinned deliberately — newer releases removed `examples/js/OrbitControls.js`).

Selecting a machine touches every layer, because the two differ in one design variable rather
than in a setting: `config.LINKAGE_PARAM` maps each machine to the parameter that sizes its
counterweight coupling (`pulley_radius` / `length_counterweight`), and `optimization.param_names(machine)`
swaps it into the five-name search space. So `OptimizationConfig.machine` is a field of its own,
not a `fixed_params` entry (it decides the search space rather than being searched), and the
CLI/UI both resolve their defaults, bounds and labels through it. In the dashboard every input's
session-state key is scoped to the machine (`_widget_key`): Streamlit keeps a keyed widget's value
across reruns, so a box can only be re-defaulted by becoming a different widget - the same trick
the length inputs use across the unit toggle. Saved defaults record the machine they were written
for and only reload onto that machine. Both animation frontends draw the linkage from the `cw_pin`
track that `sample_component_positions()` returns, and skip the pulley disc when there is none.

The Docker image (multi-stage, non-root, healthcheck) installs the wheel with the `[fast]` extra and starts via the `trebuchet-web` console script; deps are pinned in `docker-constraints.txt`. Saved dashboard defaults go to `TREBUCHET_DATA_DIR` (run.py sets the repo root, Docker sets `/app/data`, bare `trebuchet-web` falls back to `~/.trebuchet-sim`).

Simulation results use the convention: `"error" in result.metrics` means the run failed; `metrics["release_occurred"]` False means the arm never reached the release angle. Check both before rendering animations or reading release metrics.
