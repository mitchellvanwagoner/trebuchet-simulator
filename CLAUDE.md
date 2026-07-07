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

# Launch the Streamlit web UI (opens browser at localhost:8501)
python run.py

# CLI simulation and optimization
trebuchet simulate --arm-length 0.813 --animate
trebuchet optimize --target-distance 30 --lock counter_weight_mass=14
```

There is no linter configured. Saved CLI outputs (GIFs, plots) land in `outputs/` (git-ignored).

## Architecture

Python package under `src/trebuchet_sim/` (src-layout, installed editable). The core dataflow:

`config.TrebuchetParams` → `physics.TrebuchetSimulator.simulate()` → `SimulationResult` → consumed by visualization, web UI, CLI, and optimizer.

- **physics.py** — Euler-Lagrange dynamics of the launch phase (state = `[theta, theta_dot, alpha, alpha_dot]`, arm angle + string angle), integrated with `scipy.solve_ivp` until a release-angle event fires. Produces `SimulationResult` with `distance`, `efficiency`, a `metrics` dict, the dense ODE solution, optional energy history, and the post-release `trajectory`. The sling and counterweight rope are rigid links, which can push — a real rope can't. `constraint_tensions()` exposes both tensions, and every result carries `min_string_tension` / `string_compression_impulse` (∫max(0,−T)dt, N·s) metrics plus counterweight-rope equivalents: a nonzero impulse means the launch jerks the sling, the real machine would go slack, and the reported distance/efficiency are unphysical (web UI and CLI warn above the 0.05 N·s noise floor).
- **trajectory.py** — ballistic flight after release with quadratic air drag. Shared so the sim engine, matplotlib animation, and web animation all render the identical flight.
- **`sample_component_positions()`** in physics.py samples projectile/arm-tip/counterweight positions from a solved run; both animation frontends (matplotlib and Three.js) build frames from it so they can never disagree with the physics.
- **optimization.py** — differential evolution over the five optimizable params (`PARAM_NAMES` / `PARAM_BOUNDS`). The objective penalizes rope compression impulse (`slack_penalty_weight`, mirrored in both engines) so the search only returns always-taut solutions — the regime where the rigid-link model is exact. Three parameter tiers matter throughout the codebase: *free* (optimizer searches), *locked* (user pins an optimizable param), *fixed* (`fixed_params` — never optimizable, e.g. pivot height, projectile mass). Pure computation, no printing; cli.py is the presentation layer.
- **config.py** — `TrebuchetParams` dataclass; derived quantities (masses, moments of inertia, areas) are properties. `DEFAULT_OPTIMIZABLE_PARAMS` is the single source of defaults shared by CLI, web UI, and tests.
- **web/app.py** — Streamlit dashboard (3 columns: parameter inputs / animation + energy plot / results). Streamlit reruns the whole script on every widget interaction, so expensive outputs (animation HTML, energy figure) are pre-rendered once in `_store_result()` and cached in `st.session_state`.
- **web/animation3d.py** — no physics: samples the solved result into a JSON timeline (`_build_timeline`) and embeds a self-contained Three.js HTML page that plays it back with scrub/speed controls. The HTML template is a Python string with `__TIMELINE_JSON__` / `__HEIGHT__` / `__THREE_JS__` / `__ORBIT_JS__` placeholders; Three.js r128 is vendored in `web/static/` and inlined so the animation works offline (r128 pinned deliberately — newer releases removed `examples/js/OrbitControls.js`).

The Docker image (multi-stage, non-root, healthcheck) installs the wheel with the `[fast]` extra and starts via the `trebuchet-web` console script; deps are pinned in `docker-constraints.txt`. Saved dashboard defaults go to `TREBUCHET_DATA_DIR` (run.py sets the repo root, Docker sets `/app/data`, bare `trebuchet-web` falls back to `~/.trebuchet-sim`).

Simulation results use the convention: `"error" in result.metrics` means the run failed; `metrics["release_occurred"]` False means the arm never reached the release angle. Check both before rendering animations or reading release metrics.
