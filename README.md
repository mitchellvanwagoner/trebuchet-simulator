# Trebuchet Simulator

Euler-Lagrange physics simulation of a counterweight trebuchet with air drag,
joint friction, energy tracking, animation, and differential-evolution
parameter optimization.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux

pip install -e ".[dev]"
```

## Web UI

The easiest way to explore the simulator, all in your browser: a single-screen
3-column dashboard. Parameters are on the left, each design variable paired with
a lock toggle — lock it to pin the parameter to the value in its box, leave it
unlocked to let the optimizer search it (an unlocked box's contents are
ignored). Locked rows are tinted so the search space is readable at a glance.
"Search ranges" sets the bounds the optimizer looks between for each parameter,
defaulting to the built-in ones. A
live 3D animation and energy plot sit in the middle, and the range, efficiency,
release conditions, and resolved machine geometry on the right. A selector at
the top picks the counterweight linkage — pulley or traditional. The animation
traces the projectile's path and can switch between the 3D isometric view and a
flat 2D side view. A units toggle switches the whole dashboard between metric
and ft/in/lb:

```bash
python run.py
```

This opens `http://localhost:8501` automatically, or the next free port if 8501
is taken. `run.py` also rebuilds `.venv` if it is missing or stale, so a fresh
clone — or a copy of the project moved to a new folder, which leaves the venv's
recorded absolute paths dangling — needs no manual setup first. Equivalently,
once installed, you can run `trebuchet-web` from anywhere.

To serve at the friendlier `http://treb-simulator.local:8501` instead, run
this once in an elevated (admin) PowerShell, then restart `run.py`:

```powershell
Add-Content "$env:SystemRoot\System32\drivers\etc\hosts" "`r`n127.0.0.1 treb-simulator.local"
```

## Docker

Run the web UI in a container, no local Python setup required:

```bash
docker compose up --build
```

Then open `http://localhost:8501`. Equivalently, without compose:

```bash
docker build -t trebuchet-sim .
docker run -p 8501:8501 trebuchet-sim
```

The image is fully self-contained (Three.js is bundled, so the animation works
without internet access), includes the Numba fast engine for the optimizer,
runs as a non-root user, and reports readiness via a built-in healthcheck.
Dependency versions are pinned in `docker-constraints.txt` so published images
are reproducible.

To skip building locally and pull the prebuilt image published by CI instead
(auto-built and smoke-tested on every push to `main`, see
`.github/workflows/docker-publish.yml`):

```bash
docker compose -f docker-compose.ghcr.yml up
```

To reach the UI on a different host port (e.g. if 8501 is already taken), set
`WEBUI_PORT` — the container still listens on 8501 internally either way:

```bash
WEBUI_PORT=8080 docker compose up --build
```

The dashboard's saved parameter defaults (the 💾 button's `user_defaults.json`)
live in `/app/data`. `docker-compose.yml` persists them in a named Docker
volume; `docker-compose.ghcr.yml` bind-mounts a host folder chosen by
`DATA_DIR` (default `./data`), e.g. on Unraid:

```bash
DATA_DIR=/mnt/user/appdata/treb-sim docker compose -f docker-compose.ghcr.yml up
```

The app itself runs as a non-root user, but the container starts as root just
long enough to chown `/app/data` to that user before dropping privileges - so
`DATA_DIR` can point at a host folder with any existing ownership (e.g. a
fresh Unraid appdata share) without a manual `chown` first.

`TREBUCHET_OPT_WORKERS` caps the optimizer's worker processes, but only
applies to the scipy fallback engine — the image ships with Numba, whose
vectorized engine ignores it.

## Machine types

Two counterweight linkages, sharing one set of equations of motion:

- **Pulley** — the counterweight hangs from a rope over a pulley on the pivot
  axle and drops straight down, so its lever arm is the constant
  `pulley_radius`. The arm is a single beam from the pivot to the sling.
- **Traditional** — the counterweight is bolted to the arm's short end,
  `length_counterweight` behind the pivot, so it swings around with the arm and
  on its own pin as well. The beam carries mass on both sides of the pivot.

That one difference propagates: each machine has its own cocked arm angle, its
own default geometry, and its own linkage parameter in the optimizer's search
space (`pulley_radius` or `length_counterweight` — never both). Pick the
machine with the selector above the dashboard's parameters, or with `--machine`
on either CLI command; switching reloads that machine's defaults, because the
numbers don't carry across.

The Numba fast engine models both linkages, so either machine optimizes at
full speed (a 60 m traditional search takes well under a second).

## Command line

```bash
# Run a single simulation
trebuchet simulate --arm-length 0.813 --animate

# Save outputs instead of showing them interactively
trebuchet simulate --save-gif launch.gif --save-energy-plot energy.png

# Search for parameters that hit a target distance efficiently
trebuchet optimize --target-distance 30 --lock counter_weight_mass=14

# Either machine, on both commands (see "Machine types" below)
trebuchet simulate --machine traditional --counterweight-mass 80
trebuchet optimize --machine traditional --target-distance 60 --lock length_counterweight=0.4

# Narrow (or widen) what the optimizer searches, per parameter
trebuchet optimize --target-distance 30 --range arm_length=0.3:0.8 --range release_angle=-5.0:-4.0

python -m trebuchet_sim.cli simulate --help
python -m trebuchet_sim.cli optimize --help
```

Saved files land in `outputs/` (git-ignored).

## Tests

```bash
pytest
```

## Project layout

```
src/trebuchet_sim/
    config.py            TrebuchetParams dataclass and physical constants
    physics.py           Euler-Lagrange dynamics, ODE integration, energy tracking
    trajectory.py        Shared post-release ballistic trajectory with air drag
    fastsim.py           Numba-JIT engine used by the optimizer's objective
    optimization.py      Differential-evolution parameter search
    visualization.py     Matplotlib animation (CLI) and energy plots
    cli.py               `trebuchet` command-line entry point
    web/app.py           Streamlit web UI
    web/theme.py         Dashboard palette, Streamlit theme, and layout CSS
    web/units.py         Metric <-> Imperial conversions for the dashboard
    web/launcher.py      `trebuchet-web` console-script entry point
    web/animation3d.py   Live Three.js 3D animation embedded in the web UI
    web/static/          Vendored Three.js (r128), inlined so the UI works offline
tests/                  pytest suite
```
