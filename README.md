# Trebuchet Simulator

Euler-Lagrange physics simulation of a counterweight trebuchet with air drag,
joint friction, energy tracking, animation, and differential-evolution
parameter optimization.

![Sample animation](docs/assets/animation_mass15kg_range20m.gif)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux

pip install -e ".[dev]"
```

## Web UI

The easiest way to explore the simulator, all in your browser: a single-screen
3-column dashboard with locking number inputs on the left (leave a box blank to
let the optimizer solve it, type a value to lock it), a live animation and
energy plot in the middle, and a full results table on the right. The animation
traces the projectile's path and can switch between the 3D isometric view and a
flat 2D side view:

```bash
python run.py
```

This opens `http://localhost:8501` automatically. Equivalently, once
installed, you can run `trebuchet-web` from anywhere.

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

Then open `http://localhost:8501`. Outputs saved from the UI land in `./outputs`
on the host (mounted as a volume). Equivalently, without compose:

```bash
docker build -t trebuchet-sim .
docker run -p 8501:8501 -v "$(pwd)/outputs:/app/outputs" trebuchet-sim
```

To skip building locally and pull the prebuilt image published by CI instead
(auto-built on every push to `main`, see `.github/workflows/docker-publish.yml`):

```bash
docker compose -f docker-compose.ghcr.yml up
```

To reach the UI on a different host port (e.g. if 8501 is already taken), set
`WEBUI_PORT` — the container still listens on 8501 internally either way:

```bash
WEBUI_PORT=8080 docker compose up --build
```

Both compose files also persist the dashboard's saved parameter defaults (the
💾 button's `user_defaults.json`) in a named Docker volume mounted at
`/app/data`, so they survive container recreation and image updates. To point
that at a host folder instead (e.g. an Unraid appdata share), replace the
volume line with a bind mount onto the same directory path — no need to
pre-create any files, since it's a directory-to-directory mount:

```yaml
volumes:
  - ./outputs:/app/outputs
  - /mnt/user/appdata/treb-sim:/app/data
```

## Command line

```bash
# Run a single simulation
trebuchet simulate --arm-length 0.813 --animate

# Save outputs instead of showing them interactively
trebuchet simulate --save-gif launch.gif --save-energy-plot energy.png

# Search for parameters that hit a target distance efficiently
trebuchet optimize --target-distance 30 --lock counter_weight_mass=14

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
    config.py          TrebuchetParams dataclass and physical constants
    backend.py          numpy/CuPy backend selection (TREBUCHET_GPU=1 to try GPU)
    physics.py           Euler-Lagrange dynamics, ODE integration, energy tracking
    trajectory.py        Shared post-release ballistic trajectory with air drag
    optimization.py      Differential-evolution parameter search
    visualization.py     Matplotlib animation (CLI) and energy plots
    cli.py               `trebuchet` command-line entry point
    web/app.py           Streamlit web UI
    web/animation3d.py   Live Three.js 3D animation embedded in the web UI
tests/                  pytest suite
CAD/                    SolidWorks parts for the physical build
```
