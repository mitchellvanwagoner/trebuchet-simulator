"""Streamlit web UI for the trebuchet simulator.

Run with: streamlit run src/trebuchet_sim/web/app.py
Or simply: python run.py   (from the repo root)

Layout goal: the whole dashboard fits on one screen with no scrolling, so the
CSS below compacts Streamlit's default paddings and the inputs sit in
two-per-row grids.
"""

import json
import logging
import math
import os
from dataclasses import fields
from pathlib import Path

import streamlit as st

# The optimizer's differential-evolution workers touch Streamlit internals from
# a background thread, which trips this logger's "missing ScriptRunContext"
# warning on every callback. Streamlit's own message says it's safe to ignore
# in this context (no per-thread script context outside the main run), so it's
# silenced rather than left spamming the console on every optimize run.
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)

from trebuchet_sim.web import units
from trebuchet_sim.web.animation3d import build_trebuchet_3d_html, render_trebuchet_3d_html
from trebuchet_sim.config import DEFAULT_OPTIMIZABLE_PARAMS, TrebuchetParams
from trebuchet_sim.optimization import PARAM_BOUNDS, PARAM_NAMES, OptimizationConfig, optimize_trebuchet
from trebuchet_sim.physics import simulate_trebuchet
from trebuchet_sim.visualization import build_energy_figure

st.set_page_config(page_title="Trebuchet Simulator", page_icon="🏰", layout="wide")

# Compact, single-screen dashboard styling. Also hides the "Ask Google" /
# "Ask ChatGPT" links Streamlit appends to uncaught-exception messages.
st.markdown(
    """
    <style>
      header[data-testid="stHeader"] { display: none; }
      [data-testid="stToolbar"] { display: none; }
      .block-container { padding: 0.35rem 1.1rem 0.2rem; max-width: 100%; }
      [data-testid="stVerticalBlock"] { gap: 0.25rem; }
      [data-testid="stHorizontalBlock"] { gap: 0.6rem; }
      [data-testid="stWidgetLabel"] { margin-bottom: 0.05rem; min-height: 1.1rem; }
      [data-testid="stWidgetLabel"] p { font-size: 0.78rem; }
      [data-testid="stNumberInput"] input { padding: 0.15rem 0.5rem; font-size: 0.85rem; }
      [data-testid="stNumberInput"] div[data-baseweb="input"] { height: 1.75rem; }
      [data-testid="stNumberInputStepDown"], [data-testid="stNumberInputStepUp"] { display: none; }
      .stButton button { min-height: 1.9rem; padding: 0.15rem 0.75rem; }
      [data-testid="stTable"] table { font-size: 0.78rem; table-layout: fixed; width: 100%; }
      [data-testid="stTable"] th, [data-testid="stTable"] td {
        padding: 0.15rem 0.4rem;
        overflow-wrap: break-word;
        word-break: break-word;
        white-space: normal;
      }
      [data-testid="stTable"] thead th {
        font-weight: 700;
        text-transform: uppercase;
        font-size: 0.66rem;
        letter-spacing: 0.02em;
        background: rgba(120, 120, 120, 0.35);
        border-bottom: 2px solid rgba(120, 120, 120, 0.6);
      }
      /* Results panel: a fixed min-height (verified against a real render,
         since st.container(height="stretch") does not actually stretch
         inside an st.columns() cell - it only matches content height) makes
         the panel visibly span down to the rest of the dashboard instead of
         leaving blank space below a short table. */
      .st-key-results_panel { min-height: calc(100vh - 120px); }
      /* Optimizer log: fill everything between the inputs above and the bottom
         of the viewport. The offset is the measured height of the header +
         input sections; the dataframe inside stretches to the panel. */
      .st-key-opt_log_panel {
        flex: 0 0 calc(100vh - 652px) !important;
        height: calc(100vh - 652px) !important;
        min-height: 90px;
      }
      /* The status caption sits at natural height; the table container below
         it absorbs the rest of the panel and the dataframe stretches into it. */
      .st-key-opt_log_panel { display: flex; flex-direction: column; }
      /* Streamlit wraps every keyed container in its own non-growing
         "stLayoutWrapper" div; flexing the inner block is not enough; the
         wrapper itself must grow, or it caps its child at content height. */
      .st-key-opt_log_panel > div:has(> .st-key-opt_log_table) { flex: 1 1 0; min-height: 0; }
      /* flex-basis 0 (not auto): the dataframe's own height is itself a
         percentage of this container, so sizing from content would be
         circular - basis 0 makes the browser size this purely from
         flex-grow first, then percentages below resolve against that. */
      .st-key-opt_log_table { flex: 1 1 0; min-height: 0; height: 100% !important; }
      .st-key-opt_log_table [data-testid="stElementContainer"],
      .st-key-opt_log_table [data-testid="stDataFrame"] { height: 100% !important; }
      /* Animation fills the middle column so the energy plots below it end at
         the bottom of the screen; the iframe content sizes itself with 100vh. */
      /* The energy figure spans the full column width at its natural aspect
         (12:3.2), so its height scales with the viewport width; the animation
         panel's height budget subtracts that same width-proportional term. */
      .st-key-anim_panel {
        flex: 0 0 calc(100vh - 48px - 13.9vw) !important;
        height: calc(100vh - 48px - 13.9vw) !important;
      }
      /* Streamlit pins the iframe's container to the height= argument via
         flex-basis; let it grow to the panel instead. */
      .st-key-anim_panel [data-testid="stElementContainer"] { flex: 1 1 auto !important; }
      .st-key-anim_panel [data-testid="stElementContainer"],
      .st-key-anim_panel iframe { height: 100% !important; width: 100%; }
      .st-key-energy_panel img { width: 100% !important; height: auto !important; display: block; }
      /* Status alerts float over the animation rather than reflowing it. */
      .st-key-status_area { position: fixed; top: 42px; left: 25%; width: 50%; z-index: 100; }
      [data-testid="stCaptionContainer"] p { font-size: 0.72rem; }
      [data-testid="stCode"] pre { font-size: 0.7rem; padding: 0.35rem 0.55rem; }
      [data-testid="stHeading"] h3 { font-size: 0.95rem !important; padding: 0.15rem 0 0.1rem 0 !important; }
      [data-testid="stHeading"] h4 { padding: 0 !important; margin: 0 !important; font-size: 1.05rem !important; }
      /* Streamlit offsets its default (large) heading padding with a negative
         bottom margin; with the compact headings above it makes headings
         overlap the row below them. */
      [data-testid="stHeading"] [data-testid="stMarkdownContainer"] { margin-bottom: 0 !important; }
      [data-testid="stException"] a { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

title_col, unit_col = st.columns([5, 1])
title_col.markdown("#### 🏰 Trebuchet Physics Simulator")
imperial = unit_col.toggle(
    "Imperial", key="imperial_units", help="Show lengths in ft/in, masses in lb, speeds in ft/s."
)

ANIMATION_HEIGHT = 440

# Fixed system params: required, always used as given, never handed to the optimizer.
# Defaults come straight from the TrebuchetParams dataclass so they can't drift.
FIXED_PARAM_NAMES = ("pivot_height", "initial_arm_angle", "projectile_mass", "projectile_radius")
FIXED_DEFAULTS = {f.name: f.default for f in fields(TrebuchetParams) if f.name in FIXED_PARAM_NAMES}

# User-saved input defaults (💾 button), stored in canonical units (m, kg,
# radians) like TrebuchetParams. TREBUCHET_DATA_DIR picks the directory:
# run.py sets it to the repo root (git-ignored there) and the Docker image
# points it at a mountable volume; without it (e.g. bare `trebuchet-web`)
# fall back to a per-user home dir so the location never depends on the cwd.
_DATA_DIR = Path(os.environ.get("TREBUCHET_DATA_DIR") or Path.home() / ".trebuchet-sim")
USER_DEFAULTS_FILE = _DATA_DIR / "user_defaults.json"

# Optimizer worker processes (scipy's differential_evolution `workers`; see
# OptimizationConfig.workers). -1 = one process per CPU core, same as the
# dataclass default; TREBUCHET_OPT_WORKERS overrides it (e.g. to cap CPU use
# in a resource-limited container). Ignored when the fast/Numba engine runs.
try:
    _OPT_WORKERS = int(os.environ.get("TREBUCHET_OPT_WORKERS", "-1"))
except ValueError:
    _OPT_WORKERS = -1


def _load_user_defaults() -> dict:
    """Saved input defaults, read once per session; {} when absent/corrupt."""
    if "user_defaults" not in st.session_state:
        try:
            st.session_state.user_defaults = json.loads(USER_DEFAULTS_FILE.read_text())
        except Exception:
            st.session_state.user_defaults = {}
    return st.session_state.user_defaults


def _save_user_defaults(optimizable: dict, fixed: dict, target: dict) -> None:
    """Persist the current inputs (None = leave free for the optimizer)."""
    defaults = {"optimizable": optimizable, "fixed": fixed, "target": target}
    USER_DEFAULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    USER_DEFAULTS_FILE.write_text(json.dumps(defaults, indent=2))
    st.session_state.user_defaults = defaults

st.session_state.setdefault("result", None)
st.session_state.setdefault("sim_params", None)
st.session_state.setdefault("anim_html", None)
st.session_state.setdefault("energy_fig", None)
st.session_state.setdefault("opt_log_rows", [])
st.session_state.setdefault("opt_status", None)

# The optimizer log dataframe stretches to fill its panel, which the CSS above
# pins to the space between the inputs and the bottom of the viewport.
OPT_LOG_HEIGHT = "stretch"


def _log_row(generation: int, score: float, target_distance: float, result, imperial: bool) -> dict:
    """One row of the optimizer log: the DE generation's best-so-far score and progress.

    Dataframe columns need numeric values for proper sorting, so distance uses
    plain decimal feet here (not the feet+inches split in the results tables)
    when Imperial. Rows already in the log keep whichever unit system was
    active when they were generated - the log reflects history, not a live
    reconversion of past rows when the toggle changes.
    """
    dist_label = "Distance (ft)" if imperial else "Distance (m)"
    delta_label = "Δ Target (ft)" if imperial else "Δ Target (m)"

    def to_display(value_m: float) -> float:
        return value_m / units.METERS_PER_FOOT if imperial else value_m

    if result is not None and "error" not in result.metrics and result.metrics.get("release_occurred", True):
        return {
            "Gen": generation,
            "Score": round(score, 2),
            dist_label: round(to_display(result.distance), 2),
            delta_label: round(to_display(result.distance - target_distance), 2),
            "Efficiency (%)": round(result.efficiency * 100, 1),
        }
    return {
        "Gen": generation,
        "Score": round(score, 2),
        dist_label: None,
        delta_label: None,
        "Efficiency (%)": None,
    }


def _store_result(params: TrebuchetParams, result) -> None:
    """Store a new result and pre-render its expensive outputs once.

    Streamlit reruns this script on every widget interaction; caching the
    animation HTML and energy figure here keeps reruns cheap.
    """
    st.session_state.sim_params = params
    st.session_state.result = result
    releasable = "error" not in result.metrics and result.metrics.get("release_occurred", True)
    st.session_state.anim_html = build_trebuchet_3d_html(params, result, height=ANIMATION_HEIGHT) if releasable else None
    st.session_state.energy_fig = build_energy_figure(result, compact=True) if result.energy_history else None


def _length_pair_input(
    container, label: str, key_prefix: str, default_m: "float | None",
    min_value: float, max_value: "float | None" = None, help: str = None,
) -> "float | None":
    """Feet+inches sub-widget pair for a length field in Imperial mode.

    Rendered as two columns inside the caller's existing grid cell, so it keeps
    the same row height as a single metric box rather than growing the layout.
    Both boxes start blank ("free"/"use default") when the caller passes
    default_m=None. The combined value counts as set as soon as EITHER box has
    a value, treating the other as 0 - a user who only types into the inches
    box (e.g. a 6" pulley radius) must not be silently ignored just because
    they left feet untouched. Returns the combined value in meters (SI),
    clamped into [min_value, max_value], or None if both boxes are blank.
    """
    feet_default = inches_default = None
    if default_m is not None:
        clamped = max(default_m, min_value)
        if max_value is not None:
            clamped = min(clamped, max_value)
        feet_default, inches_default = units.meters_to_feet_inches(clamped)

    feet_max = math.ceil(max_value / units.METERS_PER_FOOT) + 1 if max_value is not None else None

    sub_feet, sub_inches = container.columns(2)
    feet_val = sub_feet.number_input(
        f"{label} (ft)", min_value=0, max_value=feet_max, value=feet_default,
        key=f"{key_prefix}_ft", step=1, help=help,
    )
    inches_val = sub_inches.number_input(
        "(in)", min_value=0.0, max_value=11.99, value=inches_default,
        key=f"{key_prefix}_in", step=0.1, format="%.2f",
    )

    if feet_val is None and inches_val is None:
        return None
    combined = max(units.feet_inches_to_meters(feet_val, inches_val), min_value)
    return min(combined, max_value) if max_value is not None else combined


def _optimizable_input(
    container, label: str, name: str, in_degrees: bool = False,
    kind: str = "length", imperial: bool = False,
) -> "float | None":
    """Number input for an optimizable parameter. None means free/unlocked.

    The lock state lives in session_state from the previous rerun, so the label
    icon can reflect it without an extra caption row. `kind` picks the unit
    conversion for the non-angle case ("length" -> feet+inches, "mass" -> lb)
    when `imperial`; the return value is always canonical SI (m, kg, or rad),
    matching TrebuchetParams/OptimizationConfig regardless of display units.
    """
    lo, hi = PARAM_BOUNDS[name]
    if in_degrees:
        lo, hi = math.degrees(lo), math.degrees(hi)

    # Saved default (canonical units): a value means "start locked at this",
    # absent/None means start free. Clamped in case bounds changed since saving.
    saved = _load_user_defaults().get("optimizable", {}).get(name)
    if saved is not None:
        saved = math.degrees(saved) if in_degrees else saved
        saved = min(max(saved, lo), hi)

    help_text = "Leave blank to let the optimizer solve it; type a value to lock it."

    if in_degrees:
        icon = "🔒" if st.session_state.get(f"opt_{name}", saved) is not None else "🔓"
        return container.number_input(
            f"{icon} {label} (deg)", min_value=lo, max_value=hi, value=saved,
            key=f"opt_{name}", format="%.4f", help=help_text,
        )

    if kind == "mass":
        lo_disp, hi_disp = (units.kg_to_lb(lo), units.kg_to_lb(hi)) if imperial else (lo, hi)
        saved_disp = (units.kg_to_lb(saved) if saved is not None else None) if imperial else saved
        icon = "🔒" if st.session_state.get(f"opt_{name}", saved_disp) is not None else "🔓"
        unit_label = "lb" if imperial else "kg"
        value_disp = container.number_input(
            f"{icon} {label} ({unit_label})", min_value=lo_disp, max_value=hi_disp, value=saved_disp,
            key=f"opt_{name}", format="%.4f", help=help_text,
        )
        if value_disp is None:
            return None
        return units.lb_to_kg(value_disp) if imperial else value_disp

    # kind == "length"
    if imperial:
        # The ft/in boxes' keys differ from the metric-mode single box, so the
        # icon falls back to the saved value's feet/inches components on first
        # render, before session_state has those keys - same pattern as the
        # metric case below. Locked as soon as EITHER box is set, matching
        # _length_pair_input's own "either box counts" rule.
        feet_saved = inches_saved = None
        if saved is not None:
            feet_saved, inches_saved = units.meters_to_feet_inches(min(max(saved, lo), hi))
        feet_state = st.session_state.get(f"opt_{name}_ft", feet_saved)
        inches_state = st.session_state.get(f"opt_{name}_in", inches_saved)
        icon = "🔒" if (feet_state is not None or inches_state is not None) else "🔓"
        return _length_pair_input(
            container, f"{icon} {label}", f"opt_{name}", saved, lo, hi,
            help="Leave both boxes blank to let the optimizer solve it; fill in either to lock it.",
        )

    icon = "🔒" if st.session_state.get(f"opt_{name}", saved) is not None else "🔓"
    return container.number_input(
        f"{icon} {label} (m)", min_value=lo, max_value=hi, value=saved,
        key=f"opt_{name}", format="%.4f", help=help_text,
    )


def _fixed_input(
    container, label: str, name: str, min_value: float, max_value: float,
    in_degrees: bool = False, kind: str = "length", imperial: bool = False,
) -> "float | None":
    """Number input for a fixed (never-optimized) system parameter.

    `kind` picks the unit conversion for the non-angle case ("length" -> a
    feet+inches sub-widget pair, "mass" -> pounds) when `imperial`; always
    returns canonical SI (m, kg, or rad) regardless of the display units.
    """
    default = _load_user_defaults().get("fixed", {}).get(name)
    if default is None:
        default = FIXED_DEFAULTS[name]

    if in_degrees:
        default = min(max(math.degrees(default), min_value), max_value)
        return container.number_input(
            f"{label} (deg)", min_value=min_value, max_value=max_value, value=default,
            key=f"fixed_{name}", format="%.4f",
        )

    if kind == "mass":
        unit_label = "lb" if imperial else "kg"
        lo_disp, hi_disp = (units.kg_to_lb(min_value), units.kg_to_lb(max_value)) if imperial else (min_value, max_value)
        default_disp = min(max(units.kg_to_lb(default) if imperial else default, lo_disp), hi_disp)
        value_disp = container.number_input(
            f"{label} ({unit_label})", min_value=lo_disp, max_value=hi_disp, value=default_disp,
            key=f"fixed_{name}", format="%.4f",
        )
        return units.lb_to_kg(value_disp) if imperial else value_disp

    # kind == "length"
    if imperial:
        return _length_pair_input(container, label, f"fixed_{name}", default, min_value, max_value)
    default = min(max(default, min_value), max_value)
    return container.number_input(
        f"{label} (m)", min_value=min_value, max_value=max_value, value=default,
        key=f"fixed_{name}", format="%.4f",
    )


def _fixed_input_optional(
    container, label: str, name: str, min_value: float, max_value: float,
    imperial: bool = False, help: str = None,
) -> "float | None":
    """Number input for a fixed system parameter that may be left blank.

    Unlike _fixed_input, blank is a legal value here (not "still typing") - it
    means TrebuchetParams should fall back to its own computed default. Always
    a length field in practice (CW rope length) - kept simple rather than
    generalized to mass/angle since there's no such call site today.
    """
    saved = _load_user_defaults().get("fixed", {}).get(name)
    if saved is not None:
        saved = min(max(saved, min_value), max_value)

    if imperial:
        return _length_pair_input(container, label, f"fixed_{name}", saved, min_value, max_value, help=help)

    return container.number_input(
        f"{label} (m)", min_value=min_value, max_value=max_value, value=saved,
        key=f"fixed_{name}", format="%.4f", help=help,
    )


def _fmt_length(value_m: float, imperial: bool) -> str:
    """Feet+inches split for the results tables (see _log_row for why the
    optimizer log uses plain decimal feet instead)."""
    if not imperial:
        return f"{value_m:.4f} m"
    feet, inches = units.meters_to_feet_inches(value_m)
    return f"{feet} ft {inches:.2f} in"


def _fmt_mass(value_kg: float, imperial: bool) -> str:
    return f"{units.kg_to_lb(value_kg):.3f} lb" if imperial else f"{value_kg:.3f} kg"


def _fmt_speed(value_mps: float, imperial: bool) -> str:
    return f"{units.mps_to_fps(value_mps):.2f} ft/s" if imperial else f"{value_mps:.2f} m/s"


def _render_stat_table(data: dict, columns_per_row: int = 2) -> None:
    """Render label/value pairs as small fixed-width tables, a few columns at a time.

    Putting every metric in one row (one table, N columns) forces horizontal
    scrolling once N gets past a handful; chunking keeps each table narrow
    enough to fit the column width with wrapped headers instead.
    """
    items = list(data.items())
    for start in range(0, len(items), columns_per_row):
        st.table([dict(items[start : start + columns_per_row])])


def _show_results(params: TrebuchetParams, result, imperial: bool) -> None:
    if "error" in result.metrics:
        st.error(result.metrics["error"])
        return

    if not result.metrics.get("release_occurred", True):
        st.warning("No release occurred - the arm never reached the release angle within the simulation window.")
        _render_stat_table(
            {
                "Simulation time (s)": f"{result.metrics['simulation_time']:.2f}",
                "Final arm angle (deg)": f"{result.metrics['final_arm_angle_deg']:.1f}",
                "Total arm rotation (deg)": f"{result.metrics['total_rotation_deg']:.1f}",
            }
        )
        return

    st.markdown("**Solution**")
    _render_stat_table(
        {
            "Range": _fmt_length(result.distance, imperial),
            "Efficiency": f"{result.efficiency * 100:.1f} %",
            "Release velocity": _fmt_speed(result.metrics["release_velocity"], imperial),
            "Release angle": f"{result.metrics['release_angle_deg']:.1f} deg",
            "Release height": _fmt_length(result.metrics["release_height"], imperial),
            "Time to release": f"{result.metrics['t_release']:.3f} s",
            "Flight time": f"{result.metrics.get('flight_time', 0.0):.2f} s",
            "Projectile KE at release": f"{result.metrics['ke_projectile']:.1f} J",
            "Total PE spent": f"{result.metrics['total_pe_spent']:.1f} J",
        }
    )

    st.markdown("**System**")
    _render_stat_table(
        {
            "Counterweight mass": _fmt_mass(params.counter_weight_mass, imperial),
            "Pulley radius": _fmt_length(params.pulley_radius, imperial),
            "Arm length": _fmt_length(params.arm_length, imperial),
            "String length": _fmt_length(params.string_length, imperial),
            "Release angle": f"{math.degrees(params.release_angle):.1f} deg",
            "Pivot height": _fmt_length(params.pivot_height, imperial),
            "Initial arm angle": f"{math.degrees(params.initial_arm_angle):.1f} deg",
            "CW rope length": _fmt_length(params.initial_cw_rope_length, imperial),
            "Projectile mass": _fmt_mass(params.projectile_mass, imperial),
            "Projectile radius": _fmt_length(params.projectile_radius, imperial),
            "Total mass": _fmt_mass(params.total_mass, imperial),
            "String/arm ratio": f"{params.string_arm_ratio:.3f}",
        }
    )


left, mid, right = st.columns([24, 52, 24])

with left:
    # Every input in this column stays visible with no scrolling; the optimizer
    # log at the bottom is the only element that flexes, absorbing whatever
    # vertical space the inputs leave over (see the opt_log_panel CSS).
    st.subheader("Fixed system parameters", help="Always used as given - the solver won't run without them.")
    grid3, grid4 = st.columns(2)
    pivot_height = _fixed_input(grid3, "Pivot height", "pivot_height", 0.1, 5.0, imperial=imperial)
    initial_arm_angle_deg = _fixed_input(
        grid4, "Initial arm angle", "initial_arm_angle", 5.0, 175.0, in_degrees=True
    )
    projectile_mass = _fixed_input(
        grid3, "Projectile mass", "projectile_mass", 0.001, 50.0, kind="mass", imperial=imperial
    )
    projectile_radius = _fixed_input(grid4, "Projectile radius", "projectile_radius", 0.001, 1.0, imperial=imperial)
    counter_weight_rope_length = _fixed_input_optional(
        grid3, "CW rope length", "counter_weight_rope_length", 0.001, 5.0, imperial=imperial,
        help="Rope from the pivot axle to the counterweight at t=0. Leave blank to "
        "default to 2x the pulley radius (one wrap).",
    )

    fixed_values = dict(
        pivot_height=pivot_height,
        initial_arm_angle=math.radians(initial_arm_angle_deg) if initial_arm_angle_deg is not None else None,
        projectile_mass=projectile_mass,
        projectile_radius=projectile_radius,
    )
    fixed_ready = all(v is not None for v in fixed_values.values())
    if not fixed_ready:
        st.error("All fixed system parameters are required.")

    # counter_weight_rope_length is optional (None = TrebuchetParams' own default), so
    # it's kept out of the fixed_ready/required check above but still passed through.
    fixed_params_all = dict(fixed_values, counter_weight_rope_length=counter_weight_rope_length)

    st.subheader(
        "Optimizable parameters",
        help="Leave a box blank to let the optimizer solve it (🔓 free); type a value to lock it (🔒 locked).",
    )
    grid1, grid2 = st.columns(2)
    optimizable_values = dict(
        counter_weight_mass=_optimizable_input(
            grid1, "Counterweight", "counter_weight_mass", kind="mass", imperial=imperial
        ),
        pulley_radius=_optimizable_input(grid2, "Pulley radius", "pulley_radius", imperial=imperial),
        arm_length=_optimizable_input(grid1, "Arm length", "arm_length", imperial=imperial),
        string_length=_optimizable_input(grid2, "String length", "string_length", imperial=imperial),
    )
    release_angle_deg = _optimizable_input(grid1, "Release angle", "release_angle", in_degrees=True)
    optimizable_values["release_angle"] = math.radians(release_angle_deg) if release_angle_deg is not None else None

    st.subheader(
        "Optimization target", help="Search for parameters that maximize launch efficiency at a target distance."
    )
    # Three columns keep this always-visible section at two input rows high;
    # a third row would push the buttons/log off-screen at 1366x768.
    saved_target = _load_user_defaults().get("target", {})
    grid5, grid6, grid7 = st.columns(3)
    target_min_m = 1.0
    target_default_m = max(float(saved_target.get("target_distance", 30.0)), target_min_m)
    if imperial:
        target_distance_ft = grid5.number_input(
            "Target (ft)",
            min_value=target_min_m / units.METERS_PER_FOOT,
            value=target_default_m / units.METERS_PER_FOOT,
            help="Target distance (ft)",
        )
        target_distance = target_distance_ft * units.METERS_PER_FOOT
    else:
        target_distance = grid5.number_input(
            "Target (m)", min_value=target_min_m, value=target_default_m,
            help="Target distance (m)",
        )
    efficiency_weight = grid6.number_input(
        "Eff. weight", min_value=0.0, value=max(float(saved_target.get("efficiency_weight", 5.0)), 0.0),
        help="Efficiency weight",
    )
    distance_weight = grid7.number_input(
        "Dist. weight", min_value=0.0, value=max(float(saved_target.get("distance_weight", 1.0)), 0.0),
        help="Distance weight",
    )
    population_size = grid5.number_input(
        "Population",
        min_value=5,
        max_value=500,
        value=min(max(int(saved_target.get("population_size", OptimizationConfig.population_size)), 5), 500),
        step=5,
        help="Differential-evolution population per free parameter. Larger searches more "
        "thoroughly; runtime grows proportionally.",
    )
    absolute_tolerance = grid6.number_input(
        "Tolerance",
        min_value=0.0,
        value=max(float(saved_target.get("absolute_tolerance", OptimizationConfig.absolute_tolerance)), 0.0),
        step=0.001,
        format="%.4f",
        help="Convergence tolerance on the final solution: the search stops once the "
        "population's objective spread falls below this. Smaller = more precise but slower.",
    )

    btn_sim, btn_opt, btn_save = st.columns([5, 5, 2])
    simulate_clicked = btn_sim.button("Simulate", type="primary", disabled=not fixed_ready, use_container_width=True)
    optimize_clicked = btn_opt.button("Optimize", disabled=not fixed_ready, use_container_width=True)
    if btn_save.button(
        "💾",
        help="Save the current fixed, optimizable, and optimization-target parameters as this dashboard's defaults",
        disabled=not fixed_ready,
        use_container_width=True,
    ):
        _save_user_defaults(
            optimizable_values,
            fixed_params_all,
            {
                "target_distance": target_distance,
                "efficiency_weight": efficiency_weight,
                "distance_weight": distance_weight,
                "population_size": int(population_size),
                "absolute_tolerance": absolute_tolerance,
            },
        )
        st.toast("Defaults saved")

    st.markdown("**Optimizer log**")
    with st.container(key="opt_log_panel", border=True):
        # Convergence outcome lives here in the log, not as a status banner.
        # Kept in its own (non-stretching) container so it takes only the
        # height its text needs, leaving the rest of the panel for the table.
        with st.container(key="opt_log_status"):
            opt_status_placeholder = st.empty()
            if st.session_state.opt_status:
                opt_status_placeholder.caption(st.session_state.opt_status)
        with st.container(key="opt_log_table"):
            log_placeholder = st.empty()
            if st.session_state.opt_log_rows:
                log_placeholder.dataframe(
                    st.session_state.opt_log_rows, hide_index=True, width="stretch", height=OPT_LOG_HEIGHT
                )
            else:
                log_placeholder.caption("Optimizer log will appear here.")

# Keyed so the CSS can float status messages over the animation instead of
# letting them push the animation/energy plots below the fold.
status_area = mid.container(key="status_area")

if simulate_clicked:
    sim_values = {
        name: (optimizable_values[name] if optimizable_values[name] is not None else DEFAULT_OPTIMIZABLE_PARAMS[name])
        for name in PARAM_NAMES
    }
    params = TrebuchetParams(**sim_values, **fixed_params_all)
    try:
        with st.spinner("Simulating..."):
            result = simulate_trebuchet(params, track_energy=True, simulate_aftermath=True)
        _store_result(params, result)
    except Exception as exc:  # surface a clean message, never a raw traceback
        status_area.error(f"Simulation failed: {exc}")

if optimize_clicked:
    locked = {name: value for name, value in optimizable_values.items() if value is not None}
    if len(locked) == len(PARAM_NAMES):
        status_area.error("At least one optimizable parameter must be left blank for the optimizer to search.")
    else:
        config = OptimizationConfig(
            target_distance=target_distance,
            efficiency_weight=efficiency_weight,
            distance_weight=distance_weight,
            population_size=int(population_size),
            absolute_tolerance=absolute_tolerance,
            locked_params=locked,
            fixed_params=fixed_params_all,
            workers=_OPT_WORKERS,
        )
        st.session_state.opt_log_rows = []
        st.session_state.opt_status = None
        opt_status_placeholder.empty()

        def _log_progress(generation, score, params, result) -> None:
            # Newest generation goes to the front: the grid always opens scrolled
            # to the top, so this keeps the latest row visible without the user
            # having to scroll down as more generations are appended.
            rows = st.session_state.opt_log_rows
            rows.insert(0, _log_row(generation, score, target_distance, result, imperial))
            log_placeholder.dataframe(rows, hide_index=True, width="stretch", height=OPT_LOG_HEIGHT)

        try:
            with st.spinner("Optimizing (this can take a minute)..."):
                optimal_params, opt_result, de_result = optimize_trebuchet(config, progress_callback=_log_progress)

            if de_result.success:
                st.session_state.opt_status = f"Optimization succeeded after {de_result.nfev} evaluations"
            else:
                st.session_state.opt_status = f"Optimization did not fully converge after {de_result.nfev} evaluations"
            opt_status_placeholder.caption(st.session_state.opt_status)

            _store_result(optimal_params, opt_result)
        except Exception as exc:
            status_area.error(f"Optimization failed: {exc}")

with mid:
    if st.session_state.anim_html is not None:
        # The animation panel flexes to fill the column (see CSS) so the energy
        # plots below it sit at the bottom of the screen.
        with st.container(key="anim_panel"):
            render_trebuchet_3d_html(st.session_state.anim_html, height=ANIMATION_HEIGHT)
        if st.session_state.energy_fig is not None:
            with st.container(key="energy_panel"):
                st.pyplot(st.session_state.energy_fig)
    else:
        st.info("Run a simulation or optimization to see the animation and energy graph.")

with right:
    with st.container(key="results_panel", border=True):
        if st.session_state.result is not None:
            _show_results(st.session_state.sim_params, st.session_state.result, imperial)
        else:
            st.info("Results will appear here after you run a simulation.")
