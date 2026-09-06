"""Streamlit web UI for the trebuchet simulator.

Run with: streamlit run src/trebuchet_sim/web/app.py
Or simply: python run.py   (from the repo root)

Layout goal: the whole dashboard fits on one screen with no scrolling, so the
CSS below compacts Streamlit's default paddings and the inputs sit in
two-per-row grids.
"""

import html
import json
import logging
import math
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Callable

import streamlit as st

# The optimizer's differential-evolution workers touch Streamlit internals from
# a background thread, which trips this logger's "missing ScriptRunContext"
# warning on every callback. Streamlit's own message says it's safe to ignore
# in this context (no per-thread script context outside the main run), so it's
# silenced rather than left spamming the console on every optimize run.
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)

from trebuchet_sim.web import theme, units
from trebuchet_sim.web.animation3d import build_trebuchet_3d_html, render_trebuchet_3d_html
from trebuchet_sim.config import (
    DEFAULT_INITIAL_ARM_ANGLE,
    DEFAULT_MACHINE_FIXED,
    DEFAULT_MACHINE_PARAMS,
    LINKAGE_PARAM,
    MachineType,
    TrebuchetParams,
)
from trebuchet_sim.optimization import (
    PARAM_BOUNDS,
    PARAM_LIMITS,
    OptimizationConfig,
    optimize_trebuchet,
    param_names,
)
from trebuchet_sim.physics import simulate_trebuchet
from trebuchet_sim.visualization import build_energy_figure

st.set_page_config(page_title="Trebuchet Simulator", page_icon="🏰", layout="wide")

st.markdown(theme.DASHBOARD_CSS, unsafe_allow_html=True)

title_col, unit_col = st.columns([5, 1])
title_col.markdown(
    '<div class="tb-appbar">'
    '<div class="tb-mark">🏰</div>'
    "<div><div class=\"tb-title\">Trebuchet Physics Simulator</div>"
    '<div class="tb-sub">Euler-Lagrange launch dynamics, ballistic flight, and parameter search</div>'
    "</div></div>",
    unsafe_allow_html=True,
)
imperial = unit_col.toggle(
    "Imperial units", key="imperial_units", help="Show lengths in ft/in, masses in lb, speeds in ft/s."
)

ANIMATION_HEIGHT = 440

# Decimal places every number box shows. Also what "the user changed this" means for a
# search range: a value is compared against its default as displayed, because a default
# that has been through a unit conversion and back no longer equals itself in SI.
INPUT_DECIMALS = 4

# What each design variable measures, for picking its display unit.
PARAM_KIND = {
    "counter_weight_mass": "mass",
    "release_angle": "angle",
    "pulley_radius": "length",
    "length_counterweight": "length",
    "arm_length": "length",
    "string_length": "length",
}

# Fixed system params: required, always used as given, never handed to the optimizer.
# Defaults come straight from the TrebuchetParams dataclass so they can't drift.
FIXED_PARAM_NAMES = ("pivot_height", "initial_arm_angle", "projectile_mass", "projectile_radius")
_DATACLASS_FIXED_DEFAULTS = {f.name: f.default for f in fields(TrebuchetParams) if f.name in FIXED_PARAM_NAMES}


def _fixed_defaults(machine: MachineType) -> dict:
    """Defaults for the fixed system parameters, for one machine.

    Dataclass defaults, overlaid with whatever this machine needs different (a
    traditional machine wants a much taller pivot, so its cocked tip clears the
    ground by a sling length).

    initial_arm_angle needs the explicit fill-in: its dataclass default is None
    because the cocked position only resolves per machine in __post_init__, and a
    box with no default renders blank, which the solver reads as "not ready". A
    saved user_defaults.json used to hide that, so it showed up first on a fresh
    install - an empty TREBUCHET_DATA_DIR, i.e. a newly created container volume.
    """
    defaults = dict(_DATACLASS_FIXED_DEFAULTS)
    defaults["initial_arm_angle"] = float(DEFAULT_INITIAL_ARM_ANGLE[machine])
    defaults.update(
        {name: value for name, value in DEFAULT_MACHINE_FIXED[machine].items() if name in FIXED_PARAM_NAMES}
    )
    return defaults


# The cocked arm sits on opposite sides of vertical on the two machines - the pulley
# arm starts raised behind the pivot, the traditional one nose-down in front of it - so
# one signed range cannot serve both without also admitting the poses that don't throw.
INITIAL_ARM_ANGLE_BOUNDS = {
    MachineType.PULLEY: (math.radians(5.0), math.radians(175.0)),
    MachineType.TRADITIONAL: (math.radians(-175.0), math.radians(-5.0)),
}

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


# Section names inside one machine's saved block. `optimizable` maps each parameter to
# {"value": <SI float>, "locked": <bool>} - the box's raw content and the lock toggle's
# own state are saved separately (see _optimizable_input) so switching the toggle off and
# back on later restores the last value instead of resetting it. `ranges` maps each to
# {"min": <SI float>, "max": <SI float>}, the search bounds. `target` holds the optimizer
# target and its search weights.
_SAVED_SECTIONS = ("optimizable", "ranges", "fixed", "target")


def _saved_machines() -> dict:
    """Every machine's saved block, keyed by machine value.

    Also the migration point for the original one-machine file, which held the sections
    at the top level next to a single `machine` key. Such a file is read as that
    machine's block and left alone on disk; the next save rewrites it in this shape,
    which is why the legacy keys are only consulted when `machines` is absent.
    """
    defaults = _load_user_defaults()
    saved = defaults.get("machines")
    if isinstance(saved, dict):
        return saved
    legacy = {section: defaults[section] for section in _SAVED_SECTIONS if section in defaults}
    if not legacy:
        return {}
    try:
        owner = MachineType(defaults.get("machine", MachineType.PULLEY))
    except ValueError:
        owner = MachineType.PULLEY
    return {owner.value: legacy}


def _save_user_defaults(
    machine: MachineType, optimizable: dict, ranges: dict, fixed: dict, target: dict
) -> None:
    """Persist the current inputs as this machine's defaults.

    Each machine keeps its own block, and saving one leaves the other's untouched: a
    0.4 m arm is a good pulley machine and a useless traditional one, so the two sets
    are not interchangeable and never were (see _saved_for) - but until they were stored
    separately, saving either one also threw the other away. The top-level `machine` is
    which was saved last, and only decides which the dashboard opens on.
    """
    machines = dict(_saved_machines())
    machines[machine.value] = {
        "optimizable": optimizable,
        "ranges": ranges,
        "fixed": fixed,
        "target": target,
    }
    defaults = {"machine": machine.value, "machines": machines}
    USER_DEFAULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    USER_DEFAULTS_FILE.write_text(json.dumps(defaults, indent=2))
    st.session_state.user_defaults = defaults


def _saved_machine() -> MachineType:
    """The machine saved last, which the dashboard opens on; pulley when there are none."""
    try:
        return MachineType(_load_user_defaults().get("machine", MachineType.PULLEY))
    except ValueError:  # unrecognized value in a hand-edited file
        return MachineType.PULLEY


def _saved_for(machine: MachineType, section: str) -> dict:
    """One section of one machine's saved defaults; {} when that machine has none."""
    block = _saved_machines().get(machine.value)
    return block.get(section, {}) if isinstance(block, dict) else {}


def _unit_dependent_inputs() -> "list[tuple[str, str]]":
    """(session-state key, quantity kind) for every single-box input whose stored number
    means something different in each unit system.

    Metric lengths sit in one box and imperial ones in a `_ft`/`_in` pair, so those
    widgets reseed themselves across the toggle and need no help. The boxes listed here
    are the ones that keep a single widget in both systems - masses, and the search-range
    boxes, which use decimal feet rather than a ft+in pair (a range is a coarse thing,
    and the same choice the optimizer log and the headline readout already make). Without
    conversion such a box would keep showing the metric figure under an imperial label,
    and then be read back as if it were imperial.

    Both machines' keys are listed: only one machine is on screen at a time, but the
    other's stored value still has to be in the system it will be read back in.
    """
    entries = []
    for machine in MachineType:
        entries.append((_widget_key("opt", "counter_weight_mass", machine), "mass"))
        entries.append((_widget_key("fixed", "projectile_mass", machine), "mass"))
        for name in param_names(machine):
            if PARAM_KIND[name] == "angle":
                continue  # angles are degrees in both systems
            for prefix in ("rmin", "rmax"):
                entries.append((_widget_key(prefix, name, machine), PARAM_KIND[name]))
    return entries


def _sync_unit_inputs(imperial: bool) -> None:
    """Convert the single-box unit-dependent inputs in place when the toggle flips.

    Runs before any input renders, so the widgets below see the converted value.
    """
    previous = st.session_state.get("_units_were_imperial")
    if previous is None:
        st.session_state["_units_were_imperial"] = imperial
        return
    if previous == imperial:
        return

    convert = {
        "mass": units.kg_to_lb if imperial else units.lb_to_kg,
        "length": (
            (lambda v: v / units.METERS_PER_FOOT) if imperial else (lambda v: v * units.METERS_PER_FOOT)
        ),
    }
    for key, kind in _unit_dependent_inputs():
        value = st.session_state.get(key)
        if isinstance(value, (int, float)):
            st.session_state[key] = convert[kind](float(value))
    st.session_state["_units_were_imperial"] = imperial


def _widget_key(prefix: str, name: str, machine: MachineType) -> str:
    """Session-state key for one parameter's input on one machine.

    Every input is scoped to its machine, so the two machines hold separate boxes and
    switching between them reseeds from that machine's defaults - the same trick the
    length inputs use across the unit toggle, and for the same reason: Streamlit keeps
    a keyed widget's value across reruns, so a box can only be re-defaulted by becoming
    a different widget. It also means edits to one machine survive a look at the other.
    """
    return f"{prefix}_{name}_{machine.value}"

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
    st.session_state.energy_fig = (
        build_energy_figure(result, compact=True, dark=True) if result.energy_history else None
    )


def _length_pair_input(
    container, label: str, key_prefix: str, default_m: "float | None",
    min_value: float, max_value: "float | None" = None, help: str = None,
) -> "float | None":
    """Feet+inches sub-widget pair for a length field in Imperial mode.

    Rendered as two columns inside the caller's existing grid cell, so it keeps
    the same row height as a single metric box rather than growing the layout.
    Both boxes start pre-filled with default_m's feet/inches split (blank if
    default_m is None), but can always be cleared back to blank afterwards:
    Streamlit only allows an empty number_input when its `value` argument is
    None, so the initial fill goes through session_state.setdefault() instead
    of `value=`, and `value=None` is passed on every call - otherwise a
    concrete `value=` re-passed on each rerun would make the box snap back to
    it whenever the user tried to clear it. The combined value counts as set
    as soon as EITHER box has a value, treating the other as 0 - a user who
    only types into the inches box (e.g. a 6" pulley radius) must not be
    silently ignored just because they left feet untouched. Returns the
    combined value in meters (SI), clamped into [min_value, max_value], or
    None if both boxes are blank.
    """
    feet_default = inches_default = None
    if default_m is not None:
        clamped = max(default_m, min_value)
        if max_value is not None:
            clamped = min(clamped, max_value)
        feet_default, inches_default = units.meters_to_feet_inches(clamped)

    feet_max = math.ceil(max_value / units.METERS_PER_FOOT) + 1 if max_value is not None else None

    ft_key, in_key = f"{key_prefix}_ft", f"{key_prefix}_in"
    if feet_default is not None:
        st.session_state.setdefault(ft_key, feet_default)
    if inches_default is not None:
        st.session_state.setdefault(in_key, inches_default)

    # Keyed so the stylesheet can reclaim width inside the pair (see the
    # st-key-pair_ rule); the two boxes are the narrowest inputs on the page.
    sub_feet, sub_inches = container.container(key=f"pair_{key_prefix}").columns(2)
    feet_val = sub_feet.number_input(
        f"{label} (ft)", min_value=0, max_value=feet_max, value=None,
        key=ft_key, step=1, help=help,
    )
    inches_val = sub_inches.number_input(
        "(in)", min_value=0.0, max_value=11.99, value=None,
        key=in_key, step=0.1, format="%.2f",
    )

    if feet_val is None and inches_val is None:
        return None
    combined = max(units.feet_inches_to_meters(feet_val, inches_val), min_value)
    return min(combined, max_value) if max_value is not None else combined


# --- Unit-aware parameter inputs ---------------------------------------------
#
# Every parameter box is "one number plus a conversion between canonical SI
# (m / kg / rad) and whatever the box displays". The single exception is an
# imperial length, which is edited as a feet box plus an inches box.
# `_DisplayUnit` captures both cases, so the three input builders below differ
# only in where their default comes from and whether they carry a lock toggle -
# none of them does any unit arithmetic of its own.
#
# Everything here takes and returns SI. Keeping the conversion in exactly one
# place is what stops a display value from reaching the physics: an earlier
# version returned radians from a helper whose callers then applied
# math.radians() a second time, so locking the release angle silently simulated
# a machine ~57x off the number in the box.


def _identity(value: float) -> float:
    return value


@dataclass(frozen=True)
class _DisplayUnit:
    """How one physical quantity is shown in the active unit system."""

    suffix: str                                    # widget label suffix, e.g. "kg"
    to_display: "Callable[[float], float]" = _identity   # SI -> box contents
    to_si: "Callable[[float], float]" = _identity        # box contents -> SI
    is_pair: bool = False                          # imperial length: ft + in boxes


def _display_unit(kind: str, imperial: bool) -> _DisplayUnit:
    """Display unit for `kind` ("length", "mass" or "angle") in the active system."""
    if kind == "angle":
        return _DisplayUnit("deg", math.degrees, math.radians)  # always degrees
    if kind == "mass":
        return _DisplayUnit("lb", units.kg_to_lb, units.lb_to_kg) if imperial else _DisplayUnit("kg")
    if kind == "length":
        return _DisplayUnit("ft", is_pair=True) if imperial else _DisplayUnit("m")
    raise ValueError(f"unknown quantity kind: {kind!r}")


def _range_unit(kind: str, imperial: bool) -> _DisplayUnit:
    """Display unit for a search-range box: like _display_unit, but never a ft+in pair.

    A range end is a coarse bound, not a measurement to build to, and splitting each of
    ten boxes into feet and inches would double the popover's width for no precision
    anyone needs. Decimal feet is the same call the optimizer log and the headline range
    readout already make (see _log_row and _split_length).
    """
    if kind == "length" and imperial:
        return _DisplayUnit(
            "ft",
            to_display=lambda metres: metres / units.METERS_PER_FOOT,
            to_si=lambda feet: feet * units.METERS_PER_FOOT,
        )
    return _display_unit(kind, imperial)


def _si_input(
    container, label: str, key: str, unit: _DisplayUnit,
    si_default: "float | None", si_min: float, si_max: float,
    clearable: bool, help: str = None,
) -> "float | None":
    """One parameter box (or ft/in pair), in canonical SI and out again.

    `clearable` picks between Streamlit's two number_input modes. True routes
    the initial value through session_state and passes `value=None` on every
    rerun, which is the only way Streamlit lets a box be emptied - blank is
    meaningful for the optimizable params (nothing to lock to) and for the
    optional ones. False passes the default as `value=`, so the box always
    holds a number; the required fixed params use that.

    Returns None only when the box is blank.
    """
    if unit.is_pair:
        return _length_pair_input(container, label, key, si_default, si_min, si_max, help=help)

    lo, hi = unit.to_display(si_min), unit.to_display(si_max)
    default = None if si_default is None else min(max(unit.to_display(si_default), lo), hi)

    if clearable:
        if default is not None:
            st.session_state.setdefault(key, default)
        default = None

    shown = container.number_input(
        f"{label} ({unit.suffix})", min_value=lo, max_value=hi, value=default,
        key=key, format=f"%.{INPUT_DECIMALS}f", help=help,
    )
    return None if shown is None else unit.to_si(shown)


def _optimizable_input(
    container, label: str, name: str, machine: MachineType,
    kind: str = "length", imperial: bool = False, help: str = None,
) -> "tuple[float | None, float, bool]":
    """Number input for an optimizable parameter, paired with a lock toggle.

    The toggle is the sole source of truth for locked/free - typing into the
    box no longer implies locking it. On: the optimizer/simulator use
    whatever is currently in the box. Off: they disregard the box entirely
    and treat the parameter as free, even if the box holds a leftover number.
    The box itself can always be cleared back to blank; a locked-but-blank box
    has nothing to lock to, so it's treated the same as free.

    Returns (effective, raw, locked), all in canonical SI:
      - effective: the value when locked and the box isn't blank, else None -
        what the rest of the app expects for simulate/optimize (None = free).
      - raw: the box's current content, or a fallback default if blank - saved
        so flipping the toggle back on (or just typing again) later restores
        something sensible instead of nothing.
      - locked: the toggle's own state, also saved so reloading remembers
        which params were locked.
    """
    unit = _display_unit(kind, imperial)
    si_min, si_max = PARAM_BOUNDS[name]

    saved_entry = _saved_for(machine, "optimizable").get(name)
    if not isinstance(saved_entry, dict):  # stale/pre-toggle save format - fall back to un-locked defaults
        saved_entry = {}
    saved_locked = bool(saved_entry.get("locked", False))
    # Fallback used both to pre-fill the box and to fall back on if the box gets
    # cleared - clamped in case bounds changed since the value was saved.
    fallback_si = saved_entry.get("value")
    if fallback_si is None:
        fallback_si = DEFAULT_MACHINE_PARAMS[machine][name]
    fallback_si = min(max(fallback_si, si_min), si_max)

    # Keyed wrapper so the stylesheet can tint the whole row when the lock is
    # on (see the [class*="st-key-param_"] rule) - a 20px toggle on its own is
    # too small to read a column of five locks from.
    row = container.container(key=_widget_key("param", name, machine))
    lock_col, box_col = row.columns([1, 7])
    # Spacer matches the input's label row height so the toggle lines up with
    # the box itself, not the label above it.
    lock_col.markdown("<div style='height:1.0rem'></div>", unsafe_allow_html=True)
    locked = lock_col.toggle(
        f"Lock {label}", key=_widget_key("lock", name, machine), value=saved_locked, label_visibility="collapsed",
        help="On: lock this parameter to the value in the box. Off: leave it free for the optimizer to search.",
    )

    value_si = _si_input(
        box_col, label, _widget_key("opt", name, machine), unit, fallback_si, si_min, si_max,
        clearable=True, help=help
    )
    raw = fallback_si if value_si is None else value_si
    effective = value_si if locked else None
    return effective, raw, locked


def _range_input(container, label: str, name: str, machine: MachineType, imperial: bool) -> tuple:
    """Min/max boxes for one design variable's search range, in canonical SI.

    Returns (bounds, is_custom): `bounds` is always a usable (lo, hi) pair - the default
    when the boxes are untouched - and `is_custom` says whether the user moved either end,
    which is what the "Ranges" button counts.

    Bounds are clamped to PARAM_LIMITS rather than PARAM_BOUNDS, so the search can be
    widened past the defaults as well as narrowed; OptimizationConfig enforces the same
    envelope, and min >= max is corrected here rather than raised, since half of every
    edit passes through that state as the user types.
    """
    unit = _range_unit(PARAM_KIND[name], imperial)
    default_lo, default_hi = PARAM_BOUNDS[name]
    limit_lo, limit_hi = PARAM_LIMITS[name]

    saved = _saved_for(machine, "ranges").get(name) or {}
    seed_lo = saved.get("min", default_lo)
    seed_hi = saved.get("max", default_hi)

    lo_col, hi_col = container.columns(2)
    low = _si_input(
        lo_col, f"{label} min", _widget_key("rmin", name, machine), unit,
        seed_lo, limit_lo, limit_hi, clearable=False,
    )
    high = _si_input(
        hi_col, "max", _widget_key("rmax", name, machine), unit,
        seed_hi, limit_lo, limit_hi, clearable=False,
    )

    low = default_lo if low is None else low
    high = default_hi if high is None else high
    if low >= high:
        # Mid-edit state, not an error: keep the search space non-empty by falling back
        # to the default span rather than refusing to run.
        low, high = default_lo, default_hi

    # Compared as displayed, not in SI. A box seeded with the default and left alone
    # still round-trips through the unit conversion at the box's own precision, so in
    # imperial a 0.1 m default comes back as 0.10000488 m - equal on screen, unequal in
    # SI, and an exact test would report every length as customized.
    def shown(value: float) -> float:
        return round(unit.to_display(value), INPUT_DECIMALS)

    is_custom = (shown(low), shown(high)) != (shown(default_lo), shown(default_hi))
    return (low, high), is_custom


def _fixed_input(
    container, label: str, name: str, si_min: float, si_max: float, machine: MachineType,
    kind: str = "length", imperial: bool = False,
) -> "float | None":
    """Number input for a required fixed (never-optimized) system parameter.

    Bounds and return value are canonical SI. Returns None only when the box is
    blank, which the caller treats as "not ready to solve".
    """
    default = _saved_for(machine, "fixed").get(name)
    if default is None:
        default = _fixed_defaults(machine)[name]
    default = min(max(default, si_min), si_max)
    unit = _display_unit(kind, imperial)
    return _si_input(
        container, label, _widget_key("fixed", name, machine), unit, default, si_min, si_max, clearable=False
    )


def _fixed_input_optional(
    container, label: str, name: str, si_min: float, si_max: float, machine: MachineType,
    kind: str = "length", imperial: bool = False, help: str = None,
) -> "float | None":
    """Number input for a fixed system parameter that may be left blank.

    Unlike _fixed_input, blank is a legal value here (not "still typing") - it
    means TrebuchetParams should fall back to its own computed default.
    """
    saved = _saved_for(machine, "fixed").get(name)
    if saved is None:
        saved = DEFAULT_MACHINE_FIXED[machine].get(name)
    if saved is not None:
        saved = min(max(saved, si_min), si_max)
    unit = _display_unit(kind, imperial)
    return _si_input(
        container, label, _widget_key("fixed", name, machine), unit, saved, si_min, si_max,
        clearable=True, help=help
    )


def _fmt_length(value_m: float, imperial: bool) -> str:
    """Feet+inches split for the results tables (see _log_row for why the
    optimizer log uses plain decimal feet instead)."""
    if not imperial:
        return f"{value_m:.4f} m"
    feet, inches = units.meters_to_feet_inches(value_m)
    return f"{feet} ft {inches:.2f} in"


def _split_length(value_m: float, imperial: bool) -> "tuple[str, str]":
    """Length as a (number, unit) pair for the headline readout.

    Decimal feet rather than the feet+inches split `_fmt_length` uses: the hero
    number is meant to be read at a glance, and a second unit mid-number breaks
    that (the exact ft+in figure is still in the machine spec list below).
    """
    if imperial:
        return f"{value_m / units.METERS_PER_FOOT:.2f}", "ft"
    return f"{value_m:.2f}", "m"


def _fmt_delta(delta_m: float, target_m: float, imperial: bool) -> str:
    """HTML note pairing the shot with the target it was aimed at."""
    unit = "ft" if imperial else "m"
    scale = units.METERS_PER_FOOT if imperial else 1.0
    target_disp = target_m / scale
    miss = abs(delta_m) / scale
    if miss < 0.005:
        return f"Target <b>{target_disp:.2f} {unit}</b> &middot; <b>on target</b>"
    direction = "over" if delta_m > 0 else "short"
    return f"Target <b>{target_disp:.2f} {unit}</b> &middot; <b>{miss:.2f} {unit} {direction}</b>"


def _fmt_mass(value_kg: float, imperial: bool) -> str:
    return f"{units.kg_to_lb(value_kg):.3f} lb" if imperial else f"{value_kg:.3f} kg"


def _fmt_speed(value_mps: float, imperial: bool) -> str:
    return f"{units.mps_to_fps(value_mps):.2f} ft/s" if imperial else f"{value_mps:.2f} m/s"


def _section(title: str, hint: str = "") -> None:
    """Small uppercase rule-header used to separate panel sections.

    `hint` becomes a native title tooltip rather than a Streamlit help icon:
    the dashboard is height-constrained, and a tooltip costs no vertical space.
    """
    attr = f' title="{html.escape(hint, quote=True)}"' if hint else ""
    st.markdown(f'<div class="tb-section"{attr}>{html.escape(title)}</div>', unsafe_allow_html=True)


def _empty_state(mark: str, title: str, hint: str) -> str:
    """Designed placeholder for a panel with nothing to show yet."""
    return (
        f'<div class="tb-empty"><div class="tb-empty-mark">{mark}</div>'
        f'<div class="tb-empty-title">{html.escape(title)}</div>'
        f'<div class="tb-empty-hint">{hint}</div></div>'
    )


def _metric_grid(cells: dict) -> None:
    """Two-column grid of small label/value metric cards.

    Replaces the old chunked `st.table` readout: a table forced every value
    into a fixed-width grid with wrapped uppercase headers, which scanned
    poorly and could not give the headline numbers any more weight than
    `String/arm ratio`.
    """
    body = "".join(
        f'<div class="tb-cell"><span class="tb-k">{html.escape(key)}</span>'
        f'<span class="tb-v">{html.escape(value)}</span></div>'
        for key, value in cells.items()
    )
    st.markdown(f'<div class="tb-grid">{body}</div>', unsafe_allow_html=True)


def _spec_list(specs: dict) -> None:
    """Dense label/value list for the machine's resolved geometry."""
    body = "".join(
        f'<div class="tb-spec"><span>{html.escape(key)}</span><b>{html.escape(value)}</b></div>'
        for key, value in specs.items()
    )
    st.markdown(f'<div class="tb-specs">{body}</div>', unsafe_allow_html=True)


def _show_results(params: TrebuchetParams, result, imperial: bool, target_distance: float) -> None:
    if "error" in result.metrics:
        st.error(result.metrics["error"])
        return

    if not result.metrics.get("release_occurred", True):
        if result.metrics.get("arm_ground_contact"):
            st.warning(
                f"No release - the arm reached the ground "
                f"{result.metrics.get('total_rotation_deg', 0.0):.0f}° into the throw and the "
                "launch ended there. The beam is longer than the pivot is tall, so it cannot "
                "swing past the bottom: raise the pivot height above the arm length, or shorten "
                "the arm."
            )
        else:
            st.warning(
                "No release - the arm never reached the release angle within the simulation window."
            )
        _metric_grid(
            {
                "Simulation time": f"{result.metrics['simulation_time']:.2f} s",
                "Final arm angle": f"{result.metrics['final_arm_angle_deg']:.1f} deg",
                "Total rotation": f"{result.metrics['total_rotation_deg']:.1f} deg",
            }
        )
        return

    snap_energy = result.metrics.get("sling_snap_energy", 0.0)
    slack_fraction = result.metrics.get("string_slack_fraction", 0.0)
    tension_deficit = result.metrics.get("sling_tension_deficit", 0.0)
    if slack_fraction > 1e-3 or snap_energy > 1e-3:
        total_pe = result.metrics.get("total_pe_spent", 0.0)
        loss_share = f" ({snap_energy / total_pe * 100:.0f}% of the PE spent)" if total_pe > 0 else ""
        st.warning(
            f"Sling goes slack for {slack_fraction * 100:.0f}% of the launch: the projectile "
            f"flies detached and snaps taut {result.metrics.get('sling_snap_count', 0)} time(s), "
            f"dissipating {snap_energy:.1f} J{loss_share}. The simulated flight is physical, "
            "but a smoother geometry would keep that energy."
        )
    # Only when the sling held: a run that actually detached has already said so above,
    # and two warnings about the same rope would cost a line the one-screen layout
    # hasn't got. 0.01 is the floor below which the two engines can't tell a marginal
    # launch from a clean one (see tests/test_fastsim.py).
    elif tension_deficit > 0.01:
        st.warning(
            f"Sling stays taut but runs marginal for {tension_deficit * 100:.0f}% of the launch: "
            "it never detaches here, but it is close enough that a small change in the build "
            "would make it snap. Raising the snap penalty when optimizing trades a little range "
            "for a sling that stays loaded."
        )
    # 0.05 N*s is the integration-noise floor for the rigid-link counterweight rope.
    if result.metrics.get("cw_rope_compression_impulse", 0.0) > 0.05:
        st.warning(
            f"Counterweight rope goes slack (min tension "
            f"{result.metrics.get('min_cw_rope_tension', 0.0):.1f} N): the arm out-accelerates the "
            "falling counterweight, so the results are not physical."
        )

    # Headline range, with how far it landed from the target the optimizer is
    # aiming at - the single number most runs are judged by.
    delta = result.distance - target_distance
    range_value, range_unit = _split_length(result.distance, imperial)
    st.markdown(
        '<div class="tb-hero">'
        '<div class="tb-hero-label">Range</div>'
        f'<div class="tb-hero-value">{html.escape(range_value)}'
        f'<span class="tb-unit">{html.escape(range_unit)}</span></div>'
        f'<div class="tb-hero-note">{_fmt_delta(delta, target_distance, imperial)}</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    efficiency_pct = result.efficiency * 100
    st.markdown(
        '<div class="tb-bar-row"><span>Efficiency</span>'
        f"<b>{efficiency_pct:.1f}%</b></div>"
        f'<div class="tb-bar"><div class="tb-bar-fill" style="width:{min(max(efficiency_pct, 0), 100):.1f}%"></div></div>',
        unsafe_allow_html=True,
    )

    _metric_grid(
        {
            "Release velocity": _fmt_speed(result.metrics["release_velocity"], imperial),
            "Release angle": f"{result.metrics['release_angle_deg']:.1f} deg",
            "Release height": _fmt_length(result.metrics["release_height"], imperial),
            "Time to release": f"{result.metrics['t_release']:.3f} s",
            "Flight time": f"{result.metrics.get('flight_time', 0.0):.2f} s",
            "Projectile KE": f"{result.metrics['ke_projectile']:.1f} J",
            "Total PE spent": f"{result.metrics['total_pe_spent']:.1f} J",
            "Min sling tension": f"{result.metrics.get('min_string_tension', float('nan')):.1f} N",
        }
    )

    _section("Machine")
    # Only the linkage this machine actually has: the other's parameter is carried on
    # TrebuchetParams but unused, so listing it would be reporting a number that had no
    # effect on the run above.
    linkage_spec = (
        {"Pulley radius": _fmt_length(params.pulley_radius, imperial)}
        if params.has_pulley
        else {"CW arm length": _fmt_length(params.length_counterweight, imperial)}
    )
    _spec_list(
        {
            "Machine": "Pulley" if params.has_pulley else "Traditional",
            "Counterweight mass": _fmt_mass(params.counter_weight_mass, imperial),
            **linkage_spec,
            "Arm length": _fmt_length(params.arm_length, imperial),
            "String length": _fmt_length(params.string_length, imperial),
            "String/arm ratio": f"{params.string_arm_ratio:.3f}",
            "Release angle": f"{math.degrees(params.release_angle):.1f} deg",
            "Pivot height": _fmt_length(params.pivot_height, imperial),
            "Initial arm angle": f"{math.degrees(params.initial_arm_angle):.1f} deg",
            "CW rope length": _fmt_length(params.initial_cw_rope_length, imperial),
            "Projectile mass": _fmt_mass(params.projectile_mass, imperial),
            "Projectile radius": _fmt_length(params.projectile_radius, imperial),
            "Total mass": _fmt_mass(params.total_mass, imperial),
        }
    )


# Must run before any input renders, so the widgets below see converted values.
_sync_unit_inputs(imperial)

left, mid, right = st.columns([24, 52, 24])

with left:
    # Every input in this column stays visible with no scrolling; the optimizer
    # log at the bottom is the only element that flexes, absorbing whatever
    # vertical space the inputs leave over (see the opt_log_panel CSS).
    _section("Machine", "Pick the counterweight linkage, then the fixed geometry it hangs on.")
    # Rendered before every other input, because it decides their defaults, bounds and
    # widget keys (see _widget_key).
    machine = MachineType(
        st.segmented_control(
            "Machine type",
            options=[m.value for m in MachineType],
            format_func=lambda value: {"pulley": "Pulley", "traditional": "Traditional"}[value],
            default=_saved_machine().value,
            key="machine_type",
            label_visibility="collapsed",
            help="Pulley: the counterweight hangs from a rope over the pivot axle and drops "
            "straight down. Traditional: it is bolted to the arm's short end and swings with "
            "it. Switching reloads that machine's default geometry.",
        )
        # segmented_control returns None if the active pill is clicked again; keep the
        # machine we already had rather than leaving the page with no machine at all.
        or st.session_state.get("_machine_was", MachineType.PULLEY.value)
    )
    angle_min, angle_max = INITIAL_ARM_ANGLE_BOUNDS[machine]

    grid3, grid4 = st.columns(2)
    pivot_height = _fixed_input(grid3, "Pivot height", "pivot_height", 0.1, 5.0, machine, imperial=imperial)
    initial_arm_angle = _fixed_input(
        grid4, "Initial arm angle", "initial_arm_angle", angle_min, angle_max, machine, kind="angle",
    )
    projectile_mass = _fixed_input(
        grid3, "Projectile mass", "projectile_mass", 0.001, 50.0, machine, kind="mass", imperial=imperial
    )
    projectile_radius = _fixed_input(
        grid4, "Projectile radius", "projectile_radius", 0.001, 1.0, machine, imperial=imperial
    )
    counter_weight_rope_length = _fixed_input_optional(
        grid3, "CW rope length", "counter_weight_rope_length", 0.001, 5.0, machine, imperial=imperial,
        help=(
            "Rope from the pivot axle to the counterweight at t=0. Leave blank to "
            "default to 2x the pulley radius (one wrap)."
            if machine is MachineType.PULLEY
            else "Link from the pin on the arm's short end to the counterweight. The weight "
            "swings on it, so a longer link swings more slowly."
        ),
    )

    fixed_values = dict(
        pivot_height=pivot_height,
        initial_arm_angle=initial_arm_angle,
        projectile_mass=projectile_mass,
        projectile_radius=projectile_radius,
    )
    fixed_ready = all(v is not None for v in fixed_values.values())
    if not fixed_ready:
        st.error("All fixed system parameters are required.")

    # counter_weight_rope_length is optional (None = TrebuchetParams' own default), so
    # it's kept out of the fixed_ready/required check above but still passed through.
    fixed_params_all = dict(fixed_values, counter_weight_rope_length=counter_weight_rope_length)

    _section("Design variables", "Lock a parameter to pin it to the value in its box; unlocked leaves it free for the optimizer to search.")
    grid1, grid2 = st.columns(2)
    # The linkage row is the one design variable that differs between the machines
    # (config.LINKAGE_PARAM): a pulley radius on one, the short arm's length on the
    # other. Both have their own widget keys, so each machine keeps its own box value
    # across a switch.
    linkage = LINKAGE_PARAM[machine]
    linkage_label, linkage_help = {
        "pulley_radius": ("Pulley radius", "How far the counterweight falls per radian of arm rotation."),
        "length_counterweight": ("CW arm length", "How far the counterweight sits behind the pivot."),
    }[linkage]

    # (container, label, name, kind) per row, laid out down the two columns.
    rows = [
        (grid1, "Counterweight", "counter_weight_mass", "mass", None),
        (grid2, linkage_label, linkage, "length", linkage_help),
        (grid1, "Arm length", "arm_length", "length", None),
        (grid2, "String length", "string_length", "length", None),
        (grid1, "Release angle", "release_angle", "angle", None),
    ]
    optimizable_values, optimizable_raw, optimizable_locked = {}, {}, {}
    for container, label, name, kind, help_text in rows:
        value, raw, is_locked = _optimizable_input(
            container, label, name, machine, kind=kind, imperial=imperial, help=help_text
        )
        # value is None when the parameter is free; raw is always concrete, and both it
        # and the lock state are saved by the 💾 button so unlocking doesn't lose the
        # number and reloading remembers which params were locked.
        optimizable_values[name] = value
        optimizable_raw[name] = raw
        optimizable_locked[name] = is_locked

    # Search ranges live in a popover rather than inline: two more boxes per row would
    # not fit this column (each design-variable box is already only ~130px wide), and
    # grouping them makes the whole search space readable at once. The button carries a
    # count so a narrowed search is still visible without opening it.
    param_ranges = {}
    custom_ranges = []
    with st.popover(
        "Search ranges", use_container_width=True,
        help="Bounds the optimizer searches between. Locked parameters ignore theirs.",
    ):
        st.caption("Where the optimizer may look. Locked parameters are pinned, so their range is unused.")
        for _container, label, name, _kind, _help_text in rows:
            bounds, is_custom = _range_input(st, label, name, machine, imperial)
            param_ranges[name] = bounds
            if is_custom:
                custom_ranges.append(name)
    if custom_ranges:
        st.caption(f"{len(custom_ranges)} custom range{'s' if len(custom_ranges) > 1 else ''}")

    _section("Optimizer target", "Search for parameters that maximize launch efficiency at a target distance.")
    # Target is the knob that changes per run, so it stays on the surface. The
    # four search-tuning values are set-once settings and live in a popover:
    # inline they cost a second input row, and that row is the difference
    # between a usable optimizer log and a two-line one - the log is the panel
    # that absorbs whatever vertical space the inputs leave (see theme.py).
    # Popover children execute on every rerun, so the values below are always
    # current whether or not it happens to be open.
    # Scoped to the machine like every other saved section: the two reach different
    # distances (a plain pulley machine tops out around 120 m, a traditional one around
    # 100 m) and want different weights to get there, so one target carried across would
    # be as wrong as one arm length. The boxes below are keyed per machine for the same
    # reason the parameter boxes are - a keyed widget survives a rerun, so it can only
    # be re-seeded by becoming a different widget (see _widget_key).
    saved_target = _saved_for(machine, "target")
    target_col, tuning_col = st.columns([2, 1])
    target_min_m = 1.0
    target_default_m = max(float(saved_target.get("target_distance", 30.0)), target_min_m)
    if imperial:
        target_distance_ft = target_col.number_input(
            "Target (ft)",
            min_value=target_min_m / units.METERS_PER_FOOT,
            value=target_default_m / units.METERS_PER_FOOT,
            # Separate key per unit as well as per machine, so the box reseeds across the
            # units toggle instead of showing feet under a metres label - the same split
            # the length inputs make (see _unit_dependent_inputs).
            key=_widget_key("target", "distance_ft", machine),
            help="Target distance (ft)",
        )
        target_distance = target_distance_ft * units.METERS_PER_FOOT
    else:
        target_distance = target_col.number_input(
            "Target (m)", min_value=target_min_m, value=target_default_m,
            key=_widget_key("target", "distance_m", machine),
            help="Target distance (m)",
        )

    tuning_col.markdown("<div style='height:1.0rem'></div>", unsafe_allow_html=True)
    with tuning_col.popover("Tuning", use_container_width=True, help="Search weights and convergence settings"):
        weight_col, dist_col = st.columns(2)
        efficiency_weight = weight_col.number_input(
            "Eff. weight", min_value=0.0,
            value=max(float(saved_target.get("efficiency_weight", OptimizationConfig.efficiency_weight)), 0.0),
            key=_widget_key("tune", "efficiency_weight", machine),
            help="How strongly the objective rewards launch efficiency.",
        )
        distance_weight = dist_col.number_input(
            "Dist. weight", min_value=0.0,
            value=max(float(saved_target.get("distance_weight", OptimizationConfig.distance_weight)), 0.0),
            key=_widget_key("tune", "distance_weight", machine),
            help="How strongly the objective penalizes missing the target. Against the "
            "efficiency weight this is the exchange rate: efficiency points the search will "
            "give up per 1% of target distance. Lower it to be shown the most efficient "
            "machine near the target rather than one that hits it.",
        )
        population_size = weight_col.number_input(
            "Population",
            min_value=5,
            value=max(int(saved_target.get("population_size", OptimizationConfig.population_size)), 5),
            step=5,
            key=_widget_key("tune", "population_size", machine),
            help="Differential-evolution population per free parameter. No upper cap - larger "
            "searches more thoroughly, but runtime grows proportionally.",
        )
        absolute_tolerance = dist_col.number_input(
            "Tolerance",
            min_value=0.0,
            value=max(float(saved_target.get("absolute_tolerance", OptimizationConfig.absolute_tolerance)), 0.0),
            step=0.00001,
            format="%.8f",
            key=_widget_key("tune", "absolute_tolerance", machine),
            help="Convergence tolerance on the final solution: the search stops once the "
            "population's objective spread falls below this. Smaller = more precise but slower; "
            "0 runs until the population fully converges or another stop condition is hit.",
        )
        # Full width rather than a third box in either column: its label doesn't fit the
        # half-width columns above, and the popover floats over the page, so an extra row
        # here costs the one-screen layout nothing.
        snap_penalty_weight = st.number_input(
            "Snap penalty",
            min_value=0.0,
            value=max(
                float(saved_target.get("snap_penalty_weight", OptimizationConfig.snap_penalty_weight)), 0.0
            ),
            step=50.0,
            key=_widget_key("tune", "snap_penalty_weight", machine),
            help="How strongly the objective avoids designs whose sling runs close to slack. "
            "Raise it if the winner still jerks; 0 optimizes on range and efficiency alone.",
        )

    btn_sim, btn_opt, btn_save = st.columns([5, 5, 2])
    simulate_clicked = btn_sim.button("Simulate", type="primary", disabled=not fixed_ready, use_container_width=True)
    optimize_clicked = btn_opt.button("Optimize", disabled=not fixed_ready, use_container_width=True)
    if btn_save.button(
        "💾",
        help="Save the current fixed, optimizable, and optimization-target parameters as "
        "this machine's defaults. Each machine keeps its own set; saving one leaves the "
        "other's alone",
        disabled=not fixed_ready,
        use_container_width=True,
    ):
        _save_user_defaults(
            machine,
            {
                name: {"value": optimizable_raw[name], "locked": optimizable_locked[name]}
                for name in param_names(machine)
            },
            {
                name: {"min": bounds[0], "max": bounds[1]}
                for name, bounds in param_ranges.items()
            },
            fixed_params_all,
            {
                "target_distance": target_distance,
                "efficiency_weight": efficiency_weight,
                "distance_weight": distance_weight,
                "population_size": int(population_size),
                "absolute_tolerance": absolute_tolerance,
                "snap_penalty_weight": snap_penalty_weight,
            },
        )
        st.toast("Defaults saved")

    _section("Optimizer log")
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
        name: (
            optimizable_values[name]
            if optimizable_values[name] is not None
            else DEFAULT_MACHINE_PARAMS[machine][name]
        )
        for name in param_names(machine)
    }
    params = TrebuchetParams(machine=machine, **sim_values, **fixed_params_all)
    try:
        with st.spinner("Simulating..."):
            result = simulate_trebuchet(params, track_energy=True, simulate_aftermath=True)
        _store_result(params, result)
    except Exception as exc:  # surface a clean message, never a raw traceback
        status_area.error(f"Simulation failed: {exc}")

if optimize_clicked:
    locked = {name: value for name, value in optimizable_values.items() if value is not None}
    if len(locked) == len(param_names(machine)):
        status_area.error("At least one optimizable parameter must be left blank for the optimizer to search.")
    else:
        config = OptimizationConfig(
            machine=machine,
            # Only the free parameters' ranges: a locked one is pinned, so passing its
            # range would just be noise in the saved/reported search space.
            param_bounds={
                name: bounds for name, bounds in param_ranges.items() if name not in locked
            },
            target_distance=target_distance,
            efficiency_weight=efficiency_weight,
            distance_weight=distance_weight,
            population_size=int(population_size),
            absolute_tolerance=absolute_tolerance,
            snap_penalty_weight=snap_penalty_weight,
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
        with st.container(key="empty_stage"):
            st.markdown(
                _empty_state(
                    "🏰",
                    "No launch yet",
                    "Set the machine up on the left, then hit <b>Simulate</b> to watch the "
                    "throw in 3D - or <b>Optimize</b> to search for a machine that hits the target.",
                ),
                unsafe_allow_html=True,
            )

with right:
    with st.container(key="results_panel", border=True):
        if st.session_state.result is not None:
            _show_results(st.session_state.sim_params, st.session_state.result, imperial, target_distance)
        else:
            st.markdown(
                _empty_state(
                    "📐",
                    "Awaiting a run",
                    "Range, efficiency, release conditions, and the resolved machine geometry appear here.",
                ),
                unsafe_allow_html=True,
            )
