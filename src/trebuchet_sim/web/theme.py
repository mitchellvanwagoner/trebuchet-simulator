"""Visual design system for the dashboard: palette, Streamlit theme, and CSS.

Single source of truth for the dashboard's look. Three consumers:

- `streamlit_theme_env()` - Streamlit resolves its own theme (widget chrome,
  base colors) from config *before* the app script runs, so it can only be set
  through env vars/CLI flags at server start. `web/launcher.py` (the
  `trebuchet-web` console script, also the Docker entrypoint) and `run.py` both
  apply this dict, which keeps the container and a local checkout identical
  without shipping a `.streamlit/config.toml` (that path is git-ignored).
- `DASHBOARD_CSS` - everything Streamlit's own theme can't express: the
  fixed-viewport layout, panel/card chrome, and the metric readouts.
- `SCENE_*` - handed to the Three.js animation and the matplotlib energy figure
  so the visualizations sit in the same palette as the chrome around them.

Pure constants and strings; imports nothing (not even Streamlit) so `run.py`
can read the theme before the package or its dependencies are installed.
"""

# --- Palette -----------------------------------------------------------------
# Deep navy base sampled from the 3D scene, warm amber accent taken from the
# trajectory arc it draws - the chrome and the visualization share one palette.
BG = "#080d16"
SURFACE = "#0f1724"
SURFACE_2 = "#152036"
BORDER = "#1e2b42"
BORDER_STRONG = "#2c3c59"
TEXT = "#e7edf7"
MUTED = "#8a9ab5"
FAINT = "#5d6d88"

ACCENT = "#f5a524"
ACCENT_DIM = "#b8791a"
ACCENT_SOFT = "rgba(245, 165, 36, 0.14)"

INFO = "#4c8dff"
SUCCESS = "#3ddc97"
WARNING = "#f5a524"
DANGER = "#ff6b6b"

# Colors handed to the visualizations (Three.js scene + matplotlib energy plot).
SCENE_BG = SURFACE
SCENE_GRID = "#1b2942"
SCENE_GROUND = "#2f4a35"
SCENE_TRAJECTORY = ACCENT
SCENE_TRAIL = "#ffe066"


def streamlit_theme_env() -> dict:
    """Streamlit theme settings as environment variables.

    Applied by every entry point before the server starts (see module docstring).
    Values mirror the palette above so the built-in widget chrome - inputs,
    toggles, buttons, dataframes - lands in the same palette as DASHBOARD_CSS.
    """
    return {
        "STREAMLIT_THEME_BASE": "dark",
        "STREAMLIT_THEME_PRIMARY_COLOR": ACCENT,
        "STREAMLIT_THEME_BACKGROUND_COLOR": BG,
        "STREAMLIT_THEME_SECONDARY_BACKGROUND_COLOR": SURFACE_2,
        "STREAMLIT_THEME_TEXT_COLOR": TEXT,
    }


# --- Stylesheet --------------------------------------------------------------
# Laid out as a fixed-viewport flex column rather than the `calc(100vh - Npx)`
# height math this replaced: those constants had to be re-measured by hand
# whenever a control was added, and had already drifted (the input stack
# outgrew its magic number and pushed the header off-screen). Here the page is
# pinned to the viewport and the panels that should absorb slack are marked
# `flex: 1 1 0`, so the one-screen guarantee holds no matter what changes above
# them.
DASHBOARD_CSS = f"""
<style>
  /* ---------- Chrome removal ---------- */
  header[data-testid="stHeader"], [data-testid="stToolbar"] {{ display: none !important; }}
  /* Streamlit appends "Ask Google"/"Ask ChatGPT" links to uncaught exceptions. */
  [data-testid="stException"] a {{ display: none; }}

  /* ---------- Fixed-viewport shell ---------- */
  html, body, .stApp {{ height: 100%; overflow: hidden; background: {BG}; }}
  section[data-testid="stMain"] {{ height: 100vh; overflow: hidden; }}
  [data-testid="stMainBlockContainer"] {{
    display: flex; flex-direction: column;
    height: 100vh; max-width: 100%;
    padding: 0.5rem 0.9rem 0.55rem;
    overflow: hidden;
  }}
  [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] {{
    flex: 1 1 0; min-height: 0; gap: 0.35rem;
  }}
  /* The last top-level row is the 3-column dashboard body: it takes every
     pixel the header row leaves. min-height:0 lets it shrink below content
     height, which is what allows the inner panels to scroll instead of the
     page. */
  [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"]
    > [data-testid="stLayoutWrapper"]:last-child {{ flex: 1 1 0; min-height: 0; }}
  [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"]
    > [data-testid="stLayoutWrapper"]:last-child > [data-testid="stHorizontalBlock"],
  [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"]
    > [data-testid="stLayoutWrapper"]:last-child [data-testid="stColumn"],
  [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"]
    > [data-testid="stLayoutWrapper"]:last-child [data-testid="stColumn"]
    > [data-testid="stVerticalBlock"] {{ height: 100%; min-height: 0; }}

  [data-testid="stVerticalBlock"] {{ gap: 0.28rem; }}
  [data-testid="stHorizontalBlock"] {{ gap: 0.55rem; }}

  /* ---------- Typography ---------- */
  .stApp, .stApp p, .stApp label {{ font-feature-settings: "tnum" 0; }}
  /* overflow:hidden here as well as on the <p>: the label box itself is not
     width-constrained by Streamlit, so a nowrap label would otherwise spill
     across the neighbouring column (visible on the ft/in pairs). */
  [data-testid="stWidgetLabel"] {{
    margin-bottom: 0.05rem; min-height: 0.92rem;
    overflow: hidden; max-width: 100%;
  }}
  /* Single-line labels keep every input row the same height. Imperial mode
     splits each length into a ft+in pair, halving the label width, and
     wrapped two- and three-line labels there used to knock the whole
     two-per-row grid out of rhythm and push the optimizer log off-screen. */
  [data-testid="stWidgetLabel"] p {{
    font-size: 0.7rem; font-weight: 500; letter-spacing: 0.01em;
    color: {MUTED}; text-transform: none;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    display: block; max-width: 100%;
  }}
  [data-testid="stCaptionContainer"] p {{ font-size: 0.7rem; color: {MUTED}; }}

  /* Streamlit gives every markdown block a -16px bottom margin to offset its
     default paragraph spacing. Our custom blocks have no such spacing, so the
     offset collapses them to a few pixels and the next row draws on top. Zero
     it for anything this stylesheet owns (every such block is `tb-`-prefixed). */
  [data-testid="stMarkdownContainer"]:has(> [class^="tb-"]) {{
    margin-bottom: 0 !important;
  }}

  /* ---------- App bar ---------- */
  .tb-appbar {{ display: flex; align-items: center; gap: 0.6rem; }}
  .tb-mark {{
    width: 26px; height: 26px; border-radius: 7px; flex: none;
    display: grid; place-items: center; font-size: 14px;
    background: linear-gradient(145deg, {ACCENT}, {ACCENT_DIM});
    box-shadow: 0 2px 10px rgba(245, 165, 36, 0.3);
  }}
  .tb-title {{
    font-size: 0.98rem; font-weight: 700; letter-spacing: -0.01em;
    color: {TEXT}; line-height: 1.15;
  }}
  .tb-sub {{ font-size: 0.68rem; color: {FAINT}; line-height: 1.15; }}

  /* ---------- Panels ---------- */
  /* The two bordered containers are named explicitly rather than matched by a
     generic wrapper testid: this Streamlit build paints a `border=True`
     container on the keyed stVerticalBlock itself, and every other keyed
     container (the param rows, the ft/in pairs) would be caught by anything
     broader. Without this they fall back to Streamlit's default - a washed-out
     20%-white hairline over a transparent background. */
  .st-key-results_panel, .st-key-opt_log_panel {{
    background: {SURFACE} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
  }}
  .tb-section {{
    display: flex; align-items: center; gap: 0.4rem;
    font-size: 0.66rem; font-weight: 700; letter-spacing: 0.09em;
    text-transform: uppercase; color: {FAINT};
    margin: 0.1rem 0 0.02rem;
  }}
  .tb-section::after {{
    content: ""; flex: 1; height: 1px;
    background: linear-gradient(90deg, {BORDER}, transparent);
  }}

  /* ---------- Inputs ---------- */
  [data-testid="stNumberInput"] input {{
    padding: 0.16rem 0.5rem; font-size: 0.82rem;
    font-variant-numeric: tabular-nums;
  }}
  [data-testid="stNumberInputContainer"] {{
    min-height: 0; height: 2rem;
    background: {SURFACE_2};
    border-color: {BORDER}; border-radius: 7px;
  }}
  [data-testid="stNumberInputContainer"]:focus-within {{
    border-color: {ACCENT};
  }}
  [data-testid="stNumberInputStepDown"],
  [data-testid="stNumberInputStepUp"] {{ display: none; }}
  /* Imperial splits one length into a ft box and an in box, each roughly a
     quarter of an already-narrow column. The inline clear button eats ~24px of
     that - enough to hide the value entirely - and the field can still be
     cleared by selecting its contents, which is what "leave blank" needs. */
  div[class*="st-key-pair_"] [data-testid="stNumberInputClearButton"] {{ display: none; }}

  /* Locked parameters read as locked at a glance: the toggle alone is a 20px
     dot, too small to scan a column of five by. The row's own container is
     tinted from the toggle's checked state instead. */
  div[class*="st-key-param_"]:has(input[type="checkbox"]:checked)
    [data-testid="stNumberInputContainer"] {{
    background: {ACCENT_SOFT};
    border-color: rgba(245, 165, 36, 0.55);
  }}
  div[class*="st-key-param_"]:has(input[type="checkbox"]:checked)
    [data-testid="stWidgetLabel"] p {{
    color: {ACCENT}; font-weight: 600;
  }}

  /* ---------- Buttons ---------- */
  .stButton button {{
    min-height: 1.95rem; padding: 0.15rem 0.7rem;
    border-radius: 7px; font-size: 0.82rem; font-weight: 600;
  }}
  .stButton button[kind="primary"] {{ color: #241703; border: none; }}
  .stButton button[kind="secondary"] {{
    background: {SURFACE_2}; border: 1px solid {BORDER_STRONG}; color: {TEXT};
  }}
  .stButton button[kind="secondary"]:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}

  /* ---------- Alerts ---------- */
  [data-testid="stAlert"] {{
    border-radius: 8px; padding: 0.4rem 0.65rem;
    font-size: 0.74rem; border-left: 3px solid currentColor;
  }}
  [data-testid="stAlert"] p {{ font-size: 0.74rem; line-height: 1.35; }}
  /* Run status floats over the visualization instead of reflowing the column,
     which would otherwise push the energy plots past the bottom of the screen. */
  .st-key-status_area {{
    position: fixed; top: 3rem; left: 27%; width: 46%; z-index: 100;
  }}
  .st-key-status_area [data-testid="stAlert"] {{
    box-shadow: 0 8px 26px rgba(0, 0, 0, 0.55);
    backdrop-filter: blur(6px);
  }}

  /* ---------- Column: parameters (left) ---------- */
  /* The optimizer log absorbs whatever vertical space the input stack leaves.
     Streamlit wraps every keyed container in a non-growing "stLayoutWrapper",
     so the wrapper must be told to grow too or it caps its child at content
     height. */
  [data-testid="stLayoutWrapper"]:has(> .st-key-opt_log_panel) {{ flex: 1 1 0; min-height: 0; }}
  .st-key-opt_log_panel {{
    display: flex; flex-direction: column;
    height: 100%; min-height: 62px; padding: 0.4rem 0.45rem;
  }}
  [data-testid="stLayoutWrapper"]:has(> .st-key-opt_log_table) {{ flex: 1 1 0; min-height: 0; }}
  /* flex-basis 0 (not auto): the dataframe's own height is itself a percentage
     of this container, so sizing from content would be circular - basis 0
     makes the browser size this purely from flex-grow first, then the
     percentages below resolve against that. */
  .st-key-opt_log_table {{ flex: 1 1 0; min-height: 0; height: 100% !important; }}
  .st-key-opt_log_table [data-testid="stElementContainer"],
  .st-key-opt_log_table [data-testid="stDataFrame"] {{ height: 100% !important; }}

  /* ---------- Column: visualization (middle) ---------- */
  [data-testid="stLayoutWrapper"]:has(> .st-key-anim_panel) {{ flex: 1 1 0; min-height: 0; }}
  .st-key-anim_panel {{
    height: 100%; min-height: 0; overflow: hidden;
    border-radius: 10px; border: 1px solid {BORDER}; background: {SURFACE};
  }}
  .st-key-anim_panel [data-testid="stElementContainer"] {{ flex: 1 1 auto !important; }}
  .st-key-anim_panel [data-testid="stElementContainer"],
  .st-key-anim_panel iframe {{ height: 100% !important; width: 100%; }}
  .st-key-energy_panel {{ flex: none; }}
  .st-key-energy_panel img {{
    width: 100% !important; height: auto !important; display: block;
    border-radius: 9px; border: 1px solid {BORDER};
  }}

  /* ---------- Column: results (right) ---------- */
  [data-testid="stLayoutWrapper"]:has(> .st-key-results_panel) {{ flex: 1 1 0; min-height: 0; }}
  .st-key-results_panel {{
    height: 100%; min-height: 0; overflow-y: auto;
    padding: 0.55rem 0.6rem;
  }}

  /* ---------- Metric readouts ---------- */
  .tb-hero {{
    background: linear-gradient(160deg, {SURFACE_2}, {SURFACE});
    border: 1px solid {BORDER_STRONG}; border-radius: 10px;
    padding: 0.5rem 0.7rem 0.55rem; margin-bottom: 0.4rem;
  }}
  .tb-hero-label {{
    font-size: 0.62rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: {FAINT};
  }}
  .tb-hero-value {{
    font-size: 1.85rem; font-weight: 700; line-height: 1.1;
    color: {ACCENT}; font-variant-numeric: tabular-nums;
    letter-spacing: -0.02em;
  }}
  .tb-hero-value .tb-unit {{
    font-size: 0.8rem; font-weight: 600; color: {MUTED}; margin-left: 0.22rem;
  }}
  .tb-hero-note {{ font-size: 0.68rem; color: {MUTED}; margin-top: 0.1rem; }}
  .tb-hero-note b {{ color: {TEXT}; font-weight: 600; }}

  .tb-bar-row {{
    display: flex; justify-content: space-between; align-items: baseline;
    font-size: 0.68rem; color: {MUTED}; margin-bottom: 0.2rem;
  }}
  .tb-bar-row b {{
    font-size: 0.82rem; color: {TEXT}; font-weight: 700;
    font-variant-numeric: tabular-nums;
  }}
  .tb-bar {{
    height: 5px; border-radius: 3px; background: {SURFACE_2};
    overflow: hidden; margin-bottom: 0.5rem;
  }}
  .tb-bar-fill {{
    height: 100%; border-radius: 3px;
    background: linear-gradient(90deg, {ACCENT_DIM}, {ACCENT});
  }}

  .tb-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0.3rem; }}
  .tb-cell {{
    background: {SURFACE_2}; border: 1px solid {BORDER};
    border-radius: 8px; padding: 0.32rem 0.45rem; min-width: 0;
  }}
  .tb-k {{
    display: block; font-size: 0.6rem; font-weight: 600;
    letter-spacing: 0.05em; text-transform: uppercase; color: {FAINT};
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }}
  .tb-v {{
    display: block; font-size: 0.8rem; font-weight: 600; color: {TEXT};
    font-variant-numeric: tabular-nums; margin-top: 0.05rem;
  }}

  .tb-specs {{ display: flex; flex-direction: column; gap: 1px; }}
  .tb-spec {{
    display: flex; justify-content: space-between; align-items: baseline;
    gap: 0.5rem; padding: 0.17rem 0.3rem; border-radius: 5px;
    font-size: 0.71rem; color: {MUTED};
  }}
  .tb-spec:nth-child(odd) {{ background: rgba(255, 255, 255, 0.022); }}
  .tb-spec b {{
    color: {TEXT}; font-weight: 600; white-space: nowrap;
    font-variant-numeric: tabular-nums;
  }}

  /* ---------- Empty states ---------- */
  .tb-empty {{
    height: 100%; min-height: 140px;
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; gap: 0.3rem; text-align: center;
    color: {FAINT}; padding: 1rem;
  }}
  .tb-empty-mark {{ font-size: 1.7rem; opacity: 0.5; }}
  .tb-empty-title {{ font-size: 0.84rem; font-weight: 600; color: {MUTED}; }}
  .tb-empty-hint {{ font-size: 0.7rem; max-width: 34ch; line-height: 1.4; }}
  .tb-empty-hint b {{ color: {ACCENT}; font-weight: 600; }}
  /* The middle column's empty state is the only thing in it, so it has to
     claim the column's full height itself. */
  [data-testid="stLayoutWrapper"]:has(> .st-key-empty_stage) {{ flex: 1 1 0; min-height: 0; }}
  .st-key-empty_stage {{
    height: 100%; border: 1px dashed {BORDER_STRONG};
    border-radius: 10px; background: {SURFACE};
  }}

  /* ---------- Dataframe (optimizer log) ---------- */
  [data-testid="stDataFrame"] {{ font-size: 0.72rem; }}

  /* ---------- Scrollbars ---------- */
  ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
  ::-webkit-scrollbar-track {{ background: transparent; }}
  ::-webkit-scrollbar-thumb {{
    background: {BORDER_STRONG}; border-radius: 4px;
  }}
  ::-webkit-scrollbar-thumb:hover {{ background: {FAINT}; }}
</style>
"""
