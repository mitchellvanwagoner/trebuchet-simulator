"""Trebuchet animation and energy-plot visualization."""

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from trebuchet_sim.config import TrebuchetParams
from trebuchet_sim.physics import SimulationResult, sample_full_timeline

TARGET_FPS = 30


def create_animation(params: TrebuchetParams, result: SimulationResult) -> FuncAnimation:
    """Create a dual-view (full trajectory + system close-up) trebuchet animation.

    Runs one unified timeline from launch through projectile impact: launch dynamics,
    then the independently-integrated post-release machine settling (arm/counterweight)
    stitched with the ballistic flight (projectile), both ending together when the
    projectile lands. Requires `result` to have been produced with
    `simulate_trebuchet(..., simulate_aftermath=True)` to show the settling machine;
    without it, the machine holds its release pose (see sample_full_timeline).
    """
    release_occurred = bool(result.metrics.get("release_occurred", False))
    t_release = result.metrics.get("t_release", result.metrics.get("simulation_time", 10.0))
    flight_time = result.metrics.get("flight_time", 0.0) if release_occurred else 0.0
    t_end = t_release + flight_time

    dt = 1.0 / TARGET_FPS
    t_anim = np.arange(0, t_end, dt)
    if len(t_anim) == 0 or t_anim[-1] < t_end:
        t_anim = np.append(t_anim, t_end)
    frame_interval = dt * 1000  # milliseconds, for matplotlib

    state_data = _calculate_animation_data(params, result, t_anim, t_release)

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle("Trebuchet Physics Simulation", fontsize=14)
    ax_trajectory = plt.subplot(1, 2, 1)
    ax_system = plt.subplot(1, 2, 2)

    _setup_trajectory_view(ax_trajectory, state_data, result, params)
    _setup_system_view(ax_system, state_data, params)

    def animate(frame):
        return _animate_frame(frame, ax_trajectory, ax_system, state_data, params, result)

    anim = FuncAnimation(fig, animate, frames=len(t_anim), interval=frame_interval, blit=False, repeat=True)

    plt.tight_layout()
    return anim


def _calculate_animation_data(params: TrebuchetParams, result: SimulationResult, t_anim: np.ndarray, t_release: float) -> dict:
    """Calculate component positions at each animation time step across the full timeline."""
    positions = sample_full_timeline(params, result, t_anim)
    return {
        "times": t_anim,
        "t_release": t_release,
        "positions": {key: np.array(value) for key, value in positions.items()},
    }


def _setup_trajectory_view(ax, state_data: dict, result: SimulationResult, params: TrebuchetParams):
    """Setup the full trajectory view with proper scaling."""
    proj_positions = state_data["positions"]["projectile"]
    all_x = list(proj_positions[:, 0])
    all_y = list(proj_positions[:, 1])

    if result.metrics.get("release_occurred"):
        title_suffix = f"Range: {result.distance:.1f}m, Efficiency: {result.efficiency:.3f}"
    else:
        title_suffix = f"No Release - Final: ({proj_positions[-1, 0]:.1f}m, {proj_positions[-1, 1]:.1f}m)"

    margin = 2.0
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    ax.plot(0, params.pivot_height, "ko", markersize=8, label="Pivot")
    ax.axhline(y=0, color="brown", linestyle="-", linewidth=3, label="Ground")

    ax.set_xlabel("Horizontal Distance (m)", fontsize=12)
    ax.set_ylabel("Height (m)", fontsize=12)
    ax.set_title(f"Full Trajectory View\n{title_suffix}", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)


def _setup_system_view(ax, state_data: dict, params: TrebuchetParams):
    """Setup the system close-up view with proper scaling.

    Bounds on the arm-tip/counterweight positions (present for the whole timeline) plus
    the projectile only while it's still on the machine (pre-release) - once released it
    can travel tens of meters downrange, which would blow out the close-up scale and hide
    the machine's post-release motion entirely.
    """
    times = state_data["times"]
    t_release = state_data["t_release"]
    positions = state_data["positions"]
    launch_mask = times <= t_release

    all_positions = list(positions["arm_tip"]) + list(positions["counterweight"]) + list(positions["projectile"][launch_mask])
    all_x = [pos[0] for pos in all_positions]
    all_y = [pos[1] for pos in all_positions]

    margin = 0.5
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    ax.plot(0, params.pivot_height, "ko", markersize=12, label="Pivot")
    ax.axhline(y=0, color="brown", linestyle="-", linewidth=4, label="Ground")

    ax.set_xlabel("Position (m)", fontsize=12)
    ax.set_ylabel("Height (m)", fontsize=12)
    ax.set_title("System Close-up View", fontsize=14)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)


def _animate_frame(frame: int, ax_trajectory, ax_system, state_data: dict, params: TrebuchetParams, result: SimulationResult):
    """Render a single frame of the unified launch -> settling -> flight timeline."""
    t_release = state_data["t_release"]
    current_time = float(state_data["times"][frame])
    launched = current_time > t_release

    proj_pos = state_data["positions"]["projectile"][frame]
    arm_tip_pos = state_data["positions"]["arm_tip"][frame]
    cw_pos = state_data["positions"]["counterweight"][frame]
    pivot_x, pivot_y = 0, params.pivot_height

    ax_system.clear()
    _setup_system_view(ax_system, state_data, params)
    ax_system.plot([pivot_x, arm_tip_pos[0]], [pivot_y, arm_tip_pos[1]], "k-", linewidth=8, label="Arm")
    if not launched:
        # The sling only exists pre-release; the projectile is gone from the machine afterward.
        ax_system.plot([arm_tip_pos[0], proj_pos[0]], [arm_tip_pos[1], proj_pos[1]], "r-", linewidth=4, label="String")
    ax_system.plot(proj_pos[0], proj_pos[1], "ro", markersize=12, label="Projectile")
    ax_system.plot(cw_pos[0], cw_pos[1], "s", color="gray", markersize=12, label="Counterweight")
    if params.has_pulley:
        pulley = patches.Circle((pivot_x, pivot_y), params.pulley_radius, fill=False, edgecolor="brown", linewidth=6)
        ax_system.add_patch(pulley)
    else:
        # No pulley to draw. Instead the beam continues behind the pivot to the pin the
        # counterweight hangs from, and the weight swings on a link below it - both of
        # which move with the arm, so they come from the sampled positions.
        pin_pos = state_data["positions"]["cw_pin"][frame]
        ax_system.plot([pivot_x, pin_pos[0]], [pivot_y, pin_pos[1]], "k-", linewidth=8)
        ax_system.plot([pin_pos[0], cw_pos[0]], [pin_pos[1], cw_pos[1]], color="brown", linewidth=3)
    phase = "Launching" if not launched else ("In Flight" if current_time < state_data["times"][-1] else "Landed")
    ax_system.set_title(f"System Close-up ({phase})\nTime: {current_time:.3f}s", fontsize=12)

    ax_trajectory.clear()
    _setup_trajectory_view(ax_trajectory, state_data, result, params)
    proj_positions = state_data["positions"]["projectile"]
    launch_mask = state_data["times"] <= t_release
    n_launch = int(np.count_nonzero(launch_mask))
    ax_trajectory.plot(
        proj_positions[: min(frame, n_launch) + 1, 0], proj_positions[: min(frame, n_launch) + 1, 1],
        "b-", linewidth=2, label="Launch path",
    )
    if launched:
        ax_trajectory.plot(
            proj_positions[n_launch: frame + 1, 0], proj_positions[n_launch: frame + 1, 1],
            "g-", linewidth=3, label="Flight path",
        )
    ax_trajectory.plot(proj_pos[0], proj_pos[1], "ro", markersize=12, label="Projectile")
    if launched:
        ax_trajectory.set_title(
            f"Ballistic Flight\nTime: {current_time:.3f}s, Flight: {current_time - t_release:.3f}s", fontsize=12
        )


def show_animation(anim: FuncAnimation) -> None:
    """Display animation and block until the window is closed."""
    plt.show(block=True)


def save_animation_gif(anim: FuncAnimation, filename: str, fps: int = 30) -> None:
    """Save animation as a GIF file."""
    print(f"Saving animation as GIF: {filename}")
    anim.save(filename, writer=PillowWriter(fps=fps))
    print(f"Animation saved as {filename}")


# Dark palette for the web dashboard's embedded energy figure, kept in step with
# web/theme.py. Imported lazily inside build_energy_figure so this module (used
# by the CLI, which never touches the web package) keeps no web dependency.
def _dark_plot_palette() -> dict:
    from trebuchet_sim.web import theme

    return {
        "surface": theme.SURFACE,
        "text": theme.TEXT,
        "muted": theme.MUTED,
        "grid": theme.BORDER,
        "total": theme.TEXT,
        "cw": theme.SUCCESS,
        "proj": "#ff7a5c",
        "arm": theme.INFO,
        "pulley": "#c084fc",
        "release": theme.ACCENT,
    }


_LIGHT_PLOT_PALETTE = {
    "surface": "white",
    "text": "black",
    "muted": "black",
    "grid": "gray",
    "total": "black",
    "cw": "green",
    "proj": "red",
    "arm": "blue",
    "pulley": "magenta",
    "release": "orange",
}


def _legend_above(ax, ncol: int, fontsize: int) -> None:
    """Put an axes' legend in a horizontal strip just above the plot area.

    Frameless and spanning the full axes width, so it reads as a caption rather
    than a box floating over the data.
    """
    # Anchored to the axes' top-left corner and packed left, rather than
    # stretched across the full width - with only two entries an expanded
    # legend pushes them to opposite ends and reads as two stray labels.
    ax.legend(
        fontsize=fontsize, ncol=ncol, frameon=False,
        loc="lower left", bbox_to_anchor=(0.0, 1.01),
        borderaxespad=0.0, handlelength=1.4,
        columnspacing=1.2, handletextpad=0.4,
    )


def build_energy_figure(result: SimulationResult, compact: bool = False, dark: bool = False):
    """Build the two-panel energy-components figure. Public so callers (e.g. the web UI) can embed it directly.

    compact=True lays the panels side by side in a short, wide figure sized for
    embedding in the single-screen web dashboard.

    dark=True restyles the figure for the dashboard's dark surface (the default
    light styling is what the CLI's saved PNGs want).
    """
    history = result.energy_history
    times = [e["time"] for e in history]
    palette = _dark_plot_palette() if dark else _LIGHT_PLOT_PALETTE

    if compact:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.2))
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[1, 1])
        fig.suptitle("Trebuchet Energy Components Over Time", fontsize=16, fontweight="bold", color=palette["text"])

    label_size = 9 if compact else 12
    title_size = 10 if compact else 14
    legend_size = 8 if compact else 11
    detail_legend_size = 7 if compact else 10

    # Compact mode abbreviates the series names and drops the axes titles: the
    # panels are only a few inches wide there, and the legend already says what
    # each one plots, so the title was redundant with it.
    def name(full: str, short: str) -> str:
        return short if compact else full

    total_energies = [e["total"] for e in history]
    ax1.plot(times, total_energies, "-", color=palette["total"], label=name("Total Energy", "Total"), linewidth=2.5, alpha=0.95)
    ax1.plot(times, [e["cw_pe"] for e in history], "--", color=palette["cw"], label=name("Counterweight PE", "CW PE"), linewidth=2.2, alpha=0.9)
    ax1.set_ylabel("Energy (J)", fontsize=label_size)

    ax2.plot(times, [e["proj_ke"] for e in history], "-", color=palette["proj"], label=name("Projectile KE", "Proj KE"), linewidth=2)
    ax2.plot(times, [e["arm_ke"] for e in history], "-", color=palette["arm"], label="Arm KE", linewidth=2)
    ax2.plot(times, [e["cw_ke"] for e in history], "-", color=palette["cw"], label=name("Counterweight KE", "CW KE"), linewidth=2)
    ax2.plot(times, [e["pulley_ke"] for e in history], "-", color=palette["pulley"], label=name("Pulley KE", "Pulley"), linewidth=2)
    ax2.plot(times, [e["proj_pe"] for e in history], "--", color=palette["proj"], label=name("Projectile PE", "Proj PE"), linewidth=1.8, alpha=0.75)
    ax2.plot(times, [e["arm_pe"] for e in history], "--", color=palette["arm"], label=name("Arm PE", "Arm PE"), linewidth=1.8, alpha=0.75)
    ax2.set_xlabel("Time (s)", fontsize=label_size)
    ax2.set_ylabel("Energy (J)", fontsize=label_size)

    if compact:
        # Legends go above the axes as a single horizontal strip. Inside the
        # plot they sat on top of the curves - with six series the detail panel
        # has no empty corner for `loc="best"` to find.
        _legend_above(ax1, ncol=2, fontsize=legend_size)
        _legend_above(ax2, ncol=6, fontsize=detail_legend_size)
        ax1.set_xlabel("Time (s)", fontsize=label_size)
        ax1.tick_params(labelsize=8)
        ax2.tick_params(labelsize=8)
    else:
        ax1.set_title("Total Energy and Counterweight Potential Energy", fontsize=title_size)
        ax2.set_title("Kinetic and Potential Energy Components (Detail View)", fontsize=title_size)
        ax1.legend(fontsize=legend_size)
        ax2.legend(fontsize=detail_legend_size, ncol=2)

    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)

    if result.metrics.get("release_occurred", True) and "t_release" in result.metrics:
        release_time = result.metrics["t_release"]
        ax1.axvline(x=release_time, color=palette["release"], linestyle=":", linewidth=2, alpha=0.9)
        ax2.axvline(x=release_time, color=palette["release"], linestyle=":", linewidth=2, alpha=0.9)

        if not compact:
            max_energy_top = max(total_energies)
            ax1.annotate(
                f"Release\n(t={release_time:.3f}s)",
                xy=(release_time, max_energy_top * 0.9),
                xytext=(release_time + 0.1, max_energy_top * 0.9),
                arrowprops=dict(arrowstyle="->", color=palette["release"], alpha=0.8),
                fontsize=10,
                color=palette["release"],
                fontweight="bold",
            )

    if dark:
        _apply_dark_axes(fig, (ax1, ax2), palette)

    plt.tight_layout()
    if compact:
        # tight_layout ignores legends anchored outside the axes, so the strip
        # would otherwise be clipped by the top of the figure.
        fig.subplots_adjust(top=0.84)
    return fig


def _apply_dark_axes(fig, axes, palette: dict) -> None:
    """Recolor a finished figure for the dashboard's dark panel.

    Applied after plotting rather than through an rcParams style so the CLI's
    light output is untouched by a global side effect.
    """
    fig.patch.set_facecolor(palette["surface"])
    for ax in axes:
        ax.set_facecolor(palette["surface"])
        ax.tick_params(colors=palette["muted"])
        ax.xaxis.label.set_color(palette["muted"])
        ax.yaxis.label.set_color(palette["muted"])
        # Colour alone separates the title here; DejaVu (matplotlib's bundled
        # default) has no semibold face, and asking for one logs a findfont
        # warning on every render.
        ax.title.set_color(palette["text"])
        ax.grid(True, alpha=0.18, color=palette["grid"])
        for side, spine in ax.spines.items():
            # Keep a single light baseline instead of a full box - the chart
            # reads as part of the panel rather than a pasted-in image.
            spine.set_visible(side in ("bottom", "left"))
            spine.set_color(palette["grid"])
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(palette["surface"])
            legend.get_frame().set_edgecolor(palette["grid"])
            legend.get_frame().set_alpha(0.85)
            for text in legend.get_texts():
                text.set_color(palette["muted"])


def plot_energy_history(result: SimulationResult) -> None:
    """Display kinetic/potential energy components over time in a blocking window."""
    if not result.energy_history:
        print("Energy tracking was not enabled. Run simulate_trebuchet(..., track_energy=True) first.")
        return
    build_energy_figure(result)
    plt.show()


def save_energy_plot(result: SimulationResult, filename: str) -> None:
    """Save the kinetic/potential energy plot to a file."""
    if not result.energy_history:
        print("Energy tracking was not enabled. Run simulate_trebuchet(..., track_energy=True) first.")
        return
    fig = build_energy_figure(result)
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Energy plot saved as: {filename}")
