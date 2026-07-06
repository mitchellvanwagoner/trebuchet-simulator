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
    pulley = patches.Circle((pivot_x, pivot_y), params.pulley_radius, fill=False, edgecolor="brown", linewidth=6)
    ax_system.add_patch(pulley)
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


def build_energy_figure(result: SimulationResult, compact: bool = False):
    """Build the two-panel energy-components figure. Public so callers (e.g. the web UI) can embed it directly.

    compact=True lays the panels side by side in a short, wide figure sized for
    embedding in the single-screen web dashboard.
    """
    history = result.energy_history
    times = [e["time"] for e in history]

    if compact:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.2))
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[1, 1])
        fig.suptitle("Trebuchet Energy Components Over Time", fontsize=16, fontweight="bold")

    label_size = 9 if compact else 12
    title_size = 10 if compact else 14
    legend_size = 8 if compact else 11
    detail_legend_size = 7 if compact else 10

    total_energies = [e["total"] for e in history]
    ax1.plot(times, total_energies, "k-", label="Total Energy", linewidth=3, alpha=0.9)
    ax1.plot(times, [e["cw_pe"] for e in history], "g--", label="Counterweight PE", linewidth=2.5, alpha=0.8)
    ax1.set_ylabel("Energy (J)", fontsize=label_size)
    ax1.set_title("Total and Counterweight PE" if compact else "Total Energy and Counterweight Potential Energy", fontsize=title_size)
    ax1.legend(fontsize=legend_size)
    ax1.grid(True, alpha=0.3)

    ax2.plot(times, [e["proj_ke"] for e in history], "r-", label="Projectile KE", linewidth=2)
    ax2.plot(times, [e["arm_ke"] for e in history], "b-", label="Arm KE", linewidth=2)
    ax2.plot(times, [e["cw_ke"] for e in history], "g-", label="Counterweight KE", linewidth=2)
    ax2.plot(times, [e["pulley_ke"] for e in history], "m-", label="Pulley KE", linewidth=2)
    ax2.plot(times, [e["proj_pe"] for e in history], "r--", label="Projectile PE", linewidth=2, alpha=0.8)
    ax2.plot(times, [e["arm_pe"] for e in history], "b--", label="Arm PE", linewidth=2, alpha=0.8)
    ax2.set_xlabel("Time (s)", fontsize=label_size)
    ax2.set_ylabel("Energy (J)", fontsize=label_size)
    ax2.set_title("Energy Components" if compact else "Kinetic and Potential Energy Components (Detail View)", fontsize=title_size)
    ax2.legend(fontsize=detail_legend_size, ncol=2)
    ax2.grid(True, alpha=0.3)
    if compact:
        ax1.set_xlabel("Time (s)", fontsize=label_size)
        ax1.tick_params(labelsize=8)
        ax2.tick_params(labelsize=8)

    if result.metrics.get("release_occurred", True) and "t_release" in result.metrics:
        release_time = result.metrics["t_release"]
        ax1.axvline(x=release_time, color="orange", linestyle=":", linewidth=2, alpha=0.8)
        ax2.axvline(x=release_time, color="orange", linestyle=":", linewidth=2, alpha=0.8)

        if not compact:
            max_energy_top = max(total_energies)
            ax1.annotate(
                f"Release\n(t={release_time:.3f}s)",
                xy=(release_time, max_energy_top * 0.9),
                xytext=(release_time + 0.1, max_energy_top * 0.9),
                arrowprops=dict(arrowstyle="->", color="orange", alpha=0.8),
                fontsize=10,
                color="orange",
                fontweight="bold",
            )

    plt.tight_layout()
    return fig


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
