"""The matplotlib frontend: the CLI's animation and the energy figure the dashboard embeds.

This module had no tests at all, which is a gap of the same shape as the argparse `%` that
once took down every CLI invocation: nothing here is exercised until a user passes
`--animate`, and by then it is too late. The checks below are deliberately about the two
things that actually break - whether every frame renders, and whether the figure plots the
history it was handed rather than something re-derived - rather than about how it looks.

Agg is selected before pyplot is imported anywhere, so these run headless.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from trebuchet_sim import visualization as vis  # noqa: E402
from trebuchet_sim.config import (  # noqa: E402
    DEFAULT_MACHINE_FIXED,
    DEFAULT_MACHINE_PARAMS,
    MachineType,
    TrebuchetParams,
)
from trebuchet_sim.physics import simulate_trebuchet  # noqa: E402

# A launch that lands the projectile, drags it along the ground and picks it up again -
# the case with the most going on in the sampled timeline (see tests/test_ground.py).
FOUR_REGIME_PARAMS = {
    "counter_weight_mass": 52.725642457680614,
    "pulley_radius": 0.6070760857653384,
    "arm_length": 1.0901717662568329,
    "string_length": 0.9980417041772385,
    "release_angle": -4.2438199212106875,
    "pivot_height": 1.3401717662568329,
}


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _result(params: TrebuchetParams):
    return simulate_trebuchet(params, track_energy=True, simulate_aftermath=True)


def _defaults(machine: MachineType) -> TrebuchetParams:
    return TrebuchetParams(
        machine=machine, **DEFAULT_MACHINE_PARAMS[machine], **DEFAULT_MACHINE_FIXED[machine]
    )


def _no_release_params() -> TrebuchetParams:
    """Enough joint friction that the arm never reaches the release angle."""
    return TrebuchetParams(machine=MachineType.PULLEY,
                           **DEFAULT_MACHINE_PARAMS[MachineType.PULLEY],
                           joint_friction_coefficient=50.0)


def _all_cases():
    yield "pulley", _defaults(MachineType.PULLEY)
    yield "traditional", _defaults(MachineType.TRADITIONAL)
    yield "four-regime", TrebuchetParams(**FOUR_REGIME_PARAMS)
    yield "no release", _no_release_params()


@pytest.mark.parametrize("machine", list(MachineType))
@pytest.mark.parametrize("compact,dark", [(False, False), (True, True)])
def test_the_energy_figure_plots_the_history_it_was_given(machine, compact, dark):
    """The figure is a view of `energy_history`, not a second calculation of it.

    Checking the plotted y-data against the recorded series is what makes that true: a
    figure that quietly re-derived, resampled or reordered anything would still render.
    """
    result = _result(_defaults(machine))
    history = result.energy_history
    assert history

    fig = vis.build_energy_figure(result, compact=compact, dark=dark)
    ax_totals, ax_detail = fig.axes[0], fig.axes[1]

    times = [entry["time"] for entry in history]
    # Panel one: total energy, then the counterweight's potential energy.
    assert list(ax_totals.lines[0].get_xdata()) == times
    assert list(ax_totals.lines[0].get_ydata()) == [e["total"] for e in history]
    assert list(ax_totals.lines[1].get_ydata()) == [e["cw_pe"] for e in history]
    # Panel two: the six components, in the order the module plots them.
    for line, key in zip(ax_detail.lines, ["proj_ke", "arm_ke", "cw_ke", "pulley_ke",
                                           "proj_pe", "arm_pe"]):
        assert list(line.get_ydata()) == [e[key] for e in history], key


@pytest.mark.parametrize("compact", [False, True])
def test_the_energy_figure_marks_the_release_only_when_one_happened(compact):
    """The release line is drawn from `t_release`, which a no-release run does not have.

    Reading a metric that is only present on some results is exactly where this module
    would raise, and only for the user unlucky enough to plot a machine that never threw.
    """
    released = _result(_defaults(MachineType.PULLEY))
    t_release = released.metrics["t_release"]
    fig = vis.build_energy_figure(released, compact=compact)
    # A vertical marker is a line whose x-data is the release time twice over.
    marks = [ln for ax in fig.axes[:2] for ln in ax.lines
             if len(ln.get_xdata()) == 2 and set(np.ravel(ln.get_xdata())) == {t_release}]
    assert len(marks) == 2  # one per panel

    stalled = _result(_no_release_params())
    assert stalled.metrics["release_occurred"] is False
    assert "t_release" not in stalled.metrics
    fig = vis.build_energy_figure(stalled, compact=compact)
    assert fig.axes  # it renders at all, which is the point
    assert not [ln for ax in fig.axes[:2] for ln in ax.lines if len(ln.get_xdata()) == 2]


def test_the_energy_plot_declines_to_draw_an_untracked_run(tmp_path, capsys):
    """Energy tracking is opt-in, so both entry points have to survive it being off."""
    result = simulate_trebuchet(_defaults(MachineType.PULLEY))
    assert result.energy_history is None

    target = tmp_path / "energy.png"
    vis.save_energy_plot(result, str(target))
    assert not target.exists()
    assert "Energy tracking was not enabled" in capsys.readouterr().out

    vis.plot_energy_history(result)  # same guard, no window, no exception


def test_a_saved_energy_plot_is_a_real_file(tmp_path):
    result = _result(_defaults(MachineType.PULLEY))
    target = tmp_path / "energy.png"

    vis.save_energy_plot(result, str(target))

    assert target.exists() and target.stat().st_size > 5_000  # a 300 dpi two-panel plot


def _t_release(result):
    """What create_animation uses as the release instant, including when there isn't one."""
    return result.metrics.get("t_release", result.metrics.get("simulation_time", 10.0))


def _timeline(result):
    """The time grid create_animation builds, rebuilt here so the tests can index it.

    Kept as a re-derivation rather than read off the animation: matplotlib exposes the
    frame count but not the times behind it, and the point of the checks below is that the
    two match.
    """
    t_release = _t_release(result)
    flight = result.metrics.get("flight_time", 0.0) if result.metrics.get("release_occurred") else 0.0
    t_end = t_release + flight
    times = np.arange(0, t_end, 1.0 / vis.TARGET_FPS)
    if len(times) == 0 or times[-1] < t_end:
        times = np.append(times, t_end)
    return times


# Animations are held for the session. Rendering the frames directly leaves the animation
# object itself unplayed, and matplotlib warns on collecting one of those - advising
# exactly this, that the object be kept alive. The warning fires from __del__, so it
# escapes any per-test filter anyway.
_KEEP_ALIVE = []


@pytest.mark.parametrize("label,params", list(_all_cases()),
                         ids=[label for label, _ in _all_cases()])
def test_every_animation_frame_renders(label, params):
    """The frame renderer slices the sampled arrays by frame index, so it is where an
    off-by-one lives - and a launch that never releases has no flight half to slice at all.

    Every frame is rendered rather than a sample of them: the boundaries that matter (the
    first, the release crossing, the last) are not at indices a test can name in advance
    for four different launches, and 80-odd frames is cheap.
    """
    result = _result(params)
    times = _timeline(result)
    anim = vis.create_animation(params, result)
    _KEEP_ALIVE.append(anim)

    assert isinstance(anim, FuncAnimation)
    # The public handle on how many frames it will play.
    assert len(list(anim.new_frame_seq())) == len(times)

    ax_trajectory, ax_system = anim._fig.axes[0], anim._fig.axes[1]
    state_data = vis._calculate_animation_data(params, result, times, _t_release(result))
    for frame in range(len(times)):
        vis._animate_frame(frame, ax_trajectory, ax_system, state_data, params, result)


def test_the_animation_spans_launch_through_impact():
    """One timeline, ending when the projectile lands - not when the launch ends."""
    params = _defaults(MachineType.PULLEY)
    result = _result(params)

    times = _timeline(result)
    expected_end = result.metrics["t_release"] + result.metrics["flight_time"]
    assert times[0] == 0.0
    assert times[-1] == pytest.approx(expected_end)
    # The projectile's last sampled position is where the trajectory says it landed.
    state = vis._calculate_animation_data(params, result, times, _t_release(result))
    assert state["positions"]["projectile"][-1][0] == pytest.approx(result.distance, rel=1e-6)


@pytest.mark.parametrize("machine,expect_pulley", [(MachineType.PULLEY, True),
                                                   (MachineType.TRADITIONAL, False)])
def test_the_animation_draws_each_machines_own_linkage(machine, expect_pulley):
    """Same split the 3D scene makes: a pulley disc, or a beam behind the pivot and a link.

    Drawn from the sampled `cw_pin` track rather than recomputed, so this also checks the
    sampler is handing the frontend the pin it needs.
    """
    params = _defaults(machine)
    result = _result(params)
    # Two bare axes rather than a whole animation: the renderer only needs somewhere to
    # draw, and building an animation it never plays is what makes matplotlib complain.
    _fig, (ax_trajectory, ax_system) = plt.subplots(1, 2)
    times = _timeline(result)
    state_data = vis._calculate_animation_data(params, result, times, _t_release(result))

    vis._animate_frame(0, ax_trajectory, ax_system, state_data, params, result)

    discs = [p for p in ax_system.patches if isinstance(p, Circle)]
    if expect_pulley:
        assert len(discs) == 1
        assert discs[0].get_radius() == params.pulley_radius
    else:
        assert discs == []
        pin = state_data["positions"]["cw_pin"][0]
        # The pin rides the beam behind the pivot, so it is not the axle itself.
        assert not np.allclose(pin, [0.0, params.pivot_height])
        assert any(
            np.allclose(line.get_xydata()[-1], pin) for line in ax_system.lines
        ), "no line ends at the counterweight pin"


def test_save_animation_gif_writes_a_gif(tmp_path):
    """The CLI's --save-gif path. Driven with a two-frame animation of its own: rendering
    eighty trebuchet frames through Pillow takes seventeen seconds, and what is under test
    here is the writer wiring rather than the drawing, which every frame above covers.
    """
    fig, ax = plt.subplots()
    (line,) = ax.plot([0, 1], [0, 1])
    anim = FuncAnimation(fig, lambda i: (line.set_ydata([0, i]),), frames=2, interval=50)
    target = tmp_path / "launch.gif"

    vis.save_animation_gif(anim, str(target), fps=5)

    assert target.exists() and target.stat().st_size > 0
    assert target.read_bytes()[:6] in (b"GIF87a", b"GIF89a")
