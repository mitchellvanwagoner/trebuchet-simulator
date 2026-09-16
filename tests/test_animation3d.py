"""The 3D animation's payload: what the browser is handed, rather than how it looks.

The energy plot is part of this page rather than a figure beside it, because it plays on the
animation's clock (see web/animation3d.py). That only works if the timeline and the history
agree about what time it is, which is what these check - along with which series a machine
gets, since a pulley line on a machine with no pulley is a flat zero taking up the legend.
"""

import pytest

pytest.importorskip("streamlit")

from trebuchet_sim.config import (  # noqa: E402
    DEFAULT_MACHINE_FIXED,
    DEFAULT_MACHINE_PARAMS,
    MachineType,
    TrebuchetParams,
)
from trebuchet_sim.physics import POST_RELEASE_ENERGY_SECONDS, simulate_trebuchet  # noqa: E402
from trebuchet_sim.web.animation3d import _build_timeline, build_trebuchet_3d_html  # noqa: E402


def _machine(machine: MachineType) -> TrebuchetParams:
    return TrebuchetParams(
        machine=machine, **DEFAULT_MACHINE_PARAMS[machine], **DEFAULT_MACHINE_FIXED[machine]
    )


def _timeline(machine: MachineType) -> tuple:
    params = _machine(machine)
    result = simulate_trebuchet(params, track_energy=True, simulate_aftermath=True)
    return _build_timeline(params, result), result


@pytest.mark.parametrize("machine", list(MachineType))
def test_the_energy_payload_names_only_this_machine_s_series(machine):
    timeline, _result = _timeline(machine)
    labels = [series["label"] for series in timeline["energy"]["series"]]

    assert ("Pulley" in labels) is (machine is MachineType.PULLEY)
    # The counterweight's two energies are the point of the plot on the traditional
    # machine, whose weight swings on its own pin rather than riding a rope.
    assert {"CW KE", "CW PE", "Proj KE", "Total"} <= set(labels)
    for series in timeline["energy"]["series"]:
        assert len(series["values"]) == len(timeline["energy"]["t"])


@pytest.mark.parametrize("machine", list(MachineType))
def test_the_chart_and_the_frames_run_on_one_clock(machine):
    """The lines draw as the machine moves, so the two have to end together."""
    timeline, result = _timeline(machine)
    settle = max(result.metrics["flight_time"], POST_RELEASE_ENERGY_SECONDS)

    assert timeline["total_time"] == pytest.approx(result.metrics["t_release"] + settle)
    assert timeline["energy"]["t"][-1] == pytest.approx(timeline["total_time"])
    assert timeline["aftermath_frames"][-1]["t"] == pytest.approx(settle)


def test_the_page_carries_the_chart_and_declares_its_height():
    params = _machine(MachineType.PULLEY)
    result = simulate_trebuchet(params, track_energy=True, simulate_aftermath=True)

    html = build_trebuchet_3d_html(params, result, height=440)

    assert "treb-chart-wrap" in html
    assert "function drawChart" in html
    assert '"energy":' in html


def test_a_run_without_energy_tracking_still_animates():
    """The chart is optional: an untracked run hides it and gives the scene the whole frame."""
    params = _machine(MachineType.PULLEY)
    result = simulate_trebuchet(params, simulate_aftermath=True)

    timeline = _build_timeline(params, result)

    assert timeline["energy"] is None
    assert build_trebuchet_3d_html(params, result, height=440) is not None
