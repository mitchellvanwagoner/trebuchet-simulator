"""Saved dashboard defaults are per-machine.

Driven through Streamlit's own AppTest harness rather than by importing the helpers:
web/app.py *is* the dashboard script, so importing it runs the page. That also means
these cover the half of the behaviour a function-level test could not reach - whether
the boxes on screen actually reseed when the machine changes, which depends on the
widget keys being machine-scoped (see _widget_key).
"""

import json
import logging

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest

import trebuchet_sim.web

APP_PATH = str((__import__("pathlib").Path(trebuchet_sim.web.__file__).parent / "app.py"))
TIMEOUT = 120

# AppTest runs the script outside Streamlit's runtime, so every st.* call logs a
# "missing ScriptRunContext" warning; the same filter the container smoke test uses.
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").addFilter(
    lambda record: "missing ScriptRunContext" not in record.getMessage()
)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("TREBUCHET_DATA_DIR", str(tmp_path))
    return tmp_path


def _run(app=None):
    app = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT) if app is None else app
    app = app.run(timeout=TIMEOUT)
    assert not app.exception, [str(e.value) for e in app.exception]
    return app


def _box(app, label):
    matches = [n for n in app.number_input if n.label == label]
    assert matches, f"no {label!r} box on the page"
    return matches[0]


def _set_machine(app, machine):
    return _run(app.segmented_control[0].set_value(machine))


def _save(app):
    return _run([b for b in app.button if b.label == "💾"][0].click())


def test_each_machine_keeps_its_own_target_and_weights(data_dir):
    app = _run()

    # Pulley: a target and a weight nothing else would produce.
    app = _run(_box(app, "Target (m)").set_value(47.0))
    app = _run(_box(app, "Dist. weight").set_value(7.0))
    app = _save(app)

    # Switching machines must not carry them over - that is the whole point.
    app = _set_machine(app, "traditional")
    assert _box(app, "Target (m)").value == 30.0
    assert _box(app, "Dist. weight").value == 1.0

    app = _run(_box(app, "Target (m)").set_value(88.0))
    app = _run(_box(app, "Dist. weight").set_value(3.0))
    app = _save(app)

    saved = json.loads((data_dir / "user_defaults.json").read_text())
    assert set(saved["machines"]) == {"pulley", "traditional"}
    assert saved["machines"]["pulley"]["target"]["target_distance"] == 47.0
    assert saved["machines"]["traditional"]["target"]["target_distance"] == 88.0
    assert saved["machines"]["pulley"]["target"]["distance_weight"] == 7.0
    assert saved["machines"]["traditional"]["target"]["distance_weight"] == 3.0
    # Saving the traditional machine second must not have taken the pulley's ranges
    # with it - every section is per machine, not just the target.
    assert saved["machines"]["pulley"]["ranges"]
    assert saved["machines"]["pulley"]["optimizable"]

    # And each machine gets its own back on a fresh session.
    app = _run()
    assert _box(app, "Target (m)").value == 88.0  # traditional was saved last, so it opens
    app = _set_machine(app, "pulley")
    assert _box(app, "Target (m)").value == 47.0
    assert _box(app, "Dist. weight").value == 7.0


def test_a_one_machine_defaults_file_still_loads_and_survives_the_next_save(data_dir):
    """The original format kept one machine's sections at the top level.

    Such a file has to keep working - it is what every existing install and mounted
    Docker volume holds - and saving the *other* machine must not be what finally
    deletes it.
    """
    legacy = {
        "machine": "traditional",
        "optimizable": {},
        "ranges": {},
        "fixed": {},
        "target": {"target_distance": 61.0, "efficiency_weight": 9.0},
    }
    (data_dir / "user_defaults.json").write_text(json.dumps(legacy))

    app = _run()
    # It opens on the machine it was written for, carrying that machine's values...
    assert app.segmented_control[0].value == "traditional"
    assert _box(app, "Target (m)").value == 61.0
    assert _box(app, "Eff. weight").value == 9.0

    # ...and they stay with it rather than leaking onto the other machine.
    app = _set_machine(app, "pulley")
    assert _box(app, "Target (m)").value == 30.0
    assert _box(app, "Eff. weight").value == 5.0

    app = _run(_box(app, "Target (m)").set_value(25.0))
    app = _save(app)

    saved = json.loads((data_dir / "user_defaults.json").read_text())
    assert saved["machines"]["pulley"]["target"]["target_distance"] == 25.0
    assert saved["machines"]["traditional"]["target"]["target_distance"] == 61.0
    assert saved["machines"]["traditional"]["target"]["efficiency_weight"] == 9.0
