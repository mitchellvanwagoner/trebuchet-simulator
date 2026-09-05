"""Container smoke test: prove the dashboard actually renders and throws.

Run inside a freshly started image, as the runtime user, against the real data
directory - see the "Smoke test image" step in .github/workflows/docker-publish.yml.

Why this exists beyond the workflow's health poll: Streamlit answers
`/_stcore/health` as soon as the *server* is listening, about a second after start,
whereas the app script only runs when a session connects. An exception raised while
rendering the dashboard therefore leaves the health endpoint reporting `ok`, and a
smoke test built on it alone publishes a broken image. That is not hypothetical - a
missing per-machine default once crashed every session with a KeyError while the
container looked perfectly healthy, and it only showed up on a data directory that
had no saved defaults yet, which is exactly what a new volume is.

`AppTest` is Streamlit's own headless harness (public API since 1.28): it executes
the real app script in-process and surfaces anything it raised, with no browser to
install and no Streamlit internals to reach into.
"""

import logging
import os
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

# AppTest drives the script outside Streamlit's usual runtime, so every st.* call logs a
# "missing ScriptRunContext" warning - tens of thousands of lines that would bury the
# actual result in the CI log.
#
# Dropped with a filter rather than a log level, because Streamlit owns the level:
# streamlit.logger stamps its configured level on each logger and re-applies it as the
# runtime initializes, so anything set here is overwritten (and it must be this exact
# logger either way - get_logger sets propagate=False, so the parent never sees it).
# Filters survive that, and this one drops only the one message.
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").addFilter(
    lambda record: "missing ScriptRunContext" not in record.getMessage()
)

# Simulating builds a physics run, a Three.js timeline and a matplotlib figure, so the
# default few seconds is not enough on a loaded CI runner.
TIMEOUT_SECONDS = 120


def _fail(message: str) -> "NoReturn":
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def _check_no_exception(app: AppTest, stage: str) -> None:
    if app.exception:
        details = "\n".join(str(e.value) for e in app.exception)
        _fail(f"the app raised while {stage}:\n{details}")


def _click(app: AppTest, label: str, stage: str) -> AppTest:
    matches = [button for button in app.button if button.label == label]
    if not matches:
        _fail(f"no {label!r} button on the page while {stage}")
    app = matches[0].click().run(timeout=TIMEOUT_SECONDS)
    _check_no_exception(app, stage)
    return app


def _released(app: AppTest, stage: str):
    # app.py seeds this key on every run, so a missing one means the script never got
    # that far. AppTest's session_state proxies attribute access to keys, so it has no
    # .get() to reach for.
    result = app.session_state["result"]
    if result is None:
        _fail(f"{stage} produced no result")
    if "error" in result.metrics:
        _fail(f"{stage} failed: {result.metrics['error']}")
    if not result.metrics.get("release_occurred", False):
        _fail(f"{stage} never released the projectile")
    if not result.distance > 0:
        _fail(f"{stage} threw {result.distance} m")
    return result


def main() -> int:
    # Resolve the installed app rather than a source checkout: what runs here has to be
    # the copy the image actually serves. Located via the package directory rather than
    # by importing the module, because importing app.py *is* running the dashboard -
    # outside the harness, where a failure surfaces as a bare traceback instead of a
    # reported one, and where it would then run a second time under AppTest.
    import trebuchet_sim.web

    app_path = Path(trebuchet_sim.web.__file__).parent / "app.py"
    if not app_path.is_file():
        _fail(f"no app script at {app_path}")
    print(f"app script    : {app_path}")
    print(f"data directory: {os.environ.get('TREBUCHET_DATA_DIR', '<unset>')}")

    # The Numba engine is an install-time extra, and without it the optimizer silently
    # falls back to the much slower scipy objective - correct, but not what was shipped.
    from trebuchet_sim.optimization import _FASTSIM_AVAILABLE

    if not _FASTSIM_AVAILABLE:
        _fail("the Numba fast engine is missing; the image should install the [fast] extra")
    print("fast engine   : available")

    app = AppTest.from_file(str(app_path), default_timeout=TIMEOUT_SECONDS).run()
    _check_no_exception(app, "rendering the dashboard")
    print("render        : ok")

    # Both machines, because they take different paths through the geometry, the
    # equations of motion and the animation, and each has its own set of defaults.
    app = _click(app, "Simulate", "simulating the pulley machine")
    pulley = _released(app, "the pulley machine")
    print(f"pulley        : {pulley.distance:.2f} m at {pulley.efficiency * 100:.1f}%")

    if not app.segmented_control:
        _fail("no machine selector on the page")
    app = app.segmented_control[0].set_value("traditional").run(timeout=TIMEOUT_SECONDS)
    _check_no_exception(app, "switching to the traditional machine")

    app = _click(app, "Simulate", "simulating the traditional machine")
    traditional = _released(app, "the traditional machine")
    print(f"traditional   : {traditional.distance:.2f} m at {traditional.efficiency * 100:.1f}%")

    print("Container smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
