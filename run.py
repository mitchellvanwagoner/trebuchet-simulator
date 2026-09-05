#!/usr/bin/env python3
"""Launch the Trebuchet Simulator web UI in your browser.

Usage: python run.py

Works with any Python — even one without Streamlit installed — because it
finds the project's .venv/ interpreter and launches Streamlit with that. If
that venv is missing or stale (see _repair_venv), it is rebuilt in place first,
so moving or copying the project folder needs no manual setup.
Picks the first free port at or after 8501, then opens the browser once the
server responds. Serves at http://treb-simulator.local:<port> if that
hostname is mapped in your hosts file, otherwise at http://localhost:<port>.

Every path here is derived from this file's own location, so the project can
live anywhere.
"""

import importlib.util
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).parent
APP_PATH = REPO_ROOT / "src" / "trebuchet_sim" / "web" / "app.py"
PREFERRED_PORT = 8501
FRIENDLY_HOST = "treb-simulator.local"

SETUP_HELP = """\
Streamlit is not installed. Set the project up once, then re-run this script:

  python -m venv .venv
  .venv\\Scripts\\activate        (Windows)
  source .venv/bin/activate      (macOS/Linux)
  pip install -e ".[dev]"
"""


VENV_DIR = REPO_ROOT / ".venv"


def _venv_pythons() -> "list[Path]":
    """Candidate venv interpreter paths, both platform layouts."""
    return [
        VENV_DIR / "Scripts" / "python.exe",  # Windows
        VENV_DIR / "bin" / "python",          # macOS/Linux
    ]


def _runs_streamlit(python: Path) -> bool:
    """True if `python` exists and can import Streamlit."""
    if not python.exists():
        return False
    try:
        probe = subprocess.run([str(python), "-c", "import streamlit"], capture_output=True)
    except OSError:
        return False
    return probe.returncode == 0


def _repair_venv() -> bool:
    """Rebuild .venv in place, then reinstall the project into it.

    A virtual environment is not relocatable: pyvenv.cfg records an absolute
    path to the base interpreter, and pip's editable install records an
    absolute path to src/. Moving or copying the project - or upgrading the
    system Python - leaves both dangling, and every command that uses the venv
    then fails with a confusing "did not find executable" error.

    `python -m venv` over an existing directory rewrites those records without
    deleting anything, so this is a repair rather than a wipe; the editable
    reinstall then re-points the project path and refreshes any dependency
    built for the previous Python's ABI.

    Returns False (leaving the venv untouched) when this script is *running*
    from the venv it would have to rebuild.
    """
    if Path(sys.executable).resolve().is_relative_to(VENV_DIR.resolve()):
        return False

    print("Setting up .venv (missing or stale) - this runs once after a move or a fresh clone.")
    try:
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV_DIR)])
        python = next(p for p in _venv_pythons() if p.exists())
        subprocess.check_call([str(python), "-m", "pip", "install", "--upgrade", "pip", "--quiet"])
        subprocess.check_call([str(python), "-m", "pip", "install", "-e", ".[dev]"], cwd=REPO_ROOT)
    except (subprocess.CalledProcessError, StopIteration, OSError) as exc:
        print(f"Automatic setup failed ({exc}).", file=sys.stderr)
        return False
    return True


def _find_interpreter() -> str:
    """Pick an interpreter that has Streamlit installed, preferring the venv.

    run.py itself may be started with any Python (system install, py launcher,
    double-click); only the child process that runs Streamlit needs the venv.
    """
    for python in [*_venv_pythons(), Path(sys.executable)]:
        if _runs_streamlit(python):
            return str(python)

    # Nothing usable: most often a venv that was moved with the project, or a
    # checkout that has never been set up. Both are repairable without help.
    if _repair_venv():
        for python in _venv_pythons():
            if _runs_streamlit(python):
                return str(python)

    print(SETUP_HELP, file=sys.stderr)
    raise SystemExit(1)


def _theme_env() -> dict:
    """Streamlit theme settings from the dashboard's own palette module.

    Loaded straight from the file rather than `from trebuchet_sim.web import
    theme`: that would execute the package __init__, which imports numpy and
    scipy - and run.py has to work under an interpreter where nothing is
    installed yet. theme.py itself imports nothing, so this is safe.
    """
    theme_path = REPO_ROOT / "src" / "trebuchet_sim" / "web" / "theme.py"
    try:
        spec = importlib.util.spec_from_file_location("_trebuchet_theme", theme_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.streamlit_theme_env()
    except Exception:  # theme is cosmetic - never block startup over it
        return {}


def _find_free_port(start: int, attempts: int = 20) -> int:
    """First port at or after `start` that nothing is listening on.

    Probes by connecting, not binding: on Windows, binding can succeed even
    while another server (e.g. a previous run) is already listening.
    """
    for port in range(start, start + attempts):
        with socket.socket() as probe:
            probe.settimeout(0.25)
            if probe.connect_ex(("127.0.0.1", port)) != 0:
                return port
    print(f"No free port found in {start}-{start + attempts - 1}.", file=sys.stderr)
    raise SystemExit(1)


def _friendly_host_resolves() -> bool:
    try:
        return socket.gethostbyname(FRIENDLY_HOST) == "127.0.0.1"
    except OSError:
        return False


def _ensure_streamlit_credentials() -> None:
    """Pre-create Streamlit's credentials file so its first-run email prompt
    never appears (it blocks the server from starting)."""
    credentials = Path.home() / ".streamlit" / "credentials.toml"
    if not credentials.exists():
        credentials.parent.mkdir(parents=True, exist_ok=True)
        credentials.write_text('[general]\nemail = ""\n')


def _open_browser_when_ready(url: str, timeout_s: float = 30.0) -> None:
    """Poll the server until it responds, then open the browser."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1)
            webbrowser.open(url)
            return
        except OSError:
            time.sleep(0.3)


def _child_env() -> dict:
    """Child environment with src/ on PYTHONPATH (harmless if pip-installed)."""
    env = os.environ.copy()
    src_dir = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [src_dir, env.get("PYTHONPATH")]))
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    # Saved dashboard defaults (user_defaults.json) live in the repo root for
    # local checkouts; without this the app falls back to ~/.trebuchet-sim.
    env.setdefault("TREBUCHET_DATA_DIR", str(REPO_ROOT))
    # Streamlit resolves its theme at server start, before app.py runs, so the
    # palette has to arrive as config (see web/launcher.py for the same step on
    # the installed/console-script path).
    for key, value in _theme_env().items():
        env.setdefault(key, value)
    return env


def main() -> int:
    if not APP_PATH.exists():
        print(f"App not found at {APP_PATH} - run this from a full checkout.", file=sys.stderr)
        return 1

    python = _find_interpreter()
    port = _find_free_port(PREFERRED_PORT)
    _ensure_streamlit_credentials()

    host = FRIENDLY_HOST if _friendly_host_resolves() else "localhost"
    url = f"http://{host}:{port}"
    print(f"Starting Trebuchet Simulator at {url}  (Ctrl+C to stop)")
    if port != PREFERRED_PORT:
        print(f"Note: port {PREFERRED_PORT} was busy, using {port} instead.")
    if host == "localhost":
        print(
            f"Tip: to use http://{FRIENDLY_HOST}:{port} instead, run this once in an "
            f"elevated (admin) PowerShell, then restart run.py:\n"
            f'  Add-Content "$env:SystemRoot\\System32\\drivers\\etc\\hosts" '
            f'"`r`n127.0.0.1 {FRIENDLY_HOST}"'
        )

    threading.Thread(target=_open_browser_when_ready, args=(url,), daemon=True).start()

    # Headless keeps Streamlit from fighting us over browser launching and
    # from ever prompting on stdin; we open the browser ourselves above.
    # cwd=REPO_ROOT makes the src-layout package importable even when the
    # package isn't pip-installed and run.py is started from elsewhere.
    command = [
        python,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        f"--server.port={port}",
        "--server.headless=true",
        f"--browser.serverAddress={host}",
    ]
    try:
        return subprocess.call(command, cwd=REPO_ROOT, env=_child_env())
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
