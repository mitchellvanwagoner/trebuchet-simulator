"""Console-script entry point: `trebuchet-web` launches the Streamlit UI in a browser.

Also the Docker image's CMD, and the dev preview's launch target, so the theme
setup below applies everywhere the dashboard runs.
"""

import os
import sys
from pathlib import Path

from streamlit.web import cli as stcli

from trebuchet_sim.web import theme


def _ensure_streamlit_credentials() -> None:
    """Pre-create Streamlit's credentials file so its first-run email prompt
    never appears (it blocks the server from starting).

    Duplicated in run.py on purpose: run.py must work before the package is
    installed, so it cannot import this module."""
    credentials = Path.home() / ".streamlit" / "credentials.toml"
    if not credentials.exists():
        credentials.parent.mkdir(parents=True, exist_ok=True)
        credentials.write_text('[general]\nemail = ""\n')


def _apply_theme_env() -> None:
    """Publish the dashboard palette as Streamlit theme config.

    Streamlit resolves its theme at server start, before the app script runs,
    so app.py cannot set it - and a `.streamlit/config.toml` would not travel
    with the package (that path is git-ignored, and Streamlit resolves it from
    the working directory, which differs between a checkout and the container).
    Env vars are the one channel that works identically in both.

    setdefault, not assignment: an explicit STREAMLIT_THEME_* in the
    environment still wins.
    """
    for key, value in theme.streamlit_theme_env().items():
        os.environ.setdefault(key, value)


def main() -> None:
    _ensure_streamlit_credentials()
    _apply_theme_env()
    app_path = Path(__file__).parent / "app.py"
    # Extra arguments are forwarded to Streamlit (e.g. --server.port=8765), so
    # the dev preview can pin a port while still going through this entry point.
    sys.argv = ["streamlit", "run", str(app_path), *sys.argv[1:]]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
