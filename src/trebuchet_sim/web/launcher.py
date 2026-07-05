"""Console-script entry point: `trebuchet-web` launches the Streamlit UI in a browser."""

import sys
from pathlib import Path

from streamlit.web import cli as stcli


def _ensure_streamlit_credentials() -> None:
    """Pre-create Streamlit's credentials file so its first-run email prompt
    never appears (it blocks the server from starting).

    Duplicated in run.py on purpose: run.py must work before the package is
    installed, so it cannot import this module."""
    credentials = Path.home() / ".streamlit" / "credentials.toml"
    if not credentials.exists():
        credentials.parent.mkdir(parents=True, exist_ok=True)
        credentials.write_text('[general]\nemail = ""\n')


def main() -> None:
    _ensure_streamlit_credentials()
    app_path = Path(__file__).parent / "app.py"
    sys.argv = ["streamlit", "run", str(app_path)]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
