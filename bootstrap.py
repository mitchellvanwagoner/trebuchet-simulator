#!/usr/bin/env python3
r"""Create or repair this project's .venv, and make what can be relative relative.

Usage, under any Python at all - the system one, the py launcher, a fresh
install that has never seen this project:

    py bootstrap.py                 # create/repair .venv, install the project
    py bootstrap.py --extras dev    # a different extras set (default: dev,fast)

A virtual environment is not a portable thing, and it is worth being precise
about which parts of that are fixable, because only one of them is:

  * `pyvenv.cfg` records `home` - an absolute path to the *base* interpreter
    the venv was built from (here `C:\Python314`). That path is outside the
    project, so no amount of relative-path work reaches it, and it differs on
    every machine: it is what breaks when this folder is synced between two
    Windows accounts, because the other one keeps its Python under
    `C:\Users\<name>\AppData\Local\Python\...`. Rebuilding is the only repair,
    which is what this script does.

  * `Scripts\*.exe` console wrappers (pytest.exe, trebuchet.exe, streamlit.exe)
    embed an absolute shebang to the venv's own python.exe. That launcher
    format has no relative form. Prefer `python -m pytest` / `python -m
    streamlit`, which never touch them.

  * pip's editable install drops a `.pth` naming an absolute path to `src/`.
    This one *is* fixable: site.py resolves a non-absolute `.pth` line against
    the site-packages directory it was found in, so a relative line survives
    the project folder being moved or renamed on the same machine. `relink()`
    rewrites it after every install, since pip writes it absolute each time.

  * The installed dependencies (numba, scipy, numpy) are compiled against one
    Python ABI and one OS. Even with every path relative, a Windows .venv is
    not going to run on macOS. Copying the *project* and rebuilding the venv is
    the portable operation; copying the venv is not.

So: a copied checkout needs this script run once, and nothing else.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
VENV_DIR = REPO_ROOT / ".venv"
DEFAULT_EXTRAS = "dev,fast"


def venv_pythons() -> "list[Path]":
    """Candidate interpreter paths inside the venv, both platform layouts."""
    return [
        VENV_DIR / "Scripts" / "python.exe",  # Windows
        VENV_DIR / "bin" / "python",          # macOS/Linux
    ]


def working_python(module: str = "trebuchet_sim") -> "Path | None":
    """The venv interpreter, if it both runs and can import `module`.

    Both halves matter, and they fail differently. A venv carried over from
    another machine still *has* a python.exe, but it is only a launcher: it
    reads `pyvenv.cfg` for the base interpreter and dies with "did not find
    executable" when that path does not exist here. A venv that runs but cannot
    import the project has a stale or missing editable install. Either way the
    repair is the same rebuild, so this collapses them into one test.
    """
    for python in venv_pythons():
        if not python.exists():
            continue
        try:
            probe = subprocess.run([str(python), "-c", f"import {module}"],
                                   capture_output=True)
        except OSError:
            continue
        if probe.returncode == 0:
            return python
    return None


def _site_packages(python: Path) -> "list[Path]":
    code = "import json, site; print(json.dumps(site.getsitepackages()))"
    try:
        out = subprocess.check_output([str(python), "-c", code], text=True)
    except (subprocess.CalledProcessError, OSError):
        return []
    return [Path(p) for p in json.loads(out)]


def relink(python: Path) -> "list[Path]":
    """Rewrite the editable install's `.pth` entries as relative paths.

    pip writes an absolute path to `src/` on every install, so this runs after
    each one. Only lines that are absolute paths *inside this project* are
    touched: a `.pth` may also carry `import ...` lines, which site.py executes
    rather than resolving, and absolute paths to somewhere else entirely, which
    are not ours to rewrite.

    Returns the files it changed.
    """
    changed = []
    for site_dir in _site_packages(python):
        if not site_dir.is_dir():
            continue
        for pth in sorted(site_dir.glob("*.pth")):
            try:
                lines = pth.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError):
                continue
            out, dirty = [], False
            for line in lines:
                stripped = line.strip()
                if not stripped or stripped.startswith(("#", "import ")):
                    out.append(line)
                    continue
                path = Path(stripped)
                if not path.is_absolute() or REPO_ROOT not in path.parents:
                    out.append(line)
                    continue
                out.append(os.path.relpath(path, site_dir))
                dirty = True
            if dirty:
                pth.write_text("\n".join(out) + "\n", encoding="utf-8")
                changed.append(pth)
    return changed


def bootstrap(extras: str = DEFAULT_EXTRAS, quiet: bool = False) -> "Path | None":
    """Rebuild .venv in place and reinstall the project into it.

    `python -m venv` over an existing directory rewrites `pyvenv.cfg` and the
    launchers without deleting the packages already there, so this is a repair
    rather than a wipe. The editable reinstall then re-points the project path
    and refreshes anything built for a previous Python's ABI.

    Returns the interpreter, or None if this is running from the very venv it
    would have to rebuild - those files are in use, and the caller is better
    placed to say so than a half-finished rebuild is.
    """
    if Path(sys.executable).resolve().is_relative_to(VENV_DIR):
        return None

    say = (lambda *a: None) if quiet else print
    say(f"Setting up {VENV_DIR.name} (missing or stale) - this runs once after a copy or a fresh clone.")
    target = f".[{extras}]" if extras else "."
    try:
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV_DIR)])
        python = next(p for p in venv_pythons() if p.exists())
        subprocess.check_call([str(python), "-m", "pip", "install", "--upgrade",
                               "pip", "--quiet"])
        subprocess.check_call([str(python), "-m", "pip", "install", "-e", target],
                              cwd=REPO_ROOT)
    except (subprocess.CalledProcessError, StopIteration, OSError) as exc:
        print(f"Automatic setup failed ({exc}).", file=sys.stderr)
        return None

    for pth in relink(python):
        say(f"Made {pth.name} relative to its site-packages.")
    return python


def ensure(module: str = "trebuchet_sim", extras: str = DEFAULT_EXTRAS,
           quiet: bool = False) -> "Path | None":
    """The venv interpreter, repairing the venv first if it is broken or stale."""
    return working_python(module) or bootstrap(extras, quiet=quiet)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create or repair this project's .venv.")
    parser.add_argument("--extras", default=DEFAULT_EXTRAS,
                        help=f"pip extras to install (default: {DEFAULT_EXTRAS!r}; empty for none)")
    parser.add_argument("--force", action="store_true",
                        help="rebuild even if the venv already works")
    args = parser.parse_args()

    if not args.force:
        python = working_python()
        if python is not None:
            relink(python)
            print(f"{VENV_DIR} already works ({python}).")
            return 0

    python = bootstrap(args.extras)
    if python is None:
        print("Cannot rebuild the venv from inside it - re-run with a different "
              "Python, e.g. `py bootstrap.py`.", file=sys.stderr)
        return 1
    print(f"Ready: {python}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
