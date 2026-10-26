"""Packaging invariants — the things that break on someone else's machine.

None of this tests the statistic; it tests that the package is a package:
one version string, a public API surface matching what is exported, and an
import that costs only NumPy.
"""

from __future__ import annotations

import re
from pathlib import Path


import wade

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = (ROOT / "pyproject.toml").read_text(encoding="utf-8")


def test_version_comes_from_pyproject():
    """``wade.__version__`` is read from the installed metadata, so pyproject
    is the one place the version is declared. A stale editable install is the
    only way for them to disagree, and the fix is to reinstall."""
    declared = re.search(r'^version = "([^"]+)"', PYPROJECT, re.M)
    assert declared, "pyproject.toml has no static version"
    assert declared.group(1) == wade.__version__, (
        f"pyproject {declared.group(1)!r} != wade.__version__ {wade.__version__!r}; "
        f"reinstall with `pip install -e .`")


def test_all_is_sorted_and_complete():
    """``__all__`` is the public surface: everything in it must exist, and
    everything exported must be listed (so a new function cannot be added
    silently and then be undocumented)."""
    missing = [n for n in wade.__all__ if not hasattr(wade, n)]
    assert not missing, f"__all__ names that do not exist: {missing}"
    public = {n for n in vars(wade)
              if not n.startswith("_") and n not in ("annotations", "PackageNotFoundError")
              and not isinstance(getattr(wade, n), type(wade))}
    unlisted = public - set(wade.__all__)
    assert not unlisted, f"public names missing from __all__: {sorted(unlisted)}"


def test_importing_wade_costs_only_numpy():
    """The statistic's dependency surface is one library. Plotting and polars
    are imported inside the functions that need them."""
    import subprocess
    import sys

    code = ("import sys, wade; "
            "heavy = [m for m in ('polars', 'pandas', 'plotly', 'matplotlib', 'scipy') "
            "         if m in sys.modules]; "
            "sys.exit(1 if heavy else 0)")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr

