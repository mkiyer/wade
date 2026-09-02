"""Packaging invariants — the things that break on someone else's machine.

None of this tests the statistic; it tests that the package is a package:
one version string, the optional extras actually importable when installed,
the declared pytest markers matching the ones in use, and the public API
surface matching what is exported.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import wade

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = (ROOT / "pyproject.toml").read_text(encoding="utf-8")


def test_version_is_declared_once_and_agrees():
    """``wade.__version__`` is what the manifest records, pyproject is what pip
    reports, and Cargo.toml is what the crate calls itself. A drift mislabels
    every written result.

    Cargo's was found at 0.1.0 against pyproject's 0.2.0.dev0 on 2026-09-02,
    because nothing compared them. It is compared on the **release segment**
    only: Cargo takes semver, which cannot spell a PEP 440 ``.dev0``, so a
    pre-release suffix on the Python side is allowed to have no counterpart.
    """
    declared = re.search(r'^version = "([^"]+)"', PYPROJECT, re.M)
    assert declared, "pyproject.toml has no static version"
    assert declared.group(1) == wade.__version__, (
        f"pyproject {declared.group(1)!r} != wade.__version__ {wade.__version__!r}")

    crate = re.search(r'^version = "([^"]+)"', (ROOT / "Cargo.toml").read_text(encoding="utf-8"), re.M)
    assert crate, "Cargo.toml has no version"
    release = re.match(r"\d+\.\d+\.\d+", declared.group(1)).group(0)
    assert crate.group(1) == release, (
        f"Cargo.toml {crate.group(1)!r} != pyproject's release segment {release!r}")


def test_every_declared_marker_is_used_and_every_used_marker_declared():
    declared = set(re.findall(r'^\s*"(\w+): ', PYPROJECT, re.M))
    used = set()
    for f in (ROOT / "tests").glob("test_*.py"):
        used |= set(re.findall(r"pytest\.mark\.(\w+)", f.read_text(encoding="utf-8")))
    used -= {"parametrize", "skipif", "xfail", "skip", "filterwarnings", "usefixtures"}
    assert used <= declared, f"markers used but not declared: {sorted(used - declared)}"
    assert declared <= used, f"markers declared but unused: {sorted(declared - used)}"


def test_all_is_sorted_and_complete():
    """``__all__`` is the public surface: everything in it must exist, and
    everything exported must be listed (so a new function cannot be added
    silently and then be undocumented)."""
    missing = [n for n in wade.__all__ if not hasattr(wade, n)]
    assert not missing, f"__all__ names that do not exist: {missing}"
    public = {n for n in vars(wade)
              if not n.startswith("_") and n not in ("annotations",)
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


@pytest.mark.parametrize("extra", ["test", "plot", "io"])
def test_optional_extras_are_declared(extra):
    assert re.search(rf"^{extra} = \[", PYPROJECT, re.M), f"extra {extra!r} not declared"
