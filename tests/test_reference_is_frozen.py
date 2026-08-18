"""``reference/`` is evidence, and evidence that changes is not evidence.

The reference is only an oracle for as long as it is unmodified. A "small
fix" to ``wade.R`` — the ``tail_conc`` guard is the obvious temptation and
the ``nprobs == 1`` reshape is the other — would destroy the only
independent check the port has, and would do it silently: the fixtures
would simply start describing the patched behaviour and every parity
assertion would keep passing.

So the byte-identity is asserted by the test suite rather than left to
discipline. Fixes belong in the port, with the divergence documented as
intentional (``test_divergences.py``); where a test needs a patched R, the
patch belongs in the harness, never in ``reference/``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
REF_R = REPO / "reference" / "R"
SUMS = REF_R / "sha256sums.txt"


def _expected() -> dict[str, str]:
    entries = {}
    for line in SUMS.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        digest, name = line.split(None, 1)
        entries[name.lstrip("*").strip()] = digest
    return entries


def test_checksum_manifest_is_present_and_covers_wade_R():
    assert SUMS.exists(), f"missing {SUMS}"
    entries = _expected()
    assert "wade.R" in entries
    # The cfRNA consumer layer was pruned once the port was verified; the
    # manifest tracks what remains. wade.R is the entry that matters — it is
    # the oracle the golden fixtures were generated from.
    assert not any(e.startswith("downstream/") for e in entries)
    assert len(entries) == 5, f"expected 5 checksummed files, found {len(entries)}"


@pytest.mark.parametrize("name", sorted(_expected()))
def test_reference_file_is_byte_identical(name):
    path = REF_R / name
    assert path.exists(), f"{name} is listed in sha256sums.txt but missing"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == _expected()[name], (
        f"reference/R/{name} has been MODIFIED.\n"
        f"  expected {_expected()[name]}\n"
        f"  found    {digest}\n"
        f"reference/ is frozen — it is the parity oracle. Revert the change and "
        f"put the fix in the port instead."
    )


def test_the_seam_does_not_edit_the_reference():
    """The injection wrapper must source ``wade.R``, never rewrite it.

    The seam shadows ``sample`` and ``::`` through the enclosing
    environment chain, so nothing is assigned into any package namespace
    and the file on disk is untouched. Asserted structurally so a future
    "simpler" seam that patches the source cannot land unnoticed.
    """
    seam = (REPO / "tools" / "r" / "wade_seam.R").read_text()
    assert "sys.source" in seam
    for forbidden in ("writeLines(", "cat(", "file.copy", "gsub(", "sub("):
        assert f"{forbidden}wade" not in seam.replace(" ", "")
    assert "assignInNamespace" not in seam, (
        "the seam must not mutate a package namespace; it shadows lexically"
    )
