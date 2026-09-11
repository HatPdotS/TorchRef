"""Dtype conformance: no unjustified hardcoded float dtype anywhere in the source.

The package resolves one float dtype at import (``config.get_float_dtype()``)
and every allocation on a live path is meant to honour it. A literal
``torch.float32`` / ``torch.float64`` baked into an allocation is a latent bug:
MPS has no float64, so a float64 config silently downcasts and a float32 config
silently upcasts -- neither raises, and the corruption surfaces far from its
cause.

This guard makes the rule enforceable. Outside a small set of inherently-exempt
modules (triton kernels, the config dtype maps, offline scripts), every
``torch.<floatdtype>`` reference must carry a one-line justification::

    x = torch.tensor(v, dtype=torch.float64)  # dtype-ok: SVD needs f64 stability

The marker may sit on the reference's own line or the line immediately above it.
The point is not to ban hardcoded dtypes -- some are correct (dtype validation,
deliberate high-precision accumulation, backend capability declarations) -- but
to force each one to say *why* it deviates, so a reviewer can tell a considered
choice from an oversight at a glance.
"""

from pathlib import Path

import pytest

from tests.helpers.dtype_inventory import (
    EXEMPT_PREFIXES,
    JUSTIFY_MARKER,
    find_hardcoded_dtypes,
    is_exempt,
)

_PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "torchref"

# The config getter each category should defer to, quoted in the failure message.
_GETTER = {
    "float": "config.get_float_dtype()",
    "int": "config.get_int_dtype()",
    "complex": "config.get_complex_dtype()",
}


@pytest.mark.unit
def test_no_unjustified_hardcoded_dtype():
    """Every hardcoded float/int/complex dtype on a live path is justified."""
    uses = find_hardcoded_dtypes(_PACKAGE_ROOT)
    offenders = [u for u in uses if not is_exempt(u.rel_path) and not u.justified]

    def fmt(u):
        return f"{u.where}  {u.dtype} -> {_GETTER[u.category]}  |  {u.line}"

    assert not offenders, (
        f"{len(offenders)} hardcoded dtype(s) with no justification. Either switch "
        f"to the config default for that category, or if the literal is deliberate "
        f"add a '{JUSTIFY_MARKER} <reason>' marker on the line or the block above:\n  "
        + "\n  ".join(fmt(u) for u in offenders)
    )


@pytest.mark.unit
def test_justifications_carry_a_reason():
    """A ``# dtype-ok:`` marker must be followed by an actual reason, not left blank."""
    blank = []
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if JUSTIFY_MARKER in line:
                reason = line.split(JUSTIFY_MARKER, 1)[1].strip()
                if not reason:
                    rel = path.relative_to(_PACKAGE_ROOT.parent)
                    blank.append(f"{rel}:{i}")

    assert not blank, (
        f"'{JUSTIFY_MARKER}' markers with no reason after the colon:\n  "
        + "\n  ".join(blank)
    )


@pytest.mark.unit
def test_exempt_prefixes_still_match_something():
    """Stop ``EXEMPT_PREFIXES`` accumulating entries for paths that are long gone.

    An exemption that no longer matches any source is either a typo or a stale
    excuse -- both hide the fact that the module it was meant to cover is now
    unguarded (or renamed and silently re-included).
    """
    uses = find_hardcoded_dtypes(_PACKAGE_ROOT)
    seen_paths = {u.rel_path for u in uses}
    stale = [
        p
        for p in EXEMPT_PREFIXES
        if not any(rp.startswith(p) for rp in seen_paths)
    ]
    assert not stale, (
        "EXEMPT_PREFIXES entries that match no source file with a hardcoded "
        f"float dtype (stale or mistyped): {stale}"
    )
