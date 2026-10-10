"""Static inventory of hardcoded float/int/complex dtypes in the torchref source.

The library resolves one dtype per category at import (``get_float_dtype`` /
``get_int_dtype`` / ``get_complex_dtype``) and every allocation on a live path is
expected to honour it. A literal ``torch.float32`` / ``torch.int64`` /
``torch.complex128`` baked into an allocation is a latent bug: MPS has no
float64, so a float64 config silently downcasts and a float32 config silently
upcasts -- neither raises, both corrupt results far from the cause.

This module finds every guarded ``torch.<dtype>`` reference by parsing the source
(AST, not regex, so dtypes named inside docstrings or comments do not count),
and reports whether each one carries an inline justification. The guard test in
``tests/unit/test_dtype_conformance.py`` turns that into a rule: outside a small
set of inherently-exempt modules, every hardcoded dtype must be justified with a
``# dtype-ok: <reason>`` marker on its own line or the comment block above it.

``torch.bool`` is not guarded: a mask is categorical, not numeric precision.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List, NamedTuple

__all__ = [
    "FLOAT_DTYPES",
    "INT_DTYPES",
    "COMPLEX_DTYPES",
    "GUARDED_DTYPES",
    "JUSTIFY_MARKER",
    "EXEMPT_PREFIXES",
    "DtypeUse",
    "find_hardcoded_dtypes",
    "is_exempt",
]

# The dtypes that must not be hardcoded on a live path -- each has a config
# default (get_float_dtype / get_int_dtype / get_complex_dtype) that an
# allocation is meant to honour. ``torch.bool`` is deliberately excluded: a mask
# is categorical, not numeric precision, so pinning it is correct not a deviation.
FLOAT_DTYPES = frozenset(
    {"float64", "float32", "float16", "double", "half", "bfloat16"}
)
INT_DTYPES = frozenset(
    {
        "int64", "int32", "int16", "int8",
        "uint8", "uint16", "uint32", "uint64",
        "long", "int", "short", "char", "byte",
    }
)
COMPLEX_DTYPES = frozenset(
    {"complex128", "complex64", "complex32", "cfloat", "cdouble", "chalf"}
)

# Category lookup so a finding can say which config default it should use.
_CATEGORY = {name: "float" for name in FLOAT_DTYPES}
_CATEGORY.update({name: "int" for name in INT_DTYPES})
_CATEGORY.update({name: "complex" for name in COMPLEX_DTYPES})
GUARDED_DTYPES = frozenset(_CATEGORY)

# A hardcoded float dtype is allowed when this marker appears on its line or the
# line immediately above it. The text after the colon is the required reason.
JUSTIFY_MARKER = "# dtype-ok:"

# Module path prefixes (relative to the package parent, e.g. "torchref/...")
# where hardcoded float dtypes are inherent to the file's job and a per-line
# marker would be noise rather than signal:
#   * triton kernels compile against explicit, static dtypes;
#   * config.py *defines* the dtype maps the rest of the code reads;
#   * scripts/ generate static on-disk tables offline, not model tensors.
EXEMPT_PREFIXES = (
    "torchref/base/targets/triton/",
    "torchref/base/direct_summation/triton_ds.py",
    "torchref/config.py",
    "torchref/scripts/",
)


class DtypeUse(NamedTuple):
    """One ``torch.<dtype>`` reference (float, int, or complex) in the source."""

    where: str  # "relative/path.py:lineno"
    rel_path: str  # "relative/path.py"
    lineno: int
    dtype: str  # e.g. "float64"
    category: str  # "float" | "int" | "complex"
    line: str  # the source line, stripped
    justified: bool  # carries JUSTIFY_MARKER on its line or the block above


def is_exempt(rel_path: str) -> bool:
    """Whether ``rel_path`` is in an inherently-exempt module."""
    return any(rel_path.startswith(p) for p in EXEMPT_PREFIXES)


def _is_torch_dtype(node: ast.AST) -> str | None:
    """Return the dtype name if ``node`` is a guarded ``torch.<dtype>``, else None."""
    if (
        isinstance(node, ast.Attribute)
        and node.attr in GUARDED_DTYPES
        and isinstance(node.value, ast.Name)
        and node.value.id == "torch"
    ):
        return node.attr
    return None


def find_hardcoded_dtypes(package_root: Path) -> List[DtypeUse]:
    """Every guarded ``torch.<dtype>`` reference under ``package_root``.

    Uses the AST so references inside strings and comments are not counted, then
    reads the raw source lines to decide whether each carries a justification.
    """
    uses: List[DtypeUse] = []

    for path in sorted(package_root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text, filename=str(path))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
            continue
        lines = text.splitlines()
        rel = str(path.relative_to(package_root.parent))

        for node in ast.walk(tree):
            dtype = _is_torch_dtype(node)
            if dtype is None:
                continue
            lineno = node.lineno
            this_line = lines[lineno - 1] if 0 < lineno <= len(lines) else ""
            # A marker counts if it is on the reference's own line, or anywhere in
            # the contiguous block of comment-only lines immediately above it -- so
            # a multi-line justification works with the marker on any of its lines.
            justified = JUSTIFY_MARKER in this_line
            i = lineno - 2
            while not justified and i >= 0 and lines[i].strip().startswith("#"):
                if JUSTIFY_MARKER in lines[i]:
                    justified = True
                i -= 1
            uses.append(
                DtypeUse(
                    where=f"{rel}:{lineno}",
                    rel_path=rel,
                    lineno=lineno,
                    dtype=dtype,
                    category=_CATEGORY[dtype],
                    line=this_line.strip(),
                    justified=justified,
                )
            )

    return uses
