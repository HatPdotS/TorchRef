"""Static inventory of ``DeviceMixin`` subclasses in the torchref source tree.

Why AST rather than ``DeviceMixin.__subclasses__()``: the runtime hook only
sees classes whose defining module has been imported, so a coverage guard built
on it silently shrinks whenever an import is removed or an optional dependency
is missing -- exactly when you most want to be told about a gap. Parsing the
source finds every class whether or not it is importable here.

Inheritance is resolved **transitively**: ``PositiveMixedTensor(MixedTensor)``
never names a mixin alias itself, but it is still a device-bearing class and
still needs conformance coverage.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, Set, Tuple

__all__ = ["MIXIN_ALIASES", "device_mixin_classes"]

# Every spelling of the one mixin. ``DeviceMovementMixin`` and
# ``_NonModuleDeviceMixin`` are aliases kept for backward compatibility.
MIXIN_ALIASES = frozenset(
    {"DeviceMixin", "DeviceMovementMixin", "_NonModuleDeviceMixin"}
)


def _base_names(node: ast.ClassDef) -> Set[str]:
    """Base-class names for ``node``, ignoring subscripts and keywords."""
    names = set()
    for base in node.bases:
        if isinstance(base, ast.Name):
            names.add(base.id)
        elif isinstance(base, ast.Attribute):
            names.add(base.attr)
    return names


def device_mixin_classes(package_root: Path) -> Dict[str, str]:
    """Return ``{class_name: "relative/path.py:lineno"}`` for device-bearing classes.

    A class qualifies if any of its bases is a mixin alias, or is another
    qualifying class -- iterated to a fixed point so multi-level hierarchies
    are captured.
    """
    definitions: Dict[str, Tuple[Set[str], str]] = {}

    for path in sorted(package_root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
            continue
        rel = path.relative_to(package_root.parent)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                definitions[node.name] = (_base_names(node), f"{rel}:{node.lineno}")

    qualifying: Dict[str, str] = {}
    changed = True
    while changed:
        changed = False
        for name, (bases, where) in definitions.items():
            if name in qualifying:
                continue
            if bases & MIXIN_ALIASES or bases & qualifying.keys():
                qualifying[name] = where
                changed = True

    return qualifying
