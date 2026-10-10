"""``torchref.base`` sits below the symmetry, dataset and refinement layers.

Pinned for the French-Wilson and Wilson-outlier tensor math: the space-group
bookkeeping they need belongs to the io layer that calls them.
"""

import ast
from pathlib import Path

import pytest

import torchref

_HIGHER_LAYERS = {
    "torchref.cli",
    "torchref.experimental",
    "torchref.io",
    "torchref.maps",
    "torchref.model",
    "torchref.refinement",
    "torchref.scaling",
    "torchref.symmetry",
    "torchref.topology",
}


def _upward_imports(path: Path) -> set[str]:
    """Absolute imports in ``path``, function-level ones included, above base."""
    modules = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module)
    return {
        m
        for m in modules
        if m == "torchref" or ".".join(m.split(".")[:2]) in _HIGHER_LAYERS
    }


@pytest.mark.unit
@pytest.mark.parametrize("module", ["french_wilson.py", "wilson_outliers.py"])
def test_base_module_imports_nothing_above_base(module):
    path = Path(torchref.__file__).parent / "base" / module
    assert not _upward_imports(path)
