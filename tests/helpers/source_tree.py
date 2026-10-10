"""Gate for tests that read TorchRef's source files rather than import the package.

Static checks (AST inventories of device-bearing classes, hardcoded dtypes) scan the
``torchref/`` directory beside ``tests/``. When the suite runs against an installed
wheel with only ``tests/`` present, there is nothing to scan: such a test would either
pass vacuously or report every allow-list entry as stale. It is skipped instead.
"""

from pathlib import Path

import pytest

#: The ``torchref`` source package next to ``tests/``, whether or not it exists.
PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "torchref"

#: Skip unless the source tree is present.
requires_source_tree = pytest.mark.skipif(
    not (PACKAGE_ROOT / "__init__.py").is_file(),
    reason=f"reads the torchref source tree, not present at {PACKAGE_ROOT}",
)
