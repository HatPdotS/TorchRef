"""Share loaded objects within a functional module for read-only checks.

These fixtures capture the configured dtype/device at module setup. Callers may
populate derived caches but must not change parameters, tables, grids, masks,
device, or configuration. Tests of loading, mutation, and empty caches construct
fresh objects instead. No loaded objects are shared across test modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from torchref.model import ModelFT


@pytest.fixture(scope="module")
def shared_model_ft(sample_cif_file: Path) -> ModelFT:
    """Load a read-only Fourier model with a 2 Å resolution limit per module."""
    from torchref.model import ModelFT

    return ModelFT(max_res=2.0, verbose=0).load_cif(str(sample_cif_file))
