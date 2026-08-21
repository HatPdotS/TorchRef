"""Benchmark structures and case loading.

The ten deposited structures the alignment work is measured on. Paths are
resolved relative to this file, never hardcoded: the drivers inherited from the
old worktree pointed at an absolute path inside a stale checkout, so they read
data and code from a different tree than the one under test.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

    from torchref.io.datasets.reflection_data import ReflectionData
    from torchref.model import ModelFT

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_FILES = REPO_ROOT / "tests" / "files"

#: Benchmark structures, in the order the seed formula depends on.
#: ``seed_for`` uses ``BENCH_PDBS.index(pdb)``, so **inserting or reordering
#: entries changes every seed** and silently invalidates comparisons against
#: archived results. Append only.
BENCH_PDBS: Tuple[str, ...] = (
    "1DAW",   # C2,      small  -- the fast control; use it for anything quick
    "3E98",   # P2_1,    control (note: pandas reads the string "3E98" as a float)
    "3A5V",   # I422
    "3VRJ",
    "1AK5",   # P432,    cubic ghost case
    "3K7M",   # P432,    the primary ghost case
    "3GR5",   # P6_522
    "2DQ6",   # P3_121,  tNCS
    "4BX9",   # P4_32_12, large; the only benchmark entry carrying ANISOU
    "6G9X",   # large
)

#: PDB filename stems, where they differ from the code. 1AK5 is the only one.
PDB_STEMS = {"1AK5": "1AK5_with_H"}

#: Present in tests/files but deliberately excluded from BENCH_PDBS:
#: 5BOV (a single translation-function allocation OOMs an A100-40GB) and
#: 7L84 (no matching MTZ).


def case_paths(pdb: str) -> Tuple[Path, Path]:
    """Return ``(pdb_path, mtz_path)`` for a benchmark code.

    Parameters
    ----------
    pdb : str
        Benchmark structure code, e.g. ``"1DAW"``.

    Returns
    -------
    tuple of pathlib.Path
        Model and reflection file paths.

    Raises
    ------
    FileNotFoundError
        If either file is missing, named so the caller sees which one.
    """
    stem = PDB_STEMS.get(pdb, pdb)
    pdb_path = TEST_FILES / "pdb" / f"{stem}.pdb"
    mtz_path = TEST_FILES / "mtz" / f"{pdb}.mtz"
    for p in (pdb_path, mtz_path):
        if not p.exists():
            raise FileNotFoundError(f"{pdb}: missing {p}")
    return pdb_path, mtz_path


def load_case(pdb: str, device: str = "cpu") -> Tuple["ModelFT", "ReflectionData"]:
    """Load the deposited model and its reflections.

    Parameters
    ----------
    pdb : str
        Benchmark structure code.
    device : str, optional
        Torch device for both objects. Default ``"cpu"``.

    Returns
    -------
    tuple
        ``(model, data)``.
    """
    from torchref.io.datasets.reflection_data import ReflectionData
    from torchref.model import ModelFT

    pdb_path, mtz_path = case_paths(pdb)
    model = ModelFT(device=device).load_pdb(str(pdb_path))
    data = ReflectionData(device=device).load_mtz(str(mtz_path))
    return model, data


def rotated_case(
    pdb: str, seed: int, device: str = "cpu",
) -> Tuple["ModelFT", "ReflectionData", "torch.Tensor"]:
    """Load a case and rotate a copy of the model by a seeded random rotation.

    The returned model is a **copy**: ``Model.rotate`` mutates in place and
    returns ``self``, so rotating the loaded model directly would also move the
    reference a caller may want to compare against.

    Parameters
    ----------
    pdb : str
        Benchmark structure code.
    seed : int
        Seed for :func:`~alignment_lab.lab.truth.random_rotation`.
    device : str, optional
        Torch device. Default ``"cpu"``.

    Returns
    -------
    tuple
        ``(rotated_model, data, R_true)`` with ``R_true`` in float64.
    """
    from .truth import random_rotation

    model, data = load_case(pdb, device=device)
    R_true = random_rotation(seed)
    rotated = model.copy().rotate(
        R_true.to(model.dtype_float), center=model.xyz().mean(0),
    )
    return rotated, data, R_true
