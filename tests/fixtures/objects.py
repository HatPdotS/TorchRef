"""Load fresh mutable models, reflection data, scalers, and restraints.

Function-scoped fixtures isolate test mutations. The explicitly shared device
bundle caches one model per device and must be treated as read-only by callers.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import torch

if TYPE_CHECKING:
    from torchref.io import ReflectionData
    from torchref.model import Model, ModelFT
    from torchref.scaling import Scaler


@pytest.fixture
def loaded_model(sample_cif_file: Path) -> Model:
    """Load a fresh mutable Model from the sample CIF file."""
    from torchref.model.model import Model

    model = Model()
    model.load_cif(str(sample_cif_file))
    return model


@pytest.fixture
def loaded_model_ft(sample_cif_file: Path) -> ModelFT:
    """Load a fresh mutable Fourier model with a 2 Å resolution limit."""
    from torchref.model import ModelFT

    return ModelFT(max_res=2.0, verbose=0).load_cif(str(sample_cif_file))


@pytest.fixture
def loaded_reflection_data(sample_mtz_file: Path) -> ReflectionData:
    """Load fresh mutable reflection data from the sample MTZ file."""
    from torchref.io import ReflectionData

    data = ReflectionData()
    data.load_mtz(str(sample_mtz_file))
    return data


@pytest.fixture
def model_and_data(sample_structure_pair: dict[str, Path]) -> dict[str, Any]:
    """Load a fresh matching model and reflection dataset."""
    from torchref.io import ReflectionData
    from torchref.model.model import Model

    model = Model()
    model.load_cif(str(sample_structure_pair["model"]))

    data = ReflectionData()
    data.load_mtz(str(sample_structure_pair["reflections"]))

    return {"model": model, "data": data}


@pytest.fixture
def model_with_symmetry(loaded_model: Model) -> dict[str, Any]:
    """Pair a fresh model with initialized symmetry."""
    from torchref.symmetry import SpaceGroup

    sg = SpaceGroup(loaded_model.spacegroup)
    return {"model": loaded_model, "symmetry": sg}


@pytest.fixture
def initialized_scaler(model_and_data: dict[str, Any]) -> Scaler:
    """Build a scaler around a fresh matching model and dataset."""
    from torchref.scaling.scaler import Scaler

    model = model_and_data["model"]
    data = model_and_data["data"]

    scaler = Scaler(model=model, data=data, nbins=10, verbose=0)
    return scaler


@pytest.fixture
def model_with_restraints(loaded_model: Model) -> dict[str, Any]:
    """Build restraints around a fresh model."""
    from torchref.topology.restraints import Restraints

    restraints = Restraints(
        pdb=loaded_model.pdb,
        xyz_fn=loaded_model.xyz,
        vdw_radii_fn=loaded_model.get_vdw_radii,
        verbose=0,
    )
    restraints.build_restraints()
    return {"model": loaded_model, "restraints": restraints}


@pytest.fixture(scope="session")
def all_test_structures(
    all_structure_pairs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return all loaded model/data pairs for comprehensive testing."""
    from torchref.io import ReflectionData
    from torchref.model.model import Model

    structures = []
    for pair in all_structure_pairs:
        try:
            model = Model()
            model.load_cif(str(pair["model"]))

            data = ReflectionData()
            data.load_mtz(str(pair["reflections"]))

            structures.append(
                {
                    "pdb_id": pair["pdb_id"],
                    "model": model,
                    "data": data,
                    "model_path": pair["model"],
                    "data_path": pair["reflections"],
                }
            )
        except Exception:
            # Skip structures that fail to load
            continue

    if not structures:
        pytest.skip("No structures could be loaded")

    return structures


@pytest.fixture(scope="session")
def _device_model_cache() -> dict:
    """``{device_str: ModelFT}`` built at most once per device, per session."""
    return {}


@pytest.fixture
def device_model_bundle(
    _device_model_cache: dict[str, ModelFT], pdb_dir: Path, any_device: torch.device
) -> dict[str, ModelFT]:
    """Borrow a session-shared model on the requested device.

    Notes
    -----
    Treat the model as read-only, including when a target borrows it. Moving a
    target can move its model too; tests of movement need a fresh model.
    """
    key = str(any_device)
    if key not in _device_model_cache:
        pdb = pdb_dir / "1DAW.pdb"
        if not pdb.exists():
            pytest.skip("1DAW.pdb fixture not present")
        from torchref.model import ModelFT

        _device_model_cache[key] = ModelFT(device=any_device, verbose=0).load_pdb(
            str(pdb)
        )
    return {"model": _device_model_cache[key]}
