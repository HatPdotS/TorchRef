"""Exercise explicitly named extra structure files in the slow compatibility tier."""

import pytest
import torch

from tests.helpers.structure_cases import (
    EXTENDED_PAIR_CODES,
    MODEL_CIF_FILES,
    MTZ_CODES,
    SF_CIF_CODES,
)
from torchref.config import (
    canonical_device,
    get_default_device,
    get_float_dtype,
    get_int_dtype,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "directory, expected",
    [
        ("cif", MODEL_CIF_FILES),
        ("mtz", tuple(f"{code}.mtz" for code in MTZ_CODES)),
        ("cif_sf", tuple(f"{code}-sf.cif" for code in SF_CIF_CODES)),
    ],
    ids=["models", "mtz", "sf-cif"],
)
def test_compatibility_inventory(test_files_dir, directory, expected) -> None:
    """Every bundled input has an explicit quick or extended coverage assignment."""
    suffix = ".mtz" if directory == "mtz" else ".cif"
    actual = {path.name for path in (test_files_dir / directory).glob(f"*{suffix}")}
    assert actual == set(expected)


@pytest.mark.slow
@pytest.mark.parametrize(
    "filename", [name for name in MODEL_CIF_FILES if name != "1DAW.cif"]
)
def test_model_cif_compatibility(cif_dir, filename) -> None:
    """Each extra CIF loads atoms and finite symmetry operators on the default device."""
    from torchref.model import Model

    path = cif_dir / filename
    assert path.is_file()
    model = Model(verbose=0).load_cif(str(path))
    xyz = model.xyz()
    assert xyz.shape == (len(model.pdb), 3)
    assert len(xyz) > 0
    assert xyz.dtype == get_float_dtype()
    assert canonical_device(xyz.device) == canonical_device(get_default_device())
    assert torch.isfinite(xyz).all()
    assert model.cell.data.shape == (6,)
    assert model.spacegroup.matrices.shape[0] > 0
    assert torch.isfinite(model.spacegroup.matrices).all()


@pytest.mark.slow
@pytest.mark.parametrize("code", EXTENDED_PAIR_CODES)
def test_modelft_cif_compatibility(cif_dir, code) -> None:
    """Fourier models initialize their scattering parametrization in distinct crystals."""
    from torchref.model import ModelFT

    path = cif_dir / f"{code}.cif"
    assert path.is_file()
    model = ModelFT(max_res=3.0, verbose=0).load_cif(str(path))
    assert len(model.xyz()) > 0
    assert model.parametrization
    assert all(size > 0 for size in model.grid_shape)


@pytest.mark.slow
@pytest.mark.parametrize(
    "directory, filename, loader",
    [("mtz", f"{code}.mtz", "load_mtz") for code in MTZ_CODES if code != "1DAW"]
    + [
        ("cif_sf", f"{code}-sf.cif", "load_cif")
        for code in SF_CIF_CODES
        if code != "1DAW"
    ],
    ids=[f"mtz-{code}" for code in MTZ_CODES if code != "1DAW"]
    + [f"sf-cif-{code}" for code in SF_CIF_CODES if code != "1DAW"],
)
def test_reflection_file_compatibility(
    test_files_dir, directory, filename, loader
) -> None:
    """Every named MTZ/SF-CIF must load reflections; one success cannot mask another failure."""
    from torchref.io import ReflectionData

    path = test_files_dir / directory / filename
    assert path.is_file()
    data = ReflectionData(verbose=0)
    getattr(data, loader)(str(path))
    assert data.hkl.shape == (len(data.hkl), 3)
    assert len(data.hkl) > 0
    assert data.hkl.dtype == get_int_dtype()
    assert canonical_device(data.hkl.device) == canonical_device(get_default_device())
    assert data.cell.data.shape == (6,)
    assert data.F.shape == (len(data.hkl),)
