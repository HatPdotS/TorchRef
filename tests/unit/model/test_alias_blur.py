"""F_calc from the FFT route does not depend on the resolution the grid was built for.

Pinned on deposited 1DAW (3051 atoms, C2) in float32 against direct summation of the
same atoms: with the alias blur, reflections out to the grid's ``max_res`` stay within
``torchref.aliasing_tolerance`` of the analytic answer, so F over a fixed resolution
range agrees between grids built for different ``max_res``. All ADPs are also set to
10 A^2, which without the blur aliases by up to ~19% at 3 A, so the tests cannot pass
on blunt atoms alone; ``test_unblurred_grid_fails_the_same_gate`` keeps that honest.
"""

import pytest
import torch

import torchref
from torchref.base.reciprocal import generate_possible_hkl
from torchref.model import ModelFT
from torchref.model.sf_ds import SfDS

pytestmark = pytest.mark.unit

_RESOLUTIONS = (3.0, 2.8, 2.4)
_LOW_RES_LIMIT = 3.0


@pytest.fixture(scope="module")
def scene(pdb_dir):
    path = pdb_dir / "1DAW.pdb"
    if not path.exists():
        pytest.skip("1DAW.pdb fixture not present")
    model = ModelFT(max_res=_RESOLUTIONS[0], device="cpu").load_pdb(str(path))
    xyz, adp, occ, A, B = (t.detach() for t in model.get_iso())
    sharp = (xyz, torch.full_like(adp, 10.0), occ, A, B)
    deposited = (xyz, adp, occ, A, B)

    hkl = generate_possible_hkl(model.cell.data, min(_RESOLUTIONS))
    # Every 7th reflection still spans every shell; DS over all of them would dominate
    # the module's runtime without sharpening any gate.
    hkl = hkl[::7]
    s = (hkl.to(model.dtype_float) @ model.cell.inv_fractional_matrix).norm(dim=1)
    ds = SfDS(model.ctx, device="cpu")
    oracle = {
        name: ds.compute_structure_factors(hkl, *atoms)[0]
        for name, atoms in (("sharp", sharp), ("deposited", deposited))
    }
    return model, {"sharp": sharp, "deposited": deposited}, hkl, s, oracle


@pytest.fixture
def tolerance():
    original = torchref.aliasing_tolerance.value
    yield original
    torchref.aliasing_tolerance.value = original


def _fft(model, atoms, hkl, max_res):
    model.max_res = max_res
    return model.fft.compute_structure_factors(hkl, *atoms)[0]


def _rel(a, b):
    return float((a - b).norm() / b.norm())


@pytest.mark.parametrize("kind", ["sharp", "deposited"])
@pytest.mark.parametrize("max_res", _RESOLUTIONS)
def test_outer_shell_within_tolerance(scene, tolerance, kind, max_res):
    model, atoms, hkl, s, oracle = scene
    keep = s <= 1.0 / max_res
    F = _fft(model, atoms[kind], hkl[keep], max_res)
    outer = s[keep] > 0.9 / max_res
    assert _rel(F[outer], oracle[kind][keep][outer]) <= tolerance


@pytest.mark.parametrize("kind", ["sharp", "deposited"])
def test_low_resolution_fcalc_independent_of_grid(scene, tolerance, kind):
    """The #106 symptom: F over a fixed range moved with the grid's ``max_res``."""
    model, atoms, hkl, s, oracle = scene
    keep = s <= 1.0 / _LOW_RES_LIMIT
    F = {d: _fft(model, atoms[kind], hkl[keep], d) for d in _RESOLUTIONS}
    for d in _RESOLUTIONS[1:]:
        assert _rel(F[d], F[_RESOLUTIONS[0]]) <= tolerance
    for d in _RESOLUTIONS:
        assert _rel(F[d], oracle[kind][keep]) <= tolerance


def test_unblurred_grid_fails_the_same_gate(scene, tolerance):
    model, atoms, hkl, s, oracle = scene
    torchref.aliasing_tolerance.value = 1.0 - 1e-9
    max_res = _RESOLUTIONS[0]
    keep = s <= 1.0 / max_res
    F = _fft(model, atoms["sharp"], hkl[keep], max_res)
    outer = s[keep] > 0.9 / max_res
    assert _rel(F[outer], oracle["sharp"][keep][outer]) > 5 * tolerance


def test_blur_follows_the_atoms(scene, tolerance):
    model, atoms, _, _, _ = scene
    model.max_res = _RESOLUTIONS[0]
    sharp = float(model.fft.alias_blur(atoms["sharp"][1]))
    deposited = float(model.fft.alias_blur(atoms["deposited"][1]))
    assert sharp > deposited >= 0.0
    assert float(model.fft.alias_blur(atoms["sharp"][1] + 200.0)) == 0.0
