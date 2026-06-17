"""Unit tests for ``Model.set_adp_mode`` — the isotropic/anisotropic ADP switch.

Verifies the conversion (not freeze) of atoms between isotropic ``adp`` (B) and
anisotropic ``u`` (U) representations, the per-atom ``aniso_flag`` repartition,
and that the written PDB follows the ANISOU convention (ANISOU only for atoms
refined anisotropically). Uses 7L84 (a mixed iso/aniso deposited model).
"""

import math

import pytest
import torch

EIGHT_PI_SQ = 8.0 * math.pi**2


@pytest.fixture
def aniso_model(pdb_dir):
    """A mixed iso/aniso model (7L84 carries ANISOU for heavy atoms)."""
    from torchref.model.model_ft import ModelFT

    m = ModelFT()
    m.load_pdb(str(pdb_dir / "7L84.pdb"))
    return m


def _written_anisou(model, tmp_path):
    out = tmp_path / "out.pdb"
    model.write_pdb(str(out))
    with open(out) as f:
        return sum(1 for ln in f if ln.startswith("ANISOU"))


def _resname(model):
    return model.pdb["resname"].astype(str).str.upper().str.strip()


def _element(model):
    return model.pdb["element"].astype(str).str.upper().str.strip()


@pytest.mark.unit
def test_fixture_has_aniso(aniso_model):
    """7L84 loads with anisotropic atoms (it is a deposited ANISOU model)."""
    assert aniso_model.aniso_flag.any()


@pytest.mark.unit
def test_isotropic_converts_all_atoms(aniso_model, tmp_path):
    m = aniso_model
    orig_flag = m.aniso_flag.clone()
    orig_U = m.u().detach().clone()

    m.set_adp_mode("isotropic")

    # every atom isotropic; the aniso SF subset is empty
    assert not m.aniso_flag.any()
    assert m._aniso_indices.numel() == 0
    # u carries NaN everywhere (iso convention); B is finite everywhere
    assert torch.isnan(m.u()).all()
    assert torch.isfinite(m.adp()).all()
    # B for originally-anisotropic atoms equals the equivalent B_eq of their U
    beq = (EIGHT_PI_SQ / 3.0) * (orig_U[:, 0] + orig_U[:, 1] + orig_U[:, 2])
    idx = orig_flag.nonzero(as_tuple=True)[0]
    assert torch.allclose(m.adp()[idx], beq[idx], atol=1e-3, rtol=1e-3)
    # masks: all B refinable, no U refinable
    assert bool(m.adp_mask.all())
    assert not bool(m.u_mask.any())
    # PDB convention: no ANISOU records written
    assert _written_anisou(m, tmp_path) == 0


@pytest.mark.unit
def test_anisotropic_promotes_non_water_heavy(aniso_model, tmp_path):
    m = aniso_model
    is_hoh = torch.tensor((_resname(m) == "HOH").values)
    is_h = torch.tensor((_element(m) == "H").values)
    expect = (~is_hoh) & (~is_h)

    m.set_adp_mode("anisotropic")  # default selection: not resname HOH and not element H

    assert torch.equal(m.aniso_flag.cpu(), expect)
    # selected atoms carry finite U; everything else NaN
    finite_U = torch.isfinite(m.u().detach()).all(dim=1).cpu()
    assert torch.equal(finite_U, expect)
    # no water / hydrogen got promoted
    assert not bool((m.aniso_flag.cpu() & is_hoh).any())
    assert not bool((m.aniso_flag.cpu() & is_h).any())
    # masks partition: U refinable iff anisotropic, B refinable iff isotropic
    assert torch.equal(m.u_mask.cpu(), expect)
    assert torch.equal(m.adp_mask.cpu(), ~expect)
    # ANISOU written for exactly the anisotropic atoms
    assert _written_anisou(m, tmp_path) == int(expect.sum())


@pytest.mark.unit
def test_anisotropic_expansion_is_isotropic_tensor(aniso_model):
    """Atoms promoted from iso -> aniso get U = (B / 8pi^2) * I (diagonal).

    Go fully isotropic first so the subsequent anisotropic conversion must
    expand U from B for every selected atom (exercises the iso->aniso path).
    """
    m = aniso_model
    m.set_adp_mode("isotropic")
    B0 = m.adp().detach().clone()

    m.set_adp_mode("anisotropic")

    U = m.u().detach()
    promoted = m.aniso_flag  # all were isotropic, so every aniso atom was expanded
    assert promoted.any()
    idx = promoted.nonzero(as_tuple=True)[0]
    u_iso = B0[idx] / EIGHT_PI_SQ
    # diagonal U components ~ B/8pi^2, off-diagonals ~ 0
    assert torch.allclose(U[idx, 0], u_iso, atol=1e-4, rtol=1e-3)
    assert torch.allclose(U[idx, 1], u_iso, atol=1e-4, rtol=1e-3)
    assert torch.allclose(U[idx, 2], u_iso, atol=1e-4, rtol=1e-3)
    assert torch.allclose(U[idx, 3:], torch.zeros_like(U[idx, 3:]), atol=1e-4)


@pytest.mark.unit
def test_anisotropic_selection_subset(aniso_model, tmp_path):
    m = aniso_model
    chain = sorted(set(m.pdb["chainid"].astype(str)))[0]
    in_chain = torch.tensor((m.pdb["chainid"].astype(str) == chain).values)

    m.set_adp_mode("anisotropic", aniso_selection=f"chain {chain}")

    aniso = m.aniso_flag.cpu()
    assert int(aniso.sum()) > 0
    # only chain `chain` atoms became anisotropic
    assert bool((aniso <= in_chain).all())
    assert _written_anisou(m, tmp_path) == int(aniso.sum())


@pytest.mark.unit
def test_sf_subset_inputs_finite_across_conversions(aniso_model):
    """The B fed for iso atoms and the U fed for aniso atoms must be finite
    (NaN would blow up the structure-factor FFT)."""
    m = aniso_model
    for mode in ("isotropic", "anisotropic", "isotropic"):
        m.set_adp_mode(mode)
        iso_idx = (~m.aniso_flag).nonzero(as_tuple=True)[0]
        aniso_idx = m.aniso_flag.nonzero(as_tuple=True)[0]
        assert torch.isfinite(m.adp()[iso_idx]).all()
        if aniso_idx.numel():
            assert torch.isfinite(m.u()[aniso_idx]).all()


@pytest.mark.unit
def test_roundtrip_preserves_pattern(aniso_model):
    """iso -> aniso -> iso ends fully isotropic; aniso pattern is reproducible."""
    m = aniso_model
    m.set_adp_mode("anisotropic")
    flag_a = m.aniso_flag.clone()
    m.set_adp_mode("isotropic")
    assert not m.aniso_flag.any()
    m.set_adp_mode("anisotropic")
    assert torch.equal(m.aniso_flag, flag_a)


@pytest.mark.unit
def test_invalid_mode_raises(aniso_model):
    with pytest.raises(ValueError):
        aniso_model.set_adp_mode("banana")
