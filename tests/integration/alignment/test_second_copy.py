"""Place the second chain of a homodimer with the first held fixed.

3E98 (P2(1)) has two sequence-identical chains. With chain A fixed at its
deposited position, chain B reoriented at random must come back to its own
site: with a fixed component the origin is pinned, so the only acceptable
answers are B's deposited site and its symmetry images, and A's site is
excluded by the packing check.
"""
from pathlib import Path

import pytest
import torch

from torchref.experimental.alignment import MolecularReplacementPipeline
from torchref.io.datasets.reflection_data import ReflectionData
from torchref.model import ModelFT

TEST_FILES = Path(__file__).resolve().parents[2] / "files"


def _chain_selection(model, chain):
    pdb = model.pdb
    het = sorted(set(pdb.loc[(pdb["chainid"] == chain) & (pdb["ATOM"] == "HETATM"),
                             "resname"].astype(str)))
    sel = f"chain {chain}"
    if het:
        sel += " and not (" + " or ".join(f"resname {r}" for r in het) + ")"
    return sel


def _random_rotation(seed):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(3, 3, generator=g, dtype=torch.float64)
    Q, R = torch.linalg.qr(A)
    Q = Q @ torch.diag(torch.sign(torch.diag(R)))
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def _pose_error_pinned(placed, canonical, cell, sg):
    """Rotation vs Cartesian mates; translation vs the matching image, lattice only."""
    P = canonical.to(torch.float64)
    Q = placed.to(torch.float64)
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, _, Vt = torch.linalg.svd(Qc.T @ Pc)
    d = torch.sign(torch.det(U @ Vt))
    R = U @ torch.diag(torch.tensor([1.0, 1.0, d], dtype=torch.float64)) @ Vt
    B = cell.fractional_matrix.to(torch.float64)
    Binv = torch.linalg.inv(B)
    S = sg.matrices.to(torch.float64)
    T = sg.translations.to(torch.float64)
    R_cart = B @ S @ Binv
    tr = torch.einsum("kij,ij->k", R_cart, R)
    ang = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0).arccos() * (180.0 / 3.141592653589793)
    k = int(ang.argmin())
    delta = (Binv @ Q.mean(0)) - (S[k] @ (Binv @ P.mean(0)) + T[k])
    delta = delta - delta.round()
    return float(ang[k]), float((B @ delta).norm())


@pytest.mark.integration
@pytest.mark.slow
def test_second_chain_returns_to_its_own_site_with_the_first_fixed():
    model = ModelFT().load_pdb(str(TEST_FILES / "pdb" / "3E98.pdb"))
    data = ReflectionData().load_mtz(str(TEST_FILES / "mtz" / "3E98.mtz"))
    chain_a = model.select(_chain_selection(model, "A"))
    chain_b = model.select(_chain_selection(model, "B"))
    canonical_b = chain_b.xyz().clone()

    search = chain_b.copy()
    search.spacegroup = "P 1"
    search = search.copy().rotate(_random_rotation(11).to(model.dtype_float),
                                  center=canonical_b.mean(0))
    pipe = MolecularReplacementPipeline(
        data, search, d_min=4.0, d_max=15.0, n_shells=20,
        n_rotation_peaks=200, n_rotation_candidates=10, fixed=[chain_a],
    )
    sols = pipe.run()
    rot, trans = _pose_error_pinned(sols[0].model.xyz(), canonical_b, data.cell, data.spacegroup)
    assert rot < 8.0, f"rotation {rot:.2f} deg"
    assert trans < 4.0, f"placed {trans:.1f} A from chain B's site with chain A fixed"
    assert sols[0].clash_fraction <= 0.05
