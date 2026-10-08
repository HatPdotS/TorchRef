"""Crystal symmetry handled by the OpenMM adapter itself.

The unit-cell layout must be the crystal: invariant under a lattice translation of the
model, equal to an explicit P1 expansion of it, and differentiable through the symmetry
operations. The ensemble's quasi-crystal layout with identical members is the same
system. A molecule on a special position is held once per distinct site.
"""

import warnings

import numpy as np
import pytest
import torch

from torchref.experimental.mm import CrystalLayout, OpenMMAdapter
from torchref.experimental.mm.topology import build_asu_topology
from torchref.model.model import Model
from torchref.symmetry import SpaceGroup

pytestmark = pytest.mark.openmm


@pytest.fixture(scope="module")
def protein(pdb_dir):
    """7L84 (P 43 21 2, eight operations) with TorchRef's hydrogens."""
    return (
        Model(verbose=0, device="cpu", hydrogens="add")
        .load_pdb(str(pdb_dir / "7L84.pdb"))
        .strip_altlocs()
    )


@pytest.fixture(scope="module")
def waters(protein):
    """7L84's waters in general positions: small enough for an explicit P1 expansion.

    Waters that coincide with a symmetry copy of themselves are left out; written out
    as P1 atoms they would be two molecules, which presence does not merge.
    """
    table = protein.to_dataframe()
    water = protein._derive(
        table[table.resname.str.strip().eq("HOH")].copy(), hydrogens="keep"
    )
    asu = build_asu_topology(water)
    special = np.flatnonzero(~_cell_adapter(water).present.all(axis=0))
    rows = asu.rows[np.isin(asu.molecule_of, special)]
    return water._derive(
        water.to_dataframe().drop(index=rows).reset_index(drop=True), hydrogens="keep"
    )


def _cell_adapter(model, **kwargs):
    layout = CrystalLayout.unit_cell(model.cell, model.spacegroup, cutoff=10.0)
    return OpenMMAdapter.from_model(
        model,
        layout=layout,
        nonbonded="pme",
        cutoff=10.0,
        hydrogens_added=True,
        platform="Reference",
        **kwargs,
    )


def test_lattice_translation_leaves_the_energy(waters):
    adapter = _cell_adapter(waters)
    xyz = waters.xyz().detach().double()
    a = torch.as_tensor(adapter.layout.box[:, 0])
    shifted = xyz + a
    assert adapter.energy(shifted).item() == pytest.approx(
        adapter.energy(xyz).item(), rel=1e-9
    )


def test_symmetric_layout_equals_explicit_p1_expansion(waters):
    """Every symmetry copy written out as atoms of a P1 model gives the same system."""
    adapter = _cell_adapter(waters)
    table = waters.to_dataframe()
    copies = adapter.layout.positions(waters.xyz().detach().double()).numpy()
    pieces = []
    for c, xyz in enumerate(copies):
        piece = table.copy()
        piece[["x", "y", "z"]] = xyz
        piece["chainid"] = chr(ord("A") + c)
        pieces.append(piece)
    import pandas as pd

    p1 = waters._derive(pd.concat(pieces, ignore_index=True), hydrogens="keep")
    explicit = OpenMMAdapter.from_model(
        p1,
        layout=CrystalLayout.unit_cell(p1.cell, SpaceGroup("P 1"), cutoff=10.0),
        nonbonded="pme",
        cutoff=10.0,
        hydrogens_added=True,
        platform="Reference",
    )
    assert explicit.n_particles == adapter.n_particles
    assert explicit.energy(p1.xyz().detach().double()).item() == pytest.approx(
        adapter.energy(waters.xyz().detach().double()).item(), rel=1e-6
    )


def test_gradient_through_the_expansion(protein):
    """Directional finite difference in the ASU coordinates, float32, real structure."""
    adapter = _cell_adapter(protein, max_force=float("inf"))
    xyz = protein.xyz().detach().clone().requires_grad_()
    gradient = torch.autograd.grad(adapter.energy(xyz), xyz)[0]
    direction = gradient / gradient.norm()
    step = 2e-3
    with torch.no_grad():
        finite = (
            adapter.energy(xyz + step * direction)
            - adapter.energy(xyz - step * direction)
        ) / (2 * step)
    assert finite.item() == pytest.approx(gradient.norm().item(), rel=1e-2)


def test_quasi_crystal_of_identical_members_is_the_unit_cell(protein):
    cell, spacegroup = protein.cell, protein.spacegroup
    n = int(spacegroup.n_ops)
    xyz = protein.xyz().detach().double()
    members = xyz.unsqueeze(0).repeat(n, 1, 1).requires_grad_()
    quasi = OpenMMAdapter.from_model(
        protein,
        layout=CrystalLayout.quasi_crystal(cell, spacegroup, 1),
        xyz=members,
        nonbonded="pme",
        cutoff=10.0,
        hydrogens_added=True,
        platform="Reference",
        max_force=float("inf"),
    )
    unit = _cell_adapter(protein, max_force=float("inf"))
    single = xyz.clone().requires_grad_()
    e_unit = unit.energy(single)
    e_quasi = quasi.energy(members)
    assert e_quasi.item() == pytest.approx(e_unit.item(), rel=1e-9)
    g_unit = torch.autograd.grad(e_unit, single)[0]
    g_quasi = torch.autograd.grad(e_quasi, members)[0].sum(0)
    assert torch.allclose(g_quasi, g_unit, rtol=1e-6, atol=1e-6 * g_unit.abs().max())


def test_special_position_water_is_held_once_per_site(pdb_dir):
    """3GR5's HOH 224 sits on a two-fold: six of its twelve copies are kept."""
    model = (
        Model(verbose=0, device="cpu", hydrogens="add")
        .load_pdb(str(pdb_dir / "3GR5.pdb"))
        .strip_altlocs()
    )
    columns = model.ctx.topology.columns()
    keep = columns["resname"] != "SO4"
    asu = build_asu_topology(model, atoms=keep)
    adapter = _cell_adapter(model, atoms=keep)
    labels = np.array([asu.residue_label(r) for r in asu.residue_of])
    water = asu.molecule_of[np.flatnonzero(labels == "HOH A 224")[0]]
    assert adapter.present[:, water].sum() == 6
    assert (~adapter.present).sum() == 6
    energy = adapter.energy(model.xyz().detach().double()).item()
    assert np.isfinite(energy) and abs(energy) < 1e7


def test_deposited_crystal_has_no_overlaps(protein):
    """7L84's crystal, special-position waters held once, raises no overlap report."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        adapter = _cell_adapter(protein)
    assert (~adapter.present).sum() > 0
