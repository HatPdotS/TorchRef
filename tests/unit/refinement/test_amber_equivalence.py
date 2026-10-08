"""AMBER energies and gradients pinned against independently built systems.

The references in ``tests/files/openmm`` hold coordinates, energy and gradient from
systems built without the adapter, on the same atoms:

- ``7L84.npz``: OpenMM's PDB-file route (amber14 + TIP3P-FB, reaction field at 5 Å,
  gradient clipped at 10000 kJ/mol/nm), the protein with TorchRef's hydrogens.
- ``1DAW_ANP_ligand.npz``, ``3GR5_SO4_ligand.npz``: the ligand alone, parameterised by
  tleap from the same antechamber/parmchk2 output and loaded by parmed, no cutoff.
- ``3GR5_unit_cell.npz``: the twelve symmetry copies of 3GR5 replicated force by force
  from the single-copy system, PME at 10 Å, without the water on the two-fold axis.

The live test rebuilds the 7L84 system through OpenMM's own PDB reader.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from torchref.experimental.mm import CrystalLayout, OpenMMAdapter
from torchref.experimental.mm.topology import build_asu_topology
from torchref.model.model import Model

REFERENCES = Path(__file__).parents[2] / "files" / "openmm"
TIP3P = ("amber14-all.xml", "amber14/tip3p.xml")


def _load(pdb_dir, code):
    return (
        Model(verbose=0, device="cpu", hydrogens="add")
        .load_pdb(str(pdb_dir / f"{code}.pdb"))
        .strip_altlocs()
    )


def _energy_and_gradient(adapter, xyz):
    xyz = torch.as_tensor(xyz).requires_grad_()
    energy = adapter.energy(xyz)
    return energy.item(), torch.autograd.grad(energy, xyz)[0].numpy()


def _assert_gradient(actual, expected):
    np.testing.assert_allclose(
        actual, expected, rtol=1e-4, atol=1e-4 * np.abs(expected).max()
    )


@pytest.mark.openmm
def test_protein_matches_the_pdb_file_route(pdb_dir):
    reference = np.load(REFERENCES / "7L84.npz")
    model = _load(pdb_dir, "7L84")
    assert model.n_atoms == len(reference["xyz"])
    adapter = OpenMMAdapter.from_model(model)
    energy, gradient = _energy_and_gradient(adapter, reference["xyz"])
    assert energy == pytest.approx(float(reference["energy"]), rel=1e-4)
    _assert_gradient(gradient, reference["gradient"])


@pytest.mark.openmm
def test_protein_system_is_the_one_openmm_reads_from_a_pdb(pdb_dir, tmp_path):
    """OpenMM's own reader and templates give the same energy on the same atoms."""
    import openmm
    import openmm.app as app
    import openmm.unit as unit

    model = _load(pdb_dir, "7L84")
    path = tmp_path / "model.pdb"
    model.write_pdb(str(path))
    parsed = app.PDBFile(str(path))
    atoms = list(parsed.topology.atoms())
    assert len(atoms) == model.n_atoms
    existing = {tuple(sorted((a.index, b.index))) for a, b in parsed.topology.bonds()}
    for i, j in model.restraints.topology.atoms.bonds.indices.cpu().tolist():
        if tuple(sorted((i, j))) not in existing:
            parsed.topology.addBond(atoms[i], atoms[j])
    system = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml").createSystem(
        parsed.topology,
        nonbondedMethod=app.CutoffNonPeriodic,
        nonbondedCutoff=5 * unit.angstrom,
        constraints=None,
        rigidWater=False,
    )
    xyz = model.xyz().detach().double()
    context = openmm.Context(
        system,
        openmm.VerletIntegrator(0.001),
        openmm.Platform.getPlatformByName("Reference"),
    )
    context.setPositions(xyz.numpy() * 0.1)
    expected = context.getState(getEnergy=True).getPotentialEnergy()
    adapter = OpenMMAdapter.from_model(model, platform="Reference")
    assert adapter.energy(xyz).item() == pytest.approx(
        expected.value_in_unit(unit.kilojoules_per_mole), rel=1e-9
    )


def _ligand(pdb_dir, code, resname):
    model = _load(pdb_dir, code)
    table = model.to_dataframe()
    first = table[table.resname.str.strip().eq(resname)].iloc[0]
    rows = table[
        table.resname.str.strip().eq(resname)
        & (table.chainid == first.chainid)
        & (table.resseq == first.resseq)
    ]
    return model._derive(rows.copy(), hydrogens="keep")


@pytest.mark.amber
@pytest.mark.parametrize("code, resname", [("1DAW", "ANP"), ("3GR5", "SO4")])
def test_ligand_matches_tleap(pdb_dir, code, resname):
    reference = np.load(REFERENCES / f"{code}_{resname}_ligand.npz")
    ligand = _ligand(pdb_dir, code, resname)
    assert ligand.n_atoms == len(reference["xyz"])
    adapter = OpenMMAdapter.from_model(
        ligand, forcefield=TIP3P, nonbonded="none", hydrogens_added=True
    )
    energy, forces = adapter.energy_and_forces(torch.as_tensor(reference["xyz"]))
    assert energy == pytest.approx(float(reference["energy"]), rel=1e-4)
    np.testing.assert_allclose(
        forces[0],
        reference["forces"],
        rtol=1e-4,
        atol=1e-4 * np.abs(reference["forces"]).max(),
    )


@pytest.mark.amber
def test_unit_cell_matches_the_replicated_system(pdb_dir):
    reference = np.load(REFERENCES / "3GR5_unit_cell.npz")
    model = _load(pdb_dir, "3GR5")
    asu = build_asu_topology(model)
    labels = np.array([asu.residue_label(r) for r in asu.residue_of])
    keep = ~np.isin(labels, reference["excluded"])
    adapter = OpenMMAdapter.from_model(
        model,
        atoms=keep,
        layout=CrystalLayout.unit_cell(model.cell, model.spacegroup, cutoff=10.0),
        forcefield=TIP3P,
        nonbonded="pme",
        cutoff=10.0,
        max_force=float("inf"),
    )
    energy, gradient = _energy_and_gradient(
        adapter, torch.as_tensor(reference["xyz"]).double()
    )
    assert energy == pytest.approx(float(reference["energy"]), rel=1e-4)
    _assert_gradient(gradient, reference["gradient"])
