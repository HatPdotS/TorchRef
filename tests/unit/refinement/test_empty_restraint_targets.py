"""Geometry targets on models that lack a restraint kind.

Extracts of 1DAW: its glycines alone have no chiral centres and no intra-residue
torsions, glycine A 34 alone has no plane of more than three atoms, and waters carry no
covalent restraints at all. A target with nothing to restrain returns a zero loss and an
empty statistics dict instead of raising, and that zero, like every other loss, comes
back in the coordinates' dtype.
"""

from pathlib import Path

import pytest
import torch

from torchref.base.targets.torsion import torsion_omega_math
from torchref.model.model import Model
from torchref.refinement.targets import TotalGeometryTarget
from torchref.refinement.targets.adp import (
    ADPLocalityTarget,
    NodeLoadTarget,
    NodeSmoothnessTarget,
    RigidBondTarget,
)
from torchref.refinement.targets.geometry import (
    AngleTarget,
    BondTarget,
    ChiralTarget,
    PlanarityTarget,
    TorsionTarget,
)
from torchref.utils.stats import StatEntry


def _glycines(lines):
    return [line for line in lines if line.startswith("ATOM") and line[17:20] == "GLY"]


def _glycine_a34(lines):
    return [
        line for line in _glycines(lines) if line[21] == "A" and int(line[22:26]) == 34
    ]


def _waters(lines):
    return [
        line for line in lines if line.startswith("HETATM") and line[17:20] == "HOH"
    ][:20]


EXTRACTS = [_glycines, _glycine_a34, _waters]


def _extract(pdb_dir: Path, tmp_path: Path, select, device=None) -> Model:
    """A model of 1DAW's CRYST1 record plus the coordinate records ``select`` keeps."""
    lines = (pdb_dir / "1DAW.pdb").read_text().splitlines()
    records = [line for line in lines if line.startswith("CRYST1")] + select(lines)
    path = tmp_path / f"{select.__name__.strip('_')}.pdb"
    path.write_text("\n".join(records + ["END"]) + "\n")
    model = Model(verbose=0, device=device)
    model.load_pdb(str(path))
    return model


@pytest.mark.unit
def test_torsion_target_without_intra_torsions(pdb_dir, tmp_path):
    """Glycines have phi/psi/omega torsions only, so just the omega term remains."""
    model = _extract(pdb_dir, tmp_path, _glycines)
    target = TorsionTarget(model)
    omega = model.restraints.restraints["torsion"]["omega"]
    expected = torsion_omega_math(
        model.xyz(),
        omega["indices"],
        omega["sigmas"],
        omega["is_proline"],
        target.w_cis_proline,
        target.w_cis_general,
    )

    assert target().item() == pytest.approx(expected.item())
    stats = target.stats()
    assert "n_omega" in stats and "loss" in stats
    assert "rms_delta" not in stats


@pytest.mark.cuda
def test_torsion_target_without_intra_torsions_on_cuda(pdb_dir, tmp_path, cuda_device):
    """The Triton dispatch branch treats the missing group as the eager one does."""
    from torchref.utils import use_portable

    model = _extract(pdb_dir, tmp_path, _glycines, device=cuda_device)
    target = TorsionTarget(model)
    with use_portable():
        eager = target().item()

    assert target().item() == pytest.approx(eager, rel=1e-5)


@pytest.mark.unit
def test_covalent_targets_on_waters(pdb_dir, tmp_path):
    model = _extract(pdb_dir, tmp_path, _waters)

    for cls in (BondTarget, AngleTarget, TorsionTarget):
        assert cls(model)().item() == 0.0, cls.__name__
    assert BondTarget(model).stats() == {}
    assert AngleTarget(model).stats() == {}
    assert set(TorsionTarget(model).stats()) == {"loss"}


@pytest.mark.unit
def test_chiral_target_without_chiral_centres(pdb_dir, tmp_path):
    target = ChiralTarget(_extract(pdb_dir, tmp_path, _glycines))

    assert target().item() == 0.0
    assert target.stats() == {}
    violations = target.get_violations()
    assert violations["indices"].shape == (0, 4)
    assert all(len(value) == 0 for value in violations.values())


@pytest.mark.unit
def test_planarity_stats_without_planes_over_three_atoms(pdb_dir, tmp_path):
    target = PlanarityTarget(_extract(pdb_dir, tmp_path, _glycine_a34))

    assert target().item() == 0.0
    assert target.stats() == {}


@pytest.mark.unit
@pytest.mark.parametrize("select", EXTRACTS, ids=lambda f: f.__name__.strip("_"))
def test_total_geometry_stats_are_stat_entries(pdb_dir, tmp_path, select):
    stats = TotalGeometryTarget(_extract(pdb_dir, tmp_path, select)).stats()

    for component, entries in stats.items():
        assert all(isinstance(v, StatEntry) for v in entries.values()), component


@pytest.mark.unit
@pytest.mark.parametrize("select", EXTRACTS, ids=lambda f: f.__name__.strip("_"))
def test_losses_take_the_coordinate_dtype(double_cpu, pdb_dir, tmp_path, select):
    model = _extract(pdb_dir, tmp_path, select)
    dtype = model.xyz().dtype
    assert dtype == torch.float64

    targets = dict(TotalGeometryTarget(model).items())
    targets.update(
        node_load=NodeLoadTarget(model),
        node_smoothness=NodeSmoothnessTarget(model),
        rigid_bond=RigidBondTarget(model),
        locality=ADPLocalityTarget(model),
    )
    for name, target in targets.items():
        loss = target()
        assert loss.dim() == 0 and loss.dtype == dtype, name
    assert targets["locality"]._neighbor_distances.dtype == dtype
    violations = targets["chiral"].get_violations()
    for key in ("volumes", "ideal_volumes", "deviations"):
        assert violations[key].dtype == dtype, key
