"""Registry of cheaply-constructible device-bearing objects for conformance tests.

Each :class:`DeviceCase` builds one object on a requested device. The suite in
``tests/unit/test_device_conformance.py`` then asserts that everything the
object owns actually landed there, and that it survives a device round trip.

``UNCOVERED`` is the other half of the contract: every device-bearing class in
the source tree must appear either here or there, with a reason. That is what
keeps the registry from quietly rotting as the package grows -- see
``test_every_device_bearing_class_is_accounted_for``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

import torch

__all__ = [
    "DeviceCase",
    "CASES",
    "TargetDeviceCase",
    "TARGET_CASES",
    "UNCOVERED",
    "UNCOVERED_PREFIXES",
]

_CELL = [50.0, 60.0, 70.0, 90.0, 90.0, 90.0]
_SG = "P 21 21 21"


@dataclass(frozen=True)
class DeviceCase:
    """One constructible object under device test.

    Parameters
    ----------
    name
        Test id.
    build
        ``device -> object``. Must place everything on ``device``.
    cls_name
        Class this case covers, for the coverage guard.
    tensor_free
        True when the object legitimately owns no tensors (an empty shell).
        Such objects can only be checked via their tracker.
    ignore
        Path fragments to exclude from the walk (borrowed state).
    """

    name: str
    build: Callable[[torch.device], Any]
    cls_name: str
    tensor_free: bool = False
    ignore: tuple = field(default_factory=tuple)


def _symmetry(device):
    """A bare Symmetry from an explicit operation list (no space group involved)."""
    import torch as _torch

    from torchref.config import get_float_dtype
    from torchref.symmetry import Symmetry

    dtype = get_float_dtype()
    matrices = _torch.eye(3, dtype=dtype, device=device).unsqueeze(0).repeat(2, 1, 1)
    matrices[1] = -matrices[1]
    translations = _torch.zeros(2, 3, dtype=dtype, device=device)
    return Symmetry(matrices=matrices, translations=translations)


def _map_symmetry_interpolation(device):
    """The interpolating map operator, on a grid that forbids direct indexing.

    P212121 requires even dimensions, so an odd grid forces the interpolating
    variant rather than the streaming one.
    """
    from torchref.symmetry import SpaceGroup
    from torchref.symmetry.map_symmetry_interpolation import (
        _MapSymmetryInterpolation,
    )

    return _MapSymmetryInterpolation(SpaceGroup(_SG, device=device), (15, 15, 15))


def _edge_block(device):
    """A small bond block, origin-sorted, built straight from index arrays."""
    import numpy as np

    from torchref.topology import EdgeBlock

    return EdgeBlock.from_origins(
        {"intra": np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)},
        2,
        "bond",
        device=device,
    )


def _atom_graph(device):
    """A four-atom chain: enough to exercise the edge blocks and the CSR adjacency."""
    import numpy as np

    from torchref.topology import EdgeBlock
    from torchref.topology.atom_graph import AtomGraph

    def block(rows, arity, edge_type):
        return EdgeBlock.from_origins(
            {"intra": np.asarray(rows, dtype=np.int64).reshape(-1, arity)},
            arity,
            edge_type,
            device=device,
        )

    return AtomGraph(
        name=np.array(["N", "CA", "C", "O"]),
        element=np.array(["N", "C", "C", "O"]),
        altloc=np.array([" ", " ", " ", " "]),
        residue_of=torch.zeros(4, dtype=torch.int64, device=device),
        bonds=block([[0, 1], [1, 2], [2, 3]], 2, "bond"),
        angles=block([[0, 1, 2], [1, 2, 3]], 3, "angle"),
        torsions=block([[0, 1, 2, 3]], 4, "torsion"),
        chirals=block([[1, 0, 2, 3]], 4, "chiral"),
        planes={3: block([[1, 2, 3]], 3, "plane")},
    )


def _topology(device):
    """The atom graph above under a one-residue sequence."""
    import numpy as np

    from torchref.topology.residue_graph import ResidueGraph
    from torchref.topology.topology import Topology

    residues = ResidueGraph(
        chain=np.array(["A"]),
        resseq=np.array([1], dtype=np.int64),
        icode=np.array([""]),
        resname=np.array(["GLY"]),
        template_key=np.array(["GLY"], dtype=object),
        atom_start=np.array([0], dtype=np.int64),
        atom_end=np.array([4], dtype=np.int64),
    )
    return Topology(residues=residues, atoms=_atom_graph(device))


def _cell(device):
    from torchref.symmetry import Cell

    return Cell(_CELL, device=device)


CASES: List[DeviceCase] = [
    DeviceCase("EdgeBlock", _edge_block, "EdgeBlock"),
    DeviceCase("AtomGraph", _atom_graph, "AtomGraph"),
    DeviceCase("Topology", _topology, "Topology"),
    DeviceCase("Cell", _cell, "Cell"),
    DeviceCase(
        "SpaceGroup",
        lambda d: __import__(
            "torchref.symmetry", fromlist=["SpaceGroup"]
        ).SpaceGroup(_SG, device=d),
        "SpaceGroup",
    ),
    # D4: device implied by the cell; the SpaceGroup used to be built from the
    # raw (None) device argument and land on the process default instead.
    DeviceCase(
        "SfFFT_from_cell",
        lambda d: __import__(
            "torchref.model.sf_fft", fromlist=["SfFFT"]
        ).SfFFT(cell=_cell(d), spacegroup=_SG, max_res=2.0),
        "SfFFT",
    ),
    # D4: explicit device disagreeing with the supplied cell.
    DeviceCase(
        "SfFFT_explicit_device",
        lambda d: __import__(
            "torchref.model.sf_fft", fromlist=["SfFFT"]
        ).SfFFT(cell=_cell("cpu"), spacegroup=_SG, max_res=2.0, device=d),
        "SfFFT",
    ),
    DeviceCase(
        "SfDS_from_cell",
        lambda d: __import__(
            "torchref.model.sf_ds", fromlist=["SfDS"]
        ).SfDS(cell=_cell(d), spacegroup=_SG),
        "SfDS",
    ),
    # D1: tensor-free shells, whose tracker is the only thing to check.
    DeviceCase(
        "ScalerBase_empty",
        lambda d: __import__(
            "torchref.scaling", fromlist=["ScalerBase"]
        ).ScalerBase(device=d),
        "ScalerBase",
        tensor_free=True,
    ),
    DeviceCase(
        "LossState",
        lambda d: __import__(
            "torchref.refinement.loss_state", fromlist=["LossState"]
        ).LossState(device=d),
        "LossState",
        tensor_free=True,
    ),
    DeviceCase(
        "SolventModel",
        lambda d: __import__(
            "torchref.scaling.solvent", fromlist=["SolventModel"]
        ).SolventModel(device=d),
        "SolventModel",
    ),
    # D9: empty-init wrappers used to drop the requested device entirely.
    DeviceCase(
        "MixedTensor_empty",
        lambda d: __import__(
            "torchref.model.parameter_wrappers", fromlist=["MixedTensor"]
        ).MixedTensor(device=d),
        "MixedTensor",
    ),
    DeviceCase(
        "MixedTensor_populated",
        lambda d: __import__(
            "torchref.model.parameter_wrappers", fromlist=["MixedTensor"]
        ).MixedTensor(
            torch.arange(12, dtype=torch.float32).reshape(4, 3),
            # A 2D value tensor takes a 1D per-row mask.
            refinable_mask=torch.zeros(4, dtype=torch.bool),
            device=d,
        ),
        "MixedTensor",
    ),
    DeviceCase(
        "PositiveMixedTensor",
        lambda d: __import__(
            "torchref.model.parameter_wrappers", fromlist=["PositiveMixedTensor"]
        ).PositiveMixedTensor(
            torch.arange(1, 5, dtype=torch.float32),
            refinable_mask=torch.zeros(4, dtype=torch.bool),
            device=d,
        ),
        "PositiveMixedTensor",
    ),
    DeviceCase(
        "OccupancyTensor_empty",
        lambda d: __import__(
            "torchref.model.parameter_wrappers", fromlist=["OccupancyTensor"]
        ).OccupancyTensor(device=d),
        "OccupancyTensor",
    ),
    DeviceCase(
        "RigidXYZTensor_empty",
        lambda d: __import__(
            "torchref.model.rigid_xyz", fromlist=["RigidXYZTensor"]
        ).RigidXYZTensor(device=d),
        "RigidXYZTensor",
    ),
    DeviceCase(
        "SpaceGroup",
        lambda d: __import__(
            "torchref.symmetry", fromlist=["SpaceGroup"]
        ).SpaceGroup(_SG, device=d),
        "SpaceGroup",
    ),
    DeviceCase(
        "Symmetry",
        _symmetry,
        "Symmetry",
    ),
    DeviceCase(
        "HydrogenTopology_empty",
        lambda d: __import__(
            "torchref.topology.riding",
            fromlist=["HydrogenTopology"],
        ).HydrogenTopology(device=d),
        "HydrogenTopology",
        # The builders attach every tensor later, so a fresh topology is a bare shell
        # and only its tracker can be checked.
        tensor_free=True,
    ),
    DeviceCase(
        "_MapSymmetryInterpolation",
        _map_symmetry_interpolation,
        "_MapSymmetryInterpolation",
        # ``symmetry`` is the group this operator was built from, not state it owns.
        ignore=("symmetry",),
    ),
    DeviceCase(
        "TensorMasks",
        lambda d: __import__(
            "torchref.utils", fromlist=["TensorMasks"]
        ).TensorMasks(device=d),
        "TensorMasks",
        tensor_free=True,
    ),
    DeviceCase(
        "ReflectionData_empty",
        lambda d: __import__(
            "torchref.io", fromlist=["ReflectionData"]
        ).ReflectionData(device=d),
        "ReflectionData",
    ),
    DeviceCase(
        "ManualWeighting",
        lambda d: __import__(
            "torchref.refinement.weighting", fromlist=["ManualWeighting"]
        ).ManualWeighting({"geometry": 1.0}, device=d),
        "ManualWeighting",
        tensor_free=True,
    ),
]


@dataclass(frozen=True)
class TargetDeviceCase:
    """A refinement target, which needs a real model/data/scaler to construct.

    Separate from :class:`DeviceCase` because ``build`` takes a fixture-supplied
    bundle as well as a device. The bundle is already on ``device``; ``build``
    must not cache or mutate it.

    ``ignore`` defaults to the wrapped objects: a target *borrows* its model and
    data, and the conformance walk would otherwise re-check the entire structure
    through every target that points at it.
    """

    name: str
    build: Callable[[Dict[str, Any], torch.device], Any]
    cls_name: str
    needs: tuple = ("model",)
    ignore: tuple = ("_model", "_data", "_scaler", "_model_dark", "_model_light")


TARGET_CASES: List[TargetDeviceCase] = [
    TargetDeviceCase(
        "NonBondedTarget",
        lambda b, d: __import__(
            "torchref.refinement.targets.geometry.non_bonded",
            fromlist=["NonBondedTarget"],
        ).NonBondedTarget(b["model"]),
        "NonBondedTarget",
    ),
    TargetDeviceCase(
        "ADPSimilarityTarget",
        lambda b, d: __import__(
            "torchref.refinement.targets.adp.similarity",
            fromlist=["ADPSimilarityTarget"],
        ).ADPSimilarityTarget(b["model"]),
        "ADPSimilarityTarget",
    ),
    TargetDeviceCase(
        "ADPLocalityTarget",
        lambda b, d: __import__(
            "torchref.refinement.targets.adp.locality", fromlist=["ADPLocalityTarget"]
        ).ADPLocalityTarget(b["model"]),
        "ADPLocalityTarget",
    ),
    # Owns no tensors at all -- the case that exercises the request-driven
    # tracker path rather than the owned-tensor path.
    TargetDeviceCase(
        "BondTarget",
        lambda b, d: __import__(
            "torchref.refinement.targets.geometry.bonds", fromlist=["BondTarget"]
        ).BondTarget(b["model"]),
        "BondTarget",
    ),
]


# Device-bearing classes deliberately not in CASES. Every entry needs a reason;
# "hard to build" is a reason, "didn't get to it" is not.
UNCOVERED: Dict[str, str] = {
    # --- abstract / mixin bases: never instantiated directly -----------------
    "Target": "abstract base; covered through its concrete subclasses",
    "ModelTarget": "abstract base; needs a loaded model",
    "DataTarget": "abstract base; needs model + data + scaler",
    "XrayTarget": "abstract base; needs model + data + scaler",
    "GeometryTarget": "abstract base; needs a model with restraints",
    "CrystalDataset": "abstract dataclass base; covered via ReflectionData",
    "BaseWeighting": "abstract base; covered via ManualWeighting",
    "Refinement": "abstract base; covered via LBFGSRefinement in integration",
    "PassThroughTensor": "documented non-functional stub (parameter_wrappers.py)",
    "ADPTarget": "abstract base; needs a model with ADPs",
    "CombinedTargets": "composite container; needs its component targets",
    "CombinedModelTargets": "composite container; needs a loaded model",
    "CollectionXrayTarget": "abstract base; needs a dataset collection",
    # --- need loaded structures/data: covered by integration tests -----------
    "Model": "needs a PDB; covered by tests/unit/model/test_model_state_dict_device.py",
    "ModelFT": "needs a PDB; covered by tests/unit/model/test_model_state_dict_device.py",
    "MixedModel": "needs two loaded models",
    "ModelCollection": "needs several loaded models",
    "_SharedMixedModel": "internal view owned by ModelCollection",
    "Scaler": "needs a loaded model + data; covered in integration",
    "RestraintsNew": "needs a model + monomer library",
    "FrenchWilson": "needs loaded intensities",
    "DatasetCollection": "needs several loaded datasets",
    "FcalcDataset": "needs computed structure factors",
    "Map": "needs data + model",
    "DifferenceMap": "needs two datasets + a model",
    "LBFGSRefinement": "full pipeline; covered in integration",
    "ModelContext": "needs a loaded structure to hold a cell and space group; "
    "covered through Model in tests/unit/model/test_model_state_dict_device.py",
    "_MapSymmetryDirect": "stateless view over its Symmetry: recomputes index grids "
    "per operation to keep peak memory O(grid), so it owns no tensors to move",
    "CholeskyMixedTensor": "needs a valid ADP tensor; shares MixedTensor's paths",
    "CollectionScaler": "needs a dataset collection",
    "CollectionDifferenceTarget": "needs a dataset collection",
    "CollectionMLTarget": "needs a dataset collection",
    "CollectionRiceTarget": "needs a dataset collection",
    "ADPSigdTarget": "needs a model with ADPs",
    "AngleTarget": "needs a model with restraints",
    "ChiralTarget": "needs a model with restraints",
    "ReciprocalSymmetryExtractor": "needs an hkl tensor + SpaceGroup; device is "
    "intentionally inherited from hkl, so the generic 'requested device' "
    "assertion does not apply",
    "CoordinateSimilarityTarget": "needs two loaded models",
    "MultiModelADPTarget": "needs a model collection",
    "MultiModelGeometryTarget": "needs a model collection",
    "TotalGeometryTarget": "composite; needs a model with restraints",
    "TotalADPTarget": "composite; needs a model with restraints",
    "NonBondedHTarget": "needs a model with hydrogen topology",
    "PlanarityTarget": "needs a model with restraints",
    "RamachandranTarget": "needs a model with restraints",
    "TorsionTarget": "needs a model with restraints",
    "RigidBondTarget": "needs a model with restraints",
    "ScalerLogScaleTrendTarget": "needs a scaler",
    "ScalerURegularizationTarget": "needs a scaler",
    "NLLXrayTarget": "needs model + data + scaler",
    "LeastSquaresXrayTarget": "needs model + data + scaler",
    "UnitWeightK1XrayTarget": "needs model + data + scaler",
    "SigmaAXrayTarget": "abstract base; needs model + data + scaler",
    "MLXrayTarget": "needs model + data + scaler",
    "MLNoAlphaXrayTarget": "needs model + data + scaler",
    "MLFullXrayTarget": "needs model + data + scaler",
    "NLLBetaXrayTarget": "needs model + data + scaler",
    "RiceXrayTarget": "needs model + data + scaler",
    "DifferenceXrayTarget": "needs two datasets",
    "PhaseInformedDifferenceTarget": "needs two datasets + phases",
    "RiceDifferenceTarget": "needs two datasets",
    "TaylorCorrectedDifferenceTarget": "needs two datasets",
    "RigidTransform": "alignment helper; needs a coordinate set",
    "RigidBodyRefinement": "experimental; needs model + data",
}

# Everything under torchref/experimental is out of scope for the conformance
# sweep: it is explicitly unstable, and several modules need optional
# dependencies (OpenMM, AmberTools, JAX) that are not installed in CI.
UNCOVERED_PREFIXES = ("torchref/experimental/",)
