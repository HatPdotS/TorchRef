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


def _cell(device):
    from torchref.symmetry import Cell

    return Cell(_CELL, device=device)


CASES: List[DeviceCase] = [
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
        "ReciprocalSymmetryGrid",
        lambda d: __import__(
            "torchref.symmetry", fromlist=["ReciprocalSymmetryGrid"]
        ).ReciprocalSymmetryGrid(_SG, grid_shape=(16, 16, 16), device=d),
        "ReciprocalSymmetryGrid",
    ),
    DeviceCase(
        "MapSymmetryDirect",
        lambda d: __import__(
            "torchref.symmetry", fromlist=["MapSymmetryDirect"]
        ).MapSymmetryDirect(
            _SG, map_shape=(16, 16, 16), cell_params=_CELL, device=d
        ),
        "MapSymmetryDirect",
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
    "HydrogenTopology": "needs a built restraint topology",
    "FrenchWilson": "needs loaded intensities",
    "DatasetCollection": "needs several loaded datasets",
    "FcalcDataset": "needs computed structure factors",
    "Map": "needs data + model",
    "DifferenceMap": "needs two datasets + a model",
    "LBFGSRefinement": "full pipeline; covered in integration",
    "MapSymmetry": "interpolation variant; needs a real map grid",
    "CholeskyMixedTensor": "needs a valid ADP tensor; shares MixedTensor's paths",
    "CollectionScaler": "needs a dataset collection",
    "CollectionDifferenceTarget": "needs a dataset collection",
    "CollectionMLTarget": "needs a dataset collection",
    "CollectionRiceTarget": "needs a dataset collection",
    "ADPEntropyTarget": "needs a model with ADPs",
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
