"""Device conformance: constructors place tensors where they were told, and
``.to()`` moves *everything* -- tensors and trackers alike.

The package resolves a single default device at import and expects every object
to live on it, with ``self.device`` as the authority for where new tensors are
allocated (200+ call sites pass ``device=self.device``). Two failure modes
follow, and both used to be untested:

* a constructor that builds a sub-object on the *global* default rather than
  the device it was handed, producing a split object that only errors much
  later, deep inside some unrelated op;
* a ``.to()`` that moves the tensors but leaves ``self.device`` stale, so the
  object keeps allocating on the device it just left.

Before this file, the only test that exercised a real device transition was
CUDA-gated -- meaning zero coverage on CPU-only CI and on Apple silicon.
"""

from pathlib import Path

import pytest
import torch

from torchref.config import canonical_device

# Absolute ``tests.`` imports, not bare ``helpers.``: the repo carries two
# pytest configs -- ``tests/pytest.ini`` (rootdir ``tests/``, ``pythonpath = ..``)
# and ``[tool.pytest.ini_options]`` in ``pyproject.toml`` (rootdir the repo
# root). Only the repo root is on ``sys.path`` under both, so a bare
# ``helpers.`` import works from ``tests/`` and fails from the repo root.
from tests.helpers.device_asserts import assert_device_consistent, collect_device_map
from tests.helpers.device_cases import (
    CASES,
    TARGET_CASES,
    UNCOVERED,
    UNCOVERED_PREFIXES,
)
from tests.helpers.device_inventory import device_mixin_classes

_IDS = [c.name for c in CASES]


@pytest.mark.unit
@pytest.mark.parametrize("case", CASES, ids=_IDS)
def test_constructed_on_requested_device(case, any_device):
    """Everything a constructor builds lands on the device it was given."""
    obj = case.build(any_device)
    assert_device_consistent(obj, any_device, name=case.name, ignore=case.ignore)


@pytest.mark.unit
@pytest.mark.parametrize("case", CASES, ids=_IDS)
def test_device_round_trip(case, any_device):
    """cpu -> device -> cpu, with tensors and trackers following every leg."""
    obj = case.build(torch.device("cpu"))
    assert_device_consistent(obj, "cpu", name=case.name, ignore=case.ignore)

    obj.to(any_device)
    assert_device_consistent(obj, any_device, name=case.name, ignore=case.ignore)

    obj.to("cpu")
    assert_device_consistent(obj, "cpu", name=case.name, ignore=case.ignore)


@pytest.mark.unit
@pytest.mark.parametrize(
    "case", [c for c in CASES if not c.tensor_free], ids=[c.name for c in CASES if not c.tensor_free]
)
def test_tracker_agrees_with_owned_tensor(case, any_device):
    """``obj.device`` must equal a real tensor's device, index and all.

    ``torch.device('mps') != torch.device('mps:0')``, so a tracker carrying the
    un-indexed spelling compares unequal to every tensor the object owns even
    though nothing is actually misplaced. Callers do write
    ``if t.device != self.device``, so the two spellings have to agree.
    """
    obj = case.build(any_device)
    tensors, _, _, _ = collect_device_map(obj, case.name)
    if not tensors:
        pytest.skip(f"{case.name} owns no tensors on this path")
    tracker = getattr(obj, "device", None)
    assert isinstance(tracker, torch.device), f"{case.name}.device is {tracker!r}"
    for path, dev in tensors.items():
        assert tracker == dev, (
            f"{case.name}.device == {tracker!r} but {path} is on {dev!r}; "
            "these must compare equal, not merely name the same backend"
        )


@pytest.mark.unit
def test_every_device_bearing_class_is_accounted_for():
    """A new device-bearing class must be given a case or an explicit excuse.

    Uses a static AST inventory rather than ``DeviceMixin.__subclasses__()``:
    the runtime hook only sees classes whose module happens to be imported, so
    it would silently under-report exactly when coverage regressed.
    """
    package_root = Path(__file__).resolve().parents[2] / "torchref"
    found = device_mixin_classes(package_root)

    covered = (
        {c.cls_name for c in CASES}
        | {c.cls_name for c in TARGET_CASES}
        | set(UNCOVERED)
    )
    missing = {
        name: where
        for name, where in found.items()
        if name not in covered
        and not any(where.startswith(p) for p in UNCOVERED_PREFIXES)
    }

    assert not missing, (
        "device-bearing classes with no conformance case and no entry in "
        "helpers.device_cases.UNCOVERED:\n  "
        + "\n  ".join(f"{n} ({w})" for n, w in sorted(missing.items()))
    )


@pytest.mark.unit
def test_uncovered_entries_still_exist():
    """Stop ``UNCOVERED`` accumulating excuses for classes that are long gone."""
    package_root = Path(__file__).resolve().parents[2] / "torchref"
    found = set(device_mixin_classes(package_root))
    stale = sorted(set(UNCOVERED) - found)
    assert not stale, f"UNCOVERED lists classes that no longer exist: {stale}"


# ---------------------------------------------------------------------------
# Refinement targets
#
# Targets were the largest hole in the device story: none of them carried a
# ``device`` tracker at all, so ``DeviceMixin``'s machinery was a no-op for
# every one, and their scalar tunables were allocated on CPU in float32
# regardless of the model's device or the configured dtype.
# ---------------------------------------------------------------------------

_TARGET_IDS = [c.name for c in TARGET_CASES]


@pytest.mark.unit
@pytest.mark.parametrize("case", TARGET_CASES, ids=_TARGET_IDS)
def test_target_follows_its_model_device(case, device_model_bundle, any_device):
    """A target's own buffers land beside the model it acts on."""
    target = case.build(device_model_bundle, any_device)
    assert_device_consistent(target, any_device, name=case.name, ignore=case.ignore)


@pytest.mark.unit
@pytest.mark.parametrize("case", TARGET_CASES, ids=_TARGET_IDS)
def test_target_tracker_is_set(case, device_model_bundle, any_device):
    """``target.device`` exists and is truthful.

    Tensor-free targets (``BondTarget``) can only be checked this way -- and
    they are exactly the ones that used to have no tracker at all.
    """
    target = case.build(device_model_bundle, any_device)
    assert isinstance(target.device, torch.device)
    assert canonical_device(target.device) == canonical_device(any_device)


@pytest.mark.unit
@pytest.mark.parametrize("case", TARGET_CASES, ids=_TARGET_IDS)
def test_target_forward_does_not_reallocate_buffers(case, device_model_bundle, any_device):
    """``forward()`` must not move or re-register buffers.

    Three targets used to repair their CPU-resident buffers lazily on the first
    forward. That is what this pins shut: allocation inside forward breaks
    CUDA-graph capture, and it silently costs a transfer on the hot path. The
    old code fails this on the first call and passes on the second.
    """
    target = case.build(device_model_bundle, any_device)

    def snapshot():
        return {
            n: (b.data_ptr(), b.device, b.dtype)
            for n, b in target.named_buffers(recurse=False)
            if b is not None
        }

    try:
        target()
    except Exception as exc:  # pragma: no cover - target needs state we lack
        pytest.skip(f"{case.name} cannot run forward() here: {exc}")
    before = snapshot()
    target()
    assert snapshot() == before, f"{case.name} mutated its buffers during forward()"


@pytest.mark.unit
@pytest.mark.parametrize("case", TARGET_CASES, ids=_TARGET_IDS)
def test_empty_target_state_dict_round_trip(case):
    """An empty target must round-trip its own state dict strictly.

    The empty-init path exists precisely so ``load_state_dict`` has something
    to load into. ``CoordinateSimilarityTarget`` used to fail this: its index
    buffers were registered only inside ``_build_atom_map``, which never ran
    without models.
    """
    cls = type(case.build({"model": None}, torch.device("cpu")))
    shell = cls()
    other = cls()
    other.load_state_dict(shell.state_dict(), strict=True)


@pytest.mark.unit
def test_target_scalar_buffers_follow_config_dtype(pdb_dir, monkeypatch):
    """Scalar tunables honour ``dtypes.float``, not a hardcoded float32.

    CPU-only: MPS has no float64. Before the migration every one of these was
    ``torch.tensor(float(x))`` -- silently float32 under a float64 config.
    """
    pdb = pdb_dir / "1DAW.pdb"
    if not pdb.exists():
        pytest.skip("1DAW.pdb fixture not present")

    from torchref.config import dtypes
    from torchref.model import ModelFT
    from torchref.refinement.targets.geometry.non_bonded import NonBondedTarget

    monkeypatch.setattr(dtypes, "_float", torch.float64)
    model = ModelFT(device="cpu", dtype_float=torch.float64, verbose=0).load_pdb(str(pdb))
    target = NonBondedTarget(model)

    assert target.dtype_float == torch.float64
    for name in ("_sigma_vdw", "_r_exp", "_c_rep"):
        assert getattr(target, name).dtype == torch.float64, name


@pytest.mark.unit
def test_register_target_warns_on_device_mismatch(pdb_dir):
    """A mismatched target is reported, not silently relocated.

    Moving it would drag the model and data it borrows onto the state's
    device -- registering a loss term must not have that side effect.
    """
    pdb = pdb_dir / "1DAW.pdb"
    if not pdb.exists():
        pytest.skip("1DAW.pdb fixture not present")

    from torchref.model import ModelFT
    from torchref.refinement.loss_state import LossState
    from torchref.refinement.targets.geometry.non_bonded import NonBondedTarget

    model = ModelFT(device="cpu", verbose=0).load_pdb(str(pdb))
    target = NonBondedTarget(model)
    assert target.device.type == "cpu"

    state = LossState(device=torch.device("meta"))
    with pytest.warns(UserWarning, match="is on cpu but the state is on"):
        state.register_target("nb", target, probe=False)

    # ...and the target must not have been moved.
    assert target.device.type == "cpu"
    assert target._sigma_vdw.device.type == "cpu"
