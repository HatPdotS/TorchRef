"""
Unit tests for ``torchref.utils.device_mixin``.

Covers:

* recursive traversal of plain tensor attributes and nested containers,
* automatic cache invalidation via ``reset_forward_cache`` / ``reset_cache``,
* cycle protection through the thread-local visited set,
* Cell in-place ``.to()`` semantics (object identity preserved).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from torchref.symmetry.cell import Cell
from torchref.utils.device_mixin import DeviceMixin


class _PlainTensorModule(DeviceMixin, nn.Module):
    """Module holding a plain tensor attribute that is NOT a buffer."""

    def __init__(self):
        super().__init__()
        self.register_buffer("buf", torch.arange(4, dtype=torch.float32))
        # NOT a buffer -- only the __dict__ walk can move it.
        self.plain = torch.arange(3, dtype=torch.float32)
        self.nested_list = [torch.zeros(2), torch.ones(2)]
        self.nested_dict = {"a": torch.full((2,), 7.0)}


class _CountingCacheModule(DeviceMixin, nn.Module):
    """Module whose ``reset_forward_cache`` we can count."""

    def __init__(self):
        super().__init__()
        self.register_buffer("buf", torch.zeros(3))
        self.reset_calls = 0

    def reset_forward_cache(self):  # noqa: D401
        self.reset_calls += 1


class _SelfReferringTarget(DeviceMixin, nn.Module):
    """Holds an unregistered back-reference to a parent module."""

    def __init__(self):
        super().__init__()
        self.register_buffer("payload", torch.zeros(2))
        self.parent = None  # set after parent constructed (cycle)


class _CyclicParent(DeviceMixin, nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("root", torch.zeros(2))
        self.child = _SelfReferringTarget()  # registered submodule
        self.child.parent = self  # creates the cycle


class _TrackerModule(DeviceMixin, nn.Module):
    """Module with ``device`` / ``dtype_float`` trackers that should auto-refresh."""

    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
        self.dtype_float = torch.float32
        self.register_buffer("buf", torch.zeros(3, dtype=torch.float32))


class _SharedTensorParent(DeviceMixin, nn.Module):
    """Holds a Cell and an unregistered nn.Module that aliases ``cell._data``."""

    def __init__(self):
        super().__init__()
        # Pin to CPU so float64 conversion test below works on MPS-default Macs.
        self.cell = Cell(
            [10.0, 20.0, 30.0, 90.0, 90.0, 90.0],
            dtype=torch.float32,
            device="cpu",
        )
        # Alias the cell's data tensor as a plain attribute on a child
        self.alias_holder = _AliasModule(self.cell._data)


class _AliasModule(DeviceMixin, nn.Module):
    def __init__(self, shared_tensor):
        super().__init__()
        # Plain attribute, NOT a buffer -- shares object identity with cell._data
        self.cell_params = shared_tensor


@pytest.mark.unit
def test_plain_tensor_attribute_is_moved():
    """Tensors stored as plain attributes (not buffers) must move with ``.to``."""
    mod = _PlainTensorModule()
    assert mod.plain.dtype == torch.float32

    mod.to(dtype=torch.float64)
    assert mod.buf.dtype == torch.float64, "buffer should be cast"
    assert mod.plain.dtype == torch.float64, "plain tensor attribute should be cast"
    assert mod.nested_list[0].dtype == torch.float64, "list element should be cast"
    assert mod.nested_dict["a"].dtype == torch.float64, "dict value should be cast"


@pytest.mark.unit
def test_reset_forward_cache_is_called():
    """``.to()`` must invalidate ``reset_forward_cache`` when defined."""
    mod = _CountingCacheModule()
    mod.to(dtype=torch.float64)
    assert mod.reset_calls >= 1, "reset_forward_cache must be called at least once"


@pytest.mark.unit
def test_cycle_does_not_recurse_forever():
    """Back-references between parent and child must not blow the stack."""
    parent = _CyclicParent()
    parent.to(dtype=torch.float64)
    assert parent.root.dtype == torch.float64
    assert parent.child.payload.dtype == torch.float64


@pytest.mark.unit
def test_cell_to_is_in_place():
    """Cell.to() now mutates self and returns self (no fresh instance)."""
    # Pin to CPU so float64 conversion succeeds on MPS-default Macs.
    cell = Cell(
        [50.0, 60.0, 70.0, 90.0, 90.0, 90.0], dtype=torch.float32, device="cpu"
    )
    moved = cell.to(dtype=torch.float64)
    assert moved is cell, "Cell.to should return self after the unified migration"
    assert cell._data.dtype == torch.float64
    # Cache is cleared by reset_cache hook on dtype change.
    assert cell._cache == {}


@pytest.mark.unit
def test_device_tracker_attribute_is_refreshed():
    """``self.device`` should follow ``.to(...)`` so future allocations land right."""
    mod = _TrackerModule()
    assert mod.device == torch.device("cpu")
    assert mod.dtype_float == torch.float32

    mod.to(dtype=torch.float64)
    assert mod.dtype_float == torch.float64, "dtype_float tracker did not refresh"

    # device tracker must be a torch.device instance reflecting the buffer state
    assert isinstance(mod.device, torch.device)
    assert mod.device == mod.buf.device


@pytest.mark.unit
def test_shared_tensor_alias_both_attrs_move():
    """When a tensor object is aliased from two attributes, both must migrate.

    Regression: a previous implementation tracked visited tensors by id and
    skipped the second occurrence, leaving one attribute pointing at the
    old (CPU) tensor.
    """
    parent = _SharedTensorParent()
    assert parent.cell._data.device.type == "cpu"
    assert parent.alias_holder.cell_params.device.type == "cpu"

    parent.to(dtype=torch.float64)
    # Both attributes must reflect the dtype change after the move.
    assert parent.cell._data.dtype == torch.float64, "Cell._data not converted"
    assert parent.alias_holder.cell_params.dtype == torch.float64, (
        "alias cell_params not converted -- shared-id visited check is still broken"
    )


@pytest.mark.unit
def test_nested_module_in_dict_attribute_moves():
    """An ``nn.Module`` reached only through a plain dict attribute must move.

    Regression: a previous implementation added the module's ``id`` to the
    visited set *before* calling its ``_apply``, so the inner call saw
    itself in ``visited`` and short-circuited without moving any buffers.
    This is the path used by ``LossState.targets`` — a plain ``dict``
    holding ``Target`` modules that are not registered as submodules.
    """

    class _BufferModule(DeviceMixin, nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("payload", torch.zeros(3, dtype=torch.float32))

    class _Container(DeviceMixin, nn.Module):
        def __init__(self):
            super().__init__()
            # Dict attribute, not nn.ModuleDict — so the inner module is
            # invisible to nn.Module._apply and must be moved via the
            # mixin's __dict__ walk + _apply_to_obj dispatch.
            self.unregistered = {"child": _BufferModule()}

    c = _Container()
    assert c.unregistered["child"].payload.dtype == torch.float32

    c.to(dtype=torch.float64)
    assert c.unregistered["child"].payload.dtype == torch.float64, (
        "nested unregistered nn.Module did not move -- _apply_to_obj is "
        "marking visited too eagerly and short-circuiting the inner call"
    )


@pytest.mark.unit
def test_tensormasks_traversal_moves_dict_items():
    """``TensorMasks`` stores masks as ``dict`` items, not in ``__dict__``.

    When the mixin reaches a ``TensorMasks`` via traversal (e.g. from
    ``ReflectionData.masks``), the per-key mask tensors must be moved and
    the combined-mask cache invalidated; otherwise ``m()`` returns a stale
    cached tensor while the underlying masks are still on the prior device.
    """
    from torchref.utils.utils import TensorMasks

    m = TensorMasks(device="cpu")
    m["valid"] = torch.tensor([True, True, False, True])
    m["rfree"] = torch.tensor([True, False, True, True])
    _ = m()  # populate combined-mask cache on cpu
    assert m._cache is not None

    # Simulate the mixin's traversal-driven call.
    m._apply(lambda t: t.to(dtype=torch.bool))  # no device change, just trigger path
    # _cache must be invalidated by _apply regardless of whether the move
    # actually changed anything.
    assert m._cache is None
    assert m._updated is True

    # Re-running m() must recompute from the per-key masks.
    new = m()
    assert new is not None
    assert new.dtype == torch.bool


@pytest.mark.unit
def test_cell_cache_repopulates_on_target_dtype():
    """After .to(), cached derived quantities must recompute on the new dtype."""
    cell = Cell(
        [50.0, 60.0, 70.0, 90.0, 90.0, 90.0], dtype=torch.float32, device="cpu"
    )
    _ = cell.volume  # populates _cache
    assert "volume" in cell._cache

    cell.to(dtype=torch.float64)
    assert "volume" not in cell._cache, "cache must be cleared after .to()"
    new_volume = cell.volume
    assert new_volume.dtype == torch.float64


# ---------------------------------------------------------------------------
# Probe-driven tracker inference (objects that own no tensors)
# ---------------------------------------------------------------------------


class _TensorFreeTracker(DeviceMixin, nn.Module):
    """Carries device/dtype trackers but owns no tensor to read them from.

    This is the shape of every geometry target: the trackers say where the
    object *will* allocate, and nothing it currently holds can confirm it. Such
    objects are the only consumers of ``_probe_target``.
    """

    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.device = torch.device("cpu")
        self.dtype_float = dtype


@pytest.mark.unit
def test_device_move_does_not_clobber_dtype_tracker(any_device):
    """A device-only move must leave ``dtype_float`` alone.

    Regression: the probe used a single float32 scratch for the dtype axis, so
    a plain ``.to(device)`` reported ``float32`` and overwrote a float64
    tracker. Both axes need their own contrast pair.
    """
    mod = _TensorFreeTracker(dtype=torch.float64)
    mod.to(any_device)
    assert mod.dtype_float == torch.float64, "device-only move changed dtype_float"
    assert mod.device.type == any_device.type


@pytest.mark.unit
def test_dtype_only_move_does_not_clobber_device_tracker():
    """Mirror case: ``.float()`` must leave the device tracker alone."""
    mod = _TensorFreeTracker(dtype=torch.float64)
    mod.device = torch.device("cpu")
    mod.float()
    assert mod.dtype_float == torch.float32
    assert mod.device == torch.device("cpu"), "dtype-only move changed device"


@pytest.mark.unit
@pytest.mark.parametrize(
    "fn,expected_device,expected_dtype",
    [
        (lambda t: t.to("cpu"), torch.device("cpu"), None),
        (lambda t: t.double(), None, torch.float64),
        (lambda t: t.float(), None, torch.float32),
        (lambda t: t.half(), None, torch.float16),
        (lambda t: t, None, None),
    ],
    ids=["to_cpu", "double", "float", "half", "identity"],
)
def test_probe_target_resolves_axes_independently(fn, expected_device, expected_dtype):
    """``_probe_target`` reports only the axis ``fn`` actually transforms.

    ``.double()`` is the case that pins the CPU-only dtype pair: MPS has no
    float64, so contrasting dtype on an accelerator scratch would make it
    unprobeable on Apple silicon.
    """
    from torchref.utils.device_mixin import _probe_target

    device, dtype = _probe_target(fn)
    assert device == expected_device
    assert dtype == expected_dtype

