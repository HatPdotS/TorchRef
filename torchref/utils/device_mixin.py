"""
DeviceMixin - unified device and dtype movement for TorchRef.

Provides a single mixin that hijacks ``.to()``, ``.cuda()``, ``.cpu()`` (and
indirectly ``.float()``, ``.double()``, ``.half()`` since they all funnel
through ``nn.Module._apply``) for both ``nn.Module`` subclasses and plain
Python classes. The mixin recursively traverses the object graph, moving:

* parameters, buffers and child ``nn.Module`` instances (via the standard
  ``nn.Module._apply`` machinery when applicable),
* raw ``torch.Tensor`` attributes stored directly on ``self``,
* non-Module sub-objects that expose ``_apply``,
* tensors nested inside ``list`` / ``tuple`` / ``dict`` attributes,
* unregistered ``nn.Module`` instances held as plain attributes.

A thread-local visited set keyed by ``id()`` makes traversal cycle-safe
(e.g. ``Target.refinement -> Refinement.targets -> Target``). After moving,
any node exposing ``reset_forward_cache()`` or ``reset_cache()`` is
invalidated.

Usage::

    class MyModule(DeviceMixin, nn.Module):
        ...  # nothing else required for device movement

    @dataclass
    class MyDataclass(DeviceMixin):
        _data: torch.Tensor

The legacy name ``DeviceMovementMixin`` is kept as an alias for
``DeviceMixin``. The name ``_NonModuleDeviceMixin`` (briefly a distinct
implementation) is likewise an alias for ``DeviceMixin`` — the unified
mixin now handles both the Module and non-Module cases.
"""

from __future__ import annotations

import threading

import torch
from torch import nn

# ---------------------------------------------------------------------------
# Thread-local traversal state (cycle detection across one top-level .to())
# ---------------------------------------------------------------------------

_traversal_state = threading.local()

# Attribute names on nn.Module that hold params / buffers / children and are
# already moved by ``nn.Module._apply``. Skipping them in the __dict__ walk
# avoids double-work and pointless recursion into the bookkeeping dicts.
_NN_MODULE_INTERNALS = frozenset(
    {
        "_parameters",
        "_buffers",
        "_modules",
        "_non_persistent_buffers_set",
        "_backward_hooks",
        "_backward_pre_hooks",
        "_forward_hooks",
        "_forward_hooks_with_kwargs",
        "_forward_pre_hooks",
        "_forward_pre_hooks_with_kwargs",
        "_state_dict_hooks",
        "_state_dict_pre_hooks",
        "_load_state_dict_pre_hooks",
        "_load_state_dict_post_hooks",
        "_is_full_backward_hook",
        "training",
    }
)


def _current_visited():
    """Return the active visited set, or ``None`` if not in a traversal."""
    return getattr(_traversal_state, "visited", None)


def _enter_traversal():
    """Begin a top-level traversal. Returns a token for :func:`_exit_traversal`.

    Nested calls reuse the existing visited set and return ``None`` so the
    outermost caller is the only one that tears it down.
    """
    prev = getattr(_traversal_state, "visited", None)
    if prev is None:
        _traversal_state.visited = set()
        return "owner"
    return None


def _exit_traversal(token):
    """End a top-level traversal."""
    if token == "owner":
        _traversal_state.visited = None


def _parse_to_args(args, kwargs):
    """Parse loose ``.to()`` arguments into ``(device, dtype)``.

    Supports the common forms used in TorchRef and in ``torch.Tensor.to``:
    positional ``torch.dtype`` / device-like / ``None``, and the explicit
    keyword arguments ``device=`` and ``dtype=``.
    """
    device = kwargs.get("device", None)
    dtype = kwargs.get("dtype", None)
    for a in args:
        if isinstance(a, torch.dtype):
            dtype = a
        elif isinstance(a, torch.device):
            device = a
        elif isinstance(a, str):
            device = a
        elif isinstance(a, int):
            device = a
    return device, dtype


def _apply_to_obj(val, fn, visited):
    """Apply ``fn`` to any tensors inside ``val``, recursing as needed.

    Returns the (possibly new) value to store back. ``nn.Module`` children
    that have already been visited (registered submodules of the current
    parent) are skipped; unregistered ``nn.Module`` attributes are traversed.
    """
    if isinstance(val, torch.Tensor):
        # Do NOT short-circuit on visited for tensors: the same source
        # tensor may be aliased from multiple attribute slots (e.g. a
        # ``Cell._data`` referenced both as ``cell._data`` and as
        # ``submodule.cell_params``). Each slot needs an independent
        # ``fn(val)`` invocation so its attribute is updated to the
        # moved tensor. Re-applying ``fn`` to an already-converted
        # tensor is a cheap no-op when the target matches.
        return fn(val)

    if isinstance(val, nn.Module):
        if id(val) in visited:
            return val
        # Do NOT add ``id(val)`` to ``visited`` here — ``val._apply`` does
        # that itself. Adding it first would make the inner call short-circuit
        # before it actually moves the module's tensors.
        val._apply(fn)
        return val

    # Non-Module sub-objects that implement the _apply contract.
    apply_method = getattr(val, "_apply", None)
    if callable(apply_method) and not isinstance(val, type):
        if id(val) in visited:
            return val
        try:
            apply_method(fn)
            return val
        except TypeError:
            # Object has an _apply with a different signature; skip silently.
            pass

    if isinstance(val, list):
        new_list = [_apply_to_obj(v, fn, visited) for v in val]
        if all(a is b for a, b in zip(new_list, val)):
            return val
        return new_list

    if isinstance(val, tuple):
        new_tuple = tuple(_apply_to_obj(v, fn, visited) for v in val)
        if all(a is b for a, b in zip(new_tuple, val)):
            return val
        return new_tuple

    if isinstance(val, dict):
        replaced = False
        new_dict = {}
        for k, v in val.items():
            nv = _apply_to_obj(v, fn, visited)
            if nv is not v:
                replaced = True
            new_dict[k] = nv
        return new_dict if replaced else val

    return val


def _invalidate_caches(obj):
    """Call ``reset_forward_cache`` / ``reset_cache`` if present."""
    reset_fwd = getattr(obj, "reset_forward_cache", None)
    if callable(reset_fwd):
        try:
            reset_fwd()
        except Exception:
            pass
    reset_full = getattr(obj, "reset_cache", None)
    if callable(reset_full):
        try:
            reset_full()
        except Exception:
            pass


def _representative_tensor(obj):
    """Return one tensor that reflects the current device/dtype of *obj*.

    Prefers buffers (their dtype tracks ``dtype_float`` for crystallographic
    code) and falls back to parameters, then to a plain ``torch.Tensor``
    attribute found in ``__dict__``.
    """
    if isinstance(obj, nn.Module):
        for buf in obj.buffers():
            return buf
        for param in obj.parameters():
            return param
    for val in obj.__dict__.values():
        if isinstance(val, torch.Tensor):
            return val
        if hasattr(val, "_data") and isinstance(getattr(val, "_data"), torch.Tensor):
            return val._data
    return None


def _refresh_device_trackers(obj):
    """Refresh ``device`` / ``_device`` / ``dtype_float`` / ``_dtype`` trackers.

    Only updates attributes whose current value is already a ``torch.device``
    / ``torch.dtype`` (or ``None`` / ``str`` / ``int``) — never overwrites
    unrelated attributes that happen to share the same name. The tracker
    controls where new tensors are subsequently allocated, so it must follow
    ``.to()`` moves.
    """
    rep = _representative_tensor(obj)
    if rep is None:
        return

    for attr_name in ("device", "_device"):
        if attr_name not in obj.__dict__:
            continue
        current = obj.__dict__[attr_name]
        if current is None or isinstance(current, (torch.device, str, int)):
            obj.__dict__[attr_name] = rep.device

    if rep.is_floating_point() or rep.is_complex():
        for attr_name in ("dtype_float", "_dtype"):
            if attr_name not in obj.__dict__:
                continue
            current = obj.__dict__[attr_name]
            if current is None or isinstance(current, torch.dtype):
                obj.__dict__[attr_name] = rep.dtype


def _safe_setattr(obj, name, value):
    """Update an attribute without triggering nn.Module.__setattr__ side effects."""
    try:
        obj.__dict__[name] = value
    except (AttributeError, TypeError):
        object.__setattr__(obj, name, value)


# ---------------------------------------------------------------------------
# Unified mixin
# ---------------------------------------------------------------------------


class DeviceMixin:
    """Unified device/dtype movement.

    Inherit alongside ``nn.Module`` (place before ``nn.Module`` in the MRO)::

        class Foo(DeviceMixin, nn.Module):
            ...

    Or use on a plain Python class / dataclass::

        @dataclass
        class Bar(DeviceMixin):
            data: torch.Tensor

    All of ``.to()``, ``.cuda()``, ``.cpu()``, ``.float()``, ``.double()``,
    ``.half()`` route through :meth:`_apply`, which:

    1. invokes ``nn.Module._apply`` when applicable so parameters, buffers
       and child modules are moved by the standard PyTorch path,
    2. walks ``self.__dict__`` to pick up plain tensor attributes, nested
       containers and non-Module sub-objects,
    3. calls ``reset_forward_cache()`` and ``reset_cache()`` if either is
       defined.
    """

    # ---- to / cuda / cpu -------------------------------------------------

    def to(self, *args, **kwargs):  # type: ignore[override]
        """Move ``self`` to a device and/or dtype, returning ``self``.

        Accepts the usual ``nn.Module.to`` argument forms (device, dtype,
        or both). For ``nn.Module`` subclasses this defers to the standard
        ``nn.Module.to``; for plain (non-Module) classes the (device, dtype)
        pair is parsed and applied via :meth:`_apply`. A call that resolves
        to neither a device nor a dtype is a no-op that returns ``self``.
        """
        token = _enter_traversal()
        try:
            if isinstance(self, nn.Module):
                return super().to(*args, **kwargs)
            device, dtype = _parse_to_args(args, kwargs)
            if device is None and dtype is None:
                return self

            def fn(t):
                return t.to(device=device, dtype=dtype)

            return self._apply(fn)
        finally:
            _exit_traversal(token)

    def cuda(self, device=None):  # type: ignore[override]
        """Move ``self`` to a CUDA device, returning ``self``.

        ``device=None`` targets the default ``"cuda"`` device; an integer
        ``device`` is mapped to ``"cuda:N"``. Delegates to :meth:`to`.
        """
        if device is None:
            device = "cuda"
        elif isinstance(device, int):
            device = f"cuda:{device}"
        return self.to(device=device)

    def cpu(self):  # type: ignore[override]
        """Move ``self`` to the CPU, returning ``self`` (delegates to :meth:`to`)."""
        return self.to(device="cpu")

    # ---- core traversal --------------------------------------------------

    def _apply(self, fn, recurse=True):  # type: ignore[override]
        """Cycle-safe traversal engine that applies ``fn`` to every tensor.

        Overrides ``nn.Module._apply`` and is the core of the movement
        pipeline described in the class docstring: (1) standard
        ``nn.Module`` traversal of params/buffers/child modules, (2) a
        ``__dict__`` walk for plain tensors and non-Module sub-objects,
        (3) refresh of ``device``/``_device``/``dtype`` trackers, and
        (4) cache invalidation. A thread-local ``id()`` visited set makes
        it safe against reference cycles.

        Parameters
        ----------
        fn : callable
            Tensor-to-tensor function applied to each discovered tensor.
        recurse : bool, default True
            Forwarded to ``nn.Module._apply`` to control recursion into
            child modules.
        """
        visited = _current_visited()
        token = None
        if visited is None:
            token = _enter_traversal()
            visited = _current_visited()

        try:
            if id(self) in visited:
                return self
            visited.add(id(self))

            # 1. Standard nn.Module traversal (params, buffers, child modules).
            if isinstance(self, nn.Module):
                try:
                    super()._apply(fn, recurse=recurse)
                except TypeError:
                    super()._apply(fn)

                # Mark registered children as visited so that other plain
                # attributes / back-references pointing at them do not trigger
                # redundant traversal in step 2.
                for child in self.children():
                    visited.add(id(child))

            # 2. __dict__ walk for plain tensors and non-Module sub-objects.
            for name, val in list(self.__dict__.items()):
                if name in _NN_MODULE_INTERNALS:
                    continue
                new_val = _apply_to_obj(val, fn, visited)
                if new_val is not val:
                    _safe_setattr(self, name, new_val)

            # 3. Refresh ``device`` / ``_device`` / ``dtype`` trackers so
            #    subsequent tensor allocations target the new device.
            _refresh_device_trackers(self)

            # 4. Invalidate caches.
            _invalidate_caches(self)
            return self
        finally:
            if token is not None:
                _exit_traversal(token)


# ---------------------------------------------------------------------------
# Backwards-compatibility aliases
# ---------------------------------------------------------------------------

# Older code imports ``DeviceMovementMixin``; keep it pointing at the
# active implementation so existing classes pick up the new behaviour.
DeviceMovementMixin = DeviceMixin

# ``_NonModuleDeviceMixin`` was briefly distinct; the unified mixin now
# handles both Module and non-Module cases.
_NonModuleDeviceMixin = DeviceMixin
