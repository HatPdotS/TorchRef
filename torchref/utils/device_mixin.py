"""
DeviceMixin - unified device and dtype movement for TorchRef.

One mixin hijacks ``.to()``/``.cuda()``/``.cpu()`` (and indirectly ``.float()`` and
friends, which funnel through ``nn.Module._apply``) for both ``nn.Module`` subclasses and
plain Python classes, recursively moving: params, buffers and child modules via the
standard machinery; raw tensor attributes on ``self``; non-Module sub-objects exposing
``_apply``; tensors nested in ``list``/``tuple``/``dict`` attributes; and unregistered
``nn.Module`` instances held as plain attributes.

A thread-local ``id()`` visited set makes traversal cycle-safe, and every moved node has
``reset_forward_cache()``/``reset_cache()`` called on it -- so a ``.to()`` is never free,
even when it targets the device the object is already on.
"""

from __future__ import annotations

import inspect
import threading

import torch
from torch import nn

from torchref.config import canonical_device

# The same parser ``nn.Module.to`` uses, so it accepts every overload PyTorch does --
# including ``.to(other_tensor)`` and ``.to(0)``, which the ``_parse_to_args`` fallback
# below cannot express. Private API, so probe once at import: catching per call would
# turn a genuinely invalid user argument into a silent no-op.
try:
    _PARSE_TO = torch._C._nn._parse_to
except AttributeError:  # pragma: no cover - defensive against API drift
    _PARSE_TO = None

# Whether this torch exposes ``nn.Module._apply(fn, recurse=...)``. Detected by signature,
# not by catching ``TypeError`` at the call site: that would also swallow a genuine
# ``TypeError`` raised *inside* ``_apply`` after some tensors moved, and retry the move.
_MODULE_APPLY_TAKES_RECURSE = (
    "recurse" in inspect.signature(nn.Module._apply).parameters
)

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


class _ToRequest:
    """The ``(device, dtype)`` a top-level ``.to()`` asked for.

    Recorded on the traversal thread-local because ``_apply`` receives only an opaque
    ``fn``: it is the sole way a tensor-free node can learn where it was asked to go.
    ``probe`` caches a :func:`_probe_target` result so one traversal probes at most once.
    """

    __slots__ = ("device", "dtype", "probe")

    def __init__(self, device=None, dtype=None):
        self.device = canonical_device(device)
        self.dtype = dtype
        self.probe = None


def _current_visited():
    """Return the active visited set, or ``None`` if not in a traversal."""
    return getattr(_traversal_state, "visited", None)


def _current_request():
    """Return the active :class:`_ToRequest`, or ``None``."""
    return getattr(_traversal_state, "request", None)


def _push_request(request):
    """Install ``request`` for the current traversal, returning the previous."""
    prev = getattr(_traversal_state, "request", None)
    _traversal_state.request = request
    return prev


def _pop_request(prev):
    """Restore the request saved by :func:`_push_request`."""
    _traversal_state.request = prev


def _enter_traversal():
    """Begin a top-level traversal; the token goes to :func:`_exit_traversal`.

    Nested calls reuse the visited set and return ``None``, so only the outermost caller
    tears it down.
    """
    prev = getattr(_traversal_state, "visited", None)
    if prev is None:
        _traversal_state.visited = set()
        # Never inherit a request leaked by a traversal that unwound abnormally: a stale
        # target would silently misdirect this one.
        _traversal_state.request = None
        return "owner"
    return None


def _exit_traversal(token):
    """End a top-level traversal."""
    if token == "owner":
        _traversal_state.visited = None
        _traversal_state.request = None


def _parse_to_args(args, kwargs):
    """Parse loose ``.to()`` arguments into ``(device, dtype)``.

    Fallback for a torch without ``_parse_to``: handles positional dtype/device-like and
    the ``device=``/``dtype=`` keywords only, not ``.to(other_tensor)``.
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


def _accepts_single_arg(func) -> bool:
    """Whether ``func`` can be called with exactly one positional argument."""
    try:
        inspect.signature(func).bind(None)
    except (TypeError, ValueError):
        return False
    return True


def _apply_to_obj(val, fn, visited):
    """Apply ``fn`` to any tensors inside ``val``; returns the value to store back.

    Already-visited ``nn.Module`` children are skipped; unregistered ``nn.Module``
    attributes are traversed.
    """
    if isinstance(val, torch.Tensor):
        # Never short-circuit on visited for tensors: one source tensor may be aliased
        # from several attribute slots, and each slot needs its own ``fn(val)`` to be
        # rebound to the moved tensor. Re-applying ``fn`` is a no-op when it matches.
        return fn(val)

    if isinstance(val, nn.Module):
        if id(val) in visited:
            return val
        # Do NOT add ``id(val)`` to ``visited`` here -- ``val._apply`` does it, and
        # pre-adding makes that inner call short-circuit before it moves anything.
        val._apply(fn)
        return val

    # Non-Module sub-objects that implement the _apply contract.
    apply_method = getattr(val, "_apply", None)
    if callable(apply_method) and not isinstance(val, type):
        if id(val) in visited:
            return val
        # Signature checked up front rather than catching ``TypeError`` from the call,
        # which cannot tell "wrong signature" from a real error raised mid-traversal and
        # would leave the object partially moved.
        if _accepts_single_arg(apply_method):
            apply_method(fn)
            return val

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
    """Call ``reset_forward_cache`` / ``reset_cache`` if present.

    Failures propagate: a cache that fails to clear keeps tensors from the previous
    device, i.e. silently stale numbers.

    Raises
    ------
    RuntimeError
        Chained from the hook's exception. Movement is **not atomic** -- some tensors
        have already moved by then, so the object may be left partially moved.
    """
    for hook_name in ("reset_forward_cache", "reset_cache"):
        hook = getattr(obj, hook_name, None)
        if not callable(hook):
            continue
        try:
            hook()
        except Exception as exc:
            raise RuntimeError(
                f"{type(obj).__name__}.{hook_name}() failed during device/dtype "
                f"movement: {type(exc).__name__}: {exc}. The object may be "
                "partially moved and its caches may still hold tensors on the "
                "previous device."
            ) from exc


def _owned_tensors(obj):
    """Yield the tensors *obj* itself owns, never a child's.

    Non-recursive by design: inheriting a device from a submodule would report a
    half-moved graph as consistent.
    """
    if isinstance(obj, nn.Module):
        # The registration dicts, not ``buffers()``/``parameters()``: several classes here
        # override those accessors with a no-argument signature.
        for buf in obj._buffers.values():
            if buf is not None:
                yield buf
        for param in obj._parameters.values():
            if param is not None:
                yield param
    for val in obj.__dict__.values():
        if isinstance(val, torch.Tensor):
            yield val
        else:
            data = getattr(val, "_data", None)
            if isinstance(data, torch.Tensor):
                yield data
    # Container subclasses (``TensorMasks`` is a ``dict``) keep their tensors in
    # the container's own storage, not in ``__dict__``.
    if isinstance(obj, dict):
        for val in obj.values():
            if isinstance(val, torch.Tensor):
                yield val
    elif isinstance(obj, (list, tuple)):
        for val in obj:
            if isinstance(val, torch.Tensor):
                yield val


def _representative_tensor(obj):
    """Return one tensor reflecting *obj*'s own device, or ``None``."""
    for tensor in _owned_tensors(obj):
        return tensor
    return None


def _observed_state(obj):
    """Return ``(device, floating_dtype)`` observed from *obj*'s own tensors.

    The axes resolve **independently**: the first owned tensor fixes the device, but only
    a floating/complex one may fix the dtype -- otherwise an integer buffer registered
    first (``hkl``, ``aniso_flag``) vetoes the dtype answer.
    """
    device = None
    dtype = None
    for tensor in _owned_tensors(obj):
        if device is None:
            device = tensor.device
        if dtype is None and (tensor.is_floating_point() or tensor.is_complex()):
            dtype = tensor.dtype
        if device is not None and dtype is not None:
            break
    return device, dtype


def _probe_target(fn):
    """Discover what ``fn`` does to a tensor's device and floating dtype.

    Needed only on the paths that bypass :meth:`DeviceMixin.to` and so record no request:
    ``.float()``/``.double()``/``.half()``, and an ``_apply`` driven by a plain
    (non-mixin) ``nn.Module`` parent. Scratch devices are always ones known to be valid,
    never the object's own tracker, which on a tensor-free object may name a backend this
    host does not have.

    Each axis needs its own two-point contrast: a transforming ``fn`` funnels both inputs
    to one value, a preserving one hands each input's value back. Probing an axis with a
    single reference is what makes a device-only move report the scratch's dtype and
    clobber ``dtype_float``.

    Returns
    -------
    tuple
        ``(device, dtype)``; either is ``None`` when ``fn`` leaves that axis untouched or
        could not be probed at all.
    """
    accel = None
    if torch.cuda.is_available():
        accel = torch.device("cuda", torch.cuda.current_device())
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        accel = torch.device("mps", 0)

    def run(device, dtype):
        """Apply ``fn`` to one scratch tensor; ``None`` if that is not possible."""
        try:
            scratch = torch.empty(0, device=device, dtype=dtype)
        except (RuntimeError, TypeError):
            return None  # backend advertised but unusable for this dtype
        # Broad catch is correct here: unlike the cache hooks above, a non-conforming
        # ``fn`` is an expected outcome, not a corruption signal.
        try:
            out = fn(scratch)
        except Exception:
            return None
        return out if isinstance(out, torch.Tensor) else None

    # Each axis gets its own pair, varying only along the axis it measures. Sharing one
    # pair couples them: an accelerator scratch cannot be cast to float64 on MPS, so
    # probing dtype on the device pair makes ``.double()`` unprobeable there.
    base = run(torch.device("cpu"), torch.float32)
    if base is None:
        return None, None

    device = base.device
    if accel is not None:
        # Contrast pair for the device axis: same dtype, different device.
        other = run(accel, torch.float32)
        if other is None:
            device = None
        elif other.device != base.device:
            device = None  # each kept its own device -> device-preserving

    dtype = None
    if base.is_floating_point() or base.is_complex():
        # Contrast pair for the dtype axis: same device, different dtype.
        # float16 rather than float64 so this stays cheap and universally
        # supported; the CPU pin means ``.double()`` remains probeable.
        other = run(torch.device("cpu"), torch.float16)
        if other is not None and other.dtype == base.dtype:
            dtype = base.dtype

    return device, dtype


_DEVICE_TRACKERS = ("device", "_device")
_DTYPE_TRACKERS = ("dtype_float", "_dtype")


def _refresh_device_trackers(obj, fn=None):
    """Refresh ``device`` / ``_device`` / ``dtype_float`` / ``_dtype`` trackers.

    The tracker decides where the object allocates *next*, so leaving it stale on a
    tensor-free object silently misplaces every tensor it later creates. Per axis, in
    order: a tensor the object owns; the ``.to()`` request on the traversal thread-local
    (the only source for an object holding no tensors); a probe of ``fn``
    (:func:`_probe_target`), for the ``.float()`` and plain-parent paths.

    Writes only attributes already in ``obj.__dict__`` already holding a
    device/dtype-shaped value, so an unrelated attribute named ``device`` is never
    clobbered. Written values are canonical, so ``obj.device == some_tensor.device``.
    """
    has_device_tracker = any(a in obj.__dict__ for a in _DEVICE_TRACKERS)
    has_dtype_tracker = any(a in obj.__dict__ for a in _DTYPE_TRACKERS)
    if not (has_device_tracker or has_dtype_tracker):
        return

    device, dtype = _observed_state(obj)

    if device is None or dtype is None:
        request = _current_request()
        if request is not None:
            if device is None:
                device = request.device
            if dtype is None:
                dtype = request.dtype
            if (device is None or dtype is None) and fn is not None:
                if request.probe is None:
                    request.probe = _probe_target(fn)
                probed_device, probed_dtype = request.probe
                device = device if device is not None else probed_device
                dtype = dtype if dtype is not None else probed_dtype
        elif fn is not None:
            probed_device, probed_dtype = _probe_target(fn)
            device = device if device is not None else probed_device
            dtype = dtype if dtype is not None else probed_dtype

    if device is not None:
        device = canonical_device(device)
        for attr_name in _DEVICE_TRACKERS:
            if attr_name not in obj.__dict__:
                continue
            current = obj.__dict__[attr_name]
            if current is None or isinstance(current, (torch.device, str, int)):
                obj.__dict__[attr_name] = device

    if dtype is not None:
        for attr_name in _DTYPE_TRACKERS:
            if attr_name not in obj.__dict__:
                continue
            current = obj.__dict__[attr_name]
            if current is None or isinstance(current, torch.dtype):
                obj.__dict__[attr_name] = dtype


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

    Inherit **before** ``nn.Module`` in the MRO (``class Foo(DeviceMixin, nn.Module)``),
    or use it alone on a plain class or dataclass. Every mover -- ``.to()``, ``.cuda()``,
    ``.cpu()``, ``.float()``, ``.double()``, ``.half()`` -- routes through :meth:`_apply`,
    which runs ``nn.Module._apply`` where applicable, walks ``self.__dict__`` for plain
    tensors, nested containers and non-Module sub-objects, refreshes the device/dtype
    trackers, and calls ``reset_forward_cache()``/``reset_cache()`` if defined.
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
        if _PARSE_TO is not None:
            # Let PyTorch's own parser raise on invalid arguments.
            device, dtype = _PARSE_TO(*args, **kwargs)[:2]
        else:  # pragma: no cover - only on a torch build without the private API
            device, dtype = _parse_to_args(args, kwargs)

        # Build the request BEFORE claiming the traversal: ``_ToRequest`` canonicalises the
        # device and so raises for a backend this host lacks. Constructed after
        # ``_enter_traversal`` and outside the ``try``, that raise leaks the thread-local
        # visited set, and every later ``.to()`` on the thread silently moves nothing.
        request = _ToRequest(device, dtype)

        token = _enter_traversal()
        prev_request = _push_request(request)
        try:
            if isinstance(self, nn.Module):
                return super().to(*args, **kwargs)
            if device is None and dtype is None:
                return self

            # Forward the caller's original arguments rather than the parsed
            # pair, so overloads and options the parser folds away --
            # ``.to(other_tensor)``, ``non_blocking=``, ``memory_format=`` --
            # reach the tensors intact.
            def fn(t):
                return t.to(*args, **kwargs)

            return self._apply(fn)
        finally:
            _pop_request(prev_request)
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

        Overrides ``nn.Module._apply`` and runs the numbered pipeline below.

        Parameters
        ----------
        fn : callable
            Tensor-to-tensor function applied to each discovered tensor.
        recurse : bool, default True
            Forwarded to ``nn.Module._apply`` to control recursion into child modules.
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

            # Snapshot before anything moves, so step 5 can tell a real transformation
            # from a ``.to()`` that targets the current state.
            old_device, old_dtype = _observed_state(self)

            # 1. Standard nn.Module traversal (params, buffers, child modules).
            if isinstance(self, nn.Module):
                if _MODULE_APPLY_TAKES_RECURSE:
                    super()._apply(fn, recurse=recurse)
                else:  # pragma: no cover - older torch without the kwarg
                    super()._apply(fn)

                # Mark registered children visited so back-references pointing at them do
                # not re-traverse in step 2.
                for child in self.children():
                    visited.add(id(child))

            # 2. __dict__ walk for plain tensors and non-Module sub-objects.
            for name, val in list(self.__dict__.items()):
                if name in _NN_MODULE_INTERNALS:
                    continue
                new_val = _apply_to_obj(val, fn, visited)
                if new_val is not val:
                    _safe_setattr(self, name, new_val)

            # 3. Refresh trackers so later allocations target the new device.
            _refresh_device_trackers(self, fn)

            # 4. Invalidate caches.
            _invalidate_caches(self)

            # 5. Notify the object iff something actually changed, so index rebuilds do
            #    not fire on a no-op ``.to(current_device)``.
            new_device, new_dtype = _observed_state(self)
            device_changed = (
                old_device is not None
                and new_device is not None
                and canonical_device(old_device) != canonical_device(new_device)
            )
            dtype_changed = (
                old_dtype is not None
                and new_dtype is not None
                and old_dtype != new_dtype
            )
            if device_changed or dtype_changed:
                self._after_device_apply(
                    old_device,
                    new_device,
                    old_dtype,
                    new_dtype,
                    device_changed=device_changed,
                    dtype_changed=dtype_changed,
                )
            return self
        finally:
            if token is not None:
                _exit_traversal(token)

    def _after_device_apply(
        self,
        old_device,
        new_device,
        old_dtype,
        new_dtype,
        *,
        device_changed,
        dtype_changed,
    ):
        """Hook called once per object after a *real* device/dtype change.

        No-op by default. Override to rebuild state derived from tensor placement --
        precomputed index tensors, device-specific caches -- that the generic walk cannot
        repair. Use this rather than ``reset_cache()``, which fires after *every* optimizer
        step and would put index rebuilds and GPU syncs on the hot path.

        Parameters
        ----------
        old_device, new_device : torch.device or None
            Device before and after; ``None`` when the object owned no tensors on that
            side of the move.
        old_dtype, new_dtype : torch.dtype or None
            Floating/complex dtype before and after, resolved independently of the device.
        device_changed, dtype_changed : bool
            Which axes actually changed. At least one is always True.
        """


# ---------------------------------------------------------------------------
# Backwards-compatibility aliases
# ---------------------------------------------------------------------------

DeviceMovementMixin = DeviceMixin

_NonModuleDeviceMixin = DeviceMixin
