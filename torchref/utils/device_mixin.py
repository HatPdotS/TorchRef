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

import inspect
import threading

import torch
from torch import nn

from torchref.config import canonical_device

# ``torch._C._nn._parse_to`` is the same parser ``nn.Module.to`` uses, so it
# accepts every overload PyTorch does -- including ``.to(other_tensor)`` and
# ``.to(0)``, which the hand-rolled ``_parse_to_args`` fallback below cannot
# express. It is a private API, so probe once at import rather than catching
# per call: catching ``TypeError``/``RuntimeError`` around each invocation
# would turn a genuinely invalid user argument into a silent no-op.
try:
    _PARSE_TO = torch._C._nn._parse_to
except AttributeError:  # pragma: no cover - defensive against API drift
    _PARSE_TO = None

# Whether this torch exposes ``nn.Module._apply(fn, recurse=...)``. Detected
# once by signature rather than by catching ``TypeError`` at the call site: a
# blanket catch there would also swallow a genuine ``TypeError`` raised *inside*
# ``_apply`` after some tensors had already moved, and retry the move.
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

    :meth:`DeviceMixin._apply` receives only an opaque tensor-to-tensor ``fn``,
    so an object that owns no tensors cannot work out where it was just asked
    to go. Recording the parsed request on the traversal thread-local lets
    :func:`_refresh_device_trackers` answer that question at *every* node --
    including tensor-free children, which are reached through
    ``child._apply(fn)`` and therefore never pass through their own ``to()``.

    ``probe`` caches a :func:`_probe_target` result for the traversal so a
    graph with many tensor-free nodes probes at most once.
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
    """Begin a top-level traversal. Returns a token for :func:`_exit_traversal`.

    Nested calls reuse the existing visited set and return ``None`` so the
    outermost caller is the only one that tears it down.
    """
    prev = getattr(_traversal_state, "visited", None)
    if prev is None:
        _traversal_state.visited = set()
        # Never inherit a request leaked by an earlier traversal that unwound
        # abnormally: a stale target would silently misdirect this one.
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


def _accepts_single_arg(func) -> bool:
    """Whether ``func`` can be called with exactly one positional argument."""
    try:
        inspect.signature(func).bind(None)
    except (TypeError, ValueError):
        return False
    return True


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
        # Check the signature up front instead of catching ``TypeError`` from
        # the call: a catch there cannot distinguish "wrong signature" from a
        # real ``TypeError`` raised mid-traversal, and would silently leave the
        # object partially moved.
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

    Failures propagate rather than being swallowed. A cache that fails to clear
    keeps tensors from the *previous* device, which produces either a
    cross-device error much later or -- worse -- silently stale numbers; that
    is precisely the corruption this mixin exists to prevent, so it must not be
    hidden behind a bare ``except``.

    Raises
    ------
    RuntimeError
        Chained from the hook's own exception, naming the object and hook.
        Note that movement is **not atomic**: by the time a cache hook runs,
        some of the object's tensors have already been transformed, so the
        object may be left partially moved.
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
    """Yield the tensors *obj* itself owns (never a child's).

    ``recurse=False`` is deliberate: the trackers describe where this object
    allocates, so inheriting a device from a submodule would report a
    half-moved graph as consistent.
    """
    if isinstance(obj, nn.Module):
        # Read the registration dicts rather than ``buffers()``/``parameters()``:
        # several classes here override those accessors with a no-argument
        # signature (``SolventModel.parameters``, ``MixedTensor.parameters``),
        # and the dicts are in any case the literal "owned, not inherited" set.
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

    The two axes are resolved **independently**: the first owned tensor fixes
    the device, but only a floating/complex tensor may fix ``dtype_float``.
    Deciding both from a single representative tensor lets an integer buffer
    that happens to be registered first (``hkl``, ``aniso_flag``) veto the
    dtype answer.
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

    Needed only on the paths that bypass :meth:`DeviceMixin.to` and therefore
    record no request: ``.float()`` / ``.double()`` / ``.half()``, and an
    ``_apply`` driven by a plain (non-mixin) ``nn.Module`` parent.

    ``fn`` is probed on scratch tensors whose devices are **known to be
    valid**, never on the object's own tracker -- a tensor-free object can be
    carrying a stale tracker naming a backend this host does not have (say
    ``cuda:0`` on a CPU-only machine), and allocating there would fail exactly
    when the answer matters most.

    **Both** axes need two reference points, and the scratches must differ on
    both. A transforming ``fn`` funnels its two inputs to a single value on the
    axis it touches, while a preserving ``fn`` hands each input's own value
    back:

    ==================  ===================  ===================
    ``fn``              devices agree?       dtypes agree?
    ==================  ===================  ===================
    ``.to(device=D)``   yes -> device is D   no  -> dtype is None
    ``.half()``         no  -> None          yes -> dtype is f16
    ``.to(D, f32)``     yes -> D             yes -> f32
    ==================  ===================  ===================

    Probing one axis with a single reference is what makes a device-only move
    report the scratch's own dtype and clobber ``dtype_float``.

    Scratch devices are always ones **known to be valid**, never the object's
    own tracker -- a tensor-free object can carry a stale tracker naming a
    backend this host does not have (``cuda:0`` on a CPU-only machine), and
    allocating there would fail exactly when the answer matters most.

    Returns
    -------
    tuple
        ``(device, dtype)``; either is ``None`` when ``fn`` leaves that axis
        untouched, or when ``fn`` could not be probed at all.
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
        # ``fn`` is arbitrary caller-supplied code, so a broad catch is correct
        # here -- unlike the cache hooks above, a non-conforming ``fn`` is an
        # expected outcome, not a corruption signal. The caller decides what an
        # unprobeable ``fn`` means.
        try:
            out = fn(scratch)
        except Exception:
            return None
        return out if isinstance(out, torch.Tensor) else None

    # The two axes get their own contrast pair, and each pair varies only along
    # the axis it measures. Sharing one pair for both would couple them: an
    # accelerator scratch cannot be cast to float64 on MPS, so probing dtype on
    # the device pair makes ``.double()`` unprobeable on Apple silicon.
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

    The tracker decides where the object allocates *next* -- over 200 call
    sites in the package pass ``device=self.device`` -- so leaving it stale on
    a tensor-free object silently misplaces every tensor that object later
    creates.

    Resolution order, applied per axis:

    1. a tensor the object itself owns (authoritative, already moved),
    2. the ``.to()`` request recorded on the traversal thread-local, which is
       the only source available to an object holding no tensors,
    3. a probe of ``fn`` (see :func:`_probe_target`), for the ``.float()`` and
       plain-parent paths that record no request.

    Only attributes already present in ``obj.__dict__`` and already holding a
    device/dtype-shaped value are written, so an unrelated attribute that
    happens to be named ``device`` is never clobbered. Written values are
    canonical, so ``obj.device == some_tensor.device`` holds.
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
        if _PARSE_TO is not None:
            # Let PyTorch's own parser raise on invalid arguments.
            device, dtype = _PARSE_TO(*args, **kwargs)[:2]
        else:  # pragma: no cover - only on a torch build without the private API
            device, dtype = _parse_to_args(args, kwargs)

        # Build the request BEFORE claiming the traversal. ``_ToRequest``
        # canonicalises the device, which raises for a backend this host does
        # not have (``.to('cuda')`` on a CUDA-less machine). Constructing it
        # after ``_enter_traversal`` but outside the ``try`` leaked the
        # thread-local visited set on that raise, and every later ``.to()`` on
        # the thread then short-circuited as "already visited" -- moving
        # nothing, silently.
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

            # Snapshot before anything moves, so step 5 can tell a real
            # transformation from a ``.to()`` that targets the current state.
            old_device, old_dtype = _observed_state(self)

            # 1. Standard nn.Module traversal (params, buffers, child modules).
            if isinstance(self, nn.Module):
                if _MODULE_APPLY_TAKES_RECURSE:
                    super()._apply(fn, recurse=recurse)
                else:  # pragma: no cover - older torch without the kwarg
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
            _refresh_device_trackers(self, fn)

            # 4. Invalidate caches.
            _invalidate_caches(self)

            # 5. Notify the object iff something actually changed, so index
            #    rebuilds do not fire on a no-op ``.to(current_device)``.
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

        No-op by default. Override to rebuild state derived from tensor
        placement -- precomputed index tensors, device-specific caches -- that
        the generic walk cannot repair on its own.

        Use this rather than ``reset_cache()``: ``reset_cache`` is a
        functional-cache hook that :meth:`LossState.reset_caches` fires after
        *every* optimizer step, so rebuilding indices there would put index
        reconstruction and GPU syncs on the hot path. This hook only fires when
        movement actually happened.

        Parameters
        ----------
        old_device, new_device : torch.device or None
            Device before and after. ``None`` when the object owned no tensors
            on that side of the move.
        old_dtype, new_dtype : torch.dtype or None
            Floating/complex dtype before and after, resolved independently of
            the device.
        device_changed, dtype_changed : bool
            Which axes actually changed. At least one is always True.
        """


# ---------------------------------------------------------------------------
# Backwards-compatibility aliases
# ---------------------------------------------------------------------------

# Older code imports ``DeviceMovementMixin``; keep it pointing at the
# active implementation so existing classes pick up the new behaviour.
DeviceMovementMixin = DeviceMixin

# ``_NonModuleDeviceMixin`` was briefly distinct; the unified mixin now
# handles both Module and non-Module cases.
_NonModuleDeviceMixin = DeviceMixin
