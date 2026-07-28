"""Device-consistency assertions for torchref objects.

The walker here is written **independently** of
:func:`torchref.utils.device_mixin._apply_to_obj`. That is deliberate: an oracle
that mirrors the implementation it checks can only ever confirm the
implementation's own idea of what is reachable, so it would be blind to exactly
the tensors ``DeviceMixin`` forgets to move. This walker instead enumerates
everything it can reach by generic Python/PyTorch means -- ``__dict__``,
``__slots__``, module parameters/buffers/children, containers, dataclass
fields, and :class:`~torchref.utils.utils.ModuleReference` wrappers.

Exports
-------
assert_device_consistent
    Fail unless every reachable tensor *and* every device tracker sits on the
    expected device. Reports all mismatches at once.
collect_device_map
    The underlying survey, returned as ``{path: torch.device}``. Useful when a
    test wants to assert something more specific than "all on one device".
HOST_SIDE
    Paths that are intentionally CPU-resident regardless of the object's
    device. Every entry needs a comment justifying it.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import torch
from torch import nn

from torchref.config import canonical_device

__all__ = [
    "assert_device_consistent",
    "collect_device_map",
    "HOST_SIDE",
]


# Attribute *leaf names* that are legitimately CPU-resident even when the
# owning object lives on an accelerator. Keep this list short and justified --
# it is the escape hatch that makes the conformance suite meaningful rather
# than a rubber stamp.
#
# NOTE: a registered ``nn.Module`` buffer can never be host-side in practice,
# because ``module.to(device)`` relocates it regardless of what this set says.
# Anything that genuinely must stay on the host belongs in plain Python state
# (or ``get_extra_state``), not in a buffer. If you find yourself wanting to add
# a buffer name here, convert the buffer instead.
HOST_SIDE: Set[str] = set()


# Attributes on ``nn.Module`` that hold PyTorch's own bookkeeping. They are
# traversed explicitly via ``named_parameters``/``named_buffers``/
# ``named_children``, so walking the raw dicts as well would only duplicate work
# and produce confusing double paths.
_MODULE_INTERNALS = frozenset(
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

_TRACKER_ATTRS = ("device", "_device")
_DTYPE_TRACKER_ATTRS = ("dtype_float", "_dtype")

_MAX_DEPTH = 12


def _slot_names(obj: Any) -> Iterable[str]:
    """Every ``__slots__`` entry declared anywhere in ``type(obj)``'s MRO."""
    for klass in type(obj).__mro__:
        for name in getattr(klass, "__slots__", ()) or ():
            yield name


def _iter_attributes(obj: Any) -> Iterable[Tuple[str, Any]]:
    """Yield ``(name, value)`` for the object's own state.

    Covers ``__dict__`` (skipping ``nn.Module`` bookkeeping), ``__slots__``, and
    dataclass fields -- a dataclass with ``__slots__`` exposes neither through
    ``__dict__``, which is how ``_ReflectionSubset``-style views hide tensors
    from a naive walk.
    """
    seen: Set[str] = set()

    for name, value in list(getattr(obj, "__dict__", {}).items()):
        if name in _MODULE_INTERNALS:
            continue
        seen.add(name)
        yield name, value

    for name in _slot_names(obj):
        if name in seen or name.startswith("__"):
            continue
        seen.add(name)
        try:
            yield name, getattr(obj, name)
        except AttributeError:
            continue  # declared slot, never assigned

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for field in dataclasses.fields(obj):
            if field.name in seen:
                continue
            seen.add(field.name)
            try:
                yield field.name, getattr(obj, field.name)
            except AttributeError:
                continue


def _walk(
    obj: Any,
    path: str,
    found: Dict[str, torch.device],
    trackers: Dict[str, torch.device],
    dtypes: Dict[str, torch.dtype],
    dtype_trackers: Dict[str, torch.dtype],
    visited: Set[int],
    depth: int,
) -> None:
    if obj is None or depth > _MAX_DEPTH:
        return
    if isinstance(obj, (str, bytes, int, float, bool, complex, torch.dtype, torch.device)):
        return

    if isinstance(obj, torch.Tensor):
        found[path] = obj.device
        # Only floating/complex tensors carry the configured float dtype;
        # integer index buffers and boolean masks have their own and must not
        # be swept into the comparison.
        if obj.is_floating_point() or obj.is_complex():
            dtypes[path] = obj.dtype
        return

    if id(obj) in visited:
        return
    visited.add(id(obj))

    args = (found, trackers, dtypes, dtype_trackers, visited)

    if isinstance(obj, nn.Module):
        for name, tensor in list(obj.named_parameters(recurse=False)) + list(
            obj.named_buffers(recurse=False)
        ):
            if tensor is None:
                continue
            found[f"{path}.{name}"] = tensor.device
            if tensor.is_floating_point() or tensor.is_complex():
                dtypes[f"{path}.{name}"] = tensor.dtype
        for name, child in obj.named_children():
            _walk(child, f"{path}.{name}", *args, depth + 1)

    # ``ModuleReference`` deliberately hides its payload from PyTorch's module
    # tree, so a parameters/buffers sweep never sees it -- but the referenced
    # object still has to agree on a device with the referrer.
    wrapped = getattr(obj, "_wrapped_module", None)
    if wrapped is not None and not isinstance(obj, nn.Module):
        _walk(wrapped, f"{path}->ref", *args, depth + 1)
        return

    for name, value in _iter_attributes(obj):
        if name in _TRACKER_ATTRS and isinstance(value, (torch.device, str)):
            try:
                trackers[f"{path}.{name}"] = torch.device(value)
            except (RuntimeError, TypeError):
                pass  # a ``device``-named attribute that isn't a device
            continue
        if name in _DTYPE_TRACKER_ATTRS and isinstance(value, torch.dtype):
            dtype_trackers[f"{path}.{name}"] = value
            continue
        if name in HOST_SIDE:
            continue
        _walk(value, f"{path}.{name}", *args, depth + 1)

    if isinstance(obj, dict):
        for key, value in list(obj.items()):
            _walk(value, f"{path}[{key!r}]", *args, depth + 1)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for index, value in enumerate(obj):
            _walk(value, f"{path}[{index}]", *args, depth + 1)


def collect_device_map(obj: Any, name: str = "obj") -> Tuple[Dict, Dict, Dict, Dict]:
    """Survey ``obj`` along both the device and dtype axes.

    Returns ``(tensor_devices, tracker_devices, tensor_dtypes, tracker_dtypes)``,
    each ``{path: value}``. Paths are dotted attribute chains, with ``[i]`` /
    ``['k']`` for container elements and ``->ref`` where the walk crossed a
    :class:`ModuleReference`.

    The dtype maps cover only floating/complex tensors and the
    ``dtype_float`` / ``_dtype`` trackers, mirroring the mixin's rule that the
    two axes are resolved independently -- an integer index buffer says nothing
    about the configured float dtype.
    """
    tensors: Dict[str, torch.device] = {}
    trackers: Dict[str, torch.device] = {}
    tensor_dtypes: Dict[str, torch.dtype] = {}
    tracker_dtypes: Dict[str, torch.dtype] = {}
    _walk(obj, name, tensors, trackers, tensor_dtypes, tracker_dtypes, set(), 0)
    return tensors, trackers, tensor_dtypes, tracker_dtypes


def assert_device_consistent(
    obj: Any,
    expected: Any,
    *,
    name: str = "obj",
    ignore: Optional[Iterable[str]] = None,
    expected_dtype: Optional[torch.dtype] = None,
) -> None:
    """Assert every reachable tensor and tracker sits on ``expected``.

    Parameters
    ----------
    obj
        Object to survey.
    expected
        Target device. Compared through
        :func:`torchref.config.canonical_device` on both sides, so a bare
        ``mps`` and an indexed ``mps:0`` are treated as equal -- the helper
        tests placement, not spelling.
    name
        Root label used to build the reported paths.
    ignore
        Path substrings to exclude. Use for genuinely borrowed state, not to
        paper over a split object.
    expected_dtype
        When given, also assert every floating/complex tensor and every
        ``dtype_float`` / ``_dtype`` tracker matches. This is the only way to
        catch the ``torch.tensor(float(x))`` class of bug, where a tunable is
        hardcoded float32 under a float64 configuration.

    Raises
    ------
    AssertionError
        Listing **every** mismatch. A half-moved object usually has several,
        and reporting only the first turns one debugging session into five.
    """
    expected_dev = canonical_device(expected)
    ignore = tuple(ignore or ())

    tensors, trackers, tensor_dtypes, tracker_dtypes = collect_device_map(obj, name)
    problems: List[str] = []

    def skipped(path: str) -> bool:
        return any(frag in path for frag in ignore)

    for path, dev in sorted(tensors.items()):
        if not skipped(path) and canonical_device(dev) != expected_dev:
            problems.append(f"{path}: tensor on {dev}, expected {expected_dev}")

    for path, dev in sorted(trackers.items()):
        if not skipped(path) and canonical_device(dev) != expected_dev:
            problems.append(f"{path}: tracker == {dev}, expected {expected_dev}")

    if expected_dtype is not None:
        for path, dt in sorted(tensor_dtypes.items()):
            if not skipped(path) and dt != expected_dtype:
                problems.append(f"{path}: tensor dtype {dt}, expected {expected_dtype}")
        for path, dt in sorted(tracker_dtypes.items()):
            if not skipped(path) and dt != expected_dtype:
                problems.append(
                    f"{path}: dtype tracker == {dt}, expected {expected_dtype}"
                )

    if problems:
        target = f"{expected_dev}"
        if expected_dtype is not None:
            target += f" / {expected_dtype}"
        raise AssertionError(
            f"{type(obj).__name__} is not consistently on {target} "
            f"({len(problems)} mismatch(es)):\n  " + "\n  ".join(problems)
        )
