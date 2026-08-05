"""Centralized device resolution for multi-module constructors.

Many TorchRef constructors take several device-bearing inputs (``model`` +
``data``, or ``data`` + ``data_reference`` + ``model``). :func:`resolve_device`
collapses them onto one device with a fixed precedence;
:func:`require_cell_dtype` is the dtype-axis counterpart.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Union

import torch

from torchref.config import canonical_device as _canonical
from torchref.config import get_default_device


def resolve_device(
    *modules: Any,
    device: Optional[Union[torch.device, str]] = None,
) -> torch.device:
    """Resolve one device from N device-bearing modules, moving them to agree.

    Modules must expose ``.device`` and ``.to(device)`` (``nn.Module`` or
    :class:`~torchref.utils.DeviceMixin`); ``None`` entries are skipped, so
    optional submodules can be passed unconditionally. With ``device`` given,
    all modules are moved to it. Otherwise the first module's device wins --
    callers express precedence by argument order -- and any module elsewhere is
    moved to it with a :class:`UserWarning`; with no modules at all,
    :func:`torchref.config.get_default_device`.

    Parameters
    ----------
    *modules
        Device-bearing modules. ``None`` entries are skipped.
    device : torch.device or str, optional
        Explicit override. All modules are moved to it; no warning.

    Returns
    -------
    torch.device
        The resolved device.

    Raises
    ------
    TypeError
        If passed a bare ``Tensor``/``nn.Parameter``. ``Tensor.to()`` is
        out-of-place, so the move would be silently discarded. Use
        :func:`torchref.config.normalize_device` to read a device off a tensor;
        pass owning objects here.
    """
    for m in modules:
        if isinstance(m, torch.Tensor):
            raise TypeError(
                "resolve_device() moves its inputs in place and returns only "
                f"the resolved device, but a bare {type(m).__name__} was "
                "passed. torch.Tensor.to() is out-of-place, so the move would "
                "be silently discarded and the tensor left where it was. Pass "
                "the object that owns the tensor, or move it yourself: "
                "t = t.to(normalize_device(...))."
            )

    if device is not None:
        resolved = _canonical(device)
        for m in modules:
            # Skip modules already on target. ``.to()`` invalidates caches and
            # fires ``_after_device_apply``, so a no-op move is not free --
            # ``SfDS.forward`` calls ``resolve_device`` on every evaluation.
            #
            # This is only safe because ``DeviceMixin`` now keeps ``m.device``
            # truthful; before that, the unconditional ``.to()`` here was
            # accidentally repairing objects whose tracker lied about where
            # their tensors were.
            if m is not None and _canonical(m.device) != resolved:
                m.to(resolved)
        return resolved

    present = [m for m in modules if m is not None]
    if not present:
        return get_default_device()

    target = _canonical(present[0].device)
    inconsistent = [m for m in present[1:] if _canonical(m.device) != target]
    if inconsistent:
        device_list = [m.device for m in present]
        warnings.warn(
            f"resolve_device: inputs on differing devices {device_list}; "
            f"moving {len(inconsistent)} module(s) to {target} "
            "(the first input's device).",
            stacklevel=2,
        )
        for m in inconsistent:
            m.to(target)
    return target


def require_cell_dtype(cell: Any, dtype: torch.dtype, owner: str) -> None:
    """Refuse a cell whose dtype disagrees with its owner's declared float dtype.

    The dtype-axis counterpart to :func:`resolve_device`, but it refuses rather
    than repairs: moving a tensor is lossless, casting one either discards
    precision the caller chose or manufactures digits never measured. Call it at
    the point of *use* -- :class:`~torchref.symmetry.cell.Cell` casts in place,
    so a constructor check cannot see a later recast.

    Parameters
    ----------
    cell : Cell or None
        The cell to check. ``None`` is ignored.
    dtype : torch.dtype
        The owner's declared float dtype (``self.dtype_float``).
    owner : str
        Class name of the owner, for the error message.

    Raises
    ------
    RuntimeError
        If ``cell.dtype`` is not ``dtype``.
    """
    if cell is None or cell.dtype == dtype:
        return
    raise RuntimeError(
        f"{owner} was built for {dtype} but its cell holds {cell.dtype}. Every "
        "cell-derived quantity (the fractional matrices, the reciprocal basis) is taken "
        f"from the cell, so the calculation would run at {cell.dtype} regardless of the "
        f"requested {dtype} and fail on the first mixed-dtype operation. This normally "
        "means the cell was recast or replaced after the module was constructed -- "
        f"``Cell.to()`` mutates in place. Rebuild the cell at {dtype}, or construct a new "
        f"{owner} for the dtype you want."
    )
