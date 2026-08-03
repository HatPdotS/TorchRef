"""Centralized device resolution for multi-module constructors.

Many TorchRef constructors accept several device-bearing inputs
(``model`` + ``data``, or ``data`` + ``data_reference`` + ``model``,
etc.).  Each used to have its own ad-hoc rule for picking a device,
which led to silent bugs whenever inputs disagreed (e.g.  passing
``device='cpu'`` to ``Scaler`` while ``data`` lived on cuda would
still leave the ``s`` / ``bins`` buffers on cuda).

``resolve_device`` collapses N device sources into one with a fixed,
documented precedence.
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
    """Resolve a single device from N device-bearing modules.

    Each ``module`` must expose ``.device`` and accept ``.to(device)``
    (satisfied by ``torch.nn.Module`` and by ``torchref.utils.DeviceMixin``
    non-Module subclasses such as ``Cell``).  ``None`` entries are
    skipped silently so empty-init paths can pass through optional
    submodules — ``resolve_device(model, data)`` works whether or not
    ``data`` is ``None``.

    Resolution order
    ----------------
    1. If ``device`` is given, every non-``None`` module is moved to
       it and it is returned.  No warning is emitted (the caller has
       made an explicit choice).
    2. Otherwise, after dropping ``None`` entries, if no modules
       remain, :func:`torchref.config.get_default_device` is returned.
    3. The first remaining module's device is the target.  Any other
       module on a different device is moved to the target and a
       :class:`UserWarning` is emitted once for the call.

    The "first module wins" rule is intentional: callers express
    precedence by argument order.

    Parameters
    ----------
    *modules
        Device-bearing modules.  ``None`` entries are skipped.
    device : torch.device or str, optional
        Explicit override.  If provided, all non-``None`` modules are
        moved to it and it is returned.

    Returns
    -------
    torch.device
        The resolved device.

    Raises
    ------
    TypeError
        If a bare ``torch.Tensor`` (or ``nn.Parameter``) is passed. Such an
        object satisfies the ``.device`` / ``.to()`` precondition
        *syntactically* but violates it semantically, because
        ``Tensor.to()`` returns a new tensor rather than moving in place --
        so the move would be dropped without a word. Use
        :func:`torchref.config.normalize_device` to read a device off a
        tensor; use this function only to reconcile owning objects.

    Examples
    --------
    Empty call returns the configured default::

        >>> resolve_device()  # doctest: +SKIP
        device(type='cpu')

    Explicit override moves everything::

        >>> resolve_device(model, data, device='cpu')  # doctest: +SKIP
        device(type='cpu')

    Auto-reconcile with first-wins precedence (``cpu_data`` is moved to
    cuda to match the first module)::

        >>> resolve_device(cuda_model, cpu_data)  # doctest: +SKIP
        device(type='cuda')
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

    The counterpart to :func:`resolve_device` for the *dtype* axis -- and deliberately not
    the same policy. ``resolve_device`` reconciles by moving its inputs, which is right
    because relocating a tensor is lossless. Casting is not: silently pulling a float64 cell
    down to float32 would discard precision the caller chose explicitly, and pushing a
    float32 cell up to float64 would manufacture digits that were never measured. So this
    refuses instead of repairing, and leaves the choice with the caller.

    Called at the point of *use* rather than in a constructor, which is what makes it
    load-bearing: :class:`~torchref.symmetry.cell.Cell` is mutable and its ``to()`` operates
    in place, so a cell can be recast or replaced long after the owning module was built. A
    constructor check cannot see that; this can.

    What it buys is a diagnosis instead of a symptom. Every cell-derived quantity -- the
    fractional matrices, the reciprocal basis -- inherits the *cell's* dtype, so a
    disagreement surfaces as a bare ``RuntimeError: expected mat1 and mat2 to have the same
    dtype`` from whichever ``matmul`` happens to run first, with nothing naming the cell.

    Parameters
    ----------
    cell : Cell or None
        The cell to check. ``None`` is accepted and ignored, so callers may run this
        beside their own "is the cell set at all" precondition without ordering the two.
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
