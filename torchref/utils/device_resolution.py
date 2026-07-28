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
