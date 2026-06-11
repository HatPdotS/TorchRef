"""Shared helpers for collection (multi-dataset) targets."""

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
import torch

if TYPE_CHECKING:
    from torchref.io.datasets.reflection_data import ReflectionData

_LOG_2PI = np.log(2.0 * np.pi)


def _unpack_masked_data(
    data: "ReflectionData",
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]
]:
    """Extract plain tensors + validity mask from a ReflectionData call.

    Returns
    -------
    F_obs, sigma, rfree_bool, validity, centric
        All as plain tensors (not MaskedTensors).  *rfree_bool* has
        True = work, False = free.  *centric* may be None.
    """
    F_obs, sigma = data.get_corrected_data()
    rfree = data.rfree_flags
    validity = data.masks().to(torch.bool)
    centric = data.centric if hasattr(data, "centric") else None
    return F_obs, sigma, rfree.bool(), validity, centric


def _scale_fcalc(scaler, fcalc, model):
    """Apply scaler, using forward_mixed when available."""
    if scaler is None:
        return fcalc
    if hasattr(scaler, "forward_mixed") and hasattr(model, "fractions"):
        return scaler.forward_mixed(fcalc, model.fractions)
    return scaler(fcalc)
