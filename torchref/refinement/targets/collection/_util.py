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
    _, F_obs, sigma, rfree = data()
    if hasattr(F_obs, "get_mask"):
        validity = F_obs.get_mask()
        F_obs = F_obs.get_data()
        sigma = sigma.get_data() if hasattr(sigma, "get_mask") else sigma
    else:
        validity = torch.ones(len(F_obs), dtype=torch.bool, device=F_obs.device)
    centric = data.centric if hasattr(data, "centric") else None
    return F_obs, sigma, rfree.bool(), validity, centric


def _scale_fcalc(scaler, fcalc, model):
    """Apply scaler, using forward_mixed when available."""
    if scaler is None:
        return fcalc
    if hasattr(scaler, "forward_mixed") and hasattr(model, "fractions"):
        return scaler.forward_mixed(fcalc, model.fractions)
    return scaler(fcalc)
