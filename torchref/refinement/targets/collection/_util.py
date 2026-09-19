"""Shared helpers for collection (multi-dataset) targets."""

import numpy as np

_LOG_2PI = np.log(2.0 * np.pi)


def _scale_fcalc(scaler, fcalc, model):
    """Apply scaler, using forward_mixed when available."""
    if scaler is None:
        return fcalc
    if hasattr(scaler, "forward_mixed") and hasattr(model, "fractions"):
        return scaler.forward_mixed(fcalc, model.fractions)
    return scaler(fcalc)


def common_geom(data):
    """``(epsilon, d_star_sq)`` on a dataset's HKL: multiplicity and ``1/d**2`` in A^-2."""
    import torch

    from torchref.base.reciprocal import get_scattering_vectors
    from torchref.refinement.model_error_estimation.sigma_a import epsilon_from_hkl

    eps = epsilon_from_hkl(data.hkl, getattr(data, "spacegroup", None))
    s = get_scattering_vectors(data.hkl, data.cell)
    dss = (torch.norm(s, dim=1) ** 2).to(eps.dtype)
    return eps, dss
