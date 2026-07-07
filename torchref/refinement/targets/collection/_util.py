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
