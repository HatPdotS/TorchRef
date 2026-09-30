"""Map coefficients from observed amplitudes and a model.

The one place the 2Fo-Fc and Fo-Fc coefficients are formed, for real-space maps
(:class:`torchref.maps.Map`) and for the FWT/DELFWT columns an MTZ carries.
"""

from typing import Optional, Tuple

import torch

__all__ = ["map_coefficients"]


def map_coefficients(
    fobs: torch.Tensor,
    fcalc: torch.Tensor,
    observed: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unweighted ``2Fo-Fc`` and ``Fo-Fc`` coefficients on the model phases.

    ``2Fo-Fc = (2 Fo - |Fc|) exp(i phi_c)`` and ``Fo-Fc = (Fo - |Fc|) exp(i phi_c)``,
    both signed: where ``2 Fo < |Fc|`` the 2Fo-Fc coefficient points opposite to
    the model phase, which an amplitude/phase pair must carry as a 180° flip.
    These are the m = 1, D = 1 forms, not likelihood-weighted 2mFo-DFc maps.

    Parameters
    ----------
    fobs : torch.Tensor
        Observed amplitudes, shape (N,).
    fcalc : torch.Tensor
        Complex model structure factors on the same scale, shape (N,).
    observed : torch.Tensor, optional
        Boolean (N,), reflections with a usable measurement. Unobserved ones get
        ``Fc`` in 2Fo-Fc (the model fills the missing term) and zero in Fo-Fc.
        Default: all observed.

    Returns
    -------
    two_fo_fc, fo_fc : torch.Tensor
        Complex coefficients, shape (N,).
    """
    fcalc_amp = fcalc.abs()
    phase = torch.exp(1j * torch.angle(fcalc))
    fobs = fobs.to(fcalc_amp)
    two_fo_fc = (2.0 * fobs - fcalc_amp) * phase
    fo_fc = (fobs - fcalc_amp) * phase
    if observed is not None:
        observed = observed.to(device=fcalc.device, dtype=torch.bool)
        two_fo_fc = torch.where(observed, two_fo_fc, fcalc)
        fo_fc = torch.where(observed, fo_fc, torch.zeros_like(fo_fc))
    return two_fo_fc, fo_fc
