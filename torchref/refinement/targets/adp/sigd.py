"""Shifted inverse-gamma (SIGD) prior on the ADP distribution."""

import math

import torch
from typing import TYPE_CHECKING, Dict

from torchref.base.targets.adp import adp_sigd_math, u6_b_eq
from torchref.utils.stats import (
    VERBOSITY_DEBUG,
    VERBOSITY_DETAILED,
    VERBOSITY_STANDARD,
    StatEntry,
    stat,
)

from .base import ADPTarget

if TYPE_CHECKING:
    from torchref.model.model import Model


class ADPSigdTarget(ADPTarget):
    """
    ADP distribution prior: a shifted inverse-gamma distribution on the B-factors.

    Masmaliyeva & Murshudov (2019), *Acta Cryst.* D **75**, 505-518, and
    Masmaliyeva, Babai & Murshudov (2020), *Acta Cryst.* D **76**, 926-937,
    showed both theoretically and empirically that macromolecular B values follow
    a shifted inverse-gamma distribution (SIGD). This target restrains the
    B-factor distribution toward that form, replacing an earlier term that used a
    Gaussian in ``log(B)`` (i.e. a log-normal), which fits deposited structures
    measurably worse.

    The shape parameter ``alpha`` plays exactly the role the log-normal's sigma
    did -- it fixes the log-width via ``std(log B) = sqrt(trigamma(alpha))``,
    independent of the scale -- while the scale ``beta`` is set from the detached
    mean B each call, the analogue of the detached mean in the term this
    replaces. The restraint therefore constrains only the *shape* of the
    distribution and never drives the overall B level.

    Parameters
    ----------
    model : Model
        Reference to the Model object.
    alpha : float, optional
        SIGD shape parameter. Default 3.5, the value Masmaliyeva & Murshudov
        soft-restrain to across the PDB, implying ``std(log B) = 0.575`` at
        ``b_shift = 0``. Larger alpha is a narrower reference distribution and a
        stronger restraint.
    b_shift : float, optional
        SIGD shift ``B0`` (Å²). Default 0.0. A non-zero B0 is the signature of
        sharpening/blurring of the Fourier coefficients, which the same authors
        recommend avoiding during refinement; it is also strongly degenerate with
        ``alpha``, so pinning it at zero keeps the two parameters identifiable.
    verbose : int, optional
        Verbosity level. Default is 0.
    """

    name: str = "adp/sigd"

    def __init__(
        self,
        model: "Model" = None,
        alpha: float = 3.5,
        b_shift: float = 0.0,
        verbose: int = 0,
        device=None,
    ):
        super().__init__(model, verbose, device=device)
        # Buffers, not floats: both are consumed by adp_sigd_math as tensors, and
        # they must follow the target across devices and through state_dict.
        self._register_scalar("_alpha", float(alpha))
        self._register_scalar("_b_shift", float(b_shift))

    @property
    def alpha(self) -> float:
        """SIGD shape parameter."""
        return self._alpha.item()

    @alpha.setter
    def alpha(self, value: float):
        self._alpha.fill_(value)

    @property
    def b_shift(self) -> float:
        """SIGD shift B0 (Å²)."""
        return self._b_shift.item()

    @b_shift.setter
    def b_shift(self, value: float):
        self._b_shift.fill_(value)

    def _b_values(self) -> torch.Tensor:
        """Per-atom B, using B_eq when any atom is anisotropic.

        Mirrors the iso/aniso split in :class:`ADPSimilarityTarget`: an
        all-isotropic model takes the cheaper direct path and is numerically
        identical, since ``u6_b_eq`` reduces to B for isotropic atoms.
        """
        if not getattr(self.model, "_aniso_is_empty", True):
            return u6_b_eq(self.model.adp_u6())
        return self.model.adp()

    def forward(self) -> torch.Tensor:
        """Summed SIGD NLL over atoms, offset to be non-negative per atom."""
        return adp_sigd_math(self._b_values(), self._alpha, self._b_shift)

    def stats(self) -> Dict[str, any]:
        """Get SIGD prior statistics."""
        adp = self._b_values().detach()
        log_adp = torch.log(adp.clamp(min=1e-3))
        alpha = self.alpha
        loss = self.forward()

        # The scale the restraint is currently referencing, and the log-width it
        # implies -- compare against std_log_adp to read the fit at a glance.
        beta = float((adp - self._b_shift).clamp(min=1e-3).mean()) * (alpha - 1.0)
        # std(log B) = sqrt(trigamma(alpha)); torch.polygamma(1, .) is trigamma.
        implied_std = math.sqrt(
            float(torch.polygamma(1, torch.tensor(alpha, dtype=torch.float64)))
        )

        return {
            "loss": stat(loss.item(), VERBOSITY_STANDARD),
            "n_atoms": stat(len(adp), VERBOSITY_DEBUG),
            "mean_adp": stat(adp.mean().item(), VERBOSITY_DETAILED),
            "std_adp": stat(adp.std().item(), VERBOSITY_DETAILED),
            "min_adp": stat(adp.min().item(), VERBOSITY_DETAILED),
            "max_adp": stat(adp.max().item(), VERBOSITY_DETAILED),
            "mean_log_adp": stat(log_adp.mean().item(), VERBOSITY_DEBUG),
            "std_log_adp": stat(log_adp.std().item(), VERBOSITY_DETAILED),
            "implied_std_log_adp": stat(implied_std, VERBOSITY_DETAILED),
            "alpha": stat(alpha, VERBOSITY_DEBUG),
            "beta": stat(beta, VERBOSITY_DEBUG),
            "b_shift": stat(self.b_shift, VERBOSITY_DEBUG),
        }
