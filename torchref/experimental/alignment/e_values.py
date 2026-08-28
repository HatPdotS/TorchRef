"""One place to say what ``E`` means.

``E = F / sqrt(Sigma(s))`` is a **weighting choice wearing the costume of a units
change**: correlating ``E_obs`` against ``E_calc`` *is* correlating ``F`` against
``F`` with weight ``1/Sigma(s)``. The alignment package currently answers that
question nine different times -- twice in ``frf.preprocessing``, once inside
``french_wilson_preprocess``, three times in ``ml_rotation`` and three more in
``translation`` -- and the answers disagree. The rotation function's observed
side is a French-Wilson posterior weighted by ``DFAC**2``; the rescore's is plain
per-shell Wilson with epsilon divided out. So the rescore ranks candidates
against a differently-normalised observation set than the one that produced them.

The two consumers do not need the same thing from it, which is worth stating
because it explains which of them breaks:

* The rotation function is a **correlation**. A global scale cancels -- it scales
  every SO(3) sample equally and the peak is reported as a z-score -- so only the
  *relative* weighting across resolution matters. Removing the antipodal copy
  scaled every score by exactly 4 and moved 98 of 100 truth ranks not at all.
* The rescore's LLG is a **likelihood**. It compares an observation against a
  predicted distribution, so there is no free scale to cancel; get it wrong and
  you evaluate the right data against the wrong Rice.

A convention that satisfies the likelihood satisfies the correlation for free,
so the strict requirement is the one to design against.

Conventions are constructed **from the data** rather than configured and passed
in, because a fitted ``Sigma(s)`` cannot exist before the reflections do. Engines
therefore take the class and instantiate it internally::

    FastRotationFunction(..., e_convention=FrenchWilsonE)

and anything needing configuration rides in as
``functools.partial(SmoothSigmaE, n_coeff=6)``, which is class-like and needs no
extra parameter.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from .sh import assign_shells, equal_count_shell_edges

__all__ = [
    "CalcGlobalE",
    "CalcShellE",
    "EConvention",
    "FrenchWilsonE",
    "SmoothSigmaE",
    "WilsonShellE",
    "WilsonShellEpsE",
]


class EConvention:
    """Normalised amplitudes, plus the per-reflection weight that goes with them.

    Attributes
    ----------
    E : torch.Tensor
        ``(N,)`` normalised amplitude. For most conventions this is
        ``F / sqrt(sigma)``; for :class:`FrenchWilsonE` it is a posterior
        expectation and the relation is only approximate, which is exactly the
        difference the conformance harness is there to expose.
    weight : torch.Tensor
        ``(N,)`` per-reflection information weight. ``DFAC**2`` where the
        convention models measurement error, ones where it does not. This is the
        "weight by F/sigma" lever, made explicit rather than left implicit in
        whichever normaliser a caller happened to pick.
    sigma : torch.Tensor
        ``(N,)`` the normaliser actually used, ``<F**2/eps>`` per reflection.
        Reported for every convention so they are comparable even when their
        ``E`` is not defined the same way.

    Notes
    -----
    ``eps`` divides the intensity before averaging (``E**2 = (F**2/eps) /
    <F**2/eps>``) because axial reflections are systematically stronger --
    ``<I> = eps * Sigma`` -- and would otherwise dominate. Conventions that leave
    it ``None`` are declaring that their caller handles multiplicity some other
    way; the rotation function does, by unrolling the full orbit.
    """

    #: Whether this convention consumes ``sig_F``. The conformance harness skips
    #: the shrinkage test for conventions that do not.
    uses_sigma_f: bool = False

    #: The convention to use for **calculated** amplitudes, when it cannot be
    #: this one. A French-Wilson posterior is defined for observations only --
    #: there is no measurement error on a calc set to shrink toward the mean --
    #: so a sigma_F-consuming convention has to name a companion. ``None`` means
    #: "use this class for both sides", which is what most of them do.
    #:
    #: This is not a harness convenience: it is why the rotation function pairs
    #: `french_wilson_preprocess` on obs with `wilson_normalise` on calc.
    calc_companion: Optional[type] = None

    @classmethod
    def for_calc(cls) -> type:
        """The class to normalise calculated amplitudes with."""
        return cls.calc_companion or cls

    def __init__(
        self,
        F: torch.Tensor,
        s_mag: torch.Tensor,
        centric: Optional[torch.Tensor] = None,
        *,
        sig_F: Optional[torch.Tensor] = None,
        eps: Optional[torch.Tensor] = None,
        shell_idx: Optional[torch.Tensor] = None,
        n_shells: int = 20,
    ) -> None:
        if F.ndim != 1:
            raise ValueError(f"F must be 1-D, got {tuple(F.shape)}")
        if s_mag.shape != F.shape:
            raise ValueError(
                f"s_mag {tuple(s_mag.shape)} does not match F {tuple(F.shape)}"
            )
        self.F = F
        self.s_mag = s_mag
        self.centric = (
            torch.zeros_like(F, dtype=torch.bool) if centric is None
            else centric.to(torch.bool)
        )
        self.sig_F = sig_F
        self.eps = eps
        self.n_shells = int(n_shells)
        # One shell assignment, shared by whatever the subclass needs it for.
        # Assigning here rather than in each subclass is the same fix the FRF
        # needed: two consumers deriving their own equal-count edges from the
        # same |s| disagree about the reflections sitting on a boundary.
        if shell_idx is None:
            edges, _ = equal_count_shell_edges(s_mag, self.n_shells)
            shell_idx = assign_shells(s_mag, edges)
        self.shell_idx = shell_idx.clamp(min=0)

        self.sigma = self._shell_mean_intensity()
        self.E, self.weight = self._compute()

    # -- helpers shared by the subclasses ---------------------------------

    def _intensity(self) -> torch.Tensor:
        """``F**2 / eps`` -- the quantity whose shell mean is ``Sigma``."""
        I = self.F * self.F
        if self.eps is not None:
            I = I / self.eps.clamp(min=1.0)
        return I

    def _shell_mean_intensity(self) -> torch.Tensor:
        """``<F**2/eps>`` per reflection, from the shared shell assignment."""
        I = self._intensity()
        total = torch.zeros(self.n_shells, dtype=I.dtype, device=I.device)
        total.scatter_add_(0, self.shell_idx, I)
        count = torch.bincount(
            self.shell_idx, minlength=self.n_shells,
        ).to(I.dtype).clamp(min=1.0)
        return (total / count).clamp(min=1e-30).index_select(0, self.shell_idx)

    def _ones(self) -> torch.Tensor:
        return torch.ones_like(self.F)

    def _compute(self):
        raise NotImplementedError

    def __repr__(self) -> str:                       # pragma: no cover - display
        return f"{type(self).__name__}(N={self.F.numel()}, n_shells={self.n_shells})"


class WilsonShellE(EConvention):
    """Plain per-shell Wilson: ``E = F / sqrt(<F**2>_shell)``.

    What the rotation function uses on the calc side, and on the obs side when
    the data carry no sigmas. Ignores measurement error entirely.
    """

    def _compute(self):
        return self.F / self.sigma.sqrt(), self._ones()


class WilsonShellEpsE(EConvention):
    """Epsilon-corrected Wilson, ``E**2 = (F**2/eps) / <F**2/eps>_shell``.

    What the m_LETF1 rescore uses on the observed side. Identical to
    :class:`WilsonShellE` when ``eps`` is absent, which is worth knowing: the
    difference between the two is only ever the multiplicity handling.
    """

    def _compute(self):
        E = (self._intensity() / self.sigma).clamp(min=0.0).sqrt()
        return E, self._ones()


class FrenchWilsonE(EConvention):
    """French-Wilson posterior amplitude with the Rice ``DFAC`` weight.

    The rotation function's observed-side default, and the only convention here
    that looks at ``sig_F``. ``E`` is the posterior expectation of the true
    normalised amplitude given a noisy measurement, so weak reflections shrink
    toward the shell mean instead of being taken at face value; ``weight`` is
    ``DFAC**2``, the Rice-moment D factor, which is the per-reflection
    measurement-information term.

    Requires ``sig_F``. Falling back silently to plain Wilson would hide exactly
    the difference this class exists to make visible.
    """

    uses_sigma_f = True
    calc_companion = WilsonShellE

    def _compute(self):
        if self.sig_F is None:
            raise ValueError(
                "FrenchWilsonE needs sig_F; use WilsonShellE for data without "
                "measurement errors rather than letting the difference pass "
                "silently."
            )
        from .frf.french_wilson import french_wilson_preprocess

        fw = french_wilson_preprocess(
            self.F, self.sig_F, self.s_mag, self.centric,
            n_wilson_shells=self.n_shells, shell_idx=self.shell_idx,
        )
        dfac = fw["DFAC"]
        return fw["eEobs"], dfac * dfac


class CalcShellE(WilsonShellE):
    """The rescore's calc-side normaliser: per-shell, flattening every shell to 1.

    Named separately from :class:`WilsonShellE` because it is applied to a
    *reference* orientation's ``|F_calc|`` and then reused for every rotated
    candidate. Predicted to fail the obs/calc common-scale check: forcing
    ``<E_calc**2>_shell = 1`` in every shell discards the model's inter-shell
    amplitude shape, which is the very thing the likelihood's expected intensity
    is supposed to carry.
    """


class CalcGlobalE(EConvention):
    """Single global scale: ``E = F / rms(F)``, preserving inter-shell shape.

    The rescore's ``scat_mode="absolute"``. Keeps how much the model actually
    scatters per resolution instead of flattening it, which is what makes a
    relative Wilson-B correction meaningful rather than cancelled.
    """

    def _compute(self):
        rms = self._intensity().mean().clamp(min=1e-30).sqrt()
        self.sigma = torch.full_like(self.F, float(rms * rms))
        return self.F / rms, self._ones()


class SmoothSigmaE(EConvention):
    """``Sigma(s)`` as a smooth curve rather than a step function over shells.

    Per-shell ``Sigma`` is a noisy non-parametric estimate with edges, and the
    edges are not free: two consumers binning the same ``|s|`` independently
    disagreed about 7 of 55078 reflections on 3K7M. A smooth curve has no edges,
    is the same function whichever subset it is evaluated on, and is what the
    scaler already uses for the closely-related isotropic scale.

    Basis follows ``scaling/scaler_base.py::_build_iso_design`` -- Chebyshev in
    ``sin(theta)/lambda`` mapped onto ``[-1, 1]``, evaluated per reflection, in
    log space. That abscissa rather than ``s**2`` because the modulation is
    gentle through the bulk of the range and has real structure in the first few
    percent of ``s**2``.

    Fitted as a **Gamma GLM with a log link**, which is the right likelihood
    rather than a convenience: acentric ``F**2`` is exponentially distributed
    with mean ``Sigma``, i.e. Gamma with unit shape, and centric ``F**2`` is
    Gamma with shape 1/2. Fitting the *mean* this way avoids the trap that a
    regression on ``log F**2`` walks into -- the ``E[log chi**2]`` offset has to
    go somewhere, and with no intercept it is absorbed into the shape of the
    curve. That is precisely how the overall-anisotropy fit was biased.

    Coefficients are clamped in log space for the reason the scaler clamps: a
    polynomial is unbounded at the ends of its interval, and the low-resolution
    end is where a mis-specified normaliser does its damage.
    """

    #: Chebyshev terms. Six is the scaler's default and spans a Wilson plot's
    #: curvature without chasing shell-to-shell noise.
    DEFAULT_N_COEFF = 6

    #: Log-space clamp on the fitted curve, as a factor either side of the
    #: global mean intensity. Wide enough never to bind on real data; present so
    #: an extrapolating polynomial cannot produce an arbitrary scale.
    LOG_CLAMP = 10.0

    def __init__(self, *args, n_coeff: int = DEFAULT_N_COEFF,
                 n_iter: int = 8, **kwargs) -> None:
        self.n_coeff = int(n_coeff)
        self.n_iter = int(n_iter)
        super().__init__(*args, **kwargs)

    def _design(self) -> torch.Tensor:
        """``(N, n_coeff)`` Chebyshev design in sin(theta)/lambda."""
        x = (self.s_mag * 0.5).clamp(min=0.0)
        lo, hi = x.min(), x.max()
        u = (2 * (x - lo) / (hi - lo).clamp(min=1e-12) - 1).clamp(-1.0, 1.0)
        cols = [torch.ones_like(u), u]
        for _ in range(2, self.n_coeff):
            cols.append(2 * u * cols[-1] - cols[-2])
        return torch.stack(cols[: self.n_coeff], dim=1)

    def _fit_log_sigma(self) -> torch.Tensor:
        """IRLS for a Gamma GLM with log link; returns log Sigma per reflection."""
        X = self._design().to(torch.float64)
        y = self._intensity().to(torch.float64).clamp(min=1e-30)
        # Gamma shape: 1 acentric (exponential), 1/2 centric. Used as the IRLS
        # weight, so better-determined reflections pull harder.
        w = torch.where(self.centric, 0.5, 1.0).to(torch.float64)

        # Seed at the global mean, i.e. the constant curve a single Wilson
        # scale would give. Every later iteration only adds shape.
        beta = torch.zeros(self.n_coeff, dtype=torch.float64, device=X.device)
        beta[0] = torch.log(y.mean().clamp(min=1e-30))
        for _ in range(self.n_iter):
            eta = (X @ beta).clamp(-self.LOG_CLAMP + float(beta[0]),
                                   self.LOG_CLAMP + float(beta[0]))
            mu = torch.exp(eta)
            # Log link with Gamma variance: the working response is
            # eta + (y - mu)/mu and the IRLS weight is constant in mu.
            z = eta + (y - mu) / mu.clamp(min=1e-30)
            XtW = X.transpose(0, 1) * w.unsqueeze(0)
            A = XtW @ X
            A = A + torch.eye(
                self.n_coeff, dtype=A.dtype, device=A.device,
            ) * 1e-10 * float(torch.diagonal(A).abs().max().clamp(min=1e-30))
            beta_new = torch.linalg.solve(A, XtW @ z)
            if torch.allclose(beta_new, beta, rtol=1e-10, atol=1e-12):
                beta = beta_new
                break
            beta = beta_new
        return (X @ beta).clamp(
            -self.LOG_CLAMP + float(beta[0]), self.LOG_CLAMP + float(beta[0]),
        )

    def _compute(self):
        shell_sigma = self.sigma                       # the per-shell fallback
        log_sigma = self._fit_log_sigma()
        sigma = torch.exp(log_sigma).to(self.F.dtype).clamp(min=1e-30)
        # Sanity: a fitted Sigma(s) must reproduce the data's own mean intensity.
        # A Gamma GLM on a Chebyshev basis can diverge when the calc amplitudes
        # span a huge dynamic range with near-zeros, and it did -- two of four
        # calc sets came back with <E^2> ~ 0, i.e. Sigma inflated by orders of
        # magnitude. Detect that against the quantity the fit is estimating and
        # fall back to the per-shell estimate rather than returning nonsense.
        mean_I = self._intensity().mean().clamp(min=1e-30)
        ratio = float((sigma.mean() / mean_I).clamp(min=1e-30))
        self.converged = 0.2 < ratio < 5.0
        if not self.converged:
            sigma = shell_sigma
        self.sigma = sigma
        E = (self._intensity() / self.sigma).clamp(min=0.0).sqrt()
        return E, self._ones()
