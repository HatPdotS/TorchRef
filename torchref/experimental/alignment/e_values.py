"""One place to say what ``E`` means.

``E = F / sqrt(Sigma(s))`` is a **weighting choice wearing the costume of a units
change**: correlating ``E_obs`` against ``E_calc`` *is* correlating ``F`` against
``F`` with weight ``1/Sigma(s)``. The alignment package currently answers that
question nine different times -- twice in ``frf.preprocessing``, once inside
``french_wilson_preprocess`` and three more in ``translation`` -- and the
answers disagree. The rotation function's observed
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
    "convention_class",
    "convention_for_calc",
    "convention_uses_sigma_f",
]


def convention_class(conv) -> type:
    """The class behind a convention, which may be a ``functools.partial``.

    Configuration rides in as ``partial(SmoothSigmaE, n_coeff=6)``, and a
    partial forwards ``__call__`` but not class attributes -- so asking one for
    ``uses_sigma_f`` or ``for_calc`` raises. Every attribute lookup on a
    convention goes through here for that reason.
    """
    return getattr(conv, "func", conv)


def convention_for_calc(conv):
    """The convention to normalise **calculated** amplitudes with.

    Keeps the partial's configuration when the class is its own companion, and
    drops it when a different class is named -- another class's keywords are not
    this one's.
    """
    companion = convention_class(conv).calc_companion
    return conv if companion is None else companion


def convention_uses_sigma_f(conv) -> bool:
    """Whether ``conv`` consumes ``sig_F``, partial or not."""
    return bool(getattr(convention_class(conv), "uses_sigma_f", False))


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

    #: Whether this convention divides the intensity by ``eps``. A convention
    #: that does not must ignore it on BOTH sides of the ratio: applying it to
    #: the shell mean alone leaves ``<E**2> = <eps>``, which is 2 on a centred
    #: lattice and 1 on a primitive one -- a normaliser whose scale depends on
    #: the space group. Declared rather than implied so the two halves cannot
    #: drift apart again.
    uses_epsilon: bool = True

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
        """``F**2 / eps`` -- the quantity whose shell mean is ``Sigma``.

        Both the numerator of ``E**2`` and its shell mean come through here, so
        ``uses_epsilon`` reaches the ratio consistently by construction.
        """
        I = self.F * self.F
        if self.eps is not None and self.uses_epsilon:
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
    the data carry no sigmas. Ignores measurement error, and multiplicity with
    it -- both deliberately. The calc side is a single molecular transform
    sampled in a P1 box, where multiplicity has no meaning; the obs side gets
    its multiplicity from the symmetry unroll, which puts each reflection into
    the sum once per operation that reaches it.

    So an ``eps`` passed to this class is *ignored*, not half-applied. Use
    :class:`WilsonShellEpsE` when it should count.
    """

    uses_epsilon = False

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

    ``french_wilson_preprocess`` takes no multiplicity, so neither does this --
    declared so the reported ``sigma`` describes what was actually done rather
    than what the base class would have done.
    """

    uses_sigma_f = True
    uses_epsilon = False
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


#: The observed side carries multiplicity; the calculated side is a single
#: molecular transform sampled at the same Miller indices, where multiplicity
#: has no meaning. Assigned out of the class body only because `CalcShellE` is
#: defined below `WilsonShellEpsE`.
WilsonShellEpsE.calc_companion = CalcShellE


class CalcGlobalE(EConvention):
    """Single global scale: ``E = F / rms(F)``, preserving inter-shell shape.

    The rescore's ``scat_mode="absolute"``. Keeps how much the model actually
    scatters per resolution instead of flattening it, which is what makes a
    relative Wilson-B correction meaningful rather than cancelled.

    Calculated amplitudes, so multiplicity does not apply -- see
    :class:`WilsonShellE`.
    """

    uses_epsilon = False

    def _compute(self):
        rms = self._intensity().mean().clamp(min=1e-30).sqrt()
        self.sigma = torch.full_like(self.F, float(rms * rms))
        return self.F / rms, self._ones()


class SmoothSigmaE(EConvention):
    """Adapter over the shared :class:`~torchref.scaling.WilsonNormaliser`.

    The fit itself does not live here, because "what is the mean intensity at
    this resolution" is not an alignment question -- at least five private
    answers to it grew across the repo, and consumers that disagree about it
    cannot be compared with each other. What stays here is the ``EConvention``
    protocol that this package's consumers and its conformance harness are
    written against.

    ``weight`` is ones, and that is the point of the split rather than an
    omission: this class answers *what* we compare. How much each reflection
    counts is a weight, built from ``sigI`` and model error, and belongs
    elsewhere. Returning both from one object is what made the previous
    conventions impossible to interpret -- sweeping one moved a gauge quantity
    and a real one at the same time.

    Pass ``s_lo``/``s_hi`` whenever obs and calc are fitted separately and their
    curves will be compared: the basis saturates at the ends, so a curve
    evaluated beyond its own fitted range is frozen flat rather than
    extrapolated.
    """

    #: Chebyshev terms. Provisional -- the order has never been screened against
    #: a metric sensitive to it.
    DEFAULT_N_COEFF = 6

    def __init__(self, *args, n_coeff: int = DEFAULT_N_COEFF,
                 s_lo=None, s_hi=None, **kwargs) -> None:
        self.n_coeff = int(n_coeff)
        self.s_lo = s_lo
        self.s_hi = s_hi
        super().__init__(*args, **kwargs)

    def _compute(self):
        from torchref.scaling import WilsonNormaliser

        # `_intensity()` has already divided by eps -- `uses_epsilon` is True
        # here -- so the normaliser gets a reduced intensity and no eps of its
        # own. Applying it in both places would count multiplicity twice.
        fit = WilsonNormaliser(
            self._intensity(), self.s_mag, centric=self.centric,
            n_coeff=self.n_coeff, s_lo=self.s_lo, s_hi=self.s_hi,
        )
        # Kept, not discarded: the fitted CURVE is the thing anything comparing
        # two normalisations needs. Sigma_obs/Sigma_calc is how model error is
        # measured rather than assumed, and it can only be formed from the fits
        # themselves, not from the per-reflection values they produced.
        self.fit = fit
        self.sigma = fit.sigma_wilson
        return fit.E, self._ones()

    def evaluate(self, s_mag: torch.Tensor) -> torch.Tensor:
        """``Sigma(s)`` at arbitrary ``|s|``, on this fit's own abscissa."""
        return self.fit.evaluate(s_mag)
