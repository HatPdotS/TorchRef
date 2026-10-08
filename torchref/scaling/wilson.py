"""Absolute Wilson normalisation: fit ``Sigma(s)`` and divide it out.

:class:`WilsonNormaliser` takes one dataset and fits its expected intensity as a smooth
function of resolution, so that dividing by it leaves ``<E^2> = 1``. Unlike
:class:`~torchref.scaling.scaler_base.ScalerBase`, which scales ``F_calc`` onto
``F_obs``, no second dataset enters the objective, and no weight either: how much a
reflection counts is :mod:`torchref.scaling.weighting`'s business. :func:`fit_wilson_b`
is the one-number summary, an overall Wilson B for priors and reports.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from torchref.config import get_float_dtype, get_int_dtype
from torchref.scaling._protein_gamma import protein_gamma as _protein_gamma
from torchref.scaling.basis import chebyshev_design

__all__ = [
    "WilsonNormaliser",
    "WilsonFit",
    "fit_wilson_b",
    "sum_f_squared",
    "PROTEIN_RESIDUE",
]

#: Chebyshev terms. Enough to follow a Wilson plot's curvature and the
#: low-resolution solvent deficit without chasing shell-to-shell noise.
#: Provisional: the order has never been chosen against a metric sensitive to
#: it, so screen on this class's own residual trend rather than on anything
#: downstream.
DEFAULT_N_COEFF = 6

#: Bound on ``log Sigma`` relative to its own constant term. A polynomial is
#: unbounded at the ends of its range, so without this a single extreme
#: reflection at the resolution limit can carry an arbitrary scale -- the same
#: reason ``ScalerBase.iso_log_scale`` clamps per reflection.
LOG_CLAMP = 10.0

#: Step halvings allowed per IRLS iteration before the step is abandoned.
MAX_HALVINGS = 30

#: IRLS iterations allowed before the fit is declared failed. Generous, because
#: it should never bind: at :data:`DEFAULT_RTOL` the fit converges in single
#: digits. It is a runaway guard, not a budget.
DEFAULT_MAX_ITER = 100

#: Floor on the fitted mean, to keep ``y/mu`` finite if a step overshoots.
#: Must be representable in the working dtype -- ``1e-300`` is a float64
#: constant and flushes to zero in float32, which turns the guard into the
#: division by zero it exists to prevent.
_MU_FLOOR = 1e-30

#: Relative convergence tolerance -- see ``WilsonNormaliser._irls`` for why
#: it is relative to the improvement so far rather than to the objective.
#:
#: This is a normalisation curve, not a refined parameter. The quantity it
#: decides is ``E = F / sqrt(Sigma)``, which is then compared against a model
#: that is wrong by tens of percent, so four digits is already far past what
#: anything downstream can use.
DEFAULT_RTOL = 1e-4


class WilsonNormaliser:
    """``Sigma(s)`` by Gamma GLM, so that ``<E^2> = 1`` by construction.

    The model is ``<I_h> = eps_h * Sigma(s_h)`` with ``log Sigma`` a Chebyshev
    polynomial in ``sin(theta)/lambda``, fitted by maximum likelihood under

        acentric  I ~ Exp(Sigma)          (Gamma, shape 1)
        centric   I ~ Sigma * chi^2_1     (Gamma, shape 1/2)

    i.e. a Gamma GLM with a log link and the shape as the prior weight. Unit mean is
    an identity of the fit, not a rescaling: the constant column's score equation,
    ``sum_h k_h (I_h/mu_h - 1) = 0``, is ``<E^2> = 1`` in the shape-weighted sense.

    Parameters
    ----------
    I : torch.Tensor
        ``(N,)`` intensities, **not amplitudes**. Negative values are kept: they are
        excluded from the fit (the Gamma likelihood has no support there) but still
        receive a ``Sigma`` and a signed ``E_squared``.
    s_mag : torch.Tensor
        ``(N,)`` scattering-vector magnitude ``|s| = 1/d``, in Å⁻¹.
    eps : torch.Tensor, optional
        ``(N,)`` reflection multiplicity, divided out before the fit. ``None`` means 1,
        which is right for a molecular transform sampled in a P1 box.
    centric : torch.Tensor, optional
        ``(N,)`` bool, setting the Gamma shape. ``None`` means all acentric.
    n_coeff : int, optional
        Chebyshev terms. ``1`` gives a single global scale.
    s_lo, s_hi : float, optional
        ``|s|`` range mapped onto the basis, in Å⁻¹; defaults to this dataset's own
        extremes. The basis saturates at the ends, so pass both whenever the curve
        will be evaluated outside the fitted range: beyond it the curve is flat.
    fit_mask : torch.Tensor, optional
        ``(N,)`` bool selecting which reflections inform the fit; every reflection
        still receives a ``Sigma``. :meth:`from_hkl` uses it to hold out absences.
    max_iter : int, optional
        IRLS iterations before the fit raises; a runaway guard, not a budget.
    rtol : float, optional
        Convergence when the last step's gain in the objective is below ``rtol``
        times the total gain from the constant-curve start.

    Attributes
    ----------
    coefficients : torch.Tensor
        ``(n_coeff,)`` fitted Chebyshev coefficients of ``log Sigma``.
    sigma_wilson : torch.Tensor
        ``(N,)`` fitted ``Sigma(s)``.
    mean_intensity : torch.Tensor
        ``(N,)`` ``eps * Sigma(s)``, the expected intensity of each reflection.
    E_squared : torch.Tensor
        ``(N,)`` ``I / mean_intensity``. **Signed**: negative observations stay
        negative.
    E : torch.Tensor
        ``(N,)`` ``sqrt(max(E_squared, 0))``.

    Raises
    ------
    ValueError
        If ``I`` is not 1-D, ``s_mag`` does not match it, or fewer than
        ``n_coeff + 1`` reflections are usable.
    RuntimeError
        If the IRLS fit diverges or does not converge in ``max_iter`` iterations.
    """

    MAX_HALVINGS = MAX_HALVINGS

    def __init__(
        self,
        I: torch.Tensor,
        s_mag: torch.Tensor,
        *,
        eps: Optional[torch.Tensor] = None,
        centric: Optional[torch.Tensor] = None,
        n_coeff: int = DEFAULT_N_COEFF,
        s_lo: Optional[float] = None,
        s_hi: Optional[float] = None,
        fit_mask: Optional[torch.Tensor] = None,
        max_iter: int = DEFAULT_MAX_ITER,
        rtol: float = DEFAULT_RTOL,
    ) -> None:
        if I.ndim != 1:
            raise ValueError(f"I must be 1-D, got {tuple(I.shape)}")
        if s_mag.shape != I.shape:
            raise ValueError(
                f"s_mag {tuple(s_mag.shape)} does not match I {tuple(I.shape)}"
            )
        self.dtype = I.dtype
        self.n_coeff = int(n_coeff)
        self._I = I
        self._s_mag = s_mag
        self._eps = eps
        self.s_lo = float(s_mag.min()) if s_lo is None else float(s_lo)
        self.s_hi = float(s_mag.max()) if s_hi is None else float(s_hi)

        # The configured float dtype, not float64. This is a six-coefficient
        # fit of a smooth curve whose answer is compared against a model wrong
        # by tens of percent; it does not need double, and hardcoding it here
        # would be the only double-precision path in the scaling package.
        work = get_float_dtype()
        eps_w = (
            torch.ones_like(I, dtype=work) if eps is None
            else eps.to(work).clamp(min=1.0)
        )
        # Shape 1 acentric (exponential), 1/2 centric. Enters as the IRLS weight
        # because for a Gamma with shape k the variance is mu^2/k, so the
        # log-link working weight is k itself.
        k = (
            torch.ones_like(I, dtype=work) if centric is None
            else torch.where(centric.to(torch.bool), 0.5, 1.0).to(work)
        )

        I_reduced = I.to(work) / eps_w
        # The Gamma likelihood has no support at or below zero. Absences and
        # negative measurements are held out of the fit and given a Sigma from
        # the curve like everything else -- excluding them from the *estimate*
        # is not the same as refusing to normalise them.
        usable = torch.isfinite(I_reduced) & torch.isfinite(s_mag) & (I_reduced > 0)
        if fit_mask is not None:
            usable = usable & fit_mask.to(torch.bool)
        if int(usable.sum()) < self.n_coeff + 1:
            raise ValueError(
                f"only {int(usable.sum())} usable reflections for a "
                f"{self.n_coeff}-coefficient fit; need at least {self.n_coeff + 1}"
            )
        self.n_fitted = int(usable.sum())

        design = chebyshev_design(
            (s_mag * 0.5).to(work), self.n_coeff,
            lo=self.s_lo * 0.5, hi=self.s_hi * 0.5,
        )
        self.coefficients, self.n_iter = self._irls(
            design[usable], I_reduced[usable], k[usable], max_iter, rtol,
        )

        log_sigma = self._eval_log_sigma(design)
        self.sigma_wilson = torch.exp(log_sigma).to(self.dtype)
        self.mean_intensity = (
            torch.exp(log_sigma) * eps_w
        ).clamp(min=1e-30).to(self.dtype)
        self.E_squared = I / self.mean_intensity
        self.E = self.E_squared.clamp(min=0.0).sqrt()

    # -- fitting -----------------------------------------------------------

    def _eval_log_sigma(self, design: torch.Tensor) -> torch.Tensor:
        c = self.coefficients
        return (design @ c).clamp(
            min=-LOG_CLAMP + float(c[0]), max=LOG_CLAMP + float(c[0]),
        )

    @staticmethod
    def _solve_intercept(
        beta: torch.Tensor, X: torch.Tensor, y: torch.Tensor, w: torch.Tensor,
    ) -> torch.Tensor:
        """Put the intercept exactly on its score equation, in closed form.

        Shifting ``beta[0]`` by ``d`` scales every ``mu`` by ``e^d``, so
        ``e^d = sum_h k_h (I_h/mu_h) / sum_h k_h`` satisfies
        ``sum_h k_h (I_h/mu_h - 1) = 0``, i.e. ``<E^2> = 1``, independently of how
        tightly the shape converged. Only the level changes, not the shape.
        """
        eta = X @ beta
        mu = torch.exp(eta.clamp(min=-LOG_CLAMP + float(beta[0]),
                                 max=LOG_CLAMP + float(beta[0]))).clamp(min=_MU_FLOOR)
        ratio = ((w * (y / mu)).sum() / w.sum()).clamp(min=_MU_FLOOR)
        out = beta.clone()
        out[0] = out[0] + torch.log(ratio)
        return out

    def _irls(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        max_iter: int,
        rtol: float,
    ) -> Tuple[torch.Tensor, int]:
        """Gamma GLM with a log link, by iteratively reweighted least squares.

        For this family and link the working weight is the shape ``k`` and does not
        depend on ``mu``, so each step is one weighted least-squares solve against a
        fixed matrix. Convergence and step halving both use the objective
        ``L = sum_h k_h (y_h/mu_h + log mu_h)``, the negative log-likelihood without
        its ``beta``-free terms: the coefficients wander in flat directions when the
        data cover part of the basis range, and the deviance's ``-log(y/mu)`` term
        diverges at near-zero intensities. A step that does not lower ``L`` is halved,
        which guards against overshooting into ``mu`` underflow.
        """
        # Seed at the constant curve, which is the exact MLE when Sigma has no
        # resolution dependence. Every later iteration only adds shape.
        beta = torch.zeros(self.n_coeff, dtype=X.dtype, device=X.device)
        beta[0] = torch.log(((w * y).sum() / w.sum()).clamp(min=1e-30))

        def objective(b):
            eta = (X @ b).clamp(
                min=-LOG_CLAMP + float(b[0]), max=LOG_CLAMP + float(b[0]),
            )
            mu = torch.exp(eta).clamp(min=_MU_FLOOR)
            return float((w * (y / mu + eta)).sum()), eta, mu

        L, eta, mu = objective(beta)
        L0 = L                       # the constant-curve seed, for the ratio below

        # Built and factorised ONCE. For a Gamma with a log link the IRLS
        # working weight is the shape k, which does not depend on mu -- so
        # `X^T W X` is the same matrix at every iteration and only the working
        # response changes. Rebuilding it per iteration costs an O(N n^2) pass
        # over every reflection for an answer that cannot have changed.
        XtW = X.transpose(0, 1) * w.unsqueeze(0)
        A = XtW @ X
        # Ridge proportional to the matrix's own scale: the high-order
        # Chebyshev columns go near-singular when the data cover only part
        # of the basis range.
        A = A + torch.eye(self.n_coeff, dtype=A.dtype, device=A.device) * (
            1e-10 * float(torch.diagonal(A).abs().max().clamp(min=1e-30))
        )
        # Cholesky, not LU: A is SPD by construction, and MPS has neither lu_solve
        # nor cholesky_solve, so two triangular solves keep the loop on the device.
        #
        # `cholesky_ex` reports rather than raises, because a fully collinear
        # basis is a thing this fit sees: the high-order Chebyshev columns go
        # near-singular when the data cover only part of the basis range, and
        # the ridge does not always rescue that; that case falls back to a
        # general solve, which needs no definiteness, instead of failing.
        chol = torch.linalg.cholesky_ex(A)
        L_A = chol.L if int(chol.info) == 0 else None

        def _solve(rhs):
            if L_A is None:
                return torch.linalg.solve(A, rhs)
            return torch.linalg.solve_triangular(
                L_A.mT,
                torch.linalg.solve_triangular(L_A, rhs, upper=False),
                upper=True,
            )

        for it in range(1, max_iter + 1):
            z = eta + (y - mu) / mu                      # working response
            step = _solve((XtW @ z).unsqueeze(-1)).squeeze(-1) - beta
            if not torch.isfinite(step).all():
                raise RuntimeError(
                    f"Wilson fit diverged at iteration {it}: the IRLS solve "
                    f"returned non-finite coefficients."
                )

            # Halve until the step actually improves the objective.
            accepted = False
            for _ in range(self.MAX_HALVINGS):
                L_try, eta_try, mu_try = objective(beta + step)
                if L_try <= L:
                    beta = beta + step
                    accepted = True
                    break
                step = step * 0.5
            if not accepted:
                # No downhill direction left: already at the optimum.
                return self._solve_intercept(beta, X, y, w), it

            # Relative to the improvement achieved so far, not to |L|.
            #
            # |dL|/|L| is not usable here: under I -> cI the optimum is just
            # beta[0] -> beta[0] + log c, so the fit is exactly scale invariant,
            # but L picks up an additive `log c * sum(k)` and the ratio would
            # mean something different at every scale. That additive term
            # cancels in any DIFFERENCE, so a ratio of two differences is both
            # relative and scale invariant -- which is what this is.
            #
            # The denominator is the total distance travelled from the constant
            # seed, so the test reads "the last step moved us less than rtol of
            # the way we have come". It is bounded below so a fit that starts at
            # its own optimum (Sigma genuinely flat) terminates rather than
            # dividing by zero.
            step_gain = abs(L - L_try)
            total_gain = max(abs(L0 - L_try), 1e-30)
            L, eta, mu = L_try, eta_try, mu_try
            if step_gain <= rtol * total_gain:
                return self._solve_intercept(beta, X, y, w), it
        raise RuntimeError(
            f"Wilson fit did not converge in {max_iter} IRLS iterations "
            f"(last step still worth {step_gain / total_gain:.2e} of the total "
            f"improvement, against rtol={rtol:.0e}). Raising rather than "
            f"falling back to a coarser estimate: a normaliser that silently "
            f"becomes a different normaliser on hard cases is two normalisers "
            f"wearing one name."
        )

    # -- evaluation elsewhere ---------------------------------------------

    def evaluate(self, s_mag: torch.Tensor) -> torch.Tensor:
        """``Sigma(s)`` at arbitrary ``|s|``, on the basis this fit was built on.

        The curve is smooth, so it can be fitted on one reflection set and used
        on another -- which is what makes a fit on the crystal lattice usable on
        a dense sampling of the same transform. **Only inside ``[s_lo, s_hi]``**:
        the basis saturates at the ends, so outside that range this returns the
        endpoint value, flat, rather than an extrapolation.
        """
        design = chebyshev_design(
            (s_mag * 0.5).to(self.coefficients.dtype), self.n_coeff,
            lo=self.s_lo * 0.5, hi=self.s_hi * 0.5,
        )
        return torch.exp(self._eval_log_sigma(design)).to(self.dtype)

    # -- construction from crystallography --------------------------------

    @classmethod
    def from_hkl(
        cls,
        I: torch.Tensor,
        hkl: torch.Tensor,
        spacegroup,
        cell,
        **kwargs,
    ) -> "WilsonNormaliser":
        """Build from Miller indices, deriving ``|s|``, ``eps`` and centricity.

        The core takes plain tensors because not every caller has crystal
        reflections -- a molecular transform sampled in a P1 box has no ``hkl``
        at all, and there ``eps`` is 1 with nothing centric. This constructor is
        for the case that does.

        ``epsilon(friedel=False)``: Wilson's ``<I> = eps * Sigma`` counts the
        operations mapping ``h -> h``, which add coherently and set the mean.
        The Friedel-folded count changes the *distribution* instead, and that is
        centricity -- which enters here as the Gamma shape, separately. The two
        branches feed two different parameters of the same likelihood.
        """
        work = get_float_dtype()
        hkl_l = hkl.to(get_int_dtype())
        # The cell may carry the configured default device while the reflections
        # are somewhere else; the caller should not have to reconcile them.
        rec = cell.reciprocal_basis_matrix.to(device=hkl_l.device, dtype=work)
        s_mag = (hkl_l.to(work) @ rec).norm(dim=-1).to(I.dtype)
        eps = spacegroup.epsilon(hkl_l, friedel=False).to(work)
        centric = spacegroup.is_centric(hkl_l).to(torch.bool)
        # Systematically absent reflections are zero by symmetry, not by
        # measurement, so they carry no information about Sigma and would drag
        # the Gamma fit toward zero.
        fit_mask = ~spacegroup.is_absent(hkl_l).to(torch.bool)
        user_mask = kwargs.pop("fit_mask", None)
        if user_mask is not None:
            fit_mask = fit_mask & user_mask.to(torch.bool)
        return cls(
            I, s_mag, eps=eps, centric=centric, fit_mask=fit_mask, **kwargs,
        )

    def __repr__(self) -> str:                    # pragma: no cover - display
        return (
            f"{type(self).__name__}(N={self._I.numel()}, "
            f"n_coeff={self.n_coeff}, n_fitted={self.n_fitted}, "
            f"iters={self.n_iter})"
        )


#: Average protein residue, as xtriage assumes when no composition is given.
#: Only the relative amounts matter: the absolute scale goes into ``K``.
PROTEIN_RESIDUE = {"H": 8.0, "C": 5.0, "N": 1.5, "O": 1.2}

#: Without the protein correction the plot is only linear at high resolution;
#: fit reflections from this d-spacing (Å) outward only.
_PLAIN_D_MAX = 4.5


def sum_f_squared(
    d_star_sq: torch.Tensor, composition: Optional[Dict[str, float]] = None
) -> torch.Tensor:
    """``sum_j n_j f_j(d*^2)^2`` for a composition, with ITC92 form factors.

    Parameters
    ----------
    d_star_sq : torch.Tensor
        ``1/d^2`` in Å^-2, shape (N,).
    composition : dict, optional
        Element symbol to count. Defaults to :data:`PROTEIN_RESIDUE`.

    Returns
    -------
    torch.Tensor
        Shape (N,), in electrons squared, dtype of ``d_star_sq``.
    """
    from torchref.base.direct_summation import compute_scattering_factors_batch
    from torchref.base.scattering.scattering_table import (
        elements_to_z,
        get_scattering_params_by_z,
    )

    composition = PROTEIN_RESIDUE if composition is None else composition
    elements = list(composition)
    counts = torch.tensor([composition[e] for e in elements]).to(d_star_sq)
    A, B = get_scattering_params_by_z(
        elements_to_z(elements).to(d_star_sq.device), dtype=d_star_sq.dtype
    )
    f = compute_scattering_factors_batch(d_star_sq.sqrt(), A, B)
    return (f**2 * counts[None]).sum(-1)


@dataclass
class WilsonFit:
    """Result of :func:`fit_wilson_b`.

    The model is ``<I/eps> = K * sum_f2(d*^2) * (1 + gamma(d*^2)) * exp(-B d*^2 / 2)``,
    with ``gamma`` the empirical protein correction (zero when
    ``protein_gamma`` is False).

    Attributes
    ----------
    B : float
        Wilson B in Å².
    sigma_B : float
        Standard error of ``B`` from the scatter of the shell means about the
        line; it reflects how straight the plot is, not the measurement error.
    log_scale : float
        ``ln K``.
    d_max, d_min : float
        Resolution range of the fitted reflections, Å.
    n_reflections, n_shells : int
        Reflections and equal-count shells in the fit.
    composition : dict
        Composition used for ``sum_f2``.
    protein_gamma : bool
        Whether the protein correction was applied.
    """

    B: float
    sigma_B: float
    log_scale: float
    d_max: float
    d_min: float
    n_reflections: int
    n_shells: int
    composition: Dict[str, float]
    protein_gamma: bool

    def shape(self, d: torch.Tensor) -> torch.Tensor:
        """``sum_f2 * (1 + gamma) * exp(-B d*^2 / 2)`` at resolution ``d`` (Å); no K."""
        d_star_sq = d.pow(-2)
        curve = sum_f_squared(d_star_sq, self.composition)
        if self.protein_gamma:
            curve = curve * (1.0 + _protein_gamma(d_star_sq))
        return curve * torch.exp(-0.5 * self.B * d_star_sq)

    def expected_intensity(
        self, d: torch.Tensor, epsilon: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Expected mean intensity at resolution ``d`` (Å), times ``epsilon``."""
        out = math.exp(self.log_scale) * self.shape(d)
        return out if epsilon is None else out * epsilon.to(out)


def fit_wilson_b(
    I: torch.Tensor,
    d: torch.Tensor,
    *,
    sigma: Optional[torch.Tensor] = None,
    epsilon: Optional[torch.Tensor] = None,
    amplitudes: bool = False,
    composition: Optional[Dict[str, float]] = None,
    protein_gamma: bool = True,
    n_shells: int = 20,
    min_per_shell: int = 20,
    verbose: int = 0,
) -> Optional[WilsonFit]:
    """Fit the overall Wilson B, as xtriage does, by a line through shell means.

    Fits ``ln <I / (eps sum_f2 (1 + gamma))>`` against ``d*^2`` in shells of
    equal reflection count; the slope is ``-B/2``. ``sum_f2`` is the random-atom
    mean intensity of ``composition``, ``gamma`` the empirical protein
    correction of Zwart & Lamzin (2004), which keeps the plot linear from about
    11 Å to 1.2 Å. Protein-specific: set ``protein_gamma=False`` otherwise.
    Values run below ctruncate-style Wilson B; see the scaling user guide. For
    normalising intensities use :class:`WilsonNormaliser`.

    Parameters
    ----------
    I : torch.Tensor
        Intensities of shape (N,), or amplitudes with ``amplitudes=True``.
        Pass only the reflections to use (e.g. ``data.masks()`` applied).
    d : torch.Tensor
        Resolution of each reflection in Å, shape (N,).
    sigma : torch.Tensor, optional
        Uncertainties of ``I`` (N,). Used only with ``amplitudes=True``, where
        the intensity proxy is ``F^2 + sigma_F^2``: a French-Wilson ``F^2`` alone
        underestimates the mean intensity of weak reflections.
    epsilon : torch.Tensor, optional
        Reflection multiplicity (N,) from ``SpaceGroup.epsilon``; intensities
        are divided by it.
    amplitudes : bool, optional
        ``I`` holds amplitudes.
    composition : dict, optional
        Element symbol to count; defaults to :data:`PROTEIN_RESIDUE`. The
        result barely depends on it for proteins.
    protein_gamma : bool, optional
        Apply the protein correction (default True). Without it only
        reflections with d <= 4.5 Å are fitted; turn it off for nucleic acids.
    n_shells : int, optional
        Number of equal-count shells, default 20, reduced so each holds at
        least ``min_per_shell`` reflections.
    min_per_shell : int, optional
        Default 20.
    verbose : int, optional
        Print the result when > 0.

    Returns
    -------
    WilsonFit or None
        ``None``, with a warning saying why, when the data cannot support a
        fit: fewer than three shells, a ``d*^2`` range under 0.03 Å^-2, or a
        non-positive shell mean. There is no fallback value and ``B`` is not
        clamped -- an implausible B means implausible data.
    """
    from torchref.scaling._protein_gamma import D_STAR_SQ_HIGH, D_STAR_SQ_LOW

    I = I.detach().reshape(-1)
    d = d.detach().to(I).reshape(-1)
    if amplitudes:
        y = I**2 if sigma is None else I**2 + sigma.detach().to(I) ** 2
    else:
        y = I
    if epsilon is not None:
        y = y / epsilon.detach().to(I)
    d_star_sq = d.pow(-2)
    keep = torch.isfinite(y) & torch.isfinite(d_star_sq)
    if protein_gamma:
        keep &= (d_star_sq > D_STAR_SQ_LOW) & (d_star_sq < D_STAR_SQ_HIGH)
    else:
        keep &= d <= _PLAIN_D_MAX
    y, d_star_sq = y[keep], d_star_sq[keep]

    def _give_up(reason: str) -> None:
        warnings.warn(f"No Wilson B: {reason}.", stacklevel=3)
        return None

    n_shells = min(n_shells, len(y) // min_per_shell)
    if n_shells < 3:
        return _give_up(f"{len(y)} usable reflections, need {3 * min_per_shell}")
    span = float(d_star_sq.max() - d_star_sq.min())
    if span < 0.03:
        return _give_up(f"d*^2 range {span:.3f} Å^-2 is too short for a slope")

    y = y / sum_f_squared(d_star_sq, composition)
    if protein_gamma:
        y = y / (1.0 + _protein_gamma(d_star_sq))

    order = torch.argsort(d_star_sq)
    shells = torch.tensor_split(order, n_shells)
    x = torch.stack([d_star_sq[s].mean() for s in shells])
    mean_y = torch.stack([y[s].mean() for s in shells])
    if bool((mean_y <= 0).any()):
        return _give_up("a resolution shell has non-positive mean intensity")
    ln_y = torch.log(mean_y)

    # Centred regression keeps the float32 normal equations well conditioned.
    x_c, y_c = x - x.mean(), ln_y - ln_y.mean()
    sxx = (x_c**2).sum()
    slope = (x_c * y_c).sum() / sxx
    resid = y_c - slope * x_c
    sigma_slope = torch.sqrt((resid**2).sum() / (n_shells - 2) / sxx)
    fit = WilsonFit(
        B=float(-2.0 * slope),
        sigma_B=float(2.0 * sigma_slope),
        log_scale=float(ln_y.mean() - slope * x.mean()),
        d_max=float(d_star_sq.min().rsqrt()),
        d_min=float(d_star_sq.max().rsqrt()),
        n_reflections=len(y),
        n_shells=n_shells,
        composition=dict(PROTEIN_RESIDUE if composition is None else composition),
        protein_gamma=protein_gamma,
    )
    if verbose > 0:
        print(
            f"  Wilson B: {fit.B:.1f} ± {fit.sigma_B:.1f} Å² "
            f"({fit.d_max:.2f}-{fit.d_min:.2f} Å, {fit.n_reflections} reflections)"
        )
    return fit
