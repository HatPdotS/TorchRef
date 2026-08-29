"""Absolute Wilson normalisation: fit ``Sigma(s)`` and divide it out.

Distinct from :class:`~torchref.scaling.scaler_base.ScalerBase`, which is a
*relative* scaler -- it puts ``F_calc`` onto ``F_obs`` and every target it can
minimise compares the two. This one takes a single dataset and answers "what is
the expected intensity at this resolution", so that dividing by it leaves
``<E^2> = 1``. One dataset in, one curve out, no second dataset anywhere in the
objective.

**Why this exists as one shared class.** The repo grew at least five private
answers to the same question -- ``base/wilson_outliers.robust_mean_intensity``,
``base/french_wilson.estimate_mean_intensity_by_resolution``,
``ReflectionData._calculate_wilson_b``, the ``Sigma_N`` estimator in
``refinement/model_error_estimation/sigma_a``, and a per-shell one inside the
alignment package -- differing in whether they use means or medians, whether
they divide out ``epsilon``, whether they separate centrics, and where they put
their shell edges. Consumers that disagree about what E means cannot be compared
with each other, which is exactly what went wrong between the rotation function
and its own rescore.

**Scaling, not weighting.** This class answers *what* we compare. It says
nothing about how much any reflection should count -- no ``sigI``, no model
error, no solvent. Those belong to a weight, and mixing them in here is what
made the previous convention object impossible to reason about: it returned a
normalisation and a weight together, so sweeping it moved a gauge quantity and a
real one at the same time.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from torchref.scaling.basis import chebyshev_design

__all__ = ["WilsonNormaliser"]

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


class WilsonNormaliser:
    """``Sigma(s)`` by Gamma GLM, so that ``<E^2> = 1`` by construction.

    The model is ``<I_h> = eps_h * Sigma(s_h)`` with ``log Sigma`` a Chebyshev
    polynomial in ``sin(theta)/lambda``, fitted by maximum likelihood under

        acentric  I ~ Exp(Sigma)          (Gamma, shape 1)
        centric   I ~ Sigma * chi^2_1     (Gamma, shape 1/2)

    i.e. a Gamma GLM with a log link and the shape as the prior weight.

    **Unit mean is an identity of the fit, not a normalisation step.** The
    constant basis column's score equation is ``sum_h k_h (I_h/mu_h - 1) = 0``,
    which is exactly ``<E^2> = 1`` in the shape-weighted sense. Nothing is
    rescaled afterwards and nothing can drift -- which is what makes a
    downstream ``E^2 - 1`` a true centring rather than an approximate one.

    Least squares on ``log I`` would be the obvious alternative and is wrong:
    ``E[log Gamma]`` carries a digamma offset, and with a constant term present
    it is absorbed into the curve's shape rather than into the level. That is
    the defect the overall-anisotropy fit was carrying.

    Parameters
    ----------
    I : torch.Tensor
        ``(N,)`` intensities. **Intensities, not amplitudes** -- Wilson
        statistics are exact on I and awkward on F, and measurement error is
        near-Gaussian on I but badly behaved on F for weak reflections, which is
        the whole reason the French-Wilson posterior exists. Negative values are
        allowed and kept: they are meaningful, unbiased measurements. They are
        excluded from the *fit* (the Gamma likelihood has no support there) but
        still receive a ``Sigma`` and a signed ``E_squared``.
    s_mag : torch.Tensor
        ``(N,)`` scattering-vector magnitude ``|s| = 1/d``, in inverse Angstrom.
    eps : torch.Tensor, optional
        ``(N,)`` reflection multiplicity. Divides the intensity before the fit,
        because axial reflections are systematically stronger. ``None`` means 1
        everywhere, which is correct for a molecular transform sampled in a P1
        box -- multiplicity is a property of crystal symmetry and there is none
        there.
    centric : torch.Tensor, optional
        ``(N,)`` bool, setting the Gamma shape. ``None`` means all acentric.
    n_coeff : int, optional
        Chebyshev terms. ``1`` gives a single global scale.
    s_lo, s_hi : float, optional
        ``|s|`` range mapped onto the basis. Defaults to this dataset's own
        extremes. **Pass both explicitly whenever the curve will be evaluated
        outside the fitted data's range** -- comparing two fits over different
        ranges, or fitting on a crystal lattice and evaluating on a dense
        sampling. The basis saturates at the ends, so beyond the fitted range
        the curve is frozen flat rather than extrapolated.
    fit_mask : torch.Tensor, optional
        ``(N,)`` bool selecting which reflections *inform* the fit. Everything
        still receives a ``Sigma``, because the curve is smooth and evaluable
        anywhere. Use it to hold out systematic absences -- see
        :meth:`from_hkl`, which does exactly that.

    Attributes
    ----------
    coefficients : torch.Tensor
        ``(n_coeff,)`` fitted Chebyshev coefficients of ``log Sigma``.
    sigma_wilson : torch.Tensor
        ``(N,)`` fitted ``Sigma(s)``. Deliberately not called ``sigma``: this
        package also carries ``sig_F``, a measurement error, and ``sigma_a``, a
        correlation coefficient, and the three are not interchangeable.
    mean_intensity : torch.Tensor
        ``(N,)`` ``eps * Sigma(s)``, the expected intensity of each reflection.
    E_squared : torch.Tensor
        ``(N,)`` ``I / mean_intensity``. **Signed** -- negative observations stay
        negative.
    E : torch.Tensor
        ``(N,)`` ``sqrt(max(E_squared, 0))``.
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
        max_iter: int = 100,
        tol: float = 1e-10,
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

        eps64 = (
            torch.ones_like(I, dtype=torch.float64) if eps is None
            else eps.to(torch.float64).clamp(min=1.0)
        )
        # Shape 1 acentric (exponential), 1/2 centric. Enters as the IRLS weight
        # because for a Gamma with shape k the variance is mu^2/k, so the
        # log-link working weight is k itself.
        k = (
            torch.ones_like(I, dtype=torch.float64) if centric is None
            else torch.where(centric.to(torch.bool), 0.5, 1.0).to(torch.float64)
        )

        I_reduced = I.to(torch.float64) / eps64
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
            (s_mag * 0.5).to(torch.float64), self.n_coeff,
            lo=self.s_lo * 0.5, hi=self.s_hi * 0.5,
        )
        self.coefficients, self.n_iter = self._irls(
            design[usable], I_reduced[usable], k[usable], max_iter, tol,
        )

        log_sigma = self._eval_log_sigma(design)
        self.sigma_wilson = torch.exp(log_sigma).to(self.dtype)
        self.mean_intensity = (
            torch.exp(log_sigma) * eps64
        ).clamp(min=1e-30).to(self.dtype)
        self.E_squared = I / self.mean_intensity
        self.E = self.E_squared.clamp(min=0.0).sqrt()

    # -- fitting -----------------------------------------------------------

    def _eval_log_sigma(self, design: torch.Tensor) -> torch.Tensor:
        c = self.coefficients
        return (design @ c).clamp(
            min=-LOG_CLAMP + float(c[0]), max=LOG_CLAMP + float(c[0]),
        )

    def _irls(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        max_iter: int,
        tol: float,
    ) -> Tuple[torch.Tensor, int]:
        """Gamma GLM with a log link, by iteratively reweighted least squares.

        IRLS rather than a generic optimiser: for this link and family the
        working weight does not depend on ``mu``, so each step is one weighted
        least-squares solve and there is no step size, no line search and no
        absolute tolerance to fail against an unnormalised objective.

        Convergence and step control both use the objective itself,
        ``L = sum_h k_h (y_h/mu_h + log mu_h)`` -- the negative log-likelihood
        with the terms not involving ``beta`` dropped.

        That choice is forced by what the alternatives do on real data. The
        *coefficients* are underdetermined whenever the data occupy part of the
        basis range, which is the normal case once an explicit ``s_lo``/``s_hi``
        is passed, so they wander in the flat directions long after the fit has
        settled. The *deviance* carries a ``-log(y/mu)`` term that diverges as
        ``y -> 0``, and calculated amplitudes have near-zeros at the nodes of
        the molecular transform, so a few tiny intensities dominate it. And the
        *fitted mean* cannot be compared as a ratio because it is floored, so a
        collapsed fit reads as a converged one -- which is exactly how an early
        version of this reported success while returning zeros.

        ``L`` has none of those problems: the ``log y`` term that breaks the
        deviance is constant in ``beta`` and simply absent here.

        Step halving is the other half. IRLS on a log link can overshoot into
        ``mu`` underflow, after which the working response ``y/mu`` explodes and
        the next step is worse. Rejecting any step that does not improve ``L``
        and halving it is the standard remedy and makes the fit robust to the
        ill-conditioning a partial basis range creates.
        """
        # Seed at the constant curve, which is the exact MLE when Sigma has no
        # resolution dependence. Every later iteration only adds shape.
        beta = torch.zeros(self.n_coeff, dtype=torch.float64, device=X.device)
        beta[0] = torch.log(((w * y).sum() / w.sum()).clamp(min=1e-30))

        def objective(b):
            eta = (X @ b).clamp(
                min=-LOG_CLAMP + float(b[0]), max=LOG_CLAMP + float(b[0]),
            )
            mu = torch.exp(eta).clamp(min=1e-300)
            return float((w * (y / mu + eta)).sum()), eta, mu

        L, eta, mu = objective(beta)
        for it in range(1, max_iter + 1):
            z = eta + (y - mu) / mu                      # working response
            XtW = X.transpose(0, 1) * w.unsqueeze(0)
            A = XtW @ X
            # Ridge proportional to the matrix's own scale: the high-order
            # Chebyshev columns go near-singular when the data cover only part
            # of the basis range.
            A = A + torch.eye(self.n_coeff, dtype=A.dtype, device=A.device) * (
                1e-10 * float(torch.diagonal(A).abs().max().clamp(min=1e-30))
            )
            step = torch.linalg.solve(A, XtW @ z) - beta
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
                return beta, it

            improvement = abs(L - L_try) / (abs(L) + 1e-30)
            L, eta, mu = L_try, eta_try, mu_try
            if improvement <= tol:
                return beta, it
        raise RuntimeError(
            f"Wilson fit did not converge in {max_iter} IRLS iterations "
            f"(objective still moving by {improvement:.2e} relative). Raising "
            f"rather than falling back to a coarser estimate: a normaliser that "
            f"silently becomes a different normaliser on hard cases is two "
            f"normalisers wearing one name."
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
            (s_mag * 0.5).to(torch.float64), self.n_coeff,
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
        hkl_l = hkl.to(torch.long)
        # The cell may carry the configured default device while the reflections
        # are somewhere else; the caller should not have to reconcile them.
        rec = cell.reciprocal_basis_matrix.to(device=hkl_l.device,
                                              dtype=torch.float64)
        s_mag = (hkl_l.to(torch.float64) @ rec).norm(dim=-1).to(I.dtype)
        eps = spacegroup.epsilon(hkl_l, friedel=False).to(torch.float64)
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
