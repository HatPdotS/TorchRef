"""
French-Wilson conversion of a dataset's intensities, with its symmetry.

:func:`french_wilson_auto` is the entry point every intensity load goes through.
It does the space-group bookkeeping the conversion needs and hands the rest to
the tensor math in :mod:`torchref.base.french_wilson`, which stays free of the
symmetry layer:

- centric flags, which give centric rows their own variance in the prior and
  their own posterior;
- systematic absences, which never inform the prior: their intensity is zero by
  symmetry, not a sample of the Wilson distribution;
- the multiplicity ``epsilon``, the factor by which a reflection on a symmetry
  element is stronger;
- the anisotropy the Laue class allows. Without it the prior is fitted to the
  acentric rows only: a centric zone is a single plane of reciprocal space, and
  on anisotropic data its mean intensity is that of one direction, not of the
  shell.
"""

from functools import partial

import torch

from torchref.base.french_wilson import (
    DEFAULT_MIN_I_OVER_SIGMA,
    DEFAULT_N_COEFF,
    _solve_normal,
    fit_mean_intensity,
    french_wilson,
)
from torchref.symmetry import SpaceGroup, SpaceGroupLike

__all__ = ["french_wilson_auto"]


def _anisotropy_design(
    hkl: torch.Tensor,
    space_group: SpaceGroupLike,
    radial: torch.Tensor,
    s2: torch.Tensor,
    rows: torch.Tensor,
) -> torch.Tensor:
    """Direction-only quadratic forms in the Miller indices the Laue group allows.

    An ellipsoidal fall-off ``exp(-2 pi^2 s^T U s)`` is ``exp(h^T M h)`` with
    the cell folded into ``M``, so it needs only ``hkl``. Averaging ``h_i h_j``
    over the symmetry copies of each reflection leaves the forms the Laue group
    allows. Each then loses, by least squares over ``rows``, whatever the
    radial curve or ``s^2`` itself can represent -- ``s^2 = h^T G* h`` is the
    isotropic form, which the radial curve does not reproduce exactly at low
    resolution. What is left is direction only, and nothing in the fit
    duplicates the radial columns. The result is orthonormal over ``rows``.

    How many directions are left is fixed by symmetry, not by the data: it is
    the dimension of the Laue-invariant quadratic forms, the trace of the
    averaging operator over the group's rotations, less the isotropic one -- 5
    for triclinic down to 0 for cubic. Reading it off the data instead would
    let rounding in a nearly isotropic direction pass for anisotropy.

    Parameters
    ----------
    hkl : torch.Tensor
        Miller indices, shape (n, 3).
    space_group : SpaceGroupLike
        The data's space group.
    radial : torch.Tensor
        The radial basis evaluated at every row, shape (n, m).
    s2 : torch.Tensor
        ``1/d^2`` in Å⁻², shape (n,), zero where ``d`` is not finite.
    rows : torch.Tensor
        Boolean mask of shape (n,) of the rows the fit uses.

    Returns
    -------
    torch.Tensor
        Shape (n, r) with ``r`` between 0 and 5.
    """
    dtype = radial.dtype
    group = SpaceGroup(space_group, device=hkl.device)
    copies, _, _, _ = group.equivalent_hkl(
        hkl, include_friedel=False, device=hkl.device
    )

    # Rotations acting on h, recovered as the images of the unit vectors. The
    # averaging operator maps M to mean(R M R^T); in the coordinates
    # (M00, M11, M22, M01, M02, M12) its trace counts the invariant forms.
    unit = torch.eye(3, dtype=hkl.dtype, device=hkl.device)
    images, _, _, _ = group.equivalent_hkl(
        unit, include_friedel=False, device=hkl.device
    )
    R = images.to(dtype).cpu().reshape(-1, 3, 3)
    trace = 0.0
    for i, j in [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]:
        E = torch.zeros(3, 3, dtype=dtype)
        E[i, j] = E[j, i] = 1.0
        trace += float((R @ E @ R.mT).mean(0)[i, j])
    n_directions = round(trace) - 1
    if n_directions < 1:
        return torch.zeros(hkl.shape[0], 0, dtype=dtype, device=hkl.device)
    n_ops = copies.shape[0] // hkl.shape[0]
    h = copies.to(dtype).reshape(n_ops, hkl.shape[0], 3)
    forms = torch.stack(
        [
            h[..., 0] * h[..., 0],
            h[..., 1] * h[..., 1],
            h[..., 2] * h[..., 2],
            2.0 * h[..., 0] * h[..., 1],
            2.0 * h[..., 0] * h[..., 2],
            2.0 * h[..., 1] * h[..., 2],
        ],
        dim=-1,
    ).mean(0)
    # Unit scale before any sums: raw h_i h_j reach 1e4, and their squares
    # summed over 1e5 rows lose the digits that separate the anisotropic part
    # from the isotropic one in float32.
    forms = forms / forms[rows].pow(2).mean(0).sqrt().clamp(min=1e-30)

    # Two projections in turn rather than one onto [radial, s^2]: s^2 is nearly
    # a radial function, and the joint normal equations lose in float32 the
    # very difference that is being kept.
    on_rows = radial[rows]
    gram = on_rows.T @ on_rows

    def off_radial(columns):
        coefficients = torch.stack(
            [_solve_normal(gram, on_rows.T @ col[rows]) for col in columns.T], dim=1
        )
        return columns - radial @ coefficients

    residual = off_radial(forms)
    s2_off = off_radial((s2 / s2[rows].pow(2).mean().sqrt()).unsqueeze(1))[:, 0]
    w = s2_off[rows]
    residual = residual - s2_off.unsqueeze(1) * (
        (residual[rows].T @ w) / (w @ w).clamp(min=1e-30)
    )

    # A 6x6 eigenproblem, solved on the host: it is tiny, and MPS has no eigh.
    # The leading directions are the anisotropic ones; what follows them is
    # what the symmetry or the projection removed, down to rounding.
    fitted = residual[rows]
    values, vectors = torch.linalg.eigh((fitted.T @ fitted / fitted.shape[0]).cpu())
    values, vectors = values[-n_directions:], vectors[:, -n_directions:]
    basis = (vectors / values.clamp(min=1e-30).sqrt()).to(device=hkl.device)
    return residual @ basis


def french_wilson_auto(
    I: torch.Tensor,
    sigma_I: torch.Tensor,
    hkl: torch.Tensor,
    d_spacings: torch.Tensor,
    space_group: SpaceGroupLike = "P1",
    min_i_over_sigma: float = DEFAULT_MIN_I_OVER_SIGMA,
    *,
    n_coeff: int = DEFAULT_N_COEFF,
    exclude_from_fit: torch.Tensor | None = None,
    anisotropic: bool = True,
    epsilon: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert intensities to amplitudes, fitting the prior from the data.

    Every per-reflection input must be row-aligned: the prior and centric flag
    of row ``i`` are taken from ``hkl[i]`` and ``d_spacings[i]``. The prior is
    :func:`~torchref.base.french_wilson.fit_mean_intensity`, informed by neither
    systematic absences nor, when ``anisotropic`` is False, centric rows.

    Parameters
    ----------
    I : torch.Tensor
        Measured intensities of shape (n_reflections,).
    sigma_I : torch.Tensor
        Standard deviations of intensities of shape (n_reflections,).
    hkl : torch.Tensor
        Miller indices of shape (n_reflections, 3).
    d_spacings : torch.Tensor
        Resolution (d-spacing) in Å for each reflection of shape
        (n_reflections,).
    space_group : str, int, or gemmi.SpaceGroup, optional
        Space group specification. Default is "P1".
    min_i_over_sigma : float, optional
        Lower cut on ``I/sigma_I``, as for
        :func:`~torchref.base.french_wilson.french_wilson_valid_mask`. Default -3.7.
    n_coeff : int, optional
        B-spline coefficients in ``log Sigma``, as for
        :func:`~torchref.base.french_wilson.fit_mean_intensity`.
    exclude_from_fit : torch.Tensor, optional
        Boolean mask of shape (n_reflections,) of rows that must not inform the
        prior -- the free (test) set, so that nothing fitted has seen it. They
        are still given a prior from the curve and converted like every other
        row. Ignored if it would leave no row to fit.
    anisotropic : bool, optional
        Fit an ellipsoidal anisotropy into the prior. Default True.
    epsilon : bool, optional
        Give each reflection the expected intensity ``epsilon Sigma``, with
        ``epsilon`` counted from ``space_group``, which must therefore be the
        crystal's true symmetry: a reflection on a symmetry element is that
        much stronger whatever group the data were merged in. Default True.
        Absences take the general value.

    Returns
    -------
    F : torch.Tensor
        Structure factor amplitudes of shape (n_reflections,).
    sigma_F : torch.Tensor
        Standard deviations of F of shape (n_reflections,).
    valid_mask : torch.Tensor
        Boolean mask, ``True`` = keep. ``False`` both for rows too negative
        for their own sigma and for rows with NaN ``I``, a NaN or non-positive
        ``sigma_I`` or a non-finite ``d``, whose ``F`` and ``sigma_F`` are NaN.
    """
    F = torch.full_like(I, float("nan"))
    sigma_F = torch.full_like(sigma_I, float("nan"))
    # NaN rows are never converted, so they are not kept either.
    valid_mask = torch.zeros_like(I, dtype=torch.bool)

    finite = ~(torch.isnan(I) | torch.isnan(sigma_I))
    if not finite.any():
        return F, sigma_F, valid_mask
    hkl_f = hkl[finite]

    group = SpaceGroup(space_group, device=hkl.device)
    is_centric = group.is_centric(hkl_f)
    absent = group.is_absent(hkl_f)

    fit_mask = ~absent
    if not anisotropic and bool((~is_centric).any()):
        fit_mask = fit_mask & ~is_centric
    if exclude_from_fit is not None:
        held_out = exclude_from_fit.to(device=I.device, dtype=torch.bool)[finite]
        fit_mask = fit_mask & ~held_out
    # A test set drawn on a tiny dataset can cover every row; with nothing left
    # to fit there would be no prior at all.
    if not bool(fit_mask.any()):
        fit_mask = torch.ones_like(fit_mask)

    multiplicity = None
    if epsilon:
        # Operations whose rotation is the identity are the lattice centrings;
        # their count is epsilon for a general reflection, and for an absence.
        eye = torch.eye(3, dtype=group.matrices.dtype, device=group.matrices.device)
        centrings = (group.matrices - eye).abs().amax(dim=(1, 2)) < 1e-6
        multiplicity = group.epsilon(hkl_f, friedel=False).to(I.dtype)
        multiplicity = torch.where(absent, float(centrings.sum()), multiplicity)

    mean_intensity = fit_mean_intensity(
        I[finite],
        sigma_I[finite],
        d_spacings[finite],
        fit_mask=fit_mask,
        n_coeff=n_coeff,
        anisotropy=partial(_anisotropy_design, hkl_f, group) if anisotropic else None,
        is_centric=is_centric,
        epsilon=multiplicity,
    )

    F[finite], sigma_F[finite], valid_mask[finite] = french_wilson(
        I[finite],
        sigma_I[finite],
        mean_intensity,
        is_centric=is_centric,
        min_i_over_sigma=min_i_over_sigma,
    )
    return F, sigma_F, valid_mask
