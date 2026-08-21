"""The overall-anisotropy correction: the production fit, and a corrected one.

``sh.py:445 fit_overall_anisotropy`` regresses ``ln|F|^2 - ln<|F|^2>_shell`` on
``-2 pi^2 s.U.s`` by unweighted least squares **with no intercept**, and that is
the FRF's remaining defect (job 489540/489548). Three faults, all visible in its
output:

* ``E[ln(I/<I>)]`` is ``-gamma = -0.577`` for acentric reflections and
  ``-gamma - ln 2 = -1.270`` for centric ones, not zero. With no intercept the
  offset can only be absorbed by the quadratic form, which is why the fitted
  tensor's SMALLEST B eigenvalue is 35-64 A^2 on every benchmark structure
  instead of near zero. The centric part is worse than a constant: centric
  reflections lie on the zones perpendicular to the symmetry axes, so the bias
  is direction-dependent.
* ``clamp(min=1e-30)`` turns a vanishing amplitude into ``y ~ -69``; a handful of
  those outweigh thousands of ordinary reflections in an unweighted fit.
* ``ln`` of a single-reflection intensity has variance ``pi^2/6`` (acentric) or
  ``pi^2/2`` (centric) with a heavy left tail, so the fit is dominated by the
  weak reflections carrying the least information.

Raw fitted B eigenvalue spreads come out at 70 to 5461 A^2.
``symmetrize_anisotropy`` then projects onto the point-group-invariant subspace,
which annihilates the garbage where that subspace is small (cubic -> 1 DOF) and
leaves it where it is not (trigonal/hexagonal -> diag(lambda, lambda, mu), where
a uniaxial tensor along c is symmetry-allowed).

One definition of the replacement lives here rather than in a diagnostic, since
several diagnostics need to A/B against it and it is the candidate production
change.
"""

from __future__ import annotations

import math
from contextlib import contextmanager

import torch

#: U (A^2) -> B (A^2).
B_PER_U = 8.0 * math.pi ** 2

#: Arm names accepted by :func:`aniso_arm`.
ARMS = ("production", "no_aniso", "iso_only", "fixed_fit")


def fit_aniso_intensity_space(
    F_obs: torch.Tensor,
    s_vec: torch.Tensor,
    shell_idx: torch.Tensor,
    centric: torch.Tensor,
    P: int,
    *,
    min_count: int = 20,
    n_iter: int = 12,
) -> torch.Tensor:
    """Unbiased replacement for ``fit_overall_anisotropy``.

    Fits in INTENSITY space, where ``E[I/<I>_shell] = c * exp(-2 pi^2 s.U.s)``
    holds exactly with no distributional correction. ``Var(I/<I>)`` is 1 for
    acentric and 2 for centric reflections, which gives the weights; a free
    constant ``c`` absorbs the overall scale so it cannot leak into ``U``;
    non-finite and non-positive amplitudes are dropped rather than clamped.
    Gauss-Newton from ``U = 0``.

    Returns ``U`` in A^2 in the same convention as ``fit_overall_anisotropy``
    (applied as ``exp(+pi^2 s.U.s)``), so the caller's symmetrisation and
    application are unchanged.
    """
    valid = shell_idx >= 0
    F = F_obs[valid].to(torch.float64)
    s = s_vec[valid].to(torch.float64)
    idx = shell_idx[valid]
    cen = centric[valid].bool()
    ok = torch.isfinite(F) & (F > 0)
    F, s, idx, cen = F[ok], s[ok], idx[ok], cen[ok]

    I = F * F
    cnt = torch.zeros(P, dtype=torch.int64, device=F.device)
    tot = torch.zeros(P, dtype=torch.float64, device=F.device)
    cnt.index_add_(0, idx, torch.ones_like(idx))
    tot.index_add_(0, idx, I)
    mean_I = (tot / cnt.clamp(min=1).to(torch.float64)).clamp(min=1e-30)
    keep = (cnt >= min_count)[idx]
    if int(keep.sum()) < 50:
        return torch.zeros((3, 3), dtype=F_obs.dtype, device=F_obs.device)
    r = I[keep] / mean_I[idx[keep]]
    sk, cenk = s[keep], cen[keep]

    x, y, z = sk[:, 0], sk[:, 1], sk[:, 2]
    quad = torch.stack([x * x, y * y, z * z,
                        2 * x * y, 2 * x * z, 2 * y * z], dim=1)
    A = torch.cat([torch.ones_like(x).unsqueeze(1),
                   -2.0 * (torch.pi ** 2) * quad], dim=1)
    w = torch.where(cenk, torch.full_like(r, 0.5), torch.ones_like(r))

    theta = torch.zeros(7, dtype=torch.float64, device=F.device)
    for _ in range(n_iter):
        m = torch.exp((A @ theta).clamp(min=-20.0, max=20.0))
        J = m.unsqueeze(1) * A
        Jw = J * w.unsqueeze(1)
        H = J.transpose(0, 1) @ Jw
        g = Jw.transpose(0, 1) @ (r - m)
        H = H + torch.eye(7, dtype=H.dtype, device=H.device) * 1e-12 * float(
            torch.diagonal(H).abs().max().clamp(min=1e-30))
        theta = theta + torch.linalg.solve(H, g)
    u = theta[1:]
    return torch.tensor(
        [[u[0], u[3], u[4]], [u[3], u[1], u[5]], [u[4], u[5], u[2]]],
        dtype=F_obs.dtype, device=F_obs.device)


def tensor_report(U: torch.Tensor, tag: str) -> dict:
    """B eigenvalues (A^2) of a U tensor, as result-row columns."""
    ev = torch.linalg.eigvalsh(U.to(torch.float64).cpu()) * B_PER_U
    return {f"{tag}_B_min": round(float(ev[0]), 2),
            f"{tag}_B_max": round(float(ev[2]), 2),
            f"{tag}_B_spread": round(float(ev[2] - ev[0]), 2)}


@contextmanager
def aniso_arm(arm: str, data, *, d_min: float, d_max: float, captured: dict):
    """Swap the anisotropy fit for the duration of one FRF call.

    ``captured`` receives the tensors actually fitted (``raw``, and ``fixed``
    when the arm uses the replacement) so a caller can report the artefact size
    alongside the rank it costs.

    The ``fixed_fit`` arm needs ``centric``, which ``fit_overall_anisotropy``
    is not given. ``_prepare_frf_inputs`` masks ``F_obs`` and ``centric`` with
    the same resolution window, so it is recomputed here from the same
    ``d_min``/``d_max`` and checked against the amplitude count -- a mismatch
    raises rather than silently misaligning.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown aniso arm {arm!r}; expected one of {ARMS}")
    from torchref.experimental.alignment import align as _align

    original = _align.fit_overall_anisotropy
    rec = data.cell.reciprocal_basis_matrix.to(torch.float64)
    smag = (data.hkl.to(torch.float64) @ rec).norm(dim=-1)
    keep = (smag >= 1.0 / d_max) & (smag <= 1.0 / d_min)
    centric = (data.centric[keep].to(torch.bool)
               if hasattr(data, "centric") else None)

    def wrapped(F_obs, s_vec, shell_idx, **kw):
        U = original(F_obs, s_vec, shell_idx, **kw)
        captured.setdefault("raw", U.detach().clone())
        if arm == "production":
            return U
        if arm == "no_aniso":
            return torch.zeros_like(U)
        if arm == "iso_only":
            # Radial part only; symmetrisation leaves lambda*I unchanged.
            return torch.eye(3, dtype=U.dtype, device=U.device) * (
                torch.diagonal(U).sum() / 3.0)
        if centric is None or centric.numel() != F_obs.shape[0]:
            n = 0 if centric is None else centric.numel()
            raise RuntimeError(
                f"centric mask has {n} entries against {F_obs.shape[0]} "
                f"amplitudes -- the resolution window assumed here "
                f"([{d_min}, {d_max}] A) is not the engine's")
        Ufix = fit_aniso_intensity_space(
            F_obs, s_vec, shell_idx, centric.to(F_obs.device),
            P=kw.get("P", 20), min_count=kw.get("min_count", 20))
        captured["fixed"] = Ufix.detach().clone()
        return Ufix

    setattr(_align, "fit_overall_anisotropy", wrapped)
    try:
        yield
    finally:
        setattr(_align, "fit_overall_anisotropy", original)
