"""Arms for the overall-anisotropy correction.

``sh.fit_overall_anisotropy`` now fits in intensity space with a free constant.
The version it replaced regressed ``ln|F|^2 - ln<|F|^2>_shell`` on
``-2 pi^2 s.U.s`` by unweighted least squares **with no intercept**, and that was
the rotation function's last real defect. Three faults, all visible in its
output:

* ``E[ln(I/<I>)]`` is ``-gamma = -0.577`` for acentric reflections and
  ``-gamma - ln 2 = -1.270`` for centric ones, not zero. With no intercept the
  offset can only be absorbed by the quadratic form. The centric part is worse
  than a constant: centric reflections lie on the zones perpendicular to the
  symmetry axes, so the bias is direction-dependent.
* ``clamp(min=1e-30)`` turns a vanishing amplitude into ``y ~ -69``; a handful of
  those outweigh thousands of ordinary reflections in an unweighted fit.
* ``ln`` of a single-reflection intensity has variance ``pi^2/6`` (acentric) or
  ``pi^2/2`` (centric) with a heavy left tail, so the fit is dominated by the
  weak reflections carrying the least information.

Raw fitted B eigenvalue spreads came out at 70 to 5461 A^2 over the ten
benchmark structures. ``symmetrize_anisotropy`` then annihilated the garbage
where the point-group-invariant subspace is small (cubic -> one degree of
freedom) and left it standing where it is not (trigonal/hexagonal -> two).

:func:`fit_aniso_log_space` reproduces that version, so the measurement that
justified replacing it can be re-run against the current tree rather than taken
on trust.
"""

from __future__ import annotations

import math
from contextlib import contextmanager

import torch

from torchref.experimental.alignment.sh import fit_overall_anisotropy

#: U (A^2) -> B (A^2).
B_PER_U = 8.0 * math.pi ** 2

#: Arm names accepted by :func:`aniso_arm`. ``production`` is whatever
#: ``sh.fit_overall_anisotropy`` currently does; ``legacy_log`` is the biased
#: fit it replaced.
ARMS = ("production", "legacy_log", "no_aniso", "iso_only")


def fit_aniso_log_space(
    F_obs: torch.Tensor,
    s_vectors: torch.Tensor,
    shell_idx: torch.Tensor,
    P: int,
    *,
    min_count: int = 20,
) -> torch.Tensor:
    """The superseded log-space fit, verbatim, for A/B against the current one.

    Unweighted least squares of ``ln|F|^2 - ln<|F|^2>_shell`` on
    ``-2 pi^2 s.U.s`` with no constant term, and vanishing amplitudes clamped
    rather than dropped. Returns ``U`` in A^2 in the same convention as
    :func:`~torchref.experimental.alignment.sh.fit_overall_anisotropy`.
    """
    dtype = F_obs.dtype
    device = F_obs.device
    valid = shell_idx >= 0
    F = F_obs[valid]
    s = s_vectors[valid].to(dtype)
    idx = shell_idx[valid]

    count = torch.zeros(P, dtype=torch.int64, device=device)
    count.index_add_(0, idx, torch.ones_like(idx))
    F2 = F * F
    sum_F2 = torch.zeros(P, dtype=dtype, device=device)
    sum_F2.index_add_(0, idx, F2)
    mean_F2 = sum_F2 / count.clamp(min=1).to(dtype)

    good = count >= min_count
    if int(good.sum()) == 0:
        return torch.zeros((3, 3), dtype=dtype, device=device)
    keep = good[idx]
    F2k = F2[keep].clamp(min=1e-30)
    sk = s[keep].to(torch.float64)
    mean_F2_k = mean_F2[idx[keep]].clamp(min=1e-30)

    y = (torch.log(F2k) - torch.log(mean_F2_k)).to(torch.float64)
    X = torch.stack([
        sk[:, 0] ** 2, sk[:, 1] ** 2, sk[:, 2] ** 2,
        2.0 * sk[:, 0] * sk[:, 1],
        2.0 * sk[:, 0] * sk[:, 2],
        2.0 * sk[:, 1] * sk[:, 2],
    ], dim=-1)
    A = -2.0 * (torch.pi ** 2) * X
    u, _, _, _ = torch.linalg.lstsq(A, y.unsqueeze(-1))
    Uxx, Uyy, Uzz, Uxy, Uxz, Uyz = u.squeeze(-1).tolist()
    return torch.tensor(
        [[Uxx, Uxy, Uxz], [Uxy, Uyy, Uyz], [Uxz, Uyz, Uzz]],
        dtype=dtype, device=device,
    )


def tensor_report(U: torch.Tensor, tag: str) -> dict:
    """B eigenvalues (A^2) of a U tensor, as result-row columns."""
    ev = torch.linalg.eigvalsh(U.to(torch.float64).cpu()) * B_PER_U
    return {f"{tag}_B_min": round(float(ev[0]), 2),
            f"{tag}_B_max": round(float(ev[2]), 2),
            f"{tag}_B_spread": round(float(ev[2] - ev[0]), 2)}


@contextmanager
def aniso_arm(arm: str, data, *, d_min: float, d_max: float, captured: dict):
    """Swap the anisotropy fit for the duration of one FRF call.

    ``captured`` receives the tensor actually fitted under the key ``raw``, so a
    caller can report the artefact size alongside the rank it costs.

    Patches ``sh.fit_overall_anisotropy`` where ``rotation_search`` binds it --
    that is the symbol ``fit_anisotropy`` calls, and its result is what reaches
    the engine as ``U_aniso``.

    Parameters
    ----------
    arm : str
        One of :data:`ARMS`.
    data : ReflectionData
        Used to recompute the centric mask over the same resolution window the
        fit sees. A length mismatch raises rather than misaligning silently.
    d_min, d_max : float
        The window ``prepare_frf_inputs`` was called with.
    captured : dict
        Filled in by the wrapper.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown aniso arm {arm!r}; expected one of {ARMS}")
    import importlib

    _align = importlib.import_module(
        "torchref.experimental.alignment.rotation_search")

    original = _align.fit_overall_anisotropy
    rec = data.cell.reciprocal_basis_matrix.to(torch.float64)
    smag = (data.hkl.to(torch.float64) @ rec).norm(dim=-1)
    keep = (smag >= 1.0 / d_max) & (smag <= 1.0 / d_min)
    centric_window = (data.centric[keep].to(torch.bool)
                      if hasattr(data, "centric") else None)

    def wrapped(F_obs, s_vec, shell_idx, centric, **kw):
        U = original(F_obs, s_vec, shell_idx, centric, **kw)
        captured.setdefault("raw", U.detach().clone())
        if arm == "production":
            return U
        if arm == "no_aniso":
            return torch.zeros_like(U)
        if arm == "iso_only":
            # Radial part only; symmetrisation leaves lambda*I unchanged.
            return torch.eye(3, dtype=U.dtype, device=U.device) * (
                torch.diagonal(U).sum() / 3.0)
        if centric_window is None or centric_window.numel() != F_obs.shape[0]:
            n = 0 if centric_window is None else centric_window.numel()
            raise RuntimeError(
                f"centric mask has {n} entries against {F_obs.shape[0]} "
                f"amplitudes -- the resolution window assumed here "
                f"([{d_min}, {d_max}] A) is not the engine's")
        U_legacy = fit_aniso_log_space(
            F_obs, s_vec, shell_idx, P=kw.get("P", 20),
            min_count=kw.get("min_count", 20))
        captured["legacy"] = U_legacy.detach().clone()
        return U_legacy

    setattr(_align, "fit_overall_anisotropy", wrapped)
    try:
        yield
    finally:
        setattr(_align, "fit_overall_anisotropy", original)


__all__ = ["ARMS", "B_PER_U", "aniso_arm", "fit_aniso_log_space",
           "fit_overall_anisotropy", "tensor_report"]
