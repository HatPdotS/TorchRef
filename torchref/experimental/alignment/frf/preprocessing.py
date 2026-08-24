"""Observed-side preprocessing chain.

Mirrors the chain in Phaser ``DataMR::dataMR_FRF`` (DataMR.cc:863-1133)
and the auxiliary helpers in ``lib/math_FrenchWilson.cc`` and
``lib/math_RiceLLG.cc``. The heavier ports live in sibling modules
(:mod:`~torchref.experimental.alignment.frf.french_wilson` in particular);
this module imports them and carries the Phaser source citations.

If a specific preprocessing piece turns out to be wrong (per Tier 2
synthetic tests), the fix lives here — replace the import with a fresh
implementation cited line-by-line to the corresponding Phaser source.
"""
from __future__ import annotations

import math
from typing import Optional

import torch

from .french_wilson import french_wilson_preprocess      # math_FrenchWilson.cc + Dfactor.cc
from ..sh import (
    get_high_order_axis,                              # phaser's highOrderAxis()
    compute_patterson_shell_variance,
    equal_count_shell_edges,
    assign_shells,
)


def wilson_normalise(
    F: torch.Tensor,
    s_mag: torch.Tensor,
    n_shells: int = 20,
):
    """Per-shell Wilson normalisation of amplitudes.

    Source: Phaser's ``Feff[r] / SIGMAN.sqrt_epsnSN[r]`` (``DataMR.cc:925``)
    minus French-Wilson + explicit ε (``F`` is assumed anisotropy-corrected
    by the caller).

        E_h = F_h / sqrt(<F²>_p)    where p = shell containing h.

    Returns ``(E_h, sqrt_mean_F2_per_h)``.
    """
    edges, _ = equal_count_shell_edges(s_mag, n_shells)
    shell_idx = assign_shells(s_mag, edges)
    valid = shell_idx >= 0
    F_dtype = F.dtype
    F2 = F * F
    count = torch.zeros(n_shells, dtype=torch.int64, device=F.device)
    sumF2 = torch.zeros(n_shells, dtype=F_dtype, device=F.device)
    F2_v = F2[valid]
    idx_v = shell_idx[valid]
    count.index_add_(0, idx_v, torch.ones_like(idx_v))
    sumF2.index_add_(0, idx_v, F2_v)
    mean_F2 = sumF2 / count.clamp(min=1).to(F_dtype)
    mean_F2 = mean_F2.clamp(min=1e-12)
    sqrt_mean = mean_F2.sqrt()
    per_h = torch.ones_like(F)
    per_h[valid] = sqrt_mean[idx_v]
    E = F / per_h
    return E, per_h


def eterm_sigma_a(s_mag: torch.Tensor, delta_vrms_A: float) -> torch.Tensor:
    """Phaser's σA Eterm, literal port of ``Ensemble.cc:42``:

        Eterm(s) = exp(-(2π²/3) · s² · ΔVRMS_var)

    where ``ΔVRMS_var`` is the *coordinate variance* in Å². We accept the
    RMS coordinate error ``delta_vrms_A`` (Å) per the standard σA
    convention and square it internally: ``ΔVRMS_var = delta_vrms_A²``.
    """
    s2 = s_mag * s_mag
    return torch.exp(-(2.0 / 3.0) * (math.pi ** 2) * s2 * (delta_vrms_A ** 2))

__all__ = [
    "wilson_normalise",
    "wilson_normalise_epsilon",
    "compute_epsilon",
    "eterm_sigma_a",
    "french_wilson_preprocess",
    "get_high_order_axis",
    "build_lerf1_intensity",
    "apply_shell_variance_weights",
    "detect_zsymm",
    "epsilon_aware_unroll",
    "compute_v_budget",
    "bulk_solvent_factor",
    "oeffner_vrms",
    "fit_relative_wilson_b",
]


def epsilon_aware_unroll(
    hkl_int: torch.Tensor,
    sym_mats: torch.Tensor,
):
    """Unroll each ASU reflection to the **unique** P1 positions in its orbit.

    For each ``h`` in the input list, generate the orbit ``{S_k · h}`` over the
    ``n_ops`` symop matrices and emit one entry per *distinct* position. Axial /
    special-position reflections (whose stabilizer has order ε(h) > 1) therefore
    appear ``n_ops / ε(h)`` times, **not** ``n_ops`` times.

    Mirrors Phaser's ``if (!duplicate(isym, rhkl))`` skip in
    ``DataMR.cc:954-986``. A naive ``einsum + reshape`` unroll over-counts axial
    reflections by ε(h), polluting the obs SH coefficients with spurious
    non-invariant content — the noise channel that hurts high-symmetry cases.

    Parameters
    ----------
    hkl_int : (N, 3) integer tensor
        ASU Miller indices.
    sym_mats : (n_ops, 3, 3) tensor
        Spacegroup rotation operators in the reciprocal (hkl) basis. Cast to
        ``long`` internally; values must be integer.

    Returns
    -------
    unrolled_hkl : (M, 3) long tensor — flat list of unique orbit positions
        across all ASU reflections.
    asu_idx : (M,) long tensor — index into ``hkl_int`` that each unrolled entry
        came from. Callers use it to broadcast intensities / centric / sigF:
        ``F_unrolled = F_obs[asu_idx]``.
    """
    hkl_int = hkl_int.to(torch.long)
    sym_mats = sym_mats.round().to(torch.long)
    N, n_ops = hkl_int.shape[0], sym_mats.shape[0]
    # Orbits: (N, n_ops, 3) — h.S_k, the row-vector (reciprocal-space)
    # convention, matching the unroll sites in `align.py`. Note this is the
    # TRANSPOSE contraction: `kji`, not `kij`. They coincide only for
    # orthogonal symmetry matrices, so `kij` silently works everywhere except
    # trigonal/hexagonal.
    # Integer einsum dispatches to baddbmm, which CUDA does not implement for
    # Long; compute in float64 (exact for symop 0/±1 × small Miller indices)
    # and round back so the GPU path works.
    orbits = (
        torch.einsum(
            "kji,nj->nki", sym_mats.to(torch.float64), hkl_int.to(torch.float64),
        )
        .round()
        .to(torch.long)
    )
    # Pack (h, k, l) into a single int64 key for per-row dedup.
    base = 2 * int(orbits.abs().max().item()) + 1
    key = (orbits[:, :, 0] * base + orbits[:, :, 1]) * base + orbits[:, :, 2]
    # Stable sort along dim=1 → duplicates land contiguously, lowest op-index
    # first (matches Phaser's "first occurrence wins" rule).
    sorted_keys, sort_idx = key.sort(dim=1, stable=True)
    first_in_sorted = torch.cat(
        [
            torch.ones(N, 1, dtype=torch.bool, device=key.device),
            sorted_keys[:, 1:] != sorted_keys[:, :-1],
        ],
        dim=1,
    )
    keep_mask = torch.empty_like(first_in_sorted)
    keep_mask.scatter_(1, sort_idx, first_in_sorted)
    asu_idx, op_idx = keep_mask.nonzero(as_tuple=True)
    unrolled_hkl = orbits[asu_idx, op_idx]
    return unrolled_hkl, asu_idx


def compute_epsilon(
    hkl: torch.Tensor,
    sym_mats: torch.Tensor,
) -> torch.Tensor:
    """Reflection multiplicity ε(h) — the order of the stabilizer subgroup.

    Phaser source: the ``epsn`` array in ``DataMR.cc`` (used in
    ``SIGMAN.sqrt_epsnSN``, DataMR.cc:925) — same role as
    ``cctbx::miller::index_span`` epsilons.

    ε(h) = number of point-group rotation operators W for which
    ``h · W = h`` (row-vector convention, no Friedel). For a general
    reflection ε = 1; reflections on an n-fold symmetry axis get ε = n.

    Used to epsilon-correct Wilson normalisation: axial reflections are
    systematically stronger (``⟨I_h⟩ = ε_h · Σ``), so without the
    correction they over-weight the m = 0 SH column and bias the
    rotation-function map for high-symmetry spacegroups.

    Parameters
    ----------
    hkl : (N, 3) integer-valued (any dtype) Miller indices.
    sym_mats : (n_ops, 3, 3) integer rotation operators (fractional/lattice
        rotation parts of the spacegroup).

    Returns
    -------
    epsilon : (N,) float — multiplicity ε ≥ 1.
    """
    h = hkl.to(torch.float64)
    W = sym_mats.to(torch.float64)
    eps = torch.zeros(h.shape[0], dtype=torch.float64, device=h.device)
    for k in range(W.shape[0]):
        h_t = h @ W[k]                       # row-vector: h' = h · W
        same = (h_t.round() == h).all(dim=-1)
        eps += same.to(torch.float64)
    return eps.clamp(min=1.0)


def wilson_normalise_epsilon(
    F: torch.Tensor,
    s_mag: torch.Tensor,
    epsilon: torch.Tensor,
    n_shells: int = 20,
):
    """Epsilon-corrected per-shell Wilson normalisation.

    Standard crystallographic normalisation with the multiplicity factor:

        Σ_shell = ⟨I_h / ε_h⟩_shell
        E²_h    = (I_h / ε_h) / Σ_shell

    so axial reflections (large ε) are not over-counted. Returns
    ``(E, sqrt_mean_eps_corrected)`` mirroring ``wilson_normalise``.
    """
    edges, _ = equal_count_shell_edges(s_mag, n_shells)
    shell_idx = assign_shells(s_mag, edges)
    valid = shell_idx >= 0
    I_corr = (F * F) / epsilon.clamp(min=1.0)
    count = torch.zeros(n_shells, dtype=torch.int64, device=F.device)
    sumI = torch.zeros(n_shells, dtype=F.dtype, device=F.device)
    idx_v = shell_idx[valid]
    count.index_add_(0, idx_v, torch.ones_like(idx_v))
    sumI.index_add_(0, idx_v, I_corr[valid])
    mean_I = (sumI / count.clamp(min=1).to(F.dtype)).clamp(min=1e-12)
    per_h = torch.ones_like(F)
    per_h[valid] = mean_I[idx_v]
    E = (I_corr / per_h).clamp(min=0.0).sqrt()
    return E, per_h.sqrt()


def build_lerf1_intensity(
    eEobs: torch.Tensor,
    centric_obs: torch.Tensor,
    dfac: Optional[torch.Tensor] = None,
    use_centric_weight: bool = True,
) -> torch.Tensor:
    """LERF1 observed intensity: ``cweight · (eEobs² − 1) · DFAC²``.

    Phaser source: ``DataMR::m_LETF1`` (DataMR.cc:1326-1431) — the
    intensity that gets fed into the Bessel-SH expansion. cweight is
    ε(h) · (1 for centric, 2 for acentric); we use the centric/acentric
    factor only (the ε(h) multiplicity is implicit in the symmetry
    reduction of the input reflection set).
    """
    if use_centric_weight:
        cw = torch.where(
            centric_obs.bool(),
            torch.ones_like(eEobs),
            2.0 * torch.ones_like(eEobs),
        )
    else:
        cw = torch.ones_like(eEobs)
    if dfac is None:
        dfac = torch.ones_like(eEobs)
    return cw * (eEobs * eEobs - 1.0) * (dfac * dfac)


def apply_shell_variance_weights(
    intensity: torch.Tensor,
    s_mag: torch.Tensor,
    n_var_shells: int = 20,
    shell_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-shell empirical variance reweight.

    Downweights shells whose observed Patterson intensity is dominated
    by noise. Mean-normalised so total scale doesn't shift. Closest
    Phaser analog is per-shell BINS + ``best(r)`` in ``Ensemble.cc``.

    ``shell_idx`` reuses an assignment the caller already made. Worth passing:
    binning here independently of the Wilson normalisation puts the two on
    edges that disagree for the reflections sitting on a boundary.
    """
    if shell_idx is None:
        edges, _ = equal_count_shell_edges(s_mag, n_var_shells)
        shell_idx = assign_shells(s_mag, edges)
    valid = shell_idx >= 0
    var_p = compute_patterson_shell_variance(
        intensity[valid].to(torch.float64),
        shell_idx[valid],
        P=n_var_shells,
    )
    inv_sqrt_var = 1.0 / var_p.sqrt().clamp(min=1e-30)
    inv_sqrt_var = inv_sqrt_var * (
        n_var_shells / inv_sqrt_var.sum().clamp(min=1e-30)
    )
    weights = torch.ones_like(intensity)
    weights[valid] = inv_sqrt_var[shell_idx[valid]].to(intensity.dtype)
    return intensity * weights


def detect_zsymm(sym_mats: Optional[torch.Tensor]) -> int:
    """Detect ``ZSYMM`` for the **z-axis** m-symmetry filter.

    Phaser source: ``highOrderAxis()`` in ``rotationgroup.h``. The m-filter
    zeroes obs SH coefficients with ``|m| mod ZSYMM != 0`` — but ``m`` is the
    azimuthal order **about the z axis of the SH basis**, so the filter is only
    valid when the crystal's high-order rotation axis is actually along z
    (cubic / tetragonal / hexagonal: principal axis = c ∥ z).

    For spacegroups whose high-order axis is x or y — e.g. monoclinic C2 / P2₁
    with the 2-fold along b ∥ y — applying a z-axis filter is WRONG (it filters
    about the wrong axis and corrupts the obs coefficients). Phaser rotates the
    high-order axis to z before the expansion (DataMR.cc:962-979); we do not, so
    here we conservatively return ``ZSYMM=1`` (no filter) when the axis is not z,
    rather than apply a wrong filter. Verified: for all benchmark cubic/tetra/hex
    cases the axis is z (filter unchanged); only monoclinic 1DAW/3E98/3VRJ change
    (they already rank 0-3, and a 2-fold m-filter is a weak constraint anyway).
    """
    if sym_mats is None:
        return 1
    axis, zsymm = get_high_order_axis(sym_mats.to(torch.float64).cpu())
    if axis != 2:  # high-order axis not along z → don't apply a wrong filter
        return 1
    return int(zsymm)


def compute_v_budget(
    eps_factor: torch.Tensor,
    sigma_a: torch.Tensor,
    n_mol: int = 1,
    totvar_known: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Phaser's per-reflection variance budget ``V(h)`` for the m_LETF1 LL.

    Source: ``DataMR.cc:949`` (build) + ``DataMR.cc:1411`` (use in ``m_LETF1``)::

        V = PTNCS.EPSFAC[r] − totvar_known[r] − totvar_search[r]

    where ``EPSFAC[r] = ε(h)`` (or the tNCS-corrected variance bin in the
    NCS-present case; we use plain ε for the standalone search), ``totvar_known``
    is the variance contribution from any fixed model (zero for a pure cross
    rotation function), and ``totvar_search = σ_A²(s) · n_mol`` is the variance
    explained by the moving model at the expected scattering content.

    For the cross-rotation case (no fixed model, ``totvar_known = 0``):

        V(h) = ε(h) − σ_A²(s)·n_mol

    Working in E-space (obs already Wilson-normalised), so no Σ_N factor.

    Parameters
    ----------
    eps_factor : (N,) tensor
        Per-reflection ε(h), the multiplicity (1 for general positions,
        n>1 for reflections on n-fold symmetry axes). From
        :func:`compute_epsilon`.
    sigma_a : (N,) tensor
        Per-reflection σ_A(s) (interpolated from the per-shell fit).
    n_mol : int
        Number of molecules in the unit cell summed over by the NSYMP-loop in
        the calc-side expected intensity. Equals ``NSYMP`` for the standalone
        cross-rotation search.
    totvar_known : (N,) tensor, optional
        Variance contribution from a fixed/known model. Default ``None`` →
        treated as zero (standalone cross-rotation function).

    Returns
    -------
    V : (N,) tensor
        Per-reflection variance budget. Clamped to ``> 0`` to keep the LL finite;
        a non-positive ``V`` would imply σ_A² overshoots ε, which Phaser also
        guards against via ``PHASER_ASSERT(C > 0)`` at DataMR.cc:1413.
    """
    sa = sigma_a.to(eps_factor.dtype)
    moving = (sa * sa) * float(n_mol)
    V = eps_factor - moving
    if totvar_known is not None:
        V = V - totvar_known.to(eps_factor.dtype)
    return V.clamp(min=1e-6)


# =============================================================================
# Phaser model-prep — three pieces Phaser applies before the FRF that we don't.
# See `Ensemble::setPDB` (EnsemblePDB.cc:40-100). Adding these as opt-in.
# =============================================================================


def bulk_solvent_factor(
    s_mag: torch.Tensor,
    fsol: float = 0.95,
    bsol: float = 300.0,
    sigA_min: float = 0.01,
) -> torch.Tensor:
    """Phaser's Babinet bulk-solvent term — ``solTerm.h:9``::

        solTerm(s²) = max(SIGA_MIN, 1 − fsol · exp(−bsol · s²/4))

    Models the bulk solvent's contribution to the structure factor via Babinet's
    principle. At low resolution (s→0) the term → ``1 − fsol`` ≈ 0.05 (with the
    default ``fsol=0.95``), aggressively suppressing the calc — physically, the
    model represents only the macromolecule, but the diffraction data sees
    macromolecule + bulk solvent, and at low resolution the solvent's flat
    average density partially cancels the macromolecule's contribution. At high
    resolution (s→∞) the term → 1 (no effect).

    Phaser folds this into the effective σ_A via
    ``σ_A_eff(s) = solTerm(s²) · DLuzzati(s², vrms)`` (EnsemblePDB.cc:96-100).
    For callers that work in σ_A space (the rescore, the FRF eterm), multiplying
    by this factor reproduces that behaviour.

    Defaults match Phaser (``DEF_SOLPAR_BULK_FSOL=0.95``,
    ``DEF_SOLPAR_BULK_BSOL=300``, ``DEF_SOLPAR_SIGA_MIN=0.01``).

    Parameters
    ----------
    s_mag : tensor
        Per-reflection reciprocal-space magnitude |s| (Å^-1).
    fsol, bsol, sigA_min : float
        Babinet parameters. Defaults match Phaser.

    Returns
    -------
    torch.Tensor
        Per-reflection solvent multiplier, same shape as ``s_mag``. Always in
        ``[sigA_min, 1]``.
    """
    s2 = s_mag * s_mag
    babinet = 1.0 - float(fsol) * torch.exp(-float(bsol) * s2 / 4.0)
    return babinet.clamp(min=float(sigA_min))


def oeffner_vrms(n_residues: int, identity: float = 1.0) -> float:
    """Phaser's Oeffner empirical vrms estimate — ``rms_estimate.cc:37``::

        vrms = A · (B + clamp(n_residues, 125, 1500))^(1/3) · exp(C · (1 − ident))

    with ``A = 0.0569``, ``B = 173``, ``C = 1.52``. The clamp avoids extrapolating
    beyond the well-populated range of Oeffner et al.'s training set
    (Acta Cryst. (2013) D69:2209-2215). For a perfect model (``identity=1``) and
    a typical protein (~300 residues), this gives vrms ≈ 0.47 Å; large
    assemblies (clamped at 1500) give vrms ≈ 0.67 Å. Phaser uses this as the
    Luzzati ``vrms`` for the σ_A computation.

    Parameters
    ----------
    n_residues : int
        Sequence length of the search model. Internally clamped to [125, 1500].
    identity : float, optional
        Sequence identity to expected target on [0, 1]. Default 1.0 (perfect).

    Returns
    -------
    float
        Coordinate RMS estimate in Å, suitable as ``delta_vrms_A`` for
        :func:`compute_sigma_a_luzzati` / :func:`eterm_sigma_a`.
    """
    A, B, C = 0.0569, 173.0, 1.52
    n_clamped = max(125, min(int(n_residues), 1500))
    return A * (B + n_clamped) ** (1.0 / 3.0) * math.exp(C * (1.0 - float(identity)))


def fit_relative_wilson_b(
    F_obs: torch.Tensor,
    F_calc: torch.Tensor,
    s_mag: torch.Tensor,
    n_shells: int = 20,
    clamp_b: float = 50.0,
    s_mag_calc: Optional[torch.Tensor] = None,
) -> float:
    """Phaser's relative Wilson-B fit — ``EnsemblePDB.cc:793-851``.

    Estimates the per-model relative Wilson B-factor that brings the calc's
    per-shell <|F_calc|²> into agreement with the data's per-shell <|F_obs|²>.
    Implemented as a weighted linear regression of
    ``log(<F_obs²>_shell / <F_calc²>_shell)`` against ``s²`` over equal-count
    shells; returns ``WilsonB = -2 · slope``.

    Per-shell weighting mirrors Phaser (EnsemblePDB.cc:830-835):
      - ``s² < 0.009`` (d > 10.5 Å): weight = 0 (no reliable BEST curve).
      - ``s² < 0.04`` (d > 5 Å): weight = ``(s²/0.04)²`` (down-weighted).
      - ``s² ≥ 0.04``: weight = 1.

    Apply at call sites as ``F_calc · exp(-WilsonB · s² / 4)``.

    Parameters
    ----------
    F_obs : (N_obs,) tensor
    F_calc : (N_calc,) tensor
        Calc amplitudes. May be on a different reciprocal grid than obs (e.g.
        the dense P1-box from ``dense_calc_via_box``); per-shell means handle
        the binning independently.
    s_mag : (N_obs,) tensor
        Reciprocal-space magnitudes for OBS. Used to derive shell edges by
        equal-count binning on the obs distribution.
    s_mag_calc : (N_calc,) tensor, optional
        Reciprocal-space magnitudes for CALC. Defaults to ``s_mag`` (when obs
        and calc share a grid). When given, obs and calc are binned into the
        SAME edges (derived from obs ``s_mag``) but with independent counts.
    n_shells : int
        Number of equal-count resolution shells (on obs).
    clamp_b : float
        Clamps the fitted B to ``[-clamp_b, +clamp_b]``.

    Returns
    -------
    float
        Relative Wilson B-factor (Å²). ``0`` if too few shells contribute.
    """
    if s_mag_calc is None:
        s_mag_calc = s_mag
    if F_obs.shape[0] != s_mag.shape[0]:
        raise ValueError(
            f"F_obs / s_mag length mismatch: {F_obs.shape[0]} vs {s_mag.shape[0]}"
        )
    if F_calc.shape[0] != s_mag_calc.shape[0]:
        raise ValueError(
            f"F_calc / s_mag_calc length mismatch: "
            f"{F_calc.shape[0]} vs {s_mag_calc.shape[0]}"
        )

    edges, _ = equal_count_shell_edges(s_mag, n_shells)
    # Bin obs and calc into the SAME edges (independently — different N's).
    shell_idx_obs = assign_shells(s_mag, edges)
    shell_idx_calc = assign_shells(s_mag_calc, edges)
    valid_obs = shell_idx_obs >= 0
    valid_calc = shell_idx_calc >= 0
    if not valid_obs.any() or not valid_calc.any():
        return 0.0

    F2_obs = (F_obs * F_obs).to(torch.float64)
    F2_calc = (F_calc * F_calc).to(torch.float64)
    s2_obs = (s_mag * s_mag).to(torch.float64)

    counts_obs = torch.zeros(n_shells, dtype=torch.int64, device=s_mag.device)
    counts_calc = torch.zeros(n_shells, dtype=torch.int64, device=s_mag.device)
    sum_F2obs = torch.zeros(n_shells, dtype=torch.float64, device=s_mag.device)
    sum_F2calc = torch.zeros(n_shells, dtype=torch.float64, device=s_mag.device)
    sum_s2 = torch.zeros(n_shells, dtype=torch.float64, device=s_mag.device)
    idx_v_obs = shell_idx_obs[valid_obs]
    idx_v_calc = shell_idx_calc[valid_calc]
    counts_obs.index_add_(0, idx_v_obs, torch.ones_like(idx_v_obs))
    counts_calc.index_add_(0, idx_v_calc, torch.ones_like(idx_v_calc))
    sum_F2obs.index_add_(0, idx_v_obs, F2_obs[valid_obs])
    sum_F2calc.index_add_(0, idx_v_calc, F2_calc[valid_calc])
    sum_s2.index_add_(0, idx_v_obs, s2_obs[valid_obs])  # obs-side s² for the regression abscissa

    # Drop shells empty on either side.
    keep = (counts_obs > 0) & (counts_calc > 0)
    mean_F2obs = sum_F2obs[keep] / counts_obs[keep].to(torch.float64)
    mean_F2calc = sum_F2calc[keep] / counts_calc[keep].to(torch.float64)
    mean_s2 = sum_s2[keep] / counts_obs[keep].to(torch.float64)

    # log(Σ_N / Σ_P) per shell.
    eps = 1e-30
    log_ratio = (mean_F2obs.clamp(min=eps) / mean_F2calc.clamp(min=eps)).log()

    # Phaser per-shell weights (EnsemblePDB.cc:830-835).
    weights = torch.ones_like(mean_s2)
    low_mask = mean_s2 < 0.04
    weights[low_mask] = (mean_s2[low_mask] / 0.04) ** 2
    weights[mean_s2 < 0.009] = 0.0
    if (weights > 0).sum().item() < 2:
        return 0.0

    # Weighted linear fit y = slope · x (no intercept), per the Phaser source.
    w = weights
    x = mean_s2
    y = log_ratio
    sw = w.sum()
    swx = (w * x).sum()
    swy = (w * y).sum()
    swx2 = (w * x * x).sum()
    swxy = (w * x * y).sum()
    denom = (sw * swx2 - swx * swx).item()
    if abs(denom) < 1e-30:
        return 0.0
    slope = (sw * swxy - swx * swy).item() / denom
    # Phaser: WilsonB_intensity = -4·slope, then halved → WilsonB = -2·slope.
    wilson_b = -2.0 * slope
    return float(max(-clamp_b, min(wilson_b, clamp_b)))


# Note: a naive OLS-on-log-F² fit of anisotropic Wilson U was attempted on
# 2026-05-28 and didn't work. The Wilson left tail (small F values produce huge
# negative log F²) dominates the regression, returning U components of order
# 10²–10³ Å² on real data — three orders of magnitude beyond physical, on both
# easy (1DAW) and hard (2DQ6) cases. Robustifying via |F|-weighting + ridge
# only made the fit saturate any sensible clamp. Phaser's ``scaleANIS``
# (``DataB.cc``, ~500 LoC) is an iterative ML fit on ``logSigmaEsq``; that's
# the right approach if obs-side aniso ever becomes the next lever. The 2DQ6
# benchmark failure we were chasing turned out to be tNCS
# (``<(E²−1)²>_acentric = 5.5`` vs Wilson = 1.0), not anisotropy, so this
# branch is not in the immediate critical path.
