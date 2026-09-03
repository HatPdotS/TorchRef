"""Fast rotation function: find the orientations of a search model in a crystal.

One call, three inputs::

    from torchref.experimental.alignment import rotation_search

    solutions = rotation_search(model, data, model_error_A=0.8)
    placed = model.copy().rotate(solutions.rotations[0].T)

Everything else is derived from the model, the data and that error, following
Phaser's own chain (``runMR_FRF.cc:419-448``): the spherical-harmonic bandwidth
from the model's mean radius and the data's resolution, the sigma_A fall-off
from the coordinate error, the Wilson normalisation and French-Wilson posterior
from the observations and their sigmas.

The constants below are engine settings, not tuning knobs. They are scored on
whether the true orientation lands inside the candidate window the downstream
placement search carries forward, over the benchmark structures at seeded
orientations -- not on the median rank, which hides the cases that matter. Each
one's provenance is in its own comment.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from torchref.config import get_default_device, get_float_dtype
from torchref.scaling.weighting import (DEFAULT_SNR_CAP,
                                        DEFAULT_TRUST_CAP)
from .sh import (
    apply_overall_anisotropy,
    assign_shells,
    equal_count_shell_edges,
    fit_overall_anisotropy,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...io.datasets.reflection_data import ReflectionData
    from ...model.model_ft import ModelFT
    from .frf.types import RotationPeak

__all__ = ["FRFInputs", "RotationSolutions", "prepare_frf_inputs",
           "rotation_search"]

# Note for anyone reaching for the constants below programmatically: the package
# re-exports `rotation_search` (the function) under this module's own name, so
# `from torchref.experimental.alignment import rotation_search` binds the
# function, not the module. Use `importlib.import_module` for the module object.


#: Spherical-harmonic bandwidth ceiling. Where the model and the resolution ask
#: for more, the resolution is coarsened to match instead -- see
#: ``phaser_lmax_resolution``.
#:
#: Chosen by measurement, and the optimum is interior: over ten structures at
#: ten seeded orientations, truth lands in the top twenty on 95/100 cells at 48,
#: 98/100 at 64 and 98/100 at 100, but the binding case is 1AK5 (P 4 3 2), which
#: manages 6/10, 9/10 and 8/10. Only 64 clears nine of ten on every structure.
#: Phaser's own ceiling is 100 (``DEF_CLMN_LMAX``); here that is both worse on
#: 1AK5 and six to ten times slower, and it needs more than 32 GB on three of
#: the ten.
LMAX_CAP = 64

#: SO(3) sample spacing in degrees for the rotation-function grid. Also sets the
#: peak-suppression radius, as ``max(2 * this, 6)`` degrees. Inherited from the
#: configuration every measurement on this engine was made with.
GRID_SAMPLING_DEG = 3.0

#: Edge of the P1 box the model's transform is sampled in, as a multiple of the
#: molecular diameter. The box has to be large enough that the periodic images
#: do not overlap the molecule's own Patterson.
DENSE_CALC_PAD = 2.0

#: Equal-count resolution shells for the Wilson normalisation, the shell
#: variance weights, the relative Wilson-B fit and the anisotropy fit.
N_WILSON_SHELLS = 20

#: Discard rotation-function samples this far below the mean, in standard
#: deviations, before peak finding. Generous: it exists to bound the candidate
#: set, not to select.
SIGMA_THRESHOLD = -5.0

#: Babinet bulk-solvent parameters folded into sigma_A on the model side
#: (``EnsemblePDB.cc:96-100``). Phaser's defaults.
SOLVENT_FSOL = 0.95
SOLVENT_BSOL = 300.0

#: Low-resolution cutoff in Angstrom. Effectively none: the rotation function
#: wants the low-resolution terms, which carry the molecular envelope.
LOW_RESOLUTION_CUTOFF_A = 100.0

# Two things deliberately absent, both measured and rejected on the same panel:
#
# * **Orbit-deduplicated obs unroll.** Keeping only the distinct positions in
#   each reflection's orbit, as Phaser does, rather than all n_ops copies. It
#   moves 28 of 100 cells and in both directions -- 26 better, 13 worse against
#   the shipped configuration -- with the binding structure unchanged at 9/10. A
#   quarter of the results churned for no net gain.
# * **Two-radius Patterson union.** Running the search at two integration radii
#   and merging the peak lists by z-score. Exactly double the cost (8.7 s
#   against 4.4 s median) and it changes 1 cell in 100, which is the engine's own
#   run-to-run spread. An earlier measurement had favoured it; that result does
#   not survive the anisotropy fix.

#: Resolution window ``(d_max, d_min)`` the overall anisotropy is fitted in.
#: The tensor is then applied across the full range. Inherited from the range
#: every measurement on this engine was made with, and not itself measured --
#: unlike the constants above, this pair has no evidence behind it beyond being
#: the one in use.
ANISO_FIT_WINDOW_A = (15.0, 4.0)


@dataclass
class RotationSolutions:
    """Candidate orientations for a search model, best first.

    Attributes
    ----------
    rotations : torch.Tensor
        ``(n, 3, 3)`` float64. ``rotations[i]`` maps the search-model frame onto
        the crystal frame, so the coordinate rotation that places the model is
        its transpose: ``model.copy().rotate(rotations[i].T)``. Each is
        determined only up to the crystal's rotational symmetry -- its mates are
        ``rotations[i] @ R_g`` -- and the list carries one representative per
        orbit, so consecutive entries are distinct orientations.
    scores : torch.Tensor
        ``(n,)`` rotation-function value at each orientation.
    z_scores : torch.Tensor
        ``(n,)`` standard deviations above the mean over the whole SO(3) sample
        list. This is the scale to judge a solution on; the raw score is not
        comparable between runs.
    euler_zyz : torch.Tensor
        ``(n, 3)`` Edmonds active ZYZ angles in radians, the engine's native
        parametrisation: ``R = R_z(alpha) R_y(beta) R_z(gamma)``.
    lmax : int
        Spherical-harmonic bandwidth used.
    d_min : float
        High-resolution limit actually used (Angstrom), after the
        bandwidth-resolution coupling.
    model_error_A : float
        The coordinate error the sigma_A fall-off was built from.
    """

    rotations: torch.Tensor
    scores: torch.Tensor
    z_scores: torch.Tensor
    euler_zyz: torch.Tensor
    lmax: int
    d_min: float
    model_error_A: float

    def __len__(self) -> int:
        return int(self.rotations.shape[0])


def fit_anisotropy(
    data: "ReflectionData",
    *,
    d_min: float,
    d_max: float,
    n_shells: int = N_WILSON_SHELLS,
    device=None,
) -> torch.Tensor:
    """Fit the overall anisotropy tensor and project it onto the point group.

    Returns ``U`` in Angstrom squared as a ``(3, 3)`` at the configured float
    dtype, in the convention ``F_corrected = F * exp(+pi^2 s.U.s)``. It stays on
    the data's own device unless ``device`` says otherwise, so nothing crosses a
    device boundary to be fitted and come back.

    It used to be pinned to the host in double. Neither is needed. Measured over
    the 16 datasets in ``tests/files/mtz``, the same fit in float32 reproduces
    the double one to 3.3e-5 relative in ``U`` and 4.5e-6 in the correction
    factor it exists to produce; end to end, four of five panel cases return a
    bit-identical peak list and the fifth (2DQ6, P3121, the most nearly
    isotropic ``U`` of the panel) keeps its top orientation and reshuffles two
    near-tied deep ranks.

    The projection matters: an unconstrained six-component fit can return a
    tensor the lattice forbids, and applying it then modulates the observations
    by a direction-dependent factor the crystal cannot have. Cubic lattices
    admit one degree of freedom, tetragonal/trigonal/hexagonal two,
    orthorhombic three.
    """
    from .sh import hkl_symops_to_cartesian, symmetrize_anisotropy

    real = get_float_dtype()
    dev = data.hkl.device if device is None else device
    rec_basis = data.cell.reciprocal_basis_matrix.detach().to(device=dev, dtype=real)
    hkl = data.hkl.detach().to(device=dev, dtype=real)
    s_vec_all = hkl @ rec_basis
    s_mag_all = s_vec_all.norm(dim=-1)
    keep = (s_mag_all >= 1.0 / d_max) & (s_mag_all <= 1.0 / d_min)
    if int(keep.sum()) < n_shells * 5:
        raise ValueError(
            f"Only {int(keep.sum())} reflections in [{d_min}, {d_max}] A, too "
            f"few for {n_shells} shells."
        )
    F_obs = data.F.detach().to(device=dev, dtype=real).abs()[keep]
    s_vec = s_vec_all[keep]
    s_mag = s_mag_all[keep]
    centric = (
        data.centric.detach().to(dev)[keep].to(torch.bool)
        if hasattr(data, "centric")
        else torch.zeros_like(F_obs, dtype=torch.bool)
    )

    edges, _ = equal_count_shell_edges(s_mag, n_shells)
    shell_idx = assign_shells(s_mag, edges)
    U = fit_overall_anisotropy(
        F_obs, s_vec, shell_idx, centric, P=n_shells, min_count=20,
    )
    sym_cart = hkl_symops_to_cartesian(
        data.spacegroup.matrices.detach().to(device=dev, dtype=real), rec_basis,
    )
    return symmetrize_anisotropy(U, sym_cart)



@dataclass
class FRFInputs:
    """The observations the rotation search runs on, masked and corrected.

    ``F_obs`` is anisotropy-corrected. ``sig_F`` carries the same correction,
    which is a multiplicative factor, so ``F/sigma`` survives it unchanged -- it
    is here because the engine builds its measurement weight from the sigmas and
    the earlier code discarded them immediately after the Wilson step. ``None``
    when the data carry no sigmas.
    """

    F_obs: torch.Tensor              # (N,) anisotropy-corrected amplitudes
    sig_F: Optional[torch.Tensor]    # (N,) their sigmas, same correction
    hkl: torch.Tensor                # (N, 3) integer Miller indices
    s_vec: torch.Tensor              # (N, 3) reciprocal-space Cartesian
    s_mag: torch.Tensor              # (N,) inverse Angstrom
    centric: torch.Tensor            # (N,) bool
    U_aniso: torch.Tensor            # (3, 3) Popov-Bourenkov U
    device: torch.device


def prepare_frf_inputs(
    model: "ModelFT",
    data: "ReflectionData",
    *,
    d_min: float,
    d_max: float,
    n_shells: int,
    verbose: int = 0,
) -> FRFInputs:
    """Mask the observations to ``[d_min, d_max]`` and correct their anisotropy.

    The anisotropy tensor comes from :func:`fit_anisotropy`, which is also what
    the public :func:`rotation_search` uses. It used to be refitted here by a
    second copy of the same six lines over the same window -- two paths to one
    number is how they drift apart.

    Everything lands on the configured default device, not on whichever device
    ``model`` happens to sit on.
    """
    device = get_default_device()
    real = get_float_dtype()

    F_obs = data.F.to(real).abs()
    hkl_all = data.hkl
    rec_basis = data.cell.reciprocal_basis_matrix.to(real)
    s_vec_all = hkl_all.to(real) @ rec_basis
    s_mag_all = s_vec_all.norm(dim=-1)
    keep = (s_mag_all >= 1.0 / d_max) & (s_mag_all <= 1.0 / d_min)
    if keep.sum().item() < n_shells * 5:
        raise ValueError(
            f"Too few reflections ({keep.sum().item()}) in [{d_min},{d_max}] A "
            f"for {n_shells} shells; widen the resolution range."
        )
    F_obs = F_obs[keep].to(device)
    sig_F = getattr(data, "F_sigma", None)
    if sig_F is not None:
        sig_F = sig_F.to(real)[keep].to(device)
    hkl = hkl_all[keep].to(device)
    s_vec = s_vec_all[keep].to(device)
    s_mag = s_mag_all[keep].to(device)
    centric = (
        data.centric[keep].to(torch.bool).to(device)
        if hasattr(data, "centric")
        else torch.zeros_like(F_obs, dtype=torch.bool)
    )

    U_aniso = fit_anisotropy(
        data, d_min=d_min, d_max=d_max, n_shells=n_shells, device=device,
    )
    F_obs_aniso = apply_overall_anisotropy(F_obs, s_vec, U_aniso)
    # Same multiplicative factor, so F/sigma survives the correction intact.
    sig_F_aniso = (None if sig_F is None
                   else apply_overall_anisotropy(sig_F, s_vec, U_aniso))

    return FRFInputs(
        F_obs=F_obs_aniso, sig_F=sig_F_aniso, hkl=hkl, s_vec=s_vec,
        s_mag=s_mag, centric=centric, U_aniso=U_aniso, device=device,
    )


def search_peaks(
    model: "ModelFT",
    data: "ReflectionData",
    model_error_A: float,
    *,
    U_aniso: torch.Tensor,
    n_peaks: int,
    verbose: int = 0,
    device: Optional[torch.device] = None,
    obs_weight: str = "inverse_variance",
    sigma_a_source: str = "empirical",
    apply_bulk_solvent: bool = False,
    shell_variance_weights: bool = False,
    snr_cap: float = DEFAULT_SNR_CAP,
    trust_cap: float = DEFAULT_TRUST_CAP,
) -> Tuple[List["RotationPeak"], int, float]:
    """Run the rotation function, returning the engine's own peak list.

    Returns ``(peaks, lmax, d_min)``, where ``peaks`` is a list of
    :class:`~torchref.experimental.alignment.frf.types.RotationPeak` in Edmonds
    ZYZ. For the placement pipeline, which consumes peaks directly and has
    already fitted ``U_aniso`` for its rescore stage; :func:`rotation_search` is
    the entry point for everything else.
    """
    from ...utils import resolve_device
    from .frf.api import FastRotationFunction, phaser_lmax_resolution
    from .frf.dense_calc import dense_calc_via_box

    # One device for both inputs, rather than whichever one this function
    # happened to read first: `resolve_device` moves them into agreement (with a
    # warning) and falls back to the configured default. Data first, matching
    # the rest of the codebase.
    device = resolve_device(data, model, device=device)
    real = get_float_dtype()
    with torch.no_grad():
        rec_basis = data.cell.reciprocal_basis_matrix.to(real).to(device)
        hkl_all = data.hkl.to(device)
        s_vec_all = hkl_all.to(real) @ rec_basis
        s_mag_all = s_vec_all.norm(dim=-1)

        d_min_data = float(1.0 / s_mag_all.max().item())
        d_max = float(LOW_RESOLUTION_CUTOFF_A)

        # Couple the bandwidth to the resolution FIRST, because the mask below is
        # the only place the coarsened limit is applied -- the engine takes the
        # window as given. Data finer than L can represent contributes aliasing
        # rather than signal and buries the symmetry-diluted true peak, so
        # dropping this cut is not a small error: on 3K7M it moved truth from
        # rank 18 to rank 238 and doubled the runtime.
        model_radius_A = float(
            (model.xyz() - model.xyz().mean(0)).norm(dim=-1).mean().item()
        )
        L, d_min = phaser_lmax_resolution(model_radius_A, d_min_data, LMAX_CAP)

        # Masking before the unroll rather than after: |s| is symmetry-invariant
        # (symmetry operations are isometries), so the two commute -- and this way
        # the discarded high-resolution tail is not first replicated n_ops times.
        # The low-resolution half is live, not decorative: 3K7M carries two
        # reflections beyond 100 A that it removes.
        keep = (s_mag_all >= 1.0 / d_max) & (s_mag_all <= 1.0 / d_min)

        s_asu = s_vec_all[keep]
        s_mag_asu = s_mag_all[keep]
        F_obs = apply_overall_anisotropy(
            data.F.to(real).abs().to(device)[keep], s_asu, U_aniso,
        )
        sigF = (
            data.F_sigma.to(real).to(device)[keep]
            if getattr(data, "F_sigma", None) is not None
            else None
        )
        centric = (
            data.centric[keep].to(torch.bool).to(device)
            if hasattr(data, "centric")
            else torch.zeros_like(F_obs, dtype=torch.bool)
        )
        # Unmerged Bijvoet data puts two rows on one canonical index, so the
        # unroll below would weight those reflections twice. Detectable, so say
        # so rather than quietly double-counting.
        if getattr(data, "friedel_merged", True) is False:
            warnings.warn(
                "data are Bijvoet-unmerged: both members of a pair share a "
                "canonical index, so the symmetry unroll weights those "
                "reflections twice in the Patterson. Merge first for a clean "
                "rotation function.",
                RuntimeWarning, stacklevel=2,
            )

        # Expand the observations over the space group's rotations to fill
        # reciprocal space. |F(hS)| = |F(h)|, and the harmonics need the full
        # sphere: sampling only the asymmetric unit under-determines the
        # invariant subspace, which is what breaks the high-symmetry cases.
        #
        # h' = h.S, i.e. the transpose contraction. It agrees with S.h only for
        # orthogonal symmetry matrices, so using S.h works everywhere except
        # trigonal and hexagonal, where it mixes non-equivalent reflections into
        # one orbit.
        sg_mats = data.spacegroup.matrices.to(real).to(device)
        n_ops = int(sg_mats.shape[0])
        # `expand_reciprocal` is the package's one implementation of this
        # contraction, and it carries the h.S convention so no call site has to
        # re-decide the real/reciprocal transpose. It returns (n_ops, N, 3), i.e.
        # already op-major, which is the flattening the accumulations downstream
        # were measured with -- a different row order changes the summation order
        # in the later index_add_/unique and the last bits with it.
        #
        # It rounds to int64 internally, so the products are exact and the cast
        # below loses nothing. It also returns on the SPACE GROUP's device rather
        # than the caller's, so the move is load-bearing whenever they differ.
        hkl_unrolled = (
            data.spacegroup.expand_reciprocal(hkl_all[keep])
            .reshape(-1, 3)
            .to(device=device, dtype=rec_basis.dtype)
        )
        s_obs = hkl_unrolled @ rec_basis
        # Only the GEOMETRY is unrolled. The amplitudes, sigmas and centric
        # flags stay one row per unique reflection and the engine broadcasts
        # them after its per-reflection chain, which is symmetry-invariant. The
        # op-major flattening above means unrolled row `k * N + n` came from
        # unique row `n`, so the map is `arange(N)` tiled n_ops times.
        n_unique = int(F_obs.shape[0])
        asu_idx = (
            torch.arange(n_unique, device=device)
            .unsqueeze(0)
            .expand(n_ops, -1)
            .reshape(-1)
        )

        # The model's transform on a dense P1 grid rather than at the crystal's
        # own reflections: the crystal lattice is too sparse to determine the
        # high-l harmonics for a large molecule. `L` / `d_min` came from the
        # bandwidth coupling above, which the obs mask also used.
        s_calc, F_calc = dense_calc_via_box(
            model, d_max, d_min, pad=DENSE_CALC_PAD, verbose=verbose > 0,
        )
        s_calc = s_calc.to(device)
        F_calc = F_calc.to(device)

        # No relative Wilson-B match here any more. It multiplied `F_calc` by
        # exp(-B s^2/4) -- a smooth function of |s| -- and the engine's very next
        # step divides out exactly such a function when it normalises. Measured:
        # a relative B of +-30 A^2 moves E by at most 1.3e-7, the fit's own
        # convergence tolerance. It was computing a number and having it undone.
        #
        # Not the same as the earlier finding that knocking it out was
        # rank-neutral; that was a measurement about whether it mattered, this is
        # that it is arithmetically cancelled. `fit_relative_wilson_b` survives in
        # `frf/preprocessing` with no production caller at all -- it was kept for
        # the ML rescore, and that was deleted.

        # Point-group rotations in the Cartesian frame, so the peak finder can
        # treat an orientation and its symmetry mates as one peak. As a set
        # these equal B S B^-1; `hkl_symops_to_cartesian` returns the same
        # rotations in a different order. Built on the host in double, where
        # the peak finder's 3x3 algebra runs anyway.
        from .sh import hkl_symops_to_cartesian
        sym_cart = hkl_symops_to_cartesian(
            data.spacegroup.matrices.detach().cpu().to(torch.float64),  # dtype-ok: 3x3 rotation algebra in double on the host
            data.cell.reciprocal_basis_matrix.detach().cpu().to(torch.float64),  # dtype-ok: 3x3 rotation algebra in double on the host
        )

        engine = FastRotationFunction(
            s_obs, F_obs, centric, sg_mats,
            L=L, d_min=d_min, d_max=d_max,
            delta_vrms_A=float(model_error_A),
            n_wilson_shells=N_WILSON_SHELLS,
            sig_F_obs=sigF,
            grid_sampling_deg=GRID_SAMPLING_DEG,
            asu_idx=asu_idx,
            s_mag_asu=s_mag_asu,
            obs_weight=obs_weight, snr_cap=snr_cap, trust_cap=trust_cap,
            shell_variance_weights=shell_variance_weights,
            sym_cart=sym_cart,
        )
        _arf, peaks = engine.score_model(
            s_calc, F_calc, n_peaks=n_peaks,
            sigma_threshold=SIGMA_THRESHOLD,
            sigma_a_source=sigma_a_source,
            apply_bulk_solvent=apply_bulk_solvent,
            solvent_fsol=SOLVENT_FSOL, solvent_bsol=SOLVENT_BSOL,
        )

    return peaks, int(L - 1), float(d_min)


def _solutions(peaks: List["RotationPeak"], lmax: int, d_min: float,
               model_error_A: float) -> RotationSolutions:
    """Package a peak list as the public return type."""
    from torchref.base.alignment.rotation import rotation_matrix_euler_zyz

    euler = torch.tensor(
        [[p.alpha, p.beta, p.gamma] for p in peaks], dtype=torch.float64,  # dtype-ok: RotationSolutions are documented as float64
    ).reshape(-1, 3)
    rotations = (
        rotation_matrix_euler_zyz(euler)
        if euler.numel()
        else torch.zeros((0, 3, 3), dtype=torch.float64)  # dtype-ok: RotationSolutions are documented as float64
    )
    return RotationSolutions(
        rotations=rotations,
        scores=torch.tensor([p.score for p in peaks], dtype=torch.float64),  # dtype-ok: RotationSolutions are documented as float64
        z_scores=torch.tensor([p.sigma for p in peaks], dtype=torch.float64),  # dtype-ok: RotationSolutions are documented as float64
        euler_zyz=euler,
        lmax=lmax,
        d_min=d_min,
        model_error_A=float(model_error_A),
    )


def rotation_search(
    model: "ModelFT",
    data: "ReflectionData",
    model_error_A: float,
    *,
    n_peaks: int = 500,
    verbose: int = 0,
    device: Optional[torch.device] = None,
) -> RotationSolutions:
    """Find the orientations of ``model`` consistent with ``data``.

    Parameters
    ----------
    model : ModelFT
        Search model. Its orientation in the file is the frame the returned
        rotations are relative to; its position is irrelevant, since the
        rotation function works on the Patterson.
    data : ReflectionData
        Observed amplitudes. ``F_sigma`` is used for the French-Wilson posterior
        when present.
    model_error_A : float
        Expected r.m.s. coordinate error of the model against the target, in
        Angstrom. This sets the sigma_A fall-off, and so how much weight the
        high-resolution terms carry. Use
        :func:`~torchref.experimental.alignment.frf.preprocessing.oeffner_vrms`
        to estimate it from the model's length and sequence identity if it is
        not otherwise known.
    n_peaks : int, optional
        How many candidate orientations to return, best first. Default 500. This
        bounds the answer, not the search.
    verbose : int, optional
        Progress reporting. Default 0, silent.
    device : torch.device, optional
        Where to run. Default ``None`` takes ``data``'s device, moving ``model``
        to match; an explicit value moves both. With neither carrying one, the
        configured default applies.
    Returns
    -------
    RotationSolutions
        Ranked orientations. See that class for the rotation convention.

    Raises
    ------
    RuntimeError
        If ``model`` has no coordinates loaded.
    ValueError
        If the data carry too few reflections to bin.
    """
    if not model.ctx.initialized:
        raise RuntimeError("model has no coordinates; load a PDB first.")
    d_max_fit, d_min_fit = ANISO_FIT_WINDOW_A
    U_aniso = fit_anisotropy(data, d_min=d_min_fit, d_max=d_max_fit)
    peaks, lmax, d_min = search_peaks(
        model, data, model_error_A,
        U_aniso=U_aniso, n_peaks=n_peaks, verbose=verbose, device=device,
    )
    return _solutions(peaks, lmax, d_min, model_error_A)
