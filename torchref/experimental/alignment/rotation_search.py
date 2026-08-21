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

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

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

__all__ = ["RotationSolutions", "rotation_search"]

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
        determined only up to the crystal's rotational symmetry, so a solution
        and its symmetry mates are the same answer.
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
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Fit the overall anisotropy tensor and project it onto the point group.

    Returns ``U`` in Angstrom squared, in the convention
    ``F_corrected = F * exp(+pi^2 s.U.s)``.

    The projection matters: an unconstrained six-component fit can return a
    tensor the lattice forbids, and applying it then modulates the observations
    by a direction-dependent factor the crystal cannot have. Cubic lattices
    admit one degree of freedom, tetragonal/trigonal/hexagonal two,
    orthorhombic three.
    """
    from .sh import hkl_symops_to_cartesian, symmetrize_anisotropy

    device = device or data.hkl.device
    rec_basis = data.cell.reciprocal_basis_matrix.to(torch.float64)
    s_vec_all = data.hkl.to(torch.float64) @ rec_basis
    s_mag_all = s_vec_all.norm(dim=-1)
    keep = (s_mag_all >= 1.0 / d_max) & (s_mag_all <= 1.0 / d_min)
    if int(keep.sum()) < n_shells * 5:
        raise ValueError(
            f"Only {int(keep.sum())} reflections in [{d_min}, {d_max}] A, too "
            f"few for {n_shells} shells."
        )
    F_obs = data.F.to(torch.float64).abs()[keep].to(device)
    s_vec = s_vec_all[keep].to(device)
    s_mag = s_mag_all[keep].to(device)
    centric = (
        data.centric[keep].to(torch.bool).to(device)
        if hasattr(data, "centric")
        else torch.zeros_like(F_obs, dtype=torch.bool)
    )

    edges, _ = equal_count_shell_edges(s_mag, n_shells)
    shell_idx = assign_shells(s_mag, edges)
    U = fit_overall_anisotropy(
        F_obs, s_vec, shell_idx, centric, P=n_shells, min_count=20,
    )
    sym_cart = hkl_symops_to_cartesian(
        data.spacegroup.matrices.to(torch.float64).to(device),
        rec_basis.to(device),
    )
    return symmetrize_anisotropy(U, sym_cart)


def search_peaks(
    model: "ModelFT",
    data: "ReflectionData",
    model_error_A: float,
    *,
    U_aniso: torch.Tensor,
    n_peaks: int,
    verbose: int = 0,
) -> Tuple[List["RotationPeak"], int, float]:
    """Run the rotation function, returning the engine's own peak list.

    Returns ``(peaks, lmax, d_min)``, where ``peaks`` is a list of
    :class:`~torchref.experimental.alignment.frf.types.RotationPeak` in Edmonds
    ZYZ. For the placement pipeline, which consumes peaks directly and has
    already fitted ``U_aniso`` for its rescore stage; :func:`rotation_search` is
    the entry point for everything else.
    """
    from .frf.api import FastRotationFunction, phaser_lmax_resolution
    from .frf.dense_calc import dense_calc_via_box
    from .frf.preprocessing import fit_relative_wilson_b

    device = model.xyz().device
    with torch.no_grad():
        rec_basis = data.cell.reciprocal_basis_matrix.to(torch.float64).to(device)
        hkl_all = data.hkl.to(device)
        s_vec_all = hkl_all.to(torch.float64) @ rec_basis
        s_mag_all = s_vec_all.norm(dim=-1)

        # Take the observations at the full data resolution: the bandwidth
        # coupling below coarsens the limit to whatever the harmonics can
        # represent, so pre-restricting here would only lose the terms it keeps.
        d_min_data = float(1.0 / s_mag_all.max().item())
        d_max = float(LOW_RESOLUTION_CUTOFF_A)
        keep = (s_mag_all >= 1.0 / d_max) & (s_mag_all <= 1.0 / d_min_data)

        s_obs = s_vec_all[keep]
        F_obs = apply_overall_anisotropy(
            data.F.to(torch.float64).abs().to(device)[keep], s_obs, U_aniso,
        )
        sigF = (
            data.F_sigma.to(torch.float64).to(device)[keep]
            if getattr(data, "F_sigma", None) is not None
            else None
        )
        centric = (
            data.centric[keep].to(torch.bool).to(device)
            if hasattr(data, "centric")
            else torch.zeros_like(F_obs, dtype=torch.bool)
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
        sg_mats = data.spacegroup.matrices.to(torch.float64).to(device)
        n_ops = int(sg_mats.shape[0])
        hkl_keep = hkl_all.to(torch.float64)[keep]
        hkl_unrolled = torch.einsum(
            "kji,nj->kni", sg_mats, hkl_keep).reshape(-1, 3)
        s_obs = hkl_unrolled @ rec_basis
        F_obs = F_obs.unsqueeze(0).expand(n_ops, -1).reshape(-1).contiguous()
        centric = centric.unsqueeze(0).expand(n_ops, -1).reshape(-1).contiguous()
        if sigF is not None:
            sigF = sigF.unsqueeze(0).expand(n_ops, -1).reshape(-1).contiguous()

        # The model's transform on a dense P1 grid rather than at the crystal's
        # own reflections: the crystal lattice is too sparse to determine the
        # high-l harmonics for a large molecule.
        model_radius_A = float(
            (model.xyz() - model.xyz().mean(0)).norm(dim=-1).mean().item()
        )
        L, d_min = phaser_lmax_resolution(model_radius_A, d_min_data, LMAX_CAP)
        s_calc, F_calc = dense_calc_via_box(
            model, d_max, d_min, pad=DENSE_CALC_PAD, verbose=verbose > 0,
        )
        s_calc = s_calc.to(device)
        F_calc = F_calc.to(device)

        # Put the model's amplitudes on the observations' overall B scale
        # (EnsemblePDB.cc:793-851), so the radial fall-off does not by itself
        # discriminate between orientations.
        s_calc_mag = s_calc.norm(dim=-1)
        B_rel = fit_relative_wilson_b(
            F_obs.to(torch.float64), F_calc.to(torch.float64),
            s_obs.norm(dim=-1).to(torch.float64), n_shells=N_WILSON_SHELLS,
            s_mag_calc=s_calc_mag.to(torch.float64),
        )
        if abs(B_rel) > 1e-6:
            F_calc = F_calc * torch.exp(-B_rel * (s_calc_mag * s_calc_mag) / 4.0)
            if verbose > 0:
                print(f"  relative Wilson B = {B_rel:+.2f} A^2", flush=True)

        engine = FastRotationFunction(
            s_obs, F_obs, centric, sg_mats,
            d_min=d_min_data, d_max=d_max,
            delta_vrms_A=float(model_error_A),
            n_wilson_shells=N_WILSON_SHELLS,
            sig_F_obs=sigF,
            grid_sampling_deg=GRID_SAMPLING_DEG,
            model_radius_A=model_radius_A,
            auto_lmax=True,
            lmax_cap=LMAX_CAP,
            # The spherical-harmonic contraction dominates the runtime and is
            # rate-limited in double precision on accelerators. Its float32 path
            # keeps the Bessel recurrence and the cross-chunk accumulator at
            # full precision.
            compute_dtype=torch.complex64 if device.type == "cuda" else None,
        )
        _arf, peaks = engine.score_model(
            s_calc, F_calc, n_peaks=n_peaks,
            sigma_threshold=SIGMA_THRESHOLD,
            apply_bulk_solvent=True,
            solvent_fsol=SOLVENT_FSOL, solvent_bsol=SOLVENT_BSOL,
        )

    return peaks, int(L - 1), float(d_min)


def _solutions(peaks: List["RotationPeak"], lmax: int, d_min: float,
               model_error_A: float) -> RotationSolutions:
    """Package a peak list as the public return type."""
    from .frf.rotation_utils import rotation_matrix_from_edmonds_euler_batch

    euler = torch.tensor(
        [[p.alpha, p.beta, p.gamma] for p in peaks], dtype=torch.float64,
    ).reshape(-1, 3)
    rotations = (
        rotation_matrix_from_edmonds_euler_batch(
            euler[:, 0], euler[:, 1], euler[:, 2],
        )
        if euler.numel()
        else torch.zeros((0, 3, 3), dtype=torch.float64)
    )
    return RotationSolutions(
        rotations=rotations,
        scores=torch.tensor([p.score for p in peaks], dtype=torch.float64),
        z_scores=torch.tensor([p.sigma for p in peaks], dtype=torch.float64),
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
    if not model.initialized:
        raise RuntimeError("model has no coordinates; load a PDB first.")
    d_max_fit, d_min_fit = ANISO_FIT_WINDOW_A
    U_aniso = fit_anisotropy(
        data, d_min=d_min_fit, d_max=d_max_fit, device=model.xyz().device,
    )
    peaks, lmax, d_min = search_peaks(
        model, data, model_error_A,
        U_aniso=U_aniso, n_peaks=n_peaks, verbose=verbose,
    )
    return _solutions(peaks, lmax, d_min, model_error_A)
