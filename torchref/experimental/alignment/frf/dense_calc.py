"""Dense P1-box sampling of a model's molecular transform.

The Fast Rotation Function correlates the obs
Patterson against the *model* transform, and sampling that transform at the
sparse crystal lattice under-determines the high-l spherical-harmonic modes for
large molecules. Phaser avoids this by computing the model transform on a dense,
oversampled P1-box FFT grid (``EnsemblePDB.cc:122-135``). This module does the
same: drop the (single, un-symmetry-expanded) model into a cubic P1 box and
reuse ``ModelFT``'s own structure-factor machinery (real ITC92 form factors +
per-atom B/occ) to sample ``|F_calc|`` on the box's dense reciprocal grid.

Unlike the original benchmark helper this operates on ``model.copy()`` so the
caller's ``cell``/``spacegroup``/``max_res`` are never mutated, and the whole SF
build runs under ``torch.no_grad()`` (the load-bearing memory fix — a forward-only
SF build otherwise accumulates a backward graph and OOMs on big grids).
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Tuple

import torch

if TYPE_CHECKING:
    from torchref.model import ModelFT


def dense_calc_via_box(
    model: "ModelFT",
    d_max: float,
    d_min: float,
    *,
    pad: float = 2.0,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample the model transform on a dense P1-box grid in ``[d_max, d_min]``.

    Parameters
    ----------
    model : ModelFT
        Search model (in whatever orientation the rotation search should treat
        as the reference). **Not mutated** — a ``model.copy()`` is used internally.
    d_max, d_min : float
        Low- and high-resolution limits (Å) for the returned reflections.
    pad : float, optional
        P1-box edge as a multiple of the molecular diameter (default 2.0, the
        validated v19 value). A bigger box = finer reciprocal sampling.
    verbose : bool, optional
        If True, print a one-line ``[DENSE_FT]`` summary (box edge / grid size).

    Returns
    -------
    (s_vec, F_calc) : Tuple[torch.Tensor, torch.Tensor]
        ``s_vec`` is the Cartesian reciprocal grid (N, 3) in double -- the
        expansion clusters on it and needs exact keys -- and ``F_calc`` the
        amplitudes (N,) in the model's dtype, both on the model's device.
    """
    from torchref.symmetry.cell import Cell

    # Isolate the box mutation from the caller: the three setters below replace
    # the FFT submodule, so anything the copy set up for the crystal cell is
    # thrown away. That used to cost 104 ms on 3K7M building a 250**3 x 3
    # coordinate grid the box never looks at; `real_space_grid` is built on
    # demand now rather than stored, so the copy is cheap.
    m = model.copy()
    with torch.no_grad():
        coords = m.xyz()
        dev = coords.device
        # Cubic P1 box sized to ``pad`` diameters; no symmetry keeps the grid small.
        extent = (coords - coords.mean(0)).norm(dim=-1).max().item()
        a = float(pad * 2.0 * extent)
        # The grid is derived lazily from (cell, space group, max_res) on first
        # use, so the order of these assignments no longer matters.
        m.max_res = float(d_min)
        m.spacegroup = "P 1"
        m.cell = Cell([a, a, a, 90.0, 90.0, 90.0], device=dev)

        nmax = int(math.ceil(a / d_min))
        idx = torch.arange(-nmax, nmax + 1, device=dev)
        H, K, Lg = torch.meshgrid(idx, idx, idx, indexing="ij")
        hkl = torch.stack(
            [H.reshape(-1), K.reshape(-1), Lg.reshape(-1)], dim=-1
        ).to(torch.long)  # dtype-ok: Miller indices are integers
        # Cubic box: |s| = |hkl| / a.
        smag = hkl.to(torch.float64).norm(dim=-1) / a  # dtype-ok: exact clustering key; needs double
        keep = (smag >= 1.0 / d_max) & (smag <= 1.0 / d_min)
        hkl = hkl[keep].contiguous()
        F = model_sf_abs(m, hkl)
        s_vec = hkl.to(torch.float64) / a  # dtype-ok: exact clustering key; needs double

    if verbose:
        print(
            f"[DENSE_FT] box={a:.0f}A n_grid={hkl.shape[0]} max_res={d_min:.2f}",
            flush=True,
        )
    return s_vec, F


def model_sf_abs(model: "ModelFT", hkl: torch.Tensor) -> torch.Tensor:
    """``|F_calc|`` for ``hkl`` via the model's SF machinery (no grad), in the model's dtype."""
    with torch.no_grad():
        return model.get_structure_factor(hkl, recalc=True).abs()
