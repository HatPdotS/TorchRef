"""
Ensemble atomic model: ~100 coordinate copies of the same chemistry
sharing one Fourier transform.

Implementation strategy
-----------------------
We subclass :class:`~torchref.model.model_ft.ModelFT` but **do not** keep
N separate ModelFT instances. Instead, the single underlying ModelFT is
loaded with the input PDB replicated ``n_members`` times into one flat
atom list, with occupancy ``1/N`` on every atom and a constant frozen
B-factor. The existing ``SfFFT`` machinery sums atom contributions
weighted by occupancy, so it already produces

    F_calc = (1/N) * Σ_i F_i(xyz_i)

with no new aggregation layer required. One FFT call per forward replaces
the N FFT calls that a ``MixedModel``-style wrapper would need at
``n_members ~ 100``.

Coordinates are stored as one flat ``MixedTensor`` of shape
``(N * n_atoms_per_member, 3)`` where rows ``[i*n_atoms:(i+1)*n_atoms]``
are member ``i``. The :attr:`xyz_per_member` property exposes a
``(N, n_atoms, 3)`` view (no copy) for downstream code that needs
per-member access (Amber per-member energy, ensemble entropy/spread terms).

Restraints / topology
---------------------
The model also keeps ``_pdb_single`` — the original single-copy
DataFrame — so restraint builders and Amber topology code can operate
on one chemistry and apply per-member via ``xyz_per_member[i]`` rather
than fabricating cross-member bonds from the flat replicated DataFrame.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch

from torchref.config import get_default_device, get_float_dtype
from torchref.io import pdb as pdb_io
from torchref.model.model_ft import ModelFT
from torchref.symmetry import Cell


def _strip_altlocs_df(df: pd.DataFrame) -> pd.DataFrame:
    """Drop alternate conformations from a PDB DataFrame.

    For each residue with multiple altlocs, keeps the conformer with the
    highest mean occupancy (ties broken alphabetically) and clears the
    ``altloc`` column. Ensemble refinement models disorder via the ensemble
    itself, not via PDB altlocs — leaving altlocs in would double-count
    atoms downstream (OpenMM topology, FFT grid, restraints). Mirrors
    :meth:`Model.strip_altlocs` so single-copy and ensemble paths behave
    identically.
    """
    has_altloc = df["altloc"].astype(str).str.strip() != ""
    if not has_altloc.any():
        out = df.copy()
        out["altloc"] = ""
        return out

    drop_idx: list = []
    res_cols = ["chainid", "resseq", "icode", "resname"]
    altloc_rows = df.loc[has_altloc]
    for _, grp in altloc_rows.groupby(res_cols):
        altlocs = sorted(grp["altloc"].unique())
        if len(altlocs) <= 1:
            continue
        best, best_occ = altlocs[0], -1.0
        for al in altlocs:
            occ = grp.loc[grp["altloc"] == al, "occupancy"].mean()
            if occ > best_occ:
                best, best_occ = al, occ
        for al in altlocs:
            if al != best:
                drop_idx.extend(grp.index[grp["altloc"] == al].tolist())
    filtered = df.drop(index=drop_idx).reset_index(drop=True)
    filtered["altloc"] = ""
    return filtered


def _replicate_and_perturb_pdb(
    df: pd.DataFrame,
    n_members: int,
    perturb_sigma: float,
    b_const: float,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """
    Replicate ``df`` ``n_members`` times with optional coordinate jitter.

    Each copy gets B-factor = ``b_const``, occupancy = ``1/N``, anisotropy
    cleared, and ``serial`` re-numbered to avoid collisions across copies.
    """
    if rng is None:
        rng = np.random.default_rng()
    n_atoms = len(df)
    pieces = []
    for i in range(n_members):
        copy = df.copy(deep=True)
        if perturb_sigma > 0:
            noise = rng.normal(0.0, perturb_sigma, size=(n_atoms, 3))
            copy["x"] = copy["x"].astype(float) + noise[:, 0]
            copy["y"] = copy["y"].astype(float) + noise[:, 1]
            copy["z"] = copy["z"].astype(float) + noise[:, 2]
        copy["tempfactor"] = float(b_const)
        copy["occupancy"] = 1.0 / float(n_members)
        # Clear anisotropy: ensemble disorder replaces it.
        for c in ("u11", "u22", "u33", "u12", "u13", "u23"):
            if c in copy.columns:
                copy[c] = 0.0
        if "anisou_flag" in copy.columns:
            copy["anisou_flag"] = False
        # Re-number serial across the concatenated set so they remain unique.
        copy["serial"] = np.arange(n_atoms) + i * n_atoms + 1
        pieces.append(copy)
    out = pd.concat(pieces, ignore_index=True)
    return out


def _process_model_df(df, strip_H: bool):
    """Apply the standard H-strip / altloc-strip / dropna to a parsed model df."""
    if strip_H:
        df = df.loc[
            df["element"].astype(str).str.strip() != "H"
        ].reset_index(drop=True)
    df = _strip_altlocs_df(df)
    df.dropna(subset=["x", "y", "z", "tempfactor", "occupancy"], inplace=True)
    return df.reset_index(drop=True)


def _load_model_block(header_lines: list, body_lines: list, strip_H: bool):
    """Parse one MODEL block via a temp single-model PDB + the full loader.

    Robust but expensive (one gemmi parse + a temp-file round-trip). Used for
    the chemistry-template model and as the fallback when the fast coord-only
    path can't be verified.
    """
    import tempfile
    import os as _os

    with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as tmp:
        tmp.writelines(header_lines)
        tmp.writelines(body_lines)
        tmp.write("END\n")
        tmp_path = tmp.name
    try:
        df = pdb_io.load_as_dataframe(tmp_path)
    finally:
        _os.unlink(tmp_path)
    return _process_model_df(df, strip_H)


def _coords_from_atom_lines(lines: list, strip_H: bool) -> np.ndarray:
    """Extract (N,3) xyz from raw ATOM/HETATM lines using fixed PDB columns.

    Cheap, in-memory — no gemmi, no temp files. H atoms are skipped when
    ``strip_H`` (element col 77-78, falling back to a name-based H test when
    the element column is blank). Returns coords in file order; correctness vs
    the full loader is verified by the caller before this path is trusted.
    """
    xs: list = []
    for ln in lines:
        if not (ln.startswith("ATOM") or ln.startswith("HETATM")):
            continue
        if strip_H:
            elem = ln[76:78].strip() if len(ln) >= 78 else ""
            if elem in ("H", "D"):
                continue
            if elem == "" and ln[12:16].strip()[:1] == "H":
                continue
        try:
            xs.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
        except (ValueError, IndexError):
            continue
    return np.asarray(xs, dtype=float) if xs else np.empty((0, 3))


def _parse_multi_model_pdb(filepath: str, strip_H: bool = True) -> list:
    """
    Split a multi-MODEL PDB into a list of single-model DataFrames.

    Ensemble members share identical chemistry (same atoms/order), so only the
    first model is parsed the full (gemmi) way as a chemistry template; the rest
    reuse that template with coordinates extracted directly from their raw ATOM
    lines — avoiding one gemmi parse + temp-file round-trip *per model* (which is
    catastrophically slow and memory-heavy for large ensembles on a network FS).
    The fast path is only used when it provably reproduces model 0 (matching
    count AND coordinates); otherwise it falls back to the robust per-model load.

    Falls back to ``[single_df]`` when no ``MODEL`` records are found.
    """
    with open(filepath, "r") as f:
        text = f.read()
    lines = text.splitlines(keepends=True)
    if "MODEL " not in text:
        return [_process_model_df(pdb_io.load_as_dataframe(filepath), strip_H)]

    # Header = leading non-MODEL lines (CRYST1, REMARKs); kept per block so the
    # loader finds the unit cell.
    i = 0
    header_lines: list = []
    while i < len(lines) and not lines[i].startswith("MODEL "):
        header_lines.append(lines[i])
        i += 1

    # Collect each MODEL's body lines.
    blocks: list = []
    cur = None
    for line in lines[i:]:
        if line.startswith("MODEL "):
            cur = []
        elif line.startswith("ENDMDL"):
            if cur is not None:
                blocks.append(cur)
            cur = None
        elif cur is not None:
            cur.append(line)
    if not blocks:
        return [_process_model_df(pdb_io.load_as_dataframe(filepath), strip_H)]

    # Model 0: full robust parse = chemistry template.
    df0 = _load_model_block(header_lines, blocks[0], strip_H)
    n0 = len(df0)
    # Verify the fast coord-only path reproduces model 0 exactly (same count and
    # coordinates → atom order + H/altloc filtering are consistent for this file).
    c0 = _coords_from_atom_lines(blocks[0], strip_H)
    xyz0 = df0[["x", "y", "z"]].to_numpy(dtype=float)
    fast_ok = c0.shape == (n0, 3) and np.allclose(c0, xyz0, atol=1e-3)

    models = [df0]
    for blk in blocks[1:]:
        coords = _coords_from_atom_lines(blk, strip_H) if fast_ok else None
        if fast_ok and coords.shape == (n0, 3):
            df = df0.copy(deep=True)
            df["x"], df["y"], df["z"] = coords[:, 0], coords[:, 1], coords[:, 2]
            models.append(df)
        else:
            models.append(_load_model_block(header_lines, blk, strip_H))
    return models


class _SyntheticPDBReader:
    """Minimal reader-compatible wrapper around a prepared DataFrame."""

    def __init__(self, df: pd.DataFrame, cell, spacegroup):
        self.dataframe = df
        self.cell = cell
        self.spacegroup = spacegroup
        self.links = None

    def __call__(self):
        return self.dataframe, self.cell, self.spacegroup


def build_single_copy_model(ensemble, atom_idx=None, verbose: int = 0):
    """Build a single-conformation :class:`~torchref.model.model.Model` from an
    ensemble's single-copy chemistry (``EnsembleModel._pdb_single``).

    Used as the ``chem_model`` for the Amber targets so they build one OpenMM
    topology from a genuine :class:`Model` (no duck-typed shim). Optionally
    restrict to ``atom_idx`` (row indices into the per-member atom layout) to
    drop non-standard or special-position atoms before parameterisation.

    Parameters
    ----------
    ensemble : EnsembleModel
        Source ensemble; supplies ``_pdb_single``, ``cell``, ``spacegroup``,
        ``device``.
    atom_idx : array-like of int, optional
        Rows of ``_pdb_single`` to keep (``None`` ⇒ all atoms).
    verbose : int
        Forwarded to the :class:`Model` constructor.

    Returns
    -------
    Model
        A single-conformation model exposing ``.pdb`` / ``.update_pdb()`` /
        ``.xyz()`` / ``.device`` over the selected atoms.
    """
    from torchref.model.model import Model

    df = ensemble._pdb_single
    if atom_idx is not None:
        df = df.iloc[np.asarray(atom_idx)]
    chem = Model(verbose=verbose, strip_H=False, device=ensemble.device)
    chem.load(
        _SyntheticPDBReader(
            df.reset_index(drop=True).copy(),
            ensemble.cell,
            ensemble.spacegroup,
        )
    )
    return chem


class EnsembleModel(ModelFT):
    """
    Ensemble of ``n_members`` atomic copies behind a single ``ModelFT``.

    Parameters
    ----------
    dtype_float : torch.dtype
        Floating-point dtype.
    verbose : int
        Verbosity.
    device : torch.device
        Computation device.
    strip_H : bool
        Whether to strip hydrogens (inherited).
    max_res : float
        FFT grid target resolution (inherited).
    radius_angstrom : float
        Atomic radius cutoff for the real-space scatter (inherited).

    Notes
    -----
    Instances should be constructed via :meth:`from_single` or
    :meth:`from_multimodel_pdb` rather than ``__init__`` + ``load_pdb``,
    so the replication / perturbation logic is applied consistently.
    """

    def __init__(
        self,
        dtype_float=None,
        verbose: int = 1,
        device=None,
        strip_H: bool = True,
        max_res: float = 1.0,
        radius_angstrom: float = 4.0,
        gridsize: Optional[int] = None,
        wavelength: float = 1.0,
        anomalous_threshold: float = 0.5,
    ):
        if dtype_float is None:
            dtype_float = get_float_dtype()
        if device is None:
            device = get_default_device()
        super().__init__(
            dtype_float=dtype_float,
            verbose=verbose,
            device=device,
            strip_H=strip_H,
            max_res=max_res,
            radius_angstrom=radius_angstrom,
            gridsize=gridsize,
            wavelength=wavelength,
            anomalous_threshold=anomalous_threshold,
        )
        # Filled in by ``_finalize_ensemble`` after ``load`` returns.
        self.n_members: int = 0
        self.n_atoms_per_member: int = 0
        # Single-copy PDB DataFrame, preserved for restraint / topology
        # builders that must not see the flat replicated atom list.
        self._pdb_single: Optional[pd.DataFrame] = None
        # Ensemble dropout (regularization). When active, each structure-
        # factor forward uses a random subset of members (see
        # :meth:`configure_dropout`). The per-atom occupancy multiplier is a
        # buffer (registered in ``_finalize_ensemble``) so the SF cache
        # invalidates correctly when it changes.
        self.dropout_active: bool = False
        self.dropout_min: int = 0
        self.dropout_max: int = 0

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_single(
        cls,
        pdb_path: str,
        n_members: int,
        perturb_sigma: float = 0.01,
        b_const: float = 5.0,
        seed: Optional[int] = None,
        verbose: int = 1,
        device=None,
        strip_H: bool = True,
        max_res: float = 1.0,
        radius_angstrom: float = 4.0,
        n_max: Optional[int] = None,
        **modelft_kwargs,
    ) -> "EnsembleModel":
        """
        Build an ensemble from a single-model PDB by replicate-and-perturb.

        Parameters
        ----------
        pdb_path : str
            Path to input PDB. May contain multiple models — only the first
            is used (use :meth:`from_multimodel_pdb` to consume all).
        n_members : int
            Number of ensemble members.
        perturb_sigma : float
            Std-dev (Å) of Gaussian noise added to xyz of each replicated copy.
            Default 0.01 Å — only large enough to break gradient degeneracy
            between identical copies. Larger values (≥ ~0.05 Å) introduce
            LJ clashes (atoms walking inside vdW radii), which makes any
            downstream force-field restraint (AmberTarget, geometry terms)
            return huge energies / gradients. The ensemble's real disorder
            should develop from the X-ray gradient + entropy regularizer
            during refinement, not from the initial noise.
        b_const : float
            Fixed isotropic B-factor (Å²) for every atom in every member.
            Small but non-zero to avoid FFT grid aliasing.
        seed : int, optional
            RNG seed for the perturbation.
        """
        reader = pdb_io.PDBReader(verbose=verbose).read(pdb_path)
        df, cell, spacegroup = reader()
        if strip_H:
            df = df.loc[df["element"].astype(str).str.strip() != "H"].reset_index(drop=True)
        # Strip alternate conformations: the ensemble IS the disorder model,
        # so per-residue altlocs would double-count atoms in OpenMM topology,
        # FFT, restraints, etc. Mirrors Model.strip_altlocs.
        df = _strip_altlocs_df(df)
        df.dropna(subset=["x", "y", "z", "tempfactor", "occupancy"], inplace=True)

        pool = max(int(n_members), int(n_max) if n_max else int(n_members))
        rng = np.random.default_rng(seed)
        replicated = _replicate_and_perturb_pdb(
            df, n_members=pool, perturb_sigma=perturb_sigma,
            b_const=b_const, rng=rng,
        )

        model = cls(
            verbose=verbose, device=device, strip_H=False,  # already stripped
            max_res=max_res, radius_angstrom=radius_angstrom,
            **modelft_kwargs,
        )
        model._pdb_single = df.reset_index(drop=True).copy()
        synthetic = _SyntheticPDBReader(replicated, cell, spacegroup)
        model.load(synthetic)
        model._finalize_ensemble(n_members=pool, n_atoms_per_member=len(df))
        if pool > n_members:
            model._alive[n_members:] = False
        return model

    @classmethod
    def from_multimodel_pdb(
        cls,
        pdb_path: str,
        n_members: Optional[int] = None,
        perturb_sigma: float = 0.01,
        b_const: float = 5.0,
        seed: Optional[int] = None,
        verbose: int = 1,
        device=None,
        strip_H: bool = True,
        max_res: float = 1.0,
        radius_angstrom: float = 4.0,
        n_max: Optional[int] = None,
        **modelft_kwargs,
    ) -> "EnsembleModel":
        """
        Build an ensemble from a multi-MODEL PDB.

        If the file has at least ``n_members`` MODEL records, the first
        ``n_members`` are used as-is. If it has fewer, the existing models
        are cycled and perturbed to fill out the ensemble. If
        ``n_members`` is None, every MODEL in the file is used.

        ``n_max`` (>= n_members) pre-allocates a fixed pool of member *slots*
        for birth/death population dynamics: the first ``n_members`` slots
        start alive, slots ``[n_members:n_max]`` start dead (perturbed seeds
        ready for bifurcation to reactivate). Default ``n_max = n_members``
        (no spare slots; bifurcation can only reuse slots freed by deaths).
        """
        models = _parse_multi_model_pdb(pdb_path, strip_H=strip_H)
        if len(models) == 0:
            raise ValueError(f"No usable atomic models parsed from {pdb_path}")
        if n_members is None:
            n_members = len(models)

        # Use the first model as the reference single-copy chemistry.
        # All members must have the same atom count and ordering for a flat
        # concatenated tensor to be coherent.
        n_atoms = len(models[0])
        for k, m in enumerate(models):
            if len(m) != n_atoms:
                raise ValueError(
                    f"Multi-MODEL PDB members have inconsistent atom counts: "
                    f"MODEL 1 has {n_atoms}, MODEL {k+1} has {len(m)}. "
                    "Ensemble refinement requires identical chemistry across members."
                )

        # Read cell / spacegroup via a separate reader on the unmodified file.
        cell_reader = pdb_io.PDBReader(verbose=verbose).read(pdb_path)
        cell, spacegroup = cell_reader.cell, cell_reader.spacegroup

        pool = max(int(n_members), int(n_max) if n_max else int(n_members))
        rng = np.random.default_rng(seed)
        pieces: list = []
        for i in range(pool):
            src = models[i % len(models)].copy(deep=True)
            if i >= len(models) and perturb_sigma > 0:
                # Cyclic copies beyond what's in the file: perturb them.
                noise = rng.normal(0.0, perturb_sigma, size=(n_atoms, 3))
                src["x"] = src["x"].astype(float) + noise[:, 0]
                src["y"] = src["y"].astype(float) + noise[:, 1]
                src["z"] = src["z"].astype(float) + noise[:, 2]
            src["tempfactor"] = float(b_const)
            src["occupancy"] = 1.0 / float(n_members)
            for c in ("u11", "u22", "u33", "u12", "u13", "u23"):
                if c in src.columns:
                    src[c] = 0.0
            if "anisou_flag" in src.columns:
                src["anisou_flag"] = False
            src["serial"] = np.arange(n_atoms) + i * n_atoms + 1
            pieces.append(src)
        replicated = pd.concat(pieces, ignore_index=True)

        model = cls(
            verbose=verbose, device=device, strip_H=False,
            max_res=max_res, radius_angstrom=radius_angstrom,
            **modelft_kwargs,
        )
        model._pdb_single = models[0].reset_index(drop=True).copy()
        synthetic = _SyntheticPDBReader(replicated, cell, spacegroup)
        model.load(synthetic)
        model._finalize_ensemble(n_members=pool, n_atoms_per_member=n_atoms)
        if pool > n_members:
            model._alive[n_members:] = False
        return model

    def _finalize_ensemble(self, n_members: int, n_atoms_per_member: int) -> None:
        """Stamp ensemble metadata and freeze everything except xyz."""
        self.n_members = int(n_members)
        self.n_atoms_per_member = int(n_atoms_per_member)
        # Per-atom occupancy multiplier for dropout. All-ones = full ensemble
        # (no dropout). ``resample_dropout`` rewrites it in place.
        self.register_buffer(
            "_dropout_occ_mult",
            torch.ones(
                n_members * n_atoms_per_member,
                dtype=self.dtype_float,
                device=self.device,
            ),
        )
        # ---- Population-dynamics state (opt-in via ``enable_population_refinement``)
        # Per-atom -> member index for the member-contiguous flat layout, so a
        # per-member (N,) vector expands to per-atom with one gather.
        self.register_buffer(
            "_member_index",
            torch.arange(
                n_members * n_atoms_per_member, device=self.device
            ).div(int(n_atoms_per_member), rounding_mode="floor"),
        )
        # ``alive`` marks which member slots contribute. Death flips a slot
        # dead (softmax weight -> 0); bifurcation reactivates a dead slot as a
        # perturbed copy of a parent. Shapes are FIXED at n_members (= N_max),
        # so no tensor/optimizer surgery is ever needed.
        self.register_buffer(
            "_alive", torch.ones(n_members, dtype=torch.bool, device=self.device)
        )
        # Per-member occupancy logits (softmax over alive -> Σw=1) and ADP raw
        # (softplus -> B>0). Created always (so state_dict shapes are stable)
        # but only consumed in get_iso / the optimizer when population
        # refinement is enabled.
        self._refine_population: bool = False
        self._refine_member_b: bool = False
        # Frozen at construction (requires_grad=False); a fresh ensemble refines
        # xyz only. enable_population_refinement() flips these refinable when the
        # opt-in feature is turned on.
        self.occ_logits = torch.nn.Parameter(
            torch.zeros(n_members, dtype=self.dtype_float, device=self.device),
            requires_grad=False,
        )
        with torch.no_grad():
            # Per-member initial B = mean of that member's per-atom B (all
            # equal to b_const at construction); store its softplus-inverse.
            b_per_member = (
                self.adp().detach()
                .reshape(n_members, n_atoms_per_member)
                .mean(dim=1)
                .clamp_min(1e-3)
            )
            b_raw0 = torch.log(torch.expm1(b_per_member).clamp_min(1e-6))
        self.b_raw = torch.nn.Parameter(
            b_raw0.to(self.dtype_float), requires_grad=False
        )
        # Only xyz refines — B-factors fixed (ensemble spread IS the disorder),
        # anisotropic U is unused, and occupancy is fixed at 1/N.
        for tgt in ("adp", "u", "occupancy"):
            try:
                self.freeze(tgt)
            except Exception:
                if self.verbose > 0:
                    print(f"  EnsembleModel: freeze({tgt!r}) failed (ignored)")

    # ------------------------------------------------------------------
    # Per-member occupancy + ADP + birth/death population dynamics
    # ------------------------------------------------------------------

    def enable_population_refinement(
        self, enable: bool = True, refine_b: bool = False
    ) -> None:
        """Turn per-member occupancy (and optionally ADP) injection on/off.

        When on, ``get_iso``/``get_aniso`` substitute the live per-member
        softmax occupancy ``w_m`` for the frozen 1/N, so
        ``F̄ = Σ_{m alive} w_m·DWF(B_m)·F_m`` and gradients flow to
        ``occ_logits``. With ``refine_b=True`` the per-member softplus ADP
        ``B_m`` is also injected (gradients to ``b_raw``); otherwise B stays
        frozen at ``b_const`` — the default, because free B drives the
        weighted members to B=0 (delta-function overfit; a hard stability
        risk on the FFT grid). Refinement code must add the matching
        parameters to the optimizer.
        """
        self._refine_population = bool(enable)
        self._refine_member_b = bool(refine_b)
        # Flip the per-member levers refinable exactly when the feature is on
        # (they are frozen at construction so a fresh ensemble refines xyz only).
        self.occ_logits.requires_grad_(bool(enable))
        self.b_raw.requires_grad_(bool(enable) and bool(refine_b))
        self.reset_cache()

    @property
    def n_alive(self) -> int:
        """Number of currently-alive member slots (the effective ensemble size)."""
        a = getattr(self, "_alive", None)
        return int(a.sum().item()) if a is not None else int(self.n_members)

    def member_weights(self) -> torch.Tensor:
        """Per-member occupancy ``w_m = softmax(occ_logits | alive)`` (dead -> 0)."""
        logits = self.occ_logits
        a = getattr(self, "_alive", None)
        if a is not None:
            logits = logits.masked_fill(~a, float("-inf"))
        return torch.softmax(logits, dim=0)

    def member_b(self) -> torch.Tensor:
        """Per-member isotropic B-factor ``B_m = softplus(b_raw)`` (>0)."""
        return torch.nn.functional.softplus(self.b_raw)

    def kill_member(self, idx: int) -> bool:
        """Flip member ``idx`` dead (weight redistributes via alive-softmax).

        Refuses to kill the last alive member. Returns True if a kill happened.
        """
        a = self._alive
        if not bool(a[idx]) or int(a.sum().item()) <= 1:
            return False
        a[idx] = False
        self.reset_cache()
        return True

    def bifurcate_member(self, parent_idx: int, sigma: float = 0.1) -> int:
        """Activate a free (dead) slot as a perturbed copy of ``parent_idx``.

        Child xyz = parent xyz + N(0, sigma) (symmetry break); the parent's
        weight is split between the two (``logit -= ln2`` on both); child B
        copies the parent's. Returns the reborn slot index, or -1 if the pool
        is full (no dead slot available).
        """
        a = self._alive
        free = (~a).nonzero(as_tuple=False).flatten()
        if free.numel() == 0:
            return -1
        d = int(free[0].item())
        n_at = int(self.n_atoms_per_member)
        with torch.no_grad():
            flat = self.xyz.refinable_params  # (N_max*n_atoms, 3)
            ps, pe = parent_idx * n_at, (parent_idx + 1) * n_at
            ds, de = d * n_at, (d + 1) * n_at
            flat[ds:de] = flat[ps:pe] + torch.randn_like(flat[ps:pe]) * float(sigma)
            ln2 = float(np.log(2.0))
            self.occ_logits[d] = self.occ_logits[parent_idx] - ln2
            self.occ_logits[parent_idx] = self.occ_logits[parent_idx] - ln2
            self.b_raw[d] = self.b_raw[parent_idx]
            a[d] = True
        self.reset_cache()
        return d

    # ------------------------------------------------------------------
    # Low-rank (frozen-basis PCA) reparameterization
    # ------------------------------------------------------------------

    def enable_low_rank(self, K: int) -> float:
        """Swap ``self.xyz`` to a frozen-basis low-rank parameterization.

        Computes a PCA of the current ensemble's per-member coordinates,
        freezes the mean ``mu`` and the top-``K`` principal modes ``V`` as
        buffers, and replaces the flat ``MixedTensor`` with a
        :class:`~torchref.experimental.ensemble.low_rank_ensemble.LowRankXYZ` whose only
        refinable leaf is the per-member amplitudes ``A`` (shape ``(N, K)``).
        Degrees of freedom collapse from ``N·n_atoms·3`` to ``N·K``.

        The current coordinates ARE the basis source, so this must be called
        after the ensemble is seeded with real disorder (e.g. after a
        ``--branch-from`` overlay) — a fresh replicate-and-perturb ensemble
        has only ~``perturb_sigma`` of near-degenerate spread.

        Parameters
        ----------
        K : int
            Number of principal modes to retain. Clamped to
            ``min(K, n_members - 1)`` (the centered data has rank ≤ N−1).

        Returns
        -------
        float
            Cumulative fraction of ensemble coordinate variance captured by
            the retained ``K`` modes.
        """
        from .low_rank_ensemble import LowRankXYZ

        N = int(self.n_members)
        n_atoms = int(self.n_atoms_per_member)
        K = int(K)
        max_rank = max(1, N - 1)
        if K > max_rank:
            if self.verbose > 0:
                print(
                    f"  EnsembleModel.enable_low_rank: K={K} exceeds rank "
                    f"N-1={max_rank}; clamping to {max_rank}."
                )
            K = max_rank

        with torch.no_grad():
            flat = self.xyz().detach()                       # (N*n_atoms, 3)
            X = flat.reshape(N, n_atoms * 3).to(torch.float64)
            mu = X.mean(dim=0)                               # (D,)
            Xc = X - mu.unsqueeze(0)
            # full_matrices=False → Vt is (min(N, D), D); S length min(N, D).
            U, S, Vt = torch.linalg.svd(Xc, full_matrices=False)
            Vk = Vt[:K]                                      # (K, D)
            A0 = Xc @ Vk.T                                   # (N, K)
            total_var = (S ** 2).sum().clamp_min(1e-30)
            explained = float((S[:K] ** 2).sum() / total_var)

        dtype = self.dtype_float
        lowrank = LowRankXYZ(
            mu=mu.to(dtype),
            V=Vk.to(dtype),
            amplitudes=A0.to(dtype),
            n_members=N,
            n_atoms=n_atoms,
            explained_variance=explained,
        ).to(self.device)
        self.xyz = lowrank
        self.reset_cache()

        if self.verbose > 0:
            print(
                f"  EnsembleModel.enable_low_rank: K={K} modes, "
                f"DOF {N * n_atoms * 3} -> {N * K} "
                f"({N}x{K} amplitudes), "
                f"explained variance = {explained * 100:.2f}%"
            )
        return explained

    def enable_pca(self, K: Optional[int] = None) -> float:
        """Swap ``self.xyz`` to a fully-refinable PCA parameterization.

        Like :meth:`enable_low_rank` but the mean ``mu``, basis ``V`` AND
        amplitudes ``A`` all refine (see
        :class:`~torchref.experimental.ensemble.pca_model.PCAEnsembleParam`). ``K=None`` → the
        full rank ``N-1`` (complete reparameterization). Must be called after
        the ensemble carries real disorder (e.g. after a ``--branch-from``).
        Returns the cumulative explained-variance fraction at seed time.
        """
        from .pca_model import PCAEnsembleParam

        N = int(self.n_members)
        n_atoms = int(self.n_atoms_per_member)
        with torch.no_grad():
            flat = self.xyz().detach()
        pca = PCAEnsembleParam.from_ensemble(
            flat, n_members=N, n_atoms=n_atoms, K=K,
        )
        self.xyz = pca.to(self.device)
        self.reset_cache()
        if self.verbose > 0:
            print(
                f"  EnsembleModel.enable_pca: K={self.xyz.K} modes (refine μ,A,V), "
                f"explained variance = {self.xyz.explained_variance * 100:.2f}%"
            )
        return self.xyz.explained_variance

    # ------------------------------------------------------------------
    # Dropout (member-subset regularization)
    # ------------------------------------------------------------------

    def configure_dropout(
        self,
        active: bool,
        dropout_min: Optional[int] = None,
        dropout_max: Optional[int] = None,
    ) -> None:
        """Enable/disable ensemble dropout.

        When active, each structure-factor forward uses a random subset of
        ``[dropout_min, dropout_max]`` members (rescaled to ``1/|S|`` so the
        coherent average stays unbiased), broken via :meth:`resample_dropout`.
        Members not in the subset get zero occupancy that forward, hence no
        X-ray/Wilson gradient — but their geometry is still restrained by
        Amber (which reads coordinates, not occupancy). This breaks the member
        co-adaptation that lets an overparameterized ensemble memorize
        work-set noise. Disabling restores the full ``1/N`` average.
        """
        self.dropout_active = bool(active)
        if dropout_min is not None:
            self.dropout_min = int(dropout_min)
        if dropout_max is not None:
            self.dropout_max = int(dropout_max)
        if not self.dropout_active:
            self.set_dropout_full()

    def set_dropout_full(self) -> None:
        """Reset the occupancy multiplier to all-ones (full ensemble)."""
        if getattr(self, "_dropout_occ_mult", None) is not None:
            self._dropout_occ_mult.fill_(1.0)

    def resample_dropout(self) -> int:
        """Draw a fresh member subset and rewrite the occupancy multiplier.

        Picks ``k ~ U[dropout_min, dropout_max]`` members uniformly at random,
        sets their per-atom multiplier to ``N/k`` (so effective occupancy is
        ``(1/N)·(N/k) = 1/k`` and the subset average is unbiased) and the rest
        to 0. No-op when dropout is inactive. Returns ``k`` (or ``N`` when
        inactive).
        """
        if not self.dropout_active or self._dropout_occ_mult is None:
            return self.n_members
        N = self.n_members
        lo = max(1, int(self.dropout_min))
        hi = min(int(self.dropout_max), N)
        if hi < lo:
            hi = lo
        k = int(torch.randint(lo, hi + 1, (1,)).item())
        dev = self._dropout_occ_mult.device
        dt = self._dropout_occ_mult.dtype
        keep = torch.zeros(N, device=dev, dtype=dt)
        keep[torch.randperm(N, device=dev)[:k]] = float(N) / float(k)
        self._dropout_occ_mult.copy_(keep.repeat_interleave(self.n_atoms_per_member))
        return k

    def _apply_dropout(self, occupancy: torch.Tensor) -> torch.Tensor:
        """Scale ``occupancy`` by the dropout multiplier when shapes align.

        ``get_iso`` returns occupancy for the full (all-isotropic) ensemble,
        so the multiplier aligns; ``get_aniso`` is empty for the ensemble, so
        the shape guard skips it harmlessly.
        """
        m = getattr(self, "_dropout_occ_mult", None)
        if m is not None and occupancy.numel() == m.numel():
            return occupancy * m
        return occupancy

    def _inject_population(self, occupancy: torch.Tensor, adp: torch.Tensor):
        """Substitute live per-member softmax occupancy / softplus ADP.

        Only when population refinement is on and the per-atom vector aligns
        with the full ensemble layout (``get_iso`` covers all atoms; the
        all-isotropic ensemble leaves ``get_aniso`` empty, so its shorter
        vector simply skips the swap). The returned tensors are live, so
        autograd reaches ``occ_logits``/``b_raw`` through the FFT SF path.
        """
        if not getattr(self, "_refine_population", False):
            return occupancy, adp
        if occupancy.numel() != self._member_index.numel():
            return occupancy, adp
        idx = self._member_index
        occupancy = self.member_weights()[idx]
        # B stays frozen at b_const unless explicitly refined (free B -> 0 on
        # the weighted members = delta-function overfit; default off).
        if getattr(self, "_refine_member_b", False):
            adp = self.member_b()[idx]
        return occupancy, adp

    def get_iso(self):
        xyz, adp, occupancy, A, B = super().get_iso()
        occupancy, adp = self._inject_population(occupancy, adp)
        return xyz, adp, self._apply_dropout(occupancy), A, B

    def get_aniso(self):
        xyz, u, occupancy, A, B = super().get_aniso()
        occupancy, _ = self._inject_population(occupancy, u)
        return xyz, u, self._apply_dropout(occupancy), A, B

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------

    @property
    def xyz_per_member(self) -> torch.Tensor:
        """
        ``(N, n_atoms_per_member, 3)`` view onto the flat xyz tensor.

        Live: gradients computed against this view backpropagate into the
        underlying ``MixedTensor`` parameters via the standard autograd path
        (the view is a reshape of the same storage).
        """
        flat = self.xyz()
        return flat.view(self.n_members, self.n_atoms_per_member, 3)

    @property
    def pdb_single(self) -> pd.DataFrame:
        """Single-copy PDB DataFrame for restraint / topology builders."""
        return self._pdb_single

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def write_pdb(self, filepath: str) -> None:
        """
        Write the ensemble as a multi-MODEL PDB.

        Each ensemble member becomes one MODEL record. The single-copy
        chemistry from ``self._pdb_single`` is used as the row template;
        per-member coordinates come from ``xyz_per_member``.
        """
        if self._pdb_single is None:
            raise RuntimeError(
                "EnsembleModel has no single-copy PDB; was it built from a "
                "from_single / from_multimodel_pdb classmethod?"
            )
        coords = self.xyz_per_member.detach().cpu().numpy()
        dfs = []
        for i in range(self.n_members):
            df = self._pdb_single.copy(deep=True)
            df["x"] = coords[i, :, 0]
            df["y"] = coords[i, :, 1]
            df["z"] = coords[i, :, 2]
            df["tempfactor"] = float(self.adp().detach().cpu().numpy()[0]) \
                if self.adp is not None else 5.0
            df["occupancy"] = 1.0 / self.n_members
            # Attach cell/spacegroup to first frame for the writer's CRYST1.
            if i == 0:
                if self.cell is not None:
                    df.attrs["cell"] = self.cell.data.detach().cpu().numpy().tolist()
                if self.spacegroup is not None:
                    df.attrs["spacegroup"] = self.spacegroup.hm
                df.attrs["z"] = ""
            dfs.append(df)
        pdb_io.write_multi_model(dfs, filepath)
