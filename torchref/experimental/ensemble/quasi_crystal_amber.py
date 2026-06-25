"""
Quasi-crystal Amber target for ensemble refinement.

Replaces :class:`EnsembleAmberKLTarget`'s per-member Python loop (100
sequential OpenMM round-trips) with a single unified OpenMM ``System``: the
asymmetric unit is symmetry-expanded by the spacegroup within each small
cell, and the supercell tiles the ensemble's disorder copies along axis a
(k × 1 × 1, where ``k = n_members / N_sym``). One ``setPositions ->
getState`` round-trip covers all members. PME for electrostatics under the
supercell PBC; inter-member nonbondeds are now physically meaningful crystal
contacts and act as the regularizer.

No entropy / KL term — the original ``EnsembleAmberKLTarget`` used a
per-atom variance over members as an anti-collapse prior, but in the
quasi-crystal layout members live in *different* small cells of the
supercell, so cross-member variance is dominated by the tile offset and
no longer reflects "disorder spread." Crystal contacts in OpenMM provide
the physical anti-collapse.

Design
------
Subclass of :class:`~torchref.experimental.targets.amber_target.AmberTarget`. The
base ``__init__`` builds the single-molecule chemistry against a genuine
single-conformation :class:`Model` (the ensemble's ``_pdb_single`` restricted to
non-special-position atoms), giving us as **inherited** state:

- the antechamber pipeline + GAFF2 setup for non-standard residues;
- the template OpenMM ``System`` (single-molecule, AMBER14 / GAFF2);
- the H virtual-site frame tables (``_build_h_attachment``) and the shared
  local-frame placement (``_place_hydrogens_local_frame``);
- the autograd Function ``_OpenMMAMBERFunction``.

This target then replicates the template ``System`` into the symmetry-expanded
supercell and replaces the (single-molecule) context with a PME supercell one.

After the template is built we discard most of the temporary target's
state and:

1. replicate the System into a supercell with
   :func:`_replicate_to_supercell_system`;
2. build a new ``Context`` on the supercell (CUDA > OpenCL > CPU);
3. tile the template's atom map + H-attachment indices per member.

Forward (implemented in a follow-up increment) reads
``ensemble.xyz_per_member``, applies the supercell layout's sym+tile
transform, scatters into the unified OpenMM position tensor (heavy via
``_compose_full_omm_xyz``-style scatter; H via the tiled local-frame
placement), and calls the same ``_OpenMMAMBERFunction.apply``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np
import torch

from torchref.experimental.targets.amber_target import (
    AmberTarget,
    _OpenMMAMBERFunction,
    _place_hydrogens_local_frame,
)
from .ensemble_model import build_single_copy_model
from .supercell import SupercellLayout, _replicate_to_supercell_system


def _detect_special_position_atoms(
    xyz_ang: np.ndarray,
    cell_matrix: np.ndarray,
    sym_R: np.ndarray,
    sym_t: np.ndarray,
    threshold_ang: float = 0.5,
) -> np.ndarray:
    """Return a boolean mask of ASU atoms NOT on any spacegroup symmetry element.

    An atom is "on a symmetry element" if applying ANY non-identity sym op
    to its position (+ PBC wrap) brings the result to within
    ``threshold_ang`` of the original atom. Such atoms acquire multiplicity
    in the unit cell — the crystallographer's convention is to deposit them
    with fractional occupancy (1 / site-symmetry order). When we sym-expand
    the full ASU into the supercell with full occupancy, special-position
    atoms become N coincident copies → LJ explosion.

    Parameters
    ----------
    xyz_ang : (n, 3) np.ndarray
        ASU atom positions in Cartesian (Å).
    cell_matrix : (3, 3) np.ndarray
        Cell ``B`` such that ``cart = B @ frac`` (columns are lattice vectors).
    sym_R : (n_sym, 3, 3) np.ndarray
        Spacegroup rotation matrices in fractional coordinates.
    sym_t : (n_sym, 3) np.ndarray
        Spacegroup translations in fractional coordinates.
    threshold_ang : float
        Atoms whose self-image distance (under ANY non-identity op + PBC) is
        below this are flagged as special-position. 0.5 Å catches the
        crystallographically-on-sym-element cases without flagging genuine
        close crystal contacts (salt bridges, H-bonds at ~2.3-2.7 Å).

    Returns
    -------
    keep_mask : (n,) bool np.ndarray
        True for atoms that are NOT on any sym element — safe to sym-expand.
    """
    B_inv = np.linalg.inv(cell_matrix)
    xyz_frac = xyz_ang @ B_inv.T
    keep = np.ones(xyz_ang.shape[0], dtype=bool)
    # Skip identity (op 0); any subsequent op that maps an atom near itself
    # (within threshold) means that atom is on a sym element.
    for j in range(1, sym_R.shape[0]):
        sym_frac = xyz_frac @ sym_R[j].T + sym_t[j]
        diff_frac = xyz_frac - sym_frac
        diff_frac -= np.round(diff_frac)  # minimum image
        diff_cart = diff_frac @ cell_matrix.T
        d = np.linalg.norm(diff_cart, axis=-1)
        keep &= d > threshold_ang
    return keep

if TYPE_CHECKING:
    from .ensemble_model import EnsembleModel
    from torchref.symmetry import Cell, SpaceGroup


class QuasiCrystalAmberTarget(AmberTarget):
    """Amber energy on a k × 1 × 1 supercell with full crystal sym expansion.

    The ensemble is laid out as ``k = n_members / N_sym`` disorder copies
    tiled along the small cell's a-axis; each small cell holds the
    spacegroup's full N_sym sym mates. PBC + PME for electrostatics.

    The model is an :class:`EnsembleModel`; cell + spacegroup typically come
    from the same :class:`ReflectionData` the X-ray side uses.

    Parameters
    ----------
    model : EnsembleModel
        Ensemble with ``n_members = n_disorder * N_sym``.
    cell : Cell
        Small unit cell. ``cell.fractional_matrix`` is used (column-vector
        convention: ``r_cart = B @ r_frac``).
    spacegroup : SpaceGroup
        Spacegroup with ``n_ops`` symmetry operators, ``matrices`` (N_sym, 3, 3)
        rotations in fractional coordinates, and ``translations`` (N_sym, 3).
    n_disorder : int
        Number of disorder copies tiled along axis a (k ≥ 1). Must satisfy
        ``n_disorder * spacegroup.n_ops == model.n_members``.
    pme_cutoff_ang : float
        Nonbonded cutoff in Å for PME. 10 Å is the standard for proteins.
    ewald_error_tolerance : float
        PME accuracy tolerance (5e-4 is OpenMM's production default).
    normalize_per_asu : bool
        If True (default), ``forward()`` returns supercell energy ÷ number of
        ASU copies in the supercell (= ``n_disorder · N_sym = model.n_members``),
        i.e. the per-ASU total energy in kJ/mol. This puts the loss on a
        "single structure" scale that's directly comparable to per-ASU X-ray
        and Wilson NLLs in ``EnsembleRefinement._create_loss_state``.
        If False, ``forward()`` returns the raw supercell total energy.
    residue_charges : dict, optional
        Net charges for non-standard residues; passed to the internal
        :class:`AmberTarget`.
    gaff2_files : dict, optional
        Pre-built GAFF2 parameter files per residue; passed through.
    charge_method : str
        antechamber charge method ('gas' or 'bcc'). Default 'gas' (fast,
        no QM); matches the ensemble setup.
    verbose : int
        Verbosity (0 = silent, 1 = setup messages).

    Notes
    -----
    Atom layout in the unified OpenMM System:

      atom_omm_index = m * n_omm_per_member + i

    where ``m = d * N_sym + j`` (d = disorder/tile, j = sym mate), so members
    are block-ordered. This matches :attr:`EnsembleModel.xyz_per_member`.
    """

    name: str = "quasi_crystal_amber"

    def __init__(
        self,
        model: "EnsembleModel",
        cell: "Cell",
        spacegroup: "SpaceGroup",
        n_disorder: int,
        pme_cutoff_ang: float = 10.0,
        ewald_error_tolerance: float = 5e-4,
        normalize_per_asu: bool = True,
        residue_charges: Optional[Dict[str, int]] = None,
        gaff2_files: Optional[Dict[str, Tuple[str, str]]] = None,
        charge_method: str = "gas",
        drop_special_position_threshold_ang: float = 0.0,
        relax_on_init: bool = True,
        relax_max_iterations: int = 200,
        force_clamp: float = 10000.0,
        verbose: int = 0,
    ) -> None:
        try:
            import openmm  # noqa: F401, PLC0415
        except ImportError as e:
            raise ImportError(
                "QuasiCrystalAmberTarget requires OpenMM.\n"
                "Install with:  pip install torchref[amber]\n"
                "Or via conda:  conda install -c conda-forge openmm"
            ) from e

        # --- Validate N == n_disorder · N_sym ---
        n_sym = int(spacegroup.n_ops)
        N = int(model.n_members)
        if int(n_disorder) * n_sym != N:
            raise ValueError(
                f"n_members ({N}) must equal n_disorder ({n_disorder}) "
                f"* spacegroup.n_ops ({n_sym}) = {int(n_disorder) * n_sym}"
            )

        self._n_disorder = int(n_disorder)
        self._force_clamp = float(force_clamp)
        self._n_sym = n_sym
        self._n_members = N
        self._normalize_per_asu = bool(normalize_per_asu)
        self._pme_cutoff_ang = float(pme_cutoff_ang)
        self._ewald_tol = float(ewald_error_tolerance)

        # --- Build the SupercellLayout (host tensors; will be moved at use time) ---
        cell_matrix = cell.fractional_matrix.detach().cpu()
        sym_R = spacegroup.matrices.detach().cpu()
        sym_t = spacegroup.translations.detach().cpu()
        self._layout = SupercellLayout(
            cell=cell_matrix,
            sym_rotations=sym_R,
            sym_translations=sym_t,
            n_disorder=self._n_disorder,
        )

        # --- Optional: detect & drop atoms on spacegroup sym elements ---
        # Atoms that sit on a sym element (e.g. an HOH water on a 2-fold
        # axis, like 3GR5's HOH 224) double-count when we tile N full ASU
        # copies into the supercell: the sym-mate of such an atom lands at
        # the same Cartesian position, giving LJ-12 interaction at ~0
        # distance and a huge nominal energy. The
        # ``_OpenMMAMBERFunction`` force-clamp (10000 kJ/mol/nm per atom)
        # keeps the optimizer's per-step displacement bounded, so by default
        # we let the system equilibrate — the optimizer pushes the offending
        # atom off the sym element over the first few steps. Set
        # ``drop_special_position_threshold_ang > 0`` to opt in to dropping
        # those atoms from the Amber side instead (X-ray F_calc still uses
        # them via the model).
        n_atoms_full = int(len(model._pdb_single))
        if drop_special_position_threshold_ang > 0.0:
            asu_xyz_ang = model.xyz_per_member[0].detach().cpu().numpy()
            keep_mask = _detect_special_position_atoms(
                asu_xyz_ang,
                cell_matrix.numpy(),
                sym_R.numpy(),
                sym_t.numpy(),
                threshold_ang=float(drop_special_position_threshold_ang),
            )
            n_dropped = int((~keep_mask).sum())
            self._keep_atom_idx_np = np.where(keep_mask)[0].astype(np.int64)
            if verbose >= 1 and n_dropped > 0:
                pdb_full = model._pdb_single
                dropped = pdb_full.iloc[~keep_mask]
                preview = ", ".join(
                    f"{str(r['resname']).strip()}{int(r['resseq'])}."
                    f"{str(r['name']).strip()}"
                    for _, r in dropped.head(5).iterrows()
                )
                print(
                    f"[QuasiCrystalAmberTarget] dropping {n_dropped} atom(s) "
                    f"on sym elements (sym-expansion would overlap): "
                    f"{preview}" + ("..." if n_dropped > 5 else "")
                )
        else:
            self._keep_atom_idx_np = np.arange(n_atoms_full, dtype=np.int64)

        # --- Build the single-copy chemistry topology as INHERITED state. ---
        # As an AmberTarget subclass we run the full antechamber + ForceField
        # pipeline against a genuine single-conformation Model (the ensemble's
        # ``_pdb_single`` restricted to non-special-position atoms). This
        # populates self._system / _pos_buf / _model_to_omm / _h_* for ONE
        # member; we replicate them into the supercell below. ``_model`` stays
        # the ensemble (its per-member coords drive forward()); the
        # single-molecule context the base builds is replaced by the supercell
        # context. coords baked into the build are member-0's; they are
        # overwritten on every forward() anyway.
        chem_model = build_single_copy_model(
            model, atom_idx=self._keep_atom_idx_np, verbose=verbose
        )
        super().__init__(
            model=model,
            chem_model=chem_model,
            cutoff=self._pme_cutoff_ang,
            normalize_by_atoms=False,
            residue_charges=residue_charges,
            gaff2_files=gaff2_files,
            charge_method=charge_method,
            verbose=verbose,
        )

        # --- Read the single-member template from inherited state. ---
        template_system = self._system
        template_pos_nm = self._pos_buf  # (n_omm_template, 3) numpy float64
        self._n_omm_per_member = int(self._n_omm_atoms)
        self._n_model_per_member = int(self._n_model_atoms)

        # Atom map for one member, in OpenMM indices. Tile per member at
        # forward time via the (src_model_idx, dst_omm_idx) pairs.
        template_map = np.asarray(self._model_to_omm, dtype=np.int64)
        self._template_model_to_omm = template_map
        # Index pairs: model atom `src_model_idx[k]` lives in OMM slot
        # `dst_omm_idx[k]` (single-member, in [0, n_omm_per_member)). Atoms
        # with template_map == -1 (waters, OXT, ligands tleap regenerated)
        # are excluded — their OMM slot keeps the construction-time position.
        valid_mask = template_map >= 0
        self._src_model_idx_np = np.where(valid_mask)[0].astype(np.int64)
        self._dst_omm_idx_np = template_map[valid_mask].astype(np.int64)
        # Lazy torch buffers (built on first forward, cached per device/dtype).
        self._buffers_device: Optional[torch.device] = None
        self._buffers_dtype: Optional[torch.dtype] = None

        # Build a Cartesian replicated initial-position buffer for the
        # supercell. Per-member: apply the (sym op, tile) transform to
        # template_pos_nm.  This is the position the Context is built with;
        # it gets overwritten every forward() call.
        positions_per_member_ang = torch.from_numpy(
            template_pos_nm.astype(np.float64) * 10.0
        ).unsqueeze(0).expand(N, -1, -1).contiguous()
        supercell_pos_ang = self._layout.compute_member_positions(
            positions_per_member_ang
        )
        supercell_pos_nm = (
            (supercell_pos_ang / 10.0).reshape(-1, 3).cpu().numpy().astype(np.float64)
        )
        self._pos_buf = supercell_pos_nm.copy()
        self._n_omm_total = int(supercell_pos_nm.shape[0])
        assert self._n_omm_total == N * self._n_omm_per_member

        # --- H attachment, template arrays (numpy, into [0, n_omm_per_member)).
        # The tiling per member is deferred to the forward path: each H index
        # gets ``+ m · n_omm_per_member`` added per member m.
        #
        # AmberTarget marks rigid-fallback Hs (no valid local frame) with
        # sentinel ``-1`` in ``_h_n1_idx`` / ``_h_n2_idx``. The corresponding
        # ``_h_frame_valid`` row is False, so the local-frame branch never
        # uses these indices. But ``index_select`` still evaluates the lookup
        # and errors on negative indices, so clamp the sentinels to 0 — a
        # safe in-bounds dummy whose result is then masked away by the
        # ``frame_valid`` ``torch.where`` in :meth:`_place_hydrogens`.
        self._h_idx_template = np.asarray(self._h_idx, dtype=np.int64).copy()
        self._h_parent_idx_template = np.asarray(self._h_parent_idx, dtype=np.int64).copy()
        h_n1 = np.asarray(self._h_n1_idx, dtype=np.int64).copy()
        h_n2 = np.asarray(self._h_n2_idx, dtype=np.int64).copy()
        h_n1[h_n1 < 0] = 0
        h_n2[h_n2 < 0] = 0
        self._h_n1_idx_template = h_n1
        self._h_n2_idx_template = h_n2
        self._h_local_pos_template = np.asarray(self._h_local_pos, dtype=np.float64).copy()
        self._h_frame_valid_template = np.asarray(self._h_frame_valid, dtype=bool).copy()
        self._h_offset_template = np.asarray(self._h_offset, dtype=np.float64).copy()

        # Build the supercell System (replicate + PME + PBC).
        self._system = _replicate_to_supercell_system(
            template_system,
            self._layout,
            pme_cutoff_nm=self._pme_cutoff_ang / 10.0,
            ewald_error_tolerance=self._ewald_tol,
        )

        # --- Build the Context on the supercell System (replaces the
        #     single-molecule context the base just built). ---
        self._context, self._platform_name = self._build_supercell_context(
            supercell_pos_nm
        )

        # --- Relax against Amber to resolve sym-expansion clashes BEFORE
        #     any Adam steps see them. Without this, the first few forwards
        #     hit (force-clamped, but) very large gradients on the atoms at
        #     special positions; Adam's running variance gets poisoned and
        #     later legitimate gradients are dampened. Running OpenMM's
        #     LocalEnergyMinimizer once now resolves those clashes (e.g.
        #     HOH sitting on a 2-fold axis gets pushed off into the 12
        #     distinct sym-related positions). The relaxed coords are
        #     written back to ``model.xyz_per_member`` so subsequent
        #     ``forward()`` calls see them.
        if relax_on_init:
            self._relax_against_amber(int(relax_max_iterations))

        if self.verbose >= 1:
            print(
                f"[QuasiCrystalAmberTarget] platform={self._platform_name}, "
                f"n_members={N} (n_disorder={self._n_disorder}, N_sym={n_sym}), "
                f"n_omm_per_member={self._n_omm_per_member}, "
                f"n_omm_total={self._n_omm_total}, "
                f"n_nonstandard={self._n_nonstandard}"
            )

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_supercell_context(self, init_positions_nm: np.ndarray):
        """Create an OpenMM Context on the best available platform.

        Mirrors :meth:`AmberTarget._build_context`: try each platform by
        actually constructing a Context (CUDA can be present but fail at
        kernel-load time, e.g. PTX version mismatch), with the order
        CUDA → OpenCL → CPU → Reference. The first one whose construction
        + warmup succeeds wins.
        """
        import openmm  # noqa: PLC0415
        import openmm.unit as u_omm  # noqa: PLC0415

        device_type = getattr(self._model.device, "type", "cpu")
        preferred = "CUDA" if device_type == "cuda" else "CPU"

        seen: set = set()
        order = [
            p for p in [preferred, "OpenCL", "CPU", "Reference"]
            if not (p in seen or seen.add(p))  # type: ignore[func-returns-value]
        ]

        last_exc: Optional[Exception] = None
        for name in order:
            try:
                platform = openmm.Platform.getPlatformByName(name)
                integrator = openmm.VerletIntegrator(1.0 * u_omm.femtoseconds)
                ctx = openmm.Context(self._system, integrator, platform)
                ctx.setPositions(init_positions_nm * u_omm.nanometer)
                ctx.getState(getEnergy=True, getForces=True)  # warmup + validation
                if self.verbose >= 1:
                    print(f"[QuasiCrystalAmberTarget] OpenMM platform: {name}")
                return ctx, name
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if self.verbose >= 1:
                    print(
                        f"[QuasiCrystalAmberTarget] Platform {name} unavailable: "
                        f"{exc}"
                    )
                continue

        raise RuntimeError(
            f"No usable OpenMM platform; tried {order}. Last error: {last_exc}"
        )

    # ------------------------------------------------------------------
    # Pre-refinement Amber relaxation
    # ------------------------------------------------------------------

    def _relax_against_amber(self, max_iterations: int) -> None:
        """Run OpenMM's LocalEnergyMinimizer on the supercell, then write the
        relaxed positions back into ``model.xyz_per_member``.

        The supercell starts with overlapping atoms wherever the ASU sits on
        (or near) a spacegroup symmetry element — those atoms get sym-expanded
        onto themselves. OpenMM's minimizer handles the resulting LJ
        catastrophes gracefully (internal scaling) and pushes the coincident
        atoms apart into their N_sym distinct positions. We then invert the
        sym + tile transform per member to recover ASU-frame coords and write
        them back to the model so the X-ray side sees the relaxed start too.
        """
        import openmm  # noqa: PLC0415
        import openmm.unit as u_omm  # noqa: PLC0415

        if self.verbose >= 1:
            energy_before = self._context.getState(
                getEnergy=True
            ).getPotentialEnergy().value_in_unit(u_omm.kilojoule_per_mole)
            print(
                f"[QuasiCrystalAmberTarget] Amber relaxation: "
                f"energy before = {energy_before:+.4e} kJ/mol, "
                f"running LocalEnergyMinimizer (maxIterations={max_iterations})"
            )
        openmm.LocalEnergyMinimizer.minimize(
            self._context, maxIterations=max_iterations
        )
        state = self._context.getState(getPositions=True, getEnergy=True)
        if self.verbose >= 1:
            energy_after = state.getPotentialEnergy().value_in_unit(
                u_omm.kilojoule_per_mole
            )
            print(
                f"[QuasiCrystalAmberTarget] Amber relaxation: "
                f"energy after  = {energy_after:+.4e} kJ/mol "
                f"({energy_after / self._n_members:+.4e}/ASU)"
            )
        pos_nm = state.getPositions(asNumpy=True).value_in_unit(u_omm.nanometer)
        pos_nm = np.asarray(pos_nm, dtype=np.float64)

        # Update the buffer that holds unmapped slots (e.g. tleap-added atoms
        # we don't track in the model) so future forwards see the relaxed
        # positions for those slots too.
        self._pos_buf = pos_nm.copy()
        # Invalidate the lazy torch buffer cache so it picks up the new positions.
        self._buffers_device = None
        self._buffers_dtype = None

        # Invert sym + tile per member and write the relaxed ASU-frame coords
        # back into model.xyz for the KEPT atoms.
        cell_matrix = self._layout.cell.cpu().numpy()
        cell_inv = np.linalg.inv(cell_matrix)
        a_vec = cell_matrix[:, 0]
        sym_R = self._layout.sym_rotations.cpu().numpy()
        sym_t = self._layout.sym_translations.cpu().numpy()
        R_cart = cell_matrix @ sym_R @ cell_inv     # (n_sym, 3, 3)
        t_cart = sym_t @ cell_matrix.T              # (n_sym, 3)

        n_members = self._n_members
        n_sym = self._n_sym
        n_omm = self._n_omm_per_member
        pos_ang = pos_nm * 10.0  # nm → Å
        pos_ang = pos_ang.reshape(n_members, n_omm, 3)

        # For each kept model atom k, its OMM slot within one member is
        # dst_omm_idx_np[k]. Pull the relaxed supercell position for that
        # (member, slot), then apply inverse sym + inverse tile.
        keep_idx = self._keep_atom_idx_np
        new_asu_per_member_kept = np.empty(
            (n_members, len(self._dst_omm_idx_np), 3), dtype=np.float64
        )
        # Map: per kept atom k, which model-atom row (in the FULL model layout)?
        # That's keep_idx[src_model_idx[k]] — we need to write to that row.
        # Actually src_model_idx is into the SHIM's pdb (which is already
        # subsetted to keep_idx). So:
        #   model atom row = keep_idx[ src_model_idx_np[k] ]
        for m in range(n_members):
            d = m // n_sym
            j = m % n_sym
            omm_kept = pos_ang[m, self._dst_omm_idx_np]  # (n_kept, 3) Å
            omm_kept_no_tile = omm_kept - d * a_vec
            omm_kept_no_t = omm_kept_no_tile - t_cart[j]
            # r_asu_col = R^T @ r_super_col; for row vec: r_asu_row = r_super_row @ R
            new_asu_per_member_kept[m] = omm_kept_no_t @ R_cart[j]

        # Write back to model.xyz.refinable_params at the FULL model atom indices.
        full_model_idx = keep_idx[self._src_model_idx_np]  # FULL model atom rows
        n_atoms_per_member = int(self._model.n_atoms_per_member)
        flat_rows = (
            np.arange(n_members, dtype=np.int64)[:, None] * n_atoms_per_member
            + full_model_idx[None, :]
        ).reshape(-1)
        new_flat = new_asu_per_member_kept.reshape(-1, 3)
        with torch.no_grad():
            xyz_param = self._model.xyz.refinable_params
            new_flat_t = torch.from_numpy(new_flat).to(
                device=xyz_param.device, dtype=xyz_param.dtype
            )
            xyz_param.data[flat_rows] = new_flat_t
            # MixedTensor's xyz() (and hence xyz_per_member) caches its
            # composed output via CachedForwardMixin; invalidate so the
            # next read sees the new refinable_params values. Without this
            # the relaxed coords sit in the parameter but the model's
            # downstream getters keep returning the pre-relax cached tensor.
            if hasattr(self._model.xyz, "reset_forward_cache"):
                self._model.xyz.reset_forward_cache()
            # Also invalidate the model's SF cache — heavy coords just changed.
            if hasattr(self._model, "reset_cache"):
                self._model.reset_cache()

    # ------------------------------------------------------------------
    # Lazy device buffers
    # ------------------------------------------------------------------

    def _ensure_torch_buffers(
        self, device: torch.device, dtype: torch.dtype
    ) -> None:
        """Move/build the torch buffers for ``forward``: atom maps, tiled H
        indices, and the constant init-positions tensor. Caches per
        (device, dtype). No work on repeat calls with the same key."""
        if (
            self._buffers_device == device
            and self._buffers_dtype == dtype
        ):
            return

        N = self._n_members
        n_omm = self._n_omm_per_member

        # Initial sym-tiled positions for every OMM atom (nm). Used as the
        # "fallback" position for slots that don't have a model atom mapped to
        # them (waters, OXT, etc. tleap regenerated).
        self._pos_buf_torch = torch.from_numpy(self._pos_buf).to(
            device=device, dtype=dtype
        )

        # Index pairs (long) for the scatter from model atoms into OMM slots.
        self._src_model_idx_torch = torch.from_numpy(self._src_model_idx_np).to(
            device=device, dtype=torch.long
        )
        self._dst_omm_idx_torch = torch.from_numpy(self._dst_omm_idx_np).to(
            device=device, dtype=torch.long
        )

        # Index of ensemble-model atoms (in the FULL EnsembleModel layout)
        # that survived the special-position filter — used in forward to
        # subset ``xyz_per_member`` before applying the layout transform.
        self._keep_atom_idx_torch = torch.from_numpy(
            self._keep_atom_idx_np
        ).to(device=device, dtype=torch.long)

        # Boolean mask: True where the OMM slot has NO model atom mapped to
        # it (so we keep the init position there).
        unmapped = torch.ones(n_omm, dtype=torch.bool, device=device)
        unmapped[self._dst_omm_idx_torch] = False
        self._unmapped_mask_torch = unmapped  # (n_omm,)

        # H-attachment indices tiled per member: template indices live in
        # [0, n_omm); full-tensor indices live in [0, N · n_omm).
        member_offset = (
            torch.arange(N, device=device, dtype=torch.long).unsqueeze(1)
            * n_omm
        )  # (N, 1)
        h_idx_t = torch.from_numpy(self._h_idx_template).to(
            device=device, dtype=torch.long
        )
        h_parent_t = torch.from_numpy(self._h_parent_idx_template).to(
            device=device, dtype=torch.long
        )
        h_n1_t = torch.from_numpy(self._h_n1_idx_template).to(
            device=device, dtype=torch.long
        )
        h_n2_t = torch.from_numpy(self._h_n2_idx_template).to(
            device=device, dtype=torch.long
        )

        self._h_idx_tiled = (member_offset + h_idx_t.unsqueeze(0)).reshape(-1)
        self._h_parent_idx_tiled = (
            member_offset + h_parent_t.unsqueeze(0)
        ).reshape(-1)
        self._h_n1_idx_tiled = (
            member_offset + h_n1_t.unsqueeze(0)
        ).reshape(-1)
        self._h_n2_idx_tiled = (
            member_offset + h_n2_t.unsqueeze(0)
        ).reshape(-1)

        # Per-H constants tiled by member (same value for each member's
        # corresponding H).
        self._h_local_pos_tiled = torch.from_numpy(
            self._h_local_pos_template
        ).to(device=device, dtype=dtype).repeat(N, 1)
        self._h_frame_valid_tiled = torch.from_numpy(
            self._h_frame_valid_template
        ).to(device=device, dtype=torch.bool).repeat(N)
        self._h_offset_tiled = torch.from_numpy(
            self._h_offset_template
        ).to(device=device, dtype=dtype).repeat(N, 1)

        self._buffers_device = device
        self._buffers_dtype = dtype

    # ------------------------------------------------------------------
    # Position composition
    # ------------------------------------------------------------------

    def _compose_full_omm_xyz(
        self, supercell_xyz_nm: torch.Tensor
    ) -> torch.Tensor:
        """Build the full ``(N · n_omm_per_member, 3)`` OpenMM xyz tensor.

        Mapped (heavy) OMM slots get the current model coords (sym + tile
        applied via the supercell layout); unmapped slots keep the construction-
        time positions (tleap-regenerated atoms — waters, OXT, etc. that don't
        move with the model). H atoms are then placed analytically from the
        heavy positions via the tiled local-frame machinery.

        Parameters
        ----------
        supercell_xyz_nm : torch.Tensor, (N, n_model_per_member, 3)
            Per-member supercell-Cartesian coords in nm
            (= ``layout.compute_member_positions(ensemble_xyz_ang) / 10``).

        Returns
        -------
        torch.Tensor, (N · n_omm_per_member, 3)
            Flat OpenMM-order positions in nm, differentiable in
            ``supercell_xyz_nm`` (and thus in ``model.xyz_per_member``).
        """
        N = self._n_members
        n_omm = self._n_omm_per_member
        device = supercell_xyz_nm.device
        dtype = supercell_xyz_nm.dtype

        pos_init = self._pos_buf_torch.view(N, n_omm, 3)  # (N, n_omm, 3)
        # Scatter mapped model atoms into a zero tensor at the OMM slots.
        src = supercell_xyz_nm.index_select(1, self._src_model_idx_torch)
        # index_copy is autograd-friendly and returns a new tensor.
        scattered = torch.zeros(
            (N, n_omm, 3), device=device, dtype=dtype
        ).index_copy(1, self._dst_omm_idx_torch, src)
        # Where the slot is unmapped, use the init position; otherwise use
        # scattered (the current model coord).
        mask = self._unmapped_mask_torch.view(1, n_omm, 1)
        heavy = torch.where(mask, pos_init, scattered)  # (N, n_omm, 3)

        # Place hydrogens (operates on the flat (N·n_omm, 3) view).
        full_flat = heavy.reshape(-1, 3)
        return self._place_hydrogens(full_flat)

    def _place_hydrogens(self, heavy_xyz_nm: torch.Tensor) -> torch.Tensor:
        """Vectorized H placement across all members via tiled local frames.

        Mirrors :meth:`AmberTarget._place_hydrogens` but operates on the
        ``(N·n_omm_per_member, 3)`` supercell positions with tiled parent /
        neighbour indices and per-H constants. Frame: parent + first heavy
        neighbour for ``e1``, second heavy neighbour projected for ``e2``,
        cross for ``e3``; H position is ``p + Σ local_pos[k] · e_k``. Rigid
        fallback for the small fraction of Hs without two heavy neighbours.
        """
        # Same local-frame physics as the single-molecule path — one shared
        # implementation, here applied with member-tiled index tensors.
        h_pos = _place_hydrogens_local_frame(
            heavy_xyz_nm,
            self._h_parent_idx_tiled,
            self._h_n1_idx_tiled,
            self._h_n2_idx_tiled,
            self._h_local_pos_tiled,
            self._h_frame_valid_tiled,
            self._h_offset_tiled,
        )
        # Write H positions into the heavy tensor via functional index_copy.
        return heavy_xyz_nm.index_copy(0, self._h_idx_tiled, h_pos)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self) -> torch.Tensor:
        """Amber energy on the unified supercell.

        Reads :attr:`EnsembleModel.xyz_per_member`, applies the sym + tile
        transform via the :class:`SupercellLayout`, scatters heavy atoms
        into the OpenMM atom layout, places H atoms analytically, and calls
        the existing :class:`_OpenMMAMBERFunction` autograd bridge for
        energy + analytical forces. Returns kJ/mol/ASU (default) — supercell
        total energy divided by the number of ASU copies it contains
        (``n_disorder · N_sym = model.n_members``) — or raw supercell total
        kJ/mol if ``normalize_per_asu=False``.
        """
        ensemble_xyz_ang_full = self._model.xyz_per_member  # (N, n_model_full, 3) Å
        device = ensemble_xyz_ang_full.device
        dtype = ensemble_xyz_ang_full.dtype
        self._ensure_torch_buffers(device, dtype)

        # Subset to the kept atoms (drop special-position atoms that would
        # double-count when sym-expanded). The result has shape
        # (N, n_model_per_member_kept, 3), matching the inherited single-copy
        # chemistry model, which was built on the SAME kept-atom subset.
        ensemble_xyz_ang = ensemble_xyz_ang_full.index_select(
            1, self._keep_atom_idx_torch
        )

        supercell_xyz_ang = self._layout.compute_member_positions(
            ensemble_xyz_ang
        )
        supercell_xyz_nm = supercell_xyz_ang / 10.0  # (N, n_keep, 3) nm

        full_xyz_nm = self._compose_full_omm_xyz(supercell_xyz_nm)

        energy = _OpenMMAMBERFunction.apply(
            full_xyz_nm, self._context, self._force_clamp
        )
        if self._normalize_per_asu:
            # Per-ASU total: one ASU's atoms is one member's worth, and the
            # supercell holds n_disorder * N_sym = n_members ASU copies.
            energy = energy / float(self._n_members)
        return energy
