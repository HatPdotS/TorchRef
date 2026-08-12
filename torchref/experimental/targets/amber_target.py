"""
AMBER14/GAFF2 Force Field as a Differentiable Restraint.

Uses OpenMM to evaluate the AMBER14 energy for current model coordinates.
Analytical forces from OpenMM are bridged into PyTorch autograd via a
custom Function, making the energy fully differentiable w.r.t. xyz.

Non-standard residues (HETATM not in AMBER14_STANDARD) are parameterised
automatically via antechamber/GAFF2.  Results are cached under
``PATH_TORCHREF_DATA / "amber_cache" / {resname}/``.

Intended workflow::

    # Canonical one-liner — strips altlocs, adds H, then build target:
    mh = (Model(verbose=0, strip_H=True)
          .load_pdb('structure.pdb')
          .strip_altlocs()
          .generate_hydrogens())
    target = AmberTarget(model=mh)                               # protein-only
    target = AmberTarget(model=mh, residue_charges={'LIG': -1})  # with ligand

    loss = target()          # kJ/mol per atom
    loss.backward()
    # xyz gradient is now populated with AMBER forces

Performance note
----------------
OpenMM's ``Modeller.addHydrogens()`` is faster when H atoms are already present
in the model (it refines positions rather than building from scratch).
Gradient and energy are identical either way (H are stripped from the atom map;
``n_model_atoms`` changes only the energy normalisation).

Design notes
------------
- Standard-residues path uses pdbfixer to add missing terminal/sidechain
  heavy atoms before OpenMM's Modeller adds H; the GAFF2 path uses tleap
  for H addition.
- Altloc atoms are filtered before building the OpenMM system: only the
  primary conformation (altloc == '' or 'A') is used.
- OXT and H atoms are excluded from the PDB written to tleap; tleap
  re-adds them via its C-terminal and H-addition templates.
- H positions in the OpenMM context are set once at construction and are
  NOT updated during forward() — a good approximation for small refinement
  steps (< 0.1 Å heavy-atom displacement).
- model_to_omm maps model-atom index → OpenMM atom index for HEAVY atoms
  only.  Model H atoms receive -1 and are skipped in forward().
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch

from torchref import PATH_TORCHREF_DATA
from torchref.utils.stats import (
    VERBOSITY_DEBUG,
    VERBOSITY_DETAILED,
    VERBOSITY_STANDARD,
    StatEntry,
    stat,
)

from torchref.refinement.targets.base import ModelTarget

if TYPE_CHECKING:
    from torchref.model.model import Model


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# AmberTools binaries — discovered lazily on first GAFF2 use so that
# importing torchref never fails even when ambertools is absent.
_AMBERTOOLS_BINARIES: Dict[str, Optional[str]] = {}


def _find_ambertools_binary(name: str) -> str:
    """Locate an AmberTools binary on PATH or via $AMBERHOME.

    Raises FileNotFoundError with install instructions when not found.
    """
    if name in _AMBERTOOLS_BINARIES:
        cached = _AMBERTOOLS_BINARIES[name]
        if cached is not None:
            return cached

    path = shutil.which(name)
    if path:
        _AMBERTOOLS_BINARIES[name] = path
        return path

    for env_var in ("AMBERHOME", "AMBERTOOLS_HOME"):
        home = os.environ.get(env_var)
        if home:
            candidate = os.path.join(home, "bin", name)
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                _AMBERTOOLS_BINARIES[name] = candidate
                return candidate

    raise FileNotFoundError(
        f"'{name}' not found on PATH or in $AMBERHOME.\n"
        f"Install AmberTools:  conda install -c conda-forge ambertools\n"
        f"Or provide pre-computed mol2/frcmod files via gaff2_files=."
    )

#: Residue names covered by AMBER14 force field — antechamber not needed.
AMBER14_STANDARD: frozenset = frozenset(
    {
        # Protein residues
        "ALA", "ARG", "ASN", "ASP", "CYS", "CYX", "GLN", "GLU", "GLY",
        "HID", "HIE", "HIP", "HIS", "ILE", "LEU", "LYS", "MET", "PHE",
        "PRO", "SER", "THR", "TRP", "TYR", "VAL",
        # Terminal caps
        "ACE", "NME",
        # Water and common ions
        "HOH", "WAT", "NA", "K", "CL", "MG", "ZN", "CA", "FE", "MN",
        # RNA / DNA nucleotides
        "A", "G", "C", "U", "T", "DA", "DG", "DC", "DT",
    }
)

# Atom names that tleap adds itself via terminal / template logic; must be
# excluded from the PDB handed to tleap to avoid "does not have a type" errors.
_TLEAP_SKIP_ATOMS: frozenset = frozenset({"OXT", "OT1", "OT2"})

# Residues handled by amber14-all.xml + amber14/tip3pfb.xml in OpenMM Modeller
# (does NOT include Mg/Zn/Ca/Fe etc. — those lack templates in the default XML set)
_MODELLER_FF_RESIDUES: frozenset = frozenset(
    {
        "ALA", "ARG", "ASN", "ASP", "CYS", "CYX", "GLN", "GLU", "GLY",
        "HID", "HIE", "HIP", "HIS", "ILE", "LEU", "LYS", "MET", "PHE",
        "PRO", "SER", "THR", "TRP", "TYR", "VAL",
        "ACE", "NME",
        "HOH", "WAT",
        "NA", "K", "CL",   # ions in amber14/tip3pfb.xml
        "A", "G", "C", "U", "T", "DA", "DG", "DC", "DT",
    }
)

# Residues to exclude from the protein PDB written to tleap (GAFF2 path).
# Currently empty: all AMBER14_STANDARD residues (protein, ions, water) are
# included so they participate in both LJ (steric) and Coulomb gradients.
#
# Waters ARE included because:
# - Crystal waters are poorly restrained by X-ray data (weak density, high B)
#   so their AMBER LJ/Coulomb gradient is their primary positional restraint.
# - tleap reads the PDB sequentially and preserves HOH order, so the
#   sequential residue map still matches correctly.
# - The Coulomb magnitude is wrong (no dielectric screening) but the direction
#   is correct; the AMBER weight in the total loss absorbs the scale error.
#
# Monatomic ions (MG, ZN, CA, …) are covered by leaprc.water.tip3p
# (Li/Merz 12-6 + Joung-Cheatham sets) and are critical for electrostatics
# near charged ligands.
_TLEAP_EXCLUDE_RESIDUES: frozenset = frozenset()

# ---------------------------------------------------------------------------
# Autograd bridge
# ---------------------------------------------------------------------------


class _OpenMMAMBERFunction(torch.autograd.Function):
    """
    Bridges OpenMM energy + analytical forces into PyTorch autograd.

    forward : full_xyz_nm (nm, float, [n_omm_total, 3]) → energy (kJ/mol)

    The input tensor must already contain positions for **every** OpenMM
    atom — heavy and H — in OpenMM's native atom order. Building this
    tensor (scattering model heavy atoms + computing H positions
    analytically from heavy positions) happens in
    :meth:`AmberTarget._compose_full_omm_xyz`, upstream of this Function.

    backward: ∂E/∂full_xyz = −F (full OpenMM force vector). The H
    contributions in F propagate naturally through ``_compose_full_omm_xyz``
    and ``_place_hydrogens`` upstream via PyTorch autograd, delivering
    correctly-distributed gradients to the heavy model atoms (parent +
    local-frame neighbors).
    """

    @staticmethod
    def forward(ctx, full_xyz_nm, context, max_force_nm=10000.0):
        import openmm.unit as unit  # noqa: PLC0415

        pos_np = full_xyz_nm.detach().cpu().numpy().astype(np.float64)
        context.setPositions(pos_np)

        state = context.getState(getEnergy=True, getForces=True)
        energy_kJ = state.getPotentialEnergy().value_in_unit(
            unit.kilojoules_per_mole
        )
        forces_kJ_nm = state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoules_per_mole / unit.nanometer
        )

        # Per-atom force clamp to prevent extreme LJ clashes from blowing
        # up the optimizer. 1000 kJ/mol/Å ≈ force from a ~0.3 Å LJ
        # overlap; converted to kJ/mol/nm = 10000 (the default). Raising it
        # (or passing inf) lets amber push harder against clash geometry —
        # the lever for rejecting the unphysical-geometry overfit. A huge
        # value makes scale≡1 (no clamp).
        norms = np.linalg.norm(forces_kJ_nm, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        scale = np.minimum(float(max_force_nm) / norms, 1.0)
        forces_kJ_nm = forces_kJ_nm * scale

        ctx.save_for_backward(
            torch.tensor(
                forces_kJ_nm,
                dtype=full_xyz_nm.dtype,
                device=full_xyz_nm.device,
            )
        )
        return torch.tensor(
            energy_kJ, dtype=full_xyz_nm.dtype, device=full_xyz_nm.device,
        )

    @staticmethod
    def backward(ctx, grad_output):
        (forces,) = ctx.saved_tensors
        # F = −∂E/∂full_xyz  →  ∂E/∂full_xyz = −F (kJ/mol/nm).
        # Trailing Nones are for the non-tensor ``context`` and ``max_force_nm``.
        return -forces * grad_output, None, None


# ---------------------------------------------------------------------------
# Differentiable hydrogen placement (single source of truth)
# ---------------------------------------------------------------------------


def _place_hydrogens_local_frame(
    heavy_xyz: torch.Tensor,
    parent_idx: torch.Tensor,
    n1_idx: torch.Tensor,
    n2_idx: torch.Tensor,
    local_pos: torch.Tensor,
    frame_valid: torch.Tensor,
    offset: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Place hydrogens from heavy-atom positions via captured local frames.

    The one and only implementation of the H-placement physics, shared by the
    single-molecule / per-member path (:meth:`AmberTarget._place_hydrogens`)
    and the tiled supercell path (``QuasiCrystalAmberTarget._place_hydrogens``).
    Differentiable in ``heavy_xyz``: autograd distributes each H force onto its
    parent + the two frame-reference atoms via the exact local-frame Jacobian.

    For each H, an orthonormal frame is built from its parent ``p`` and two
    heavy neighbours ``n1, n2``::

        e1 = û(n1 − p)
        e2 = û((n2 − p) ⊥ e1)
        e3 = e1 × e2
        h  = p + lx·e1 + ly·e2 + lz·e3

    Hs flagged ``frame_valid == False`` (no two heavy neighbours) fall back to
    the rigid translation ``p + offset``.

    Parameters
    ----------
    heavy_xyz : torch.Tensor, ``(M, 3)``
        Positions (nm) with all heavy-atom slots populated. May be a single
        topology (``M = n_omm``) or a tiled supercell (``M = N · n_omm``).
    parent_idx, n1_idx, n2_idx : torch.Tensor, ``(H,)`` long
        Indices into ``heavy_xyz``. Invalid-frame neighbour indices must be
        pre-clamped to a safe in-bounds value (their result is masked out).
    local_pos : torch.Tensor, ``(H, 3)``
        Captured local-frame coordinates of each H.
    frame_valid : torch.Tensor, ``(H,)`` bool
        Whether the local-frame placement is used (else the rigid fallback).
    offset : torch.Tensor, ``(H, 3)``
        Rigid-fallback ``p → H`` vector.
    eps : float
        Norm floor guarding degenerate frames.

    Returns
    -------
    torch.Tensor, ``(H, 3)``
        H positions (nm). The caller writes these into the H slots.
    """
    p = heavy_xyz.index_select(0, parent_idx)
    n1 = heavy_xyz.index_select(0, n1_idx)
    n2 = heavy_xyz.index_select(0, n2_idx)

    a = n1 - p
    e1 = a / a.norm(dim=-1, keepdim=True).clamp(min=eps)
    b = n2 - p
    b_perp = b - (b * e1).sum(-1, keepdim=True) * e1
    e2 = b_perp / b_perp.norm(dim=-1, keepdim=True).clamp(min=eps)
    e3 = torch.cross(e1, e2, dim=-1)

    h_frame = (
        p
        + local_pos[:, 0:1] * e1
        + local_pos[:, 1:2] * e2
        + local_pos[:, 2:3] * e3
    )
    h_rigid = p + offset
    return torch.where(frame_valid.unsqueeze(-1), h_frame, h_rigid)


# ---------------------------------------------------------------------------
# AmberTarget
# ---------------------------------------------------------------------------


class AmberTarget(ModelTarget):
    """
    Differentiable AMBER14/GAFF2 force-field energy restraint.

    On construction the target:

    1. Detects non-standard residues (HETATM not in :data:`AMBER14_STANDARD`).
    2. Runs antechamber + parmchk2 (parallel, cached) for each non-standard
       residue.
    3. Builds an OpenMM system:

       * **Standard path** (no non-standard residues): filter model PDB to
         primary conformation + heavy atoms, use ``openmm.app.Modeller`` to
         re-add H with AMBER14-compatible names, create system with
         ``ForceField('amber14-all.xml')``.
       * **GAFF2 path** (with non-standard residues): same protein PDB
         (additionally removing OXT) handed to tleap together with each
         ligand's mol2 via ``combine{}``.  Combined AMBER14+GAFF2 topology
         is parameterised by parmed.

    4. Creates an OpenMM Context on the platform that matches the model's
       device: CUDA for ``model.device.type == 'cuda'``, CPU otherwise.
       Falls back CUDA → OpenCL → CPU if the preferred platform is unavailable.
    5. Builds a model-atom → OpenMM-atom index map so that only heavy atoms
       are transferred; H positions are kept from the initial OpenMM setup.

    Parameters
    ----------
    model : Model
        TorchRef model.  Heavy-atom-only models (``strip_H=True``) are
        accepted.  H atoms are added internally by OpenMM's Modeller or
        tleap and are NOT included in the atom map or gradient.

        Passing a model that already has H atoms (via
        ``model.generate_hydrogens()`` or loading a PDB with H) speeds up
        initialisation because ``Modeller.addHydrogens()`` converges
        faster from existing positions.

        **GAFF2 ligands**: antechamber's BCC charge scheme runs a
        semiempirical QM step (sqm) that needs a fully protonated molecule.
        Heavy-only ligands are auto-protonated from the monomer library
        (``generate_hydrogens``) first; an error is raised only if no
        monomer CIF resolves AND the heavy-atom electron count is odd.
        Calling ``model.generate_hydrogens()`` or loading the PDB with
        ``strip_H=False`` beforehand avoids relying on that fallback.
    cutoff : float
        Non-bonded cutoff in Angstroms.  Default 5.0.
    normalize_by_atoms : bool
        If True the energy is divided by the number of model atoms.
        Default True.
    residue_charges : dict[str, int], optional
        Net formal charge per non-standard residue name,
        e.g. ``{'LIG': -1, 'ATP': -4}``.  Residues not listed default to 0
        with a warning.
    gaff2_files : dict[str, tuple[str, str]], optional
        Escape hatch for pre-parameterised non-standard residues: maps a
        residue name to a ``(mol2, frcmod)`` file pair, bypassing the
        antechamber/parmchk2 step for that residue.  Referenced by the
        parameterisation error messages as a manual override.
    charge_method : str
        Antechamber charge method (``-c`` flag), one of
        ``bcc``/``gas``/``gascharge``/``rc``/``esp``/``mul``/``abcg2``.
        Default ``"gas"`` (Gasteiger; empirical, no QM, always succeeds).
        ``"bcc"`` (AM1-BCC) is more accurate but runs the sqm QM step and
        can fail to converge on multi-residue batches.
    verbose : int
        Verbosity level (0 = silent, 1 = informational, 2 = debug).
    chem_model : Model, optional
        Single-conformation topology source.  When ``model`` is a
        multi-member ensemble, ``chem_model`` supplies the one conformation
        used to build the chemistry/topology; defaults to ``model`` for the
        single-molecule case.
    """

    name: str = "amber"

    def __init__(
        self,
        model: "Model" = None,
        cutoff: float = 5.0,
        normalize_by_atoms: bool = True,
        residue_charges: Optional[Dict[str, int]] = None,
        gaff2_files: Optional[Dict[str, Tuple[str, str]]] = None,
        charge_method: str = "gas",
        verbose: int = 0,
        chem_model: "Model" = None,
    ):
        try:
            import openmm  # noqa: F401, PLC0415
        except ImportError:
            raise ImportError(
                "AmberTarget requires OpenMM.\n"
                "Install with:  pip install torchref[amber]\n"
                "Or via conda:  conda install -c conda-forge openmm"
            ) from None

        super().__init__(model=model, verbose=verbose)

        # The chemistry/topology is built from a SINGLE-conformation model
        # (``_chem_model``). ``_model`` may be a multi-member ensemble whose
        # per-member coordinates are fed through ``_energy`` by subclasses;
        # for the single-molecule case the two are the same object.
        self._chem_model = chem_model if chem_model is not None else model

        # Antechamber charge method. Options (per antechamber -c flag):
        #   'bcc'  — AM1-BCC; runs sqm semi-empirical QM, accurate but can
        #            fail to converge on multi-residue batches.
        #   'gas'  — Gasteiger; empirical, no QM, always succeeds. Less
        #            accurate Coulomb terms but fine when bonded geometry
        #            dominates (e.g. ensemble geometry restraints).
        #   'gascharge', 'rc', 'esp', 'mul', etc. — see antechamber docs.
        if charge_method not in {"bcc", "gas", "gascharge", "rc", "esp", "mul", "abcg2"}:
            raise ValueError(
                f"charge_method must be one of bcc/gas/gascharge/rc/esp/mul/abcg2; "
                f"got {charge_method!r}"
            )
        self._charge_method = charge_method
        self._normalize = normalize_by_atoms
        self._residue_charges = dict(residue_charges) if residue_charges else {}
        self._gaff2_files = dict(gaff2_files) if gaff2_files else {}

        self.register_buffer("_cutoff_buf", torch.tensor(float(cutoff)))

        # Internal state (None until fully initialised)
        self._context = None
        self._platform_name: str = "none"
        self._model_to_omm: Optional[np.ndarray] = None
        self._pos_buf: Optional[np.ndarray] = None
        self._n_omm_atoms: int = 0
        self._n_model_atoms: int = 0
        self._n_nonstandard: int = 0
        # GAFF2 path: ordered residue map for atom matching (None = standard path)
        self._tleap_residue_map: Optional[List[Dict[str, int]]] = None
        # Cached protonated chemistry PDB (filled lazily by the first ligand
        # parameterisation that needs H). None = not yet computed; False =
        # generate_hydrogens failed (don't retry).
        self._protonated_pdb_cache = None

        if self._chem_model is None:
            return  # Allow empty init for state_dict loading

        self._build()

    # ------------------------------------------------------------------
    # Top-level build orchestration
    # ------------------------------------------------------------------

    def _build(self) -> None:
        """Detect → antechamber → build OpenMM system → map atoms.

        Builds the OpenMM topology from ``self._chem_model`` — a single
        conformation. (``self._model`` may be a multi-member ensemble.)
        """
        # Reject models with alternate conformations — OpenMM only handles
        # a single conformation.  Call model.strip_altlocs() first.
        altlocs = self._chem_model.pdb["altloc"].astype(str).str.strip()
        if (altlocs != "").any():
            raise ValueError(
                "[AmberTarget] Model contains alternate conformations. "
                "OpenMM requires a single conformation.\n"
                "Fix: model = model.strip_altlocs() before creating AmberTarget."
            )
        nonstandard = self._detect_nonstandard_residues()
        self._n_nonstandard = len(nonstandard)

        gaff2_params = self._run_antechamber_parallel(nonstandard)

        system, topology, positions_nm = self._build_omm_system(gaff2_params)
        self._system = system
        self._topology = topology

        # Make tleap positions available to _build_atom_map (GAFF2 path uses
        # position-based matching; positions_nm will be cleaned up afterward).
        self._tleap_pos_nm = positions_nm
        self._build_atom_map()
        del self._tleap_pos_nm

        self._build_context(positions_nm)

        # Pre-allocate nm position buffer: H positions pre-filled from OpenMM init
        self._pos_buf = positions_nm.copy()
        self._n_model_atoms = len(self._chem_model.pdb)
        # Build (H, parent, offset) table so we can rigidly re-attach H atoms
        # to their parent heavy atom each forward. Without this, H positions
        # stay frozen at construction time while heavy atoms move, blowing up
        # bond-stretch terms by orders of magnitude (the dominant pathology
        # for any model.xyz() that excludes H).
        self._build_h_attachment(positions_nm)

        if self.verbose >= 1:
            print(
                f"[AmberTarget] platform={self._platform_name}, "
                f"n_omm={self._n_omm_atoms}, n_model={self._n_model_atoms}, "
                f"n_nonstandard={self._n_nonstandard}"
            )

    # ------------------------------------------------------------------
    # Step 1 — Detect non-standard residues
    # ------------------------------------------------------------------

    def _detect_nonstandard_residues(self) -> List[Tuple[str, int]]:
        """
        Return ``(resname, net_charge)`` for HETATM residues not in
        :data:`AMBER14_STANDARD`.  ATOM records with unknown resnames warn.
        """
        pdb = self._chem_model.pdb
        nonstandard: List[Tuple[str, int]] = []
        seen: set = set()

        records = pdb["ATOM"].astype(str).str.strip()
        resnames = pdb["resname"].astype(str).str.strip()

        for record, resname in zip(records, resnames):
            if resname in seen:
                continue
            seen.add(resname)

            if resname in AMBER14_STANDARD:
                continue

            if record == "HETATM":
                charge = self._residue_charges.get(resname, None)
                if charge is None:
                    warnings.warn(
                        f"[AmberTarget] Non-standard residue '{resname}' has no "
                        f"charge in residue_charges; assuming 0. "
                        f"Pass residue_charges={{'{resname}': <charge>}} to suppress.",
                        UserWarning,
                        stacklevel=4,
                    )
                    charge = 0
                nonstandard.append((resname, charge))
            else:  # ATOM record with unrecognised name
                warnings.warn(
                    f"[AmberTarget] ATOM record with unrecognised residue name "
                    f"'{resname}'. It will be dropped from the OpenMM system "
                    f"(zero AMBER gradient on its atoms) unless a residue_charges "
                    f"or gaff2_files entry is supplied for it.",
                    UserWarning,
                    stacklevel=4,
                )

        return nonstandard

    # ------------------------------------------------------------------
    # Step 2 — Antechamber pipeline
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(resname: str, atom_names: List[str], charge: int,
                   charge_method: str = "bcc") -> str:
        content = f"{resname}:{':'.join(sorted(atom_names))}:{charge}:{charge_method}"
        return hashlib.sha1(content.encode()).hexdigest()

    def _get_cache_dir(self, resname: str) -> Path:
        d = PATH_TORCHREF_DATA / "amber_cache" / resname
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _write_residue_pdb(self, res_atoms, path: Path) -> None:
        """Write a minimal single-residue PDB file for antechamber input."""
        with open(path, "w") as f:
            for serial, (_, row) in enumerate(res_atoms.iterrows(), 1):
                name = str(row["name"]).strip()
                resname = str(row["resname"]).strip()
                x, y, z = float(row["x"]), float(row["y"]), float(row["z"])
                elem = str(row.get("element", name[0])).strip()
                chain = str(row.get("chainid", "A")).strip() or "A"
                resseq = int(row.get("resseq", 1))
                f.write(
                    f"HETATM{serial:5d} {name:<4s} {resname:3s} {chain}"
                    f"{resseq:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}"
                    f"  1.00  0.00          {elem:>2s}\n"
                )
            f.write("END\n")

    def _protonated_chem_pdb(self):
        """Protonated chemistry-model PDB DataFrame (cached), or ``None``.

        Uses :meth:`Model.generate_hydrogens` once on the whole chemistry model
        (which has a unit cell + full residue context, so gemmi's topology engine
        is well-posed). H come from the monomer-library CIF at ideal geometry via
        TorchRef's auto-fetching monomer library — no full CCP4 install needed.
        Cached so repeated ligand parameterisations don't re-run it.
        """
        if self._protonated_pdb_cache is None:
            try:
                m_h = self._chem_model.generate_hydrogens()
                self._protonated_pdb_cache = (
                    m_h.update_pdb() if hasattr(m_h, "update_pdb") else m_h.pdb
                )
            except Exception as exc:  # missing CIF/lib, gemmi failure, etc.
                if self.verbose >= 1:
                    print(f"[AmberTarget] generate_hydrogens failed: {exc}")
                self._protonated_pdb_cache = False
        if self._protonated_pdb_cache is False:
            return None
        return self._protonated_pdb_cache

    def _protonate_residue_pdb(self, resname: str, out_pdb: Path) -> bool:
        """Write a protonated single-residue PDB for ``resname`` to ``out_pdb``.

        antechamber/GAFF2 needs a protonated, valence-satisfied molecule because
        the model is heavy-atom-only. Only topologically-correct H are required
        here — charges are Gasteiger (connectivity-based, no QM) and the running-
        system H are re-placed analytically each step — so the monomer library's
        ideal geometry (via :meth:`_protonated_chem_pdb`) is ample.

        Returns ``True`` iff H were added for ``resname`` (a monomer CIF
        resolved); ``False`` lets the caller fall back.
        """
        pdb_h = self._protonated_chem_pdb()
        if pdb_h is None:
            return False
        res = pdb_h[pdb_h["resname"].astype(str).str.strip() == resname]
        h_mask = res["element"].astype(str).str.strip().isin(["H", "D"])
        if not bool(h_mask.any()):
            return False
        self._write_residue_pdb(res, out_pdb)
        if self.verbose >= 1:
            print(
                f"[AmberTarget] protonated '{resname}' via monomer library: "
                f"+{int(h_mask.sum())} H"
            )
        return True

    def _run_antechamber_one(
        self, resname: str, charge: int
    ) -> Tuple[str, Path, Path]:
        """
        Run antechamber + parmchk2 for one non-standard residue.

        Cache is checked first.  On a miss, work happens in a temp dir and
        results are atomically moved to the cache (write-then-rename).
        """
        pdb = self._chem_model.pdb
        res_atoms = pdb[pdb["resname"].astype(str).str.strip() == resname]
        atom_names = res_atoms["name"].astype(str).str.strip().tolist()

        # antechamber needs a fully protonated molecule (sqm — used for BCC
        # charges — needs an even electron count, and GAFF2 atom typing needs
        # satisfied valences). The model is heavy-atom-only, so a ligand with no
        # H is protonated below from the monomer library before antechamber runs.
        # Compute the heavy-atom electron parity here to sanity-check the result.
        _Z = {"H":1,"He":2,"Li":3,"Be":4,"B":5,"C":6,"N":7,"O":8,"F":9,"Ne":10,
               "Na":11,"Mg":12,"Al":13,"Si":14,"P":15,"S":16,"Cl":17,"Ar":18,
               "K":19,"Ca":20,"Cr":24,"Mn":25,"Fe":26,"Co":27,"Ni":28,"Cu":29,
               "Zn":30,"Br":35,"I":53,"Se":34,"Mo":42,"W":74,"Pt":78,"Au":79}
        elems = res_atoms["element"].astype(str).str.strip().str.capitalize()
        n_protons = sum(_Z.get(e, 0) for e in elems)
        n_electrons = n_protons - charge
        has_h = bool(elems.isin(["H", "D"]).any())

        key = self._cache_key(resname, atom_names, charge, self._charge_method)
        cache_dir = self._get_cache_dir(resname)

        mol2_cached = cache_dir / f"{key}.mol2"
        frcmod_cached = cache_dir / f"{key}.frcmod"

        if mol2_cached.exists() and frcmod_cached.exists():
            if self.verbose >= 1:
                print(f"[AmberTarget] Cache hit: {resname} ({key[:8]}...)")
            return resname, mol2_cached, frcmod_cached

        if self.verbose >= 1:
            print(f"[AmberTarget] antechamber: {resname} (charge={charge:+d})")

        work_dir = Path(tempfile.mkdtemp(prefix=f"amber_{resname}_"))
        try:
            lig_pdb = work_dir / "lig.pdb"
            lig_mol2 = work_dir / "lig.mol2"
            lig_frcmod = work_dir / "lig.frcmod"

            self._write_residue_pdb(res_atoms, lig_pdb)

            # Heavy-atom-only ligand → protonate before antechamber so GAFF2
            # typing sees satisfied valences (and sqm, if BCC, gets a closed-
            # shell molecule). Hydrogens come from TorchRef's monomer-library
            # placement at ideal geometry.
            antechamber_input = lig_pdb
            if not has_h:
                lig_h_pdb = work_dir / "lig_h.pdb"
                if self._protonate_residue_pdb(resname, lig_h_pdb):
                    antechamber_input = lig_h_pdb
                elif n_electrons % 2 != 0:
                    raise RuntimeError(
                        f"[AmberTarget] Cannot parameterise '{resname}': odd "
                        f"electron count ({n_electrons}) for charge {charge:+d} "
                        f"and no hydrogens could be added (no monomer-library CIF "
                        f"resolved for '{resname}').\nFix: pass an explicit charge "
                        f"via residue_charges={{'{resname}': <charge>}}, supply "
                        f"gaff2_files for this residue, or make a monomer CIF "
                        f"resolvable for auto-protonation (TORCHREF_MONOMER_LIB, "
                        f"or CLIBD_MON as an optional override)."
                    )

            # antechamber
            r = subprocess.run(
                [
                    _find_ambertools_binary("antechamber"),
                    "-i", str(antechamber_input), "-fi", "pdb",
                    "-o", str(lig_mol2), "-fo", "mol2",
                    "-c", self._charge_method, "-nc", str(charge),
                    "-s", "2", "-at", "gaff2", "-dr", "no",
                ],
                cwd=str(work_dir),
                capture_output=True, text=True, timeout=600,
            )
            if r.returncode != 0 or not lig_mol2.exists():
                raise RuntimeError(
                    f"antechamber failed for '{resname}':\n"
                    f"STDOUT: {r.stdout}\nSTDERR: {r.stderr}"
                )

            # parmchk2
            r = subprocess.run(
                [
                    _find_ambertools_binary("parmchk2"),
                    "-i", str(lig_mol2), "-f", "mol2",
                    "-o", str(lig_frcmod), "-s", "gaff2",
                ],
                cwd=str(work_dir),
                capture_output=True, text=True, timeout=120,
            )
            if r.returncode != 0 or not lig_frcmod.exists():
                raise RuntimeError(
                    f"parmchk2 failed for '{resname}':\n"
                    f"STDOUT: {r.stdout}\nSTDERR: {r.stderr}"
                )

            # Atomic cache write (temp file → rename)
            shutil.copy2(lig_mol2, cache_dir / f"{key}.mol2.tmp")
            shutil.copy2(lig_frcmod, cache_dir / f"{key}.frcmod.tmp")
            (cache_dir / f"{key}.mol2.tmp").rename(mol2_cached)
            (cache_dir / f"{key}.frcmod.tmp").rename(frcmod_cached)

            (cache_dir / f"{key}.meta.json").write_text(
                json.dumps(
                    {
                        "resname": resname,
                        "charge": charge,
                        "atom_names": sorted(atom_names),
                        "cache_key": key,
                    },
                    indent=2,
                )
            )

            if self.verbose >= 1:
                print(f"[AmberTarget] Cached: {resname} → {cache_dir}")
            return resname, mol2_cached, frcmod_cached
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def _run_antechamber_parallel(
        self, nonstandard: List[Tuple[str, int]]
    ) -> Dict[str, Tuple[Path, Path]]:
        """Resolve GAFF2 parameters for non-standard residues.

        Checks (in order): user-supplied gaff2_files → cache → antechamber.
        """
        if not nonstandard:
            return {}

        results: Dict[str, Tuple[Path, Path]] = {}
        need_antechamber: List[Tuple[str, int]] = []

        for rn, charge in nonstandard:
            # 1. User-supplied files
            if rn in self._gaff2_files:
                mol2, frcmod = self._gaff2_files[rn]
                if self.verbose >= 1:
                    print(f"[AmberTarget] Using supplied files for '{rn}'")
                results[rn] = (Path(mol2), Path(frcmod))
            else:
                need_antechamber.append((rn, charge))

        if not need_antechamber:
            return results

        # 2. Cache + antechamber for remaining residues
        with ThreadPoolExecutor(max_workers=min(len(need_antechamber), 4)) as pool:
            futures = {
                pool.submit(self._run_antechamber_one, rn, ch): rn
                for rn, ch in need_antechamber
            }
            for fut in as_completed(futures):
                rn = futures[fut]
                try:
                    rn_out, mol2, frcmod = fut.result()
                    results[rn_out] = (mol2, frcmod)
                except Exception as exc:
                    raise RuntimeError(
                        f"[AmberTarget] Failed to parameterise '{rn}': {exc}"
                    ) from exc

        return results

    # ------------------------------------------------------------------
    # Step 3 — Build OpenMM system
    # ------------------------------------------------------------------

    def _filter_pdb_for_omm(self, include_nonstandard: bool = False):
        """
        Return a filtered copy of model.pdb suitable for OpenMM / tleap:
        - Primary conformation only (altloc == '' or 'A')
        - Heavy atoms only (element != H or D)
        - Optionally exclude non-standard residues (standard path)

        The returned DataFrame keeps the original model.pdb integer index
        so that ``df.index`` can be used as model row indices in the atom map.
        """
        pdb = self._chem_model.update_pdb()

        mask = pdb["altloc"].astype(str).str.strip().isin(["", "A"])
        mask &= ~pdb["element"].astype(str).str.strip().isin(["H", "D"])

        if not include_nonstandard:
            ns_resnames = {
                rn for rn in pdb["resname"].astype(str).str.strip().unique()
                if rn not in _MODELLER_FF_RESIDUES
            }
            if ns_resnames:
                mask &= ~pdb["resname"].astype(str).str.strip().isin(ns_resnames)

        # Do NOT reset_index: keep original model.pdb row positions as index
        return pdb[mask].copy()

    def _filter_pdb_for_tleap(self):
        """
        Filter model.pdb for the tleap protein PDB (GAFF2 path):

        - Primary conformation only (altloc == '' or 'A')
        - Heavy atoms only (element != H or D)
        - Standard AMBER residues only (``AMBER14_STANDARD``) — non-standard
          HETATM residues are handled via antechamber / mol2 separately
        - Waters (HOH/WAT) ARE included — ``_TLEAP_EXCLUDE_RESIDUES`` is
          empty, so all ``AMBER14_STANDARD`` residues participate in the
          LJ/Coulomb gradients (atom matching is position-based, so tleap's
          water ordering does not break the map)
        - Monatomic ions (MG, ZN, CA, …) ARE included — covered by
          ``leaprc.water.tip3p`` (Li/Merz 12-6 set), appear in fixed PDB
          order, important for electrostatics near charged ligands
        - Terminal atoms tleap regenerates (OXT …) excluded

        Note: uses ``AMBER14_STANDARD`` (not ``_MODELLER_FF_RESIDUES``)
        so that ions absent from amber14-all.xml are still sent to tleap.
        Index is preserved (original model.pdb row positions).
        """
        pdb = self._chem_model.update_pdb()

        mask = pdb["altloc"].astype(str).str.strip().isin(["", "A"])
        mask &= ~pdb["element"].astype(str).str.strip().isin(["H", "D"])

        # Allow AMBER-standard residues (including HOH/WAT, since
        # _TLEAP_EXCLUDE_RESIDUES is empty); non-standard HETATM are excluded
        res_col = pdb["resname"].astype(str).str.strip()
        tleap_allowed = AMBER14_STANDARD - _TLEAP_EXCLUDE_RESIDUES
        mask &= res_col.isin(tleap_allowed)

        # Strip tleap-regenerated terminal atoms
        mask &= ~pdb["name"].astype(str).str.strip().isin(_TLEAP_SKIP_ATOMS)

        return pdb[mask].copy()

    def _build_omm_system(
        self, gaff2_params: Dict[str, Tuple[Path, Path]]
    ) -> Tuple:
        """
        Build OpenMM system.  Returns ``(system, omm_topology, pos_nm_array)``.

        Standard path (no non-standard residues)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Filter model PDB → heavy atoms, primary conformation, standard residues.
        Use ``openmm.app.Modeller.addHydrogens()`` to re-add H with AMBER names.
        Create system with ``ForceField('amber14-all.xml')``.

        GAFF2 path (non-standard residues present)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Write protein PDB (no OXT, no H) + mol2 per ligand.
        Combine via tleap ``combine{}`` command → prmtop/inpcrd.
        Load with parmed → ``AmberParm.createSystem()``.
        """
        import openmm as mm  # noqa: PLC0415
        import openmm.app as app  # noqa: PLC0415
        import openmm.unit as unit  # noqa: PLC0415

        cutoff_A = float(self._cutoff_buf.item())

        if not gaff2_params:
            system, topology, pos_nm = self._build_standard(cutoff_A, app, unit)
        else:
            system, topology, pos_nm = self._build_gaff2(
                gaff2_params, cutoff_A, app, unit
            )

        # Remove CMMotionRemover so raw per-atom forces are available
        for i in range(system.getNumForces() - 1, -1, -1):
            if isinstance(system.getForce(i), mm.CMMotionRemover):
                system.removeForce(i)

        return system, topology, pos_nm

    def _build_standard(self, cutoff_A: float, app, unit) -> Tuple:
        """
        AMBER14 standard-residue path using gemmi + pdbfixer + OpenMM.

        gemmi writes proper chain termination / TER records so pdbfixer
        can detect and fix missing terminal atoms (OXT).  pdbfixer also
        handles missing sidechain atoms and non-standard residue names.
        """
        import gemmi  # noqa: PLC0415
        from pdbfixer import PDBFixer  # noqa: PLC0415
        from torchref.io import pdb as pdbio  # noqa: PLC0415

        # Standard path: Modeller preserves chain/resseq → use key-based mapping
        self._tleap_residue_map = None

        pdb_heavy = self._filter_pdb_for_omm(include_nonstandard=False)

        tmp = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
        tmp2 = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
        tmp.close()
        tmp2.close()
        try:
            # Write via torchref, then re-read/write with gemmi to get
            # proper chain breaks and TER records that pdbfixer needs.
            pdbio.write(pdb_heavy, tmp.name)
            st = gemmi.read_structure(tmp.name)
            st.setup_entities()
            st.assign_subchains()
            st.write_pdb(tmp2.name)

            # pdbfixer: add missing terminal atoms and sidechain atoms
            fixer = PDBFixer(filename=tmp2.name)
            fixer.findMissingResidues()
            fixer.missingResidues = {}  # don't fill gaps
            fixer.findMissingAtoms()

            if self.verbose >= 1:
                n_missing = sum(len(v) for v in fixer.missingAtoms.values())
                n_terminals = sum(
                    1 for v in fixer.missingTerminals.values() if v
                )
                if n_missing or n_terminals:
                    print(
                        f"[AmberTarget] pdbfixer: {n_missing} missing atoms, "
                        f"{n_terminals} terminal fixes"
                    )

            fixer.addMissingAtoms()
        finally:
            os.unlink(tmp.name)
            os.unlink(tmp2.name)

        ff = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
        modeller = app.Modeller(fixer.topology, fixer.positions)
        modeller.addHydrogens(ff)

        system = ff.createSystem(
            modeller.topology,
            nonbondedMethod=app.CutoffNonPeriodic,
            nonbondedCutoff=cutoff_A * unit.angstrom,
            constraints=None,
        )

        # positions in nm
        pos_nm = np.array(
            modeller.positions.value_in_unit(unit.nanometer), dtype=np.float64
        )
        return system, modeller.topology, pos_nm

    def _build_gaff2(
        self,
        gaff2_params: Dict[str, Tuple[Path, Path]],
        cutoff_A: float,
        app,
        unit,
    ) -> Tuple:
        """
        AMBER14 + GAFF2 path via tleap + parmed.

        All AMBER14-standard heavy atoms (protein, ions, waters — no OXT, no H,
        no non-standard HETATM) plus each ligand mol2 are combined by tleap
        ``combine{}``.  parmed loads the resulting prmtop/inpcrd.

        Atom mapping uses position-based matching (see :meth:`_build_atom_map`):
        tleap's initial coordinates are taken directly from the PDB we write,
        so model and tleap positions agree to 3 decimal places (PDB precision),
        making a KD-tree nearest-neighbour search unambiguous.  This avoids
        relying on tleap's residue-sequential numbering, which is fragile for
        water molecules.
        """
        import parmed as pmd  # noqa: PLC0415
        from torchref.io import pdb as pdbio  # noqa: PLC0415

        work_dir = Path(tempfile.mkdtemp(prefix="amber_gaff2_"))
        try:
            prot_pdb = work_dir / "protein.pdb"
            pdb_tleap = self._filter_pdb_for_tleap()

            # Signal GAFF2 path to _build_atom_map (position-based + name fallback)
            self._tleap_residue_map = True  # type: ignore[assignment]
            # Store GAFF2 resnames so _build_atom_map can do name-based fallback
            # for ligand atoms (mol2 may have old coords if model was refined first)
            self._gaff2_resnames: set = set(gaff2_params.keys())

            pdbio.write(pdb_tleap.reset_index(drop=True), str(prot_pdb))

            prmtop = work_dir / "complex.prmtop"
            inpcrd = work_dir / "complex.inpcrd"

            # Build tleap source lines + one mol2 load per ligand
            lig_loads = []
            lig_names = []
            for rn, (mol2, frcmod) in gaff2_params.items():
                lig_loads.append(f"loadAmberParams {frcmod}")
                lig_loads.append(f"{rn} = loadMol2 {mol2}")
                lig_names.append(rn)

            combine_list = " ".join(["protein"] + lig_names)
            tleap_script = "\n".join(
                [
                    "source leaprc.protein.ff14SB",
                    "source leaprc.water.tip3p",
                    "source leaprc.gaff2",
                ]
                + lig_loads
                + [
                    f"protein = loadPdb {prot_pdb}",
                    f"complex = combine {{{combine_list}}}",
                    f"saveAmberParm complex {prmtop} {inpcrd}",
                    "quit",
                ]
            )
            (work_dir / "tleap.in").write_text(tleap_script + "\n")

            r = subprocess.run(
                [_find_ambertools_binary("tleap"), "-f", str(work_dir / "tleap.in")],
                cwd=str(work_dir),
                capture_output=True, text=True, timeout=300,
            )
            if not inpcrd.exists():
                raise RuntimeError(
                    f"[AmberTarget] tleap failed (GAFF2 path).\n"
                    f"STDOUT (last 1000 chars):\n{r.stdout[-1000:]}\n"
                    f"STDERR: {r.stderr[-500:]}"
                )

            combined = pmd.amber.AmberParm(str(prmtop), str(inpcrd))
            system = combined.createSystem(
                nonbondedMethod=app.CutoffNonPeriodic,
                nonbondedCutoff=cutoff_A * unit.angstrom,
                constraints=None,
            )
            topology = combined.topology
            pos_nm = np.array(
                combined.positions.value_in_unit(unit.nanometer), dtype=np.float64
            )
            return system, topology, pos_nm
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Step 4 — Atom map (model index → OpenMM index)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_hydrogen(omm_atom) -> bool:
        elem = omm_atom.element
        if elem is not None:
            return elem.symbol == "H"
        return omm_atom.name.startswith("H")  # heuristic fallback

    def _build_atom_map(self) -> None:
        """
        Build ``self._model_to_omm``: int32 array [n_model] where entry *i*
        is the OpenMM atom index corresponding to model atom *i*, or -1 for
        unmatched atoms (H atoms, altloc-B atoms, non-standard HETATM, …).

        Two strategies depending on how the system was built:

        **Standard path** (``_tleap_residue_map is None``):
        OpenMM Modeller preserves chain IDs and residue numbers from the input
        PDB, so matching uses the key ``(chain_id, resseq, icode, atom_name)``.

        **GAFF2 path** (``_tleap_residue_map is not None``):
        tleap strips chain IDs and renumbers residues sequentially, making
        name/number-based matching unreliable (especially for waters).
        Instead, the tleap initial positions are taken from the exact
        coordinates we wrote to the PDB (via ``update_pdb()``), so model and
        tleap positions agree to within PDB precision (0.001 Å = 0.0001 nm).
        A KD-tree nearest-neighbour search with a tight threshold (0.005 nm)
        unambiguously identifies each tleap heavy atom's model counterpart.
        """
        from scipy.spatial import cKDTree  # noqa: PLC0415

        pdb = self._chem_model.pdb
        n_model = len(pdb)
        model_to_omm = np.full(n_model, -1, dtype=np.int32)

        if self._tleap_residue_map is None:
            # ---- Standard path: match by (chain, resseq, icode, atom_name) ----
            # No altlocs at this point (checked in _build).
            model_key_to_idx: Dict[Tuple, int] = {}
            for i in range(n_model):
                row = pdb.iloc[i]
                key = (
                    str(row["chainid"]).strip(),
                    int(row["resseq"]),
                    str(row.get("icode", "")).strip(),
                    str(row["name"]).strip(),
                )
                model_key_to_idx[key] = i

            for omm_atom in self._topology.atoms():
                if self._is_hydrogen(omm_atom):
                    continue
                chain_id = omm_atom.residue.chain.id.strip()
                try:
                    resseq = int(omm_atom.residue.id)
                except ValueError:
                    raw = omm_atom.residue.id.strip()
                    resseq = int(raw.rstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ") or "0")
                icode = (omm_atom.residue.insertionCode or "").strip()
                idx = model_key_to_idx.get(
                    (chain_id, resseq, icode, omm_atom.name.strip())
                )
                if idx is not None:
                    model_to_omm[idx] = omm_atom.index

        else:
            # ---- GAFF2 path: position-based matching via KD-tree ----
            # Collect tleap heavy-atom positions (nm) and their indices.
            tleap_pos_nm = self._tleap_pos_nm  # set by _build() before this call
            tleap_ha_omm_idx: List[int] = []
            tleap_ha_pos: List[np.ndarray] = []
            for omm_atom in self._topology.atoms():
                if not self._is_hydrogen(omm_atom):
                    tleap_ha_omm_idx.append(omm_atom.index)
                    tleap_ha_pos.append(tleap_pos_nm[omm_atom.index])

            tleap_ha_pos_arr = np.array(tleap_ha_pos)  # (N_tleap_heavy, 3) nm
            tree = cKDTree(tleap_ha_pos_arr)

            # Collect model primary-altloc heavy-atom positions (nm) and indices.
            # Use update_pdb() coords — same values that were written to tleap PDB.
            fresh_pdb = self._chem_model.update_pdb()
            altloc_ok = fresh_pdb["altloc"].astype(str).str.strip().isin(["", "A"])
            not_h = ~fresh_pdb["element"].astype(str).str.strip().isin(["H", "D"])
            primary_heavy = np.where((altloc_ok & not_h).values)[0]

            model_pos_nm = np.column_stack([
                fresh_pdb["x"].values[primary_heavy],
                fresh_pdb["y"].values[primary_heavy],
                fresh_pdb["z"].values[primary_heavy],
            ]) * 0.1  # Å → nm

            # Match: threshold = 0.005 nm (50× PDB precision of 0.0001 nm)
            dists, nn_idx = tree.query(model_pos_nm, k=1)
            matched = dists < 0.005
            for local_i, (model_i, nn_i) in enumerate(zip(primary_heavy, nn_idx)):
                if matched[local_i]:
                    model_to_omm[model_i] = tleap_ha_omm_idx[nn_i]

            # Name-based fallback for GAFF2 ligand residues whose mol2 positions
            # differ from the current model (e.g. after refinement steps).
            # The cached mol2 retains original antechamber coordinates, so a
            # second AmberTarget init after LBFGS will have position shifts.
            gaff2_resnames = getattr(self, "_gaff2_resnames", set())
            if gaff2_resnames:
                # Build (resname, atom_name) → model positional index for primary heavy
                lig_key_to_model: Dict[Tuple[str, str], int] = {}
                for arr_pos in primary_heavy:
                    rn = str(fresh_pdb["resname"].values[arr_pos]).strip()
                    if rn not in gaff2_resnames:
                        continue
                    aname = str(fresh_pdb["name"].values[arr_pos]).strip()
                    lig_key_to_model[(rn, aname)] = int(arr_pos)

                for omm_atom in self._topology.atoms():
                    if self._is_hydrogen(omm_atom):
                        continue
                    rn = omm_atom.residue.name
                    if rn not in gaff2_resnames:
                        continue
                    aname = omm_atom.name.strip()
                    model_arr_pos = lig_key_to_model.get((rn, aname))
                    if model_arr_pos is not None and model_to_omm[model_arr_pos] < 0:
                        model_to_omm[model_arr_pos] = omm_atom.index

        # Warn about UNEXPECTED unmatched heavy atoms.
        # Expected to be unmatched (silently skipped in gradient):
        #   - H / D atoms
        #   - Waters, ions excluded from tleap (_TLEAP_EXCLUDE_RESIDUES)
        #   - C-terminal OXT regenerated by tleap (_TLEAP_SKIP_ATOMS)
        #   - Alternate conformer atoms (altloc != '' and != 'A')
        elem_col  = pdb["element"].astype(str).str.strip()
        altloc_col = pdb["altloc"].astype(str).str.strip()
        resname_col = pdb["resname"].astype(str).str.strip()
        name_col    = pdb["name"].astype(str).str.strip()
        heavy_mask = ~elem_col.isin(["H", "D"])
        # Residues in AMBER14_STANDARD but without an amber14-all.xml template:
        # unmatched on the standard (Modeller) path; matched via tleap on GAFF2 path.
        _no_modeller_template = AMBER14_STANDARD - _MODELLER_FF_RESIDUES
        expected_mask = (
            # Waters always excluded from tleap; no AMBER gradient expected
            resname_col.isin(_TLEAP_EXCLUDE_RESIDUES) |
            # Ions that lack Modeller templates (matched in GAFF2 path, not standard)
            resname_col.isin(_no_modeller_template) |
            # tleap-regenerated terminal atoms (OXT etc.)
            name_col.isin(_TLEAP_SKIP_ATOMS) |
            # alternate conformers (altloc B, C, …)
            (~altloc_col.isin(["", "A"]))
        )
        unexpected_unmatched = np.where(
            heavy_mask.values & ~expected_mask.values & (model_to_omm < 0)
        )[0]
        if len(unexpected_unmatched) > 0:
            ex = [
                f"{pdb.iloc[i]['name'].strip()} "
                f"({pdb.iloc[i]['resname'].strip()} {pdb.iloc[i]['resseq']})"
                for i in unexpected_unmatched[:5]
            ]
            warnings.warn(
                f"[AmberTarget] {len(unexpected_unmatched)} heavy model atom(s) "
                f"could not be matched to OpenMM topology "
                f"(e.g. {', '.join(ex)}). Their gradients will be zero.",
                UserWarning,
                stacklevel=3,
            )
        elif self.verbose >= 2:
            unmatched_heavy = int(heavy_mask.values.sum()) - int(
                (heavy_mask.values & (model_to_omm >= 0)).sum()
            )
            print(
                f"[AmberTarget] {unmatched_heavy} heavy atoms have model_to_omm=-1 "
                f"(expected: non-standard HETATM / altloc-B / OXT)"
            )

        self._model_to_omm = model_to_omm
        self._n_omm_atoms = self._system.getNumParticles()

        if self.verbose >= 2:
            matched = int((model_to_omm >= 0).sum())
            print(
                f"[AmberTarget] atom map: {matched}/{n_model} model atoms matched "
                f"({self._n_omm_atoms} total OpenMM atoms)"
            )

    # ------------------------------------------------------------------
    # Hydrogen re-attachment
    # ------------------------------------------------------------------

    def _build_h_attachment(self, pos_nm: np.ndarray) -> None:
        """
        Build the local-frame placement table for every H atom.

        Each H is placed at construction-time according to OpenMM's
        ``Modeller.addHydrogens`` output. We freeze that placement in a
        local frame defined by the parent heavy atom and 2 reference
        heavy atoms. At forward time, the H position is recomputed in
        differentiable PyTorch from the current heavy positions:

            e1 = (n1 − p) / |n1 − p|
            e2 = perp(n2 − p, e1) / |perp(n2 − p, e1)|
            e3 = e1 × e2
            h = p + lx·e1 + ly·e2 + lz·e3

        where (lx, ly, lz) = (h − p) · [e1, e2, e3] is captured once.

        Backward through this formula in PyTorch autograd produces the
        exact local-frame Jacobian — so the force on H from OpenMM gets
        correctly distributed across p, n1, n2, not just onto p.

        Reference-atom selection per H:
        - parent ``p``       : the unique heavy atom bonded to H
        - neighbor ``n1``    : any heavy atom bonded to ``p`` (≠ H)
        - neighbor ``n2``    : another heavy atom bonded to ``p``; if
                               ``p`` has only one heavy neighbour, fall
                               back to a heavy atom bonded to ``n1``
                               (i.e. walk one bond further out).

        Hs with no usable triple — extremely rare in real chemistry —
        fall back to the legacy ``h = p + offset`` rigid translation
        path, with ``h_frame_valid=False``.
        """
        if not hasattr(self, "_topology") or self._topology is None:
            self._h_idx = None
            self._h_parent_idx = None
            self._h_n1_idx = None
            self._h_n2_idx = None
            self._h_local_pos = None
            self._h_frame_valid = None
            self._h_offset = None
            return

        # ---- Walk topology bonds. Build heavy-neighbor adjacency and
        # parent map for H atoms in one pass. ------------------------------
        from collections import defaultdict

        def _is_h(atom) -> bool:
            return atom.element is not None and atom.element.symbol == "H"

        parent_of_h: Dict[int, int] = {}
        heavy_neighbors: Dict[int, list] = defaultdict(list)
        for bond in self._topology.bonds():
            a, b = bond[0], bond[1]
            a_is_h = _is_h(a)
            b_is_h = _is_h(b)
            if a_is_h and not b_is_h:
                parent_of_h[a.index] = b.index
            elif b_is_h and not a_is_h:
                parent_of_h[b.index] = a.index
            elif not a_is_h and not b_is_h:
                heavy_neighbors[a.index].append(b.index)
                heavy_neighbors[b.index].append(a.index)
            # H-H bonds are nonsense; ignored.

        if not parent_of_h:
            self._h_idx = None
            self._h_parent_idx = None
            self._h_n1_idx = None
            self._h_n2_idx = None
            self._h_local_pos = None
            self._h_frame_valid = None
            self._h_offset = None
            return

        # ---- Resolve (parent, n1, n2) per H ----------------------------
        h_indices = sorted(parent_of_h.keys())
        n_h = len(h_indices)
        p_arr = np.empty(n_h, dtype=np.int64)
        n1_arr = np.empty(n_h, dtype=np.int64)
        n2_arr = np.empty(n_h, dtype=np.int64)
        valid = np.zeros(n_h, dtype=bool)

        for k, h in enumerate(h_indices):
            p = parent_of_h[h]
            p_arr[k] = p
            neigh = heavy_neighbors.get(p, [])
            if len(neigh) >= 2:
                n1_arr[k] = neigh[0]
                n2_arr[k] = neigh[1]
                valid[k] = True
            elif len(neigh) == 1:
                n1 = neigh[0]
                further = [
                    j for j in heavy_neighbors.get(n1, []) if j != p
                ]
                if further:
                    n1_arr[k] = n1
                    n2_arr[k] = further[0]
                    valid[k] = True
                else:
                    n1_arr[k] = n1
                    n2_arr[k] = -1
                    valid[k] = False
            else:
                n1_arr[k] = -1
                n2_arr[k] = -1
                valid[k] = False

        # ---- Compute local-frame coordinates from initial positions ----
        h_pos = pos_nm[np.asarray(h_indices, dtype=np.int64)]
        p_pos = pos_nm[p_arr]
        local_pos = np.zeros((n_h, 3), dtype=np.float64)
        eps = 1e-12

        valid_idx = np.where(valid)[0]
        if valid_idx.size > 0:
            n1_pos = pos_nm[n1_arr[valid_idx]]
            n2_pos = pos_nm[n2_arr[valid_idx]]
            a = n1_pos - p_pos[valid_idx]
            b = n2_pos - p_pos[valid_idx]
            e1 = a / np.maximum(
                np.linalg.norm(a, axis=-1, keepdims=True), eps,
            )
            b_perp = b - (b * e1).sum(-1, keepdims=True) * e1
            e2 = b_perp / np.maximum(
                np.linalg.norm(b_perp, axis=-1, keepdims=True), eps,
            )
            e3 = np.cross(e1, e2)
            offset_v = h_pos[valid_idx] - p_pos[valid_idx]
            local_pos[valid_idx, 0] = (offset_v * e1).sum(-1)
            local_pos[valid_idx, 1] = (offset_v * e2).sum(-1)
            local_pos[valid_idx, 2] = (offset_v * e3).sum(-1)

        # Rigid fallback offset (used for !valid Hs only)
        offset_all = h_pos - p_pos

        # Stash numpy arrays for the forward path (the autograd Function
        # converts to torch tensors lazily on the model's device).
        self._h_idx = np.asarray(h_indices, dtype=np.int64)
        self._h_parent_idx = p_arr
        self._h_n1_idx = n1_arr
        self._h_n2_idx = n2_arr
        self._h_local_pos = local_pos.astype(np.float64)
        self._h_frame_valid = valid
        self._h_offset = offset_all  # legacy field, used only when !valid

        if self.verbose >= 1:
            n_frame = int(valid.sum())
            n_fallback = int((~valid).sum())
            print(
                f"[AmberTarget] H-attachment: {n_h} H atoms — "
                f"{n_frame} via local-frame placement, "
                f"{n_fallback} via rigid fallback"
            )

    # ------------------------------------------------------------------
    # Differentiable PyTorch placement (called from AmberTarget.forward)
    # ------------------------------------------------------------------

    def _place_hydrogens(self, heavy_omm_xyz_nm: torch.Tensor) -> torch.Tensor:
        """
        Compute H positions from the current heavy-atom OpenMM-order tensor.

        Uses the local-frame data captured in :meth:`_build_h_attachment`:
        for each H, build an orthonormal frame from (parent, n1, n2) and
        place the H at its captured local-frame coordinates. Hs with
        ``h_frame_valid=False`` fall back to ``parent + h_offset``.

        Differentiable: backward through this function distributes the
        H force across the parent + n1 + n2 reference atoms via the exact
        local-frame Jacobian (handled by PyTorch autograd).

        Parameters
        ----------
        heavy_omm_xyz_nm : (n_omm_total, 3) tensor in nm, OpenMM atom order.
            H slot values are ignored — they will be overwritten in the
            returned tensor.

        Returns
        -------
        h_xyz_nm : (n_H, 3) tensor in nm. Empty if no Hs.
        """
        if self._h_idx is None or self._h_idx.size == 0:
            return torch.zeros(
                (0, 3),
                dtype=heavy_omm_xyz_nm.dtype,
                device=heavy_omm_xyz_nm.device,
            )

        device = heavy_omm_xyz_nm.device
        dtype = heavy_omm_xyz_nm.dtype
        # Lazily cache tensor views on the right device/dtype.
        if (
            getattr(self, "_h_tensors_dev", None) != device
            or getattr(self, "_h_tensors_dtype", None) != dtype
        ):
            self._h_parent_idx_t = torch.as_tensor(
                self._h_parent_idx, dtype=torch.long, device=device,
            )
            # For invalid frames clamp neighbor indices to 0 so the gather is
            # safe; the value is masked out by `where` below.
            n1 = np.where(self._h_n1_idx >= 0, self._h_n1_idx, 0)
            n2 = np.where(self._h_n2_idx >= 0, self._h_n2_idx, 0)
            self._h_n1_idx_t = torch.as_tensor(n1, dtype=torch.long, device=device)
            self._h_n2_idx_t = torch.as_tensor(n2, dtype=torch.long, device=device)
            self._h_local_pos_t = torch.as_tensor(
                self._h_local_pos, dtype=dtype, device=device,
            )
            self._h_offset_t = torch.as_tensor(
                self._h_offset, dtype=dtype, device=device,
            )
            self._h_frame_valid_t = torch.as_tensor(
                self._h_frame_valid, dtype=torch.bool, device=device,
            )
            self._h_tensors_dev = device
            self._h_tensors_dtype = dtype

        return _place_hydrogens_local_frame(
            heavy_omm_xyz_nm,
            self._h_parent_idx_t,
            self._h_n1_idx_t,
            self._h_n2_idx_t,
            self._h_local_pos_t,
            self._h_frame_valid_t,
            self._h_offset_t,
        )

    def _compose_full_omm_xyz(
        self,
        heavy_model_xyz_ang: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build the full OpenMM-order position tensor (heavy + H) in nm.

        Heavy model atoms are scattered into their OpenMM slots via
        ``self._model_to_omm``. Unmatched OpenMM heavy slots (e.g.
        non-standard residues without a model match) are filled from
        the construction-time ``_pos_buf`` snapshot so they're at least
        consistent. H slots are filled by :meth:`_place_hydrogens`.

        Differentiable through ``heavy_model_xyz_ang`` — autograd routes
        gradients on H slots back to their parent/n1/n2 reference atoms.
        """
        device = heavy_model_xyz_ang.device
        dtype = heavy_model_xyz_ang.dtype
        n_omm = self._n_omm_atoms

        # Lazy tensorize index maps.
        if (
            getattr(self, "_omm_tensors_dev", None) != device
            or getattr(self, "_omm_tensors_dtype", None) != dtype
        ):
            valid_np = self._model_to_omm >= 0
            self._model_valid_t = torch.as_tensor(
                valid_np, dtype=torch.bool, device=device,
            )
            self._model_valid_model_idx_t = torch.as_tensor(
                np.where(valid_np)[0], dtype=torch.long, device=device,
            )
            self._model_valid_omm_idx_t = torch.as_tensor(
                self._model_to_omm[valid_np], dtype=torch.long, device=device,
            )
            # Construction-time snapshot for unmatched heavy slots + initial Hs
            self._pos_buf_t = torch.as_tensor(
                self._pos_buf, dtype=dtype, device=device,
            )
            self._omm_tensors_dev = device
            self._omm_tensors_dtype = dtype

        # Start from the construction-time snapshot (provides values for
        # unmatched heavy atoms and any non-frame-placed atom). Heavy and
        # H slots will be overwritten below.
        full = self._pos_buf_t.clone()

        heavy_model_xyz_nm = heavy_model_xyz_ang * 0.1
        heavy_matched = heavy_model_xyz_nm.index_select(
            0, self._model_valid_model_idx_t,
        )
        full = full.index_copy(0, self._model_valid_omm_idx_t, heavy_matched)

        # Now derive H positions from the fully-populated heavy tensor.
        if self._h_idx is not None and self._h_idx.size > 0:
            h_xyz = self._place_hydrogens(full)
            if not hasattr(self, "_h_idx_t_for_omm"):
                self._h_idx_t_for_omm = torch.as_tensor(
                    self._h_idx, dtype=torch.long, device=device,
                )
            elif self._h_idx_t_for_omm.device != device:
                self._h_idx_t_for_omm = self._h_idx_t_for_omm.to(device)
            full = full.index_copy(0, self._h_idx_t_for_omm, h_xyz)

        return full

    # ------------------------------------------------------------------
    # Step 5 — OpenMM Context
    # ------------------------------------------------------------------

    def _build_context(self, pos_nm: np.ndarray) -> None:
        """
        Create an OpenMM Context on the platform that matches the model's device.

        Mapping: ``model.device.type == 'cuda'`` → CUDA, otherwise CPU.
        Falls back CUDA → OpenCL → CPU if the preferred platform is unavailable.
        """
        import openmm as mm  # noqa: PLC0415

        device_type = getattr(self._chem_model.device, "type", "cpu")
        preferred = "CUDA" if device_type == "cuda" else "CPU"

        seen: set = set()
        platforms = [
            p for p in [preferred, "OpenCL", "CPU"]
            if not (p in seen or seen.add(p))  # type: ignore[func-returns-value]
        ]

        for name in platforms:
            try:
                platform = mm.Platform.getPlatformByName(name)
                integrator = mm.VerletIntegrator(1.0)
                context = mm.Context(self._system, integrator, platform)
                context.setPositions(pos_nm)
                # Warmup + validation
                context.getState(getEnergy=True, getForces=True)
                self._context = context
                self._platform_name = name
                if self.verbose >= 1:
                    print(f"[AmberTarget] OpenMM platform: {name}")
                return
            except Exception as exc:
                if self.verbose >= 1:
                    print(f"[AmberTarget] Platform {name} unavailable: {exc}")

        raise RuntimeError(
            f"[AmberTarget] No usable OpenMM platform (tried {platforms})."
        )

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def _energy(self, xyz_ang: torch.Tensor) -> torch.Tensor:
        """AMBER14 energy for one conformation's heavy-atom coords.

        Parameters
        ----------
        xyz_ang : torch.Tensor
            ``(n_model_atoms, 3)`` heavy-atom coordinates in Å, in the order
            of ``self._chem_model.pdb`` (the topology the system was built on).

        Returns
        -------
        torch.Tensor
            Scalar energy in kJ/mol (or kJ/mol/atom if ``normalize_by_atoms``).
            Gradient flows to ``xyz_ang`` via OpenMM analytical forces
            (heavy atoms direct) and via :meth:`_place_hydrogens` / PyTorch
            autograd (H positions, redistributed onto their parent +
            local-frame neighbors).

        Notes
        -----
        Subclasses feed per-member coordinates here; the single-molecule
        :meth:`forward` passes ``self._model.xyz()``.
        """
        if self._context is None:
            raise RuntimeError(
                "[AmberTarget] Not initialised. Pass model= to constructor."
            )

        full_xyz_nm = self._compose_full_omm_xyz(xyz_ang)
        energy = _OpenMMAMBERFunction.apply(full_xyz_nm, self._context)

        if self._normalize:
            energy = energy / self._n_model_atoms

        return energy

    def forward(self) -> torch.Tensor:
        """Compute the AMBER14 energy for the model's current coordinates."""
        return self._energy(self._model.xyz())

    # ------------------------------------------------------------------
    # stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, "StatEntry"]:
        """Return target statistics for the logging pipeline."""
        with torch.no_grad():
            e_per_atom = self.forward().item()

        e_total = (
            e_per_atom * self._n_model_atoms if self._normalize else e_per_atom
        )

        return {
            "loss": stat(e_per_atom, VERBOSITY_STANDARD),
            "energy_kJ_mol": stat(e_total, VERBOSITY_DETAILED),
            "n_atoms": stat(self._n_model_atoms, VERBOSITY_DEBUG),
            "platform": stat(self._platform_name, VERBOSITY_DETAILED),
            "n_nonstandard": stat(self._n_nonstandard, VERBOSITY_DEBUG),
        }
