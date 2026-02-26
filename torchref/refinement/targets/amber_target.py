"""
AMBER14/GAFF2 Force Field as a Differentiable Restraint.

Uses OpenMM to evaluate the AMBER14 energy for current model coordinates.
Analytical forces from OpenMM are bridged into PyTorch autograd via a
custom Function, making the energy fully differentiable w.r.t. xyz.

Non-standard residues (HETATM not in AMBER14_STANDARD) are parameterised
automatically via antechamber/GAFF2.  Results are cached under
``PATH_TORCHREF_DATA / "amber_cache" / {resname}/``.

Intended workflow::

    model_h = model.generate_hydrogens()
    target  = AmberTarget(model=model_h, residue_charges={'LIG': -1})
    loss    = target()          # kJ/mol per atom
    loss.backward()
    # xyz gradient is now populated with AMBER forces

Design notes
------------
- No pdbfixer dependency.  H atoms are handled by OpenMM's Modeller
  (standard-residues path) or tleap (GAFF2 path).
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

from .base import ModelTarget

if TYPE_CHECKING:
    from torchref.model.model import Model


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Full path to antechamber binary (dedicated conda env).
ANTECHAMBER = "/das/work/units/LBR-FEL/p17490/CONDA/antechamber/bin/antechamber"
#: Full path to parmchk2 binary.
PARMCHK2 = "/das/work/units/LBR-FEL/p17490/CONDA/antechamber/bin/parmchk2"
#: Full path to tleap binary (needed for GAFF2 path).
TLEAP = "/das/work/units/LBR-FEL/p17490/CONDA/antechamber/bin/tleap"

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

# For the GAFF2 tleap path: exclude water from the protein PDB.
# tleap can reorder HOH residues relative to the model (PDB numbering vs
# tleap sequential numbering), which breaks the sequential atom-map strategy
# when hundreds of waters are present.  Waters are excluded: they get
# model_to_omm = -1 (no AMBER gradient — acceptable since crystal waters
# are not primary targets of force-field restraints and would anyway alter
# the effective dielectric for in-vacuo protein calculations).
#
# Monatomic ions (MG, ZN, CA, …) are NOT excluded: leaprc.water.tip3p
# already loads their parameters (Li/Merz 12-6 + Joung-Cheatham sets),
# they appear in a fixed, predictable order in the PDB we write, and their
# point-charge + LJ treatment is exactly the AMBER approach.  Including them
# is important for electrostatics near charged ligands.
_TLEAP_EXCLUDE_RESIDUES: frozenset = frozenset({"HOH", "WAT"})


# ---------------------------------------------------------------------------
# Autograd bridge
# ---------------------------------------------------------------------------


class _OpenMMAMBERFunction(torch.autograd.Function):
    """
    Bridges OpenMM energy + analytical forces into PyTorch autograd.

    forward : xyz_ang (Å, float, [n_model, 3]) → energy (kJ/mol, scalar)
    backward: ∂loss/∂xyz = −F (OpenMM forces, exact gradients)

    Non-tensor arguments are passed as plain Python objects; they receive
    ``None`` gradients and are not differentiated.
    """

    @staticmethod
    def forward(ctx, xyz_ang, context, model_to_omm, pos_buf):
        import openmm.unit as unit  # noqa: PLC0415

        # Update heavy-atom positions in the pre-allocated nm buffer (Å → nm)
        model_xyz_nm = xyz_ang.detach().cpu().numpy().astype(np.float64) * 0.1
        valid = model_to_omm >= 0
        pos_buf[model_to_omm[valid]] = model_xyz_nm[valid]

        # Transfer positions to OpenMM context (CPU → GPU inside OpenMM)
        context.setPositions(pos_buf)

        state = context.getState(getEnergy=True, getForces=True)
        energy_kJ = state.getPotentialEnergy().value_in_unit(
            unit.kilojoules_per_mole
        )
        forces_kJ_nm = state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoules_per_mole / unit.nanometer
        )

        # Map forces: OpenMM-indexed → model-indexed;  kJ/mol/nm → kJ/mol/Å
        n_model = xyz_ang.shape[0]
        model_forces = np.zeros((n_model, 3), dtype=np.float64)
        model_forces[valid] = forces_kJ_nm[model_to_omm[valid]] * 0.1

        ctx.save_for_backward(
            torch.tensor(
                model_forces, dtype=xyz_ang.dtype, device=xyz_ang.device
            )
        )
        return torch.tensor(energy_kJ, dtype=xyz_ang.dtype, device=xyz_ang.device)

    @staticmethod
    def backward(ctx, grad_output):
        (forces,) = ctx.saved_tensors
        # F = −∂E/∂x  →  ∂E/∂x = −F
        return -forces * grad_output, None, None, None


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

    4. Creates an OpenMM Context on the requested platform
       (CUDA → OpenCL → CPU).
    5. Builds a model-atom → OpenMM-atom index map so that only heavy atoms
       are transferred; H positions are kept from the initial OpenMM setup.

    Parameters
    ----------
    model : Model
        TorchRef model.  Should include hydrogens (call
        ``model.generate_hydrogens()`` beforehand) so atom counts match the
        intended refinement state.  Heavy-atom-only models also work — H
        atoms are added internally by OpenMM.
    cutoff : float
        Non-bonded cutoff in Angstroms.  Default 5.0.
    normalize_by_atoms : bool
        If True the energy is divided by the number of model atoms.
        Default True.
    residue_charges : dict[str, int], optional
        Net formal charge per non-standard residue name,
        e.g. ``{'LIG': -1, 'ATP': -4}``.  Residues not listed default to 0
        with a warning.
    platform : str
        Preferred OpenMM platform.  Tried first; falls back CUDA → OpenCL → CPU.
    verbose : int
        Verbosity level (0 = silent, 1 = informational, 2 = debug).
    """

    name: str = "amber"

    def __init__(
        self,
        model: "Model" = None,
        cutoff: float = 5.0,
        normalize_by_atoms: bool = True,
        residue_charges: Optional[Dict[str, int]] = None,
        platform: str = "CUDA",
        verbose: int = 0,
    ):
        super().__init__(model=model, verbose=verbose)

        self._normalize = normalize_by_atoms
        self._preferred_platform = platform
        self._residue_charges = dict(residue_charges) if residue_charges else {}

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

        if model is None:
            return  # Allow empty init for state_dict loading

        self._build(model)

    # ------------------------------------------------------------------
    # Top-level build orchestration
    # ------------------------------------------------------------------

    def _build(self, model: "Model") -> None:
        """Detect → antechamber → build OpenMM system → map atoms."""
        nonstandard = self._detect_nonstandard_residues()
        self._n_nonstandard = len(nonstandard)

        gaff2_params = self._run_antechamber_parallel(nonstandard)

        system, topology, positions_nm = self._build_omm_system(gaff2_params)
        self._system = system
        self._topology = topology

        self._build_atom_map()
        self._build_context(positions_nm)

        # Pre-allocate nm position buffer: H positions pre-filled from OpenMM init
        self._pos_buf = positions_nm.copy()
        self._n_model_atoms = len(model.pdb)

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
        pdb = self._model.pdb
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
                    f"'{resname}'. pdbfixer / tleap may or may not handle it.",
                    UserWarning,
                    stacklevel=4,
                )

        return nonstandard

    # ------------------------------------------------------------------
    # Step 2 — Antechamber pipeline
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(resname: str, atom_names: List[str], charge: int) -> str:
        content = f"{resname}:{':'.join(sorted(atom_names))}:{charge}"
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

    def _run_antechamber_one(
        self, resname: str, charge: int
    ) -> Tuple[str, Path, Path]:
        """
        Run antechamber + parmchk2 for one non-standard residue.

        Cache is checked first.  On a miss, work happens in a temp dir and
        results are atomically moved to the cache (write-then-rename).
        """
        pdb = self._model.pdb
        res_atoms = pdb[pdb["resname"].astype(str).str.strip() == resname]
        atom_names = res_atoms["name"].astype(str).str.strip().tolist()
        key = self._cache_key(resname, atom_names, charge)
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

            # antechamber
            r = subprocess.run(
                [
                    ANTECHAMBER,
                    "-i", str(lig_pdb), "-fi", "pdb",
                    "-o", str(lig_mol2), "-fo", "mol2",
                    "-c", "bcc", "-nc", str(charge),
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
                    PARMCHK2,
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
        """Run antechamber for all non-standard residues in parallel (≤ 4)."""
        if not nonstandard:
            return {}

        results: Dict[str, Tuple[Path, Path]] = {}
        with ThreadPoolExecutor(max_workers=min(len(nonstandard), 4)) as pool:
            futures = {
                pool.submit(self._run_antechamber_one, rn, ch): rn
                for rn, ch in nonstandard
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
        pdb = self._model.update_pdb()

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
        - Waters excluded (``_TLEAP_EXCLUDE_RESIDUES``) — tleap reorders
          waters, breaking sequential atom-map strategy; no gradient loss
          since crystal waters are not primary refinement targets
        - Monatomic ions (MG, ZN, CA, …) ARE included — covered by
          ``leaprc.water.tip3p`` (Li/Merz 12-6 set), appear in fixed PDB
          order, important for electrostatics near charged ligands
        - Terminal atoms tleap regenerates (OXT …) excluded

        Note: uses ``AMBER14_STANDARD`` (not ``_MODELLER_FF_RESIDUES``)
        so that ions absent from amber14-all.xml are still sent to tleap.
        Index is preserved (original model.pdb row positions).
        """
        pdb = self._model.update_pdb()

        mask = pdb["altloc"].astype(str).str.strip().isin(["", "A"])
        mask &= ~pdb["element"].astype(str).str.strip().isin(["H", "D"])

        # Allow AMBER-standard residues; exclude HOH/WAT and non-standard HETATM
        res_col = pdb["resname"].astype(str).str.strip()
        tleap_allowed = AMBER14_STANDARD - _TLEAP_EXCLUDE_RESIDUES
        mask &= res_col.isin(tleap_allowed)

        # Strip tleap-regenerated terminal atoms
        mask &= ~pdb["name"].astype(str).str.strip().isin(_TLEAP_SKIP_ATOMS)

        return pdb[mask].copy()

    # ------------------------------------------------------------------
    # Helpers: ordered residue → model-row mapping (for atom map)
    # ------------------------------------------------------------------

    @staticmethod
    def _ordered_residue_map(df) -> List[Dict[str, int]]:
        """
        Given a DataFrame with original model.pdb index, produce an ordered
        list of dicts ``{atom_name: model_row_index}`` — one per unique
        residue in the order the residues appear.  Used to match the
        sequential tleap topology back to model atoms.
        """
        result: List[Dict[str, int]] = []
        seen: list = []  # ordered unique residue keys

        # Use a stable groupby preserving first-appearance order
        res_key_col = list(
            zip(
                df["chainid"].astype(str).str.strip(),
                df["resseq"].astype(int),
                df.get("icode", "").astype(str).str.strip()
                if "icode" in df.columns else [""] * len(df),
                df["resname"].astype(str).str.strip(),
            )
        )

        rk_to_idx: Dict[tuple, int] = {}
        for row_idx, rk in zip(df.index, res_key_col):
            if rk not in rk_to_idx:
                rk_to_idx[rk] = len(result)
                result.append({})
            atom_name = str(df.loc[row_idx, "name"]).strip()
            result[rk_to_idx[rk]][atom_name] = int(row_idx)

        return result

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
        AMBER14 standard-residue path using OpenMM Modeller (no tleap/pdbfixer).
        """
        from torchref.io import pdb as pdbio  # noqa: PLC0415

        # Standard path: Modeller preserves chain/resseq → use key-based mapping
        self._tleap_residue_map = None

        pdb_heavy = self._filter_pdb_for_omm(include_nonstandard=False)

        tmp = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
        tmp.close()
        try:
            pdbio.write(pdb_heavy, tmp.name)
            pdb_omm = app.PDBFile(tmp.name)
        finally:
            os.unlink(tmp.name)

        ff = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
        modeller = app.Modeller(pdb_omm.topology, pdb_omm.positions)
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

        Protein heavy atoms (no OXT, no H, no non-standard) + each ligand
        mol2 are combined by tleap ``combine{}``.  parmed loads the resulting
        prmtop/inpcrd and creates the OpenMM system.

        Also builds ``self._tleap_residue_map``: ordered list of
        ``{atom_name: model_row_idx}`` dicts, one per tleap residue, used
        by :meth:`_build_atom_map` to recover the model↔OpenMM correspondence
        (tleap strips chain IDs and renumbers residues sequentially).
        """
        import parmed as pmd  # noqa: PLC0415
        from torchref.io import pdb as pdbio  # noqa: PLC0415

        work_dir = Path(tempfile.mkdtemp(prefix="amber_gaff2_"))
        try:
            prot_pdb = work_dir / "protein.pdb"
            pdb_tleap = self._filter_pdb_for_tleap()

            # Build ordered residue map BEFORE writing (needed for atom map)
            protein_res_map = self._ordered_residue_map(pdb_tleap)

            # Add each ligand residue: atom_name → model row idx (primary, heavy)
            model_pdb = self._model.pdb
            ligand_res_maps: List[Dict[str, int]] = []
            for rn in gaff2_params:
                lig_df = model_pdb[
                    model_pdb["resname"].astype(str).str.strip() == rn
                ]
                lig_df = lig_df[
                    lig_df["altloc"].astype(str).str.strip().isin(["", "A"]) &
                    ~lig_df["element"].astype(str).str.strip().isin(["H", "D"])
                ]
                d: Dict[str, int] = {}
                for row_idx in lig_df.index:
                    aname = str(model_pdb.loc[row_idx, "name"]).strip()
                    d[aname] = int(row_idx)
                ligand_res_maps.append(d)

            self._tleap_residue_map: Optional[List[Dict[str, int]]] = (
                protein_res_map + ligand_res_maps
            )

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
                [TLEAP, "-f", str(work_dir / "tleap.in")],
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
        unmatched atoms (H atoms, altloc atoms excluded from topology, …).

        Two strategies depending on how the system was built:

        **Standard path** (``_tleap_residue_map is None``):
        Modeller preserves chain IDs and residue numbers from the input PDB,
        so matching uses the key ``(chain_id, resseq, icode, atom_name)``.

        **GAFF2 path** (``_tleap_residue_map is not None``):
        tleap strips chain IDs and renumbers residues sequentially.  Matching
        uses the ordered list of per-residue ``{atom_name: model_row_idx}``
        dicts stored in ``_tleap_residue_map`` during :meth:`_build_gaff2`.
        """
        pdb = self._model.pdb
        n_model = len(pdb)
        model_to_omm = np.full(n_model, -1, dtype=np.int32)

        if self._tleap_residue_map is None:
            # ---- Standard path: match by (chain, resseq, icode, atom_name) ----
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
            # ---- GAFF2 path: match by sequential residue order + atom_name ----
            res_map = self._tleap_residue_map  # list of {name: model_row_idx}

            for res_seq_idx, omm_res in enumerate(self._topology.residues()):
                if res_seq_idx >= len(res_map):
                    break  # topology has more residues than expected (water?)
                name_to_model = res_map[res_seq_idx]
                for omm_atom in omm_res.atoms():
                    if self._is_hydrogen(omm_atom):
                        continue
                    model_idx = name_to_model.get(omm_atom.name.strip())
                    if model_idx is not None:
                        model_to_omm[model_idx] = omm_atom.index

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
                f"(all expected: waters/ions/altloc-B/OXT)"
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
    # Step 5 — OpenMM Context
    # ------------------------------------------------------------------

    def _build_context(self, pos_nm: np.ndarray) -> None:
        """
        Create an OpenMM Context on the best available platform.
        Tries preferred → OpenCL → CPU.
        """
        import openmm as mm  # noqa: PLC0415

        seen: set = set()
        platforms = [
            p for p in [self._preferred_platform, "OpenCL", "CPU"]
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

    def forward(self) -> torch.Tensor:
        """
        Compute AMBER14 energy for current model coordinates.

        Returns
        -------
        torch.Tensor
            Scalar energy in kJ/mol (or kJ/mol/atom if normalize_by_atoms).
            Gradient flows to ``model.xyz`` via OpenMM analytical forces.
        """
        if self._context is None:
            raise RuntimeError(
                "[AmberTarget] Not initialised. Pass model= to constructor."
            )

        xyz = self._model.xyz()  # (n_model_atoms, 3), Å

        energy = _OpenMMAMBERFunction.apply(
            xyz,
            self._context,
            self._model_to_omm,
            self._pos_buf,
        )

        if self._normalize:
            energy = energy / self._n_model_atoms

        return energy

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
