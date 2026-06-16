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
OpenMM's ``Modeller.addHydrogens()`` is 4–5× faster when H atoms are already
present in the model (it refines positions rather than building from scratch).
For pure-protein structures: init ~3 s with H, ~11 s from heavy atoms only.
Gradient and energy are identical either way (H are stripped from the atom map;
``n_model_atoms`` changes only the energy normalisation).

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

        # Clamp per-atom force magnitude to prevent extreme LJ clashes
        # from producing gradients that blow up the optimizer.
        # 1000 kJ/mol/Å ≈ force from a ~0.3 Å LJ overlap.
        max_force = 1000.0  # kJ/mol/Å
        norms = np.linalg.norm(model_forces, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        scale = np.minimum(max_force / norms, 1.0)
        model_forces *= scale

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
        initialisation ~4× because ``Modeller.addHydrogens()`` converges
        faster from existing positions.

        **Required for GAFF2 ligands**: antechamber's BCC charge scheme
        runs a semiempirical QM step (sqm) that requires a fully protonated
        molecule.  If the model has no H atoms for a non-standard residue,
        an explicit error is raised.  Call ``model.generate_hydrogens()``
        or load the PDB with ``strip_H=False`` before creating the target.
    cutoff : float
        Non-bonded cutoff in Angstroms.  Default 5.0.
    normalize_by_atoms : bool
        If True the energy is divided by the number of model atoms.
        Default True.
    residue_charges : dict[str, int], optional
        Net formal charge per non-standard residue name,
        e.g. ``{'LIG': -1, 'ATP': -4}``.  Residues not listed default to 0
        with a warning.
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
        gaff2_files: Optional[Dict[str, Tuple[str, str]]] = None,
        verbose: int = 0,
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

        if model is None:
            return  # Allow empty init for state_dict loading

        self._build(model)

    # ------------------------------------------------------------------
    # Top-level build orchestration
    # ------------------------------------------------------------------

    def _build(self, model: "Model") -> None:
        """Detect → antechamber → build OpenMM system → map atoms."""
        # Reject models with alternate conformations — OpenMM only handles
        # a single conformation.  Call model.strip_altlocs() first.
        altlocs = model.pdb["altloc"].astype(str).str.strip()
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

        # antechamber's BCC charges require a fully protonated molecule so that
        # the semiempirical QM step (sqm) has an even electron count.
        # Detect odd-electron count early and emit a clear error.
        _Z = {"H":1,"He":2,"Li":3,"Be":4,"B":5,"C":6,"N":7,"O":8,"F":9,"Ne":10,
               "Na":11,"Mg":12,"Al":13,"Si":14,"P":15,"S":16,"Cl":17,"Ar":18,
               "K":19,"Ca":20,"Cr":24,"Mn":25,"Fe":26,"Co":27,"Ni":28,"Cu":29,
               "Zn":30,"Br":35,"I":53,"Se":34,"Mo":42,"W":74,"Pt":78,"Au":79}
        elems = res_atoms["element"].astype(str).str.strip().str.capitalize()
        n_protons = sum(_Z.get(e, 0) for e in elems)
        n_electrons = n_protons - charge
        if n_electrons % 2 != 0:
            # Odd electron count → sqm cannot converge.  Almost certainly caused
            # by missing H atoms in an organic molecule.
            has_h = elems.isin(["H", "D"]).any()
            raise RuntimeError(
                f"[AmberTarget] Cannot run antechamber for '{resname}': "
                f"odd electron count ({n_electrons}) for charge {charge:+d}.\n"
                + (
                    "The model has no H atoms for this residue; sqm (inside antechamber) "
                    "requires a fully protonated molecule.\n"
                    "Fix: call model.generate_hydrogens() or load the PDB with "
                    "strip_H=False before creating AmberTarget."
                    if not has_h else
                    f"Check the net charge (residue_charges={{'{resname}': <charge>}}) "
                    f"and verify the protonation state."
                )
            )

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
                    _find_ambertools_binary("antechamber"),
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
        pdb = self._model.update_pdb()

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

        pdb = self._model.pdb
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
            fresh_pdb = self._model.update_pdb()
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
    # Step 5 — OpenMM Context
    # ------------------------------------------------------------------

    def _build_context(self, pos_nm: np.ndarray) -> None:
        """
        Create an OpenMM Context on the platform that matches the model's device.

        Mapping: ``model.device.type == 'cuda'`` → CUDA, otherwise CPU.
        Falls back CUDA → OpenCL → CPU if the preferred platform is unavailable.
        """
        import openmm as mm  # noqa: PLC0415

        device_type = getattr(self._model.device, "type", "cpu")
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
