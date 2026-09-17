"""Evaluate AMBER energies and forces for TorchRef-owned atomic coordinates.

The model must already contain the atoms and protonation state required by the
force field. Construction validates a one-to-one atom map; it does not add atoms
to the model. Each evaluation reorders all coordinates, including hydrogens,
converts Cartesian Å to nm, and returns OpenMM forces through PyTorch autograd.
Riding geometry and orientation parameters belong to the model.
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
from torchref.config import get_float_dtype, get_int_dtype
from torchref.refinement.targets.base import ModelTarget
from torchref.utils.stats import (
    VERBOSITY_DEBUG,
    VERBOSITY_DETAILED,
    VERBOSITY_STANDARD,
    StatEntry,
    stat,
)

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

_TLEAP_SKIP_ATOMS: frozenset = frozenset({"OXT", "OT1", "OT2"})

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

    The input contains every model atom in OpenMM order. Backward returns
    minus the force in kJ/mol/nm; the upstream gather and Å-to-nm conversion
    return each gradient to its TorchRef coordinate or riding parameter.
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
# AmberTarget
# ---------------------------------------------------------------------------


class AmberTarget(ModelTarget):
    """
    Differentiable AMBER14/GAFF2 force-field energy restraint.

    Build chemistry once, then supply current coordinates for every atom to
    OpenMM. The loss never generates or independently places hydrogens.

    Parameters
    ----------
    model : Model
        Fully prepared, single-conformation model, including hydrogens and
        terminal atoms required by AMBER. Existing atoms and coordinates are
        preserved. Prepare protonation before constructing this target and
        enable riding mode on the model when hydrogen geometry is constrained.
        Incomplete or incompatible chemistry raises ValueError during setup.
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

    Notes
    -----
    Reconstruct the target after changing atom identities, atom order, or
    connectivity. Cartesian coordinate and riding-parameter changes need no
    rebuild. OpenMM evaluation transfers coordinates and forces through CPU
    memory and supports first derivatives only. Forces above 10000 kJ/mol/nm
    are clipped per atom; in that regime the returned gradient is clipped
    rather than the exact energy derivative.
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
    ) -> None:
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

        self.register_buffer(
            "_cutoff_buf",
            torch.tensor(float(cutoff), dtype=get_float_dtype(), device=self.device),
        )

        # Internal state (None until fully initialised)
        self._context = None
        self._platform_name: str = "none"
        self._model_to_omm: Optional[np.ndarray] = None
        self._pos_buf: Optional[np.ndarray] = None
        self._n_omm_atoms: int = 0
        self._n_model_atoms: int = 0
        self._n_nonstandard: int = 0
        # tleap renumbers residues; identify their original model instances.
        self._tleap_residue_map: Optional[bool] = None
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

        xyz = self._chem_model.xyz().detach().cpu().numpy()
        positions_nm = np.asarray(
            xyz[np.argsort(self._model_to_omm)] * 0.1, dtype=np.float64
        )
        self._build_context(positions_nm)

        self._pos_buf = positions_nm.copy()
        self._n_model_atoms = len(self._chem_model.pdb)

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





    def _run_antechamber_one(
        self, resname: str, charge: int
    ) -> Tuple[str, Path, Path]:
        """
        Run antechamber + parmchk2 for one non-standard residue.

        Cache is checked first.  On a miss, work happens in a temp dir and
        results are atomically moved to the cache (write-then-rename).
        """
        pdb = self._chem_model.pdb.copy()
        pdb[["x", "y", "z"]] = self._chem_model.xyz().detach().cpu().numpy()
        res_atoms = pdb[pdb["resname"].astype(str).str.strip() == resname]
        first = res_atoms.iloc[0]
        for column in ("chainid", "resseq", "icode"):
            res_atoms = res_atoms[res_atoms[column] == first[column]]
        atom_names = res_atoms["name"].astype(str).str.strip().tolist()

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

            antechamber_input = lig_pdb

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



    def _filter_pdb_for_tleap(self):
        """Export standard heavy atoms for tleap template parameterisation.

        The resulting topology must map back to every model atom, including
        hydrogens and terminal oxygens, before a context can be constructed.
        """
        pdb = self._chem_model.pdb.copy()
        pdb[["x", "y", "z"]] = self._chem_model.xyz().detach().cpu().numpy()

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
        """Parameterise the model with AMBER14 or AMBER14/GAFF2."""
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
        """Parameterise existing atoms, retaining PDB serials through name aliases."""
        from torchref.io import pdb as pdbio

        self._tleap_residue_map = None
        pdb = self._chem_model.pdb.copy()
        xyz = self._chem_model.xyz().detach().cpu().numpy()
        pdb[["x", "y", "z"]] = xyz
        pdb["serial"] = np.arange(1, len(pdb) + 1)
        with tempfile.TemporaryDirectory(prefix="torchref_amber_") as directory:
            filename = str(Path(directory) / "model.pdb")
            pdbio.write(pdb, filename)
            parsed = app.PDBFile(filename)

        topology = parsed.topology
        source_rows = np.array([int(a.id) - 1 for a in topology.atoms()])
        if len(source_rows) != len(pdb) or not np.array_equal(
            np.sort(source_rows), np.arange(len(pdb))
        ):
            raise ValueError(
                "[AmberTarget] PDB atom identities are ambiguous or duplicated. "
                "Every TorchRef atom must correspond to exactly one AMBER particle."
            )
        self._source_model_rows = source_rows
        # Preserve explicit covalent links that the PDB bond templates cannot
        # infer. PDB parsing also resolves standard hydrogen-name aliases.
        atoms = list(topology.atoms())
        inverse = np.argsort(source_rows)
        existing_bonds = {tuple(sorted((a.index, b.index))) for a, b in topology.bonds()}
        for i, j in self._chem_model.restraints.topology.atoms.bonds.indices.cpu().tolist():
            pair = tuple(sorted((int(inverse[i]), int(inverse[j]))))
            if pair not in existing_bonds:
                topology.addBond(atoms[pair[0]], atoms[pair[1]])
                existing_bonds.add(pair)
        ff = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
        try:
            system = ff.createSystem(
                topology,
                nonbondedMethod=app.CutoffNonPeriodic,
                nonbondedCutoff=cutoff_A * unit.angstrom,
                constraints=None,
                rigidWater=False,
            )
        except ValueError as exc:
            raise ValueError(
                "[AmberTarget] TorchRef model is not AMBER-compatible. "
                "Prepare missing atoms, terminal groups and protonation in the "
                "model before constructing the loss; no atoms were added. "
                f"OpenMM: {exc}"
            ) from exc
        xyz = self._chem_model.xyz().detach().cpu().numpy()
        return system, topology, np.asarray(xyz[source_rows] * 0.1, dtype=np.float64)

    def _build_gaff2(
        self,
        gaff2_params: Dict[str, Tuple[Path, Path]],
        cutoff_A: float,
        app,
        unit,
    ) -> Tuple:
        """Build AMBER14/GAFF2 templates with one copy per ligand instance.

        tleap may reorder or rename atoms; the complete map is validated before
        its system is used. Runtime coordinates always come from TorchRef.
        """
        import parmed as pmd  # noqa: PLC0415
        from torchref.io import pdb as pdbio  # noqa: PLC0415

        work_dir = Path(tempfile.mkdtemp(prefix="amber_gaff2_"))
        try:
            prot_pdb = work_dir / "protein.pdb"
            pdb_tleap = self._filter_pdb_for_tleap()

            # tleap does not preserve the original chain and residue identifiers.
            self._tleap_residue_map = True
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

            ligand_copies = []
            ligand_keys = []
            pdb = self._chem_model.pdb
            for rn in lig_names:
                rows = pdb[pdb["resname"].astype(str).str.strip() == rn]
                for key, _ in rows.groupby(["chainid", "resseq", "icode"], sort=False):
                    copy_name = f"ligand{len(ligand_copies)}"
                    lig_loads.append(f"{copy_name} = copy {rn}")
                    ligand_copies.append(copy_name)
                    ligand_keys.append(tuple(key))
            self._gaff2_residue_keys = ligand_keys
            combine_list = " ".join(["protein"] + ligand_copies)
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
                rigidWater=False,
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
        """Require a bijection between model rows and all OpenMM particles."""
        pdb = self._chem_model.pdb
        n_model = len(pdb)
        atoms = list(self._topology.atoms())
        mapping = np.full(n_model, -1, dtype=np.int32)
        if self._tleap_residue_map is None:
            mapping[self._source_model_rows] = np.arange(len(atoms))
        else:
            self._map_gaff2_atoms(mapping, atoms)

        mapped = mapping[mapping >= 0]
        missing_model = np.flatnonzero(mapping < 0)
        missing_omm = sorted(set(range(len(atoms))) - set(mapped.tolist()))
        duplicate = len(np.unique(mapped)) != len(mapped)
        if (
            len(atoms) != self._system.getNumParticles()
            or duplicate
            or len(missing_model)
            or missing_omm
        ):
            model_examples = [
                f"{pdb.iloc[i]['chainid']}:{pdb.iloc[i]['resseq']}:"
                f"{pdb.iloc[i]['name']}"
                for i in missing_model[:5]
            ]
            omm_examples = [
                f"{atoms[i].residue.name}:{atoms[i].residue.id}:{atoms[i].name}"
                for i in missing_omm[:5]
            ]
            raise ValueError(
                "[AmberTarget] AMBER atom mapping is not one-to-one: "
                f"unmatched model atoms={model_examples}, "
                f"unmatched AMBER atoms={omm_examples}, duplicate matches={duplicate}. "
                "Prepare matching atoms and protonation in TorchRef before "
                "constructing the loss."
            )
        self._model_to_omm = mapping
        self._n_omm_atoms = len(atoms)
        inverse = np.argsort(mapping)
        self.register_buffer(
            "_omm_to_model",
            torch.as_tensor(inverse, dtype=get_int_dtype(), device=self._chem_model.device),
        )

    def _map_gaff2_atoms(self, mapping: np.ndarray, atoms: list) -> None:
        """Match residue instances using heavy anchors, then names and H parents."""
        from scipy.spatial import cKDTree

        pdb = self._chem_model.pdb
        keys = [
            tuple(row)
            for row in pdb[["chainid", "resseq", "icode"]].itertuples(
                index=False, name=None
            )
        ]
        groups = {}
        for i, key in enumerate(keys):
            groups.setdefault(key, []).append(i)
        residues = list(self._topology.residues())
        ligand_keys = self._gaff2_residue_keys
        ligand_residues = (
            residues[len(residues) - len(ligand_keys) :] if ligand_keys else []
        )
        residue_map = {res.index: key for res, key in zip(ligand_residues, ligand_keys)}
        xyz_nm = self._chem_model.xyz().detach().cpu().numpy() * 0.1
        names = pdb["name"].astype(str).str.strip().to_numpy()
        elements = pdb["element"].astype(str).str.strip().str.upper().to_numpy()
        heavy_rows = np.flatnonzero(~np.isin(elements, ["H", "D"]))
        tree = cKDTree(xyz_nm[heavy_rows])
        for residue in residues:
            if residue.index in residue_map:
                continue
            candidates = set()
            for atom in residue.atoms():
                if self._is_hydrogen(atom):
                    continue
                for local in tree.query_ball_point(self._tleap_pos_nm[atom.index], 0.005):
                    row = heavy_rows[local]
                    if elements[row] == atom.element.symbol.upper():
                        candidates.add(keys[row])
            if len(candidates) != 1:
                raise ValueError(
                    f"[AmberTarget] Cannot uniquely identify AMBER residue "
                    f"{residue.name} {residue.id} in TorchRef."
                )
            residue_map[residue.index] = candidates.pop()
        if len(set(residue_map.values())) != len(residue_map):
            raise ValueError(
                "[AmberTarget] Multiple AMBER residues match one model residue."
            )

        model_parents = {}
        graph = self._chem_model.restraints.topology.atoms
        for i, j in graph.bonds.indices.cpu().tolist():
            if elements[i] in {"H", "D"} and elements[j] not in {"H", "D"}:
                model_parents[i] = j
            elif elements[j] in {"H", "D"} and elements[i] not in {"H", "D"}:
                model_parents[j] = i
        omm_parents = {}
        for a, b in self._topology.bonds():
            if self._is_hydrogen(a) and not self._is_hydrogen(b):
                omm_parents[a.index] = b.index
            elif self._is_hydrogen(b) and not self._is_hydrogen(a):
                omm_parents[b.index] = a.index
        for residue in residues:
            rows = groups.get(residue_map[residue.index], [])
            by_name = {names[i]: i for i in rows}
            if len(by_name) != len(rows):
                raise ValueError(
                    "[AmberTarget] Duplicate atom names within a model residue."
                )
            for atom in residue.atoms():
                row = by_name.get(atom.name)
                if row is None and atom.name in {"H", "H1"}:
                    row = by_name.get("H1" if atom.name == "H" else "H")
                if row is None and not self._is_hydrogen(atom):
                    candidates = [
                        i
                        for i in rows
                        if mapping[i] < 0
                        and elements[i] == atom.element.symbol.upper()
                        and (
                            len(rows) == 1
                            or np.linalg.norm(xyz_nm[i] - self._tleap_pos_nm[atom.index])
                            < 0.005
                        )
                    ]
                    if len(candidates) == 1:
                        row = candidates[0]
                if row is not None:
                    symbol = "H" if elements[row] == "D" else elements[row]
                    if symbol == atom.element.symbol.upper() and mapping[row] < 0:
                        mapping[row] = atom.index
            for row in rows:
                if mapping[row] >= 0 and row in model_parents:
                    expected_parent = mapping[model_parents[row]]
                    if omm_parents.get(mapping[row]) != expected_parent:
                        raise ValueError(
                            "[AmberTarget] Hydrogen attachment differs between "
                            f"TorchRef and AMBER: {residue.name} {names[row]}."
                        )
            # Equivalent hydrogens may use different numbering conventions. Only
            # pair remaining H atoms attached to the same already-mapped parent.
            used = set(mapping[mapping >= 0].tolist())
            for row in rows:
                if mapping[row] >= 0 or row not in model_parents:
                    continue
                parent = mapping[model_parents[row]]
                choices = sorted(
                    a.index
                    for a in residue.atoms()
                    if a.index not in used and omm_parents.get(a.index) == parent
                )
                if choices:
                    mapping[row] = choices[0]
                    used.add(choices[0])

    def _compose_full_omm_xyz(self, model_xyz_ang: torch.Tensor) -> torch.Tensor:
        """Gather all Cartesian model coordinates into OpenMM order and nm."""
        if model_xyz_ang.shape != (self._n_model_atoms, 3):
            raise ValueError(
                "[AmberTarget] Atom count changed; rebuild the target after "
                "changing model topology."
            )
        if self._omm_to_model.device != model_xyz_ang.device:
            self._omm_to_model = self._omm_to_model.to(model_xyz_ang.device)
        return model_xyz_ang.index_select(0, self._omm_to_model) * 0.1

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
        """Evaluate all-atom Cartesian coordinates in Å, in chemistry-model order.

        Return a scalar in kJ/mol, divided by the atom count when normalization
        is enabled. Gradients flow through the model's own coordinate wrapper.
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
