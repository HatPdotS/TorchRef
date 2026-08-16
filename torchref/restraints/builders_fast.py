"""Restraint builders that walk the whole structure inside a single ``build()``.

One ``build(pdb, cif_dict, device)`` call per restraint type handles every residue
internally -- callers do not loop -- or :func:`build_all_restraints` does all the
intra-residue types at once. Inter-residue links go through the
``InterResidue*Builder`` classes, whose ``build()`` returns directly like the others;
only their disulfide path is stateful -- it accumulates over ``process_disulfide_*``
calls and emits nothing until ``finalize()`` (``finalize_disulfide()`` on the torsion
builder).

Nothing here is re-exported at the ``torchref.restraints`` level; import from
``torchref.restraints.builders_fast``.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from torchref.config import get_float_dtype

# Import the Numba-accelerated matching functions
from torchref.restraints.builders_numba import (
    match_angles_numba,
    match_bonds_numba,
    match_chirals_numba,
    match_torsions_numba,
)


# =============================================================================
# Pre-processing utilities
# =============================================================================


class PreprocessedPDB:
    """
    Pre-processed PDB data as NumPy arrays for fast iteration.

    Converts DataFrame to arrays once, computes residue boundaries,
    enabling O(1) access to residue data without DataFrame operations.

    Supports altloc expansion: residues with alternate conformations can be
    expanded into multiple conformations, each with common atoms plus the
    specific altloc atoms. The constructor only normalizes altlocs and
    computes residue boundaries; the expansion itself is performed on demand
    by :meth:`get_altloc_conformations`, not at preprocessing time.
    """

    def __init__(self, pdb: pd.DataFrame):
        """
        Initialize from PDB DataFrame.

        Parameters
        ----------
        pdb : pd.DataFrame
            PDB DataFrame with standard columns.
        """
        self.n_atoms = len(pdb)

        # Core arrays
        self.atom_names = pdb["name"].values.astype(str)
        self.atom_indices = pdb["index"].values.astype(np.int64)
        self.chain_ids = pdb["chainid"].values.astype(str)
        self.resseqs = pdb["resseq"].values.astype(np.int64)
        self.resnames = pdb["resname"].values.astype(str)

        # Optional columns - normalize altloc (treat '' as ' ')
        if "altloc" in pdb.columns:
            altlocs = pdb["altloc"].values.astype(str)
            # Normalize: treat '' as ' ' (no altloc)
            altlocs = np.where(altlocs == "", " ", altlocs)
            self.altlocs = altlocs
        else:
            self.altlocs = np.full(self.n_atoms, " ", dtype="<U1")

        if "ATOM" in pdb.columns:
            self.atom_types = pdb["ATOM"].values.astype(str)
        else:
            self.atom_types = np.full(self.n_atoms, "ATOM", dtype="<U6")

        # Pre-compute residue boundaries
        self._compute_residue_boundaries()

    def _compute_residue_boundaries(self):
        """Compute start/end indices for each residue."""
        # Create residue key for grouping
        residue_keys = np.char.add(
            np.char.add(self.chain_ids.astype("<U10"), "_"), self.resseqs.astype(str)
        )

        # Find boundaries where residue changes
        changes = np.where(residue_keys[:-1] != residue_keys[1:])[0] + 1
        self.residue_starts = np.concatenate([[0], changes])
        self.residue_ends = np.concatenate([changes, [self.n_atoms]])
        self.n_residues = len(self.residue_starts)

        # Store residue metadata (from first atom of each residue)
        self.residue_chain_ids = self.chain_ids[self.residue_starts]
        self.residue_resseqs = self.resseqs[self.residue_starts]
        self.residue_resnames = self.resnames[self.residue_starts]

    def get_residue_data(self, residue_idx: int) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Get atom data for a residue.

        Returns
        -------
        atom_names : np.ndarray
            Atom names for this residue.
        atom_indices : np.ndarray
            Global atom indices.
        resname : str
            Residue name.
        """
        start = self.residue_starts[residue_idx]
        end = self.residue_ends[residue_idx]
        return (
            self.atom_names[start:end],
            self.atom_indices[start:end],
            self.residue_resnames[residue_idx],
        )

    def residue_keys(
        self, mapping: Optional[Mapping[Tuple[str, int], str]] = None
    ) -> List[str]:
        """Return the restraint-dictionary key of each residue, by residue index.

        Parameters
        ----------
        mapping : mapping, optional
            ``{(chain_id, resseq): key}`` overriding the residue name for those
            residues -- how a linked residue is pointed at a modified copy of its
            component (see :mod:`torchref.restraints.modifications`). Residues
            absent from it, and every residue when this is None, key on their own
            residue name, which is the unmodified behaviour.

        Returns
        -------
        list of str
            One key per residue, indexed as ``residue_resnames``.
        """
        if not mapping:
            return list(self.residue_resnames)
        return [
            mapping.get(
                (str(self.residue_chain_ids[i]), int(self.residue_resseqs[i])),
                self.residue_resnames[i],
            )
            for i in range(self.n_residues)
        ]

    def has_duplicate_atoms(self, residue_idx: int) -> bool:
        """Check if residue has duplicate atom names (altlocs)."""
        start = self.residue_starts[residue_idx]
        end = self.residue_ends[residue_idx]
        names = self.atom_names[start:end]
        return len(names) != len(set(names))

    def has_altlocs(self, residue_idx: int) -> bool:
        """Check if residue has any alternate conformations."""
        start = self.residue_starts[residue_idx]
        end = self.residue_ends[residue_idx]
        altlocs = self.altlocs[start:end]
        # Has altlocs if any altloc is not ' ' (normalized no-altloc marker)
        return np.any(altlocs != " ")

    def get_altloc_conformations(
        self, residue_idx: int
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, str]]:
        """
        Iterate over altloc conformations for a residue.

        For residues without altlocs, yields once with all atoms.
        For residues with altlocs, yields once per unique altloc,
        each time with common atoms (no altloc) + altloc-specific atoms.

        Yields
        ------
        atom_names : np.ndarray
            Atom names for this conformation.
        atom_indices : np.ndarray
            Global atom indices.
        resname : str
            Residue name.
        """
        start = self.residue_starts[residue_idx]
        end = self.residue_ends[residue_idx]

        names = self.atom_names[start:end]
        indices = self.atom_indices[start:end]
        altlocs = self.altlocs[start:end]
        resname = self.residue_resnames[residue_idx]

        unique_altlocs = np.unique(altlocs)

        if len(unique_altlocs) == 1 and unique_altlocs[0] == " ":
            # No altlocs - yield all atoms once
            yield names, indices, resname
        elif " " in unique_altlocs:
            # Has common atoms (no altloc) and altloc-specific atoms
            # Common atoms mask
            common_mask = altlocs == " "
            common_names = names[common_mask]
            common_indices = indices[common_mask]

            # Yield once per specific altloc (A, B, etc.)
            for alt in unique_altlocs:
                if alt == " ":
                    continue
                alt_mask = altlocs == alt
                # Combine common + altloc-specific
                combined_names = np.concatenate([common_names, names[alt_mask]])
                combined_indices = np.concatenate([common_indices, indices[alt_mask]])
                yield combined_names, combined_indices, resname
        else:
            # No common atoms - yield each altloc separately
            for alt in unique_altlocs:
                alt_mask = altlocs == alt
                yield names[alt_mask], indices[alt_mask], resname


class PreprocessedCIF:
    """
    Pre-processed CIF restraints as NumPy arrays per residue type.
    """

    def __init__(self, cif_dict: Dict):
        """
        Initialize from CIF dictionary.

        Parameters
        ----------
        cif_dict : dict
            CIF dictionary with restraints per residue type.
        """
        self.residue_types = list(cif_dict.keys())

        # Pre-process each restraint type
        self.bonds = {}
        self.angles = {}
        self.torsions = {}
        self.planes = {}
        self.chirals = {}

        for restype, data in cif_dict.items():
            if "bonds" in data and len(data["bonds"]) > 0:
                self.bonds[restype] = self._preprocess_bonds(data["bonds"])
            if "angles" in data and len(data["angles"]) > 0:
                self.angles[restype] = self._preprocess_angles(data["angles"])
            if "torsions" in data and len(data["torsions"]) > 0:
                result = self._preprocess_torsions(data["torsions"])
                if result is not None:
                    self.torsions[restype] = result
            if "planes" in data and len(data["planes"]) > 0:
                self.planes[restype] = self._preprocess_planes(data["planes"])
            if "chirals" in data and len(data["chirals"]) > 0:
                self.chirals[restype] = self._preprocess_chirals(data["chirals"])

    def _preprocess_bonds(self, bonds_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Convert bonds DataFrame to NumPy arrays."""
        return {
            "atom1": bonds_df["atom1"].values.astype(str),
            "atom2": bonds_df["atom2"].values.astype(str),
            "value": bonds_df["value"].values.astype(np.float64),
            "sigma": bonds_df["sigma"].values.astype(np.float64),
        }

    def _preprocess_angles(self, angles_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Convert angles DataFrame to NumPy arrays."""
        return {
            "atom1": angles_df["atom1"].values.astype(str),
            "atom2": angles_df["atom2"].values.astype(str),
            "atom3": angles_df["atom3"].values.astype(str),
            "value": angles_df["value"].values.astype(np.float64),
            "sigma": angles_df["sigma"].values.astype(np.float64),
        }

    # Backbone heavy atoms — torsions where ALL four atoms fall in this set
    # are phi/psi-equivalent and must NOT be restrained as intra-residue
    # torsions (they conflict with Ramachandran-favored angles).
    # Example: CIF "sp2_sp3_1  O C CA N  0.0 10.0 6" directly restrains psi.
    _BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O", "OXT"})

    def _preprocess_torsions(self, torsions_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Convert torsions to NumPy arrays, dropping backbone-only torsions.

        A torsion over four backbone heavy atoms (N, CA, C, O, OXT) is
        phi/psi-equivalent and would fight the Ramachandran term, so it is removed
        here rather than downweighted.
        """
        bb = self._BACKBONE_ATOMS
        keep = np.array([
            not ({a1, a2, a3, a4} <= bb)
            for a1, a2, a3, a4 in zip(
                torsions_df["atom1"].values,
                torsions_df["atom2"].values,
                torsions_df["atom3"].values,
                torsions_df["atom4"].values,
            )
        ], dtype=bool)
        torsions_df = torsions_df[keep].reset_index(drop=True)

        if len(torsions_df) == 0:
            return None

        # Handle both 'period' and 'periodicity' column names
        if "periodicity" in torsions_df.columns:
            periods = torsions_df["periodicity"].values.astype(np.int64)
        elif "period" in torsions_df.columns:
            periods = torsions_df["period"].values.astype(np.int64)
        else:
            periods = np.ones(len(torsions_df), dtype=np.int64)

        return {
            "atom1": torsions_df["atom1"].values.astype(str),
            "atom2": torsions_df["atom2"].values.astype(str),
            "atom3": torsions_df["atom3"].values.astype(str),
            "atom4": torsions_df["atom4"].values.astype(str),
            "value": torsions_df["value"].values.astype(np.float64),
            "sigma": torsions_df["sigma"].values.astype(np.float64),
            "period": periods,
        }

    def _preprocess_planes(self, planes_df: pd.DataFrame) -> List[Dict]:
        """Convert planes DataFrame to list of plane data."""
        plane_ids = planes_df["plane_id"].unique()
        planes_data = []

        for plane_id in plane_ids:
            plane_atoms = planes_df[planes_df["plane_id"] == plane_id]
            sigma_col = "sigma" if "sigma" in plane_atoms.columns else None
            planes_data.append(
                {
                    "atoms": plane_atoms["atom"].values.astype(str),
                    "sigmas": (
                        plane_atoms[sigma_col].values.astype(np.float64)
                        if sigma_col
                        else np.full(len(plane_atoms), 0.02)
                    ),
                }
            )

        return planes_data

    def _preprocess_chirals(self, chirals_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Convert chirals DataFrame to NumPy arrays."""
        # Convert volume_sign strings to floats. The CCP4 library writes both the
        # full and the truncated spelling ("positiv", "negativ"); an unrecognised
        # sign becomes NaN and the restraint is then dropped in builders_numba,
        # so the short forms have to be matched here or those chirals vanish.
        volume_signs = []
        for sign in chirals_df["volume_sign"].values:
            sign = str(sign).strip().lower()
            if sign.startswith("positiv"):
                volume_signs.append(1.0)
            elif sign.startswith("negativ"):
                volume_signs.append(-1.0)
            elif sign in ["both", "either"]:
                volume_signs.append(0.0)
            else:
                volume_signs.append(np.nan)

        sigma_col = "sigma" if "sigma" in chirals_df.columns else None

        return {
            "center": chirals_df["atom_centre"].values.astype(str),
            "atom1": chirals_df["atom1"].values.astype(str),
            "atom2": chirals_df["atom2"].values.astype(str),
            "atom3": chirals_df["atom3"].values.astype(str),
            "volume_sign": np.array(volume_signs, dtype=np.float64),
            "sigma": (
                chirals_df[sigma_col].values.astype(np.float64)
                if sigma_col
                else np.full(len(chirals_df), 0.2)
            ),
        }


# =============================================================================
# Residue pairing shared by the inter-residue builders
# =============================================================================


def build_residue_conformation_maps(
    pp_pdb: PreprocessedPDB,
) -> List[List[Dict[str, int]]]:
    """Atom-name-to-index maps per residue per conformer (outer, then inner list).

    One map for a residue without altlocs; otherwise one per conformer, each
    holding the common atoms plus that conformer's own.
    """
    all_maps = []
    for res_idx in range(pp_pdb.n_residues):
        res_maps = []
        for atom_names, atom_indices, _ in pp_pdb.get_altloc_conformations(res_idx):
            res_maps.append(dict(zip(atom_names, atom_indices)))
        all_maps.append(res_maps)
    return all_maps


def find_consecutive_residue_pairs(pp_pdb: PreprocessedPDB) -> List[Tuple[int, int]]:
    """Find pairs of residues numbered consecutively within one chain.

    Sequence numbering is the only criterion: no distance check, and insertion
    codes are invisible here because :class:`PreprocessedPDB` groups residues on
    ``(chain, resseq)`` alone.
    """
    pairs = []
    by_chain: Dict[Any, List[Tuple[int, int]]] = {}
    for res_idx in range(pp_pdb.n_residues):
        chain = pp_pdb.residue_chain_ids[res_idx]
        by_chain.setdefault(chain, []).append(
            (pp_pdb.residue_resseqs[res_idx], res_idx)
        )

    for residues in by_chain.values():
        residues_sorted = sorted(residues, key=lambda x: x[0])
        for i in range(len(residues_sorted) - 1):
            resseq_i, idx_i = residues_sorted[i]
            resseq_next, idx_next = residues_sorted[i + 1]
            if resseq_next == resseq_i + 1:
                pairs.append((idx_i, idx_next))
    return pairs


def find_peptide_link_pairs(pp_pdb: PreprocessedPDB) -> List[Tuple[int, int]]:
    """Consecutive residue pairs that actually carry a C-N peptide bond.

    Narrows :func:`find_consecutive_residue_pairs` to pairs where the first
    residue has a ``C`` and the second an ``N`` -- the condition
    :meth:`InterResidueBondBuilder.build` applies implicitly when it looks the two
    atoms up. Deciding which residues are peptide-linked has to agree with that
    builder exactly, or a residue could be given linked restraint targets without
    getting the link itself.

    Parameters
    ----------
    pp_pdb : PreprocessedPDB
        Preprocessed atoms, already filtered to polymer atoms by the caller.

    Returns
    -------
    list of tuple of int
        ``(residue index donating C, residue index donating N)`` pairs.
    """
    pairs = []
    for res_i, res_next in find_consecutive_residue_pairs(pp_pdb):
        names_i, _, _ = pp_pdb.get_residue_data(res_i)
        names_next, _, _ = pp_pdb.get_residue_data(res_next)
        if "C" in names_i and "N" in names_next:
            pairs.append((res_i, res_next))
    return pairs


# =============================================================================
# Builder Base Class
# =============================================================================


class RestraintBuilder(ABC):
    """
    Abstract base class for restraint builders.

    All builders share the same API:
        builder = SomeRestraintBuilder(verbose=0)
        result = builder.build(pdb, cif_dict, device)
    """

    def __init__(self, verbose: int = 0):
        """Initialize builder."""
        self.verbose = verbose

    @abstractmethod
    def build(
        self,
        pdb: pd.DataFrame,
        cif_dict: Dict,
        device: torch.device,
        sort_indices: bool = True,
        residue_keys: Optional[Mapping[Tuple[str, int], str]] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Build restraints from PDB and CIF data.

        Parameters
        ----------
        pdb : pd.DataFrame
            PDB DataFrame with atom data.
        cif_dict : dict
            CIF dictionary with restraints per residue type.
        device : torch.device
            Target device for output tensors.
        sort_indices : bool, default True
            Whether to sort by first atom index for cache efficiency.
        residue_keys : mapping, optional
            ``{(chain_id, resseq): cif_dict key}`` for residues whose restraints
            come from somewhere other than their residue name -- a peptide-linked
            residue draws from a modified copy of its component. Residues absent
            from it key on their residue name, so None reproduces the plain
            per-residue-type lookup.

        Returns
        -------
        dict or None
            Dictionary with restraint tensors, or None if no restraints found.
        """
        pass


# =============================================================================
# Bond Builder
# =============================================================================


class BondRestraintBuilder(RestraintBuilder):
    """
    Fast bond restraint builder.

    Usage:
        builder = BondRestraintBuilder()
        result = builder.build(pdb, cif_dict, device)
        # result = {'indices': tensor, 'references': tensor, 'sigmas': tensor}
    """

    def build(
        self,
        pdb: pd.DataFrame,
        cif_dict: Dict,
        device: torch.device,
        sort_indices: bool = True,
        residue_keys: Optional[Mapping[Tuple[str, int], str]] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Build all bond restraints."""
        # Pre-process data
        pp_pdb = PreprocessedPDB(pdb)
        pp_cif = PreprocessedCIF(cif_dict)
        keys = pp_pdb.residue_keys(residue_keys)

        # Allocate work arrays
        max_per_residue = 50
        work_idx1 = np.zeros(max_per_residue, dtype=np.int64)
        work_idx2 = np.zeros(max_per_residue, dtype=np.int64)
        work_refs = np.zeros(max_per_residue, dtype=np.float64)
        work_sigmas = np.zeros(max_per_residue, dtype=np.float64)

        # Accumulate results
        all_indices = []
        all_refs = []
        all_sigmas = []

        # Process all residues (with altloc expansion)
        for res_idx in range(pp_pdb.n_residues):
            key = keys[res_idx]

            # Skip if no bond restraints for this residue type
            if key not in pp_cif.bonds:
                continue

            cif_bonds = pp_cif.bonds[key]
            n_cif = len(cif_bonds["atom1"])

            # Resize work arrays if needed
            if n_cif > max_per_residue:
                max_per_residue = n_cif * 2
                work_idx1 = np.zeros(max_per_residue, dtype=np.int64)
                work_idx2 = np.zeros(max_per_residue, dtype=np.int64)
                work_refs = np.zeros(max_per_residue, dtype=np.float64)
                work_sigmas = np.zeros(max_per_residue, dtype=np.float64)

            # Iterate over altloc conformations (yields once if no altlocs)
            for atom_names, atom_indices, _ in pp_pdb.get_altloc_conformations(res_idx):
                # Use Numba-accelerated matching
                count = match_bonds_numba(
                    atom_names,
                    atom_indices,
                    cif_bonds["atom1"],
                    cif_bonds["atom2"],
                    cif_bonds["value"],
                    cif_bonds["sigma"],
                    work_idx1,
                    work_idx2,
                    work_refs,
                    work_sigmas,
                )

                if count > 0:
                    all_indices.append(
                        np.column_stack(
                            [work_idx1[:count].copy(), work_idx2[:count].copy()]
                        )
                    )
                    all_refs.append(work_refs[:count].copy())
                    all_sigmas.append(work_sigmas[:count].copy())

        # Finalize
        if not all_indices:
            return None

        indices = np.concatenate(all_indices, axis=0)
        references = np.concatenate(all_refs)
        sigmas = np.concatenate(all_sigmas)

        if sort_indices and len(indices) > 0:
            order = np.argsort(indices[:, 0])
            indices = indices[order]
            references = references[order]
            sigmas = sigmas[order]

        # Replace zero sigmas
        sigmas = np.where(sigmas == 0, 1e-4, sigmas)

        return {
            "indices": torch.tensor(indices, dtype=torch.long, device=device),
            "references": torch.tensor(references, dtype=get_float_dtype(), device=device),
            "sigmas": torch.tensor(sigmas, dtype=get_float_dtype(), device=device),
        }


# =============================================================================
# Angle Builder
# =============================================================================


class AngleRestraintBuilder(RestraintBuilder):
    """
    Fast angle restraint builder.

    Usage:
        builder = AngleRestraintBuilder()
        result = builder.build(pdb, cif_dict, device)
    """

    def build(
        self,
        pdb: pd.DataFrame,
        cif_dict: Dict,
        device: torch.device,
        sort_indices: bool = True,
        residue_keys: Optional[Mapping[Tuple[str, int], str]] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Build all angle restraints."""
        pp_pdb = PreprocessedPDB(pdb)
        pp_cif = PreprocessedCIF(cif_dict)
        keys = pp_pdb.residue_keys(residue_keys)

        max_per_residue = 100
        work_idx1 = np.zeros(max_per_residue, dtype=np.int64)
        work_idx2 = np.zeros(max_per_residue, dtype=np.int64)
        work_idx3 = np.zeros(max_per_residue, dtype=np.int64)
        work_refs = np.zeros(max_per_residue, dtype=np.float64)
        work_sigmas = np.zeros(max_per_residue, dtype=np.float64)

        all_indices = []
        all_refs = []
        all_sigmas = []

        for res_idx in range(pp_pdb.n_residues):
            key = keys[res_idx]

            if key not in pp_cif.angles:
                continue

            cif_angles = pp_cif.angles[key]
            n_cif = len(cif_angles["atom1"])

            if n_cif > max_per_residue:
                max_per_residue = n_cif * 2
                work_idx1 = np.zeros(max_per_residue, dtype=np.int64)
                work_idx2 = np.zeros(max_per_residue, dtype=np.int64)
                work_idx3 = np.zeros(max_per_residue, dtype=np.int64)
                work_refs = np.zeros(max_per_residue, dtype=np.float64)
                work_sigmas = np.zeros(max_per_residue, dtype=np.float64)

            # Iterate over altloc conformations (yields once if no altlocs)
            for atom_names, atom_indices, _ in pp_pdb.get_altloc_conformations(res_idx):
                count = match_angles_numba(
                    atom_names,
                    atom_indices,
                    cif_angles["atom1"],
                    cif_angles["atom2"],
                    cif_angles["atom3"],
                    cif_angles["value"],
                    cif_angles["sigma"],
                    work_idx1,
                    work_idx2,
                    work_idx3,
                    work_refs,
                    work_sigmas,
                )

                if count > 0:
                    all_indices.append(
                        np.column_stack(
                            [
                                work_idx1[:count].copy(),
                                work_idx2[:count].copy(),
                                work_idx3[:count].copy(),
                            ]
                        )
                    )
                    all_refs.append(work_refs[:count].copy())
                    all_sigmas.append(work_sigmas[:count].copy())

        if not all_indices:
            return None

        indices = np.concatenate(all_indices, axis=0)
        references = np.concatenate(all_refs)
        sigmas = np.concatenate(all_sigmas)

        if sort_indices and len(indices) > 0:
            order = np.argsort(indices[:, 0])
            indices = indices[order]
            references = references[order]
            sigmas = sigmas[order]

        sigmas = np.where(sigmas == 0, 1e-4, sigmas)

        return {
            "indices": torch.tensor(indices, dtype=torch.long, device=device),
            "references": torch.tensor(references, dtype=get_float_dtype(), device=device),
            "sigmas": torch.tensor(sigmas, dtype=get_float_dtype(), device=device),
        }


# =============================================================================
# Torsion Builder
# =============================================================================


class TorsionRestraintBuilder(RestraintBuilder):
    """
    Fast torsion restraint builder.

    Usage:
        builder = TorsionRestraintBuilder()
        result = builder.build(pdb, cif_dict, device)
        # result includes 'periods' tensor
    """

    def build(
        self,
        pdb: pd.DataFrame,
        cif_dict: Dict,
        device: torch.device,
        sort_indices: bool = True,
        residue_keys: Optional[Mapping[Tuple[str, int], str]] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Build all torsion restraints."""
        pp_pdb = PreprocessedPDB(pdb)
        pp_cif = PreprocessedCIF(cif_dict)
        keys = pp_pdb.residue_keys(residue_keys)

        max_per_residue = 50
        work_idx1 = np.zeros(max_per_residue, dtype=np.int64)
        work_idx2 = np.zeros(max_per_residue, dtype=np.int64)
        work_idx3 = np.zeros(max_per_residue, dtype=np.int64)
        work_idx4 = np.zeros(max_per_residue, dtype=np.int64)
        work_refs = np.zeros(max_per_residue, dtype=np.float64)
        work_sigmas = np.zeros(max_per_residue, dtype=np.float64)
        work_periods = np.zeros(max_per_residue, dtype=np.int64)

        all_indices = []
        all_refs = []
        all_sigmas = []
        all_periods = []

        for res_idx in range(pp_pdb.n_residues):
            key = keys[res_idx]

            if key not in pp_cif.torsions:
                continue

            cif_torsions = pp_cif.torsions[key]
            n_cif = len(cif_torsions["atom1"])

            if n_cif > max_per_residue:
                max_per_residue = n_cif * 2
                work_idx1 = np.zeros(max_per_residue, dtype=np.int64)
                work_idx2 = np.zeros(max_per_residue, dtype=np.int64)
                work_idx3 = np.zeros(max_per_residue, dtype=np.int64)
                work_idx4 = np.zeros(max_per_residue, dtype=np.int64)
                work_refs = np.zeros(max_per_residue, dtype=np.float64)
                work_sigmas = np.zeros(max_per_residue, dtype=np.float64)
                work_periods = np.zeros(max_per_residue, dtype=np.int64)

            # Iterate over altloc conformations (yields once if no altlocs)
            for atom_names, atom_indices, _ in pp_pdb.get_altloc_conformations(res_idx):
                count = match_torsions_numba(
                    atom_names,
                    atom_indices,
                    cif_torsions["atom1"],
                    cif_torsions["atom2"],
                    cif_torsions["atom3"],
                    cif_torsions["atom4"],
                    cif_torsions["value"],
                    cif_torsions["sigma"],
                    cif_torsions["period"],
                    work_idx1,
                    work_idx2,
                    work_idx3,
                    work_idx4,
                    work_refs,
                    work_sigmas,
                    work_periods,
                )

                if count > 0:
                    all_indices.append(
                        np.column_stack(
                            [
                                work_idx1[:count].copy(),
                                work_idx2[:count].copy(),
                                work_idx3[:count].copy(),
                                work_idx4[:count].copy(),
                            ]
                        )
                    )
                    all_refs.append(work_refs[:count].copy())
                    all_sigmas.append(work_sigmas[:count].copy())
                    all_periods.append(work_periods[:count].copy())

        if not all_indices:
            return None

        indices = np.concatenate(all_indices, axis=0)
        references = np.concatenate(all_refs)
        sigmas = np.concatenate(all_sigmas)
        periods = np.concatenate(all_periods)

        if sort_indices and len(indices) > 0:
            order = np.argsort(indices[:, 0])
            indices = indices[order]
            references = references[order]
            sigmas = sigmas[order]
            periods = periods[order]

        sigmas = np.where(sigmas == 0, 1e-4, sigmas)

        return {
            "indices": torch.tensor(indices, dtype=torch.long, device=device),
            "references": torch.tensor(references, dtype=get_float_dtype(), device=device),
            "sigmas": torch.tensor(sigmas, dtype=get_float_dtype(), device=device),
            "periods": torch.tensor(periods, dtype=torch.long, device=device),
        }


# =============================================================================
# Plane Builder
# =============================================================================


class PlaneRestraintBuilder(RestraintBuilder):
    """
    Fast plane restraint builder.

    Returns planes grouped by atom count (e.g., '4_atoms', '5_atoms').

    Usage:
        builder = PlaneRestraintBuilder()
        result = builder.build(pdb, cif_dict, device)
        # result = {'4_atoms': {'indices': ..., 'sigmas': ...}, '5_atoms': {...}}
    """

    def build(
        self,
        pdb: pd.DataFrame,
        cif_dict: Dict,
        device: torch.device,
        sort_indices: bool = True,
        residue_keys: Optional[Mapping[Tuple[str, int], str]] = None,
    ) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        """Build all plane restraints, grouped by atom count."""
        pp_pdb = PreprocessedPDB(pdb)
        pp_cif = PreprocessedCIF(cif_dict)
        keys = pp_pdb.residue_keys(residue_keys)

        # Group planes by size: {n_atoms: [(indices_array, sigmas_array), ...]}
        planes_by_size: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}

        for res_idx in range(pp_pdb.n_residues):
            key = keys[res_idx]

            if key not in pp_cif.planes:
                continue

            # Iterate over altloc conformations (yields once if no altlocs)
            for atom_names, atom_indices, _ in pp_pdb.get_altloc_conformations(res_idx):
                # Build name to index map
                name_to_idx = {name: idx for name, idx in zip(atom_names, atom_indices)}

                for plane_data in pp_cif.planes[key]:
                    plane_atom_names = plane_data["atoms"]
                    plane_sigmas = plane_data["sigmas"]

                    # Find atoms in this plane
                    plane_indices = []
                    plane_sigma_values = []
                    for i, atom_name in enumerate(plane_atom_names):
                        if atom_name in name_to_idx:
                            plane_indices.append(name_to_idx[atom_name])
                            plane_sigma_values.append(plane_sigmas[i])

                    # Need at least 3 atoms for a plane
                    if len(plane_indices) >= 3:
                        n_atoms = len(plane_indices)
                        indices_array = np.array(plane_indices, dtype=np.int64)
                        sigmas_array = np.array(plane_sigma_values, dtype=np.float64)

                        if n_atoms not in planes_by_size:
                            planes_by_size[n_atoms] = []
                        planes_by_size[n_atoms].append((indices_array, sigmas_array))

        if not planes_by_size:
            return None

        # Finalize each size group
        result = {}
        for n_atoms, planes_list in planes_by_size.items():
            indices = np.stack([p[0] for p in planes_list], axis=0)
            sigmas = np.stack([p[1] for p in planes_list], axis=0)

            if sort_indices and len(indices) > 0:
                order = np.argsort(indices[:, 0])
                indices = indices[order]
                sigmas = sigmas[order]

            sigmas = np.where(sigmas == 0, 1e-4, sigmas)

            key = f"{n_atoms}_atoms"
            result[key] = {
                "indices": torch.tensor(indices, dtype=torch.long, device=device),
                "sigmas": torch.tensor(sigmas, dtype=get_float_dtype(), device=device),
            }

        return result


# =============================================================================
# Chiral Builder
# =============================================================================


class ChiralRestraintBuilder(RestraintBuilder):
    """
    Fast chiral restraint builder.

    Usage:
        builder = ChiralRestraintBuilder()
        result = builder.build(pdb, cif_dict, device)
        # result includes 'ideal_volumes' tensor
    """

    def build(
        self,
        pdb: pd.DataFrame,
        cif_dict: Dict,
        device: torch.device,
        sort_indices: bool = True,
        residue_keys: Optional[Mapping[Tuple[str, int], str]] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Build all chiral restraints."""
        pp_pdb = PreprocessedPDB(pdb)
        pp_cif = PreprocessedCIF(cif_dict)
        keys = pp_pdb.residue_keys(residue_keys)

        max_per_residue = 20
        work_center = np.zeros(max_per_residue, dtype=np.int64)
        work_idx1 = np.zeros(max_per_residue, dtype=np.int64)
        work_idx2 = np.zeros(max_per_residue, dtype=np.int64)
        work_idx3 = np.zeros(max_per_residue, dtype=np.int64)
        work_signs = np.zeros(max_per_residue, dtype=np.float64)
        work_sigmas = np.zeros(max_per_residue, dtype=np.float64)

        all_indices = []
        all_ideal_volumes = []
        all_sigmas = []

        for res_idx in range(pp_pdb.n_residues):
            key = keys[res_idx]

            if key not in pp_cif.chirals:
                continue

            cif_chirals = pp_cif.chirals[key]
            n_cif = len(cif_chirals["center"])

            if n_cif > max_per_residue:
                max_per_residue = n_cif * 2
                work_center = np.zeros(max_per_residue, dtype=np.int64)
                work_idx1 = np.zeros(max_per_residue, dtype=np.int64)
                work_idx2 = np.zeros(max_per_residue, dtype=np.int64)
                work_idx3 = np.zeros(max_per_residue, dtype=np.int64)
                work_signs = np.zeros(max_per_residue, dtype=np.float64)
                work_sigmas = np.zeros(max_per_residue, dtype=np.float64)

            # Iterate over altloc conformations (yields once if no altlocs)
            for atom_names, atom_indices, _ in pp_pdb.get_altloc_conformations(res_idx):
                count = match_chirals_numba(
                    atom_names,
                    atom_indices,
                    cif_chirals["center"],
                    cif_chirals["atom1"],
                    cif_chirals["atom2"],
                    cif_chirals["atom3"],
                    cif_chirals["volume_sign"],
                    cif_chirals["sigma"],
                    work_center,
                    work_idx1,
                    work_idx2,
                    work_idx3,
                    work_signs,
                    work_sigmas,
                )

                if count > 0:
                    all_indices.append(
                        np.column_stack(
                            [
                                work_center[:count].copy(),
                                work_idx1[:count].copy(),
                                work_idx2[:count].copy(),
                                work_idx3[:count].copy(),
                            ]
                        )
                    )
                    # Ideal volume = sign * 2.5 (typical tetrahedral volume).
                    # For volume_sign 'both'/'either' the sign is 0.0, so this
                    # stores exactly 0.0. That 0.0 is a sentinel: the chiral
                    # target treats ideal_volume == 0 as an achiral centre and
                    # restrains |volume| toward 2.5 (not toward a target of 0).
                    # Note: chirals with an unknown sign were mapped to NaN by
                    # PreprocessedCIF._preprocess_chirals and dropped upstream
                    # by match_chirals_numba, so they never reach here.
                    all_ideal_volumes.append(work_signs[:count].copy() * 2.5)
                    all_sigmas.append(work_sigmas[:count].copy())

        if not all_indices:
            return None

        indices = np.concatenate(all_indices, axis=0)
        ideal_volumes = np.concatenate(all_ideal_volumes)
        sigmas = np.concatenate(all_sigmas)

        if sort_indices and len(indices) > 0:
            order = np.argsort(indices[:, 0])
            indices = indices[order]
            ideal_volumes = ideal_volumes[order]
            sigmas = sigmas[order]

        sigmas = np.where(sigmas == 0, 1e-4, sigmas)

        return {
            "indices": torch.tensor(indices, dtype=torch.long, device=device),
            "ideal_volumes": torch.tensor(
                ideal_volumes, dtype=get_float_dtype(), device=device
            ),
            "sigmas": torch.tensor(sigmas, dtype=get_float_dtype(), device=device),
        }


# =============================================================================
# Convenience function to build all restraints at once
# =============================================================================


def build_all_restraints(
    pdb: pd.DataFrame, cif_dict: Dict, device: torch.device, verbose: int = 0
) -> Dict[str, Any]:
    """
    Build every intra-residue restraint type at once, on ``device``.

    Returns
    -------
    dict
        Present keys only, so a type with no matches is absent rather than empty:
        ``bond``/``angle`` as ``{indices, references, sigmas}``, ``torsion`` also
        with ``periods``, ``chiral`` also with ``ideal_volumes``, and ``plane``
        nested by atom count (``{'4_atoms': {...}, '5_atoms': {...}}``).
    """
    result = {}

    bond_result = BondRestraintBuilder(verbose).build(pdb, cif_dict, device)
    if bond_result:
        result["bond"] = bond_result
        if verbose > 0:
            print(f"Built {bond_result['indices'].shape[0]} bond restraints")

    angle_result = AngleRestraintBuilder(verbose).build(pdb, cif_dict, device)
    if angle_result:
        result["angle"] = angle_result
        if verbose > 0:
            print(f"Built {angle_result['indices'].shape[0]} angle restraints")

    torsion_result = TorsionRestraintBuilder(verbose).build(pdb, cif_dict, device)
    if torsion_result:
        result["torsion"] = torsion_result
        if verbose > 0:
            print(f"Built {torsion_result['indices'].shape[0]} torsion restraints")

    plane_result = PlaneRestraintBuilder(verbose).build(pdb, cif_dict, device)
    if plane_result:
        result["plane"] = plane_result
        if verbose > 0:
            n_planes = sum(v["indices"].shape[0] for v in plane_result.values())
            print(f"Built {n_planes} plane restraints")

    chiral_result = ChiralRestraintBuilder(verbose).build(pdb, cif_dict, device)
    if chiral_result:
        result["chiral"] = chiral_result
        if verbose > 0:
            print(f"Built {chiral_result['indices'].shape[0]} chiral restraints")

    return result


# =============================================================================
# Fast Inter-Residue Builders
# =============================================================================


class PreprocessedLinkData:
    """
    Pre-processed link restraint data for fast inter-residue matching.

    Converts link DataFrames to NumPy arrays for efficient access.
    """

    def __init__(self, link_dict: Dict):
        """
        Initialize from link dictionary.

        Parameters
        ----------
        link_dict : dict
            Dictionary with link restraints (e.g., TRANS, disulf).
        """
        self.bonds = None
        self.angles = None
        self.torsions = None
        self.planes = None

        if "bonds" in link_dict and link_dict["bonds"] is not None:
            bonds_df = link_dict["bonds"]
            self.bonds = {
                "comp1": bonds_df["atom_1_comp_id"].values.astype(str),
                "comp2": bonds_df["atom_2_comp_id"].values.astype(str),
                "atom1": bonds_df["atom1"].values.astype(str),
                "atom2": bonds_df["atom2"].values.astype(str),
                "value": bonds_df["value"].values.astype(np.float64),
                "sigma": bonds_df["sigma"].values.astype(np.float64),
            }

        if "angles" in link_dict and link_dict["angles"] is not None:
            angles_df = link_dict["angles"]
            self.angles = {
                "comp1": angles_df["atom_1_comp_id"].values.astype(str),
                "comp2": angles_df["atom_2_comp_id"].values.astype(str),
                "comp3": angles_df["atom_3_comp_id"].values.astype(str),
                "atom1": angles_df["atom1"].values.astype(str),
                "atom2": angles_df["atom2"].values.astype(str),
                "atom3": angles_df["atom3"].values.astype(str),
                "value": angles_df["value"].values.astype(np.float64),
                "sigma": angles_df["sigma"].values.astype(np.float64),
            }

        if "torsions" in link_dict and link_dict["torsions"] is not None:
            torsions_df = link_dict["torsions"]
            self.torsions = {
                "comp1": torsions_df["atom_1_comp_id"].values.astype(str),
                "comp2": torsions_df["atom_2_comp_id"].values.astype(str),
                "comp3": torsions_df["atom_3_comp_id"].values.astype(str),
                "comp4": torsions_df["atom_4_comp_id"].values.astype(str),
                "atom1": torsions_df["atom1"].values.astype(str),
                "atom2": torsions_df["atom2"].values.astype(str),
                "atom3": torsions_df["atom3"].values.astype(str),
                "atom4": torsions_df["atom4"].values.astype(str),
                "id": (
                    torsions_df["id"].values.astype(str)
                    if "id" in torsions_df.columns
                    else np.full(len(torsions_df), "", dtype="<U10")
                ),
                "value": torsions_df["value"].values.astype(np.float64),
                "sigma": torsions_df["sigma"].values.astype(np.float64),
                "period": (
                    torsions_df["period"].values.astype(np.int64)
                    if "period" in torsions_df.columns
                    else np.ones(len(torsions_df), dtype=np.int64)
                ),
            }

        if "planes" in link_dict and link_dict["planes"] is not None:
            planes_df = link_dict["planes"]
            self.planes = self._preprocess_planes(planes_df)

    def _preprocess_planes(self, planes_df: pd.DataFrame) -> List[Dict]:
        """Convert planes DataFrame to list of plane data."""
        plane_ids = planes_df["plane_id"].unique()
        planes_data = []

        for plane_id in plane_ids:
            plane_atoms = planes_df[planes_df["plane_id"] == plane_id]
            planes_data.append(
                {
                    "comp_ids": plane_atoms["atom_comp_id"].values.astype(str),
                    "atoms": plane_atoms["atom"].values.astype(str),
                    "sigmas": plane_atoms["sigma"].values.astype(np.float64),
                }
            )

        return planes_data


class InterResidueBondBuilder:
    """
    Fast builder for inter-residue bond restraints.

    Usage:
        builder = InterResidueBondBuilder()
        result = builder.build(pdb, link_dict, device)

        # Or for disulfides (incremental):
        builder = InterResidueBondBuilder()
        for sg1_idx, sg2_idx, length, sigma in disulfide_pairs:
            builder.process_disulfide_bond(sg1_idx, sg2_idx, length, sigma)
        result = builder.finalize(device)
    """

    def __init__(self, verbose: int = 0):
        """Initialize builder with accumulators for disulfide bonds."""
        self.verbose = verbose
        # Accumulators for incremental disulfide building
        self._indices: List[np.ndarray] = []
        self._references: List[np.ndarray] = []
        self._sigmas: List[np.ndarray] = []
        self._count: int = 0

    def reset(self):
        """Clear all accumulated data."""
        self._indices.clear()
        self._references.clear()
        self._sigmas.clear()
        self._count = 0

    def process_disulfide_bond(
        self, sg1_idx: int, sg2_idx: int, bond_length: float, bond_sigma: float
    ) -> int:
        """
        Process a single disulfide bond restraint.

        Parameters
        ----------
        sg1_idx : int
            Index of first SG atom.
        sg2_idx : int
            Index of second SG atom.
        bond_length : float
            Target bond length in Å.
        bond_sigma : float
            Sigma for restraint in Å.

        Returns
        -------
        int
            Always returns 1.
        """
        self._indices.append(np.array([[sg1_idx, sg2_idx]], dtype=np.int64))
        self._references.append(np.array([bond_length], dtype=np.float32))
        self._sigmas.append(np.array([bond_sigma], dtype=np.float32))
        self._count += 1
        return 1

    def finalize(
        self, device: torch.device, sort_indices: bool = True, min_sigma: float = 1e-4
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Convert accumulated disulfide bond data to tensors.

        Parameters
        ----------
        device : torch.device
            Target device for tensors.
        sort_indices : bool, default True
            Whether to sort by first atom index.
        min_sigma : float, default 1e-4
            Minimum sigma value.

        Returns
        -------
        dict or None
            Dictionary with 'indices', 'references', 'sigmas' tensors.
        """
        if not self._indices:
            return None

        indices = np.concatenate(self._indices, axis=0)
        references = np.concatenate(self._references)
        sigmas = np.concatenate(self._sigmas)

        if sort_indices and len(indices) > 0:
            sort_order = np.argsort(indices[:, 0])
            indices = indices[sort_order]
            references = references[sort_order]
            sigmas = sigmas[sort_order]

        sigmas = np.where(sigmas == 0, min_sigma, sigmas)

        return {
            "indices": torch.tensor(indices, dtype=torch.long, device=device),
            "references": torch.tensor(references, dtype=get_float_dtype(), device=device),
            "sigmas": torch.tensor(sigmas, dtype=get_float_dtype(), device=device),
        }

    @property
    def count(self) -> int:
        """Return total number of restraints accumulated."""
        return self._count

    def build(
        self,
        pdb: pd.DataFrame,
        link_dict: Dict,
        device: torch.device,
        filter_atom_type: str = "ATOM",
        sort_indices: bool = True,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Build all inter-residue bond restraints.

        Parameters
        ----------
        pdb : pd.DataFrame
            PDB DataFrame.
        link_dict : dict
            Link dictionary with 'bonds' DataFrame.
        device : torch.device
            Target device.
        filter_atom_type : str, optional
            Filter to only this atom type (e.g., 'ATOM' for protein).
        sort_indices : bool
            Whether to sort output by first atom index.

        Returns
        -------
        dict or None
            Dictionary with restraint tensors.
        """
        if "bonds" not in link_dict or link_dict["bonds"] is None:
            return None

        # Pre-process link data
        link_data = PreprocessedLinkData(link_dict)
        if link_data.bonds is None:
            return None

        # Pre-process PDB
        if filter_atom_type:
            pdb = pdb[pdb["ATOM"] == filter_atom_type]
        pp_pdb = PreprocessedPDB(pdb)

        # Build per-conformation maps for each residue (altloc-aware)
        conf_maps = build_residue_conformation_maps(pp_pdb)

        # Find consecutive residue pairs
        pairs = find_consecutive_residue_pairs(pp_pdb)

        # Accumulate restraints
        all_indices = []
        all_refs = []
        all_sigmas = []

        bonds = link_data.bonds
        n_bonds = len(bonds["atom1"])

        for res_i_idx, res_next_idx in pairs:
            # Iterate over all conformer pairs (Cartesian product)
            for map_i in conf_maps[res_i_idx]:
                for map_next in conf_maps[res_next_idx]:

                    for b in range(n_bonds):
                        comp1, comp2 = bonds["comp1"][b], bonds["comp2"][b]
                        atom1_name, atom2_name = bonds["atom1"][b], bonds["atom2"][b]

                        map1 = map_i if comp1 == "1" else map_next
                        map2 = map_i if comp2 == "1" else map_next

                        if atom1_name in map1 and atom2_name in map2:
                            idx1 = map1[atom1_name]
                            idx2 = map2[atom2_name]
                            all_indices.append([idx1, idx2])
                            all_refs.append(bonds["value"][b])
                            all_sigmas.append(bonds["sigma"][b])

        if not all_indices:
            return None

        indices = np.array(all_indices, dtype=np.int64)
        references = np.array(all_refs, dtype=np.float64)
        sigmas = np.array(all_sigmas, dtype=np.float64)

        if sort_indices and len(indices) > 0:
            order = np.argsort(indices[:, 0])
            indices = indices[order]
            references = references[order]
            sigmas = sigmas[order]

        sigmas = np.where(sigmas == 0, 1e-4, sigmas)

        return {
            "indices": torch.tensor(indices, dtype=torch.long, device=device),
            "references": torch.tensor(references, dtype=get_float_dtype(), device=device),
            "sigmas": torch.tensor(sigmas, dtype=get_float_dtype(), device=device),
        }


class InterResidueAngleBuilder:
    """
    Fast builder for inter-residue angle restraints.

    Usage:
        builder = InterResidueAngleBuilder()
        result = builder.build(pdb, link_dict, device)

        # Or for disulfides (incremental):
        builder = InterResidueAngleBuilder()
        builder.process_disulfide_angles(res1_atoms, res2_atoms, link_angles)
        result = builder.finalize(device)
    """

    def __init__(self, verbose: int = 0):
        """Initialize builder with accumulators for disulfide angles."""
        self.verbose = verbose
        # Accumulators for incremental disulfide building
        self._indices: List[np.ndarray] = []
        self._references: List[np.ndarray] = []
        self._sigmas: List[np.ndarray] = []
        self._count: int = 0

    def reset(self):
        """Clear all accumulated data."""
        self._indices.clear()
        self._references.clear()
        self._sigmas.clear()
        self._count = 0

    @staticmethod
    def _get_atom_index(residue: pd.DataFrame, atom_name: str) -> Optional[int]:
        """Get atom index from residue, handling alternate conformations."""
        atoms = residue[residue["name"] == atom_name]
        if len(atoms) == 0:
            return None
        if " " in atoms["altloc"].values:
            return int(atoms[atoms["altloc"] == " "].iloc[0]["index"])
        elif "A" in atoms["altloc"].values:
            return int(atoms[atoms["altloc"] == "A"].iloc[0]["index"])
        else:
            return int(atoms.iloc[0]["index"])

    def process_disulfide_angles(
        self,
        res1_atoms: pd.DataFrame,
        res2_atoms: pd.DataFrame,
        link_angles: pd.DataFrame,
    ) -> int:
        """
        Process disulfide angle restraints.

        Parameters
        ----------
        res1_atoms : pd.DataFrame
            First cysteine residue atoms.
        res2_atoms : pd.DataFrame
            Second cysteine residue atoms.
        link_angles : pd.DataFrame
            Angle definitions from disulfide link.

        Returns
        -------
        int
            Number of angle restraints added.
        """
        count = 0
        for _, angle_row in link_angles.iterrows():
            comp1 = angle_row["atom_1_comp_id"]
            comp2 = angle_row["atom_2_comp_id"]
            comp3 = angle_row["atom_3_comp_id"]
            atom1_name = angle_row["atom1"]
            atom2_name = angle_row["atom2"]
            atom3_name = angle_row["atom3"]

            res1 = res1_atoms if comp1 == "1" else res2_atoms
            res2 = res1_atoms if comp2 == "1" else res2_atoms
            res3 = res1_atoms if comp3 == "1" else res2_atoms

            idx1 = self._get_atom_index(res1, atom1_name)
            idx2 = self._get_atom_index(res2, atom2_name)
            idx3 = self._get_atom_index(res3, atom3_name)

            if idx1 is not None and idx2 is not None and idx3 is not None:
                self._indices.append(np.array([[idx1, idx2, idx3]], dtype=np.int64))
                self._references.append(
                    np.array([float(angle_row["value"])], dtype=np.float32)
                )
                self._sigmas.append(
                    np.array([float(angle_row["sigma"])], dtype=np.float32)
                )
                count += 1

        self._count += count
        return count

    def finalize(
        self, device: torch.device, sort_indices: bool = True, min_sigma: float = 1e-4
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Convert accumulated data to sorted tensors."""
        if not self._indices:
            return None

        indices = np.concatenate(self._indices, axis=0)
        references = np.concatenate(self._references)
        sigmas = np.concatenate(self._sigmas)

        if sort_indices and len(indices) > 0:
            sort_order = np.argsort(indices[:, 0])
            indices = indices[sort_order]
            references = references[sort_order]
            sigmas = sigmas[sort_order]

        sigmas = np.where(sigmas == 0, min_sigma, sigmas)

        return {
            "indices": torch.tensor(indices, dtype=torch.long, device=device),
            "references": torch.tensor(references, dtype=get_float_dtype(), device=device),
            "sigmas": torch.tensor(sigmas, dtype=get_float_dtype(), device=device),
        }

    @property
    def count(self) -> int:
        """Return total number of restraints accumulated."""
        return self._count

    def build(
        self,
        pdb: pd.DataFrame,
        link_dict: Dict,
        device: torch.device,
        filter_atom_type: str = "ATOM",
        sort_indices: bool = True,
        next_resname_filter: Optional[str] = None,
        exclude_next_resname: Optional[str] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Build all inter-residue angle restraints.

        Parameters
        ----------
        pdb : pd.DataFrame
            Atom DataFrame.
        link_dict : Dict
            Link definition dictionary containing angle parameters.
        device : torch.device
            Target device for tensors.
        filter_atom_type : str, optional
            Filter to this ATOM type (default "ATOM").
        sort_indices : bool, optional
            Sort output by first atom index (default True).
        next_resname_filter : str, optional
            If set, only build angles for pairs where the second (next)
            residue has this residue name (e.g. "PRO" for proline links).
        exclude_next_resname : str, optional
            If set, skip pairs where the second (next) residue has this
            residue name.  Useful for excluding PRO from TRANS angles
            when PTRANS is handled separately.
        """
        if "angles" not in link_dict or link_dict["angles"] is None:
            return None

        link_data = PreprocessedLinkData(link_dict)
        if link_data.angles is None:
            return None

        if filter_atom_type:
            pdb = pdb[pdb["ATOM"] == filter_atom_type]
        pp_pdb = PreprocessedPDB(pdb)
        conf_maps = build_residue_conformation_maps(pp_pdb)
        pairs = find_consecutive_residue_pairs(pp_pdb)

        all_indices = []
        all_refs = []
        all_sigmas = []

        angles = link_data.angles
        n_angles = len(angles["atom1"])

        for res_i_idx, res_next_idx in pairs:
            # Filter by next residue name if requested
            if next_resname_filter is not None:
                if pp_pdb.residue_resnames[res_next_idx] != next_resname_filter:
                    continue
            if exclude_next_resname is not None:
                if pp_pdb.residue_resnames[res_next_idx] == exclude_next_resname:
                    continue

            for map_i in conf_maps[res_i_idx]:
                for map_next in conf_maps[res_next_idx]:

                    for a in range(n_angles):
                        comp1, comp2, comp3 = (
                            angles["comp1"][a],
                            angles["comp2"][a],
                            angles["comp3"][a],
                        )
                        atom1, atom2, atom3 = (
                            angles["atom1"][a],
                            angles["atom2"][a],
                            angles["atom3"][a],
                        )

                        map1 = map_i if comp1 == "1" else map_next
                        map2 = map_i if comp2 == "1" else map_next
                        map3 = map_i if comp3 == "1" else map_next

                        if atom1 in map1 and atom2 in map2 and atom3 in map3:
                            idx1, idx2, idx3 = map1[atom1], map2[atom2], map3[atom3]
                            all_indices.append([idx1, idx2, idx3])
                            all_refs.append(angles["value"][a])
                            all_sigmas.append(angles["sigma"][a])

        if not all_indices:
            return None

        indices = np.array(all_indices, dtype=np.int64)
        references = np.array(all_refs, dtype=np.float64)
        sigmas = np.array(all_sigmas, dtype=np.float64)

        if sort_indices and len(indices) > 0:
            order = np.argsort(indices[:, 0])
            indices = indices[order]
            references = references[order]
            sigmas = sigmas[order]

        sigmas = np.where(sigmas == 0, 1e-4, sigmas)

        return {
            "indices": torch.tensor(indices, dtype=torch.long, device=device),
            "references": torch.tensor(references, dtype=get_float_dtype(), device=device),
            "sigmas": torch.tensor(sigmas, dtype=get_float_dtype(), device=device),
        }


class InterResidueTorsionBuilder:
    """
    Fast builder for inter-residue torsion restraints (phi, psi, omega).

    Usage:
        builder = InterResidueTorsionBuilder()
        result = builder.build(pdb, link_dict, device)
        # result = {'phi': {...}, 'psi': {...}, 'omega': {...},
        #           'ramachandran': {...}}

        # Or for disulfides (incremental):
        builder = InterResidueTorsionBuilder()
        builder.process_disulfide_torsions(res1_atoms, res2_atoms, link_torsions)
        result = builder.finalize_disulfide(device)
    """

    def __init__(self, verbose: int = 0):
        """Initialize builder with accumulators for disulfide torsions."""
        self.verbose = verbose
        # Accumulators for disulfide torsions
        self._disulfide_indices: List[np.ndarray] = []
        self._disulfide_references: List[np.ndarray] = []
        self._disulfide_sigmas: List[np.ndarray] = []
        self._disulfide_periods: List[np.ndarray] = []
        self._disulfide_count: int = 0

    def reset(self):
        """Clear all accumulated disulfide data."""
        self._disulfide_indices.clear()
        self._disulfide_references.clear()
        self._disulfide_sigmas.clear()
        self._disulfide_periods.clear()
        self._disulfide_count = 0

    @staticmethod
    def _get_atom_index(residue: pd.DataFrame, atom_name: str) -> Optional[int]:
        """Get atom index from residue, handling alternate conformations."""
        atoms = residue[residue["name"] == atom_name]
        if len(atoms) == 0:
            return None
        if " " in atoms["altloc"].values:
            return int(atoms[atoms["altloc"] == " "].iloc[0]["index"])
        elif "A" in atoms["altloc"].values:
            return int(atoms[atoms["altloc"] == "A"].iloc[0]["index"])
        else:
            return int(atoms.iloc[0]["index"])

    def process_disulfide_torsions(
        self,
        res1_atoms: pd.DataFrame,
        res2_atoms: pd.DataFrame,
        link_torsions: pd.DataFrame,
    ) -> int:
        """
        Process disulfide torsion restraints.

        Parameters
        ----------
        res1_atoms : pd.DataFrame
            First cysteine residue atoms.
        res2_atoms : pd.DataFrame
            Second cysteine residue atoms.
        link_torsions : pd.DataFrame
            Torsion definitions from disulfide link.

        Returns
        -------
        int
            Number of torsion restraints added.
        """
        count = 0
        for _, torsion_row in link_torsions.iterrows():
            comp1 = torsion_row["atom_1_comp_id"]
            comp2 = torsion_row["atom_2_comp_id"]
            comp3 = torsion_row["atom_3_comp_id"]
            comp4 = torsion_row["atom_4_comp_id"]
            atom1_name = torsion_row["atom1"]
            atom2_name = torsion_row["atom2"]
            atom3_name = torsion_row["atom3"]
            atom4_name = torsion_row["atom4"]

            res1 = res1_atoms if comp1 == "1" else res2_atoms
            res2 = res1_atoms if comp2 == "1" else res2_atoms
            res3 = res1_atoms if comp3 == "1" else res2_atoms
            res4 = res1_atoms if comp4 == "1" else res2_atoms

            idx1 = self._get_atom_index(res1, atom1_name)
            idx2 = self._get_atom_index(res2, atom2_name)
            idx3 = self._get_atom_index(res3, atom3_name)
            idx4 = self._get_atom_index(res4, atom4_name)

            if idx1 is None or idx2 is None or idx3 is None or idx4 is None:
                continue

            self._disulfide_indices.append(
                np.array([[idx1, idx2, idx3, idx4]], dtype=np.int64)
            )
            self._disulfide_references.append(
                np.array([float(torsion_row["value"])], dtype=np.float32)
            )
            self._disulfide_sigmas.append(
                np.array([float(torsion_row["sigma"])], dtype=np.float32)
            )
            self._disulfide_periods.append(
                np.array([2], dtype=np.int64)
            )  # Period 2 for disulfide
            count += 1

        self._disulfide_count += count
        return count

    def finalize_disulfide(
        self, device: torch.device, sort_indices: bool = True
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Finalize disulfide torsion restraints."""
        if not self._disulfide_indices:
            return None

        indices = np.concatenate(self._disulfide_indices, axis=0)
        references = np.concatenate(self._disulfide_references)
        sigmas = np.concatenate(self._disulfide_sigmas)
        periods = np.concatenate(self._disulfide_periods)

        if sort_indices and len(indices) > 0:
            sort_order = np.argsort(indices[:, 0])
            indices = indices[sort_order]
            references = references[sort_order]
            sigmas = sigmas[sort_order]
            periods = periods[sort_order]

        return {
            "indices": torch.tensor(indices, dtype=torch.long, device=device),
            "references": torch.tensor(references, dtype=get_float_dtype(), device=device),
            "sigmas": torch.tensor(sigmas, dtype=get_float_dtype(), device=device),
            "periods": torch.tensor(periods, dtype=torch.long, device=device),
        }

    @property
    def disulfide_count(self) -> int:
        """Return total number of disulfide torsion restraints accumulated."""
        return self._disulfide_count

    @staticmethod
    def _torsion_angle_np(coords: np.ndarray, i1, i2, i3, i4) -> float:
        """Compute torsion angle (degrees) from coordinates for 4 atom indices."""
        p = coords[[i1, i2, i3, i4]]
        b1, b2, b3 = p[1] - p[0], p[2] - p[1], p[3] - p[2]
        n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
        n1_len, n2_len = np.linalg.norm(n1), np.linalg.norm(n2)
        if n1_len < 1e-10 or n2_len < 1e-10:
            return 180.0
        n1, n2 = n1 / n1_len, n2 / n2_len
        m1 = np.cross(n1, b2 / np.linalg.norm(b2))
        return float(np.degrees(np.arctan2(np.dot(m1, n2), np.dot(n1, n2))))

    def build(
        self,
        pdb: pd.DataFrame,
        link_dict: Dict,
        device: torch.device,
        filter_atom_type: str = "ATOM",
        sort_indices: bool = True,
    ) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        """
        Build all inter-residue torsion restraints.

        Returns separate phi, psi, omega, and ramachandran results.
        """
        if "torsions" not in link_dict or link_dict["torsions"] is None:
            return None

        link_data = PreprocessedLinkData(link_dict)
        if link_data.torsions is None:
            return None

        if filter_atom_type:
            pdb = pdb[pdb["ATOM"] == filter_atom_type]
        pp_pdb = PreprocessedPDB(pdb)
        conf_maps = build_residue_conformation_maps(pp_pdb)
        pairs = find_consecutive_residue_pairs(pp_pdb)

        # Build coordinate array for omega angle computation (cis/trans PRO)
        max_idx = int(pdb["index"].max()) + 1
        coords_np = np.zeros((max_idx, 3))
        coords_np[pdb["index"].values] = pdb[["x", "y", "z"]].values

        # Separate accumulators for phi, psi, omega
        phi_data = {"indices": [], "periods": []}
        psi_data = {"indices": [], "periods": []}
        omega_data = {
            "indices": [],
            "references": [],
            "sigmas": [],
            "periods": [],
            "is_proline": [],
        }
        # Ramachandran: collect phi/psi per residue, then match afterwards
        # phi from pair (i, j) belongs to residue j (second residue)
        # psi from pair (i, j) belongs to residue i (first residue)
        phi_by_residue = {}  # res_idx -> atom indices
        psi_by_residue = {}  # res_idx -> atom indices
        omega_by_residue = {}  # res_idx -> omega_deg (for cis/trans PRO detection)
        resname_by_residue = {}  # res_idx -> resname
        next_resname_by_residue = {}  # res_idx -> next resname (for pre-PRO)

        torsions = link_data.torsions
        n_torsions = len(torsions["atom1"])

        from torchref.restraints.ramachandran import classify_residue

        for res_i_idx, res_next_idx in pairs:
            resname_i = pp_pdb.residue_resnames[res_i_idx]
            resname_next = pp_pdb.residue_resnames[res_next_idx]
            is_proline = resname_next == "PRO"

            for map_i in conf_maps[res_i_idx]:
                for map_next in conf_maps[res_next_idx]:

                    # Track which residue each phi/psi belongs to
                    pair_phi = None   # phi from this pair belongs to res_next_idx
                    pair_psi = None   # psi from this pair belongs to res_i_idx

                    for t in range(n_torsions):
                        comp1 = torsions["comp1"][t]
                        comp2 = torsions["comp2"][t]
                        comp3 = torsions["comp3"][t]
                        comp4 = torsions["comp4"][t]
                        atom1 = torsions["atom1"][t]
                        atom2 = torsions["atom2"][t]
                        atom3 = torsions["atom3"][t]
                        atom4 = torsions["atom4"][t]
                        torsion_id = torsions["id"][t]

                        map1 = map_i if comp1 == "1" else map_next
                        map2 = map_i if comp2 == "1" else map_next
                        map3 = map_i if comp3 == "1" else map_next
                        map4 = map_i if comp4 == "1" else map_next

                        if not (
                            atom1 in map1 and atom2 in map2
                            and atom3 in map3 and atom4 in map4
                        ):
                            continue

                        idx1, idx2, idx3, idx4 = (
                            map1[atom1],
                            map2[atom2],
                            map3[atom3],
                            map4[atom4],
                        )
                        period = int(torsions["period"][t])

                        if torsion_id == "phi":
                            phi_data["indices"].append([idx1, idx2, idx3, idx4])
                            phi_data["periods"].append(period)
                            pair_phi = [idx1, idx2, idx3, idx4]
                        elif torsion_id == "psi":
                            psi_data["indices"].append([idx1, idx2, idx3, idx4])
                            psi_data["periods"].append(period)
                            pair_psi = [idx1, idx2, idx3, idx4]
                        elif torsion_id == "omega":
                            omega_data["indices"].append([idx1, idx2, idx3, idx4])
                            omega_data["references"].append(float(torsions["value"][t]))
                            omega_data["sigmas"].append(float(torsions["sigma"][t]))
                            omega_data["periods"].append(period)
                            omega_data["is_proline"].append(is_proline)

                    # Store phi/psi by the residue they actually belong to:
                    # phi: C(i) - N(j) - CA(j) - C(j)  → belongs to residue j
                    # psi: N(i) - CA(i) - C(i)  - N(j)  → belongs to residue i
                    if pair_phi is not None:
                        phi_by_residue[res_next_idx] = pair_phi
                    if pair_psi is not None:
                        psi_by_residue[res_i_idx] = pair_psi
                    # Track residue names and next-residue names for classification
                    resname_by_residue[res_i_idx] = resname_i
                    resname_by_residue[res_next_idx] = resname_next
                    next_resname_by_residue[res_i_idx] = resname_next
                    # Compute omega for PRO cis/trans detection
                    if omega_data["indices"]:
                        omega_deg = self._torsion_angle_np(
                            coords_np, *omega_data["indices"][-1]
                        )
                        omega_by_residue[res_next_idx] = omega_deg

        result = {}

        # Finalize phi
        if phi_data["indices"]:
            indices = np.array(phi_data["indices"], dtype=np.int64)
            periods = np.array(phi_data["periods"], dtype=np.int64)
            if sort_indices:
                order = np.argsort(indices[:, 0])
                indices = indices[order]
                periods = periods[order]
            result["phi"] = {
                "indices": torch.tensor(indices, dtype=torch.long, device=device),
                "periods": torch.tensor(periods, dtype=torch.long, device=device),
            }

        # Finalize psi
        if psi_data["indices"]:
            indices = np.array(psi_data["indices"], dtype=np.int64)
            periods = np.array(psi_data["periods"], dtype=np.int64)
            if sort_indices:
                order = np.argsort(indices[:, 0])
                indices = indices[order]
                periods = periods[order]
            result["psi"] = {
                "indices": torch.tensor(indices, dtype=torch.long, device=device),
                "periods": torch.tensor(periods, dtype=torch.long, device=device),
            }

        # Finalize omega
        if omega_data["indices"]:
            indices = np.array(omega_data["indices"], dtype=np.int64)
            references = np.array(omega_data["references"], dtype=np.float64)
            sigmas = np.array(omega_data["sigmas"], dtype=np.float64)
            periods = np.array(omega_data["periods"], dtype=np.int64)
            is_proline = np.array(omega_data["is_proline"], dtype=bool)
            if sort_indices:
                order = np.argsort(indices[:, 0])
                indices = indices[order]
                references = references[order]
                sigmas = sigmas[order]
                periods = periods[order]
                is_proline = is_proline[order]
            result["omega"] = {
                "indices": torch.tensor(indices, dtype=torch.long, device=device),
                "references": torch.tensor(
                    references, dtype=get_float_dtype(), device=device
                ),
                "sigmas": torch.tensor(sigmas, dtype=get_float_dtype(), device=device),
                "periods": torch.tensor(periods, dtype=torch.long, device=device),
                "is_proline": torch.tensor(is_proline, dtype=torch.bool, device=device),
            }

        # Finalize ramachandran — match phi and psi for the SAME residue
        # phi_by_residue[r] = phi atom indices for residue r
        # psi_by_residue[r] = psi atom indices for residue r
        # A residue needs both phi and psi for a Ramachandran restraint
        rama_residues = sorted(
            set(phi_by_residue.keys()) & set(psi_by_residue.keys())
        )
        if rama_residues:
            rama_phi = []
            rama_psi = []
            rama_types = []
            for res_idx in rama_residues:
                resname = resname_by_residue[res_idx]
                next_rn = next_resname_by_residue.get(res_idx, "")
                omega_deg = omega_by_residue.get(res_idx, 180.0)
                rama_type = classify_residue(resname, next_rn, omega_deg)
                rama_phi.append(phi_by_residue[res_idx])
                rama_psi.append(psi_by_residue[res_idx])
                rama_types.append(rama_type)

            phi_idx = np.array(rama_phi, dtype=np.int64)
            psi_idx = np.array(rama_psi, dtype=np.int64)
            stypes = np.array(rama_types, dtype=np.int64)
            if sort_indices:
                order = np.argsort(phi_idx[:, 0])
                phi_idx = phi_idx[order]
                psi_idx = psi_idx[order]
                stypes = stypes[order]
            result["ramachandran"] = {
                "phi_indices": torch.tensor(
                    phi_idx, dtype=torch.long, device=device
                ),
                "psi_indices": torch.tensor(
                    psi_idx, dtype=torch.long, device=device
                ),
                "surface_type": torch.tensor(
                    stypes, dtype=torch.long, device=device
                ),
            }

        return result if result else None


class InterResiduePlaneBuilder:
    """
    Fast builder for inter-residue plane restraints (peptide planes).

    Usage:
        builder = InterResiduePlaneBuilder()
        result = builder.build(pdb, link_dict, device)
    """

    def __init__(self, verbose: int = 0):
        """Initialize builder."""
        self.verbose = verbose

    def build(
        self,
        pdb: pd.DataFrame,
        link_dict: Dict,
        device: torch.device,
        filter_atom_type: str = "ATOM",
        sort_indices: bool = True,
    ) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        """Build all inter-residue plane restraints, grouped by atom count."""
        if "planes" not in link_dict or link_dict["planes"] is None:
            return None

        link_data = PreprocessedLinkData(link_dict)
        if link_data.planes is None:
            return None

        if filter_atom_type:
            pdb = pdb[pdb["ATOM"] == filter_atom_type]
        pp_pdb = PreprocessedPDB(pdb)
        conf_maps = build_residue_conformation_maps(pp_pdb)
        pairs = find_consecutive_residue_pairs(pp_pdb)

        # Group planes by atom count
        planes_by_size: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}

        for res_i_idx, res_next_idx in pairs:
          for map_i in conf_maps[res_i_idx]:
            for map_next in conf_maps[res_next_idx]:

              for plane_data in link_data.planes:
                comp_ids = plane_data["comp_ids"]
                atom_names = plane_data["atoms"]
                sigmas = plane_data["sigmas"]

                plane_indices = []
                plane_sigmas = []
                all_found = True

                for i, (comp_id, atom_name, sigma) in enumerate(
                    zip(comp_ids, atom_names, sigmas)
                ):
                    atom_map = map_i if comp_id == "1" else map_next
                    if atom_name in atom_map:
                        plane_indices.append(atom_map[atom_name])
                        plane_sigmas.append(sigma)
                    else:
                        all_found = False
                        break

                if all_found and len(plane_indices) >= 3:
                    n_atoms = len(plane_indices)
                    if n_atoms not in planes_by_size:
                        planes_by_size[n_atoms] = []
                    planes_by_size[n_atoms].append(
                        (
                            np.array(plane_indices, dtype=np.int64),
                            np.array(plane_sigmas, dtype=np.float64),
                        )
                    )

        if not planes_by_size:
            return None

        result = {}
        for n_atoms, planes_list in planes_by_size.items():
            indices = np.stack([p[0] for p in planes_list], axis=0)
            sigmas = np.stack([p[1] for p in planes_list], axis=0)

            if sort_indices and len(indices) > 0:
                order = np.argsort(indices[:, 0])
                indices = indices[order]
                sigmas = sigmas[order]

            sigmas = np.where(sigmas == 0, 1e-4, sigmas)

            key = f"{n_atoms}_atoms"
            result[key] = {
                "indices": torch.tensor(indices, dtype=torch.long, device=device),
                "sigmas": torch.tensor(sigmas, dtype=get_float_dtype(), device=device),
            }

        return result


# =============================================================================
# Legacy-compatible ResidueIterator (for code that still needs it)
# =============================================================================


class ResidueIterator:
    """
    Efficient iterator over residues.

    .. deprecated::
        Retained only for backward compatibility with code that still relies
        on residue-by-residue iteration. Prefer :func:`build_all_restraints`
        or the individual ``builder.build()`` methods instead.
    """

    def __init__(self, pdb: pd.DataFrame, filter_atom_type: Optional[str] = None):
        """Initialize with pre-grouping of residues."""
        if filter_atom_type is not None:
            pdb = pdb[pdb["ATOM"] == filter_atom_type]

        self.pdb = pdb
        self._grouped = pdb.groupby(["chainid", "resseq"], sort=False)
        self.groups = list(self._grouped.groups.keys())

    def __iter__(self) -> Iterator[Tuple[str, int, pd.DataFrame]]:
        """Iterate over residues."""
        for chain_id, resseq in self.groups:
            residue = self._grouped.get_group((chain_id, resseq))
            yield chain_id, resseq, residue

    def __len__(self) -> int:
        """Return number of residues."""
        return len(self.groups)

    def get_consecutive_pairs(self) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Iterate over consecutive residue pairs within each chain."""
        by_chain = {}
        for chain_id, resseq in self.groups:
            if chain_id not in by_chain:
                by_chain[chain_id] = []
            by_chain[chain_id].append(resseq)

        for chain_id, resseqs in by_chain.items():
            resseqs_sorted = sorted(resseqs)
            for i in range(len(resseqs_sorted) - 1):
                resseq_i = resseqs_sorted[i]
                resseq_next = resseqs_sorted[i + 1]

                if resseq_next != resseq_i + 1:
                    continue

                residue_i = self._grouped.get_group((chain_id, resseq_i))
                residue_next = self._grouped.get_group((chain_id, resseq_next))
                yield residue_i, residue_next
