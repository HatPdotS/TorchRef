"""

Fast Restraint Builder Classes using NumPy and Numba

This module provides optimized builder classes that avoid Pandas operations
in the hot loop.

"""

from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

try:
    import numba
    from numba import njit, prange

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    # Fallback decorator that does nothing
    def njit(*args, **kwargs):
        def decorator(func):
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

    prange = range


# =============================================================================
# Numba-accelerated helper functions
# =============================================================================


@njit(cache=True)
def find_atom_index(atom_names: np.ndarray, target: str) -> int:
    """
    Find index of target atom name in array.

    Returns -1 if not found.
    """
    for i in range(len(atom_names)):
        if atom_names[i] == target:
            return i
    return -1


@njit(cache=True)
def match_bonds_numba(
    residue_atom_names: np.ndarray,  # atom names for this residue
    residue_atom_indices: np.ndarray,  # global atom indices
    bond_atom1: np.ndarray,  # CIF bond atom1 names
    bond_atom2: np.ndarray,  # CIF bond atom2 names
    bond_values: np.ndarray,  # CIF bond reference values
    bond_sigmas: np.ndarray,  # CIF bond sigmas
    out_idx1: np.ndarray,  # output arrays (pre-allocated)
    out_idx2: np.ndarray,
    out_refs: np.ndarray,
    out_sigmas: np.ndarray,
) -> int:
    """
    Match bond restraints for a single residue.

    Returns number of matched bonds.
    """
    count = 0
    n_bonds = len(bond_atom1)
    n_atoms = len(residue_atom_names)

    for i in range(n_bonds):
        # Find atom1
        idx1 = -1
        for j in range(n_atoms):
            if residue_atom_names[j] == bond_atom1[i]:
                idx1 = j
                break
        if idx1 < 0:
            continue

        # Find atom2
        idx2 = -1
        for j in range(n_atoms):
            if residue_atom_names[j] == bond_atom2[i]:
                idx2 = j
                break
        if idx2 < 0:
            continue

        # Both atoms found - add restraint
        out_idx1[count] = residue_atom_indices[idx1]
        out_idx2[count] = residue_atom_indices[idx2]
        out_refs[count] = bond_values[i]
        out_sigmas[count] = bond_sigmas[i]
        count += 1

    return count


@njit(cache=True)
def match_angles_numba(
    residue_atom_names: np.ndarray,
    residue_atom_indices: np.ndarray,
    angle_atom1: np.ndarray,
    angle_atom2: np.ndarray,
    angle_atom3: np.ndarray,
    angle_values: np.ndarray,
    angle_sigmas: np.ndarray,
    out_idx1: np.ndarray,
    out_idx2: np.ndarray,
    out_idx3: np.ndarray,
    out_refs: np.ndarray,
    out_sigmas: np.ndarray,
) -> int:
    """Match angle restraints for a single residue."""
    count = 0
    n_angles = len(angle_atom1)
    n_atoms = len(residue_atom_names)

    for i in range(n_angles):
        # Find all three atoms
        idx1, idx2, idx3 = -1, -1, -1
        for j in range(n_atoms):
            name = residue_atom_names[j]
            if name == angle_atom1[i]:
                idx1 = j
            elif name == angle_atom2[i]:
                idx2 = j
            elif name == angle_atom3[i]:
                idx3 = j

        if idx1 < 0 or idx2 < 0 or idx3 < 0:
            continue

        out_idx1[count] = residue_atom_indices[idx1]
        out_idx2[count] = residue_atom_indices[idx2]
        out_idx3[count] = residue_atom_indices[idx3]
        out_refs[count] = angle_values[i]
        out_sigmas[count] = angle_sigmas[i]
        count += 1

    return count


@njit(cache=True)
def match_torsions_numba(
    residue_atom_names: np.ndarray,
    residue_atom_indices: np.ndarray,
    torsion_atom1: np.ndarray,
    torsion_atom2: np.ndarray,
    torsion_atom3: np.ndarray,
    torsion_atom4: np.ndarray,
    torsion_values: np.ndarray,
    torsion_sigmas: np.ndarray,
    torsion_periods: np.ndarray,
    out_idx1: np.ndarray,
    out_idx2: np.ndarray,
    out_idx3: np.ndarray,
    out_idx4: np.ndarray,
    out_refs: np.ndarray,
    out_sigmas: np.ndarray,
    out_periods: np.ndarray,
) -> int:
    """Match torsion restraints for a single residue."""
    count = 0
    n_torsions = len(torsion_atom1)
    n_atoms = len(residue_atom_names)

    for i in range(n_torsions):
        # Skip if sigma is zero
        if torsion_sigmas[i] == 0:
            continue

        # Find all four atoms
        idx1, idx2, idx3, idx4 = -1, -1, -1, -1
        for j in range(n_atoms):
            name = residue_atom_names[j]
            if name == torsion_atom1[i]:
                idx1 = j
            elif name == torsion_atom2[i]:
                idx2 = j
            elif name == torsion_atom3[i]:
                idx3 = j
            elif name == torsion_atom4[i]:
                idx4 = j

        if idx1 < 0 or idx2 < 0 or idx3 < 0 or idx4 < 0:
            continue

        out_idx1[count] = residue_atom_indices[idx1]
        out_idx2[count] = residue_atom_indices[idx2]
        out_idx3[count] = residue_atom_indices[idx3]
        out_idx4[count] = residue_atom_indices[idx4]
        out_refs[count] = torsion_values[i]
        out_sigmas[count] = torsion_sigmas[i]
        out_periods[count] = torsion_periods[i]
        count += 1

    return count


@njit(cache=True)
def match_chirals_numba(
    residue_atom_names: np.ndarray,
    residue_atom_indices: np.ndarray,
    chiral_center: np.ndarray,
    chiral_atom1: np.ndarray,
    chiral_atom2: np.ndarray,
    chiral_atom3: np.ndarray,
    chiral_volume_signs: np.ndarray,  # +1, -1, 0, or NaN
    chiral_sigmas: np.ndarray,
    out_center: np.ndarray,
    out_idx1: np.ndarray,
    out_idx2: np.ndarray,
    out_idx3: np.ndarray,
    out_signs: np.ndarray,
    out_sigmas: np.ndarray,
) -> int:
    """Match chiral restraints for a single residue."""
    count = 0
    n_chirals = len(chiral_center)
    n_atoms = len(residue_atom_names)

    for i in range(n_chirals):
        # Skip if unknown chiral sign (NaN)
        if np.isnan(chiral_volume_signs[i]):
            continue

        # Find all four atoms (center + 3 neighbors)
        idx_c, idx1, idx2, idx3 = -1, -1, -1, -1
        for j in range(n_atoms):
            name = residue_atom_names[j]
            if name == chiral_center[i]:
                idx_c = j
            elif name == chiral_atom1[i]:
                idx1 = j
            elif name == chiral_atom2[i]:
                idx2 = j
            elif name == chiral_atom3[i]:
                idx3 = j

        if idx_c < 0 or idx1 < 0 or idx2 < 0 or idx3 < 0:
            continue

        out_center[count] = residue_atom_indices[idx_c]
        out_idx1[count] = residue_atom_indices[idx1]
        out_idx2[count] = residue_atom_indices[idx2]
        out_idx3[count] = residue_atom_indices[idx3]
        out_signs[count] = chiral_volume_signs[i]
        out_sigmas[count] = chiral_sigmas[i]
        count += 1

    return count


# =============================================================================
# Pre-processed data structures
# =============================================================================


class PreprocessedPDB:
    """
    Pre-processed PDB data as NumPy arrays for fast access.

    Converts DataFrame operations to array lookups.
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

        # Optional
        if "altloc" in pdb.columns:
            self.altlocs = pdb["altloc"].values.astype(str)
        else:
            self.altlocs = np.full(self.n_atoms, " ", dtype=str)

        if "ATOM" in pdb.columns:
            self.atom_types = pdb["ATOM"].values.astype(str)
        else:
            self.atom_types = np.full(self.n_atoms, "ATOM", dtype=str)

        # Pre-compute residue boundaries
        self._compute_residue_boundaries()

    def _compute_residue_boundaries(self):
        """Compute start/end indices for each residue."""
        # Create residue key for grouping
        residue_keys = np.char.add(
            np.char.add(self.chain_ids, "_"), self.resseqs.astype(str)
        )

        # Find unique residues and their boundaries
        unique_keys, first_indices = np.unique(residue_keys, return_index=True)

        # Sort by first occurrence to maintain order
        order = np.argsort(first_indices)
        self.residue_keys = unique_keys[order]

        # Compute boundaries
        sorted_first = first_indices[order]
        self.residue_starts = sorted_first
        self.residue_ends = np.append(sorted_first[1:], self.n_atoms)
        self.n_residues = len(self.residue_keys)

        # Store residue metadata
        self.residue_chain_ids = self.chain_ids[self.residue_starts]
        self.residue_resseqs = self.resseqs[self.residue_starts]
        self.residue_resnames = self.resnames[self.residue_starts]

    def get_residue_atoms(self, residue_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get atom names and indices for a residue.

        Parameters
        ----------
        residue_idx : int
            Index into residue arrays.

        Returns
        -------
        atom_names : np.ndarray
            Atom names for this residue.
        atom_indices : np.ndarray
            Global atom indices.
        """
        start = self.residue_starts[residue_idx]
        end = self.residue_ends[residue_idx]
        return self.atom_names[start:end], self.atom_indices[start:end]


class PreprocessedCIF:
    """
    Pre-processed CIF restraints as NumPy arrays.

    Stores restraint data per residue type for fast lookup.
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

        # Pre-process each restraint type for each residue
        self.bonds = {}
        self.angles = {}
        self.torsions = {}
        self.planes = {}
        self.chirals = {}

        for restype, data in cif_dict.items():
            if "bonds" in data:
                self.bonds[restype] = self._preprocess_bonds(data["bonds"])
            if "angles" in data:
                self.angles[restype] = self._preprocess_angles(data["angles"])
            if "torsions" in data:
                self.torsions[restype] = self._preprocess_torsions(data["torsions"])
            if "planes" in data:
                self.planes[restype] = self._preprocess_planes(data["planes"])
            if "chirals" in data:
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

    def _preprocess_torsions(self, torsions_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Convert torsions DataFrame to NumPy arrays."""
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

    def _preprocess_planes(self, planes_df: pd.DataFrame) -> Dict[str, Any]:
        """Convert planes DataFrame to structured data."""
        # Planes are more complex - group by plane_id
        plane_ids = planes_df["plane_id"].unique()
        planes_data = []

        for plane_id in plane_ids:
            plane_atoms = planes_df[planes_df["plane_id"] == plane_id]
            planes_data.append(
                {
                    "atoms": plane_atoms["atom"].values.astype(str),
                    "sigmas": (
                        plane_atoms["sigma"].values.astype(np.float64)
                        if "sigma" in plane_atoms.columns
                        else np.full(len(plane_atoms), 0.02)
                    ),
                }
            )

        return planes_data

    def _preprocess_chirals(self, chirals_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Convert chirals DataFrame to NumPy arrays."""
        # Use string volume_sign to determine sign
        volume_signs = []
        for sign in chirals_df["volume_sign"].values:
            if sign == "positive":
                volume_signs.append(1.0)
            elif sign == "negative":
                volume_signs.append(-1.0)
            elif sign in ["both", "either"]:
                volume_signs.append(0.0)  # Achiral/racemic
            else:
                volume_signs.append(np.nan)  # Will be filtered out

        return {
            "center": chirals_df["atom_centre"].values.astype(str),
            "atom1": chirals_df["atom1"].values.astype(str),
            "atom2": chirals_df["atom2"].values.astype(str),
            "atom3": chirals_df["atom3"].values.astype(str),
            "volume_sign": np.array(volume_signs, dtype=np.float64),
            "sigma": (
                chirals_df["sigma"].values.astype(np.float64)
                if "sigma" in chirals_df.columns
                else np.full(len(chirals_df), 0.2)
            ),
        }


# =============================================================================
# Fast Builder Classes
# =============================================================================


class FastResidueIterator:
    """
    Fast residue iterator using pre-processed PDB data.
    """

    def __init__(self, preprocessed: PreprocessedPDB, filter_protein: bool = False):
        """
        Initialize from preprocessed PDB.

        Parameters
        ----------
        preprocessed : PreprocessedPDB
            Pre-processed PDB data.
        filter_protein : bool
            If True, only iterate over ATOM records.
        """
        self.pdb = preprocessed
        self.filter_protein = filter_protein

        if filter_protein:
            # Create mask for protein atoms
            self.protein_mask = preprocessed.atom_types == "ATOM"
        else:
            self.protein_mask = None

    def __iter__(self) -> Iterator[Tuple[int, str, int, str, np.ndarray, np.ndarray]]:
        """
        Iterate over residues.

        Yields
        ------
        residue_idx : int
            Index of residue.
        chain_id : str
            Chain identifier.
        resseq : int
            Residue sequence number.
        resname : str
            Residue name.
        atom_names : np.ndarray
            Atom names in this residue.
        atom_indices : np.ndarray
            Global atom indices.
        """
        for i in range(self.pdb.n_residues):
            atom_names, atom_indices = self.pdb.get_residue_atoms(i)

            if self.filter_protein and self.protein_mask is not None:
                start = self.pdb.residue_starts[i]
                end = self.pdb.residue_ends[i]
                mask = self.protein_mask[start:end]
                atom_names = atom_names[mask]
                atom_indices = atom_indices[mask]

                if len(atom_names) == 0:
                    continue

            yield (
                i,
                self.pdb.residue_chain_ids[i],
                self.pdb.residue_resseqs[i],
                self.pdb.residue_resnames[i],
                atom_names,
                atom_indices,
            )

    def __len__(self) -> int:
        return self.pdb.n_residues


class FastBondBuilder:
    """
    Fast bond restraint builder using Numba.
    """

    def __init__(self, verbose: int = 0):
        self.verbose = verbose
        self._indices = []
        self._references = []
        self._sigmas = []
        self._count = 0

        # Pre-allocate work arrays (will grow if needed)
        self._max_bonds = 50  # per residue
        self._work_idx1 = np.zeros(self._max_bonds, dtype=np.int64)
        self._work_idx2 = np.zeros(self._max_bonds, dtype=np.int64)
        self._work_refs = np.zeros(self._max_bonds, dtype=np.float64)
        self._work_sigmas = np.zeros(self._max_bonds, dtype=np.float64)

    def process_residue(
        self,
        atom_names: np.ndarray,
        atom_indices: np.ndarray,
        cif_bonds: Dict[str, np.ndarray],
    ) -> int:
        """
        Process bonds for a single residue.

        Parameters
        ----------
        atom_names : np.ndarray
            Atom names in this residue.
        atom_indices : np.ndarray
            Global atom indices.
        cif_bonds : dict
            Pre-processed bond data from CIF.

        Returns
        -------
        int
            Number of bonds added.
        """
        # Skip if duplicate atom names (altlocs not expanded)
        if len(atom_names) != len(set(atom_names)):
            return 0

        n_cif_bonds = len(cif_bonds["atom1"])

        # Ensure work arrays are large enough
        if n_cif_bonds > self._max_bonds:
            self._max_bonds = n_cif_bonds * 2
            self._work_idx1 = np.zeros(self._max_bonds, dtype=np.int64)
            self._work_idx2 = np.zeros(self._max_bonds, dtype=np.int64)
            self._work_refs = np.zeros(self._max_bonds, dtype=np.float64)
            self._work_sigmas = np.zeros(self._max_bonds, dtype=np.float64)

        # Use Numba-accelerated matching
        count = match_bonds_numba(
            atom_names,
            atom_indices,
            cif_bonds["atom1"],
            cif_bonds["atom2"],
            cif_bonds["value"],
            cif_bonds["sigma"],
            self._work_idx1,
            self._work_idx2,
            self._work_refs,
            self._work_sigmas,
        )

        if count > 0:
            # Collect results
            indices = np.column_stack(
                [self._work_idx1[:count].copy(), self._work_idx2[:count].copy()]
            )
            self._indices.append(indices)
            self._references.append(self._work_refs[:count].copy())
            self._sigmas.append(self._work_sigmas[:count].copy())
            self._count += count

        return count

    def finalize(
        self, device: torch.device, sort_indices: bool = True, min_sigma: float = 1e-4
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Convert accumulated data to tensors."""
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
            "references": torch.tensor(references, dtype=torch.float32, device=device),
            "sigmas": torch.tensor(sigmas, dtype=torch.float32, device=device),
        }

    def reset(self):
        self._indices.clear()
        self._references.clear()
        self._sigmas.clear()
        self._count = 0


class FastAngleBuilder:
    """Fast angle restraint builder using Numba."""

    def __init__(self, verbose: int = 0):
        self.verbose = verbose
        self._indices = []
        self._references = []
        self._sigmas = []
        self._count = 0

        self._max_angles = 100
        self._work_idx1 = np.zeros(self._max_angles, dtype=np.int64)
        self._work_idx2 = np.zeros(self._max_angles, dtype=np.int64)
        self._work_idx3 = np.zeros(self._max_angles, dtype=np.int64)
        self._work_refs = np.zeros(self._max_angles, dtype=np.float64)
        self._work_sigmas = np.zeros(self._max_angles, dtype=np.float64)

    def process_residue(
        self,
        atom_names: np.ndarray,
        atom_indices: np.ndarray,
        cif_angles: Dict[str, np.ndarray],
    ) -> int:
        # Skip if duplicate atom names (altlocs not expanded)
        if len(atom_names) != len(set(atom_names)):
            return 0

        n_cif_angles = len(cif_angles["atom1"])

        if n_cif_angles > self._max_angles:
            self._max_angles = n_cif_angles * 2
            self._work_idx1 = np.zeros(self._max_angles, dtype=np.int64)
            self._work_idx2 = np.zeros(self._max_angles, dtype=np.int64)
            self._work_idx3 = np.zeros(self._max_angles, dtype=np.int64)
            self._work_refs = np.zeros(self._max_angles, dtype=np.float64)
            self._work_sigmas = np.zeros(self._max_angles, dtype=np.float64)

        count = match_angles_numba(
            atom_names,
            atom_indices,
            cif_angles["atom1"],
            cif_angles["atom2"],
            cif_angles["atom3"],
            cif_angles["value"],
            cif_angles["sigma"],
            self._work_idx1,
            self._work_idx2,
            self._work_idx3,
            self._work_refs,
            self._work_sigmas,
        )

        if count > 0:
            indices = np.column_stack(
                [
                    self._work_idx1[:count].copy(),
                    self._work_idx2[:count].copy(),
                    self._work_idx3[:count].copy(),
                ]
            )
            self._indices.append(indices)
            self._references.append(self._work_refs[:count].copy())
            self._sigmas.append(self._work_sigmas[:count].copy())
            self._count += count

        return count

    def finalize(
        self, device: torch.device, sort_indices: bool = True, min_sigma: float = 1e-4
    ) -> Optional[Dict[str, torch.Tensor]]:
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
            "references": torch.tensor(references, dtype=torch.float32, device=device),
            "sigmas": torch.tensor(sigmas, dtype=torch.float32, device=device),
        }

    def reset(self):
        self._indices.clear()
        self._references.clear()
        self._sigmas.clear()
        self._count = 0


class FastTorsionBuilder:
    """Fast torsion restraint builder using Numba."""

    def __init__(self, verbose: int = 0):
        self.verbose = verbose
        self._indices = []
        self._references = []
        self._sigmas = []
        self._periods = []
        self._count = 0

        self._max_torsions = 50
        self._work_idx1 = np.zeros(self._max_torsions, dtype=np.int64)
        self._work_idx2 = np.zeros(self._max_torsions, dtype=np.int64)
        self._work_idx3 = np.zeros(self._max_torsions, dtype=np.int64)
        self._work_idx4 = np.zeros(self._max_torsions, dtype=np.int64)
        self._work_refs = np.zeros(self._max_torsions, dtype=np.float64)
        self._work_sigmas = np.zeros(self._max_torsions, dtype=np.float64)
        self._work_periods = np.zeros(self._max_torsions, dtype=np.int64)

    def process_residue(
        self,
        atom_names: np.ndarray,
        atom_indices: np.ndarray,
        cif_torsions: Dict[str, np.ndarray],
    ) -> int:
        # Skip if duplicate atom names (altlocs not expanded)
        if len(atom_names) != len(set(atom_names)):
            return 0

        n_cif_torsions = len(cif_torsions["atom1"])

        if n_cif_torsions > self._max_torsions:
            self._max_torsions = n_cif_torsions * 2
            self._work_idx1 = np.zeros(self._max_torsions, dtype=np.int64)
            self._work_idx2 = np.zeros(self._max_torsions, dtype=np.int64)
            self._work_idx3 = np.zeros(self._max_torsions, dtype=np.int64)
            self._work_idx4 = np.zeros(self._max_torsions, dtype=np.int64)
            self._work_refs = np.zeros(self._max_torsions, dtype=np.float64)
            self._work_sigmas = np.zeros(self._max_torsions, dtype=np.float64)
            self._work_periods = np.zeros(self._max_torsions, dtype=np.int64)

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
            self._work_idx1,
            self._work_idx2,
            self._work_idx3,
            self._work_idx4,
            self._work_refs,
            self._work_sigmas,
            self._work_periods,
        )

        if count > 0:
            indices = np.column_stack(
                [
                    self._work_idx1[:count].copy(),
                    self._work_idx2[:count].copy(),
                    self._work_idx3[:count].copy(),
                    self._work_idx4[:count].copy(),
                ]
            )
            self._indices.append(indices)
            self._references.append(self._work_refs[:count].copy())
            self._sigmas.append(self._work_sigmas[:count].copy())
            self._periods.append(self._work_periods[:count].copy())
            self._count += count

        return count

    def finalize(
        self, device: torch.device, sort_indices: bool = True, min_sigma: float = 1e-4
    ) -> Optional[Dict[str, torch.Tensor]]:
        if not self._indices:
            return None

        indices = np.concatenate(self._indices, axis=0)
        references = np.concatenate(self._references)
        sigmas = np.concatenate(self._sigmas)
        periods = np.concatenate(self._periods)

        if sort_indices and len(indices) > 0:
            sort_order = np.argsort(indices[:, 0])
            indices = indices[sort_order]
            references = references[sort_order]
            sigmas = sigmas[sort_order]
            periods = periods[sort_order]

        sigmas = np.where(sigmas == 0, min_sigma, sigmas)

        return {
            "indices": torch.tensor(indices, dtype=torch.long, device=device),
            "references": torch.tensor(references, dtype=torch.float32, device=device),
            "sigmas": torch.tensor(sigmas, dtype=torch.float32, device=device),
            "periods": torch.tensor(periods, dtype=torch.long, device=device),
        }

    def reset(self):
        self._indices.clear()
        self._references.clear()
        self._sigmas.clear()
        self._periods.clear()
        self._count = 0


class FastPlaneBuilder:
    """Fast plane restraint builder."""

    def __init__(self, verbose: int = 0):
        self.verbose = verbose
        # Group planes by number of atoms - store (indices, sigmas_array) tuples
        self._planes_by_size: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}
        self._count = 0

    def process_residue(
        self, atom_names: np.ndarray, atom_indices: np.ndarray, cif_planes: List[Dict]
    ) -> int:
        """Process plane restraints for a residue."""
        count = 0

        # Check for duplicate atom names (skip if found)
        if len(atom_names) != len(set(atom_names)):
            return 0

        # Build name to index map
        name_to_idx = {name: idx for name, idx in zip(atom_names, atom_indices)}

        for plane_data in cif_planes:
            plane_atom_names = plane_data["atoms"]
            plane_sigmas = plane_data["sigmas"]

            # Find all atoms in this plane
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

                if n_atoms not in self._planes_by_size:
                    self._planes_by_size[n_atoms] = []

                self._planes_by_size[n_atoms].append((indices_array, sigmas_array))
                count += 1

        self._count += count
        return count

    def finalize(
        self, device: torch.device, sort_indices: bool = True, min_sigma: float = 1e-4
    ) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        """Finalize plane restraints, grouped by atom count."""
        if not self._planes_by_size:
            return None

        result = {}

        for n_atoms, planes_list in self._planes_by_size.items():
            indices = np.stack([p[0] for p in planes_list], axis=0)
            sigmas = np.stack([p[1] for p in planes_list], axis=0)

            if sort_indices and len(indices) > 0:
                sort_order = np.argsort(indices[:, 0])
                indices = indices[sort_order]
                sigmas = sigmas[sort_order]

            sigmas = np.where(sigmas == 0, min_sigma, sigmas)

            # Use same key format as original: '{n}_atoms'
            key = f"{n_atoms}_atoms"
            result[key] = {
                "indices": torch.tensor(indices, dtype=torch.long, device=device),
                "sigmas": torch.tensor(sigmas, dtype=torch.float32, device=device),
            }

        return result

    def reset(self):
        self._planes_by_size.clear()
        self._count = 0


class FastChiralBuilder:
    """Fast chiral restraint builder using Numba."""

    def __init__(self, verbose: int = 0):
        self.verbose = verbose
        self._indices = []
        self._ideal_volumes = []
        self._sigmas = []
        self._count = 0

        self._max_chirals = 20
        self._work_center = np.zeros(self._max_chirals, dtype=np.int64)
        self._work_idx1 = np.zeros(self._max_chirals, dtype=np.int64)
        self._work_idx2 = np.zeros(self._max_chirals, dtype=np.int64)
        self._work_idx3 = np.zeros(self._max_chirals, dtype=np.int64)
        self._work_signs = np.zeros(self._max_chirals, dtype=np.float64)
        self._work_sigmas = np.zeros(self._max_chirals, dtype=np.float64)

    def process_residue(
        self,
        atom_names: np.ndarray,
        atom_indices: np.ndarray,
        cif_chirals: Dict[str, np.ndarray],
    ) -> int:
        # Skip if duplicate atom names (altlocs not expanded)
        if len(atom_names) != len(set(atom_names)):
            return 0

        n_cif_chirals = len(cif_chirals["center"])

        if n_cif_chirals > self._max_chirals:
            self._max_chirals = n_cif_chirals * 2
            self._work_center = np.zeros(self._max_chirals, dtype=np.int64)
            self._work_idx1 = np.zeros(self._max_chirals, dtype=np.int64)
            self._work_idx2 = np.zeros(self._max_chirals, dtype=np.int64)
            self._work_idx3 = np.zeros(self._max_chirals, dtype=np.int64)
            self._work_signs = np.zeros(self._max_chirals, dtype=np.float64)
            self._work_sigmas = np.zeros(self._max_chirals, dtype=np.float64)

        count = match_chirals_numba(
            atom_names,
            atom_indices,
            cif_chirals["center"],
            cif_chirals["atom1"],
            cif_chirals["atom2"],
            cif_chirals["atom3"],
            cif_chirals["volume_sign"],
            cif_chirals["sigma"],
            self._work_center,
            self._work_idx1,
            self._work_idx2,
            self._work_idx3,
            self._work_signs,
            self._work_sigmas,
        )

        if count > 0:
            indices = np.column_stack(
                [
                    self._work_center[:count].copy(),
                    self._work_idx1[:count].copy(),
                    self._work_idx2[:count].copy(),
                    self._work_idx3[:count].copy(),
                ]
            )
            # Ideal volume = sign * 2.5 (typical tetrahedral volume)
            ideal_volumes = self._work_signs[:count].copy() * 2.5

            self._indices.append(indices)
            self._ideal_volumes.append(ideal_volumes)
            self._sigmas.append(self._work_sigmas[:count].copy())
            self._count += count

        return count

    def finalize(
        self, device: torch.device, sort_indices: bool = True, min_sigma: float = 1e-4
    ) -> Optional[Dict[str, torch.Tensor]]:
        if not self._indices:
            return None

        indices = np.concatenate(self._indices, axis=0)
        ideal_volumes = np.concatenate(self._ideal_volumes)
        sigmas = np.concatenate(self._sigmas)

        if sort_indices and len(indices) > 0:
            sort_order = np.argsort(indices[:, 0])
            indices = indices[sort_order]
            ideal_volumes = ideal_volumes[sort_order]
            sigmas = sigmas[sort_order]

        sigmas = np.where(sigmas == 0, min_sigma, sigmas)

        return {
            "indices": torch.tensor(indices, dtype=torch.long, device=device),
            "ideal_volumes": torch.tensor(
                ideal_volumes, dtype=torch.float32, device=device
            ),
            "sigmas": torch.tensor(sigmas, dtype=torch.float32, device=device),
        }

    def reset(self):
        self._indices.clear()
        self._ideal_volumes.clear()
        self._sigmas.clear()
        self._count = 0


# =============================================================================
# High-level API matching original builders
# =============================================================================


def build_all_restraints_fast(
    pdb: pd.DataFrame, cif_dict: Dict, device: torch.device, verbose: int = 0
) -> Dict[str, Any]:
    """
    Build all restraints using the fast implementation.

    Parameters
    ----------
    pdb : pd.DataFrame
        PDB DataFrame.
    cif_dict : dict
        CIF dictionary with restraints.
    device : torch.device
        Target device.
    verbose : int
        Verbosity level.

    Returns
    -------
    dict
        Dictionary with all restraint tensors.
    """
    # Pre-process data
    preprocessed_pdb = PreprocessedPDB(pdb)
    preprocessed_cif = PreprocessedCIF(cif_dict)

    # Create builders
    bond_builder = FastBondBuilder(verbose=verbose)
    angle_builder = FastAngleBuilder(verbose=verbose)
    torsion_builder = FastTorsionBuilder(verbose=verbose)
    plane_builder = FastPlaneBuilder(verbose=verbose)
    chiral_builder = FastChiralBuilder(verbose=verbose)

    # Process all residues
    iterator = FastResidueIterator(preprocessed_pdb)

    for res_idx, chain_id, resseq, resname, atom_names, atom_indices in iterator:
        # Bonds
        if resname in preprocessed_cif.bonds:
            bond_builder.process_residue(
                atom_names, atom_indices, preprocessed_cif.bonds[resname]
            )

        # Angles
        if resname in preprocessed_cif.angles:
            angle_builder.process_residue(
                atom_names, atom_indices, preprocessed_cif.angles[resname]
            )

        # Torsions
        if resname in preprocessed_cif.torsions:
            torsion_builder.process_residue(
                atom_names, atom_indices, preprocessed_cif.torsions[resname]
            )

        # Planes
        if resname in preprocessed_cif.planes:
            plane_builder.process_residue(
                atom_names, atom_indices, preprocessed_cif.planes[resname]
            )

        # Chirals
        if resname in preprocessed_cif.chirals:
            chiral_builder.process_residue(
                atom_names, atom_indices, preprocessed_cif.chirals[resname]
            )

    # Finalize
    result = {}

    bond_result = bond_builder.finalize(device)
    if bond_result:
        result["bond"] = bond_result

    angle_result = angle_builder.finalize(device)
    if angle_result:
        result["angle"] = angle_result

    torsion_result = torsion_builder.finalize(device)
    if torsion_result:
        result["torsion"] = torsion_result

    plane_result = plane_builder.finalize(device)
    if plane_result:
        result["plane"] = plane_result

    chiral_result = chiral_builder.finalize(device)
    if chiral_result:
        result["chiral"] = chiral_result

    return result
