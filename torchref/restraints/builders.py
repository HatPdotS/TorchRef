"""
Restraint Builder Classes for Crystallographic Refinement

This module provides a hierarchy of builder classes for constructing
geometry restraints efficiently. The base class handles common functionality
while specialized child classes implement restraint-type-specific logic.

The builder pattern allows:
- Single-pass iteration over residues
- Easy unit testing of individual components
- Memory-efficient accumulation and finalization
- Sorted indices for cache-friendly access

Classes
-------
ResidueIterator
    Efficient iterator over residues with pre-grouped data.
RestraintBuilder
    Abstract base class for all restraint builders.
BondRestraintBuilder
    Builder for bond length restraints.
AngleRestraintBuilder
    Builder for angle restraints.
TorsionRestraintBuilder
    Builder for torsion angle restraints.
PlaneRestraintBuilder
    Builder for planarity restraints.
ChiralRestraintBuilder
    Builder for chiral volume restraints.
"""

from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from torchref.config import get_float_dtype


class ResidueIterator:
    """
    Efficient iterator over residues with pre-grouped data.

    Pre-groups PDB data by (chainid, resseq) once at initialization,
    avoiding O(N) DataFrame filtering per residue. This reduces the
    complexity from O(N×R) to O(N log N) where N is the number of atoms
    and R is the number of residues.

    Parameters
    ----------
    pdb : pd.DataFrame
        PDB DataFrame with columns 'chainid', 'resseq', 'resname', 'name', 'index'.
    filter_atom_type : str, optional
        If provided, only include atoms with this ATOM type (e.g., 'ATOM' for protein).

    Attributes
    ----------
    pdb : pd.DataFrame
        Reference to the input DataFrame.
    groups : list of tuple
        List of (chainid, resseq) tuples for iteration.

    Examples
    --------
    ::

        iterator = ResidueIterator(model.pdb)
        for chain_id, resseq, residue_df in iterator:
            print(f"Processing {chain_id}:{resseq}")
    """

    def __init__(self, pdb: pd.DataFrame, filter_atom_type: Optional[str] = None):
        """Initialize with pre-grouping of residues."""
        if filter_atom_type is not None:
            pdb = pdb[pdb["ATOM"] == filter_atom_type]

        self.pdb = pdb
        # Pre-group once - O(N log N) instead of O(N×R)
        self._grouped = pdb.groupby(["chainid", "resseq"], sort=False)
        self.groups = list(self._grouped.groups.keys())

    def __iter__(self) -> Iterator[Tuple[str, int, pd.DataFrame]]:
        """
        Iterate over residues.

        Yields
        ------
        chain_id : str
            Chain identifier.
        resseq : int
            Residue sequence number.
        residue : pd.DataFrame
            DataFrame containing all atoms in this residue.
        """
        for chain_id, resseq in self.groups:
            residue = self._grouped.get_group((chain_id, resseq))
            yield chain_id, resseq, residue

    def __len__(self) -> int:
        """Return number of residues."""
        return len(self.groups)

    def get_consecutive_pairs(self) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Iterate over consecutive residue pairs within each chain.

        Useful for building inter-residue restraints (peptide bonds, etc.).

        Yields
        ------
        residue_i : pd.DataFrame
            First residue in the pair.
        residue_next : pd.DataFrame
            Next consecutive residue (resseq + 1).
        """
        # Group by chain first
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

                # Skip if not consecutive (chain break)
                if resseq_next != resseq_i + 1:
                    continue

                residue_i = self._grouped.get_group((chain_id, resseq_i))
                residue_next = self._grouped.get_group((chain_id, resseq_next))
                yield residue_i, residue_next


class RestraintBuilder(ABC):
    """
    Abstract base class for restraint builders.

    Provides common functionality for accumulating restraint data during
    residue iteration and finalizing to sorted tensors. Child classes
    implement the specific logic for each restraint type.

    Parameters
    ----------
    verbose : int, default 0
        Verbosity level for debug output.

    Attributes
    ----------
    _indices : list of np.ndarray
        Accumulated index arrays.
    _references : list of np.ndarray
        Accumulated reference value arrays.
    _sigmas : list of np.ndarray
        Accumulated sigma arrays.
    _count : int
        Total number of restraints added.
    """

    # Class attributes to be overridden by child classes
    restraint_type: str = "base"
    atom_columns: List[str] = []
    n_atoms: int = 0

    def __init__(self, verbose: int = 0):
        """Initialize empty accumulator lists."""
        self.verbose = verbose
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

    @abstractmethod
    def process_residue(
        self, residue: pd.DataFrame, cif_restraints: pd.DataFrame
    ) -> int:
        """
        Process a single residue, matching atoms to CIF restraints.

        Parameters
        ----------
        residue : pd.DataFrame
            Residue atoms with 'name' and 'index' columns.
        cif_restraints : pd.DataFrame
            Restraints from CIF dictionary for this residue type.

        Returns
        -------
        int
            Number of restraints added.
        """
        pass

    def _filter_usable_restraints(
        self, cif_restraints: pd.DataFrame, atom_names: set
    ) -> pd.DataFrame:
        """
        Filter CIF restraints to those where all required atoms are present.

        Parameters
        ----------
        cif_restraints : pd.DataFrame
            All restraints from CIF for this residue type.
        atom_names : set
            Set of atom names present in the residue.

        Returns
        -------
        pd.DataFrame
            Filtered restraints where all atoms exist.
        """
        mask = np.all(
            [cif_restraints[col].isin(atom_names).values for col in self.atom_columns],
            axis=0,
        )
        return cif_restraints.loc[mask]

    def _build_name_to_index_map(
        self, residue: pd.DataFrame
    ) -> Optional[Dict[str, int]]:
        """
        Create atom name to index mapping, checking for duplicates.

        Parameters
        ----------
        residue : pd.DataFrame
            Residue atoms with 'name' and 'index' columns.

        Returns
        -------
        dict or None
            Mapping from atom name to index, or None if duplicates exist.
        """
        residue_indexed = residue.set_index("name")
        if residue_indexed.index.has_duplicates:
            if self.verbose > 2:
                resname = residue["resname"].iloc[0] if len(residue) > 0 else "UNKNOWN"
                chain_id = residue["chainid"].iloc[0] if len(residue) > 0 else "UNKNOWN"
                resseq = residue["resseq"].iloc[0] if len(residue) > 0 else "UNKNOWN"
                print(
                    f"WARNING: Skipping {self.restraint_type} restraints for "
                    f"{resname} {chain_id} {resseq} (duplicate atom names)"
                )
            return None
        return dict(zip(residue["name"], residue["index"]))

    def _map_atoms_to_indices(
        self, usable_restraints: pd.DataFrame, name_to_idx: Dict[str, int]
    ) -> np.ndarray:
        """
        Map atom names in restraints to atom indices.

        Parameters
        ----------
        usable_restraints : pd.DataFrame
            Filtered restraints.
        name_to_idx : dict
            Mapping from atom name to index.

        Returns
        -------
        np.ndarray
            Array of shape (n_restraints, n_atoms) with atom indices.
        """
        indices = np.array(
            [
                [name_to_idx[name] for name in usable_restraints[col].values]
                for col in self.atom_columns
            ]
        ).T
        return indices

    def _accumulate(
        self, indices: np.ndarray, references: np.ndarray, sigmas: np.ndarray
    ):
        """
        Add restraint data to accumulator lists.

        Parameters
        ----------
        indices : np.ndarray
            Atom indices array of shape (n_restraints, n_atoms).
        references : np.ndarray
            Reference values array of shape (n_restraints,).
        sigmas : np.ndarray
            Sigma values array of shape (n_restraints,).
        """
        self._indices.append(indices)
        self._references.append(references)
        self._sigmas.append(sigmas)
        self._count += len(indices)

    def finalize(
        self, device: torch.device, sort_indices: bool = True, min_sigma: float = 1e-4
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Convert accumulated data to sorted tensors.

        Parameters
        ----------
        device : torch.device
            Target device for tensors.
        sort_indices : bool, default True
            Whether to sort by first atom index for memory locality.
        min_sigma : float, default 1e-4
            Minimum sigma value to avoid division by zero.

        Returns
        -------
        dict or None
            Dictionary with 'indices', 'references', 'sigmas' tensors,
            or None if no restraints were accumulated.
        """
        if not self._indices:
            return None

        indices = np.concatenate(self._indices, axis=0)
        references = np.concatenate(self._references)
        sigmas = np.concatenate(self._sigmas)

        # Sort by first atom index for cache-friendly access
        if sort_indices and len(indices) > 0:
            sort_order = np.argsort(indices[:, 0])
            indices = indices[sort_order]
            references = references[sort_order]
            sigmas = sigmas[sort_order]

        # Replace zero sigmas
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


class BondRestraintBuilder(RestraintBuilder):
    """
    Builder for bond length restraints.

    Bond restraints define expected distances between pairs of bonded atoms.
    Each restraint specifies: atom1, atom2, target_distance, sigma.

    Examples
    --------
    ::

        builder = BondRestraintBuilder()
        for residue in residues:
            builder.process_residue(residue, cif_bonds)
        result = builder.finalize(device)
        print(result['indices'].shape)  # (n_bonds, 2)
    """

    restraint_type = "bond"
    atom_columns = ["atom1", "atom2"]
    n_atoms = 2

    def process_residue(
        self, residue: pd.DataFrame, cif_restraints: pd.DataFrame
    ) -> int:
        """
        Process bond restraints for a single residue.

        Parameters
        ----------
        residue : pd.DataFrame
            Residue atoms with 'name' and 'index' columns.
        cif_restraints : pd.DataFrame
            Bond restraints with 'atom1', 'atom2', 'value', 'sigma' columns.

        Returns
        -------
        int
            Number of bond restraints added.
        """
        atom_names = set(residue["name"].values)
        usable = self._filter_usable_restraints(cif_restraints, atom_names)

        if len(usable) == 0:
            return 0

        name_to_idx = self._build_name_to_index_map(residue)
        if name_to_idx is None:
            return 0

        indices = self._map_atoms_to_indices(usable, name_to_idx)
        references = usable["value"].values.astype(np.float32)
        sigmas = usable["sigma"].values.astype(np.float32)

        self._accumulate(indices, references, sigmas)
        return len(usable)


class AngleRestraintBuilder(RestraintBuilder):
    """
    Builder for angle restraints.

    Angle restraints define expected angles between three atoms (1-2-3).
    The angle is measured at the central atom (atom2).

    Examples
    --------
    ::

        builder = AngleRestraintBuilder()
        for residue in residues:
            builder.process_residue(residue, cif_angles)
        result = builder.finalize(device)
        print(result['indices'].shape)  # (n_angles, 3)
    """

    restraint_type = "angle"
    atom_columns = ["atom1", "atom2", "atom3"]
    n_atoms = 3

    def process_residue(
        self, residue: pd.DataFrame, cif_restraints: pd.DataFrame
    ) -> int:
        """
        Process angle restraints for a single residue.

        Parameters
        ----------
        residue : pd.DataFrame
            Residue atoms with 'name' and 'index' columns.
        cif_restraints : pd.DataFrame
            Angle restraints with 'atom1', 'atom2', 'atom3', 'value', 'sigma' columns.

        Returns
        -------
        int
            Number of angle restraints added.
        """
        atom_names = set(residue["name"].values)
        usable = self._filter_usable_restraints(cif_restraints, atom_names)

        if len(usable) == 0:
            return 0

        name_to_idx = self._build_name_to_index_map(residue)
        if name_to_idx is None:
            return 0

        indices = self._map_atoms_to_indices(usable, name_to_idx)
        references = usable["value"].values.astype(np.float32)
        sigmas = usable["sigma"].values.astype(np.float32)

        self._accumulate(indices, references, sigmas)
        return len(usable)


class TorsionRestraintBuilder(RestraintBuilder):
    """
    Builder for torsion angle restraints.

    Torsion restraints define expected dihedral angles between four atoms.
    Includes periodicity information for symmetric torsions (e.g., period=3
    for methyl groups).

    Attributes
    ----------
    _periods : list of np.ndarray
        Accumulated periodicity arrays.

    Examples
    --------
    ::

        builder = TorsionRestraintBuilder()
        for residue in residues:
            builder.process_residue(residue, cif_torsions)
        result = builder.finalize(device)
        print(result['indices'].shape)  # (n_torsions, 4)
        print(result['periods'].shape)  # (n_torsions,)
    """

    restraint_type = "torsion"
    atom_columns = ["atom1", "atom2", "atom3", "atom4"]
    n_atoms = 4

    def __init__(self, verbose: int = 0):
        """Initialize with additional periods accumulator."""
        super().__init__(verbose)
        self._periods: List[np.ndarray] = []

    def reset(self):
        """Clear all accumulated data including periods."""
        super().reset()
        self._periods.clear()

    def process_residue(
        self, residue: pd.DataFrame, cif_restraints: pd.DataFrame
    ) -> int:
        """
        Process torsion restraints for a single residue.

        Filters out torsions with sigma=0 (undefined/flexible torsions).

        Parameters
        ----------
        residue : pd.DataFrame
            Residue atoms with 'name' and 'index' columns.
        cif_restraints : pd.DataFrame
            Torsion restraints with 'atom1'-'atom4', 'value', 'sigma',
            and optionally 'periodicity' columns.

        Returns
        -------
        int
            Number of torsion restraints added.
        """
        atom_names = set(residue["name"].values)
        usable = self._filter_usable_restraints(cif_restraints, atom_names)

        if len(usable) == 0:
            return 0

        name_to_idx = self._build_name_to_index_map(residue)
        if name_to_idx is None:
            return 0

        indices = self._map_atoms_to_indices(usable, name_to_idx)
        references = usable["value"].values.astype(np.float32)
        sigmas = usable["sigma"].values.astype(np.float32)

        # Get periodicity if available
        if "periodicity" in usable.columns:
            periods = usable["periodicity"].values.astype(np.int64)
        else:
            periods = np.ones(len(usable), dtype=np.int64)

        # Filter out torsions with sigma=0 (these are flexible/undefined)
        valid_mask = sigmas != 0
        if not np.any(valid_mask):
            return 0

        indices = indices[valid_mask]
        references = references[valid_mask]
        sigmas = sigmas[valid_mask]
        periods = periods[valid_mask]

        self._accumulate(indices, references, sigmas)
        self._periods.append(periods)
        return len(indices)

    def finalize(
        self, device: torch.device, sort_indices: bool = True, min_sigma: float = 1e-4
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Convert accumulated data to sorted tensors, including periods.

        Parameters
        ----------
        device : torch.device
            Target device for tensors.
        sort_indices : bool, default True
            Whether to sort by first atom index for memory locality.
        min_sigma : float, default 1e-4
            Minimum sigma value.

        Returns
        -------
        dict or None
            Dictionary with 'indices', 'references', 'sigmas', 'periods' tensors.
        """
        if not self._indices:
            return None

        indices = np.concatenate(self._indices, axis=0)
        references = np.concatenate(self._references)
        sigmas = np.concatenate(self._sigmas)
        periods = np.concatenate(self._periods)

        # Sort by first atom index
        if sort_indices and len(indices) > 0:
            sort_order = np.argsort(indices[:, 0])
            indices = indices[sort_order]
            references = references[sort_order]
            sigmas = sigmas[sort_order]
            periods = periods[sort_order]

        # Replace zero sigmas (shouldn't happen after filtering, but safety)
        sigmas = np.where(sigmas == 0, min_sigma, sigmas)

        return {
            "indices": torch.tensor(indices, dtype=torch.long, device=device),
            "references": torch.tensor(references, dtype=get_float_dtype(), device=device),
            "sigmas": torch.tensor(sigmas, dtype=get_float_dtype(), device=device),
            "periods": torch.tensor(periods, dtype=torch.long, device=device),
        }


class PlaneRestraintBuilder(RestraintBuilder):
    """
    Builder for planarity restraints.

    Plane restraints define groups of atoms that should be coplanar.
    Unlike other restraints, planes have variable atom counts (3-10 atoms).
    Results are grouped by atom count for efficient tensor operations.

    Attributes
    ----------
    _planes_by_count : dict
        Dictionary mapping atom_count -> {'indices': [...], 'sigmas': [...]}.

    Examples
    --------
    ::

        builder = PlaneRestraintBuilder()
        for residue in residues:
            builder.process_residue(residue, cif_planes)
        result = builder.finalize(device)
        # Returns dict like {'4_atoms': {...}, '6_atoms': {...}}
    """

    restraint_type = "plane"
    atom_columns = ["atom"]  # Planes use single atom column with plane_id grouping
    n_atoms = None  # Variable

    def __init__(self, verbose: int = 0):
        """Initialize with planes-by-count dictionary."""
        super().__init__(verbose)
        self._planes_by_count: Dict[int, Dict[str, List]] = {}

    def reset(self):
        """Clear all accumulated data."""
        super().reset()
        self._planes_by_count.clear()

    def process_residue(
        self, residue: pd.DataFrame, cif_restraints: pd.DataFrame
    ) -> int:
        """
        Process plane restraints for a single residue.

        Groups plane atoms by plane_id and stores separately by atom count.

        Parameters
        ----------
        residue : pd.DataFrame
            Residue atoms with 'name' and 'index' columns.
        cif_restraints : pd.DataFrame
            Plane restraints with 'plane_id', 'atom', 'sigma' columns.

        Returns
        -------
        int
            Number of plane restraints added.
        """
        atom_names = set(residue["name"].values)

        # Filter to atoms that exist in residue
        usable = cif_restraints[cif_restraints["atom"].isin(atom_names)]
        if len(usable) == 0:
            return 0

        name_to_idx = self._build_name_to_index_map(residue)
        if name_to_idx is None:
            return 0

        count = 0
        # Process each plane separately
        for plane_id in usable["plane_id"].unique():
            plane_atoms = usable[usable["plane_id"] == plane_id]

            atom_count = len(plane_atoms)
            # Skip invalid planes (fewer than 3 atoms)
            if atom_count < 3:
                continue

            # Get indices and sigmas for this plane
            atom_indices = np.array(
                [name_to_idx[name] for name in plane_atoms["atom"].values]
            )
            sigmas = plane_atoms["sigma"].values.astype(np.float32)

            # Store by atom count
            if atom_count not in self._planes_by_count:
                self._planes_by_count[atom_count] = {"indices": [], "sigmas": []}

            self._planes_by_count[atom_count]["indices"].append(atom_indices)
            self._planes_by_count[atom_count]["sigmas"].append(sigmas)
            count += 1

        self._count += count
        return count

    def finalize(
        self, device: torch.device, sort_indices: bool = True, min_sigma: float = 1e-4
    ) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        """
        Convert accumulated data to tensors grouped by atom count.

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
            Dictionary with keys like '4_atoms', '6_atoms', each containing
            'indices' and 'sigmas' tensors.
        """
        if not self._planes_by_count:
            return None

        result = {}
        for atom_count, data in self._planes_by_count.items():
            if not data["indices"]:
                continue

            indices = np.stack(data["indices"], axis=0)
            sigmas = np.stack(data["sigmas"], axis=0)

            # Sort by first atom index
            if sort_indices and len(indices) > 0:
                sort_order = np.argsort(indices[:, 0])
                indices = indices[sort_order]
                sigmas = sigmas[sort_order]

            # Replace zero sigmas
            sigmas = np.where(sigmas == 0, min_sigma, sigmas)

            key = f"{atom_count}_atoms"
            result[key] = {
                "indices": torch.tensor(indices, dtype=torch.long, device=device),
                "sigmas": torch.tensor(sigmas, dtype=get_float_dtype(), device=device),
            }

        return result if result else None


class ChiralRestraintBuilder(RestraintBuilder):
    """
    Builder for chiral volume restraints.

    Chiral restraints define the expected handedness (R/S configuration)
    of tetrahedral centers. Each restraint specifies a center atom and
    three neighbors, with a signed ideal volume.

    Attributes
    ----------
    _ideal_volumes : list of np.ndarray
        Accumulated ideal volume arrays (signed based on chirality).
    ideal_volume : float
        Magnitude of ideal chiral volume in Å³.
    sigma : float
        Standard deviation for restraint in Å³.

    Examples
    --------
    ::

        builder = ChiralRestraintBuilder(ideal_volume=2.5, sigma=0.2)
        for residue in residues:
            builder.process_residue(residue, cif_chirals)
        result = builder.finalize(device)
        print(result['indices'].shape)  # (n_chirals, 4)
        print(result['ideal_volumes'].shape)  # (n_chirals,)
    """

    restraint_type = "chiral"
    atom_columns = ["atom_centre", "atom1", "atom2", "atom3"]
    n_atoms = 4

    def __init__(self, ideal_volume: float = 2.5, sigma: float = 0.2, verbose: int = 0):
        """
        Initialize with chiral volume parameters.

        Parameters
        ----------
        ideal_volume : float, default 2.5
            Magnitude of ideal chiral volume in Å³.
        sigma : float, default 0.2
            Standard deviation for restraint in Å³.
        verbose : int, default 0
            Verbosity level.
        """
        super().__init__(verbose)
        self.ideal_volume = ideal_volume
        self.sigma = sigma
        self._ideal_volumes: List[np.ndarray] = []

    def reset(self):
        """Clear all accumulated data."""
        super().reset()
        self._ideal_volumes.clear()

    def process_residue(
        self, residue: pd.DataFrame, cif_restraints: pd.DataFrame
    ) -> int:
        """
        Process chiral restraints for a single residue.

        Parameters
        ----------
        residue : pd.DataFrame
            Residue atoms with 'name' and 'index' columns.
        cif_restraints : pd.DataFrame
            Chiral restraints with 'atom_centre', 'atom1', 'atom2', 'atom3',
            'volume_sign' columns.

        Returns
        -------
        int
            Number of chiral restraints added.
        """
        if cif_restraints.empty:
            return 0

        atom_names = set(residue["name"].values)
        name_to_idx = self._build_name_to_index_map(residue)
        if name_to_idx is None:
            return 0

        indices_list = []
        volumes_list = []
        sigmas_list = []

        for _, row in cif_restraints.iterrows():
            center = row["atom_centre"]
            atom1 = row["atom1"]
            atom2 = row["atom2"]
            atom3 = row["atom3"]
            volume_sign = row["volume_sign"]

            # Check all atoms are present
            if not all(a in atom_names for a in [center, atom1, atom2, atom3]):
                continue

            # Get indices
            try:
                idx_center = name_to_idx[center]
                idx1 = name_to_idx[atom1]
                idx2 = name_to_idx[atom2]
                idx3 = name_to_idx[atom3]
            except KeyError:
                continue

            # Determine signed ideal volume
            if volume_sign == "positive":
                signed_volume = self.ideal_volume
            elif volume_sign == "negative":
                signed_volume = -self.ideal_volume
            elif volume_sign in ["both", "either"]:
                signed_volume = 0.0  # Achiral/racemic
            else:
                if self.verbose > 2:
                    print(f"WARNING: Unknown chiral sign '{volume_sign}'")
                continue

            indices_list.append([idx_center, idx1, idx2, idx3])
            volumes_list.append(signed_volume)
            sigmas_list.append(self.sigma)

        if not indices_list:
            return 0

        self._indices.append(np.array(indices_list, dtype=np.int64))
        self._ideal_volumes.append(np.array(volumes_list, dtype=np.float32))
        self._sigmas.append(np.array(sigmas_list, dtype=np.float32))
        self._count += len(indices_list)
        return len(indices_list)

    def finalize(
        self, device: torch.device, sort_indices: bool = True, min_sigma: float = 1e-4
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Convert accumulated data to sorted tensors.

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
            Dictionary with 'indices', 'ideal_volumes', 'sigmas' tensors.
        """
        if not self._indices:
            return None

        indices = np.concatenate(self._indices, axis=0)
        ideal_volumes = np.concatenate(self._ideal_volumes)
        sigmas = np.concatenate(self._sigmas)

        # Sort by center atom index
        if sort_indices and len(indices) > 0:
            sort_order = np.argsort(indices[:, 0])
            indices = indices[sort_order]
            ideal_volumes = ideal_volumes[sort_order]
            sigmas = sigmas[sort_order]

        return {
            "indices": torch.tensor(indices, dtype=torch.long, device=device),
            "ideal_volumes": torch.tensor(
                ideal_volumes, dtype=get_float_dtype(), device=device
            ),
            "sigmas": torch.tensor(sigmas, dtype=get_float_dtype(), device=device),
        }


# =============================================================================
# Inter-Residue Restraint Builders
# =============================================================================


class InterResidueBondBuilder:
    """
    Builder for inter-residue bond restraints (peptide bonds, disulfide bonds).

    Unlike intra-residue builders, this handles bonds between atoms in
    different residues. It processes residue pairs and link definitions.

    Parameters
    ----------
    verbose : int, default 0
        Verbosity level for debug output.

    Attributes
    ----------
    _indices : list of np.ndarray
        Accumulated bond index arrays.
    _references : list of np.ndarray
        Accumulated reference distance arrays.
    _sigmas : list of np.ndarray
        Accumulated sigma arrays.

    Examples
    --------
    ::

        builder = InterResidueBondBuilder()
        for res_i, res_next in iterator.get_consecutive_pairs():
            builder.process_peptide_bond(res_i, res_next, trans_link)
        result = builder.finalize(device)
    """

    def __init__(self, verbose: int = 0):
        """Initialize empty accumulator lists."""
        self.verbose = verbose
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
        """
        Get atom index from residue, handling alternate conformations.

        Prefers atoms with no altloc (' '), then 'A', then first available.

        Parameters
        ----------
        residue : pd.DataFrame
            Residue atoms.
        atom_name : str
            Name of atom to find.

        Returns
        -------
        int or None
            Atom index, or None if not found.
        """
        atoms = residue[residue["name"] == atom_name]
        if len(atoms) == 0:
            return None
        if " " in atoms["altloc"].values:
            return int(atoms[atoms["altloc"] == " "].iloc[0]["index"])
        elif "A" in atoms["altloc"].values:
            return int(atoms[atoms["altloc"] == "A"].iloc[0]["index"])
        else:
            return int(atoms.iloc[0]["index"])

    def process_peptide_bond(
        self,
        residue_i: pd.DataFrame,
        residue_next: pd.DataFrame,
        link_bonds: pd.DataFrame,
    ) -> int:
        """
        Process peptide bond restraints between consecutive residues.

        Parameters
        ----------
        residue_i : pd.DataFrame
            First residue (C-terminal of bond).
        residue_next : pd.DataFrame
            Second residue (N-terminal of bond).
        link_bonds : pd.DataFrame
            Bond definitions from link dictionary with columns:
            'atom_1_comp_id', 'atom1', 'atom_2_comp_id', 'atom2', 'value', 'sigma'.

        Returns
        -------
        int
            Number of bond restraints added.
        """
        count = 0
        for _, bond_row in link_bonds.iterrows():
            comp1 = bond_row["atom_1_comp_id"]
            comp2 = bond_row["atom_2_comp_id"]
            atom1_name = bond_row["atom1"]
            atom2_name = bond_row["atom2"]

            # Get residue based on comp_id ('1' = residue_i, '2' = residue_next)
            res1 = residue_i if comp1 == "1" else residue_next
            res2 = residue_i if comp2 == "1" else residue_next

            idx1 = self._get_atom_index(res1, atom1_name)
            idx2 = self._get_atom_index(res2, atom2_name)

            if idx1 is not None and idx2 is not None:
                self._indices.append(np.array([[idx1, idx2]], dtype=np.int64))
                self._references.append(
                    np.array([float(bond_row["value"])], dtype=np.float32)
                )
                self._sigmas.append(
                    np.array([float(bond_row["sigma"])], dtype=np.float32)
                )
                count += 1

        self._count += count
        return count

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
        Convert accumulated data to sorted tensors.

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


class InterResidueAngleBuilder:
    """
    Builder for inter-residue angle restraints (peptide angles).

    Handles angles that span two consecutive residues.

    Parameters
    ----------
    verbose : int, default 0
        Verbosity level for debug output.
    """

    def __init__(self, verbose: int = 0):
        """Initialize empty accumulator lists."""
        self.verbose = verbose
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

    def process_peptide_angles(
        self,
        residue_i: pd.DataFrame,
        residue_next: pd.DataFrame,
        link_angles: pd.DataFrame,
    ) -> int:
        """
        Process peptide angle restraints between consecutive residues.

        Parameters
        ----------
        residue_i : pd.DataFrame
            First residue.
        residue_next : pd.DataFrame
            Second residue.
        link_angles : pd.DataFrame
            Angle definitions from link dictionary.

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

            res1 = residue_i if comp1 == "1" else residue_next
            res2 = residue_i if comp2 == "1" else residue_next
            res3 = residue_i if comp3 == "1" else residue_next

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
        return self.process_peptide_angles(res1_atoms, res2_atoms, link_angles)

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


class InterResidueTorsionBuilder:
    """
    Builder for inter-residue torsion restraints (phi, psi, omega).

    Handles backbone torsion angles that span consecutive residues.
    Separates phi, psi, and omega angles for different treatment.

    Parameters
    ----------
    verbose : int, default 0
        Verbosity level for debug output.
    """

    def __init__(self, verbose: int = 0):
        """Initialize empty accumulator lists for each torsion type."""
        self.verbose = verbose

        # Separate accumulators for phi, psi, omega
        self._phi_indices: List[np.ndarray] = []
        self._phi_periods: List[np.ndarray] = []

        self._psi_indices: List[np.ndarray] = []
        self._psi_periods: List[np.ndarray] = []

        self._omega_indices: List[np.ndarray] = []
        self._omega_references: List[np.ndarray] = []
        self._omega_sigmas: List[np.ndarray] = []
        self._omega_periods: List[np.ndarray] = []
        self._omega_is_proline: List[bool] = []

        # For disulfide torsions
        self._disulfide_indices: List[np.ndarray] = []
        self._disulfide_references: List[np.ndarray] = []
        self._disulfide_sigmas: List[np.ndarray] = []
        self._disulfide_periods: List[np.ndarray] = []

    def reset(self):
        """Clear all accumulated data."""
        self._phi_indices.clear()
        self._phi_periods.clear()
        self._psi_indices.clear()
        self._psi_periods.clear()
        self._omega_indices.clear()
        self._omega_references.clear()
        self._omega_sigmas.clear()
        self._omega_periods.clear()
        self._omega_is_proline.clear()
        self._disulfide_indices.clear()
        self._disulfide_references.clear()
        self._disulfide_sigmas.clear()
        self._disulfide_periods.clear()

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

    def process_peptide_torsions(
        self,
        residue_i: pd.DataFrame,
        residue_next: pd.DataFrame,
        link_torsions: pd.DataFrame,
    ) -> Tuple[int, int, int]:
        """
        Process backbone torsion restraints between consecutive residues.

        Parameters
        ----------
        residue_i : pd.DataFrame
            First residue.
        residue_next : pd.DataFrame
            Second residue.
        link_torsions : pd.DataFrame
            Torsion definitions from link dictionary with 'id' column
            indicating 'phi', 'psi', or 'omega'.

        Returns
        -------
        tuple of (int, int, int)
            Number of (phi, psi, omega) torsions added.
        """
        resname_next = residue_next["resname"].iloc[0]
        is_proline = resname_next == "PRO"

        phi_count = 0
        psi_count = 0
        omega_count = 0

        for _, torsion_row in link_torsions.iterrows():
            comp1 = torsion_row["atom_1_comp_id"]
            comp2 = torsion_row["atom_2_comp_id"]
            comp3 = torsion_row["atom_3_comp_id"]
            comp4 = torsion_row["atom_4_comp_id"]
            atom1_name = torsion_row["atom1"]
            atom2_name = torsion_row["atom2"]
            atom3_name = torsion_row["atom3"]
            atom4_name = torsion_row["atom4"]
            torsion_id = torsion_row["id"]

            res1 = residue_i if comp1 == "1" else residue_next
            res2 = residue_i if comp2 == "1" else residue_next
            res3 = residue_i if comp3 == "1" else residue_next
            res4 = residue_i if comp4 == "1" else residue_next

            idx1 = self._get_atom_index(res1, atom1_name)
            idx2 = self._get_atom_index(res2, atom2_name)
            idx3 = self._get_atom_index(res3, atom3_name)
            idx4 = self._get_atom_index(res4, atom4_name)

            if idx1 is None or idx2 is None or idx3 is None or idx4 is None:
                continue

            period = (
                int(torsion_row["period"])
                if "period" in torsion_row and pd.notna(torsion_row["period"])
                else 0
            )

            if torsion_id == "phi":
                self._phi_indices.append(
                    np.array([[idx1, idx2, idx3, idx4]], dtype=np.int64)
                )
                self._phi_periods.append(np.array([period], dtype=np.int64))
                phi_count += 1
            elif torsion_id == "psi":
                self._psi_indices.append(
                    np.array([[idx1, idx2, idx3, idx4]], dtype=np.int64)
                )
                self._psi_periods.append(np.array([period], dtype=np.int64))
                psi_count += 1
            elif torsion_id == "omega":
                self._omega_indices.append(
                    np.array([[idx1, idx2, idx3, idx4]], dtype=np.int64)
                )
                self._omega_references.append(
                    np.array([float(torsion_row["value"])], dtype=np.float32)
                )
                self._omega_sigmas.append(
                    np.array([float(torsion_row["sigma"])], dtype=np.float32)
                )
                self._omega_periods.append(np.array([period], dtype=np.int64))
                self._omega_is_proline.append(is_proline)
                omega_count += 1

        return phi_count, psi_count, omega_count

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

        return count

    def finalize_phi(
        self, device: torch.device, sort_indices: bool = True
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Finalize phi angle indices and periods."""
        if not self._phi_indices:
            return None

        indices = np.concatenate(self._phi_indices, axis=0)
        periods = np.concatenate(self._phi_periods)

        if sort_indices and len(indices) > 0:
            sort_order = np.argsort(indices[:, 0])
            indices = indices[sort_order]
            periods = periods[sort_order]

        return {
            "indices": torch.tensor(indices, dtype=torch.long, device=device),
            "periods": torch.tensor(periods, dtype=torch.long, device=device),
        }

    def finalize_psi(
        self, device: torch.device, sort_indices: bool = True
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Finalize psi angle indices and periods."""
        if not self._psi_indices:
            return None

        indices = np.concatenate(self._psi_indices, axis=0)
        periods = np.concatenate(self._psi_periods)

        if sort_indices and len(indices) > 0:
            sort_order = np.argsort(indices[:, 0])
            indices = indices[sort_order]
            periods = periods[sort_order]

        return {
            "indices": torch.tensor(indices, dtype=torch.long, device=device),
            "periods": torch.tensor(periods, dtype=torch.long, device=device),
        }

    def finalize_omega(
        self, device: torch.device, sort_indices: bool = True
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Finalize omega angle restraints."""
        if not self._omega_indices:
            return None

        indices = np.concatenate(self._omega_indices, axis=0)
        references = np.concatenate(self._omega_references)
        sigmas = np.concatenate(self._omega_sigmas)
        periods = np.concatenate(self._omega_periods)
        is_proline = np.array(self._omega_is_proline, dtype=bool)

        if sort_indices and len(indices) > 0:
            sort_order = np.argsort(indices[:, 0])
            indices = indices[sort_order]
            references = references[sort_order]
            sigmas = sigmas[sort_order]
            periods = periods[sort_order]
            is_proline = is_proline[sort_order]

        return {
            "indices": torch.tensor(indices, dtype=torch.long, device=device),
            "references": torch.tensor(references, dtype=get_float_dtype(), device=device),
            "sigmas": torch.tensor(sigmas, dtype=get_float_dtype(), device=device),
            "periods": torch.tensor(periods, dtype=torch.long, device=device),
            "is_proline": torch.tensor(is_proline, dtype=torch.bool, device=device),
        }

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


class InterResiduePlaneBuilder:
    """
    Builder for inter-residue plane restraints (peptide planes).

    Handles planes that span consecutive residues (e.g., the planar
    peptide bond group).

    Parameters
    ----------
    verbose : int, default 0
        Verbosity level for debug output.
    """

    def __init__(self, verbose: int = 0):
        """Initialize with planes-by-count dictionary."""
        self.verbose = verbose
        self._planes_by_count: Dict[int, Dict[str, List]] = {}
        self._count: int = 0

    def reset(self):
        """Clear all accumulated data."""
        self._planes_by_count.clear()
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

    def process_peptide_planes(
        self,
        residue_i: pd.DataFrame,
        residue_next: pd.DataFrame,
        link_planes: pd.DataFrame,
    ) -> int:
        """
        Process peptide plane restraints between consecutive residues.

        Parameters
        ----------
        residue_i : pd.DataFrame
            First residue.
        residue_next : pd.DataFrame
            Second residue.
        link_planes : pd.DataFrame
            Plane definitions from link dictionary with 'plane_id',
            'atom_comp_id', 'atom', 'sigma' columns.

        Returns
        -------
        int
            Number of plane restraints added.
        """
        count = 0

        # Group by plane_id
        for plane_id in link_planes["plane_id"].unique():
            plane_atoms = link_planes[link_planes["plane_id"] == plane_id]

            # Collect atom indices for this plane
            atom_indices = []
            sigmas = []
            all_found = True

            for _, plane_atom_row in plane_atoms.iterrows():
                comp_id = plane_atom_row["atom_comp_id"]
                atom_name = plane_atom_row["atom"]
                sigma = float(plane_atom_row["sigma"])

                residue = residue_i if comp_id == "1" else residue_next
                atom_idx = self._get_atom_index(residue, atom_name)

                if atom_idx is None:
                    all_found = False
                    break

                atom_indices.append(atom_idx)
                sigmas.append(sigma)

            if not all_found or len(atom_indices) < 3:
                continue

            atom_count = len(atom_indices)
            if atom_count not in self._planes_by_count:
                self._planes_by_count[atom_count] = {"indices": [], "sigmas": []}

            self._planes_by_count[atom_count]["indices"].append(
                np.array(atom_indices, dtype=np.int64)
            )
            self._planes_by_count[atom_count]["sigmas"].append(
                np.array(sigmas, dtype=np.float32)
            )
            count += 1

        self._count += count
        return count

    def finalize(
        self, device: torch.device, sort_indices: bool = True, min_sigma: float = 1e-4
    ) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        """Convert accumulated data to tensors grouped by atom count."""
        if not self._planes_by_count:
            return None

        result = {}
        for atom_count, data in self._planes_by_count.items():
            if not data["indices"]:
                continue

            indices = np.stack(data["indices"], axis=0)
            sigmas = np.stack(data["sigmas"], axis=0)

            if sort_indices and len(indices) > 0:
                sort_order = np.argsort(indices[:, 0])
                indices = indices[sort_order]
                sigmas = sigmas[sort_order]

            sigmas = np.where(sigmas == 0, min_sigma, sigmas)

            key = f"{atom_count}_atoms"
            result[key] = {
                "indices": torch.tensor(indices, dtype=torch.long, device=device),
                "sigmas": torch.tensor(sigmas, dtype=get_float_dtype(), device=device),
            }

        return result if result else None

    @property
    def count(self) -> int:
        """Return total number of restraints accumulated."""
        return self._count
