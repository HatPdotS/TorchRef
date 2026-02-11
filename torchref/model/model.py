"""
A base model class for atomic structure models using PyTorch.

Space groups are stored as gemmi.SpaceGroup objects for consistency
and direct access to symmetry operations.

Variable naming conventions:
- adp: Atomic displacement parameters (model-level, replaces b_factor)
- xyz: Cartesian coordinates
- xyz_fractional: Fractional coordinates
- F_calc/F_obs: Structure factor amplitudes (uppercase = amplitudes)
- f_calc/f_obs: Complex structure factors (lowercase = complex)
"""

from typing import Optional, Union

import gemmi
import torch
import torch.nn as nn

import torchref.base.get_scattering_factor_torch as gsf
import torchref.base.math_numpy as mnp
from torchref.io import cif, pdb
from torchref.base import math_torch
from torchref.model.parameter_wrappers import (
    MixedTensor,
    OccupancyTensor,
    PositiveMixedTensor,
)
from torchref.symmetry import Cell, SpaceGroup
from torchref.utils.debug_utils import DebugMixin
from torchref.utils.device_mixin import DeviceMovementMixin
from torchref.utils.utils import sanitize_pdb_dataframe


class Model(DeviceMovementMixin, DebugMixin, nn.Module):
    """
    Base model class for atomic structure models using PyTorch.

    This class provides the foundation for managing atomic structure data
    including coordinates, atomic displacement parameters (ADPs),
    and occupancies. It supports both empty initialization for state_dict
    loading and file-based initialization from PDB/CIF files.

    Parameters
    ----------
    dtype_float : torch.dtype, optional
        Data type for floating point tensors. Default is torch.float32.
    verbose : int, optional
        Verbosity level for logging. Default is 1.
    device : torch.device, optional
        Computation device. Default is torch.device('cpu').
    strip_H : bool, optional
        Whether to strip hydrogen atoms when loading. Default is True.

    Attributes
    ----------
    xyz : MixedTensor
        Atomic coordinates tensor with shape (n_atoms, 3).
    adp : PositiveMixedTensor
        Atomic displacement parameters (isotropic B-factors) with shape (n_atoms,).
    u : MixedTensor
        Anisotropic displacement parameters with shape (n_atoms, 6).
    occupancy : OccupancyTensor
        Atomic occupancies with values in [0, 1].
    pdb : pandas.DataFrame
        DataFrame containing atomic model data.
    cell : Cell
        Unit cell object with parameters [a, b, c, alpha, beta, gamma].
    spacegroup : gemmi.SpaceGroup
        Space group object.
    symmetry : Symmetry
        Symmetry operations handler for this space group.
    initialized : bool
        Whether the model has been initialized with data.

    Examples
    --------
    Empty initialization for state_dict loading::

        model = Model()
        model.load_state_dict(torch.load('model.pt'))

    File-based initialization::

        model = Model()
        model.load_pdb('structure.pdb')
    """

    def __init__(
        self,
        dtype_float=torch.float32,
        verbose=1,
        device=torch.device("cpu"),
        strip_H: bool = True,
    ):
        """
        Initialize an empty Model shell.

        Creates a model shell ready for file loading via load_pdb()/load_cif()
        or state restoration via load_state_dict().

        Parameters
        ----------
        dtype_float : torch.dtype, optional
            Data type for floating point tensors. Default is torch.float32.
        verbose : int, optional
            Verbosity level for logging. Default is 1.
        device : torch.device, optional
            Computation device. Default is torch.device('cpu').
        strip_H : bool, optional
            Whether to strip hydrogen atoms when loading. Default is True.
        """
        super().__init__()
        # Configuration
        self.dtype_float = dtype_float
        self.verbose = verbose
        self.device = device
        self.strip_H = strip_H

        # State tracking
        self.initialized = False
        self.altloc_pairs = []

        # These will be set during load() or load_state_dict()
        self.pdb = None
        self._cell: Optional[Cell] = None
        self._spacegroup: Optional[SpaceGroup] = None

        # Submodules (created during load or load_state_dict)
        self.xyz = None
        self.adp = None
        self.u = None
        self.occupancy = None

        # Scattering factor parametrization (built lazily on first access)
        self._parametrization = None

        # Restraints (built lazily on first access)
        self._restraints = None
        self._cif_path = None

    def __bool__(self):
        """Return the initialization status when used in boolean context."""
        return self.initialized

    # =========================================================================
    # Cell, SpaceGroup, and Symmetry properties
    # =========================================================================

    @property
    def cell(self) -> Optional[Cell]:
        """
        Unit cell object with parameters [a, b, c, alpha, beta, gamma].

        Returns
        -------
        Cell or None
            The unit cell object, or None if not set.
        """
        return self._cell

    @cell.setter
    def cell(self, value: Cell):
        """
        Set the unit cell.

        Parameters
        ----------
        value : Cell
            The unit cell object to set.
        """
        self._cell = value

    @property
    def spacegroup(self) -> Optional[gemmi.SpaceGroup]:
        """
        Space group object.

        Returns
        -------
        gemmi.SpaceGroup or None
            The space group object, or None if not set.
        """
        return self._spacegroup

    @spacegroup.setter
    def spacegroup(self, value):
        """
        Set the space group and update the symmetry object.

        Parameters
        ----------
        value : gemmi.SpaceGroup or str or int
            The space group to set. Can be a gemmi.SpaceGroup object,
            a space group name string, or a space group number.
        """
        if value is not None:
            self._spacegroup = SpaceGroup(value)
        else:
            self._spacegroup = None

    @property
    def symmetry(self) -> Optional[SpaceGroup]:
        """
        Symmetry operations handler for this space group.

        Returns the same SpaceGroup object as `self.spacegroup` — the separate
        Symmetry wrapper was redundant since Symmetry is just an alias.

        Returns
        -------
        SpaceGroup or None
            The space group object, or None if not set.
        """
        return self._spacegroup

    @symmetry.setter
    def symmetry(self, value: Optional[SpaceGroup]):
        """
        Set the symmetry / space group object directly.

        Parameters
        ----------
        value : SpaceGroup or None
            The space group object to set.
        """
        self._spacegroup = value

    # =========================================================================
    # Crystallographic matrix properties (delegated to Cell)
    # =========================================================================

    @property
    def inv_fractional_matrix(self) -> torch.Tensor:
        """
        Fractionalization matrix B^-1 (Cartesian -> fractional).

        Delegates to Cell for automatic caching and device/dtype handling.

        Returns
        -------
        torch.Tensor
            Shape (3, 3) fractionalization matrix.
        """
        return self.cell.inv_fractional_matrix.to(dtype=self.dtype_float)

    @property
    def fractional_matrix(self) -> torch.Tensor:
        """
        Orthogonalization matrix B (fractional -> Cartesian).

        Delegates to Cell for automatic caching and device/dtype handling.

        Returns
        -------
        torch.Tensor
            Shape (3, 3) orthogonalization matrix.
        """
        return self.cell.fractional_matrix.to(dtype=self.dtype_float)

    @property
    def recB(self) -> torch.Tensor:
        """
        Reciprocal basis matrix with [a*, b*, c*] as rows.

        Delegates to Cell for automatic caching and device/dtype handling.

        Returns
        -------
        torch.Tensor
            Shape (3, 3) matrix where rows are the reciprocal basis vectors.
        """
        return self.cell.reciprocal_basis_matrix.to(dtype=self.dtype_float)

    # =========================================================================
    # Atomic Number (Z) Property
    # =========================================================================

    @property
    def Z(self) -> torch.Tensor:
        """
        Atomic numbers for all atoms.

        Returns
        -------
        torch.Tensor
            Tensor of atomic numbers with shape (n_atoms,).
        """
        return self._build_z_tensor()

    def _build_z_tensor(self) -> torch.Tensor:
        """
        Build atomic number tensor from element column.

        Converts element symbols to atomic numbers using the pre-loaded
        element-to-Z mapping from the scattering table.

        Returns
        -------
        torch.Tensor
            Tensor of atomic numbers with shape (n_atoms,).
        """
        if hasattr(self, "_Z") and self._Z is not None:
            return self._Z

        if not self.initialized or self.pdb is None:
            raise RuntimeError(
                "Cannot build Z tensor: model not initialized. "
                "Load data first with load_pdb() or load_cif()."
            )

        from torchref.base.scattering.scattering_table import get_element_to_z_mapping

        element_to_z = get_element_to_z_mapping()
        z_values = [
            element_to_z.get(elem.strip().capitalize(), 0)
            for elem in self.pdb["element"]
        ]
        self.register_buffer("_Z", torch.tensor(z_values, dtype=torch.int32))
        return self._Z

    # =========================================================================
    # Scattering Factor Parametrization
    # =========================================================================

    def get_P1_parameters_iso(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get model parameters transformed to P1 space for optimization.

        This is useful for optimizers that do not handle symmetry directly or MD.

        Returns
        -------
        xyz_p1 : torch.Tensor
            Fractional coordinates expanded to P1 space.
        adp_p1 : torch.Tensor
            Isotropic ADPs expanded to P1 space.
        occupancy_p1 : torch.Tensor
            Occupancies expanded to P1 space.
        A : torch.Tensor
            Scattering factor A coefficients expanded to P1 space.
        B : torch.Tensor
            Scattering factor B coefficients expanded to P1 space.
        """
        Nops = self.spacegroup.n_ops
        xyz_initial = self.xyz()
        xyz_fractional = self.cell.cartesian_to_fractional(xyz_initial)
        xyz_p1 = self.spacegroup.expand_coords_to_P1(xyz_fractional)
        adp_p1 = self.adp().expand(Nops, -1).reshape(-1)
        occupancy_p1 = self.occupancy().expand(Nops, -1).reshape(-1)
        A = self._A.expand(Nops, -1).reshape(-1, 5)
        B = self._B.expand(Nops, -1).reshape(-1, 5)
        return xyz_p1, adp_p1, occupancy_p1, A, B

    def get_MD_parameters(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get model parameters prepared for molecular dynamics simulation.

        Returns all P1-expanded parameters plus atomic numbers for MD engines.

        Returns
        -------
        xyz_p1 : torch.Tensor
            Fractional coordinates expanded to P1 space.
        adp_p1 : torch.Tensor
            Isotropic ADPs expanded to P1 space.
        occupancy_p1 : torch.Tensor
            Occupancies expanded to P1 space.
        A : torch.Tensor
            Scattering factor A coefficients expanded to P1 space.
        B : torch.Tensor
            Scattering factor B coefficients expanded to P1 space.
        Z_p1 : torch.Tensor
            Atomic numbers expanded to P1 space.
        """
        xyz_p1, adp_p1, occupancy_p1, A, B = self.get_P1_parameters_iso()
        Nops = self.spacegroup.n_ops
        Z_p1 = self.Z.expand(Nops, -1).reshape(-1)
        return xyz_p1, adp_p1, occupancy_p1, A, B, Z_p1

    def _build_parametrization(self):
        """
        Build ITC92 parametrization for all atoms in the model.

        Uses vectorized Z-based table lookup from pre-computed scattering
        factor table. Also builds a backward-compatible parametrization
        dictionary for legacy code. Registers the _A and _B parameter
        tensors as internal buffers.

        This method is called lazily on first access to `parametrization` or
        scattering parameters.

        Returns
        -------
        dict
            Parametrization dictionary {element: (A_tensor, B_tensor)}.
        """
        if self._parametrization is not None:
            return self._parametrization

        if not self.initialized or self.pdb is None:
            raise RuntimeError(
                "Cannot build parametrization: model not initialized. "
                "Load data first with load_pdb() or load_cif()."
            )

        if self.verbose > 1:
            print("Building ITC92 parametrization via table lookup...")

        # Use Z-based vectorized lookup
        from torchref.base.scattering.scattering_table import get_scattering_params_by_z

        z_tensor = self.Z
        A, B = get_scattering_params_by_z(
            z_tensor, device=self.device, dtype=self.dtype_float
        )

        self.register_buffer("_A", A)
        self.register_buffer("_B", B)

        # Build backward-compatible parametrization dict
        # Group by element to create {element: (A, B)} mapping
        elements = self.pdb.element.tolist()
        unique_elements = list(set(elements))
        self._parametrization = {}

        for elem in unique_elements:
            # Find first occurrence of this element
            idx = elements.index(elem)
            self._parametrization[elem] = (
                A[idx : idx + 1],  # Keep shape (1, 5)
                B[idx : idx + 1],
            )

        if self.verbose > 0:
            print(
                f"Parametrization built for {len(self._parametrization)} unique atom types"
            )
        if self.verbose > 1:
            print("Elements with parametrization:", list(self._parametrization.keys()))

        return self._parametrization

    @property
    def parametrization(self):
        """
        ITC92 parametrization dictionary {element: (A, B)}.

        The parametrization is built lazily on first access.

        Returns
        -------
        dict
            Dictionary mapping element symbols to tuples of (A, B) tensors.
        """
        return self._build_parametrization()

    def get_scattering_params_iso(self):
        """
        Get ITC92 scattering parameters (A, B) for isotropic atoms.

        Returns
        -------
        A : torch.Tensor
            ITC92 A parameters (amplitudes) with shape (n_iso_atoms, 5).
        B : torch.Tensor
            ITC92 B parameters (widths) with shape (n_iso_atoms, 5).
        """
        self._build_parametrization()
        mask = ~self.aniso_flag
        return self._A[mask], self._B[mask]

    def get_scattering_params_aniso(self):
        """
        Get ITC92 scattering parameters (A, B) for anisotropic atoms.

        Returns
        -------
        A : torch.Tensor
            ITC92 A parameters (amplitudes) with shape (n_aniso_atoms, 5).
        B : torch.Tensor
            ITC92 B parameters (widths) with shape (n_aniso_atoms, 5).
        """
        self._build_parametrization()
        mask = self.aniso_flag
        return self._A[mask], self._B[mask]

    # =========================================================================
    # Restraints (Geometry Restraints)
    # =========================================================================

    def set_restraints_cif(self, cif_path):
        """
        Set CIF path for lazy restraint building.

        Parameters
        ----------
        cif_path : str or list of str
            Path(s) to CIF restraints dictionary file(s).
        return self
            For method chaining
        """
        self._cif_path = cif_path
        # Reset restraints so they will be rebuilt on next access
        self._restraints = None
        return self

    def _build_restraints(self):
        """
        Build restraints lazily on first access.

        This method creates RestraintsNew with the model's pdb DataFrame
        and callables for xyz, adp, and vdw_radii.

        Returns
        -------
        RestraintsNew
            The restraints object.
        """
        if self._restraints is not None:
            return self._restraints

        if not self.initialized:
            raise RuntimeError(
                "Cannot build restraints: model not initialized. "
                "Load data first with load_pdb() or load_cif()."
            )

        from torchref.restraints.restraints import RestraintsNew

        if self.verbose > 0:
            print("Building restraints...")

        self._restraints = RestraintsNew(
            pdb=self.pdb,
            cif_path=self._cif_path,
            xyz_fn=self.xyz,
            adp_fn=self.adp,
            vdw_radii_fn=self.get_vdw_radii,
            verbose=self.verbose,
        )

        return self._restraints

    @property
    def restraints(self):
        """
        Lazy restraints property.

        The restraints are built on first access using the model's pdb DataFrame
        and the CIF path set via set_restraints_cif().

        Returns
        -------
        RestraintsNew
            The restraints object containing bond, angle, torsion, etc. restraints.
        """
        return self._build_restraints()

    # =========================================================================
    # Restraint Evaluation Wrappers
    # =========================================================================

    def bond_deviations(self):
        """
        Compute bond length deviations using current xyz coordinates.

        Returns
        -------
        deviations : torch.Tensor
            Calculated minus expected bond lengths in Angstroms.
        sigmas : torch.Tensor
            Standard deviations from CIF library in Angstroms.
        """
        return self.restraints.bond_deviations(self.xyz())

    def angle_deviations(self):
        """
        Compute angle deviations using current xyz coordinates.

        Returns
        -------
        deviations : torch.Tensor
            Calculated minus expected angles in radians.
        sigmas : torch.Tensor
            Standard deviations in radians.
        """
        return self.restraints.angle_deviations(self.xyz())

    def torsion_deviations_with_sigmas(self):
        """
        Compute torsion deviations (wrapped for periodicity) and sigmas.

        Returns
        -------
        deviations_rad : torch.Tensor
            Wrapped deviations in radians.
        sigmas_deg : torch.Tensor
            Standard deviations in degrees (for von Mises NLL).
        """
        return self.restraints.torsion_deviations_with_sigmas(self.xyz())

    def load(self, reader):
        self.pdb, cell, spacegroup = reader()

        self.pdb = (
            self.pdb.loc[self.pdb["element"] != "H"].reset_index(drop=True)
            if self.strip_H
            else self.pdb
        )
        self.pdb.dropna(subset=["x", "y", "z", "tempfactor", "occupancy"], inplace=True)
        self.pdb["index"] = self.pdb.index.to_numpy(dtype=int)

        # Store Cell object directly and use its cached derived quantities
        self.cell = Cell(cell, dtype=self.dtype_float, device=self.device)

        # Store space group - setter also updates symmetry automatically
        self.spacegroup = spacegroup

        # Register aniso_flag buffer (crystallographic matrices are delegated to Cell)
        self.register_buffer(
            "aniso_flag", torch.tensor(self.pdb["anisou_flag"].values, dtype=torch.bool)
        )

        # Create MixedTensors for model parameters
        self.xyz = MixedTensor(
            torch.tensor(self.pdb[["x", "y", "z"]].values, dtype=self.dtype_float),
            name="xyz",
            device=self.device,
        )
        self.adp = PositiveMixedTensor(
            torch.tensor(self.pdb["tempfactor"].values, dtype=self.dtype_float),
            name="adp",
            device=self.device,
        )
        self.u = MixedTensor(
            torch.tensor(
                self.pdb[["u11", "u22", "u33", "u12", "u13", "u23"]].values,
                dtype=self.dtype_float,
            ),
            name="aniso_U",
            device=self.device,
        )

        # Create OccupancyTensor with residue-level sharing and altloc support
        initial_occ = torch.tensor(self.pdb["occupancy"].values, dtype=self.dtype_float)
        sharing_groups, altloc_groups, refinable_mask = self._create_occupancy_groups(
            self.pdb, initial_occ
        )
        self.occupancy = OccupancyTensor(
            initial_values=initial_occ,
            sharing_groups=sharing_groups,
            altloc_groups=altloc_groups,
            refinable_mask=refinable_mask,
            dtype=self.dtype_float,
            device=self.device,
            name="occupancy",
        )

        self.set_default_masks()
        self.register_alternative_conformations()
        self.initialized = True
        return self

    def load_pdb(self, file):
        """
        Load atomic model from PDB file.

        Parameters
        ----------
        file : str
            Path to PDB file.

        Returns
        -------
        Model
            Self, for method chaining.
        """
        reader = pdb.PDBReader(verbose=self.verbose).read(file)
        return self.load(reader)

    def load_cif(self, file):
        """
        Load atomic model from mmCIF file.

        Parameters
        ----------
        file : str
            Path to CIF/mmCIF file.

        Returns
        -------
        Model
            Self, for method chaining.
        """
        if self.verbose > 0:
            print(f"Loading CIF file: {file}")

        # Read CIF file
        cif_reader = cif.ModelCIFReader(file)

        return self.load(cif_reader)

    def _create_occupancy_groups(self, pdb_df, initial_occ):
        """
        Create sharing groups and altloc groups for occupancy.

        This method identifies atoms that should share occupancy values and
        groups alternative conformations for proper constraint handling.

        Logic:
        1. First identify alternative conformations (multiple altlocs per residue)
        2. For altloc groups: ALL atoms in each conformation share one collapsed index
        3. For non-altloc residues: group by similar occupancy (within 0.01 tolerance)
        4. Only refine occupancies that differ from 1.0

        Parameters
        ----------
        pdb_df : pandas.DataFrame
            PDB DataFrame with atom information.
        initial_occ : torch.Tensor
            Tensor of initial occupancy values with shape (n_atoms,).

        Returns
        -------
        sharing_groups_tensor : torch.Tensor
            Tensor of shape (n_atoms,) where each value is the collapsed index
            for that atom.
        altloc_groups : list of tuple
            List of tuples of atom index lists for alternative conformations.
        refinable_mask : torch.Tensor
            Boolean tensor indicating which atoms should be refined.
        """
        n_atoms = len(initial_occ)
        altloc_groups = []
        refinable_mask = torch.zeros(n_atoms, dtype=torch.bool)

        # Initialize sharing groups tensor - each atom maps to its own index initially
        sharing_groups_tensor = torch.arange(n_atoms, dtype=torch.long)
        collapsed_idx = 0

        # First pass: identify and process alternative conformations
        # For altloc atoms: ALL atoms in a conformation MUST share the same collapsed index
        # regardless of their individual occupancy values
        pdb_with_altlocs = pdb_df[pdb_df["altloc"] != ""]
        altloc_residues = set()  # Track which residues have altlocs

        if len(pdb_with_altlocs) > 0:
            grouped_by_residue = pdb_with_altlocs.groupby(
                ["resname", "resseq", "chainid"]
            )

            for (resname, resseq, chainid), group in grouped_by_residue:
                unique_altlocs = sorted(group["altloc"].unique())

                # Only process if there are multiple conformations
                if len(unique_altlocs) > 1:
                    altloc_residues.add((resname, resseq, chainid))
                    conformation_atom_lists = []

                    for altloc in unique_altlocs:
                        # Get all atoms for this specific altloc
                        altloc_atoms = group[group["altloc"] == altloc]
                        indices = altloc_atoms["index"].tolist()

                        # Assign ALL atoms in this conformation to the same collapsed index
                        sharing_groups_tensor[indices] = collapsed_idx

                        # Check if any atom in this conformation has occupancy != 1.0
                        for idx in indices:
                            if abs(initial_occ[idx].item() - 1.0) > 0.01:
                                refinable_mask[idx] = True

                        conformation_atom_lists.append(indices)
                        collapsed_idx += 1

                    # Add to altloc_groups
                    altloc_groups.append(tuple(conformation_atom_lists))

        # Second pass: process non-altloc residues
        # Group by residue, and create sharing groups based on occupancy similarity
        grouped = pdb_df.groupby(["resname", "resseq", "chainid", "altloc"])

        for (resname, resseq, chainid, altloc), group in grouped:
            # Skip if this residue has alternative conformations (already processed)
            if (resname, resseq, chainid) in altloc_residues:
                continue

            indices = group["index"].tolist()

            if len(indices) == 0:
                continue

            # Get occupancies for this residue
            residue_occs = initial_occ[indices]

            # Check if all occupancies are within tolerance
            occ_min = residue_occs.min().item()
            occ_max = residue_occs.max().item()
            occ_mean = residue_occs.mean().item()

            if (occ_max - occ_min) <= 0.01:
                # All atoms in residue have similar occupancy - create sharing group
                sharing_groups_tensor[indices] = collapsed_idx
                collapsed_idx += 1

                # Only refine if mean occupancy differs from 1.0
                if abs(occ_mean - 1.0) > 0.01:
                    for idx in indices:
                        refinable_mask[idx] = True
            else:
                # Occupancies differ within residue - each atom independent
                # Refine those that differ from 1.0
                for idx in indices:
                    if abs(initial_occ[idx].item() - 1.0) > 0.01:
                        refinable_mask[idx] = True

        # Compact the indices - make them contiguous from 0 to n_collapsed-1
        unique_indices = torch.unique(sharing_groups_tensor, sorted=True)
        index_map = torch.zeros(n_atoms, dtype=torch.long)
        for new_idx, old_idx in enumerate(unique_indices):
            mask = sharing_groups_tensor == old_idx
            sharing_groups_tensor[mask] = new_idx

        n_collapsed = len(unique_indices)

        if self.verbose > 1:
            n_groups = n_collapsed
            n_independent = n_atoms - n_collapsed  # Atoms not sharing with others
            n_refinable = refinable_mask.sum().item()
            n_altloc_groups = len(altloc_groups)

            print("\nOccupancy Setup:")
            print(f"  Total atoms: {n_atoms}")
            print(f"  Collapsed indices: {n_collapsed}")
            print(f"  Alternative conformation groups: {n_altloc_groups}")
            print(f"  Refinable atoms: {n_refinable}")
            print(f"  Compression ratio: {n_atoms / n_collapsed:.2f}x")

        return sharing_groups_tensor, altloc_groups, refinable_mask

    def update_pdb(self):
        self.pdb.loc[:, ["x", "y", "z"]] = self.xyz().cpu().detach().numpy()
        self.pdb.loc[:, ["u11", "u22", "u33", "u12", "u13", "u23"]] = (
            self.u().cpu().detach().numpy()
        )
        self.pdb.loc[:, "tempfactor"] = self.adp().cpu().detach().numpy()
        self.pdb.loc[:, "occupancy"] = self.occupancy().cpu().detach().numpy()
        return self.pdb

    def get_vdw_radii(self):
        """
        Get van der Waals radii for all atoms based on their elements.

        Caches the result in self.vdw_radii for future calls.

        Returns
        -------
        torch.Tensor
            Van der Waals radii for each atom with shape (n_atoms,).
        """
        import os
        from torchref import PATH_TORCHREF_DATA
        import pandas as pd

        if hasattr(self, "vdw_radii"):
            return self.vdw_radii
        elements = self.pdb.loc[:, "element"]
        path = os.path.join(
            PATH_TORCHREF_DATA,
            "atomic_vdw_radii.csv",
        )
        vdw_df = pd.read_csv(path, comment="#")
        vdw_df["element"] = vdw_df["element"].str.strip().str.capitalize()
        elements = elements.str.strip().str.capitalize()
        elements_not_in = elements[~elements.isin(vdw_df["element"])]
        if len(elements_not_in) > 0:
            # Add missing elements with default vdW radius 1.9 Å
            missing = sorted(set(e.strip().capitalize() for e in elements_not_in))
            if missing:
                add_df = pd.DataFrame(
                    {"element": missing, "vdW_Radius_Angstrom": [1.9] * len(missing)}
                )
                vdw_df = pd.concat([vdw_df, add_df], ignore_index=True)

        vdw_radii = (
            vdw_df.set_index("element").loc[elements]["vdW_Radius_Angstrom"].values
        )
        self.register_buffer(
            "vdw_radii",
            torch.tensor(vdw_radii, dtype=self.dtype_float, device=self.device),
        )
        assert len(self.vdw_radii) == len(
            self.pdb
        ), f"vdW radii length mismatch with number of atoms {len(self.vdw_radii)} != {len(self.pdb)}"
        return self.vdw_radii

    def to(self, device=None, dtype=None):
        """
        Move Model to specified device and/or dtype.

        Parameters
        ----------
        device : torch.device or str, optional
            Target device.
        dtype : torch.dtype, optional
            Target data type.

        Returns
        -------
        Model
            Self, for method chaining.
        """
        # Move Cell object
        if self.cell is not None:
            self.cell = self.cell.to(device=device, dtype=dtype)

        # Move altloc_pairs tensors
        if self.altloc_pairs:
            self.altloc_pairs = [
                tuple(tensor.to(device=device) for tensor in group)
                for group in self.altloc_pairs
            ]

        # Update device tracking
        if device is not None:
            self.device = torch.device(device)

        # Move restraints if they exist (not a registered submodule, so move explicitly)
        if self._restraints is not None:
            self._restraints.to(device=device, dtype=dtype)

        # Call parent to move all registered buffers and parameters
        result = super().to(device=device, dtype=dtype)
        if self.verbose > 0:
            print(f"Model moved to device: {self.device}")
        return result

    def copy(self):
        """
        Create a deep copy of the Model.

        Creates a complete independent copy including all registered buffers,
        module parameters, PDB DataFrame, and spacegroup information.

        Returns
        -------
        Model
            A new Model instance with copied data.

        Examples
        --------
        ::

            model = Model().load_pdb('structure.pdb')
            model_copy = model.copy()
            # model_copy is independent, changes won't affect model
        """
        if not self.initialized:
            raise RuntimeError("Cannot copy an uninitialized Model. Load data first.")

        # Create new model instance with same configuration
        model_copy = Model(
            dtype_float=self.dtype_float,
            verbose=self.verbose,
            device=self.device,
            strip_H=self.strip_H,
        )

        # Deep copy the PDB DataFrame
        model_copy.pdb = self.pdb.copy(deep=True)

        # Copy scalar attributes - spacegroup setter also sets symmetry
        model_copy.spacegroup = self.spacegroup  # gemmi.SpaceGroup is immutable
        model_copy.initialized = True

        # Copy Cell object
        if self.cell is not None:
            model_copy.cell = self.cell.clone()

        # Copy all registered buffers using PyTorch's _buffers dict
        for buffer_name, buffer_value in self._buffers.items():
            if buffer_value is not None:
                model_copy.register_buffer(buffer_name, buffer_value.clone())

        # Copy all modules (parameter wrappers) using their .copy() methods
        for module_name, module in self._modules.items():
            if module is not None and hasattr(module, "copy"):
                setattr(model_copy, module_name, module.copy())

        # Copy alternative conformation pairs
        if hasattr(self, "altloc_pairs") and self.altloc_pairs:
            model_copy.altloc_pairs = [
                tuple(tensor.clone() for tensor in group) for group in self.altloc_pairs
            ]
        else:
            model_copy.altloc_pairs = []

        if self.verbose > 0:
            print(f"✓ Model copied successfully ({len(model_copy.pdb)} atoms)")

        return model_copy

    def write_pdb(self, filename):
        self.update_pdb()
        self.pdb = sanitize_pdb_dataframe(self.pdb)
        self.pdb.attrs["spacegroup"] = self.spacegroup.hm if self.spacegroup else "P 1"
        pdb.write(self.pdb, filename)

    def get_iso(self):
        xyz = self.xyz()[~self.aniso_flag]
        adp = self.adp()[~self.aniso_flag]
        occupancy = self.occupancy()[~self.aniso_flag]
        return xyz, adp, occupancy

    def set_default_masks(self):
        self.register_buffer(
            "xyz_mask", torch.ones(len(self.pdb), dtype=torch.bool, device=self.device)
        )
        self.xyz.update_refinable_mask(self.xyz_mask)
        self.register_buffer("adp_mask", ~self.adp().detach().isnan())
        self.adp.update_refinable_mask(self.adp_mask)
        self.register_buffer("u_mask", ~self.u().detach().isnan().any(dim=1))
        self.u.update_refinable_mask(self.u_mask)
        self.register_buffer("occupancy_mask", self.occupancy() < 0.999)
        self.occupancy.update_refinable_mask(self.occupancy_mask)

    def freeze(self, target: str):
        if target == "xyz":
            self.xyz.fix_all()
        elif target == "adp":
            self.adp.fix_all()
        elif target == "u":
            self.u.fix_all()
        elif target == "occupancy":
            self.occupancy.freeze_all()  # OccupancyTensor uses freeze_all() not fix_all()

    def freeze_all(self):
        self.freeze("xyz")
        self.freeze("adp")
        self.freeze("u")
        self.freeze("occupancy")

    def unfreeze_all(self):
        self.unfreeze("xyz")
        self.unfreeze("adp")
        self.unfreeze("u")
        self.unfreeze("occupancy")

    def unfreeze(self, target: str):
        if target == "xyz":
            self.xyz.update_refinable_mask(self.xyz_mask)
        elif target == "adp":
            self.adp.update_refinable_mask(self.adp_mask)
        elif target == "u":
            self.u.update_refinable_mask(self.u_mask)
        elif target == "occupancy":
            # OccupancyTensor uses unfreeze_all() or update_refinable_mask() with full atom space mask
            self.occupancy.update_refinable_mask(
                self.occupancy_mask, in_compressed_space=False
            )

    def update_mask_from_selection(
        self, selection_string: str, target: str, mode: str = "set", freeze: bool = True
    ):
        """
        Update the refinable mask for a parameter using Phenix-style selection syntax.

        This method updates the internal mask buffer (xyz_mask, adp_mask, u_mask, or
        occupancy_mask) based on the selection. The updated mask is NOT automatically
        applied to the parameter tensors - use apply_mask_to_parameter() to apply it.

        Parameters
        ----------
        selection_string : str
            Phenix-style selection string (see parse_phenix_selection docs).
        target : str
            Parameter to update: 'xyz', 'adp', 'u', or 'occupancy'.
        mode : str, optional
            How to combine with current mask:
            - 'set': Replace mask with selection (default)
            - 'add': Add selection to current mask
            - 'remove': Remove selection from current mask
        freeze : bool, optional
            If True (default), selected atoms will be frozen (mask=False).
            If False, selected atoms will be unfrozen (mask=True).

        Raises
        ------
        ValueError
            If target is not recognized or selection syntax is invalid.

        Examples
        --------
        ::

            # Freeze chain A coordinates
            model.update_mask_from_selection("chain A", "xyz", mode='set', freeze=True)
            model.apply_mask_to_parameter("xyz")

            # Unfreeze backbone atoms
            model.update_mask_from_selection("name CA or name C or name N", "xyz", freeze=False)
            model.apply_mask_to_parameter("xyz")
        """
        from torchref.utils.utils import create_selection_mask

        # Map target to the corresponding mask buffer
        mask_map = {
            "xyz": "xyz_mask",
            "adp": "adp_mask",
            "u": "u_mask",
            "occupancy": "occupancy_mask",
        }

        if target not in mask_map:
            raise ValueError(
                f"Invalid target: '{target}'. Must be one of: {list(mask_map.keys())}"
            )

        mask_name = mask_map[target]
        current_mask = getattr(self, mask_name)

        # Get selection mask
        selection_mask = create_selection_mask(
            selection_string,
            self.pdb,
            current_mask=current_mask if mode != "set" else None,
            mode=mode,
        )

        # Invert selection if we're freezing (refinable_mask=False means frozen)
        if freeze:
            updated_mask = current_mask & ~selection_mask
        else:
            updated_mask = selection_mask

        # Update the buffer
        setattr(self, mask_name, updated_mask)

        if self.verbose > 0:
            n_selected = selection_mask.sum().item()
            n_refinable = updated_mask.sum().item()
            action = "frozen" if freeze else "unfrozen"
            print(
                f"Selection '{selection_string}' ({n_selected} atoms) {action} for {target}"
            )
            print(
                f"  Total refinable atoms for {target}: {n_refinable}/{len(self.pdb)}"
            )

    def apply_mask_to_parameter(self, target: str):
        """
        Apply the current mask buffer to the parameter tensor.

        Takes the current state of the mask buffer (xyz_mask, adp_mask, etc.)
        and applies it to the corresponding parameter tensor's refinable mask.

        Parameters
        ----------
        target : str
            Parameter to update: 'xyz', 'adp', 'u', or 'occupancy'.

        Raises
        ------
        ValueError
            If target is not recognized.

        Examples
        --------
        ::

            model.update_mask_from_selection("chain A", "xyz", freeze=True)
            model.apply_mask_to_parameter("xyz")
        """
        if target == "xyz":
            self.xyz.update_refinable_mask(self.xyz_mask)
        elif target == "adp":
            self.adp.update_refinable_mask(self.adp_mask)
        elif target == "u":
            self.u.update_refinable_mask(self.u_mask)
        elif target == "occupancy":
            self.occupancy.update_refinable_mask(
                self.occupancy_mask, in_compressed_space=False
            )
        else:
            raise ValueError(
                f"Invalid target: '{target}'. Must be 'xyz', 'adp', 'u', or 'occupancy'"
            )

        if self.verbose > 0:
            n_refinable = getattr(self, f"{target}_mask").sum().item()
            print(f"  Applied mask to {target}: {n_refinable} atoms refinable")

    def freeze_selection(
        self, selection_string: str, targets: Union[str, list] = "all"
    ):
        """
        Freeze atoms matching a Phenix-style selection for specified parameters.

        Convenience method that combines update_mask_from_selection() and
        apply_mask_to_parameter() into a single call.

        Parameters
        ----------
        selection_string : str
            Phenix-style selection string.
        targets : str or list of str, optional
            Parameter(s) to freeze. Can be:
            - 'all': Freeze xyz, adp, u, and occupancy (default)
            - str: Single parameter ('xyz', 'adp', 'u', 'occupancy')
            - list: List of parameters, e.g., ['xyz', 'adp']

        Examples
        --------
        ::

            # Freeze all parameters for chain A
            model.freeze_selection("chain A", targets='all')

            # Freeze only coordinates for residues 10-20
            model.freeze_selection("resseq 10:20", targets='xyz')
        """
        # Handle 'all' target
        if targets == "all":
            targets = ["xyz", "adp", "u", "occupancy"]
        elif isinstance(targets, str):
            targets = [targets]

        # Update and apply masks for each target
        for target in targets:
            self.update_mask_from_selection(
                selection_string, target, mode="set", freeze=True
            )
            self.apply_mask_to_parameter(target)

    def unfreeze_selection(
        self, selection_string: str, targets: Union[str, list] = "all"
    ):
        """
        Unfreeze atoms matching a Phenix-style selection for specified parameters.

        Convenience method that combines update_mask_from_selection() and
        apply_mask_to_parameter() into a single call.

        Parameters
        ----------
        selection_string : str
            Phenix-style selection string.
        targets : str or list of str, optional
            Parameter(s) to unfreeze. Can be:
            - 'all': Unfreeze xyz, adp, u, and occupancy (default)
            - str: Single parameter ('xyz', 'adp', 'u', 'occupancy')
            - list: List of parameters, e.g., ['xyz', 'adp']

        Examples
        --------
        ::

            # Unfreeze all parameters for chain A
            model.unfreeze_selection("chain A", targets='all')

            # Unfreeze only coordinates for backbone atoms
            model.unfreeze_selection("name CA or name C or name N", targets='xyz')
        """
        # Handle 'all' target
        if targets == "all":
            targets = ["xyz", "adp", "u", "occupancy"]
        elif isinstance(targets, str):
            targets = [targets]

        # Update and apply masks for each target
        for target in targets:
            self.update_mask_from_selection(
                selection_string, target, mode="set", freeze=False
            )
            self.apply_mask_to_parameter(target)

    def get_aniso(self):
        xyz = self.xyz()[self.aniso_flag]
        u = self.u()[self.aniso_flag]
        occupancy = self.occupancy()[self.aniso_flag]
        return xyz, u, occupancy

    def parameters(self, recurse: bool = True):
        return (p for p in super().parameters(recurse) if p.numel() > 0)

    def named_mixed_tensors(self):
        """
        Iterate over all MixedTensor attributes with their names.

        Yields:
            Tuple of (name, MixedTensor)
        """
        for name, module in self.named_modules():
            if isinstance(module, MixedTensor) and module != self:
                yield name, module

    def print_parameters_info(self):
        """Print information about all MixedTensor parameters."""
        print("=" * 80)
        print("Model Parameters Summary")
        print("=" * 80)
        for attr_name, mixed_tensor in self.named_mixed_tensors():
            print(f"\n{attr_name}: {mixed_tensor}")
            if mixed_tensor.get_refinable_count() > 0:
                print(
                    f"  Refinable values: min={mixed_tensor.refinable_params.min().item():.4f}, "
                    f"max={mixed_tensor.refinable_params.max().item():.4f}, "
                    f"mean={mixed_tensor.refinable_params.mean().item():.4f}"
                )
        print("=" * 80)

    def register_alternative_conformations(self):
        """
        Identify and register all alternative conformation groups in the structure.

        For each residue that has alternative conformations (altloc A, B, C, etc.),
        this method identifies all atoms belonging to each conformation and stores
        their indices as tensors in a tuple.

        The result is stored in self.altloc_pairs as a list of tuples, where each
        tuple contains tensors of atom indices for each alternative conformation.

        Examples
        --------
        For a residue with conformations A and B::

            # Conformation A has atoms at indices [100, 101, 102, ...]
            # Conformation B has atoms at indices [110, 111, 112, ...]
            # Result: [(tensor([100, 101, 102, ...]), tensor([110, 111, 112, ...])), ...]

        For a residue with conformations A, B, C::

            # Result: [(tensor([200, 201, ...]), tensor([210, 211, ...]), tensor([220, 221, ...])), ...]
        """
        # Initialize the list to store alternative conformation groups
        self.altloc_pairs = []

        # Get all atoms with alternative conformations (non-empty altloc field)
        pdb_with_altlocs = self.pdb[self.pdb["altloc"] != ""]

        if len(pdb_with_altlocs) == 0:
            # No alternative conformations in this structure
            return

        # Group by residue (resname, resseq, chainid) to find all residues
        # that have alternative conformations
        grouped = pdb_with_altlocs.groupby(["resname", "resseq", "chainid"])

        for (resname, resseq, chainid), group in grouped:
            # Get all unique altloc identifiers for this residue
            unique_altlocs = sorted(group["altloc"].unique())

            # Only register if there are actually multiple conformations
            if len(unique_altlocs) > 1:
                # For each altloc, collect all atom indices belonging to that conformation
                conformation_tensors = []
                for altloc in unique_altlocs:
                    # Get all atoms for this specific altloc
                    altloc_atoms = group[group["altloc"] == altloc]
                    # Get their indices and convert to tensor
                    indices = torch.tensor(
                        altloc_atoms["index"].tolist(), dtype=torch.long
                    )
                    conformation_tensors.append(indices)

                # Store as a tuple of tensors
                self.altloc_pairs.append(tuple(conformation_tensors))

    def shake_coords(self, stddev: float):
        """
        Apply random Gaussian noise to atomic coordinates.

        Perturbs the atomic coordinates by adding Gaussian noise with a
        specified standard deviation. The noise is applied to all atoms.

        Parameters
        ----------
        stddev : float
            Standard deviation of the Gaussian noise to be added, in Angstroms.
        """
        xyz = self.xyz().detach()
        new_xyz = xyz + torch.normal(
            mean=0.0, std=stddev, size=xyz.shape, device=self.device
        )
        self.xyz = MixedTensor(
            new_xyz, refinable_mask=self.xyz.refinable_mask, name="xyz"
        )

    def shake_adp(self, stddev: float):
        """
        Apply random Gaussian noise to ADPs (atomic displacement parameters).

        Perturbs the ADPs by adding Gaussian noise with a specified
        standard deviation. The noise is applied to all atoms.

        Parameters
        ----------
        stddev : float
            Standard deviation of the Gaussian noise to be added, in Angstrom^2.
        """
        adp_values = self.adp().detach()
        new_adp = adp_values + torch.normal(
            mean=0.0, std=stddev, size=adp_values.shape, device=self.device
        )
        self.adp = PositiveMixedTensor(
            new_adp, refinable_mask=self.adp.refinable_mask, name="adp"
        )

    def adp_loss(self):
        """
        Compute the ADP regularization loss.

        This loss encourages ADPs to have similar values across the
        structure, helping to prevent overfitting during refinement.

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the ADP loss.
        """
        adp_current = self.adp()
        adp_mean = torch.mean(adp_current)
        loss = torch.mean((adp_current - adp_mean) ** 2)
        return loss

    def adp_nll_loss(self, target_log_std: float = 0.2):
        """
        Compute negative log-likelihood of ADPs assuming Gaussian distribution in log-space.

        This regularization penalizes ADPs that deviate from a target distribution
        with a FIXED standard deviation (hyperparameter), avoiding circular dependency
        on the current distribution's statistics.

        The NLL for a Gaussian distribution in log-space is::

            NLL = 0.5 * mean[(log_adp - mu)^2 / sigma^2 + log(2*pi*sigma^2)]

        Where mu is the mean of log-space ADPs (computed from current data) and
        sigma is the FIXED target standard deviation (hyperparameter).

        Parameters
        ----------
        target_log_std : float, optional
            Target standard deviation in log-space. Default is 0.2.
            - 0.1 = very tight (ADPs within ~10% of mean)
            - 0.2 = moderate spread (ADPs within ~20% of mean) [RECOMMENDED]
            - 0.3 = looser spread (ADPs within ~30% of mean)

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the NLL. Lower values indicate the distribution
            is closer to the target Gaussian with fixed sigma.

        Examples
        --------
        ::

            # During refinement
            structure_factor_loss = compute_structure_factor_loss()
            nll_reg = model.adp_nll_loss(target_log_std=0.2)
            total_loss = structure_factor_loss + 0.01 * nll_reg
            total_loss.backward()

        Notes
        -----
        Uses FIXED sigma (no circular dependency on current distribution).
        Smaller target_log_std = stronger regularization (tighter distribution).
        """
        # Access the internal log-space values directly from the PositiveMixedTensor
        # The parent MixedTensor.forward() returns log-space values before exp()
        log_adp = super(PositiveMixedTensor, self.adp).forward()

        # Compute mean in log-space (target center of distribution)
        mu = torch.mean(log_adp).detach()

        # Use FIXED target_log_std (not computed from data)
        sigma = target_log_std

        # Compute NLL for Gaussian distribution
        # NLL = 0.5 * [(log_adp - μ)² / σ² + log(2πσ²)]
        ln_2pi_sigma2 = torch.log(
            torch.tensor(
                2.0 * torch.pi * sigma**2, dtype=self.dtype_float, device=self.device
            )
        )

        squared_deviations = (log_adp - mu) ** 2
        nll_per_atom = 0.5 * (squared_deviations / (sigma**2) + ln_2pi_sigma2)

        # Return mean NLL across all atoms
        nll = torch.mean(nll_per_atom)

        return nll

    def adp_nll_loss_per_atom(self, target_log_std: float = 0.2):
        """
        Compute per-atom negative log-likelihood for ADPs in log-space.

        Returns the NLL contribution for each individual atom, useful for
        identifying outliers or applying atom-specific regularization weights.

        The per-atom NLL is::

            NLL_i = 0.5 * [(log_adp_i - mu)^2 / sigma^2 + log(2*pi*sigma^2)]

        Parameters
        ----------
        target_log_std : float, optional
            Fixed target standard deviation in log-space. Default is 0.2.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_atoms,) with per-atom NLL values.
            Higher values indicate atoms farther from the mean.

        Examples
        --------
        ::

            # Get per-atom NLL
            atom_nll = model.adp_nll_loss_per_atom(target_log_std=0.2)
            # Identify outlier atoms (high NLL)
            threshold = atom_nll.mean() + 2 * atom_nll.std()
            outliers = atom_nll > threshold
        """
        # Access the internal log-space values
        log_adp = super(PositiveMixedTensor, self.adp).forward()

        # Compute mean in log-space
        mu = torch.mean(log_adp)

        # Use FIXED target_log_std
        sigma = target_log_std

        # Compute per-atom NLL
        ln_2pi_sigma2 = torch.log(
            torch.tensor(
                2.0 * torch.pi * sigma**2, dtype=self.dtype_float, device=self.device
            )
        )

        squared_deviations = (log_adp - mu) ** 2
        nll_per_atom = 0.5 * (squared_deviations / (sigma**2) + ln_2pi_sigma2)

        return nll_per_atom

    def adp_kl_divergence_loss(self, target_log_std: float = 0.2):
        """
        Compute KL divergence between log ADP distribution and target Gaussian.

        Measures how different the current log ADP distribution is from a
        target Gaussian distribution with the current mean of log ADPs and
        a fixed target standard deviation.

        KL divergence formula for two Gaussians with same mean::

            KL(q || p) = log(sigma_target/sigma_data) + sigma_data^2 / (2*sigma_target^2) - 0.5

        Parameters
        ----------
        target_log_std : float, optional
            Target standard deviation in log-space. Default is 0.2.
            Controls how tightly ADPs should cluster.

        Returns
        -------
        torch.Tensor
            Scalar KL divergence value (always >= 0).
            0 means distributions match perfectly.
            Higher values mean more deviation from target.

        Examples
        --------
        ::

            # Use in loss function
            loss = xray_loss + w_adp * model.adp_kl_divergence_loss(0.2)

        Notes
        -----
        Lower target_log_std = stronger regularization (tighter distribution).
        Mean is detached so it adapts to the natural scale of the data.
        """

        # Access the internal log-space values
        log_adp = super(PositiveMixedTensor, self.adp).forward()

        # Compute statistics of actual distribution
        mu_data = torch.mean(log_adp).detach()  # Detached mean (adapts to data)
        sigma_data = torch.std(log_adp)  # Current std (to be regularized)

        # Target distribution parameters
        mu_target = mu_data  # Same mean as data
        sigma_target = target_log_std  # Fixed target std

        # KL divergence: KL(actual || target) for Gaussians with same mean
        # KL = log(σ_target/σ_data) + σ_data² / (2σ_target²) - 0.5
        log_sigma_ratio = torch.log(
            torch.tensor(sigma_target, dtype=self.dtype_float, device=self.device)
            / sigma_data
        )
        variance_ratio = (sigma_data**2) / (2 * sigma_target**2)

        kl_divergence = log_sigma_ratio + variance_ratio - 0.5

        return kl_divergence

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """
        Return a dictionary containing the complete state of the Model.

        Includes all registered buffers, model parameters (xyz, b, u, occupancy),
        PDB DataFrame, and metadata (spacegroup, device, dtype, etc.).

        Parameters
        ----------
        destination : dict, optional
            Optional dict to populate with state.
        prefix : str, optional
            Prefix for parameter names. Default is ''.
        keep_vars : bool, optional
            Whether to keep variables in computational graph. Default is False.

        Returns
        -------
        dict
            Complete state dictionary.
        """
        # Get parent class state_dict (includes all registered buffers)
        state = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )

        # Add model-specific state
        state[prefix + "pdb"] = (
            self.pdb.copy() if hasattr(self, "pdb") and self.pdb is not None else None
        )
        # Store Cell tensor data for serialization
        state[prefix + "cell"] = (
            self.cell.data.cpu() if self.cell is not None else None
        )
        # Store spacegroup as string for serialization (gemmi.SpaceGroup is not picklable)
        state[prefix + "spacegroup"] = (
            self.spacegroup.xhm if self.spacegroup else None
        )
        state[prefix + "initialized"] = self.initialized
        state[prefix + "dtype_float"] = self.dtype_float
        state[prefix + "device"] = self.device
        state[prefix + "strip_H"] = self.strip_H
        state[prefix + "altloc_pairs"] = (
            self.altloc_pairs if hasattr(self, "altloc_pairs") else []
        )

        return state

    def save_state(self, path: str):
        """
        Save the complete state of the model to a file.

        Parameters
        ----------
        path : str
            Path to save the state dictionary to.
        """
        torch.save(self.state_dict(), path)
        if self.verbose > 0:
            print(f"Saved model state to {path}")

    def load_state(self, path: str, strict: bool = True):
        """
        Load the complete state of the model from a file.

        Parameters
        ----------
        path : str
            Path to load the state dictionary from.
        strict : bool, optional
            Whether to strictly enforce that keys match. Default is True.
        """
        state_dict = torch.load(path, map_location=self.device, weights_only=False)
        loaded = type(self).create_from_state_dict(
            state_dict, device=self.device, verbose=self.verbose
        )
        # Copy loaded state to self
        self.__dict__.update(loaded.__dict__)
        if self.verbose > 0:
            print(f"Loaded model state from {path}")

    @classmethod
    def create_from_state_dict(
        cls,
        state_dict: dict,
        device: torch.device = torch.device("cpu"),
        verbose: int = 1,
        dtype_float: torch.dtype = torch.float32,
    ) -> "Model":
        """
        Create a fully initialized Model from a state dictionary.

        This is the recommended way to restore a Model from a saved state.
        Creates an instance with properly initialized submodules, then loads the state.

        Parameters
        ----------
        state_dict : dict
            State dictionary from torch.save(model.state_dict(), ...).
        device : torch.device, optional
            Device to place tensors on. Default is torch.device('cpu').
        verbose : int, optional
            Verbosity level. Default is 1.
        dtype_float : torch.dtype, optional
            Float dtype for tensors. Default is torch.float32.

        Returns
        -------
        Model
            Fully initialized instance with restored state.
        """
        # Extract metadata (non-tensor data that we handle specially)
        pdb = state_dict.pop("pdb", None)
        cell_tensor = state_dict.pop("cell", None)
        spacegroup = state_dict.pop("spacegroup", None)
        initialized = state_dict.pop("initialized", False)
        saved_dtype = state_dict.pop("dtype_float", dtype_float)
        saved_device = state_dict.pop("device", device)
        strip_H = state_dict.pop("strip_H", True)
        altloc_pairs = state_dict.pop("altloc_pairs", [])

        # Create instance
        instance = cls(
            dtype_float=saved_dtype, verbose=verbose, device=device, strip_H=strip_H
        )

        # Set metadata
        instance.pdb = pdb
        instance.initialized = initialized
        instance.altloc_pairs = altloc_pairs

        # Setup spacegroup (setter also sets symmetry automatically)
        instance.spacegroup = spacegroup

        # Create Cell object from saved tensor data
        if cell_tensor is not None:
            instance.cell = Cell(cell_tensor, dtype=saved_dtype, device=device)

        # If PDB exists, create the parameter wrappers with correct shapes
        if pdb is not None:
            n_atoms = len(pdb)

            # Create MixedTensors with initial values from PDB (will be overwritten by load_state_dict)
            # Get refinable masks from state_dict if available
            xyz_mask = state_dict.get("xyz.refinable_mask")
            adp_mask = state_dict.get("adp.refinable_mask")
            u_mask = state_dict.get("u.refinable_mask")

            instance.xyz = MixedTensor(
                torch.tensor(pdb[["x", "y", "z"]].values, dtype=saved_dtype),
                refinable_mask=xyz_mask,
                name="xyz",
            )
            instance.adp = PositiveMixedTensor(
                torch.tensor(pdb["tempfactor"].values, dtype=saved_dtype),
                refinable_mask=adp_mask,
                name="adp",
            )
            instance.u = MixedTensor(
                torch.tensor(
                    pdb[["u11", "u22", "u33", "u12", "u13", "u23"]].values,
                    dtype=saved_dtype,
                ),
                refinable_mask=u_mask,
                name="aniso_U",
            )

            # Create OccupancyTensor
            initial_occ = torch.tensor(pdb["occupancy"].values, dtype=saved_dtype)
            sharing_groups, altloc_groups, refinable_mask = (
                instance._create_occupancy_groups(pdb, initial_occ)
            )

            # Override mask if present in state_dict
            saved_occ_mask = state_dict.get("occupancy.refinable_mask")
            if saved_occ_mask is not None:
                if saved_occ_mask.device != sharing_groups.device:
                    saved_occ_mask = saved_occ_mask.to(sharing_groups.device)
                refinable_mask = saved_occ_mask[sharing_groups]

            instance.occupancy = OccupancyTensor(
                initial_values=initial_occ,
                sharing_groups=sharing_groups,
                altloc_groups=altloc_groups,
                refinable_mask=refinable_mask,
                dtype=saved_dtype,
                device=device,
                name="occupancy",
            )

            # Register buffers that are needed
            if "aniso_flag" not in instance._buffers or instance.aniso_flag is None:
                instance.register_buffer(
                    "aniso_flag",
                    torch.tensor(pdb["anisou_flag"].values, dtype=torch.bool),
                )

            # Register mask buffers
            instance.register_buffer(
                "xyz_mask", torch.ones(n_atoms, dtype=torch.bool, device=device)
            )
            instance.register_buffer(
                "adp_mask", torch.ones(n_atoms, dtype=torch.bool, device=device)
            )
            instance.register_buffer(
                "u_mask", torch.ones(n_atoms, dtype=torch.bool, device=device)
            )
            instance.register_buffer(
                "occupancy_mask", torch.ones(n_atoms, dtype=torch.bool, device=device)
            )

            # Register other buffers based on state_dict
            # Note: inv_fractional_matrix, fractional_matrix, recB are now properties
            # delegating to Cell, so they're not registered as buffers
            buffer_names = ["vdw_radii"]
            for name in buffer_names:
                if name in state_dict and state_dict[name] is not None:
                    instance.register_buffer(
                        name, torch.zeros_like(state_dict[name], device=device)
                    )

        # Now use PyTorch's default load_state_dict
        state_dict = {k: v for k, v in state_dict.items() if k.shape[0] > 0}
        instance.load_state_dict(state_dict, strict=False)

        if verbose > 0:
            n_atoms = len(instance.pdb) if instance.pdb is not None else 0
            print(f"Created Model from state_dict: {n_atoms} atoms")

        return instance

    def get_selection_mask(self, selection: str) -> torch.Tensor:
        """
        Return a boolean mask for atoms matching a Phenix-style selection.

        This is a convenience method that wraps parse_phenix_selection() to
        return a mask that can be used directly with MixedTensor.set() or
        other operations requiring atom selection.

        Parameters
        ----------
        selection : str
            Phenix-style selection string. Supports:
            - chain <id>: Select by chain (e.g., "chain A")
            - resseq <num>: Select by residue number (e.g., "resseq 10")
            - resseq <start>:<end>: Select residue range (e.g., "resseq 10:20")
            - resname <name>: Select by residue name (e.g., "resname ALA")
            - name <atom>: Select by atom name (e.g., "name CA")
            - element <elem>: Select by element (e.g., "element C")
            - altloc <id>: Select by alternate location (e.g., "altloc A")
            - all: Select all atoms
            - not <selection>: Negate selection
            - <sel1> and <sel2>: Intersection
            - <sel1> or <sel2>: Union
            - Parentheses for grouping

        Returns
        -------
        torch.Tensor
            Boolean tensor of shape (n_atoms,) where True indicates selected atoms.

        Raises
        ------
        RuntimeError
            If the model has not been initialized.
        ValueError
            If selection syntax is invalid.

        Examples
        --------
        ::

            model = Model().load_pdb('structure.pdb')
            # Get mask for chain A
            mask = model.get_selection_mask("chain A")
            # Use mask to update coordinates
            new_coords = model.xyz()[mask] + translation
            model.xyz.set(new_coords, mask)
            # Get mask for backbone atoms
            backbone_mask = model.get_selection_mask("name CA or name C or name N or name O")
            # Complex selection with parentheses
            mask = model.get_selection_mask("chain A and (resname ALA or resname GLY)")
        """
        from torchref.utils.utils import parse_phenix_selection

        if not self.initialized:
            raise RuntimeError(
                "Cannot get selection mask from an uninitialized Model. Load data first."
            )

        return parse_phenix_selection(selection, self.pdb)

    def select(self, selection: str) -> "Model":
        """
        Return a new Model containing only atoms matching the Phenix-style selection.

        Creates an independent copy of the model containing only the selected atoms.
        All tensor data (coordinates, ADPs, occupancies, etc.) and metadata
        are properly subsetted.

        Parameters
        ----------
        selection : str
            Phenix-style selection string. Supports:
            - chain <id>: Select by chain (e.g., "chain A")
            - resseq <num>: Select by residue number (e.g., "resseq 10")
            - resseq <start>:<end>: Select residue range (e.g., "resseq 10:20")
            - resname <name>: Select by residue name (e.g., "resname ALA")
            - name <atom>: Select by atom name (e.g., "name CA")
            - element <elem>: Select by element (e.g., "element C")
            - altloc <id>: Select by alternate location (e.g., "altloc A")
            - all: Select all atoms
            - not <selection>: Negate selection
            - <sel1> and <sel2>: Intersection
            - <sel1> or <sel2>: Union
            - Parentheses for grouping

        Returns
        -------
        Model
            New instance of the same class containing only selected atoms.
            If called on a subclass, returns an instance of that subclass.

        Raises
        ------
        RuntimeError
            If the model has not been initialized.
        ValueError
            If selection syntax is invalid or no atoms are selected.

        Examples
        --------
        ::

            model = Model().load_pdb('structure.pdb')
            # Select chain A
            chain_a = model.select("chain A")
            # Select backbone atoms
            backbone = model.select("name CA or name C or name N or name O")
            # Select residues 10-50 of chain B
            region = model.select("chain B and resseq 10:50")
            # Select all except water
            no_water = model.select("not resname HOH")
            # Complex selection with parentheses
            complex_sel = model.select("chain A and (resname ALA or resname GLY)")

        Notes
        -----
        This method preserves the class type, so subclasses will return
        instances of themselves, not the base Model class.
        """
        from torchref.utils.utils import parse_phenix_selection

        if not self.initialized:
            raise RuntimeError(
                "Cannot select from an uninitialized Model. Load data first."
            )

        # Parse selection and get boolean mask
        selection_mask = parse_phenix_selection(selection, self.pdb)

        # Check that at least one atom is selected
        n_selected = selection_mask.sum().item()
        if n_selected == 0:
            raise ValueError(f"Selection '{selection}' matched no atoms.")

        # Get indices of selected atoms
        selected_indices = torch.where(selection_mask)[0]

        # Create new instance of the SAME class (preserves subclass type)
        # Use type(self) to ensure subclasses return their own type
        selected_model = type(self)(
            dtype_float=self.dtype_float,
            verbose=self.verbose,
            device=self.device,
            strip_H=self.strip_H,
        )

        # Subset PDB DataFrame and reset index
        # Convert to numpy for indexing, then back to tensor indices
        mask_np = selection_mask.cpu().numpy()
        selected_model.pdb = self.pdb.loc[mask_np].copy()
        selected_model.pdb = selected_model.pdb.reset_index(drop=True)
        selected_model.pdb["index"] = selected_model.pdb.index.to_numpy(dtype=int)

        # Copy scalar attributes - spacegroup setter also sets symmetry
        selected_model.spacegroup = self.spacegroup  # gemmi.SpaceGroup is immutable

        # Copy cell (as Cell object) - crystallographic matrices are properties
        # that delegate to Cell, so copying the Cell is sufficient
        if self.cell is not None:
            selected_model.cell = self.cell.clone()

        # Subset per-atom buffers
        if hasattr(self, "aniso_flag") and self.aniso_flag is not None:
            selected_model.register_buffer(
                "aniso_flag", self.aniso_flag[selection_mask].clone()
            )

        # Create new MixedTensors with selected atoms
        selected_model.xyz = MixedTensor(
            self.xyz()[selection_mask].clone().detach(),
            refinable_mask=(
                self.xyz.refinable_mask[selection_mask]
                if self.xyz.refinable_mask is not None
                else None
            ),
            name="xyz",
        )

        selected_model.adp = PositiveMixedTensor(
            self.adp()[selection_mask].clone().detach(),
            refinable_mask=(
                self.adp.refinable_mask[selection_mask]
                if self.adp.refinable_mask is not None
                else None
            ),
            name="adp",
        )

        selected_model.u = MixedTensor(
            self.u()[selection_mask].clone().detach(),
            refinable_mask=(
                self.u.refinable_mask[selection_mask]
                if self.u.refinable_mask is not None
                else None
            ),
            name="aniso_U",
        )

        # Handle occupancy (needs special handling due to sharing groups)
        initial_occ = self.occupancy()[selection_mask].clone().detach()
        sharing_groups, altloc_groups, refinable_mask = (
            selected_model._create_occupancy_groups(selected_model.pdb, initial_occ)
        )
        selected_model.occupancy = OccupancyTensor(
            initial_values=initial_occ,
            sharing_groups=sharing_groups,
            altloc_groups=altloc_groups,
            refinable_mask=refinable_mask,
            dtype=self.dtype_float,
            device=self.device,
            name="occupancy",
        )

        # Set default masks for the selected model
        selected_model.set_default_masks()

        # Register alternative conformations for the selected subset
        selected_model.register_alternative_conformations()

        # Mark as initialized
        selected_model.initialized = True

        if self.verbose > 0:
            print(f"Selected {n_selected}/{len(self.pdb)} atoms with '{selection}'")

        return selected_model

    def xyz_fractional(self) -> torch.Tensor:
        """
        Return atomic coordinates in fractional space.

        Converts Cartesian coordinates to fractional coordinates
        using the inverse fractional matrix.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_atoms, 3) with fractional coordinates.
        """
        if not self.initialized:
            raise RuntimeError(
                "Model must be initialized to compute fractional coordinates."
            )

        # Get Cartesian coordinates
        cartesian_coords = self.xyz()

        fractional_coords = math_torch.cartesian_to_fractional_torch(
            cartesian_coords, self.cell.data, self.inv_fractional_matrix
        )

        return fractional_coords

    def rotate(
        self, rotation_matrix: torch.Tensor, center: Optional[torch.Tensor] = None
    ) -> "Model":
        """
        Apply rotation to atomic coordinates (in-place).

        Rotates all atoms around a specified center point. The rotation is
        applied using the formula: xyz_new = R @ (xyz - center) + center

        Parameters
        ----------
        rotation_matrix : torch.Tensor
            3x3 rotation matrix. Should be orthogonal (R^T @ R = I).
        center : torch.Tensor, optional
            Center of rotation with shape (3,). If None, uses the centroid
            of all atomic coordinates.

        Returns
        -------
        Model
            Self, for method chaining.

        Examples
        --------
        ::

            # Rotate 90 degrees around Z-axis
            import math
            angle = math.pi / 2
            R = torch.tensor([
                [math.cos(angle), -math.sin(angle), 0],
                [math.sin(angle), math.cos(angle), 0],
                [0, 0, 1]
            ])
            model.rotate(R)

            # Rotate around a specific point
            center = torch.tensor([10.0, 20.0, 30.0])
            model.rotate(R, center=center)
        """
        if not self.initialized:
            raise RuntimeError("Model must be initialized to apply rotation.")

        xyz = self.xyz()
        if center is None:
            center = xyz.mean(dim=0)

        # Ensure tensors are on the same device
        rotation_matrix = rotation_matrix.to(device=xyz.device, dtype=xyz.dtype)
        center = center.to(device=xyz.device, dtype=xyz.dtype)

        # Apply rotation: xyz_new = R @ (xyz - center) + center
        xyz_centered = xyz - center
        xyz_rotated = xyz_centered @ rotation_matrix.T + center

        # Update coordinates in-place
        self.xyz[:] = xyz_rotated

        return self

    def translate(
        self, translation: torch.Tensor, fractional: bool = False
    ) -> "Model":
        """
        Apply translation to atomic coordinates (in-place).

        Translates all atoms by a specified vector. The translation can be
        given in either Cartesian or fractional coordinates.

        Parameters
        ----------
        translation : torch.Tensor
            Translation vector with shape (3,).
        fractional : bool, optional
            If True, the translation is interpreted as fractional coordinates
            and converted to Cartesian before applying. Default is False
            (translation is in Cartesian Angstroms).

        Returns
        -------
        Model
            Self, for method chaining.

        Examples
        --------
        ::

            # Translate by 5 Angstroms along X
            model.translate(torch.tensor([5.0, 0.0, 0.0]))

            # Translate by half a unit cell along each axis
            model.translate(torch.tensor([0.5, 0.5, 0.5]), fractional=True)
        """
        if not self.initialized:
            raise RuntimeError("Model must be initialized to apply translation.")

        xyz = self.xyz()
        translation = translation.to(device=xyz.device, dtype=xyz.dtype)

        if fractional:
            # Convert fractional to Cartesian using the fractional matrix
            # fractional_matrix transforms fractional -> Cartesian
            translation_cart = translation @ self.fractional_matrix
        else:
            translation_cart = translation

        # Apply translation in-place
        xyz_translated = xyz + translation_cart
        self.xyz[:] = xyz_translated

        return self

    def get_centroid(self) -> torch.Tensor:
        """
        Compute the centroid (center of mass) of all atoms.

        Returns
        -------
        torch.Tensor
            Centroid coordinates with shape (3,).
        """
        if not self.initialized:
            raise RuntimeError("Model must be initialized to compute centroid.")

        return self.xyz().mean(dim=0)

    def use_internal_coordinates(
        self,
        n_aa_per_segment: int = 3,
        bond_cutoff: float = 2.0,
        cif_dict: dict = None,
        requires_grad: bool = True,
    ) -> "Model":
        """
        Switch xyz to segmented internal coordinate parametrization.

        Replaces the current xyz MixedTensor with a SegmentedInternalCoordinateTensor
        that parametrizes atomic positions using bond lengths, angles, torsion angles,
        and per-segment rigid body parameters. The molecule is broken into independent
        segments to avoid the "lever arm problem" where small torsion changes near
        the root cause large displacements at distant atoms.

        Parameters
        ----------
        n_aa_per_segment : int, optional
            Number of amino acids per segment. Default is 3.
            - Smaller values (1-2): More segments, shallower trees, less lever arm
            - Larger values (5-10): Fewer segments, deeper trees, more lever arm
        bond_cutoff : float, optional
            Distance cutoff for bond detection in Angstroms. Default is 2.0.
            Only used when cif_dict is not provided.
        cif_dict : dict, optional
            CIF dictionary containing bond definitions per residue type.
            If provided, bonds are determined from chemical definitions rather
            than distances, which is more robust for structures with poor geometry.
            Expected format: cif_dict[resname]['bonds'] DataFrame with 'atom1', 'atom2'.
        requires_grad : bool, optional
            Whether internal coordinate parameters should have gradients.
            Default is True.

        Returns
        -------
        Model
            Self, for method chaining.

        Examples
        --------
        ::

            model = Model()
            model.load_pdb('structure.pdb')
            model.use_internal_coordinates(n_aa_per_segment=3)

            # Now model.xyz() returns coordinates reconstructed from
            # segmented internal coordinates

            # Shake the structure using internal coordinates
            new_xyz = model.xyz.shake(magnitude=0.1)

            # Each segment has independent internal coordinates and
            # rigid body parameters (position + orientation)

        Notes
        -----
        After calling this method, model.xyz will be a SegmentedInternalCoordinateTensor
        instead of a MixedTensor. This provides:
        - Shallow spanning trees within segments (depth ~10-30 vs ~1000)
        - Independent segments that don't propagate changes to distant atoms
        - Rigid body parameters (position + orientation) per segment
        - forward() / __call__(): Reconstruct Cartesian coordinates
        - shake(magnitude): Add noise to internal parameters
        - Gradient flow through all internal coordinate parameters
        """
        if not self.initialized:
            raise RuntimeError(
                "Model must be initialized before switching to internal coordinates. "
                "Load data first with load_pdb() or load_cif()."
            )

        from torchref.model.segmented_internal_coordinates import (
            SegmentedInternalCoordinateTensor
        )

        # Get current coordinates
        current_xyz = self.xyz().detach()

        # Create segmented internal coordinate tensor
        self.xyz = SegmentedInternalCoordinateTensor(
            current_xyz,
            pdb=self.pdb,
            n_aa_per_segment=n_aa_per_segment,
            bond_cutoff=bond_cutoff,
            cif_dict=cif_dict,
            requires_grad=requires_grad,
            dtype=self.dtype_float,
            device=self.device,
        )

        if self.verbose > 0:
            print(f"Switched to internal coordinate parametrization: {self.xyz}")

        return self

    def use_segmented_internal_coordinates(
        self,
        n_aa_per_segment: int = 3,
        bond_cutoff: float = 2.0,
        cif_dict: dict = None,
        requires_grad: bool = True,
    ) -> "Model":
        """
        Alias for use_internal_coordinates().

        This method is provided for backward compatibility. It calls
        use_internal_coordinates() with the same parameters.

        See Also
        --------
        use_internal_coordinates : The main method for internal coordinate conversion.
        """
        return self.use_internal_coordinates(
            n_aa_per_segment=n_aa_per_segment,
            bond_cutoff=bond_cutoff,
            cif_dict=cif_dict,
            requires_grad=requires_grad,
        )
