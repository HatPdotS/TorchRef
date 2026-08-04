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

from typing import Dict, Iterable, List, Optional, Tuple, Union

import gemmi
import torch
import torch.nn as nn

from torchref.base import math_torch
from torchref.config import get_float_dtype, normalize_device
from torchref.io import cif, pdb
from torchref.model.parameter_wrappers import (
    CholeskyMixedTensor,
    MixedTensor,
    OccupancyTensor,
    PositiveMixedTensor,
)
from torchref.symmetry import Cell, SpaceGroup
from torchref.utils.debug_utils import DebugMixin
from torchref.utils.device_mixin import DeviceMovementMixin
from torchref.utils.utils import sanitize_pdb_dataframe

# Standard 3-letter to 1-letter amino acid code mapping
_THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
    # Common modified residues
    "MSE": "M",
    "CSE": "C",
    "SEP": "S",
    "TPO": "T",
    "PTR": "Y",
}


class Model(DeviceMovementMixin, DebugMixin, nn.Module):
    """
    Base model class for atomic structure models using PyTorch.

    Owns the atomic data -- coordinates, atomic displacement parameters and
    occupancies -- each held in a parameter wrapper that decides which atoms are
    refinable. Build it empty (``Model()`` then ``load_pdb`` / ``load_cif`` /
    ``load_state_dict``); ``if model:`` tests *initialization*, not existence.

    Parameters
    ----------
    dtype_float : torch.dtype, optional
        Data type for floating point tensors. Defaults to the configured dtypes.float.
    verbose : int, optional
        Verbosity level for logging. Default is 1.
    device : torch.device, optional
        Computation device. Defaults to the configured device.current.
    strip_H : bool, optional
        Whether to strip hydrogen atoms when loading. Default is True.

    Attributes
    ----------
    xyz : MixedTensor
        Atomic coordinates tensor with shape (n_atoms, 3).
    adp : PositiveMixedTensor
        Atomic displacement parameters (isotropic B-factors, Å²) with shape (n_atoms,).
    u : CholeskyMixedTensor
        Anisotropic displacement parameters with shape (n_atoms, 6), kept
        positive-definite by construction. Isotropic atoms carry ``U = NaN``.
    occupancy : OccupancyTensor
        Atomic occupancies with values in [0, 1].
    pdb : pandas.DataFrame
        DataFrame containing atomic model data. Only refreshed from the tensors
        by :meth:`update_pdb`.
    cell : Cell
        Unit cell object with parameters [a, b, c, alpha, beta, gamma].
    spacegroup, symmetry : SpaceGroup
        Space group object; ``symmetry`` is the same object under its old name.
    initialized : bool
        Whether the model has been initialized with data.
    """

    def __init__(
        self,
        dtype_float=None,
        verbose=1,
        device=None,
        strip_H: bool = True,
    ):
        """
        Initialize an empty Model shell.

        Creates a model shell ready for file loading via load_pdb()/load_cif()
        or state restoration via load_state_dict().

        Parameters
        ----------
        dtype_float : torch.dtype, optional
            Data type for floating point tensors. Defaults to the configured dtypes.float.
        verbose : int, optional
            Verbosity level for logging. Default is 1.
        device : torch.device, optional
            Computation device. Defaults to the configured device.current.
        strip_H : bool, optional
            Whether to strip hydrogen atoms when loading. Default is True.
        """
        super().__init__()
        # Resolve dtype/device at call time (not import time) so a runtime
        # ``dtypes.float`` / ``device.current`` change is honored.
        if dtype_float is None:
            dtype_float = get_float_dtype()
        device = normalize_device(device)
        self.dtype_float = dtype_float
        self.verbose = verbose
        self.device = device
        self.strip_H = strip_H
        self._exclude_H_from_sf = False

        # State tracking
        self.initialized = False
        self.altloc_pairs = []

        # These will be set during load() or load_state_dict()
        self.pdb = None
        self.links = None
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
        """Return the initialization status when used in boolean context.

        Note that ``if model:`` tests *initialization*, not non-``None``-ness:
        an uninitialized (but non-``None``) model is falsy. Use
        ``if model is not None`` when you mean an existence check.
        """
        return self.initialized

    @property
    def exclude_H_from_sf(self) -> bool:
        """Drop H from ``get_iso()`` / ``get_aniso()`` (so from Fcalc) while
        keeping them in the geometry and VDW restraints. Default False.
        """
        return self._exclude_H_from_sf

    @exclude_H_from_sf.setter
    def exclude_H_from_sf(self, value: bool):
        self._exclude_H_from_sf = bool(value)
        # The cached iso/aniso indices encode the H choice, so rebuild them.
        if self.initialized and self.pdb is not None:
            self._rebuild_sf_indices()

    def _rebuild_sf_indices(self):
        """Rebuild cached iso/aniso index arrays from aniso_flag and H mask."""
        iso_mask = ~self.aniso_flag
        aniso_mask = self.aniso_flag

        if self._exclude_H_from_sf and self.pdb is not None:
            if not hasattr(self, "_heavy_atom_mask"):
                h_mask = torch.tensor(
                    (self.pdb["element"].str.strip() != "H").values,
                    dtype=torch.bool,
                    device=self.device,
                )
                self.register_buffer("_heavy_atom_mask", h_mask)
            iso_mask = iso_mask & self._heavy_atom_mask
            aniso_mask = aniso_mask & self._heavy_atom_mask

        self._iso_indices = iso_mask.nonzero(as_tuple=True)[0]
        self._aniso_indices = aniso_mask.nonzero(as_tuple=True)[0]
        # Fast-path flags: an everywhere-True iso_mask lets ``get_iso()`` skip the
        # gather (and its ``index_put_`` backward) entirely, and
        # ``_aniso_is_empty`` lets ``get_aniso()`` short-circuit — the typical
        # macromolecular case.
        self._iso_covers_all = bool(iso_mask.all().item())
        self._aniso_is_empty = int(self._aniso_indices.numel()) == 0

    # =========================================================================
    # Cell, SpaceGroup, and Symmetry properties
    # =========================================================================

    @property
    def cell(self) -> Optional[Cell]:
        """Unit cell object with parameters [a, b, c, alpha, beta, gamma]."""
        return self._cell

    @cell.setter
    def cell(self, value: Cell):
        """Set the unit cell."""
        self._cell = value

    @property
    def spacegroup(self) -> Optional[gemmi.SpaceGroup]:
        """Space group object, or None if not set."""
        return self._spacegroup

    @spacegroup.setter
    def spacegroup(self, value):
        """Set the space group from a SpaceGroup, gemmi object, name or number."""
        if value is not None:
            # ``device=self.device``: SpaceGroup falls back to the global
            # default otherwise, so setting a spacegroup on a CPU-pinned Model
            # would silently plant accelerator-resident matrices on it.
            self._spacegroup = SpaceGroup(value, device=self.device)
        else:
            self._spacegroup = None

    @property
    def symmetry(self) -> Optional[SpaceGroup]:
        """The same object as :attr:`spacegroup`, under its older name."""
        return self._spacegroup

    @symmetry.setter
    def symmetry(self, value: Optional[SpaceGroup]):
        """Set the space group object directly (no coercion, unlike ``spacegroup``)."""
        self._spacegroup = value

    # =========================================================================
    # Crystallographic matrix properties (delegated to Cell)
    # =========================================================================

    @property
    def inv_fractional_matrix(self) -> torch.Tensor:
        """``(3, 3)`` fractionalization matrix B^-1 (Cartesian -> fractional)."""
        return self.cell.inv_fractional_matrix.to(dtype=self.dtype_float)

    @property
    def fractional_matrix(self) -> torch.Tensor:
        """``(3, 3)`` orthogonalization matrix B (fractional -> Cartesian)."""
        return self.cell.fractional_matrix.to(dtype=self.dtype_float)

    @property
    def recB(self) -> torch.Tensor:
        """``(3, 3)`` reciprocal basis matrix with [a*, b*, c*] as rows."""
        return self.cell.reciprocal_basis_matrix.to(dtype=self.dtype_float)

    # =========================================================================
    # Atomic Number (Z) Property
    # =========================================================================

    @property
    def Z(self) -> torch.Tensor:
        """Atomic numbers, shape ``(n_atoms,)``; built and cached on first access."""
        return self._build_z_tensor()

    def _build_z_tensor(self) -> torch.Tensor:
        """Cached ``(n_atoms,)`` atomic numbers from the element column.

        Unknown elements map to 0.
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
        self.register_buffer(
            "_Z", torch.tensor(z_values, dtype=torch.int32, device=self.device)
        )
        return self._Z

    # =========================================================================
    # Scattering Factor Parametrization
    # =========================================================================

    def _build_parametrization(self):
        """Register the ``_A`` / ``_B`` ITC92 buffers by Z-based table lookup and
        return the ``{element: (A, B)}`` dict. Cached; called lazily on first
        access to :attr:`parametrization` or the scattering parameters.
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

        from torchref.base.scattering.scattering_table import get_scattering_params_by_z

        z_tensor = self.Z
        A, B = get_scattering_params_by_z(
            z_tensor, device=self.device, dtype=self.dtype_float
        )

        self.register_buffer("_A", A)
        self.register_buffer("_B", B)

        # Legacy per-element view: one representative row per element.
        elements = self.pdb.element.tolist()
        unique_elements = list(set(elements))
        self._parametrization = {}

        for elem in unique_elements:
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
        """ITC92 ``{element: (A, B)}`` dict, built on first access."""
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

        Notes
        -----
        ``n_iso_atoms`` honors ``exclude_H_from_sf``: when H exclusion is
        active the isotropic count is the H-excluded count (mirroring
        :meth:`get_iso`).
        """
        self._build_parametrization()
        idx = self._iso_indices
        return self._A[idx], self._B[idx]

    def get_scattering_params_aniso(self):
        """
        Get ITC92 scattering parameters (A, B) for anisotropic atoms.

        Returns
        -------
        A : torch.Tensor
            ITC92 A parameters (amplitudes) with shape (n_aniso_atoms, 5).
        B : torch.Tensor
            ITC92 B parameters (widths) with shape (n_aniso_atoms, 5).

        Notes
        -----
        ``n_aniso_atoms`` honors ``exclude_H_from_sf``: when H exclusion is
        active the anisotropic count is the H-excluded count (mirroring
        :meth:`get_aniso`).
        """
        self._build_parametrization()
        idx = self._aniso_indices
        return self._A[idx], self._B[idx]

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

        Returns
        -------
        Model
            Self, for method chaining.
        """
        self._cif_path = cif_path
        # Reset restraints so they will be rebuilt on next access
        self._restraints = None
        return self

    def _build_restraints(self):
        """Build and cache ``RestraintsNew`` over this model's DataFrame, wiring in
        the live ``xyz`` / ``adp`` / ``vdw_radii`` callables.
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
            cell=self._cell,
            spacegroup=self._spacegroup,
            links=self.links,
            verbose=self.verbose,
        )

        return self._restraints

    @property
    def restraints(self):
        """Bond/angle/torsion/... restraints, built on first access from the
        DataFrame and the CIF path given to :meth:`set_restraints_cif`.
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
        """
        Populate the model from a reader callable.

        The central loader that ``load_pdb`` / ``load_cif`` /
        ``_new_model_from_df`` funnel through: it strips hydrogens (when
        ``strip_H``), drops rows with NaN coordinates / B-factors / occupancies,
        builds the cell and space group, and constructs the four parameter
        wrappers (``u`` as a :class:`CholeskyMixedTensor`, as
        :meth:`create_from_state_dict` also does, so the parametrization
        round-trips).

        Parameters
        ----------
        reader : callable
            Zero-argument callable returning ``(pdb_df, cell, spacegroup)``. An
            optional ``.links`` attribute on it is stored on ``self.links``.

        Returns
        -------
        Model
            Self, for method chaining.

        Notes
        -----
        Side effects: sets ``pdb``, ``links``, ``cell``, ``spacegroup``, the
        ``aniso_flag`` buffer, the four wrappers, the default masks, the altloc
        registration and ``initialized = True``.
        """
        self.pdb, cell, spacegroup = reader()
        self.links = getattr(reader, "links", None)

        self.pdb = (
            self.pdb.loc[self.pdb["element"] != "H"].reset_index(drop=True)
            if self.strip_H
            else self.pdb
        )
        self.pdb.dropna(subset=["x", "y", "z", "tempfactor", "occupancy"], inplace=True)
        self.pdb["index"] = self.pdb.index.to_numpy(dtype=int)

        self.cell = Cell(cell, dtype=self.dtype_float, device=self.device)

        # Setter also updates symmetry.
        self.spacegroup = spacegroup

        self.register_buffer(
            "aniso_flag",
            torch.tensor(
                self.pdb["anisou_flag"].values, dtype=torch.bool, device=self.device
            ),
        )
        # Pre-compute integer indices for SF calculation (respects exclude_H_from_sf)
        self._rebuild_sf_indices()

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
        # Cholesky parametrization keeps U positive-definite by construction
        # (U = L Lᵀ), so refinement cannot drive it indefinite and NaN the FFT.
        self.u = CholeskyMixedTensor(
            torch.tensor(
                self.pdb[["u11", "u22", "u33", "u12", "u13", "u23"]].values,
                dtype=self.dtype_float,
            ),
            name="aniso_U",
            device=self.device,
        )

        # Residue-level sharing plus altloc sum-to-1 groups.
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
        self._input_file = str(file)
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
        self._input_file = str(file)
        if self.verbose > 0:
            print(f"Loading CIF file: {file}")

        # Read CIF file
        cif_reader = cif.ModelCIFReader(file)

        return self.load(cif_reader)

    @property
    def chain_sequences(self) -> List[Tuple[str, str]]:
        """Per-chain one-letter sequences, ``[(chain_id, sequence), ...]``.

        HETATM records are excluded, numbering gaps become ``?`` and unrecognized
        residues ``X``.
        """
        if self.pdb is None:
            return []

        atom_df = self.pdb[self.pdb["ATOM"] == "ATOM"]
        result = []

        for chain in atom_df["chainid"].unique():
            chain_df = atom_df[atom_df["chainid"] == chain]
            residues = chain_df.drop_duplicates(subset=["resseq", "icode"]).sort_values(
                "resseq"
            )
            resseqs = residues["resseq"].values
            resnames = residues["resname"].values

            seq_chars = []
            for i, (rseq, rname) in enumerate(zip(resseqs, resnames)):
                if i > 0:
                    gap = int(rseq) - int(resseqs[i - 1]) - 1
                    if gap > 0:
                        seq_chars.extend(["?"] * gap)
                code = _THREE_TO_ONE.get(str(rname).strip(), "X")
                seq_chars.append(code)

            result.append((str(chain), "".join(seq_chars)))

        return result

    def get_chain_residues(self) -> List[Tuple[str, List[str]]]:
        """
        Per-chain residue names as 3-letter codes (for IHM/CIF writing).

        Excludes HETATM records. Unlike :attr:`chain_sequences`, returns
        the raw 3-letter codes without gap filling.

        Returns
        -------
        list of (str, list of str)
            Ordered list of ``(chain_id, [resname, ...])``.
        """
        if self.pdb is None:
            return []

        atom_df = self.pdb[self.pdb["ATOM"] == "ATOM"]
        result = []

        for chain in atom_df["chainid"].unique():
            chain_df = atom_df[atom_df["chainid"] == chain]
            residues = chain_df.drop_duplicates(subset=["resseq", "icode"]).sort_values(
                "resseq"
            )
            resnames = [str(r).strip() for r in residues["resname"].values]
            result.append((str(chain), resnames))

        return result

    def _create_occupancy_groups(self, pdb_df, initial_occ):
        """Build ``(sharing_groups, altloc_groups, refinable_mask)`` for
        :class:`OccupancyTensor`.

        Altloc conformations share one collapsed index each; other residues share
        one only when their occupancies agree to within 0.01, and an occupancy is
        refinable only if it differs from 1.0 by more than that same deadband.
        """
        n_atoms = len(initial_occ)
        altloc_groups = []
        refinable_mask = torch.zeros(n_atoms, dtype=torch.bool)

        sharing_groups_tensor = torch.arange(n_atoms, dtype=torch.long)
        collapsed_idx = 0

        # First pass: altlocs. ALL atoms of one conformation must share a collapsed
        # index whatever their individual occupancies, or the sum-to-1
        # normalization in OccupancyTensor.forward() acts on the wrong group.
        pdb_with_altlocs = pdb_df[pdb_df["altloc"] != ""]
        altloc_residues = set()

        if len(pdb_with_altlocs) > 0:
            grouped_by_residue = pdb_with_altlocs.groupby(
                ["resname", "resseq", "chainid"]
            )

            for (resname, resseq, chainid), group in grouped_by_residue:
                unique_altlocs = sorted(group["altloc"].unique())

                if len(unique_altlocs) > 1:
                    altloc_residues.add((resname, resseq, chainid))
                    conformation_atom_lists = []

                    for altloc in unique_altlocs:
                        altloc_atoms = group[group["altloc"] == altloc]
                        indices = altloc_atoms["index"].tolist()

                        sharing_groups_tensor[indices] = collapsed_idx

                        for idx in indices:
                            if abs(initial_occ[idx].item() - 1.0) > 0.01:
                                refinable_mask[idx] = True

                        conformation_atom_lists.append(indices)
                        collapsed_idx += 1

                    altloc_groups.append(tuple(conformation_atom_lists))

        # Second pass: non-altloc residues, sharing by occupancy similarity.
        grouped = pdb_df.groupby(["resname", "resseq", "chainid", "altloc"])

        for (resname, resseq, chainid, altloc), group in grouped:
            if (resname, resseq, chainid) in altloc_residues:
                continue

            indices = group["index"].tolist()

            if len(indices) == 0:
                continue

            residue_occs = initial_occ[indices]

            occ_min = residue_occs.min().item()
            occ_max = residue_occs.max().item()
            occ_mean = residue_occs.mean().item()

            if (occ_max - occ_min) <= 0.01:
                sharing_groups_tensor[indices] = collapsed_idx
                collapsed_idx += 1

                if abs(occ_mean - 1.0) > 0.01:
                    for idx in indices:
                        refinable_mask[idx] = True
            else:
                # Occupancies disagree within the residue: keep atoms independent.
                for idx in indices:
                    if abs(initial_occ[idx].item() - 1.0) > 0.01:
                        refinable_mask[idx] = True

        # Compact to contiguous indices 0..n_collapsed-1.
        unique_indices = torch.unique(sharing_groups_tensor, sorted=True)
        index_map = torch.zeros(n_atoms, dtype=torch.long)
        for new_idx, old_idx in enumerate(unique_indices):
            mask = sharing_groups_tensor == old_idx
            sharing_groups_tensor[mask] = new_idx

        n_collapsed = len(unique_indices)

        if self.verbose > 1:
            n_groups = n_collapsed
            n_independent = n_atoms - n_collapsed
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
        """
        Write the current refinable parameters back into ``self.pdb``.

        Copies the live values of ``xyz`` (x/y/z), ``u`` (u11..u23), ``adp``
        (tempfactor), and ``occupancy`` from the parameter wrappers into the
        corresponding columns of the ``self.pdb`` DataFrame. Called by every
        writer and by ``hydrogenate`` / ``generate_hydrogens`` before output.

        Returns
        -------
        pandas.DataFrame
            The updated ``self.pdb`` DataFrame.

        Notes
        -----
        This does **not** touch ``anisou_flag``: the iso/aniso classification
        of each atom is left unchanged (see ``_apply_adp_partition`` for the
        partition logic that owns that flag).
        """
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

        import pandas as pd

        from torchref import PATH_TORCHREF_DATA

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

    def _after_device_apply(
        self, old_device, new_device, old_dtype, new_dtype, *,
        device_changed, dtype_changed,
    ):
        """Regenerate the iso/aniso index tensors on the new device.

        The movement hook, not a ``to()`` override (``_apply`` bypasses ``to()``)
        and not ``reset_cache()`` (which fires after every optimizer step).
        """
        if getattr(self, "aniso_flag", None) is not None:
            self._rebuild_sf_indices()
        if self.verbose > 0:
            print(f"Model moved to device: {self.device}")

    def copy(self):
        """
        Create a deep copy of the Model.

        Creates a complete independent copy including all registered buffers,
        module parameters, PDB DataFrame, and spacegroup information.

        Returns
        -------
        Model
            A new, fully independent Model instance with copied data.
        """
        if not self.initialized:
            raise RuntimeError("Cannot copy an uninitialized Model. Load data first.")

        model_copy = Model(
            dtype_float=self.dtype_float,
            verbose=self.verbose,
            device=self.device,
            strip_H=self.strip_H,
        )

        model_copy.pdb = self.pdb.copy(deep=True)

        # Setter also sets symmetry; gemmi.SpaceGroup is immutable, so shared.
        model_copy.spacegroup = self.spacegroup
        model_copy.initialized = True

        if self.cell is not None:
            model_copy.cell = self.cell.clone()

        for buffer_name, buffer_value in self._buffers.items():
            if buffer_value is not None:
                model_copy.register_buffer(buffer_name, buffer_value.clone())

        # Parameter wrappers via their own .copy(), which preserves each
        # wrapper's parametrization (log-space, Cholesky, collapsed logits).
        for module_name, module in self._modules.items():
            if module is not None and hasattr(module, "copy"):
                setattr(model_copy, module_name, module.copy())

        if hasattr(self, "altloc_pairs") and self.altloc_pairs:
            model_copy.altloc_pairs = [
                tuple(tensor.clone() for tensor in group) for group in self.altloc_pairs
            ]
        else:
            model_copy.altloc_pairs = []

        if self.verbose > 0:
            print(f"✓ Model copied successfully ({len(model_copy.pdb)} atoms)")

        return model_copy

    def write_pdb(self, filename, metadata=None):
        """Write model to PDB file with optional metadata header.

        Parameters
        ----------
        filename : str
            Output PDB file path.
        metadata : RefinementMetadata, optional
            Metadata to render as PDB header (REMARK 3, TITLE, etc.).
        """
        self.update_pdb()
        self.pdb = sanitize_pdb_dataframe(self.pdb)
        self.pdb.attrs["spacegroup"] = self.spacegroup.hm if self.spacegroup else "P 1"
        pdb.write(self.pdb, filename, metadata=metadata)

    def write_cif(self, filename, metadata=None):
        """Write model to mmCIF file with optional metadata.

        Parameters
        ----------
        filename : str
            Output mmCIF file path.
        metadata : RefinementMetadata, optional
            Metadata to include (refinement statistics, title, etc.).
        """
        self.update_pdb()
        self.pdb = sanitize_pdb_dataframe(self.pdb)
        self.pdb.attrs["spacegroup"] = self.spacegroup.hm if self.spacegroup else "P 1"
        cif.write_model(self.pdb, filename, metadata=metadata)

    def get_iso(self):
        """
        Return per-atom parameters for the isotropic atom subset.

        Selects atoms whose ADP is a single scalar ``b``: ``~self.aniso_flag``,
        intersected with the heavy-atom mask when ``exclude_H_from_sf`` is on.

        Returns
        -------
        xyz : torch.Tensor, shape ``(n_iso, 3)``
            Cartesian coordinates of the isotropic atoms (Å).
        adp : torch.Tensor, shape ``(n_iso,)``
            Isotropic B-factors (Å²).
        occupancy : torch.Tensor, shape ``(n_iso,)``
            Occupancies in ``[0, 1]``.

        Notes
        -----
        When the subset is everything (the common all-isotropic, H-included case)
        the wrapper outputs are returned directly, skipping a redundant gather and
        its backward scatter. :meth:`get_aniso` covers the complement.
        """
        if self._iso_covers_all:
            return self.xyz(), self.adp(), self.occupancy()
        # Use pre-computed integer indices to avoid boolean indexing GPU sync.
        idx = self._iso_indices
        xyz = self.xyz()[idx]
        adp = self.adp()[idx]
        occupancy = self.occupancy()[idx]
        return xyz, adp, occupancy

    def set_default_masks(self):
        """
        Register the default refinable masks for all four parameter wrappers.

        Builds and registers ``xyz_mask`` (all atoms), ``adp_mask`` (non-NaN
        B-factors), ``u_mask`` (atoms with no NaN U component), and
        ``occupancy_mask`` (occupancies below 0.999), then pushes each mask
        into the corresponding parameter wrapper via ``update_refinable_mask``.
        Called from :meth:`load` after the wrappers are constructed.
        """
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

    PARAM_TYPES: Tuple[str, ...] = ("xyz", "adp", "u", "occupancy")

    def parameters_of_types(self, types: Iterable[str]) -> List[nn.Parameter]:
        """Return the leaf ``nn.Parameter``s for the named parameter types.

        Used by refinement entry points (``refine_xyz``, ``refine_adp``, ...)
        to construct an optimizer over only the leaves the caller intends to
        update. ``LossState.step`` then uses the optimizer's param groups as
        intent and disables ``requires_grad`` on any other leaves the loss
        also touches.

        Parameters
        ----------
        types : Iterable[str]
            Subset of ``Model.PARAM_TYPES``: ``"xyz"``, ``"adp"``, ``"u"``,
            ``"occupancy"``. Unknown names are silently skipped.

        Returns
        -------
        list of nn.Parameter
            The ``refinable_params`` leaf for each requested type, in the
            order the types were given.
        """
        out: List[nn.Parameter] = []
        for t in types:
            wrapper = getattr(self, t, None)
            if wrapper is None:
                continue
            rp = getattr(wrapper, "refinable_params", None)
            if rp is not None:
                out.append(rp)
        return out

    def freeze(self, target: str):
        """
        Freeze (stop refining) one parameter type.

        Parameters
        ----------
        target : str
            One of ``"xyz"``, ``"adp"``, ``"u"``, ``"occupancy"``.
            Unrecognized names are ignored. (``occupancy`` is frozen via the
            OccupancyTensor's ``freeze_all`` rather than ``fix_all``.)
        """
        if target == "xyz":
            self.xyz.fix_all()
        elif target == "adp":
            self.adp.fix_all()
        elif target == "u":
            self.u.fix_all()
        elif target == "occupancy":
            self.occupancy.freeze_all()  # OccupancyTensor uses freeze_all() not fix_all()

    def freeze_all(self):
        """Freeze every parameter type (``xyz``, ``adp``, ``u``, ``occupancy``)."""
        self.freeze("xyz")
        self.freeze("adp")
        self.freeze("u")
        self.freeze("occupancy")

    def unfreeze_all(self):
        """Unfreeze every parameter type, restoring each wrapper's default mask."""
        self.unfreeze("xyz")
        self.unfreeze("adp")
        self.unfreeze("u")
        self.unfreeze("occupancy")

    def unfreeze(self, target: str):
        """
        Unfreeze (resume refining) one parameter type.

        Restores the parameter's default refinable mask (``xyz_mask`` /
        ``adp_mask`` / ``u_mask`` / ``occupancy_mask``).

        Parameters
        ----------
        target : str
            One of ``"xyz"``, ``"adp"``, ``"u"``, ``"occupancy"``.
            Unrecognized names are ignored.
        """
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

    def set_adp_mode(self, mode: str = "isotropic", aniso_selection: str = None):
        """Set the atomic displacement parameter (ADP) parametrization.

        Repartitions atoms between isotropic (a single B in ``adp``) and
        anisotropic (a 6-component U in ``u``), *converting* the stored values and
        refreshing everything keyed off the split: ``aniso_flag``, the cached SF
        index arrays, the refinable masks, the PDB ``anisou_flag`` column (which
        gates ANISOU output) and the forward caches.

        A true conversion, not a freeze: an anisotropic atom's structure factor
        uses only its ``u``, so freezing ``u`` instead would leave most atoms' ADPs
        merely fixed rather than isotropic.

        Parameters
        ----------
        mode : {"isotropic", "anisotropic"}, optional
            ``"isotropic"`` (default) converts every atom, previously anisotropic
            ones to ``B_eq = (8 pi^2 / 3)(U11 + U22 + U33)``. ``"anisotropic"``
            converts those matching ``aniso_selection``, expanding isotropic atoms
            to ``U = (B / 8 pi^2) I``.
        aniso_selection : str, optional
            Phenix-style selection for ``mode="anisotropic"``, default
            ``"not resname HOH and not element H"``; ignored otherwise.

        Notes
        -----
        Run once at model setup, before scaling / restraints / targets. The
        isotropic result matches a freshly-loaded isotropic-only model.
        """
        if not getattr(self, "initialized", False) or self.pdb is None:
            return
        if mode == "isotropic":
            aniso_mask = torch.zeros(
                len(self.pdb), dtype=torch.bool, device=self.device
            )
        elif mode == "anisotropic":
            from torchref.utils.utils import create_selection_mask

            sel = aniso_selection or "not resname HOH and not element H"
            aniso_mask = torch.as_tensor(
                create_selection_mask(sel, self.pdb), dtype=torch.bool
            ).to(self.device)
        else:
            raise ValueError(
                f"Unknown ADP mode: {mode!r}. Use 'isotropic' or 'anisotropic'."
            )
        self._apply_adp_partition(aniso_mask)

    def _apply_adp_partition(self, aniso_mask: torch.Tensor):
        """Convert ADP storage to match a target anisotropic-atom mask.

        The body of :meth:`set_adp_mode`: rebuilds both wrappers and refreshes
        ``aniso_flag``, the SF index cache, the masks, ``anisou_flag`` and caches.
        """
        import math

        eight_pi_sq = 8.0 * math.pi**2
        aniso_mask = torch.as_tensor(
            aniso_mask, dtype=torch.bool, device=self.device
        )
        with torch.no_grad():
            B = self.adp().detach().clone()
            U = self.u().detach().clone()
            finite_U = torch.isfinite(U).all(dim=1)

            # --- target U (NaN row == isotropic atom) ---
            U_target = U.clone()
            entering = aniso_mask & ~finite_U  # iso -> aniso: expand B to U_iso*I
            u_iso = (B / eight_pi_sq)[entering]
            z = torch.zeros_like(u_iso)
            U_target[entering] = torch.stack([u_iso, u_iso, u_iso, z, z, z], dim=1)
            U_target[~aniso_mask] = float("nan")  # isotropic atoms carry U = NaN

            # --- target B (equivalent isotropic B_eq for atoms leaving aniso) ---
            B_target = B.clone()
            leaving = (~aniso_mask) & finite_U
            beq = (eight_pi_sq / 3.0) * (U[:, 0] + U[:, 1] + U[:, 2])
            B_target[leaving] = beq[leaving]

        # Rebuild the parameter wrappers from the converted values (mirrors load()).
        self.adp = PositiveMixedTensor(
            B_target.to(self.dtype_float), name="adp", device=self.device
        )
        self.u = CholeskyMixedTensor(
            U_target.to(self.dtype_float), name="aniso_U", device=self.device
        )

        # Update the per-atom iso/aniso split and everything keyed off it.
        self.aniso_flag = aniso_mask.clone()
        self._rebuild_sf_indices()
        # Clean partition: isotropic atoms refine B (adp), anisotropic atoms refine U.
        self.adp_mask = ~aniso_mask
        self.u_mask = aniso_mask.clone()
        self.adp.update_refinable_mask(self.adp_mask)
        self.u.update_refinable_mask(self.u_mask)

        # The PDB/mmCIF writers gate ANISOU on this column; update_pdb() does not
        # touch it, so keep it in sync with the chosen parametrization.
        if self.pdb is not None:
            self.pdb["anisou_flag"] = aniso_mask.detach().cpu().numpy()

        # Anisotropy change invalidates structure-factor + wrapper forward caches.
        if hasattr(self, "reset_cache"):
            self.reset_cache()

    def update_mask_from_selection(
        self, selection_string: str, target: str, mode: str = "set", freeze: bool = True
    ):
        """
        Update the refinable mask for a parameter using Phenix-style selection syntax.

        Updates only the mask buffer (``xyz_mask`` / ``adp_mask`` / ``u_mask`` /
        ``occupancy_mask``); the parameter tensors keep their old split until
        :meth:`apply_mask_to_parameter` is called.

        Parameters
        ----------
        selection_string : str
            Phenix-style selection string (see parse_phenix_selection docs).
        target : str
            Parameter to update: 'xyz', 'adp', 'u', or 'occupancy'.
        mode : str, optional
            How to combine with current mask: ``'set'`` (default) replaces,
            ``'add'`` unions, ``'remove'`` subtracts.
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

            model.update_mask_from_selection("chain A", "xyz", freeze=True)
            model.apply_mask_to_parameter("xyz")
        """
        from torchref.utils.utils import create_selection_mask

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

        selection_mask = create_selection_mask(
            selection_string,
            self.pdb,
            current_mask=current_mask if mode != "set" else None,
            mode=mode,
        )

        # Masks name the REFINABLE atoms, so freezing clears the selection.
        if freeze:
            updated_mask = current_mask & ~selection_mask
        else:
            updated_mask = selection_mask

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
        Push the current mask buffer into the parameter wrapper's refinable split.

        The counterpart to :meth:`update_mask_from_selection`, which only edits the
        buffer. Replaces the wrapper's ``refinable_params``, so rebuild any
        optimizer afterwards.

        Parameters
        ----------
        target : str
            Parameter to update: 'xyz', 'adp', 'u', or 'occupancy'.

        Raises
        ------
        ValueError
            If target is not recognized.
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
            ``'all'`` (default) for xyz + adp + u + occupancy, one parameter name,
            or a list of them.

        Examples
        --------
        ::

            model.freeze_selection("chain A")                     # everything
            model.freeze_selection("resseq 10:20", targets='xyz')  # coords only
        """
        if targets == "all":
            targets = ["xyz", "adp", "u", "occupancy"]
        elif isinstance(targets, str):
            targets = [targets]

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
            ``'all'`` (default) for xyz + adp + u + occupancy, one parameter name,
            or a list of them.

        Examples
        --------
        ::

            model.unfreeze_selection("chain A")
            model.unfreeze_selection("name CA or name C or name N", targets='xyz')
        """
        if targets == "all":
            targets = ["xyz", "adp", "u", "occupancy"]
        elif isinstance(targets, str):
            targets = [targets]

        for target in targets:
            self.update_mask_from_selection(
                selection_string, target, mode="set", freeze=False
            )
            self.apply_mask_to_parameter(target)

    def get_aniso(self):
        """
        Return per-atom parameters for the anisotropic atom subset.

        Selects atoms whose ADP is the 6-element tensor
        ``u = (u11, u22, u33, u12, u13, u23)``: ``self.aniso_flag``, intersected
        with the heavy-atom mask when ``exclude_H_from_sf`` is on.

        Returns
        -------
        xyz : torch.Tensor, shape ``(n_aniso, 3)``
            Cartesian coordinates of the anisotropic atoms (Å). Empty
            tensor when there are no anisotropic atoms.
        u : torch.Tensor, shape ``(n_aniso, 6)``
            Anisotropic U components (Å²) in the order
            ``(u11, u22, u33, u12, u13, u23)``. Empty when ``n_aniso == 0``.
        occupancy : torch.Tensor, shape ``(n_aniso,)``
            Occupancies in ``[0, 1]``. Empty when ``n_aniso == 0``.

        Notes
        -----
        With no anisotropic atoms (the common protein case) three empty
        placeholders are returned without touching the wrappers at all, avoiding
        both their forward ``.clone()`` and the slow ``index_put_`` backward the
        gather would generate.
        """
        if self._aniso_is_empty:
            xyz_buf = self.xyz.fixed_values
            empty_xyz = xyz_buf.new_empty(0, 3)
            empty_u = xyz_buf.new_empty(0, 6)
            empty_occ = xyz_buf.new_empty(0)
            return empty_xyz, empty_u, empty_occ
        # Use pre-computed integer indices to avoid boolean indexing GPU sync.
        idx = self._aniso_indices
        xyz = self.xyz()[idx]
        u = self.u()[idx]
        occupancy = self.occupancy()[idx]
        return xyz, u, occupancy

    def adp_u6(self) -> "torch.Tensor":
        """Unified per-atom Cartesian U tensor ``(N, 6)`` for ALL atoms.

        Anisotropic atoms use their refined ``u`` (the 6 components
        ``u11, u22, u33, u12, u13, u23``); isotropic atoms are lifted to the
        equivalent isotropic tensor ``U = (B / 8 pi^2) I`` (off-diagonals 0).
        This gives every atom a common representation so the ADP restraints
        (similarity, locality) act uniformly and handle iso<->aniso pairs
        natively.

        Differentiable: anisotropic rows carry gradient to ``u`` (the Cholesky
        parameters), isotropic rows to ``adp``. ``u()`` returns NaN rows for
        isotropic atoms; those are scrubbed *before* :func:`torch.where` so the
        zeroed (unselected) branch cannot poison the backward pass via
        ``0 * NaN``.

        Returns
        -------
        torch.Tensor
            ``(N, 6)`` Cartesian U components (Å²).
        """
        import math

        B = self.adp()
        U = self.u()
        diag = B.new_tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        u_from_b = (B / (8.0 * math.pi**2)).unsqueeze(-1) * diag
        flag = self.aniso_flag.to(B.device).unsqueeze(-1)
        return torch.where(flag, torch.nan_to_num(U), u_from_b)

    def parameters(self, recurse: bool = True):
        """
        Iterate over refinable parameters, skipping empty ones.

        Wraps :meth:`torch.nn.Module.parameters` and filters out any
        parameter with zero elements (e.g. the ``u`` leaf when there are no
        anisotropic atoms), so an optimizer is never handed an empty tensor.

        Parameters
        ----------
        recurse : bool, optional
            If True (default), include parameters of submodules.

        Yields
        ------
        torch.nn.Parameter
            Each non-empty parameter.
        """
        return (p for p in super().parameters(recurse) if p.numel() > 0)

    def named_mixed_tensors(self):
        """Yield ``(name, wrapper)`` for every :class:`MixedTensor` submodule.

        Subclasses of ``MixedTensor`` are included; ``RigidXYZTensor`` is not.
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
        Rebuild ``self.altloc_pairs`` from the ``altloc`` column.

        One tuple per residue that has multiple conformations, holding one
        index tensor per conformation (in sorted altloc order), e.g.
        ``[(tensor([100, 101]), tensor([110, 111])), ...]``. Overwrites any
        previous content, so call it after the atom numbering changes.
        """
        self.altloc_pairs = []

        pdb_with_altlocs = self.pdb[self.pdb["altloc"] != ""]

        if len(pdb_with_altlocs) == 0:
            return

        grouped = pdb_with_altlocs.groupby(["resname", "resseq", "chainid"])

        for (resname, resseq, chainid), group in grouped:
            unique_altlocs = sorted(group["altloc"].unique())

            # A lone altloc label is not an alternative conformation.
            if len(unique_altlocs) > 1:
                conformation_tensors = []
                for altloc in unique_altlocs:
                    altloc_atoms = group[group["altloc"] == altloc]
                    indices = torch.tensor(
                        altloc_atoms["index"].tolist(), dtype=torch.long
                    )
                    conformation_tensors.append(indices)

                self.altloc_pairs.append(tuple(conformation_tensors))

    def shake_coords(self, stddev: float):
        """
        Perturb every atom's coordinates with Gaussian noise of width *stddev* (Å).

        Rebuilds the ``xyz`` wrapper (mask preserved), so optimizer state built on
        the old ``refinable_params`` is stale.
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
        Perturb every atom's isotropic ADP with Gaussian noise of width *stddev* (Å²).

        Rebuilds the ``adp`` wrapper (mask preserved), so optimizer state built on
        the old ``refinable_params`` is stale.
        """
        adp_values = self.adp().detach()
        new_adp = adp_values + torch.normal(
            mean=0.0, std=stddev, size=adp_values.shape, device=self.device
        )
        self.adp = PositiveMixedTensor(
            new_adp, refinable_mask=self.adp.refinable_mask, name="adp"
        )

    def generate_hydrogens(self, mon_lib_path: str = None) -> "Model":
        """
        Generate hydrogen atoms for the current model using gemmi.

        Places hydrogens at ideal geometry using the CCP4 monomer library and
        gemmi's topology engine. Returns a new Model instance with hydrogens
        added; the original model is not modified.

        Parameters
        ----------
        mon_lib_path : str, optional
            Path to CCP4 monomer library directory. If None, uses the monomer
            library bundled with torchref (covers standard amino acids and
            common small molecules).

        Returns
        -------
        Model
            A new Model instance with hydrogen atoms added (strip_H=False).
            Unknown residues are skipped silently.

        Notes
        -----
        Reads the *current* coordinates (via :meth:`update_pdb`), so run it after
        any coordinate change that should be reflected in the H positions.
        """
        import os
        import tempfile

        import gemmi

        from torchref import PATH_TORCHREF_DATA

        # ``mgr`` is set when we fall back to TorchRef's auto-fetching monomer
        # library manager; per-residue CIFs are then resolved through it (which
        # downloads/caches on demand) rather than from ``mon_lib_path`` directly.
        mgr = None
        if mon_lib_path is None:
            import os as _os

            # In priority order: CCP4's own env var, a library bundled next to the
            # repo, then the partial one shipped inside torchref.
            candidates = [
                _os.environ.get("CLIBD_MON", ""),
                str(PATH_TORCHREF_DATA.parent.parent / "external_monomer_library"),
                str(PATH_TORCHREF_DATA / "monomer_library"),
            ]
            mon_lib_path = None
            for c in candidates:
                if c and _os.path.isfile(_os.path.join(c, "ener_lib.cif")):
                    mon_lib_path = c
                    break
            if mon_lib_path is None:
                # No complete CCP4 library: fall back to TorchRef's manager, which
                # ships standard residues and auto-downloads the rest. This is the
                # normal path — a CCP4 install is not required.
                from torchref.restraints.library import get_library_manager

                mgr = get_library_manager(verbose=self.verbose)
                mon_lib_path = str(mgr.ensure_gemmi_base())

        # gemmi reads from a file, so the live tensors must reach the DataFrame.
        self.update_pdb()

        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            tmp_heavy = f.name
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            tmp_with_h = f.name

        try:
            from torchref.io import pdb as io_pdb
            from torchref.utils.utils import sanitize_pdb_dataframe

            pdb_out = sanitize_pdb_dataframe(self.pdb.copy())
            pdb_out.attrs["spacegroup"] = (
                self.spacegroup.hm if self.spacegroup else "P 1"
            )
            io_pdb.write(pdb_out, tmp_heavy)

            st = gemmi.read_structure(tmp_heavy)
            st.setup_entities()

            # Per-residue CIFs come from the manager (bundled → cache → download)
            # when we fell back to it, else from the explicit library directory.
            monlib = gemmi.read_monomer_lib(mon_lib_path, [])
            resnames = set(r.name for m in st for c in m for r in c)
            for rn in resnames:
                if mgr is not None:
                    cif = mgr.get_cif_file(rn)
                    cif_path = str(cif) if cif is not None else None
                else:
                    cif_path = os.path.join(mon_lib_path, rn[0].lower(), rn + ".cif")
                    if not os.path.exists(cif_path):
                        cif_path = None
                if cif_path is None:
                    continue
                doc = gemmi.cif.read(cif_path)
                for block in doc:
                    if block.name == rn or block.name.startswith("comp_" + rn):
                        monlib.add_monomer_if_present(block)
                        break

            gemmi.prepare_topology(st, monlib, h_change=gemmi.HydrogenChange.ReAdd)
            st.write_pdb(tmp_with_h)

            # strip_H=False, or the hydrogens we just placed would be dropped again.
            new_model = self.__class__(
                dtype_float=self.dtype_float,
                verbose=self.verbose,
                device=self.device,
                strip_H=False,
            )
            new_model.load_pdb(tmp_with_h)

        finally:
            for p in (tmp_heavy, tmp_with_h):
                try:
                    os.unlink(p)
                except OSError:
                    pass

        return new_model

    def _new_model_from_df(self, df, *, strip_H=None):
        """Build a fresh model of the same class from a DataFrame."""
        import inspect

        sh = self.strip_H if strip_H is None else strip_H
        ctor_kw = dict(
            dtype_float=self.dtype_float,
            verbose=0,
            device=self.device,
            strip_H=sh,
        )
        sig = inspect.signature(self.__class__.__init__)
        for pname, param in sig.parameters.items():
            if pname in ("self",) or pname in ctor_kw:
                continue
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            if hasattr(self, pname):
                ctor_kw[pname] = getattr(self, pname)
        if "gridsize" in sig.parameters and hasattr(self, "_explicit_gridsize"):
            ctor_kw["gridsize"] = self._explicit_gridsize

        new_model = self.__class__(**ctor_kw)
        sg_str = self.spacegroup.xhm if self.spacegroup else "P 1"
        new_model.load(lambda: (df, self.pdb.attrs.get("cell"), sg_str))
        if hasattr(new_model, "setup_grid"):
            new_model.setup_grid()
        # Propagate CIF restraint paths so restraints are rebuilt correctly
        if self._cif_path is not None:
            new_model._cif_path = self._cif_path
        return new_model

    def strip_altlocs(self) -> "Model":
        """Return a new model with alternate conformations removed.

        For each residue that has multiple altlocs, the conformer with
        highest average occupancy is kept (ties broken alphabetically).
        The ``altloc`` column is cleared to ``""`` in the returned model.
        The original model is not modified.
        """
        import pandas as pd

        pdb = self.pdb.copy()
        has_altloc = pdb["altloc"].astype(str).str.strip() != ""
        if not has_altloc.any():
            return self._new_model_from_df(pdb)

        drop_idx = []
        res_cols = ["chainid", "resseq", "icode", "resname"]
        altloc_rows = pdb.loc[has_altloc]
        for _, grp in altloc_rows.groupby(res_cols):
            altlocs = sorted(grp["altloc"].unique())
            if len(altlocs) <= 1:
                continue
            # Pick conformer with highest mean occupancy
            best, best_occ = altlocs[0], -1.0
            for al in altlocs:
                occ = grp.loc[grp["altloc"] == al, "occupancy"].mean()
                if occ > best_occ:
                    best, best_occ = al, occ
            # Drop rows belonging to non-best conformers
            for al in altlocs:
                if al != best:
                    drop_idx.extend(grp.index[grp["altloc"] == al].tolist())

        filtered = pdb.drop(index=drop_idx).reset_index(drop=True)
        filtered["altloc"] = ""
        filtered["serial"] = range(1, len(filtered) + 1)
        filtered["index"] = range(len(filtered))

        # Preserve DataFrame attrs
        filtered.attrs = pdb.attrs.copy()

        return self._new_model_from_df(filtered)

    def strip_hydrogens(self) -> "Model":
        """Return a new model with hydrogen atoms removed.

        The returned model has consistent DataFrame and tensors (xyz, adp,
        occupancy) with H atoms excluded.  The original model is not
        modified.

        Returns
        -------
        Model
            New model without hydrogen atoms.
        """
        self.update_pdb()
        pdb = self.pdb.copy()
        h_mask = pdb["element"].str.strip() == "H"
        if not h_mask.any():
            return self._new_model_from_df(pdb, strip_H=True)

        filtered = pdb[~h_mask].reset_index(drop=True)
        filtered["index"] = range(len(filtered))
        filtered.attrs = pdb.attrs.copy()
        return self._new_model_from_df(filtered, strip_H=True)

    # Module-level cache for CIF monomer data (shared across calls)
    _hydrogenate_cif_cache = {}

    def hydrogenate(
        self,
        verbose: int = 0,
        optimize: bool = False,
        lbfgs_steps: int = 3,
        max_iter: int = 20,
    ) -> "Model":
        """
        Return a new model with hydrogen atoms placed via Kabsch alignment.

        Uses torchref's monomer library to identify missing H atoms, places
        them by SVD-aligning ideal monomer coordinates onto the current model
        coordinates, then corrects each H to sit at ideal bond length from its
        parent atom. The original model is not modified.

        Parameters
        ----------
        verbose : int, optional
            Verbosity level (0=silent, 1=summary, 2=detailed). Default 0.
        optimize : bool, optional
            If True, run a short LBFGS geometry optimization on H positions
            after placement. Default False (Kabsch placement only).
        lbfgs_steps : int, optional
            Number of LBFGS outer steps (only when optimize=True). Default 3.
        max_iter : int, optional
            Max line-search iterations per LBFGS step. Default 20.

        Returns
        -------
        Model
            New model with hydrogen atoms added.
            All parameters are unfrozen in the returned model.
        """
        import numpy as np
        import pandas as pd

        from torchref.restraints.library import MonomerLibraryManager

        # Sync current coordinates into DataFrame
        self.update_pdb()

        lib = MonomerLibraryManager(verbose=0)
        cache = Model._hydrogenate_cif_cache

        # --- Phase A: build per-residue-type lookup tables (cached) ---
        for rn in self.pdb["resname"].unique():
            rn_str = str(rn).strip()
            if not rn_str:
                continue
            if rn_str in cache:
                if cache[rn_str] is None or "heavy_neighbor_map" in cache[rn_str]:
                    continue
                del cache[rn_str]  # Stale entry, re-read
            cif_path = lib.get_cif_file(rn_str)
            if cif_path is None:
                cache[rn_str] = None
                continue
            try:
                from torchref.io.cif_readers import RestraintCIFReader

                reader = RestraintCIFReader(str(cif_path))
                all_data = reader.get_all_restraints()
                comp_data = all_data.get(rn_str) or all_data.get(rn_str.upper())
                if comp_data is None:
                    cache[rn_str] = None
                    continue
                atom_df = comp_data.get("atoms", comp_data.get("atom"))
                bond_df = comp_data.get("bonds", comp_data.get("bond"))
                if atom_df is None or atom_df.empty or "x" not in atom_df.columns:
                    cache[rn_str] = None
                    continue
            except Exception:
                cache[rn_str] = None
                continue

            ids = atom_df["atom_id"].astype(str).str.strip().values
            elems = atom_df["type_symbol"].astype(str).str.strip().values
            coords = atom_df[["x", "y", "z"]].values.astype(np.float64)
            is_h = np.array([e.upper() == "H" for e in elems])
            id_to_idx = {n: i for i, n in enumerate(ids)}

            # H→parent map + ideal bond lengths + heavy adjacency
            parent_map = {}  # h_name -> parent_name
            ideal_bl = {}  # h_name -> ideal bond length (Angstrom)
            heavy_neighbor_map = {}  # heavy_name -> [bonded heavy names]
            if bond_df is not None and not bond_df.empty:
                a1s = bond_df["atom1"].astype(str).str.strip().values
                a2s = bond_df["atom2"].astype(str).str.strip().values
                vals = pd.to_numeric(bond_df["value"], errors="coerce").values
                h_set = set(ids[is_h])
                for i in range(len(a1s)):
                    b1, b2 = a1s[i], a2s[i]
                    if b1 in h_set and b2 in id_to_idx and not is_h[id_to_idx[b2]]:
                        parent_map[b1] = b2
                        if np.isfinite(vals[i]):
                            ideal_bl[b1] = float(vals[i])
                    elif b2 in h_set and b1 in id_to_idx and not is_h[id_to_idx[b1]]:
                        parent_map[b2] = b1
                        if np.isfinite(vals[i]):
                            ideal_bl[b2] = float(vals[i])
                    # Heavy-atom adjacency for local Kabsch
                    i1, i2 = id_to_idx.get(b1), id_to_idx.get(b2)
                    if (
                        i1 is not None
                        and i2 is not None
                        and not is_h[i1]
                        and not is_h[i2]
                    ):
                        heavy_neighbor_map.setdefault(b1, []).append(b2)
                        heavy_neighbor_map.setdefault(b2, []).append(b1)

            cache[rn_str] = {
                "ids": ids,
                "elems": elems,
                "coords": coords,
                "is_h": is_h,
                "id_to_idx": id_to_idx,
                "heavy_names": ids[~is_h],
                "heavy_coords": coords[~is_h],
                "h_names": ids[is_h],
                "h_coords": coords[is_h],
                "parent_map": parent_map,
                "ideal_bl": ideal_bl,
                "heavy_neighbor_map": heavy_neighbor_map,
            }

        # Filter to available residue types
        available = {
            rn: cache[rn]
            for rn in self.pdb["resname"].unique()
            if str(rn).strip() in cache and cache.get(str(rn).strip()) is not None
        }
        if not available:
            if verbose > 0:
                print("No monomer library data found; returning copy.")
            return self.copy()

        # --- Phase B: place H atoms via Kabsch alignment ---
        model_names_arr = self.pdb["name"].astype(str).str.strip().values
        model_xyz_arr = self.pdb[["x", "y", "z"]].values.astype(np.float64)
        model_occ_arr = self.pdb["occupancy"].values.astype(np.float64)
        model_bfac_arr = self.pdb["tempfactor"].values.astype(np.float64)
        model_atom_type_arr = self.pdb["ATOM"].values
        model_altloc_arr = self.pdb["altloc"].values.astype(str)

        group_cols = ["chainid", "resseq", "icode", "resname"]
        group_keys = self.pdb[group_cols].values
        changes = np.zeros(len(group_keys), dtype=bool)
        changes[0] = True
        for c in range(4):
            changes[1:] |= group_keys[1:, c] != group_keys[:-1, c]
        group_starts = np.nonzero(changes)[0]
        group_ends = np.append(group_starts[1:], len(group_keys))

        # Pre-allocate lists for H atom data columns
        h_x, h_y, h_z = [], [], []
        h_names_out, h_altlocs, h_resnames = [], [], []
        h_chainids, h_resseqs, h_icodes = [], [], []
        h_occ, h_bfac, h_atom_types = [], [], []
        h_insert_after = []

        max_bond_dist = 1.5  # Reject H atoms placed > this from parent
        _std_val = {"C": 4, "N": 3, "O": 2, "S": 2}

        # Heavy-atom mask for distance-based neighbor detection
        model_elem_arr = self.pdb["element"].astype(str).str.strip().values
        model_heavy_mask_full = np.array([e.upper() != "H" for e in model_elem_arr])

        for gi in range(len(group_starts)):
            s, e = group_starts[gi], group_ends[gi]
            rn = str(group_keys[s, 3]).strip()
            info = cache.get(rn)
            if info is None:
                continue
            chainid = group_keys[s, 0]
            resseq = group_keys[s, 1]
            icode = group_keys[s, 2]

            names_in_model = set(model_names_arr[s:e])
            h_to_add_mask = np.array(
                [n not in names_in_model for n in info["h_names"]], dtype=bool
            )
            if not h_to_add_mask.any():
                continue
            h_names_add = info["h_names"][h_to_add_mask]
            h_coords_ideal = info["h_coords"][h_to_add_mask]

            # Altloc handling
            altlocs_in_res = set(model_altloc_arr[s:e])
            altloc_list = (
                [""]
                if altlocs_in_res <= {""}
                else sorted(a for a in altlocs_in_res if a != "")
            )

            for altloc in altloc_list:
                if altloc == "":
                    mask = np.ones(e - s, dtype=bool)
                else:
                    al = model_altloc_arr[s:e]
                    mask = (al == altloc) | (al == "")

                conf_names = model_names_arr[s:e][mask]
                conf_xyz = model_xyz_arr[s:e][mask]
                conf_occ = model_occ_arr[s:e][mask]
                conf_bfac = model_bfac_arr[s:e][mask]
                conf_atom_type = model_atom_type_arr[s:e][mask]

                # Name→index lookup for this conformer
                name_to_idx = {}
                for j, cn in enumerate(conf_names):
                    if cn not in name_to_idx:
                        name_to_idx[cn] = j

                conf_name_set = set(conf_names)
                common_mask = np.array(
                    [n in conf_name_set for n in info["heavy_names"]],
                    dtype=bool,
                )
                n_common = common_mask.sum()

                # Global Kabsch when ≥ 3 matching heavy atoms
                R_global = t_global = None
                if n_common >= 3:
                    P = info["heavy_coords"][common_mask]
                    Q = np.array(
                        [
                            conf_xyz[name_to_idx[n]]
                            for n in info["heavy_names"][common_mask]
                        ],
                        dtype=np.float64,
                    )
                    cp, cq = P.mean(0), Q.mean(0)
                    Hm = (P - cp).T @ (Q - cq)
                    U, S, Vt = np.linalg.svd(Hm)
                    d = np.linalg.det(Vt.T @ U.T)
                    sign_d = np.diag([1.0, 1.0, 1.0 if d > 0 else -1.0])
                    R_global = Vt.T @ sign_d @ U.T
                    t_global = cq - R_global @ cp

                # Group H atoms by parent for placement
                parent_to_hi = {}
                for hi, h_name in enumerate(h_names_add):
                    pn = info["parent_map"].get(h_name)
                    if pn is not None and pn in name_to_idx:
                        parent_to_hi.setdefault(pn, []).append(hi)

                hnm = info.get("heavy_neighbor_map", {})
                id2i = info["id_to_idx"]
                all_coords = info["coords"]
                mask_idx = np.where(mask)[0]  # conformer indices in [s:e]

                for par_name, hi_list in parent_to_hi.items():
                    pidx = name_to_idx[par_name]
                    parent_pos = conf_xyz[pidx]
                    parent_full = s + mask_idx[pidx]

                    # Heavy neighbors in the model (distance-based,
                    # includes cross-residue bonds like C-N peptide)
                    dvec = model_xyz_arr - model_xyz_arr[parent_full]
                    dists_sq = (dvec**2).sum(1)
                    bonded = np.where(
                        (dists_sq > 0.09) & (dists_sq < 3.61) & model_heavy_mask_full
                    )[0]
                    bonded = bonded[bonded != parent_full]
                    n_model_heavy = len(bonded)

                    # Expected H count from standard valence
                    par_elem = info["elems"][id2i[par_name]].upper()
                    expected_h = max(
                        0,
                        _std_val.get(par_elem, 4) - n_model_heavy,
                    )

                    # --- Step 1: local Kabsch for initial placement ---
                    local_set = {par_name}
                    for nb in hnm.get(par_name, []):
                        local_set.add(nb)
                        for nb2 in hnm.get(nb, []):
                            local_set.add(nb2)
                    local_names = [
                        n for n in local_set if n in name_to_idx and n in id2i
                    ]

                    if len(local_names) >= 3:
                        Pl = np.array([all_coords[id2i[n]] for n in local_names])
                        Ql = np.array([conf_xyz[name_to_idx[n]] for n in local_names])
                        cpl, cql = Pl.mean(0), Ql.mean(0)
                        Hl = (Pl - cpl).T @ (Ql - cql)
                        Ul, _, Vtl = np.linalg.svd(Hl)
                        dl = np.linalg.det(Vtl.T @ Ul.T)
                        sl = np.diag([1.0, 1.0, 1.0 if dl > 0 else -1.0])
                        R_use = Vtl.T @ sl @ Ul.T
                        t_use = cql - R_use @ cpl
                    elif R_global is not None:
                        R_use, t_use = R_global, t_global
                    else:
                        R_use = None  # Will use random placement

                    # Kabsch-place and filter by distance
                    valid_h = []
                    if R_use is not None:
                        for hi in hi_list:
                            h_name = h_names_add[hi]
                            h_cif = all_coords[id2i[h_name]]
                            h_pos = R_use @ h_cif + t_use
                            direction = h_pos - parent_pos
                            dist = np.linalg.norm(direction)
                            if dist < 1e-6 or dist > max_bond_dist:
                                continue
                            bl = info["ideal_bl"].get(h_name, dist)
                            h_pos = parent_pos + direction * (bl / dist)
                            valid_h.append((h_name, h_pos, bl))
                    else:
                        # Random-rotation placement (< 3 matching atoms)
                        # Apply a random SO(3) rotation to ideal CIF
                        # geometry so internal angles are preserved.
                        # Random rotation via QR decomposition.
                        M = np.random.randn(3, 3)
                        Q_r, _ = np.linalg.qr(M)
                        if np.linalg.det(Q_r) < 0:
                            Q_r[:, 0] = -Q_r[:, 0]
                        par_cif = all_coords[id2i[par_name]]
                        for hi in hi_list:
                            h_name = h_names_add[hi]
                            h_cif = all_coords[id2i[h_name]]
                            bl = info["ideal_bl"].get(h_name, 0.97)
                            d_ideal = h_cif - par_cif
                            d_rot = Q_r @ d_ideal
                            dn = np.linalg.norm(d_rot)
                            if dn > 1e-6:
                                d_rot = d_rot * (bl / dn)
                            else:
                                d_rot = np.array([bl, 0.0, 0.0])
                            valid_h.append((h_name, parent_pos + d_rot, bl))

                    # Limit to expected count (removes terminal H)
                    if len(valid_h) > expected_h:
                        valid_h.sort(key=lambda x: x[0])  # alphabetical
                        valid_h = valid_h[:expected_h]

                    # --- Step 2: geometric re-placement ---
                    if n_model_heavy >= 2:
                        nvecs = model_xyz_arr[bonded] - model_xyz_arr[parent_full]
                        svec = nvecs.sum(0)
                        snorm = np.linalg.norm(svec)

                        if len(valid_h) == 1 and snorm > 1e-6:
                            # Single H: place opposite to neighbors
                            h_nm, _, bl = valid_h[0]
                            h_pos = parent_pos - bl * svec / snorm
                            valid_h[0] = (h_nm, h_pos, bl)

                        elif len(valid_h) == 2 and n_model_heavy == 2 and snorm > 1e-6:
                            # CH2-like: sp3 tetrahedral placement
                            v1, v2 = nvecs[0], nvecs[1]
                            base = -svec / snorm
                            perp = np.cross(v1, v2)
                            pn = np.linalg.norm(perp)
                            if pn > 1e-6:
                                perp = perp / pn
                                n1 = np.linalg.norm(v1)
                                n2 = np.linalg.norm(v2)
                                c12 = np.dot(v1, v2) / (n1 * n2)
                                denom = 3.0 * np.sqrt(max(1e-12, (1 + c12) / 2))
                                a = min(1.0, 1.0 / denom)
                                b = np.sqrt(max(0, 1 - a * a))
                                d_up = a * base + b * perp
                                d_dn = a * base - b * perp
                                # Assign Kabsch-nearest to each
                                _, pos0, bl0 = valid_h[0]
                                _, pos1, bl1 = valid_h[1]
                                g_up = parent_pos + bl0 * d_up
                                g_dn = parent_pos + bl1 * d_dn
                                if pos0 is not None and pos1 is not None:
                                    d_same = np.linalg.norm(
                                        pos0 - g_up
                                    ) + np.linalg.norm(pos1 - g_dn)
                                    d_swap = np.linalg.norm(
                                        pos0 - g_dn
                                    ) + np.linalg.norm(pos1 - g_up)
                                    if d_swap < d_same:
                                        g_up, g_dn = g_dn, g_up
                                valid_h[0] = (valid_h[0][0], g_up, bl0)
                                valid_h[1] = (valid_h[1][0], g_dn, bl1)

                    elif n_model_heavy == 1:
                        # One heavy neighbor: place H opposite to it
                        nvec = model_xyz_arr[bonded[0]] - model_xyz_arr[parent_full]
                        nn = np.linalg.norm(nvec)
                        if nn > 1e-6:
                            d_opp = -nvec / nn
                            for vi in range(len(valid_h)):
                                if valid_h[vi][1] is None:
                                    nm, _, bl = valid_h[vi]
                                    valid_h[vi] = (nm, parent_pos + bl * d_opp, bl)

                    # Fill remaining None positions with random dirs
                    for vi in range(len(valid_h)):
                        if valid_h[vi][1] is not None:
                            continue
                        nm, _, bl = valid_h[vi]
                        # Random unit vector via Marsaglia method
                        while True:
                            u = np.random.uniform(-1, 1, 3)
                            n2 = (u * u).sum()
                            if 0.01 < n2 < 1.0:
                                break
                        d = u / np.sqrt(n2)
                        # Push away from already-placed H siblings
                        for vj in range(len(valid_h)):
                            if vj == vi or valid_h[vj][1] is None:
                                continue
                            sep = parent_pos + bl * d - valid_h[vj][1]
                            if np.linalg.norm(sep) < 0.5 * bl:
                                d = -d  # flip to other hemisphere
                                break
                        valid_h[vi] = (nm, parent_pos + bl * d, bl)

                    # --- Step 3: emit placed H atoms ---
                    for h_nm, h_pos, _ in valid_h:
                        h_x.append(h_pos[0])
                        h_y.append(h_pos[1])
                        h_z.append(h_pos[2])
                        h_names_out.append(h_nm)
                        h_altlocs.append(altloc)
                        h_resnames.append(rn)
                        h_chainids.append(chainid)
                        h_resseqs.append(resseq)
                        h_icodes.append(icode)
                        h_occ.append(conf_occ[pidx])
                        h_bfac.append(conf_bfac[pidx])
                        h_atom_types.append(conf_atom_type[pidx])
                        h_insert_after.append(e - 1)

        n_h_placed = len(h_x)
        if n_h_placed == 0:
            if verbose > 0:
                print("No hydrogen atoms to add; returning copy.")
            return self.copy()

        if verbose > 0:
            print(f"Placing {n_h_placed} hydrogen atoms...")

        # Build H DataFrame in one shot
        h_df = pd.DataFrame(
            {
                "ATOM": h_atom_types,
                "serial": 0,
                "name": h_names_out,
                "altloc": h_altlocs,
                "resname": h_resnames,
                "chainid": h_chainids,
                "resseq": h_resseqs,
                "icode": h_icodes,
                "x": h_x,
                "y": h_y,
                "z": h_z,
                "occupancy": h_occ,
                "tempfactor": h_bfac,
                "element": "H",
                "charge": 0,
                "anisou_flag": False,
                "u11": 0.0,
                "u22": 0.0,
                "u33": 0.0,
                "u12": 0.0,
                "u13": 0.0,
                "u23": 0.0,
            }
        )
        insert_after = np.array(h_insert_after)

        # Interleave: assign sort keys
        n_orig = len(self.pdb)
        sort_key = np.empty(n_orig + n_h_placed, dtype=np.float64)
        sort_key[:n_orig] = np.arange(n_orig, dtype=np.float64)
        _, inv, counts = np.unique(
            insert_after, return_inverse=True, return_counts=True
        )
        cumcount = np.zeros(n_h_placed, dtype=np.float64)
        group_running = np.zeros(len(counts), dtype=np.float64)
        for i in range(n_h_placed):
            g = inv[i]
            cumcount[i] = group_running[g]
            group_running[g] += 1
        sort_key[n_orig:] = (
            insert_after + 0.5 + cumcount * (0.4 / np.maximum(counts[inv], 1))
        )

        augmented_df = pd.concat([self.pdb, h_df], ignore_index=True)
        augmented_df = augmented_df.iloc[
            np.argsort(sort_key, kind="stable")
        ].reset_index(drop=True)
        augmented_df["serial"] = np.arange(1, len(augmented_df) + 1)
        augmented_df["index"] = np.arange(len(augmented_df))

        for col in (
            "x",
            "y",
            "z",
            "occupancy",
            "tempfactor",
            "u11",
            "u22",
            "u33",
            "u12",
            "u13",
            "u23",
        ):
            augmented_df[col] = pd.to_numeric(
                augmented_df[col], errors="coerce"
            ).astype(float)
        augmented_df["serial"] = augmented_df["serial"].astype(int)
        augmented_df["resseq"] = augmented_df["resseq"].astype(int)
        augmented_df["charge"] = augmented_df["charge"].fillna(0).astype(int)
        augmented_df["anisou_flag"] = augmented_df["anisou_flag"].astype(bool)
        augmented_df[["altloc", "icode"]] = augmented_df[["altloc", "icode"]].fillna("")
        augmented_df["element"] = (
            augmented_df["element"].astype(str).str.strip().str.capitalize()
        )
        augmented_df.attrs["cell"] = self.pdb.attrs.get("cell")
        augmented_df.attrs["spacegroup"] = self.pdb.attrs.get("spacegroup", "P 1")

        new_model = self._new_model_from_df(augmented_df, strip_H=False)

        if verbose > 0:
            n_h = (new_model.pdb["element"] == "H").sum()
            print(f"  New model: {len(new_model.pdb)} atoms ({n_h} H)")

        # --- Phase C (optional): LBFGS geometry optimization ---
        if optimize:
            new_model.freeze_all()
            new_model.unfreeze_selection("element H", targets="xyz")
            refinable_params = [p for p in new_model.parameters() if p.numel() > 0]
            if refinable_params:
                try:
                    from torchref.refinement.targets.combined import (
                        TotalGeometryTarget,
                    )

                    geom_target = TotalGeometryTarget(new_model, verbose=0)
                    targets = {
                        n: geom_target[n]
                        for n in ("bond", "angle", "torsion", "chiral")
                    }

                    def _geom_loss():
                        total = torch.tensor(0.0, device=self.device)
                        for t in targets.values():
                            val = t()
                            if torch.isfinite(val):
                                total = total + val
                        return total

                    if verbose > 0:
                        with torch.no_grad():
                            init_l = _geom_loss()
                        print(f"  Geometry loss before: {init_l.item():.4f}")
                        for m in new_model.modules():
                            if hasattr(m, "reset_forward_cache"):
                                m.reset_forward_cache()

                    opt = torch.optim.LBFGS(
                        refinable_params,
                        lr=0.1,
                        max_iter=max_iter,
                        history_size=100,
                        line_search_fn="strong_wolfe",
                    )
                    best_loss = float("inf")
                    best_params = [p.data.clone() for p in refinable_params]

                    def closure():
                        opt.zero_grad()
                        loss = _geom_loss()
                        if loss.requires_grad and torch.isfinite(loss):
                            loss.backward()
                            for p in refinable_params:
                                if p.grad is not None:
                                    p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                        return loss

                    for _ in range(lbfgs_steps):
                        opt.step(closure)
                        with torch.no_grad():
                            cur = _geom_loss()
                        if torch.isfinite(cur) and cur.item() < best_loss:
                            best_loss = cur.item()
                            best_params = [p.data.clone() for p in refinable_params]
                    with torch.no_grad():
                        for p, bp in zip(refinable_params, best_params):
                            p.data.copy_(bp)
                    if verbose > 0:
                        with torch.no_grad():
                            fin_l = _geom_loss()
                        print(f"  Geometry loss after:  {fin_l.item():.4f}")
                except Exception as e:
                    if verbose > 0:
                        print(f"  Warning: optimization failed: {e}")
            new_model.set_default_masks()
            new_model.unfreeze_all()

        if verbose > 0:
            print("  Hydrogenation complete.")

        return new_model

    def adp_kl_divergence_loss(self, target_log_std: float = 0.2):
        """
        Compute KL divergence between log ADP distribution and target Gaussian.

        Regularizer measuring how far the spread of ``log(B)`` is from a Gaussian
        of the same (detached, so data-adaptive) mean and fixed
        ``target_log_std``::

            KL(q || p) = log(sigma_target/sigma_data)
                         + sigma_data^2 / (2 sigma_target^2) - 0.5

        Parameters
        ----------
        target_log_std : float, optional
            Target standard deviation in log space, default 0.2. Smaller means
            stronger regularization (a tighter ADP distribution).

        Returns
        -------
        torch.Tensor
            Scalar KL divergence, ``>= 0``, zero when the spreads match.

        Examples
        --------
        ::

            loss = xray_loss + w_adp * model.adp_kl_divergence_loss(0.2)
        """

        # The internal LOG-space values, not self.adp() (which is exp of them).
        log_adp = super(PositiveMixedTensor, self.adp).forward()

        mu_data = torch.mean(log_adp).detach()  # Detached: adapts to the data
        # Clamp the std off zero: a uniform ADP distribution (every B reset to one
        # constant) gives std=0, so KL=+inf, which LBFGS rejects as non-finite --
        # the B-factors then stay uniform and the divergence recurs every cycle.
        sigma_data = torch.std(log_adp).clamp(min=1e-6)

        mu_target = mu_data  # Same mean as data
        sigma_target = target_log_std

        # log(σ_target) as a Python scalar: synthesizing a CUDA tensor from a host
        # scalar per call is forbidden during CUDA Graph capture.
        import math

        log_sigma_target = math.log(float(sigma_target))
        log_sigma_ratio = log_sigma_target - torch.log(sigma_data)
        variance_ratio = (sigma_data**2) / (2 * sigma_target**2)

        kl_divergence = log_sigma_ratio + variance_ratio - 0.5

        return kl_divergence

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """
        Return a dictionary containing the complete state of the Model.

        Registered buffers, the four parameter wrappers, the PDB DataFrame and the
        metadata (space group as a string, cell as a CPU tensor, dtype, device,
        ``strip_H``, altloc pairs). Restore with :meth:`create_from_state_dict`,
        which is what knows how to rebuild the wrappers.

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
        state = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )

        state[prefix + "pdb"] = (
            self.pdb.copy() if hasattr(self, "pdb") and self.pdb is not None else None
        )
        state[prefix + "cell"] = self.cell.data.cpu() if self.cell is not None else None
        # As a string: gemmi.SpaceGroup is not picklable.
        state[prefix + "spacegroup"] = self.spacegroup.xhm if self.spacegroup else None
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
            Accepted for signature compatibility; the restore goes through
            :meth:`create_from_state_dict`, which is never strict.
        """
        state_dict = torch.load(path, map_location=self.device, weights_only=False)
        loaded = type(self).create_from_state_dict(
            state_dict, device=self.device, verbose=self.verbose
        )
        # Adopt the fully-built model's state wholesale.
        self.__dict__.update(loaded.__dict__)
        if self.verbose > 0:
            print(f"Loaded model state from {path}")

    @classmethod
    def create_from_state_dict(
        cls,
        state_dict: dict,
        device: torch.device = None,
        verbose: int = 1,
        dtype_float: torch.dtype = None,
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
            Device to place tensors on. Defaults to the configured device.current.
        verbose : int, optional
            Verbosity level. Default is 1.
        dtype_float : torch.dtype, optional
            Float dtype for tensors. Defaults to the configured dtypes.float.

        Returns
        -------
        Model
            Fully initialized instance with restored state.

        Notes
        -----
        Consumes ``state_dict``: the metadata keys are popped off it. The
        anisotropic ``u`` is rebuilt as a :class:`CholeskyMixedTensor`, matching
        :meth:`load`, so the positive-definite parametrization round-trips.
        """
        # Resolve dtype/device at call time so the fallbacks below use the
        # current config, not the import-time default.
        device = normalize_device(device)
        if dtype_float is None:
            dtype_float = get_float_dtype()
        pdb = state_dict.pop("pdb", None)
        cell_tensor = state_dict.pop("cell", None)
        spacegroup = state_dict.pop("spacegroup", None)
        initialized = state_dict.pop("initialized", False)
        saved_dtype = state_dict.pop("dtype_float", dtype_float)
        saved_device = state_dict.pop("device", device)
        strip_H = state_dict.pop("strip_H", True)
        altloc_pairs = state_dict.pop("altloc_pairs", [])

        instance = cls(
            dtype_float=saved_dtype, verbose=verbose, device=device, strip_H=strip_H
        )

        instance.pdb = pdb
        instance.initialized = initialized
        instance.altloc_pairs = altloc_pairs

        # Setter also sets symmetry.
        instance.spacegroup = spacegroup

        if cell_tensor is not None:
            instance.cell = Cell(cell_tensor, dtype=saved_dtype, device=device)

        # The wrappers are built from the PDB purely to get the right shapes and
        # masks; load_state_dict below overwrites their values.
        if pdb is not None:
            n_atoms = len(pdb)

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
            # Match load(): the anisotropic U is a CholeskyMixedTensor so the
            # restored model refines it in the same positive-definite-by-
            # construction parametrization as a freshly-loaded one.
            instance.u = CholeskyMixedTensor(
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
            # Pre-compute SF indices (respects exclude_H_from_sf)
            instance._rebuild_sf_indices()

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

        # Drop only empty-in-dim-0 tensors (placeholders from an atom-less state);
        # scalars and non-tensor entries must survive for load_state_dict.
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if not (torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] == 0)
        }
        instance.load_state_dict(state_dict, strict=False)

        if verbose > 0:
            n_atoms = len(instance.pdb) if instance.pdb is not None else 0
            print(f"Created Model from state_dict: {n_atoms} atoms")

        return instance

    def get_selection_mask(self, selection: str) -> torch.Tensor:
        """
        Return a boolean mask for atoms matching a Phenix-style selection.

        Wraps :func:`~torchref.utils.utils.parse_phenix_selection`; the result can
        be handed straight to ``MixedTensor.set()``.

        Parameters
        ----------
        selection : str
            Phenix-style selection: ``chain``, ``resseq`` (single or ``10:20``),
            ``resname``, ``name``, ``element``, ``altloc``, ``all``, combined with
            ``not`` / ``and`` / ``or`` and parentheses.

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

            mask = model.get_selection_mask("chain A and (resname ALA or resname GLY)")
            model.xyz.set(model.xyz()[mask] + translation, mask)
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

        An independent model with every per-atom tensor, buffer and metadata field
        subsetted, built as ``type(self)`` so subclasses return their own class.

        Parameters
        ----------
        selection : str
            Phenix-style selection; see :meth:`get_selection_mask` for the syntax.

        Returns
        -------
        Model
            New instance of the same class holding only the selected atoms.

        Raises
        ------
        RuntimeError
            If the model has not been initialized.
        ValueError
            If selection syntax is invalid or no atoms are selected.

        Notes
        -----
        The subclass constructor is called with the base kwargs only, so
        subclass-specific settings fall back to their defaults (see
        :meth:`ModelFT.select`). ``u`` is rebuilt as a plain
        :class:`MixedTensor`, *not* a :class:`CholeskyMixedTensor`, so the
        selected model loses the positive-definite parametrization of its
        anisotropic ADPs.

        Examples
        --------
        ::

            chain_a = model.select("chain A")
            no_water = model.select("not resname HOH")
        """
        from torchref.utils.utils import parse_phenix_selection

        if not self.initialized:
            raise RuntimeError(
                "Cannot select from an uninitialized Model. Load data first."
            )

        selection_mask = parse_phenix_selection(selection, self.pdb)

        n_selected = selection_mask.sum().item()
        if n_selected == 0:
            raise ValueError(f"Selection '{selection}' matched no atoms.")

        selected_indices = torch.where(selection_mask)[0]

        # type(self), so a subclass returns its own type.
        selected_model = type(self)(
            dtype_float=self.dtype_float,
            verbose=self.verbose,
            device=self.device,
            strip_H=self.strip_H,
        )

        # ``index`` must be renumbered: the occupancy grouping below reads it.
        mask_np = selection_mask.cpu().numpy()
        selected_model.pdb = self.pdb.loc[mask_np].copy()
        selected_model.pdb = selected_model.pdb.reset_index(drop=True)
        selected_model.pdb["index"] = selected_model.pdb.index.to_numpy(dtype=int)

        # Setter also sets symmetry; gemmi.SpaceGroup is immutable, so shared.
        selected_model.spacegroup = self.spacegroup

        # The fractional / reciprocal matrices are properties over the Cell, so
        # cloning the Cell carries all of them.
        if self.cell is not None:
            selected_model.cell = self.cell.clone()

        if hasattr(self, "aniso_flag") and self.aniso_flag is not None:
            selected_model.register_buffer(
                "aniso_flag", self.aniso_flag[selection_mask].clone()
            )
            # Pre-compute SF indices (respects exclude_H_from_sf)
            selected_model._rebuild_sf_indices()

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

        # Occupancy sharing/altloc groups must be rebuilt for the new numbering.
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

        selected_model.set_default_masks()
        selected_model.register_alternative_conformations()
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

        ``xyz_new = R @ (xyz - center) + center``, writing back through
        ``self.xyz[:]`` -- so this replaces ``refinable_params`` and invalidates
        any optimizer state built on it.

        Parameters
        ----------
        rotation_matrix : torch.Tensor
            3x3 rotation matrix. Should be orthogonal (R^T @ R = I).
        center : torch.Tensor, optional
            Center of rotation with shape (3,). Defaults to the centroid of all
            atomic coordinates.

        Returns
        -------
        Model
            Self, for method chaining.
        """
        if not self.initialized:
            raise RuntimeError("Model must be initialized to apply rotation.")

        xyz = self.xyz()
        if center is None:
            center = xyz.mean(dim=0)

        rotation_matrix = rotation_matrix.to(device=xyz.device, dtype=xyz.dtype)
        center = center.to(device=xyz.device, dtype=xyz.dtype)

        xyz_centered = xyz - center
        xyz_rotated = xyz_centered @ rotation_matrix.T + center

        self.xyz[:] = xyz_rotated

        return self

    def translate(self, translation: torch.Tensor, fractional: bool = False) -> "Model":
        """
        Apply translation to atomic coordinates (in-place).

        Writes back through ``self.xyz[:]``, so this replaces
        ``refinable_params`` and invalidates optimizer state built on it.

        Parameters
        ----------
        translation : torch.Tensor
            Translation vector with shape (3,).
        fractional : bool, optional
            If True, ``translation`` is fractional and converted to Cartesian
            first. Default False (Cartesian Ångströms).

        Returns
        -------
        Model
            Self, for method chaining.

        Examples
        --------
        ::

            model.translate(torch.tensor([5.0, 0.0, 0.0]))                  # 5 Å in x
            model.translate(torch.tensor([0.5, 0.5, 0.5]), fractional=True)  # half cell
        """
        if not self.initialized:
            raise RuntimeError("Model must be initialized to apply translation.")

        xyz = self.xyz()
        translation = translation.to(device=xyz.device, dtype=xyz.dtype)

        if fractional:
            # Convert fractional -> Cartesian. The orthogonalization matrix B
            # (fractional_matrix) follows the convention cart = frac @ B.T
            # (see Cell.fractional_to_cartesian); the transpose matters for
            # non-orthogonal (monoclinic/triclinic) cells.
            translation_cart = translation @ self.fractional_matrix.T
        else:
            translation_cart = translation

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

    def use_rigid_xyz(self) -> "Model":
        """
        Swap ``self.xyz`` for a per-chain :class:`RigidXYZTensor`.

        The only refinable leaves become per-chain Euler angles and translations,
        with chains auto-detected from ``self.pdb["chainid"]`` (waters and
        single-atom non-polymer residues are held fixed). The original container is
        stashed for :meth:`restore_xyz_from_rigid`.

        Also freezes ``adp`` / ``u`` / ``occupancy`` so only rigid-body parameters
        refine; :meth:`restore_xyz_from_rigid` re-enables exactly those that were
        refinable beforehand.

        Returns
        -------
        Model
            Self, for method chaining.
        """
        from torchref.model.rigid_xyz import RigidXYZTensor

        if not self.initialized:
            raise RuntimeError(
                "Model must be initialized before use_rigid_xyz(). "
                "Load data first with load_pdb() or load_cif()."
            )
        if isinstance(self.xyz, RigidXYZTensor):
            return self

        with torch.no_grad():
            current_xyz = self.xyz().detach().clone()
        chain_ids = list(self.pdb["chainid"].values)

        # Phenix-style polymer filter: drop waters and single-atom non-peptide
        # residues (ions), keep multi-atom HET ligands so they ride along with
        # their parent chain.
        _WATERS = {"HOH", "WAT", "DOD", "H2O"}
        _STD_POLYMER = {
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
            "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
            "TYR", "VAL", "MSE", "SEC", "PYL",
            "A", "C", "G", "T", "U", "I",
            "DA", "DC", "DG", "DT", "DU", "DI",
        }
        resname = self.pdb["resname"].astype(str).str.strip()
        is_water = resname.isin(_WATERS).values
        is_std = resname.isin(_STD_POLYMER).values
        residue_atom_count = (
            self.pdb.groupby(["chainid", "resseq", "icode"])["serial"].transform("count").values
        )
        is_single_atom = residue_atom_count == 1
        drop = is_water | (is_single_atom & ~is_std)
        mobile_arr = ~drop
        if mobile_arr.sum() == 0:
            raise RuntimeError(
                "Polymer filter removed every atom — cannot build rigid bodies."
            )
        mobile_mask = torch.from_numpy(mobile_arr).to(device=self.device)
        if self.verbose > 0 and int(drop.sum()) > 0:
            n_water = int(is_water.sum())
            n_ion = int((is_single_atom & ~is_std & ~is_water).sum())
            print(
                f"Rigid-body filter: {n_water} water + {n_ion} ion atoms "
                f"held fixed ({int(drop.sum())} total of {len(drop)})."
            )

        # Atomic Z stands in for mass: near-proportional for the elements that
        # dominate biological structures (C/N/O/S/P), so the rotation centre
        # becomes a centre of mass, as in Phenix.
        atom_weights = self.Z.to(dtype=self.dtype_float)

        rigid_xyz = RigidXYZTensor(
            original_xyz=current_xyz,
            chain_ids=chain_ids,
            dtype=self.dtype_float,
            device=self.device,
            mobile_mask=mobile_mask,
            atom_weights=atom_weights,
        )

        # Pop from _modules first, so assigning the new container registers a
        # submodule cleanly rather than colliding with the old one.
        self._rigid_original_xyz_container = self._modules.pop("xyz")
        self.xyz = rigid_xyz

        # Snapshot which groups were refinable BEFORE freezing them, so the
        # restore re-enables exactly those and leaves already-frozen ones alone.
        # Without it the handoff back to per-atom refinement builds an optimizer
        # over an empty parameter set and crashes.
        self._rigid_frozen_targets = [
            t
            for t in ("adp", "u", "occupancy")
            if getattr(self, t).refinable_params.numel() > 0
        ]

        self.freeze("adp")
        self.freeze("u")
        self.freeze("occupancy")

        if hasattr(self, "reset_cache"):
            self.reset_cache()

        if self.verbose > 0:
            print(
                f"Switched to rigid-body parametrization: {rigid_xyz} "
                f"({rigid_xyz.n_chains} chain(s))"
            )
        return self

    def restore_xyz_from_rigid(self, commit: bool = True) -> "Model":
        """
        Inverse of :meth:`use_rigid_xyz`.

        Parameters
        ----------
        commit : bool, optional
            If ``True`` (default), bake the current rotated/translated
            coordinates into a fresh :class:`MixedTensor` and install that
            as ``self.xyz``. If ``False``, restore the original container
            untouched (discarding the rigid transform).

        Returns
        -------
        Model
            Self, for method chaining.
        """
        from torchref.model.parameter_wrappers import MixedTensor
        from torchref.model.rigid_xyz import RigidXYZTensor

        if not isinstance(self.xyz, RigidXYZTensor):
            return self

        if commit:
            with torch.no_grad():
                current = self.xyz().detach().clone()
            new_xyz = MixedTensor(current, name="xyz", device=self.device)
            self._modules.pop("xyz", None)
            self.xyz = new_xyz
            xyz_mask = getattr(self, "xyz_mask", None)
            if xyz_mask is not None and xyz_mask.shape[0] == new_xyz.shape[0]:
                self.xyz.update_refinable_mask(xyz_mask)
            self.pdb.loc[:, ["x", "y", "z"]] = current.cpu().numpy()
        else:
            original = getattr(self, "_rigid_original_xyz_container", None)
            if original is None:
                raise RuntimeError(
                    "No stashed xyz container to restore. Did you call "
                    "use_rigid_xyz() first?"
                )
            self._modules.pop("xyz", None)
            self.xyz = original

        if hasattr(self, "_rigid_original_xyz_container"):
            del self._rigid_original_xyz_container

        # Re-enable exactly the groups use_rigid_xyz() froze, so subsequent
        # per-atom / ADP refinement has parameters to optimize.
        for target in getattr(self, "_rigid_frozen_targets", []):
            self.unfreeze(target)
        if hasattr(self, "_rigid_frozen_targets"):
            del self._rigid_frozen_targets

        if hasattr(self, "reset_cache"):
            self.reset_cache()
        return self
