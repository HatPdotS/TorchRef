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
from torchref.config import canonical_device, get_float_dtype, normalize_device
from torchref.io import cif, pdb
from torchref.model.context import ModelContext
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

    Owns the refinable atomic data -- coordinates, atomic displacement parameters and
    occupancies -- each held in a parameter wrapper that decides which atoms are
    refinable. Everything the structure was *loaded from* rather than refined lives on
    :attr:`ctx`, a :class:`~torchref.model.context.ModelContext`. Build the model empty
    (``Model()`` then ``load_pdb`` / ``load_cif`` / ``load_state_dict``); ``if model:``
    tests *initialization*, not existence.

    Parameters
    ----------
    dtype_float : torch.dtype, optional
        Data type for floating point tensors. Defaults to the configured dtypes.float.
    verbose : int, optional
        Verbosity level for logging. Default is 1.
    device : torch.device, optional
        Computation device. Defaults to the configured device.current.
    strip_H : bool, optional
        Whether to strip hydrogen atoms when loading. Default False: hydrogens are kept
        where the file has them and generated where it does not.
    add_hydrogens : bool, optional
        Generate hydrogens on load for residues that arrive without them. Default True;
        ignored when ``strip_H`` is set.

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
    ctx : ModelContext
        The unit cell, space group, atom table, link records, provenance and
        configuration. The fields not forwarded below are reached through it, e.g.
        ``model.ctx.strip_H`` and ``model.ctx.initialized``.
    pdb : pandas.DataFrame
        Atom table, forwarded to :attr:`ctx`. Only refreshed from the tensors by
        :meth:`update_pdb`.
    cell : Cell
        Unit cell, forwarded to :attr:`ctx`.
    spacegroup : SpaceGroup
        Space group, forwarded to :attr:`ctx`.
    device : torch.device
        Where the tensors live. Kept on the model rather than the context because the
        device-movement machinery rewrites it in place.
    """

    def __init__(
        self,
        dtype_float=None,
        verbose=1,
        device=None,
        strip_H: bool = False,
        add_hydrogens: bool = True,
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
            Whether to strip hydrogen atoms when loading. Default False: hydrogens are
            kept where the file has them and generated where it does not.
        add_hydrogens : bool, optional
            Generate hydrogens on load for residues that arrive without them. Default
            True; ignored when ``strip_H`` is set.
        """
        super().__init__()
        # Resolve dtype/device at call time (not import time) so a runtime
        # ``dtypes.float`` / ``device.current`` change is honored.
        if dtype_float is None:
            dtype_float = get_float_dtype()
        device = normalize_device(device)
        # ``device`` and ``dtype_float`` stay here rather than moving into the context:
        # they are live ``DeviceMixin`` trackers, rewritten in place by the traversal on
        # whichever object owns the tensors.
        self.dtype_float = dtype_float
        self.device = device

        # Everything the model is loaded from and sits in, as opposed to what is
        # refined. Populated by load() / create_from_state_dict().
        self.ctx = ModelContext(
            verbose=verbose, strip_H=strip_H, add_hydrogens=add_hydrogens
        )

        # Submodules (created during load or load_state_dict)
        self.xyz = None
        self.adp = None
        self.u = None
        self.occupancy = None

        # Scattering factor parametrization (built lazily on first access)
        self._parametrization = None

        # Restraints (built lazily on first access)
        self._restraints = None

    def __bool__(self):
        """Return the initialization status when used in boolean context.

        Note that ``if model:`` tests *initialization*, not non-``None``-ness:
        an uninitialized (but non-``None``) model is falsy. Use
        ``if model is not None`` when you mean an existence check.
        """
        return self.ctx.initialized

    @property
    def exclude_H_from_sf(self) -> bool:
        """Drop H from ``get_iso()`` / ``get_aniso()`` (so from Fcalc) while
        keeping them in the geometry and VDW restraints. Default False.
        """
        return self.ctx.exclude_H_from_sf

    @exclude_H_from_sf.setter
    def exclude_H_from_sf(self, value: bool):
        self.ctx.exclude_H_from_sf = bool(value)
        # The cached iso/aniso indices encode the H choice, so rebuild them.
        if self.ctx.initialized and self.pdb is not None:
            self._rebuild_sf_indices()

    def _rebuild_sf_indices(self):
        """Rebuild cached iso/aniso index arrays from aniso_flag and H mask."""
        iso_mask = ~self.aniso_flag
        aniso_mask = self.aniso_flag

        if self.ctx.exclude_H_from_sf and self.pdb is not None:
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
    def pdb(self) -> Optional["pandas.DataFrame"]:
        """Atom table. Only refreshed from the tensors by :meth:`update_pdb`."""
        return self.ctx.pdb

    @pdb.setter
    def pdb(self, value):
        self.ctx.pdb = value

    @property
    def cell(self) -> Optional[Cell]:
        """Unit cell object with parameters [a, b, c, alpha, beta, gamma]."""
        return self.ctx.cell

    @cell.setter
    def cell(self, value: Cell):
        """Set the unit cell."""
        self.ctx.cell = value

    @property
    def spacegroup(self) -> Optional[SpaceGroup]:
        """Space group object, or None if not set."""
        return self.ctx.spacegroup

    @spacegroup.setter
    def spacegroup(self, value):
        """Set the space group from a SpaceGroup, gemmi object, name or number."""
        if value is not None:
            # ``device=self.device``: SpaceGroup falls back to the global
            # default otherwise, so setting a spacegroup on a CPU-pinned Model
            # would silently plant accelerator-resident matrices on it.
            self.ctx.spacegroup = SpaceGroup(value, device=self.device)
        else:
            self.ctx.spacegroup = None

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

        if not self.ctx.initialized or self.pdb is None:
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

        if not self.ctx.initialized or self.pdb is None:
            raise RuntimeError(
                "Cannot build parametrization: model not initialized. "
                "Load data first with load_pdb() or load_cif()."
            )

        if self.ctx.verbose > 1:
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

        if self.ctx.verbose > 0:
            print(
                f"Parametrization built for {len(self._parametrization)} unique atom types"
            )
        if self.ctx.verbose > 1:
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
        self.ctx.cif_path = cif_path
        # Reset restraints so they will be rebuilt on next access
        self._restraints = None
        return self

    def _build_restraints(self):
        """Build and cache ``Restraints`` over this model's DataFrame, wiring in
        the live ``xyz`` / ``adp`` / ``vdw_radii`` callables.
        """
        if self._restraints is not None:
            return self._restraints

        if not self.ctx.initialized:
            raise RuntimeError(
                "Cannot build restraints: model not initialized. "
                "Load data first with load_pdb() or load_cif()."
            )

        from torchref.topology.restraints import Restraints

        if self.ctx.verbose > 0:
            print("Building restraints...")

        self._restraints = Restraints(
            pdb=self.pdb,
            cif_path=self.ctx.cif_path,
            xyz_fn=self.xyz,
            adp_fn=self.adp,
            vdw_radii_fn=self.get_vdw_radii,
            cell=self.ctx.cell,
            spacegroup=self.ctx.spacegroup,
            links=self.ctx.links,
            verbose=self.ctx.verbose,
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

    #: Per-atom buffers built lazily on first use and cached. Each is sized to the atom
    #: table, so all of them go stale the moment the atom set changes.
    _ATOM_DERIVED_BUFFERS = (
        "vdw_radii",
        "_Z",
        "_A",
        "_B",
        "_heavy_atom_mask",
    )

    def _invalidate_atom_derived_caches(self) -> None:
        """Drop the lazily-cached per-atom buffers.

        Each is guarded by ``hasattr`` and returned as-is once built, so a load that
        changes the atom count would otherwise hand back a buffer sized for the previous
        one. That surfaced when hydrogen generation began extending the table in place:
        the van der Waals radii stayed at the heavy-atom count while the pair list
        indexed the full set, and the non-bonded build raised ``IndexError``. Rebuilding
        a new model each time had hidden it.
        """
        for name in self._ATOM_DERIVED_BUFFERS:
            if hasattr(self, name):
                delattr(self, name)

    def load(self, reader, add_hydrogens: bool = None):
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
            optional ``.links`` attribute on it is stored on ``self.ctx.links``.
        add_hydrogens : bool, optional
            Whether to top up missing hydrogens once the model is built. Defaults to the
            context's setting, and is forced off for the re-entry that
            :meth:`_add_missing_hydrogens` makes, so generation happens once per load.

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
        if add_hydrogens is None:
            add_hydrogens = self.ctx.add_hydrogens and not self.ctx.strip_H
        self._invalidate_atom_derived_caches()
        self.pdb, cell, spacegroup = reader()
        self.ctx.links = getattr(reader, "links", None)

        self.pdb = (
            self.pdb.loc[self.pdb["element"] != "H"].reset_index(drop=True)
            if self.ctx.strip_H
            else self.pdb
        )
        self.pdb.dropna(subset=["x", "y", "z", "tempfactor", "occupancy"], inplace=True)
        # Reindex before deriving the ``index`` column: every consumer uses it to
        # address length-N per-atom tensors positionally (see
        # ``_create_occupancy_groups``), so a gapped index from the drop above sends
        # them past the end. Only the strip_H branch reset, so a model losing rows to
        # the dropna instead -- an atom with no coordinates or no B -- raised
        # IndexError at load. Hit on roughly one PDB-REDO entry in six.
        self.pdb.reset_index(drop=True, inplace=True)
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
        self.ctx.initialized = True

        if add_hydrogens:
            self._add_missing_hydrogens()
        return self

    def _add_missing_hydrogens(self) -> None:
        """Top up the hydrogens the atom table is missing, in place.

        Per parent, not per file: a structure deposited with some hydrogens gets the
        rest, because the plan only ever proposes a hydrogen the template names and the
        model does not have. 1AK5 arrives with 675 of roughly 2500, and a
        does-it-have-any test would have left it there.

        Re-enters :meth:`load` on the augmented atom table, which rebuilds the parameter
        wrappers and per-atom buffers at the new size. The re-entry is told not to
        consider hydrogens again, so this runs once per load rather than recursing to a
        fixed point.

        Costs a restraint build that is then discarded, because the plan needs the
        topology and the topology is built over the atoms as loaded. Set
        ``add_hydrogens=False`` to skip it for a model that will never be refined.
        """
        from torchref.topology.hydrogens import (
            augment_atom_table,
            optimise_free_torsions,
            plan_hydrogens,
        )

        restraints = self.restraints
        xyz = self.xyz().detach()
        plan = plan_hydrogens(
            restraints.topology, restraints.cif_dict, xyz, verbose=self.ctx.verbose
        )
        if plan.n_hydrogens == 0:
            return
        optimise_free_torsions(plan, restraints.topology, xyz)
        augmented = augment_atom_table(self.pdb, plan, restraints.topology)

        if self.ctx.verbose > 0:
            print(f"Generated {plan.n_hydrogens} hydrogens")

        # The topology and every per-atom tensor are sized for the old atom set.
        self._restraints = None
        cell, spacegroup = self.cell, self.spacegroup
        links = self.ctx.links

        def reader():
            return augmented, cell.data.cpu().numpy(), spacegroup

        # Carried explicitly: ``load`` reads links off the reader, so a bare callable
        # would drop the LINK records the first read resolved.
        reader.links = links
        self.load(reader, add_hydrogens=False)

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
        self.ctx.input_file = str(file)
        reader = pdb.PDBReader(verbose=self.ctx.verbose).read(file)
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
        self.ctx.input_file = str(file)
        if self.ctx.verbose > 0:
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

        if self.ctx.verbose > 1:
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

        Copies the live values of ``xyz`` (x/y/z), ``u`` (u11..u23) and
        ``occupancy`` from the parameter wrappers into the corresponding columns of
        the ``self.pdb`` DataFrame. Called by every writer and by ``hydrogenate``
        before output.

        ``tempfactor`` is the equivalent isotropic B whenever any atom is
        anisotropic, so the column agrees with the ANISOU records written beside it;
        with no anisotropic atoms it is the isotropic wrapper directly.

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
        # The B column must agree with the ANISOU records beside it: for an
        # anisotropic atom the PDB convention is B_eq = (8 pi^2 / 3) tr(U), not
        # whatever the isotropic wrapper still happens to hold. That wrapper stops
        # being refined the moment an atom goes anisotropic, so writing it directly
        # emits a stale B alongside a live U.
        if getattr(self, "_aniso_is_empty", True):
            self.pdb.loc[:, "tempfactor"] = self.adp().cpu().detach().numpy()
        else:
            from torchref.base.targets.adp import u6_b_eq

            self.pdb.loc[:, "tempfactor"] = (
                u6_b_eq(self.adp_u6()).cpu().detach().numpy()
            )
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
        if self.ctx.verbose > 0:
            print(f"Model moved to device: {self.device}")

    def copy(self):
        """
        Create a deep copy of the Model.

        Independent in every part: the context is copied via
        :meth:`~torchref.model.context.ModelContext.copy`, buffers are cloned and each
        parameter wrapper is copied through its own ``copy`` so its parametrization
        survives.

        Returns
        -------
        Model
            A new, fully independent Model instance with copied data.
        """
        if not self.ctx.initialized:
            raise RuntimeError("Cannot copy an uninitialized Model. Load data first.")

        model_copy = Model(
            dtype_float=self.dtype_float,
            verbose=self.ctx.verbose,
            device=self.device,
            strip_H=self.ctx.strip_H,
        )

        # One call carries the atom table, cell, space group, altloc groups and
        # provenance, each deep-copied or cloned -- see ``ModelContext.copy``.
        model_copy.ctx = self.ctx.copy()

        for buffer_name, buffer_value in self._buffers.items():
            if buffer_value is not None:
                model_copy.register_buffer(buffer_name, buffer_value.clone())

        # Parameter wrappers via their own .copy(), which preserves each
        # wrapper's parametrization (log-space, Cholesky, collapsed logits).
        for module_name, module in self._modules.items():
            if module is not None and hasattr(module, "copy"):
                setattr(model_copy, module_name, module.copy())

        # A wrapper that borrows the coordinates carries that reference through its
        # own ``copy``, so it still points at THIS model's ``xyz``. Re-point it, or
        # the two models silently share coordinates and the copy is not independent.
        for module in model_copy._modules.values():
            if module is not None and hasattr(module, "set_xyz_fn"):
                module.set_xyz_fn(model_copy.xyz)

        if self.ctx.verbose > 0:
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

    def set_adp_mode(
        self,
        mode: str = "isotropic",
        aniso_selection: str = None,
        n_nodes: int = None,
        k_neighbors: int = 12,
        refine_node_positions: bool = True,
        mode_set: str = None,
        init: str = "fit",
    ):
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
        mode : {"isotropic", "anisotropic", "field", "field_aniso", "preserve"}, optional
            ``"isotropic"`` (default) converts every atom, previously anisotropic
            ones to ``B_eq = (8 pi^2 / 3)(U11 + U22 + U33)``. ``"anisotropic"``
            converts those matching ``aniso_selection``, expanding isotropic atoms
            to ``U = (B / 8 pi^2) I``. ``"field"`` replaces the per-atom isotropic B
            with a :class:`~torchref.model.disorder_field.DisorderFieldTensor`, whose
            node values are least-squares fitted to the B it replaces, so the atom
            count stops setting the ADP parameter count.
            ``"preserve"`` is a no-op, leaving the ADPs exactly as the file supplied
            them: use it when the starting model's own ADPs are what is being measured.
        aniso_selection : str, optional
            Phenix-style selection for ``mode="anisotropic"``, default
            ``"not resname HOH and not element H"``; ignored otherwise.
        n_nodes : int, optional
            Nodes for ``mode="field"``. Defaults to one per 25 atoms, floored at 4.
        k_neighbors : int, optional
            Candidate nodes per atom for ``mode="field"``. Default 12.
        refine_node_positions : bool, optional
            Give each node a refinable offset from its anchor centroid, at three extra
            parameters per node. On by default: it is what lets the load-balancing
            restraint move a node toward atoms instead of only widening its kernel.
        init : {"fit", "flat"}, optional
            What a field mode fits its nodes to: ``"fit"`` (default) the model's current
            per-atom ADPs, ``"flat"`` a single level with their spatial structure
            discarded. See :meth:`_install_disorder_field`.
        mode_set : str, optional
            For ``mode="field_aniso"``, a key of
            :data:`~torchref.model.disorder_field.MODE_SETS` --- ``"rigid"`` is TLS,
            ``"affine"`` adds shear and extension. The node then stores the covariance
            of its displacement modes, so the U it gives an atom depends on where that
            atom sits inside the node's region rather than being constant across it.
            Default ``None`` keeps the constant-U payload.

        Notes
        -----
        Run once at model setup, before scaling / restraints / targets. The
        isotropic result matches a freshly-loaded isotropic-only model.

        Leaving ``"field"`` needs no special case: the conversion reads ``adp()``,
        which a field evaluates per atom, so the field materialises into a per-atom
        wrapper on the way out.
        """
        if not self.ctx.initialized or self.pdb is None:
            return
        if mode == "preserve":
            # Leave the ADPs exactly as loaded. Constructing a Refinement otherwise
            # reparametrises them before anything else runs, which silently discards a
            # deposited model's anisotropy -- use this when the starting model's own
            # ADPs are the thing being measured.
            return
        if mode in ("field", "field_aniso"):
            aniso = mode == "field_aniso"
            # Run the partition first either way: it owns every buffer keyed off the
            # iso/aniso split, and it converts the stored values in the right direction
            # (B -> U_iso*I entering anisotropic, U -> B_eq entering isotropic), so the
            # field is fitted to a target that is already in its own representation.
            if aniso:
                # Every atom, unless the caller narrows it. A node field is not the
                # per-atom parametrisation that "not water, not hydrogen" exists to
                # ration -- its cost is set by node count, not atom count -- and a
                # partial selection would leave half the ADPs coming from the field and
                # half from the per-atom wrapper, which is not a representation anyone
                # asked for.
                if aniso_selection is None:
                    target_mask = torch.ones(
                        len(self.pdb), dtype=torch.bool, device=self.device
                    )
                else:
                    from torchref.utils.utils import create_selection_mask

                    target_mask = torch.as_tensor(
                        create_selection_mask(aniso_selection, self.pdb),
                        dtype=torch.bool,
                    ).to(self.device)
            else:
                target_mask = torch.zeros(
                    len(self.pdb), dtype=torch.bool, device=self.device
                )
            self._apply_adp_partition(target_mask)
            self._install_disorder_field(
                n_nodes=n_nodes,
                k_neighbors=k_neighbors,
                refine_node_positions=refine_node_positions,
                anisotropic=aniso,
                mode_set=mode_set,
                init=init,
            )
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
                f"Unknown ADP mode: {mode!r}. Use 'isotropic', 'anisotropic', "
                "'field', 'field_aniso' or 'preserve'."
            )
        self._apply_adp_partition(aniso_mask)

    @property
    def adp_is_field(self) -> bool:
        """Whether either ADP slot holds a node field rather than a per-atom wrapper."""
        from torchref.model.disorder_field import DisorderFieldTensor

        return isinstance(self.adp, DisorderFieldTensor) or isinstance(
            self.u, DisorderFieldTensor
        )

    @property
    def adp_field(self):
        """The node field driving the ADPs, or ``None`` if neither slot holds one."""
        from torchref.model.disorder_field import DisorderFieldTensor

        for wrapper in (self.u, self.adp):
            if isinstance(wrapper, DisorderFieldTensor):
                return wrapper
        return None

    def _install_disorder_field(
        self,
        n_nodes: int = None,
        k_neighbors: int = 12,
        refine_node_positions: bool = False,
        anisotropic: bool = False,
        mode_set: str = None,
        init: str = "fit",
    ):
        """Replace a per-atom ADP wrapper with a node field fitted to it.

        The field lands in the slot its payload feeds: an isotropic payload takes over
        ``adp`` and leaves the model isotropic, an anisotropic one takes over ``u`` and
        the model refines every selected atom anisotropically. Both expect the partition
        to have run first, which :meth:`set_adp_mode` arranges.

        ``mode_set`` selects a displacement-mode payload in place of the constant-U one,
        which is the difference between a node holding a single ADP and a node holding a
        motion whose ADP varies across its region.

        ``init`` chooses what the field is fitted to:

        ``"fit"``
            The per-atom ADPs the model currently holds. Right when those mean something
            --- a deposited or already-refined model --- because the field then starts
            from a state whose R-factor is known.
        ``"flat"``
            A single value, the median of those ADPs. Right when they do not mean
            anything. An AlphaFold model's B values come from a pLDDT conversion, and
            fitting a smooth basis to them spends the field's parameters reproducing
            structure it cannot hold and that is not worth holding: measured on 2A25, the
            fitted field starts 0.025 R-free WORSE than a flat one, before any
            refinement. The level is kept because it is close to right and the scaler
            owns it anyway; only the spatial structure is discarded.
        """
        from torchref.model.disorder_field import (
            AnisotropicPayload,
            DisorderFieldTensor,
            IsotropicPayload,
            ModeCovariancePayload,
            density_anchor_rows,
        )

        if mode_set is not None and not anisotropic:
            raise ValueError(
                "mode_set describes an anisotropic displacement field and has no "
                "isotropic form; use mode='field_aniso'."
            )

        with torch.no_grad():
            xyz = self.xyz().detach()
            # The fit target is whatever the partition just produced: per-atom U6 for
            # the anisotropic payload, per-atom B for the isotropic one.
            target = (
                self.adp_u6().detach().clone()
                if anisotropic
                else self.adp().detach().clone()
            )
            if init == "flat":
                # Flatten through the equivalent isotropic B, and hand the payload a 1-D
                # target: its ``fit`` lifts that to U_iso * I. Taking a median over all
                # six U components instead would set the off-diagonals equal to the
                # diagonals, giving eigenvalues (3L, 0, 0) -- singular, and NaN once the
                # Cholesky encode takes log(diag - epsilon).
                b = (
                    (8.0 * math.pi**2 / 3.0) * target[:, :3].sum(dim=1)
                    if target.ndim == 2
                    else target
                )
                finite = torch.isfinite(b)
                if not bool(finite.any()):
                    raise ValueError("cannot flatten an all-NaN ADP target")
                level = b[finite].median()
                target = torch.where(finite, level.expand_as(b), b)
            elif init != "fit":
                raise ValueError(
                    f"init={init!r}; expected 'fit' (use the model's own ADPs) or "
                    "'flat' (discard their spatial structure, keep the level)."
                )
            B = target
        if n_nodes is None:
            n_nodes = max(4, int(round(len(self.pdb) / 25.0)))

        # Anchor on density clusters, not single atoms: a node placed exactly on an atom
        # can isolate that atom by narrowing its kernel, which is per-atom refinement
        # wearing a node's clothes.
        anchor_rows = density_anchor_rows(xyz, min(n_nodes, len(self.pdb)))

        if mode_set is not None:
            payload = ModeCovariancePayload(mode_set)
        elif anisotropic:
            payload = AnisotropicPayload()
        else:
            payload = IsotropicPayload()

        field = DisorderFieldTensor(
            initial_values=target.to(self.dtype_float),
            xyz_fn=self.xyz,
            n_nodes=n_nodes,
            refine_positions=refine_node_positions,
            payload=payload,
            anchor_rows=anchor_rows,
            k_neighbors=k_neighbors,
            name="aniso_U" if anisotropic else "adp",
            dtype=self.dtype_float,
            device=self.device,
        )
        if anisotropic:
            self.u = field
            # The mask is in atom space either way; the field collapses it onto nodes.
            self.u.update_refinable_mask(self.u_mask)
        else:
            self.adp = field
            self.adp.update_refinable_mask(self.adp_mask)

        if self.ctx.verbose > 0:
            kind = mode_set if mode_set else ("aniso U" if anisotropic else "iso B")
            was = len(self.pdb) * (6 if anisotropic else 1)
            print(
                f"ADP field ({kind}): {field.n_nodes} nodes, k={k_neighbors}, "
                f"{int(field.get_refinable_count())} refinable nodes, "
                f"{int(field.refinable_params.numel())} parameters "
                f"(was {was} per-atom)"
            )
        if hasattr(self, "reset_cache"):
            self.reset_cache()

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

        if self.ctx.verbose > 0:
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

        if self.ctx.verbose > 0:
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
        Rebuild ``self.ctx.altloc_pairs`` from the ``altloc`` column.

        One tuple per residue that has multiple conformations, holding one
        index tensor per conformation (in sorted altloc order), e.g.
        ``[(tensor([100, 101]), tensor([110, 111])), ...]``. Overwrites any
        previous content, so call it after the atom numbering changes.
        """
        self.ctx.altloc_pairs = []

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

                self.ctx.altloc_pairs.append(tuple(conformation_tensors))

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


    def _new_model_from_df(self, df, *, strip_H=None, add_hydrogens=False):
        """Build a fresh model of the same class from a DataFrame.

        ``add_hydrogens`` defaults to False, unlike the constructor: the caller has
        already settled which atoms the table holds, and generating more would fight
        that. :meth:`hydrogenate` passes an already-augmented table for the same reason.
        """
        import inspect

        sh = self.ctx.strip_H if strip_H is None else strip_H
        ctor_kw = dict(
            dtype_float=self.dtype_float,
            verbose=0,
            device=self.device,
            strip_H=sh,
            add_hydrogens=add_hydrogens,
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
        if self.ctx.cif_path is not None:
            new_model._cif_path = self.ctx.cif_path
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

    def hydrogenate(self, verbose: int = 0, optimize: bool = True) -> "Model":
        """Return a new model with hydrogens added from the monomer templates.

        Hydrogen generation is template instantiation over the topology: each residue's
        library template is aligned onto the heavy atoms present and its hydrogens read
        off, and the bond graph decides how many hydrogens a parent can carry and which
        of them have a free torsion. The original model is not modified.

        Parameters
        ----------
        verbose : int, default 0
            Verbosity level.
        optimize : bool, default True
            Scan each free torsion -- hydroxyl, thiol, amine, methyl -- for the
            least-clashing angle. The template's dihedral for those is arbitrary, so this
            is on by default; it is a rotation about one bond and costs little.

        Returns
        -------
        Model
            New model with hydrogens, built with ``strip_H=False`` so they survive the
            load.
        """
        from torchref.topology.hydrogens import (
            augment_atom_table,
            optimise_free_torsions,
            plan_hydrogens,
        )

        self.update_pdb()
        restraints = self.restraints  # builds the topology this reads
        xyz = self.xyz().detach()

        plan = plan_hydrogens(
            restraints.topology, restraints.cif_dict, xyz, verbose=verbose
        )
        if optimize:
            optimise_free_torsions(plan, restraints.topology, xyz)

        if verbose > 0:
            print(f"Adding {plan.n_hydrogens} hydrogens")
        augmented = augment_atom_table(self.pdb, plan, restraints.topology)
        return self._new_model_from_df(augmented, strip_H=False)


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

        state[prefix + "pdb"] = self.pdb.copy() if self.pdb is not None else None
        state[prefix + "cell"] = self.cell.data.cpu() if self.cell is not None else None
        # As a string: gemmi.SpaceGroup is not picklable.
        state[prefix + "spacegroup"] = self.spacegroup.xhm if self.spacegroup else None
        state[prefix + "initialized"] = self.ctx.initialized
        state[prefix + "dtype_float"] = self.dtype_float
        state[prefix + "device"] = self.device
        state[prefix + "strip_H"] = self.ctx.strip_H
        state[prefix + "altloc_pairs"] = self.ctx.altloc_pairs

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
        if self.ctx.verbose > 0:
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
            state_dict, device=self.device, verbose=self.ctx.verbose
        )
        # Adopt the fully-built model's state wholesale.
        self.__dict__.update(loaded.__dict__)
        if self.ctx.verbose > 0:
            print(f"Loaded model state from {path}")

    @staticmethod
    def _restore_adp_slot(prefix, state_dict, pdb, saved_dtype, xyz_wrapper):
        """Rebuild the ``adp`` or ``u`` wrapper, as a node field when the state was one.

        Built from the PDB for its shapes and masks only; ``load_state_dict`` overwrites
        every value afterwards.

        A saved :class:`~torchref.model.disorder_field.DisorderFieldTensor` is recognised
        by its ``neighbor_list``, not by the shape of its storage: the ``u`` slot holds a
        2-D tensor either way, so shape alone cannot tell a ``(K, 10)`` node field from a
        ``(n_atoms, 6)`` per-atom U.

        Parameters
        ----------
        prefix : {"adp", "u"}
            Which slot to rebuild. ``"u"`` carries the anisotropic representation.
        state_dict : dict
            The state being restored, read but not consumed.
        pdb : pandas.DataFrame
            Atom table supplying the initial values.
        saved_dtype : torch.dtype
            Float dtype the state was saved in.
        xyz_wrapper : MixedTensor
            The already-rebuilt coordinate wrapper; a node field derives its node
            positions from it.
        """
        from torchref.model.parameter_wrappers import (
            CholeskyMixedTensor,
            PositiveMixedTensor,
        )

        aniso = prefix == "u"
        name = "aniso_U" if aniso else "adp"
        mask = state_dict.get(f"{prefix}.refinable_mask")
        if aniso:
            initial = torch.tensor(
                pdb[["u11", "u22", "u33", "u12", "u13", "u23"]].values,
                dtype=saved_dtype,
            )
        else:
            initial = torch.tensor(pdb["tempfactor"].values, dtype=saved_dtype)

        saved_nl = state_dict.get(f"{prefix}.neighbor_list")
        if saved_nl is None:
            # Match load(): the anisotropic U is a CholeskyMixedTensor so a restored
            # model refines it in the same positive-definite-by-construction
            # parametrization as a freshly-loaded one.
            wrapper = CholeskyMixedTensor if aniso else PositiveMixedTensor
            return wrapper(initial, refinable_mask=mask, name=name)

        from torchref.model.disorder_field import (
            AnisotropicPayload,
            DisorderFieldTensor,
            IsotropicPayload,
            payload_from_code,
        )

        # The saved code names the payload exactly. Fall back to inferring it from the
        # slot for state dicts written before the code existed, where the only payloads
        # were the two the slot already implies.
        saved_code = state_dict.get(f"{prefix}.payload_code")
        if saved_code is not None:
            payload = payload_from_code(int(saved_code))
        else:
            payload = AnisotropicPayload() if aniso else IsotropicPayload()
        saved_values = state_dict[f"{prefix}.fixed_values"]
        # Rebuild with the SAVED anchor rows: cluster anchoring makes these length
        # n_atoms where single-atom anchoring makes them length K, so reconstructing
        # them from scratch would shape-mismatch on load.
        saved_anchor_atom = state_dict.get(f"{prefix}.anchor_atom")
        saved_anchor_node = state_dict.get(f"{prefix}.anchor_node")
        return DisorderFieldTensor(
            initial_values=initial,
            xyz_fn=xyz_wrapper,
            n_nodes=int(saved_values.shape[0]),
            k_neighbors=int(saved_nl.shape[1]),
            payload=payload,
            # Storage is [payload | log sigma | offset], so the extra three columns
            # say whether node positions carry a refinable offset.
            refine_positions=bool(saved_values.shape[1] == payload.width + 4),
            anchor_rows=(
                (saved_anchor_atom, saved_anchor_node)
                if saved_anchor_atom is not None
                else None
            ),
            refinable_mask=mask,
            mask_in_node_space=True,
            name=name,
            dtype=saved_dtype,
        )

    @classmethod
    def _rebuild_wrappers_from_pdb(cls, instance, pdb, state_dict, saved_dtype, device):
        """Give ``instance`` parameter wrappers and per-atom buffers of the right shape.

        The half of :meth:`create_from_state_dict` that every subclass needs
        identically, so subclasses call this rather than restating it: a per-class copy
        drifts, and a restore that rebuilds the wrong wrapper type fails on a shape
        mismatch rather than on anything that names the real cause.

        Values are placeholders throughout --- the caller's ``load_state_dict`` is what
        puts the saved numbers in. Only shapes, masks and dtypes matter here.
        """
        from torchref.model.parameter_wrappers import MixedTensor, OccupancyTensor

        n_atoms = len(pdb)

        instance.xyz = MixedTensor(
            torch.tensor(pdb[["x", "y", "z"]].values, dtype=saved_dtype),
            refinable_mask=state_dict.get("xyz.refinable_mask"),
            name="xyz",
        )
        instance.adp = cls._restore_adp_slot(
            "adp", state_dict, pdb, saved_dtype, instance.xyz
        )
        instance.u = cls._restore_adp_slot(
            "u", state_dict, pdb, saved_dtype, instance.xyz
        )

        initial_occ = torch.tensor(pdb["occupancy"].values, dtype=saved_dtype)
        sharing_groups, altloc_groups, refinable_mask = (
            instance._create_occupancy_groups(pdb, initial_occ)
        )
        # A saved mask is in group space; expand it back over atoms.
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

        if "aniso_flag" not in instance._buffers or instance.aniso_flag is None:
            instance.register_buffer(
                "aniso_flag",
                torch.tensor(pdb["anisou_flag"].values, dtype=torch.bool),
            )
        # Pre-compute SF indices (respects exclude_H_from_sf)
        instance._rebuild_sf_indices()

        for mask_name in ("xyz_mask", "adp_mask", "u_mask", "occupancy_mask"):
            instance.register_buffer(
                mask_name, torch.ones(n_atoms, dtype=torch.bool, device=device)
            )

        # Note: inv_fractional_matrix, fractional_matrix and recB are properties
        # delegating to Cell, so they are not registered as buffers.
        if state_dict.get("vdw_radii") is not None:
            instance.register_buffer(
                "vdw_radii", torch.zeros_like(state_dict["vdw_radii"], device=device)
            )

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
            Move the restored model here once it is built. The restore itself always
            runs on CPU, and ``None`` leaves it there rather than resolving to
            ``device.current`` -- loading a file is not a reason to claim an
            accelerator. Move it yourself, or pass one here.
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
        # Build on CPU throughout, then move once at the end if the caller named a
        # device. One device for the whole model is the invariant that matters: the
        # wrappers are built from the atom table and land on CPU whatever is asked for,
        # so resolving an accelerator up front splits the model rather than placing it.
        target_device = canonical_device(device) if device is not None else None
        device = torch.device("cpu")
        if dtype_float is None:
            dtype_float = get_float_dtype()
        pdb = state_dict.pop("pdb", None)
        cell_tensor = state_dict.pop("cell", None)
        spacegroup = state_dict.pop("spacegroup", None)
        initialized = state_dict.pop("initialized", False)
        saved_dtype = state_dict.pop("dtype_float", dtype_float)
        state_dict.pop("device", None)  # popped so it never reaches load_state_dict
        strip_H = state_dict.pop("strip_H", True)
        altloc_pairs = state_dict.pop("altloc_pairs", [])

        instance = cls(
            dtype_float=saved_dtype, verbose=verbose, device=device, strip_H=strip_H
        )

        instance.pdb = pdb
        instance.ctx.initialized = initialized
        instance.ctx.altloc_pairs = altloc_pairs

        # Setter also sets symmetry.
        instance.spacegroup = spacegroup

        if cell_tensor is not None:
            instance.cell = Cell(cell_tensor, dtype=saved_dtype, device=device)

        # The wrappers are built from the PDB purely to get the right shapes and
        # masks; load_state_dict below overwrites their values.
        if pdb is not None:
            cls._rebuild_wrappers_from_pdb(instance, pdb, state_dict, saved_dtype, device)

        # Drop only empty-in-dim-0 tensors (placeholders from an atom-less state);
        # scalars and non-tensor entries must survive for load_state_dict.
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if not (torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] == 0)
        }
        instance.load_state_dict(state_dict, strict=False)

        if target_device is not None:
            instance.to(target_device)

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

        if not self.ctx.initialized:
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

        if not self.ctx.initialized:
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
            verbose=self.ctx.verbose,
            device=self.device,
            strip_H=self.ctx.strip_H,
        )

        # ``index`` must be renumbered: the occupancy grouping below reads it.
        mask_np = selection_mask.cpu().numpy()
        selected_model.pdb = self.pdb.loc[mask_np].copy()
        selected_model.pdb = selected_model.pdb.reset_index(drop=True)
        selected_model.pdb["index"] = selected_model.pdb.index.to_numpy(dtype=int)

        # The setter rebuilds a SpaceGroup, so the selection gets its own.
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
        selected_model.ctx.initialized = True

        if self.ctx.verbose > 0:
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
        if not self.ctx.initialized:
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
        if not self.ctx.initialized:
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
        if not self.ctx.initialized:
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
        if not self.ctx.initialized:
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

        if not self.ctx.initialized:
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
        if self.ctx.verbose > 0 and int(drop.sum()) > 0:
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

        if self.ctx.verbose > 0:
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
