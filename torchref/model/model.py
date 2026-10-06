"""The atomic model: refinable parameters over a loaded structure.

:class:`Model` holds coordinates, isotropic and anisotropic ADPs and occupancies as
parameter wrappers that decide which atoms are refinable, over a
:class:`~torchref.model.context.ModelContext` that carries what the structure was
loaded with: cell, space group, atom identity and restraints.
:class:`~torchref.model.model_ft.ModelFT` adds the structure factors.
"""

import math
import warnings
from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from torchref.base import math_torch
from torchref.config import (
    canonical_device,
    get_default_device,
    get_float_dtype,
    get_int_dtype,
    normalize_device,
)
from torchref.io import cif, pdb
from torchref.model.context import AtomValues, ModelContext, own_spacegroup
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

if TYPE_CHECKING:
    import pandas


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
    hydrogens : {"keep", "add", "strip"}, optional
        What loading does with the file's hydrogens: keep them (default), also generate
        the missing ones from the monomer templates, or remove them all.
    hydrogen_mode : {"atoms", "riding"}, optional
        Hydrogens as ordinary refinable atoms (default) or riding on their parents.
        ``"riding"`` with ``hydrogens="strip"`` raises ``ValueError``.
    cif_path : str or list of str, optional
        Restraint dictionary file(s) for residues the monomer library does not know, or
        whose library entry should be overridden. Given here rather than after loading so
        that hydrogen generation on load reads the same dictionary the restraints will.
    hydrogens_in_xray : bool, optional
        Whether hydrogens contribute to the structure factors. Default True.

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
        The unit cell, space group, atom identity (``ctx.topology``), link records,
        provenance and configuration. The fields not forwarded below are reached
        through it, e.g. ``model.ctx.hydrogens`` and ``model.ctx.initialized``.
    n_atoms : int
        Number of atoms.
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
        hydrogens: str = "keep",
        hydrogen_mode: str = "atoms",
        cif_path: Optional[Union[str, List[str]]] = None,
        hydrogens_in_xray: bool = True,
    ):
        """Initialize an empty Model shell; see the class docstring for the arguments.

        Load a structure with :meth:`load_pdb` / :meth:`load_cif`, or restore one with
        :meth:`create_from_state_dict`.
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

        # Settings only until a structure is loaded, which replaces the context with
        # one built by ``ModelContext.from_atoms``.
        self.ctx = ModelContext(
            verbose=verbose,
            hydrogens=hydrogens,
            hydrogen_mode=hydrogen_mode,
            cif_path=cif_path,
            hydrogens_in_xray=hydrogens_in_xray,
        )

        # Parameter wrappers, installed by _install_parameters.
        self.xyz = None
        self.adp = None
        self.u = None
        self.occupancy = None

    def __bool__(self):
        """Return the initialization status when used in boolean context.

        Note that ``if model:`` tests *initialization*, not non-``None``-ness:
        an uninitialized (but non-``None``) model is falsy. Use
        ``if model is not None`` when you mean an existence check.
        """
        return self.ctx.initialized

    @property
    def hydrogens_in_xray(self) -> bool:
        """Whether hydrogens enter ``get_iso()`` / ``get_aniso()`` and so Fcalc.

        Restraints and the non-bonded term see the hydrogens either way, and the
        bulk-solvent mask never does. Default True. Changing it re-keys the
        iso/aniso partition on the next access; no cache needs clearing.
        """
        return self.ctx.hydrogens_in_xray

    @hydrogens_in_xray.setter
    def hydrogens_in_xray(self, value: bool):
        self.ctx.hydrogens_in_xray = bool(value)

    def _sf_atom_mask(self) -> Optional[torch.Tensor]:
        """Atoms that enter Fcalc, or None when every atom does.

        Boolean ``(N,)`` over the atom table. Built lazily as the ``_heavy_atom_mask``
        buffer, which is dropped with the other per-atom caches when the atom set
        changes, so it never outlives the table it was built for.
        """
        if self.ctx.hydrogens_in_xray or self.ctx.topology is None:
            return None
        if getattr(self, "_heavy_atom_mask", None) is None:
            self.register_buffer(
                "_heavy_atom_mask",
                ~self.ctx.topology.atoms.is_hydrogen.to(self.device),
            )
        return self._heavy_atom_mask

    # -- iso/aniso partition, derived on access ---------------------------
    #
    # These four are a cache over ``aniso_flag`` and the H choice, and caches in
    # this codebase are recomputed on access rather than copied. Keying them on a
    # fingerprint of their inputs means there is no invalidation to remember:
    # every mutation that could change them changes the fingerprint, including an
    # in-place edit of ``aniso_flag`` (``_version`` moves) and a whole-tensor
    # replacement (``data_ptr`` moves).
    #
    # Eager rebuilding is what made ``copy()`` fragile. A fresh copy is
    # constructed, then has its context replaced and its buffers cloned, so
    # indices built during construction describe the wrong ``aniso_flag`` --
    # and they do not raise, they silently gather the wrong atoms, with
    # ``_aniso_is_empty`` able to skip anisotropic atoms outright.

    def _sf_partition(self):
        """``(iso_idx, aniso_idx, iso_covers_all, aniso_is_empty)``, cached."""
        flag = self.aniso_flag
        heavy = getattr(self, "_heavy_atom_mask", None)
        fp = (
            (flag.data_ptr(), flag._version) if flag is not None else None,
            bool(self.ctx.hydrogens_in_xray),
            None if heavy is None else (heavy.data_ptr(), heavy._version),
            self.n_atoms,
        )
        cached = getattr(self, "_sf_partition_cache", None)
        if cached is not None and self._sf_partition_fp == fp:
            return cached

        iso_mask = ~flag
        aniso_mask = flag
        sf_atoms = self._sf_atom_mask()
        if sf_atoms is not None:
            # The mask is part of the key, so re-key after building it.
            fp = (fp[0], fp[1], (sf_atoms.data_ptr(), sf_atoms._version), fp[3])
            iso_mask = iso_mask & sf_atoms
            aniso_mask = aniso_mask & sf_atoms

        iso_idx = iso_mask.nonzero(as_tuple=True)[0]
        aniso_idx = aniso_mask.nonzero(as_tuple=True)[0]
        # Fast-path flags: an everywhere-True iso_mask lets ``get_iso()`` skip the
        # gather (and its ``index_put_`` backward) entirely, and
        # ``_aniso_is_empty`` lets ``get_aniso()`` short-circuit -- the typical
        # macromolecular case.
        out = (iso_idx, aniso_idx,
               bool(iso_mask.all().item()), int(aniso_idx.numel()) == 0)
        self._sf_partition_cache = out
        self._sf_partition_fp = fp
        return out

    @property
    def _iso_indices(self) -> torch.Tensor:
        return self._sf_partition()[0]

    @property
    def _aniso_indices(self) -> torch.Tensor:
        return self._sf_partition()[1]

    @property
    def _iso_covers_all(self) -> bool:
        return self._sf_partition()[2]

    @property
    def _aniso_is_empty(self) -> bool:
        return self._sf_partition()[3]

    # =========================================================================
    # Cell, SpaceGroup, and Symmetry properties
    # =========================================================================

    @property
    def n_atoms(self) -> int:
        """Number of atoms; 0 before a structure is loaded."""
        return self.ctx.n_atoms

    @property
    def pdb(self) -> Optional["pandas.DataFrame"]:
        """The atom table, freshly joined from identity and current values.

        .. deprecated::
            Use :meth:`to_dataframe` for output and ``model.ctx.topology`` for atom
            identity. Each access builds a new table, so writing into it changes
            nothing.
        """
        warnings.warn(
            "Model.pdb is deprecated: use model.to_dataframe() for a table and "
            "model.ctx.topology for atom identity",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.to_dataframe() if self.ctx.topology is not None else None

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
        """Set the space group from a SpaceGroup, gemmi object, name or number.

        The model owns its space group: an incoming ``SpaceGroup`` is copied rather
        than shared (see :func:`~torchref.model.context.own_spacegroup`), and lands on
        the model's device and float dtype.
        """
        self.ctx.spacegroup = own_spacegroup(value, self.dtype_float, self.device)

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

        if not self.ctx.initialized:
            raise RuntimeError(
                "Cannot build Z tensor: model not initialized. "
                "Load data first with load_pdb() or load_cif()."
            )

        from torchref.base.scattering.scattering_table import get_element_to_z_mapping

        element_to_z = get_element_to_z_mapping()
        z_values = [
            element_to_z.get(elem.strip().capitalize(), 0)
            for elem in self.ctx.topology.atoms.element
        ]
        self.register_buffer(
            "_Z", torch.tensor(z_values, dtype=get_int_dtype(), device=self.device)
        )
        return self._Z

    # =========================================================================
    # Scattering Factor Parametrization
    # =========================================================================

    def _build_parametrization(self) -> None:
        """Register the ``_A`` / ``_B`` ITC92 buffers by Z-based table lookup.

        Called lazily on first access to the scattering parameters; the buffers are
        kept until the atom set changes.
        """
        if getattr(self, "_A", None) is not None:
            return

        if not self.ctx.initialized:
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

        if self.ctx.verbose > 0:
            elements = sorted(set(self.ctx.topology.atoms.element))
            print(f"Parametrization built for {len(elements)} unique atom types")
            if self.ctx.verbose > 1:
                print("Elements with parametrization:", elements)

    @property
    def parametrization(self) -> dict:
        """ITC92 ``{element: (A, B)}``, each a ``(1, 5)`` row of the per-atom buffers.

        Built on each access; the rows alias the ``_A`` / ``_B`` buffers, which are
        what the structure-factor code reads.
        """
        self._build_parametrization()
        elements, first = np.unique(self.ctx.topology.atoms.element, return_index=True)
        return {
            str(elem): (self._A[i : i + 1], self._B[i : i + 1])
            for elem, i in zip(elements, first)
        }

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
        ``n_iso_atoms`` honors ``hydrogens_in_xray``: when hydrogens are excluded
        the isotropic count is the heavy-atom count (mirroring :meth:`get_iso`).
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
        ``n_aniso_atoms`` honors ``hydrogens_in_xray``: when hydrogens are excluded
        the anisotropic count is the heavy-atom count (mirroring :meth:`get_aniso`).
        """
        self._build_parametrization()
        idx = self._aniso_indices
        return self._A[idx], self._B[idx]

    # =========================================================================
    # Restraints (Geometry Restraints)
    # =========================================================================

    @property
    def restraints(self):
        """Geometry restraints over the atom table, on :attr:`ctx`.

        Built on first access over the current coordinates and cached on the context
        until the atom table or ``ctx.cif_path`` changes. Evaluations take the
        coordinates as an argument, e.g.
        ``model.restraints.bond_deviations(model.xyz())``.
        """
        if self.ctx.restraints is None:
            if not self.ctx.initialized:
                raise RuntimeError(
                    "Cannot build restraints: model not initialized. "
                    "Load data first with load_pdb() or load_cif()."
                )
            if self.ctx.verbose > 0:
                print("Building restraints...")
            self.ctx.build_restraints(self.xyz())
        return self.ctx.restraints

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

        Each is returned as-is once built, so a load that changes the atom count would
        otherwise reuse buffers sized for the previous atom set.
        """
        for name in self._ATOM_DERIVED_BUFFERS:
            if hasattr(self, name):
                delattr(self, name)

    def load(self, reader):
        """
        Populate the model from a reader callable, through
        :meth:`~torchref.model.context.ModelContext.from_atoms`; ``load_pdb`` /
        ``load_cif`` come through here.

        Parameters
        ----------
        reader : callable
            Zero-argument callable returning ``(pdb_df, cell, spacegroup)``. An
            optional ``.links`` attribute on it is kept as ``ctx.links``.

        Returns
        -------
        Model
            Self, for method chaining.
        """
        pdb, cell, spacegroup = reader()
        self._invalidate_atom_derived_caches()
        self.ctx, values = ModelContext.from_atoms(
            pdb,
            cell,
            spacegroup,
            dtype=self.dtype_float,
            device=self.device,
            links=getattr(reader, "links", None),
            **self.ctx.settings(),
        )
        self._install_parameters(values)
        return self

    def _install_parameters(
        self, values: AtomValues, state: Optional[dict] = None, xyz=None
    ) -> None:
        """Build the parameter wrappers and per-atom buffers over ``ctx.topology``.

        The only place the wrappers are constructed: load, restore, select and every
        derived model come through here.

        Parameters
        ----------
        values : AtomValues
            Starting values, row-aligned with ``ctx.topology``.
        state : dict, optional
            A state dict about to be loaded. Its saved refinable masks, occupancy
            groups, riding frames and node-field ADP layout fix the wrappers' shapes,
            and the default masks are **not** applied; ``load_state_dict`` supplies
            the values afterwards.
        xyz : MixedTensor, optional
            Coordinate wrapper to install as is, instead of building one from
            ``values``.
        """
        from torchref.model.riding_xyz import RidingXYZTensor

        dtype = self.dtype_float
        restoring = state is not None
        state = {} if state is None else state

        self.register_buffer(
            "aniso_flag",
            torch.as_tensor(values.aniso, dtype=torch.bool, device=self.device),
        )
        self.xyz = self._build_xyz(values, state) if xyz is None else xyz
        self.adp = self._restore_adp_slot(
            "adp", state, values, dtype, self.xyz, self.device
        )
        self.u = self._restore_adp_slot(
            "u", state, values, dtype, self.xyz, self.device
        )
        self.occupancy = self._build_occupancy(values, state)

        if restoring:
            # Placeholders: the saved masks arrive with load_state_dict, and applying
            # the defaults here would resize the refinable sets it has to match.
            for mask_name in ("xyz_mask", "adp_mask", "u_mask", "occupancy_mask"):
                self.register_buffer(
                    mask_name,
                    torch.ones(self.n_atoms, dtype=torch.bool, device=self.device),
                )
            if state.get("vdw_radii") is not None:
                self.register_buffer(
                    "vdw_radii",
                    torch.zeros_like(state["vdw_radii"], device=self.device),
                )
            return

        self.set_default_masks()
        if self.ctx.hydrogen_mode == "riding" and not isinstance(
            self.xyz, RidingXYZTensor
        ):
            self.xyz = RidingXYZTensor.from_mixed_tensor(
                self.xyz, self.hydrogen_frames()
            )
            self._repoint_coordinate_accessors()

    def _build_occupancy(self, values: AtomValues, state: dict) -> OccupancyTensor:
        """The occupancy wrapper over ``values``, grouped as ``state`` saved it.

        A saved grouping is taken as saved, never re-derived from the saved
        occupancies: which atoms share a group depends on the values (the 0.01
        deadband of :meth:`ModelContext.occupancy_groups`), and refinement moves them,
        so a re-derivation can disagree with the group-space parameters being loaded.
        Without one -- a fresh load -- the context groups the atoms.
        """
        initial = torch.tensor(values.occupancy, dtype=self.dtype_float)
        settings = {
            "dtype": self.dtype_float,
            "device": self.device,
            "name": "occupancy",
        }
        if state.get("occupancy.expansion_mask") is not None:
            return OccupancyTensor.from_saved_groups(
                initial, state, prefix="occupancy.", **settings
            )
        sharing_groups, altloc_groups, refinable_mask = self.ctx.occupancy_groups(
            initial
        )
        return OccupancyTensor(
            initial_values=initial,
            sharing_groups=sharing_groups,
            altloc_groups=altloc_groups,
            refinable_mask=refinable_mask,
            **settings,
        )

    def _build_xyz(self, values: AtomValues, state: dict):
        """The coordinate wrapper over ``values``, riding if ``state`` saved one.

        A saved riding wrapper is recognised by its frame buffers, never by shape: its
        storage is ``(n_base, 3)`` and a plain wrapper's ``(n_atoms, 3)``, both 2-D.
        """
        coords = torch.tensor(values.xyz, dtype=self.dtype_float)
        mask = state.get("xyz.refinable_mask")
        if state.get("xyz.h_row") is None:
            return MixedTensor(
                coords, refinable_mask=mask, name="xyz", device=self.device
            )

        from torchref.model.riding_xyz import RidingXYZTensor
        from torchref.topology.hydrogens import HydrogenFrames

        frames = HydrogenFrames.from_tensors(
            state["xyz.h_row"],
            state["xyz.parent_row"],
            state["xyz.n1_row"],
            state["xyz.n2_row"],
            state["xyz.frame_valid"],
            state.get("xyz.torsion_group"),
            state.get("xyz.rotation_group"),
        )
        return RidingXYZTensor(
            coords,
            frames,
            refinable_mask=mask,
            mask_in_base_space=True,
            name="xyz",
            device=self.device,
        )

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

    def to_dataframe(self) -> "pandas.DataFrame":
        """The atom table: identity from the topology, values from the wrappers.

        Built fresh on every call and never kept, so it cannot go stale and editing it
        changes nothing. Columns are the readers' (``ATOM``, ``serial``, ``name``,
        ``altloc``, ``resname``, ``chainid``, ``resseq``, ``icode``, ``x``/``y``/``z``,
        ``occupancy``, ``tempfactor``, ``element``, ``charge``, ``anisou_flag``,
        ``u11`` ... ``u23``, ``index``); ``serial`` runs 1..N and a blank altloc is
        ``''``. ``attrs`` carries the cell and the space group.

        ``tempfactor`` is the equivalent isotropic B whenever any atom is anisotropic,
        so the column agrees with the ANISOU records written beside it: for an
        anisotropic atom the PDB convention is B_eq = (8 pi^2 / 3) tr(U), not whatever
        the isotropic wrapper still holds, which stops being refined the moment an atom
        goes anisotropic.

        Returns
        -------
        pandas.DataFrame
        """
        import pandas as pd

        identity = self.ctx.topology.columns()
        # Detached, not under no_grad: the wrappers cache their forward, and a
        # gradient-free result cached here would be served to the next loss.
        xyz = self.xyz().detach().cpu().numpy()
        u = self.u().detach().cpu().numpy()
        b = self._b_eq().detach().cpu().numpy()
        occupancy = self.occupancy().detach().cpu().numpy()
        n = self.n_atoms
        table = pd.DataFrame(
            {
                "ATOM": np.where(identity["is_hetatm"], "HETATM", "ATOM"),
                "serial": np.arange(1, n + 1),
                "name": identity["name"],
                "altloc": np.where(identity["altloc"] == " ", "", identity["altloc"]),
                "resname": identity["resname"],
                "chainid": identity["chain"],
                "resseq": identity["resseq"],
                "icode": identity["icode"],
                "x": xyz[:, 0],
                "y": xyz[:, 1],
                "z": xyz[:, 2],
                "occupancy": occupancy,
                "tempfactor": b,
                "element": identity["element"],
                "charge": identity["charge"],
                "anisou_flag": self.aniso_flag.detach().cpu().numpy(),
                **{
                    column: u[:, i]
                    for i, column in enumerate(
                        ("u11", "u22", "u33", "u12", "u13", "u23")
                    )
                },
                "index": np.arange(n),
            }
        )
        if self.cell is not None:
            # The shortest decimal that round-trips the stored precision, so a float32
            # cell writes as 143.11 rather than 143.110001.
            cell = self.cell.data.detach().cpu().numpy()
            table.attrs["cell"] = [float(str(v)) for v in cell]
        table.attrs["spacegroup"] = self.spacegroup.hm if self.spacegroup else "P 1"
        if self.ctx.z_value is not None:
            table.attrs["z"] = self.ctx.z_value
        return table

    def _current_values(self) -> AtomValues:
        """The wrappers' current values, as starting values for a derived model.

        ``b`` is the isotropic wrapper's value, not the anisotropic B_eq that
        :meth:`to_dataframe` writes.
        """
        # Detached rather than under no_grad; see to_dataframe.
        return AtomValues(
            xyz=self.xyz().detach().cpu().numpy().astype(np.float64),
            b=self.adp().detach().cpu().numpy().astype(np.float64),
            u=self.u().detach().cpu().numpy().astype(np.float64),
            occupancy=self.occupancy().detach().cpu().numpy().astype(np.float64),
            aniso=self.aniso_flag.detach().cpu().numpy().astype(bool),
        )

    def get_vdw_radii(self):
        """
        Get van der Waals radii for all atoms based on their elements.

        Caches the result in self.vdw_radii for future calls.

        Returns
        -------
        torch.Tensor
            Van der Waals radii for each atom with shape (n_atoms,).
        """
        from torchref.topology.nonbonded import vdw_radii_for_elements

        if hasattr(self, "vdw_radii"):
            return self.vdw_radii
        vdw_radii = vdw_radii_for_elements(self.ctx.topology.atoms.element)
        self.register_buffer(
            "vdw_radii",
            torch.tensor(vdw_radii, dtype=self.dtype_float, device=self.device),
        )
        return self.vdw_radii

    def _after_device_apply(
        self, old_device, new_device, old_dtype, new_dtype, *,
        device_changed, dtype_changed,
    ):
        """Report the move when verbose."""
        if self.ctx.verbose > 0:
            print(f"Model moved to device: {self.device}")

    def copy(self):
        """
        Create a deep copy of the model, of the same class.

        Independent in every part: the context -- restraints included -- is copied via
        :meth:`~torchref.model.context.ModelContext.copy`, buffers are cloned and each
        parameter wrapper is copied through its own ``copy`` so its parametrization
        survives. Subclass settings carry over through ``_subclass_kwargs``.

        Returns
        -------
        Model
            A new, fully independent instance with copied data.
        """
        if not self.ctx.initialized:
            raise RuntimeError("Cannot copy an uninitialized Model. Load data first.")

        duplicate = self._spawn(self.ctx.copy())
        for name, buffer in self._buffers.items():
            if buffer is not None:
                duplicate.register_buffer(name, buffer.clone().detach())
        for name, module in self._modules.items():
            # Submodules the constructor already built (ModelFT's engine) derive from
            # the context and are not copied.
            if (
                module is None
                or name in duplicate._modules
                or not hasattr(module, "copy")
            ):
                continue
            setattr(duplicate, name, module.copy())

        # Anything that borrows the coordinates -- the ADP node field -- carries the
        # reference through its own ``copy`` and still points at THIS model's ``xyz``.
        duplicate._repoint_coordinate_accessors()
        if hasattr(duplicate, "reset_cache"):
            duplicate.reset_cache()

        if self.ctx.verbose > 0:
            print(f"Copied {type(self).__name__} ({duplicate.n_atoms} atoms)")
        return duplicate

    def _spawn(self, ctx: ModelContext) -> "Model":
        """An empty instance of this class around ``ctx``.

        Carries this model's dtype, device and subclass settings
        (:meth:`_subclass_kwargs`); the caller installs the parameters.
        """
        model = type(self)(
            dtype_float=self.dtype_float,
            verbose=ctx.verbose,
            device=self.device,
            **self._subclass_kwargs(),
        )
        model.ctx = ctx
        return model

    def _subclass_kwargs(self) -> dict:
        """Constructor arguments a subclass adds, as this instance holds them.

        :meth:`copy`, :meth:`select` and the derived-model helpers build new instances
        through it. Model adds none.
        """
        return {}

    def _derive(self, pdb, **overrides) -> "Model":
        """A new, quiet model of this class built from an atom table in this crystal.

        Construction, so the table is read: see :meth:`ModelContext.from_atoms`.

        Parameters
        ----------
        pdb : pandas.DataFrame
            Atom table for the new model.
        **overrides
            Context settings to change, e.g. ``hydrogens="strip"``.
        """
        from torchref.topology import Topology

        values = AtomValues.from_table(pdb.reset_index(drop=True))
        return self._derive_from(Topology.from_table(pdb), values, **overrides)

    def _derive_from(
        self, topology, values: AtomValues, xyz=None, **overrides
    ) -> "Model":
        """A new model of this class over ``topology`` and ``values`` in this crystal.

        Parameters
        ----------
        topology : Topology
            Node-only identity of the new atoms.
        values : AtomValues
            Their starting values.
        xyz : MixedTensor, optional
            A coordinate wrapper to install instead of one built from ``values``; only
            valid when the hydrogen policy leaves the atom set unchanged.
        **overrides
            Context settings to change; ``verbose`` defaults to 0.
        """
        ctx, values = self.ctx.derive(topology, values, **{"verbose": 0, **overrides})
        model = self._spawn(ctx)
        model._install_parameters(values, xyz=xyz)
        return model

    def _kept_hydrogens(self) -> str:
        """The policy for a table derived from this one: never generate again."""
        return "strip" if self.ctx.hydrogens == "strip" else "keep"

    def write_pdb(self, filename, metadata=None):
        """Write model to PDB file with optional metadata header.

        Parameters
        ----------
        filename : str
            Output PDB file path.
        metadata : RefinementMetadata, optional
            Metadata to render as PDB header (REMARK 3, TITLE, etc.).
        """
        table = sanitize_pdb_dataframe(self.to_dataframe())
        table.attrs["spacegroup"] = self.spacegroup.hm if self.spacegroup else "P 1"
        pdb.write(table, filename, metadata=metadata)

    def write_cif(self, filename, metadata=None):
        """Write model to mmCIF file with optional metadata.

        Parameters
        ----------
        filename : str
            Output mmCIF file path.
        metadata : RefinementMetadata, optional
            Metadata to include (refinement statistics, title, etc.).
        """
        table = sanitize_pdb_dataframe(self.to_dataframe())
        table.attrs["spacegroup"] = self.spacegroup.hm if self.spacegroup else "P 1"
        cif.write_model(table, filename, metadata=metadata)

    def get_iso(self):
        """
        Return per-atom parameters for the isotropic atom subset.

        Selects atoms whose ADP is a single scalar ``b``: ``~self.aniso_flag``,
        intersected with the heavy-atom mask when ``hydrogens_in_xray`` is off.

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
        Called from ``_install_parameters`` after the wrappers are constructed.
        """
        self.register_buffer(
            "xyz_mask", torch.ones(self.n_atoms, dtype=torch.bool, device=self.device)
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
        """Return the leaf ``nn.Parameter`` objects for the named parameter types.

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
            Leaves for each requested type, in the order the types were given.
            A coordinate wrapper may expose several: a riding one adds torsion and
            rotation leaves, a rigid one has rotation and translation leaves.
        """
        out: List[nn.Parameter] = []
        for t in types:
            wrapper = getattr(self, t, None)
            if wrapper is None:
                continue
            if hasattr(wrapper, "optimization_parameters"):
                out.extend(wrapper.optimization_parameters())
            else:
                rp = getattr(wrapper, "refinable_params", None)
                if rp is not None:
                    out.append(rp)
        return out

    def freeze(self, target: str):
        """
        Freeze (stop refining) one parameter type.

        A temporary toggle: the refinable set (the mask buffer) is left as it is,
        and :meth:`unfreeze` re-applies it. Frozen values are the current ones.

        Parameters
        ----------
        target : str
            One of ``"xyz"``, ``"adp"``, ``"u"``, ``"occupancy"``.
            Unrecognized names are ignored.
        """
        if target in self.PARAM_TYPES:
            # Every wrapper takes an atom-space mask and collapses it onto its own
            # storage (nodes of a field, occupancy groups, rigid bodies).
            getattr(self, target).update_refinable_mask(
                torch.zeros(self.n_atoms, dtype=torch.bool, device=self.device)
            )

    def freeze_all(self):
        """Freeze every parameter type (``xyz``, ``adp``, ``u``, ``occupancy``)."""
        for target in self.PARAM_TYPES:
            self.freeze(target)

    def unfreeze_all(self):
        """Unfreeze every parameter type, re-applying each one's refinable set."""
        for target in self.PARAM_TYPES:
            self.unfreeze(target)

    def unfreeze(self, target: str):
        """
        Unfreeze (resume refining) one parameter type.

        Re-applies the parameter's refinable set: its mask buffer (``xyz_mask`` /
        ``adp_mask`` / ``u_mask`` / ``occupancy_mask``), set at load and edited by
        :meth:`freeze_selection` and :meth:`unfreeze_selection`.

        Parameters
        ----------
        target : str
            One of ``"xyz"``, ``"adp"``, ``"u"``, ``"occupancy"``.
            Unrecognized names are ignored.
        """
        if target in self.PARAM_TYPES:
            getattr(self, target).update_refinable_mask(getattr(self, f"{target}_mask"))

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
            count stops setting the ADP parameter count. ``"field_aniso"`` is the same
            representation carrying a full U per node, which takes over ``u`` rather
            than ``adp``. ``"preserve"`` is a no-op: the ADPs stay exactly as the file
            supplied them, anisotropic where the file was anisotropic.
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
            discarded. See ``_install_disorder_field``.
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
        if not self.ctx.initialized:
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
                        self.n_atoms, dtype=torch.bool, device=self.device
                    )
                else:
                    target_mask = self.get_selection_mask(aniso_selection).to(
                        self.device
                    )
            else:
                target_mask = torch.zeros(
                    self.n_atoms, dtype=torch.bool, device=self.device
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
                self.n_atoms, dtype=torch.bool, device=self.device
            )
        elif mode == "anisotropic":
            sel = aniso_selection or "not resname HOH and not element H"
            aniso_mask = self.get_selection_mask(sel).to(self.device)
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
            n_nodes = max(4, int(round(self.n_atoms / 25.0)))

        # Anchor on density clusters, not single atoms: a node placed exactly on an atom
        # can isolate that atom by narrowing its kernel, which is per-atom refinement
        # wearing a node's clothes.
        anchor_rows = density_anchor_rows(xyz, min(n_nodes, self.n_atoms))

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
            was = self.n_atoms * (6 if anisotropic else 1)
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
        ``aniso_flag`` (which the writers' ``anisou_flag`` column comes from), the SF
        index cache, the masks and caches.
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
        # Clean partition: isotropic atoms refine B (adp), anisotropic atoms refine U.
        self.adp_mask = ~aniso_mask
        self.u_mask = aniso_mask.clone()
        self.adp.update_refinable_mask(self.adp_mask)
        self.u.update_refinable_mask(self.u_mask)

        # Anisotropy change invalidates structure-factor + wrapper forward caches.
        if hasattr(self, "reset_cache"):
            self.reset_cache()

    def update_mask_from_selection(
        self, selection_string: str, target: str, freeze: bool = True
    ):
        """
        Remove a Phenix-style selection from a parameter's refinable set, or add it.

        The refinable set is the mask buffer (``xyz_mask`` / ``adp_mask`` / ``u_mask``
        / ``occupancy_mask``) that :meth:`unfreeze` and :meth:`unfreeze_all` re-apply.
        Only the buffer changes; the parameter tensors keep their old split until
        :meth:`apply_mask_to_parameter` is called.

        Parameters
        ----------
        selection_string : str
            Phenix-style selection; grammar in :mod:`torchref.utils.selection`.
        target : str
            Parameter to update: 'xyz', 'adp', 'u', or 'occupancy'.
        freeze : bool, optional
            True (default) removes the selected atoms from the set, False adds them.
            Atoms outside the selection keep their state either way.

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

        selected = self.get_selection_mask(selection_string).to(current_mask.device)
        # Masks name the REFINABLE atoms, so freezing clears the selection.
        updated_mask = current_mask & ~selected if freeze else current_mask | selected

        setattr(self, mask_name, updated_mask)

        if self.ctx.verbose > 0:
            n_selected = selected.sum().item()
            n_refinable = updated_mask.sum().item()
            action = "frozen" if freeze else "unfrozen"
            print(
                f"Selection '{selection_string}' ({n_selected} atoms) {action} for {target}"
            )
            print(
                f"  Total refinable atoms for {target}: {n_refinable}/{self.n_atoms}"
            )

    def apply_mask_to_parameter(self, target: str):
        """
        Push the current mask buffer into the parameter wrapper's refinable split.

        The counterpart to :meth:`update_mask_from_selection`, which only edits the
        buffer, and the repartition :meth:`unfreeze` makes, raising on an unknown
        target. Replaces the wrapper's ``refinable_params``, so rebuild any
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
        if target not in self.PARAM_TYPES:
            raise ValueError(
                f"Invalid target: '{target}'. Must be 'xyz', 'adp', 'u', or 'occupancy'"
            )
        self.unfreeze(target)

        if self.ctx.verbose > 0:
            n_refinable = getattr(self, f"{target}_mask").sum().item()
            print(f"  Applied mask to {target}: {n_refinable} atoms refinable")

    def freeze_selection(
        self, selection_string: str, targets: Union[str, list] = "all"
    ):
        """
        Freeze atoms matching a Phenix-style selection for specified parameters.

        Removes them from each target's refinable set
        (:meth:`update_mask_from_selection`) and applies the set
        (:meth:`apply_mask_to_parameter`); atoms outside the selection keep their state.

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
        self._edit_refinable_sets(selection_string, targets, freeze=True)

    def unfreeze_selection(
        self, selection_string: str, targets: Union[str, list] = "all"
    ):
        """
        Unfreeze atoms matching a Phenix-style selection for specified parameters.

        Adds them to each target's refinable set (:meth:`update_mask_from_selection`)
        and applies the set (:meth:`apply_mask_to_parameter`); atoms outside the
        selection keep their state. :meth:`freeze` and :meth:`freeze_all` leave the
        sets untouched, so after them this makes the whole set refinable again, not
        just the selection: to refine only a selection, start from
        ``freeze_selection("all")``.

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

            model.freeze_selection("all", targets='xyz')
            model.unfreeze_selection("name CA or name C or name N", targets='xyz')
        """
        self._edit_refinable_sets(selection_string, targets, freeze=False)

    def _edit_refinable_sets(
        self, selection_string: str, targets: Union[str, list], freeze: bool
    ) -> None:
        """Edit each target's refinable set by a selection, then apply the set."""
        if targets == "all":
            targets = list(self.PARAM_TYPES)
        elif isinstance(targets, str):
            targets = [targets]

        for target in targets:
            self.update_mask_from_selection(selection_string, target, freeze=freeze)
            self.apply_mask_to_parameter(target)

    def get_aniso(self):
        """
        Return per-atom parameters for the anisotropic atom subset.

        Selects atoms whose ADP is the 6-element tensor
        ``u = (u11, u22, u33, u12, u13, u23)``: ``self.aniso_flag``, intersected
        with the heavy-atom mask when ``hydrogens_in_xray`` is off.

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

    def _b_eq(self) -> torch.Tensor:
        """Per-atom equivalent isotropic B in Å², ``(N,)``, differentiable.

        ``B_eq = (8 pi^2 / 3) tr(U)`` from :meth:`adp_u6` when any atom is
        anisotropic, else ``adp()`` directly, which is the same number for an
        isotropic atom without the U path. The written ``tempfactor`` column reads
        it, and so should any ADP restraint that needs one B per atom.
        """
        if self._aniso_is_empty:
            return self.adp()
        from torchref.base.targets.adp import u6_b_eq

        return u6_b_eq(self.adp_u6())

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
        """Yield ``(name, wrapper)`` for every ``MixedTensor`` submodule.

        Subclasses of :class:`~torchref.model.parameter_wrappers.MixedTensor` are
        included; ``RigidXYZTensor`` is not.
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
        if hasattr(self.xyz, "with_values"):
            # A riding wrapper keeps its frames; only the stored rows take the noise.
            self.xyz = self.xyz.with_values(new_xyz)
        else:
            self.xyz = MixedTensor(
                new_xyz, refinable_mask=self.xyz.refinable_mask, name="xyz"
            )
        self._repoint_coordinate_accessors()

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

    def strip_altlocs(self) -> "Model":
        """Return a new model with alternate conformations removed.

        Conformers are compared within one topology residue, ``(chain, resseq,
        icode)``, so residues 100 and 100A never compete, and alternates carrying
        different residue names (microheterogeneity) are treated as the alternates they
        are. In each residue with more than one altloc the conformer with the highest
        mean current occupancy is kept (ties to the first in sorted order), together
        with the residue's blank-altloc atoms. The returned model has no altlocs; the
        original is not modified.
        """
        from torchref.topology import Topology

        topology = self.ctx.topology
        occupancy = self.occupancy().detach().cpu().numpy()
        keep = np.ones(self.n_atoms, dtype=bool)
        resnames = topology.columns()["resname"]
        for residue, labels, conformers in self.ctx.altloc_residues():
            means = [occupancy[conformers[label]].mean() for label in labels]
            best = labels[int(np.argmax(means))]
            # Shared atoms belong to the retained chemical conformer in a model
            # with no altlocs, even when their deposited name was the other type.
            rows = list(topology.residues.atom_rows(residue))
            resnames[rows] = resnames[conformers[best][0]]
            for label in labels:
                if label != best:
                    keep[conformers[label]] = False

        rows = np.nonzero(keep)[0]
        columns = {key: value[rows] for key, value in topology.columns().items()}
        columns["altloc"] = np.full(len(rows), " ")
        columns["resname"] = resnames[rows]
        return self._derive_from(
            Topology.from_columns(columns),
            self._current_values().gather(rows),
            hydrogens=self._kept_hydrogens(),
        )

    def strip_hydrogens(self) -> "Model":
        """Return a new model with hydrogen atoms removed.

        Built from the current parameter values with ``hydrogens="strip"`` and
        ``hydrogen_mode="atoms"``. The original model is not modified.

        Returns
        -------
        Model
            New model without hydrogen atoms.
        """
        return self._derive_from(
            self.ctx.topology,
            self._current_values(),
            hydrogens="strip",
            hydrogen_mode="atoms",
        )

    def hydrogenate(self, verbose: int = 0) -> "Model":
        """Return a new model with missing hydrogens added from the monomer templates.

        Built from the current parameter values with ``hydrogens="add"``: each residue's
        library template is aligned onto the heavy atoms present and its hydrogens read
        off, and every free torsion (hydroxyl, thiol, amine, methyl) is scanned for the
        least-clashing angle. Missing HOH hydrogens get the dictionary geometry with a
        random orientation drawn from ``torch.manual_seed``. Existing hydrogens are
        retained. The original model is not modified.

        Parameters
        ----------
        verbose : int, default 0
            Verbosity level.

        Returns
        -------
        Model
        """
        return self._derive_from(
            self.ctx.topology, self._current_values(), hydrogens="add", verbose=verbose
        )

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """
        Return a dictionary containing the complete state of the Model.

        Registered buffers, the four parameter wrappers, the context's entries
        (:meth:`~torchref.model.context.ModelContext.state`), the atom table
        (:meth:`to_dataframe`), dtype and device. Restore with
        :meth:`create_from_state_dict`, which is what knows how to rebuild the wrappers.

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

        for key, value in self.ctx.state().items():
            state[prefix + key] = value
        # The atom table is the checkpoint format for identity and starting values;
        # restoring is construction, so it is split again there.
        state[prefix + "pdb"] = (
            self.to_dataframe() if self.ctx.topology is not None else None
        )
        state[prefix + "dtype_float"] = self.dtype_float
        state[prefix + "device"] = self.device

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

    def load_state(self, path: str, strict: bool = True, device=None):
        """
        Load the complete state of the model from a file.

        Parameters
        ----------
        path : str
            Path to load the state dictionary from.
        strict : bool, optional
            Accepted for signature compatibility; the restore goes through
            :meth:`create_from_state_dict`, which is never strict.
        device : torch.device, optional
            Device to restore onto. Defaults to this model's current device, so an
            in-place reload keeps its placement; pass one to restore elsewhere.
        """
        target_device = self.device if device is None else device
        state_dict = torch.load(path, map_location=target_device, weights_only=False)
        loaded = type(self).create_from_state_dict(
            state_dict, device=target_device, verbose=self.ctx.verbose
        )
        # Replace rather than merge: an empty model's ``None`` wrapper placeholders are
        # plain attributes, and left in place they would shadow the restored modules.
        self.__dict__.clear()
        self.__dict__.update(loaded.__dict__)
        if self.ctx.verbose > 0:
            print(f"Loaded model state from {path}")

    @staticmethod
    def _restore_adp_slot(prefix, state_dict, values, saved_dtype, xyz_wrapper, device):
        """Build the ``adp`` or ``u`` wrapper, as a node field when the state was one.

        With an empty ``state_dict`` this is the per-atom wrapper a fresh load uses.
        Otherwise the values are placeholders that ``load_state_dict`` overwrites.

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
        values : AtomValues
            Supplies the initial values.
        saved_dtype : torch.dtype
            Float dtype the state was saved in.
        xyz_wrapper : MixedTensor
            The already-built coordinate wrapper; a node field derives its node
            positions from it.
        device : torch.device
        """
        from torchref.model.parameter_wrappers import (
            CholeskyMixedTensor,
            PositiveMixedTensor,
        )

        aniso = prefix == "u"
        name = "aniso_U" if aniso else "adp"
        mask = state_dict.get(f"{prefix}.refinable_mask")
        if aniso:
            initial = torch.tensor(values.u, dtype=saved_dtype)
        else:
            initial = torch.tensor(values.b, dtype=saved_dtype)

        saved_nl = state_dict.get(f"{prefix}.neighbor_list")
        if saved_nl is None:
            # Match load(): the anisotropic U is a CholeskyMixedTensor so a restored
            # model refines it in the same positive-definite-by-construction
            # parametrization as a freshly-loaded one.
            wrapper = CholeskyMixedTensor if aniso else PositiveMixedTensor
            return wrapper(initial, refinable_mask=mask, name=name, device=device)

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
            device=device,
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
        Create a fully initialized model of this class from a state dictionary.

        Parameters
        ----------
        state_dict : dict
            State dictionary from ``torch.save(model.state_dict(), ...)``.
        device : torch.device, optional
            Move the restored model here once it is built. The restore itself always
            runs on CPU; ``None`` then moves it to the configured default device
            (``get_default_device()``), so a round-trip lands beside a same-config
            model rather than stranding itself on CPU.
        verbose : int, optional
            Verbosity level. Default is 1.
        dtype_float : torch.dtype, optional
            Float dtype for tensors when the state does not record one. Defaults to
            the configured dtypes.float.

        Returns
        -------
        Model
            Fully initialized instance with restored state.

        Notes
        -----
        Consumes ``state_dict``: the metadata keys are popped off it. Checkpoints
        written before the hydrogen policy existed are mapped onto it; see
        :meth:`~torchref.model.context.ModelContext.from_state`.
        """
        # Build on CPU throughout, then move once: the wrappers are built from the atom
        # table and land on CPU whatever is asked for, so resolving an accelerator up
        # front would split the model rather than place it.
        target_device = (
            canonical_device(device) if device is not None else get_default_device()
        )
        cpu = torch.device("cpu")
        if dtype_float is None:
            dtype_float = get_float_dtype()
        saved_dtype = state_dict.pop("dtype_float", dtype_float)
        state_dict.pop("device", None)

        instance = cls(
            dtype_float=saved_dtype,
            verbose=verbose,
            device=cpu,
            **cls._pop_subclass_state(state_dict),
        )
        instance.ctx, values = ModelContext.from_state(
            state_dict, dtype=saved_dtype, device=cpu, verbose=verbose
        )
        if values is not None:
            instance._install_parameters(values, state=state_dict)
        instance.load_state_dict(instance._restorable_entries(state_dict), strict=False)
        instance.to(target_device)
        if hasattr(instance, "reset_cache"):
            instance.reset_cache()

        if verbose > 0:
            print(f"Created {cls.__name__} from state_dict: {instance.n_atoms} atoms")
        return instance

    @classmethod
    def _pop_subclass_state(cls, state_dict: dict) -> dict:
        """Pop a subclass's own metadata keys and return them as constructor kwargs."""
        return {}

    def _restorable_entries(self, state_dict: dict) -> dict:
        """The entries of ``state_dict`` that ``load_state_dict`` should see.

        Drops tensors empty along dim 0 (placeholders from an atom-less state); scalars
        and non-tensor entries survive.
        """
        return {
            k: v
            for k, v in state_dict.items()
            if not (torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] == 0)
        }

    def get_selection_mask(self, selection: str) -> torch.Tensor:
        """
        Return a boolean mask for atoms matching a Phenix-style selection.

        Evaluated on the topology
        (:meth:`~torchref.topology.topology.Topology.select`); the result can be handed
        straight to ``MixedTensor.set()``.

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
        if not self.ctx.initialized:
            raise RuntimeError(
                "Cannot get selection mask from an uninitialized Model. Load data first."
            )

        return self.ctx.topology.select(selection)

    def select(self, selection: str) -> "Model":
        """
        Return a new model of the same class holding only the atoms a selection matches.

        Parameters
        ----------
        selection : str
            Phenix-style selection; see :meth:`get_selection_mask` for the syntax.

        Returns
        -------
        Model
            Built from the current parameter values, with default refinable masks. A
            riding wrapper keeps its frames and orientations, and a hydrogen whose
            parent is cut becomes an ordinary row. Restraints rebuild on first access.

        Raises
        ------
        RuntimeError
            If the model has not been initialized.
        ValueError
            If selection syntax is invalid or no atoms are selected.
        """
        if not self.ctx.initialized:
            raise RuntimeError(
                "Cannot select from an uninitialized Model. Load data first."
            )

        mask = self.get_selection_mask(selection)
        n_selected = int(mask.sum())
        if n_selected == 0:
            raise ValueError(f"Selection '{selection}' matched no atoms.")

        rows = np.nonzero(mask.cpu().numpy())[0]
        riding_xyz = (
            self.xyz.select_rows(mask) if hasattr(self.xyz, "select_rows") else None
        )
        selected = self._derive_from(
            self.ctx.topology.gather(rows),
            self._current_values().gather(rows),
            xyz=riding_xyz,
            hydrogens=self._kept_hydrogens(),
            verbose=self.ctx.verbose,
        )

        if self.ctx.verbose > 0:
            print(f"Selected {n_selected}/{self.n_atoms} atoms with '{selection}'")
        return selected

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
            Center of rotation with shape (3,), in Å. Defaults to :meth:`get_centroid`.

        Returns
        -------
        Model
            Self, for method chaining.
        """
        if not self.ctx.initialized:
            raise RuntimeError("Model must be initialized to apply rotation.")

        xyz = self.xyz()
        if center is None:
            center = self.get_centroid()

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
        """Return the unweighted mean of all Cartesian coordinates, ``(3,)`` in Å."""
        if not self.ctx.initialized:
            raise RuntimeError("Model must be initialized to compute centroid.")

        return self.xyz().mean(dim=0)

    # ------------------------------------------------------------------
    # Hydrogen parametrisation
    # ------------------------------------------------------------------

    @property
    def hydrogen_mode(self) -> str:
        """``"atoms"`` or ``"riding"``, as held by the context.

        See :class:`~torchref.model.context.ModelContext`.
        """
        return self.ctx.hydrogen_mode

    def hydrogen_frames(self):
        """Which rows ride on which heavy atoms, for the current atom table.

        Read off the riding coordinate wrapper when one is installed, else derived
        from the bond graph, which costs a restraint build the first time.

        Returns
        -------
        HydrogenFrames
        """
        if hasattr(self.xyz, "hydrogen_frames"):
            return self.xyz.hydrogen_frames()
        from torchref.topology.hydrogens import hydrogen_frames

        return hydrogen_frames(self.restraints.topology)

    def _repoint_coordinate_accessors(self) -> None:
        """Make every borrowed coordinate accessor read the current ``xyz`` wrapper.

        The ADP node field borrows the coordinates through ``set_xyz_fn``; after the
        wrapper slot is replaced it would otherwise keep reading a dead module.
        """
        for module in self._modules.values():
            if module is not None and hasattr(module, "set_xyz_fn"):
                module.set_xyz_fn(self.xyz)

    def set_hydrogen_mode(self, mode: str, frames=None) -> "Model":
        """Switch how the hydrogen rows of the current atom table are parametrised.

        Parameters
        ----------
        mode : {"atoms", "riding"}
            ``"riding"``: hydrogen coordinates derive from their parents each forward;
            rotatable groups keep shared torsion or orientation parameters.
            ``"atoms"``: hydrogens are ordinary refinable atoms.
        frames : HydrogenFrames, optional
            Riding frames for the current table; default :meth:`hydrogen_frames`.

        Returns
        -------
        Model
            Self, for chaining.

        Raises
        ------
        ValueError
            For an unknown mode, or ``"riding"`` on a model loaded with
            ``hydrogens="strip"``.

        Notes
        -----
        The atom table never changes here: hydrogens a table lacks are generated only
        at load, with ``hydrogens="add"``. Replaces the ``xyz`` wrapper, so any
        optimizer or ``LossState`` built over the old parameters is stale;
        :meth:`~torchref.refinement.base_refinement.Refinement.set_hydrogen_mode`
        does the engine-side reset. The refinable set carries over row for row (a
        hydrogen released to ``"atoms"`` follows its parent's mask).
        """
        from torchref.model.context import check_hydrogen_policy
        from torchref.model.riding_xyz import RidingXYZTensor

        if not self.ctx.initialized:
            raise RuntimeError("Load a structure before setting the hydrogen mode.")
        check_hydrogen_policy(self.ctx.hydrogens, mode)

        riding = isinstance(self.xyz, RidingXYZTensor)
        if mode == "atoms":
            new_xyz = self.xyz.to_mixed_tensor() if riding else self.xyz
        elif riding and frames is None:
            new_xyz = self.xyz
        else:
            base = self.xyz.to_mixed_tensor() if riding else self.xyz
            if frames is None:
                frames = self.hydrogen_frames()
            new_xyz = RidingXYZTensor.from_mixed_tensor(base, frames)

        if new_xyz is not self.xyz:
            # Pop first so the new wrapper registers as a fresh submodule.
            self._modules.pop("xyz")
            self.xyz = new_xyz
            self._repoint_coordinate_accessors()
        self.ctx.hydrogen_mode = mode
        if hasattr(self, "reset_cache"):
            self.reset_cache()
        if self.ctx.verbose > 0:
            print(f"Hydrogen mode: {mode} ({self.xyz})")
        return self

    def use_rigid_xyz(self) -> "Model":
        """
        Swap ``self.xyz`` for a per-chain
        :class:`~torchref.model.rigid_xyz.RigidXYZTensor`.

        The only refinable leaves become per-chain Euler angles and translations,
        with chains auto-detected from the topology's chain ids (waters and
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
        topology = self.ctx.topology
        residue_of = topology.atoms.residue_of.cpu().numpy()
        chain_ids = list(topology.residues.chain[residue_of])

        # Phenix-style polymer filter: drop waters and single-atom non-peptide
        # residues (ions), keep multi-atom HET ligands so they ride along with
        # their parent chain.
        _STD_POLYMER = {
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
            "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
            "TYR", "VAL", "MSE", "SEC", "PYL",
            "A", "C", "G", "T", "U", "I",
            "DA", "DC", "DG", "DT", "DU", "DI",
        }
        is_water = topology.is_water
        is_std = np.isin(topology.residues.resname[residue_of], list(_STD_POLYMER))
        residue_sizes = topology.residues.atom_end - topology.residues.atom_start
        residue_atom_count = residue_sizes[residue_of]
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
        self._repoint_coordinate_accessors()

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
            coordinates into a fresh
            :class:`~torchref.model.parameter_wrappers.MixedTensor` and install that
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
            stashed = getattr(self, "_rigid_original_xyz_container", None)
            if stashed is not None and hasattr(stashed, "with_values"):
                # A riding wrapper keeps its frames; a rigid motion leaves every
                # local offset unchanged.
                new_xyz = stashed.with_values(current)
            else:
                new_xyz = MixedTensor(current, name="xyz", device=self.device)
            self._modules.pop("xyz", None)
            self.xyz = new_xyz
            xyz_mask = getattr(self, "xyz_mask", None)
            if xyz_mask is not None and xyz_mask.shape[0] == new_xyz.shape[0]:
                self.xyz.update_refinable_mask(xyz_mask)
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
        self._repoint_coordinate_accessors()

        # Re-enable exactly the groups use_rigid_xyz() froze, so subsequent
        # per-atom / ADP refinement has parameters to optimize.
        for target in getattr(self, "_rigid_frozen_targets", []):
            self.unfreeze(target)
        if hasattr(self, "_rigid_frozen_targets"):
            del self._rigid_frozen_targets

        if hasattr(self, "reset_cache"):
            self.reset_cache()
        return self
