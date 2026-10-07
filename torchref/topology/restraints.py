"""The restraint layer over a topology, and what it takes to build one.

:class:`Restraints` is the orchestrator. Given a node-only
:class:`~torchref.topology.topology.Topology` and the coordinates to build over, it
resolves the monomer dictionaries, connects the topology, layers the ideal values over
its edges, derives the non-bonded pair list, and exposes the whole thing as
``restraints[edge_type][origin][property]`` -- three dict lookups into a mapping
assembled once, because the geometry targets read it on every iteration.

Three kinds of thing live here, and only the first is really connectivity:

* the topology and the values keyed to its edges, which are constants for the lifetime
  of an atom set;
* the non-bonded pair list, which is distance-derived and rebuilt as the model moves, so
  it is held apart from the rest;
* the Ramachandran map, a residue-level product of the same build.

Deliberately decoupled from :class:`~torchref.model.Model`: it takes a topology and
holds no reference back to whatever owns the coordinates. Every evaluation takes the
coordinates (or ADPs) it scores as an argument, and the pair list is rebuilt from the
coordinates it is handed, so the same object serves any model that shares the atom set.
"""

import numpy as np
import pandas as pd
import torch
from torch.nn import Module

from torchref.base.targets._common import torsions_from_xyz
from torchref.config import get_float_dtype
from torchref.topology.monomer.cif import (
    find_cif_file_in_library,
    read_cif,
    read_link_definitions,
)
from torchref.utils.debug_utils import DebugMixin
from torchref.utils.device_mixin import DeviceMixin


class Restraints(DeviceMixin, DebugMixin, Module):
    """
    Restraints handler for crystallographic model refinement.

    Parameters
    ----------
    topology : Topology, optional
        The atoms to restrain -- a node-only topology is enough
        (:meth:`~torchref.topology.Topology.from_table`); it is connected here. If
        None, creates an empty shell.
    cif_path : str or list of str, optional
        Path to the CIF restraints dictionary file(s).
    xyz : torch.Tensor, optional
        Cartesian coordinates in Å, shape ``(n_atoms, 3)``, that the topology and the
        first pair list are built over. Required with ``topology``. The build lands on
        this tensor's device. Not retained.
    cell : Cell, optional
        Crystallographic unit cell. Together with ``spacegroup``, enables
        symmetry-aware VDW restraints (contacts with symmetry mates). Without both,
        the pair list is searched in an isolated P1 box. Copied, so moving these
        restraints never moves the caller's cell.
    spacegroup : SpaceGroup, optional
        Space group. Together with ``cell``, enables symmetry-aware VDW restraints.
        Copied, like ``cell``.
    links : pd.DataFrame, optional
        Parsed PDB LINK records; each accepted record adds one bond restraint
        between the two named atoms.
    verbose : int, default 1
        Verbosity level (0=silent, 1=normal, 2=detailed).
    nonbonded : bool, default True
        Build the non-bonded pair list. ``False`` gives connectivity and ideal values
        only, for a caller that needs the topology and nothing else, such as hydrogen
        planning; the pair search is the largest part of a build.

    Attributes
    ----------
    restraints : dict
        Restraint groups as ``restraints["bond"]["intra"]["indices"]``. A plain nested
        dict; the per-origin indices are views into ``topology``'s edge blocks.
    topology : Topology
        The connected topology the geometry restraints are defined over; a new object,
        the input is not modified.
    cif_dict : dict
        Parsed CIF restraints keyed by residue type; ``missing_residues`` lists
        the types that could not be resolved.
    h_topo
        Riding-hydrogen map, built only when the model carries no hydrogens of its
        own. Empty otherwise; see :mod:`torchref.topology.riding`.
    link_dict, link_list
        Link-type definitions from the monomer library, set only when ``topology``
        was provided.
    """

    def __init__(
        self,
        topology=None,
        cif_path=None,
        xyz: torch.Tensor = None,
        cell=None,
        spacegroup=None,
        links: pd.DataFrame = None,
        verbose: int = 1,
        nonbonded: bool = True,
    ):
        """Initialize the Restraints handler."""
        super().__init__()
        self.cif_path = cif_path
        self.verbose = verbose
        self.links = links
        self._nonbonded = bool(nonbonded)

        # Own copies: DeviceMixin moves a Cell or SpaceGroup in place, and these
        # usually belong to the model's context.
        self._cell = None if cell is None else cell.clone()
        self._spacegroup = None if spacegroup is None else spacegroup.copy()

        # Connectivity, the values layered over it, and the non-bonded pair list, which
        # is rebuilt on displacement and so is kept apart from the rest.
        self.topology = None
        self._values = {}
        self._vdw = {}
        # Derived: per-origin views into the topology's edge blocks. Rebuilt by
        # _rebuild_entries, which runs at build time and after any device move.
        self._entries = {}
        self._torsion_max_period = 1

        # Empty initialization
        if topology is None:
            self.cif_dict = {}
            self.unique_residues = []
            return
        if xyz is None:
            raise ValueError("Restraints over a topology need the coordinates, xyz=")

        self._nodes = topology
        self.unique_residues, single_atom = self._dictionary_resnames(topology)

        # Parse CIF files
        self._load_cif_dictionaries(cif_path, single_atom)

        # Load link definitions for inter-residue restraints
        if verbose > 1:
            print("Loading link definitions from monomer library...")
        self.link_dict, self.link_list = read_link_definitions()
        if verbose > 1:
            print(f"Loaded {len(self.link_dict)} link types")

        self.build_restraints(xyz)
        if self.verbose > 0:
            self.summary()

    @staticmethod
    def _dictionary_resnames(topology) -> tuple:
        """Residue names in first-seen order, as two lists: those whose atoms carry more
        than one name, whose dictionaries the build needs, and those of a single atom.
        """
        names_by_resname: dict = {}
        columns = topology.columns()
        for resname, atom_name in zip(columns["resname"], columns["name"]):
            names_by_resname.setdefault(str(resname), set()).add(str(atom_name))
        multi = [name for name, atoms in names_by_resname.items() if len(atoms) > 1]
        return multi, [name for name in names_by_resname if name not in multi]

    def _riding_table(self, xyz: torch.Tensor) -> pd.DataFrame:
        """The identity-plus-coordinates table :mod:`torchref.topology.riding` reads.

        That module takes an atom table rather than a topology, so one is assembled
        here from :attr:`topology` and ``xyz`` (Cartesian, Å, shape ``(n_atoms, 3)``).
        """
        columns = self.topology.columns()
        coords = xyz.detach().cpu().numpy()
        return pd.DataFrame(
            {
                "name": columns["name"],
                "element": columns["element"],
                "resname": columns["resname"],
                "chainid": columns["chain"],
                "resseq": columns["resseq"],
                "icode": columns["icode"],
                "x": coords[:, 0],
                "y": coords[:, 1],
                "z": coords[:, 2],
            }
        )

    @property
    def restraints(self) -> dict:
        """Restraint groups as ``[edge type][origin][property]``.

        A plain nested dict of tensors, assembled once at build time. Reading it costs
        three dict lookups and no allocation, which matters because the geometry targets
        do it on every iteration. Per-origin indices are **views** into the topology's
        contiguous edge blocks, so an in-place edit to a block is visible here at once,
        and taking a subset costs nothing.
        """
        return self._entries

    def _rebuild_entries(self) -> None:
        """Re-derive the entry views from the topology and its values.

        Cheap -- a handful of slices -- and idempotent. Runs at the end of a build and
        again after any device or dtype move, because moving a tensor rebinds it and
        leaves the old views pointing at freed storage.
        """
        if self.topology is None:
            return
        from torchref.topology import assemble_entries, max_period

        self._entries = assemble_entries(self.topology, self._values)
        if self._vdw:
            self._entries["vdw"] = self._vdw
        self._torsion_max_period = max_period(self._entries)

    def _apply(self, fn, recurse: bool = True):
        """Drop the derived views before the traversal, re-slice them after.

        ``DeviceMixin``'s ``__dict__`` walk recurses into dicts, so leaving the entries
        in place would move each slice on its own and quietly turn every view into an
        independent tensor -- doubling the memory and breaking the aliasing the design
        rests on. Rebuilding unconditionally rather than in ``_after_device_apply``,
        because that hook only fires when the device or dtype actually changed, and a
        ``.to()`` onto the current device must not leave the entries empty.
        """
        self._entries = {}
        result = super()._apply(fn, recurse)
        self._rebuild_entries()
        return result

    def _load_cif_dictionaries(self, cif_path, single_atom):
        """Load CIF dictionaries from provided paths and monomer library.

        A residue of ``single_atom`` takes its library entry if that reads as a
        restraint dictionary, as one with hydrogens does (a lone water oxygen, an NH2
        cap): it types the atom and holds the hydrogens it rides. An ion's entry has no
        bonds and the reader rejects it, so an ion stays untyped and out of
        ``missing_residues``.
        """
        if cif_path:
            if isinstance(cif_path, str):
                self.cif_dict = read_cif(cif_path)
            elif isinstance(cif_path, list):
                self.cif_dict = {}
                for cif_file in cif_path:
                    self.cif_dict.update(read_cif(cif_file))
            else:
                raise ValueError("cif_path must be a string or a list of strings")
        else:
            self.cif_dict = {}

        # Load missing residues from monomer library
        self.missing_residues = [
            res for res in self.unique_residues if res not in self.cif_dict
        ]
        from pathlib import Path
        from torchref import PATH_TORCHREF_DATA

        lookups = self.missing_residues + [
            res for res in single_atom if res not in self.cif_dict
        ]
        additional_files = [
            (
                Path(PATH_TORCHREF_DATA) / "monomer_library/h/HOH.cif"
                if res == "HOH"
                else find_cif_file_in_library(res)
            )
            for res in lookups
        ]

        for res, cif_file in zip(lookups, additional_files):
            if cif_file is not None:
                if self.verbose > 1:
                    print(cif_file)
                try:
                    additional_cif_dict = read_cif(cif_file)
                    self.cif_dict.update(additional_cif_dict)
                except Exception as e:
                    if res not in single_atom:
                        print("Error reading CIF file:", e)
                        print("This residue will have no restraints applied.")

        self.missing_residues = [
            res for res in self.unique_residues if res not in self.cif_dict
        ]

        if len(self.missing_residues) >= 1:
            if self.verbose > 0:
                print(
                    f"Warning: The following residues are missing from the CIF dictionary "
                    f"and will have no restraints applied: {self.missing_residues}"
                )

    def _load_rama_surfaces(self, device: torch.device):
        """Load pre-computed Ramachandran NLL surfaces as a buffer."""
        from torchref.topology.ramachandran import load_nll_surfaces

        surfaces = load_nll_surfaces(device)
        self.register_buffer("_rama_surfaces", surfaces)

    def build_restraints(self, xyz: torch.Tensor):
        """Build the topology, the values over it, and the non-bonded pair list.

        Parameters
        ----------
        xyz : torch.Tensor
            Cartesian coordinates in Å, shape ``(n_atoms, 3)``. The pair list is
            skipped when constructed with ``nonbonded=False``. Builds on CPU and moves
            the result to ``xyz``'s device at the end.
        """
        try:
            target_device = xyz.device
            device = torch.device("cpu")

            from torchref.topology import build_topology_with_values

            self.topology, self._values, extras = build_topology_with_values(
                self._nodes,
                self.cif_dict,
                xyz.detach().to(device),
                link_dict=self.link_dict,
                link_list=self.link_list,
                links=self.links,
                device=device,
                verbose=self.verbose,
            )
            self._rebuild_entries()

            rama = extras.get("ramachandran")
            if rama is not None:
                self.register_buffer("_rama_phi_indices", rama["phi_indices"])
                self.register_buffer("_rama_psi_indices", rama["psi_indices"])
                self.register_buffer("_rama_surface_type", rama["surface_type"])
                self._load_rama_surfaces(device)

            # cutoff sits ~1 Å beyond the largest heavy-atom VDW sum (~3.6 Å) plus
            # expected drift, so a displacement-triggered rebuild stays inside the
            # margin and cannot miss a newly-formed contact.
            if self._nonbonded:
                self._build_vdw_restraints(xyz, cutoff=6.0, inter_residue_only=False)

            if target_device.type != "cpu":
                self.to(target_device)

        except Exception as e:
            self.debug_on_error(e, context="Restraints.build_restraints")
            raise

    @property
    def h_topo(self):
        """Access riding hydrogen topology (None if not built)."""
        return getattr(self, "_h_topo", None)

    def _build_h_exclusion_hash(self, h_topo, device):
        """Sorted 1-D hash tensor of H-specific 1-2 and 1-3 exclusions.

        Hashes are ``min(i, j) * max_idx + max(i, j)``; the sort is required for
        ``torch.searchsorted`` lookup.
        """
        if h_topo is None or h_topo.n_hydrogens == 0:
            # dtype-ok: packed pair key min*max_idx+max overflows int32 beyond ~46k atoms; searchsorted needs both sides int64
            return torch.tensor([], dtype=torch.long, device=device)

        n_heavy = self.topology.n_atoms
        n_h = h_topo.n_hydrogens
        exclusions = set()

        parent_idx = h_topo.h_parent_idx.cpu().numpy()
        nb_idx = h_topo.parent_neighbor_idx.cpu().numpy()
        nb_count = h_topo.parent_neighbor_count.cpu().numpy()

        for hi in range(n_h):
            # H index in the combined array is n_heavy + hi
            h_combined = n_heavy + hi
            p = int(parent_idx[hi])

            # 1-2: H — parent
            exclusions.add((min(h_combined, p), max(h_combined, p)))

            # 1-3: H — parent's heavy neighbours
            for ni in range(int(nb_count[hi])):
                nb = int(nb_idx[hi, ni])
                if nb >= 0:
                    exclusions.add((min(h_combined, nb), max(h_combined, nb)))

        if not exclusions:
            # dtype-ok: packed pair key min*max_idx+max overflows int32 beyond ~46k atoms; searchsorted needs both sides int64
            return torch.tensor([], dtype=torch.long, device=device)

        arr = np.array(list(exclusions), dtype=np.int64)
        max_idx = max(n_heavy + n_h, int(arr.max()) + 1)
        hashes = arr[:, 0] * max_idx + arr[:, 1]
        hashes.sort()
        # dtype-ok: packed pair key min*max_idx+max overflows int32 beyond ~46k atoms; searchsorted needs both sides int64
        return torch.tensor(hashes, dtype=torch.long, device=device)

    def _build_vdw_restraints(self, xyz, cutoff=6.0, inter_residue_only=True):
        """Build van der Waals (non-bonded contact) restraints.

        With cell and spacegroup present, includes contacts to symmetry mates.
        Without them the same periodic search runs in an isolated P1 box, one
        cutoff wider than the model on every side, so no image comes within range.
        Also builds the riding-hydrogen topology for H-VDW evaluation.

        Parameters
        ----------
        xyz : torch.Tensor
            Cartesian coordinates in Å, shape ``(n_atoms, 3)``.
        cutoff : float, default 6.0
            Contact-search cutoff in Angstroms. Keep it ~1 Å beyond the largest
            heavy-atom VDW sum so the rebuild threshold has margin.
        inter_residue_only : bool, default True
            If True, only build contacts between atoms in different residues.

        Notes
        -----
        Caches the kwargs in ``_vdw_build_kwargs`` and a detached ASU coordinate
        snapshot in ``_last_vdw_build_xyz``; :meth:`rebuild_vdw_restraints` and
        ``NonBondedTarget.maintenance`` both read those.
        """
        self._vdw_build_kwargs = dict(
            cutoff=cutoff,
            inter_residue_only=inter_residue_only,
        )

        if self.verbose > 0:
            print("\nBuilding VDW (non-bonded) restraints...")

        # The build (neighbour search, H topology, exclusion hashing) runs on CPU;
        # everything it registers is migrated to target_device at the end, which
        # the maintenance-triggered rebuild path depends on.
        cpu = torch.device("cpu")
        target_device = xyz.device
        xyz_cpu = xyz.detach().to(cpu)
        radii_cpu = torch.as_tensor(
            self.topology.atoms.vdw_radii, dtype=get_float_dtype()
        )

        from torchref.symmetry import SpaceGroup
        from torchref.symmetry.cell import Cell

        # Fresh CPU copies: Cell/SpaceGroup ``.to()`` mutates in place, which would
        # relocate the copies this object keeps on the model's device.
        if self._cell is not None and self._spacegroup is not None:
            cell_cpu = Cell(
                self._cell._data.detach(), device=cpu, dtype=self._cell.dtype
            )
            sg_cpu = self._spacegroup.copy().to(cpu)
        else:
            span = xyz_cpu.max(dim=0).values - xyz_cpu.min(dim=0).values
            extent = float(span.max())
            side = extent + 2.0 * cutoff
            cell_cpu = Cell([side, side, side, 90.0, 90.0, 90.0], device=cpu)
            sg_cpu = SpaceGroup("P 1", device=cpu)

        from torchref.topology.nonbonded import build_vdw_restraints_gpu

        self._vdw = build_vdw_restraints_gpu(
            xyz=xyz_cpu,
            vdw_radii=radii_cpu,
            cell=cell_cpu,
            sg=sg_cpu,
            topology=self.topology,
            exclusion_set=self.topology.atoms.exclusions_12_13_14(),
            cutoff=cutoff,
            inter_residue_only=inter_residue_only,
            verbose=self.verbose,
        )

        # Publish the new pair list before anything reads it back below. Unlike the
        # geometry edges it is not derived from the topology, so it is held separately
        # and re-inserted here and by _rebuild_entries.
        self._entries["vdw"] = self._vdw

        # Riding hydrogens stand in for the sterics of hydrogens the model does not
        # carry. Once it carries them they are ordinary atoms in the pair list above, and
        # placing riding ones as well would put phantom hydrogens in the structure that
        # push real atoms around. The two also disagree about how many belong on a
        # parent -- the riding builder counts bonded neighbours by distance, the
        # generator reads them off the bond graph -- so the leftovers are not even the
        # hydrogens the generator declined to add.
        from torchref.topology.riding import (
            HydrogenTopology,
            build_h_candidate_pairs,
            build_hydrogen_topology,
            candidate_contact_distances,
        )

        if bool(self.topology.atoms.is_hydrogen.any()):
            self._h_topo = HydrogenTopology(device=cpu)
        else:
            riding_table = self._riding_table(xyz)
            self._h_topo = build_hydrogen_topology(
                pdb=riding_table,
                device=cpu,
                verbose=self.verbose,
                cif_dict=self.cif_dict,
            )
        self._h_excl_hash = self._build_h_exclusion_hash(self._h_topo, cpu)

        # Precompute H candidate pairs from heavy-atom VDW pair list
        vdw_data = self.restraints.get("vdw")
        if vdw_data is not None and self._h_topo.n_hydrogens > 0:
            build_h_candidate_pairs(
                h_topo=self._h_topo,
                vdw_data=vdw_data,
                pdb=riding_table,
                h_excl_hash=self._h_excl_hash,
                device=cpu,
                verbose=self.verbose,
            )
            if self._h_topo.has_candidates:
                self._h_topo.cand_min_dist = candidate_contact_distances(
                    self._h_topo, radii_cpu, self.topology.atoms.hb_type
                )

        # Snapshot at build time so maintenance() can diff current positions
        # against it; kept on the model device so the compare is one op.
        self._last_vdw_build_xyz = xyz.detach().clone()

        # Move the CPU-built pair list, h_topo and h_excl_hash to the model device.
        # The rebuild path has no surrounding migration, so this cannot be dropped.
        if target_device.type != "cpu":
            self.to(target_device)

    def rebuild_vdw_restraints(self, xyz: torch.Tensor) -> None:
        """Refresh the VDW pair list over ``xyz`` with the initial build's kwargs.

        Called by :meth:`NonBondedTarget.maintenance` once max atomic displacement
        since the last build exceeds its threshold. Raises ``RuntimeError`` if no
        initial build has run.

        Parameters
        ----------
        xyz : torch.Tensor
            Current Cartesian coordinates in Å, shape ``(n_atoms, 3)``.
        """
        if not hasattr(self, "_vdw_build_kwargs"):
            raise RuntimeError(
                "rebuild_vdw_restraints called before initial build "
                "— _vdw_build_kwargs is missing"
            )
        self._build_vdw_restraints(xyz, **self._vdw_build_kwargs)

    # Device movement goes through DeviceMixin: the topology and the value tensors are
    # walked and moved, and _apply re-slices the derived entry views afterwards.

    def summary(self):
        """Print a detailed summary of all restraints."""
        print("=" * 80)
        print("Restraints Summary")
        print("=" * 80)
        print(f"CIF file: {self.cif_path}")
        print(f"Residue types in dictionary: {len(self.cif_dict)}")
        print()

        def get_count(rtype, origin):
            indices = self.restraints.get(rtype, {}).get(origin, {}).get("indices")
            return 0 if indices is None else indices.shape[0]

        print("INTRA-RESIDUE RESTRAINTS:")
        print("-" * 80)
        print(f"  Bonds: {get_count('bond', 'intra')}")
        print(f"  Angles: {get_count('angle', 'intra')}")
        print(f"  Torsions: {get_count('torsion', 'intra')}")

        # Count planes
        n_planes = 0
        for key in self.restraints.get("plane", {}).keys():
            n_planes += get_count("plane", key)
        print(f"  Planes: {n_planes}")

        # Chiral
        chiral_count = 0
        if "chiral" in self.restraints:
            indices = self.restraints["chiral"].get("indices")
            chiral_count = 0 if indices is None else indices.shape[0]
        print(f"  Chirals: {chiral_count}")

        print()
        print("INTER-RESIDUE RESTRAINTS:")
        print("-" * 80)
        print(f"  Peptide bonds: {get_count('bond', 'peptide')}")
        print(f"  Peptide angles: {get_count('angle', 'peptide')}")
        print(f"  Disulfide bonds: {get_count('bond', 'disulfide')}")
        print(f"  Disulfide angles: {get_count('angle', 'disulfide')}")
        print(f"  Disulfide torsions: {get_count('torsion', 'disulfide')}")
        print(f"  LINK bonds: {get_count('bond', 'link')}")

        print()
        print("BACKBONE TORSIONS:")
        print("-" * 80)
        print(f"  Phi: {get_count('torsion', 'phi')}")
        print(f"  Psi: {get_count('torsion', 'psi')}")
        print(f"  Omega: {get_count('torsion', 'omega')}")

        # Ramachandran
        rama_count = 0
        if hasattr(self, "_rama_phi_indices") and self._rama_phi_indices is not None:
            rama_count = self._rama_phi_indices.shape[0]
        if rama_count > 0:
            print(f"  Ramachandran: {rama_count}")

        print()
        print("VDW RESTRAINTS:")
        print("-" * 80)
        vdw_count = 0
        vdw_sym_count = 0
        if "vdw" in self.restraints:
            indices = self.restraints["vdw"].get("indices")
            vdw_count = 0 if indices is None else indices.shape[0]
            symop_indices = self.restraints["vdw"].get("symop_indices")
            cell_offsets = self.restraints["vdw"].get("cell_offsets")
            if symop_indices is not None and len(symop_indices) > 0:
                from torchref.base.coordinates.symmetry_images import (
                    is_symmetry_image,
                )

                is_sym = is_symmetry_image(symop_indices, cell_offsets)
                vdw_sym_count = int(is_sym.sum().item())
        vdw_asu_count = vdw_count - vdw_sym_count
        if vdw_sym_count > 0:
            print(f"  Non-bonded contacts: {vdw_count} ({vdw_asu_count} intra-ASU, {vdw_sym_count} symmetry)")
        else:
            print(f"  Non-bonded contacts: {vdw_count}")

        print("=" * 80)

    def __repr__(self):
        """Return a one-line string representation.

        Surfaces only a subset of restraint counts (intra-residue bonds,
        angles, torsions and peptide bonds); see :meth:`summary` for the full
        breakdown.
        """

        def get_count(rtype, origin):
            indices = self.restraints.get(rtype, {}).get(origin, {}).get("indices")
            return 0 if indices is None else indices.shape[0]

        n_bonds = get_count("bond", "intra")
        n_angles = get_count("angle", "intra")
        n_torsions = get_count("torsion", "intra")
        n_bonds_peptide = get_count("bond", "peptide")

        return (
            f"Restraints(bonds={n_bonds}, angles={n_angles}, "
            f"torsions={n_torsions}, peptide_bonds={n_bonds_peptide})"
        )

    def bond_lengths(self, idx, xyz: torch.Tensor):
        """
        Compute current bond lengths from atomic coordinates.

        Parameters
        ----------
        idx : torch.Tensor
            Bond indices tensor of shape (N, 2).
        xyz : torch.Tensor
            Cartesian coordinates in Å, shape (n_atoms, 3).

        Returns
        -------
        torch.Tensor
            Tensor of bond lengths of shape (N,).
        """
        if idx is None:
            return xyz.new_zeros(0)
        pos1 = xyz[idx[:, 0], :]
        pos2 = xyz[idx[:, 1], :]
        return torch.linalg.norm(pos2 - pos1, dim=-1)

    def copy(self):
        """An independent copy, sharing no state with this one.

        The entry views are re-sliced afterwards rather than left as deep-copied
        tensors: ``deepcopy`` duplicates a view and the block it points into as two
        unrelated tensors, so the copy would still hold the right values but would no
        longer alias, and an in-place edit to one would stop being visible through the
        other.

        Returns
        -------
        Restraints
        """
        import copy

        duplicate = copy.deepcopy(self)
        duplicate._rebuild_entries()
        return duplicate

    def bond_deviations(self, xyz: torch.Tensor):
        """
        Compute bond length deviations and sigmas.

        Parameters
        ----------
        xyz : torch.Tensor
            Cartesian coordinates in Å, shape (n_atoms, 3).

        Returns
        -------
        deviations : torch.Tensor
            Calculated minus expected bond lengths in Angstroms, shape ``(n_bonds,)``;
            empty when there are no bond restraints.
        sigmas : torch.Tensor
            Standard deviations from CIF library in Angstroms, shape ``(n_bonds,)``.
        """
        group = self.restraints.get("bond", {}).get("all")
        if group is None:
            return xyz.new_zeros(0), xyz.new_zeros(0)

        idx = group["indices"]
        references = group["references"]
        sigmas = group["sigmas"]

        # Get current bond lengths
        bond_lengths = self.bond_lengths(idx, xyz)
        deviations = bond_lengths - references

        return deviations, sigmas

    def angles(self, idx, xyz: torch.Tensor):
        """
        Compute current angle values for all angle restraints.

        Parameters
        ----------
        idx : torch.Tensor
            Angle indices tensor of shape (N, 3).
        xyz : torch.Tensor
            Cartesian coordinates in Å, shape (n_atoms, 3).

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_angles,) with current angle values in degrees.
        """
        pos1 = xyz[idx[:, 0], :]
        pos2 = xyz[idx[:, 1], :]
        pos3 = xyz[idx[:, 2], :]

        # Compute vectors
        v1 = pos1 - pos2  # Vector from atom2 to atom1
        v2 = pos3 - pos2  # Vector from atom2 to atom3

        # Compute angle using dot product
        # cos(θ) = (v1 · v2) / (|v1| * |v2|)
        dot_product = torch.sum(v1 * v2, dim=-1)
        norm1 = torch.linalg.norm(v1, dim=-1)
        norm2 = torch.linalg.norm(v2, dim=-1)

        # Clamp to avoid numerical issues with arccos
        cos_angle = torch.clamp(dot_product / (norm1 * norm2), -1.0, 1.0)

        # Return angle in degrees
        angles_rad = torch.acos(cos_angle)
        angles_deg = torch.rad2deg(angles_rad)

        return angles_deg

    def angle_deviations(self, xyz: torch.Tensor):
        """
        Compute angle deviations and sigmas.

        Parameters
        ----------
        xyz : torch.Tensor
            Cartesian coordinates in Å, shape (n_atoms, 3).

        Returns
        -------
        deviations : torch.Tensor
            Calculated minus expected angles, in radians, shape ``(n_angles,)``;
            empty when there are no angle restraints. The CIF library references
            are stored in degrees and converted to radians here before differencing.
        sigmas : torch.Tensor
            CIF library standard deviations, converted from degrees to radians.
        """
        group = self.restraints.get("angle", {}).get("all")
        if group is None:
            return xyz.new_zeros(0), xyz.new_zeros(0)

        idx = group["indices"]
        references_rad = group["references"] * (torch.pi / 180.0)
        sigmas_rad = group["sigmas"] * (torch.pi / 180.0)

        calculated_rad = self.angles(idx, xyz) * (torch.pi / 180.0)
        deviations = calculated_rad - references_rad

        return deviations, sigmas_rad

    def torsions(self, idx: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        """Compute current torsion angles, IUPAC sign, in degrees.

        Delegates to :func:`torchref.base.targets._common.torsions_from_xyz`, the
        package's one eager dihedral, whose sign is the convention the monomer-library
        references are written in (the same as ``gemmi.calculate_dihedral``).

        Parameters
        ----------
        idx : torch.Tensor
            Torsion atom indices of shape (n_torsions, 4), integer dtype.
        xyz : torch.Tensor
            Cartesian coordinates in Å, shape (n_atoms, 3).

        Returns
        -------
        torch.Tensor
            Torsion angles in degrees in [-180, 180], shape (n_torsions,).
        """
        return torsions_from_xyz(xyz, idx)

    def _wrap_torsion_periodicity(self, diff_rad, periods):
        """Smallest angular deviation under n-fold rotational symmetry.

        ``diff_rad`` and ``periods`` share a shape; period 0 or 1 means plain
        wrapping to [-π, π], period n folds by 360°/n and returns the equivalent
        deviation nearest zero (period 6, benzene: 10°/70°/130°/... all give 10°).
        """
        # Clamp periods to minimum of 1 to avoid division by zero
        periods_safe = torch.clamp(periods, min=1)
        # Use cached max_period to avoid .item() GPU sync every iteration
        max_period = getattr(self, "_torsion_max_period", None)
        if max_period is None:
            max_period = int(periods_safe.max().item())

        if max_period > 1:
            # Vectorized approach: generate all equivalent angles
            device = diff_rad.device
            original_shape = diff_rad.shape

            # Flatten input for processing
            diff_rad_flat = diff_rad.flatten()
            periods_flat = periods_safe.flatten()
            n_angles = len(diff_rad_flat)

            # Create offset matrix: k * (2π / period) for k in [0, 1, ..., period-1]
            # Shape: (n_angles, max_period)
            k_range = torch.arange(max_period, device=device).unsqueeze(
                0
            )  # (1, max_period)
            periods_expanded = periods_flat.unsqueeze(1).to(diff_rad.dtype)

            # Offsets for each angle: k * 2π/period
            offsets = k_range * (
                2.0 * torch.pi / periods_expanded
            )  # (n_angles, max_period)

            # Apply offsets to differences: (n_angles, max_period)
            diff_rad_expanded = diff_rad_flat.unsqueeze(1)  # (n_angles, 1)
            equiv_diffs = diff_rad_expanded - offsets  # (n_angles, max_period)

            # Wrap all equivalent angles to [-pi, pi]
            equiv_diffs_wrapped = torch.remainder(
                equiv_diffs + torch.pi, 2.0 * torch.pi
            ) - torch.pi

            # Mask out invalid offsets (where k >= period for each angle)
            valid_mask = k_range < periods_expanded  # (n_angles, max_period)

            # Set invalid positions to large value so they won't be selected
            equiv_diffs_wrapped_masked = torch.where(
                valid_mask,
                torch.abs(equiv_diffs_wrapped),
                torch.tensor(float("inf"), device=device),
            )

            # Find minimum absolute difference for each angle
            min_indices = torch.argmin(equiv_diffs_wrapped_masked, dim=1)  # (n_angles,)

            # Gather the best wrapped difference for each angle
            diff_wrapped_best = equiv_diffs_wrapped[
                torch.arange(n_angles, device=device), min_indices
            ]

            # Reshape back to original shape
            return diff_wrapped_best.reshape(original_shape)
        else:
            # All periods are 0 or 1, simple wrapping
            return torch.remainder(diff_rad + torch.pi, 2.0 * torch.pi) - torch.pi

    def torsion_deviations_with_sigmas(self, xyz: torch.Tensor):
        """
        Compute torsion deviations (wrapped for periodicity) and sigmas.

        Parameters
        ----------
        xyz : torch.Tensor
            Cartesian coordinates in Å, shape (n_atoms, 3).

        Returns
        -------
        deviations_rad : torch.Tensor
            Wrapped deviations in radians, shape ``(n_torsions,)``; empty when there
            are no torsion restraints.
        sigmas_deg : torch.Tensor
            Standard deviations in degrees (for von Mises NLL).
        """
        group = self.restraints.get("torsion", {}).get("all")
        if group is None:
            return xyz.new_zeros(0), xyz.new_zeros(0)

        idx = group["indices"]
        expected = group["references"]
        sigmas_deg = group["sigmas"]
        periods = group["periods"]

        calculated = self.torsions(idx, xyz)

        # Wrap for periodicity
        diff_rad = (calculated - expected) * (torch.pi / 180.0)
        deviations_rad = self._wrap_torsion_periodicity(diff_rad, periods)

        return deviations_rad, sigmas_deg

    def adp_b_differences(self, adp: torch.Tensor):
        """
        Compute B-factor differences between bonded atoms.

        Parameters
        ----------
        adp : torch.Tensor
            Isotropic B-factors in Å², shape (n_atoms,).

        Returns
        -------
        torch.Tensor
            B-factor differences (B_i - B_j) in Å² for all bonds, in ``adp``'s dtype;
            empty when there are no bonds.
        """
        b_factors = adp

        diffs_list = []
        if "bond" in self.restraints:
            for origin, restraint_group in self.restraints["bond"].items():
                if origin == "all":
                    continue
                indices = restraint_group.get("indices")
                if indices is not None and len(indices) > 0:
                    b1 = b_factors[indices[:, 0]]
                    b2 = b_factors[indices[:, 1]]
                    diffs_list.append(b1 - b2)

        if diffs_list:
            return torch.cat(diffs_list, dim=0)
        return b_factors.new_zeros(0)
