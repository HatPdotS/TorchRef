"""The restraint layer over a topology, and what it takes to build one.

:class:`Restraints` is the orchestrator. Given an atom table it resolves the monomer
dictionaries, builds the :class:`~torchref.topology.topology.Topology`, layers the ideal
values over its edges, derives the non-bonded pair list, and exposes the whole thing as
``restraints[edge_type][origin][property]`` -- three dict lookups into a mapping
assembled once, because the geometry targets read it on every iteration.

Three kinds of thing live here, and only the first is really connectivity:

* the topology and the values keyed to its edges, which are constants for the lifetime
  of an atom set;
* the non-bonded pair list, which is distance-derived and rebuilt as the model moves, so
  it is held apart from the rest;
* the Ramachandran map, a residue-level product of the same build.

Deliberately decoupled from :class:`~torchref.model.Model`: it takes an atom table plus
callables for coordinates, ADPs and van der Waals radii, so it can be built and tested
without one.
"""

from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.nn import Module

from torchref.topology.monomer.cif import (
    find_cif_file_in_library,
    read_cif,
    read_link_definitions,
)
from torchref.config import get_float_dtype
from torchref.utils.debug_utils import DebugMixin
from torchref.utils.device_mixin import DeviceMixin



class Restraints(DeviceMixin, DebugMixin, Module):
    """
    Restraints handler for crystallographic model refinement.

    Builds restraint tensors via the builder classes in ``builders_fast``.
    Decoupled from Model: takes a pdb DataFrame plus callables for coordinates,
    ADPs and VDW radii.

    Parameters
    ----------
    pdb : pd.DataFrame, optional
        DataFrame containing atomic structure data. If None, creates empty shell.
    cif_path : str or list of str, optional
        Path to the CIF restraints dictionary file(s).
    xyz_fn : callable, optional
        Returns current xyz coordinates. Required to build/evaluate when ``pdb``
        is provided.
    adp_fn : callable, optional
        Returns current ADP values. Required for ADP-based restraints.
    vdw_radii_fn : callable, optional
        Returns VDW radii. Required for VDW restraints.
    cell : Cell, optional
        Crystallographic unit cell. Together with ``spacegroup``, enables
        symmetry-aware VDW restraints (contacts with symmetry mates).
    spacegroup : SpaceGroup or str, optional
        Space group. Together with ``cell``, enables symmetry-aware VDW restraints.
    links : pd.DataFrame, optional
        Parsed PDB LINK records; each accepted record adds one bond restraint
        between the two named atoms.
    verbose : int, default 1
        Verbosity level (0=silent, 1=normal, 2=detailed).

    Attributes
    ----------
    restraints : dict
        Restraint groups as ``restraints["bond"]["intra"]["indices"]``. A plain nested
        dict; the per-origin indices are views into ``topology``'s edge blocks.
    topology : Topology
        The connectivity the geometry restraints are defined over.
    cif_dict : dict
        Parsed CIF restraints keyed by residue type; ``missing_residues`` lists
        the types that could not be resolved.
    h_topo
        Riding-hydrogen map, built only when the model carries no hydrogens of its
        own. Empty otherwise; see :mod:`torchref.topology.riding`.
    link_dict, link_list
        Link-type definitions from the monomer library, set only when ``pdb``
        was provided.
    """

    def __init__(
        self,
        pdb: pd.DataFrame = None,
        cif_path=None,
        xyz_fn: Callable[[], torch.Tensor] = None,
        adp_fn: Callable[[], torch.Tensor] = None,
        vdw_radii_fn: Callable[[], torch.Tensor] = None,
        cell=None,
        spacegroup=None,
        links: pd.DataFrame = None,
        verbose: int = 1,
    ):
        """Initialize the Restraints handler."""
        super().__init__()
        self.cif_path = cif_path
        self.verbose = verbose
        self.links = links

        # Store callable functions for coordinate/ADP access
        self._xyz_fn = xyz_fn
        self._adp_fn = adp_fn
        self._vdw_radii_fn = vdw_radii_fn

        # Store crystallographic info for symmetry VDW restraints
        self._cell = cell
        self._spacegroup = spacegroup

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
        if pdb is None:
            self.pdb = None
            self.cif_dict = {}
            self.unique_residues = []
            return

        # Full initialization with pdb
        self.pdb = pdb
        self.unique_residues = pdb.resname.unique()
        self.unique_residues = [
            residue
            for residue in self.unique_residues
            if self.pdb.loc[self.pdb["resname"] == residue, "name"].nunique() > 1
        ]

        # Parse CIF files
        self._load_cif_dictionaries(cif_path)

        # Load link definitions for inter-residue restraints
        if verbose > 1:
            print("Loading link definitions from monomer library...")
        self.link_dict, self.link_list = read_link_definitions()
        if verbose > 1:
            print(f"Loaded {len(self.link_dict)} link types")

        # Build restraints using the new builder pattern
        self.build_restraints()
        if self.verbose > 0:
            self.summary()

    def xyz(self, xyz: torch.Tensor = None) -> torch.Tensor:
        """
        Get current xyz coordinates.

        Parameters
        ----------
        xyz : torch.Tensor, optional
            If provided, returns this tensor directly.
            Otherwise calls the stored xyz_fn callable.

        Returns
        -------
        torch.Tensor
            Current xyz coordinates of shape (n_atoms, 3).
        """
        if xyz is not None:
            return xyz
        if self._xyz_fn is None:
            raise RuntimeError(
                "No xyz callable provided. Initialize with xyz_fn or pass xyz argument."
            )
        return self._xyz_fn()

    def adp(self, adp: torch.Tensor = None) -> torch.Tensor:
        """
        Get current ADP values.

        Parameters
        ----------
        adp : torch.Tensor, optional
            If provided, returns this tensor directly.
            Otherwise calls the stored adp_fn callable.

        Returns
        -------
        torch.Tensor
            Current ADP values. Shape ``(n_atoms,)`` for isotropic B-factors,
            or ``(n_atoms, 6)`` for anisotropic ADPs (the six unique
            components of the U tensor per atom), depending on what the stored
            ``adp_fn`` (or the ``adp`` argument) supplies.
        """
        if adp is not None:
            return adp
        if self._adp_fn is None:
            raise RuntimeError(
                "No adp callable provided. Initialize with adp_fn or pass adp argument."
            )
        return self._adp_fn()

    def get_vdw_radii(self) -> torch.Tensor:
        """
        Get VDW radii for all atoms.

        Returns
        -------
        torch.Tensor
            VDW radii of shape (n_atoms,).
        """
        if self._vdw_radii_fn is None:
            raise RuntimeError(
                "No vdw_radii callable provided. Initialize with vdw_radii_fn."
            )
        return self._vdw_radii_fn()

    # =========================================================================
    # Restraint storage
    # =========================================================================








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

    def _load_cif_dictionaries(self, cif_path):
        """Load CIF dictionaries from provided paths and monomer library."""
        if cif_path:
            if isinstance(cif_path, str):
                try:
                    self.cif_dict = read_cif(cif_path)
                except ValueError as e:
                    print("Error reading CIF file:", e)
                    raise
                except Exception as e:
                    print("Error reading CIF file:", e)
                    self.cif_dict = {}
            elif isinstance(cif_path, list):
                self.cif_dict = {}
                for cif_file in cif_path:
                    try:
                        cif_dict_part = read_cif(cif_file)
                        self.cif_dict.update(cif_dict_part)
                    except ValueError as e:
                        print("Error reading CIF file:", e)
                        raise
                    except Exception as e:
                        print("Error reading CIF file:", e)
            else:
                raise ValueError("cif_path must be a string or a list of strings")
        else:
            self.cif_dict = {}

        # Load missing residues from monomer library
        self.missing_residues = [
            res for res in self.unique_residues if res not in self.cif_dict
        ]
        additional_files = [
            find_cif_file_in_library(res) for res in self.missing_residues
        ]

        for cif_file in additional_files:
            if cif_file is not None:
                if self.verbose > 1:
                    print(cif_file)
                try:
                    additional_cif_dict = read_cif(cif_file)
                    self.cif_dict.update(additional_cif_dict)
                except Exception as e:
                    print("Error reading CIF file:", e)
                    print("This residue will have no restraints applied.")

        self.missing_residues = [
            res for res in self.unique_residues if res not in self.cif_dict
        ]

        if len(self.missing_residues) > 1:
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

    def build_restraints(self):
        """Build the topology, the values over it, and the non-bonded pair list.

        Builds on CPU and moves the result to the ``xyz()`` device at the end.
        """
        try:
            target_device = self.xyz().device
            device = torch.device("cpu")

            from torchref.topology import build_topology_with_values

            self.topology, self._values, extras = build_topology_with_values(
                self.pdb,
                self.cif_dict,
                link_dict=self.link_dict,
                link_list=self.link_list,
                links=self.links,
                xyz=self.xyz().detach().to(device),
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
            self._build_vdw_restraints(
                cutoff=6.0, sigma=0.05, inter_residue_only=False, use_spatial_hash=True
            )

            if target_device.type != "cpu":
                self.to(target_device)

        except Exception as e:
            self.debug_on_error(e, context="Restraints.build_restraints")
            raise







    def _find_nearby_pairs_spatial_hash(self, xyz, cutoff=6.0):
        """Atom pairs within ``cutoff`` of each other, as (M, 2) rows with i < j.

        Cell-list search over cubic cells of side ``cutoff``, checking only the 14
        unique offsets (self + 13 forward neighbours): O(N) memory instead of the
        O(N^2) distance matrix. Runs on CPU regardless of ``xyz``'s device.
        """
        device = xyz.device
        n_atoms = xyz.shape[0]

        if n_atoms == 0:
            return torch.tensor([], dtype=torch.long, device=device).reshape(0, 2)  # dtype-ok: empty atom-pair index tensor; int64 index required

        # Work on CPU to avoid per-iteration GPU kernel launch overhead
        coords = xyz.detach().cpu()
        cell_size = cutoff

        # Assign each atom to a cubic cell
        xyz_min = coords.min(dim=0).values
        cell_idx = ((coords - xyz_min) / cell_size).long()  # (N, 3)

        grid_dims = cell_idx.max(dim=0).values + 1
        gx, gy, gz = grid_dims[0].item(), grid_dims[1].item(), grid_dims[2].item()
        gyz = gy * gz

        # Flat cell index per atom
        flat = cell_idx[:, 0] * gyz + cell_idx[:, 1] * gz + cell_idx[:, 2]

        # Sort atoms by cell so each cell's atoms are contiguous
        order = flat.argsort()
        sorted_flat = flat[order]

        unique_cells, counts = torch.unique_consecutive(
            sorted_flat, return_counts=True
        )
        n_unique = len(unique_cells)
        starts = torch.zeros(n_unique + 1, dtype=torch.long)  # dtype-ok: grid-cell CSR start offsets; int64 index required
        starts[1:] = counts.cumsum(0)

        # Lookup: flat_cell -> index in unique_cells (-1 if empty)
        n_grid = gx * gyz
        cell_lookup = torch.full((n_grid,), -1, dtype=torch.long)  # dtype-ok: cell lookup table (-1 sentinel); int64 index required
        cell_lookup[unique_cells] = torch.arange(n_unique)

        # 14 unique neighbour offsets: self (0,0,0) + 13 forward neighbours.
        # "Forward" = first non-zero component is positive, avoiding double counting.
        offsets_list = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    if (
                        dx > 0
                        or (dx == 0 and dy > 0)
                        or (dx == 0 and dy == 0 and dz >= 0)
                    ):
                        offsets_list.append(
                            (dx, dy, dz, dx * gyz + dy * gz + dz)
                        )

        cutoff_sq = cutoff * cutoff
        pair_chunks = []

        # Move to numpy for tight loop (faster item access than torch on CPU)
        unique_np = unique_cells.numpy()
        starts_np = starts.numpy()
        order_np = order.numpy()
        coords_np = coords.numpy()

        for ci in range(n_unique):
            cell_flat = int(unique_np[ci])
            sa, ea = int(starts_np[ci]), int(starts_np[ci + 1])
            atoms_a = order_np[sa:ea]
            xyz_a = coords_np[atoms_a]  # (na, 3)

            cx = cell_flat // gyz
            cy = (cell_flat % gyz) // gz
            cz = cell_flat % gz

            for dx, dy, dz, off_flat in offsets_list:
                ncx, ncy, ncz = cx + dx, cy + dy, cz + dz
                if (
                    ncx < 0 or ncx >= gx
                    or ncy < 0 or ncy >= gy
                    or ncz < 0 or ncz >= gz
                ):
                    continue

                nb_flat = ncx * gyz + ncy * gz + ncz
                nb_ci = int(cell_lookup[nb_flat])
                if nb_ci < 0:
                    continue

                sb, eb = int(starts_np[nb_ci]), int(starts_np[nb_ci + 1])
                atoms_b = order_np[sb:eb]
                xyz_b = coords_np[atoms_b]  # (nb, 3)

                # Vectorised distance² via broadcasting: (na, nb, 3)
                diff = xyz_a[:, None, :] - xyz_b[None, :, :]
                dist_sq = (diff * diff).sum(axis=-1)  # (na, nb)

                if off_flat == 0:
                    # Self-cell: upper triangle only
                    na = len(atoms_a)
                    if na < 2:
                        continue
                    ii, jj = np.triu_indices(na, k=1)
                    mask = dist_sq[ii, jj] < cutoff_sq
                    if mask.any():
                        ai = atoms_a[ii[mask]]
                        aj = atoms_a[jj[mask]]
                        pairs = np.stack(
                            [np.minimum(ai, aj), np.maximum(ai, aj)], axis=1
                        )
                        pair_chunks.append(pairs)
                else:
                    # Inter-cell: all pairs
                    ii, jj = np.where(dist_sq < cutoff_sq)
                    if len(ii) > 0:
                        ai = atoms_a[ii]
                        bj = atoms_b[jj]
                        pairs = np.stack(
                            [np.minimum(ai, bj), np.maximum(ai, bj)], axis=1
                        )
                        pair_chunks.append(pairs)

        if pair_chunks:
            all_pairs = np.concatenate(pair_chunks, axis=0)
            return torch.from_numpy(all_pairs).to(dtype=torch.long, device=device)  # dtype-ok: atom-pair index array from numpy; int64 index required
        else:
            return torch.tensor([], dtype=torch.long, device=device).reshape(0, 2)  # dtype-ok: empty atom-pair index tensor; int64 index required

    def _expand_with_symmetry_mates(self, xyz, cutoff):
        """Append symmetry-mate positions to ASU ``xyz`` for neighbour search.

        Centroid pre-filtering skips mates that cannot reach within ``cutoff``.
        Returns ``(combined_xyz, provenance)``; the ASU occupies rows ``[:N]`` and
        ``provenance`` maps every row back with numpy ``asu_source_indices``,
        ``symop_indices`` and ``(N, 3)`` ``cell_offsets``.
        """
        from torchref.config import dtypes
        from torchref.symmetry import SpaceGroup

        cell = self._cell
        sg = self._spacegroup
        if not isinstance(sg, SpaceGroup):
            sg = SpaceGroup(sg)

        n_asu = xyz.shape[0]
        device = xyz.device
        fdtype = dtypes.float

        # Work on the model's device throughout
        xyz_det = xyz.detach().to(fdtype)
        xyz_frac = cell.cartesian_to_fractional(xyz_det)

        # Compute centroid and molecule radius for pre-filtering
        centroid_frac = xyz_frac.mean(dim=0)
        centroid_cart = xyz_det.mean(dim=0)
        molecule_radius = (xyz_det - centroid_cart).norm(dim=1).max().item()
        threshold = 2 * molecule_radius + cutoff

        B = cell.fractional_matrix.to(device=device, dtype=fdtype)
        I_mat = torch.eye(3, dtype=fdtype, device=device)

        # Phase 1: centroid pre-filter to find which (symop, offset) combos
        # can produce contacts.  This is a small loop over scalar ops.
        n_ops = sg.n_ops
        matrices = sg.matrices.to(device=device, dtype=fdtype)
        translations = sg.translations.to(device=device, dtype=fdtype)

        valid_ops = []  # list of (op_idx, dx, dy, dz)
        for op_idx in range(n_ops):
            R = matrices[op_idx]
            t = translations[op_idx]
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    for dz in range(-1, 2):
                        if op_idx == 0 and dx == 0 and dy == 0 and dz == 0:
                            continue
                        offset = torch.tensor([dx, dy, dz], dtype=fdtype,
                                              device=device)
                        d_frac = (R - I_mat) @ centroid_frac + t + offset
                        d_cart = B @ d_frac
                        if d_cart.norm().item() <= threshold:
                            valid_ops.append((op_idx, dx, dy, dz))

        if not valid_ops:
            provenance = {
                "asu_source_indices": np.arange(n_asu, dtype=np.int64),
                "symop_indices": np.zeros(n_asu, dtype=np.int64),
                "cell_offsets": np.zeros((n_asu, 3), dtype=np.int64),
            }
            if self.verbose > 0:
                print("  Symmetry expansion: 0 mate(s) within range "
                      f"({n_asu} total atoms for neighbor search)")
            return xyz_det, provenance

        # Phase 2: batch-generate all mate coordinates in one go
        n_valid = len(valid_ops)
        op_indices = [v[0] for v in valid_ops]
        cell_offs = torch.tensor(
            [[v[1], v[2], v[3]] for v in valid_ops], dtype=fdtype,
            device=device,
        )  # (n_valid, 3)

        # Gather rotation matrices and translations for valid ops
        R_batch = matrices[op_indices]          # (n_valid, 3, 3)
        t_batch = translations[op_indices]      # (n_valid, 3)

        # Batched transform: for each valid op, compute R @ xyz_frac.T + t + offset
        # xyz_frac: (N, 3), R_batch: (n_valid, 3, 3)
        # -> (n_valid, 3, N) via batched matmul, then transpose to (n_valid, N, 3)
        xyz_frac_T = xyz_frac.T.unsqueeze(0).expand(n_valid, -1, -1)  # (n_valid, 3, N)
        mate_frac_all = torch.bmm(R_batch, xyz_frac_T).permute(0, 2, 1)  # (n_valid, N, 3)
        mate_frac_all = mate_frac_all + t_batch.unsqueeze(1) + cell_offs.unsqueeze(1)

        # Convert all to Cartesian: (n_valid * N, 3)
        mate_frac_flat = mate_frac_all.reshape(-1, 3)
        mate_cart_flat = cell.fractional_to_cartesian(mate_frac_flat)

        # Build combined coordinate array: ASU + all mates
        combined_xyz = torch.cat(
            [xyz_det, mate_cart_flat], dim=0
        )

        # Build provenance arrays
        asu_source = np.arange(n_asu, dtype=np.int64)
        # ASU block
        all_asu_sources = [asu_source]
        all_symops = [np.zeros(n_asu, dtype=np.int64)]
        all_offsets = [np.zeros((n_asu, 3), dtype=np.int64)]
        # Mate blocks (each has n_asu atoms)
        for op_idx, dx, dy, dz in valid_ops:
            all_asu_sources.append(asu_source)
            all_symops.append(np.full(n_asu, op_idx, dtype=np.int64))
            all_offsets.append(np.tile([dx, dy, dz], (n_asu, 1)).astype(np.int64))

        provenance = {
            "asu_source_indices": np.concatenate(all_asu_sources),
            "symop_indices": np.concatenate(all_symops),
            "cell_offsets": np.concatenate(all_offsets),
        }

        if self.verbose > 0:
            print(f"  Symmetry expansion: {n_valid} mate(s) within range "
                  f"({combined_xyz.shape[0]} total atoms for neighbor search)")

        return combined_xyz, provenance

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
            return torch.tensor([], dtype=torch.long, device=device)  # dtype-ok: empty index tensor; int64 index required

        n_heavy = len(self.pdb)
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
            return torch.tensor([], dtype=torch.long, device=device)  # dtype-ok: empty index tensor; int64 index required

        arr = np.array(list(exclusions), dtype=np.int64)
        max_idx = max(n_heavy + n_h, int(arr.max()) + 1)
        hashes = arr[:, 0] * max_idx + arr[:, 1]
        hashes.sort()
        return torch.tensor(hashes, dtype=torch.long, device=device)  # dtype-ok: grid-cell hash values used as keys/index; int64 required

    def _build_vdw_restraints(
        self, cutoff=6.0, sigma=0.2, inter_residue_only=True, use_spatial_hash=True
    ):
        """Build van der Waals (non-bonded contact) restraints.

        With cell and spacegroup present, includes contacts to symmetry mates via
        the GPU-native periodic grid search; otherwise falls back to
        :meth:`_build_vdw_restraints_legacy` (whose own default cutoff is 5.0).
        Also builds the riding-hydrogen topology for H-VDW evaluation.

        Parameters
        ----------
        cutoff : float, default 6.0
            Contact-search cutoff in Angstroms. Keep it ~1 Å beyond the largest
            heavy-atom VDW sum so the rebuild threshold has margin.
        sigma : float, default 0.2
            Restraint sigma in Angstroms (the production caller passes 0.05).
        inter_residue_only : bool, default True
            If True, only build contacts between atoms in different residues.
        use_spatial_hash : bool, default True
            Use the spatial-hash neighbour search in the legacy (no-symmetry) path.

        Notes
        -----
        Caches the kwargs in ``_vdw_build_kwargs`` and a detached ASU coordinate
        snapshot in ``_last_vdw_build_xyz``; :meth:`rebuild_vdw_restraints` and
        ``NonBondedTarget.maintenance`` both read those.
        """
        self._vdw_build_kwargs = dict(
            cutoff=cutoff,
            sigma=sigma,
            inter_residue_only=inter_residue_only,
            use_spatial_hash=use_spatial_hash,
        )

        if self.verbose > 0:
            print("\nBuilding VDW (non-bonded) restraints...")

        has_symmetry = (
            self._cell is not None
            and self._spacegroup is not None
        )

        # The build (neighbour search, H topology, exclusion hashing) runs on CPU;
        # everything it registers is migrated to target_device at the end, which
        # the maintenance-triggered rebuild path depends on.
        cpu = torch.device("cpu")
        target_device = self.xyz().device if self._xyz_fn is not None else cpu

        def xyz_cpu():
            return self.xyz().detach().to(cpu)

        def vdw_radii_cpu():
            return self.get_vdw_radii().detach().to(cpu)

        # Construct fresh CPU copies — Cell/SpaceGroup ``.to()`` mutates
        # in place, which would silently relocate the model's own Cell/SG.
        if self._cell is not None:
            from torchref.symmetry.cell import Cell
            cell_cpu = Cell(self._cell._data.detach(), device=cpu,
                            dtype=self._cell.dtype)
        else:
            cell_cpu = None
        if self._spacegroup is not None:
            sg_cpu = self._spacegroup.copy().to(cpu)
        else:
            sg_cpu = None

        if has_symmetry:
            from torchref.topology.nonbonded import build_vdw_restraints_gpu

            exclusions = self.topology.atoms.exclusions_from_restraint_edges()
            self._vdw = build_vdw_restraints_gpu(
                xyz_fn=xyz_cpu,
                vdw_radii_fn=vdw_radii_cpu,
                cell=cell_cpu,
                sg=sg_cpu,
                pdb=self.pdb,
                exclusion_set=exclusions,
                cutoff=cutoff,
                sigma=sigma,
                inter_residue_only=inter_residue_only,
                verbose=self.verbose,
            )
        else:
            self._build_vdw_restraints_legacy(
                cutoff=cutoff, sigma=sigma,
                inter_residue_only=inter_residue_only,
                use_spatial_hash=use_spatial_hash,
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
        )

        elements = self.pdb["element"].astype(str).str.strip().values
        if (elements == "H").any():
            self._h_topo = HydrogenTopology(device=cpu)
        else:
            self._h_topo = build_hydrogen_topology(
                pdb=self.pdb,
                device=cpu,
                verbose=self.verbose,
            )
        self._h_excl_hash = self._build_h_exclusion_hash(self._h_topo, cpu)

        # Precompute H candidate pairs from heavy-atom VDW pair list
        vdw_data = self.restraints.get("vdw")
        if vdw_data is not None and self._h_topo.n_hydrogens > 0:
            build_h_candidate_pairs(
                h_topo=self._h_topo,
                vdw_data=vdw_data,
                pdb=self.pdb,
                h_excl_hash=self._h_excl_hash,
                device=cpu,
                verbose=self.verbose,
            )
            # Fill in VDW min distances using combined radii array
            if self._h_topo.has_candidates:
                heavy_radii = vdw_radii_cpu()                 # (N_heavy,)
                h_radii = self._h_topo.h_vdw_radius           # (N_h,) on CPU
                all_radii = torch.cat([heavy_radii, h_radii])
                self._h_topo.cand_min_dist = (
                    all_radii[self._h_topo.cand_idx_i]
                    + all_radii[self._h_topo.cand_idx_j]
                )

        # Snapshot at build time so maintenance() can diff current positions
        # against it; kept on the model device so the compare is one op.
        if self._xyz_fn is not None:
            self._last_vdw_build_xyz = self.xyz().detach().clone()

        # Move the CPU-built pair list, h_topo and h_excl_hash to the model device.
        # The rebuild path has no surrounding migration, so this cannot be dropped.
        if target_device.type != "cpu":
            self.to(target_device)

    def rebuild_vdw_restraints(self) -> None:
        """Refresh the VDW pair list with the kwargs the initial build was given.

        Called by :meth:`NonBondedTarget.maintenance` once max atomic displacement
        since the last build exceeds its threshold. Raises ``RuntimeError`` if no
        initial build has run.
        """
        if not hasattr(self, "_vdw_build_kwargs"):
            raise RuntimeError(
                "rebuild_vdw_restraints called before initial build "
                "— _vdw_build_kwargs is missing"
            )
        self._build_vdw_restraints(**self._vdw_build_kwargs)

    def _build_vdw_restraints_legacy(
        self, cutoff=5.0, sigma=0.2, inter_residue_only=True, use_spatial_hash=True
    ):
        """Legacy VDW restraint builder (no symmetry or CPU fallback)."""

        exclusions = self.topology.atoms.exclusions_from_restraint_edges()
        vdw_radii = self.get_vdw_radii()
        xyz = self.xyz()
        device = xyz.device
        pdb = self.pdb
        n_asu = xyz.shape[0]

        # Expand with symmetry mates if crystallographic info is available
        has_symmetry = (
            self._cell is not None
            and self._spacegroup is not None
        )
        if has_symmetry:
            combined_xyz, provenance = self._expand_with_symmetry_mates(xyz, cutoff)
        else:
            combined_xyz = xyz
            provenance = None

        # Find nearby pairs in the (potentially expanded) coordinate set
        if use_spatial_hash:
            nearby_pairs = self._find_nearby_pairs_spatial_hash(combined_xyz, cutoff)
        else:
            n_total = combined_xyz.shape[0]
            pairs_list = []
            cutoff_sq = cutoff**2
            for i in range(n_total):
                for j in range(i + 1, n_total):
                    dist_sq = ((combined_xyz[i] - combined_xyz[j]) ** 2).sum()
                    if dist_sq < cutoff_sq:
                        pairs_list.append([i, j])
            nearby_pairs = (
                torch.tensor(pairs_list, dtype=torch.long, device=device)  # dtype-ok: atom-pair index tensor; int64 index required
                if pairs_list
                else torch.tensor([], dtype=torch.long, device=device).reshape(0, 2)  # dtype-ok: empty atom-pair index tensor; int64 index required
            )

        empty_result = {
            "indices": torch.tensor([], dtype=torch.long, device=device).reshape(0, 2),  # dtype-ok: empty atom-pair index tensor; int64 index required
            "min_distances": torch.tensor([], dtype=get_float_dtype(), device=device),
            "sigmas": torch.tensor([], dtype=get_float_dtype(), device=device),
            "symop_indices": torch.tensor([], dtype=torch.long, device=device),  # dtype-ok: empty symop index tensor; int64 index required
            "cell_offsets": torch.tensor([], dtype=torch.long, device=device).reshape(0, 3),  # dtype-ok: empty cell-offset index tensor; int64 index required
        }

        if len(nearby_pairs) == 0:
            self._vdw = empty_result
            return

        pairs_np = nearby_pairs.cpu().numpy()

        # Map indices through provenance to get ASU source atoms and symop info
        if provenance is not None:
            prov_asu = provenance["asu_source_indices"]
            prov_sym = provenance["symop_indices"]
            prov_off = provenance["cell_offsets"]

            # Get provenance for each atom in each pair
            idx0 = pairs_np[:, 0]
            idx1 = pairs_np[:, 1]

            asu_src_0 = prov_asu[idx0]
            asu_src_1 = prov_asu[idx1]
            sym_0 = prov_sym[idx0]
            sym_1 = prov_sym[idx1]
            off_0 = prov_off[idx0]
            off_1 = prov_off[idx1]

            is_asu_0 = (sym_0 == 0) & (off_0 == 0).all(axis=1)
            is_asu_1 = (sym_1 == 0) & (off_1 == 0).all(axis=1)

            # Keep only pairs where at least one atom is from the ASU
            has_asu = is_asu_0 | is_asu_1
            pairs_np = pairs_np[has_asu]
            asu_src_0 = asu_src_0[has_asu]
            asu_src_1 = asu_src_1[has_asu]
            sym_0 = sym_0[has_asu]
            sym_1 = sym_1[has_asu]
            off_0 = off_0[has_asu]
            off_1 = off_1[has_asu]
            is_asu_0 = is_asu_0[has_asu]
            is_asu_1 = is_asu_1[has_asu]

            # Normalize: put the ASU atom in position 0, mate in position 1
            # For intra-ASU pairs (both ASU), keep as-is (both are ASU anyway)
            # For symmetry pairs: swap so ASU is first
            swap = ~is_asu_0 & is_asu_1
            if swap.any():
                asu_src_0[swap], asu_src_1[swap] = asu_src_1[swap].copy(), asu_src_0[swap].copy()
                sym_0[swap], sym_1[swap] = sym_1[swap].copy(), sym_0[swap].copy()
                off_0[swap], off_1[swap] = off_1[swap].copy(), off_0[swap].copy()
                is_asu_0[swap] = True
                is_asu_1[swap] = False

            # Final indices: ASU atom indices for both atoms in each pair
            final_i1 = asu_src_0
            final_i2 = asu_src_1
            # Symmetry info comes from the mate atom (position 1)
            final_symop = sym_1
            final_offsets = off_1

            is_both_asu = is_asu_0 & is_asu_1
        else:
            # No symmetry: all pairs are intra-ASU
            final_i1 = pairs_np[:, 0]
            final_i2 = pairs_np[:, 1]
            final_symop = np.zeros(len(pairs_np), dtype=np.int64)
            final_offsets = np.zeros((len(pairs_np), 3), dtype=np.int64)
            is_both_asu = np.ones(len(pairs_np), dtype=bool)

        # --- Filtering ---
        # Bonded exclusions, same-residue, and altloc filters apply only to
        # intra-ASU pairs. Symmetry pairs cannot be bonded.

        # Start with all pairs kept
        keep_mask = np.ones(len(final_i1), dtype=bool)

        # Exclusion mask (bonded 1-2, 1-3, 1-4) -- intra-ASU only
        if exclusions and is_both_asu.any():
            exclusion_arr = np.array(list(exclusions), dtype=np.int64)
            max_idx = max(
                pdb["index"].max() + 1,
                final_i1[is_both_asu].max() + 1,
                final_i2[is_both_asu].max() + 1,
            )
            # Normalize pair order for comparison
            norm_i1 = np.minimum(final_i1, final_i2)
            norm_i2 = np.maximum(final_i1, final_i2)
            pair_hash = norm_i1 * max_idx + norm_i2
            excl_hash = exclusion_arr[:, 0] * max_idx + exclusion_arr[:, 1]
            is_excluded = np.isin(pair_hash, excl_hash)
            # Only apply to intra-ASU pairs
            keep_mask &= ~(is_excluded & is_both_asu)

        # Inter-residue mask -- intra-ASU only
        if inter_residue_only:
            chainid_array = pdb["chainid"].values
            resseq_array = pdb["resseq"].values
            same_residue = (
                (chainid_array[final_i1] == chainid_array[final_i2])
                & (resseq_array[final_i1] == resseq_array[final_i2])
            )
            keep_mask &= ~(same_residue & is_both_asu)

        # Altloc compatibility -- intra-ASU only
        if "altloc" in pdb.columns:
            altloc_array = pdb["altloc"].values.astype(str)
            altloc_array = np.where(
                np.isin(altloc_array, ["", " "]), " ", altloc_array
            )
            altloc_i = altloc_array[final_i1]
            altloc_j = altloc_array[final_i2]
            incompatible_altloc = (
                (altloc_i != " ") & (altloc_j != " ") & (altloc_i != altloc_j)
            )
            keep_mask &= ~(incompatible_altloc & is_both_asu)

        # Apply filter
        final_i1 = final_i1[keep_mask]
        final_i2 = final_i2[keep_mask]
        final_symop = final_symop[keep_mask]
        final_offsets = final_offsets[keep_mask]

        if len(final_i1) == 0:
            self._vdw = empty_result
            return

        # Compute min distances using VDW radii of ASU source atoms.
        vdw_np = vdw_radii.cpu().numpy()
        min_distances = vdw_np[final_i1] + vdw_np[final_i2]

        # Store results
        final_pairs = np.stack([final_i1, final_i2], axis=1)
        self._vdw = {
            "indices": torch.tensor(final_pairs, dtype=torch.long, device=device),  # dtype-ok: final atom-pair index tensor; int64 index required
            "min_distances": torch.tensor(
                min_distances, dtype=get_float_dtype(), device=device
            ),
            "sigmas": torch.full(
                (len(final_pairs),), sigma, dtype=get_float_dtype(), device=device
            ),
            "symop_indices": torch.tensor(
                final_symop, dtype=torch.long, device=device  # dtype-ok: symop index tensor; int64 index required
            ),
            "cell_offsets": torch.tensor(
                final_offsets, dtype=torch.long, device=device  # dtype-ok: cell-offset index tensor; int64 index required
            ),
        }

        if self.verbose > 0:
            scope = "inter-residue" if inter_residue_only else "all"
            msg = f"  Built {len(final_pairs)} VDW restraints ({scope} contacts)"
            if has_symmetry:
                is_sym_pair = (final_symop != 0) | (final_offsets != 0).any(axis=1)
                n_sym_count = int(is_sym_pair.sum())
                msg += f", {n_sym_count} symmetry contacts"
            print(msg)

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
                import torch as _torch
                is_sym = (symop_indices != 0) | (cell_offsets != 0).any(dim=-1)
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



    def bond_lengths(self, idx, xyz: torch.Tensor = None):
        """
        Compute current bond lengths from atomic coordinates.

        Parameters
        ----------
        idx : torch.Tensor
            Bond indices tensor of shape (N, 2).
        xyz : torch.Tensor, optional
            Coordinates tensor of shape (n_atoms, 3).
            If None, uses the stored xyz_fn callable.

        Returns
        -------
        torch.Tensor
            Tensor of bond lengths of shape (N,).
        """
        xyz = self.xyz(xyz)
        if idx is None:
            return torch.tensor([], device=xyz.device)
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

    def bond_deviations(self, xyz: torch.Tensor = None):
        """
        Compute bond length deviations and sigmas.

        Parameters
        ----------
        xyz : torch.Tensor, optional
            Coordinates tensor. If None, uses the stored xyz_fn callable.

        Returns
        -------
        deviations : torch.Tensor
            Calculated minus expected bond lengths in Angstroms.
        sigmas : torch.Tensor
            Standard deviations from CIF library in Angstroms.
        """
        if "all" not in self.restraints["bond"]:
            self.cat_dict()

        idx = self.restraints["bond"]["all"]["indices"]
        references = self.restraints["bond"]["all"]["references"]
        sigmas = self.restraints["bond"]["all"]["sigmas"]

        # Get current bond lengths
        bond_lengths = self.bond_lengths(idx, xyz)
        deviations = bond_lengths - references

        return deviations, sigmas

    def nll_bonds(self, xyz: torch.Tensor = None):
        """
        Compute negative log-likelihood for bond length restraints.

        For Gaussian distribution: NLL = -log(P(x|μ,σ))
        NLL = 0.5 * ((x - μ) / σ)^2 + log(σ) + 0.5 * log(2π)

        This is the true NLL where exp(-NLL) = probability density.

        Parameters
        ----------
        xyz : torch.Tensor, optional
            Coordinates tensor. If None, uses the stored xyz_fn callable.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_bonds,) with negative log-likelihood values.
        """
        from torchref.refinement.targets import gaussian_nll

        deviations, sigmas = self.bond_deviations(xyz)
        return gaussian_nll(deviations, sigmas)

    def angles(self, idx, xyz: torch.Tensor = None):
        """
        Compute current angle values for all angle restraints.

        Parameters
        ----------
        idx : torch.Tensor
            Angle indices tensor of shape (N, 3).
        xyz : torch.Tensor, optional
            Coordinates tensor. If None, uses the stored xyz_fn callable.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_angles,) with current angle values in degrees.
        """
        xyz = self.xyz(xyz)
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

    def angle_deviations(self, xyz: torch.Tensor = None):
        """
        Compute angle deviations and sigmas.

        Parameters
        ----------
        xyz : torch.Tensor, optional
            Coordinates tensor. If None, uses the stored xyz_fn callable.

        Returns
        -------
        deviations : torch.Tensor
            Calculated minus expected angles, in radians. The CIF library
            references are stored in degrees and converted to radians here
            before differencing.
        sigmas : torch.Tensor
            CIF library standard deviations, converted from degrees to radians.
        """
        if "all" not in self.restraints["angle"]:
            self.cat_dict()

        idx = self.restraints["angle"]["all"]["indices"]
        references_rad = self.restraints["angle"]["all"]["references"] * (
            torch.pi / 180.0
        )
        sigmas_rad = self.restraints["angle"]["all"]["sigmas"] * (torch.pi / 180.0)

        calculated_rad = self.angles(idx, xyz) * (torch.pi / 180.0)
        deviations = calculated_rad - references_rad

        return deviations, sigmas_rad

    def nll_angles(self, xyz: torch.Tensor = None):
        """
        Compute negative log-likelihood for angle restraints.

        For Gaussian distribution: NLL = -log(P(x|μ,σ))
        NLL = 0.5 * ((x - μ) / σ)^2 + log(σ) + 0.5 * log(2π)

        This is the true NLL where exp(-NLL) = probability density.

        Parameters
        ----------
        xyz : torch.Tensor, optional
            Coordinates tensor. If None, uses the stored xyz_fn callable.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_angles,) with negative log-likelihood values.
        """
        from torchref.refinement.targets import gaussian_nll

        deviations, sigmas = self.angle_deviations(xyz)
        return gaussian_nll(deviations, sigmas)

    def cat_dict(self):
        """Ensure the combined ``all`` groups are present. Idempotent.

        They are assembled with everything else at build time, so this normally has
        nothing to do; it exists because the geometry targets guard their reads with
        ``if "all" not in ...`` and call it when the guard trips.

        The previous implementation concatenated the origins on each call, and because
        writing ``restraints['bond']['all']`` also registered ``'all'`` as an origin, a
        second call folded the combined group into itself and doubled every restraint.
        Deriving the group from the topology instead makes that impossible: ``all`` is a
        span of the edge block, never an origin in its own right.
        """
        if self.topology is not None and "all" not in self._entries.get("bond", {}):
            self._rebuild_entries()

    def torsions(self, idx, xyz: torch.Tensor = None):
        """
        Compute current torsion angle values for all torsion restraints.

        Parameters
        ----------
        idx : torch.Tensor
            Torsion indices tensor of shape (N, 4).
        xyz : torch.Tensor, optional
            Coordinates tensor. If None, uses the stored xyz_fn callable.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_torsions,) with current torsion values in degrees.
        """
        xyz = self.xyz(xyz)

        pos1 = xyz[idx[:, 0], :]
        pos2 = xyz[idx[:, 1], :]
        pos3 = xyz[idx[:, 2], :]
        pos4 = xyz[idx[:, 3], :]

        # Compute torsion angles using vector math
        b1 = pos2 - pos1
        b2 = pos3 - pos2
        b3 = pos4 - pos3

        # Normalize b2 for projection
        b2_norm = torch.linalg.norm(b2, dim=-1, keepdim=True)
        b2_unit = b2 / b2_norm

        # Compute normals to planes
        n1 = torch.cross(b1, b2, dim=-1)
        n2 = torch.cross(b2, b3, dim=-1)

        # Normalize normals
        n1_unit = n1 / torch.linalg.norm(n1, dim=-1, keepdim=True)
        n2_unit = n2 / torch.linalg.norm(n2, dim=-1, keepdim=True)

        # Compute angle between normals
        m1 = torch.cross(n1_unit, b2_unit, dim=-1)

        x = torch.sum(n1_unit * n2_unit, dim=-1)
        y = torch.sum(m1 * n2_unit, dim=-1)

        torsions_rad = torch.atan2(y, x)
        torsions_deg = torch.rad2deg(torsions_rad)
        return torsions_deg

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
            periods_expanded = periods_flat.unsqueeze(1).float()  # (n_angles, 1)

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


    def torsion_deviations_with_sigmas(self, xyz: torch.Tensor = None):
        """
        Compute torsion deviations (wrapped for periodicity) and sigmas.

        Parameters
        ----------
        xyz : torch.Tensor, optional
            Coordinates tensor. If None, uses the stored xyz_fn callable.

        Returns
        -------
        deviations_rad : torch.Tensor
            Wrapped deviations in radians.
        sigmas_deg : torch.Tensor
            Standard deviations in degrees (for von Mises NLL).
        """
        if "all" not in self.restraints["torsion"]:
            self.cat_dict()

        idx = self.restraints["torsion"]["all"]["indices"]
        expected = self.restraints["torsion"]["all"]["references"]
        sigmas_deg = self.restraints["torsion"]["all"]["sigmas"]
        periods = self.restraints["torsion"]["all"]["periods"]

        calculated = self.torsions(idx, xyz)

        # Wrap for periodicity
        diff_rad = (calculated - expected) * (torch.pi / 180.0)
        deviations_rad = self._wrap_torsion_periodicity(diff_rad, periods)

        return deviations_rad, sigmas_deg




    def adp_b_differences(self, adp: torch.Tensor = None):
        """
        Compute B-factor differences between bonded atoms.

        Parameters
        ----------
        adp : torch.Tensor, optional
            ADP values. If None, uses the stored adp_fn callable.

        Returns
        -------
        torch.Tensor
            Tensor of B-factor differences (B_i - B_j) for all bonds.
        """
        b_factors = self.adp(adp)

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
        return torch.tensor([], device=b_factors.device)

