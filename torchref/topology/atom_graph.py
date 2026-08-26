"""The atom level of a topology: atoms as nodes, typed edge blocks over them.

Bonds are promoted to a real adjacency structure -- a CSR pair built once from the bond
block -- so :meth:`AtomGraph.neighbors` answers "what is atom *i* bonded to" without
inferring it from restraint index lists. Angles, torsions, chirals and planes stay typed
hyperedge sets read from the monomer library, because the library deliberately does not
restrain every path the bond graph implies.

Every indexing structure here is a tensor, so it moves with ``.to(device)``
alongside the edge blocks. Only the per-atom identifiers are NumPy, because they are
strings.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Tuple

import numpy as np
import torch

from torchref.topology.edges import EdgeBlock
from torchref.utils.device_mixin import DeviceMixin


def _build_csr(bonds: torch.Tensor, n_atoms: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric CSR adjacency from an ``(E, 2)`` bond list.

    Parameters
    ----------
    bonds : torch.Tensor
        Bond atom indices, shape ``(E, 2)``, dtype ``int64``.
    n_atoms : int
        Number of atoms, so isolated trailing atoms still get an entry.

    Returns
    -------
    indptr, indices : torch.Tensor
        ``indices[indptr[i]:indptr[i + 1]]`` are atom ``i``'s bonded neighbours,
        ascending. Each bond contributes both directions.
    """
    device = bonds.device
    if bonds.numel() == 0:
        return (
            torch.zeros(n_atoms + 1, dtype=torch.int64, device=device),
            torch.zeros(0, dtype=torch.int64, device=device),
        )

    src = torch.cat([bonds[:, 0], bonds[:, 1]])
    dst = torch.cat([bonds[:, 1], bonds[:, 0]])

    # Sort by (src, dst). Two stable passes, least significant first, give the same
    # order as a lexicographic sort without materialising a composite key.
    order = torch.argsort(dst, stable=True)
    src, dst = src[order], dst[order]
    order = torch.argsort(src, stable=True)
    src, dst = src[order], dst[order]

    counts = torch.bincount(src, minlength=n_atoms)
    indptr = torch.zeros(n_atoms + 1, dtype=torch.int64, device=device)
    torch.cumsum(counts, dim=0, out=indptr[1:])
    return indptr, dst.to(torch.int64)


def _extend_paths(
    indptr: torch.Tensor, indices: torch.Tensor, paths: torch.Tensor
) -> torch.Tensor:
    """Extend each bonded path by one bonded step, without doubling back.

    Parameters
    ----------
    indptr, indices : torch.Tensor
        CSR adjacency.
    paths : torch.Tensor
        Existing paths, shape ``(P, L)`` with ``L >= 2``, each row a chain of bonded
        atoms.

    Returns
    -------
    torch.Tensor
        Shape ``(P', L + 1)``. A path is extended by every bonded neighbour of its last
        atom except the one it just came from, so ``(i, j, k)`` never yields ``k = i``.
    """
    device = paths.device
    if paths.numel() == 0:
        return torch.zeros((0, paths.shape[1] + 1), dtype=torch.int64, device=device)

    last, prev = paths[:, -1], paths[:, -2]
    counts = indptr[last + 1] - indptr[last]
    total = int(counts.sum())
    if total == 0:
        return torch.zeros((0, paths.shape[1] + 1), dtype=torch.int64, device=device)

    row = torch.repeat_interleave(torch.arange(len(paths), device=device), counts)
    # Offset of each slot within its own neighbour list.
    exclusive = torch.cumsum(counts, dim=0) - counts
    pos = torch.arange(total, device=device) - torch.repeat_interleave(
        exclusive, counts
    )
    nxt = indices[torch.repeat_interleave(indptr[last], counts) + pos]

    keep = nxt != prev[row]
    return torch.cat([paths[row][keep], nxt[keep, None]], dim=1)


@dataclass(eq=False, repr=False)
class AtomGraph(DeviceMixin):
    """Atoms as nodes, typed edges over them, with bond adjacency.

    Parameters
    ----------
    name, element, altloc : numpy.ndarray
        Per-atom identifiers, shape ``(N,)``. Strings, so NumPy rather than tensors;
        residue-level identity is reached through ``residue_of`` rather than duplicated
        here.
    residue_of : torch.Tensor
        Residue index per atom, shape ``(N,)``, dtype ``int64``.
    bonds, angles, torsions, chirals : EdgeBlock
        Typed edge blocks. ``bonds`` also backs the adjacency.
    planes : dict
        ``{n_atoms_in_plane: EdgeBlock}`` -- planes are ragged, so they are grouped by
        atom count the way the plane restraints already are.

    Notes
    -----
    Holds no refinable parameters, so this is a dataclass rather than an ``nn.Module``.
    The adjacency is derived from ``bonds`` at construction and rebuilt by
    :meth:`rebuild_adjacency` if the bond block is replaced.
    """

    name: np.ndarray
    element: np.ndarray
    altloc: np.ndarray
    residue_of: torch.Tensor
    bonds: EdgeBlock
    angles: EdgeBlock
    torsions: EdgeBlock
    chirals: EdgeBlock
    planes: Dict[int, EdgeBlock] = field(default_factory=dict)

    _adj_indptr: Optional[torch.Tensor] = field(default=None, repr=False)
    _adj_indices: Optional[torch.Tensor] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self._adj_indptr is None:
            self.rebuild_adjacency()

    @property
    def device(self) -> torch.device:
        """Where the indexing tensors live. Derived from the bond block."""
        return self.bonds.indices.device

    @property
    def n_atoms(self) -> int:
        """Number of atom nodes."""
        return len(self.name)

    @property
    def is_hydrogen(self) -> torch.Tensor:
        """Boolean mask of hydrogen atoms, shape ``(N,)``."""
        flags = np.char.upper(np.char.strip(self.element.astype(str))) == "H"
        return torch.as_tensor(flags, device=self.bonds.indices.device)

    def rebuild_adjacency(self) -> None:
        """Rebuild the CSR adjacency from the current bond block."""
        self._adj_indptr, self._adj_indices = _build_csr(
            self.bonds.indices, self.n_atoms
        )

    def neighbors(self, i: int) -> torch.Tensor:
        """Atoms bonded to atom ``i``, ascending.

        Returns
        -------
        torch.Tensor
            Neighbour indices, a view into the adjacency, on the graph's device.
        """
        return self._adj_indices[self._adj_indptr[i] : self._adj_indptr[i + 1]]

    def degree(self, i: int = None) -> torch.Tensor:
        """Bonded-neighbour count, for atom ``i`` or for every atom."""
        deg = self._adj_indptr[1:] - self._adj_indptr[:-1]
        return deg if i is None else deg[i]

    def _directed_bonds(self) -> torch.Tensor:
        """Bonds as ``(2E, 2)`` directed pairs."""
        b = self.bonds.indices
        return torch.cat([b, b.flip(1)], dim=0)

    # ------------------------------------------------------------------
    # Non-bonded exclusions
    # ------------------------------------------------------------------

    @staticmethod
    def _pair_set(pairs: torch.Tensor) -> Set[Tuple[int, int]]:
        """``(low, high)`` tuples of a ``(P, 2)`` index tensor, self-pairs dropped."""
        if pairs.numel() == 0:
            return set()
        lo = torch.minimum(pairs[:, 0], pairs[:, 1])
        hi = torch.maximum(pairs[:, 0], pairs[:, 1])
        keep = lo != hi
        return set(zip(lo[keep].cpu().tolist(), hi[keep].cpu().tolist()))

    def exclusions_from_restraint_edges(self) -> Set[Tuple[int, int]]:
        """1-2, 1-3 and 1-4 pairs taken from the bond, angle and torsion **edges**.

        1-2 from every bond, 1-3 from each angle's outer pair, 1-4 from each torsion's
        outer pair. Reproduces exactly the set the non-bonded term has always been
        given.

        This is *not* the same as :meth:`exclusions_12_13_14`: a pair that is 1-3 bonded
        but whose angle the monomer library does not restrain appears there and not
        here, and so takes a repulsion it should not. Kept because switching the
        non-bonded term to the connectivity-derived set changes its value and wants its
        own measurement.

        Returns
        -------
        set of tuple of int
            ``(low, high)`` atom index pairs.
        """
        excl: Set[Tuple[int, int]] = set()
        for block, cols in (
            (self.bonds, (0, 1)),
            (self.angles, (0, 2)),
            (self.torsions, (0, 3)),
        ):
            if block.n_edges:
                excl |= self._pair_set(block.indices[:, cols])
        return excl

    def exclusions_12_13_14(self) -> Set[Tuple[int, int]]:
        """1-2, 1-3 and 1-4 pairs derived from bond **connectivity** alone.

        Walks the adjacency two and three steps out, so the result does not depend on
        which angles and torsions the monomer library happens to restrain. This is the
        physically correct exclusion set; :meth:`exclusions_from_restraint_edges` is the
        one currently wired into the non-bonded term.

        Returns
        -------
        set of tuple of int
            ``(low, high)`` atom index pairs.
        """
        p2 = self._directed_bonds()
        p3 = _extend_paths(self._adj_indptr, self._adj_indices, p2)
        p4 = _extend_paths(self._adj_indptr, self._adj_indices, p3)
        return (
            self._pair_set(p2)
            | self._pair_set(p3[:, (0, 2)])
            | self._pair_set(p4[:, (0, 3)])
        )

    def hydrogen_parents(self) -> Dict[int, torch.Tensor]:
        """``{hydrogen atom: heavy neighbours of its bonded parent}``.

        Taken from bond connectivity, so it does not depend on current coordinates the
        way a distance criterion does.

        Returns
        -------
        dict
            Empty when the graph carries no hydrogens.
        """
        is_h = self.is_hydrogen
        out: Dict[int, torch.Tensor] = {}
        for h in torch.nonzero(is_h, as_tuple=False).flatten().tolist():
            nb = self.neighbors(h)
            heavy = nb[~is_h[nb]]
            if heavy.numel() == 0:
                continue
            parent = int(heavy[0])
            parent_nb = self.neighbors(parent)
            out[h] = parent_nb[~is_h[parent_nb] & (parent_nb != h)]
        return out

    def __repr__(self) -> str:
        return (
            f"AtomGraph(n_atoms={self.n_atoms}, bonds={self.bonds.n_edges}, "
            f"angles={self.angles.n_edges}, torsions={self.torsions.n_edges}, "
            f"chirals={self.chirals.n_edges}, "
            f"planes={sum(b.n_edges for b in self.planes.values())})"
        )


__all__ = ["AtomGraph"]
