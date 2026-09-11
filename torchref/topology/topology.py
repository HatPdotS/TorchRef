"""The topology container: a residue graph over an atom graph.

:class:`Topology` is what the model's connectivity lives in. The residue level carries
the sequence and the inter-residue links; the atom level carries the atoms, the typed
edge blocks and the bond adjacency. Per-atom residue identity is reached through
``atoms.residue_of`` rather than duplicated per atom.

Mutable by design; prefer :meth:`Topology.copy` over editing in place.
"""

from dataclasses import dataclass
from typing import Dict, Set, Tuple

import numpy as np
import torch

from torchref.topology.atom_graph import AtomGraph
from torchref.topology.residue_graph import ResidueGraph
from torchref.utils.device_mixin import DeviceMixin


@dataclass(eq=False, repr=False)
class Topology(DeviceMixin):
    """Connectivity of one model, at both the residue and the atom level.

    Parameters
    ----------
    residues : ResidueGraph
        Sequence level -- residues as nodes, links as edges.
    atoms : AtomGraph
        Atom level -- atoms as nodes, typed edge blocks, bond adjacency.

    Notes
    -----
    Holds no refinable parameters, so this is a dataclass rather than an ``nn.Module``.
    Edge indices are ``int64`` constants and no gradient reaches them; gradients reach
    the coordinates that the indices gather.
    """

    residues: ResidueGraph
    atoms: AtomGraph

    @property
    def device(self) -> torch.device:
        """Where the indexing tensors live. Derived from the atom graph."""
        return self.atoms.device

    @property
    def n_atoms(self) -> int:
        """Number of atom nodes."""
        return self.atoms.n_atoms

    @property
    def n_residues(self) -> int:
        """Number of residue nodes."""
        return self.residues.n_residues

    def copy(self) -> "Topology":
        """An independent copy sharing no storage with this one."""
        return Topology(residues=self.residues.copy(), atoms=self.atoms.copy())

    def subset(self, keep) -> "Topology":
        """The topology over a subset of the atoms.

        Reindexes what survives instead of rebuilding: no CIF is re-read and no template
        is re-matched, which is what made ``Model.select`` expensive.

        Parameters
        ----------
        keep : torch.Tensor or numpy.ndarray
            Boolean mask over atoms, shape ``(N,)``, or integer atom indices. Indices
            are taken as a set, not an order -- the result keeps the topology's own atom
            order, because the edge blocks stay canonical only under a monotone
            relabelling.

        Returns
        -------
        Topology
            Atoms in their original relative order. A residue with no surviving atoms is
            dropped, and any link edge touching it goes with it.

        Notes
        -----
        Selecting part of a residue leaves that residue's restraints partial: an edge
        loses its whole restraint as soon as one of its atoms goes. That is the honest
        outcome -- half a peptide plane is not a plane -- but it means a subset is a
        weaker geometric model, not merely a smaller one.
        """
        mask = torch.as_tensor(keep)
        if mask.dtype != torch.bool:
            selected = torch.zeros(self.n_atoms, dtype=torch.bool)
            selected[mask.to(torch.int64)] = True  # dtype-ok: boolean-mask->index cast for scatter select; int64 index required
            mask = selected
        mask = mask.to(device=self.atoms.residue_of.device)

        if int(mask.sum()) == 0:
            raise ValueError("subset would keep no atoms")

        n_kept = int(mask.sum())
        remap = torch.full((self.n_atoms,), -1, dtype=torch.int64, device=mask.device)  # dtype-ok: atom remap index array (-1 sentinel); int64 index required
        remap[mask] = torch.arange(n_kept, dtype=torch.int64, device=mask.device)  # dtype-ok: arange remap indices; int64 index required

        # A residue survives if any of its atoms does. Counting per residue also
        # gives the new atom ranges, contiguous because the atom order is unchanged.
        residue_of = self.atoms.residue_of
        per_residue = (
            torch.bincount(residue_of[mask], minlength=self.n_residues).cpu().numpy()
        )
        residue_keep = per_residue > 0
        counts = per_residue[residue_keep]
        atom_end = np.cumsum(counts)
        atom_start = atom_end - counts

        residue_remap = torch.full(
            (self.n_residues,), -1, dtype=torch.int64, device=mask.device  # dtype-ok: residue remap index array (-1 sentinel); int64 index required
        )
        residue_remap[torch.as_tensor(residue_keep, device=mask.device)] = torch.arange(
            int(residue_keep.sum()), dtype=torch.int64, device=mask.device  # dtype-ok: arange residue remap indices; int64 index required
        )

        return Topology(
            residues=self.residues.subset(
                residue_keep,
                atom_start.astype(np.int64),
                atom_end.astype(np.int64),
            ),
            atoms=self.atoms.subset(remap, residue_remap),
        )

    def neighbors(self, i: int) -> torch.Tensor:
        """Atoms bonded to atom ``i``. Delegates to :meth:`AtomGraph.neighbors`."""
        return self.atoms.neighbors(i)

    def residue_of_atom(self, i: int) -> int:
        """Residue index of atom ``i``."""
        return int(self.atoms.residue_of[i])

    def resname_of_atom(self, i: int) -> str:
        """Residue name of atom ``i``, joined through the residue graph."""
        return str(self.residues.resname[self.residue_of_atom(i)])

    def edge_block(self, edge_type: str):
        """The :class:`~torchref.topology.edges.EdgeBlock` for one edge type.

        Parameters
        ----------
        edge_type : str
            ``'bond'``, ``'angle'``, ``'torsion'`` or ``'chiral'``. Planes are ragged
            and reached through ``atoms.planes``.
        """
        return {
            "bond": self.atoms.bonds,
            "angle": self.atoms.angles,
            "torsion": self.atoms.torsions,
            "chiral": self.atoms.chirals,
        }[edge_type]

    def tuple_sets(self) -> Dict[str, Dict[str, Set[Tuple[int, ...]]]]:
        """Every edge as ``{edge type: {origin: set of index tuples}}``.

        Order-free, so this is what an equivalence check against another builder should
        compare.
        """
        out: Dict[str, Dict[str, Set[Tuple[int, ...]]]] = {}
        for name in ("bond", "angle", "torsion", "chiral"):
            block = self.edge_block(name)
            out[name] = {o: block.tuple_set(o) for o in block.origins()}
        out["plane"] = {}
        for size, block in self.atoms.planes.items():
            for origin in block.origins():
                out["plane"][f"{size}_atoms/{origin}"] = block.tuple_set(origin)
        return out

    def __repr__(self) -> str:
        return f"Topology({self.residues!r}, {self.atoms!r})"


__all__ = ["Topology"]
