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

    def neighbors(self, i: int) -> np.ndarray:
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
