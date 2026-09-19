"""The sequence level of a topology: residues as nodes, links as edges.

A residue node is one instance of a monomer-library template. Its identity is
``(chain, resseq, icode)`` -- the insertion code included, so residues 100 and 100A are
distinct nodes. Edges are the inter-residue links: peptide bonds, disulfides, and
explicit ``LINK`` records.

Holds no tensors, so this is a plain dataclass; the atom-level tensors live on
:class:`~torchref.topology.atom_graph.AtomGraph`.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

#: SG-SG separation below which two cysteines are taken to be disulfide-bonded.
DISULFIDE_MAX_DISTANCE = 2.5

#: Lower bound guarding against an atom being paired with itself through a
#: coordinate duplicate.
DISULFIDE_MIN_DISTANCE = 0.1


def _residue_runs(
    chain: np.ndarray, resseq: np.ndarray, icode: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Start and end row of each contiguous ``(chain, resseq, icode)`` run.

    Contiguity is assumed rather than checked, matching the reader's guarantee that a
    structure's atoms arrive grouped by residue. A residue split across two
    non-adjacent runs would become two nodes.

    Returns
    -------
    starts, ends : numpy.ndarray
        Half-open row ranges, shape ``(R,)`` each.
    """
    n = len(chain)
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    changed = np.zeros(n, dtype=bool)
    changed[0] = True
    changed[1:] = (
        (chain[1:] != chain[:-1])
        | (resseq[1:] != resseq[:-1])
        | (icode[1:] != icode[:-1])
    )
    starts = np.nonzero(changed)[0].astype(np.int64)
    ends = np.append(starts[1:], n).astype(np.int64)
    return starts, ends


@dataclass
class ResidueGraph:
    """Residues as nodes, inter-residue links as edges.

    Parameters
    ----------
    chain, resseq, icode, resname : numpy.ndarray
        Per-residue identity, shape ``(R,)``.
    template_key : numpy.ndarray
        Restraint-dictionary key per residue, shape ``(R,)``. Either the residue name
        or a link-modified variant such as ``'ALA:DEL-HN1+DEL-OXT'``.
    atom_start, atom_end : numpy.ndarray
        Half-open row range of each residue's atoms, shape ``(R,)``.
    link_pairs : numpy.ndarray
        Residue index pairs, shape ``(L, 2)``. For a peptide link the first entry
        donates its ``C`` and the second its ``N``.
    link_kind : numpy.ndarray
        Link type per edge, shape ``(L,)``: ``'TRANS'``, ``'PTRANS'``, ``'disulf'`` or
        ``'LINK'``.
    """

    chain: np.ndarray
    resseq: np.ndarray
    icode: np.ndarray
    resname: np.ndarray
    template_key: np.ndarray
    atom_start: np.ndarray
    atom_end: np.ndarray
    link_pairs: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), dtype=np.int64)
    )
    link_kind: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype="<U8"))

    @property
    def n_residues(self) -> int:
        """Number of residue nodes."""
        return len(self.chain)

    def key(self, i: int) -> Tuple[str, int, str]:
        """Identity of residue ``i`` as ``(chain, resseq, icode)``."""
        return (str(self.chain[i]), int(self.resseq[i]), str(self.icode[i]))

    def atom_rows(self, i: int) -> range:
        """Row range of residue ``i``'s atoms."""
        return range(int(self.atom_start[i]), int(self.atom_end[i]))

    def copy(self) -> "ResidueGraph":
        """An independent copy sharing no arrays with this one."""
        return ResidueGraph(
            chain=self.chain.copy(),
            resseq=self.resseq.copy(),
            icode=self.icode.copy(),
            resname=self.resname.copy(),
            template_key=self.template_key.copy(),
            atom_start=self.atom_start.copy(),
            atom_end=self.atom_end.copy(),
            link_pairs=self.link_pairs.copy(),
            link_kind=self.link_kind.copy(),
        )

    def subset(
        self, keep: np.ndarray, atom_start: np.ndarray, atom_end: np.ndarray
    ) -> "ResidueGraph":
        """The residues in ``keep``, with the atom ranges the caller recomputed.

        Parameters
        ----------
        keep : numpy.ndarray
            Boolean mask over residues, shape ``(R,)``.
        atom_start, atom_end : numpy.ndarray
            New half-open atom ranges for the surviving residues, in their order, shape
            ``(R_kept,)``. Passed in rather than derived here because only the caller
            knows how the atoms were renumbered.

        Returns
        -------
        ResidueGraph
            Link edges are kept only where **both** endpoints survive, and reindexed. A
            peptide bond to a residue that is gone is not a peptide bond, and keeping it
            would leave an edge pointing outside the graph.
        """
        remap = np.full(self.n_residues, -1, dtype=np.int64)
        remap[keep] = np.arange(int(keep.sum()), dtype=np.int64)

        if len(self.link_pairs):
            mapped = remap[self.link_pairs]
            survives = (mapped >= 0).all(axis=1)
            link_pairs = mapped[survives]
            link_kind = self.link_kind[survives]
        else:
            link_pairs = np.zeros((0, 2), dtype=np.int64)
            link_kind = np.zeros(0, dtype="<U8")

        return ResidueGraph(
            chain=self.chain[keep],
            resseq=self.resseq[keep],
            icode=self.icode[keep],
            resname=self.resname[keep],
            template_key=self.template_key[keep],
            atom_start=atom_start,
            atom_end=atom_end,
            link_pairs=link_pairs,
            link_kind=link_kind,
        )

    def links_of_kind(self, kind: str) -> np.ndarray:
        """Link edges of one kind, shape ``(L_k, 2)``."""
        if len(self.link_kind) == 0:
            return np.zeros((0, 2), dtype=np.int64)
        return self.link_pairs[self.link_kind == kind]

    def __repr__(self) -> str:
        kinds = (
            {k: int((self.link_kind == k).sum()) for k in np.unique(self.link_kind)}
            if len(self.link_kind)
            else {}
        )
        return f"ResidueGraph(n_residues={self.n_residues}, links={kinds})"


def build_residue_nodes(
    chain: np.ndarray,
    resseq: np.ndarray,
    icode: np.ndarray,
    resname: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Per-residue identity arrays and atom ranges from per-atom columns.

    Returns
    -------
    dict
        ``chain``, ``resseq``, ``icode``, ``resname``, ``atom_start``, ``atom_end``,
        each shape ``(R,)``.
    """
    starts, ends = _residue_runs(chain, resseq, icode)
    return {
        "chain": chain[starts],
        "resseq": resseq[starts],
        "icode": icode[starts],
        "resname": resname[starts],
        "atom_start": starts,
        "atom_end": ends,
    }


def find_peptide_links(
    nodes: Dict[str, np.ndarray],
    names_by_residue: List[set],
) -> List[Tuple[int, int]]:
    """Sequence-adjacent residue pairs carrying a C-N peptide bond.

    Two residues are sequence-adjacent when they are neighbours in their chain's
    ``(resseq, icode)`` ordering **and** either share a ``resseq`` -- an insertion-code
    step such as 100 to 100A -- or differ by exactly one. The second condition is what
    stops a chain break being bridged: residues 49 and 56 are neighbours in the ordering
    but not in the sequence.

    The C/N test then narrows to pairs that actually carry the bond, which is the
    condition the link builders apply implicitly when they look the two atoms up.

    Parameters
    ----------
    nodes : dict
        Output of :func:`build_residue_nodes`.
    names_by_residue : list of set
        Atom names present in each residue.

    Returns
    -------
    list of tuple of int
        ``(residue donating C, residue donating N)`` pairs.
    """
    by_chain: Dict[str, List[int]] = {}
    for i in range(len(nodes["chain"])):
        by_chain.setdefault(str(nodes["chain"][i]), []).append(i)

    pairs = []
    for members in by_chain.values():
        ordered = sorted(
            members, key=lambda i: (int(nodes["resseq"][i]), str(nodes["icode"][i]))
        )
        for a, b in zip(ordered, ordered[1:]):
            step = int(nodes["resseq"][b]) - int(nodes["resseq"][a])
            if step not in (0, 1):
                continue
            if "C" in names_by_residue[a] and "N" in names_by_residue[b]:
                pairs.append((a, b))
    return pairs


def find_disulfide_links(
    sg_rows: Sequence[int],
    residue_of_row: Dict[int, int],
    xyz: torch.Tensor,
) -> List[Tuple[int, int]]:
    """``SG`` atom pairs within :data:`DISULFIDE_MAX_DISTANCE` in different residues.

    Pairing is per **atom**, not per residue, because a cysteine modelled in two
    alternative conformations carries two ``SG`` atoms and each can form its own bond.
    Pairing per residue would keep only one of them. Two ``SG`` atoms of the same
    residue -- its own alternative conformers -- are within bonding distance of each
    other and are excluded by the differing-residue test.

    Parameters
    ----------
    sg_rows : sequence of int
        Atom rows of every ``SG`` under consideration.
    residue_of_row : dict
        ``{atom row: residue index}`` for those rows.
    xyz : torch.Tensor
        Cartesian coordinates, shape ``(N, 3)``.

    Returns
    -------
    list of tuple of int
        Atom-row pairs, lower row first, ascending.
    """
    rows = list(sg_rows)
    if len(rows) < 2:
        return []
    idx = torch.as_tensor(rows, dtype=torch.int64, device=xyz.device)  # dtype-ok: residue-atom index tensor; int64 index required
    dist = torch.cdist(xyz[idx], xyz[idx])
    close = (dist > DISULFIDE_MIN_DISTANCE) & (dist < DISULFIDE_MAX_DISTANCE)

    a, b = torch.triu_indices(len(rows), len(rows), offset=1, device=xyz.device)
    hit = close[a, b]
    pairs = []
    for i, j in zip(a[hit].cpu().tolist(), b[hit].cpu().tolist()):
        row_i, row_j = rows[i], rows[j]
        if residue_of_row[row_i] != residue_of_row[row_j]:
            pairs.append((row_i, row_j))
    return pairs


__all__ = [
    "ResidueGraph",
    "build_residue_nodes",
    "find_peptide_links",
    "find_disulfide_links",
    "DISULFIDE_MAX_DISTANCE",
    "DISULFIDE_MIN_DISTANCE",
]
