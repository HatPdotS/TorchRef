"""The topology container: a residue graph over an atom graph.

:class:`Topology` is where a model's atom identity and connectivity live. The residue
level carries the sequence and the inter-residue links; the atom level carries the
atoms, the typed edge blocks and the bond adjacency. Sequence position is reached
through ``atoms.residue_of``; chemical residue identity is per atom so alternate
residue types can share a sequence position.

Identity comes first. :meth:`Topology.from_table` is the one place an atom table's
identity columns become arrays; the result is a node-only topology -- names, elements,
altlocs, residues, chains, record types -- with empty edge blocks and
``connected=False``. Connecting it against the monomer dictionaries is the restraint
build's job. Refinable values (coordinates, B-factors, occupancies) never live here;
they belong to the model's parameter wrappers.

Mutable by design; prefer :meth:`Topology.copy` over editing in place.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Mapping, Set, Tuple

import numpy as np
import torch

from torchref.config import get_int_dtype
from torchref.topology.atom_graph import AtomGraph
from torchref.topology.residue_graph import ResidueGraph
from torchref.utils.device_mixin import DeviceMixin

if TYPE_CHECKING:
    import pandas

#: Per-atom identity columns, as :meth:`Topology.columns` returns them.
IDENTITY_COLUMNS = (
    "name",
    "element",
    "altloc",
    "chain",
    "resseq",
    "icode",
    "resname",
    "is_hetatm",
    "charge",
)


def identity_columns(pdb: "pandas.DataFrame") -> Dict[str, np.ndarray]:
    """The identity half of an atom table, as per-atom arrays.

    Parameters
    ----------
    pdb : pandas.DataFrame
        Atom table with ``name``, ``chainid``, ``resseq`` and ``resname`` columns;
        ``element``, ``altloc``, ``icode``, ``ATOM`` and ``charge`` are optional.

    Returns
    -------
    dict
        :data:`IDENTITY_COLUMNS`, each shape ``(N,)``. Strings are stripped, so a
        padded ``' ALA'`` reads as ``'ALA'``; a blank altloc reads as ``' '``; a
        missing or non-numeric charge as 0.
    """
    import pandas as pd

    n = len(pdb)

    def text(column: str, default: str) -> np.ndarray:
        if column in pdb.columns:
            return np.char.strip(pdb[column].values.astype(str))
        return np.full(n, default)

    altloc = text("altloc", "")
    charge = (
        pd.to_numeric(pdb["charge"], errors="coerce").fillna(0).to_numpy()
        if "charge" in pdb.columns
        else np.zeros(n)
    )
    return {
        "name": text("name", ""),
        "element": text("element", ""),
        "altloc": np.where(np.char.strip(altloc) == "", " ", altloc),
        "chain": text("chainid", ""),
        "resseq": pdb["resseq"].values.astype(np.int64),
        "icode": text("icode", ""),
        "resname": text("resname", ""),
        "is_hetatm": text("ATOM", "ATOM") == "HETATM",
        "charge": charge.astype(np.int64),
    }


@dataclass(eq=False, repr=False)
class Topology(DeviceMixin):
    """Connectivity of one model, at both the residue and the atom level.

    Parameters
    ----------
    residues : ResidueGraph
        Sequence level -- residues as nodes, links as edges.
    atoms : AtomGraph
        Atom level -- atoms as nodes, typed edge blocks, bond adjacency.
    connected : bool, default False
        Whether the edge blocks and residue links have been built. A node-only topology
        (from :meth:`from_table`) answers every identity question but has no edges.

    Notes
    -----
    Holds no refinable parameters, so this is a dataclass rather than an ``nn.Module``.
    Edge indices are ``int64`` constants and no gradient reaches them; gradients reach
    the coordinates that the indices gather.
    """

    residues: ResidueGraph
    atoms: AtomGraph
    connected: bool = False

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @classmethod
    def from_table(cls, pdb: "pandas.DataFrame", device=None) -> "Topology":
        """A node-only topology over an atom table's identity columns.

        Parameters
        ----------
        pdb : pandas.DataFrame
            Atom table; see :func:`identity_columns`. Row order is atom order.
        device : torch.device, optional
            Where ``residue_of`` and the (empty) edge blocks live.

        Returns
        -------
        Topology
            ``connected=False``.
        """
        return cls.from_columns(identity_columns(pdb), device=device)

    @classmethod
    def from_columns(cls, columns: Mapping[str, np.ndarray], device=None) -> "Topology":
        """A node-only topology over per-atom identity arrays.

        Parameters
        ----------
        columns : mapping
            :data:`IDENTITY_COLUMNS`, each shape ``(N,)``. Residues are the contiguous
            runs of ``(chain, resseq, icode)``.
        device : torch.device, optional

        Returns
        -------
        Topology
            ``connected=False``.
        """
        from torchref.topology.residue_graph import build_residue_nodes

        nodes = build_residue_nodes(
            columns["chain"], columns["resseq"], columns["icode"], columns["resname"]
        )
        n_residues = len(nodes["chain"])
        residues = ResidueGraph(
            chain=nodes["chain"],
            resseq=nodes["resseq"],
            icode=nodes["icode"],
            resname=nodes["resname"],
            atom_start=nodes["atom_start"],
            atom_end=nodes["atom_end"],
        )
        residue_of = torch.as_tensor(
            np.repeat(
                np.arange(n_residues, dtype=np.int64),
                nodes["atom_end"] - nodes["atom_start"],
            ),
            dtype=get_int_dtype(),
            device=device,
        )
        atoms = AtomGraph(
            resname=np.asarray(columns["resname"]).copy(),
            name=np.asarray(columns["name"]),
            element=np.asarray(columns["element"]),
            altloc=np.asarray(columns["altloc"]),
            residue_of=residue_of,
            is_hetatm=np.asarray(columns["is_hetatm"], dtype=bool),
            charge=np.asarray(columns["charge"], dtype=np.int64),
        )
        return cls(residues=residues, atoms=atoms)

    def columns(self) -> Dict[str, np.ndarray]:
        """Per-atom identity arrays, sequence-position fields broadcast to atoms.

        Returns
        -------
        dict
            :data:`IDENTITY_COLUMNS`, each shape ``(N,)``, freshly allocated.
        """
        of = self.atoms.residue_of.cpu().numpy()
        return {
            "name": self.atoms.name.copy(),
            "element": self.atoms.element.copy(),
            "altloc": self.atoms.altloc.copy(),
            "chain": self.residues.chain[of],
            "resseq": self.residues.resseq[of],
            "icode": self.residues.icode[of],
            "resname": (
                self.residues.resname[of]
                if self.atoms.resname is None
                else self.atoms.resname.copy()
            ),
            "is_hetatm": self.atoms.is_hetatm.copy(),
            "charge": self.atoms.charge.copy(),
        }

    def gather(self, rows: np.ndarray) -> "Topology":
        """A node-only topology whose atom ``i`` is this one's atom ``rows[i]``.

        Parameters
        ----------
        rows : numpy.ndarray
            Source atom per new atom, shape ``(N_new,)``. Rows may repeat -- a hydrogen
            gathered from its parent -- and the caller overwrites what differs.

        Returns
        -------
        Topology
            ``connected=False``: edges are not carried, and residues are re-derived from
            the gathered order, so the gathered atoms of one residue must stay
            contiguous.
        """
        rows = np.asarray(rows, dtype=np.int64)
        columns = {key: value[rows] for key, value in self.columns().items()}
        return Topology.from_columns(columns, device=self.atoms.residue_of.device)

    def select(self, selection: str) -> torch.Tensor:
        """Atoms matching a Phenix-style selection.

        Parameters
        ----------
        selection : str
            Grammar in :mod:`torchref.utils.selection`, e.g.
            ``"chain A and not resname HOH"``.

        Returns
        -------
        torch.Tensor
            Boolean mask, shape ``(N,)``, on the CPU.
        """
        from torchref.utils.selection import select_atoms

        return select_atoms(self.columns(), selection)

    @property
    def is_water(self) -> np.ndarray:
        """True for atoms of water residues, shape ``(N,)``."""
        from torchref.topology.residue_graph import WATER_RESNAMES

        return np.isin(self.columns()["resname"], list(WATER_RESNAMES))

    @property
    def is_polymer(self) -> np.ndarray:
        """True for atoms of polymer residues, shape ``(N,)``.

        A residue is polymer when its first atom is an ATOM record, the same rule the
        peptide-link search uses.
        """
        first = self.residues.atom_start.astype(np.int64)
        per_residue = ~self.atoms.is_hetatm[first] if len(first) else np.zeros(0, bool)
        return per_residue[self.atoms.residue_of.cpu().numpy()]

    def with_hydrogens(
        self, plan
    ) -> Tuple["Topology", np.ndarray, np.ndarray, np.ndarray]:
        """This topology's atoms with a hydrogen plan's atoms inserted.

        Each residue's planned hydrogens go immediately after its own atoms, never at
        the end: residues are contiguous runs, so appending would split every
        hydrogenated residue into two nodes. A hydrogen inherits its parent's identity
        and takes the plan's ``name``, ``element`` and ``altloc``.

        Parameters
        ----------
        plan : HydrogenPlan
            From :func:`torchref.topology.hydrogens.plan_hydrogens` over this topology.

        Returns
        -------
        topology : Topology
            Node-only, ``connected=False``.
        source : numpy.ndarray
            Row of this topology each new atom was gathered from, ``(N_new,)`` -- the
            parent for a hydrogen. Gather per-atom values with it too.
        old_to_new : numpy.ndarray
            New row of every existing atom, ``(N,)``.
        plan_to_new : numpy.ndarray
            New row of every planned hydrogen, ``(plan.n_hydrogens,)``.
        """
        n_old = self.n_atoms
        if plan.n_hydrogens == 0:
            rows = np.arange(n_old, dtype=np.int64)
            return self.gather(rows), rows, rows.copy(), np.zeros(0, dtype=np.int64)

        by_residue: Dict[int, list] = {}
        for i, residue in enumerate(np.asarray(plan.residue).tolist()):
            by_residue.setdefault(int(residue), []).append(i)

        source, old_to_new = [], np.empty(n_old, dtype=np.int64)
        plan_to_new = np.empty(plan.n_hydrogens, dtype=np.int64)
        offset = 0
        for residue in range(self.n_residues):
            start = int(self.residues.atom_start[residue])
            end = int(self.residues.atom_end[residue])
            source.append(np.arange(start, end, dtype=np.int64))
            old_to_new[start:end] = offset + np.arange(end - start)
            offset += end - start
            members = by_residue.get(residue)
            if members:
                source.append(np.asarray(plan.parent, dtype=np.int64)[members])
                plan_to_new[members] = offset + np.arange(len(members))
                offset += len(members)
        source = np.concatenate(source)

        columns = {key: value[source] for key, value in self.columns().items()}
        altloc = np.asarray(plan.altloc).astype(str)
        for key, values in (
            ("name", np.asarray(plan.name).astype(str)),
            ("element", np.asarray(plan.element).astype(str)),
            ("altloc", np.where(np.char.strip(altloc) == "", " ", altloc)),
        ):
            column = columns[key].astype(
                np.result_type(columns[key].dtype, values.dtype)
            )
            column[plan_to_new] = values
            columns[key] = column
        topology = Topology.from_columns(columns, device=self.atoms.residue_of.device)
        return topology, source, old_to_new, plan_to_new

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
        return Topology(
            residues=self.residues.copy(),
            atoms=self.atoms.copy(),
            connected=self.connected,
        )

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
            selected[mask.to(get_int_dtype())] = True
            mask = selected
        mask = mask.to(device=self.atoms.residue_of.device)

        if int(mask.sum()) == 0:
            raise ValueError("subset would keep no atoms")

        n_kept = int(mask.sum())
        remap = torch.full(
            (self.n_atoms,), -1, dtype=get_int_dtype(), device=mask.device
        )
        remap[mask] = torch.arange(n_kept, dtype=get_int_dtype(), device=mask.device)

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
            (self.n_residues,), -1, dtype=get_int_dtype(), device=mask.device
        )
        residue_remap[torch.as_tensor(residue_keep, device=mask.device)] = torch.arange(
            int(residue_keep.sum()), dtype=get_int_dtype(), device=mask.device
        )

        return Topology(
            residues=self.residues.subset(
                residue_keep,
                atom_start.astype(np.int64),
                atom_end.astype(np.int64),
            ),
            atoms=self.atoms.subset(remap, residue_remap),
            connected=self.connected,
        )

    def neighbors(self, i: int) -> torch.Tensor:
        """Atoms bonded to atom ``i``. Delegates to :meth:`AtomGraph.neighbors`."""
        return self.atoms.neighbors(i)

    def residue_of_atom(self, i: int) -> int:
        """Residue index of atom ``i``."""
        return int(self.atoms.residue_of[i])

    def resname_of_atom(self, i: int) -> str:
        """Chemical residue name of atom ``i``, including alternate residue types."""
        if self.atoms.resname is not None:
            return str(self.atoms.resname[i])
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


__all__ = ["Topology", "identity_columns", "IDENTITY_COLUMNS"]
