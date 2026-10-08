"""OpenMM topology over a TorchRef model's own atoms and bonds.

:func:`build_asu_topology` reads atom identity from ``model.ctx.topology`` and the bonds
from the connected restraint topology, so OpenMM's atoms are TorchRef's rows in TorchRef's
order: the map between the two is written down here once, as :attr:`AsuTopology.rows`,
and never re-derived from names, serial numbers or positions. :meth:`AsuTopology.to_openmm`
lays out any number of copies of those atoms -- one for an isolated model, one per
symmetry copy for a crystal -- with some molecules absent from some copies. Nothing is
written to disk.

The hydrogen policy is checked here too (:func:`check_hydrogens`): AMBER's templates need
every hydrogen, and hydrogens TorchRef did not generate are used as they are.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

if TYPE_CHECKING:
    from torchref.model.model import Model
    from torchref.topology.atom_graph import AtomGraph
    from torchref.topology.restraints import Restraints

#: Elements whose bonds are coordination rather than covalent. AMBER models these ions
#: as non-bonded particles, so a bond to one would leave no template matching either
#: partner.
METALS = frozenset(
    "Li Be Na Mg Al K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Rb Sr Y Zr Nb Mo Tc Ru Rh Pd "
    "Ag Cd In Sn Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir "
    "Pt Au Hg Tl Pb Bi U".split()
)

#: Fraction of the hydrogens the monomer dictionaries call for below which a model is
#: rejected. A heavy-atom model has none of them and 1AK5 as deposited about a quarter;
#: a protonation state AMBER accepts in place of the dictionary's (HID or HIE for the
#: doubly protonated HIS, neutral LYS) costs a few percent.
MIN_HYDROGEN_FRACTION = 0.5

#: Longest O3'-P distance, in Å, read as a nucleic-acid backbone bond. The restraint
#: build links peptides but not nucleotides, so these bonds are found here.
_NUCLEIC_BOND_CUTOFF = 2.0

AtomSelection = Union[None, str, np.ndarray, torch.Tensor, Sequence[int]]


@dataclass(eq=False)
class AsuTopology:
    """The selected atoms of one model, laid out the way OpenMM sees them.

    Parameters
    ----------
    rows : numpy.ndarray
        Model row of each atom, shape ``(n,)``, increasing. Atom ``k`` here and in
        every OpenMM topology built from this one is model row ``rows[k]``.
    n_model_atoms : int
        Atom count of the whole model, the stride between copies in
        :meth:`to_openmm`'s particle index.
    name, symbol : numpy.ndarray
        Atom name and element symbol (``'C'``, ``'Fe'``; deuterium as ``'D'``), shape
        ``(n,)``.
    residue_start : numpy.ndarray
        Offsets of each residue's first atom, shape ``(R + 1,)``; residues are
        contiguous runs of atoms.
    residue_name, residue_id, insertion_code, chain_id : numpy.ndarray
        Per-residue identity, shape ``(R,)``.
    bonds : numpy.ndarray
        Covalent bonds as local atom indices, shape ``(B, 2)``.
    molecule_of : numpy.ndarray
        Molecule of each atom, shape ``(n,)``: the connected components of the bonds,
        with every residue kept whole.
    restraints : Restraints
        The connected restraint topology the bonds came from; its ``cif_dict`` holds the
        dictionary chemistry of every residue.
    """

    rows: np.ndarray
    n_model_atoms: int
    name: np.ndarray
    symbol: np.ndarray
    residue_start: np.ndarray
    residue_name: np.ndarray
    residue_id: np.ndarray
    insertion_code: np.ndarray
    chain_id: np.ndarray
    bonds: np.ndarray
    molecule_of: np.ndarray
    restraints: "Restraints"

    @property
    def n_atoms(self) -> int:
        """Number of selected atoms."""
        return len(self.rows)

    @property
    def n_residues(self) -> int:
        """Number of selected residues."""
        return len(self.residue_name)

    @property
    def n_molecules(self) -> int:
        """Number of molecules among the selected atoms."""
        return int(self.molecule_of.max()) + 1 if self.n_atoms else 0

    @property
    def residue_of(self) -> np.ndarray:
        """Residue of each atom, shape ``(n,)``."""
        return np.repeat(np.arange(self.n_residues), np.diff(self.residue_start))

    @property
    def is_heavy(self) -> np.ndarray:
        """True for atoms other than hydrogen and deuterium, shape ``(n,)``."""
        return ~np.isin(self.symbol, ["H", "D"])

    def residue_label(self, r: int) -> str:
        """``'ALA A 12'``-style label of residue ``r``."""
        icode = str(self.insertion_code[r]).strip()
        return f"{self.residue_name[r]} {self.chain_id[r]} {self.residue_id[r]}{icode}"

    def to_openmm(
        self, present: Optional[np.ndarray] = None, box_nm: Optional[np.ndarray] = None
    ) -> Tuple["object", np.ndarray, np.ndarray]:
        """Build an ``openmm.app.Topology`` holding the present copies of each molecule.

        Parameters
        ----------
        present : numpy.ndarray, optional
            Shape ``(C, n_molecules)``, bool: whether copy ``c`` holds molecule ``m``.
            Default: one copy holding everything.
        box_nm : numpy.ndarray, optional
            Periodic box vectors as columns, shape ``(3, 3)``, in nm, already in OpenMM's
            reduced form. Leave None for a non-periodic system.

        Returns
        -------
        topology : openmm.app.Topology
            Copies in order, each in this topology's atom order; a new chain starts
            wherever the chain id changes and at every copy.
        particles : numpy.ndarray
            For OpenMM atom ``k``, the row ``c * n_model_atoms + rows[i]`` of the
            flattened ``(C, n_model_atoms, 3)`` copy coordinates it takes its position
            from, shape ``(n_particles,)``.
        residue_origin : numpy.ndarray
            For each OpenMM residue, the residue of this topology it copies, shape
            ``(n_openmm_residues,)``.
        """
        import openmm
        import openmm.app as app
        import openmm.unit as unit

        if present is None:
            present = np.ones((1, self.n_molecules), dtype=bool)
        elements = [_element(s) for s in self.symbol]
        residue_molecule = self.molecule_of[self.residue_start[:-1]]

        topology = app.Topology()
        particles: List[np.ndarray] = []
        residue_origin: List[int] = []
        for copy, holds in enumerate(present):
            atoms: Dict[int, object] = {}
            chain, chain_key = None, None
            for r in np.flatnonzero(holds[residue_molecule]):
                if chain is None or self.chain_id[r] != chain_key:
                    chain_key = self.chain_id[r]
                    chain = topology.addChain(id=str(chain_key))
                residue = topology.addResidue(
                    str(self.residue_name[r]),
                    chain,
                    id=str(self.residue_id[r]),
                    insertionCode=str(self.insertion_code[r]).strip(),
                )
                residue_origin.append(int(r))
                for k in range(self.residue_start[r], self.residue_start[r + 1]):
                    atoms[k] = topology.addAtom(str(self.name[k]), elements[k], residue)
            local = np.fromiter(atoms.keys(), dtype=np.int64, count=len(atoms))
            particles.append(copy * self.n_model_atoms + self.rows[local])
            for i, j in self.bonds:
                if i in atoms:
                    topology.addBond(atoms[i], atoms[j])
        if box_nm is not None:
            topology.setPeriodicBoxVectors(
                [openmm.Vec3(*(float(v) for v in box_nm[:, k])) for k in range(3)]
                * unit.nanometer
            )
        return topology, np.concatenate(particles), np.asarray(residue_origin)


def build_asu_topology(model: "Model", atoms: AtomSelection = None) -> AsuTopology:
    """Lay out a model's atoms, residues, chains and covalent bonds for OpenMM.

    Parameters
    ----------
    model : Model
        A single-conformation model. Its restraints are reused when built; otherwise
        a connectivity-only build is made and discarded.
    atoms : str, array-like or None
        The atoms to include: a selection string (``Topology.select`` grammar), a
        boolean mask of shape ``(N,)`` or model row indices. Default: all atoms. Must
        hold whole molecules.

    Returns
    -------
    AsuTopology

    Raises
    ------
    ValueError
        If the model has alternate conformations or ``atoms`` splits a molecule.

    Notes
    -----
    Bonds come from the restraint topology. Its ``link`` bonds to a metal
    (:data:`METALS`) are dropped -- AMBER's ions are non-bonded -- and nucleotide
    O3'-P bonds, which the restraint build does not make, are added by distance.
    """
    ctx = model.ctx
    topology = ctx.topology
    if (topology.atoms.altloc != " ").any():
        raise ValueError(
            "OpenMM needs a single conformation; call model.strip_altlocs() first."
        )
    restraints = ctx.restraints
    if restraints is None:
        restraints = ctx.build_restraints(model.xyz(), nonbonded=False, verbose=0)
    graph = restraints.topology.atoms
    if graph.n_atoms != topology.n_atoms:
        raise ValueError(
            "The restraint topology no longer matches the model's atoms; rebuild the "
            "restraints after changing the atom set."
        )

    n = topology.n_atoms
    columns = topology.columns()
    symbol = np.where(
        np.char.upper(np.char.strip(columns["element"].astype(str))) == "D",
        "D",
        graph.symbols,
    )
    bonds = _covalent_bonds(graph, symbol)
    xyz = model.xyz().detach().cpu().numpy()
    bonds = np.concatenate([bonds, _nucleic_backbone_bonds(columns, xyz)])
    residue_of = graph.residue_of.cpu().numpy().astype(np.int64)
    molecule_of = _molecules(n, bonds, residue_of)

    rows = _selected_rows(atoms, topology, n)
    keep = np.zeros(n, dtype=bool)
    keep[rows] = True
    split = np.unique(molecule_of[rows])
    if not np.isin(np.flatnonzero(np.isin(molecule_of, split)), rows).all():
        raise ValueError(
            "The atom selection splits a molecule; select whole molecules (a protein "
            "chain, a ligand with its hydrogens, a water)."
        )

    local = np.full(n, -1, dtype=np.int64)
    local[rows] = np.arange(len(rows))
    both = keep[bonds[:, 0]] & keep[bonds[:, 1]]
    local_bonds = local[bonds[both]]
    _, local_molecule = np.unique(molecule_of[rows], return_inverse=True)

    sel_residue = residue_of[rows]
    starts = np.flatnonzero(np.r_[True, sel_residue[1:] != sel_residue[:-1]])
    first = rows[starts]
    return AsuTopology(
        rows=rows,
        n_model_atoms=n,
        name=columns["name"][rows].astype(str),
        symbol=symbol[rows],
        residue_start=np.r_[starts, len(rows)].astype(np.int64),
        residue_name=columns["resname"][first].astype(str),
        residue_id=np.asarray(columns["resseq"][first]).astype(str),
        insertion_code=columns["icode"][first].astype(str),
        chain_id=columns["chain"][first].astype(str),
        bonds=local_bonds,
        molecule_of=local_molecule.astype(np.int64),
        restraints=restraints,
    )


def check_hydrogens(
    graph: "AtomGraph", rows: np.ndarray, hydrogens_added: bool
) -> Tuple[int, int]:
    """Warn about hydrogens TorchRef did not add, and reject a model missing most.

    Parameters
    ----------
    graph : AtomGraph
        Connected atom graph of the model, carrying the dictionaries' hydrogen counts.
    rows : numpy.ndarray
        Model rows to check, shape ``(n,)``.
    hydrogens_added : bool
        Whether TorchRef generated the model's missing hydrogens. When False a
        ``UserWarning`` is issued every time, whatever the count.

    Returns
    -------
    present, expected : int
        Hydrogens the dictionaries call for that the model holds, and how many they
        call for. Atoms without a dictionary count neither way.

    Raises
    ------
    ValueError
        If fewer than :data:`MIN_HYDROGEN_FRACTION` of the expected hydrogens are
        present.
    """
    if graph.template_h_count is None:
        return 0, 0
    template = graph.template_h_count.cpu().numpy()[rows].astype(np.int64)
    missing = graph.implicit_h_count().cpu().numpy()[rows].astype(np.int64)
    expected = int(template[template > 0].sum())
    present = expected - int(missing.sum())
    if not hydrogens_added:
        warnings.warn(
            f"OpenMM uses the model's own hydrogens, which TorchRef did not add: "
            f"{present} of the {expected} the monomer dictionaries call for are "
            "present. Load with hydrogens='add' or call Model.hydrogenate() to "
            "complete them.",
            UserWarning,
            stacklevel=3,
        )
    if expected and present < MIN_HYDROGEN_FRACTION * expected:
        raise ValueError(
            f"Only {present} of the {expected} hydrogens the monomer dictionaries call "
            "for are present; AMBER needs every one. Load with hydrogens='add' or call "
            "Model.hydrogenate() first."
        )
    return present, expected


def missing_hydrogens(asu: AsuTopology, residues: Sequence[int]) -> Dict[str, int]:
    """Hydrogens each listed residue lacks against its dictionary, where any.

    Parameters
    ----------
    asu : AsuTopology
    residues : sequence of int
        Residues of ``asu``.

    Returns
    -------
    dict
        ``{residue label: missing count}`` for the residues missing at least one.
    """
    missing = asu.restraints.topology.atoms.implicit_h_count()
    if missing is None:
        return {}
    per_atom = missing.cpu().numpy()[asu.rows]
    result = {}
    for r in residues:
        count = int(per_atom[asu.residue_start[r] : asu.residue_start[r + 1]].sum())
        if count:
            result[asu.residue_label(r)] = count
    return result


def _element(symbol: str):
    """The ``openmm.app.Element`` of a symbol; deuterium takes hydrogen's templates."""
    import openmm.app.element as element

    if symbol == "D":
        return element.hydrogen
    try:
        return element.Element.getBySymbol(symbol)
    except KeyError:
        raise ValueError(f"OpenMM knows no element {symbol!r}") from None


def _covalent_bonds(graph: "AtomGraph", symbol: np.ndarray) -> np.ndarray:
    """The restraint bonds, without ``link`` bonds to a metal; shape ``(B, 2)``."""
    bonds = graph.bonds.indices.cpu().numpy().astype(np.int64)
    if "link" not in graph.bonds.origin_bounds:
        return bonds
    start, end = graph.bonds.origin_bounds["link"]
    metal = np.isin(symbol, list(METALS))
    coordination = np.zeros(len(bonds), dtype=bool)
    link = bonds[start:end]
    coordination[start:end] = metal[link[:, 0]] | metal[link[:, 1]]
    return bonds[~coordination]


def _nucleic_backbone_bonds(
    columns: Dict[str, np.ndarray], xyz: np.ndarray
) -> np.ndarray:
    """O3'(i)-P(i+1) bonds between consecutive nucleotides of a chain, ``(B, 2)``."""
    name = columns["name"].astype(str)
    o3 = np.flatnonzero(np.isin(name, ["O3'", "O3*"]))
    phosphorus = np.flatnonzero(name == "P")
    if not len(o3) or not len(phosphorus):
        return np.zeros((0, 2), dtype=np.int64)
    key = np.char.add(
        np.char.add(columns["chain"].astype(str), ":"),
        np.char.add(
            np.asarray(columns["resseq"]).astype(str), columns["icode"].astype(str)
        ),
    )
    # Residue order follows the atom order, so the next residue is the next new key.
    residue = np.cumsum(np.r_[0, key[1:] != key[:-1]])
    p_of_residue = {int(residue[p]): int(p) for p in phosphorus}
    pairs = []
    for i in o3:
        p = p_of_residue.get(int(residue[i]) + 1)
        if (
            p is not None
            and columns["chain"][p] == columns["chain"][i]
            and np.linalg.norm(xyz[p] - xyz[i]) < _NUCLEIC_BOND_CUTOFF
        ):
            pairs.append((int(i), p))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def _molecules(n: int, bonds: np.ndarray, residue_of: np.ndarray) -> np.ndarray:
    """Connected components of the bonds, every residue kept whole; shape ``(n,)``."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    # Chaining each atom to the next one of its residue keeps a residue whose
    # dictionary leaves it in pieces (or that lacks atoms) in one molecule.
    same = np.flatnonzero(residue_of[1:] == residue_of[:-1])
    edges = np.concatenate([bonds, np.stack([same, same + 1], axis=1)])
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return labels.astype(np.int64)


def _selected_rows(atoms: AtomSelection, topology, n: int) -> np.ndarray:
    """Model rows named by a selection string, mask or index list, increasing."""
    if atoms is None:
        return np.arange(n, dtype=np.int64)
    if isinstance(atoms, str):
        mask = topology.select(atoms).cpu().numpy()
        return np.flatnonzero(mask).astype(np.int64)
    values = (
        atoms.detach().cpu().numpy()
        if isinstance(atoms, torch.Tensor)
        else np.asarray(atoms)
    )
    if values.dtype == bool:
        if values.shape != (n,):
            raise ValueError(f"An atom mask must have shape ({n},), got {values.shape}")
        return np.flatnonzero(values).astype(np.int64)
    rows = np.unique(values.astype(np.int64))
    if len(rows) and (rows[0] < 0 or rows[-1] >= n):
        raise ValueError(f"Atom rows must lie in [0, {n})")
    return rows
