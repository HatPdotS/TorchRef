"""Atom identity from a sequence: which atoms a chain of residues has, and in what order.

:func:`sequence_columns` expands chains of residue names into the per-atom identity
arrays :meth:`~torchref.topology.topology.Topology.from_columns` takes, reading each
residue's atoms from its monomer-library template. No atom table and no coordinates are
involved, so the result is the same whatever structure the sequence later folds into;
:meth:`Topology.from_sequence <torchref.topology.topology.Topology.from_sequence>` is
the entry point.

A residue joined to its neighbours is not the free molecule its template describes. The
atoms the peptide link removes are taken from the same ``chem_link`` modifications the
restraint build patches the templates with (:mod:`torchref.topology.templates`), so the
two agree on which atoms a linked residue has.
"""

from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np

from torchref.topology.monomer.cif import (
    find_cif_file_in_library,
    read_cif,
    read_link_definitions,
)
from torchref.topology.monomer.modifications import (
    link_modifications,
    read_mod_definitions,
)
from torchref.topology.residue_graph import polymer_type

#: One-letter amino-acid code to the residue it names. Selenocysteine and pyrrolysine
#: have letters; modified residues do not, and are given by name.
ONE_TO_THREE = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
    "U": "SEC",
    "O": "PYL",
}

_HYDROGENS = frozenset({"H", "D"})

ChainSequence = Union[str, Sequence[str]]


def _residue_names(chain: str, sequence: ChainSequence) -> List[str]:
    """Residue names of one chain; a string is read as one-letter amino-acid codes."""
    if isinstance(sequence, str):
        names = []
        for position, letter in enumerate(sequence.strip().upper(), start=1):
            if letter not in ONE_TO_THREE:
                raise ValueError(
                    f"Chain {chain!r}: no amino acid has the one-letter code "
                    f"{letter!r} (position {position}). Give the chain as a list of "
                    "residue names to use anything else."
                )
            names.append(ONE_TO_THREE[letter])
    else:
        names = [str(name).strip() for name in sequence]
    if not names:
        raise ValueError(f"Chain {chain!r} has no residues.")
    return names


def _component_atoms(resname: str, custom: Mapping) -> Tuple[List[str], List[str]]:
    """Atom names and element symbols of one component, in template order.

    Read from ``custom`` if it defines the component, else from the monomer library.
    The library file is read for its atom list alone, so a component with no bond
    section -- an ion -- is as good as any other.
    """
    if resname in custom and custom[resname].get("atoms") is not None:
        table = custom[resname]["atoms"]
        names = [str(name).strip() for name in table["atom_id"]]
        elements = [str(symbol).strip() for symbol in table["type_symbol"]]
        return names, elements

    import gemmi

    path = find_cif_file_in_library(resname)
    if path is None:
        raise ValueError(
            f"No monomer-library entry for residue {resname!r}. Pass a restraint "
            "dictionary for it with cif_path, or point TORCHREF_MONOMER_LIB at a "
            "library that has it."
        )
    for block in gemmi.cif.read(str(path)):
        table = block.find("_chem_comp_atom.", ["comp_id", "atom_id", "type_symbol"])
        rows = [row for row in table if row.str(0) == resname]
        if rows:
            return [row.str(1) for row in rows], [row.str(2) for row in rows]
    raise ValueError(f"{path} lists no atoms for residue {resname!r}.")


def _deleted_atoms(modification: Optional[str], definitions: Mapping) -> Set[str]:
    """Names of the atoms a ``chem_mod`` modification deletes."""
    table = definitions.get(modification, {}).get("atoms") if modification else None
    if table is None:
        return set()
    deleted = table[table["function"].astype(str).str.strip() == "delete"]
    return {str(name).strip() for name in deleted["atom_id"]}


def sequence_columns(
    chains: Mapping[str, ChainSequence],
    *,
    start: Union[int, Mapping[str, int]] = 1,
    oxt: bool = True,
    cif_path: Union[None, str, Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    """Per-atom identity arrays for chains of residues, without hydrogens.

    Each residue contributes the heavy atoms of its monomer-library template, in
    template order, less the ones a peptide link to a neighbour removes. Consecutive
    residues of a chain are linked when both are peptides by their chemical-component
    type; anything else in a chain -- waters, ions, ligands -- stands alone.

    Parameters
    ----------
    chains : mapping of str to (str or sequence of str)
        Residues per chain ID, in chain order. A string is one-letter amino-acid
        codes (:data:`ONE_TO_THREE`); a sequence holds residue names, which is how
        modified residues, nucleotides and non-polymers are given, e.g.
        ``{"A": "MKV", "B": ["MSE", "ALA"], "W": ["HOH"] * 20}``.
    start : int or mapping of str to int, default 1
        Number of each chain's first residue; the rest count up from it. A mapping
        gives it per chain ID, 1 for chains it leaves out.
    oxt : bool, default True
        Keep the terminal oxygen of each peptide's last residue, as deposited models
        do. Structure generators often leave it out.
    cif_path : str or sequence of str, optional
        Restraint dictionary file(s), read before the monomer library for the atoms
        of the residues they define.

    Returns
    -------
    dict
        :data:`~torchref.topology.topology.IDENTITY_COLUMNS`, each shape ``(N,)``.
        Altlocs and insertion codes are blank, formal charges 0, and ``is_hetatm``
        marks every residue that is neither peptide nor nucleotide.

    Raises
    ------
    ValueError
        For an empty chain, an unknown one-letter code, or a residue with no template.

    Notes
    -----
    Resolving a residue outside the bundled library downloads its template on first
    use. Nucleotides are placed but not linked: the restraint build links peptides
    only.
    """
    custom: Dict = {}
    for path in [cif_path] if isinstance(cif_path, str) else (cif_path or []):
        custom.update(read_cif(path))
    _, link_list = read_link_definitions()
    modifications = link_modifications(link_list)
    definitions = read_mod_definitions()
    trans = modifications.get("TRANS", (None, None))
    proline = modifications.get("PTRANS", trans)

    def link_of(acceptor: str) -> Tuple[Optional[str], Optional[str]]:
        return proline if acceptor == "PRO" else trans

    templates: Dict[str, Tuple[List[str], List[str]]] = {}
    columns: Dict[str, list] = {
        key: [] for key in ("name", "element", "chain", "resseq", "resname")
    }
    hetero: List[bool] = []
    for chain, sequence in chains.items():
        chain = str(chain)
        names = _residue_names(chain, sequence)
        first = start.get(chain, 1) if isinstance(start, Mapping) else start
        kinds = polymer_type(names)
        peptide = kinds == "protein"
        for i, resname in enumerate(names):
            if resname not in templates:
                templates[resname] = _component_atoms(resname, custom)
            removed: Set[str] = set()
            if peptide[i]:
                donates = i + 1 < len(names) and peptide[i + 1]
                if donates:
                    removed |= _deleted_atoms(link_of(names[i + 1])[0], definitions)
                elif not oxt:
                    removed |= _deleted_atoms(trans[0], definitions)
                if i > 0 and peptide[i - 1]:
                    removed |= _deleted_atoms(link_of(resname)[1], definitions)
            for name, element in zip(*templates[resname]):
                if element.upper() in _HYDROGENS or name in removed:
                    continue
                columns["name"].append(name)
                columns["element"].append(element)
                columns["chain"].append(chain)
                columns["resseq"].append(first + i)
                columns["resname"].append(resname)
                hetero.append(kinds[i] == "")

    n = len(hetero)
    return {
        "name": np.asarray(columns["name"], dtype=str),
        "element": np.asarray(columns["element"], dtype=str),
        "altloc": np.full(n, " "),
        "chain": np.asarray(columns["chain"], dtype=str),
        "resseq": np.asarray(columns["resseq"], dtype=np.int64),
        "icode": np.full(n, ""),
        "resname": np.asarray(columns["resname"], dtype=str),
        "is_hetatm": np.asarray(hetero, dtype=bool),
        "charge": np.zeros(n, dtype=np.int64),
    }


__all__ = ["sequence_columns", "ONE_TO_THREE"]
