"""Connect a node-only :class:`~torchref.topology.Topology` to the dictionaries.

The input carries identity only (:meth:`Topology.from_table`); this module adds the
edges. Intra-residue edges are matched template by template through the matchers in
:mod:`torchref.topology.matchers`. Inter-residue edges come from the
``InterResidue*Builder`` classes over the residue graph's peptide links, disulfides are
found by SG-SG distance, and ``LINK`` records are resolved by residue identity. Every
edge index is an atom row of the topology.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from torchref.config import get_int_dtype
from torchref.topology.builders import (
    InterResidueAngleBuilder,
    InterResidueBondBuilder,
    InterResiduePlaneBuilder,
    InterResidueTorsionBuilder,
    PeptideResidues,
    PreprocessedCIF,
)
from torchref.topology.matchers import (
    match_angles,
    match_bonds,
    match_chirals,
    match_torsions,
)
from torchref.topology.atom_graph import AtomGraph
from torchref.topology.edges import EdgeBlock, assemble_origins
from torchref.topology.residue_graph import (
    ResidueGraph,
    find_disulfide_links,
    find_peptide_links,
)
from torchref.topology.restraint_sets import to_tensor
from torchref.topology.templates import resolve_template_keys
from torchref.topology.topology import Topology

#: Initial size of the matcher work arrays, grown on demand.
_WORK = 64


def _atom_columns(topology: Topology) -> Dict[str, np.ndarray]:
    """Per-atom identity arrays the matchers read, ``record`` and ``index`` included.

    ``index`` is the atom row: edge indices are rows of the topology.
    """
    cols = topology.columns()
    cols["record"] = np.where(cols.pop("is_hetatm"), "HETATM", "ATOM")
    cols["index"] = np.arange(topology.n_atoms, dtype=np.int64)
    return cols


def _chemical_nodes(cols, nodes, peptide_pairs):
    """Expand sequence positions into chemical identities for template matching.

    Blank-altloc atoms participate in every chemical conformer at their position.
    ``index`` continues to address the original atom order, including when shared
    atoms are duplicated in this temporary matching view.
    """
    rows, identities, owners, starts, ends = [], [], [], [], []
    variants = {}
    for r, (start, end) in enumerate(zip(nodes["atom_start"], nodes["atom_end"])):
        source = np.arange(int(start), int(end))
        names = list(dict.fromkeys(cols["resname"][source].tolist()))
        variants[r] = []
        for rn in names:
            variants[r].append(len(owners))
            chosen = source[
                (cols["resname"][source] == rn) | (cols["altloc"][source] == " ")
            ]
            starts.append(len(rows))
            rows.extend(chosen.tolist())
            identities.extend([rn] * len(chosen))
            ends.append(len(rows))
            owners.append(r)
    rows = np.asarray(rows, dtype=np.int64)
    owners = np.asarray(owners, dtype=np.int64)
    expanded = {k: v[rows] for k, v in cols.items()}
    expanded["resname"] = np.asarray(identities)
    chemical = {k: v[owners] for k, v in nodes.items()}
    chemical["resname"] = expanded["resname"][starts]
    chemical["atom_start"] = np.asarray(starts, dtype=np.int64)
    chemical["atom_end"] = np.asarray(ends, dtype=np.int64)
    pairs = [(a, b) for i, j in peptide_pairs for a in variants[i] for b in variants[j]]
    return expanded, chemical, pairs, rows, owners


def _conformers(
    cols: Dict[str, np.ndarray], start: int, end: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Atom name/index arrays per alternative conformation of one residue.

    A residue without altlocs yields one conformation holding all its atoms. Otherwise
    one per altloc, each holding the blank-altloc atoms plus that altloc's own -- so a
    restraint spanning a shared backbone and a branching side chain is emitted once per
    conformer.
    """
    names = cols["name"][start:end]
    indices = cols["index"][start:end]
    altlocs = cols["altloc"][start:end]
    unique = np.unique(altlocs)

    if len(unique) == 1 and unique[0] == " ":
        return [(names, indices)]
    if " " in unique:
        common = altlocs == " "
        out = []
        for alt in unique:
            if alt == " ":
                continue
            m = altlocs == alt
            out.append(
                (
                    np.concatenate([names[common], names[m]]),
                    np.concatenate([indices[common], indices[m]]),
                )
            )
        return out
    return [(names[altlocs == a], indices[altlocs == a]) for a in unique]


def _atom_types(
    cols: Dict[str, np.ndarray],
    nodes: Dict[str, np.ndarray],
    template_key: np.ndarray,
    comp_dict: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-atom energy type and template hydrogen count, by name in the patched template.

    Returns
    -------
    energy_type : numpy.ndarray
        Shape ``(N,)``, ``''`` where the residue has no template or the atom is not
        in it.
    template_h_count : numpy.ndarray
        Shape ``(N,)``, ``int8``; hydrogens the atom carries in its template, ``0``
        for template atoms with none (hydrogens included), ``-1`` where unknown.
    """
    from torchref.topology.hydrogens import template_atom_types

    n_atoms = len(cols["name"])
    energy = np.full(n_atoms, "", dtype="<U8")
    counts = np.full(n_atoms, -1, dtype=np.int8)
    cache: Dict[str, Tuple[Dict[str, str], Dict[str, int]]] = {}
    for r in range(len(nodes["chain"])):
        key = str(template_key[r])
        component = comp_dict.get(key)
        if component is None:
            continue
        if key not in cache:
            cache[key] = template_atom_types(component)
        types, h_count = cache[key]
        if not types:
            continue
        start, end = int(nodes["atom_start"][r]), int(nodes["atom_end"][r])
        for row in range(start, end):
            name = cols["name"][row]
            if name in types:
                energy[row] = types[name]
                counts[row] = h_count.get(name, 0)
    return energy, counts


def _match_intra(
    cols: Dict[str, np.ndarray],
    nodes: Dict[str, np.ndarray],
    template_key: np.ndarray,
    pp_cif: PreprocessedCIF,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
    """Intra-residue edges and the ideal values that belong to them.

    Keyed ``bonds`` / ``angles`` / ``torsions`` / ``chirals``.

    Emitted only where every named atom of a library restraint is present in the
    conformation, which is the condition the matchers apply.

    Returns
    -------
    indices : dict
        ``{kind: (E, k) array}``.
    values : dict
        ``{kind: {property: (E,) array}}``, accumulated row-for-row with the indices.
    """
    acc: Dict[str, List[np.ndarray]] = {
        "bonds": [],
        "angles": [],
        "torsions": [],
        "chirals": [],
    }
    val: Dict[str, Dict[str, List[np.ndarray]]] = {
        "bonds": {"references": [], "sigmas": []},
        "angles": {"references": [], "sigmas": []},
        "torsions": {"references": [], "sigmas": [], "periods": []},
        "chirals": {"ideal_volumes": [], "sigmas": []},
    }
    work = {k: np.zeros(_WORK, dtype=np.int64) for k in ("i1", "i2", "i3", "i4", "per")}
    work["f1"] = np.zeros(_WORK, dtype=np.float64)
    work["f2"] = np.zeros(_WORK, dtype=np.float64)
    size = _WORK

    for r in range(len(nodes["chain"])):
        key = str(template_key[r])
        start, end = int(nodes["atom_start"][r]), int(nodes["atom_end"][r])
        needed = max(
            len(pp_cif.bonds.get(key, {}).get("atom1", ())),
            len(pp_cif.angles.get(key, {}).get("atom1", ())),
            len(pp_cif.torsions.get(key, {}).get("atom1", ())),
            len(pp_cif.chirals.get(key, {}).get("atom1", ())),
        )
        if needed == 0:
            continue
        if needed > size:
            size = needed * 2
            work = {k: np.zeros(size, dtype=v.dtype) for k, v in work.items()}

        for names, indices in _conformers(cols, start, end):
            if key in pp_cif.bonds:
                b = pp_cif.bonds[key]
                n = match_bonds(
                    names,
                    indices,
                    b["atom1"],
                    b["atom2"],
                    b["value"],
                    b["sigma"],
                    work["i1"],
                    work["i2"],
                    work["f1"],
                    work["f2"],
                )
                if n:
                    acc["bonds"].append(
                        np.column_stack([work["i1"][:n].copy(), work["i2"][:n].copy()])
                    )
                    val["bonds"]["references"].append(work["f1"][:n].copy())
                    val["bonds"]["sigmas"].append(work["f2"][:n].copy())
            if key in pp_cif.angles:
                a = pp_cif.angles[key]
                n = match_angles(
                    names,
                    indices,
                    a["atom1"],
                    a["atom2"],
                    a["atom3"],
                    a["value"],
                    a["sigma"],
                    work["i1"],
                    work["i2"],
                    work["i3"],
                    work["f1"],
                    work["f2"],
                )
                if n:
                    acc["angles"].append(
                        np.column_stack(
                            [
                                work["i1"][:n].copy(),
                                work["i2"][:n].copy(),
                                work["i3"][:n].copy(),
                            ]
                        )
                    )
                    val["angles"]["references"].append(work["f1"][:n].copy())
                    val["angles"]["sigmas"].append(work["f2"][:n].copy())
            if key in pp_cif.torsions:
                t = pp_cif.torsions[key]
                n = match_torsions(
                    names,
                    indices,
                    t["atom1"],
                    t["atom2"],
                    t["atom3"],
                    t["atom4"],
                    t["value"],
                    t["sigma"],
                    t["period"],
                    work["i1"],
                    work["i2"],
                    work["i3"],
                    work["i4"],
                    work["f1"],
                    work["f2"],
                    work["per"],
                )
                if n:
                    acc["torsions"].append(
                        np.column_stack(
                            [
                                work["i1"][:n].copy(),
                                work["i2"][:n].copy(),
                                work["i3"][:n].copy(),
                                work["i4"][:n].copy(),
                            ]
                        )
                    )
                    val["torsions"]["references"].append(work["f1"][:n].copy())
                    val["torsions"]["sigmas"].append(work["f2"][:n].copy())
                    val["torsions"]["periods"].append(work["per"][:n].copy())
            if key in pp_cif.chirals:
                c = pp_cif.chirals[key]
                n = match_chirals(
                    names,
                    indices,
                    c["center"],
                    c["atom1"],
                    c["atom2"],
                    c["atom3"],
                    c["volume_sign"],
                    c["sigma"],
                    work["i1"],
                    work["i2"],
                    work["i3"],
                    work["i4"],
                    work["f1"],
                    work["f2"],
                )
                if n:
                    acc["chirals"].append(
                        np.column_stack(
                            [
                                work["i1"][:n].copy(),
                                work["i2"][:n].copy(),
                                work["i3"][:n].copy(),
                                work["i4"][:n].copy(),
                            ]
                        )
                    )
                    # Ideal volume is the sign times a typical tetrahedral volume. A
                    # sign of 0 ('both' / 'either') stays exactly 0, which the chiral
                    # target reads as an achiral centre and restrains |volume| instead.
                    val["chirals"]["ideal_volumes"].append(work["f1"][:n].copy() * 2.5)
                    val["chirals"]["sigmas"].append(work["f2"][:n].copy())

    arity = {"bonds": 2, "angles": 3, "torsions": 4, "chirals": 4}
    indices = {
        k: (np.concatenate(v, axis=0) if v else np.zeros((0, arity[k]), dtype=np.int64))
        for k, v in acc.items()
    }
    values: Dict[str, Dict[str, np.ndarray]] = {}
    for kind, properties in val.items():
        joined: Dict[str, np.ndarray] = {}
        for prop, chunks in properties.items():
            if not chunks:
                continue
            array = np.concatenate(chunks)
            if prop == "sigmas":
                # A zero sigma divides by zero in the loss, so it is floored.
                array = np.where(array == 0, 1e-4, array)
            joined[prop] = array
        values[kind] = joined
    return indices, values


def _match_intra_planes(
    cols: Dict[str, np.ndarray],
    nodes: Dict[str, np.ndarray],
    template_key: np.ndarray,
    pp_cif: PreprocessedCIF,
) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict[str, np.ndarray]]]:
    """Intra-residue planes grouped by how many atoms survived matching.

    Missing atoms are dropped rather than voiding the plane; a plane is kept once at
    least three of its atoms are present, so its arity depends on the model. Sigmas are
    per atom, not per plane, so they carry the same ``(E, k)`` shape as the indices.
    """
    by_size: Dict[int, List[np.ndarray]] = {}
    sigmas_by_size: Dict[int, List[np.ndarray]] = {}
    for r in range(len(nodes["chain"])):
        key = str(template_key[r])
        if key not in pp_cif.planes:
            continue
        start, end = int(nodes["atom_start"][r]), int(nodes["atom_end"][r])
        for names, indices in _conformers(cols, start, end):
            # Last-wins on a duplicate name, matching PlaneRestraintBuilder.
            name_to_idx = dict(zip(names, indices))
            for plane in pp_cif.planes[key]:
                present = []
                present_sigmas = []
                for position, atom_name in enumerate(plane["atoms"]):
                    if atom_name in name_to_idx:
                        present.append(name_to_idx[atom_name])
                        present_sigmas.append(plane["sigmas"][position])
                if len(present) >= 3:
                    by_size.setdefault(len(present), []).append(
                        np.asarray(present, dtype=np.int64)
                    )
                    sigmas_by_size.setdefault(len(present), []).append(
                        np.asarray(present_sigmas, dtype=np.float64)
                    )

    indices_out = {n: np.stack(rows, axis=0) for n, rows in by_size.items()}
    values_out = {
        n: {
            "sigmas": np.where(
                np.stack(rows, axis=0) == 0, 1e-4, np.stack(rows, axis=0)
            )
        }
        for n, rows in sigmas_by_size.items()
    }
    return indices_out, values_out


def _inter_residue_edges(
    residues: PeptideResidues,
    link_dict: Optional[Dict],
    verbose: int,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict], Dict[str, Dict]]:
    """Peptide edges, their values, and the Ramachandran pairing, from the builders.

    Reuses ``InterResidue*Builder`` rather than reimplementing the link geometry. The
    pairs are the residue graph's peptide links, so an insertion-code step (100 to
    100A) is linked like any other.

    Returns
    -------
    indices : dict
        ``{edge type: {origin: (E, k) array}}``.
    values : dict
        ``{edge type: {origin: {property: array}}}`` -- every property a builder
        returned besides the indices, so ``omega``'s ``is_proline`` comes along without
        being named here.
    extras : dict
        Non-edge products of the same pass, currently the ``ramachandran`` phi/psi
        pairing and its surface types.
    """
    indices: Dict[str, Dict[str, np.ndarray]] = {
        "bond": {},
        "angle": {},
        "torsion": {},
        "plane": {},
    }
    values: Dict[str, Dict] = {"bond": {}, "angle": {}, "torsion": {}, "plane": {}}
    extras: Dict[str, Dict] = {}
    if not link_dict or "TRANS" not in link_dict:
        return indices, values, extras

    cpu = torch.device("cpu")
    trans = link_dict["TRANS"]
    ptrans = link_dict.get("PTRANS")

    def split(group):
        """A builder group as ``(indices array, {property: array})``."""
        rows = group["indices"].cpu().numpy()
        rest = {
            prop: tensor.cpu().numpy()
            for prop, tensor in group.items()
            if prop != "indices" and tensor is not None
        }
        return rows, rest

    def per_link(builder):
        """One builder's groups, X-Pro pairs from PTRANS and the rest from TRANS.

        PTRANS carries the X-Pro C-N length, the C(i-1)-N-CD angle and the
        C(i-1)-N-CA-CD plane, so proline pairs are excluded from the TRANS pass rather
        than restrained by both.
        """
        if ptrans is None:
            return [builder.build(residues, trans, cpu)]
        return [
            builder.build(residues, trans, cpu, exclude_next_resname="PRO"),
            builder.build(residues, ptrans, cpu, next_resname_filter="PRO"),
        ]

    def joined(groups):
        """Builder groups as one ``(indices array, {property: array})``, or None."""
        parts = [split(g) for g in groups if g]
        if not parts:
            return None
        shared = set.intersection(*(set(p[1]) for p in parts))
        return (
            np.concatenate([p[0] for p in parts], axis=0),
            {prop: np.concatenate([p[1][prop] for p in parts]) for prop in shared},
        )

    for edge_type, builder in (
        ("bond", InterResidueBondBuilder),
        ("angle", InterResidueAngleBuilder),
    ):
        group = joined(per_link(builder(verbose=verbose)))
        if group is not None:
            indices[edge_type]["peptide"], values[edge_type]["peptide"] = group

    # One TRANS pass: the PTRANS torsions are the same, and the Ramachandran pairing
    # needs each residue's phi and psi from a single pass.
    tors = InterResidueTorsionBuilder(verbose=verbose).build(residues, trans, cpu)
    if tors:
        for origin in ("phi", "psi", "omega"):
            if origin in tors:
                indices["torsion"][origin], values["torsion"][origin] = split(
                    tors[origin]
                )
        if "ramachandran" in tors:
            extras["ramachandran"] = tors["ramachandran"]

    planes = [g for g in per_link(InterResiduePlaneBuilder(verbose=verbose)) if g]
    for key in sorted({key for group in planes for key in group}):
        indices["plane"][key], values["plane"][key] = joined(
            [group.get(key) for group in planes]
        )
    return indices, values, extras


def _origins(
    intra: np.ndarray,
    inter: Dict[str, np.ndarray],
    disulfide: Optional[np.ndarray],
) -> Dict[str, np.ndarray]:
    """Collect one edge type's per-origin arrays, dropping the empty ones."""
    per_origin: Dict[str, np.ndarray] = {}
    if intra is not None and len(intra):
        per_origin["intra"] = intra
    for origin, rows in inter.items():
        if rows is not None and len(rows):
            per_origin[origin] = rows
    if disulfide is not None and len(disulfide):
        per_origin["disulfide"] = disulfide
    return per_origin


def _disulfide_edges(
    topology: Topology,
    cols: Dict[str, np.ndarray],
    residue_of_row: Dict[int, int],
    pairs: Sequence[Tuple[int, int]],
    link_dict: Optional[Dict],
    verbose: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
    """Bond, angle and torsion edges for the detected disulfide links, with values.

    Drives the ``InterResidue*Builder`` disulfide paths from the residue graph's
    ``disulf`` edges, so the link geometry comes from the ``disulf`` dictionary entry
    rather than being restated here.

    Returns
    -------
    indices : dict
        ``{'bond'|'angle'|'torsion': (E, k) array}``, omitting types with no edges.
    values : dict
        ``{edge type: {property: array}}`` for the same edges.
    """
    out: Dict[str, np.ndarray] = {}
    if not pairs or not link_dict or "disulf" not in link_dict:
        return out, {}

    disulf = link_dict["disulf"]
    bonds = disulf.get("bonds")
    if bonds is None:
        return out, {}
    sg_sg = bonds[(bonds["atom1"] == "SG") & (bonds["atom2"] == "SG")]
    if len(sg_sg) == 0:
        return out, {}
    length = float(sg_sg["value"].values[0])
    sigma = float(sg_sg["sigma"].values[0])

    cpu = torch.device("cpu")
    bond_builder = InterResidueBondBuilder(verbose=verbose)
    angle_builder = InterResidueAngleBuilder(verbose=verbose)
    torsion_builder = InterResidueTorsionBuilder(verbose=verbose)

    for row_a, row_b in pairs:
        bond_builder.process_disulfide_bond(int(row_a), int(row_b), length, sigma)
        res_a, res_b = residue_of_row[row_a], residue_of_row[row_b]
        if disulf.get("angles") is not None:
            angle_builder.process_disulfide_angles(
                topology, res_a, res_b, disulf["angles"]
            )
        if disulf.get("torsions") is not None:
            torsion_builder.process_disulfide_torsions(
                topology, res_a, res_b, disulf["torsions"]
            )

    values: Dict[str, Dict[str, np.ndarray]] = {}
    for edge_type, group in (
        ("bond", bond_builder.finalize(cpu)),
        ("angle", angle_builder.finalize(cpu)),
        ("torsion", torsion_builder.finalize_disulfide(cpu)),
    ):
        if not group:
            continue
        out[edge_type] = group["indices"].cpu().numpy()
        values[edge_type] = {
            prop: tensor.cpu().numpy()
            for prop, tensor in group.items()
            if prop != "indices" and tensor is not None
        }
    return out, values


def _lookup_link_atom(
    topology: Topology,
    residue_by_key: Dict[Tuple[str, int, str], List[int]],
    chainid: str,
    resseq: int,
    icode: str,
    resname: str,
    name: str,
    altloc: str,
):
    """Resolve one ``LINK`` record's atom to a row of the topology, or None.

    Matches on ``(chainid, resseq, icode, name)`` with ``resname`` as a tie-breaker.
    Where a residue has alternative conformations the requested altloc wins, then the
    blank one, then ``'A'``, then whatever is left -- a LINK naming a specific conformer
    should reach that conformer, but one naming none should still resolve.
    """
    key = (str(chainid), int(resseq), str(icode).strip())
    candidates = residue_by_key.get(key, [])
    wanted = str(resname).strip() if resname else ""
    rows = [
        row
        for r in candidates
        for row in topology.residues.atom_rows(r)
        if str(topology.atoms.name[row]).strip() == str(name).strip()
    ]
    if wanted:
        tied = [row for row in rows if topology.resname_of_atom(row).strip() == wanted]
        rows = tied or rows
    if not rows:
        return None
    altlocs = [str(topology.atoms.altloc[row]) for row in rows]
    requested = str(altloc).strip() if altloc else ""
    for candidate in ((requested, " ") if requested else ()) + (" ", "A"):
        for row, alt in zip(rows, altlocs):
            if alt == candidate:
                return int(row)
    return int(rows[0])

def _link_record_edges(
    topology: Topology,
    links,
    disulfide_bonds: Optional[np.ndarray],
    verbose: int,
) -> Tuple[np.ndarray, List[Tuple[int, int]], Dict[str, np.ndarray]]:
    """Bond edges for the accepted ``LINK`` records, and the atom pairs they join.

    A record duplicating an auto-detected disulfide is dropped, since that link already
    contributed its bond, angles and torsions; so is a record repeating an earlier one,
    which would otherwise add a second bond edge and a second restraint on the same pair.

    Returns
    -------
    edges : numpy.ndarray
        Shape ``(L, 2)``; empty when nothing resolved.
    atom_pairs : list of tuple of int
        The same pairs, for lifting to residue-level link edges.
    values : dict
        ``references`` from each record's ``length`` (1.5 A where blank or unusable) and
        a fixed ``sigmas`` of 0.02 A.
    """
    if links is None or len(links) == 0:
        return np.zeros((0, 2), dtype=np.int64), [], {}

    existing = set()
    if disulfide_bonds is not None:
        for a, b in disulfide_bonds:
            existing.add((min(int(a), int(b)), max(int(a), int(b))))

    residue_by_key: Dict[Tuple[str, int, str], List[int]] = {}
    for r in range(topology.n_residues):
        chain, resseq, icode = topology.residues.key(r)
        residue_by_key.setdefault((chain, resseq, icode.strip()), []).append(r)

    rows: List[Tuple[int, int]] = []
    lengths: List[float] = []
    n_unresolved = 0
    for _, link in links.iterrows():
        idx1 = _lookup_link_atom(
            topology,
            residue_by_key,
            chainid=link["chainid1"],
            resseq=int(link["resseq1"]),
            icode=link["icode1"],
            resname=link["resname1"],
            name=link["name1"],
            altloc=link["altloc1"],
        )
        idx2 = _lookup_link_atom(
            topology,
            residue_by_key,
            chainid=link["chainid2"],
            resseq=int(link["resseq2"]),
            icode=link["icode2"],
            resname=link["resname2"],
            name=link["name2"],
            altloc=link["altloc2"],
        )
        if idx1 is None or idx2 is None or idx1 == idx2:
            n_unresolved += 1
            continue
        pair = (min(idx1, idx2), max(idx1, idx2))
        if pair in existing:
            continue
        existing.add(pair)
        rows.append((idx1, idx2))
        length = link["length"]
        usable = isinstance(length, (int, float)) and length == length and length > 0
        lengths.append(float(length) if usable else 1.5)

    if verbose > 1 and n_unresolved:
        print(f"{n_unresolved} LINK records did not resolve to a pair of atoms")
    if not rows:
        return np.zeros((0, 2), dtype=np.int64), [], {}
    values = {
        "references": np.asarray(lengths, dtype=np.float64),
        "sigmas": np.full(len(rows), 0.02, dtype=np.float64),
    }
    return np.asarray(rows, dtype=np.int64), rows, values


def _first_occurrences(
    rows: np.ndarray, properties: Dict[str, np.ndarray]
) -> np.ndarray:
    """Positions of the rows that do not repeat an earlier row's atoms and values.

    Parameters
    ----------
    rows : numpy.ndarray
        Edge atom indices, shape ``(E, k)``.
    properties : dict
        ``{property: array}``, each indexed by row on axis 0.

    Returns
    -------
    numpy.ndarray
        Ascending positions into ``rows``, the first of each repeated set.
    """
    if len(rows) < 2:
        return np.arange(len(rows))
    # Atom rows and values side by side in float64, which holds both exactly.
    key = np.column_stack(
        [np.asarray(rows, dtype=np.float64).reshape(len(rows), -1)]
        + [
            np.asarray(values, dtype=np.float64).reshape(len(rows), -1)
            for _, values in sorted(properties.items())
            if values is not None
        ]
    )
    _, first = np.unique(key, axis=0, return_index=True)
    return np.sort(first)


def _block_with_values(
    per_origin: Dict[str, np.ndarray],
    payload: Dict[str, Dict[str, np.ndarray]],
    arity: int,
    edge_type: str,
    device,
) -> Tuple[EdgeBlock, Dict[str, Dict[str, torch.Tensor]]]:
    """One canonical edge block plus its per-origin value tensors.

    The block and the values come out of a single :func:`assemble_origins` call, so the
    same permutation is applied to both -- which is the only thing keeping a sigma
    attached to the edge it belongs to.

    Within an origin, a row repeating an earlier row's atoms and values is dropped: a
    restraint over atoms that every altloc conformer shares is matched once per
    conformer, and the copies would weight it once per conformer. Rows over the same
    atoms with different values are kept.
    """
    unique_rows: Dict[str, np.ndarray] = {}
    unique_payload: Dict[str, Dict[str, np.ndarray]] = {}
    for origin, rows in per_origin.items():
        properties = payload.get(origin) or {}
        keep = _first_occurrences(np.asarray(rows), properties)
        unique_rows[origin] = np.asarray(rows)[keep]
        unique_payload[origin] = {
            prop: None if values is None else np.asarray(values)[keep]
            for prop, values in properties.items()
        }
    indices, bounds, sorted_payload = assemble_origins(
        unique_rows, arity, edge_type, unique_payload
    )
    block = EdgeBlock(
        indices=torch.as_tensor(indices, dtype=get_int_dtype(), device=device),
        origin_bounds=bounds,
    )
    values = {
        origin: {
            prop: to_tensor(array, prop, device=device)
            for prop, array in properties.items()
        }
        for origin, properties in sorted_payload.items()
    }
    return block, values


def build_topology(
    topology: Topology,
    cif_dict: Dict,
    xyz: torch.Tensor,
    link_dict: Optional[Dict] = None,
    link_list=None,
    links=None,
    device=None,
    verbose: int = 0,
) -> Topology:
    """Build a topology, discarding the restraint values built along the way.

    See :func:`build_topology_with_values` for the parameters; this is the connectivity
    half on its own, for callers that need the graph and no ideal geometry.
    """
    topology, _, _ = build_topology_with_values(
        topology,
        cif_dict,
        xyz,
        link_dict=link_dict,
        link_list=link_list,
        links=links,
        device=device,
        verbose=verbose,
    )
    return topology


def build_topology_with_values(
    topology: Topology,
    cif_dict: Dict,
    xyz: torch.Tensor,
    link_dict: Optional[Dict] = None,
    link_list=None,
    links=None,
    device=None,
    verbose: int = 0,
) -> Tuple[Topology, Dict[str, Dict], Dict[str, Dict]]:
    """Connect a node-only topology against the restraint dictionaries.

    Parameters
    ----------
    topology : Topology
        Identity to connect, e.g. from :meth:`Topology.from_table`. Not modified; its
        edges, if any, are ignored.
    cif_dict : dict
        Restraint dictionary keyed by residue name.
    xyz : torch.Tensor
        Cartesian coordinates in Å, shape ``(N, 3)``. Disulfides are detected by SG-SG
        distance and proline omega classified cis or trans from them.
    link_dict : dict, optional
        Link-type definitions. Without it no inter-residue edges are built.
    link_list : pandas.DataFrame, optional
        Link table used to resolve which modifications a peptide link applies.
    links : pandas.DataFrame, optional
        Parsed PDB ``LINK`` records. Each record that resolves to two distinct atoms and
        does not duplicate an auto-detected disulfide contributes one bond edge.
    device : torch.device, optional
        Where to place the edge blocks.
    verbose : int, default 0
        Verbosity level.

    Returns
    -------
    topology : Topology
        A new, connected topology over the same atoms.
    values : dict
        ``{edge_type: {origin: {property: tensor}}}`` for bonds, angles and torsions;
        ``{'chiral': {property: tensor}}`` and ``{'plane': {size: {property: tensor}}}``
        for the two types that carry no origin. Row-aligned to the edge blocks.
    extras : dict
        Products of the same pass that are not edges -- currently ``ramachandran``.
    """
    cols = _atom_columns(topology)
    residue_nodes = topology.residues
    nodes = {
        field: getattr(residue_nodes, field)
        for field in ("chain", "resseq", "icode", "resname", "atom_start", "atom_end")
    }
    n_res = len(nodes["chain"])

    names_by_residue = [
        set(cols["name"][int(nodes["atom_start"][r]) : int(nodes["atom_end"][r])])
        for r in range(n_res)
    ]
    is_polymer = np.array(
        [cols["record"][int(nodes["atom_start"][r])] == "ATOM" for r in range(n_res)],
        dtype=bool,
    )

    polymer_nodes = {k: v[is_polymer] for k, v in nodes.items()}
    polymer_map = np.nonzero(is_polymer)[0]
    polymer_names = [names_by_residue[r] for r in polymer_map]
    peptide_local = find_peptide_links(polymer_nodes, polymer_names)
    peptide_pairs = [
        (int(polymer_map[a]), int(polymer_map[b])) for a, b in peptide_local
    ]

    match_cols, chemical_nodes, chemical_pairs, source_rows, owners = _chemical_nodes(
        cols, nodes, peptide_pairs
    )
    comp_dict, chemical_keys = resolve_template_keys(
        chemical_nodes["resname"], chemical_pairs, cif_dict, link_list, verbose=verbose
    )
    template_key = np.asarray(nodes["resname"], dtype=object).copy()
    _, first_variant = np.unique(owners, return_index=True)
    template_key[:] = chemical_keys[first_variant]
    pp_cif = PreprocessedCIF(comp_dict)
    match_cols["name"] = match_cols["name"].copy()
    # PDB terminal H1 is the monomer dictionary's H. Resolve the alias only
    # for matching, preserving the model's atom names and row identities.
    for r in range(len(chemical_keys)):
        start, end = int(chemical_nodes["atom_start"][r]), int(
            chemical_nodes["atom_end"][r]
        )
        names = match_cols["name"][start:end]
        if "H1" not in names or "H" in names:
            continue
        component = comp_dict.get(str(chemical_keys[r]), {})
        atom_table = component.get("atoms")
        if atom_table is None:
            continue
        template_names = set(atom_table["atom_id"].astype(str).str.strip())
        if "H" in template_names and "H1" not in template_names:
            names[names == "H1"] = "H"
    chemical_energy, chemical_h_count = _atom_types(
        match_cols, chemical_nodes, chemical_keys, comp_dict
    )
    energy_type = np.full(topology.n_atoms, "", dtype=chemical_energy.dtype)
    template_h_count = np.full(topology.n_atoms, -1, dtype=np.int8)
    own_identity = match_cols["resname"] == cols["resname"][source_rows]
    energy_type[source_rows[own_identity]] = chemical_energy[own_identity]
    template_h_count[source_rows[own_identity]] = chemical_h_count[own_identity]

    intra, intra_values = _match_intra(
        match_cols, chemical_nodes, chemical_keys, pp_cif
    )
    intra_planes, intra_plane_values = _match_intra_planes(
        match_cols, chemical_nodes, chemical_keys, pp_cif
    )
    inter, inter_values, extras = _inter_residue_edges(
        PeptideResidues(topology, peptide_pairs, xyz.detach().cpu().numpy()),
        link_dict,
        verbose,
    )

    residue_of_row = {}
    for r in range(n_res):
        for row in range(int(nodes["atom_start"][r]), int(nodes["atom_end"][r])):
            residue_of_row[row] = r

    sg_rows = [
        row
        for row in range(len(cols["name"]))
        if cols["name"][row] == "SG" and cols["record"][row] == "ATOM"
    ]
    disulfide_pairs = find_disulfide_links(sg_rows, residue_of_row, xyz)
    disulfide, disulfide_values = _disulfide_edges(
        topology, cols, residue_of_row, disulfide_pairs, link_dict, verbose
    )

    link_edges, link_atom_pairs, link_values = _link_record_edges(
        topology, links, disulfide.get("bond"), verbose
    )

    # LINK edges carry ``index`` values, so lift them through that column.
    index_to_residue = {int(cols["index"][row]): r for row, r in residue_of_row.items()}

    link_pairs = [(a, b, "TRANS") for a, b in peptide_pairs]
    disulf_residue_pairs = sorted(
        {
            (
                min(residue_of_row[a], residue_of_row[b]),
                max(residue_of_row[a], residue_of_row[b]),
            )
            for a, b in disulfide_pairs
        }
    )
    link_pairs += [(a, b, "disulf") for a, b in disulf_residue_pairs]
    for a, b in link_atom_pairs:
        ra, rb = index_to_residue.get(a), index_to_residue.get(b)
        if ra is not None and rb is not None and ra != rb:
            link_pairs.append((ra, rb, "LINK"))

    residues = ResidueGraph(
        chain=nodes["chain"],
        resseq=nodes["resseq"],
        icode=nodes["icode"],
        resname=nodes["resname"],
        template_key=template_key,
        atom_start=nodes["atom_start"],
        atom_end=nodes["atom_end"],
        link_pairs=(
            np.array([(a, b) for a, b, _ in link_pairs], dtype=np.int64)
            if link_pairs
            else np.zeros((0, 2), dtype=np.int64)
        ),
        link_kind=np.array([k for _, _, k in link_pairs], dtype="<U8"),
    )

    plane_blocks: Dict[int, EdgeBlock] = {}
    plane_values: Dict[int, Dict[str, torch.Tensor]] = {}
    plane_sizes = set(intra_planes)
    for key in inter.get("plane", {}):
        plane_sizes.add(int(str(key).split("_")[0]))
    for size in sorted(plane_sizes):
        per_origin: Dict[str, np.ndarray] = {}
        payload: Dict[str, Dict[str, np.ndarray]] = {}
        if size in intra_planes:
            per_origin["intra"] = intra_planes[size]
            payload["intra"] = intra_plane_values.get(size, {})
        peptide = inter.get("plane", {}).get(f"{size}_atoms")
        if peptide is not None and len(peptide):
            per_origin["peptide"] = peptide
            payload["peptide"] = inter_values["plane"].get(f"{size}_atoms", {})
        if not per_origin:
            continue
        block, per_origin_values = _block_with_values(
            per_origin, payload, size, "plane", device
        )
        plane_blocks[size] = block
        # Planes carry no origin downstream, so the origins are concatenated back into
        # one group -- in block order, which is what the block's own layout already is.
        plane_values[size] = {
            prop: torch.cat(
                [
                    per_origin_values[o][prop]
                    for o in block.origins()
                    if prop in per_origin_values.get(o, {})
                ]
            )
            for prop in {p for v in per_origin_values.values() for p in v}
        }

    bond_block, bond_values = _block_with_values(
        _origins(
            intra["bonds"],
            {**inter["bond"], "link": link_edges},
            disulfide.get("bond"),
        ),
        {
            "intra": intra_values["bonds"],
            **inter_values["bond"],
            "link": link_values,
            "disulfide": disulfide_values.get("bond", {}),
        },
        2,
        "bond",
        device,
    )
    angle_block, angle_values = _block_with_values(
        _origins(intra["angles"], inter["angle"], disulfide.get("angle")),
        {
            "intra": intra_values["angles"],
            **inter_values["angle"],
            "disulfide": disulfide_values.get("angle", {}),
        },
        3,
        "angle",
        device,
    )
    torsion_block, torsion_values = _block_with_values(
        _origins(intra["torsions"], inter["torsion"], disulfide.get("torsion")),
        {
            "intra": intra_values["torsions"],
            **inter_values["torsion"],
            "disulfide": disulfide_values.get("torsion", {}),
        },
        4,
        "torsion",
        device,
    )
    chiral_block, chiral_values = _block_with_values(
        _origins(intra["chirals"], {}, None),
        {"intra": intra_values["chirals"]},
        4,
        "chiral",
        device,
    )

    atoms = AtomGraph(
        resname=cols["resname"].copy(),
        name=cols["name"],
        element=cols["element"],
        altloc=cols["altloc"],
        is_hetatm=topology.atoms.is_hetatm.copy(),
        charge=topology.atoms.charge.copy(),
        residue_of=torch.as_tensor(
            np.repeat(
                np.arange(n_res, dtype=np.int64),
                nodes["atom_end"] - nodes["atom_start"],
            ),
            dtype=get_int_dtype(),
            device=device,
        ),
        bonds=bond_block,
        angles=angle_block,
        torsions=torsion_block,
        chirals=chiral_block,
        planes=plane_blocks,
        energy_type=energy_type,
        template_h_count=torch.as_tensor(
            template_h_count,
            dtype=torch.int8,  # dtype-ok: small per-atom count; AtomGraph storage
            device=device,
        ),
    )

    values: Dict[str, Dict] = {
        "bond": bond_values,
        "angle": angle_values,
        "torsion": torsion_values,
        # Chirals carry no origin downstream, so the single origin is unwrapped.
        "chiral": chiral_values.get("intra", {}),
        "plane": plane_values,
    }
    return Topology(residues=residues, atoms=atoms, connected=True), values, extras


__all__ = ["build_topology", "build_topology_with_values"]
