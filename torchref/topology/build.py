"""Assemble a :class:`~torchref.topology.topology.Topology` from an atom table.

Intra-residue edges are matched here, template by template, through the Numba matchers
in :mod:`torchref.restraints.builders_numba`. Inter-residue edges come from the
``InterResidue*Builder`` classes, which already encode the link geometry and are reused
rather than reimplemented.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from torchref.restraints.builders_fast import (
    InterResidueAngleBuilder,
    InterResidueBondBuilder,
    InterResiduePlaneBuilder,
    InterResidueTorsionBuilder,
    PreprocessedCIF,
)
from torchref.restraints.builders_numba import (
    match_angles_numba,
    match_bonds_numba,
    match_chirals_numba,
    match_torsions_numba,
)
from torchref.topology.atom_graph import AtomGraph
from torchref.topology.edges import EdgeBlock, assemble_origins
from torchref.topology.residue_graph import (
    ResidueGraph,
    build_residue_nodes,
    find_disulfide_links,
    find_peptide_links,
)
from torchref.topology.restraint_sets import to_tensor
from torchref.topology.templates import resolve_template_keys
from torchref.topology.topology import Topology

#: Initial size of the matcher work arrays, grown on demand.
_WORK = 64


def _atom_columns(pdb: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Per-atom identity arrays, with altlocs normalised so blank reads as ``' '``."""
    altloc = pdb["altloc"].values.astype(str) if "altloc" in pdb.columns else None
    if altloc is None:
        altloc = np.full(len(pdb), " ", dtype="<U1")
    else:
        altloc = np.where(altloc == "", " ", altloc)
    icode = (
        pdb["icode"].values.astype(str)
        if "icode" in pdb.columns
        else np.full(len(pdb), "", dtype="<U1")
    )
    return {
        "name": pdb["name"].values.astype(str),
        "element": (
            pdb["element"].values.astype(str)
            if "element" in pdb.columns
            else np.full(len(pdb), "", dtype="<U2")
        ),
        "altloc": altloc,
        "chain": pdb["chainid"].values.astype(str),
        "resseq": pdb["resseq"].values.astype(np.int64),
        "icode": icode,
        "resname": pdb["resname"].values.astype(str),
        "record": (
            pdb["ATOM"].values.astype(str)
            if "ATOM" in pdb.columns
            else np.full(len(pdb), "ATOM", dtype="<U6")
        ),
        "index": pdb["index"].values.astype(np.int64),
    }


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
                n = match_bonds_numba(
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
                n = match_angles_numba(
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
                n = match_torsions_numba(
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
                n = match_chirals_numba(
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
    pdb: pd.DataFrame,
    link_dict: Optional[Dict],
    verbose: int,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict], Dict[str, Dict]]:
    """Peptide edges, their values, and the Ramachandran pairing, from the builders.

    Reuses ``InterResidue*Builder`` rather than reimplementing the link geometry.

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

    bond = InterResidueBondBuilder(verbose=verbose).build(
        pdb, trans, cpu, filter_atom_type="ATOM"
    )
    if bond:
        indices["bond"]["peptide"], values["bond"]["peptide"] = split(bond)

    ab = InterResidueAngleBuilder(verbose=verbose)
    if ptrans is not None:
        # PTRANS carries the extra C(i-1)-N-CD angle, so proline pairs are built from it
        # and excluded from the TRANS pass to avoid two restraints on the same atoms.
        groups = [
            ab.build(
                pdb, trans, cpu, filter_atom_type="ATOM", exclude_next_resname="PRO"
            ),
            ab.build(
                pdb, ptrans, cpu, filter_atom_type="ATOM", next_resname_filter="PRO"
            ),
        ]
    else:
        groups = [ab.build(pdb, trans, cpu, filter_atom_type="ATOM")]
    parts = [split(g) for g in groups if g]
    if parts:
        indices["angle"]["peptide"] = np.concatenate([p[0] for p in parts], axis=0)
        shared = set.intersection(*(set(p[1]) for p in parts))
        values["angle"]["peptide"] = {
            prop: np.concatenate([p[1][prop] for p in parts]) for prop in shared
        }

    tors = InterResidueTorsionBuilder(verbose=verbose).build(
        pdb, trans, cpu, filter_atom_type="ATOM"
    )
    if tors:
        for origin in ("phi", "psi", "omega"):
            if origin in tors:
                indices["torsion"][origin], values["torsion"][origin] = split(
                    tors[origin]
                )
        if "ramachandran" in tors:
            extras["ramachandran"] = tors["ramachandran"]

    planes = InterResiduePlaneBuilder(verbose=verbose).build(
        pdb, trans, cpu, filter_atom_type="ATOM"
    )
    if planes:
        for key, group in planes.items():
            indices["plane"][key], values["plane"][key] = split(group)
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
    pdb: pd.DataFrame,
    nodes: Dict[str, np.ndarray],
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
        # The edge indices are the atom table's ``index`` column, not its row number.
        bond_builder.process_disulfide_bond(
            int(cols["index"][row_a]), int(cols["index"][row_b]), length, sigma
        )
        res_a, res_b = residue_of_row[row_a], residue_of_row[row_b]
        atoms_a = pdb.iloc[
            int(nodes["atom_start"][res_a]) : int(nodes["atom_end"][res_a])
        ]
        atoms_b = pdb.iloc[
            int(nodes["atom_start"][res_b]) : int(nodes["atom_end"][res_b])
        ]
        if disulf.get("angles") is not None:
            angle_builder.process_disulfide_angles(atoms_a, atoms_b, disulf["angles"])
        if disulf.get("torsions") is not None:
            torsion_builder.process_disulfide_torsions(
                atoms_a, atoms_b, disulf["torsions"]
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
    pdb: pd.DataFrame,
    chainid: str,
    resseq: int,
    icode: str,
    resname: str,
    name: str,
    altloc: str,
):
    """Resolve one ``LINK`` record's atom to a row of the atom table, or None.

    Matches on ``(chainid, resseq, icode, name)`` with ``resname`` as a tie-breaker.
    Where a residue has alternative conformations the requested altloc wins, then the
    blank one, then ``'A'``, then whatever is left -- a LINK naming a specific conformer
    should reach that conformer, but one naming none should still resolve.
    """
    sel = pdb[
        (pdb["chainid"].astype(str) == str(chainid))
        & (pdb["resseq"].astype(int) == int(resseq))
        & (pdb["icode"].astype(str) == str(icode))
        & (pdb["name"].astype(str).str.strip() == str(name).strip())
    ]
    if len(sel) == 0:
        return None
    if resname:
        tied = sel[sel["resname"].astype(str).str.strip() == str(resname).strip()]
        if len(tied) > 0:
            sel = tied

    if altloc:
        for candidate in (altloc, ""):
            hit = sel[sel["altloc"].astype(str) == candidate]
            if len(hit) > 0:
                return int(hit.iloc[0]["index"])
    for candidate in ("", "A"):
        hit = sel[sel["altloc"].astype(str) == candidate]
        if len(hit) > 0:
            return int(hit.iloc[0]["index"])
    return int(sel.iloc[0]["index"])


def _link_record_edges(
    pdb: pd.DataFrame,
    links,
    disulfide_bonds: Optional[np.ndarray],
    verbose: int,
) -> Tuple[np.ndarray, List[Tuple[int, int]], Dict[str, np.ndarray]]:
    """Bond edges for the accepted ``LINK`` records, and the atom pairs they join.

    A record duplicating an auto-detected disulfide is dropped, since that link already
    contributed its bond, angles and torsions.

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

    rows: List[Tuple[int, int]] = []
    lengths: List[float] = []
    n_unresolved = 0
    for _, link in links.iterrows():
        idx1 = _lookup_link_atom(
            pdb,
            chainid=link["chainid1"],
            resseq=int(link["resseq1"]),
            icode=link["icode1"],
            resname=link["resname1"],
            name=link["name1"],
            altloc=link["altloc1"],
        )
        idx2 = _lookup_link_atom(
            pdb,
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
    """
    indices, bounds, sorted_payload = assemble_origins(
        per_origin, arity, edge_type, payload
    )
    block = EdgeBlock(
        indices=torch.as_tensor(indices, dtype=torch.int64, device=device),
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
    pdb: pd.DataFrame,
    cif_dict: Dict,
    link_dict: Optional[Dict] = None,
    link_list=None,
    links=None,
    xyz: Optional[torch.Tensor] = None,
    device=None,
    verbose: int = 0,
) -> Topology:
    """Build a topology, discarding the restraint values built along the way.

    See :func:`build_topology_with_values` for the parameters; this is the connectivity
    half on its own, for callers that need the graph and no ideal geometry.
    """
    topology, _, _ = build_topology_with_values(
        pdb,
        cif_dict,
        link_dict=link_dict,
        link_list=link_list,
        links=links,
        xyz=xyz,
        device=device,
        verbose=verbose,
    )
    return topology


def build_topology_with_values(
    pdb: pd.DataFrame,
    cif_dict: Dict,
    link_dict: Optional[Dict] = None,
    link_list=None,
    links=None,
    xyz: Optional[torch.Tensor] = None,
    device=None,
    verbose: int = 0,
) -> Tuple[Topology, Dict[str, Dict], Dict[str, Dict]]:
    """Build a topology from an atom table and the restraint dictionaries.

    Parameters
    ----------
    pdb : pandas.DataFrame
        Atom table, with ``name``, ``element``, ``altloc``, ``chainid``, ``resseq``,
        ``icode``, ``resname``, ``ATOM`` and ``index`` columns.
    cif_dict : dict
        Restraint dictionary keyed by residue name.
    link_dict : dict, optional
        Link-type definitions. Without it no inter-residue edges are built.
    link_list : pandas.DataFrame, optional
        Link table used to resolve which modifications a peptide link applies.
    links : pandas.DataFrame, optional
        Parsed PDB ``LINK`` records. Each record that resolves to two distinct atoms and
        does not duplicate an auto-detected disulfide contributes one bond edge.
    xyz : torch.Tensor, optional
        Coordinates, shape ``(N, 3)``. Needed only to detect disulfide links, which are
        found by SG-SG distance.
    device : torch.device, optional
        Where to place the edge blocks.
    verbose : int, default 0
        Verbosity level.

    Returns
    -------
    topology : Topology
        The connectivity.
    values : dict
        ``{edge_type: {origin: {property: tensor}}}`` for bonds, angles and torsions;
        ``{'chiral': {property: tensor}}`` and ``{'plane': {size: {property: tensor}}}``
        for the two types that carry no origin. Row-aligned to the edge blocks.
    extras : dict
        Products of the same pass that are not edges -- currently ``ramachandran``.
    """
    cols = _atom_columns(pdb)
    nodes = build_residue_nodes(
        cols["chain"], cols["resseq"], cols["icode"], cols["resname"]
    )
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

    comp_dict, template_key = resolve_template_keys(
        nodes["resname"], peptide_pairs, cif_dict, link_list, verbose=verbose
    )
    pp_cif = PreprocessedCIF(comp_dict)

    intra, intra_values = _match_intra(cols, nodes, template_key, pp_cif)
    intra_planes, intra_plane_values = _match_intra_planes(
        cols, nodes, template_key, pp_cif
    )
    inter, inter_values, extras = _inter_residue_edges(pdb, link_dict, verbose)

    residue_of_row = {}
    for r in range(n_res):
        for row in range(int(nodes["atom_start"][r]), int(nodes["atom_end"][r])):
            residue_of_row[row] = r

    disulfide_pairs: List[Tuple[int, int]] = []
    disulfide: Dict[str, np.ndarray] = {}
    disulfide_values: Dict[str, Dict[str, np.ndarray]] = {}
    if xyz is not None:
        sg_rows = [
            row
            for row in range(len(cols["name"]))
            if cols["name"][row] == "SG" and cols["record"][row] == "ATOM"
        ]
        disulfide_pairs = find_disulfide_links(sg_rows, residue_of_row, xyz)
        disulfide, disulfide_values = _disulfide_edges(
            pdb, nodes, cols, residue_of_row, disulfide_pairs, link_dict, verbose
        )

    link_edges, link_atom_pairs, link_values = _link_record_edges(
        pdb, links, disulfide.get("bond"), verbose
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
        name=cols["name"],
        element=cols["element"],
        altloc=cols["altloc"],
        residue_of=torch.as_tensor(
            np.repeat(
                np.arange(n_res, dtype=np.int64),
                nodes["atom_end"] - nodes["atom_start"],
            ),
            dtype=torch.int64,
            device=device,
        ),
        bonds=bond_block,
        angles=angle_block,
        torsions=torsion_block,
        chirals=chiral_block,
        planes=plane_blocks,
    )

    values: Dict[str, Dict] = {
        "bond": bond_values,
        "angle": angle_values,
        "torsion": torsion_values,
        # Chirals carry no origin downstream, so the single origin is unwrapped.
        "chiral": chiral_values.get("intra", {}),
        "plane": plane_values,
    }
    return Topology(residues=residues, atoms=atoms), values, extras


__all__ = ["build_topology", "build_topology_with_values"]
