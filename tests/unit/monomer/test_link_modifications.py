"""Contracts for CCP4 ``chem_mod`` records applied at peptide links.

The monomer library defines amino acids free, so a residue's restraints only
describe the in-chain form once the link's modifications have been applied. What
is pinned here is that they are, that chain termini keep the free form, and that
the resulting targets close: the restrained angles around a peptide carbonyl
carbon or amide nitrogen sum to 360 degrees.
"""

import numpy as np
import pandas as pd
import pytest

from torchref.topology.monomer.modifications import (
    apply_modifications,
    link_modifications,
    read_mod_definitions,
)
from torchref.topology.monomer.cif import read_link_definitions

TOL = 1e-3


def _component():
    """A miniature component in the library's free-amino-acid form."""
    return {
        "bonds": pd.DataFrame(
            {
                "atom1": ["N", "CA", "C", "C"],
                "atom2": ["CA", "C", "O", "OXT"],
                "value": [1.483, 1.531, 1.251, 1.251],
                "sigma": [0.010, 0.010, 0.0183, 0.0183],
            }
        ),
        "angles": pd.DataFrame(
            {
                "atom1": ["CA", "CA", "O"],
                "atom2": ["C", "C", "C"],
                "atom3": ["O", "OXT", "OXT"],
                "value": [117.191, 117.191, 125.618],
                "sigma": [2.33, 2.33, 1.50],
            }
        ),
        "planes": pd.DataFrame(
            {
                "atom": ["C", "CA", "O", "OXT"],
                "plane_id": ["plan-1"] * 4,
                "sigma": [0.02] * 4,
            }
        ),
    }


def _modification(**sections):
    return {name: pd.DataFrame(rows) for name, rows in sections.items()}


# =============================================================================
# The modification engine
# =============================================================================


@pytest.mark.unit
def test_change_overwrites_value_and_sigma():
    mods = {
        "M": _modification(
            angles={
                "function": ["change"],
                "atom1": ["CA"],
                "atom2": ["C"],
                "atom3": ["O"],
                "value": [120.614],
                "sigma": [1.50],
            }
        )
    }
    result = apply_modifications(_component(), ["M"], mods)

    row = result["angles"].iloc[0]
    assert row["value"] == pytest.approx(120.614)
    assert row["sigma"] == pytest.approx(1.50)


@pytest.mark.unit
def test_atom_order_within_a_restraint_is_irrelevant():
    """``O C CA`` in a modification must find ``CA C O`` in the component."""
    mods = {
        "M": _modification(
            angles={
                "function": ["change"],
                "atom1": ["O"],
                "atom2": ["C"],
                "atom3": ["CA"],
                "value": [120.614],
                "sigma": [1.50],
            }
        )
    }
    result = apply_modifications(_component(), ["M"], mods)

    assert result["angles"].iloc[0]["value"] == pytest.approx(120.614)


@pytest.mark.unit
def test_delete_removes_rows_and_leaves_the_rest():
    mods = {
        "M": _modification(
            bonds={
                "function": ["delete"],
                "atom1": ["C"],
                "atom2": ["OXT"],
                "value": [np.nan],
                "sigma": [np.nan],
            }
        )
    }
    result = apply_modifications(_component(), ["M"], mods)

    pairs = set(zip(result["bonds"]["atom1"], result["bonds"]["atom2"]))
    assert ("C", "OXT") not in pairs
    assert len(result["bonds"]) == 3


@pytest.mark.unit
def test_add_appends_a_new_restraint():
    mods = {
        "M": _modification(
            bonds={
                "function": ["add"],
                "atom1": ["CA"],
                "atom2": ["CB"],
                "value": [1.520],
                "sigma": [0.015],
            }
        )
    }
    result = apply_modifications(_component(), ["M"], mods)

    added = result["bonds"].iloc[-1]
    assert (added["atom1"], added["atom2"]) == ("CA", "CB")
    assert added["value"] == pytest.approx(1.520)


@pytest.mark.unit
def test_deleting_every_atom_of_a_plane_deletes_the_plane():
    mods = {
        "M": _modification(
            planes={
                "function": ["delete"] * 4,
                "plane_id": ["plan-1"] * 4,
                "atom": ["C", "CA", "O", "OXT"],
                "sigma": [0.02] * 4,
            }
        )
    }
    result = apply_modifications(_component(), ["M"], mods)

    assert len(result["planes"]) == 0


@pytest.mark.unit
def test_source_component_is_not_modified():
    component = _component()
    mods = {
        "M": _modification(
            angles={
                "function": ["change"],
                "atom1": ["CA"],
                "atom2": ["C"],
                "atom3": ["O"],
                "value": [120.614],
                "sigma": [1.50],
            }
        )
    }
    apply_modifications(component, ["M"], mods)

    assert component["angles"].iloc[0]["value"] == pytest.approx(117.191)


@pytest.mark.unit
def test_no_modifications_is_a_no_op():
    component = _component()
    result = apply_modifications(component, [], {})

    assert result["angles"].iloc[0]["value"] == pytest.approx(117.191)


# =============================================================================
# The library's own records
# =============================================================================


@pytest.mark.unit
def test_peptide_links_name_their_modifications():
    _, link_list = read_link_definitions()
    modifications = link_modifications(link_list)

    assert modifications["TRANS"] == ("DEL-OXT", "DEL-HN1")
    assert modifications["PTRANS"] == ("DEL-OXT", "DEL-HNP")
    assert modifications["gap"] == (None, None)


@pytest.mark.unit
def test_peptide_modifications_carry_the_linked_backbone_targets():
    mod_dict = read_mod_definitions()

    del_oxt = mod_dict["DEL-OXT"]["angles"]
    ca_c_o = del_oxt[(del_oxt["function"] == "change") & (del_oxt["atom3"] == "O")]
    assert ca_c_o["value"].iloc[0] == pytest.approx(120.614)

    del_hn1 = mod_dict["DEL-HN1"]["angles"]
    ca_n_h = del_hn1[(del_hn1["function"] == "change") & (del_hn1["atom3"] == "H")]
    assert ca_n_h["value"].iloc[0] == pytest.approx(118.729)

    del_hnp = mod_dict["DEL-HNP"]["angles"]
    ca_n_cd = del_hnp[(del_hnp["function"] == "change") & (del_hnp["atom3"] == "CD")]
    assert ca_n_cd["value"].iloc[0] == pytest.approx(112.597)


# =============================================================================
# Built restraint tables
# =============================================================================


def _built(pdb_path, strip_H=True):
    """Build a model's restraints and return ``(model, table accessor)``.

    ``add_hydrogens=False``: these tests read the restraint targets of the hydrogens the
    file carries. Generating more would add a chain-terminal ``CA-N-H``, which correctly
    keeps the free-amino-acid target of 109.6 degrees rather than the linked 118.7 and so
    is outside what they assert.
    """
    from torchref import Model

    model = Model(verbose=0, strip_H=strip_H, add_hydrogens=False)
    model.load_pdb(str(pdb_path))
    return model, model.restraints.restraints


def _table(tables, rtype, origin):
    group = tables[rtype][origin]
    return tuple(
        group[field].cpu().numpy()
        for field in ("indices", "references", "sigmas")
    )


def _triples(tables, origin, names):
    indices, references, sigmas = _table(tables, "angle", origin)
    return [
        ((names[a], names[b], names[c]), int(b), float(v), float(s))
        for (a, b, c), v, s in zip(indices, references, sigmas)
    ]


def _named_angle(rows, apex, outer):
    return [row for row in rows if row[0][1] == apex and set(row[0]) == set(outer)]


@pytest.fixture(scope="module")
def daw(pdb_dir):
    model, tables = _built(pdb_dir / "1DAW.pdb")
    return model.pdb["name"].values.astype(str), model.pdb, tables


@pytest.mark.unit
def test_linked_residues_get_the_peptide_ca_c_o_target(daw):
    names, _, tables = daw
    rows = _named_angle(_triples(tables, "intra", names), "C", ("CA", "C", "O"))

    linked = [row for row in rows if abs(row[2] - 120.614) < TOL]
    assert len(linked) == len(rows) - 1  # every residue but the C-terminus
    assert all(abs(row[3] - 1.50) < TOL for row in linked)


@pytest.mark.unit
def test_c_terminal_residue_keeps_its_free_carboxylate(daw):
    names, _, tables = daw
    rows = _triples(tables, "intra", names)

    # OXT restraints survive only where there is no peptide bond to delete them.
    assert len(_named_angle(rows, "C", ("CA", "C", "OXT"))) == 1
    assert len(_named_angle(rows, "C", ("O", "C", "OXT"))) == 1

    free = [
        row
        for row in _named_angle(rows, "C", ("CA", "C", "O"))
        if abs(row[2] - 120.614) > TOL
    ]
    assert len(free) == 1
    assert free[0][2] == pytest.approx(117.199, abs=1e-2)


@pytest.mark.unit
def test_proline_gets_its_own_modification(daw):
    names, pdb, tables = daw
    resnames = pdb["resname"].values.astype(str)
    rows = _named_angle(_triples(tables, "intra", names), "N", ("CA", "N", "CD"))

    assert rows, "1DAW has prolines"
    assert all(abs(row[2] - 112.597) < TOL for row in rows)
    assert all(resnames[row[1]] == "PRO" for row in rows)


@pytest.mark.unit
def test_backbone_bond_targets_switch_to_the_linked_values(daw):
    names, pdb, tables = daw
    resnames = pdb["resname"].values.astype(str)
    indices, references, _ = _table(tables, "bond", "intra")

    def targets(first, second):
        return [
            (float(v), resnames[a])
            for (a, b), v in zip(indices, references)
            if {names[a], names[b]} == {first, second}
        ]

    # DEL-HN1 gives 1.453, DEL-HNP 1.459; only the free N-terminus keeps the
    # per-residue-type value its own component defines.
    n_ca = targets("N", "CA")
    unlinked = [
        value
        for value, _ in n_ca
        if abs(value - 1.453) > TOL and abs(value - 1.459) > TOL
    ]
    assert len(unlinked) == 1
    assert all(
        abs(value - 1.459) < TOL
        for value, resname in n_ca
        if resname == "PRO"
    )

    # DEL-OXT gives 1.229; only the free C-terminus keeps the carboxylate 1.251.
    c_o = [value for value, _ in targets("C", "O")]
    assert sum(abs(value - 1.229) < TOL for value in c_o) == len(c_o) - 1


@pytest.mark.unit
def test_angles_around_a_peptide_carbonyl_close_to_360_degrees(daw):
    """The crystallographic invariant the missing modifications used to break."""
    names, _, tables = daw
    intra = {
        row[1]: row[2]
        for row in _named_angle(_triples(tables, "intra", names), "C", ("CA", "C", "O"))
    }

    link = {}
    for trio, apex, value, _ in _triples(tables, "peptide", names):
        if trio[1] == "C":
            link.setdefault(apex, []).append(value)

    sums = [
        intra[apex] + sum(values)
        for apex, values in link.items()
        if len(values) == 2 and apex in intra
    ]
    assert sums, "1DAW has peptide links"
    # Exactly 360 for TRANS; the PTRANS values close to within 0.05 degrees.
    assert max(abs(total - 360.0) for total in sums) < 0.05


@pytest.mark.unit
def test_amide_hydrogen_angles_close_to_360_degrees(pdb_dir):
    names_pdb = pdb_dir / "1AK5_with_H.pdb"
    model, tables = _built(names_pdb, strip_H=False)
    names = model.pdb["name"].values.astype(str)

    intra = {
        row[1]: row[2]
        for row in _named_angle(_triples(tables, "intra", names), "N", ("CA", "N", "H"))
    }
    assert intra, "1AK5_with_H has amide hydrogens"
    assert all(abs(value - 118.729) < TOL for value in intra.values())

    link = {}
    for trio, apex, value, _ in _triples(tables, "peptide", names):
        if trio[1] == "N":
            link.setdefault(apex, []).append(value)

    sums = [
        intra[apex] + sum(values)
        for apex, values in link.items()
        if len(values) == 2 and apex in intra
    ]
    assert sums
    assert max(abs(total - 360.0) for total in sums) < 0.01


@pytest.mark.unit
def test_deposited_coordinates_fit_the_corrected_targets_better(daw):
    """Deposited geometry sits closer to the linked target than to the free one."""
    names, pdb, tables = daw
    xyz = pdb[["x", "y", "z"]].values.astype(np.float64)
    indices, _, _ = _table(tables, "angle", "intra")

    observed = []
    for a, b, c in indices:
        if names[b] != "C" or {names[a], names[c]} != {"CA", "O"}:
            continue
        v1 = xyz[a] - xyz[b]
        v2 = xyz[c] - xyz[b]
        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)
        observed.append(np.degrees(np.arccos(np.clip(v1 @ v2, -1.0, 1.0))))
    observed = np.array(observed)

    linked_rms = np.sqrt(((observed - 120.614) ** 2).mean())
    free_rms = np.sqrt(((observed - 117.191) ** 2).mean())
    assert linked_rms < free_rms
