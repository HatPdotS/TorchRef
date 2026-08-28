"""CCP4 ``chem_mod`` records: how a monomer's restraints change when it is linked.

The monomer library describes each amino acid in its **free** form -- ``ALA.cif``
carries ``OXT`` and a protonated ``N``, with carboxylate and ammonium geometry. The
in-chain form is produced by the modifications named in the ``chem_link`` table:
``TRANS`` applies ``DEL-OXT`` to the residue donating its C and ``DEL-HN1`` to the
residue donating its N (``DEL-HNP`` for proline). Those modifications delete the
restraints that the link makes meaningless and **overwrite** the ones whose ideal
value changes, notably ``CA-C-O`` (117.2 deg free, 120.6 deg linked) and ``CA-N-H``
(109.6 deg free, 118.7 deg linked).

Skipping them leaves a residue restrained toward geometry it cannot reach: around a
peptide carbonyl carbon the intra-residue ``CA-C-O`` plus the link's ``CA-C-N`` and
``O-C-N`` sum to 360 deg only once ``DEL-OXT`` has been applied.

``_chem_mod_atom`` and ``_chem_mod_tree`` are deliberately ignored. The restraint
builders match library restraints against the atoms actually present in the model
and silently skip any whose atoms are missing, so adding or deleting atom
*definitions* changes nothing downstream; only the restraint sections matter.
"""

from functools import lru_cache
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import pandas as pd

from torchref.topology.monomer.cif import read_library_blocks

#: CIF category -> section name, matching :func:`read_link_definitions`.
_CATEGORY_MAP = {
    "chem_mod_bond": "bonds",
    "chem_mod_angle": "angles",
    "chem_mod_tor": "torsions",
    "chem_mod_plane_atom": "planes",
    "chem_mod_chir": "chirals",
}

#: Per section: the columns identifying a restraint, and the columns a
#: ``change``/``add`` row may overwrite.
_ATOM_COLUMNS = {
    "bonds": ("atom1", "atom2"),
    "angles": ("atom1", "atom2", "atom3"),
    "torsions": ("atom1", "atom2", "atom3", "atom4"),
    "planes": ("plane_id", "atom"),
    "chirals": ("atom_centre", "atom1", "atom2", "atom3"),
}
_VALUE_COLUMNS = {
    "bonds": ("value", "sigma"),
    "angles": ("value", "sigma"),
    "torsions": ("value", "sigma", "periodicity"),
    "planes": ("sigma",),
    "chirals": ("volume_sign",),
}
#: Sections whose row is meaningless without a target value, so an ``add`` row
#: carrying only ``.`` placeholders is dropped rather than appended as NaN.
_REQUIRES_VALUE = {"bonds", "angles", "torsions"}


def _restraint_key(section: str, row: Mapping) -> tuple:
    """Return the order-insensitive identity of one restraint row.

    Bonds match on the unordered atom pair, angles on the apex plus the unordered
    outer pair, torsions on the atom quadruple in either direction, chirals on the
    centre plus the unordered substituents. Planes key on ``(plane_id, atom)``.
    """
    if section == "bonds":
        return tuple(sorted((row["atom1"], row["atom2"])))
    if section == "angles":
        return (row["atom2"], tuple(sorted((row["atom1"], row["atom3"]))))
    if section == "torsions":
        forward = (row["atom1"], row["atom2"], row["atom3"], row["atom4"])
        return min(forward, forward[::-1])
    if section == "planes":
        return (row["plane_id"], row["atom"])
    return (
        row["atom_centre"],
        tuple(sorted((row["atom1"], row["atom2"], row["atom3"]))),
    )


@lru_cache(maxsize=1)
def read_mod_definitions() -> Dict[str, Dict[str, pd.DataFrame]]:
    """Read every ``data_mod_*`` block from ``mon_lib_list.cif``.

    Returns
    -------
    dict
        Keyed by modification ID (``'DEL-OXT'``, ``'DEL-HN1'``, ...). Each value
        holds DataFrames under ``'bonds'``, ``'angles'``, ``'torsions'``,
        ``'planes'`` and ``'chirals'``, with the same column names the monomer
        CIF reader produces plus a ``'function'`` column holding ``'add'``,
        ``'change'`` or ``'delete'``. Sections absent from a block are absent from
        its dict; a modification with no restraint sections at all maps to ``{}``.

    Warnings
    --------
    The result is cached process-wide and shared by every caller -- treat the
    DataFrames as read-only. Distances are the X-ray values
    (``new_value_dist``), matching :class:`~torchref.io.cif_readers.RestraintCIFReader`;
    the ``_nucleus`` columns are ignored.
    """
    from torchref.io.cif_readers import CIFReader

    blocks = read_library_blocks()

    mod_dict: Dict[str, Dict[str, pd.DataFrame]] = {}
    for block_name, block_text in blocks.items():
        if not block_name.startswith("mod_") or block_name == "mod_list":
            continue
        mod_id = block_name[4:]

        reader = CIFReader.from_string(block_text)
        sections = {}
        for cif_category, section in _CATEGORY_MAP.items():
            if cif_category not in reader.data:
                continue
            df = reader.data[cif_category].copy()
            df.columns = [c.split(".")[-1] for c in df.columns]
            sections[section] = _standardize_mod_columns(df, section)
        mod_dict[mod_id] = sections

    return mod_dict


def _standardize_mod_columns(df: pd.DataFrame, section: str) -> pd.DataFrame:
    """Rename ``_chem_mod`` columns to the restraint-CIF names, on a copy.

    ``atom_id_N -> atomN`` and the ``new_value_*``/``new_*_esd`` pairs to
    ``value``/``sigma``, so a modification row and a component row are directly
    comparable. Missing numbers become NaN, which the caller reads as "keep the
    current value".
    """
    df = df.rename(
        columns={
            "atom_id_1": "atom1",
            "atom_id_2": "atom2",
            "atom_id_3": "atom3",
            "atom_id_4": "atom4",
            "atom_id_centre": "atom_centre",
            "atom_id": "atom",
            "new_value_dist": "value",
            "new_value_dist_esd": "sigma",
            "new_value_angle": "value",
            "new_value_angle_esd": "sigma",
            "new_dist_esd": "sigma",
            "new_period": "periodicity",
            "new_volume_sign": "volume_sign",
        }
    )

    columns = ["function"] + list(_ATOM_COLUMNS[section])
    for column in _VALUE_COLUMNS[section]:
        if column not in df.columns:
            df[column] = pd.NA
        elif column != "volume_sign":
            df[column] = pd.to_numeric(df[column], errors="coerce")
        columns.append(column)

    df["function"] = df["function"].astype(str).str.strip().str.lower()
    return df[columns].reset_index(drop=True)


def link_modifications(link_list: Optional[pd.DataFrame]) -> Dict[str, Tuple]:
    """Map each link ID to the modifications it applies to its two partners.

    Parameters
    ----------
    link_list : pandas.DataFrame or None
        The ``chem_link`` table returned by
        :func:`~torchref.topology.monomer.cif.read_link_definitions`.

    Returns
    -------
    dict
        ``{link_id: (mod_id_1, mod_id_2)}``, where ``mod_id_1`` applies to the
        first partner of the link (for ``TRANS``, the residue donating its C).
        A CIF null (``.``) becomes None.
    """
    if link_list is None or len(link_list) == 0:
        return {}

    def clean(value) -> Optional[str]:
        text = str(value).strip()
        return None if text in ("", ".", "?", "nan") else text

    return {
        str(row["id"]).strip(): (clean(row["mod_id_1"]), clean(row["mod_id_2"]))
        for _, row in link_list.iterrows()
    }


def apply_modifications(
    comp: Mapping[str, pd.DataFrame],
    mod_ids: Sequence[str],
    mod_dict: Mapping[str, Mapping[str, pd.DataFrame]],
) -> Dict[str, pd.DataFrame]:
    """Return ``comp``'s restraints with the named modifications applied.

    Parameters
    ----------
    comp : mapping of str to pandas.DataFrame
        One component's restraints, as produced by
        :func:`~torchref.topology.monomer.cif.read_cif`.
    mod_ids : sequence of str
        Modification IDs to apply, in order. Unknown IDs are ignored.
    mod_dict : mapping
        Modification definitions from :func:`read_mod_definitions`.

    Returns
    -------
    dict
        A new dict with new DataFrames; ``comp`` is not touched. Within each
        modification the ``delete`` rows are applied first, then ``change``, then
        ``add``, so a modification that deletes and re-adds the same restraint
        ends up with the added one.
    """
    if not mod_ids:
        return dict(comp)

    result = {
        key: (value.copy() if isinstance(value, pd.DataFrame) else value)
        for key, value in comp.items()
    }
    for mod_id in mod_ids:
        modification = mod_dict.get(mod_id)
        if not modification:
            continue
        for section, mod_rows in modification.items():
            if mod_rows is None or mod_rows.empty:
                continue
            target = result.get(section)
            if target is None:
                target = pd.DataFrame(
                    columns=list(_ATOM_COLUMNS[section]) + list(_VALUE_COLUMNS[section])
                )
            result[section] = _apply_section(target, mod_rows, section)
    return result


def _apply_section(
    target: pd.DataFrame, mod_rows: pd.DataFrame, section: str
) -> pd.DataFrame:
    """Apply one modification's rows for one restraint section."""
    if target.empty and not len(mod_rows):
        return target

    keys = [_restraint_key(section, row) for _, row in target.iterrows()]
    by_key = {}
    for position, key in enumerate(keys):
        by_key.setdefault(key, []).append(position)

    dropped: set = set()
    additions = []
    for _, mod_row in mod_rows.iterrows():
        function = mod_row["function"]
        key = _restraint_key(section, mod_row)
        positions = [p for p in by_key.get(key, []) if p not in dropped]

        if function == "delete":
            dropped.update(by_key.get(key, []))
        elif function == "change" or (function == "add" and positions):
            # 'add' on a restraint the component already defines is treated as a
            # change: the library expects one row per restraint.
            for position in positions:
                for column in _VALUE_COLUMNS[section]:
                    value = mod_row[column]
                    if pd.notna(value):
                        target.iloc[position, target.columns.get_loc(column)] = value
        elif function == "add":
            if section in _REQUIRES_VALUE and pd.isna(mod_row["value"]):
                continue
            additions.append(_addition_row(target, mod_row, section))

    if dropped:
        target = target.drop(target.index[sorted(dropped)])
    if additions:
        target = pd.concat([target, pd.DataFrame(additions)], ignore_index=True)
    return target.reset_index(drop=True)


def _addition_row(target: pd.DataFrame, mod_row: pd.Series, section: str) -> dict:
    """Build one new restraint row, defaulting values the modification omits."""
    row = {column: mod_row[column] for column in _ATOM_COLUMNS[section]}
    for column in _VALUE_COLUMNS[section]:
        value = mod_row[column]
        if pd.isna(value) and section == "planes" and column == "sigma":
            value = 0.02
        row[column] = value
    for column in target.columns:
        row.setdefault(column, pd.NA)
    return row


def modification_ids_for_links(
    link_ids: Iterable[str], link_list: Optional[pd.DataFrame]
) -> Dict[str, Tuple]:
    """Return ``{link_id: (mod_id_1, mod_id_2)}`` for the requested links only."""
    modifications = link_modifications(link_list)
    return {link_id: modifications.get(link_id, (None, None)) for link_id in link_ids}


__all__ = [
    "apply_modifications",
    "link_modifications",
    "modification_ids_for_links",
    "read_mod_definitions",
]
