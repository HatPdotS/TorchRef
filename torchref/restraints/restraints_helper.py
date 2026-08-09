"""Helper functions for reading and validating restraint CIF dictionaries
and link definitions used by the restraints module.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
import torch

from torchref.io import cif


def validate_restraint_data(residue_data, cif_path):
    """Raise ``ValueError`` unless every compound carries real restraint parameters.

    Rejects structure-only CIFs: a compound must have a bond section, and that
    section must carry ``value`` and ``sigma`` columns.
    """
    if not residue_data:
        raise ValueError(f"CIF file {cif_path} contains no compound definitions")

    for comp_id, data in residue_data.items():
        if data is None or not data:
            raise ValueError(
                f"CIF file {cif_path}: No data found for compound '{comp_id}'\n"
                f"This may be a structure-only CIF file without restraint parameters."
            )

        if "bonds" in data or "bond" in data:
            bond_key = "bonds" if "bonds" in data else "bond"
            bond_df = data[bond_key]
            required_cols = ["value", "sigma"]
            missing_cols = [col for col in required_cols if col not in bond_df.columns]

            if missing_cols:
                raise ValueError(
                    f"CIF file {cif_path}: Compound '{comp_id}' is missing restraint parameters.\n"
                    f"Missing columns in bond restraints: {missing_cols}\n"
                    f"Available columns: {list(bond_df.columns)}\n\n"
                    f"This appears to be a structure definition file (e.g., from PDB) rather than\n"
                    f"a proper restraint file. Restraint files must include ideal geometry parameters\n"
                    f"such as 'value' and 'sigma' for bonds.\n\n"
                    f"Solution: Remove this file or use the monomer library files which contain\n"
                    f"proper restraint parameters (from the CCP4 Monomer Library)."
                )
        else:
            raise ValueError(
                f"CIF file {cif_path}: Compound '{comp_id}' has no bond restraint data.\n"
                f"Available data types: {list(data.keys())}\n\n"
                f"This is not a valid restraint file. Please use proper restraint files from\n"
                f"the monomer library or pass None to use the default library."
            )


def read_cif(cif_path):
    """
    Read a restraint CIF into ``{comp_id: {section: DataFrame}}``.

    Sections are the standardized keys ``bond``, ``angle``, ``torsion``, ``plane``,
    ``chiral`` and ``atom``. Runs :func:`validate_restraint_data`, so a
    structure-only CIF raises ``ValueError`` here rather than yielding empty
    restraints later.
    """
    reader = cif.RestraintCIFReader(cif_path)

    all_restraints = reader.get_all_restraints()

    validate_restraint_data(all_restraints, cif_path)

    return all_restraints


def find_cif_file_in_library(resname):
    """Path to ``resname``'s CIF in the monomer library, or None.

    Delegates to :meth:`MonomerLibraryManager.get_cif_file`, whose last resort is
    an on-demand download.
    """
    from torchref.restraints.library import get_library_manager

    return get_library_manager().get_cif_file(resname)


def split_data_blocks(content):
    """Split a multi-block CIF into ``{block_name: block_text}``.

    ``mon_lib_list.cif`` holds hundreds of ``data_`` blocks in one file, so the
    readers below slice it up themselves and hand each block to
    :class:`~torchref.io.cif_readers.CIFReader` separately. The ``data_`` prefix
    is stripped from the name but kept in the text.
    """
    import re

    blocks = {}
    # re.split with a lookahead keeps the "data_" line at the head of each part.
    for part in re.split(r"(?=^data_)", content, flags=re.MULTILINE):
        part = part.strip()
        if not part.startswith("data_"):
            continue
        first_newline = part.find("\n")
        if first_newline == -1:
            blocks[part[5:].strip()] = ""
        else:
            blocks[part[5:first_newline].strip()] = part
    return blocks


@lru_cache(maxsize=1)
def read_library_blocks():
    """Return ``mon_lib_list.cif`` split into ``{block_name: block_text}``.

    Shared by :func:`read_link_definitions` and
    :func:`~torchref.restraints.modifications.read_mod_definitions`, which read
    disjoint parts of the same 4 MB file.

    Warnings
    --------
    Cached process-wide; the returned dict is shared, so do not mutate it.
    """
    from torchref.restraints.library import get_library_manager

    path = str(get_library_manager().get_link_definitions_path())
    with open(path) as handle:
        return split_data_blocks(handle.read())


@lru_cache(maxsize=1)
def read_link_definitions():
    """
    Read inter-residue link definitions from mon_lib_list.cif.

    Returns
    -------
    link_dict : dict
        Keyed by link ID ('TRANS', 'CIS', 'disulf', ...), each holding DataFrames
        under 'bonds', 'angles', 'torsions', 'planes' and 'chirals'.
    link_list : DataFrame or None
        The ``chem_link`` table, or None if the file has no ``link_list`` block.
        Its ``mod_id_1``/``mod_id_2`` columns name the modifications each link
        applies to its partners -- see :mod:`torchref.restraints.modifications`.

    Warnings
    --------
    Parsing the 4 MB library takes a few tenths of a second, so the result is
    cached process-wide and shared by every caller -- treat it as read-only.
    """
    from torchref.io.cif_readers import CIFReader

    blocks = read_library_blocks()

    # --- Parse the link_list block ---
    link_list = None
    if "link_list" in blocks:
        reader = CIFReader.from_string(blocks["link_list"])
        if "chem_link" in reader.data:
            df = reader.data["chem_link"]
            # Strip CIF prefixes from column names
            df.columns = [c.split(".")[-1] for c in df.columns]
            link_list = df
            print(f"Found {len(link_list)} link definitions")

    # --- Parse each individual link block ---
    link_dict = {}

    # Map CIFReader category names to our section types
    category_map = {
        "chem_link_bond": "bonds",
        "chem_link_angle": "angles",
        "chem_link_tor": "torsions",
        "chem_link_plane": "planes",
        "chem_link_chir": "chirals",
    }

    for block_name, block_text in blocks.items():
        if not block_name.startswith("link_") or block_name == "link_list":
            continue

        link_id = block_name[5:]  # strip "link_" prefix

        # Parse this block independently
        reader = CIFReader.from_string(block_text)

        link_data = {}
        for cif_category, section_type in category_map.items():
            if cif_category in reader.data:
                df = reader.data[cif_category]
                # Strip CIF prefixes (e.g. "_chem_link_bond.atom_id_1" -> "atom_id_1")
                df.columns = [c.split(".")[-1] for c in df.columns]
                df = _standardize_link_columns(df, section_type)
                link_data[section_type] = df

        if link_data:
            link_dict[link_id] = link_data

    return link_dict, link_list


def _standardize_link_columns(df, section_type):
    """Rename ``_chem_link`` columns to the restraint-CIF names, on a copy.

    ``atom_id_N -> atomN``, and ``value_dist``/``value_angle`` plus their ``_esd``
    partners to ``value``/``sigma``; ``section_type`` picks which pair applies.
    """
    if df.empty:
        return df

    df = df.copy()

    # Column mapping
    column_map = {
        "atom_id_1": "atom1",
        "atom_id_2": "atom2",
        "atom_id_3": "atom3",
        "atom_id_4": "atom4",
    }

    # Apply common mappings
    df = df.rename(columns=column_map)

    # Handle value/sigma based on section type
    if section_type == "bonds":
        if "value_dist" in df.columns:
            df = df.rename(columns={"value_dist": "value"})
        if "value_dist_esd" in df.columns:
            df = df.rename(columns={"value_dist_esd": "sigma"})
    elif section_type in ["angles", "torsions"]:
        if "value_angle" in df.columns:
            df = df.rename(columns={"value_angle": "value"})
        if "value_angle_esd" in df.columns:
            df = df.rename(columns={"value_angle_esd": "sigma"})
    elif section_type == "planes":
        if "atom_id" in df.columns:
            df = df.rename(columns={"atom_id": "atom"})
        if "dist_esd" in df.columns:
            df = df.rename(columns={"dist_esd": "sigma"})
            # Clip sigma: default 0.02 Å, minimum 0.001 Å (consistent with
            # monomer CIF reader in cif_readers.py:_standardize_planes)
            df["sigma"] = (
                pd.to_numeric(df["sigma"], errors="coerce").fillna(0.02).clip(lower=0.001)
            )

    return df


