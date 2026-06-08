
import numpy as np
import pandas as pd
import torch

from torchref.io import cif


def validate_restraint_data(residue_data, cif_path):
    """
    Validate that the CIF file contains actual restraint parameters.

    Parameters
    ----------
    residue_data : dict
        Dictionary of residue restraint data from CIF file.
    cif_path : str or Path
        Path to the CIF file being validated.

    Raises
    ------
    ValueError
        If the file doesn't contain proper restraint data.
    """
    if not residue_data:
        raise ValueError(f"CIF file {cif_path} contains no compound definitions")

    for comp_id, data in residue_data.items():
        if data is None or not data:
            raise ValueError(
                f"CIF file {cif_path}: No data found for compound '{comp_id}'\n"
                f"This may be a structure-only CIF file without restraint parameters."
            )

        # Check if bond data exists and has restraint parameters (standardized column names)
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
            # No bond data at all - definitely not a restraint file
            raise ValueError(
                f"CIF file {cif_path}: Compound '{comp_id}' has no bond restraint data.\n"
                f"Available data types: {list(data.keys())}\n\n"
                f"This is not a valid restraint file. Please use proper restraint files from\n"
                f"the monomer library or pass None to use the default library."
            )


def read_cif(cif_path):
    """
    Read restraint CIF file using the new RestraintCIFReader.

    Returns dictionary with standardized keys for compatibility with restraints.py.

    Parameters
    ----------
    cif_path : str or Path
        Path to restraint CIF file.

    Returns
    -------
    dict
        Dictionary mapping compound IDs to restraint data with standardized keys::

            {
                'comp_id': {
                    'bond': DataFrame with bond restraints,
                    'angle': DataFrame with angle restraints,
                    'torsion': DataFrame with torsion restraints,
                    'plane': DataFrame with planarity restraints,
                    'chiral': DataFrame with chirality definitions,
                    'atom': DataFrame with atom definitions
                }
            }
    """
    # Use the new RestraintCIFReader
    reader = cif.RestraintCIFReader(cif_path)

    # Get all restraints
    all_restraints = reader.get_all_restraints()

    # Validate the data
    validate_restraint_data(all_restraints, cif_path)

    return all_restraints


def find_cif_file_in_library(resname):
    """
    Find a CIF file in the monomer library based on residue name.

    Resolves files using the MonomerLibraryManager priority chain:
    environment variable > bundled package data > user cache >
    legacy external_monomer_library > on-demand download.

    Parameters
    ----------
    resname : str
        Residue name (e.g., 'ALA', 'GLY', 'ATP').

    Returns
    -------
    Path or None
        Path object pointing to the CIF file, or None if not found.
    """
    from torchref.restraints.library import get_library_manager

    return get_library_manager().get_cif_file(resname)


def read_link_definitions():
    """
    Read link definitions from mon_lib_list.cif.

    Returns
    -------
    tuple
        A tuple of (link_dict, link_list) where:

        - link_dict : dict
            Dictionary where keys are link IDs (e.g., 'TRANS', 'CIS')
            and values are dictionaries containing:

            - 'bonds': DataFrame of inter-residue bonds
            - 'angles': DataFrame of inter-residue angles
            - 'torsions': DataFrame of inter-residue torsions

        - link_list : DataFrame
            DataFrame containing the list of all link definitions.
    """
    from torchref.io.cif_readers import CIFReader
    from torchref.restraints.library import get_library_manager
    import os
    import re

    link_file_path = str(get_library_manager().get_link_definitions_path())
    with open(link_file_path) as f:
        content = f.read()

    # Split content into blocks at every "data_" line.
    # re.split keeps the delimiter when using a capture group.
    parts = re.split(r"(?=^data_)", content, flags=re.MULTILINE)

    blocks = {}  # block_name -> block_text
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part.startswith("data_"):
            first_newline = part.find("\n")
            if first_newline == -1:
                block_name = part[5:].strip()
                block_body = ""
            else:
                block_name = part[5:first_newline].strip()
                block_body = part
            blocks[block_name] = block_body

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
    """
    Standardize column names in link definitions to match restraint CIF format.

    Converts from _chem_link format to standardized format:

    - atom_id_1/2/3/4 -> atom1/2/3/4
    - value_dist -> value
    - value_dist_esd -> sigma
    - value_angle -> value
    - value_angle_esd -> sigma

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with link restraint data.
    section_type : str
        Type of restraint section ('bonds', 'angles', 'torsions', 'planes').

    Returns
    -------
    pandas.DataFrame
        DataFrame with standardized column names.
    """
    if df.empty:
        return df

    # Create a copy to avoid modifying original
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


