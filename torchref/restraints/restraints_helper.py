from pathlib import Path

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
                    f"proper restraint parameters (in external_monomer_library/)."
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


def split_respecting_quotes(line):
    """
    Split a line by whitespace, but preserve quoted strings intact.

    Handles both single and double quotes.

    Parameters
    ----------
    line : str
        Input line to split.

    Returns
    -------
    list of str
        List of tokens split by whitespace, with quoted strings preserved.
    """
    line_new = ""
    in_quotes = False
    quote_char = None
    for character in line:
        if (character == "'" or character == '"') and not in_quotes:
            # Starting a quoted section
            in_quotes = True
            quote_char = character
        elif character == quote_char and in_quotes:
            # Ending a quoted section
            in_quotes = False
            quote_char = None
        elif in_quotes and character == " ":
            # Skip spaces inside quotes
            continue
        line_new += character
    return line_new.split()


def find_cif_file_in_library(resname):
    """
    Find a CIF file in the external monomer library based on residue name.

    The library is organized by first character (e.g., 'ALA' -> 'a/ALA.cif').
    This function works regardless of the current working directory by
    calculating the path relative to this script's location.

    Parameters
    ----------
    resname : str
        Residue name (e.g., 'ALA', 'GLY', 'ATP').

    Returns
    -------
    Path or None
        Path object pointing to the CIF file, or None if not found.
    """
    from torchref.restraints import MONOMER_LIB_PATH

    if MONOMER_LIB_PATH is None:
        raise RuntimeError(
            "MONOMER_LIB_PATH is not set. External monomer library not found."
        )

    # The library organizes files by first character (lowercase)
    first_char = resname[0].lower()

    # Construct the expected path
    cif_file = MONOMER_LIB_PATH / first_char / f"{resname}.cif"

    # Check if file exists
    if cif_file.exists():
        return cif_file
    else:
        # Try uppercase version as fallback
        cif_file_upper = MONOMER_LIB_PATH / first_char / f"{resname.upper()}.cif"
        if cif_file_upper.exists():
            return cif_file_upper
        return None


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
    link_file_path = (
        Path(__file__).parent.parent.parent
        / "external_monomer_library"
        / "list"
        / "mon_lib_list.cif"
    )
    with open(link_file_path) as f:
        lines = f.readlines()

    link_dict = {}
    link_list = None

    # First, read the link list to get all link IDs
    lines_iter = iter(lines)
    for line in lines_iter:
        if line.strip() == "data_link_list":
            # Read the link list
            for line in lines_iter:
                if line.strip() == "loop_":
                    comp_list = []
                    values = []
                    for line in lines_iter:
                        if line.startswith("#") or (
                            line.strip() == "" and len(values) > 0
                        ):
                            break
                        if line.startswith("_chem_link."):
                            comp_list.append(line.split(".")[1].strip())
                        else:
                            split_items = split_respecting_quotes(line.strip())
                            if split_items:
                                values.append(split_items)

                    if len(values) > 0:
                        link_list = pd.DataFrame(values, columns=comp_list)
                        print(f"Found {len(link_list)} link definitions")
                        break
                    break
            break

    # Now read each individual link definition
    lines_iter = iter(lines)
    for line in lines_iter:
        if line.startswith("data_link_") and line.strip() != "data_link_list":
            link_id = line.strip().replace("data_link_", "")

            link_data = {}

            # Read bonds, angles, torsions for this link
            while True:
                line = next(lines_iter, None)
                if line is None:
                    break

                # Stop if we hit the next link
                if line.startswith("data_link_"):
                    break

                if line.strip() == "loop_":
                    # Read the next line to see what type of data this is
                    first_col_line = next(lines_iter)

                    if "_chem_link_bond." in first_col_line:
                        section_type = "bonds"
                        prefix = "_chem_link_bond."
                    elif "_chem_link_angle." in first_col_line:
                        section_type = "angles"
                        prefix = "_chem_link_angle."
                    elif "_chem_link_tor." in first_col_line:
                        section_type = "torsions"
                        prefix = "_chem_link_tor."
                    elif "_chem_link_plane." in first_col_line:
                        section_type = "planes"
                        prefix = "_chem_link_plane."
                    else:
                        # Skip unknown sections
                        continue

                    # Read column names
                    columns = [first_col_line.split(".")[1].strip()]
                    for line in lines_iter:
                        if line.startswith(prefix):
                            columns.append(line.split(".")[1].strip())
                        else:
                            # First data line
                            break

                    # Read data values
                    values = []
                    while line is not None:
                        if (
                            line.strip() == ""
                            or line.startswith("loop_")
                            or line.startswith("data_")
                        ):
                            break
                        split_items = split_respecting_quotes(line.strip())
                        if split_items and not line.startswith("_"):
                            values.append(split_items)

                        line = next(lines_iter, None)

                    if len(values) > 0:
                        df = pd.DataFrame(values, columns=columns)
                        # Standardize column names for consistency
                        df = _standardize_link_columns(df, section_type)
                        link_data[section_type] = df

            if len(link_data) > 0:
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
            # Clip sigma to minimum of 0.1 Å (same as in CIF reader)
            df["sigma"] = (
                pd.to_numeric(df["sigma"], errors="coerce").fillna(0.1).clip(lower=0.1)
            )

    return df


def build_restraints_bondlength(cif, pdb):
    """
    Build bond length restraints from CIF dictionary and PDB DataFrame.

    Parameters
    ----------
    cif : dict
        CIF dictionary containing bond restraint definitions.
    pdb : pandas.DataFrame
        PDB DataFrame with atomic coordinates.

    Returns
    -------
    list
        List containing [column1, column2, references, sigmas] tensors.
    """
    columns1 = []
    columns2 = []
    references = []
    sigmas = []
    for chain_id in pdb["chainid"].unique():
        chain = pdb.loc[pdb["chainid"] == chain_id]
        for resseq in chain["resseq"].unique():
            residue = pdb.loc[(pdb["resseq"] == resseq) & (pdb["chainid"] == chain_id)]
            if residue.ATOM.values[0] == "HETATM":
                continue
            resname = residue["resname"].values[0]
            if resname not in cif:
                continue
            cif_residue = cif[resname]
            cif_bonds_residue = cif_residue["_chem_comp_bond"]
            usable_dict = cif_bonds_residue.loc[
                cif_bonds_residue["atom_id_1"].isin(residue["name"])
                & cif_bonds_residue["atom_id_2"].isin(residue["name"])
            ]
            not_found = cif_bonds_residue.loc[
                ~(
                    cif_bonds_residue["atom_id_1"].isin(residue["name"])
                    & cif_bonds_residue["atom_id_2"].isin(residue["name"])
                )
            ]
            residue.set_index("name", inplace=True)
            column1 = residue.loc[usable_dict["atom_id_1"], "index"].values
            column2 = residue.loc[usable_dict["atom_id_2"], "index"].values
            reference = usable_dict["value_dist"].values.astype(float)
            sigma = usable_dict["value_dist_esd"].values.astype(float)
            columns1.append(column1)
            columns2.append(column2)
            references.append(reference)
            sigmas.append(sigma)
    column1 = torch.tensor(np.concatenate(columns1, dtype=int))
    column2 = torch.tensor(np.concatenate(columns2, dtype=int))
    references = torch.tensor(np.concatenate(references, dtype=float))
    sigmas = torch.tensor(np.concatenate(sigmas, dtype=float))
    return [column1, column2, references, sigmas]


def build_restraints_angles(cif, pdb):
    """
    Build angle restraints from CIF dictionary and PDB DataFrame.

    Parameters
    ----------
    cif : dict
        CIF dictionary containing angle restraint definitions.
    pdb : pandas.DataFrame
        PDB DataFrame with atomic coordinates.

    Returns
    -------
    list
        List containing [column1, column2, column3, references, sigmas] tensors.
    """
    columns1 = []
    columns2 = []
    columns3 = []
    references = []
    sigmas = []
    for chain_id in pdb["chainid"].unique():
        chain = pdb.loc[pdb["chainid"] == chain_id]
        for resseq in chain["resseq"].unique():
            residue = pdb.loc[(pdb["resseq"] == resseq) & (pdb["chainid"] == chain_id)]
            if residue.ATOM.values[0] == "HETATM":
                continue
            resname = residue["resname"].values[0]
            if resname not in cif:
                continue
            cif_residue = cif[resname]
            cif_bonds_residue = cif_residue["_chem_comp_angle"]
            usable_dict = cif_bonds_residue.loc[
                cif_bonds_residue["atom_id_1"].isin(residue["name"])
                & cif_bonds_residue["atom_id_2"].isin(residue["name"])
            ]
            not_found = cif_bonds_residue.loc[
                ~(
                    cif_bonds_residue["atom_id_1"].isin(residue["name"])
                    & cif_bonds_residue["atom_id_2"].isin(residue["name"])
                )
            ]
            residue.set_index("name", inplace=True)
            column1 = residue.loc[usable_dict["atom_id_1"], "index"].values
            column2 = residue.loc[usable_dict["atom_id_2"], "index"].values
            column3 = residue.loc[usable_dict["atom_id_3"], "index"].values
            reference = usable_dict["value_angle"].values.astype(float)
            sigma = usable_dict["value_angle_esd"].values.astype(float)
            columns1.append(column1)
            columns2.append(column2)
            columns3.append(column3)
            references.append(reference)
            sigmas.append(sigma)
    column1 = torch.tensor(np.concatenate(columns1, dtype=int))
    column2 = torch.tensor(np.concatenate(columns2, dtype=int))
    column3 = torch.tensor(np.concatenate(columns3, dtype=int))
    references = torch.tensor(np.concatenate(references, dtype=float))
    sigmas = torch.tensor(np.concatenate(sigmas, dtype=float))
    return [column1, column2, column3, references, sigmas]


def build_restraints_torsion(cif, pdb):
    """
    Build torsion angle restraints from CIF dictionary and PDB DataFrame.

    Parameters
    ----------
    cif : dict
        CIF dictionary containing torsion restraint definitions.
    pdb : pandas.DataFrame
        PDB DataFrame with atomic coordinates.

    Returns
    -------
    list
        List containing [column1, column2, column3, column4, references, sigmas] tensors.
    """
    columns1 = []
    columns2 = []
    columns3 = []
    columns4 = []
    references = []
    sigmas = []
    for chain_id in pdb["chainid"].unique():
        chain = pdb.loc[pdb["chainid"] == chain_id]
        for resseq in chain["resseq"].unique():
            residue = pdb.loc[(pdb["resseq"] == resseq) & (pdb["chainid"] == chain_id)]
            if residue.ATOM.values[0] == "HETATM":
                continue
            resname = residue["resname"].values[0]
            if resname not in cif:
                continue
            cif_residue = cif[resname]
            cif_bonds_residue = cif_residue["_chem_comp_tor"]
            usable_dict = cif_bonds_residue.loc[
                cif_bonds_residue["atom_id_1"].isin(residue["name"])
                & cif_bonds_residue["atom_id_2"].isin(residue["name"])
            ]
            not_found = cif_bonds_residue.loc[
                ~(
                    cif_bonds_residue["atom_id_1"].isin(residue["name"])
                    & cif_bonds_residue["atom_id_2"].isin(residue["name"])
                )
            ]
            residue.set_index("name", inplace=True)
            column1 = residue.loc[usable_dict["atom_id_1"], "index"].values
            column2 = residue.loc[usable_dict["atom_id_2"], "index"].values
            column3 = residue.loc[usable_dict["atom_id_3"], "index"].values
            column4 = residue.loc[usable_dict["atom_id_4"], "index"].values
            reference = usable_dict["value_angle"].values.astype(float)
            sigma = usable_dict["value_angle_esd"].values.astype(float)
            columns1.append(column1)
            columns2.append(column2)
            columns3.append(column3)
            columns4.append(column4)
            references.append(reference)
            sigmas.append(sigma)
    column1 = torch.tensor(np.concatenate(columns1, dtype=int))
    column2 = torch.tensor(np.concatenate(columns2, dtype=int))
    column3 = torch.tensor(np.concatenate(columns3, dtype=int))
    column4 = torch.tensor(np.concatenate(columns4, dtype=int))
    references = torch.tensor(np.concatenate(references, dtype=float))
    sigmas = torch.tensor(np.concatenate(sigmas, dtype=float))
    return [column1, column2, column3, column4, references, sigmas]


def build_restraints_planes(cif, pdb):
    """
    Build planarity restraints from CIF dictionary and PDB DataFrame.

    Parameters
    ----------
    cif : dict
        CIF dictionary containing plane restraint definitions.
    pdb : pandas.DataFrame
        PDB DataFrame with atomic coordinates.

    Returns
    -------
    list
        List containing [column1, plane_numbers, sigmas] tensors.
    """
    columns1 = []
    planenrs = []
    references = []
    sigmas = []
    last_plane = 0
    for chain_id in pdb["chainid"].unique():
        chain = pdb.loc[pdb["chainid"] == chain_id]
        for resseq in chain["resseq"].unique():
            residue = pdb.loc[(pdb["resseq"] == resseq) & (pdb["chainid"] == chain_id)]
            if residue.ATOM.values[0] == "HETATM":
                continue
            resname = residue["resname"].values[0]
            if resname not in cif:
                continue
            cif_residue = cif[resname]
            if "_chem_comp_plane_atom" not in cif_residue:
                continue
            cif_bonds_residue = cif_residue["_chem_comp_plane_atom"]
            usable_dict = cif_bonds_residue.loc[
                cif_bonds_residue["atom_id"].isin(residue["name"])
                & cif_bonds_residue["atom_id"].isin(residue["name"])
            ]
            not_found = cif_bonds_residue.loc[
                ~(
                    cif_bonds_residue["atom_id"].isin(residue["name"])
                    & cif_bonds_residue["atom_id"].isin(residue["name"])
                )
            ]
            residue.set_index("name", inplace=True)
            column1 = residue.loc[usable_dict["atom_id"], "index"].values
            plane_nr = (
                usable_dict.plane_id.str.split("-").str[1].astype(int).values
                + last_plane
            )
            last_plane = plane_nr.max()
            sigma = usable_dict["dist_esd"].values.astype(float)
            columns1.append(column1)
            planenrs.append(plane_nr)
            sigmas.append(sigma)
    column1 = torch.tensor(np.concatenate(columns1, dtype=int))
    planenrs = np.concatenate(planenrs, dtype=int)
    sigmas = torch.tensor(np.concatenate(sigmas, dtype=float))
    return [column1, planenrs, sigmas]


def build_restraints(cif, pdb):
    """
    Build all restraints from CIF dictionary and PDB DataFrame.

    Parameters
    ----------
    cif : dict
        CIF dictionary containing restraint definitions.
    pdb : pandas.DataFrame
        PDB DataFrame with atomic coordinates.

    Returns
    -------
    dict
        Dictionary with keys 'bondlength', 'angles', 'torsion', 'planes'
        containing the respective restraint data.
    """
    bondlength = build_restraints_bondlength(cif, pdb)
    angles = build_restraints_angles(cif, pdb)
    torsion = build_restraints_torsion(cif, pdb)
    planes = build_restraints_planes(cif, pdb)
    restraints = dict()
    restraints["bondlength"] = bondlength
    restraints["angles"] = angles
    restraints["torsion"] = torsion
    restraints["planes"] = planes
    return restraints


def calculate_restraints_bondlength(xyz, restraints_bondlength):
    """
    Calculate bond length restraint energy.

    Parameters
    ----------
    xyz : torch.Tensor
        Atomic coordinates tensor of shape (N, 3).
    restraints_bondlength : list
        List containing [column1, column2, reference, sigma] tensors.

    Returns
    -------
    torch.Tensor
        Total bond length restraint energy.
    """
    column1 = restraints_bondlength[0]
    column2 = restraints_bondlength[1]
    reference = restraints_bondlength[2]
    sigma = restraints_bondlength[3]
    distances = torch.sum((xyz[column1] - xyz[column2]) ** 2, axis=1) ** 0.5
    return torch.sum(torch.exp((torch.abs(distances - reference) / sigma) ** 2))


def calculate_restraints_angles(xyz, restraints_angles):
    """
    Calculate angle restraint energy.

    Parameters
    ----------
    xyz : torch.Tensor
        Atomic coordinates tensor of shape (N, 3).
    restraints_angles : list
        List containing [column1, column2, column3, reference, sigma] tensors.

    Returns
    -------
    torch.Tensor
        Total angle restraint energy.
    """
    column1 = restraints_angles[0]
    column2 = restraints_angles[1]
    column3 = restraints_angles[2]
    reference = restraints_angles[3]
    sigma = restraints_angles[4]
    v1 = xyz[column1] - xyz[column2]
    v2 = xyz[column3] - xyz[column2]
    v1 = v1 / torch.sum(v1**2, axis=1).reshape(-1, 1) ** 0.5
    v2 = v2 / torch.sum(v2**2, axis=1).reshape(-1, 1) ** 0.5
    angle = torch.arccos(torch.sum(v1 * v2, axis=1)) * 180 / np.pi
    return torch.sum(torch.exp(torch.abs((angle - reference)) / sigma))


def calculate_restraints_torsion(xyz, restraints_torsion):
    """
    Calculate torsion angle restraint energy.

    Parameters
    ----------
    xyz : torch.Tensor
        Atomic coordinates tensor of shape (N, 3).
    restraints_torsion : list
        List containing [column1, column2, column3, column4, reference, sigma] tensors.

    Returns
    -------
    torch.Tensor
        Total torsion angle restraint energy.
    """
    column1 = restraints_torsion[0]
    column2 = restraints_torsion[1]
    column3 = restraints_torsion[2]
    column4 = restraints_torsion[3]
    reference = restraints_torsion[4]
    sigma = restraints_torsion[5]
    v1 = xyz[column1] - xyz[column2]
    v2 = xyz[column3] - xyz[column2]
    v3 = xyz[column3] - xyz[column4]
    n1 = torch.linalg.cross(v1, v2)
    n2 = torch.linalg.cross(v2, v3)
    n1 = n1 / torch.sum(n1**2, axis=1).reshape(-1, 1) ** 0.5
    n2 = n2 / torch.sum(n2**2, axis=1).reshape(-1, 1) ** 0.5
    angle = torch.arccos(torch.sum(n1 * n2, axis=1)) * 180 / np.pi
    dif = angle - reference
    dif = torch.min(
        torch.vstack(
            (
                torch.abs(dif),
                torch.abs(dif + 180),
                torch.abs(dif - 180),
                torch.abs(dif - 360),
                torch.abs(dif + 360),
            )
        ),
        axis=0,
    )[0]
    return torch.sum(torch.exp(torch.abs(dif) / sigma))


def calculate_restraints_all(xyz, restraints):
    """
    Calculate total restraint energy for all restraint types.

    Parameters
    ----------
    xyz : torch.Tensor
        Atomic coordinates tensor of shape (N, 3).
    restraints : dict
        Dictionary containing 'bondlength', 'angles', and 'torsion' restraints.

    Returns
    -------
    torch.Tensor
        Total restraint energy (sum of bondlength, angles, and torsion energies).
    """
    bondlength = calculate_restraints_bondlength(xyz, restraints["bondlength"])
    angles = calculate_restraints_angles(xyz, restraints["angles"])
    torsion = calculate_restraints_torsion(xyz, restraints["torsion"])
    return bondlength + angles + torsion


def read_for_component(lines, comp_id):
    """
    Read CIF data for a specific component/compound ID.

    Parameters
    ----------
    lines : list of str
        Lines from CIF file.
    comp_id : str
        Component ID to search for.

    Returns
    -------
    dict or None
        Dictionary of DataFrames for different data types, or None if not found.
    """
    lines = iter(lines)
    for line in lines:
        # Handle both formats:
        # - "data_comp_XXX" (multi-compound files)
        # - "data_XXX" (single-compound files)
        if line.startswith("data_comp_" + comp_id) or line.strip() == "data_" + comp_id:
            line = next(lines)
            dfs = {}
            while True:
                # Check if we've left this component's section
                if line.startswith("data_comp_") and not line.startswith(
                    "data_comp_" + comp_id
                ):
                    break
                if (
                    line.strip().startswith("data_")
                    and line.strip() != "data_" + comp_id
                ):
                    break

                if line.strip() == "loop_":
                    line = next(lines)
                    id = line.split(".")[0].strip()
                    comp_list = [line.split(".")[1].strip()]
                    values = []
                    in_data_section = False

                    for line in lines:
                        # Stop on comment line
                        if line.startswith("#"):
                            break
                        # Stop on blank line if we've already seen data
                        if in_data_section and line.strip() == "":
                            break
                        # Stop if we hit a new component section
                        if line.strip().startswith("data_comp_"):
                            break
                        # Stop if we hit a new loop - but don't consume the line!
                        if line.strip() == "loop_":
                            # Don't break - we'll process this loop in the outer while loop
                            # But we need to exit this inner loop
                            break
                        # Collect column names for this loop
                        if line.startswith(id):
                            comp_list.append(line.split(".")[1].strip())
                        else:
                            # Only process non-empty lines as data
                            split_items = split_respecting_quotes(line.strip())
                            if split_items:
                                values.append(split_items)
                                in_data_section = True

                    try:
                        data = pd.DataFrame(values, columns=comp_list)
                        dfs[id] = data

                        # Apply esd column handling
                        esd_columns = [col for col in data if col.endswith("_esd")]
                        for col in esd_columns:
                            try:
                                data[col] = data[col].astype(float)
                                data.loc[data[col] == 0, col] = 1e-4
                            except:
                                print(
                                    f"Failed to convert esd column to float for: {col}"
                                )

                    except Exception as e:
                        print(f"Failed to create dataframe for: {comp_id}")
                        print(f"Columns ({len(comp_list)}): {comp_list}")
                        print(f"Values ({len(values)} rows):")
                        for i, v in enumerate(values[:5]):  # Show first 5
                            print(f"  Row {i} ({len(v)} items): {v}")
                        if len(values) > 5:
                            print(f"  ... and {len(values)-5} more rows")
                        print(f"Error: {e}")

                    # If we broke because of 'loop_', the line variable now contains 'loop_'
                    # and the while loop will process it in the next iteration
                    # If we broke for another reason, we need to read the next line
                    if line.strip() == "loop_":
                        continue  # Don't read next line, process this loop_ in the while loop

                # Read the next line for the while loop
                try:
                    line = next(lines)
                except StopIteration:
                    break

            return dfs


def read_comp_list(lines):
    """
    Read the compound list from CIF file lines.

    Parameters
    ----------
    lines : list of str
        Lines from CIF file.

    Returns
    -------
    pandas.DataFrame or None
        DataFrame with compound list data, or None if not found.
    """
    lines = iter(lines)
    for line in lines:
        if line.strip() == "data_comp_list":
            for line in lines:
                if line.strip() == "loop_":
                    comp_list = []
                    values = []
                    in_data_section = False
                    for line in lines:
                        # Stop on comment line
                        if line.startswith("#"):
                            break
                        # Stop on blank line if we've already seen data
                        if in_data_section and line.strip() == "":
                            break
                        # Stop if we hit a new section marker
                        if line.strip().startswith("data_") or line.strip().startswith(
                            "loop_"
                        ):
                            break
                        # Collect column names
                        if line.startswith("_chem_comp"):
                            comp_list.append(line.split(".")[1].strip())
                        else:
                            # Only process non-empty lines as data
                            split_items = split_respecting_quotes(line.strip())
                            if split_items:
                                values.append(split_items)
                                in_data_section = True
                    data = pd.DataFrame(values, columns=comp_list)
                    return data
