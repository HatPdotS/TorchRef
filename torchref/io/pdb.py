"""
PDB file format reading and writing.

This module provides functions for reading and writing PDB files containing
atomic coordinate data.

Functions
---------
read
    Read a PDB file and return a reader object.
write
    Write atomic coordinates to a PDB file.
find_header_length
    Find the number of header lines in a PDB file.
load_as_dataframe
    Load a PDB file into a pandas DataFrame.
read_crystallographic_info
    Extract unit cell and space group from a PDB file.

Classes
-------
PDBReader
    Reader class for PDB files.

Examples
--------
>>> from torchref.io import pdb
>>>
>>> # Reading
>>> reader = pdb.read('structure.pdb', verbose=1)
>>> df, cell, spacegroup = reader()
>>>
>>> # Writing
>>> pdb.write(df, 'output.pdb')
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def find_header_length(filepath: str, max_header_length: int = 100000) -> int:
    """
    Find the number of header lines in a PDB file.

    Scans the file line by line until an ATOM or HETATM record is found.

    Parameters
    ----------
    filepath : str
        Path to the PDB file.
    max_header_length : int, optional
        Maximum number of header lines to scan. Default is 100000.

    Returns
    -------
    int
        Number of header lines before the first ATOM/HETATM record.

    Raises
    ------
    ValueError
        If header length exceeds max_header_length.
    """
    skipheader = 0
    with open(filepath, "r") as f:
        for line in f:
            if "ATOM" in line[0:4] or "HETATM" in line[0:6]:
                break
            skipheader += 1
            if skipheader > max_header_length:
                raise ValueError("Header length is too long, check file")
    return skipheader


def read_crystallographic_info(
    filepath: str,
) -> Tuple[Optional[List[float]], Optional[str], Optional[str]]:
    """
    Extract crystallographic information from a PDB file.

    Reads the CRYST1 record to obtain unit cell parameters and space group.

    Parameters
    ----------
    filepath : str
        Path to the PDB file.

    Returns
    -------
    cell : list of float or None
        Unit cell parameters [a, b, c, alpha, beta, gamma] in A and degrees.
    spacegroup : str or None
        Space group symbol.
    z : str or None
        Number of molecules per unit cell.
    """
    with open(filepath, "r") as f:
        for line in f:
            if "CRYST1" in line:
                a = line[6:15]
                b = line[15:24]
                c = line[24:33]
                alpha = line[33:40]
                beta = line[40:47]
                gamma = line[47:54]
                spacegroup = line[55:68].strip()
                z = line[68:].strip()
                cell = [
                    float(a),
                    float(b),
                    float(c),
                    float(alpha),
                    float(beta),
                    float(gamma),
                ]
                return cell, spacegroup, z
    return None, None, None


def load_as_dataframe(
    filepath: str, skipheader: int = 0, skipfooter: int = 1
) -> pd.DataFrame:
    """
    Load a PDB file into a pandas DataFrame.

    Parses ATOM, HETATM, and ANISOU records from a PDB file and returns
    a structured DataFrame with all atomic properties.

    Parameters
    ----------
    filepath : str
        Path to the PDB file.
    skipheader : int, optional
        Number of header lines to skip. If 0, automatically detected.
    skipfooter : int, optional
        Number of footer lines to skip. Default is 1.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ATOM, serial, name, altloc, resname, chainid,
        resseq, icode, x, y, z, occupancy, tempfactor, element, charge,
        anisou_flag, u11, u22, u33, u12, u13, u23, index.
        DataFrame attributes include 'cell', 'spacegroup', and 'z'.
    """
    if skipheader == 0:
        skipheader = find_header_length(filepath)

    colspecs = [
        (0, 6),
        (6, 11),
        (12, 16),
        (16, 17),
        (17, 20),
        (21, 22),
        (22, 26),
        (26, 27),
        (30, 38),
        (38, 46),
        (46, 54),
        (54, 60),
        (60, 66),
        (76, 78),
        (78, 80),
    ]
    names = [
        "ATOM",
        "serial",
        "name",
        "altloc",
        "resname",
        "chainid",
        "resseq",
        "icode",
        "x",
        "y",
        "z",
        "occupancy",
        "tempfactor",
        "element",
        "charge",
    ]

    pdb = pd.read_fwf(
        filepath,
        names=names,
        colspecs=colspecs,
        skiprows=skipheader,
        skipfooter=skipfooter,
        keep_default_na=False,
        na_values=[""],
    )
    pdb["anisou_flag"] = False

    # Read ANISOU records
    anisou_names = [
        "ATOM",
        "serial",
        "name",
        "altloc",
        "resname",
        "chainid",
        "resseq",
        "u11",
        "u22",
        "u33",
        "u12",
        "u13",
        "u23",
        "element",
    ]
    anisou_colspecs = [
        (0, 6),
        (6, 11),
        (12, 16),
        (16, 17),
        (17, 20),
        (21, 22),
        (22, 26),
        (29, 35),
        (36, 42),
        (43, 49),
        (50, 56),
        (57, 63),
        (63, 70),
        (76, 78),
    ]
    anisou = pd.read_fwf(
        filepath,
        names=anisou_names,
        colspecs=anisou_colspecs,
        skiprows=skipheader,
        skipfooter=skipfooter,
        keep_default_na=False,
        na_values=[""],
    )
    anisou = anisou.loc[anisou["ATOM"] == "ANISOU"]
    pdb = pdb.loc[(pdb["ATOM"] == "ATOM") | (pdb["ATOM"] == "HETATM")]

    anisou.drop(columns=["ATOM"], inplace=True)
    pdb = pdb.merge(
        anisou,
        on=["serial", "name", "altloc", "resname", "chainid", "resseq", "element"],
        how="left",
    )
    pdb.loc[pdb["u11"].notnull(), "anisou_flag"] = True
    pdb[["u11", "u22", "u33", "u12", "u13", "u23"]] = (
        pdb[["u11", "u22", "u33", "u12", "u13", "u23"]].astype(float) / 1e4
    )
    pdb[["serial", "resseq"]] = pdb[["serial", "resseq"]].astype(int)
    pdb[["x", "y", "z", "occupancy", "tempfactor"]] = pdb[
        ["x", "y", "z", "occupancy", "tempfactor"]
    ].astype(float)
    pdb[["altloc", "icode"]] = pdb[["altloc", "icode"]].fillna("")
    pdb["charge"] = (
        pdb["charge"]
        .astype(str)
        .str.strip("+")
        .str.replace("1-", "-1")
        .str.replace("2-", "-2")
        .astype(float)
        .fillna(0)
        .astype(int)
    )
    pdb["element"] = pdb["element"].astype(str).str.strip().str.capitalize()
    pdb["index"] = np.arange(pdb.shape[0]).astype(int)

    try:
        cell, spacegroup, z = read_crystallographic_info(filepath)
    except:
        cell, spacegroup, z = None, None, None

    pdb.attrs["cell"] = cell
    pdb.attrs["spacegroup"] = spacegroup
    pdb.attrs["z"] = z

    return pdb


class PDBReader:
    """
    Reader for PDB files containing atomic coordinate data.

    This class reads PDB files and extracts atomic coordinates, properties,
    and crystallographic metadata.

    Attributes
    ----------
    verbose : int
        Verbosity level for logging.
    dataframe : pd.DataFrame
        DataFrame containing atomic data.
    cell : list or None
        Unit cell parameters [a, b, c, alpha, beta, gamma].
    spacegroup : str or None
        Space group symbol.

    Examples
    --------
    >>> reader = pdb.read('structure.pdb', verbose=1)
    >>> df, cell, spacegroup = reader()
    >>> print(f"Loaded {len(df)} atoms")
    """

    def __init__(self, verbose: int = 0):
        """
        Initialize PDB reader.

        Parameters
        ----------
        verbose : int, optional
            Verbosity level (0=silent, 1=normal, 2=debug). Default is 0.
        """
        self.verbose = verbose
        self.dataframe = None
        self.cell = None
        self.spacegroup = None
        self.z = None

    def read(self, filepath: str) -> "PDBReader":
        """
        Read a PDB file and extract atomic data.

        Parameters
        ----------
        filepath : str
            Path to the PDB file.

        Returns
        -------
        PDBReader
            Self, for method chaining.
        """
        if self.verbose > 1:
            print(f"Reading PDB file: {filepath}")

        self.dataframe = load_as_dataframe(filepath)
        self.cell, self.spacegroup, self.z = read_crystallographic_info(filepath)

        if self.verbose > 0:
            print(f"Loaded {len(self.dataframe)} atoms")

        return self

    def __call__(self) -> Tuple[pd.DataFrame, Optional[np.ndarray], Optional[str]]:
        """
        Return extracted data in a standardized format.

        Returns
        -------
        dataframe : pd.DataFrame
            DataFrame with atomic data.
        cell : np.ndarray or None
            Unit cell parameters [a, b, c, alpha, beta, gamma].
        spacegroup : str or None
            Space group symbol.
        """
        if self.dataframe is None:
            raise ValueError("No data loaded. Call read() first.")
        return self.dataframe, self.cell, self.spacegroup


def read(filepath: str, verbose: int = 0) -> PDBReader:
    """
    Read a PDB file.

    Parameters
    ----------
    filepath : str
        Path to the PDB file.
    verbose : int, optional
        Verbosity level. Default is 0.

    Returns
    -------
    PDBReader
        Reader object with data loaded.
    """
    return PDBReader(verbose=verbose).read(filepath)


def write(df: pd.DataFrame, filepath: str, template: str = None) -> None:
    """
    Write a DataFrame to a PDB file.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing atom data with columns: ATOM, serial, name,
        altloc, resname, chainid, resseq, icode, x, y, z, occupancy,
        tempfactor, element, charge.
    filepath : str
        Output PDB filename.
    template : str, optional
        PDB template file to copy header from.
    """
    with open(filepath, "w") as n:
        # Write CRYST1 record if cell info available
        try:
            cell = df.attrs["cell"]
            spacegroup = df.attrs["spacegroup"]
            cell_abc = cell[:3]
            cell_angles = cell[3:]
            z = df.attrs.get("z", "")
            try:
                strz = str(int(z))
            except:
                strz = ""
            line = (
                "CRYST1"
                + "".join([f"{i:>9.3f}" for i in cell_abc])
                + "".join([f"{i:>7.2f}" for i in cell_angles])
                + " "
                + f"{spacegroup:<14}"
                + strz
                + "\n"
            )
            n.write(line)
        except:
            print("No cell information found, writing without cell and spacegroup")

        # Copy template header if provided
        if template is not None:
            with open(template) as t:
                for line in t:
                    if "REMARK" not in line and "ATOM" in line:
                        break
                    n.write(line)

        # Write atom records
        for i, row in df.iterrows():
            (
                ATOM,
                serial,
                name,
                altloc,
                resname,
                chainid,
                resseq,
                icode,
                x,
                y,
                z_coord,
                occupancy,
                tempfactor,
                element,
                charge,
            ) = row[
                [
                    "ATOM",
                    "serial",
                    "name",
                    "altloc",
                    "resname",
                    "chainid",
                    "resseq",
                    "icode",
                    "x",
                    "y",
                    "z",
                    "occupancy",
                    "tempfactor",
                    "element",
                    "charge",
                ]
            ]

            if charge > 0:
                charge = "+" + str(charge)
            elif charge == 0:
                charge = ""
            else:
                charge = str(charge)

            if len(name) > 3:
                name = name[-3:]
            if len(name) < 3:
                name = name + " " * (3 - len(name))

            if chainid is None or str(chainid) == "nan":
                chainid = ""

            try:
                s = (
                    f"{str(ATOM):<6}{int(serial):>5}{str(name):>5}{str(altloc):>1}"
                    f"{str(resname):>3}{str(chainid):>2}{int(resseq):>4}{str(icode):>4}"
                    f"{round(x, 3):>8}{round(y, 3):>8}{round(z_coord, 3):>8}"
                    f"{round(occupancy, 3):>6.2f}{round(tempfactor, 2):>6}"
                    f"{str(element):>12}{charge:>2}\n"
                )
                n.write(s)
            except:
                print("row", i, "failed")
                print(row)

            # Write ANISOU record if present
            if row["anisou_flag"]:
                u11, u22, u33, u12, u13, u23 = row[
                    ["u11", "u22", "u33", "u12", "u13", "u23"]
                ]
                s = (
                    f"ANISOU{int(serial):>5}{str(name):>5}{str(altloc):>1}"
                    f"{str(resname):>3}{str(chainid):>2}{int(resseq):>4}  "
                    f"{int(u11 * 1e4):>{7}}{int(u22 * 1e4):>{7}}{int(u33 * 1e4):>{7}}"
                    f"{int(u12 * 1e4):>{7}}{int(u13 * 1e4):>{7}}{int(u23 * 1e4):>{7}}"
                    f"      {str(element):>{2}}{str(charge):>2}\n"
                )
                n.write(s)

        n.write("END")


# Legacy aliases for backwards compatibility during transition
PDB = PDBReader
find_header_length_pdb_file = find_header_length
load_pdb_as_pd = load_as_dataframe
