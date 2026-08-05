"""
PDB file reading and writing (atoms, ANISOU, CRYST1, LINK records).

``read`` returns a reader object, not the data -- call it for the tuple::

    df, cell, spacegroup = pdb.read('structure.pdb')()
    pdb.write(df, 'output.pdb')

Cell and space group travel on ``df.attrs`` (``cell`` / ``spacegroup`` / ``z``),
not in columns, so a DataFrame rebuilt from scratch loses them and
:func:`write` then emits no CRYST1 record.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def _format_pdb_atom_name(name, element="") -> str:
    """Format an atom name into the 4-character PDB atom-name field (cols 13-16).

    PDB / gemmi convention: 4-character names, and atoms with a two-letter
    element symbol (FE, MG), start in column 13; shorter single-letter-element
    names are indented one space. Never truncates below 4 characters.

    Parameters
    ----------
    name : str
        Atom name (any length).
    element : str, optional
        Element symbol, used to decide indentation for short names.

    Returns
    -------
    str
        Exactly 4 characters, to be placed in columns 13-16.
    """
    name = str(name).strip()
    element = str(element).strip()
    if len(name) >= 4:
        return name[:4]
    if len(element) == 2:
        return f"{name:<4}"
    return f" {name:<3}"


def find_header_length(filepath: str, max_header_length: int = 100000) -> int:
    """
    Find the number of header lines in a PDB file.

    Stops at the first line whose leading columns *contain* ``"ATOM"`` (cols
    1-4) or ``"HETATM"`` (cols 1-6) -- a substring test, not a record-type
    ``startswith``, so a header line with those letters there ends the scan.

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
            # Must be an actual CRYST1 record, not a header/REMARK line that
            # merely mentions the word (e.g. "REVDAT ... EXPDTA CRYST1").
            if not line.startswith("CRYST1"):
                continue
            # Fixed columns per the PDB spec (1-indexed):
            #   a 7-15, b 16-24, c 25-33, alpha 34-40, beta 41-47, gamma 48-54,
            #   sGroup 56-66, Z 67-70.
            try:
                cell = [
                    float(line[6:15]),
                    float(line[15:24]),
                    float(line[24:33]),
                    float(line[33:40]),
                    float(line[40:47]),
                    float(line[47:54]),
                ]
            except ValueError:
                # Malformed/short CRYST1 record: treat cell as unavailable
                # rather than crashing the whole read.
                return None, None, None
            spacegroup = line[55:66].strip()
            z = line[66:70].strip()
            return cell, spacegroup, z
    return None, None, None


def load_as_dataframe(
    filepath: str, skipheader: int = 0, skipfooter: int = 1
) -> pd.DataFrame:
    """
    Load a PDB file into a pandas DataFrame.

    Parses ATOM, HETATM and ANISOU records by fixed column positions.

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
        DataFrame whose columns include (among others, in no contractual
        order): ATOM, serial, name, altloc, resname, chainid, resseq, icode,
        x, y, z, occupancy, tempfactor, element, charge, anisou_flag, u11,
        u22, u33, u12, u13, u23, index.
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
    Reader for PDB files: atoms, crystallographic metadata and LINK records.

    Populated by :meth:`read`; calling the instance returns
    ``(dataframe, cell, spacegroup)`` and raises if ``read`` has not run.

    Parameters
    ----------
    verbose : int, optional
        Verbosity level (0=silent, 1=normal, 2=debug). Default is 0.

    Attributes
    ----------
    dataframe : pd.DataFrame
        Atomic data.
    cell : list or None
        Unit cell parameters [a, b, c, alpha, beta, gamma].
    spacegroup : str or None
        Space group symbol.
    z, links
        Molecules per cell, and the parsed LINK records.
    """

    def __init__(self, verbose: int = 0):
        self.verbose = verbose
        self.dataframe = None
        self.cell = None
        self.spacegroup = None
        self.z = None
        self.links = None

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
        self.links = extract_link_records(filepath, verbose=self.verbose)

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
        cell : list of float or None
            Unit cell parameters [a, b, c, alpha, beta, gamma] (the list
            returned by ``read_crystallographic_info``), or None if absent.
        spacegroup : str or None
            Space group symbol, or None if absent.
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
        Reader object; call it for ``(df, cell, spacegroup)``.
    """
    return PDBReader(verbose=verbose).read(filepath)


def extract_pdb_headers(filepath: str) -> list:
    """Read all header lines (before first ATOM/HETATM) from a PDB file.

    Parameters
    ----------
    filepath : str
        Path to the PDB file.

    Returns
    -------
    list of str
        Header lines (without trailing newlines).
    """
    headers = []
    with open(filepath, "r") as f:
        for line in f:
            record = line[:6].strip()
            if record in ("ATOM", "HETATM"):
                break
            headers.append(line.rstrip("\n"))
    return headers


def extract_link_records(filepath: str, verbose: int = 0) -> pd.DataFrame:
    """Parse LINK records from a PDB file (PDB v3.3 format).

    Symmetry-mate links (sym1 or sym2 not blank/``1555``) are skipped with a
    warning, since the asymmetric unit holds no copy of the symmetry mate
    that the bond can attach to.

    Parameters
    ----------
    filepath : str
        Path to the PDB file.
    verbose : int, optional
        If ``> 0``, prints a one-line summary; if ``> 1``, also warns about
        skipped symmetry-mate or malformed records.

    Returns
    -------
    pd.DataFrame
        One row per accepted LINK record with columns ``name1``, ``altloc1``,
        ``resname1``, ``chainid1``, ``resseq1``, ``icode1`` (and the matching
        ``*2`` set), plus ``length`` (NaN if blank). Empty DataFrame if none.
    """
    rows = []
    skipped_sym = 0
    skipped_bad = 0
    with open(filepath, "r") as f:
        for line in f:
            if line[:6] != "LINK  ":
                continue
            try:
                sym1 = line[59:65].strip() if len(line) >= 65 else ""
                sym2 = line[66:72].strip() if len(line) >= 72 else ""
                if sym1 not in ("", "1555") or sym2 not in ("", "1555"):
                    skipped_sym += 1
                    if verbose > 1:
                        print(
                            f"Warning: skipping symmetry-mate LINK "
                            f"(sym1={sym1!r}, sym2={sym2!r}): {line.rstrip()}"
                        )
                    continue

                length_str = line[73:78].strip() if len(line) >= 74 else ""
                length = float(length_str) if length_str else float("nan")

                rows.append(
                    {
                        "name1": line[12:16].strip(),
                        "altloc1": line[16:17].strip(),
                        "resname1": line[17:20].strip(),
                        "chainid1": line[21:22].strip(),
                        "resseq1": int(line[22:26]),
                        "icode1": line[26:27].strip(),
                        "name2": line[42:46].strip(),
                        "altloc2": line[46:47].strip(),
                        "resname2": line[47:50].strip(),
                        "chainid2": line[51:52].strip(),
                        "resseq2": int(line[52:56]),
                        "icode2": line[56:57].strip(),
                        "length": length,
                    }
                )
            except (ValueError, IndexError):
                skipped_bad += 1
                if verbose > 1:
                    print(f"Warning: skipping malformed LINK: {line.rstrip()}")

    df = pd.DataFrame(
        rows,
        columns=[
            "name1", "altloc1", "resname1", "chainid1", "resseq1", "icode1",
            "name2", "altloc2", "resname2", "chainid2", "resseq2", "icode2",
            "length",
        ],
    )
    if verbose > 0 and (len(df) or skipped_sym or skipped_bad):
        print(
            f"LINK records: parsed {len(df)}, "
            f"skipped {skipped_sym} symmetry-mate, {skipped_bad} malformed"
        )
    return df


def write(df: pd.DataFrame, filepath: str, template: str = None, metadata=None) -> None:
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
        PDB template file to copy header from. Deprecated in favour of
        ``metadata``; no ``DeprecationWarning`` is emitted when it is used.
    metadata : RefinementMetadata, optional
        Metadata to render as PDB header (REMARK 3, TITLE, etc.).

    Notes
    -----
    The CRYST1 record is sourced from the DataFrame attributes
    ``df.attrs["cell"]``, ``df.attrs["spacegroup"]`` and
    ``df.attrs.get("z")`` (not from columns). If any of these are missing,
    the file is written without a CRYST1 record and a warning is printed.

    Rows that fail to format are skipped with a printed warning; the
    remaining rows are still written.
    """
    with open(filepath, "w") as n:
        # Write metadata header if provided (before CRYST1)
        if metadata is not None:
            n.write(metadata.render_pdb_header())

        # Copy template header if provided (deprecated path)
        if template is not None:
            with open(template) as t:
                for line in t:
                    if "REMARK" not in line and "ATOM" in line:
                        break
                    n.write(line)

        # Write CRYST1 record if cell info available (directly before atoms)
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

            # 4-character PDB atom-name field (cols 13-16); preceded by the
            # blank col 12 in the format string below.
            name_field = _format_pdb_atom_name(name, element)

            if chainid is None or str(chainid) == "nan":
                chainid = ""

            try:
                s = (
                    f"{str(ATOM):<6}{int(serial):>5} {name_field}{str(altloc):>1}"
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
                    f"ANISOU{int(serial):>5} {name_field}{str(altloc):>1}"
                    f"{str(resname):>3}{str(chainid):>2}{int(resseq):>4}  "
                    f"{int(u11 * 1e4):>{7}}{int(u22 * 1e4):>{7}}{int(u33 * 1e4):>{7}}"
                    f"{int(u12 * 1e4):>{7}}{int(u13 * 1e4):>{7}}{int(u23 * 1e4):>{7}}"
                    f"      {str(element):>{2}}{str(charge):>2}\n"
                )
                n.write(s)

        n.write("END")


def write_multi_model(
    dataframes: List[pd.DataFrame],
    filepath: str,
    model_names: Optional[List[str]] = None,
) -> None:
    """
    Write multiple models to a single PDB file with MODEL/ENDMDL records.

    Each DataFrame is wrapped in a MODEL/ENDMDL pair, producing a
    multi-model PDB file suitable for ensemble or time-resolved data.

    Parameters
    ----------
    dataframes : list of pandas.DataFrame
        List of atom DataFrames (same format as ``write()`` expects).
    filepath : str
        Output PDB filename.
    model_names : list of str, optional
        Names for each model (written as REMARK before each MODEL record).
        If None, models are numbered sequentially.
    """
    if not dataframes:
        return

    with open(filepath, "w") as f:
        # Write CRYST1 from first model if available
        first_df = dataframes[0]
        try:
            cell = first_df.attrs["cell"]
            spacegroup = first_df.attrs["spacegroup"]
            cell_abc = cell[:3]
            cell_angles = cell[3:]
            z = first_df.attrs.get("z", "")
            try:
                strz = str(int(z))
            except Exception:
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
            f.write(line)
        except Exception:
            pass

        for model_idx, df in enumerate(dataframes):
            model_num = model_idx + 1
            if model_names and model_idx < len(model_names):
                f.write(f"REMARK   3  MODEL {model_num}: {model_names[model_idx]}\n")
            f.write(f"MODEL     {model_num:>4}\n")

            for i, row in df.iterrows():
                ATOM = row.get("ATOM", "ATOM")
                serial = row.get("serial", i + 1)
                name = str(row.get("name", "CA"))
                altloc = str(row.get("altloc", ""))
                resname = str(row.get("resname", "UNK"))
                chainid = str(row.get("chainid", ""))
                resseq = int(row.get("resseq", 1))
                icode = str(row.get("icode", ""))
                x = float(row.get("x", 0.0))
                y = float(row.get("y", 0.0))
                z_coord = float(row.get("z", 0.0))
                occupancy = float(row.get("occupancy", 1.0))
                tempfactor = float(row.get("tempfactor", 20.0))
                element = str(row.get("element", "C"))
                charge = row.get("charge", 0)

                if charge > 0:
                    charge_str = "+" + str(charge)
                elif charge == 0:
                    charge_str = ""
                else:
                    charge_str = str(charge)

                # 4-character PDB atom-name field (cols 13-16); preceded by the
                # blank col 12 in the format string below.
                name_field = _format_pdb_atom_name(name, element)

                if chainid is None or chainid == "nan":
                    chainid = ""

                try:
                    s = (
                        f"{str(ATOM):<6}{int(serial):>5} {name_field}{altloc:>1}"
                        f"{resname:>3}{chainid:>2}{resseq:>4}{icode:>4}"
                        f"{round(x, 3):>8}{round(y, 3):>8}{round(z_coord, 3):>8}"
                        f"{round(occupancy, 3):>6.2f}{round(tempfactor, 2):>6}"
                        f"{element:>12}{charge_str:>2}\n"
                    )
                    f.write(s)
                except Exception:
                    pass

            f.write("ENDMDL\n")

        f.write("END\n")


# Deprecated aliases kept for backwards compatibility; prefer the canonical
# names (PDBReader, find_header_length, load_as_dataframe). Slated for removal
# in a future release. These are public symbols.
PDB = PDBReader
find_header_length_pdb_file = find_header_length
load_pdb_as_pd = load_as_dataframe
