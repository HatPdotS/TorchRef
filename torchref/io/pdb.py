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


def _model_numbers(filepath: str, skipheader: int, skipfooter: int) -> dict:
    """MODEL number of each atom and ANISOU record that load_as_dataframe reads.

    Records outside any MODEL record are model 1, as in ModelCIFReader. The number
    is the first field after MODEL, or the count of MODEL records if it has none.

    Returns
    -------
    dict
        ``"ATOM"`` (ATOM and HETATM records) and ``"ANISOU"``: lists of model
        numbers, one per record in file order.
    """
    with open(filepath, "r") as f:
        lines = f.readlines()
    numbers = {"ATOM": [], "ANISOU": []}
    current, n_models = 1, 0
    for i, line in enumerate(lines[: len(lines) - skipfooter]):
        record = line[:6].strip()
        if record == "MODEL":
            n_models += 1
            fields = line[6:].split()
            current = int(fields[0]) if fields and fields[0].isdigit() else n_models
        elif i >= skipheader and record in ("ATOM", "HETATM", "ANISOU"):
            numbers["ANISOU" if record == "ANISOU" else "ATOM"].append(current)
    return numbers


def load_as_dataframe(
    filepath: str, skipheader: int = 0, skipfooter: int = 0
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
        Number of lines to skip at the end of the file. Default is 0; END, TER
        and MASTER records are dropped with every other non-atom record.

    Returns
    -------
    pd.DataFrame
        DataFrame whose columns include (among others, in no contractual
        order): ATOM, serial, name, altloc, resname, chainid, resseq, icode,
        x, y, z, occupancy, tempfactor, element, charge, model_num,
        anisou_flag, u11, u22, u33, u12, u13, u23, index. ``model_num`` is the
        MODEL record number, 1 throughout a file without MODEL records; the
        models of a multi-model file are concatenated in file order.
        DataFrame attributes include 'cell', 'spacegroup', and 'z'.

    Raises
    ------
    ValueError
        If an ATOM or HETATM record has a blank element field (columns 77-78).
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
    models = _model_numbers(filepath, skipheader, skipfooter)
    anisou = anisou.loc[anisou["ATOM"] == "ANISOU"].assign(model_num=models["ANISOU"])
    pdb = pdb.loc[(pdb["ATOM"] == "ATOM") | (pdb["ATOM"] == "HETATM")].assign(
        model_num=models["ATOM"]
    )

    anisou.drop(columns=["ATOM"], inplace=True)
    # Every model repeats the same atom identities, so ANISOU records are matched
    # within their own model.
    pdb = pdb.merge(
        anisou,
        on=[
            "serial",
            "name",
            "altloc",
            "resname",
            "chainid",
            "resseq",
            "element",
            "model_num",
        ],
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
    # Not guessed from the atom name: an unknown element scatters as Z = 0, so a
    # wrong guess, like a blank, would load without complaint.
    blank = pdb["element"].isna()
    if blank.any():
        first = pdb.loc[blank].head(5)
        atoms = ", ".join(
            f"{serial} {name} {resname}"
            for serial, name, resname in zip(
                first["serial"], first["name"], first["resname"]
            )
        )
        raise ValueError(
            f"{filepath}: {int(blank.sum())} atoms have a blank element field "
            f"(columns 77-78), starting with {atoms}. Add the element symbols "
            "first, e.g. with gemmi, pdbset or phenix.pdbtools."
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


#: Columns of the LINK-record table that ``Model.load`` reads off a reader's ``.links``.
#: Shared by the PDB and mmCIF readers so the topology builder sees one schema.
LINK_COLUMNS = (
    "name1", "altloc1", "resname1", "chainid1", "resseq1", "icode1",
    "name2", "altloc2", "resname2", "chainid2", "resseq2", "icode2",
    "length",
)


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

    df = pd.DataFrame(rows, columns=list(LINK_COLUMNS))
    if verbose > 0 and (len(df) or skipped_sym or skipped_bad):
        print(
            f"LINK records: parsed {len(df)}, "
            f"skipped {skipped_sym} symmetry-mate, {skipped_bad} malformed"
        )
    return df


_ATOM_COLUMNS = (
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
)

_U_COLUMNS = ("u11", "u22", "u33", "u12", "u13", "u23")


def _text(value) -> str:
    """``value`` as stripped text, with None, NaN and the string ``'nan'`` blank.

    The reader leaves a blank chain ID as NaN, which ``astype(str)`` downstream
    turns into ``'nan'``; both mean the field is empty.
    """
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text == "nan" else text


def _format_charge(charge) -> str:
    """Formal charge for columns 79-80: blank when neutral, else ``+1`` / ``-2``."""
    charge = 0 if pd.isna(charge) else int(charge)
    return f"{charge:+d}" if charge else ""


def _format_atom_identity(row) -> str:
    """Columns 7-27 of an ATOM, HETATM or ANISOU record: which atom it describes.

    wwPDB v3.3 layout: serial 7-11, atom name 13-16, altLoc 17, resName 18-20,
    chainID 22, resSeq 23-26, iCode 27. A two-character chain ID takes columns
    21-22, where gemmi reads and writes it.

    Parameters
    ----------
    row : mapping
        One atom; reads ``serial``, ``name``, ``element``, ``altloc``,
        ``resname``, ``chainid``, ``resseq`` and ``icode``.

    Returns
    -------
    str
        Exactly 21 characters for in-range values. A value wider than its field
        (serial > 99999, a 4-character residue name) is not truncated and shifts
        every later column.
    """
    name = _format_pdb_atom_name(row["name"], _text(row["element"]))
    return (
        f"{int(row['serial']):>5} {name}{_text(row['altloc']):1}"
        f"{_text(row['resname']):>3}{_text(row['chainid']):>2}"
        f"{int(row['resseq']):>4}{_text(row['icode']):1}"
    )


def _format_atom_records(row, anisou: bool) -> str:
    """The ATOM or HETATM record of one atom, then its ANISOU record if ``anisou``.

    Both records take columns 7-27 from :func:`_format_atom_identity`, so they
    cannot disagree about the atom. ANISOU holds round(U * 10^4) with U in Å².

    Parameters
    ----------
    row : mapping
        One atom with the columns :func:`write` requires, plus ``u11`` ...
        ``u23`` when ``anisou`` is true.
    anisou : bool
        Whether to append the ANISOU record.

    Returns
    -------
    str
        One or two newline-terminated 80-column records.
    """
    identity = _format_atom_identity(row)
    element_charge = f"{_text(row['element']):>2}{_format_charge(row['charge']):>2}"
    records = (
        f"{_text(row['ATOM']):<6}{identity}   "
        f"{row['x']:8.3f}{row['y']:8.3f}{row['z']:8.3f}"
        f"{row['occupancy']:6.2f}{row['tempfactor']:6.2f}"
        f"{'':10}{element_charge}\n"
    )
    if anisou:
        u = "".join(f"{round(float(row[c]) * 1e4):7d}" for c in _U_COLUMNS)
        records += f"ANISOU{identity} {u}{'':6}{element_charge}\n"
    return records


def _write_atom_records(handle, df: pd.DataFrame, anisou: bool) -> None:
    """Write one ATOM/HETATM record per row of ``df``, in row order.

    With ``anisou``, rows whose ``anisou_flag`` is set also get an ANISOU
    record. A row that cannot be formatted is skipped whole, with a printed
    warning, so one bad value costs one atom rather than the file.
    """
    anisou = anisou and "anisou_flag" in df.columns and bool(df["anisou_flag"].any())
    columns = list(_ATOM_COLUMNS)
    if anisou:
        columns += ["anisou_flag", *_U_COLUMNS]
    for i, row in enumerate(df[columns].to_dict("records")):
        try:
            records = _format_atom_records(row, anisou and bool(row["anisou_flag"]))
        except (TypeError, ValueError) as error:
            print(f"Skipping atom row {i}, which cannot be formatted: {error}")
            continue
        handle.write(records)


def write(df: pd.DataFrame, filepath: str, metadata=None) -> None:
    """
    Write a DataFrame to a PDB file.

    Parameters
    ----------
    df : pandas.DataFrame
        Atom table with columns ATOM, serial, name, altloc, resname, chainid,
        resseq, icode, x, y, z (Cartesian, Å), occupancy, tempfactor (Å²),
        element and charge. Rows whose optional ``anisou_flag`` is set also get
        an ANISOU record from ``u11`` ... ``u23`` (Å²).
    filepath : str
        Output PDB filename.
    metadata : RefinementMetadata, optional
        Metadata to render as PDB header (REMARK 3, TITLE, etc.).

    Raises
    ------
    KeyError
        If a required column is missing.

    Notes
    -----
    The CRYST1 record is sourced from the DataFrame attributes
    ``df.attrs["cell"]``, ``df.attrs["spacegroup"]`` and
    ``df.attrs.get("z")`` (not from columns). If any of these are missing,
    the file is written without a CRYST1 record and a warning is printed.

    Rows that fail to format are skipped with a printed warning; the
    remaining rows are still written. Nothing is renumbered: duplicated atom
    identifiers are written as they are (see
    :func:`torchref.utils.sanitize_pdb_dataframe`).
    """
    with open(filepath, "w") as n:
        # Write metadata header if provided (before CRYST1)
        if metadata is not None:
            n.write(metadata.render_pdb_header())

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

        _write_atom_records(n, df, anisou=True)
        n.write("END")


def write_multi_model(
    dataframes: List[pd.DataFrame],
    filepath: str,
    model_names: Optional[List[str]] = None,
) -> None:
    """
    Write multiple models to a single PDB file with MODEL/ENDMDL records.

    Each DataFrame is wrapped in a MODEL/ENDMDL pair, producing a
    multi-model PDB file suitable for ensemble or time-resolved data. Atom
    records are formatted as by :func:`write`, but without ANISOU records:
    each model's ADPs are its isotropic ``tempfactor``.

    Parameters
    ----------
    dataframes : list of pandas.DataFrame
        List of atom DataFrames, with the columns :func:`write` requires.
    filepath : str
        Output PDB filename.
    model_names : list of str, optional
        Names for each model (written as REMARK before each MODEL record).
        If None, models are numbered sequentially.

    Raises
    ------
    KeyError
        If a DataFrame lacks a required column.
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
            _write_atom_records(f, df, anisou=False)
            f.write("ENDMDL\n")

        f.write("END\n")
