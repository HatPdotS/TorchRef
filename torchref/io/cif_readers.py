"""
CIF readers, one per data type: :class:`CIFReader` (base, dict-like),
:class:`ReflectionCIFReader` (structure factors), :class:`ModelCIFReader`
(coordinates) and :class:`RestraintCIFReader` (monomer dictionaries).

Space groups always come back as plain ``str`` Hermann-Mauguin names
(``"P 1"``), validated against gemmi -- never as objects.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gemmi
import numpy as np
import pandas as pd

#: Column holding the ``data_`` block a loop row was read from. Added only when
#: ``parse_all_blocks`` is set, because that is the only mode in which rows from
#: different blocks share a category. Dictionaries whose loops carry no
#: ``comp_id`` still have to be split per compound, and the block name is the
#: only thing that can do it. Stripped again by
#: :meth:`RestraintCIFReader._filter_by_comp`, so it never reaches a caller.
_SOURCE_BLOCK_COLUMN = "_source_block"


class CIFReader:
    """
    A dictionary-like reader for CIF/mmCIF files.

    Loops are stored as pandas DataFrames.
    Other data is stored in a hierarchical dictionary structure.

    Parameters
    ----------
    filepath : str, optional
        Path to CIF file to load immediately.
    data_block : str, optional
        Specific data block name to read (e.g., 'r1vlmsf').
        If None and parse_all_blocks=False, reads the first data block.
        If None and parse_all_blocks=True, reads all data blocks.
    parse_all_blocks : bool, default False
        If True, parse all data blocks and merge them into a single
        dictionary (useful for restraint files). If False, parse only
        the specified block or the first block.

    Attributes
    ----------
    data : dict
        Dictionary storing parsed CIF data.
    filepath : Path or None
        Path to the loaded CIF file.
    available_blocks : list
        List of data block names found in the file.
    """

    def __init__(
        self,
        filepath: Optional[str] = None,
        data_block: Optional[str] = None,
        parse_all_blocks: bool = False,
    ):
        """See the class docstring for the parameters."""
        self.data = {}
        self.filepath = None
        self.data_block = data_block
        self.parse_all_blocks = parse_all_blocks
        self.available_blocks = []
        self.verbose = 0
        self._current_block = None

        if filepath:
            self.load(filepath)

    @classmethod
    def from_string(cls, content: str, **kwargs) -> "CIFReader":
        """Create CIFReader from string content instead of a file."""
        reader = cls(**kwargs)
        reader._parse(content)
        return reader

    def load(self, filepath: str):
        """Load and parse ``filepath``, replacing any previously loaded data."""
        self.filepath = Path(filepath)
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        self._parse(content)

    def _parse(self, content: str):
        """Parse the file text into ``self.data``.

        Which blocks: all merged when ``parse_all_blocks``, else the named
        ``data_block``, else only the first.
        """
        lines = content.split("\n")
        i = 0
        self._current_block = None
        target_block_found = False

        # First pass: find all data blocks
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("data_"):
                block_name = stripped[5:].strip()  # Remove 'data_' prefix
                self.available_blocks.append(block_name)

        # Determine parsing strategy
        if self.parse_all_blocks:
            # Parse all blocks - don't filter by block name
            parse_all = True
            if self.verbose > 0 and len(self.available_blocks) > 1:
                print(f"Parsing all {len(self.available_blocks)} data blocks")
        else:
            parse_all = False
            if self.data_block is None and self.available_blocks:
                # No specific block requested, use first one
                self.data_block = self.available_blocks[0]

            if self.verbose > 0 and len(self.available_blocks) > 1:
                print(f"Multiple data blocks found: {self.available_blocks}")
                print(f"Reading block: {self.data_block}")

        # Second pass: parse the target block(s)
        while i < len(lines):
            line = lines[i].strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                i += 1
                continue

            # Check for data block
            if line.startswith("data_"):
                block_name = line[5:].strip()  # Remove 'data_' prefix
                self._current_block = block_name

                if parse_all:
                    # Continue parsing - don't skip any blocks
                    i += 1
                    continue

                # Check if this is our target block
                if self.data_block and block_name == self.data_block:
                    target_block_found = True
                    i += 1
                    continue
                elif self.data_block and target_block_found:
                    # We've finished reading our target block, stop parsing
                    break
                else:
                    # Skip this block
                    i += 1
                    continue

            # Only parse if we're in parse_all mode OR in the target block
            if parse_all or (not self.data_block or target_block_found):
                # Check for loop
                if line.startswith("loop_"):
                    i = self._parse_loop(lines, i + 1)
                    continue

                # Parse single key-value pairs
                if line.startswith("_"):
                    i = self._parse_keyvalue(lines, i)
                    continue

            i += 1

        # Validate that we found the requested block (if not parsing all)
        if not parse_all and self.data_block and not target_block_found:
            raise ValueError(
                f"Data block '{self.data_block}' not found in CIF file.\n"
                f"Available blocks: {self.available_blocks}"
            )

    def _parse_loop(self, lines: List[str], start_idx: int) -> int:
        """Parse one ``loop_`` into a DataFrame; returns the next line index."""
        # Collect column names
        columns = []
        i = start_idx

        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith("#"):
                i += 1
                continue

            if line.startswith("_"):
                columns.append(line)
                i += 1
            else:
                break

        if not columns:
            return i

        # Collect data rows
        data_rows = []
        current_row = []
        in_multiline = False
        multiline_value = []

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Check if we've reached the end of the loop
            if not in_multiline and (
                not stripped
                or stripped.startswith("_")
                or stripped.startswith("loop_")
                or stripped.startswith("data_")
            ):
                if current_row:
                    data_rows.append(current_row)
                break

            # Handle multiline strings (starting with semicolon)
            if not in_multiline and line.startswith(";"):
                in_multiline = True
                multiline_value = [line[1:]]  # Remove leading semicolon
                i += 1
                continue

            if in_multiline:
                if line.startswith(";"):
                    # End of multiline string
                    in_multiline = False
                    current_row.append("\n".join(multiline_value))
                    multiline_value = []

                    # Check if row is complete
                    if len(current_row) == len(columns):
                        data_rows.append(current_row)
                        current_row = []
                else:
                    multiline_value.append(line)
                i += 1
                continue

            # Parse regular data line
            if stripped and not stripped.startswith("#"):
                tokens = self._tokenize_line(stripped)
                for token in tokens:
                    current_row.append(token)

                    # Check if row is complete
                    if len(current_row) == len(columns):
                        data_rows.append(current_row)
                        current_row = []

            i += 1

        # Create DataFrame
        if data_rows:
            df = pd.DataFrame(data_rows, columns=columns)

            # Store in hierarchical dictionary based on category
            # Extract category from first column name (e.g., _atom_site.id -> atom_site)
            if columns:
                category = self._extract_category(columns[0])
                if category:
                    self.data[category] = self._merge_category(category, df)

        return i

    def _merge_category(self, category: str, df: pd.DataFrame) -> pd.DataFrame:
        """Combine ``df`` with anything already stored under ``category``.

        Only relevant when ``parse_all_blocks`` is set: a multi-compound restraint
        dictionary repeats the same category (``chem_comp_bond``, ...) once per
        ``data_comp_*`` block, and replacing rather than appending would keep only
        the block that happens to come last. Rows are tagged with their source
        block so they can be split apart again even when the loop itself carries
        no ``comp_id`` column.

        Loops of one category with differing columns are aligned by name and
        NaN-filled; the surplus columns belong to other compounds and are removed
        by :meth:`RestraintCIFReader._filter_by_comp` before standardisation.
        """
        if not self.parse_all_blocks:
            return df

        df[_SOURCE_BLOCK_COLUMN] = self._current_block
        previous = self.data.get(category)
        if isinstance(previous, pd.DataFrame):
            return pd.concat([previous, df], ignore_index=True)
        return df

    def _parse_keyvalue(self, lines: List[str], start_idx: int) -> int:
        """Parse one key-value pair; returns the next line index."""
        line = lines[start_idx].strip()

        # Handle multiline values
        if start_idx + 1 < len(lines) and lines[start_idx + 1].startswith(";"):
            key = line
            value_lines = []
            i = start_idx + 2

            while i < len(lines):
                if lines[i].startswith(";"):
                    break
                value_lines.append(lines[i])
                i += 1

            value = "\n".join(value_lines)
            self._store_keyvalue(key, value)
            return i + 1

        # Handle single line key-value
        parts = line.split(None, 1)
        if len(parts) == 2:
            key, value = parts
            # Remove quotes if present
            if value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
            elif value.startswith('"') and value.endswith('"'):
                value = value[1:-1]

            self._store_keyvalue(key, value)

        return start_idx + 1

    def _tokenize_line(self, line: str) -> List[str]:
        """Split a data line into tokens, keeping quoted strings intact."""
        tokens = []
        current_token = []
        in_quotes = False
        quote_char = None

        i = 0
        while i < len(line):
            char = line[i]

            # Handle quotes
            if char in ('"', "'") and not in_quotes:
                in_quotes = True
                quote_char = char
                i += 1
                continue

            if char == quote_char and in_quotes:
                in_quotes = False
                quote_char = None
                if current_token:
                    tokens.append("".join(current_token))
                    current_token = []
                i += 1
                continue

            # Handle whitespace outside quotes
            if char.isspace() and not in_quotes:
                if current_token:
                    tokens.append("".join(current_token))
                    current_token = []
                i += 1
                continue

            current_token.append(char)
            i += 1

        if current_token:
            tokens.append("".join(current_token))

        return tokens

    def _extract_category(self, key: str) -> str:
        """Category of a CIF key: ``'_atom_site.id'`` -> ``'atom_site'``."""
        if key.startswith("_"):
            key = key[1:]

        if "." in key:
            return key.split(".")[0]

        return key

    def _store_keyvalue(self, key: str, value: str):
        """Store one key-value pair into the hierarchical ``self.data`` dict."""
        # Extract category and attribute
        if key.startswith("_"):
            key = key[1:]

        if "." in key:
            category, attribute = key.split(".", 1)

            if category not in self.data:
                self.data[category] = {}

            if isinstance(self.data[category], dict):
                self.data[category][attribute] = value
        else:
            self.data[key] = value

    def write(self, filepath: str):
        """Write the parsed data back out as CIF."""
        with open(filepath, "w") as f:
            f.write("data_structure\n")
            f.write("#\n")

            # Write single key-value pairs first
            for category, content in sorted(self.data.items()):
                if isinstance(content, dict):
                    for key, value in sorted(content.items()):
                        # Handle multiline values
                        if "\n" in str(value):
                            f.write(f"_{category}.{key}\n")
                            f.write(";\n")
                            f.write(str(value))
                            f.write("\n;\n")
                        else:
                            # Quote values with spaces
                            if " " in str(value):
                                f.write(f"_{category}.{key} '{value}'\n")
                            else:
                                f.write(f"_{category}.{key} {value}\n")
                    f.write("#\n")

            # Write loops (DataFrames)
            for category, content in sorted(self.data.items()):
                if isinstance(content, pd.DataFrame):
                    f.write("loop_\n")

                    # Write column names
                    for col in content.columns:
                        f.write(f"{col}\n")

                    # Write data rows
                    for _, row in content.iterrows():
                        row_values = []
                        for val in row:
                            val_str = str(val)
                            # Quote values with spaces or special characters
                            if " " in val_str or any(c in val_str for c in ['"', "'"]):
                                row_values.append(f"'{val_str}'")
                            else:
                                row_values.append(val_str)
                        f.write(" ".join(row_values) + "\n")

                    f.write("#\n")

    # Dictionary-like interface
    def __getitem__(self, key: str) -> Union[pd.DataFrame, Dict, Any]:
        """Get item by key."""
        return self.data[key]

    def __setitem__(self, key: str, value: Union[pd.DataFrame, Dict, Any]):
        """Set item by key."""
        self.data[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return key in self.data

    def __len__(self) -> int:
        """Return number of top-level categories."""
        return len(self.data)

    def keys(self):
        """Return dictionary keys."""
        return self.data.keys()

    def values(self):
        """Return dictionary values."""
        return self.data.values()

    def items(self):
        """Return dictionary items."""
        return self.data.items()

    def get(self, key: str, default=None):
        """Get item with default value."""
        return self.data.get(key, default)

    def __repr__(self) -> str:
        """String representation."""
        categories = list(self.keys())
        loops = [k for k, v in self.items() if isinstance(v, pd.DataFrame)]
        dicts = [k for k, v in self.items() if isinstance(v, dict)]

        return (
            f"CIFReader(categories={len(categories)}, "
            f"loops={len(loops)}, "
            f"key-value_groups={len(dicts)})"
        )

    def summary(self):
        """Print a summary of the CIF contents."""
        print(f"CIF File: {self.filepath}")
        print(f"Total categories: {len(self.data)}")
        print("\nLoops (DataFrames):")
        for key, value in sorted(self.items()):
            if isinstance(value, pd.DataFrame):
                print(f"  {key}: {len(value)} rows × {len(value.columns)} columns")

        print("\nKey-Value Groups (Dictionaries):")
        for key, value in sorted(self.items()):
            if isinstance(value, dict):
                print(f"  {key}: {len(value)} items")


class ReflectionCIFReader:
    """
    Reader for structure factor CIF files (e.g. ``*-sf.cif`` from the PDB):
    Miller indices, F/σF, I/σI, phases, figures of merit, R-free flags, cell and
    space group.

    Calling the instance gives the same unpack order as ``MTZReader.__call__``::

        data_dict, cell, spacegroup = ReflectionCIFReader('7JI4-sf.cif')()

    Mind the two naming schemes: the :meth:`get_reflection_data` DataFrame uses
    ``F_obs``/``sigma_F_obs`` and ``I_obs``/``sigma_I_obs``, while the standalone
    :meth:`get_amplitudes` / :meth:`get_intensities` dicts use ``'F'``/
    ``'sigma_F'`` and ``'I'``/``'sigma_I'``.
    """

    def __init__(
        self,
        filepath: str,
        verbose: int = 0,
        data_block: Optional[str] = None,
        anomalous: Optional[bool] = None,
    ):
        """
        Initialize and load structure factor CIF file.

        Parameters
        ----------
        filepath : str
            Path to structure factor CIF file.
        verbose : int, default 0
            Verbosity level (0=silent, 1=info, 2=debug).
        data_block : str, optional
            Specific data block name to read (e.g., 'r1vlmsf').
            If None, reads the first data block. Useful for files with
            multiple datasets.
        anomalous : bool, optional
            Anomalous (Bijvoet) handling. If None (default), ``pdbx_F_plus/minus``
            (or ``I``) columns are auto-detected and unstacked into explicit
            Friedel pairs when present (anomalous preferred). True forces this;
            False forces a merged load (Bijvoet mates averaged) even when present.
        """
        self.filepath = Path(filepath)
        self.verbose = verbose
        self.anomalous = anomalous
        self.cif_reader = CIFReader(filepath, data_block=data_block)
        self.cif_reader.verbose = verbose
        self._validate()
        self._extract_data()

    def _validate(self):
        """Validate that this is a structure factor CIF file."""
        if "refln" not in self.cif_reader:
            error_msg = (
                f"File {self.filepath} does not contain reflection data (_refln loop) "
                f"in the selected data block '{self.cif_reader.data_block}'.\n"
                f"Available data blocks in file: {self.cif_reader.available_blocks}\n"
                f"Available categories in selected block: {list(self.cif_reader.keys())}\n\n"
            )

            if len(self.cif_reader.available_blocks) > 1:
                error_msg += (
                    f"This file contains multiple data blocks. Try specifying a different block:\n"
                    f"  reader = ReflectionCIFReader('{self.filepath}', data_block='BLOCKNAME')\n"
                    f"where BLOCKNAME is one of: {self.cif_reader.available_blocks}"
                )

            raise ValueError(error_msg)

    def _extract_data(self):
        """Extract data in legacy MTZ-compatible format."""
        self.data = {}

        # Extract reflection data
        refln_df = self.get_reflection_data()

        # Store combined HKL array (like MTZ reader) - this is the primary format
        hkl = np.column_stack(
            [
                refln_df["h"].to_numpy(),
                refln_df["k"].to_numpy(),
                refln_df["l"].to_numpy(),
            ]
        ).astype(np.int32)
        self.data["HKL"] = hkl
        self.data["HKL_key"] = refln_df["hkl_key"]
        # Friedel merge state (set by get_reflection_data); False when F(+)/F(-) were
        # expanded into explicit signed-HKL Bijvoet pairs.
        self.data["friedel_merged"] = getattr(self, "_friedel_merged", True)

        # Store amplitudes if available (standardized keys matching MTZ reader)
        if refln_df["F_obs"].notna().any():
            self.data["F"] = refln_df["F_obs"].to_numpy().astype(np.float32)
            self.data["F_col"] = refln_df["F_obs_key"]

        if refln_df["sigma_F_obs"].notna().any():
            self.data["SIGF"] = refln_df["sigma_F_obs"].to_numpy().astype(np.float32)
            self.data["SIGF_col"] = refln_df["sigma_F_obs_key"]

        # Store intensities if available (standardized keys matching MTZ reader)
        if refln_df["I_obs"].notna().any():
            self.data["I"] = refln_df["I_obs"].to_numpy().astype(np.float32)
            self.data["I_col"] = refln_df["I_obs_key"]

        if refln_df["sigma_I_obs"].notna().any():
            self.data["SIGI"] = refln_df["sigma_I_obs"].to_numpy().astype(np.float32)
            self.data["SIGI_col"] = refln_df["sigma_I_obs_key"]

        # A structure-factor CIF must carry observed amplitudes or intensities.
        # Calculated columns (e.g. _refln.F_calc) are intentionally not used as
        # observations, so a calc-only file lands here with neither F nor I.
        if "F" not in self.data and "I" not in self.data:
            raise ValueError(
                f"No observed amplitudes or intensities found in {self.filepath} "
                f"(data block '{self.cif_reader.data_block}'). The reflection loop "
                f"has no measured F/I column; calculated columns such as "
                f"_refln.F_calc are not used as observations."
            )

        # Store R-free flags if available (standardized keys matching MTZ reader)
        if refln_df["free_flag"].notna().any():
            rfree_characters = (
                refln_df["free_flag"].str.lower().map({"f": 0, "x": -1, "o": 1})
            )
            percentage_work = (
                (rfree_characters == 1).sum() / len(rfree_characters) * 100.0
            )
            percentage_test = (
                (rfree_characters == 0).sum() / len(rfree_characters) * 100.0
            )
            if percentage_work < 0.9:
                if self.verbose > 0:
                    print(
                        f"WARNING: R-free flags indicate only {percentage_work:.2f}% work reflections. Skipping R-free flags. >90% expected. Generating new Rfree flags"
                    )
                self.data["R-free-source"] = "None"
            elif percentage_test < 0.01:
                if self.verbose > 0:
                    print(
                        f"WARNING: R-free flags indicate only {percentage_test:.2f}% test reflections. Skipping R-free flags. >1% expected. Generating new Rfree flags"
                    )
                self.data["R-free-source"] = "None"
            else:
                self.data["R-free-flags"] = rfree_characters.to_numpy().astype(np.int32)
                self.data["R-free-source"] = refln_df["free_flag_key"]

        # Extract cell and spacegroup
        self.cell = self.get_cell_parameters()
        if self.cell is None:
            raise ValueError(
                f"Unit cell parameters not found in CIF file: {self.filepath}"
            )
        else:
            self.cell = np.array(self.cell)

        self.spacegroup = self.get_space_group()

        if self.verbose > 1:
            print(f"Loaded CIF file: {self.filepath}")
            print(f"  Reflections: {len(refln_df)}")
            print(f"  Has F: {'F' in self.data}")
            print(f"  Has I: {'I' in self.data}")
            print(f"  Has R-free: {'R-free-flags' in self.data}")
            print(f"  Cell: {self.cell}")
            print(f"  Spacegroup: {self.spacegroup}")

    def read(self, filepath: str = None):
        """Re-read ``filepath`` (default: the init path); returns ``self``."""
        if filepath is not None:
            self.__init__(filepath, verbose=self.verbose)
        return self

    def __call__(self) -> Tuple[Dict[str, np.ndarray], np.ndarray, str]:
        """
        Get data in legacy MTZ-compatible format.

        Returns
        -------
        data : dict
            Dictionary with extracted data arrays:
            - 'HKL': Nx3 int32 array of Miller indices (plus 'HKL_key')
            - 'F', 'SIGF': Amplitudes and sigmas (if available)
            - 'I', 'SIGI': Intensities and sigmas (if available)
            - 'R-free-flags': R-free test set flags (if available)
        cell : numpy.ndarray
            Cell parameters [a, b, c, alpha, beta, gamma].
        spacegroup : str
            Space group Hermann-Mauguin name (e.g. "P 1").
        """
        try:
            return self.data, self.cell, self.spacegroup
        except AttributeError as e:
            raise ValueError(
                "Data not loaded. Call read() first or provide filepath in __init__"
            ) from e

    def get_reflection_data(self) -> pd.DataFrame:
        """
        Extract reflection data with standardized column names.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns:
            - h, k, l: Miller indices
            - F_obs, sigma_F_obs: Observed amplitudes (if available)
            - I_obs, sigma_I_obs: Observed intensities (if available)
            - phase, fom: Phase and figure of merit (if available)
            - free_flag: R-free flags (if available)

        Notes
        -----
        Missing columns will be filled with NaN or appropriate defaults.
        """
        refln_df = self.cif_reader["refln"].copy()

        # Standardize column names
        result = pd.DataFrame()

        # Friedel merge state: set False below if anomalous F(+)/F(-) (or I(+)/I(-))
        # are detected, in which case rows are expanded into signed-HKL Bijvoet pairs.
        self._friedel_merged = True

        # Miller indices (required)
        result["h"], hkey = self._extract_numeric(
            refln_df, ["_refln.index_h", "_refln.h"], required=True, target_type="int"
        )
        result["k"], kkey = self._extract_numeric(
            refln_df, ["_refln.index_k", "_refln.k"], required=True, target_type="int"
        )
        result["l"], lkey = self._extract_numeric(
            refln_df, ["_refln.index_l", "_refln.l"], required=True, target_type="int"
        )
        result["hkl_key"] = f"{hkey},{kkey},{lkey}"

        # Structure factors - check for anomalous data first
        F_plus_col = (
            "_refln.pdbx_F_plus" if "_refln.pdbx_F_plus" in refln_df.columns else None
        )
        F_minus_col = (
            "_refln.pdbx_F_minus" if "_refln.pdbx_F_minus" in refln_df.columns else None
        )
        sigF_plus_col = (
            "_refln.pdbx_F_plus_sigma"
            if "_refln.pdbx_F_plus_sigma" in refln_df.columns
            else None
        )
        sigF_minus_col = (
            "_refln.pdbx_F_minus_sigma"
            if "_refln.pdbx_F_minus_sigma" in refln_df.columns
            else None
        )

        if F_plus_col and F_minus_col:
            # Keep F(+)/F(-) separate for expansion into signed-HKL Bijvoet pairs
            # (see _expand_friedel_pairs). Placeholder F_obs is filled there.
            self._friedel_merged = False
            result["_F_plus"] = pd.to_numeric(
                refln_df[F_plus_col].replace(["?", "."], np.nan), errors="coerce"
            )
            result["_F_minus"] = pd.to_numeric(
                refln_df[F_minus_col].replace(["?", "."], np.nan), errors="coerce"
            )
            result["F_obs"] = np.nan
            result["F_obs_key"] = f"{F_plus_col}/{F_minus_col}_unstacked"

            if sigF_plus_col and sigF_minus_col:
                result["_sigF_plus"] = pd.to_numeric(
                    refln_df[sigF_plus_col].replace(["?", "."], np.nan), errors="coerce"
                )
                result["_sigF_minus"] = pd.to_numeric(
                    refln_df[sigF_minus_col].replace(["?", "."], np.nan),
                    errors="coerce",
                )
                result["sigma_F_obs"] = np.nan
                result["sigma_F_obs_key"] = (
                    f"{sigF_plus_col}/{sigF_minus_col}_unstacked"
                )
            else:
                sigF, sigma_F_obs_key = self._extract_numeric(
                    refln_df,
                    [
                        "_refln.F_meas_sigma_au",
                        "_refln.F_meas_sigma",
                        "_refln.F_squared_sigma",
                        "_refln.SIGF-obs",
                    ],
                    target_type="float",
                )
                # Same sigma for both mates when per-mate sigmas are unavailable.
                result["_sigF_plus"] = sigF
                result["_sigF_minus"] = sigF
                result["sigma_F_obs"] = np.nan
                result["sigma_F_obs_key"] = sigma_F_obs_key

            if self.verbose > 0:
                F_plus, F_minus = result["_F_plus"], result["_F_minus"]
                n_both = ((pd.notna(F_plus)) & (pd.notna(F_minus))).sum()
                n_plus_only = ((pd.notna(F_plus)) & (pd.isna(F_minus))).sum()
                n_minus_only = ((pd.isna(F_plus)) & (pd.notna(F_minus))).sum()
                print("Anomalous data detected: unstacking F+ and F- into pairs")
                print(f"  Reflections with both F+/F-: {n_both}")
                print(f"  Reflections with F+ only: {n_plus_only}")
                print(f"  Reflections with F- only: {n_minus_only}")
        else:
            # Standard non-anomalous data
            # NOTE: do NOT include calculated columns (e.g. _refln.F_calc) here.
            # Treating calc amplitudes as observations gives meaningless
            # R-factors / refinement against the model's own F_calc.
            result["F_obs"], F_obs_key = self._extract_numeric(
                refln_df,
                [
                    "_refln.F_meas_au",
                    "_refln.F_meas",
                    "_refln.pdbx_F_plus",
                    "_refln.F-obs",
                    "_refln.F_squared_meas",
                ],
                target_type="float",
            )
            result["F_obs_key"] = F_obs_key
            result["sigma_F_obs"], sigma_F_obs_key = self._extract_numeric(
                refln_df,
                [
                    "_refln.F_meas_sigma_au",
                    "_refln.F_meas_sigma",
                    "_refln.F_squared_sigma",
                    "_refln.SIGF-obs",
                ],
                target_type="float",
            )
            result["sigma_F_obs_key"] = sigma_F_obs_key

        # Intensities - check for anomalous intensities
        I_plus_col = (
            "_refln.pdbx_I_plus" if "_refln.pdbx_I_plus" in refln_df.columns else None
        )
        I_minus_col = (
            "_refln.pdbx_I_minus" if "_refln.pdbx_I_minus" in refln_df.columns else None
        )
        sigI_plus_col = (
            "_refln.pdbx_I_plus_sigma"
            if "_refln.pdbx_I_plus_sigma" in refln_df.columns
            else None
        )
        sigI_minus_col = (
            "_refln.pdbx_I_minus_sigma"
            if "_refln.pdbx_I_minus_sigma" in refln_df.columns
            else None
        )

        if I_plus_col and I_minus_col:
            # Keep I(+)/I(-) separate for expansion into signed-HKL Bijvoet pairs.
            self._friedel_merged = False
            result["_I_plus"] = pd.to_numeric(
                refln_df[I_plus_col].replace(["?", "."], np.nan), errors="coerce"
            )
            result["_I_minus"] = pd.to_numeric(
                refln_df[I_minus_col].replace(["?", "."], np.nan), errors="coerce"
            )
            result["I_obs"] = np.nan
            result["I_obs_key"] = f"{I_plus_col}/{I_minus_col}_unstacked"

            if sigI_plus_col and sigI_minus_col:
                result["_sigI_plus"] = pd.to_numeric(
                    refln_df[sigI_plus_col].replace(["?", "."], np.nan), errors="coerce"
                )
                result["_sigI_minus"] = pd.to_numeric(
                    refln_df[sigI_minus_col].replace(["?", "."], np.nan),
                    errors="coerce",
                )
                result["sigma_I_obs"] = np.nan
                result["sigma_I_obs_key"] = (
                    f"{sigI_plus_col}/{sigI_minus_col}_unstacked"
                )
            else:
                sigI, sigIobskey = self._extract_numeric(
                    refln_df,
                    [
                        "_refln.intensity_sigma",
                        "_refln.I_sigma",
                        "_refln.SIGI-obs",
                        "_refln.pdbx_I_sigma",
                    ],
                    target_type="float",
                )
                result["_sigI_plus"] = sigI
                result["_sigI_minus"] = sigI
                result["sigma_I_obs"] = np.nan
                result["sigma_I_obs_key"] = sigIobskey

            if self.verbose > 0:
                I_plus, I_minus = result["_I_plus"], result["_I_minus"]
                n_both = ((pd.notna(I_plus)) & (pd.notna(I_minus))).sum()
                n_plus_only = ((pd.notna(I_plus)) & (pd.isna(I_minus))).sum()
                n_minus_only = ((pd.isna(I_plus)) & (pd.notna(I_minus))).sum()
                print(
                    "Anomalous intensity data detected: unstacking I+ and I- into pairs"
                )
                print(f"  Reflections with both I+/I-: {n_both}")
                print(f"  Reflections with I+ only: {n_plus_only}")
                print(f"  Reflections with I- only: {n_minus_only}")
        else:
            # Standard non-anomalous intensities
            result["I_obs"], Iobskey = self._extract_numeric(
                refln_df,
                [
                    "_refln.intensity_meas",
                    "_refln.I_meas",
                    "_refln.pdbx_I_plus",
                    "_refln.I-obs",
                    "_refln.pdbx_I",
                ],
                target_type="float",
            )
            result["I_obs_key"] = Iobskey
            result["sigma_I_obs"], sigIobskey = self._extract_numeric(
                refln_df,
                [
                    "_refln.intensity_sigma",
                    "_refln.I_sigma",
                    "_refln.pdbx_I_plus_sigma",
                    "_refln.SIGI-obs",
                    "_refln.pdbx_I_sigma",
                ],
                target_type="float",
            )
            result["sigma_I_obs_key"] = sigIobskey

        # Phase information
        result["phase"], phase_key = self._extract_numeric(
            refln_df,
            ["_refln.phase_meas", "_refln.phase_calc", "_refln.pdbx_PHIB"],
            target_type="float",
        )

        result["phase_key"] = phase_key
        result["fom"], fom_key = self._extract_numeric(
            refln_df, ["_refln.fom", "_refln.pdbx_FOM"], target_type="float"
        )
        result["fom_key"] = fom_key

        # R-free flags
        result["free_flag"], free_flag_key = self._extract_numeric(
            refln_df,
            ["_refln.status", "_refln.pdbx_r_free_flag", "_refln.free_flag"],
            target_type="None",
        )
        result["free_flag_key"] = free_flag_key

        if not self._friedel_merged:
            # Anomalous columns were detected. Honor the caller's preference:
            # anomalous=False forces a merged (averaged) load; otherwise unstack.
            if self.anomalous is False:
                result = self._merge_friedel_pairs(result)
                self._friedel_merged = True
            else:
                try:
                    result = self._expand_friedel_pairs(result)
                except Exception as e:  # fall back to merged on any failure
                    if self.verbose > 0:
                        print(
                            f"Anomalous unstacking skipped ({e}); "
                            "averaging F+/F- instead."
                        )
                    result = self._merge_friedel_pairs(result)
                    self._friedel_merged = True
        elif self.anomalous is True and self.verbose > 0:
            print(
                "anomalous=True requested but no F(+)/F(-) columns found; "
                "loading merged."
            )

        return result

    def _expand_friedel_pairs(self, result: pd.DataFrame) -> pd.DataFrame:
        """Expand F(+)/F(-) (or I(+)/I(-)) into explicit signed-HKL Bijvoet rows.

        The Friedel-plus members keep their Miller index; the Friedel-minus members
        get the negated index. Centric reflections are not split (their mates are
        equivalent), and rows whose chosen observation is missing are dropped. All
        other per-reflection columns (phase, fom, free_flag, ...) are duplicated to
        both mates. Temporary ``_F_plus``/``_F_minus``/... columns are removed.
        """
        hkl = result[["h", "k", "l"]].to_numpy().astype(np.int32)
        gops = gemmi.SpaceGroup(self.get_space_group()).operations()
        centric = np.asarray(gops.centric_flag_array(hkl), dtype=bool)

        has_F = "_F_plus" in result.columns
        has_I = "_I_plus" in result.columns

        def member(df: pd.DataFrame, sign: str) -> pd.DataFrame:
            out = df.copy()
            if has_F:
                out["F_obs"] = df[f"_F_{sign}"]
                out["sigma_F_obs"] = df[f"_sigF_{sign}"]
            if has_I:
                out["I_obs"] = df[f"_I_{sign}"]
                out["sigma_I_obs"] = df[f"_sigI_{sign}"]
            # Keep only rows where the primary observation is present.
            present = pd.Series(False, index=out.index)
            if has_I:
                present |= out["I_obs"].notna()
            if has_F:
                present |= out["F_obs"].notna()
            return out[present]

        plus = member(result, "plus")

        minus = result.copy()
        minus[["h", "k", "l"]] = -minus[["h", "k", "l"]].to_numpy()
        minus = minus[~centric]  # centric mates are equivalent; do not duplicate
        minus = member(minus, "minus")

        combined = pd.concat([plus, minus], ignore_index=True)
        temp_cols = [
            c
            for c in (
                "_F_plus",
                "_F_minus",
                "_sigF_plus",
                "_sigF_minus",
                "_I_plus",
                "_I_minus",
                "_sigI_plus",
                "_sigI_minus",
            )
            if c in combined.columns
        ]
        return combined.drop(columns=temp_cols)

    def _merge_friedel_pairs(self, result: pd.DataFrame) -> pd.DataFrame:
        """Fallback: average F(+)/F(-) (and I(+)/I(-)) into merged single rows."""

        def avg(a: str, b: str) -> pd.Series:
            return result[a].combine(
                result[b],
                lambda x, y: (
                    (x + y) / 2
                    if pd.notna(x) and pd.notna(y)
                    else (x if pd.notna(x) else y)
                ),
                fill_value=np.nan,
            )

        out = result.copy()
        if "_F_plus" in out.columns:
            out["F_obs"] = avg("_F_plus", "_F_minus")
            out["sigma_F_obs"] = (
                np.sqrt((out["_sigF_plus"] ** 2 + out["_sigF_minus"] ** 2) / 4)
                .combine_first(out["_sigF_plus"])
                .combine_first(out["_sigF_minus"])
            )
        if "_I_plus" in out.columns:
            out["I_obs"] = avg("_I_plus", "_I_minus")
            out["sigma_I_obs"] = (
                np.sqrt((out["_sigI_plus"] ** 2 + out["_sigI_minus"] ** 2) / 4)
                .combine_first(out["_sigI_plus"])
                .combine_first(out["_sigI_minus"])
            )
        temp_cols = [
            c
            for c in (
                "_F_plus",
                "_F_minus",
                "_sigF_plus",
                "_sigF_minus",
                "_I_plus",
                "_I_minus",
                "_sigI_plus",
                "_sigI_minus",
            )
            if c in out.columns
        ]
        return out.drop(columns=temp_cols)

    def _extract_numeric(
        self,
        df: pd.DataFrame,
        possible_cols: List[str],
        required: bool = False,
        target_type: str = "float",
    ) -> pd.Series:
        """First of ``possible_cols`` present in ``df``, coerced to ``target_type``.

        Returns ``(series, column_name)``, or an all-NaN series paired with the
        string ``'None'`` when nothing matched and ``required`` is False.
        """
        for col in possible_cols:
            if col in df.columns:
                try:
                    # Handle '?' as missing data
                    data = df[col].replace(["?", "."], np.nan)
                    if target_type == "int":
                        return (
                            pd.to_numeric(data, errors="coerce").fillna(0).astype(int),
                            col,
                        )
                    elif target_type == "float":
                        return pd.to_numeric(data, errors="coerce"), col
                    elif target_type == "None":
                        return data.astype(str), col
                    else:
                        print(f"Unknown target_type: {target_type}")
                except Exception:
                    continue

        if required:
            raise ValueError(
                f"Required column not found. Tried: {possible_cols}\n"
                f"Available columns: {list(df.columns)}"
            )

        # Return NaN series
        return pd.Series([np.nan] * len(df)), "None"

    def has_miller_indices(self) -> bool:
        """Check if file contains Miller indices."""
        if "refln" not in self.cif_reader:
            return False
        df = self.cif_reader["refln"]
        h_cols = ["_refln.index_h", "_refln.h"]
        return any(col in df.columns for col in h_cols)

    def has_amplitudes(self) -> bool:
        """Check if file contains structure factor amplitudes.

        Notes
        -----
        This counts *calculated* amplitudes (``_refln.F_calc``) as well as
        observed ones, so it is not equivalent to "has observed amplitudes".
        A calc-only file returns ``True`` here but is still rejected by the
        loader (``_extract_data``), which uses only measured F/I as
        observations.
        """
        if "refln" not in self.cif_reader:
            return False
        df = self.cif_reader["refln"]
        f_cols = [
            "_refln.F_meas_au",
            "_refln.F_meas",
            "_refln.pdbx_F_plus",
            "_refln.F_calc",
            "_refln.F-obs",
        ]
        return any(col in df.columns for col in f_cols)

    def has_intensities(self) -> bool:
        """Check if file contains intensity measurements."""
        if "refln" not in self.cif_reader:
            return False
        df = self.cif_reader["refln"]
        i_cols = [
            "_refln.intensity_meas",
            "_refln.I_meas",
            "_refln.pdbx_I_plus",
            "_refln.I-obs",
            "_refln.pdbx_I",
        ]
        return any(col in df.columns for col in i_cols)

    def has_phases(self) -> bool:
        """Check if file contains phase information."""
        if "refln" not in self.cif_reader:
            return False
        df = self.cif_reader["refln"]
        phase_cols = ["_refln.phase_meas", "_refln.phase_calc", "_refln.pdbx_PHIB"]
        return any(col in df.columns for col in phase_cols)

    def has_rfree_flags(self) -> bool:
        """Check if file contains R-free flags."""
        if "refln" not in self.cif_reader:
            return False
        df = self.cif_reader["refln"]
        flag_cols = ["_refln.status", "_refln.pdbx_r_free_flag", "_refln.free_flag"]
        return any(col in df.columns for col in flag_cols)

    def get_miller_indices(self) -> Optional[np.ndarray]:
        """Miller indices as an (N, 3) array, or None if absent."""
        data = self.get_reflection_data()
        if data is None or "h" not in data.columns:
            return None
        return data[["h", "k", "l"]].values

    def get_amplitudes(self) -> Optional[Dict[str, np.ndarray]]:
        """``{'F': ..., 'sigma_F': ...}``, or None if absent."""
        data = self.get_reflection_data()
        if data is None or "F_obs" not in data.columns:
            return None
        if data["F_obs"].isna().all():
            return None
        return {"F": data["F_obs"].values, "sigma_F": data["sigma_F_obs"].values}

    def get_intensities(self) -> Optional[Dict[str, np.ndarray]]:
        """``{'I': ..., 'sigma_I': ...}``, or None if absent."""
        data = self.get_reflection_data()
        if data is None or "I_obs" not in data.columns:
            return None
        if data["I_obs"].isna().all():
            return None
        return {"I": data["I_obs"].values, "sigma_I": data["sigma_I_obs"].values}

    def get_cell_parameters(self) -> Optional[List[float]]:
        """Unit cell ``[a, b, c, alpha, beta, gamma]`` as 6 floats, or None."""
        if "cell" not in self.cif_reader:
            return None

        cell_data = self.cif_reader["cell"]
        try:
            a = float(self._get_value(cell_data, ["_cell.length_a", "length_a"], "1.0"))
            b = float(self._get_value(cell_data, ["_cell.length_b", "length_b"], "1.0"))
            c = float(self._get_value(cell_data, ["_cell.length_c", "length_c"], "1.0"))
            alpha = float(
                self._get_value(cell_data, ["_cell.angle_alpha", "angle_alpha"], "90.0")
            )
            beta = float(
                self._get_value(cell_data, ["_cell.angle_beta", "angle_beta"], "90.0")
            )
            gamma = float(
                self._get_value(cell_data, ["_cell.angle_gamma", "angle_gamma"], "90.0")
            )
            return [a, b, c, alpha, beta, gamma]
        except Exception:
            return None

    def get_space_group(self) -> str:
        """Hermann-Mauguin space group name, falling back to ``"P 1"``."""
        sg_name = "P 1"
        if "symmetry" in self.cif_reader:
            sym_data = self.cif_reader["symmetry"]
            sg_name = self._get_value(
                sym_data,
                [
                    "_symmetry.space_group_name_H-M",
                    "space_group_name_H-M",
                    "_space_group.name_H-M_alt",
                ],
                "P 1",
            )

        # Validate the name by trying to parse it
        try:
            gemmi.SpaceGroup(sg_name)
            return sg_name
        except Exception:
            try:
                gemmi.SpaceGroup(sg_name.replace(" ", ""))
                return sg_name.replace(" ", "")
            except Exception:
                return "P 1"

    def _get_value(self, data, possible_keys: List[str], default: Any = None) -> Any:
        """Get value from DataFrame or dict, trying multiple keys."""
        if isinstance(data, pd.DataFrame):
            for key in possible_keys:
                if key in data.columns and len(data) > 0:
                    return data[key].iloc[0]
        elif isinstance(data, dict):
            for key in possible_keys:
                if key in data:
                    return data[key]
        return default


class ModelCIFReader:
    """
    Reader for model/structure CIF files (e.g. ``*.cif`` from the PDB):
    coordinates, altlocs, ANISOU, cell and space group.

    Calling the instance gives the same unpack order as the PDB reader::

        dataframe, cell, spacegroup = ModelCIFReader('3E98.cif')()

    Multi-model files come out concatenated from :meth:`get_atom_data`; use
    :meth:`get_atom_data_by_model` to keep them apart.
    """

    def __init__(self, filepath: str, verbose: int = 0):
        """Load ``filepath`` immediately; ``verbose`` is 0=silent, 1=info, 2=debug."""
        self.filepath = Path(filepath)
        self.verbose = verbose
        self.cif = CIFReader(filepath)
        self._validate()
        self._extract_data()

    def _validate(self):
        """Validate that this is a model CIF file."""
        if "atom_site" not in self.cif.data:
            raise ValueError(
                f"File {self.filepath} does not contain atomic coordinate data (_atom_site loop).\n"
                f"This does not appear to be a model CIF file.\n"
                f"Available data blocks: {list(self.cif.data.keys())}"
            )

    def _extract_data(self):
        """Extract data in legacy PDB-compatible format."""
        # Get atom data as DataFrame
        self.dataframe = self.get_atom_data()

        # Extract cell and spacegroup
        cell_params = self.get_cell_parameters()
        if cell_params is None:
            self.cell = None
        else:
            self.cell = cell_params

        self.spacegroup = self.get_space_group()

        # Store as DataFrame attributes (like legacy PDB reader)
        self.dataframe.attrs["cell"] = self.cell
        self.dataframe.attrs["spacegroup"] = self.spacegroup
        self.dataframe.attrs["z"] = None  # CIF files typically don't have Z value

        if self.verbose > 1:
            print(f"Loaded CIF model file: {self.filepath}")
            print(f"  Atoms: {len(self.dataframe)}")
            print(f"  Cell: {self.cell}")
            print(f"  Spacegroup: {self.spacegroup}")

    def read(self, filepath: str = None):
        """Re-read ``filepath`` (default: the init path); returns ``self``."""
        if filepath is not None:
            self.__init__(filepath, verbose=self.verbose)
        return self

    def __call__(self) -> Tuple[pd.DataFrame, List[float], str]:
        """
        Get data in legacy PDB-compatible format.

        Returns
        -------
        dataframe : pandas.DataFrame
            Atom data with columns: ATOM, serial, name, altloc, resname, chainid,
            resseq, icode, x, y, z, occupancy, tempfactor, element, charge,
            anisou_flag, u11, u22, u33, u12, u13, u23.
        cell : list
            Cell parameters [a, b, c, alpha, beta, gamma].
        spacegroup : str
            Space group Hermann-Mauguin name (e.g. "P 1").
        """
        try:
            return self.dataframe, self.cell, self.spacegroup
        except AttributeError as e:
            raise ValueError(
                "Data not loaded. Call read() first or provide filepath in __init__"
            ) from e

    def get_atom_data(self) -> pd.DataFrame:
        """
        Extract atomic coordinate data in PDB-compatible format.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns matching PDB format:
            - ATOM, serial, name, altloc, resname, chainid, resseq, icode
            - x, y, z, occupancy, tempfactor
            - element, charge
            - anisou_flag, u11, u22, u33, u12, u13, u23
        """
        atom_df = self.cif.data["atom_site"].copy()
        result = pd.DataFrame()

        # Record type (ATOM or HETATM)
        result["ATOM"] = self._extract_string(
            atom_df, ["_atom_site.group_PDB"], default="ATOM"
        )

        # Serial number
        result["serial"] = self._extract_int(
            atom_df, ["_atom_site.id"], default_range=True
        )

        # Atom identification
        result["name"] = self._extract_string(
            atom_df,
            ["_atom_site.label_atom_id", "_atom_site.auth_atom_id"],
            required=True,
        )

        result["altloc"] = self._extract_string(
            atom_df, ["_atom_site.label_alt_id"], default="", replace_dot=True
        )

        result["resname"] = self._extract_string(
            atom_df,
            ["_atom_site.label_comp_id", "_atom_site.auth_comp_id"],
            required=True,
        )

        result["chainid"] = self._extract_string(
            atom_df,
            ["_atom_site.auth_asym_id", "_atom_site.label_asym_id"],
            default="",
            replace_dot=True,
        )

        result["resseq"] = self._extract_int(
            atom_df, ["_atom_site.auth_seq_id", "_atom_site.label_seq_id"], default=0
        )

        result["icode"] = self._extract_string(
            atom_df, ["_atom_site.pdbx_PDB_ins_code"], default="", replace_dot=True
        )

        # Coordinates
        result["x"] = self._extract_float(
            atom_df, ["_atom_site.Cartn_x"], required=True
        )
        result["y"] = self._extract_float(
            atom_df, ["_atom_site.Cartn_y"], required=True
        )
        result["z"] = self._extract_float(
            atom_df, ["_atom_site.Cartn_z"], required=True
        )

        # Properties
        result["occupancy"] = self._extract_float(
            atom_df, ["_atom_site.occupancy"], default=1.0
        )
        result["tempfactor"] = self._extract_float(
            atom_df, ["_atom_site.B_iso_or_equiv"], default=20.0
        )

        # Element and charge
        result["element"] = self._extract_string(
            atom_df, ["_atom_site.type_symbol"], required=True
        )
        result["charge"] = self._extract_int(
            atom_df, ["_atom_site.pdbx_formal_charge"], default=0
        )

        # Model number (for multi-model structures / ensembles)
        result["model_num"] = self._extract_int(
            atom_df, ["_atom_site.pdbx_PDB_model_num"], default=1
        )

        # Anisotropic displacement parameters
        aniso_cols = [
            "_atom_site.aniso_U[1][1]",
            "_atom_site.aniso_U[2][2]",
            "_atom_site.aniso_U[3][3]",
            "_atom_site.aniso_U[1][2]",
            "_atom_site.aniso_U[1][3]",
            "_atom_site.aniso_U[2][3]",
        ]

        if all(col in atom_df.columns for col in aniso_cols):
            result["u11"] = pd.to_numeric(
                atom_df["_atom_site.aniso_U[1][1]"], errors="coerce"
            )
            result["u22"] = pd.to_numeric(
                atom_df["_atom_site.aniso_U[2][2]"], errors="coerce"
            )
            result["u33"] = pd.to_numeric(
                atom_df["_atom_site.aniso_U[3][3]"], errors="coerce"
            )
            result["u12"] = pd.to_numeric(
                atom_df["_atom_site.aniso_U[1][2]"], errors="coerce"
            )
            result["u13"] = pd.to_numeric(
                atom_df["_atom_site.aniso_U[1][3]"], errors="coerce"
            )
            result["u23"] = pd.to_numeric(
                atom_df["_atom_site.aniso_U[2][3]"], errors="coerce"
            )
            result["anisou_flag"] = ~pd.isna(result["u11"])
        else:
            result["u11"] = np.nan
            result["u22"] = np.nan
            result["u33"] = np.nan
            result["u12"] = np.nan
            result["u13"] = np.nan
            result["u23"] = np.nan
            result["anisou_flag"] = False

        # Add index column for compatibility with legacy PDB format
        result["index"] = np.arange(len(result), dtype=int)
        result["element"] = result["element"].str.strip().str.capitalize()
        return result

    def get_atom_data_by_model(self) -> Dict[int, pd.DataFrame]:
        """
        Split atom data by ``pdbx_PDB_model_num``.

        For single-model files, returns ``{1: dataframe}``.
        For multi-model files, returns one DataFrame per model number.

        Returns
        -------
        dict of int -> pandas.DataFrame
            Mapping of model number to atom DataFrame.
        """
        df = self.get_atom_data()
        if "model_num" not in df.columns:
            return {1: df}
        return {
            int(num): group.reset_index(drop=True)
            for num, group in df.groupby("model_num")
        }

    def _extract_string(
        self,
        df: pd.DataFrame,
        possible_cols: List[str],
        required: bool = False,
        default: str = "",
        replace_dot: bool = False,
    ) -> pd.Series:
        """Extract string column with fallbacks."""
        for col in possible_cols:
            if col in df.columns:
                data = df[col].fillna(default)
                if replace_dot:
                    data = data.replace([".", "?"], default)
                return data

        if required:
            raise ValueError(f"Required column not found. Tried: {possible_cols}")

        return pd.Series([default] * len(df))

    def _extract_float(
        self,
        df: pd.DataFrame,
        possible_cols: List[str],
        required: bool = False,
        default: float = np.nan,
    ) -> pd.Series:
        """Extract float column with fallbacks."""
        for col in possible_cols:
            if col in df.columns:
                return pd.to_numeric(
                    df[col].replace(["?", "."], np.nan), errors="coerce"
                ).fillna(default)

        if required:
            raise ValueError(f"Required column not found. Tried: {possible_cols}")

        return pd.Series([default] * len(df))

    def _extract_int(
        self,
        df: pd.DataFrame,
        possible_cols: List[str],
        required: bool = False,
        default: int = 0,
        default_range: bool = False,
    ) -> pd.Series:
        """Extract integer column with fallbacks."""
        for col in possible_cols:
            if col in df.columns:
                # Replace missing values and convert to numeric
                # Use mask to avoid FutureWarning about downcasting in replace
                series = df[col].copy()
                series = series.mask(series.isin(["?", "."]), default)
                return (
                    pd.to_numeric(series, errors="coerce").fillna(default).astype(int)
                )

        if required:
            raise ValueError(f"Required column not found. Tried: {possible_cols}")

        if default_range:
            return pd.Series(range(1, len(df) + 1))

        return pd.Series([default] * len(df))

    def get_cell_parameters(self) -> Optional[List[float]]:
        """Extract unit cell parameters [a, b, c, alpha, beta, gamma]."""
        if "cell" not in self.cif.data:
            return None

        cell_data = self.cif.data["cell"]
        try:
            a = float(
                self._get_first_value(cell_data, ["_cell.length_a", "length_a"], "1.0")
            )
            b = float(
                self._get_first_value(cell_data, ["_cell.length_b", "length_b"], "1.0")
            )
            c = float(
                self._get_first_value(cell_data, ["_cell.length_c", "length_c"], "1.0")
            )
            alpha = float(
                self._get_first_value(
                    cell_data, ["_cell.angle_alpha", "angle_alpha"], "90.0"
                )
            )
            beta = float(
                self._get_first_value(
                    cell_data, ["_cell.angle_beta", "angle_beta"], "90.0"
                )
            )
            gamma = float(
                self._get_first_value(
                    cell_data, ["_cell.angle_gamma", "angle_gamma"], "90.0"
                )
            )
            return [a, b, c, alpha, beta, gamma]
        except Exception:
            return None

    def get_space_group(self) -> str:
        """Hermann-Mauguin space group name, falling back to ``"P 1"``."""
        sg_name = "P 1"
        if "symmetry" in self.cif.data:
            sym_data = self.cif.data["symmetry"]
            sg_name = self._get_first_value(
                sym_data,
                [
                    "_symmetry.space_group_name_H-M",
                    "space_group_name_H-M",
                    "_space_group.name_H-M_alt",
                ],
                "P 1",
            )

        # Validate the name by trying to parse it
        try:
            gemmi.SpaceGroup(sg_name)
            return sg_name
        except Exception:
            try:
                gemmi.SpaceGroup(sg_name.replace(" ", ""))
                return sg_name.replace(" ", "")
            except Exception:
                return "P 1"

    def _get_first_value(
        self, data, possible_keys: List[str], default: Any = None
    ) -> Any:
        """Get value from DataFrame or dict, trying multiple keys."""
        if isinstance(data, pd.DataFrame):
            for key in possible_keys:
                if key in data.columns and len(data) > 0:
                    return data[key].iloc[0]
        elif isinstance(data, dict):
            for key in possible_keys:
                if key in data:
                    return data[key]
        return default

    # Convenience methods for testing
    def has_coordinates(self) -> bool:
        """Check if atomic coordinates are available."""
        return "atom_site" in self.cif.data

    def has_cell_parameters(self) -> bool:
        """Check if unit cell parameters are available."""
        return "cell" in self.cif.data

    def has_space_group(self) -> bool:
        """Check if space group information is available."""
        return "symmetry" in self.cif.data

    def has_occupancy(self) -> bool:
        """Check if occupancy data is available."""
        if "atom_site" not in self.cif.data:
            return False
        return "_atom_site.occupancy" in self.cif.data["atom_site"].columns

    def has_bfactor(self) -> bool:
        """Check if B-factor/temperature factor data is available."""
        if "atom_site" not in self.cif.data:
            return False
        return "_atom_site.B_iso_or_equiv" in self.cif.data["atom_site"].columns

    def has_anisotropic_data(self) -> bool:
        """Check if anisotropic displacement parameters are available."""
        if "atom_site" not in self.cif.data:
            return False
        aniso_cols = [
            "_atom_site.aniso_U[1][1]",
            "_atom_site.aniso_U[2][2]",
            "_atom_site.aniso_U[3][3]",
        ]
        return all(col in self.cif.data["atom_site"].columns for col in aniso_cols)

    def get_coordinates(self) -> Optional[np.ndarray]:
        """Coordinates as an (N, 3) array of [x, y, z], or None if absent."""
        if not self.has_coordinates():
            return None

        atom_data = self.get_atom_data()
        return atom_data[["x", "y", "z"]].values

    def get_atom_info(self) -> pd.DataFrame:
        """Atom names, residue info and elements, without the coordinates."""
        atom_data = self.get_atom_data()
        return atom_data[
            [
                "serial",
                "name",
                "altloc",
                "resname",
                "chainid",
                "resseq",
                "icode",
                "element",
                "charge",
            ]
        ]


class RestraintCIFReader:
    """
    Reader for monomer-library restraint dictionaries: bonds, angles, torsions,
    planes and chirality, each with its ideal value and ESD.

    Validates that the file really carries restraint parameters, not just a
    structure definition::

        comp_data = RestraintCIFReader('.../ALA.cif').get_all_restraints()
        bond_df = comp_data['ALA']['bonds']
    """

    def __init__(self, filepath: str, verbose: int = 0):
        """Load the restraint dictionary at ``filepath`` immediately.

        ``verbose`` above 0 reports a dictionary that defines more than one
        compound, which is otherwise indistinguishable from a single-compound one
        at the call site.
        """
        self.filepath = Path(filepath)
        self.verbose = verbose
        # Every data block is read, not just the first: a dictionary may define
        # several compounds, each in its own ``data_comp_<ID>`` block.
        self.cif = CIFReader(filepath, parse_all_blocks=True)
        self.compounds = self._extract_compounds()
        self._validate()

        if self.verbose > 0 and len(self.compounds) > 1:
            print(
                f"  {self.filepath.name}: {len(self.compounds)} compounds "
                f"({', '.join(self.compounds)})"
            )

    def _extract_compounds(self) -> List[str]:
        """Compound IDs present in the file, e.g. ``['ALA']``."""
        compounds = []

        # Check for comp_list (monomer library format)
        if "comp_list" in self.cif.data:
            df = self.cif.data["comp_list"]
            if "id" in df.columns:
                compounds = df["id"].tolist()
            elif "_chem_comp.id" in df.columns:
                compounds = df["_chem_comp.id"].tolist()

        # Check for chem_comp (eLBOW/phenix format)
        if not compounds and "chem_comp" in self.cif.data:
            df = self.cif.data["chem_comp"]
            if "_chem_comp.id" in df.columns and len(df) > 0:
                compounds = df["_chem_comp.id"].tolist()
            elif "id" in df.columns and len(df) > 0:
                compounds = df["id"].tolist()

        if not compounds and "comp" in self.cif.data:
            df = self.cif.data["comp"]
            if "id" in df.columns and len(df) > 0:
                compounds = [df["id"].iloc[0]]
            elif "_chem_comp.id" in df.columns and len(df) > 0:
                compounds = [df["_chem_comp.id"].iloc[0]]

        # If no comp_list/chem_comp header, derive the compound ID(s) from the
        # restraint data blocks themselves. Many user-supplied dictionaries
        # (eLBOW, AceDRG, Grade) ship a single ``data_comp_<ID>`` block with no
        # ``comp_list``; without this the ID would fall back to the filename
        # stem (see ``_validate``), which rarely matches the PDB residue name,
        # so ``_filter_by_comp`` returns nothing and the custom restraints are
        # silently overshadowed by the bundled monomer library.
        if not compounds:
            for block in (
                "comp_atom",
                "chem_comp_atom",
                "comp_bond",
                "chem_comp_bond",
            ):
                df = self.cif.data.get(block)
                if df is None or len(df) == 0:
                    continue
                id_col = next(
                    (c for c in df.columns if c == "comp_id" or c.endswith(".comp_id")),
                    None,
                )
                if id_col is not None:
                    compounds = [c for c in df[id_col].unique().tolist() if c]
                    if compounds:
                        break

        # Last resort before the filename stem: the ``data_comp_<ID>`` block names.
        # A dictionary can omit both ``comp_list`` and the per-row ``comp_id``
        # columns, and then the block name is the only record of what it defines.
        if not compounds:
            compounds = [
                block[len("comp_") :]
                for block in self.cif.available_blocks
                if block.startswith("comp_") and block != "comp_list"
            ]

        # A compound may be named by both the header and its own data rows, and
        # ``get_all_restraints`` would then build it twice.
        return list(dict.fromkeys(c for c in compounds if c))

    def _validate(self):
        """
        Validate that this is a proper restraint file with geometry parameters.
        """
        if not self.compounds:
            # Try to infer compound ID from filename
            comp_id = self.filepath.stem
            if comp_id:
                self.compounds = [comp_id]

        # Check for bond restraints with proper parameters
        # Try both naming conventions: comp_bond and chem_comp_bond
        bond_df = None
        if "comp_bond" in self.cif.data:
            bond_df = self.cif.data["comp_bond"]
        elif "chem_comp_bond" in self.cif.data:
            bond_df = self.cif.data["chem_comp_bond"]

        if bond_df is not None:
            required_cols = ["value_dist", "value_dist_esd"]
            missing_cols = [
                col
                for col in required_cols
                if not any(col in c for c in bond_df.columns)
            ]

            if missing_cols:
                raise ValueError(
                    f"Restraint file {self.filepath} is missing required bond parameters.\n"
                    f"Missing columns: {missing_cols}\n"
                    f"Available columns: {list(bond_df.columns)}\n\n"
                    f"This appears to be a structure definition file (from PDB) rather than\n"
                    f"a proper restraint dictionary. Restraint files must include ideal\n"
                    f"geometry parameters such as 'value_dist' and 'value_dist_esd'.\n\n"
                    f"Solution: Use files from the CCP4 Monomer Library\n"
                    f"which contain proper restraint parameters."
                )
        else:
            raise ValueError(
                f"File {self.filepath} does not contain bond restraint data (_chem_comp_bond).\n"
                f"Available data blocks: {list(self.cif.data.keys())}\n\n"
                f"This is not a valid restraint dictionary file."
            )

    def get_all_restraints(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Extract all restraint data for all compounds with standardized column names.

        Returns
        -------
        dict
            Dictionary mapping compound ID to dict of restraint types::

                {
                    'ALA': {
                        'bonds': DataFrame(atom1, atom2, value, sigma),
                        'angles': DataFrame(atom1, atom2, atom3, value, sigma),
                        'torsions': DataFrame(atom1, atom2, atom3, atom4, value, sigma, periodicity),
                        'planes': DataFrame(atom, plane_id),
                        'chirals': DataFrame(atom_centre, atom1, atom2, atom3, volume_sign)
                    },
                    ...
                }
        """
        result = {}

        for comp_id in self.compounds:
            result[comp_id] = self.get_compound_restraints(comp_id)

        # If no compounds found, try to get data directly
        if not result:
            comp_id = self.filepath.stem
            raw_bonds = self.cif.data.get(
                "comp_bond", self.cif.data.get("chem_comp_bond", pd.DataFrame())
            )
            raw_angles = self.cif.data.get(
                "comp_angle", self.cif.data.get("chem_comp_angle", pd.DataFrame())
            )
            raw_torsions = self.cif.data.get(
                "comp_tor", self.cif.data.get("chem_comp_tor", pd.DataFrame())
            )
            raw_planes = self.cif.data.get(
                "comp_plane_atom",
                self.cif.data.get("chem_comp_plane_atom", pd.DataFrame()),
            )
            raw_chirals = self.cif.data.get(
                "comp_chir", self.cif.data.get("chem_comp_chir", pd.DataFrame())
            )
            raw_atoms = self.cif.data.get(
                "comp_atom", self.cif.data.get("chem_comp_atom", pd.DataFrame())
            )

            result[comp_id] = {
                "bonds": self._standardize_bonds(raw_bonds),
                "angles": self._standardize_angles(raw_angles),
                "torsions": self._standardize_torsions(raw_torsions),
                "planes": self._standardize_planes(raw_planes),
                "chirals": self._standardize_chirals(raw_chirals),
                "atoms": self._standardize_atoms(raw_atoms),
            }

        return result

    def get_compound_restraints(self, comp_id: str) -> Dict[str, pd.DataFrame]:
        """
        Extract restraints for a specific compound with standardized column names.

        Parameters
        ----------
        comp_id : str
            Compound identifier (e.g., 'ALA').

        Returns
        -------
        dict
            Dictionary of restraint DataFrames with standardized columns::

                {
                    'bonds': DataFrame(atom1, atom2, value, sigma)
                    'angles': DataFrame(atom1, atom2, atom3, value, sigma)
                    'torsions': DataFrame(atom1, atom2, atom3, atom4, value, sigma, periodicity)
                    'planes': DataFrame(atom, plane_id)
                    'chirals': DataFrame(atom_centre, atom1, atom2, atom3, volume_sign)
                    'atoms': DataFrame(atom_id, type_symbol, charge, etc.)
                }
        """
        restraints = {}

        # Extract and standardize each restraint type
        raw_bonds = self._filter_by_comp(
            self.cif.data.get(
                "comp_bond", self.cif.data.get("chem_comp_bond", pd.DataFrame())
            ),
            comp_id,
        )
        restraints["bonds"] = self._standardize_bonds(raw_bonds)

        raw_angles = self._filter_by_comp(
            self.cif.data.get(
                "comp_angle", self.cif.data.get("chem_comp_angle", pd.DataFrame())
            ),
            comp_id,
        )
        restraints["angles"] = self._standardize_angles(raw_angles)

        raw_torsions = self._filter_by_comp(
            self.cif.data.get(
                "comp_tor", self.cif.data.get("chem_comp_tor", pd.DataFrame())
            ),
            comp_id,
        )
        restraints["torsions"] = self._standardize_torsions(raw_torsions)

        raw_planes = self._filter_by_comp(
            self.cif.data.get(
                "comp_plane_atom",
                self.cif.data.get("chem_comp_plane_atom", pd.DataFrame()),
            ),
            comp_id,
        )
        restraints["planes"] = self._standardize_planes(raw_planes)

        raw_chirals = self._filter_by_comp(
            self.cif.data.get(
                "comp_chir", self.cif.data.get("chem_comp_chir", pd.DataFrame())
            ),
            comp_id,
        )
        restraints["chirals"] = self._standardize_chirals(raw_chirals)

        raw_atoms = self._filter_by_comp(
            self.cif.data.get(
                "comp_atom", self.cif.data.get("chem_comp_atom", pd.DataFrame())
            ),
            comp_id,
        )
        restraints["atoms"] = self._standardize_atoms(raw_atoms)

        return restraints

    def _standardize_bonds(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize bond restraint columns to: atom1, atom2, value, sigma."""
        if df.empty:
            return pd.DataFrame(columns=["atom1", "atom2", "value", "sigma"])

        result = pd.DataFrame()
        result["atom1"] = self._extract_col(
            df, ["atom_id_1", "_chem_comp_bond.atom_id_1", "atom1"]
        )
        result["atom2"] = self._extract_col(
            df, ["atom_id_2", "_chem_comp_bond.atom_id_2", "atom2"]
        )
        result["value"] = pd.to_numeric(
            self._extract_col(
                df, ["value_dist", "_chem_comp_bond.value_dist", "value"]
            ),
            errors="coerce",
        )
        result["sigma"] = pd.to_numeric(
            self._extract_col(
                df, ["value_dist_esd", "_chem_comp_bond.value_dist_esd", "sigma", "esd"]
            ),
            errors="coerce",
        )
        return result

    def _standardize_angles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize angle restraint columns to: atom1, atom2, atom3, value, sigma."""
        if df.empty:
            return pd.DataFrame(columns=["atom1", "atom2", "atom3", "value", "sigma"])

        result = pd.DataFrame()
        result["atom1"] = self._extract_col(
            df, ["atom_id_1", "_chem_comp_angle.atom_id_1", "atom1"]
        )
        result["atom2"] = self._extract_col(
            df, ["atom_id_2", "_chem_comp_angle.atom_id_2", "atom2"]
        )
        result["atom3"] = self._extract_col(
            df, ["atom_id_3", "_chem_comp_angle.atom_id_3", "atom3"]
        )
        result["value"] = pd.to_numeric(
            self._extract_col(
                df, ["value_angle", "_chem_comp_angle.value_angle", "value"]
            ),
            errors="coerce",
        )
        result["sigma"] = pd.to_numeric(
            self._extract_col(
                df,
                ["value_angle_esd", "_chem_comp_angle.value_angle_esd", "sigma", "esd"],
            ),
            errors="coerce",
        )
        return result

    def _standardize_torsions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize torsion restraint columns to: atom1, atom2, atom3, atom4, value, sigma, periodicity."""
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "atom1",
                    "atom2",
                    "atom3",
                    "atom4",
                    "value",
                    "sigma",
                    "periodicity",
                ]
            )

        result = pd.DataFrame()
        result["atom1"] = self._extract_col(
            df, ["atom_id_1", "_chem_comp_tor.atom_id_1", "atom1"]
        )
        result["atom2"] = self._extract_col(
            df, ["atom_id_2", "_chem_comp_tor.atom_id_2", "atom2"]
        )
        result["atom3"] = self._extract_col(
            df, ["atom_id_3", "_chem_comp_tor.atom_id_3", "atom3"]
        )
        result["atom4"] = self._extract_col(
            df, ["atom_id_4", "_chem_comp_tor.atom_id_4", "atom4"]
        )
        result["value"] = pd.to_numeric(
            self._extract_col(
                df, ["value_angle", "_chem_comp_tor.value_angle", "value"]
            ),
            errors="coerce",
        )
        result["sigma"] = pd.to_numeric(
            self._extract_col(
                df,
                ["value_angle_esd", "_chem_comp_tor.value_angle_esd", "sigma", "esd"],
            ),
            errors="coerce",
        )
        result["periodicity"] = pd.to_numeric(
            self._extract_col(df, ["period", "_chem_comp_tor.period", "periodicity"]),
            errors="coerce",
        )
        return result

    def _standardize_planes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize plane restraint columns to: atom, plane_id, sigma."""
        if df.empty:
            return pd.DataFrame(columns=["atom", "plane_id", "sigma"])

        result = pd.DataFrame()
        result["atom"] = self._extract_col(
            df, ["atom_id", "_chem_comp_plane_atom.atom_id", "atom"]
        )
        result["plane_id"] = self._extract_col(
            df, ["plane_id", "_chem_comp_plane_atom.plane_id", "id"]
        )

        # Extract sigma (dist_esd) and convert to numeric
        sigma = pd.to_numeric(
            self._extract_col(
                df, ["dist_esd", "_chem_comp_plane_atom.dist_esd", "sigma"]
            ),
            errors="coerce",
        )

        # Fill missing values with 0.01 Å default, then clip minimum to 0.001 Å
        # (avoid overly tight restraints while allowing looser ones)
        sigma = sigma.fillna(0.01)
        result["sigma"] = sigma.clip(lower=0.001)  # Minimum 0.001 Å, no maximum

        return result

    def _standardize_chirals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize chirality columns to: atom_centre, atom1, atom2, atom3, volume_sign."""
        if df.empty:
            return pd.DataFrame(
                columns=["atom_centre", "atom1", "atom2", "atom3", "volume_sign"]
            )

        result = pd.DataFrame()
        result["atom_centre"] = self._extract_col(
            df, ["atom_id_centre", "_chem_comp_chir.atom_id_centre", "atom_centre"]
        )
        result["atom1"] = self._extract_col(
            df, ["atom_id_1", "_chem_comp_chir.atom_id_1", "atom1"]
        )
        result["atom2"] = self._extract_col(
            df, ["atom_id_2", "_chem_comp_chir.atom_id_2", "atom2"]
        )
        result["atom3"] = self._extract_col(
            df, ["atom_id_3", "_chem_comp_chir.atom_id_3", "atom3"]
        )
        result["volume_sign"] = self._extract_col(
            df, ["volume_sign", "_chem_comp_chir.volume_sign", "sign"]
        )
        return result

    def _standardize_atoms(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize atom definition columns to: atom_id, type_symbol, charge, etc."""
        if df.empty:
            return pd.DataFrame(columns=["atom_id", "type_symbol", "charge"])

        result = pd.DataFrame()
        result["atom_id"] = self._extract_col(
            df, ["atom_id", "_chem_comp_atom.atom_id", "id"]
        )
        result["type_symbol"] = self._extract_col(
            df, ["type_symbol", "_chem_comp_atom.type_symbol", "symbol"]
        )
        result["charge"] = pd.to_numeric(
            self._extract_col(
                df, ["charge", "_chem_comp_atom.charge", "partial_charge"]
            ),
            errors="coerce",
        )

        # Include x,y,z if present (for ideal coordinates)
        for coord in ["x", "y", "z"]:
            coord_cols = [
                f"pdbx_model_Cartn_{coord}_ideal",
                f"_chem_comp_atom.pdbx_model_Cartn_{coord}_ideal",
                f"_chem_comp_atom.{coord}",
                coord,
            ]
            if any(col in df.columns for col in coord_cols):
                result[coord] = pd.to_numeric(
                    self._extract_col(df, coord_cols), errors="coerce"
                )

        return result

    def _filter_by_comp(self, df: pd.DataFrame, comp_id: str) -> pd.DataFrame:
        """Rows of ``df`` belonging to ``comp_id``.

        The index is reset because the caller assembles its result column by
        column: :meth:`_extract_col` preserves this frame's index for a column it
        finds but returns a fresh ``RangeIndex`` for one it does not, so a
        non-zero-based index makes those two disagree and pandas aligns the
        mismatch away to NaN -- dropping values that are present. Rows only reach
        a non-zero index once several blocks are concatenated, i.e. exactly on the
        multi-compound dictionaries this reader now supports.
        """
        if df.empty:
            return df.drop(columns=[_SOURCE_BLOCK_COLUMN], errors="ignore")

        # Try different possible column names for compound ID
        # Include all naming conventions: monomer library, eLBOW/phenix, short forms
        id_cols = [
            "comp_id",
            "_chem_comp.id",
            "_chem_comp_bond.comp_id",
            "_chem_comp_angle.comp_id",
            "_chem_comp_tor.comp_id",
            "_chem_comp_atom.comp_id",
            "_chem_comp_plane_atom.comp_id",
            "_chem_comp_chir.comp_id",
            "id",
        ]

        selected = None
        for col in id_cols:
            if col in df.columns:
                selected = df[df[col] == comp_id]
                break

        # No comp_id column: fall back to the block the rows were read from.
        # Dictionaries from eLBOW/Grade often omit comp_id, and without this the
        # compounds in a multi-block file would be indistinguishable.
        if selected is None and _SOURCE_BLOCK_COLUMN in df.columns:
            selected = df[df[_SOURCE_BLOCK_COLUMN].isin([f"comp_{comp_id}", comp_id])]

        # Single-compound file with no comp_id anywhere: every row is its own.
        if selected is None:
            selected = df

        return (
            selected.drop(columns=[_SOURCE_BLOCK_COLUMN], errors="ignore")
            .reset_index(drop=True)
            .copy()
        )

    def get_bond_restraints(self, comp_id: str) -> pd.DataFrame:
        """
        Get bond restraints with standardized column names.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns:
                - atom1, atom2: Atom names
                - value: Ideal bond length (Å)
                - sigma: Estimated standard deviation (Å)
        """
        restraints = self.get_compound_restraints(comp_id)
        return restraints["bonds"]

    def _extract_col(self, df: pd.DataFrame, possible_cols: List[str]) -> pd.Series:
        """Extract column trying multiple names."""
        for col in possible_cols:
            if col in df.columns:
                return df[col]
        return pd.Series([None] * len(df))

    # Convenience methods for testing
    def get_compound_id(self) -> str:
        """Get the primary compound ID from this file."""
        if self.compounds:
            return self.compounds[0]
        return self.filepath.stem

    def has_bond_restraints(self) -> bool:
        """Check if bond restraints are available."""
        return "comp_bond" in self.cif.data or "chem_comp_bond" in self.cif.data

    def has_angle_restraints(self) -> bool:
        """Check if angle restraints are available."""
        return "comp_angle" in self.cif.data or "chem_comp_angle" in self.cif.data

    def has_torsion_restraints(self) -> bool:
        """Check if torsion restraints are available."""
        return "comp_tor" in self.cif.data or "chem_comp_tor" in self.cif.data

    def has_plane_restraints(self) -> bool:
        """Check if plane restraints are available."""
        return (
            "comp_plane_atom" in self.cif.data
            or "chem_comp_plane_atom" in self.cif.data
        )

    def has_chirality_restraints(self) -> bool:
        """Check if chirality definitions are available."""
        return "comp_chir" in self.cif.data or "chem_comp_chir" in self.cif.data
