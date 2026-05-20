from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

import gemmi
import numpy as np
import pandas as pd
import torch
import json

from torchref.utils.device_mixin import DeviceMovementMixin


class ModuleReference:
    """
    A wrapper class to hold references to PyTorch modules without registering them.

    When you assign a nn.Module to an attribute of another nn.Module, PyTorch
    automatically registers it as a submodule, which adds its parameters to the
    parent's parameter tree. This wrapper prevents tlog_normal_stdhat automatic registration.

    This is useful when you want to:

    - Hold references to modules without including their parameters
    - Avoid circular dependencies in the module tree
    - Reference external modules that should be managed separately

    Attributes
    ----------
    _wrapped_module : torch.nn.Module
        The wrapped PyTorch module.

    Examples
    --------
    ::

        model = MyModel()
        scaler = Scaler()
        scaler._model = ModuleReference(model)  # Won't register as submodule
        # Access the module via .module property
        output = scaler._model.module(input_data)
    """

    def __init__(self, module):
        """
        Wrap a module to prevent automatic registration.

        Parameters
        ----------
        module : torch.nn.Module
            The PyTorch module to wrap.
        """
        # Store in __dict__ directly to avoid any attribute interception
        object.__setattr__(self, "_wrapped_module", module)

    @property
    def module(self):
        """Access the wrapped module."""
        return object.__getattribute__(self, "_wrapped_module")

    def __getattr__(self, name):
        """Forward attribute access to the wrapped module."""
        return getattr(self.module, name)

    def __call__(self, *args, **kwargs):
        """Forward calls to the wrapped module."""
        return self.module(*args, **kwargs)

    def __repr__(self):
        return f"ModuleReference({self.module.__class__.__name__})"


class CIFReader:
    """
    A dictionary-like reader for CIF/mmCIF files.

    Loops are stored as pandas DataFrames.
    Other data is stored in a hierarchical dictionary structure.

    Attributes
    ----------
    data : dict
        Dictionary storing parsed CIF data.
    filepath : pathlib.Path or None
        Path to the loaded CIF file.
    """

    def __init__(self, filepath: Optional[str] = None):
        """
        Initialize CIF reader.

        Parameters
        ----------
        filepath : str, optional
            Path to CIF file to load immediately.
        """
        self.data = {}
        self.filepath = None

        if filepath:
            self.load(filepath)

    def load(self, filepath: str):
        """
        Load and parse a CIF file.

        Parameters
        ----------
        filepath : str
            Path to CIF file.
        """
        self.filepath = Path(filepath)
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        self._parse(content)

    def _parse(self, content: str):
        """
        Parse CIF file content.

        Parameters
        ----------
        content : str
            String content of CIF file.
        """
        lines = content.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                i += 1
                continue

            # Check for data block (usually just one in mmCIF)
            if line.startswith("data_"):
                i += 1
                continue

            # Check for loop
            if line.startswith("loop_"):
                i = self._parse_loop(lines, i + 1)
                continue

            # Parse single key-value pairs
            if line.startswith("_"):
                i = self._parse_keyvalue(lines, i)
                continue

            i += 1

    def _parse_loop(self, lines: List[str], start_idx: int) -> int:
        """
        Parse a loop structure into a pandas DataFrame.

        Parameters
        ----------
        lines : list of str
            All lines of the file.
        start_idx : int
            Starting line index (after 'loop_').

        Returns
        -------
        int
            Index of the next line to process.
        """
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
                    self.data[category] = df

        return i

    def _parse_keyvalue(self, lines: List[str], start_idx: int) -> int:
        """
        Parse a single key-value pair.

        Parameters
        ----------
        lines : list of str
            All lines of the file.
        start_idx : int
            Starting line index.

        Returns
        -------
        int
            Index of the next line to process.
        """
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
        """
        Tokenize a data line, handling quoted strings.

        Parameters
        ----------
        line : str
            Line to tokenize.

        Returns
        -------
        list of str
            List of tokens.
        """
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
        """
        Extract category from a CIF key (e.g., '_atom_site.id' -> 'atom_site').

        Parameters
        ----------
        key : str
            CIF key.

        Returns
        -------
        str
            Category name.
        """
        if key.startswith("_"):
            key = key[1:]

        if "." in key:
            return key.split(".")[0]

        return key

    def _store_keyvalue(self, key: str, value: str):
        """
        Store a key-value pair in the hierarchical dictionary.

        Parameters
        ----------
        key : str
            CIF key (e.g., '_entry.id').
        value : str
            Value to store.
        """
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
        """
        Write the CIF data back to a file.

        Parameters
        ----------
        filepath : str
            Output file path.
        """
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


def save_map(array, cell, filename):
    """
    Save a 3D map to a CCP4 file.

    Parameters
    ----------
    array : numpy.ndarray or torch.Tensor
        3D array representing the map.
    cell : list, tuple, numpy.ndarray, torch.Tensor, or gemmi.UnitCell
        Unit cell parameters [a, b, c, alpha, beta, gamma].
    filename : str
        Output CCP4 file name.

    Returns
    -------
    bool
        True if save was successful.
    """

    if isinstance(array, torch.Tensor):
        np_map = array.detach().cpu().numpy().astype(np.float32)
    else:
        np_map = array.astype(np.float32)
    if isinstance(cell, gemmi.UnitCell):
        cell = cell.parameters
    elif isinstance(cell, np.ndarray):
        cell = cell.tolist()
    elif isinstance(cell, list):
        cell = cell
    elif isinstance(cell, tuple):
        cell = list(cell)
    elif isinstance(cell, torch.Tensor):
        cell = cell.tolist()
    map_ccp = gemmi.Ccp4Map()
    map_ccp.grid = gemmi.FloatGrid(
        np_map, gemmi.UnitCell(*cell), gemmi.find_spacegroup_by_name("P1")
    )
    map_ccp.setup(0.0)
    map_ccp.update_ccp4_header()
    map_ccp.write_ccp4_map(filename)
    print("Map saved successfully")

    return True


import torch.nn as nn


class TensorDict(nn.Module):
    """
    A dictionary-like container for PyTorch tensors that:
    - Supports standard dict syntax
    - Automatically moves with the module
    - Registers tensors as buffers so they are included in state_dict
    """

    def __init__(self, initial_dict: Optional[Dict[str, torch.Tensor]] = None):
        super().__init__()
        self._keys = []
        if initial_dict:
            for k, v in initial_dict.items():
                self[k] = v

    def __setitem__(self, key: str, tensor: torch.Tensor):
        name = f"_buf_{key}"
        if not hasattr(self, name):
            # Register as buffer
            self.register_buffer(name, tensor)
            self._keys.append(key)
        else:
            existing = getattr(self, name)
            if existing.shape == tensor.shape:
                # Update existing buffer in-place (same shape)
                existing.data.copy_(tensor)
            else:
                # Shape changed - re-register the buffer with new tensor
                delattr(self, name)
                self.register_buffer(name, tensor)

    def __getitem__(self, key: str) -> torch.Tensor:
        name = f"_buf_{key}"
        if not hasattr(self, name):
            raise KeyError(key)
        return getattr(self, name)

    def __contains__(self, key: str):
        return key in self._keys

    def keys(self):
        return self._keys.copy()

    def values(self):
        return [getattr(self, f"_buf_{k}") for k in self._keys]

    def items(self):
        return [(k, getattr(self, f"_buf_{k}")) for k in self._keys]

    def __len__(self):
        return len(self._keys)

    def __repr__(self):
        return (
            "TensorDict({"
            + ", ".join(f'{k}: {getattr(self, f"_buf_{k}")}' for k in self._keys)
            + "}})"
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Override to dynamically register buffers during loading."""
        local_keys = [k for k in state_dict.keys() if k.startswith(prefix + "_buf_")]

        for key in local_keys:
            buffer_name = key[len(prefix) :]
            original_key = buffer_name[5:]  # remove "_buf_"

            if not hasattr(self, buffer_name):
                tensor = state_dict[key]
                self.register_buffer(buffer_name, torch.zeros_like(tensor))
                self._keys.append(original_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class TensorMasks(DeviceMovementMixin, dict):
    """
    A dictionary for managing boolean mask tensors with device support.

    This is a lightweight dict subclass that:
    - Ensures all tensors are boolean dtype
    - Supports device movement via to(), cuda(), cpu()
    - Provides combined mask via __call__()

    Parameters
    ----------
    data : dict, optional
        Initial mask data.
    device : str or torch.device, optional
        Device for tensors. Defaults to the configured device.current.

    Examples
    --------
    ::

        masks = TensorMasks(device='cuda')
        masks['valid'] = torch.ones(100, dtype=torch.bool)
        masks['rfree'] = rfree_flags > 0
        combined = masks()  # Get combined mask (AND of all)
        masks.cpu()  # Move all to CPU
    """

    def __init__(self, data=None, device=None):
        super().__init__()
        if device is None:
            from torchref.config import get_default_device

            device = get_default_device()
        self.device = torch.device(device)
        self._cache = None
        self._updated = True

        # Initialize with provided data
        if data:
            for k, v in data.items():
                self[k] = v

    def __setitem__(self, key: str, tensor: torch.Tensor):
        """Set mask tensor, ensuring boolean dtype and correct device."""
        if tensor is not None:
            if tensor.dtype != torch.bool:
                raise ValueError(
                    f"Mask '{key}' must be boolean dtype, got {tensor.dtype}"
                )
            if tensor.sum() == 0:
                raise ValueError(f"Mask '{key}' cannot be all False, this would mask all data.")
            tensor = tensor.to(self.device)
        super().__setitem__(key, tensor)
        self._updated = True

    def _apply(self, fn):
        """Move mask tensors stored as ``dict`` items and invalidate the cache.

        ``TensorMasks`` is a ``dict`` subclass — its mask tensors live in the
        dict's own storage, **not** in ``self.__dict__`` — so the standard
        :class:`DeviceMixin` ``__dict__`` walk would otherwise miss them and
        only move the cached combined mask, leaving the per-key masks on
        the previous device.
        """
        # Walk the dict storage and move each mask tensor.
        for k in list(self.keys()):
            v = self[k]
            if isinstance(v, torch.Tensor):
                dict.__setitem__(self, k, fn(v))

        # Invalidate the combined-mask cache so the next call to ``self()``
        # recomputes from the moved masks rather than returning the stale
        # combined tensor.
        self._cache = None
        self._updated = True

        # Refresh the ``device`` tracker so future ``__setitem__`` calls
        # (which migrate incoming tensors to ``self.device``) land correctly.
        for v in self.values():
            if isinstance(v, torch.Tensor):
                self.device = v.device
                break
        return self

    def reset_cache(self) -> None:
        """Invalidate the cached combined mask."""
        self._cache = None
        self._updated = True

    def __call__(self) -> torch.Tensor:
        """
        Return combined mask (AND of all masks).

        Returns
        -------
        torch.Tensor
            Combined boolean mask, or None if no masks.
        """
        if not self:
            return None

        if self._updated or self._cache is None:
            self._cache = self._get_combined_mask()
            self._updated = False

        return self._cache

    def _get_combined_mask(self) -> torch.Tensor:
        """Compute combined mask using logical AND."""
        masks = [v for v in self.values() if v is not None]
        if not masks:
            return None

        combined = masks[0].clone()
        for m in masks[1:]:
            combined &= m
        return combined

    def __repr__(self):
        mask_info = ", ".join(
            f"'{k}': shape={v.shape}" for k, v in self.items() if v is not None
        )
        return f"TensorMasks({{{mask_info}}}, device={self.device})"


def sanitize_pdb_dataframe(pdb: pd.DataFrame, verbose: int = 0) -> pd.DataFrame:
    """
    Sanitize a PDB DataFrame to ensure unique atom identifiers.

    This function fixes common issues in PDB/CIF files:

    1. HETATM records (especially waters) with duplicate resseq values (e.g., all 0)
    2. Residue names longer than 3 characters (truncates to 3)
    3. Ensures unique (chainid, resseq, name, altloc) combinations

    Parameters
    ----------
    pdb : pandas.DataFrame
        DataFrame with PDB data (must have columns: ATOM, chainid, resseq,
        name, altloc, resname, serial).
    verbose : int, default 0
        Verbosity level (0=silent, 1=info, 2=debug).

    Returns
    -------
    pandas.DataFrame
        Sanitized DataFrame with unique atom identifiers.

    Examples
    --------
    ::

        from torchref.model import Model
        from torchref.utils import sanitize_pdb_dataframe
        model = Model()
        model.load_cif('structure.cif')
        model.pdb = sanitize_pdb_dataframe(model.pdb, verbose=1)
    """
    pdb = pdb.copy()

    if verbose > 0:
        print("Sanitizing PDB DataFrame...")
        print(f"  Initial atoms: {len(pdb)}")

    # 1. Standardize residue names to max 3 characters
    long_resnames = pdb["resname"].str.len() > 3
    if long_resnames.any():
        n_long = long_resnames.sum()
        if verbose > 0:
            unique_long = pdb.loc[long_resnames, "resname"].unique()
            print(
                f"  Truncating {n_long} atoms with resname > 3 chars: {unique_long[:5]}"
            )
        pdb.loc[long_resnames, "resname"] = pdb.loc[long_resnames, "resname"].str[:3]

    # 2. Fix duplicate atom identifiers by reassigning resseq
    # Check for duplicates
    dup_mask = pdb.duplicated(
        subset=["chainid", "resseq", "name", "altloc"], keep=False
    )

    if dup_mask.any():
        n_dup = dup_mask.sum()
        if verbose > 0:
            print(f"  Found {n_dup} atoms with duplicate identifiers")

        # Group by (chainid, resname, ATOM) to handle each group separately
        # This ensures we only renumber within the same molecule type and chain
        for (chainid, resname, atom_type), group in pdb.groupby(
            ["chainid", "resname", "ATOM"]
        ):
            group_indices = group.index

            # Check if this group has duplicates
            group_dup_mask = group.duplicated(
                subset=["chainid", "resseq", "name", "altloc"], keep=False
            )

            if group_dup_mask.any():
                # Find the maximum resseq in this chain to start numbering from there
                chain_data = pdb[pdb["chainid"] == chainid]
                max_resseq = chain_data["resseq"].max()

                # Start numbering from max_resseq + 1
                new_resseq_start = (
                    max_resseq + 1 if pd.notna(max_resseq) and max_resseq > 0 else 1
                )

                # Assign new sequential resseq values to all atoms in this group
                # Group by (serial) to keep atoms of the same residue together
                unique_serials = group["serial"].unique()
                residue_counter = new_resseq_start

                for serial in unique_serials:
                    serial_mask = pdb["serial"] == serial
                    pdb.loc[serial_mask, "resseq"] = residue_counter
                    residue_counter += 1

                if verbose > 1:
                    n_fixed = len(unique_serials)
                    print(
                        f"    Fixed {n_fixed} {resname} residues in chain {chainid} (resseq {new_resseq_start}-{residue_counter-1})"
                    )

        # Verify duplicates are fixed
        final_dup_mask = pdb.duplicated(
            subset=["chainid", "resseq", "name", "altloc"], keep=False
        )
        if final_dup_mask.any():
            remaining_dups = final_dup_mask.sum()
            if verbose > 0:
                print(
                    f"  WARNING: Still have {remaining_dups} duplicate identifiers after sanitization"
                )
                dups = pdb[final_dup_mask].sort_values(["chainid", "resseq", "name"])
                print(
                    dups[
                        [
                            "ATOM",
                            "serial",
                            "name",
                            "resname",
                            "chainid",
                            "resseq",
                            "altloc",
                        ]
                    ].head(10)
                )
        else:
            if verbose > 0:
                print("  ✓ All duplicate identifiers resolved")
    else:
        if verbose > 0:
            print("  ✓ No duplicate atom identifiers found")

    if verbose > 0:
        print(f"  Final atoms: {len(pdb)}")

    return pdb


def _parse_with_parentheses(
    selection_string: str, pdb_df: pd.DataFrame
) -> torch.Tensor:
    """
    Helper function to handle parentheses in selection strings.
    Recursively evaluates innermost parentheses first.
    """
    import re

    # Find innermost parentheses
    while True:
        match = re.search(r"\(([^()]+)\)", selection_string)
        if not match:
            break

        # Evaluate the innermost parenthesized expression
        inner = match.group(1)
        inner_mask = _parse_without_parentheses(inner, pdb_df)

        # Replace with a placeholder that we'll substitute back
        # Use a unique placeholder that won't appear in normal selection
        placeholder = f"__MASK_{id(inner_mask)}__"
        selection_string = (
            selection_string[: match.start()]
            + placeholder
            + selection_string[match.end() :]
        )

        # Store the mask result in a temporary global dict
        # (not ideal but works for this recursive evaluation)
        if not hasattr(_parse_with_parentheses, "_mask_cache"):
            _parse_with_parentheses._mask_cache = {}
        _parse_with_parentheses._mask_cache[placeholder] = inner_mask

    # Now parse the expression without parentheses, substituting cached masks
    return _parse_without_parentheses(selection_string, pdb_df)


def _parse_without_parentheses(
    selection_string: str, pdb_df: pd.DataFrame
) -> torch.Tensor:
    """
    Parse selection string without parentheses.
    Handles logical operators and basic keywords.
    """
    import re

    selection_string = selection_string.strip()

    if not selection_string:
        raise ValueError("Selection string cannot be empty")

    # Check if this is a cached mask placeholder
    if selection_string.startswith("__MASK_") and selection_string.endswith("__"):
        if hasattr(_parse_with_parentheses, "_mask_cache"):
            return _parse_with_parentheses._mask_cache.get(
                selection_string, torch.ones(len(pdb_df), dtype=torch.bool)
            )
        return torch.ones(len(pdb_df), dtype=torch.bool)

    # Handle "all" keyword
    if selection_string.lower() == "all":
        return torch.ones(len(pdb_df), dtype=torch.bool)

    # Parse logical operators (or, and, not) with proper precedence
    # Priority: not > and > or

    # First, handle "or" (lowest precedence)
    if " or " in selection_string.lower():
        parts = re.split(r"\s+or\s+", selection_string, flags=re.IGNORECASE)
        masks = [_parse_without_parentheses(part.strip(), pdb_df) for part in parts]
        result = masks[0]
        for mask in masks[1:]:
            result = result | mask
        return result

    # Then, handle "and"
    if " and " in selection_string.lower():
        parts = re.split(r"\s+and\s+", selection_string, flags=re.IGNORECASE)
        masks = [_parse_without_parentheses(part.strip(), pdb_df) for part in parts]
        result = masks[0]
        for mask in masks[1:]:
            result = result & mask
        return result

    # Then, handle "not"
    if selection_string.lower().startswith("not "):
        inner_selection = selection_string[4:].strip()
        return ~_parse_without_parentheses(inner_selection, pdb_df)

    # Now handle individual selection keywords
    parts = selection_string.split(None, 1)
    if len(parts) < 2:
        raise ValueError(f"Invalid selection syntax: '{selection_string}'")

    keyword, value = parts[0].lower(), parts[1]

    # Initialize mask as all False
    mask = torch.zeros(len(pdb_df), dtype=torch.bool)

    if keyword == "chain":
        # Select by chain ID
        chain_id = value.strip()
        selected = pdb_df["chainid"] == chain_id
        mask = torch.tensor(selected.values, dtype=torch.bool)

    elif keyword == "resseq":
        # Select by residue sequence number or range
        if ":" in value:
            # Range selection
            start, end = value.split(":")
            start, end = int(start.strip()), int(end.strip())
            selected = (pdb_df["resseq"] >= start) & (pdb_df["resseq"] <= end)
        else:
            # Single residue
            resseq_num = int(value.strip())
            selected = pdb_df["resseq"] == resseq_num
        mask = torch.tensor(selected.values, dtype=torch.bool)

    elif keyword == "resname":
        # Select by residue name
        resname = value.strip().upper()
        selected = pdb_df["resname"].str.upper() == resname
        mask = torch.tensor(selected.values, dtype=torch.bool)

    elif keyword == "name":
        # Select by atom name
        atom_name = value.strip().upper()
        selected = pdb_df["name"].str.upper() == atom_name
        mask = torch.tensor(selected.values, dtype=torch.bool)

    elif keyword == "element":
        # Select by element
        element = value.strip().capitalize()
        selected = pdb_df["element"].str.capitalize() == element
        mask = torch.tensor(selected.values, dtype=torch.bool)

    elif keyword == "altloc":
        # Select by alternate location
        altloc = value.strip()
        selected = pdb_df["altloc"] == altloc
        mask = torch.tensor(selected.values, dtype=torch.bool)

    else:
        raise ValueError(f"Unknown selection keyword: '{keyword}'")

    return mask


def parse_phenix_selection(selection_string: str, pdb_df: pd.DataFrame) -> torch.Tensor:
    """
    Parse Phenix-style atom selection syntax and return a boolean mask.

    Supports common Phenix selection keywords:

    - chain <id>: Select atoms by chain ID (e.g., "chain A")
    - resseq <num>: Select atoms by residue sequence number (e.g., "resseq 10")
    - resseq <start>:<end>: Select residue range (e.g., "resseq 10:20")
    - resname <name>: Select atoms by residue name (e.g., "resname ALA")
    - name <atom>: Select atoms by atom name (e.g., "name CA")
    - element <elem>: Select atoms by element (e.g., "element C")
    - altloc <id>: Select atoms by alternate location (e.g., "altloc A")
    - all: Select all atoms
    - not <selection>: Negate selection
    - <sel1> and <sel2>: Intersection of selections
    - <sel1> or <sel2>: Union of selections
    - Parentheses for grouping: (selection)

    Parameters
    ----------
    selection_string : str
        Phenix-style selection string.
    pdb_df : pandas.DataFrame
        DataFrame containing atomic data with columns:
        'chainid', 'resseq', 'resname', 'name', 'element', 'altloc'.

    Returns
    -------
    torch.Tensor
        Boolean tensor of shape (n_atoms,) where True indicates selected atoms.

    Raises
    ------
    ValueError
        If selection syntax is invalid.

    Examples
    --------
    ::

        # Select chain A
        mask = parse_phenix_selection("chain A", pdb_df)

        # Select residues 10-20 in chain A
        mask = parse_phenix_selection("chain A and resseq 10:20", pdb_df)

        # Select all CA atoms
        mask = parse_phenix_selection("name CA", pdb_df)

        # Select backbone atoms
        mask = parse_phenix_selection("name CA or name C or name N or name O", pdb_df)

        # Select everything except water
        mask = parse_phenix_selection("not resname HOH", pdb_df)

        # Use parentheses for grouping
        mask = parse_phenix_selection("chain A and (name CA or name CB)", pdb_df)
    """
    # Clear any cached masks from previous calls
    if hasattr(_parse_with_parentheses, "_mask_cache"):
        _parse_with_parentheses._mask_cache.clear()

    # Check if there are parentheses
    if "(" in selection_string:
        return _parse_with_parentheses(selection_string, pdb_df)
    else:
        return _parse_without_parentheses(selection_string, pdb_df)


def create_selection_mask(
    selection_string: str,
    pdb_df: pd.DataFrame,
    current_mask: Optional[torch.Tensor] = None,
    mode: str = "set",
) -> torch.Tensor:
    """
    Create or modify a refinable mask based on a Phenix-style selection.

    This function allows you to update refinable masks by selecting specific atoms
    using Phenix-style syntax. You can either replace the current mask, add to it,
    or remove from it.

    Parameters
    ----------
    selection_string : str
        Phenix-style selection string.
    pdb_df : pandas.DataFrame
        DataFrame containing atomic data.
    current_mask : torch.Tensor, optional
        Current refinable mask. If None, starts with all False.
    mode : str, default 'set'
        How to combine with current mask:

        - 'set': Replace mask with selection (default)
        - 'add': Add selection to current mask (OR operation)
        - 'remove': Remove selection from current mask (AND NOT operation)

    Returns
    -------
    torch.Tensor
        Updated boolean mask of shape (n_atoms,).

    Raises
    ------
    ValueError
        If mode is not one of 'set', 'add', 'remove'.

    Examples
    --------
    ::

        # Create new mask selecting chain A
        mask = create_selection_mask("chain A", pdb_df, mode='set')

        # Add residues 10-20 to existing mask
        mask = create_selection_mask("resseq 10:20", pdb_df, current_mask=mask, mode='add')

        # Remove water from mask
        mask = create_selection_mask("resname HOH", pdb_df, current_mask=mask, mode='remove')
    """
    # Parse the selection
    selection_mask = parse_phenix_selection(selection_string, pdb_df)

    # Initialize current mask if not provided
    if current_mask is None:
        current_mask = torch.zeros(len(pdb_df), dtype=torch.bool)

    # Apply mode
    if mode == "set":
        return selection_mask
    elif mode == "add":
        return current_mask | selection_mask
    elif mode == "remove":
        return current_mask & ~selection_mask
    else:
        raise ValueError(f"Invalid mode: '{mode}'. Must be 'set', 'add', or 'remove'")


def state_dict_to_json_serializable(sd: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Convert a state_dict with tensors to a JSON-serializable format.

    Parameters
    ----------
    sd : Dict[str, torch.Tensor]
        State dict with tensor values.

    Returns
    -------
    Dict[str, Any]
        JSON-serializable dictionary.
    """
    result = {}
    for k, v in sd.items():
        result[k] = {
            "data": v.tolist() if v.numel() > 1 else [v.item()],
            "dtype": str(v.dtype),
            "shape": list(v.shape) if v.shape else [1],
        }
    return result


def dict_to_state_dict(sd_raw):
    """
    Convert a dict with serialized tensor info to a PyTorch state_dict.

    Parameters
    ----------
    sd_raw : dict
        Dictionary where values are dicts with 'data', 'dtype', 'shape' keys.

    Returns
    -------
    dict
        State dict with torch.Tensor values.
    """
    sd = {}
    for k, v in sd_raw.items():
        tensor = torch.tensor(
            v["data"], dtype=getattr(torch, v["dtype"].split(".")[-1])
        )
        tensor = tensor.reshape(v["shape"])
        sd[k] = tensor
    return sd


def json_to_state_dicts_separate(
    json_path: str,
) -> Tuple[
    Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor], list
]:
    """
    Parse hyperparameter JSON and return state_dicts for component_weighting, geometry_target, and adp_target.

    Parameters
    ----------
    json_path : str
        Path to the JSON file containing hyperparameters.

    Returns
    -------
    Tuple[Dict, Dict, Dict, list]
        Three state_dicts and a list of unassigned keys:
        (component_weighting_state, geometry_target_state, adp_target_state, unassigned_keys)
    """
    with open(json_path) as f:
        data = json.load(f)

    component_weighting_state = {}
    geometry_target_state = {}
    adp_target_state = {}
    unassigned_keys = []

    # Geometry target components
    geometry_components = {
        "bond",
        "angle",
        "torsion",
        "planarity",
        "chiral",
        "nonbonded",
    }
    # ADP target components
    adp_components = {"simu", "locality", "KL"}

    for key, value in data.items():
        tensor_value = torch.tensor(value)
        assigned = False

        if key.startswith("schemes."):
            # ComponentWeighting state_dict key
            component_weighting_state[key] = tensor_value
            assigned = True
        elif key.startswith("_targets."):
            # Extract component name (e.g., "_targets.bond._sigma" -> "bond")
            parts = key.split(".")
            component_name = parts[1]

            if component_name in geometry_components:
                geometry_target_state[key] = tensor_value
                assigned = True
            elif component_name in adp_components:
                adp_target_state[key] = tensor_value
                assigned = True

        if not assigned:
            unassigned_keys.append(key)

    return (
        component_weighting_state,
        geometry_target_state,
        adp_target_state,
        unassigned_keys,
    )


def disable_grad_outside_optimizer(optimized_params, all_params):
    """Set ``requires_grad=False`` on parameters not being optimized.

    Call this once after creating the optimizer. Non-optimized parameters
    will no longer contribute to the autograd graph, which means
    ``ModelFT.get_structure_factor`` will produce structure factors
    without ``grad_fn`` for frozen models — enabling indefinite caching
    until parameters change.

    Parameters
    ----------
    optimized_params : iterable of torch.Tensor
        Parameters passed to the optimizer.
    all_params : iterable of torch.Tensor
        All model parameters (e.g. ``model.parameters()``).
    """
    optimized_ids = set(id(p) for p in optimized_params)
    for p in all_params:
        if id(p) not in optimized_ids:
            p.requires_grad_(False)
