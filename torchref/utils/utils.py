"""
Core utility containers and atom-selection parsing for TorchRef.

This module defines the lower-level building blocks re-exported from
``torchref.utils``:

- :class:`ModuleReference` -- holds a reference to an ``nn.Module`` without
  registering it as a submodule (keeps its parameters out of the parent tree).
- :class:`TensorDict` -- an ``nn.Module``-backed, dict-like tensor container
  whose entries are registered as buffers (so they move with the module and
  appear in ``state_dict``).
- :class:`TensorMasks` -- a ``dict`` subclass for boolean mask tensors with
  device movement and a cached combined (logical-AND) mask.
- :func:`sanitize_pdb_dataframe` -- repair duplicate atom identifiers and
  over-long residue names in a PDB/CIF DataFrame.
- :func:`parse_phenix_selection` / :func:`create_selection_mask` -- parse
  Phenix-style atom-selection strings into boolean masks.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
import torch

from torchref.utils.device_mixin import DeviceMovementMixin


class ModuleReference:
    """
    A wrapper class to hold references to PyTorch modules without registering them.

    When you assign a nn.Module to an attribute of another nn.Module, PyTorch
    automatically registers it as a submodule, which adds its parameters to the
    parent's parameter tree. This wrapper prevents this automatic registration.

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


# NOTE: ``torch.nn`` is imported here (mid-module) rather than at the top
# with the other imports; left in place to avoid a non-docstring code move.
import torch.nn as nn


class TensorDict(nn.Module):
    """A dictionary-like container for PyTorch tensors.

    Backed by :class:`torch.nn.Module`: each stored tensor is registered as a
    buffer, so the container's tensors move with the module (``.to()`` /
    ``.cuda()`` / ``.cpu()``) and are included in ``state_dict``. Standard
    dict-style access (``td[key]``, ``key in td``, ``keys``/``values``/
    ``items``, ``len``) is supported. Insertion order of keys is preserved.

    Parameters
    ----------
    initial_dict : dict of str to torch.Tensor, optional
        Initial key/tensor pairs to populate the container.

    Examples
    --------
    ::

        td = TensorDict({'coords': torch.zeros(10, 3)})
        td['weights'] = torch.ones(10)
        td.cuda()                 # buffers move with the module
        coords = td['coords']     # standard dict access
        'weights' in td           # -> True
    """

    def __init__(self, initial_dict: Optional[Dict[str, torch.Tensor]] = None):
        super().__init__()
        self._keys = []
        if initial_dict:
            for k, v in initial_dict.items():
                self[k] = v

    def __setitem__(self, key: str, tensor: torch.Tensor):
        """Store ``tensor`` under ``key`` as a registered buffer.

        On a new key the tensor is registered as a buffer. On an existing
        key, a same-shape tensor is copied in-place into the buffer;
        a different-shape tensor causes the buffer to be re-registered.
        """
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
        """Return the tensor stored under ``key``.

        Raises
        ------
        KeyError
            If ``key`` is not present.
        """
        name = f"_buf_{key}"
        if not hasattr(self, name):
            raise KeyError(key)
        return getattr(self, name)

    def __contains__(self, key: str):
        """Return True if ``key`` is stored in the container."""
        return key in self._keys

    def keys(self):
        """Return a list copy of the stored keys (in insertion order)."""
        return self._keys.copy()

    def values(self):
        """Return a list of the stored tensors (in key order)."""
        return [getattr(self, f"_buf_{k}") for k in self._keys]

    def items(self):
        """Return a list of ``(key, tensor)`` pairs (in key order)."""
        return [(k, getattr(self, f"_buf_{k}")) for k in self._keys]

    def __len__(self):
        return len(self._keys)

    def __repr__(self):
        # Note: the closing literal is "}})" (one stray extra "}") so the
        # rendered repr is not perfectly brace-balanced. Kept as-is to avoid
        # changing user-visible output.
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
    - Rejects an all-False mask (it would mask out all data)
    - Supports device movement via to(), cuda(), cpu()
    - Provides combined mask via __call__()

    Parameters
    ----------
    data : dict, optional
        Initial mask data.
    device : str or torch.device, optional
        Device for tensors. Defaults to
        :func:`torchref.config.get_default_device`.

    Raises
    ------
    ValueError
        On assignment of a mask that is not boolean dtype, or that is
        entirely False (which would mask out all data).

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
        """Set mask tensor, ensuring boolean dtype and correct device.

        The tensor is moved to ``self.device``. A ``None`` value is stored
        as-is (no validation).

        Raises
        ------
        ValueError
            If ``tensor`` is not boolean dtype, or if it is all-False
            (which would mask out all data).
        """
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

        # Refresh the ``device`` tracker (future ``__setitem__`` calls migrate
        # incoming tensors to it) through the shared helper rather than a local
        # copy: it also handles the empty-``TensorMasks`` case, where there is
        # no mask tensor to read a device from and the tracker has to come from
        # the recorded ``.to()`` request.
        from torchref.utils.device_mixin import _refresh_device_trackers

        _refresh_device_trackers(self, fn)
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


