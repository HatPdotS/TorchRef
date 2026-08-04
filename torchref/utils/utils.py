"""
Core utility containers and atom-selection parsing, re-exported from ``torchref.utils``.

- :class:`ModuleReference` -- reference an ``nn.Module`` without registering it as a
  submodule, keeping its parameters out of the parent tree.
- :class:`TensorDict` -- dict-like tensor container backed by ``nn.Module`` buffers.
- :class:`TensorMasks` -- ``dict`` of boolean masks with device movement and a cached
  combined (logical-AND) mask.
- :func:`sanitize_pdb_dataframe` -- repair duplicate atom identifiers and over-long
  residue names in a PDB/CIF DataFrame.
- :func:`parse_phenix_selection` / :func:`create_selection_mask` -- Phenix-style
  atom-selection strings to boolean masks.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
import torch

from torchref.utils.device_mixin import DeviceMovementMixin


class ModuleReference:
    """
    Hold a reference to an ``nn.Module`` without registering it as a submodule.

    Assigning an ``nn.Module`` to an attribute of another registers it, adding its
    parameters to the parent's tree; wrapping it here does not. Attribute access and
    ``__call__`` are forwarded, so a wrapped module is mostly a drop-in -- but it is
    absent from ``state_dict`` and from ``.to()``, so the referent must be moved by
    whoever owns it.

    Attributes
    ----------
    _wrapped_module : torch.nn.Module
        The wrapped PyTorch module.
    """

    def __init__(self, module):
        """Wrap ``module`` to prevent automatic submodule registration."""
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


import torch.nn as nn


class TensorDict(nn.Module):
    """A dictionary-like container for PyTorch tensors.

    Backed by :class:`torch.nn.Module`: each stored tensor is registered as a buffer, so
    the container's tensors move with the module and appear in ``state_dict``. Standard
    dict-style access is supported and key insertion order is preserved.

    Parameters
    ----------
    initial_dict : dict of str to torch.Tensor, optional
        Initial key/tensor pairs to populate the container.
    """

    def __init__(self, initial_dict: Optional[Dict[str, torch.Tensor]] = None):
        super().__init__()
        self._keys = []
        if initial_dict:
            for k, v in initial_dict.items():
                self[k] = v

    def __setitem__(self, key: str, tensor: torch.Tensor):
        """Store ``tensor`` under ``key`` as a registered buffer.

        On an existing key of the *same* shape the value is copied **in place**, so a
        previously-read reference to ``self[key]`` sees the new data; a shape change
        re-registers the buffer instead, and old references then go stale.
        """
        name = f"_buf_{key}"
        if not hasattr(self, name):
            self.register_buffer(name, tensor)
            self._keys.append(key)
        else:
            existing = getattr(self, name)
            if existing.shape == tensor.shape:
                existing.data.copy_(tensor)
            else:
                delattr(self, name)
                self.register_buffer(name, tensor)

    def __getitem__(self, key: str) -> torch.Tensor:
        """Return the tensor stored under ``key``; ``KeyError`` if absent."""
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
        # The closing literal is "}})" -- one stray "}", kept so the user-visible repr
        # does not change.
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
    A ``dict`` of boolean mask tensors with device movement and a combined mask.

    Every stored mask is forced to ``self.device``; calling the instance returns the
    logical AND of all masks, cached until the next assignment or ``.to()``.

    Parameters
    ----------
    data : dict, optional
        Initial mask data.
    device : str or torch.device, optional
        Device for tensors. Defaults to :func:`torchref.config.get_default_device`.

    Raises
    ------
    ValueError
        On assignment of a mask that is not boolean dtype, or that is entirely False
        (which would mask out all data).
    """

    def __init__(self, data=None, device=None):
        super().__init__()
        from torchref.config import normalize_device

        # ``normalize_device`` rather than ``torch.device(...)``: the latter
        # keeps an un-indexed spelling ("mps"), which compares unequal to the
        # indexed device every real tensor reports.
        self.device = normalize_device(device)
        self._cache = None
        self._updated = True

        # Initialize with provided data
        if data:
            for k, v in data.items():
                self[k] = v

    def __setitem__(self, key: str, tensor: torch.Tensor):
        """Store a boolean mask, moved to ``self.device``.

        A ``None`` value is stored as-is, unvalidated.

        Raises
        ------
        ValueError
            If ``tensor`` is not boolean dtype, or is all-False (which would mask out
            all data).
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

        Needed because the masks live in the ``dict``'s own storage, not in
        ``self.__dict__``, so the standard :class:`DeviceMixin` walk moves only the
        cached combined mask and leaves the per-key masks behind.
        """
        for k in list(self.keys()):
            v = self[k]
            if isinstance(v, torch.Tensor):
                dict.__setitem__(self, k, fn(v))

        self._cache = None
        self._updated = True

        # Shared helper rather than a local device read: it also covers the empty
        # ``TensorMasks``, where the tracker must come from the recorded ``.to()``.
        from torchref.utils.device_mixin import _refresh_device_trackers

        _refresh_device_trackers(self, fn)
        return self

    def reset_cache(self) -> None:
        """Invalidate the cached combined mask."""
        self._cache = None
        self._updated = True

    def __call__(self) -> torch.Tensor:
        """Combined boolean mask (AND of all masks), or ``None`` if there are none."""
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
    Repair a PDB/CIF DataFrame so ``(chainid, resseq, name, altloc)`` is unique.

    Fixes duplicate ``resseq`` on HETATM records (waters are often all 0) by renumbering
    within the chain, and truncates residue names to 3 characters. Returns a copy; the
    input is not modified.

    Parameters
    ----------
    pdb : pandas.DataFrame
        DataFrame with PDB data (must have columns: ATOM, chainid, resseq, name, altloc,
        resname, serial).
    verbose : int, default 0
        Verbosity level (0=silent, 1=info, 2=debug).

    Returns
    -------
    pandas.DataFrame
        Sanitized copy. Renumbering can fail to converge on pathological input, in which
        case duplicates remain and a warning is printed at ``verbose > 0``.
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
    dup_mask = pdb.duplicated(
        subset=["chainid", "resseq", "name", "altloc"], keep=False
    )

    if dup_mask.any():
        n_dup = dup_mask.sum()
        if verbose > 0:
            print(f"  Found {n_dup} atoms with duplicate identifiers")

        # This ensures we only renumber within the same molecule type and chain
        for (chainid, resname, atom_type), group in pdb.groupby(
            ["chainid", "resname", "ATOM"]
        ):
            group_indices = group.index

            group_dup_mask = group.duplicated(
                subset=["chainid", "resseq", "name", "altloc"], keep=False
            )

            if group_dup_mask.any():
                chain_data = pdb[pdb["chainid"] == chainid]
                max_resseq = chain_data["resseq"].max()

                new_resseq_start = (
                    max_resseq + 1 if pd.notna(max_resseq) and max_resseq > 0 else 1
                )

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

    if selection_string.startswith("__MASK_") and selection_string.endswith("__"):
        if hasattr(_parse_with_parentheses, "_mask_cache"):
            return _parse_with_parentheses._mask_cache.get(
                selection_string, torch.ones(len(pdb_df), dtype=torch.bool)
            )
        return torch.ones(len(pdb_df), dtype=torch.bool)

    if selection_string.lower() == "all":
        return torch.ones(len(pdb_df), dtype=torch.bool)

    # Priority: not > and > or

    if " or " in selection_string.lower():
        parts = re.split(r"\s+or\s+", selection_string, flags=re.IGNORECASE)
        masks = [_parse_without_parentheses(part.strip(), pdb_df) for part in parts]
        result = masks[0]
        for mask in masks[1:]:
            result = result | mask
        return result

    if " and " in selection_string.lower():
        parts = re.split(r"\s+and\s+", selection_string, flags=re.IGNORECASE)
        masks = [_parse_without_parentheses(part.strip(), pdb_df) for part in parts]
        result = masks[0]
        for mask in masks[1:]:
            result = result & mask
        return result

    if selection_string.lower().startswith("not "):
        inner_selection = selection_string[4:].strip()
        return ~_parse_without_parentheses(inner_selection, pdb_df)

    parts = selection_string.split(None, 1)
    if len(parts) < 2:
        raise ValueError(f"Invalid selection syntax: '{selection_string}'")

    keyword, value = parts[0].lower(), parts[1]

    mask = torch.zeros(len(pdb_df), dtype=torch.bool)

    if keyword == "chain":
        chain_id = value.strip()
        selected = pdb_df["chainid"] == chain_id
        mask = torch.tensor(selected.values, dtype=torch.bool)

    elif keyword == "resseq":
        if ":" in value:
            start, end = value.split(":")
            start, end = int(start.strip()), int(end.strip())
            selected = (pdb_df["resseq"] >= start) & (pdb_df["resseq"] <= end)
        else:
            resseq_num = int(value.strip())
            selected = pdb_df["resseq"] == resseq_num
        mask = torch.tensor(selected.values, dtype=torch.bool)

    elif keyword == "resname":
        resname = value.strip().upper()
        selected = pdb_df["resname"].str.upper() == resname
        mask = torch.tensor(selected.values, dtype=torch.bool)

    elif keyword == "name":
        atom_name = value.strip().upper()
        selected = pdb_df["name"].str.upper() == atom_name
        mask = torch.tensor(selected.values, dtype=torch.bool)

    elif keyword == "element":
        element = value.strip().capitalize()
        selected = pdb_df["element"].str.capitalize() == element
        mask = torch.tensor(selected.values, dtype=torch.bool)

    elif keyword == "altloc":
        altloc = value.strip()
        selected = pdb_df["altloc"] == altloc
        mask = torch.tensor(selected.values, dtype=torch.bool)

    else:
        raise ValueError(f"Unknown selection keyword: '{keyword}'")

    return mask


def parse_phenix_selection(selection_string: str, pdb_df: pd.DataFrame) -> torch.Tensor:
    """
    Parse Phenix-style atom selection syntax and return a boolean mask.

    The grammar is the contract, so it is spelled out. Terms:
    ``chain <id>``, ``resseq <num>``, ``resseq <start>:<end>`` (inclusive),
    ``resname <name>``, ``name <atom>``, ``element <elem>``, ``altloc <id>``, ``all``.
    Combined with ``not``, ``and``, ``or`` (that precedence) and ``(...)`` for grouping,
    e.g. ``"chain A and (name CA or name CB)"``. ``resname``/``name`` match
    case-insensitively, ``chain``/``altloc`` do not.

    Parameters
    ----------
    selection_string : str
        Phenix-style selection string.
    pdb_df : pandas.DataFrame
        Atomic data with columns 'chainid', 'resseq', 'resname', 'name', 'element',
        'altloc'.

    Returns
    -------
    torch.Tensor
        Boolean tensor of shape (n_atoms,), on the CPU regardless of where ``pdb_df``'s
        consumers live.

    Raises
    ------
    ValueError
        On an unknown keyword, an empty selection, or a bare term with no value.
    """
    # Clear any cached masks from previous calls
    if hasattr(_parse_with_parentheses, "_mask_cache"):
        _parse_with_parentheses._mask_cache.clear()

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
    Create or modify a refinable mask from a Phenix-style selection.

    Parameters
    ----------
    selection_string : str
        Phenix-style selection string (see :func:`parse_phenix_selection`).
    pdb_df : pandas.DataFrame
        DataFrame containing atomic data.
    current_mask : torch.Tensor, optional
        Current refinable mask. If None, starts with all False. Never mutated -- a new
        tensor is returned.
    mode : str, default 'set'
        How to combine with ``current_mask``: ``'set'`` replaces it with the selection
        (and so ignores it entirely), ``'add'`` ORs, ``'remove'`` AND-NOTs.

    Returns
    -------
    torch.Tensor
        Updated boolean mask of shape (n_atoms,).

    Raises
    ------
    ValueError
        If ``mode`` is not one of 'set', 'add', 'remove'.
    """
    selection_mask = parse_phenix_selection(selection_string, pdb_df)

    if current_mask is None:
        current_mask = torch.zeros(len(pdb_df), dtype=torch.bool)

    if mode == "set":
        return selection_mask
    elif mode == "add":
        return current_mask | selection_mask
    elif mode == "remove":
        return current_mask & ~selection_mask
    else:
        raise ValueError(f"Invalid mode: '{mode}'. Must be 'set', 'add', or 'remove'")


