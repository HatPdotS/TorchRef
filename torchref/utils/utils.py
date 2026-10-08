"""
Core utility containers and atom-table sanitizing, re-exported from ``torchref.utils``.

- :class:`ModuleReference` -- reference an ``nn.Module`` without registering it as a
  submodule, keeping its parameters out of the parent tree.
- :class:`TensorDict` -- dict-like tensor container backed by ``nn.Module`` buffers.
- :class:`TensorMasks` -- ``dict`` of boolean masks with device movement and a cached
  combined (logical-AND) mask.
- :func:`sanitize_pdb_dataframe` -- renumber HETATM residues whose atom identifiers
  repeat, and truncate over-long residue names, before an atom table is written.
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
    parameters to the parent's tree; wrapping it here does not. Public attribute access
    and ``__call__`` are forwarded, so a wrapped module is mostly a drop-in -- but it is
    absent from ``state_dict`` and from ``.to()``, so the referent must be moved by
    whoever owns it. ``copy.copy`` shares the referent; ``copy.deepcopy`` and pickle
    copy it, through the memo, so a deep copy of a whole object graph stays consistent.

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
        """Forward public attribute access to the wrapped module."""
        # Underscore names stop here: DeviceMixin probes ``_apply``/``_data`` to decide
        # what to move, and copy/pickle probe ``__setstate__`` on an instance whose
        # ``_wrapped_module`` is not set yet, where ``self.module`` would recurse.
        if name.startswith("_"):
            raise AttributeError(
                f"ModuleReference does not forward {name!r}; read it from .module"
            )
        return getattr(self.__dict__.get("_wrapped_module"), name)

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
        re-registers the buffer instead, and old references then go stale. The in-place
        copy bumps the buffer's version: cached forwards that read it recompute, and a
        graph that saved the old value can no longer be backpropagated.
        """
        name = f"_buf_{key}"
        if not hasattr(self, name):
            self.register_buffer(name, tensor)
            self._keys.append(key)
        else:
            existing = getattr(self, name)
            if existing.shape == tensor.shape:
                # Not ``.data.copy_``: only a tracked write bumps ``_version``, which is
                # how a cached forward that read this buffer learns it changed.
                with torch.no_grad():
                    existing.copy_(tensor)
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
        return (
            "TensorDict({"
            + ", ".join(f'{k}: {getattr(self, f"_buf_{k}")}' for k in self._keys)
            + "})"
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

    # ``dict`` routes none of its other mutators through ``__setitem__``, so each needs
    # its own override: insertions must be validated, moved and invalidate the combined
    # mask exactly as assignment does, and without the removal overrides a removed mask
    # keeps constraining ``__call__``'s cached result until something else assigns.
    def update(self, *args, **kwargs):
        for key, tensor in dict(*args, **kwargs).items():
            self[key] = tensor

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self[key]

    def __ior__(self, other):
        self.update(other)
        return self

    def popitem(self):
        out = super().popitem()
        self._updated = True
        return out

    def __delitem__(self, key: str):
        super().__delitem__(key)
        self._updated = True

    def pop(self, key, *default):
        out = super().pop(key, *default)
        self._updated = True
        return out

    def clear(self):
        super().clear()
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
        return self._and(self.values())

    @staticmethod
    def _and(masks) -> torch.Tensor:
        """Logical AND over an iterable of masks, skipping ``None``."""
        masks = [v for v in masks if v is not None]
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


#: What identifies one atom in a PDB or mmCIF file.
_ATOM_KEY = ["chainid", "resseq", "icode", "name", "altloc"]

#: What the rows of one residue share.
_RESIDUE_KEY = ["chainid", "resseq", "icode", "resname"]


def _residue_blocks(pdb: pd.DataFrame) -> np.ndarray:
    """Residue index of every row, shape ``(n_atoms,)``, non-decreasing down the table.

    A residue is a contiguous run of one ``(chainid, resseq, icode, resname)``, split
    wherever an atom ``(name, altloc)`` repeats: unnumbered waters share that whole key,
    yet each is a residue of its own.
    """
    keys = pdb.groupby(_RESIDUE_KEY, sort=False, dropna=False).ngroup().to_numpy()
    names = ["name", "altloc"]
    atoms = pdb.groupby(names, sort=False, dropna=False).ngroup().to_numpy()
    blocks = np.empty(len(pdb), dtype=np.int64)
    block, current, seen = -1, None, set()
    for row, (key, atom) in enumerate(zip(keys.tolist(), atoms.tolist())):
        if key != current or atom in seen:
            block, current, seen = block + 1, key, set()
        seen.add(atom)
        blocks[row] = block
    return blocks


def sanitize_pdb_dataframe(pdb: pd.DataFrame, verbose: int = 0) -> pd.DataFrame:
    """
    Prepare an atom table for writing: unique HETATM residues, 3-character names.

    Truncates residue names to the 3 characters a PDB file holds. Then each HETATM
    residue that repeats an atom identifier ``(chainid, resseq, icode, name, altloc)``
    already taken by an ATOM record or an earlier residue -- typically waters all
    numbered 0, or a ligand copied without renumbering -- gets a new ``resseq``, counting
    up from the chain's highest. A residue is a contiguous run of rows sharing
    ``(chainid, resseq, icode, resname)``, split where an atom ``(name, altloc)``
    repeats, so it moves whole and a run of unnumbered waters becomes one residue per
    water. ATOM records are never renumbered, and residues kept apart by an insertion
    code (52 and 52A) are not duplicates. Returns a copy; the input is not modified.

    Parameters
    ----------
    pdb : pandas.DataFrame
        Atom table with columns ATOM, chainid, resseq, icode, resname, name and altloc.
    verbose : int, default 0
        Verbosity level (0=silent, 1=info, 2=debug).

    Returns
    -------
    pandas.DataFrame
        Sanitized copy. Duplicated identifiers among ATOM records are left as they are,
        with a warning printed at ``verbose > 0``.
    """
    pdb = pdb.copy()

    if verbose > 0:
        print("Sanitizing PDB DataFrame...")
        print(f"  Initial atoms: {len(pdb)}")

    long_resnames = pdb["resname"].str.len() > 3
    if long_resnames.any():
        n_long = long_resnames.sum()
        if verbose > 0:
            unique_long = pdb.loc[long_resnames, "resname"].unique()
            print(
                f"  Truncating {n_long} atoms with resname > 3 chars: {unique_long[:5]}"
            )
        pdb.loc[long_resnames, "resname"] = pdb.loc[long_resnames, "resname"].str[:3]

    het = (pdb["ATOM"].astype(str).str.strip() == "HETATM").to_numpy()
    residue = _residue_blocks(pdb)
    # ATOM records come first, so of a polymer residue and a HETATM residue that
    # collide it is always the HETATM one that moves.
    order = np.argsort(het, kind="stable")
    taken = np.empty(len(pdb), dtype=bool)
    taken[order] = pdb.iloc[order].duplicated(subset=_ATOM_KEY).to_numpy()
    moved = np.unique(residue[taken & het])

    if len(moved):
        first_row = np.searchsorted(residue, moved)
        chain = pd.Series(pdb["chainid"].to_numpy()[first_row])
        by_chain = pdb.groupby("chainid", sort=False, dropna=False)["resseq"]
        top = by_chain.transform("max").to_numpy()[first_row]
        offset = chain.groupby(chain, sort=False, dropna=False).cumcount().to_numpy()
        new = np.where(top > 0, top + 1, 1) + offset
        rows = np.isin(residue, moved)
        new_resseq = pd.Series(new, index=moved).loc[residue[rows]]
        pdb.loc[rows, "resseq"] = new_resseq.to_numpy()
        if verbose > 0:
            print(
                f"  Renumbered {len(moved)} HETATM residues ({rows.sum()} atoms) "
                "whose atom identifiers were already taken"
            )
        if verbose > 1:
            for chainid, numbers in pd.Series(new).groupby(chain, dropna=False):
                print(f"    chain {chainid}: resseq {numbers.min()}-{numbers.max()}")
    elif verbose > 0:
        print("  No HETATM residue needed renumbering")

    if verbose > 0:
        remaining = pdb.duplicated(subset=_ATOM_KEY, keep=False)
        if remaining.any():
            print(
                f"  WARNING: {remaining.sum()} ATOM records share an atom identifier "
                "and are left as they are"
            )
            print(pdb.loc[remaining, ["ATOM", "resname", *_ATOM_KEY]].head(10))
        print(f"  Final atoms: {len(pdb)}")

    return pdb
