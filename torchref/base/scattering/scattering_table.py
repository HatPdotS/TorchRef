"""
Table-based ITC92 scattering factor lookup.

This module provides fast, vectorized lookup of scattering factor parameters
using a pre-computed table stored as a .pt file. The table eliminates
runtime gemmi dependency for scattering parameter access.

The table supports:
- Neutral atoms for Z=1 to 103 (H to Lr)
- Common ions with various charge states
- Element symbol to atomic number mapping

Example
-------
::

    from torchref.base.scattering.scattering_table import (
        load_scattering_table,
        get_scattering_params_by_z,
        get_element_to_z_mapping,
    )

    # Load the table
    table = load_scattering_table(device='cuda')

    # Get parameters for multiple atoms at once
    z_tensor = torch.tensor([6, 7, 8])  # C, N, O
    A, B = get_scattering_params_by_z(z_tensor)
"""

import os
from typing import Dict, Optional, Tuple

import torch

from torchref.config import get_float_dtype

# Global cache for the loaded table
_TABLE_CACHE: Optional[dict] = None


def _get_table_path() -> str:
    """Get the path to the pre-computed scattering table."""
    from torchref import PATH_TORCHREF_DATA

    return os.path.join(PATH_TORCHREF_DATA, "itc92_scattering_factors.pt")


def load_scattering_table(
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    force_reload: bool = False,
) -> dict:
    """
    Load pre-computed ITC92 scattering factors from .pt file.

    The table is cached globally after first load for efficiency.
    Use force_reload=True to reload from disk.

    Parameters
    ----------
    device : torch.device, optional
        Device to place tensors on. Default is None (keeps original device).
    dtype : torch.dtype, optional
        Data type for floating point tensors. Default is None (keeps original dtype).
    force_reload : bool, optional
        Force reload from disk even if cached. Default is False.

    Returns
    -------
    dict
        Dictionary containing:
        - 'A': Tensor(max_z + 1, 5) - neutral A coefficients indexed by Z
        - 'B': Tensor(max_z + 1, 5) - neutral B coefficients indexed by Z
        - 'element_to_z': dict mapping element symbols to atomic numbers
        - 'z_to_element': dict mapping atomic numbers to element symbols
        - 'ions': dict mapping ion keys to (A, B) tuples
        - 'metadata': dict with source information

    Raises
    ------
    FileNotFoundError
        If the pre-computed table file does not exist.

    Examples
    --------
    ::

        table = load_scattering_table(device='cuda', dtype=torch.float32)
        A = table['A']  # Shape (104, 5)
        z_to_elem = table['z_to_element']  # {1: 'H', 6: 'C', ...}
    """
    global _TABLE_CACHE

    if _TABLE_CACHE is not None and not force_reload:
        table = _TABLE_CACHE
    else:
        table_path = _get_table_path()
        if not os.path.exists(table_path):
            raise FileNotFoundError(
                f"Scattering factor table not found at {table_path}. "
                "Run 'python -m torchref.scripts.generate_scattering_table' to generate it."
            )

        table = torch.load(table_path, map_location="cpu", weights_only=False)
        _TABLE_CACHE = table

    # Apply device/dtype transformations if requested
    if device is not None or dtype is not None:
        result = {}
        for key, value in table.items():
            if isinstance(value, torch.Tensor):
                if device is not None:
                    value = value.to(device=device)
                if dtype is not None and value.is_floating_point():
                    value = value.to(dtype=dtype)
                result[key] = value
            elif key == "ions" and isinstance(value, dict):
                # Transform ion tensors
                ions_result = {}
                for ion_key, (A, B) in value.items():
                    if device is not None:
                        A = A.to(device=device)
                        B = B.to(device=device)
                    if dtype is not None:
                        A = A.to(dtype=dtype)
                        B = B.to(dtype=dtype)
                    ions_result[ion_key] = (A, B)
                result[key] = ions_result
            else:
                result[key] = value
        return result

    return table


def get_element_to_z_mapping() -> Dict[str, int]:
    """
    Return element symbol to atomic number mapping.

    This function loads the scattering table if not already cached
    and returns the element_to_z dictionary.

    Returns
    -------
    dict
        Mapping of element symbols to atomic numbers.
        Example: {'H': 1, 'C': 6, 'N': 7, 'O': 8, ...}

    Examples
    --------
    ::

        element_to_z = get_element_to_z_mapping()
        z_carbon = element_to_z['C']  # Returns 6
    """
    table = load_scattering_table()
    return table["element_to_z"]


def get_z_to_element_mapping() -> Dict[int, str]:
    """
    Return atomic number to element symbol mapping.

    Returns
    -------
    dict
        Mapping of atomic numbers to element symbols.
        Example: {1: 'H', 6: 'C', 7: 'N', 8: 'O', ...}
    """
    table = load_scattering_table()
    return table["z_to_element"]


def get_scattering_params_by_z(
    z_tensor: torch.Tensor,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fast vectorized lookup of scattering parameters by atomic number.

    This function provides efficient batch lookup of ITC92 scattering
    parameters using tensor indexing. All atoms are looked up in a
    single operation.

    Parameters
    ----------
    z_tensor : torch.Tensor
        Atomic numbers for all atoms, shape (n_atoms,).
        Each value must be a valid table index in range 0-103 (the table is
        sized ``max_z + 1`` = 104). Z=1..103 are the elements; Z=0 is the
        reserved "unknown element" slot that ``elements_to_z`` maps unknown
        symbols to.
    device : torch.device, optional
        Device to place output tensors on. Default uses z_tensor's device.
    dtype : torch.dtype, optional
        Data type for output tensors. Default is the configured float dtype
        (``get_float_dtype()``).

    Returns
    -------
    A : torch.Tensor
        ITC92 A parameters (amplitudes) with shape (n_atoms, 5).
    B : torch.Tensor
        ITC92 B parameters (widths) with shape (n_atoms, 5).

    Examples
    --------
    ::

        # Get parameters for C, N, O atoms
        z = torch.tensor([6, 7, 8, 6, 6, 7])
        A, B = get_scattering_params_by_z(z)
        # A.shape == (6, 5), B.shape == (6, 5)

        # Use on GPU
        z_gpu = z.cuda()
        A, B = get_scattering_params_by_z(z_gpu, device='cuda')
    """
    if device is None:
        device = z_tensor.device
    if dtype is None:
        dtype = get_float_dtype()

    table = load_scattering_table(device=device, dtype=dtype)

    # Ensure z_tensor is on the right device and is long type for indexing
    z_idx = z_tensor.to(device=device, dtype=torch.long)

    # Direct tensor indexing for vectorized lookup
    A = table["A"][z_idx]
    B = table["B"][z_idx]

    return A, B


def get_scattering_params_for_ion(
    element: str,
    charge: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Get scattering parameters for a specific ion.

    Parameters
    ----------
    element : str
        Element symbol (e.g., 'Fe', 'O').
    charge : int
        Ionic charge (positive or negative integer).
    device : torch.device, optional
        Device to place tensors on.
    dtype : torch.dtype, optional
        Data type for tensors.

    Returns
    -------
    tuple or None
        (A, B) tensors of shape (5,), or None if ion not found.

    Examples
    --------
    ::

        # Get Fe2+ parameters
        A, B = get_scattering_params_for_ion('Fe', 2)

        # Get O2- parameters
        A, B = get_scattering_params_for_ion('O', -2)
    """
    if dtype is None:
        dtype = get_float_dtype()

    table = load_scattering_table(device=device, dtype=dtype)

    # Build ion key
    if charge > 0:
        key = f"{element}{charge}+"
    elif charge < 0:
        key = f"{element}{abs(charge)}-"
    else:
        # For neutral, use Z-based lookup
        element_to_z = table["element_to_z"]
        z = element_to_z.get(element)
        if z is None:
            return None
        A = table["A"][z]
        B = table["B"][z]
        return A, B

    ions = table.get("ions", {})
    if key in ions:
        return ions[key]

    return None


def elements_to_z(elements: list, normalize: bool = True) -> torch.Tensor:
    """
    Convert a list of element symbols to atomic numbers.

    Parameters
    ----------
    elements : list of str
        Element symbols (e.g., ['C', 'N', 'O', 'C', 'C']).
    normalize : bool, optional
        If True, normalize element names (strip whitespace, capitalize).
        Default is True.

    Returns
    -------
    torch.Tensor
        Tensor of atomic numbers with shape (n_atoms,).
        Unknown elements are assigned Z=0.

    Examples
    --------
    ::

        z = elements_to_z(['C', 'N', 'O'])
        # Returns tensor([6, 7, 8])

        z = elements_to_z([' c ', 'N', 'O'], normalize=True)
        # Returns tensor([6, 7, 8])
    """
    element_to_z = get_element_to_z_mapping()

    z_values = []
    for elem in elements:
        if normalize:
            elem = elem.strip().capitalize()
        z = element_to_z.get(elem, 0)
        z_values.append(z)

    return torch.tensor(z_values, dtype=torch.int32)
