"""Table-based ITC92 scattering factor lookup.

Vectorized lookup from a pre-computed ``.pt`` table, which removes gemmi from
the runtime path: neutral atoms Z=1..103 (H to Lr), common ions at several
charge states, and the element-symbol/atomic-number mappings.
:func:`get_scattering_params_by_z` is the batch entry point;
:func:`load_scattering_table` caches the raw table process-wide.
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
    Load pre-computed ITC92 scattering factors from the packaged ``.pt`` file.

    The CPU table is cached process-wide after the first load. Only that raw
    table is cached: passing ``device`` or ``dtype`` builds and returns a fresh
    converted dict on every call, so hoist it out of hot loops yourself.

    Parameters
    ----------
    device : torch.device, optional
        Device to place tensors on. Default None (keep the loaded device).
    dtype : torch.dtype, optional
        Float dtype for tensors. Default None (keep the loaded dtype).
    force_reload : bool, optional
        Re-read from disk even if cached. Default is False.

    Returns
    -------
    dict
        - 'A', 'B': Tensor(max_z + 1, 5), neutral coefficients indexed by Z
        - 'element_to_z' / 'z_to_element': symbol/number mappings
        - 'ions': ion key -> (A, B)
        - 'metadata': source information

    Raises
    ------
    FileNotFoundError
        If the table file is missing (generate it with
        ``python -m torchref.scripts.generate_scattering_table``).
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

    # Converted copies are built per call, never written back to the cache.
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
    Return element symbol to atomic number mapping, loading the table if needed.

    Returns
    -------
    dict
        ``{'H': 1, 'C': 6, 'N': 7, 'O': 8, ...}``. The live cached dict, not a
        copy -- do not mutate it.
    """
    table = load_scattering_table()
    return table["element_to_z"]


def get_z_to_element_mapping() -> Dict[int, str]:
    """
    Return atomic number to element symbol mapping.

    Returns
    -------
    dict
        ``{1: 'H', 6: 'C', 7: 'N', 8: 'O', ...}``. The live cached dict, not a
        copy -- do not mutate it.
    """
    table = load_scattering_table()
    return table["z_to_element"]


def get_scattering_params_by_z(
    z_tensor: torch.Tensor,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized lookup of ITC92 scattering parameters by atomic number.

    One tensor-index operation for all atoms.

    Parameters
    ----------
    z_tensor : torch.Tensor
        Atomic numbers, shape (n_atoms,). Every value must be a valid table
        index in 0-103 (the table is ``max_z + 1`` = 104 rows): 1..103 are the
        elements, 0 is the reserved "unknown element" slot that
        :func:`elements_to_z` assigns. Out-of-range Z raises from the index.
    device : torch.device, optional
        Output device. Default is ``z_tensor``'s.
    dtype : torch.dtype, optional
        Output dtype. Default is ``get_float_dtype()``.

    Returns
    -------
    A : torch.Tensor
        ITC92 A parameters (amplitudes), shape (n_atoms, 5).
    B : torch.Tensor
        ITC92 B parameters (widths), shape (n_atoms, 5).
    """
    if device is None:
        device = z_tensor.device
    if dtype is None:
        dtype = get_float_dtype()

    table = load_scattering_table(device=device, dtype=dtype)

    # Long, not the caller's int32: torch indexing requires it.
    z_idx = z_tensor.to(device=device, dtype=torch.long)  # dtype-ok: z cast to long for scattering-table index lookup; indexing requires long

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
    Get ITC92 scattering parameters for one ion, e.g. ``('Fe', 2)``/``('O', -2)``.

    Parameters
    ----------
    element : str
        Element symbol (e.g. 'Fe', 'O'). Case-sensitive: it is used verbatim as
        a table key, unlike :func:`elements_to_z`.
    charge : int
        Ionic charge. ``0`` falls back to the neutral Z-indexed row.
    device : torch.device, optional
        Device to place tensors on.
    dtype : torch.dtype, optional
        Float dtype for tensors. Default is ``get_float_dtype()``.

    Returns
    -------
    tuple or None
        (A, B) tensors of shape (5,), or None if the ion is not tabulated.
    """
    if dtype is None:
        dtype = get_float_dtype()

    table = load_scattering_table(device=device, dtype=dtype)

    if charge > 0:
        key = f"{element}{charge}+"
    elif charge < 0:
        key = f"{element}{abs(charge)}-"
    else:
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
        int32 atomic numbers, shape (n_atoms,). Unknown elements silently
        become Z=0 (the reserved unknown slot) rather than raising.
    """
    element_to_z = get_element_to_z_mapping()

    z_values = []
    for elem in elements:
        if normalize:
            elem = elem.strip().capitalize()
        z = element_to_z.get(elem, 0)
        z_values.append(z)

    return torch.tensor(z_values, dtype=torch.int32)  # dtype-ok: atomic-number Z categorical codes; fixed int32 lookup keys
