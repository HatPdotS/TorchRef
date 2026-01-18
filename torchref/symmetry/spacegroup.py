"""
Space group utilities using gemmi as the canonical representation.

This module provides a unified interface for space group handling throughout
torchref. All space groups are stored and passed as gemmi.SpaceGroup objects.

gemmi.SpaceGroup objects are:
- Hashable (can be used as dict keys)
- Immutable
- Provide direct access to symmetry operations
- Handle all name aliasing internally

Example
-------
>>> from torchref.symmetry.spacegroup import SpaceGroup
>>>
>>> # Create from various inputs
>>> sg = SpaceGroup('P 21')
>>> sg = SpaceGroup('P21')  # Same result
>>> sg = SpaceGroup(19)     # From space group number
>>>
>>> # Access properties
>>> print(sg.hm)            # 'P 21 21 21' (Hermann-Mauguin)
>>> print(sg.number)        # 19
>>> print(sg.short_name())  # 'P212121'
"""

from typing import Union

import gemmi

# Type alias for space group input
SpaceGroupLike = Union[str, int, gemmi.SpaceGroup, None]


def SpaceGroup(spacegroup: SpaceGroupLike) -> gemmi.SpaceGroup:
    """
    Normalize space group input to a gemmi.SpaceGroup object.

    This is the canonical way to create/convert space groups in torchref.
    Accepts various input formats and returns a gemmi.SpaceGroup.

    Parameters
    ----------
    spacegroup : str, int, gemmi.SpaceGroup, or None
        Space group specification:
        - str: Hermann-Mauguin symbol (e.g., 'P21', 'P 21', 'P212121')
        - int: Space group number (1-230)
        - gemmi.SpaceGroup: Passed through unchanged
        - None: Returns P1

    Returns
    -------
    gemmi.SpaceGroup
        Normalized space group object.

    Raises
    ------
    ValueError
        If the space group cannot be recognized.

    Examples
    --------
    >>> sg = SpaceGroup('P21')
    >>> sg = SpaceGroup('P 21')      # Same as above
    >>> sg = SpaceGroup(4)           # P21 by number
    >>> sg = SpaceGroup(None)        # Returns P1
    >>> sg2 = SpaceGroup(sg)         # Pass-through
    """
    if spacegroup is None:
        return gemmi.SpaceGroup("P 1")

    if isinstance(spacegroup, gemmi.SpaceGroup):
        return spacegroup

    if isinstance(spacegroup, int):
        # Space group number
        try:
            return gemmi.SpaceGroup(spacegroup)
        except Exception as e:
            raise ValueError(f"Invalid space group number: {spacegroup}") from e

    if isinstance(spacegroup, str):
        # Try to parse as string
        # Clean up common variations
        sg_clean = spacegroup.strip()

        # Handle double spaces that sometimes appear
        while "  " in sg_clean:
            sg_clean = sg_clean.replace("  ", " ")

        try:
            return gemmi.SpaceGroup(sg_clean)
        except Exception:
            pass

        # Try without spaces
        sg_nospace = sg_clean.replace(" ", "")
        try:
            return gemmi.SpaceGroup(sg_nospace)
        except Exception:
            pass

        # Try common substitutions
        substitutions = [
            (sg_clean, sg_clean),
            (sg_nospace, sg_nospace),
            (sg_clean.upper(), sg_clean.upper()),
            (sg_nospace.upper(), sg_nospace.upper()),
        ]

        for _, variant in substitutions:
            try:
                return gemmi.SpaceGroup(variant)
            except Exception:
                continue

        raise ValueError(
            f"Space group '{spacegroup}' not recognized. "
            f"Use Hermann-Mauguin notation (e.g., 'P 21', 'P212121', 'C 2 2 21') "
            f"or space group number (1-230)."
        )

    raise TypeError(
        f"spacegroup must be str, int, gemmi.SpaceGroup, or None, "
        f"got {type(spacegroup).__name__}"
    )


def spacegroup_to_str(spacegroup: SpaceGroupLike, style: str = "short") -> str:
    """
    Convert space group to string representation.

    Parameters
    ----------
    spacegroup : SpaceGroupLike
        Space group in any supported format.
    style : str, default 'short'
        Output style:
        - 'short': No spaces (e.g., 'P212121')
        - 'hm': Hermann-Mauguin with spaces (e.g., 'P 21 21 21')
        - 'xhm': Extended Hermann-Mauguin (e.g., 'P 21 21 21')

    Returns
    -------
    str
        Space group name in requested style.
    """
    sg = SpaceGroup(spacegroup)

    if style == "short":
        return sg.short_name()
    elif style == "hm":
        return sg.hm
    elif style == "xhm":
        return sg.xhm()
    else:
        raise ValueError(f"Unknown style: {style}. Use 'short', 'hm', or 'xhm'.")


def get_symmetry_operations(spacegroup: SpaceGroupLike):
    """
    Get symmetry operations from a space group.

    Parameters
    ----------
    spacegroup : SpaceGroupLike
        Space group in any supported format.

    Returns
    -------
    list of gemmi.Op
        List of symmetry operations.
    """
    sg = SpaceGroup(spacegroup)
    return list(sg.operations())


def get_operations_as_tensors(spacegroup: SpaceGroupLike, dtype=None, device=None):
    """
    Get symmetry operations as PyTorch tensors.

    Parameters
    ----------
    spacegroup : SpaceGroupLike
        Space group in any supported format.
    dtype : torch.dtype, optional
        Data type for tensors. Default is torch.float64.
    device : torch.device, optional
        Device for tensors. Default is CPU.

    Returns
    -------
    matrices : torch.Tensor, shape (n_ops, 3, 3)
        Rotation matrices.
    translations : torch.Tensor, shape (n_ops, 3)
        Translation vectors (in fractional coordinates).
    """
    import torch

    if dtype is None:
        dtype = torch.float64
    if device is None:
        device = torch.device("cpu")

    sg = SpaceGroup(spacegroup)

    # Extract rotation matrices and translations from gemmi operations
    # gemmi stores values as integers multiplied by 24, divide to get actual values
    gemmi_ops = [
        (
            torch.tensor(op.rot, dtype=dtype, device=device) / 24.0,
            torch.tensor(op.tran, dtype=dtype, device=device) / 24.0,
        )
        for op in sg.operations()
    ]
    matrices, translations = zip(*gemmi_ops)

    return torch.stack(matrices), torch.stack(translations)


def is_same_spacegroup(sg1: SpaceGroupLike, sg2: SpaceGroupLike) -> bool:
    """
    Check if two space groups are the same.

    Parameters
    ----------
    sg1, sg2 : SpaceGroupLike
        Space groups to compare.

    Returns
    -------
    bool
        True if the space groups are identical.
    """
    return SpaceGroup(sg1).number == SpaceGroup(sg2).number


def get_point_group(spacegroup: SpaceGroupLike) -> str:
    """
    Get the point group symbol for a space group.

    Parameters
    ----------
    spacegroup : SpaceGroupLike
        Space group in any supported format.

    Returns
    -------
    str
        Point group symbol (e.g., '222', 'mmm', '4/mmm').
    """
    sg = SpaceGroup(spacegroup)
    return sg.point_group_hm()


def get_crystal_system(spacegroup: SpaceGroupLike) -> str:
    """
    Get the crystal system for a space group.

    Parameters
    ----------
    spacegroup : SpaceGroupLike
        Space group in any supported format.

    Returns
    -------
    str
        Crystal system name (triclinic, monoclinic, orthorhombic,
        tetragonal, trigonal, hexagonal, or cubic).
    """
    sg = SpaceGroup(spacegroup)
    return sg.crystal_system_str()


def is_centrosymmetric(spacegroup: SpaceGroupLike) -> bool:
    """
    Check if a space group is centrosymmetric.

    Parameters
    ----------
    spacegroup : SpaceGroupLike
        Space group in any supported format.

    Returns
    -------
    bool
        True if the space group has an inversion center.
    """
    sg = SpaceGroup(spacegroup)
    return sg.centrosymmetric()


def n_operations(spacegroup: SpaceGroupLike) -> int:
    """
    Get the number of symmetry operations in a space group.

    Parameters
    ----------
    spacegroup : SpaceGroupLike
        Space group in any supported format.

    Returns
    -------
    int
        Number of symmetry operations.
    """
    sg = SpaceGroup(spacegroup)
    return len(list(sg.operations()))
