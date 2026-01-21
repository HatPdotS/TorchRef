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
::

    from torchref.symmetry.spacegroup import SpaceGroup

    # Create from various inputs
    sg = SpaceGroup('P 21')
    sg = SpaceGroup('P21')  # Same result
    sg = SpaceGroup(19)     # From space group number

    # Access properties
    print(sg.hm)            # 'P 21 21 21' (Hermann-Mauguin)
    print(sg.number)        # 19
    print(sg.short_name())  # 'P212121'
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
    ::

        sg = SpaceGroup('P21')
        sg = SpaceGroup('P 21')      # Same as above
        sg = SpaceGroup(4)           # P21 by number
        sg = SpaceGroup(None)        # Returns P1
        sg2 = SpaceGroup(sg)         # Pass-through
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


# =============================================================================
# Grid size utilities (combined FFT-friendly and symmetry-friendly)
# =============================================================================


def is_fft_friendly(n: int) -> bool:
    """
    Check if a number has only factors of 2, 3, and 5.

    These are optimal for radix-2,3,5 FFT algorithms used by PyTorch/cuFFT.

    Parameters
    ----------
    n : int
        Number to check.

    Returns
    -------
    bool
        True if n has only factors of 2, 3, 5.

    Examples
    --------
    ::

        is_fft_friendly(128)  # True (2^7)
        is_fft_friendly(135)  # True (3^3 * 5)
        is_fft_friendly(131)  # False (prime)
    """
    if n <= 0:
        return False

    # Remove all factors of 2, 3, 5
    while n % 2 == 0:
        n //= 2
    while n % 3 == 0:
        n //= 3
    while n % 5 == 0:
        n //= 5

    # If we're left with 1, the number is FFT-friendly
    return n == 1


def find_fft_friendly_size(n: int, divisibility: int = 1) -> int:
    """
    Find the nearest FFT-friendly size >= n that satisfies divisibility constraint.

    FFT-friendly means factors only of 2, 3, and 5 (radix-2,3,5 FFT algorithms).

    Parameters
    ----------
    n : int
        Minimum grid size.
    divisibility : int, default 1
        Required divisibility (e.g., 2 for screw axes).

    Returns
    -------
    int
        Optimal grid size.

    Examples
    --------
    ::

        find_fft_friendly_size(131)      # 135
        find_fft_friendly_size(131, 2)   # 160 (divisible by 2, FFT-friendly)
    """
    candidate = n

    # Make sure it satisfies divisibility
    if candidate % divisibility != 0:
        candidate = ((candidate // divisibility) + 1) * divisibility

    # Now find nearest FFT-friendly size
    while not is_fft_friendly(candidate):
        candidate += divisibility

    return candidate


def get_grid_requirements(spacegroup: SpaceGroupLike) -> dict:
    """
    Analyze symmetry operations to determine grid size requirements.

    Examines all rotation matrices and translations to determine which
    grid dimensions must satisfy divisibility constraints for exact
    integer indexing (interpolation-free symmetry expansion).

    Parameters
    ----------
    spacegroup : SpaceGroupLike
        Space group in any supported format.

    Returns
    -------
    dict
        {'nx_mod': int, 'ny_mod': int, 'nz_mod': int}
        Required divisibility for each axis.

    Examples
    --------
    ::

        get_grid_requirements('P21')
        # {'nx_mod': 1, 'ny_mod': 2, 'nz_mod': 1}

        get_grid_requirements('P212121')
        # {'nx_mod': 2, 'ny_mod': 2, 'nz_mod': 2}
    """
    import math
    from fractions import Fraction

    sg = SpaceGroup(spacegroup)

    # Start with no requirements
    nx_lcm = 1
    ny_lcm = 1
    nz_lcm = 1

    # Analyze each symmetry operation
    for op in sg.operations():
        # gemmi stores translations as integers multiplied by 24
        trans = [t / 24.0 for t in op.tran]

        # For each axis, check if translation has fractional component
        for axis_idx, t in enumerate(trans):
            if abs(t) > 1e-9:
                # Convert to fraction and get denominator
                frac = Fraction(t).limit_denominator(24)
                denom = frac.denominator

                if axis_idx == 0:
                    nx_lcm = math.lcm(nx_lcm, denom)
                elif axis_idx == 1:
                    ny_lcm = math.lcm(ny_lcm, denom)
                else:
                    nz_lcm = math.lcm(nz_lcm, denom)

    return {"nx_mod": nx_lcm, "ny_mod": ny_lcm, "nz_mod": nz_lcm}


def check_grid_compatibility(grid_shape: tuple, spacegroup: SpaceGroupLike) -> dict:
    """
    Check if a grid is compatible with space group symmetry and FFT requirements.

    Verifies that the grid satisfies both:
    1. Symmetry requirements (divisibility for screw axes)
    2. FFT-friendly sizes (factors of 2, 3, 5 only)

    Parameters
    ----------
    grid_shape : tuple of int
        Grid dimensions (nx, ny, nz).
    spacegroup : SpaceGroupLike
        Space group in any supported format.

    Returns
    -------
    dict
        Dictionary with the following keys:

        - 'compatible' : bool
            True if grid satisfies all requirements.
        - 'symmetry_compatible' : bool
            True if grid satisfies symmetry requirements.
        - 'fft_friendly' : bool
            True if all dimensions are FFT-friendly.
        - 'can_use_direct_indexing' : bool
            True if interpolation-free expansion is possible.
        - 'issues' : list of str
            Descriptions of incompatibilities (empty if compatible).
        - 'requirements' : dict
            Required divisibility from get_grid_requirements().

    Examples
    --------
    ::

        check_grid_compatibility((131, 163, 148), 'P21')
        # {'compatible': False, 'issues': ['ny=163 not divisible by 2', ...]}

        check_grid_compatibility((135, 164, 150), 'P21')
        # {'compatible': True, 'issues': []}
    """
    nx, ny, nz = grid_shape
    sg = SpaceGroup(spacegroup)
    requirements = get_grid_requirements(sg)

    issues = []
    sg_name = sg.short_name()

    # Check symmetry requirements
    if nx % requirements["nx_mod"] != 0:
        issues.append(
            f"nx={nx} not divisible by {requirements['nx_mod']} "
            f"(required for {sg_name} symmetry)"
        )

    if ny % requirements["ny_mod"] != 0:
        issues.append(
            f"ny={ny} not divisible by {requirements['ny_mod']} "
            f"(required for {sg_name} symmetry)"
        )

    if nz % requirements["nz_mod"] != 0:
        issues.append(
            f"nz={nz} not divisible by {requirements['nz_mod']} "
            f"(required for {sg_name} symmetry)"
        )

    symmetry_compatible = len(issues) == 0

    # Check FFT-friendly
    fft_x = is_fft_friendly(nx)
    fft_y = is_fft_friendly(ny)
    fft_z = is_fft_friendly(nz)
    fft_friendly = fft_x and fft_y and fft_z

    if not fft_x:
        issues.append(f"nx={nx} is not FFT-friendly (not a product of 2, 3, 5)")
    if not fft_y:
        issues.append(f"ny={ny} is not FFT-friendly (not a product of 2, 3, 5)")
    if not fft_z:
        issues.append(f"nz={nz} is not FFT-friendly (not a product of 2, 3, 5)")

    return {
        "compatible": symmetry_compatible and fft_friendly,
        "symmetry_compatible": symmetry_compatible,
        "fft_friendly": fft_friendly,
        "can_use_direct_indexing": symmetry_compatible,
        "issues": issues,
        "requirements": requirements,
    }


def suggest_grid_size(
    min_grid_shape: tuple,
    spacegroup: SpaceGroupLike,
    make_fft_friendly: bool = True,
) -> tuple:
    """
    Suggest an optimal grid size that satisfies symmetry and FFT requirements.

    Given a minimum grid size, finds the nearest larger size that:
    1. Satisfies symmetry requirements (divisibility constraints)
    2. Optionally, is FFT-friendly (factors of 2, 3, 5 only)

    Parameters
    ----------
    min_grid_shape : tuple of int
        Minimum (nx, ny, nz) grid dimensions.
    spacegroup : SpaceGroupLike
        Space group in any supported format.
    make_fft_friendly : bool, default True
        If True, ensures result has only factors of 2, 3, 5.

    Returns
    -------
    tuple of int
        Suggested grid dimensions (nx, ny, nz).

    Examples
    --------
    ::

        suggest_grid_size((131, 163, 148), 'P21')
        # (135, 164, 150) or similar

        suggest_grid_size((131, 163, 148), 'P212121')
        # (135, 164, 150) - all divisible by 2 and FFT-friendly
    """
    requirements = get_grid_requirements(spacegroup)

    def find_next_valid(n, divisibility):
        """Find next number >= n that satisfies divisibility and FFT constraints."""
        if n % divisibility == 0:
            candidate = n
        else:
            candidate = ((n // divisibility) + 1) * divisibility

        if not make_fft_friendly:
            return candidate

        # Find FFT-friendly size that also satisfies divisibility
        while not is_fft_friendly(candidate):
            candidate += divisibility

        return candidate

    nx = find_next_valid(min_grid_shape[0], requirements["nx_mod"])
    ny = find_next_valid(min_grid_shape[1], requirements["ny_mod"])
    nz = find_next_valid(min_grid_shape[2], requirements["nz_mod"])

    return (nx, ny, nz)
