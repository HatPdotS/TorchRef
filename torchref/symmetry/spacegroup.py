"""Space group utilities using gemmi as the canonical representation.

:class:`SpaceGroup` is the interface used throughout torchref: an ``nn.Module``
that normalizes its input (string, int, ``gemmi.SpaceGroup``, another
``SpaceGroup``, or None for P1), holds the rotation matrices and translations as
registered buffers, and applies them. The module-level functions here are
stateless equivalents plus the FFT/symmetry grid-size helpers.

Real and reciprocal space use *transposed* conventions -- ``x' = R·x + t`` versus
``h' = Rᵀ·h`` -- so :meth:`SpaceGroup.apply` and :meth:`SpaceGroup.apply_to_hkl`
are not interchangeable.
"""

from __future__ import annotations

from typing import Union

import gemmi
import torch
import torch.nn as nn

from torchref.config import get_float_dtype, normalize_device
from torchref.utils.debug_utils import DebugMixin
from torchref.utils.device_mixin import DeviceMovementMixin

# Type alias for space group input - includes SpaceGroup class itself
SpaceGroupLike = Union[str, int, gemmi.SpaceGroup, "SpaceGroup", None]


def _normalize_spacegroup(spacegroup: SpaceGroupLike) -> gemmi.SpaceGroup:
    """Normalize any ``SpaceGroupLike`` to a ``gemmi.SpaceGroup``.

    Accepts a Hermann-Mauguin string (spacing-insensitive, retried upper-cased),
    a number 1-230, a ``gemmi.SpaceGroup`` (returned unchanged), a
    :class:`SpaceGroup` (unwrapped), or None (P1). Raises ``ValueError`` for an
    unrecognised name/number, ``TypeError`` for any other type.
    """
    if spacegroup is None:
        return gemmi.SpaceGroup("P 1")

    if isinstance(spacegroup, gemmi.SpaceGroup):
        return spacegroup

    # Handle SpaceGroup class instances (forward reference resolved at runtime)
    if hasattr(spacegroup, "_sg_hm") and hasattr(spacegroup, "matrices"):
        return gemmi.find_spacegroup_by_name(spacegroup._sg_hm)

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
        f"spacegroup must be str, int, gemmi.SpaceGroup, SpaceGroup, or None, "
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
        - 'xhm': Extended Hermann-Mauguin, including the setting/cell-choice
          token where applicable (e.g., 'P 1 21 1' for a unique-axis-b setting)

    Returns
    -------
    str
        Space group name in requested style.
    """
    sg = _normalize_spacegroup(spacegroup)

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
    sg = _normalize_spacegroup(spacegroup)
    return list(sg.operations())


def get_operations_as_tensors(
    spacegroup: SpaceGroupLike,
    dtype: torch.dtype = None,
    device: torch.device = None,
):
    """
    Get symmetry operations as PyTorch tensors.

    Parameters
    ----------
    spacegroup : SpaceGroupLike
        Space group in any supported format.
    dtype : torch.dtype, optional
        Data type for tensors. Defaults to the configured ``dtypes.float``.
    device : torch.device, optional
        Device for tensors. Defaults to the configured ``device.current``.

    Returns
    -------
    matrices : torch.Tensor, shape (n_ops, 3, 3)
        Rotation matrices.
    translations : torch.Tensor, shape (n_ops, 3)
        Translation vectors (in fractional coordinates).
    """
    if dtype is None:
        dtype = get_float_dtype()
    device = normalize_device(device)
    sg = _normalize_spacegroup(spacegroup)

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
    return _normalize_spacegroup(sg1).number == _normalize_spacegroup(sg2).number


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
    sg = _normalize_spacegroup(spacegroup)
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
    sg = _normalize_spacegroup(spacegroup)
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
    sg = _normalize_spacegroup(spacegroup)
    return sg.is_centrosymmetric()


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
    sg = _normalize_spacegroup(spacegroup)
    return len(list(sg.operations()))


# =============================================================================
# Grid size utilities (combined FFT-friendly and symmetry-friendly)
# =============================================================================


def is_fft_friendly(n: int) -> bool:
    """True if ``n`` factors into 2, 3 and 5 only (radix-2,3,5 FFT sizes).

    ``n <= 0`` is False; 128 and 135 are True, 131 is not.
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
    """Smallest size >= ``n`` that is FFT-friendly and divisible by ``divisibility``.

    Parameters
    ----------
    n : int
        Minimum grid size.
    divisibility : int, default 1
        Required divisibility (e.g. 2 for a screw axis).

    Returns
    -------
    int
        Optimal grid size (131 -> 135; 131 with divisibility 2 -> 160).
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
    """Per-axis grid divisibility required for interpolation-free symmetry expansion.

    Derived from the denominators of the fractional translations, so a grid meeting
    them indexes symmetry mates at exact integers.

    Returns
    -------
    dict
        ``{'nx_mod': int, 'ny_mod': int, 'nz_mod': int}`` -- e.g. P21 gives
        ``(1, 2, 1)``, P212121 gives ``(2, 2, 2)``.
    """
    import math
    from fractions import Fraction

    sg = _normalize_spacegroup(spacegroup)

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
    """Check a grid against both space-group divisibility and FFT-friendliness.

    Parameters
    ----------
    grid_shape : tuple of int
        Grid dimensions (nx, ny, nz).
    spacegroup : SpaceGroupLike
        Space group in any supported format.

    Returns
    -------
    dict
        ``compatible`` (both tests pass), ``symmetry_compatible``,
        ``fft_friendly``, ``can_use_direct_indexing`` (interpolation-free
        expansion possible -- equal to ``symmetry_compatible``), ``issues``
        (per-axis descriptions, empty when compatible) and ``requirements``
        (from :func:`get_grid_requirements`).
    """
    nx, ny, nz = grid_shape
    sg = _normalize_spacegroup(spacegroup)
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
    """Smallest grid >= ``min_grid_shape`` meeting the symmetry divisibility.

    Parameters
    ----------
    min_grid_shape : tuple of int
        Minimum (nx, ny, nz) grid dimensions.
    spacegroup : SpaceGroupLike
        Space group in any supported format.
    make_fft_friendly : bool, default True
        If True, the result also factors into 2, 3, 5 only.

    Returns
    -------
    tuple of int
        Suggested grid dimensions (nx, ny, nz).
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


# =============================================================================
# SpaceGroup class - unified interface combining normalization and operations
# =============================================================================


class SpaceGroup(DeviceMovementMixin, DebugMixin, nn.Module):
    """
    Unified space group handler for crystallographic symmetry operations.

    Normalizes its input, holds the operations as buffers, applies them to
    fractional coordinates (``__call__``) or Miller indices, and exposes the
    grid-size helpers as methods. Only the derived metadata is retained -- no
    persistent ``gemmi`` reference is held (see :attr:`_gemmi`).

    Parameters
    ----------
    space_group : str, int, gemmi.SpaceGroup, SpaceGroup, or None
        Hermann-Mauguin symbol, number 1-230, gemmi object, another instance,
        or None for P1.
    dtype : torch.dtype, optional
        Data type for matrices and translations. Defaults to the configured
        ``dtypes.float`` (float32 unless ``TORCHREF_DTYPE_FLOAT=float64``).
    device : torch.device, default: configured device.current
        Device for computation.

    Attributes
    ----------
    matrices : torch.Tensor, shape (n_ops, 3, 3)
        Rotation matrices for all symmetry operations (registered buffer).
    translations : torch.Tensor, shape (n_ops, 3)
        Translation vectors for all symmetry operations (registered buffer).
    n_ops : int
        Number of symmetry operations.
    """

    def __init__(
        self,
        space_group: SpaceGroupLike = None,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ):
        super(SpaceGroup, self).__init__()
        if dtype is None:
            dtype = get_float_dtype()
        device = normalize_device(device)
        self._device = device
        self._dtype = dtype

        # Normalize to gemmi.SpaceGroup, extract metadata, then release
        gemmi_sg = _normalize_spacegroup(space_group)
        self._sg_number: int = gemmi_sg.number
        self._sg_hm: str = gemmi_sg.hm
        self._sg_short_name: str = gemmi_sg.short_name()
        self._sg_xhm: str = gemmi_sg.xhm()
        self._sg_point_group: str = gemmi_sg.point_group_hm()
        self._sg_crystal_system: str = gemmi_sg.crystal_system_str()
        self._sg_centrosymmetric: bool = gemmi_sg.is_centrosymmetric()

        # Get symmetry operations as tensors
        matrices, translations = get_operations_as_tensors(
            gemmi_sg, dtype=dtype, device=device
        )

        self.register_buffer("matrices", matrices)
        self.register_buffer("translations", translations)
        # gemmi_sg goes out of scope here — no persistent gemmi reference

    # =========================================================================
    # Core properties
    # =========================================================================

    @property
    def n_ops(self) -> int:
        """Number of symmetry operations."""
        return self.matrices.shape[0]

    @property
    def _gemmi(self) -> gemmi.SpaceGroup:
        """Fresh gemmi.SpaceGroup each access; never cache it -- a persistent
        reference to the C++ singleton produces nanobind leak warnings at shutdown."""
        return gemmi.find_spacegroup_by_name(self._sg_hm)

    @property
    def name(self) -> str:
        """Short space group name (e.g., 'P21')."""
        return self._sg_short_name

    @property
    def hm(self) -> str:
        """Hermann-Mauguin notation with spaces (e.g., 'P 21')."""
        return self._sg_hm

    @property
    def xhm(self) -> str:
        """Extended Hermann-Mauguin notation."""
        return self._sg_xhm

    @property
    def number(self) -> int:
        """Space group number (1-230)."""
        return self._sg_number

    @property
    def gemmi(self) -> gemmi.SpaceGroup:
        """Access a gemmi.SpaceGroup object (created on demand, not stored)."""
        return self._gemmi

    @property
    def point_group(self) -> str:
        """Point group symbol (e.g., '222', 'mmm')."""
        return self._sg_point_group

    @property
    def crystal_system(self) -> str:
        """Crystal system name."""
        return self._sg_crystal_system

    @property
    def centrosymmetric(self) -> bool:
        """True if space group has inversion center."""
        return self._sg_centrosymmetric

    @property
    def dtype(self) -> torch.dtype:
        """Data type used for matrices."""
        return self._dtype

    @property
    def device(self) -> torch.device:
        """Device for matrices."""
        return self._device

    # =========================================================================
    # Backward compatibility aliases
    # =========================================================================

    @property
    def spacegroup(self) -> gemmi.SpaceGroup:
        """Alias for gemmi property (backward compatibility)."""
        return self._gemmi

    @property
    def space_group(self) -> gemmi.SpaceGroup:
        """Alias for gemmi property (backward compatibility)."""
        return self._gemmi

    @property
    def space_group_name(self) -> str:
        """Alias for name property (backward compatibility)."""
        return self.name

    @property
    def space_group_number(self) -> int:
        """Alias for number property (backward compatibility)."""
        return self.number

    # =========================================================================
    # Gemmi method delegation for backward compatibility
    # =========================================================================

    def short_name(self) -> str:
        """Get short space group name."""
        return self._sg_short_name

    def operations(self):
        """Get symmetry operations (creates temporary gemmi object on demand)."""
        return self._gemmi.operations()

    # =========================================================================
    # Symmetry operation methods
    # =========================================================================

    def apply(
        self, xyz_fractional: torch.Tensor, apply_translation: bool = True
    ) -> torch.Tensor:
        """
        Apply symmetry operations to fractional coordinates (rotation + translation).

        For real space coordinates, applies the full symmetry operation: x' = R·x + t

        Parameters
        ----------
        xyz_fractional : torch.Tensor
            Input tensor of shape (N, 3) representing fractional coordinates.
        apply_translation : bool, default True
            If True, apply the full operation x' = R·x + t. If False, apply
            the rotational part only (x' = R·x), as used for Miller indices.

        Returns
        -------
        torch.Tensor
            Transformed coordinates of shape (N, 3, ops) where ops is the
            number of symmetry operations.

        See Also
        --------
        apply_to_hkl : For reciprocal space (Miller indices), rotation only.
        """
        coords = xyz_fractional.to(self.matrices.device).to(self.matrices.dtype)
        # coords: (N, 3), matrices: (ops, 3, 3)
        # Apply rotation: result[n, i, o] = sum_j(matrices[o, i, j] * coords[n, j])
        transformed = torch.einsum("oij,nj->nio", self.matrices, coords)
        # transformed: (N, 3, ops)
        # Add translations: translations (ops, 3) -> (1, 3, ops) for broadcasting
        if apply_translation:
            transformed = transformed + self.translations.T.unsqueeze(0)
        return transformed  # (N, 3, ops)

    def apply_to_hkl(self, hkl: torch.Tensor) -> torch.Tensor:
        """
        Apply symmetry operations to Miller indices (rotation only, no translation).

        Reciprocal space uses the *transpose*: ``h' = h·R = Rᵀ·h``. Substituting
        ``R·h`` gives the wrong equivalents wherever the fractional rotation is
        non-symmetric (trigonal, hexagonal, permutation-type cubic ops), silently
        corrupting centric flags and epsilon multiplicities. Translations shift
        structure-factor phases, not indices, so they are not applied here.

        Parameters
        ----------
        hkl : torch.Tensor
            Input tensor of shape (N, 3) representing Miller indices.

        Returns
        -------
        torch.Tensor
            Transformed Miller indices of shape (N, 3, ops).

        See Also
        --------
        apply : Real-space coordinates (``R·x + t``), the transposed convention.
        """
        coords = hkl.to(self.matrices.device).to(self.matrices.dtype)
        # result[n, i, o] = sum_j matrices[o, j, i] * coords[n, j] = (Rᵀ·h)_i
        return torch.einsum("oji,nj->nio", self.matrices, coords)

    def expand_coords_to_P1(self, xyz_fractional: torch.Tensor) -> torch.Tensor:
        """
        Expand fractional coordinates by applying all symmetry operations.

        Parameters
        ----------
        xyz_fractional : torch.Tensor
            Input tensor of shape (N, 3) representing fractional coordinates.

        Returns
        -------
        torch.Tensor
            Expanded coordinates of shape (N * ops, 3).
        """
        transformed = self.apply(xyz_fractional)  # (N, 3, ops)
        N = xyz_fractional.shape[0]
        ops = self.n_ops
        # (N, 3, ops) -> (N, ops, 3) -> (N * ops, 3)
        expanded = transformed.permute(0, 2, 1).reshape(N * ops, 3)
        return expanded

    def forward(self, xyz_fractional: torch.Tensor) -> torch.Tensor:
        """Forward pass applies symmetry operations."""
        return self.apply(xyz_fractional)

    # =========================================================================
    # Grid utilities
    # =========================================================================

    def get_grid_requirements(self) -> dict:
        """Per-axis grid divisibility; see :func:`get_grid_requirements`."""
        return get_grid_requirements(self)

    def check_grid_compatibility(self, grid_shape: tuple) -> dict:
        """Report whether ``(nx, ny, nz)`` suits this group and the FFT.

        Returns the report dict documented in :func:`check_grid_compatibility`.
        """
        return check_grid_compatibility(grid_shape, self)

    def suggest_grid_size(
        self, min_grid_shape: tuple, make_fft_friendly: bool = True
    ) -> tuple:
        """Smallest valid grid >= ``min_grid_shape``; see :func:`suggest_grid_size`."""
        return suggest_grid_size(min_grid_shape, self, make_fft_friendly)

    # =========================================================================
    # Dunder methods
    # =========================================================================

    def __repr__(self) -> str:
        return f"SpaceGroup('{self.name}', number={self.number}, n_ops={self.n_ops})"

    def __hash__(self) -> int:
        """Hash based on space group number."""
        return hash(self._sg_number)

    def __eq__(self, other) -> bool:
        """Equality based on space group number."""
        if isinstance(other, SpaceGroup):
            return self._sg_number == other._sg_number
        if isinstance(other, gemmi.SpaceGroup):
            return self._sg_number == other.number
        return False

    # =========================================================================
    # Device movement
    # =========================================================================

    def copy(self) -> "SpaceGroup":
        """A new SpaceGroup with the same symmetry, dtype and device (fresh buffers)."""
        new_sg = SpaceGroup(self._sg_hm, dtype=self._dtype, device=self._device)
        return new_sg
