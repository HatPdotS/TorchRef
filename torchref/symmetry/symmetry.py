"""
Core symmetry operations for crystallographic calculations.

This module provides the Symmetry class for handling space group symmetry
operations on fractional coordinates.
"""

import gemmi
import torch
import torch.nn as nn

from torchref.symmetry.spacegroup import (
    SpaceGroup,
    SpaceGroupLike,
    get_operations_as_tensors,
)
from torchref.utils.debug_utils import DebugMixin


class Symmetry(DebugMixin, nn.Module):
    """
    Crystallographic symmetry operations handler.

    Applies space group symmetry operations to fractional coordinates.
    Uses gemmi.SpaceGroup as the canonical space group representation.

    Parameters
    ----------
    space_group : str, int, gemmi.SpaceGroup, or None
        Space group specification. Accepts:
        - Hermann-Mauguin symbol (e.g., 'P21', 'P 21 21 21')
        - Space group number (1-230)
        - gemmi.SpaceGroup object
        - None (defaults to P1)
    dtype : torch.dtype, default torch.float64
        Data type for rotation matrices and translations.
    device : torch.device, default torch.device('cpu')
        Device for computation.

    Attributes
    ----------
    spacegroup : gemmi.SpaceGroup
        The canonical space group object.
    matrices : torch.Tensor, shape (n_ops, 3, 3)
        Rotation matrices for all symmetry operations.
    translations : torch.Tensor, shape (n_ops, 3)
        Translation vectors for all symmetry operations.
    n_ops : int
        Number of symmetry operations.

    Examples
    --------
    >>> sym = Symmetry('P21')
    >>> coords = torch.tensor([[0.1, 0.2, 0.3]])
    >>> transformed = sym(coords)  # Apply all symmetry operations
    >>> print(transformed.shape)   # (3, 1, 2) for P21 with 2 operations
    """

    def __init__(
        self,
        space_group: SpaceGroupLike,
        dtype: torch.dtype = torch.float64,
        device: torch.device = torch.device("cpu"),
    ):
        super(Symmetry, self).__init__()
        self.device = device
        self.dtype = dtype

        # Normalize to gemmi.SpaceGroup
        self.spacegroup = SpaceGroup(space_group)

        # Get symmetry operations as tensors
        matrices, translations = get_operations_as_tensors(
            self.spacegroup, dtype=dtype, device=device
        )

        self.register_buffer("matrices", matrices)
        self.register_buffer("translations", translations)
        self.n_ops = matrices.shape[0]

    @property
    def space_group(self) -> gemmi.SpaceGroup:
        """Alias for spacegroup (for backward compatibility)."""
        return self.spacegroup

    @property
    def space_group_name(self) -> str:
        """Get space group name as string (short form)."""
        return self.spacegroup.short_name()

    @property
    def space_group_number(self) -> int:
        """Get space group number."""
        return self.spacegroup.number

    def apply(self, fractional_coords: torch.Tensor) -> torch.Tensor:
        """
        Apply symmetry operations to fractional coordinates.

        Parameters
        ----------
        fractional_coords : torch.Tensor
            Input tensor of shape (N, 3) representing fractional coordinates.

        Returns
        -------
        torch.Tensor
            Transformed coordinates of shape (3, N, ops) where ops is the
            number of symmetry operations.
        """
        coords = (
            fractional_coords.reshape(3, -1)
            .to(self.matrices.device)
            .to(self.matrices.dtype)
        )  # (3, N)
        coords = coords.unsqueeze(0)  # (1, 3, N)
        transformed = torch.matmul(self.matrices, coords) + self.translations.unsqueeze(
            2
        )
        # transformed: (ops, 3, N)
        return transformed.permute(1, 2, 0)  # (3, N, ops)

    def forward(self, fractional_coords: torch.Tensor) -> torch.Tensor:
        """Forward pass applies symmetry operations."""
        return self.apply(fractional_coords)

    def get_grid_requirements(self) -> dict:
        """
        Analyze symmetry operations to determine grid size requirements.

        Examines all rotation matrices and translations to determine which
        grid dimensions must satisfy divisibility constraints for exact
        integer indexing (interpolation-free symmetry expansion).

        Returns
        -------
        dict
            {'nx_mod': int, 'ny_mod': int, 'nz_mod': int}
            Required divisibility for each axis.
            For example: {'nx_mod': 1, 'ny_mod': 2, 'nz_mod': 1}
            means ny must be divisible by 2.

        Examples
        --------
        >>> sym = Symmetry('P21')
        >>> req = sym.get_grid_requirements()
        >>> print(req)  # {'nx_mod': 1, 'ny_mod': 2, 'nz_mod': 1}
        """
        import math
        from fractions import Fraction

        # Start with no requirements
        nx_lcm = 1
        ny_lcm = 1
        nz_lcm = 1

        # Analyze each symmetry operation
        for i in range(self.matrices.shape[0]):
            trans = self.translations[i].cpu()

            # For each axis, check if translation has fractional component
            for axis_idx in range(3):
                t = float(trans[axis_idx])

                # Convert to fraction and get denominator
                # The grid size must be divisible by this denominator
                frac = Fraction(t).limit_denominator(
                    12
                )  # Limit to denominators up to 12
                denom = frac.denominator

                if axis_idx == 0:
                    nx_lcm = math.lcm(nx_lcm, denom)
                elif axis_idx == 1:
                    ny_lcm = math.lcm(ny_lcm, denom)
                else:
                    nz_lcm = math.lcm(nz_lcm, denom)

        return {"nx_mod": nx_lcm, "ny_mod": ny_lcm, "nz_mod": nz_lcm}

    def check_grid_compatibility(self, grid_shape: tuple) -> dict:
        """
        Check if a grid size is compatible with the symmetry operations.

        Parameters
        ----------
        grid_shape : tuple of int
            (nx, ny, nz) grid dimensions.

        Returns
        -------
        dict
            Dictionary with the following keys:

            - 'compatible' : bool
                True if grid satisfies all symmetry requirements.
            - 'can_use_direct_indexing' : bool
                True if interpolation-free expansion is possible.
            - 'issues' : list of str
                Descriptions of incompatibilities (empty if compatible).
            - 'requirements' : dict
                Required divisibility from get_grid_requirements().

        Examples
        --------
        >>> sym = Symmetry('P21')
        >>> result = sym.check_grid_compatibility((131, 163, 148))
        >>> print(result['compatible'])  # False
        >>> print(result['issues'])  # ['ny=163 not divisible by 2']
        """
        nx, ny, nz = grid_shape
        requirements = self.get_grid_requirements()

        issues = []
        sg_name = self.spacegroup.short_name()

        if nx % requirements["nx_mod"] != 0:
            issues.append(
                f"nx={nx} not divisible by {requirements['nx_mod']} "
                f"(required for {sg_name})"
            )

        if ny % requirements["ny_mod"] != 0:
            issues.append(
                f"ny={ny} not divisible by {requirements['ny_mod']} "
                f"(required for {sg_name})"
            )

        if nz % requirements["nz_mod"] != 0:
            issues.append(
                f"nz={nz} not divisible by {requirements['nz_mod']} "
                f"(required for {sg_name})"
            )

        compatible = len(issues) == 0

        return {
            "compatible": compatible,
            "can_use_direct_indexing": compatible,
            "issues": issues,
            "requirements": requirements,
        }

    def suggest_grid_size(
        self, min_grid_shape: tuple, make_fft_friendly: bool = True
    ) -> tuple:
        """
        Suggest an optimal grid size that satisfies symmetry requirements.

        Given a minimum grid size, finds the nearest larger size that:

        1. Satisfies symmetry requirements (divisibility constraints)
        2. Optionally, is FFT-friendly (factors of 2, 3, 5 only)

        Parameters
        ----------
        min_grid_shape : tuple of int
            Minimum (nx, ny, nz) grid dimensions.
        make_fft_friendly : bool, default True
            If True, ensures result has only factors of 2, 3, 5.

        Returns
        -------
        tuple of int
            Suggested grid dimensions (nx, ny, nz).

        Examples
        --------
        >>> sym = Symmetry('P21')
        >>> suggested = sym.suggest_grid_size((131, 163, 148))
        >>> print(suggested)  # (135, 164, 150) or similar
        """
        requirements = self.get_grid_requirements()

        def find_next_valid(n, divisibility):
            """Find next number >= n that satisfies divisibility."""
            if n % divisibility == 0:
                candidate = n
            else:
                candidate = ((n // divisibility) + 1) * divisibility

            if not make_fft_friendly:
                return candidate

            # Find next FFT-friendly number
            while not self._is_fft_friendly(candidate):
                candidate += divisibility

            return candidate

        nx_min, ny_min, nz_min = min_grid_shape

        nx = find_next_valid(nx_min, requirements["nx_mod"])
        ny = find_next_valid(ny_min, requirements["ny_mod"])
        nz = find_next_valid(nz_min, requirements["nz_mod"])

        return (nx, ny, nz)

    @staticmethod
    def _is_fft_friendly(n: int) -> bool:
        """Check if number has only factors of 2, 3, and 5."""
        if n <= 0:
            return False

        # Remove all factors of 2, 3, 5
        while n % 2 == 0:
            n //= 2
        while n % 3 == 0:
            n //= 3
        while n % 5 == 0:
            n //= 5

        return n == 1

    def __repr__(self) -> str:
        return (
            f"Symmetry(spacegroup={self.spacegroup.short_name()}, "
            f"number={self.spacegroup.number}, n_ops={self.n_ops})"
        )

    def __hash__(self) -> int:
        """Hash based on space group number."""
        return hash(self.spacegroup.number)

    def __eq__(self, other) -> bool:
        """Equality based on space group number."""
        if isinstance(other, Symmetry):
            return self.spacegroup.number == other.spacegroup.number
        return False
