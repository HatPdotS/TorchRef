"""Symmetry groups and everything derivable from their operations alone.

:class:`Symmetry` holds a group as rotation matrices and fractional translations and
derives what needs nothing else: expansion of positions and Miller indices,
translation phases, the reflection predicates (centric, systematically absent,
multiplicity), symmetry-compatible grid sizes, and real-space map symmetrization.
Nothing here knows about crystals, so a group assembled from a raw operation list
serves non-crystallographic symmetry equally well.
:class:`~torchref.symmetry.spacegroup.SpaceGroup` specialises it with the
crystallographic identity and the CCP4 asymmetric-unit conventions.

Rotation composes; translation does not. Translation acts *additively* on real-space
positions and as a *phase* ``exp(2 pi i h.t)`` in reciprocal space, so the two are
separate primitives rather than one method behind a flag. And ``h' = R^T h`` is not a
third law: it is :meth:`Symmetry.apply_rotations` on :attr:`Symmetry.reciprocal`, the
same group carrying the transposed rotations. Routing every caller through those
primitives is what keeps the real/reciprocal transpose from being re-decided, and got
wrong, at each site.

Every expansion returns operations on the *leading* axis -- ``(n_ops, ...)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import TYPE_CHECKING

import torch

from torchref.config import get_float_dtype
from torchref.utils.device_mixin import DeviceMixin

if TYPE_CHECKING:
    from torchref.symmetry.cell import Cell

# Fractional translations are exact multiples of 1/24 (the denominator gemmi stores
# them over), which is what lets :meth:`Symmetry.grid_requirements` recover exact
# denominators instead of guessing at a float tolerance.
_TRANSLATION_DENOMINATOR = 24

# Tolerance for "this dot product is an integer" when testing a translation against a
# reflection. Miller indices are small and translations are k/24, so the true values
# are either integral or off by at least 1/24 -- far outside float32 noise.
_PHASE_TOL = 1e-6


def is_fft_friendly(n: int) -> bool:
    """Whether ``n`` factors into 2, 3 and 5 only, as radix-2,3,5 FFTs want.

    Parameters
    ----------
    n : int
        Candidate grid length.

    Returns
    -------
    bool
        True for 128 and 135, False for 131 and for any ``n <= 0``.
    """
    if n <= 0:
        return False

    for factor in (2, 3, 5):
        while n % factor == 0:
            n //= factor

    return n == 1


def find_fft_friendly_size(n: int, divisibility: int = 1) -> int:
    """Smallest FFT-friendly size at or above ``n`` that ``divisibility`` divides.

    Parameters
    ----------
    n : int
        Minimum grid length.
    divisibility : int, default 1
        Required divisor, e.g. 2 for a screw axis.

    Returns
    -------
    int
        131 gives 135; 131 at divisibility 2 gives 160.
    """
    candidate = n
    if candidate % divisibility != 0:
        candidate = ((candidate // divisibility) + 1) * divisibility

    while not is_fft_friendly(candidate):
        candidate += divisibility

    return candidate


@dataclass(eq=False, repr=False)
class Symmetry(DeviceMixin):
    """A symmetry group as operations, plus everything they imply.

    Mutable by design; prefer :meth:`copy` over editing in place. Derived quantities
    live in one cache that :meth:`reset_cache` clears -- and that ``.to()`` clears for
    you, since :class:`~torchref.utils.device_mixin.DeviceMixin` invalidates caches on
    every move.

    Parameters
    ----------
    matrices : torch.Tensor
        Rotation matrices, shape ``(n_ops, 3, 3)``, in the fractional basis.
    translations : torch.Tensor
        Fractional translations, shape ``(n_ops, 3)``. Coerced onto ``matrices``'
        device and dtype, so the two can never end up split.

    Attributes
    ----------
    matrices, translations : torch.Tensor
        The operations, as above.

    Notes
    -----
    Holds no refinable parameters -- symmetry operations are fixed constants -- so it
    is a plain dataclass rather than an ``nn.Module``, and there is no gradient path
    through it.

    Writing into :attr:`matrices` or :attr:`translations` in place does **not**
    invalidate the cache, so the reciprocal stack and any built operator would keep
    answering for the old operations. Nothing here mutates them, and :meth:`copy` is
    the intended way to vary a group; if you must edit in place, call
    :meth:`reset_cache` afterwards.
    """

    matrices: torch.Tensor
    translations: torch.Tensor
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Validate the operation shapes and put both tensors on one device/dtype."""
        if self.matrices.ndim != 3 or self.matrices.shape[-2:] != (3, 3):
            raise ValueError(
                f"matrices must have shape (n_ops, 3, 3), got "
                f"{tuple(self.matrices.shape)}"
            )
        if self.translations.ndim != 2 or self.translations.shape[-1] != 3:
            raise ValueError(
                f"translations must have shape (n_ops, 3), got "
                f"{tuple(self.translations.shape)}"
            )
        if self.translations.shape[0] != self.matrices.shape[0]:
            raise ValueError(
                f"matrices and translations disagree on n_ops: "
                f"{self.matrices.shape[0]} vs {self.translations.shape[0]}"
            )

        # One device and dtype for the pair. A split here would surface much later, as
        # a device mismatch inside whichever expansion happened to touch both.
        self.translations = self.translations.to(
            device=self.matrices.device, dtype=self.matrices.dtype
        )

    # =========================================================================
    # Identity
    # =========================================================================

    @property
    def n_ops(self) -> int:
        """Number of symmetry operations."""
        return int(self.matrices.shape[0])

    @property
    def device(self) -> torch.device:
        """Device the operations live on."""
        return self.matrices.device

    @property
    def dtype(self) -> torch.dtype:
        """Floating dtype of the operations."""
        return self.matrices.dtype

    # =========================================================================
    # The primitives
    # =========================================================================

    @property
    def reciprocal(self) -> "Symmetry":
        """The same group carrying the transposed rotations, for reciprocal space.

        ``h' = R^T h``, so Miller-index expansion is :meth:`apply_rotations` on this
        object rather than a law of its own. Cached.

        Returns
        -------
        Symmetry
            Group with ``matrices = R^T`` and this group's translations.

        Notes
        -----
        The translations come along because :meth:`phase_factors` needs them, but in
        reciprocal space a translation is a *phase*, not a shift: calling
        :meth:`apply_translations` on this object is meaningless.
        """
        cached = self._cache.get("reciprocal")
        if cached is None:
            cached = Symmetry(
                matrices=self.matrices.transpose(-2, -1).contiguous(),
                translations=self.translations,
            )
            self._cache["reciprocal"] = cached
        return cached

    def apply_rotations(self, v: torch.Tensor) -> torch.Tensor:
        """Rotate ``v`` by every operation: ``R v``.

        The one batched matmul behind every expansion. Translation is not applied --
        compose with :meth:`apply_translations` for real-space positions, or use
        :attr:`reciprocal` for Miller indices.

        Parameters
        ----------
        v : torch.Tensor
            Vectors of shape ``(N, 3)``, in the fractional basis.

        Returns
        -------
        torch.Tensor
            Shape ``(n_ops, N, 3)``, rotated by each operation in turn.
        """
        v = v.to(device=self.device, dtype=self.dtype)
        # result[o, n, i] = sum_j matrices[o, i, j] * v[n, j]
        return torch.einsum("oij,nj->oni", self.matrices, v)

    def apply_translations(self, v: torch.Tensor) -> torch.Tensor:
        """Add each operation's fractional translation: ``v + t``.

        Real space only; in reciprocal space a translation is a phase, see
        :meth:`phase_factors`.

        Parameters
        ----------
        v : torch.Tensor
            Shape ``(n_ops, N, 3)`` -- typically straight out of
            :meth:`apply_rotations` -- or ``(N, 3)`` to broadcast one set of vectors
            across all operations.

        Returns
        -------
        torch.Tensor
            Shape ``(n_ops, N, 3)``.
        """
        v = v.to(device=self.device, dtype=self.dtype)
        if v.ndim == 2:
            v = v.unsqueeze(0).expand(self.n_ops, -1, -1)
        elif v.shape[0] != self.n_ops:
            raise ValueError(
                f"expected a leading axis of n_ops={self.n_ops} or a bare (N, 3), "
                f"got {tuple(v.shape)}"
            )
        return v + self.translations.unsqueeze(1)

    def phase_factors(self, hkl: torch.Tensor) -> torch.Tensor:
        """Structure-factor phase shift per operation: ``exp(2 pi i h.t)``.

        The reciprocal-space action of a translation. Pairs with
        ``self.reciprocal.apply_rotations(hkl)`` to combine symmetry-equivalent
        structure factors.

        Parameters
        ----------
        hkl : torch.Tensor
            Miller indices, shape ``(N, 3)``, integer or float.

        Returns
        -------
        torch.Tensor
            Complex phase factors of shape ``(n_ops, N)``.

        Notes
        -----
        The phase stays at this group's floating dtype rather than being forced to
        float32, so a float64 configuration yields complex128 instead of silently
        narrowing to complex64.

        This is the *complex factor* form, ``exp(+2 pi i h.t)``, for combining
        structure factors. Expanding the *phases* of reflection data instead needs a
        signed offset in radians, ``-2 pi h.t``, which
        :meth:`~torchref.symmetry.spacegroup.SpaceGroup.expand_hkl` computes itself.
        The two are not interchangeable and the sign difference is invisible in
        P21/P212121/C2 -- see ``tests/unit/symmetry/test_phase_convention.py``.
        """
        hkl = hkl.to(device=self.device, dtype=self.dtype)
        h_dot_t = torch.matmul(hkl, self.translations.T).T  # (n_ops, N)
        return torch.exp(1j * (2.0 * math.pi * h_dot_t))

    # =========================================================================
    # Named expansions
    # =========================================================================

    def expand_positions(self, xyz_fractional: torch.Tensor) -> torch.Tensor:
        """Expand fractional positions by every operation: ``R x + t``.

        Fixes the composition order in one place -- ``R x + t``, not ``R (x + t)``.

        Parameters
        ----------
        xyz_fractional : torch.Tensor
            Fractional coordinates, shape ``(N, 3)``.

        Returns
        -------
        torch.Tensor
            Shape ``(n_ops, N, 3)``, unwrapped (values may fall outside ``[0, 1)``).
        """
        return self.apply_translations(self.apply_rotations(xyz_fractional))

    def expand_directions(self, v_fractional: torch.Tensor) -> torch.Tensor:
        """Expand fractional directions or displacements: ``R v``, no translation.

        Parameters
        ----------
        v_fractional : torch.Tensor
            Fractional vectors, shape ``(N, 3)``.

        Returns
        -------
        torch.Tensor
            Shape ``(n_ops, N, 3)``.
        """
        return self.apply_rotations(v_fractional)

    def expand_to_P1(self, xyz_fractional: torch.Tensor) -> torch.Tensor:
        """Flatten :meth:`expand_positions` into one P1 coordinate list.

        Parameters
        ----------
        xyz_fractional : torch.Tensor
            Fractional coordinates, shape ``(N, 3)``.

        Returns
        -------
        torch.Tensor
            Shape ``(n_ops * N, 3)``, operation-major.
        """
        return self.expand_positions(xyz_fractional).reshape(-1, 3)

    def expand_reciprocal(self, hkl: torch.Tensor) -> torch.Tensor:
        """Symmetry-equivalent Miller indices: ``h' = R^T h``.

        Parameters
        ----------
        hkl : torch.Tensor
            Miller indices, shape ``(N, 3)``.

        Returns
        -------
        torch.Tensor
            Shape ``(n_ops, N, 3)``, rounded to ``int64``. Rounding is exact for valid
            operations on integer indices and only mops up float error.
        """
        equivalents = self.reciprocal.apply_rotations(hkl)
        return torch.round(equivalents).to(torch.int64)  # dtype-ok: rounded Miller equivalents; int64 for exact integer compare/index

    # =========================================================================
    # Reflection predicates
    # =========================================================================

    def is_centric(self, hkl: torch.Tensor) -> torch.Tensor:
        """Whether each reflection is centric, i.e. some operation maps ``h -> -h``.

        A centric reflection has its phase restricted to 0 or pi.

        Parameters
        ----------
        hkl : torch.Tensor
            Miller indices, shape ``(..., 3)``.

        Returns
        -------
        torch.Tensor
            Boolean mask of shape ``(...)``, on ``hkl``'s device.
        """
        original_shape = hkl.shape[:-1]
        with torch.no_grad():
            flat = hkl.reshape(-1, 3)
            equivalents = self.expand_reciprocal(flat)  # (n_ops, N, 3)
            target = -flat.to(device=equivalents.device, dtype=torch.int64)  # dtype-ok: compare target for int64 equivalents; dtype must match
            centric = (equivalents == target).all(dim=-1).any(dim=0)
        return centric.reshape(original_shape).to(hkl.device)

    def is_absent(self, hkl: torch.Tensor) -> torch.Tensor:
        """Whether each reflection is systematically absent.

        Absent when some operation maps ``h -> h`` while ``h.t`` is non-integral: the
        reflection is destroyed by interference from that translation.

        Parameters
        ----------
        hkl : torch.Tensor
            Miller indices, shape ``(..., 3)``.

        Returns
        -------
        torch.Tensor
            Boolean mask of shape ``(...)``, on ``hkl``'s device.
        """
        original_shape = hkl.shape[:-1]
        with torch.no_grad():
            flat = hkl.reshape(-1, 3)
            equivalents = self.expand_reciprocal(flat)  # (n_ops, N, 3)
            target = flat.to(device=equivalents.device, dtype=torch.int64)  # dtype-ok: compare target for int64 equivalents; dtype must match
            maps_to_self = (equivalents == target).all(dim=-1)  # (n_ops, N)

            h_dot_t = torch.matmul(
                flat.to(device=self.device, dtype=self.dtype), self.translations.T
            ).T  # (n_ops, N)
            non_integral = (h_dot_t - torch.round(h_dot_t)).abs() > _PHASE_TOL

            absent = (maps_to_self & non_integral).any(dim=0)
        return absent.reshape(original_shape).to(hkl.device)

    def epsilon(self, hkl: torch.Tensor, *, friedel: bool = True) -> torch.Tensor:
        """Reflection multiplicity: operations mapping ``h`` to ``h``, or also to ``-h``.

        Parameters
        ----------
        hkl : torch.Tensor
            Miller indices, shape ``(N, 3)``.
        friedel : bool, default True
            Whether operations mapping ``h -> -h`` count alongside ``h -> h``.

        Returns
        -------
        torch.Tensor
            Multiplicities of shape ``(N,)`` at the configured float dtype, floored at
            1, returned on ``hkl``'s device so it can weight data sitting beside it.

        Notes
        -----
        The two settings answer different questions and both are wanted.

        Operations mapping ``h -> h`` add coherently and set the **mean**,
        ``<|F|^2> = eps * Sigma``. That is the conventional crystallographic epsilon,
        and what a Wilson normalisation or a likelihood's variance budget asks for.
        Operations mapping ``h -> -h`` leave the mean alone and instead make ``F``
        real, changing the **distribution** from exponential to chi2_1 -- that is
        centricity, and :meth:`is_centric` already carries it. Folding Friedel into
        epsilon therefore mixes a mean effect with a distribution effect.

        The default keeps Friedel folded in because downstream sigma_A estimation is
        calibrated against that convention; flipping it would silently decalibrate the
        refinement path. Pass ``friedel=False`` for the conventional count, as the
        molecular-replacement likelihood does -- counting Friedel there doubles
        epsilon on exactly the reflections whose distribution the Woolfson branch is
        already handling, inflating their ``V = eps - sigma_A**2``.

        The two differ on centric reflections and *only* there: measured across the
        ten benchmark structures every disagreement was centric, and the counts are
        not small -- 12360 reflections on 2DQ6, 7555 on 4BX9, 6680 on 3K7M.

        Both settings count lattice-centring cosets, so on a centred lattice every
        reflection carries the centring order as a factor: C2 gives 2 for general
        reflections where a primitive lattice gives 1. That is a separate axis from
        this switch. Being uniform per lattice it is absorbed into ``Sigma`` wherever
        epsilon is a factor -- which is why the refinement path never saw it -- and
        bites only where epsilon is a term.
        """
        float_dtype = get_float_dtype()
        with torch.no_grad():
            equivalents = self.expand_reciprocal(hkl)  # (n_ops, N, 3)
            target = hkl.to(device=equivalents.device, dtype=torch.int64)  # dtype-ok: compare target for int64 equivalents; dtype must match
            fixes = (equivalents == target).all(dim=-1)
            if friedel:
                fixes = fixes | (equivalents == -target).all(dim=-1)
            eps = fixes.sum(dim=0).clamp(min=1).to(float_dtype)
        return eps.to(hkl.device)

    # =========================================================================
    # Grid compatibility
    # =========================================================================

    def grid_requirements(self) -> dict:
        """Per-axis grid divisibility for interpolation-free symmetry expansion.

        Read off the denominators of the fractional translations, so a grid meeting
        them indexes every symmetry mate at an exact integer.

        Returns
        -------
        dict
            ``{'nx_mod': int, 'ny_mod': int, 'nz_mod': int}`` -- P21 gives
            ``(1, 2, 1)``, P212121 gives ``(2, 2, 2)``.
        """
        mods = [1, 1, 1]
        # Recover each denominator from the integer numerator over 1/24 rather than
        # from the float directly: 1/3 is not representable in float32, so
        # ``Fraction(float)`` would need a tolerance where this is exact.
        numerators = torch.round(
            self.translations.detach().cpu().double() * _TRANSLATION_DENOMINATOR
        ).to(torch.int64)  # dtype-ok: integer translation numerators for exact Fraction recovery

        for op_numerators in numerators.tolist():
            for axis, numerator in enumerate(op_numerators):
                if numerator % _TRANSLATION_DENOMINATOR == 0:
                    continue
                denominator = Fraction(
                    int(numerator), _TRANSLATION_DENOMINATOR
                ).denominator
                mods[axis] = math.lcm(mods[axis], denominator)

        return {"nx_mod": mods[0], "ny_mod": mods[1], "nz_mod": mods[2]}

    def check_grid_compatibility(self, grid_shape: tuple) -> dict:
        """Check a grid against this group's divisibility and against the FFT.

        Parameters
        ----------
        grid_shape : tuple of int
            Grid dimensions ``(nx, ny, nz)``.

        Returns
        -------
        dict
            ``compatible`` (both tests pass), ``symmetry_compatible``,
            ``fft_friendly``, ``can_use_direct_indexing`` (interpolation-free expansion
            possible; equal to ``symmetry_compatible``), ``issues`` (per-axis
            descriptions, empty when compatible) and ``requirements`` (from
            :meth:`grid_requirements`).
        """
        requirements = self.grid_requirements()
        issues = []

        for axis, name in enumerate(("nx", "ny", "nz")):
            modulus = requirements[f"{name}_mod"]
            length = int(grid_shape[axis])
            if length % modulus != 0:
                issues.append(
                    f"{name}={length} not divisible by {modulus} "
                    f"(required by the symmetry)"
                )

        symmetry_compatible = len(issues) == 0

        fft_friendly = True
        for axis, name in enumerate(("nx", "ny", "nz")):
            length = int(grid_shape[axis])
            if not is_fft_friendly(length):
                fft_friendly = False
                issues.append(
                    f"{name}={length} is not FFT-friendly (not a product of 2, 3, 5)"
                )

        return {
            "compatible": symmetry_compatible and fft_friendly,
            "symmetry_compatible": symmetry_compatible,
            "fft_friendly": fft_friendly,
            "can_use_direct_indexing": symmetry_compatible,
            "issues": issues,
            "requirements": requirements,
        }

    def can_index_directly(self, grid_shape: tuple) -> bool:
        """Whether ``grid_shape`` admits interpolation-free symmetry expansion.

        The question :meth:`symmetrize_map` answers internally when choosing an
        implementation, exposed so callers can ask it without building an operator.

        Parameters
        ----------
        grid_shape : tuple of int
            Grid dimensions ``(nx, ny, nz)``.

        Returns
        -------
        bool
            True when every symmetry mate lands on an exact grid point.
        """
        return bool(self.check_grid_compatibility(grid_shape)["symmetry_compatible"])

    def suggest_grid_size(
        self, min_grid_shape: tuple, make_fft_friendly: bool = True
    ) -> tuple:
        """Smallest grid at or above ``min_grid_shape`` meeting the divisibility.

        Parameters
        ----------
        min_grid_shape : tuple of int
            Minimum dimensions ``(nx, ny, nz)``.
        make_fft_friendly : bool, default True
            Also require factors of 2, 3 and 5 only.

        Returns
        -------
        tuple of int
            Suggested ``(nx, ny, nz)``.
        """
        requirements = self.grid_requirements()

        def next_valid(length: int, divisibility: int) -> int:
            if make_fft_friendly:
                return find_fft_friendly_size(length, divisibility)
            if length % divisibility == 0:
                return length
            return ((length // divisibility) + 1) * divisibility

        return tuple(
            next_valid(int(min_grid_shape[axis]), requirements[f"{name}_mod"])
            for axis, name in enumerate(("nx", "ny", "nz"))
        )

    def optimal_grid_size(
        self, cell: "Cell", max_res: float, make_fft_friendly: bool = True
    ) -> tuple:
        """Smallest grid that samples ``cell`` to ``max_res`` and suits this group.

        Composes the cell's Shannon-Nyquist minimum with
        :meth:`suggest_grid_size`; the oversampling factor is the cell's, so every
        grid-sizing path shares one setting.

        Parameters
        ----------
        cell : Cell
            Unit cell.
        max_res : float
            Maximum resolution in Angstroms.
        make_fft_friendly : bool, default True
            Also require factors of 2, 3 and 5 only.

        Returns
        -------
        tuple of int
            Grid dimensions ``(nx, ny, nz)``.
        """
        return self.suggest_grid_size(
            cell.compute_grid_size(max_res), make_fft_friendly=make_fft_friendly
        )

    # =========================================================================
    # Real-space maps
    # =========================================================================

    def map_operator(self, map_shape):
        """Cached operator that applies this group to maps of ``map_shape``.

        Two implementations: exact integer indexing when the grid permits it, and
        ``grid_sample`` interpolation otherwise. Which one you get depends on the grid,
        so a mis-sized grid costs accuracy -- :meth:`can_index_directly` reports the
        distinction, and :meth:`suggest_grid_size` fixes it.

        Parameters
        ----------
        map_shape : tuple of int
            Density map dimensions ``(nx, ny, nz)``.

        Returns
        -------
        _MapSymmetryDirect or _MapSymmetryInterpolation
            The operator, memoized for the most recent shape only.

        Notes
        -----
        Only the last shape is kept. The interpolating operator holds sampling grids of
        shape ``(n_ops, nx, ny, nz, 3)`` -- hundreds of megabytes at production grid
        sizes -- so a dictionary keyed on shape would quietly make this object
        expensive to hold and to move between devices.
        """
        from torchref.symmetry.map_symmetry import build_map_operator

        key = tuple(int(n) for n in map_shape)
        cached = self._cache.get("map_operator")
        if cached is not None and cached[0] == key:
            return cached[1]

        operator = build_map_operator(self, key)
        self._cache["map_operator"] = (key, operator)
        return operator

    def symmetrize_map(
        self, density_map: torch.Tensor, combine: str = "sum"
    ) -> torch.Tensor:
        """Apply every operation to a density map and combine the mates.

        Parameters
        ----------
        density_map : torch.Tensor
            Asymmetric-unit density, shape ``(nx, ny, nz)``.
        combine : {'sum', 'max'}, default 'sum'
            ``'sum'`` for electron density, ``'max'`` for masks and boolean data.

        Returns
        -------
        torch.Tensor
            Symmetrized map, same shape as the input. Returned unchanged for a
            one-operation group.
        """
        if self.n_ops == 1:
            return density_map
        return self.map_operator(density_map.shape).symmetrize(density_map, combine)

    def expand_map_to_P1(self, density_map: torch.Tensor) -> torch.Tensor:
        """Every symmetry mate of a density map, stacked.

        Parameters
        ----------
        density_map : torch.Tensor
            Asymmetric-unit density, shape ``(nx, ny, nz)``.

        Returns
        -------
        torch.Tensor
            Shape ``(n_ops, nx, ny, nz)``.
        """
        return self.map_operator(density_map.shape).all_mates(density_map)

    # =========================================================================
    # Reciprocal-space extraction
    # =========================================================================

    def reciprocal_extractor(self, hkl: torch.Tensor, grid_shape: tuple):
        """Cached extractor pulling symmetrized structure factors off a grid.

        Precomputes the equivalent indices, phases and flat gather indices for a fixed
        ``hkl`` and ``grid_shape``, so each later call is one gather, multiply and sum.

        Parameters
        ----------
        hkl : torch.Tensor
            Target Miller indices, shape ``(N, 3)``.
        grid_shape : tuple of int
            Reciprocal grid dimensions ``(nx, ny, nz)``.

        Returns
        -------
        ReciprocalSymmetryExtractor
            Memoized against ``hkl``'s identity and ``grid_shape``; a different tensor
            or shape rebuilds it.
        """
        from torchref.base.reciprocal.symmetry import ReciprocalSymmetryExtractor
        from torchref.utils.caching import ParameterFingerprint

        key = tuple(int(n) for n in grid_shape)
        cached = self._cache.get("reciprocal_extractor")
        if cached is not None:
            cached_key, fingerprint, extractor = cached
            if cached_key == key and fingerprint.matches([hkl]):
                return extractor

        extractor = ReciprocalSymmetryExtractor(hkl, self, key)
        self._cache["reciprocal_extractor"] = (
            key,
            ParameterFingerprint([hkl]),
            extractor,
        )
        return extractor

    # =========================================================================
    # Cache and copy
    # =========================================================================

    def reset_cache(self) -> None:
        """Drop every derived quantity.

        Called for you by :class:`~torchref.utils.device_mixin.DeviceMixin` on any
        ``.to()``, including one targeting the current device.
        """
        self._cache = {}

    def _apply(self, fn, recurse: bool = True):
        """Clear the cache *before* the traversal moves anything.

        The base traversal walks ``__dict__`` first and invalidates caches afterwards,
        which would transfer the cached sampling grids to the new device only to
        discard them -- hundreds of megabytes of pointless copying at production grid
        sizes.
        """
        self.reset_cache()
        return super()._apply(fn, recurse)

    def copy(self) -> "Symmetry":
        """An independent copy with cloned operations and an empty cache.

        Returns
        -------
        Symmetry
            New instance; mutating its tensors cannot affect this one.
        """
        return type(self)(
            matrices=self.matrices.clone(),
            translations=self.translations.clone(),
        )

    # =========================================================================
    # Dunder
    # =========================================================================

    def __len__(self) -> int:
        """Number of symmetry operations."""
        return self.n_ops

    def __repr__(self) -> str:
        return f"Symmetry(n_ops={self.n_ops})"


__all__ = ["Symmetry", "find_fft_friendly_size", "is_fft_friendly"]
