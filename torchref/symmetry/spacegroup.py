"""Crystallographic space groups, using gemmi as the canonical source of truth.

:class:`SpaceGroup` is the interface used throughout torchref. It specialises
:class:`~torchref.symmetry.symmetry.Symmetry` -- which owns the operations and
everything derivable from them -- with the crystallographic identity (Hermann-Mauguin
naming, number, point group, crystal system) and the CCP4 asymmetric-unit
conventions, the two things a bare operation list cannot supply.

Construction normalizes any ``SpaceGroupLike``: a Hermann-Mauguin string, a number
1-230, a ``gemmi.SpaceGroup``, another :class:`SpaceGroup`, or None for P1. Only the
derived metadata is retained; no persistent ``gemmi`` reference is held, because a
lasting reference to the C++ singleton produces nanobind leak warnings at shutdown.

Real and reciprocal space use *transposed* conventions. That is handled once, in
:attr:`~torchref.symmetry.symmetry.Symmetry.reciprocal`, rather than being re-decided
per call site.
"""

from __future__ import annotations

from typing import Optional, Union

import gemmi
import torch

from torchref.config import get_float_dtype, normalize_device
from torchref.symmetry.symmetry import Symmetry

# Type alias for space group input - includes SpaceGroup class itself
SpaceGroupLike = Union[str, int, gemmi.SpaceGroup, "SpaceGroup", None]

# gemmi stores rotations and translations as integers scaled by 24.
_GEMMI_SCALE = 24.0


def _normalize_spacegroup(spacegroup: SpaceGroupLike) -> gemmi.SpaceGroup:
    """Normalize any ``SpaceGroupLike`` to a ``gemmi.SpaceGroup``.

    Accepts a Hermann-Mauguin string (spacing-insensitive, retried upper-cased), a
    number 1-230, a ``gemmi.SpaceGroup`` (returned unchanged), a :class:`SpaceGroup`
    (unwrapped), or None (P1).

    Parameters
    ----------
    spacegroup : SpaceGroupLike
        Space group in any supported form.

    Returns
    -------
    gemmi.SpaceGroup
        The normalized space group.

    Raises
    ------
    ValueError
        For an unrecognised name or number.
    TypeError
        For any other type.
    """
    if spacegroup is None:
        return gemmi.SpaceGroup("P 1")

    if isinstance(spacegroup, gemmi.SpaceGroup):
        return spacegroup

    # Duck-typed rather than an isinstance check against SpaceGroup, so this stays
    # usable from module scope before the class below is defined.
    if hasattr(spacegroup, "_sg_xhm") and hasattr(spacegroup, "matrices"):
        # The extended symbol carries the setting; the plain H-M symbol does not
        # (``R 3:R`` would come back as ``R 3:H``).
        return gemmi.find_spacegroup_by_name(spacegroup._sg_xhm)

    if isinstance(spacegroup, int):
        try:
            return gemmi.SpaceGroup(spacegroup)
        except Exception as e:
            raise ValueError(f"Invalid space group number: {spacegroup}") from e

    if isinstance(spacegroup, str):
        sg_clean = spacegroup.strip()
        while "  " in sg_clean:
            sg_clean = sg_clean.replace("  ", " ")
        sg_nospace = sg_clean.replace(" ", "")

        for variant in (
            sg_clean,
            sg_nospace,
            sg_clean.upper(),
            sg_nospace.upper(),
        ):
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


def _operations_as_tensors(
    sg: gemmi.SpaceGroup,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract a gemmi space group's operations as tensors.

    Parameters
    ----------
    sg : gemmi.SpaceGroup
        Normalized space group.
    dtype : torch.dtype
        Floating dtype for both tensors.
    device : torch.device
        Device for both tensors.

    Returns
    -------
    matrices : torch.Tensor
        Rotation matrices, shape ``(n_ops, 3, 3)``.
    translations : torch.Tensor
        Fractional translations, shape ``(n_ops, 3)``.
    """
    ops = [
        (
            torch.tensor(op.rot, dtype=dtype, device=device) / _GEMMI_SCALE,
            torch.tensor(op.tran, dtype=dtype, device=device) / _GEMMI_SCALE,
        )
        for op in sg.operations()
    ]
    matrices, translations = zip(*ops)
    return torch.stack(matrices), torch.stack(translations)


class SpaceGroup(Symmetry):
    """A crystallographic space group: symmetry operations plus their identity.

    Inherits the whole operation-derived surface from
    :class:`~torchref.symmetry.symmetry.Symmetry` -- expansion, phases, reflection
    predicates, grid sizing, map symmetrization -- and adds the crystallographic
    naming and the CCP4 asymmetric-unit conventions.

    Parameters
    ----------
    space_group : str, int, gemmi.SpaceGroup, SpaceGroup, or None
        Hermann-Mauguin symbol, number 1-230, gemmi object, another instance, or None
        for P1.
    dtype : torch.dtype, optional
        Dtype for the operations. Defaults to the configured ``dtypes.float``.
    device : torch.device, optional
        Device for the operations. Defaults to the configured ``device.current``.

    Attributes
    ----------
    matrices : torch.Tensor
        Rotation matrices, shape ``(n_ops, 3, 3)``.
    translations : torch.Tensor
        Fractional translations, shape ``(n_ops, 3)``.

    Examples
    --------
    >>> sg = SpaceGroup("P212121")
    >>> sg.n_ops
    4
    >>> sg.crystal_system
    'orthorhombic'
    """

    def __init__(
        self,
        space_group: SpaceGroupLike = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        if dtype is None:
            dtype = get_float_dtype()
        device = normalize_device(device)

        gemmi_sg = _normalize_spacegroup(space_group)

        self._sg_number: int = gemmi_sg.number
        self._sg_hm: str = gemmi_sg.hm
        self._sg_short_name: str = gemmi_sg.short_name()
        self._sg_xhm: str = gemmi_sg.xhm()
        self._sg_point_group: str = gemmi_sg.point_group_hm()
        self._sg_crystal_system: str = gemmi_sg.crystal_system_str()
        self._sg_centrosymmetric: bool = gemmi_sg.is_centrosymmetric()

        matrices, translations = _operations_as_tensors(gemmi_sg, dtype, device)
        # gemmi_sg goes out of scope here -- no persistent gemmi reference.

        super().__init__(matrices=matrices, translations=translations)

    # =========================================================================
    # Crystallographic identity
    # =========================================================================

    @property
    def _gemmi(self) -> gemmi.SpaceGroup:
        """A fresh ``gemmi.SpaceGroup`` on each access.

        Never cached: a persistent reference to the C++ singleton produces nanobind
        leak warnings at interpreter shutdown.
        """
        return gemmi.find_spacegroup_by_name(self._sg_xhm)

    @property
    def name(self) -> str:
        """Short space group name, e.g. ``'P21'``."""
        return self._sg_short_name

    @property
    def hm(self) -> str:
        """Hermann-Mauguin notation with spaces, e.g. ``'P 21'``."""
        return self._sg_hm

    @property
    def xhm(self) -> str:
        """Extended Hermann-Mauguin notation, including the setting token."""
        return self._sg_xhm

    @property
    def key(self) -> str:
        """Value identity of this space group: the extended H-M symbol.

        Two groups with the same key have the same operations in the same
        setting. The group number alone would not do -- ``P 1 21 1`` and
        ``P 1 1 21`` share number 4 but place the screw axis differently.
        """
        return self._sg_xhm

    @property
    def number(self) -> int:
        """Space group number, 1-230."""
        return self._sg_number

    @property
    def gemmi(self) -> gemmi.SpaceGroup:
        """A ``gemmi.SpaceGroup``, created on demand and not stored."""
        return self._gemmi

    @property
    def point_group(self) -> str:
        """Point group symbol, e.g. ``'222'`` or ``'mmm'``."""
        return self._sg_point_group

    @property
    def crystal_system(self) -> str:
        """Crystal system name, e.g. ``'orthorhombic'``."""
        return self._sg_crystal_system

    @property
    def centrosymmetric(self) -> bool:
        """Whether the group has an inversion centre."""
        return self._sg_centrosymmetric

    def short_name(self) -> str:
        """Short space group name; the callable form of :attr:`name`."""
        return self._sg_short_name

    def operations(self):
        """The gemmi operations, from a temporary gemmi object.

        Returns
        -------
        gemmi.GroupOps
            The operation list.
        """
        return self._gemmi.operations()

    # =========================================================================
    # Asymmetric-unit conventions
    # =========================================================================
    #
    # These need the CCP4 asymmetric unit, which is keyed by Laue class, so they live
    # here rather than on ``Symmetry`` -- "the canonical ASU" is meaningless for a bare
    # operation list. The algorithms are in ``reciprocal_symmetry``; these methods are
    # the only public way in.

    def expand_hkl(
        self,
        hkl: torch.Tensor,
        include_friedel: bool = True,
        remove_absences: bool = True,
        device: Optional[torch.device] = None,
        *,
        return_friedel: bool = False,
    ):
        """Expand Miller indices from the asymmetric unit to P1.

        Parameters
        ----------
        hkl : torch.Tensor
            Input Miller indices, shape ``(N, 3)``.
        include_friedel : bool, default True
            Include Friedel mates ``(-h, -k, -l)``.
        remove_absences : bool, default True
            Drop systematically absent reflections.
        device : torch.device, optional
            Computation device. Defaults to ``hkl``'s.
        return_friedel : bool, default False
            Also return ``is_friedel``. Expanding phases with ``include_friedel``
            needs it: a Friedel copy carries the negated phase.

        Returns
        -------
        expanded_hkl : torch.Tensor
            Expanded indices, shape ``(M, 3)``, in the configured int dtype.
        orig_indices : torch.Tensor
            Map expanded -> original, shape ``(M,)``, in the configured int dtype:
            ``F_exp = F_orig[orig_indices]``.
        phase_shifts : torch.Tensor
            Translation phase offsets in radians, shape ``(M,)``: ``phase_exp =
            where(is_friedel, -phase_orig, phase_orig)[orig_indices] + phase_shifts``.
        is_friedel : torch.Tensor
            Boolean, shape ``(M,)``, True for the rows that are Friedel copies of a
            rotated index; returned only with ``return_friedel``. All False when
            ``include_friedel`` is False.

        Raises
        ------
        ValueError
            If two input rows are symmetry-equivalent, since one would be
            silently dropped. Merge them first; anomalous data expand from their
            signed indices (see ``ReflectionData.expand_to_p1``).
        """
        from torchref.symmetry.reciprocal_symmetry import _expand_hkl

        return _expand_hkl(
            self,
            hkl,
            include_friedel=include_friedel,
            remove_absences=remove_absences,
            device=device,
            return_friedel=return_friedel,
        )

    def equivalent_hkl(
        self,
        hkl: torch.Tensor,
        include_friedel: bool = True,
        device: Optional[torch.device] = None,
    ):
        """Every symmetry copy of every input row, not deduplicated.

        The primitive behind :meth:`expand_hkl`, for callers that must know
        which input row each copy came from even where copies coincide (e.g.
        merging observations into another space group).

        Parameters
        ----------
        hkl : torch.Tensor
            Input Miller indices, shape ``(N, 3)``.
        include_friedel : bool, default True
            Append the Friedel copy of every rotated index.
        device : torch.device, optional
            Output device. Defaults to ``hkl``'s.

        Returns
        -------
        copies : torch.Tensor
            Shape ``(M, 3)``, in the configured int dtype, ordered by operation
            then row, Friedel copies last; ``M = n_ops * N``, doubled with
            ``include_friedel``.
        source : torch.Tensor
            Input row of each copy, shape ``(M,)``, in the configured int dtype.
        phase_shifts : torch.Tensor
            Translation phase offset in radians, shape ``(M,)``.
        is_friedel : torch.Tensor
            Boolean, shape ``(M,)``, True for the Friedel copies.
        """
        from torchref.symmetry.reciprocal_symmetry import _equivalent_hkl

        return _equivalent_hkl(
            self, hkl, include_friedel=include_friedel, device=device
        )

    def canonicalize_hkl(
        self,
        hkl: torch.Tensor,
        include_friedel: bool = True,
        device: Optional[torch.device] = None,
        *,
        sort: bool = True,
    ):
        """Map Miller indices onto their canonical CCP4 ASU representatives.

        Parameters
        ----------
        hkl : torch.Tensor
            Input Miller indices, shape ``(N, 3)``.
        include_friedel : bool, default True
            Treat Friedel mates as equivalent. With ``False`` the Friedel half of
            reciprocal space has no pure-rotation representative in the Laue-based
            CCP4 ASU, and unmappable reflections raise.
        device : torch.device, optional
            Output device. Defaults to ``hkl``'s. The lookup itself runs on CPU
            whatever device this group is on, because the ASU tables are numpy-backed.
        sort : bool, default True
            Sort the rows lexicographically by canonical ``(h, k, l)``. ``False``
            keeps the input row order and skips the sort, which callers that only
            need the per-row mapping should prefer on large inputs.

        Returns
        -------
        canonical_hkl : torch.Tensor
            Remapped indices, shape ``(N, 3)``; sorted lexicographically when
            ``sort``.
        phase_shifts : torch.Tensor
            Additive phase correction in radians, shape ``(N,)``, same row order.
        friedel_flags : torch.Tensor
            Boolean, shape ``(N,)``, True where Friedel conjugation was applied,
            same row order.
        sort_indices : torch.Tensor or None
            Permutation from original to sorted order, shape ``(N,)``; ``None``
            when ``sort=False``.

        Notes
        -----
        ``phase_shifts`` assumes the caller conjugates first: the contract is
        ``phi_new = torch.where(friedel_flags, -phi_old, phi_old) + phase_shifts``.
        """
        from torchref.symmetry.reciprocal_symmetry import _canonicalize_hkl

        return _canonicalize_hkl(
            self, hkl, include_friedel=include_friedel, device=device, sort=sort
        )

    # =========================================================================
    # Copy and dunder
    # =========================================================================

    def copy(self) -> "SpaceGroup":
        """An independent copy with cloned operations and an empty cache.

        Returns
        -------
        SpaceGroup
            New instance carrying the same symmetry, dtype and device.
        """
        new = SpaceGroup.__new__(SpaceGroup)
        new._sg_number = self._sg_number
        new._sg_hm = self._sg_hm
        new._sg_short_name = self._sg_short_name
        new._sg_xhm = self._sg_xhm
        new._sg_point_group = self._sg_point_group
        new._sg_crystal_system = self._sg_crystal_system
        new._sg_centrosymmetric = self._sg_centrosymmetric
        # Through Symmetry's own initializer, so the operand-consistency checks and the
        # device/dtype reconciliation in ``__post_init__`` run on the copy too.
        Symmetry.__init__(
            new,
            matrices=self.matrices.clone(),
            translations=self.translations.clone(),
        )
        return new

    def __hash__(self) -> int:
        """Hash on :attr:`key` (the extended H-M symbol)."""
        return hash(self._sg_xhm)

    def __eq__(self, other) -> bool:
        """Equality on :attr:`key`; also compares to a ``gemmi.SpaceGroup``."""
        if isinstance(other, SpaceGroup):
            return self._sg_xhm == other._sg_xhm
        if isinstance(other, gemmi.SpaceGroup):
            return self._sg_xhm == other.xhm()
        return False

    def __repr__(self) -> str:
        return f"SpaceGroup('{self.name}', number={self.number}, n_ops={self.n_ops})"


__all__ = ["SpaceGroup", "SpaceGroupLike"]
