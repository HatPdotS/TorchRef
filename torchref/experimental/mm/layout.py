"""Copies of a model's atoms in a crystal: where each sits and which molecules it holds.

A :class:`CrystalLayout` says how an OpenMM system is assembled from TorchRef
coordinates. Copy ``c`` takes coordinate set ``source[c]`` -- the model, or one member of
an ensemble -- and places it with a Cartesian rotation and translation (a symmetry
operation plus a lattice translation) inside an optional periodic box.
:meth:`CrystalLayout.positions` is that map as a differentiable tensor operation, so the
forces on every copy return to the coordinates the copy came from.

:meth:`CrystalLayout.presence` decides which molecules a copy holds. A molecule whose
copies overlap -- a water on a two-fold axis, a ligand straddling one -- is kept in the
first copy and left out of the later ones that coincide with it, so the crystal holds it
once per distinct site: the occupancy the deposited model gives it.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np
import torch

from torchref.config import get_int_dtype

if TYPE_CHECKING:
    from torchref.symmetry import Cell, SpaceGroup


@dataclass(eq=False)
class CrystalLayout:
    """Copies of one or more coordinate sets, optionally in a periodic box.

    Parameters
    ----------
    rotation : numpy.ndarray
        Cartesian rotation of each copy, shape ``(C, 3, 3)``; ``x_c = R_c x + t_c``.
    translation : numpy.ndarray
        Cartesian translation of each copy in Å, shape ``(C, 3)``.
    source : numpy.ndarray
        Coordinate set each copy is built from, shape ``(C,)``, integer.
    box : numpy.ndarray, optional
        Periodic box vectors as columns in Å, shape ``(3, 3)``, in OpenMM's reduced form
        (:func:`reduce_box`). None for a non-periodic system.

    Notes
    -----
    Holds NumPy float64 constants and converts them to the coordinates' device and
    dtype on use, caching one copy per (device, dtype).
    """

    rotation: np.ndarray
    translation: np.ndarray
    source: np.ndarray
    box: Optional[np.ndarray] = None
    _tensors: Dict[Tuple[torch.device, torch.dtype], tuple] = field(
        default_factory=dict, repr=False
    )

    def __post_init__(self) -> None:
        self.rotation = np.asarray(self.rotation, dtype=np.float64)
        self.translation = np.asarray(self.translation, dtype=np.float64)
        self.source = np.asarray(self.source, dtype=np.int64)
        n = len(self.source)
        if self.rotation.shape != (n, 3, 3) or self.translation.shape != (n, 3):
            raise ValueError(
                f"rotation {self.rotation.shape} and translation "
                f"{self.translation.shape} must be (C, 3, 3) and (C, 3) for C={n}"
            )
        if self.box is not None:
            self.box = np.asarray(self.box, dtype=np.float64)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def isolated(cls) -> "CrystalLayout":
        """One untransformed copy of one coordinate set, no periodic box."""
        return cls(np.eye(3)[None], np.zeros((1, 3)), np.zeros(1, dtype=np.int64))

    @classmethod
    def unit_cell(
        cls, cell: "Cell", spacegroup: "SpaceGroup", cutoff: float
    ) -> "CrystalLayout":
        """Every symmetry copy of one model in the unit cell, under periodic boundaries.

        The box is the unit cell, repeated along an axis as often as needed for OpenMM's
        rule that the non-bonded cutoff not exceed half the box height.

        Parameters
        ----------
        cell : Cell
        spacegroup : SpaceGroup
            Every operation, centring included, becomes a copy.
        cutoff : float
            Non-bonded cutoff in Å.

        Returns
        -------
        CrystalLayout
            ``n_ops * n_a * n_b * n_c`` copies, all of source 0.
        """
        B = _orthogonalisation(cell)
        heights = np.diag(B)
        repeats = [max(1, math.ceil(2.0 * cutoff / h)) for h in heights]
        R, t = _operations(spacegroup)
        shifts = np.array(list(itertools.product(*(range(k) for k in repeats))))
        ops = np.tile(np.arange(len(R)), len(shifts))
        lattice = np.repeat(shifts, len(R), axis=0)
        return cls._from_fractional(
            B,
            R[ops],
            t[ops] + lattice,
            source=np.zeros(len(ops), dtype=np.int64),
            box=B @ np.diag(np.asarray(repeats, dtype=np.float64)),
        )

    @classmethod
    def quasi_crystal(
        cls, cell: "Cell", spacegroup: "SpaceGroup", n_disorder: int
    ) -> "CrystalLayout":
        """Ensemble members as symmetry copies in a ``n_disorder × 1 × 1`` supercell.

        Member ``m = d * n_ops + j`` is placed with operation ``j`` in the ``d``-th cell
        along a, so the first ``n_ops`` members fill the first cell.

        Parameters
        ----------
        cell : Cell
        spacegroup : SpaceGroup
        n_disorder : int
            Cells along a, at least 1.

        Returns
        -------
        CrystalLayout
            ``n_disorder * n_ops`` copies, copy ``m`` of source ``m``.
        """
        if int(n_disorder) < 1:
            raise ValueError(f"n_disorder must be >= 1, got {n_disorder}")
        B = _orthogonalisation(cell)
        R, t = _operations(spacegroup)
        n_ops, k = len(R), int(n_disorder)
        ops = np.tile(np.arange(n_ops), k)
        lattice = np.zeros((k * n_ops, 3))
        lattice[:, 0] = np.repeat(np.arange(k), n_ops)
        return cls._from_fractional(
            B,
            R[ops],
            t[ops] + lattice,
            source=np.arange(k * n_ops, dtype=np.int64),
            box=B @ np.diag([float(k), 1.0, 1.0]),
        )

    @classmethod
    def _from_fractional(cls, B, R_frac, t_frac, source, box) -> "CrystalLayout":
        """Copies given in the fractional basis of orthogonalisation matrix ``B``."""
        B_inv = np.linalg.inv(B)
        return cls(
            rotation=B @ R_frac @ B_inv,
            translation=t_frac @ B.T,
            source=source,
            box=reduce_box(box),
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def n_copies(self) -> int:
        """Number of copies ``C``."""
        return len(self.source)

    @property
    def n_sources(self) -> int:
        """Number of coordinate sets the copies are built from."""
        return int(self.source.max()) + 1

    @property
    def periodic(self) -> bool:
        """Whether the copies live in a periodic box."""
        return self.box is not None

    @property
    def is_identity(self) -> bool:
        """One copy of one coordinate set, untransformed."""
        return (
            self.n_copies == 1
            and np.array_equal(self.rotation[0], np.eye(3))
            and not self.translation.any()
        )

    def positions(self, xyz: torch.Tensor) -> torch.Tensor:
        """Place every copy, differentiably in ``xyz``.

        Parameters
        ----------
        xyz : torch.Tensor
            Cartesian coordinates in Å, shape ``(S, n, 3)``, or ``(n, 3)`` for a single
            coordinate set.

        Returns
        -------
        torch.Tensor
            Shape ``(C, n, 3)`` in Å, on ``xyz``'s device and dtype. The identity layout
            returns ``xyz`` itself, unsqueezed.
        """
        if xyz.dim() == 2:
            xyz = xyz.unsqueeze(0)
        if xyz.dim() != 3 or xyz.shape[-1] != 3 or xyz.shape[0] != self.n_sources:
            raise ValueError(
                f"Expected ({self.n_sources}, n_atoms, 3) coordinates, got "
                f"{tuple(xyz.shape)}"
            )
        if self.is_identity:
            return xyz
        rotation, translation, source = self._on(xyz)
        picked = xyz.index_select(0, source)
        return torch.einsum("cij,cnj->cni", rotation, picked) + translation[:, None]

    def to_source_frame(self, positions: np.ndarray) -> np.ndarray:
        """Undo each copy's placement: ``x = R_cᵀ (y - t_c)``.

        Parameters
        ----------
        positions : numpy.ndarray
            Shape ``(C, n, 3)`` in Å.

        Returns
        -------
        numpy.ndarray
            Shape ``(C, n, 3)``: each copy back in the frame of its source.
        """
        shifted = positions - self.translation[:, None]
        return np.einsum("cji,cnj->cni", self.rotation, shifted)

    def presence(
        self,
        xyz: np.ndarray,
        molecule_of: np.ndarray,
        heavy: np.ndarray,
        cutoff: float,
    ) -> np.ndarray:
        """Which copies hold each molecule.

        Copies are visited in order. A copy of a molecule is left out when one of its
        heavy atoms lies within ``cutoff`` of a heavy atom of an earlier copy of the
        same molecule that was kept, under the minimum image of the box. Overlaps
        between different molecules are not resolved.

        Parameters
        ----------
        xyz : numpy.ndarray
            Shape ``(S, n, 3)`` in Å, the atoms ``molecule_of`` labels.
        molecule_of : numpy.ndarray
            Molecule of each atom, shape ``(n,)``, labels ``0 .. M-1``.
        heavy : numpy.ndarray
            True for non-hydrogen atoms, shape ``(n,)``. A molecule without heavy atoms
            is compared on all of its atoms.
        cutoff : float
            Overlap distance in Å; ``0`` keeps every copy.

        Returns
        -------
        numpy.ndarray
            Shape ``(C, M)``, bool. All True for a non-periodic layout.
        """
        from scipy.spatial import cKDTree

        n_molecules = int(molecule_of.max()) + 1 if len(molecule_of) else 0
        present = np.ones((self.n_copies, n_molecules), dtype=bool)
        if not self.periodic or self.n_copies == 1 or cutoff <= 0:
            return present
        placed = (
            np.einsum(
                "cij,cnj->cni",
                self.rotation,
                np.asarray(xyz, dtype=np.float64)[self.source],
            )
            + self.translation[:, None]
        )
        box_inv = np.linalg.inv(self.box)
        for m in range(n_molecules):
            atoms = np.flatnonzero((molecule_of == m) & heavy)
            if not len(atoms):
                atoms = np.flatnonzero(molecule_of == m)
            copies = placed[:, atoms]
            centre = copies.mean(axis=1)
            reach = (
                2.0 * np.linalg.norm(copies - centre[:, None], axis=-1).max() + cutoff
            )
            for c in range(1, self.n_copies):
                kept = np.flatnonzero(present[:c, m])
                frac = (centre[c] - centre[kept]) @ box_inv.T
                image = np.round(frac)
                near = np.linalg.norm((frac - image) @ self.box.T, axis=-1) < reach
                for k in np.flatnonzero(near):
                    other = copies[kept[k]] + image[k] @ self.box.T
                    distance, _ = cKDTree(other).query(copies[c], k=1)
                    if distance.min() < cutoff:
                        present[c, m] = False
                        break
        return present

    def _on(self, xyz: torch.Tensor):
        """``(rotation, translation, source)`` as tensors on ``xyz``'s device and dtype."""
        key = (xyz.device, xyz.dtype)
        if key not in self._tensors:
            self._tensors[key] = (
                torch.as_tensor(self.rotation, dtype=xyz.dtype, device=xyz.device),
                torch.as_tensor(self.translation, dtype=xyz.dtype, device=xyz.device),
                torch.as_tensor(self.source, dtype=get_int_dtype(), device=xyz.device),
            )
        return self._tensors[key]


def reduce_box(box: np.ndarray) -> np.ndarray:
    """Bring periodic box vectors into the reduced form OpenMM requires.

    OpenMM wants ``a = (a_x, 0, 0)``, ``b = (b_x, b_y, 0)``, ``c = (c_x, c_y, c_z)``
    with positive diagonal and ``|b_x| < a_x/2``, ``|c_x| < a_x/2``, ``|c_y| < b_y/2``.
    A crystallographic orthogonalisation matrix already has the triangular shape;
    lattice vectors are subtracted until the off-diagonal bounds hold.

    Parameters
    ----------
    box : numpy.ndarray
        Box vectors as columns in Å, shape ``(3, 3)``, a along x and b in the xy-plane.

    Returns
    -------
    numpy.ndarray
        Shape ``(3, 3)``, the same lattice in reduced form.

    Raises
    ------
    ValueError
        If the box is not in the standard orientation.
    """
    a, b, c = (np.array(box[:, k], dtype=np.float64) for k in range(3))
    # Entries that should be exactly zero carry float noise from cos(90°).
    tol = 1e-5 * max(np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(c))
    for vector, index in ((a, 1), (a, 2), (b, 2), (c, 0), (c, 1)):
        if abs(vector[index]) < tol:
            vector[index] = 0.0
    if a[1] or a[2] or b[2] or min(a[0], b[1], c[2]) <= 0:
        raise ValueError("Box vectors must have a along x and b in the xy-plane")
    c = c - b * np.round(c[1] / b[1])
    c = c - a * np.round(c[0] / a[0])
    b = b - a * np.round(b[0] / a[0])

    # A hexagonal cell puts b_x exactly on the -a_x/2 boundary, which OpenMM's strict
    # inequality rejects; a 1e-9 relative shift moves it inside.
    def inside(value: float, half: float) -> float:
        if abs(value) >= half * (1 - 1e-12):
            return math.copysign(half * (1.0 - 1e-9), value)
        return value

    b[0] = inside(b[0], a[0] / 2)
    c[0] = inside(c[0], a[0] / 2)
    c[1] = inside(c[1], b[1] / 2)
    return np.stack([a, b, c], axis=1)


def _orthogonalisation(cell: "Cell") -> np.ndarray:
    """``B`` with ``cart = B @ frac``, as float64."""
    return cell.fractional_matrix.detach().cpu().numpy().astype(np.float64)


def _operations(spacegroup: "SpaceGroup") -> Tuple[np.ndarray, np.ndarray]:
    """Fractional rotations ``(n_ops, 3, 3)`` and translations ``(n_ops, 3)``."""
    return (
        spacegroup.matrices.detach().cpu().numpy().astype(np.float64),
        spacegroup.translations.detach().cpu().numpy().astype(np.float64),
    )
