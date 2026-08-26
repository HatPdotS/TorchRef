"""Contiguous, origin-sorted blocks of k-ary edges.

An :class:`EdgeBlock` is the storage for one edge type -- bonds, angles, torsions,
chirals, or the planes of one atom count. Rows are sorted by origin first and
lexicographically by atom index within an origin, and ``origin_bounds`` records where
each origin's rows begin and end. Every per-origin subset is therefore a **slice** of
the block: a view that shares storage, costs nothing to take, and reflects an in-place
edit to the block immediately.

Nothing here is refinable. Indices are ``int64``, no gradient reaches them, and the
block is a constant for the lifetime of a topology unless the topology is mutated.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch

from torchref.utils.device_mixin import DeviceMixin

#: Origin order per edge type. Fixes the block layout so a rebuild on the same
#: structure lays rows out identically, in any process. The previous storage derived
#: this order from Python ``set`` iteration, which is string-hash-seed dependent.
ORIGIN_ORDER: Dict[str, Tuple[str, ...]] = {
    "bond": ("intra", "peptide", "disulfide", "link"),
    "angle": ("intra", "peptide", "disulfide"),
    # intra and disulfide lead because together they are the ``all`` group the
    # torsion target reads; adjacent origins make that group a view rather than a copy.
    "torsion": ("intra", "disulfide", "phi", "psi", "omega"),
    "chiral": ("intra",),
    "plane": ("intra", "peptide"),
}


def _lexsort_rows(rows: np.ndarray) -> np.ndarray:
    """Row order that sorts ``rows`` lexicographically, left column most significant.

    Parameters
    ----------
    rows : numpy.ndarray
        Integer array of shape ``(E, k)``.

    Returns
    -------
    numpy.ndarray
        Permutation of ``arange(E)``. Total on the row values, so it does not depend
        on the incoming order the way a single-column ``argsort`` does.
    """
    if rows.size == 0:
        return np.zeros(0, dtype=np.int64)
    return np.lexsort(tuple(rows[:, c] for c in reversed(range(rows.shape[1]))))


def assemble_origins(
    per_origin: Dict[str, np.ndarray],
    arity: int,
    edge_type: str,
    payload: Dict[str, Dict[str, np.ndarray]] = None,
) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]], Dict[str, Dict[str, np.ndarray]]]:
    """Lay origins out in canonical order, carrying per-edge values through the sort.

    Origins follow :data:`ORIGIN_ORDER` for ``edge_type``, and rows within an origin
    are sorted lexicographically. Anything in ``payload`` is permuted by the same order,
    so a value array stays aligned row-for-row with the indices it belongs to, which
    is the whole reason values cannot be concatenated separately.

    Parameters
    ----------
    per_origin : dict
        ``{origin: (E_o, k) integer array}``. Empty entries are skipped.
    arity : int
        Atoms per edge.
    edge_type : str
        Key into :data:`ORIGIN_ORDER`.
    payload : dict, optional
        ``{origin: {property: array}}``, each array indexed by row on axis 0. A property
        need not be present for every origin -- ``phi`` and ``psi`` carry no reference
        value or sigma, and must not acquire one here.

    Returns
    -------
    indices : numpy.ndarray
        Shape ``(E, k)``, canonical order.
    bounds : dict
        ``{origin: (start, end)}``, contiguous and covering the block.
    sorted_payload : dict
        ``{origin: {property: array}}``, permuted to match ``indices``.
    """
    order = ORIGIN_ORDER.get(edge_type, tuple(sorted(per_origin)))
    unknown = set(per_origin) - set(order)
    if unknown:
        raise ValueError(
            f"{edge_type}: origins {sorted(unknown)} are not in ORIGIN_ORDER"
            f"[{edge_type!r}] = {order}. Add them there so the layout stays "
            f"deterministic."
        )

    payload = payload or {}
    chunks: List[np.ndarray] = []
    bounds: Dict[str, Tuple[int, int]] = {}
    sorted_payload: Dict[str, Dict[str, np.ndarray]] = {}
    cursor = 0

    for origin in order:
        rows = per_origin.get(origin)
        if rows is None or len(rows) == 0:
            continue
        rows = np.asarray(rows, dtype=np.int64).reshape(-1, arity)
        permutation = _lexsort_rows(rows)
        chunks.append(rows[permutation])
        bounds[origin] = (cursor, cursor + len(rows))
        cursor += len(rows)

        origin_payload = payload.get(origin) or {}
        if origin_payload:
            sorted_payload[origin] = {
                prop: np.asarray(values)[permutation]
                for prop, values in origin_payload.items()
                if values is not None
            }

    indices = (
        np.concatenate(chunks, axis=0)
        if chunks
        else np.zeros((0, arity), dtype=np.int64)
    )
    return indices, bounds, sorted_payload


@dataclass(eq=False, repr=False)
class EdgeBlock(DeviceMixin):
    """One edge type's index block plus its per-origin bounds.

    Parameters
    ----------
    indices : torch.Tensor
        Atom indices, shape ``(E, k)``, dtype ``int64``, in canonical order.
    origin_bounds : dict
        ``{origin: (start, end)}`` half-open row ranges into ``indices``. Ranges are
        contiguous and cover the block.

    Notes
    -----
    Holds no refinable parameters, so this is a dataclass rather than an
    ``nn.Module``; ``DeviceMixin`` still moves ``indices`` with ``.to(device)``.
    """

    indices: torch.Tensor
    origin_bounds: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    @classmethod
    def empty(cls, arity: int, device=None) -> "EdgeBlock":
        """An edge-free block of the given arity."""
        return cls(
            indices=torch.zeros((0, arity), dtype=torch.int64, device=device),
            origin_bounds={},
        )

    @classmethod
    def from_origins(
        cls,
        per_origin: Dict[str, np.ndarray],
        arity: int,
        edge_type: str,
        device=None,
    ) -> "EdgeBlock":
        """Assemble a canonical block from ``{origin: (E_o, k) index array}``.

        Origins are laid out in :data:`ORIGIN_ORDER` for ``edge_type``, and rows
        within an origin are sorted lexicographically. Origins with no rows are
        omitted from ``origin_bounds`` rather than recorded as empty ranges.

        Parameters
        ----------
        per_origin : dict
            Integer index arrays keyed by origin. Empty arrays are skipped.
        arity : int
            Atoms per edge (2 for bonds, 3 for angles, ...).
        edge_type : str
            Key into :data:`ORIGIN_ORDER`.
        device : torch.device, optional
            Where to place the block.

        Returns
        -------
        EdgeBlock
        """
        indices, bounds, _ = assemble_origins(per_origin, arity, edge_type)
        if len(indices) == 0:
            return cls.empty(arity, device=device)
        return cls(
            indices=torch.as_tensor(indices, dtype=torch.int64, device=device),
            origin_bounds=bounds,
        )

    @property
    def device(self) -> torch.device:
        """Where the block lives. Derived, so it cannot fall out of step."""
        return self.indices.device

    @property
    def n_edges(self) -> int:
        """Number of rows in the block."""
        return int(self.indices.shape[0])

    @property
    def arity(self) -> int:
        """Atoms per edge."""
        return int(self.indices.shape[1])

    def origins(self) -> List[str]:
        """Origins present, in block layout order."""
        return sorted(self.origin_bounds, key=lambda o: self.origin_bounds[o][0])

    def origin(self, name: str) -> torch.Tensor:
        """Rows contributed by one origin, as a **view** into the block.

        Parameters
        ----------
        name : str
            Origin key.

        Returns
        -------
        torch.Tensor
            Shape ``(E_o, k)``. Shares storage with :attr:`indices`, so an in-place
            edit to either is visible through the other.

        Raises
        ------
        KeyError
            If the origin contributed no rows.
        """
        start, end = self.origin_bounds[name]
        return self.indices[start:end]

    def tuple_set(self, origin: str = None) -> set:
        """Edges as a set of index tuples, for order-free comparison.

        Parameters
        ----------
        origin : str, optional
            Restrict to one origin. None means the whole block.

        Returns
        -------
        set of tuple of int
        """
        rows = self.indices if origin is None else self.origin(origin)
        return {tuple(int(v) for v in row) for row in rows.cpu().numpy()}

    def __repr__(self) -> str:
        return (
            f"EdgeBlock(arity={self.arity}, n_edges={self.n_edges}, "
            f"origins={self.origins()})"
        )


__all__ = ["EdgeBlock", "ORIGIN_ORDER", "assemble_origins"]
