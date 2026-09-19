"""Ideal values and sigmas, layered over a topology's edges.

The topology says what is connected. What the ideal geometry *is* lives here, keyed to
the same edges, so one connectivity can carry monomer-library targets, force-field
parameters or ADP-similarity sigmas without a second copy of the edges.

:func:`assemble_entries` turns a topology plus its values into the nested mapping the
restraint consumers read, ``entries[edge_type][origin][property]``. Indices are
**views** into the contiguous edge blocks, so taking a per-origin subset costs nothing
and an in-place edit to a block is visible through every view of it. Access is three
dict lookups with no allocation, which is the point: it sits on the geometry targets'
hot path.
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from torchref.config import get_float_dtype

#: Origins making up each edge type's ``all`` group -- what the geometry targets read.
#: ``None`` means every origin present. ``phi`` and ``psi`` are conformationally free
#: and carry no target, and ``omega`` has its own von Mises target, so the torsion group
#: holds only the two origins that are ordinary restrained torsions.
ALL_MEMBERS: Dict[str, Optional[Tuple[str, ...]]] = {
    "bond": None,
    "angle": None,
    "torsion": ("intra", "disulfide"),
}

#: Integer-valued edge properties, kept as ``int64`` rather than the float dtype.
_INTEGER_PROPERTIES = frozenset({"periods", "symop_indices", "cell_offsets"})

#: Boolean edge properties.
_BOOL_PROPERTIES = frozenset({"is_proline"})


def to_tensor(values, prop: str, device=None) -> torch.Tensor:
    """A per-edge value array as a tensor of the dtype that property calls for."""
    if isinstance(values, torch.Tensor):
        return values.to(device=device) if device is not None else values
    if prop in _INTEGER_PROPERTIES:
        dtype = torch.int64  # dtype-ok: dtype var for index tensors; int64 index required
    elif prop in _BOOL_PROPERTIES:
        dtype = torch.bool
    else:
        dtype = get_float_dtype()
    return torch.as_tensor(np.asarray(values), dtype=dtype, device=device)


def _contiguous_span(
    bounds: Dict[str, Tuple[int, int]], members: Sequence[str]
) -> Optional[Tuple[int, int]]:
    """The single row range covering ``members``, or None if they are not adjacent.

    Adjacency is what makes the ``all`` group a view instead of a copy, so the block
    layout in :data:`~torchref.topology.edges.ORIGIN_ORDER` deliberately keeps each edge
    type's ``all`` members together.
    """
    present = [m for m in members if m in bounds]
    if not present:
        return None
    spans = sorted(bounds[m] for m in present)
    for (_, end), (start, _) in zip(spans, spans[1:]):
        if end != start:
            return None
    return spans[0][0], spans[-1][1]


def _group(
    indices: torch.Tensor, values: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """One restraint group: its indices plus whatever properties it carries."""
    group = {"indices": indices}
    group.update(values)
    return group


def _all_group(
    block, per_origin_values: Dict[str, Dict[str, torch.Tensor]], members
) -> Optional[Dict[str, torch.Tensor]]:
    """The combined group a geometry target reads, or None when it would be empty.

    Carries only properties present in **every** member origin, so a property one member
    lacks does not silently become a partial array.
    """
    origins = (
        block.origins()
        if members is None
        else [m for m in members if m in block.origin_bounds]
    )
    if not origins:
        return None

    span = _contiguous_span(block.origin_bounds, origins)
    if span is not None:
        start, end = span
        indices = block.indices[start:end]
    else:
        # Not reachable with the shipped layouts; kept so a future origin order that
        # separates the members degrades to a copy rather than silently misaligning.
        indices = torch.cat([block.origin(o) for o in origins], dim=0)

    shared = set(per_origin_values.get(origins[0], {}))
    for origin in origins[1:]:
        shared &= set(per_origin_values.get(origin, {}))

    values = {
        prop: torch.cat([per_origin_values[o][prop] for o in origins], dim=0)
        for prop in sorted(shared)
    }
    return _group(indices, values)


def assemble_entries(
    topology,
    values: Dict[str, Dict[str, Dict[str, torch.Tensor]]],
) -> Dict[str, Dict]:
    """Build the nested mapping the restraint consumers read.

    Parameters
    ----------
    topology : Topology
        Supplies the edge blocks; per-origin indices come out as views into them.
    values : dict
        ``{edge_type: {origin: {property: tensor}}}`` for the keyed types, plus
        ``{'chiral': {property: tensor}}`` and ``{'plane': {size: {property: tensor}}}``
        for the two that carry no origin.

    Returns
    -------
    dict
        ``entries[edge_type][origin][property]`` for bonds, angles and torsions;
        ``entries['plane']['4_atoms'][property]``; ``entries['chiral'][property]``.
        ``entries['vdw']`` starts empty and is filled when the pair list is built.
    """
    entries: Dict[str, Dict] = {}

    for edge_type in ("bond", "angle", "torsion"):
        block = topology.edge_block(edge_type)
        per_origin = values.get(edge_type, {})
        group: Dict[str, Dict[str, torch.Tensor]] = {}
        for origin in block.origins():
            group[origin] = _group(block.origin(origin), per_origin.get(origin, {}))
        combined = _all_group(block, per_origin, ALL_MEMBERS[edge_type])
        if combined is not None:
            group["all"] = combined
        entries[edge_type] = group

    chirals = topology.atoms.chirals
    entries["chiral"] = (
        _group(chirals.indices, values.get("chiral", {})) if chirals.n_edges else {}
    )

    entries["plane"] = {
        f"{size}_atoms": _group(block.indices, values.get("plane", {}).get(size, {}))
        for size, block in sorted(topology.atoms.planes.items())
    }

    entries["vdw"] = {}
    return entries


def max_period(entries: Dict[str, Dict]) -> int:
    """Largest torsion period in the ``all`` group.

    Read once at build time so the torsion target does not pay a device sync for it on
    every iteration.
    """
    group = entries.get("torsion", {}).get("all")
    if not group:
        return 1
    periods = group.get("periods")
    if periods is None or periods.numel() == 0:
        return 1
    return int(periods.max().item())


__all__ = ["assemble_entries", "max_period", "to_tensor", "ALL_MEMBERS"]
