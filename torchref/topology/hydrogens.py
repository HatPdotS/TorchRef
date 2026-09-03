"""Hydrogen generation as a graph operation: expand the template, map it on.

A monomer template already carries its hydrogens, with coordinates and with bonds naming
each one's parent. Generating hydrogens is therefore template instantiation, not
geometry reconstruction: align the template onto the heavy atoms that are present,
read the hydrogen positions off it, and correct each to its ideal bond length.

Two things the bond graph decides that a distance criterion previously guessed at:

* **How many hydrogens a parent can carry.** The count is the parent's standard valence
  minus the heavy atoms actually bonded to it, taken from the graph. A distance sweep
  gets this wrong on a distorted or predicted model, where a bond can fall outside the
  window; and it cannot distinguish a real bond from two atoms that merely sit close.
* **Which hydrogens have a free torsion.** A hydrogen whose parent has exactly one heavy
  neighbour -- hydroxyl, thiol, amine, methyl -- can rotate about the parent-neighbour
  axis, and the template's angle for it is arbitrary. Those get scanned; the rest are
  fully determined by the template and are left alone.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

#: Standard heavy-atom valences, used to cap how many hydrogens a parent may take. The
#: fallback of 4 matches the previous behaviour for elements not listed.
STANDARD_VALENCE = {"C": 4, "N": 3, "O": 2, "S": 2}
_DEFAULT_VALENCE = 4

#: A template hydrogen further than this from its parent after alignment is discarded
#: rather than corrected: the alignment for that centre is too poor to trust.
MAX_PLACEMENT_DISTANCE = 1.5

#: Angles tried when scanning a free torsion. 24 gives 15-degree resolution, which is
#: finer than the placement error the alignment itself carries.
TORSION_SCAN_STEPS = 24

#: Heavy atoms beyond this distance cannot clash with a hydrogen, so the scan ignores
#: them.
CLASH_CUTOFF = 4.0


@dataclass
class HydrogenPlan:
    """Hydrogens to add, and where to put them.

    Parameters
    ----------
    name, element, altloc : numpy.ndarray
        Per-hydrogen identity, shape ``(H,)``.
    residue : numpy.ndarray
        Residue index each hydrogen belongs to, shape ``(H,)``.
    parent : numpy.ndarray
        Atom-table index of each hydrogen's parent, shape ``(H,)``.
    position : numpy.ndarray
        Cartesian coordinates, shape ``(H, 3)``.
    bond_length : numpy.ndarray
        Ideal parent-hydrogen distance, shape ``(H,)``.
    group : numpy.ndarray
        Free-torsion group id, shape ``(H,)``; ``-1`` for a hydrogen whose position the
        template determines. Hydrogens on one parent that rotate together share an id.
    """

    name: np.ndarray
    element: np.ndarray
    altloc: np.ndarray
    residue: np.ndarray
    parent: np.ndarray
    position: np.ndarray
    bond_length: np.ndarray
    group: np.ndarray

    @property
    def n_hydrogens(self) -> int:
        """How many hydrogens the plan adds."""
        return len(self.name)

    @property
    def rotatable(self) -> np.ndarray:
        """Mask of hydrogens whose torsion is free."""
        return self.group >= 0

    def __repr__(self) -> str:
        return (
            f"HydrogenPlan(n_hydrogens={self.n_hydrogens}, "
            f"free_torsions={len(set(self.group[self.rotatable].tolist()))})"
        )


def _kabsch(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rotation and translation carrying ``source`` onto ``target``.

    Reflections are excluded, so the template's chirality survives the alignment.

    Returns
    -------
    rotation, translation : numpy.ndarray
        Shapes ``(3, 3)`` and ``(3,)``; ``rotation @ p + translation`` maps a source
        point.
    """
    source_centre, target_centre = source.mean(0), target.mean(0)
    covariance = (source - source_centre).T @ (target - target_centre)
    u, _, vt = np.linalg.svd(covariance)
    flip = np.diag([1.0, 1.0, 1.0 if np.linalg.det(vt.T @ u.T) > 0 else -1.0])
    rotation = vt.T @ flip @ u.T
    return rotation, target_centre - rotation @ source_centre


def _template(cif_dict: Dict, resname: str) -> Optional[Dict]:
    """Template atoms, hydrogen parents, ideal bond lengths and heavy adjacency.

    Read from the restraint dictionary the caller already loaded. The link modifications
    are deliberately not consulted: they rewrite restraint sections, not atom lists, so
    they cannot say which hydrogens a linked residue keeps. The valence cap answers that
    from the bond graph instead.

    Returns
    -------
    dict or None
        None when the component is absent or its atoms carry no coordinates.
    """
    component = cif_dict.get(resname)
    if component is None:
        return None
    atoms = component.get("atoms")
    if atoms is None or len(atoms) == 0:
        return None
    if not all(column in atoms.columns for column in ("x", "y", "z")):
        return None

    ids = atoms["atom_id"].astype(str).str.strip().values
    elements = atoms["type_symbol"].astype(str).str.strip().values.astype("<U4")
    coords = atoms[["x", "y", "z"]].values.astype(np.float64)
    if not np.isfinite(coords).all():
        return None
    is_h = np.char.upper(elements) == "H"
    id_to_index = {name: i for i, name in enumerate(ids)}

    parent_of: Dict[str, str] = {}
    ideal_length: Dict[str, float] = {}
    heavy_adjacency: Dict[str, List[str]] = {}

    bonds = component.get("bonds")
    if bonds is not None and len(bonds) > 0:
        import pandas as pd

        first = bonds["atom1"].astype(str).str.strip().values
        second = bonds["atom2"].astype(str).str.strip().values
        values = pd.to_numeric(bonds["value"], errors="coerce").values
        for i in range(len(first)):
            a, b = first[i], second[i]
            ia, ib = id_to_index.get(a), id_to_index.get(b)
            if ia is None or ib is None:
                continue
            if is_h[ia] and not is_h[ib]:
                parent_of[a] = b
                if np.isfinite(values[i]):
                    ideal_length[a] = float(values[i])
            elif is_h[ib] and not is_h[ia]:
                parent_of[b] = a
                if np.isfinite(values[i]):
                    ideal_length[b] = float(values[i])
            elif not is_h[ia] and not is_h[ib]:
                heavy_adjacency.setdefault(a, []).append(b)
                heavy_adjacency.setdefault(b, []).append(a)

    return {
        "ids": ids,
        "elements": elements,
        "coords": coords,
        "is_h": is_h,
        "id_to_index": id_to_index,
        "heavy_names": ids[~is_h],
        "heavy_coords": coords[~is_h],
        "h_names": ids[is_h],
        "parent_of": parent_of,
        "ideal_length": ideal_length,
        "heavy_adjacency": heavy_adjacency,
    }


def _orthonormal_frame(axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Two unit vectors completing ``axis`` into a right-handed frame."""
    seed = np.array([1.0, 0.0, 0.0])
    if abs(float(axis @ seed)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    first = seed - axis * float(axis @ seed)
    first = first / np.linalg.norm(first)
    return first, np.cross(axis, first)


def _axis_frame_placement(
    template: Dict,
    parent_name: str,
    neighbour_name: str,
    parent_position: np.ndarray,
    neighbour_position: np.ndarray,
    h_names: List[str],
) -> Optional[np.ndarray]:
    """Hydrogen positions for a centre with a single heavy neighbour.

    Maps the template's local geometry onto the model by carrying the parent-neighbour
    axis across and completing the frame arbitrarily. Every bond angle at the parent is
    preserved exactly; only the rotation about the axis is arbitrary, which is correct:
    that is the degree of freedom the template cannot know, and
    :func:`optimise_free_torsions` chooses it.
    """
    index = template["id_to_index"]
    if parent_name not in index or neighbour_name not in index:
        return None

    template_axis = (
        template["coords"][index[parent_name]]
        - template["coords"][index[neighbour_name]]
    )
    model_axis = parent_position - neighbour_position
    for vector in (template_axis, model_axis):
        if np.linalg.norm(vector) < 1e-8:
            return None
    template_axis = template_axis / np.linalg.norm(template_axis)
    model_axis = model_axis / np.linalg.norm(model_axis)

    t_first, t_second = _orthonormal_frame(template_axis)
    m_first, m_second = _orthonormal_frame(model_axis)

    positions = []
    for name in h_names:
        if name not in index:
            return None
        offset = (
            template["coords"][index[name]] - template["coords"][index[parent_name]]
        )
        positions.append(
            parent_position
            + float(offset @ template_axis) * model_axis
            + float(offset @ t_first) * m_first
            + float(offset @ t_second) * m_second
        )
    return np.array(positions)


def _half_hydrogen_angle(template: Dict, parent_name: str, h_names: List[str]) -> float:
    """Half the hydrogen-parent-hydrogen angle, from the template where it has one."""
    index = template["id_to_index"]
    if len(h_names) == 2 and all(n in index for n in h_names):
        parent = template["coords"][index[parent_name]]
        first = template["coords"][index[h_names[0]]] - parent
        second = template["coords"][index[h_names[1]]] - parent
        norms = np.linalg.norm(first) * np.linalg.norm(second)
        if norms > 1e-12:
            cosine = float(np.clip((first @ second) / norms, -1.0, 1.0))
            return 0.5 * float(np.arccos(cosine))
    # Tetrahedral, as a fallback for a template that does not carry both hydrogens.
    return 0.5 * np.arccos(-1.0 / 3.0)


def _split_neighbours(topology, atom_index: int) -> Tuple[np.ndarray, int]:
    """Heavy neighbour rows of ``atom_index``, and how many hydrogens it already has.

    Coordinate-independent, unlike a distance sweep: a stretched bond in a predicted or
    mid-refinement model still counts, and two atoms that merely sit close do not.

    The hydrogen count is what makes generation idempotent and makes a partially
    hydrogenated structure top up correctly. Both consume the parent's valence, so
    subtracting only the heavy neighbours leaves budget for a hydrogen the parent
    already carries -- which is how a second pass came to add the free-amino-acid ``H2``
    to every backbone nitrogen that already had its ``H``.
    """
    neighbours = topology.atoms.neighbors(atom_index)
    if neighbours.numel() == 0:
        return np.zeros(0, dtype=np.int64), 0
    is_h = topology.atoms.is_hydrogen[neighbours]
    return neighbours[~is_h].cpu().numpy(), int(is_h.sum())


def _template_bond_length(template: Dict, parent_name: str, h_name: str) -> float:
    """Parent-hydrogen distance in the template, or NaN if either atom is missing."""
    index = template["id_to_index"]
    if parent_name not in index or h_name not in index:
        return float("nan")
    return float(
        np.linalg.norm(
            template["coords"][index[h_name]] - template["coords"][index[parent_name]]
        )
    )


def _place_group(
    template: Dict,
    parent_name: str,
    parent_position: np.ndarray,
    neighbour_positions: np.ndarray,
    heavy_bonded: int,
    h_names: List[str],
    lengths: np.ndarray,
    name_to_row: Dict[str, int],
    coords: np.ndarray,
    template_names_of: List[str],
) -> Optional[np.ndarray]:
    """Positions for the hydrogens on one parent, by the first strategy that applies.

    In order:

    1. **Template frame.** A Kabsch fit over the parent and its immediate heavy
       neighbours, used only when the template knows every heavy atom actually bonded to
       the parent. Reproduces the library geometry exactly, torsions included.
    2. **Geometric construction.** Directions from the bonded neighbours alone, for a
       centre whose template is missing a real substituent -- a peptide-linked backbone
       nitrogen being the common case.
    3. **Axis frame.** For a single-neighbour centre, the template geometry carried over
       about the one bond, leaving the rotation about it for the scan to choose.

    Returns None when none applies, so the caller can count the hydrogen as undetermined
    rather than putting it somewhere arbitrary.
    """
    covered = _template_covers_neighbours(template, parent_name, heavy_bonded)

    if covered:
        alignment = _alignment_for(template, parent_name, name_to_row, coords)
        if alignment is not None:
            matrix, offset = alignment
            index = template["id_to_index"]
            positions = []
            for name, length in zip(h_names, lengths):
                direction = (
                    matrix @ template["coords"][index[name]] + offset
                ) - parent_position
                distance = float(np.linalg.norm(direction))
                if distance < 1e-6 or distance > MAX_PLACEMENT_DISTANCE:
                    positions = None
                    break
                positions.append(parent_position + direction * (length / distance))
            if positions is not None:
                return np.array(positions)

    if heavy_bonded >= 2:
        directions = _construct_directions(
            parent_position,
            neighbour_positions,
            len(h_names),
            _half_hydrogen_angle(template, parent_name, h_names),
        )
        if directions is not None:
            return parent_position + directions * lengths[:, None]

    if heavy_bonded == 1 and template_names_of:
        positions = _axis_frame_placement(
            template,
            parent_name,
            template_names_of[0],
            parent_position,
            neighbour_positions[0],
            h_names,
        )
        if positions is None:
            return None
        # Rescale along each direction so the bond length is the library value rather
        # than the template's own geometry, which differs from it by ~0.002 A. Scaling
        # along the direction leaves every bond angle untouched.
        offsets = positions - parent_position
        norms = np.linalg.norm(offsets, axis=1)
        if (norms < 1e-8).any():
            return None
        return parent_position + offsets * (lengths / norms)[:, None]

    return None


def plan_hydrogens(topology, cif_dict: Dict, xyz, verbose: int = 0) -> HydrogenPlan:
    """Decide which hydrogens to add and place them from the template.

    Parameters
    ----------
    topology : Topology
        Supplies the residue partition, the per-residue template key and the bond graph
        the valence cap and the free-torsion test read.
    cif_dict : dict
        Restraint dictionary, keyed by residue name; must carry an ``atoms`` section
        with coordinates for a residue to be hydrogenated.
    xyz : torch.Tensor
        Current coordinates, shape ``(N, 3)``.
    verbose : int, default 0
        Verbosity level.

    Returns
    -------
    HydrogenPlan
    """
    coords = np.asarray(xyz.detach().cpu(), dtype=np.float64)
    residues = topology.residues
    atoms = topology.atoms
    names = atoms.name.astype(str)
    altlocs = atoms.altloc.astype(str)

    out: Dict[str, list] = {
        k: []
        for k in (
            "name",
            "altloc",
            "residue",
            "parent",
            "position",
            "bond_length",
            "group",
        )
    }
    next_group = 0
    n_unplaceable = 0
    n_no_template = 0

    for residue in range(residues.n_residues):
        resname = str(residues.resname[residue]).strip()
        template = _template(cif_dict, resname)
        if template is None:
            # No usable template. In practice these are the single-atom residues --
            # waters and ions -- which the restraint dictionary omits because they carry
            # no intra-residue geometry. They could not be hydrogenated anyway: one
            # heavy atom gives no frame to orient a template against and no bond to
            # rotate about, so a water's hydrogens would point somewhere arbitrary.
            n_no_template += 1
            continue

        rows = np.arange(
            int(residues.atom_start[residue]), int(residues.atom_end[residue])
        )
        present = set(names[rows])
        candidates = [h for h in template["h_names"] if h not in present]
        if not candidates:
            continue

        for altloc, conformer in _conformer_rows(rows, altlocs):
            name_to_row = {}
            for row in conformer:
                name_to_row.setdefault(names[row], row)

            # Hydrogens grouped by the parent they hang off, in name order so the cap
            # below takes a deterministic subset.
            by_parent: Dict[str, List[str]] = {}
            for h in sorted(candidates):
                parent = template["parent_of"].get(h)
                if parent is not None and parent in name_to_row:
                    by_parent.setdefault(parent, []).append(h)

            for parent_name, group in by_parent.items():
                parent_row = name_to_row[parent_name]
                parent_position = coords[parent_row]

                heavy_rows, existing_h = _split_neighbours(topology, parent_row)
                heavy_bonded = len(heavy_rows)
                element = str(
                    template["elements"][template["id_to_index"][parent_name]]
                ).upper()
                valence = STANDARD_VALENCE.get(element, _DEFAULT_VALENCE)
                allowed = max(0, valence - heavy_bonded - existing_h)
                group = group[:allowed]
                if not group:
                    continue

                lengths = np.array(
                    [
                        template["ideal_length"].get(
                            h, _template_bond_length(template, parent_name, h)
                        )
                        for h in group
                    ]
                )
                if not np.isfinite(lengths).all():
                    n_unplaceable += len(group)
                    continue

                placed = _place_group(
                    template,
                    parent_name,
                    parent_position,
                    coords[heavy_rows] if heavy_bonded else np.zeros((0, 3)),
                    heavy_bonded,
                    group,
                    lengths,
                    name_to_row,
                    coords,
                    template_names_of=[
                        n
                        for n in template["heavy_adjacency"].get(parent_name, [])
                        if n in name_to_row
                    ],
                )
                if placed is None:
                    n_unplaceable += len(group)
                    continue

                # One free-torsion group per parent with a single heavy neighbour: its
                # hydrogens rotate together about that one bond.
                group_id = -1
                if heavy_bonded == 1:
                    group_id = next_group
                    next_group += 1

                for h, position, length in zip(group, placed, lengths):
                    out["name"].append(h)
                    out["altloc"].append(altloc)
                    out["residue"].append(residue)
                    out["parent"].append(int(parent_row))
                    out["position"].append(position)
                    out["bond_length"].append(float(length))
                    out["group"].append(group_id)

    if verbose > 1:
        print(
            f"Planned {len(out['name'])} hydrogens "
            f"({n_no_template} residues with no usable template, "
            f"{n_unplaceable} hydrogens whose direction was not determined)"
        )

    return HydrogenPlan(
        name=np.array(out["name"], dtype="<U8"),
        element=np.full(len(out["name"]), "H", dtype="<U2"),
        altloc=np.array(out["altloc"], dtype="<U1"),
        residue=np.array(out["residue"], dtype=np.int64),
        parent=np.array(out["parent"], dtype=np.int64),
        position=(
            np.array(out["position"], dtype=np.float64).reshape(-1, 3)
            if out["position"]
            else np.zeros((0, 3))
        ),
        bond_length=np.array(out["bond_length"], dtype=np.float64),
        group=np.array(out["group"], dtype=np.int64),
    )


def _conformer_rows(rows: np.ndarray, altlocs: np.ndarray):
    """``(altloc, rows)`` per alternative conformation of one residue.

    A residue without altlocs yields one conformation over all its rows; otherwise one
    per altloc, each carrying the blank-altloc atoms too, so a hydrogen on a shared
    backbone atom is placed once per conformer.
    """
    residue_altlocs = altlocs[rows]
    unique = sorted(set(residue_altlocs) - {"", " "})
    if not unique:
        return [("", rows)]
    shared = np.isin(residue_altlocs, ["", " "])
    return [(altloc, rows[shared | (residue_altlocs == altloc)]) for altloc in unique]


def _alignment_for(
    template: Dict,
    parent_name: str,
    name_to_row: Dict[str, int],
    coords: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Template-to-model transform for one hydrogen-bearing centre.

    Fitted over the parent and its **immediate** heavy neighbours only. That set is the
    rigid unit which fixes the hydrogen directions: the bond lengths and angles at the
    parent are library constants, while anything further out sits across a rotatable
    torsion whose value is the model's, not the template's.

    Reaching one bond further -- as a whole-residue or two-shell fit does -- makes the
    rotation compromise between the real local geometry and a torsion the model does not
    share, which lands hydrogens well off their parent. Measured on 7L84 the two-shell
    fit around ``CB`` aligned to 0.75 A RMSD and put 12% of side-chain hydrogens beyond
    1.5 A of the atom they belong to.

    Returns None when fewer than three neighbours match, which leaves the rotation
    undetermined; the caller then constructs the direction from the bond graph instead.
    """
    local = [parent_name] + [
        n for n in sorted(template["heavy_adjacency"].get(parent_name, []))
    ]
    matched = [n for n in local if n in name_to_row and n in template["id_to_index"]]
    if len(matched) < 3:
        return None
    source = np.array([template["coords"][template["id_to_index"][n]] for n in matched])
    target = np.array([coords[name_to_row[n]] for n in matched])
    return _kabsch(source, target)


def _template_covers_neighbours(
    template: Dict, parent_name: str, graph_heavy_count: int
) -> bool:
    """Whether the template knows every heavy atom actually bonded to the parent.

    A peptide-linked backbone nitrogen is bonded to the preceding residue's carbon,
    which the free-amino-acid template has never heard of. Placing its hydrogen from the
    template frame would ignore that substituent and drop the hydrogen on top of it, so
    those centres are built from the graph instead.
    """
    return len(template["heavy_adjacency"].get(parent_name, [])) >= graph_heavy_count


def _construct_directions(
    parent: np.ndarray,
    neighbours: np.ndarray,
    n_hydrogens: int,
    half_angle: float,
) -> Optional[np.ndarray]:
    """Unit directions for hydrogens on a centre, from its bonded neighbours alone.

    Two rules cover every centre whose orientation the neighbours determine:

    * one hydrogen, any number of neighbours -- it opposes the sum of the bond unit
      vectors, which is where the remaining valence points. This is the backbone amide
      and alpha hydrogens.
    * two hydrogens on a two-neighbour centre -- they straddle that same direction,
      opened out by ``half_angle`` in the plane perpendicular to the neighbour pair.

    Anything else (three hydrogens, or two on a one-neighbour centre) has a free
    torsion and is handled by the scan, not here.

    Returns
    -------
    numpy.ndarray or None
        Shape ``(n_hydrogens, 3)`` unit vectors, or None when the rules do not apply or
        the geometry is degenerate.
    """
    bonds = neighbours - parent
    lengths = np.linalg.norm(bonds, axis=1)
    if (lengths < 1e-8).any():
        return None
    units = bonds / lengths[:, None]

    total = units.sum(0)
    norm = np.linalg.norm(total)
    if norm < 1e-6:
        return None
    opposed = -total / norm

    if n_hydrogens == 1:
        return opposed[None, :]

    if n_hydrogens == 2 and len(units) == 2:
        perpendicular = np.cross(units[0], units[1])
        perpendicular_norm = np.linalg.norm(perpendicular)
        if perpendicular_norm < 1e-6:
            return None
        perpendicular = perpendicular / perpendicular_norm
        cos, sin = np.cos(half_angle), np.sin(half_angle)
        return np.array(
            [
                opposed * cos + perpendicular * sin,
                opposed * cos - perpendicular * sin,
            ]
        )

    return None


def optimise_free_torsions(
    plan: HydrogenPlan,
    topology,
    xyz,
    steps: int = TORSION_SCAN_STEPS,
) -> HydrogenPlan:
    """Rotate each free-torsion hydrogen group to the least-clashing angle.

    The template's dihedral for a hydroxyl, thiol, amine or methyl hydrogen is whatever
    the library happened to deposit, so it carries no information. Each group is scanned
    about its parent-neighbour axis and scored by repulsion against nearby heavy atoms;
    the best angle wins.

    Repulsion only: this removes clashes but does not seek hydrogen bonds, so a hydroxyl
    is placed out of the way rather than donated to an acceptor. Scoring donors properly
    is a separate piece of physics.

    Returns
    -------
    HydrogenPlan
        The same plan with ``position`` updated in place for the scanned groups.
    """
    if plan.n_hydrogens == 0 or not plan.rotatable.any():
        return plan

    coords = np.asarray(xyz.detach().cpu(), dtype=np.float64)
    is_h = topology.atoms.is_hydrogen.cpu().numpy()
    heavy_rows = np.nonzero(~is_h)[0]
    heavy_coords = coords[heavy_rows]

    angles = np.linspace(0.0, 2.0 * np.pi, steps, endpoint=False)

    for group_id in sorted(set(plan.group[plan.rotatable].tolist())):
        members = np.nonzero(plan.group == group_id)[0]
        parent_row = int(plan.parent[members[0]])
        parent = coords[parent_row]

        neighbours = topology.atoms.neighbors(parent_row).cpu().numpy()
        heavy_neighbours = neighbours[~is_h[neighbours]]
        if len(heavy_neighbours) != 1:
            continue
        axis = parent - coords[int(heavy_neighbours[0])]
        norm = np.linalg.norm(axis)
        if norm < 1e-8:
            continue
        axis = axis / norm

        # Heavy atoms that could clash, excluding the parent and its own neighbour.
        near = np.nonzero(
            (np.linalg.norm(heavy_coords - parent, axis=1) < CLASH_CUTOFF)
        )[0]
        exclude = {parent_row, int(heavy_neighbours[0])}
        near_coords = np.array(
            [heavy_coords[i] for i in near if heavy_rows[i] not in exclude]
        )

        offsets = plan.position[members] - parent
        best_angle, best_score = 0.0, np.inf
        for angle in angles:
            rotated = _rotate_about(offsets, axis, angle)
            if len(near_coords) == 0:
                best_angle = 0.0
                break
            trial = parent + rotated
            separation = np.linalg.norm(
                trial[:, None, :] - near_coords[None, :, :], axis=-1
            )
            score = float((1.0 / np.maximum(separation, 0.5) ** 2).sum())
            if score < best_score:
                best_score, best_angle = score, float(angle)

        if best_angle != 0.0:
            plan.position[members] = parent + _rotate_about(offsets, axis, best_angle)

    return plan


def _rotate_about(vectors: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotate ``vectors`` about a unit ``axis`` through the origin, Rodrigues form."""
    cos, sin = np.cos(angle), np.sin(angle)
    return (
        vectors * cos
        + np.cross(axis, vectors) * sin
        + np.outer(vectors @ axis, axis) * (1.0 - cos)
    )


__all__ = [
    "HydrogenPlan",
    "plan_hydrogens",
    "optimise_free_torsions",
    "augment_atom_table",
    "STANDARD_VALENCE",
    "MAX_PLACEMENT_DISTANCE",
    "TORSION_SCAN_STEPS",
]


def augment_atom_table(pdb, plan: HydrogenPlan, topology):
    """Insert a plan's hydrogens into an atom table.

    Each hydrogen is inserted immediately after the residue it belongs to, not appended
    at the end: the residue partition is built from contiguous runs of
    ``(chain, resseq, icode)``, so appending would split every hydrogenated residue into
    two nodes.

    Rows are copied from the parent, then the hydrogen's own name, element and position
    are written over them. Everything else -- chain, residue, altloc, occupancy,
    B-factor, record type -- is inherited, so a hydrogen starts from its parent's
    displacement parameter and refines from there.

    Parameters
    ----------
    pdb : pandas.DataFrame
        Atom table to extend.
    plan : HydrogenPlan
    topology : Topology
        Supplies the residue partition the insertion points come from.

    Returns
    -------
    pandas.DataFrame
        A new table with ``serial`` and ``index`` renumbered.
    """
    import pandas as pd

    if plan.n_hydrogens == 0:
        return pdb.copy()

    by_residue: Dict[int, List[int]] = {}
    for i, residue in enumerate(plan.residue.tolist()):
        by_residue.setdefault(residue, []).append(i)

    pieces = []
    for residue in range(topology.n_residues):
        start = int(topology.residues.atom_start[residue])
        end = int(topology.residues.atom_end[residue])
        pieces.append(pdb.iloc[start:end])

        members = by_residue.get(residue)
        if not members:
            continue
        rows = pdb.loc[pdb.index[plan.parent[members]]].copy()
        rows["name"] = plan.name[members]
        rows["element"] = plan.element[members]
        rows["altloc"] = plan.altloc[members]
        rows[["x", "y", "z"]] = plan.position[members]
        if "anisou_flag" in rows.columns:
            rows["anisou_flag"] = False
        for column in ("u11", "u22", "u33", "u12", "u13", "u23"):
            if column in rows.columns:
                rows[column] = float("nan")
        pieces.append(rows)

    augmented = pd.concat(pieces, ignore_index=True)
    augmented["index"] = augmented.index.to_numpy(dtype=int)
    if "serial" in augmented.columns:
        augmented["serial"] = augmented.index.to_numpy(dtype=int) + 1
    augmented.attrs = dict(pdb.attrs)
    return augmented
