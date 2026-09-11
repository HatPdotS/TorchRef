"""Numba-accelerated CIF-to-restraint matchers, free of Pandas in the hot loop.

Every ``match_*_numba`` shares one calling convention: the caller pre-allocates
the ``out_*`` arrays, the function fills entries ``[0:count]`` in place and
returns ``count``. Anything past ``count`` is stale.

Numba is optional -- without it ``njit`` degrades to a no-op decorator and
``prange`` to ``range``, so the same code runs orders of magnitude slower rather
than failing.
"""

from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

try:
    import numba
    from numba import njit, prange

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    # Fallback decorator that does nothing
    def njit(*args, **kwargs):
        def decorator(func):
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

    prange = range


# =============================================================================
# Numba-accelerated helper functions
# =============================================================================


@njit(cache=True)
def match_bonds_numba(
    residue_atom_names: np.ndarray,  # atom names for this residue
    residue_atom_indices: np.ndarray,  # global atom indices
    bond_atom1: np.ndarray,  # CIF bond atom1 names
    bond_atom2: np.ndarray,  # CIF bond atom2 names
    bond_values: np.ndarray,  # CIF bond reference values
    bond_sigmas: np.ndarray,  # CIF bond sigmas
    out_idx1: np.ndarray,  # output arrays (pre-allocated)
    out_idx2: np.ndarray,
    out_refs: np.ndarray,
    out_sigmas: np.ndarray,
) -> int:
    """
    Match bond restraints for a single residue.

    A CIF bond is emitted only if both named atoms appear in
    ``residue_atom_names``; the emitted indices come from
    ``residue_atom_indices`` (global), and the reference/sigma from
    ``bond_values``/``bond_sigmas``.

    Returns
    -------
    int
        Number of matched bonds. The ``out_*`` arrays are filled in place for
        entries ``[0:count]`` only -- the tail is left untouched.
    """
    count = 0
    n_bonds = len(bond_atom1)
    n_atoms = len(residue_atom_names)

    for i in range(n_bonds):
        # Find atom1
        idx1 = -1
        for j in range(n_atoms):
            if residue_atom_names[j] == bond_atom1[i]:
                idx1 = j
                break
        if idx1 < 0:
            continue

        # Find atom2
        idx2 = -1
        for j in range(n_atoms):
            if residue_atom_names[j] == bond_atom2[i]:
                idx2 = j
                break
        if idx2 < 0:
            continue

        # Both atoms found - add restraint
        out_idx1[count] = residue_atom_indices[idx1]
        out_idx2[count] = residue_atom_indices[idx2]
        out_refs[count] = bond_values[i]
        out_sigmas[count] = bond_sigmas[i]
        count += 1

    return count


@njit(cache=True)
def match_angles_numba(
    residue_atom_names: np.ndarray,
    residue_atom_indices: np.ndarray,
    angle_atom1: np.ndarray,
    angle_atom2: np.ndarray,
    angle_atom3: np.ndarray,
    angle_values: np.ndarray,
    angle_sigmas: np.ndarray,
    out_idx1: np.ndarray,
    out_idx2: np.ndarray,
    out_idx3: np.ndarray,
    out_refs: np.ndarray,
    out_sigmas: np.ndarray,
) -> int:
    """
    Match angle restraints for a single residue.

    A CIF angle is emitted only if all three named atoms appear in
    ``residue_atom_names`` (``angle_atom2`` is the vertex); the emitted indices
    come from ``residue_atom_indices`` (global), and the reference/sigma from
    ``angle_values``/``angle_sigmas``, both in degrees.

    Returns
    -------
    int
        Number of matched angles. The ``out_*`` arrays are filled in place for
        entries ``[0:count]`` only -- the tail is left untouched.
    """
    count = 0
    n_angles = len(angle_atom1)
    n_atoms = len(residue_atom_names)

    for i in range(n_angles):
        # Find all three atoms
        idx1, idx2, idx3 = -1, -1, -1
        for j in range(n_atoms):
            name = residue_atom_names[j]
            if name == angle_atom1[i]:
                idx1 = j
            elif name == angle_atom2[i]:
                idx2 = j
            elif name == angle_atom3[i]:
                idx3 = j

        if idx1 < 0 or idx2 < 0 or idx3 < 0:
            continue

        out_idx1[count] = residue_atom_indices[idx1]
        out_idx2[count] = residue_atom_indices[idx2]
        out_idx3[count] = residue_atom_indices[idx3]
        out_refs[count] = angle_values[i]
        out_sigmas[count] = angle_sigmas[i]
        count += 1

    return count


@njit(cache=True)
def match_torsions_numba(
    residue_atom_names: np.ndarray,
    residue_atom_indices: np.ndarray,
    torsion_atom1: np.ndarray,
    torsion_atom2: np.ndarray,
    torsion_atom3: np.ndarray,
    torsion_atom4: np.ndarray,
    torsion_values: np.ndarray,
    torsion_sigmas: np.ndarray,
    torsion_periods: np.ndarray,
    out_idx1: np.ndarray,
    out_idx2: np.ndarray,
    out_idx3: np.ndarray,
    out_idx4: np.ndarray,
    out_refs: np.ndarray,
    out_sigmas: np.ndarray,
    out_periods: np.ndarray,
) -> int:
    """
    Match torsion restraints for a single residue.

    A CIF torsion is emitted only if all four named atoms appear in
    ``residue_atom_names`` **and** its sigma is non-zero -- zero-sigma torsions
    are dropped, not clamped. Indices come from ``residue_atom_indices``
    (global); references and sigmas are in degrees.

    Returns
    -------
    int
        Number of matched torsions. The ``out_*`` arrays are filled in place for
        entries ``[0:count]`` only -- the tail is left untouched.
    """
    count = 0
    n_torsions = len(torsion_atom1)
    n_atoms = len(residue_atom_names)

    for i in range(n_torsions):
        # Skip if sigma is zero
        if torsion_sigmas[i] == 0:
            continue

        # Find all four atoms
        idx1, idx2, idx3, idx4 = -1, -1, -1, -1
        for j in range(n_atoms):
            name = residue_atom_names[j]
            if name == torsion_atom1[i]:
                idx1 = j
            elif name == torsion_atom2[i]:
                idx2 = j
            elif name == torsion_atom3[i]:
                idx3 = j
            elif name == torsion_atom4[i]:
                idx4 = j

        if idx1 < 0 or idx2 < 0 or idx3 < 0 or idx4 < 0:
            continue

        out_idx1[count] = residue_atom_indices[idx1]
        out_idx2[count] = residue_atom_indices[idx2]
        out_idx3[count] = residue_atom_indices[idx3]
        out_idx4[count] = residue_atom_indices[idx4]
        out_refs[count] = torsion_values[i]
        out_sigmas[count] = torsion_sigmas[i]
        out_periods[count] = torsion_periods[i]
        count += 1

    return count


@njit(cache=True)
def match_chirals_numba(
    residue_atom_names: np.ndarray,
    residue_atom_indices: np.ndarray,
    chiral_center: np.ndarray,
    chiral_atom1: np.ndarray,
    chiral_atom2: np.ndarray,
    chiral_atom3: np.ndarray,
    chiral_volume_signs: np.ndarray,  # +1, -1, 0, or NaN
    chiral_sigmas: np.ndarray,
    out_center: np.ndarray,
    out_idx1: np.ndarray,
    out_idx2: np.ndarray,
    out_idx3: np.ndarray,
    out_signs: np.ndarray,
    out_sigmas: np.ndarray,
) -> int:
    """
    Match chiral restraints for a single residue.

    A CIF chiral is emitted only if its centre and all three substituents appear
    in ``residue_atom_names`` **and** its volume sign is not ``NaN`` -- unknown
    signs are dropped. ``+1``/``-1``/``0`` are all kept. Indices come from
    ``residue_atom_indices`` (global).

    Returns
    -------
    int
        Number of matched chirals. The ``out_*`` arrays are filled in place for
        entries ``[0:count]`` only -- the tail is left untouched.
    """
    count = 0
    n_chirals = len(chiral_center)
    n_atoms = len(residue_atom_names)

    for i in range(n_chirals):
        # Skip if unknown chiral sign (NaN)
        if np.isnan(chiral_volume_signs[i]):
            continue

        # Find all four atoms (center + 3 neighbors)
        idx_c, idx1, idx2, idx3 = -1, -1, -1, -1
        for j in range(n_atoms):
            name = residue_atom_names[j]
            if name == chiral_center[i]:
                idx_c = j
            elif name == chiral_atom1[i]:
                idx1 = j
            elif name == chiral_atom2[i]:
                idx2 = j
            elif name == chiral_atom3[i]:
                idx3 = j

        if idx_c < 0 or idx1 < 0 or idx2 < 0 or idx3 < 0:
            continue

        out_center[count] = residue_atom_indices[idx_c]
        out_idx1[count] = residue_atom_indices[idx1]
        out_idx2[count] = residue_atom_indices[idx2]
        out_idx3[count] = residue_atom_indices[idx3]
        out_signs[count] = chiral_volume_signs[i]
        out_sigmas[count] = chiral_sigmas[i]
        count += 1

    return count


