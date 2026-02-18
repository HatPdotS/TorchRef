"""
Backbone identification and junction placement utilities for chain closure.

Provides functions to identify backbone atoms, compute backbone torsion angles,
estimate secondary structure, and plan junction placement between segments.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# Standard amino acid residue names
AA_NAMES = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
    "TYR", "VAL", "MSE", "SEC",
})

# Backbone atom names
BACKBONE_ATOMS = ("N", "CA", "C")


def identify_backbone_atoms(
    pdb: pd.DataFrame,
) -> Dict[Tuple[str, int], Dict[str, int]]:
    """
    Map (chainid, resseq) to backbone atom indices {N: idx, CA: idx, C: idx}.

    Parameters
    ----------
    pdb : pd.DataFrame
        PDB DataFrame with columns 'chainid', 'resseq', 'name', 'index', 'resname'.

    Returns
    -------
    dict
        Mapping from (chainid, resseq) to dict of atom name -> atom index
        for backbone atoms N, CA, C. Only residues with all three atoms present
        are included.
    """
    # Only look at protein residues
    is_protein = pdb["resname"].str.upper().isin(AA_NAMES)
    protein_pdb = pdb[is_protein].copy()
    protein_pdb["name_stripped"] = protein_pdb["name"].str.strip()

    backbone_mask = protein_pdb["name_stripped"].isin(BACKBONE_ATOMS)
    backbone_df = protein_pdb[backbone_mask]

    # Group by residue and build mapping
    backbone_map = {}
    for (chainid, resseq), group in backbone_df.groupby(["chainid", "resseq"]):
        name_to_idx = dict(zip(group["name_stripped"], group["index"]))
        if all(atom in name_to_idx for atom in BACKBONE_ATOMS):
            backbone_map[(chainid, resseq)] = name_to_idx

    return backbone_map


def get_chain_residues(
    pdb: pd.DataFrame,
) -> Dict[str, List[Tuple[str, int]]]:
    """
    Get ordered list of protein residue keys per chain.

    Parameters
    ----------
    pdb : pd.DataFrame
        PDB DataFrame.

    Returns
    -------
    dict
        Mapping from chainid to sorted list of (chainid, resseq) tuples.
    """
    is_protein = pdb["resname"].str.upper().isin(AA_NAMES)
    protein_pdb = pdb[is_protein]

    chain_residues = {}
    for chainid in protein_pdb["chainid"].unique():
        chain_pdb = protein_pdb[protein_pdb["chainid"] == chainid]
        resseqs = sorted(chain_pdb["resseq"].unique())
        chain_residues[chainid] = [(chainid, rs) for rs in resseqs]

    return chain_residues


def compute_backbone_torsions(
    xyz: torch.Tensor,
    backbone_map: Dict[Tuple[str, int], Dict[str, int]],
    chain_residues: Dict[str, List[Tuple[str, int]]],
) -> Dict[Tuple[str, int], Dict[str, float]]:
    """
    Compute phi, psi, omega torsion angles for each residue.

    Parameters
    ----------
    xyz : torch.Tensor
        Atomic coordinates of shape (N, 3).
    backbone_map : dict
        From identify_backbone_atoms().
    chain_residues : dict
        From get_chain_residues().

    Returns
    -------
    dict
        Mapping from (chainid, resseq) to {'phi': float, 'psi': float, 'omega': float}.
        Values are in radians. Missing angles are set to NaN.
    """
    torsions = {}

    for chainid, residues in chain_residues.items():
        for i, res_key in enumerate(residues):
            if res_key not in backbone_map:
                continue

            result = {"phi": float("nan"), "psi": float("nan"), "omega": float("nan")}
            atoms = backbone_map[res_key]

            # phi: C(i-1) - N(i) - CA(i) - C(i)
            if i > 0 and residues[i - 1] in backbone_map:
                prev_atoms = backbone_map[residues[i - 1]]
                result["phi"] = _torsion_angle(
                    xyz[prev_atoms["C"]],
                    xyz[atoms["N"]],
                    xyz[atoms["CA"]],
                    xyz[atoms["C"]],
                ).item()

            # psi: N(i) - CA(i) - C(i) - N(i+1)
            if i < len(residues) - 1 and residues[i + 1] in backbone_map:
                next_atoms = backbone_map[residues[i + 1]]
                result["psi"] = _torsion_angle(
                    xyz[atoms["N"]],
                    xyz[atoms["CA"]],
                    xyz[atoms["C"]],
                    xyz[next_atoms["N"]],
                ).item()

            # omega: CA(i-1) - C(i-1) - N(i) - CA(i)
            if i > 0 and residues[i - 1] in backbone_map:
                prev_atoms = backbone_map[residues[i - 1]]
                result["omega"] = _torsion_angle(
                    xyz[prev_atoms["CA"]],
                    xyz[prev_atoms["C"]],
                    xyz[atoms["N"]],
                    xyz[atoms["CA"]],
                ).item()

            torsions[res_key] = result

    return torsions


def _torsion_angle(
    p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, p4: torch.Tensor,
) -> torch.Tensor:
    """Compute dihedral angle between four points."""
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    n1 = torch.linalg.cross(b1, b2)
    n2 = torch.linalg.cross(b2, b3)
    n1 = n1 / torch.linalg.norm(n1).clamp(min=1e-10)
    n2 = n2 / torch.linalg.norm(n2).clamp(min=1e-10)
    b2_norm = b2 / torch.linalg.norm(b2).clamp(min=1e-10)
    m1 = torch.linalg.cross(n1, b2_norm)
    x = torch.dot(n1, n2)
    y = torch.dot(m1, n2)
    return torch.atan2(y, x)


def estimate_secondary_structure(
    torsions: Dict[Tuple[str, int], Dict[str, float]],
) -> Dict[Tuple[str, int], str]:
    """
    Simple Ramachandran region classification: H (helix), E (sheet), L (loop).

    Parameters
    ----------
    torsions : dict
        From compute_backbone_torsions().

    Returns
    -------
    dict
        Mapping from (chainid, resseq) to 'H', 'E', or 'L'.
    """
    ss = {}
    for res_key, angles in torsions.items():
        phi = angles["phi"]
        psi = angles["psi"]

        if np.isnan(phi) or np.isnan(psi):
            ss[res_key] = "L"
            continue

        # Convert to degrees for readability
        phi_deg = np.degrees(phi)
        psi_deg = np.degrees(psi)

        # Alpha helix: phi ~ -60, psi ~ -47
        if -120 < phi_deg < -20 and -80 < psi_deg < -10:
            ss[res_key] = "H"
        # Beta sheet: phi ~ -120, psi ~ 120
        elif -180 < phi_deg < -60 and 60 < psi_deg < 180:
            ss[res_key] = "E"
        else:
            ss[res_key] = "L"

    return ss


def plan_junction_placement(
    chain_residues: Dict[str, List[Tuple[str, int]]],
    backbone_map: Dict[Tuple[str, int], Dict[str, int]],
    n_aa_per_segment: int = 18,
    junction_size: int = 3,
    ss: Optional[Dict[Tuple[str, int], str]] = None,
    prefer_loops: bool = True,
) -> Tuple[List[List[Tuple[str, int]]], List[List[Tuple[str, int]]]]:
    """
    Plan segment and junction placement along protein chains.

    Divides each chain into segments of ~n_aa_per_segment residues with
    junction_size-residue junctions between them. Optionally slides junctions
    to prefer loop regions.

    The algorithm:
    1. Determine nominal junction positions at every n_aa_per_segment residues.
    2. Optionally slide each junction within +-slide_range to prefer loops.
    3. Build segments from the non-junction gaps between junctions.

    Parameters
    ----------
    chain_residues : dict
        From get_chain_residues().
    backbone_map : dict
        From identify_backbone_atoms().
    n_aa_per_segment : int
        Target number of residues per free-DOF segment.
    junction_size : int
        Number of residues per junction (slave DOFs).
    ss : dict, optional
        Secondary structure assignments from estimate_secondary_structure().
    prefer_loops : bool
        If True and ss is provided, slide junctions to prefer loop regions.

    Returns
    -------
    segments : list of list
        Each inner list contains (chainid, resseq) keys for one segment.
    junctions : list of list
        Each inner list contains (chainid, resseq) keys for one junction.
        Junction i connects segment i to segment i+1.
    """
    all_segments = []
    all_junctions = []

    for chainid, residues in chain_residues.items():
        # Filter to residues that have backbone atoms
        valid_residues = [r for r in residues if r in backbone_map]

        if not valid_residues:
            continue

        n_res = len(valid_residues)
        min_chain_length = n_aa_per_segment + junction_size + 1
        if n_res < min_chain_length:
            # Short chain: single segment, no junctions
            all_segments.append(valid_residues)
            continue

        # Step 1: Determine nominal junction start indices
        nominal_starts = []
        pos = n_aa_per_segment
        while pos + junction_size <= n_res:
            nominal_starts.append(pos)
            pos += n_aa_per_segment + junction_size

        if not nominal_starts:
            all_segments.append(valid_residues)
            continue

        # Step 2: Slide junctions to prefer loops (constrained to not overlap)
        junction_starts = []
        for nom_start in nominal_starts:
            if prefer_loops and ss is not None:
                best_start = _find_best_junction_start(
                    valid_residues, nom_start, junction_size, ss,
                    slide_range=3, existing_junctions=junction_starts,
                )
            else:
                best_start = nom_start
            junction_starts.append(best_start)

        # Step 3: Build junction lists
        junctions_chain = []
        for start in junction_starts:
            junctions_chain.append(valid_residues[start:start + junction_size])

        # Step 4: Build segments from non-junction residues
        junction_index_set = set()
        for start in junction_starts:
            for i in range(start, start + junction_size):
                junction_index_set.add(i)

        segments_chain = []
        current_seg = []
        for i, res in enumerate(valid_residues):
            if i in junction_index_set:
                if current_seg:
                    segments_chain.append(current_seg)
                    current_seg = []
            else:
                current_seg.append(res)
        if current_seg:
            segments_chain.append(current_seg)

        all_segments.extend(segments_chain)
        all_junctions.extend(junctions_chain)

    return all_segments, all_junctions


def _find_best_junction_start(
    residues: List[Tuple[str, int]],
    nominal_start: int,
    junction_size: int,
    ss: Dict[Tuple[str, int], str],
    slide_range: int = 3,
    existing_junctions: Optional[List[int]] = None,
) -> int:
    """
    Find the best junction start position within slide_range of nominal_start.

    Maximizes loop content while ensuring no overlap with existing junctions
    and staying within bounds.
    """
    best_start = nominal_start
    best_loop_count = -1

    # Compute occupied indices from existing junctions
    occupied = set()
    if existing_junctions:
        for prev_start in existing_junctions:
            for i in range(prev_start, prev_start + junction_size):
                occupied.add(i)

    for offset in range(-slide_range, slide_range + 1):
        start = nominal_start + offset
        end = start + junction_size
        if start < 0 or end > len(residues):
            continue

        # Must not overlap with existing junctions
        junc_range = set(range(start, end))
        if junc_range & occupied:
            continue

        # Must leave at least 1 residue for segment before this junction
        # (unless this is the very first junction)
        if existing_junctions:
            prev_end = max(s + junction_size for s in existing_junctions)
            if start <= prev_end:
                continue
        else:
            if start < 1:
                continue

        # Count loop residues
        loop_count = sum(
            1 for r in residues[start:end]
            if ss.get(r, "L") == "L"
        )

        if loop_count > best_loop_count:
            best_loop_count = loop_count
            best_start = start

    return best_start


def get_junction_backbone_indices(
    junction_residues: List[Tuple[str, int]],
    backbone_map: Dict[Tuple[str, int], Dict[str, int]],
) -> List[Dict[str, int]]:
    """
    Get ordered backbone atom indices for junction residues.

    Parameters
    ----------
    junction_residues : list
        List of (chainid, resseq) tuples for the junction.
    backbone_map : dict
        From identify_backbone_atoms().

    Returns
    -------
    list
        List of dicts with 'N', 'CA', 'C' atom indices, one per residue.

    Raises
    ------
    ValueError
        If any junction residue lacks backbone atoms.
    """
    result = []
    for res_key in junction_residues:
        if res_key not in backbone_map:
            raise ValueError(
                f"Junction residue {res_key} lacks backbone atoms (N, CA, C)"
            )
        result.append(backbone_map[res_key])
    return result
