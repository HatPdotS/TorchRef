"""Inter-residue restraint builders, and the dictionary preprocessing they share.

The ``InterResidue*Builder`` classes turn a link definition (``TRANS``, ``PTRANS``,
``disulf``) into edges over a topology. Their ``build()`` reads a
:class:`PeptideResidues` -- the peptide-linked residue pairs and each residue's
conformer maps, prepared once from the topology -- and returns directly; only the
disulfide path is stateful, accumulating over ``process_disulfide_*`` calls until
``finalize()`` (``finalize_disulfide()`` on the torsion builder). Edge indices are atom
rows of the topology.

Nothing here is re-exported at the package level; import from
``torchref.topology.builders``.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from torchref.base.targets._common import torsions_from_xyz
from torchref.config import get_float_dtype, get_int_dtype


# =============================================================================
# Pre-processing utilities
# =============================================================================


def _conformer_maps(
    topology, residue: int
) -> List[Tuple[Optional[str], Dict[str, int]]]:
    """Atom-name-to-row maps for one residue, one per alternative conformation.

    A residue without altlocs gives one map, labelled None. One with altlocs gives one
    per altloc, labelled with it, each holding the residue's blank-altloc atoms plus
    that altloc's own; one with no blank atoms gives one per altloc on its own.

    Returns
    -------
    list of tuple
        ``(altloc label or None, {atom name: row})`` per conformation.
    """
    start = int(topology.residues.atom_start[residue])
    end = int(topology.residues.atom_end[residue])
    names = topology.atoms.name[start:end]
    rows = np.arange(start, end, dtype=np.int64)
    altlocs = topology.atoms.altloc[start:end]
    unique = np.unique(altlocs)
    if len(unique) == 1 and unique[0] == " ":
        return [(None, dict(zip(names, rows)))]
    common = altlocs == " "
    maps = []
    for alt in unique:
        if alt == " ":
            continue
        chosen = common | (altlocs == alt)
        maps.append((str(alt), dict(zip(names[chosen], rows[chosen]))))
    return maps


def _matching_conformers(
    conformers_a: List[Tuple[Optional[str], Dict[str, int]]],
    conformers_b: List[Tuple[Optional[str], Dict[str, int]]],
) -> List[Tuple[Dict[str, int], Dict[str, int]]]:
    """Conformer maps of two linked residues that belong to one state of the model.

    Two maps match when they carry the same altloc label, or when either label is
    absent from the other residue. An unlabelled map therefore matches every map, and
    conformer ``A`` of one residue never meets conformer ``B`` of the other while both
    residues carry both.

    Parameters
    ----------
    conformers_a, conformers_b : list of tuple
        :func:`_conformer_maps` of the two residues.

    Returns
    -------
    list of tuple of dict
        ``(map_a, map_b)`` pairs, in conformer order.
    """
    labels_a = {label for label, _ in conformers_a}
    labels_b = {label for label, _ in conformers_b}
    return [
        (map_a, map_b)
        for label_a, map_a in conformers_a
        for label_b, map_b in conformers_b
        if label_a == label_b or label_a not in labels_b or label_b not in labels_a
    ]


class PeptideResidues:
    """What the peptide-link builders read, prepared once from a topology.

    Parameters
    ----------
    topology : Topology
        Supplies atom names, altlocs and residue ranges; edges are not needed.
    pairs : sequence of tuple of int
        ``(residue donating C, residue donating N)`` pairs, from
        :func:`~torchref.topology.residue_graph.find_peptide_links`.
    xyz : numpy.ndarray
        Cartesian coordinates in Å, shape ``(N, 3)``; the torsion builder classifies
        each proline's omega as cis or trans from them.

    Attributes
    ----------
    conformer_maps : dict
        ``{residue: [(altloc label or None, {atom name: row}), ...]}`` for every
        residue in a pair, as :func:`_conformer_maps` gives them.
    resnames : numpy.ndarray
        Residue name per residue, shape ``(R,)``.
    """

    def __init__(self, topology, pairs, xyz):
        self.pairs = [(int(a), int(b)) for a, b in pairs]
        self.resnames = np.char.strip(np.asarray(topology.residues.resname).astype(str))
        self.atom_altlocs = topology.atoms.altloc
        self.atom_resnames = topology.columns()["resname"]
        self.xyz = np.asarray(xyz, dtype=np.float64)
        involved = sorted({r for pair in self.pairs for r in pair})
        self.conformer_maps = {r: _conformer_maps(topology, r) for r in involved}

    def conformer_resname(self, mapping: Dict[str, int]) -> str:
        """Return the chemical identity of a conformer atom-name map.

        Parameters
        ----------
        mapping : dict
            Atom names mapped to topology rows for one conformer.

        Returns
        -------
        str
            Residue name of the labelled atoms, or the shared atoms if unlabelled.
        """
        rows = list(mapping.values())
        names = self.atom_resnames[rows]
        # Shared atoms can retain the first conformer's name. A conformer's
        # distinct chemical identity belongs to its labelled atoms.
        for row in rows:
            if self.atom_altlocs[row] != " ":
                return str(self.atom_resnames[row])
        return str(names[0])

    def conformer_pairs(
        self,
        next_resname_filter: Optional[str] = None,
        exclude_next_resname: Optional[str] = None,
    ):
        """Yield the linked conformers of every pair, as the builders iterate them.

        Conformers are paired by :func:`_matching_conformers`, so no restraint joins
        two different conformations of the model.

        Parameters
        ----------
        next_resname_filter : str, optional
            Keep only pairs whose second (N-donating) conformer has this residue name,
            e.g. ``'PRO'`` for the proline links.
        exclude_next_resname : str, optional
            Skip pairs whose second conformer has this residue name.

        Yields
        ------
        tuple
            ``(residue donating C, residue donating N, map_i, map_next)``, the last
            two ``{atom name: row}`` for the paired conformers.
        """
        for res_i, res_next in self.pairs:
            for map_i, map_next in _matching_conformers(
                self.conformer_maps[res_i], self.conformer_maps[res_next]
            ):
                next_name = self.conformer_resname(map_next)
                if next_resname_filter is not None and next_name != next_resname_filter:
                    continue
                if (
                    exclude_next_resname is not None
                    and next_name == exclude_next_resname
                ):
                    continue
                yield res_i, res_next, map_i, map_next


class PreprocessedCIF:
    """
    Pre-processed CIF restraints as NumPy arrays per residue type.

    ``torsions`` holds every torsion except a template's alternative sugar-pucker
    sets, which ``puckers`` keeps as ``{residue type: {id prefix: arrays}}`` so that a
    residue can be matched against one of them.
    """

    def __init__(self, cif_dict: Dict):
        """
        Initialize from CIF dictionary.

        Parameters
        ----------
        cif_dict : dict
            CIF dictionary with restraints per residue type.
        """
        self.residue_types = list(cif_dict.keys())

        # Pre-process each restraint type
        self.bonds = {}
        self.angles = {}
        self.torsions = {}
        self.puckers = {}
        self.planes = {}
        self.chirals = {}

        for restype, data in cif_dict.items():
            if "bonds" in data and len(data["bonds"]) > 0:
                self.bonds[restype] = self._preprocess_bonds(data["bonds"])
            if "angles" in data and len(data["angles"]) > 0:
                self.angles[restype] = self._preprocess_angles(data["angles"])
            if "torsions" in data and len(data["torsions"]) > 0:
                result = self._preprocess_torsions(data["torsions"])
                if result is not None:
                    common, puckers = self._split_puckers(result)
                    if len(common["atom1"]):
                        self.torsions[restype] = common
                    if puckers:
                        self.puckers[restype] = puckers
            if "planes" in data and len(data["planes"]) > 0:
                self.planes[restype] = self._preprocess_planes(data["planes"])
            if "chirals" in data and len(data["chirals"]) > 0:
                self.chirals[restype] = self._preprocess_chirals(data["chirals"])

    def _preprocess_bonds(self, bonds_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Convert bonds DataFrame to NumPy arrays."""
        return {
            "atom1": bonds_df["atom1"].values.astype(str),
            "atom2": bonds_df["atom2"].values.astype(str),
            "value": bonds_df["value"].values.astype(np.float64),
            "sigma": bonds_df["sigma"].values.astype(np.float64),
        }

    def _preprocess_angles(self, angles_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Convert angles DataFrame to NumPy arrays."""
        return {
            "atom1": angles_df["atom1"].values.astype(str),
            "atom2": angles_df["atom2"].values.astype(str),
            "atom3": angles_df["atom3"].values.astype(str),
            "value": angles_df["value"].values.astype(np.float64),
            "sigma": angles_df["sigma"].values.astype(np.float64),
        }

    # Backbone heavy atoms — torsions where ALL four atoms fall in this set
    # are phi/psi-equivalent and must NOT be restrained as intra-residue
    # torsions (they conflict with Ramachandran-favored angles).
    # Example: CIF "sp2_sp3_1  O C CA N  0.0 10.0 6" directly restrains psi.
    _BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O", "OXT"})

    def _preprocess_torsions(self, torsions_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Convert torsions to NumPy arrays, dropping backbone-only torsions.

        A torsion over four backbone heavy atoms (N, CA, C, O, OXT) is
        phi/psi-equivalent and would fight the Ramachandran term, so it is removed
        here rather than downweighted.
        """
        bb = self._BACKBONE_ATOMS
        keep = np.array([
            not ({a1, a2, a3, a4} <= bb)
            for a1, a2, a3, a4 in zip(
                torsions_df["atom1"].values,
                torsions_df["atom2"].values,
                torsions_df["atom3"].values,
                torsions_df["atom4"].values,
            )
        ], dtype=bool)
        torsions_df = torsions_df[keep].reset_index(drop=True)

        if len(torsions_df) == 0:
            return None

        # Handle both 'period' and 'periodicity' column names
        if "periodicity" in torsions_df.columns:
            periods = torsions_df["periodicity"].values.astype(np.int64)
        elif "period" in torsions_df.columns:
            periods = torsions_df["period"].values.astype(np.int64)
        else:
            periods = np.ones(len(torsions_df), dtype=np.int64)

        return {
            "atom1": torsions_df["atom1"].values.astype(str),
            "atom2": torsions_df["atom2"].values.astype(str),
            "atom3": torsions_df["atom3"].values.astype(str),
            "atom4": torsions_df["atom4"].values.astype(str),
            "id": (
                torsions_df["id"].values.astype(str)
                if "id" in torsions_df.columns
                else np.full(len(torsions_df), "")
            ),
            "value": torsions_df["value"].values.astype(np.float64),
            "sigma": torsions_df["sigma"].values.astype(np.float64),
            "period": periods,
        }

    #: ``_chem_comp_tor.id`` prefixes of the C2'-endo and C3'-endo sugar torsion sets
    #: the monomer library gives every nucleotide. They restrain the same ring torsions
    #: to incompatible values, so a residue is matched against one set only.
    SUGAR_PUCKERS = ("C2e", "C3e")

    @classmethod
    def _split_puckers(
        cls, torsions: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
        """Split a template's alternative sugar-pucker sets off its torsions.

        Returns
        -------
        common : dict
            The torsion arrays without the pucker sets.
        puckers : dict
            ``{id prefix: torsion arrays}`` per :data:`SUGAR_PUCKERS` set; empty, and
            ``common`` unchanged, unless the template carries more than one set.
        """
        prefix = np.array([tid.split("-", 1)[0] for tid in torsions["id"].tolist()])
        present = [p for p in cls.SUGAR_PUCKERS if (prefix == p).any()]
        if len(present) < 2:
            return torsions, {}
        common = ~np.isin(prefix, present)
        return {k: v[common] for k, v in torsions.items()}, {
            p: {k: v[prefix == p] for k, v in torsions.items()} for p in present
        }

    def _preprocess_planes(self, planes_df: pd.DataFrame) -> List[Dict]:
        """Convert planes DataFrame to list of plane data."""
        plane_ids = planes_df["plane_id"].unique()
        planes_data = []

        for plane_id in plane_ids:
            plane_atoms = planes_df[planes_df["plane_id"] == plane_id]
            sigma_col = "sigma" if "sigma" in plane_atoms.columns else None
            planes_data.append(
                {
                    "atoms": plane_atoms["atom"].values.astype(str),
                    "sigmas": (
                        plane_atoms[sigma_col].values.astype(np.float64)
                        if sigma_col
                        else np.full(len(plane_atoms), 0.02)
                    ),
                }
            )

        return planes_data

    def _preprocess_chirals(self, chirals_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Convert chirals DataFrame to NumPy arrays."""
        # Convert volume_sign strings to floats. The CCP4 library writes both the
        # full and the truncated spelling ("positiv", "negativ"); an unrecognised
        # sign becomes NaN and the restraint is then dropped by match_chirals,
        # so the short forms have to be matched here or those chirals vanish.
        volume_signs = []
        for sign in chirals_df["volume_sign"].values:
            sign = str(sign).strip().lower()
            if sign.startswith("positiv"):
                volume_signs.append(1.0)
            elif sign.startswith("negativ"):
                volume_signs.append(-1.0)
            elif sign in ["both", "either"]:
                volume_signs.append(0.0)
            else:
                volume_signs.append(np.nan)

        sigma_col = "sigma" if "sigma" in chirals_df.columns else None

        return {
            "center": chirals_df["atom_centre"].values.astype(str),
            "atom1": chirals_df["atom1"].values.astype(str),
            "atom2": chirals_df["atom2"].values.astype(str),
            "atom3": chirals_df["atom3"].values.astype(str),
            "volume_sign": np.array(volume_signs, dtype=np.float64),
            "sigma": (
                chirals_df[sigma_col].values.astype(np.float64)
                if sigma_col
                else np.full(len(chirals_df), 0.2)
            ),
        }


# =============================================================================
# Fast Inter-Residue Builders
# =============================================================================


class PreprocessedLinkData:
    """
    Pre-processed link restraint data for fast inter-residue matching.

    Converts link DataFrames to NumPy arrays for efficient access.
    """

    def __init__(self, link_dict: Dict):
        """
        Initialize from link dictionary.

        Parameters
        ----------
        link_dict : dict
            Dictionary with link restraints (e.g., TRANS, disulf).
        """
        self.bonds = None
        self.angles = None
        self.torsions = None
        self.planes = None

        if "bonds" in link_dict and link_dict["bonds"] is not None:
            bonds_df = link_dict["bonds"]
            self.bonds = {
                "comp1": bonds_df["atom_1_comp_id"].values.astype(str),
                "comp2": bonds_df["atom_2_comp_id"].values.astype(str),
                "atom1": bonds_df["atom1"].values.astype(str),
                "atom2": bonds_df["atom2"].values.astype(str),
                "value": bonds_df["value"].values.astype(np.float64),
                "sigma": bonds_df["sigma"].values.astype(np.float64),
            }

        if "angles" in link_dict and link_dict["angles"] is not None:
            angles_df = link_dict["angles"]
            self.angles = {
                "comp1": angles_df["atom_1_comp_id"].values.astype(str),
                "comp2": angles_df["atom_2_comp_id"].values.astype(str),
                "comp3": angles_df["atom_3_comp_id"].values.astype(str),
                "atom1": angles_df["atom1"].values.astype(str),
                "atom2": angles_df["atom2"].values.astype(str),
                "atom3": angles_df["atom3"].values.astype(str),
                "value": angles_df["value"].values.astype(np.float64),
                "sigma": angles_df["sigma"].values.astype(np.float64),
            }

        if "torsions" in link_dict and link_dict["torsions"] is not None:
            torsions_df = link_dict["torsions"]
            self.torsions = {
                "comp1": torsions_df["atom_1_comp_id"].values.astype(str),
                "comp2": torsions_df["atom_2_comp_id"].values.astype(str),
                "comp3": torsions_df["atom_3_comp_id"].values.astype(str),
                "comp4": torsions_df["atom_4_comp_id"].values.astype(str),
                "atom1": torsions_df["atom1"].values.astype(str),
                "atom2": torsions_df["atom2"].values.astype(str),
                "atom3": torsions_df["atom3"].values.astype(str),
                "atom4": torsions_df["atom4"].values.astype(str),
                "id": (
                    torsions_df["id"].values.astype(str)
                    if "id" in torsions_df.columns
                    else np.full(len(torsions_df), "", dtype="<U10")
                ),
                "value": torsions_df["value"].values.astype(np.float64),
                "sigma": torsions_df["sigma"].values.astype(np.float64),
                "period": (
                    torsions_df["period"].values.astype(np.int64)
                    if "period" in torsions_df.columns
                    else np.ones(len(torsions_df), dtype=np.int64)
                ),
            }

        if "planes" in link_dict and link_dict["planes"] is not None:
            planes_df = link_dict["planes"]
            self.planes = self._preprocess_planes(planes_df)

    def _preprocess_planes(self, planes_df: pd.DataFrame) -> List[Dict]:
        """Convert planes DataFrame to list of plane data."""
        plane_ids = planes_df["plane_id"].unique()
        planes_data = []

        for plane_id in plane_ids:
            plane_atoms = planes_df[planes_df["plane_id"] == plane_id]
            planes_data.append(
                {
                    "comp_ids": plane_atoms["atom_comp_id"].values.astype(str),
                    "atoms": plane_atoms["atom"].values.astype(str),
                    "sigmas": plane_atoms["sigma"].values.astype(np.float64),
                }
            )

        return planes_data


class InterResidueBondBuilder:
    """
    Fast builder for inter-residue bond restraints.

    Usage:
        builder = InterResidueBondBuilder()
        result = builder.build(residues, link_dict, device)

        # Or for disulfides (incremental):
        builder = InterResidueBondBuilder()
        for sg1_idx, sg2_idx, length, sigma in disulfide_pairs:
            builder.process_disulfide_bond(sg1_idx, sg2_idx, length, sigma)
        result = builder.finalize(device)
    """

    def __init__(self, verbose: int = 0):
        """Initialize builder with accumulators for disulfide bonds."""
        self.verbose = verbose
        # Accumulators for incremental disulfide building
        self._indices: List[np.ndarray] = []
        self._references: List[np.ndarray] = []
        self._sigmas: List[np.ndarray] = []
        self._count: int = 0

    def reset(self):
        """Clear all accumulated data."""
        self._indices.clear()
        self._references.clear()
        self._sigmas.clear()
        self._count = 0

    def process_disulfide_bond(
        self, sg1_idx: int, sg2_idx: int, bond_length: float, bond_sigma: float
    ) -> int:
        """
        Process a single disulfide bond restraint.

        Parameters
        ----------
        sg1_idx : int
            Index of first SG atom.
        sg2_idx : int
            Index of second SG atom.
        bond_length : float
            Target bond length in Å.
        bond_sigma : float
            Sigma for restraint in Å.

        Returns
        -------
        int
            Always returns 1.
        """
        self._indices.append(np.array([[sg1_idx, sg2_idx]], dtype=np.int64))
        self._references.append(np.array([bond_length], dtype=np.float64))
        self._sigmas.append(np.array([bond_sigma], dtype=np.float64))
        self._count += 1
        return 1

    def finalize(
        self, device: torch.device, sort_indices: bool = True, min_sigma: float = 1e-4
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Convert accumulated disulfide bond data to tensors.

        Parameters
        ----------
        device : torch.device
            Target device for tensors.
        sort_indices : bool, default True
            Whether to sort by first atom index.
        min_sigma : float, default 1e-4
            Minimum sigma value.

        Returns
        -------
        dict or None
            Dictionary with 'indices', 'references', 'sigmas' tensors.
        """
        if not self._indices:
            return None

        indices = np.concatenate(self._indices, axis=0)
        references = np.concatenate(self._references)
        sigmas = np.concatenate(self._sigmas)

        if sort_indices and len(indices) > 0:
            sort_order = np.argsort(indices[:, 0])
            indices = indices[sort_order]
            references = references[sort_order]
            sigmas = sigmas[sort_order]

        sigmas = np.where(sigmas == 0, min_sigma, sigmas)

        return {
            "indices": torch.tensor(indices, dtype=get_int_dtype(), device=device),
            "references": torch.tensor(
                references, dtype=get_float_dtype(), device=device
            ),
            "sigmas": torch.tensor(sigmas, dtype=get_float_dtype(), device=device),
        }

    @property
    def count(self) -> int:
        """Return total number of restraints accumulated."""
        return self._count

    def build(
        self,
        residues: "PeptideResidues",
        link_dict: Dict,
        device: torch.device,
        sort_indices: bool = True,
        next_resname_filter: Optional[str] = None,
        exclude_next_resname: Optional[str] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Build all inter-residue bond restraints.

        Parameters
        ----------
        residues : PeptideResidues
            The linked residue pairs and their atoms.
        link_dict : dict
            Link dictionary with 'bonds' DataFrame.
        device : torch.device
            Target device.
        sort_indices : bool
            Whether to sort output by first atom index.
        next_resname_filter, exclude_next_resname : str, optional
            Select pairs by the second residue's name, as
            :meth:`PeptideResidues.conformer_pairs` does.

        Returns
        -------
        dict or None
            Dictionary with restraint tensors.
        """
        if "bonds" not in link_dict or link_dict["bonds"] is None:
            return None

        # Pre-process link data
        link_data = PreprocessedLinkData(link_dict)
        if link_data.bonds is None:
            return None

        if not residues.pairs:
            return None

        # Accumulate restraints
        all_indices = []
        all_refs = []
        all_sigmas = []

        bonds = link_data.bonds
        n_bonds = len(bonds["atom1"])

        for _, _, map_i, map_next in residues.conformer_pairs(
            next_resname_filter=next_resname_filter,
            exclude_next_resname=exclude_next_resname,
        ):
            for b in range(n_bonds):
                comp1, comp2 = bonds["comp1"][b], bonds["comp2"][b]
                atom1_name, atom2_name = bonds["atom1"][b], bonds["atom2"][b]

                map1 = map_i if comp1 == "1" else map_next
                map2 = map_i if comp2 == "1" else map_next

                if atom1_name in map1 and atom2_name in map2:
                    idx1 = map1[atom1_name]
                    idx2 = map2[atom2_name]
                    all_indices.append([idx1, idx2])
                    all_refs.append(bonds["value"][b])
                    all_sigmas.append(bonds["sigma"][b])

        if not all_indices:
            return None

        indices = np.array(all_indices, dtype=np.int64)
        references = np.array(all_refs, dtype=np.float64)
        sigmas = np.array(all_sigmas, dtype=np.float64)

        if sort_indices and len(indices) > 0:
            order = np.argsort(indices[:, 0])
            indices = indices[order]
            references = references[order]
            sigmas = sigmas[order]

        sigmas = np.where(sigmas == 0, 1e-4, sigmas)

        return {
            "indices": torch.tensor(indices, dtype=get_int_dtype(), device=device),
            "references": torch.tensor(
                references, dtype=get_float_dtype(), device=device
            ),
            "sigmas": torch.tensor(sigmas, dtype=get_float_dtype(), device=device),
        }


class InterResidueAngleBuilder:
    """
    Fast builder for inter-residue angle restraints.

    Usage:
        builder = InterResidueAngleBuilder()
        result = builder.build(residues, link_dict, device)

        # Or for disulfides (incremental), per pair of cysteine conformers:
        builder = InterResidueAngleBuilder()
        builder.process_disulfide_angles(map_1, map_2, link_angles)
        result = builder.finalize(device)
    """

    def __init__(self, verbose: int = 0):
        """Initialize builder with accumulators for disulfide angles."""
        self.verbose = verbose
        # Accumulators for incremental disulfide building
        self._indices: List[np.ndarray] = []
        self._references: List[np.ndarray] = []
        self._sigmas: List[np.ndarray] = []
        self._count: int = 0

    def reset(self):
        """Clear all accumulated data."""
        self._indices.clear()
        self._references.clear()
        self._sigmas.clear()
        self._count = 0

    def process_disulfide_angles(
        self,
        map_1: Dict[str, int],
        map_2: Dict[str, int],
        link_angles: pd.DataFrame,
    ) -> int:
        """
        Process disulfide angle restraints.

        Parameters
        ----------
        map_1, map_2 : dict
            ``{atom name: row}`` of the two cysteine conformers the bond joins, as
            :func:`_conformer_maps` gives them.
        link_angles : pd.DataFrame
            Angle definitions from disulfide link.

        Returns
        -------
        int
            Number of angle restraints added.
        """
        count = 0
        for _, angle_row in link_angles.iterrows():
            maps = [
                map_1 if angle_row[f"atom_{k}_comp_id"] == "1" else map_2
                for k in (1, 2, 3)
            ]
            names = [angle_row[f"atom{k}"] for k in (1, 2, 3)]
            idx1, idx2, idx3 = (m.get(name) for m, name in zip(maps, names))

            if idx1 is not None and idx2 is not None and idx3 is not None:
                self._indices.append(np.array([[idx1, idx2, idx3]], dtype=np.int64))
                self._references.append(
                    np.array([float(angle_row["value"])], dtype=np.float64)
                )
                self._sigmas.append(
                    np.array([float(angle_row["sigma"])], dtype=np.float64)
                )
                count += 1

        self._count += count
        return count

    def finalize(
        self, device: torch.device, sort_indices: bool = True, min_sigma: float = 1e-4
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Convert accumulated data to sorted tensors."""
        if not self._indices:
            return None

        indices = np.concatenate(self._indices, axis=0)
        references = np.concatenate(self._references)
        sigmas = np.concatenate(self._sigmas)

        if sort_indices and len(indices) > 0:
            sort_order = np.argsort(indices[:, 0])
            indices = indices[sort_order]
            references = references[sort_order]
            sigmas = sigmas[sort_order]

        sigmas = np.where(sigmas == 0, min_sigma, sigmas)

        return {
            "indices": torch.tensor(indices, dtype=get_int_dtype(), device=device),
            "references": torch.tensor(
                references, dtype=get_float_dtype(), device=device
            ),
            "sigmas": torch.tensor(sigmas, dtype=get_float_dtype(), device=device),
        }

    @property
    def count(self) -> int:
        """Return total number of restraints accumulated."""
        return self._count

    def build(
        self,
        residues: "PeptideResidues",
        link_dict: Dict,
        device: torch.device,
        sort_indices: bool = True,
        next_resname_filter: Optional[str] = None,
        exclude_next_resname: Optional[str] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Build all inter-residue angle restraints.

        Parameters
        ----------
        residues : PeptideResidues
            The linked residue pairs and their atoms.
        link_dict : Dict
            Link definition dictionary containing angle parameters.
        device : torch.device
            Target device for tensors.
        sort_indices : bool, optional
            Sort output by first atom index (default True).
        next_resname_filter, exclude_next_resname : str, optional
            Select pairs by the second residue's name, as
            :meth:`PeptideResidues.conformer_pairs` does.
        """
        if "angles" not in link_dict or link_dict["angles"] is None:
            return None

        link_data = PreprocessedLinkData(link_dict)
        if link_data.angles is None:
            return None

        if not residues.pairs:
            return None

        all_indices = []
        all_refs = []
        all_sigmas = []

        angles = link_data.angles
        n_angles = len(angles["atom1"])

        for _, _, map_i, map_next in residues.conformer_pairs(
            next_resname_filter=next_resname_filter,
            exclude_next_resname=exclude_next_resname,
        ):
            for a in range(n_angles):
                comp1, comp2, comp3 = (
                    angles["comp1"][a],
                    angles["comp2"][a],
                    angles["comp3"][a],
                )
                atom1, atom2, atom3 = (
                    angles["atom1"][a],
                    angles["atom2"][a],
                    angles["atom3"][a],
                )

                map1 = map_i if comp1 == "1" else map_next
                map2 = map_i if comp2 == "1" else map_next
                map3 = map_i if comp3 == "1" else map_next

                if atom1 in map1 and atom2 in map2 and atom3 in map3:
                    idx1, idx2, idx3 = map1[atom1], map2[atom2], map3[atom3]
                    all_indices.append([idx1, idx2, idx3])
                    all_refs.append(angles["value"][a])
                    all_sigmas.append(angles["sigma"][a])

        if not all_indices:
            return None

        indices = np.array(all_indices, dtype=np.int64)
        references = np.array(all_refs, dtype=np.float64)
        sigmas = np.array(all_sigmas, dtype=np.float64)

        if sort_indices and len(indices) > 0:
            order = np.argsort(indices[:, 0])
            indices = indices[order]
            references = references[order]
            sigmas = sigmas[order]

        sigmas = np.where(sigmas == 0, 1e-4, sigmas)

        return {
            "indices": torch.tensor(indices, dtype=get_int_dtype(), device=device),
            "references": torch.tensor(
                references, dtype=get_float_dtype(), device=device
            ),
            "sigmas": torch.tensor(sigmas, dtype=get_float_dtype(), device=device),
        }


class InterResidueTorsionBuilder:
    """
    Fast builder for inter-residue torsion restraints (phi, psi, omega).

    Usage:
        builder = InterResidueTorsionBuilder()
        result = builder.build(residues, link_dict, device)
        # result = {'phi': {...}, 'psi': {...}, 'omega': {...},
        #           'ramachandran': {...}}

        # Or for disulfides (incremental), per pair of cysteine conformers:
        builder = InterResidueTorsionBuilder()
        builder.process_disulfide_torsions(map_1, map_2, link_torsions)
        result = builder.finalize_disulfide(device)
    """

    def __init__(self, verbose: int = 0):
        """Initialize builder with accumulators for disulfide torsions."""
        self.verbose = verbose
        # Accumulators for disulfide torsions
        self._disulfide_indices: List[np.ndarray] = []
        self._disulfide_references: List[np.ndarray] = []
        self._disulfide_sigmas: List[np.ndarray] = []
        self._disulfide_periods: List[np.ndarray] = []
        self._disulfide_count: int = 0

    def reset(self):
        """Clear all accumulated disulfide data."""
        self._disulfide_indices.clear()
        self._disulfide_references.clear()
        self._disulfide_sigmas.clear()
        self._disulfide_periods.clear()
        self._disulfide_count = 0

    def process_disulfide_torsions(
        self,
        map_1: Dict[str, int],
        map_2: Dict[str, int],
        link_torsions: pd.DataFrame,
    ) -> int:
        """
        Process disulfide torsion restraints.

        Parameters
        ----------
        map_1, map_2 : dict
            ``{atom name: row}`` of the two cysteine conformers the bond joins, as
            :func:`_conformer_maps` gives them.
        link_torsions : pd.DataFrame
            Torsion definitions from disulfide link.

        Returns
        -------
        int
            Number of torsion restraints added.
        """
        count = 0
        for _, torsion_row in link_torsions.iterrows():
            maps = [
                map_1 if torsion_row[f"atom_{k}_comp_id"] == "1" else map_2
                for k in (1, 2, 3, 4)
            ]
            names = [torsion_row[f"atom{k}"] for k in (1, 2, 3, 4)]
            idx1, idx2, idx3, idx4 = (m.get(name) for m, name in zip(maps, names))

            if idx1 is None or idx2 is None or idx3 is None or idx4 is None:
                continue

            self._disulfide_indices.append(
                np.array([[idx1, idx2, idx3, idx4]], dtype=np.int64)
            )
            self._disulfide_references.append(
                np.array([float(torsion_row["value"])], dtype=np.float64)
            )
            self._disulfide_sigmas.append(
                np.array([float(torsion_row["sigma"])], dtype=np.float64)
            )
            self._disulfide_periods.append(
                np.array([2], dtype=np.int64)
            )  # Period 2 for disulfide
            count += 1

        self._disulfide_count += count
        return count

    def finalize_disulfide(
        self, device: torch.device, sort_indices: bool = True
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Finalize disulfide torsion restraints."""
        if not self._disulfide_indices:
            return None

        indices = np.concatenate(self._disulfide_indices, axis=0)
        references = np.concatenate(self._disulfide_references)
        sigmas = np.concatenate(self._disulfide_sigmas)
        periods = np.concatenate(self._disulfide_periods)

        if sort_indices and len(indices) > 0:
            sort_order = np.argsort(indices[:, 0])
            indices = indices[sort_order]
            references = references[sort_order]
            sigmas = sigmas[sort_order]
            periods = periods[sort_order]

        return {
            "indices": torch.tensor(indices, dtype=get_int_dtype(), device=device),
            "references": torch.tensor(
                references, dtype=get_float_dtype(), device=device
            ),
            "sigmas": torch.tensor(sigmas, dtype=get_float_dtype(), device=device),
            "periods": torch.tensor(periods, dtype=get_int_dtype(), device=device),
        }

    @property
    def disulfide_count(self) -> int:
        """Return total number of disulfide torsion restraints accumulated."""
        return self._disulfide_count

    def build(
        self,
        residues: "PeptideResidues",
        link_dict: Dict,
        device: torch.device,
        sort_indices: bool = True,
    ) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        """
        Build all inter-residue torsion restraints.

        Returns separate phi, psi, omega, and ramachandran results.
        """
        if "torsions" not in link_dict or link_dict["torsions"] is None:
            return None

        link_data = PreprocessedLinkData(link_dict)
        if link_data.torsions is None:
            return None

        if not residues.pairs:
            return None

        # Separate accumulators for phi, psi, omega
        phi_data = {"indices": [], "periods": []}
        psi_data = {"indices": [], "periods": []}
        omega_data = {
            "indices": [],
            "references": [],
            "sigmas": [],
            "periods": [],
            "is_proline": [],
        }
        # Ramachandran: collect phi/psi per residue, then match afterwards
        # phi from pair (i, j) belongs to residue j (second residue)
        # psi from pair (i, j) belongs to residue i (first residue)
        phi_by_residue = {}  # res_idx -> atom indices
        psi_by_residue = {}  # res_idx -> atom indices
        omega_idx_by_residue = {}  # res_idx -> omega atom indices (cis/trans PRO)
        resname_by_residue = {}  # res_idx -> resname
        next_resname_by_residue = {}  # res_idx -> next resname (for pre-PRO)

        torsions = link_data.torsions
        n_torsions = len(torsions["atom1"])

        from torchref.topology.ramachandran import classify_residue

        for res_i_idx, res_next_idx, map_i, map_next in residues.conformer_pairs():
            resname_i = residues.conformer_resname(map_i)
            resname_next = residues.conformer_resname(map_next)
            is_proline = resname_next == "PRO"
            key_i = (res_i_idx, resname_i)
            key_next = (res_next_idx, resname_next)

            # Track which residue each phi/psi belongs to
            pair_phi = None  # phi from this pair belongs to res_next_idx
            pair_psi = None  # psi from this pair belongs to res_i_idx

            for t in range(n_torsions):
                comp1 = torsions["comp1"][t]
                comp2 = torsions["comp2"][t]
                comp3 = torsions["comp3"][t]
                comp4 = torsions["comp4"][t]
                atom1 = torsions["atom1"][t]
                atom2 = torsions["atom2"][t]
                atom3 = torsions["atom3"][t]
                atom4 = torsions["atom4"][t]
                torsion_id = torsions["id"][t]

                map1 = map_i if comp1 == "1" else map_next
                map2 = map_i if comp2 == "1" else map_next
                map3 = map_i if comp3 == "1" else map_next
                map4 = map_i if comp4 == "1" else map_next

                if not (
                    atom1 in map1 and atom2 in map2 and atom3 in map3 and atom4 in map4
                ):
                    continue

                idx1, idx2, idx3, idx4 = (
                    map1[atom1],
                    map2[atom2],
                    map3[atom3],
                    map4[atom4],
                )
                period = int(torsions["period"][t])

                if torsion_id == "phi":
                    phi_data["indices"].append([idx1, idx2, idx3, idx4])
                    phi_data["periods"].append(period)
                    pair_phi = [idx1, idx2, idx3, idx4]
                elif torsion_id == "psi":
                    psi_data["indices"].append([idx1, idx2, idx3, idx4])
                    psi_data["periods"].append(period)
                    pair_psi = [idx1, idx2, idx3, idx4]
                elif torsion_id == "omega":
                    omega_data["indices"].append([idx1, idx2, idx3, idx4])
                    omega_data["references"].append(float(torsions["value"][t]))
                    omega_data["sigmas"].append(float(torsions["sigma"][t]))
                    omega_data["periods"].append(period)
                    omega_data["is_proline"].append(is_proline)

            # Store phi/psi by the residue they actually belong to:
            # phi: C(i) - N(j) - CA(j) - C(j)  → belongs to residue j
            # psi: N(i) - CA(i) - C(i)  - N(j)  → belongs to residue i
            if pair_phi is not None:
                phi_by_residue[key_next] = pair_phi
            if pair_psi is not None:
                psi_by_residue[key_i] = pair_psi
            # Track residue names and next-residue names for classification
            resname_by_residue[key_i] = resname_i
            resname_by_residue[key_next] = resname_next
            next_resname_by_residue[key_i] = resname_next
            # The omega that decides PRO cis/trans, measured after the loop
            if omega_data["indices"]:
                omega_idx_by_residue[key_next] = omega_data["indices"][-1]

        result = {}

        # Finalize phi
        if phi_data["indices"]:
            indices = np.array(phi_data["indices"], dtype=np.int64)
            periods = np.array(phi_data["periods"], dtype=np.int64)
            if sort_indices:
                order = np.argsort(indices[:, 0])
                indices = indices[order]
                periods = periods[order]
            result["phi"] = {
                "indices": torch.tensor(indices, dtype=get_int_dtype(), device=device),
                "periods": torch.tensor(periods, dtype=get_int_dtype(), device=device),
            }

        # Finalize psi
        if psi_data["indices"]:
            indices = np.array(psi_data["indices"], dtype=np.int64)
            periods = np.array(psi_data["periods"], dtype=np.int64)
            if sort_indices:
                order = np.argsort(indices[:, 0])
                indices = indices[order]
                periods = periods[order]
            result["psi"] = {
                "indices": torch.tensor(indices, dtype=get_int_dtype(), device=device),
                "periods": torch.tensor(periods, dtype=get_int_dtype(), device=device),
            }

        # Finalize omega
        if omega_data["indices"]:
            indices = np.array(omega_data["indices"], dtype=np.int64)
            references = np.array(omega_data["references"], dtype=np.float64)
            sigmas = np.array(omega_data["sigmas"], dtype=np.float64)
            periods = np.array(omega_data["periods"], dtype=np.int64)
            is_proline = np.array(omega_data["is_proline"], dtype=bool)
            if sort_indices:
                order = np.argsort(indices[:, 0])
                indices = indices[order]
                references = references[order]
                sigmas = sigmas[order]
                periods = periods[order]
                is_proline = is_proline[order]
            result["omega"] = {
                "indices": torch.tensor(indices, dtype=get_int_dtype(), device=device),
                "references": torch.tensor(
                    references, dtype=get_float_dtype(), device=device
                ),
                "sigmas": torch.tensor(sigmas, dtype=get_float_dtype(), device=device),
                "periods": torch.tensor(periods, dtype=get_int_dtype(), device=device),
                "is_proline": torch.tensor(is_proline, dtype=torch.bool, device=device),
            }

        # Finalize ramachandran — match phi and psi for the SAME residue
        # phi_by_residue[r] = phi atom indices for residue r
        # psi_by_residue[r] = psi atom indices for residue r
        # A residue needs both phi and psi for a Ramachandran restraint
        rama_residues = sorted(
            set(phi_by_residue.keys()) & set(psi_by_residue.keys())
        )
        if rama_residues:
            omega_keys = [r for r in rama_residues if r in omega_idx_by_residue]
            omega_by_residue = {}
            if omega_keys:
                omega_values = torsions_from_xyz(
                    torch.as_tensor(residues.xyz, dtype=get_float_dtype()),
                    torch.as_tensor(
                        [omega_idx_by_residue[r] for r in omega_keys],
                        dtype=get_int_dtype(),
                    ),
                )
                omega_by_residue = dict(zip(omega_keys, omega_values.tolist()))
            rama_phi = []
            rama_psi = []
            rama_types = []
            for res_idx in rama_residues:
                resname = resname_by_residue[res_idx]
                next_rn = next_resname_by_residue.get(res_idx, "")
                omega_deg = omega_by_residue.get(res_idx, 180.0)
                rama_type = classify_residue(resname, next_rn, omega_deg)
                rama_phi.append(phi_by_residue[res_idx])
                rama_psi.append(psi_by_residue[res_idx])
                rama_types.append(rama_type)

            phi_idx = np.array(rama_phi, dtype=np.int64)
            psi_idx = np.array(rama_psi, dtype=np.int64)
            stypes = np.array(rama_types, dtype=np.int64)
            if sort_indices:
                order = np.argsort(phi_idx[:, 0])
                phi_idx = phi_idx[order]
                psi_idx = psi_idx[order]
                stypes = stypes[order]
            result["ramachandran"] = {
                "phi_indices": torch.tensor(
                    phi_idx, dtype=get_int_dtype(), device=device
                ),
                "psi_indices": torch.tensor(
                    psi_idx, dtype=get_int_dtype(), device=device
                ),
                "surface_type": torch.tensor(
                    stypes, dtype=get_int_dtype(), device=device
                ),
            }

        return result if result else None


class InterResiduePlaneBuilder:
    """
    Fast builder for inter-residue plane restraints (peptide planes).

    Usage:
        builder = InterResiduePlaneBuilder()
        result = builder.build(residues, link_dict, device)
    """

    def __init__(self, verbose: int = 0):
        """Initialize builder."""
        self.verbose = verbose

    def build(
        self,
        residues: "PeptideResidues",
        link_dict: Dict,
        device: torch.device,
        sort_indices: bool = True,
        next_resname_filter: Optional[str] = None,
        exclude_next_resname: Optional[str] = None,
    ) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        """Build all inter-residue plane restraints, grouped by atom count.

        ``next_resname_filter`` and ``exclude_next_resname`` select pairs by the second
        residue's name, as :meth:`PeptideResidues.conformer_pairs` does.
        """
        if "planes" not in link_dict or link_dict["planes"] is None:
            return None

        link_data = PreprocessedLinkData(link_dict)
        if link_data.planes is None:
            return None

        if not residues.pairs:
            return None

        # Group planes by atom count
        planes_by_size: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}

        for _, _, map_i, map_next in residues.conformer_pairs(
            next_resname_filter=next_resname_filter,
            exclude_next_resname=exclude_next_resname,
        ):
            for plane_data in link_data.planes:
                comp_ids = plane_data["comp_ids"]
                atom_names = plane_data["atoms"]
                sigmas = plane_data["sigmas"]

                plane_indices = []
                plane_sigmas = []
                all_found = True

                for comp_id, atom_name, sigma in zip(comp_ids, atom_names, sigmas):
                    atom_map = map_i if comp_id == "1" else map_next
                    if atom_name in atom_map:
                        plane_indices.append(atom_map[atom_name])
                        plane_sigmas.append(sigma)
                    else:
                        all_found = False
                        break

                if all_found and len(plane_indices) >= 3:
                    n_atoms = len(plane_indices)
                    if n_atoms not in planes_by_size:
                        planes_by_size[n_atoms] = []
                    planes_by_size[n_atoms].append(
                        (
                            np.array(plane_indices, dtype=np.int64),
                            np.array(plane_sigmas, dtype=np.float64),
                        )
                    )

        if not planes_by_size:
            return None

        result = {}
        for n_atoms, planes_list in planes_by_size.items():
            indices = np.stack([p[0] for p in planes_list], axis=0)
            sigmas = np.stack([p[1] for p in planes_list], axis=0)

            if sort_indices and len(indices) > 0:
                order = np.argsort(indices[:, 0])
                indices = indices[order]
                sigmas = sigmas[order]

            sigmas = np.where(sigmas == 0, 1e-4, sigmas)

            key = f"{n_atoms}_atoms"
            result[key] = {
                "indices": torch.tensor(indices, dtype=get_int_dtype(), device=device),
                "sigmas": torch.tensor(sigmas, dtype=get_float_dtype(), device=device),
            }

        return result
