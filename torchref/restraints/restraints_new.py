"""
Restraints Class (Refactored) for Crystallographic Model Refinement

This module provides a refactored restraints handler using the builder pattern.
It maintains the same interface as the original Restraints class but uses
the more efficient and testable builder classes internally.

Key improvements:
- Single-pass iteration over residues (vs multiple passes in original)
- Pre-grouped residue data for O(N log N) vs O(N×R) complexity
- Sorted indices for cache-friendly tensor access
- Separated builder classes for easier testing and maintenance
"""

from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.nn import Module

from torchref.model.model import Model
from torchref.restraints.builders_fast import (
    AngleRestraintBuilder,
    BondRestraintBuilder,
    ChiralRestraintBuilder,
    InterResidueAngleBuilder,
    InterResidueBondBuilder,
    InterResiduePlaneBuilder,
    InterResidueTorsionBuilder,
    PlaneRestraintBuilder,
    TorsionRestraintBuilder,
)
from torchref.restraints.restraints_helper import (
    find_cif_file_in_library,
    read_cif,
    read_link_definitions,
)
from torchref.utils.debug_utils import DebugMixin
from torchref.utils.utils import ModuleReference


class RestraintsNew(DebugMixin, Module):
    """
    Refactored restraints handler for crystallographic model refinement.

    This class uses the builder pattern internally for efficient construction
    of restraint tensors. It maintains API compatibility with the original
    Restraints class.

    Parameters
    ----------
    model : Model, optional
        Model instance containing the atomic structure. If None, creates empty shell.
    cif_path : str or list of str, optional
        Path to the CIF restraints dictionary file(s).
    verbose : int, default 1
        Verbosity level (0=silent, 1=normal, 2=detailed).

    Attributes
    ----------
    model : ModuleReference
        Reference to the Model instance.
    cif_dict : dict
        Parsed CIF dictionary with restraints for each residue type.
    restraints : dict
        Hierarchical dictionary containing all restraints.
    """

    def __init__(self, model: Model = None, cif_path=None, verbose: int = 1):
        """Initialize the Restraints handler."""
        super().__init__()
        self.cif_path = cif_path
        self.verbose = verbose

        # Empty initialization
        if model is None:
            self.model = None
            self.cif_dict = {}
            self.unique_residues = []
            self.restraints = {}
            return

        # Full initialization with model
        self.model = ModuleReference(model)
        self.unique_residues = model.pdb.resname.unique()
        self.unique_residues = [
            residue
            for residue in self.unique_residues
            if self.model.pdb.loc[
                self.model.pdb["resname"] == residue, "name"
            ].nunique()
            > 1
        ]

        # Parse CIF files
        self._load_cif_dictionaries(cif_path)

        # Load link definitions for inter-residue restraints
        if verbose > 1:
            print("Loading link definitions from monomer library...")
        self.link_dict, self.link_list = read_link_definitions()
        if verbose > 1:
            print(f"Loaded {len(self.link_dict)} link types")

        # Initialize hierarchical restraints dictionary
        self.restraints = {"bond": {}, "angle": {}, "torsion": {}, "plane": {}}

        # Build restraints using the new builder pattern
        self.build_restraints()
        if self.verbose > 0:
            self.summary()

    def _load_cif_dictionaries(self, cif_path):
        """Load CIF dictionaries from provided paths and monomer library."""
        if cif_path:
            if isinstance(cif_path, str):
                try:
                    self.cif_dict = read_cif(cif_path)
                except ValueError as e:
                    print("Error reading CIF file:", e)
                    raise
                except Exception as e:
                    print("Error reading CIF file:", e)
                    self.cif_dict = {}
            elif isinstance(cif_path, list):
                self.cif_dict = {}
                for cif_file in cif_path:
                    try:
                        cif_dict_part = read_cif(cif_file)
                        self.cif_dict.update(cif_dict_part)
                    except ValueError as e:
                        print("Error reading CIF file:", e)
                        raise
                    except Exception as e:
                        print("Error reading CIF file:", e)
            else:
                raise ValueError("cif_path must be a string or a list of strings")
        else:
            self.cif_dict = {}

        # Load missing residues from monomer library
        self.missing_residues = [
            res for res in self.unique_residues if res not in self.cif_dict
        ]
        additional_files = [
            find_cif_file_in_library(res) for res in self.missing_residues
        ]

        for cif_file in additional_files:
            if cif_file is not None:
                if self.verbose > 1:
                    print(cif_file)
                try:
                    additional_cif_dict = read_cif(cif_file)
                    self.cif_dict.update(additional_cif_dict)
                except Exception as e:
                    print("Error reading CIF file:", e)
                    print("This residue will have no restraints applied.")

        self.missing_residues = [
            res for res in self.unique_residues if res not in self.cif_dict
        ]

        if len(self.missing_residues) > 1:
            if self.verbose > 0:
                print(
                    f"Warning: The following residues are missing from the CIF dictionary "
                    f"and will have no restraints applied: {self.missing_residues}"
                )

    def expand_altloc(self, residue):
        """
        Expand residue with alternative conformations into separate conformations.

        Yields one DataFrame per altloc (with common atoms included in each).
        """
        residue = residue.copy()
        residue.loc[residue["altloc"].isin(["", " "]), "altloc"] = " "

        alt_conf = residue["altloc"].unique()
        if " " in alt_conf:
            residue_no_alt = residue.loc[residue["altloc"] == " "]
            for alt in alt_conf:
                if alt == " ":
                    continue
                residue_alt = residue.loc[residue["altloc"] == alt]
                residue_combined = pd.concat(
                    [residue_no_alt, residue_alt], ignore_index=True
                )
                yield residue_combined
        else:
            for alt_loc in alt_conf:
                residue_alt = residue.loc[residue["altloc"] == alt_loc]
                yield residue_alt

    def build_restraints(self):
        """
        Build all restraints using the fast builder API.

        This method uses the optimized builders that handle all residues
        internally with Numba-accelerated matching (~10x faster).
        """
        try:
            device = self.model.xyz().device
            pdb = self.model.pdb

            # Build intra-residue restraints using fast builders
            # Each builder.build() handles all residues internally - no looping needed!

            bond_result = BondRestraintBuilder(verbose=self.verbose).build(
                pdb, self.cif_dict, device
            )
            if bond_result:
                self.restraints["bond"]["intra"] = bond_result

            angle_result = AngleRestraintBuilder(verbose=self.verbose).build(
                pdb, self.cif_dict, device
            )
            if angle_result:
                self.restraints["angle"]["intra"] = angle_result

            torsion_result = TorsionRestraintBuilder(verbose=self.verbose).build(
                pdb, self.cif_dict, device
            )
            if torsion_result:
                self.restraints["torsion"]["intra"] = torsion_result

            plane_result = PlaneRestraintBuilder(verbose=self.verbose).build(
                pdb, self.cif_dict, device
            )
            if plane_result:
                for key, data in plane_result.items():
                    self.restraints["plane"][key] = data

            chiral_result = ChiralRestraintBuilder(verbose=self.verbose).build(
                pdb, self.cif_dict, device
            )
            if chiral_result:
                self.restraints["chiral"] = chiral_result

            # Build inter-residue restraints
            self._build_peptide_restraints(device)
            self._build_disulfide_restraints(device)

            # Build VDW restraints
            self._build_vdw_restraints(
                cutoff=5.0, sigma=0.05, inter_residue_only=False, use_spatial_hash=True
            )

        except Exception as e:
            self.debug_on_error(e, context="RestraintsNew.build_restraints")
            raise

    def _build_peptide_restraints(self, device: torch.device):
        """Build peptide bond restraints using fast inter-residue builders."""
        if "TRANS" not in self.link_dict:
            if self.verbose > 0:
                print(
                    "Warning: TRANS link not found in link dictionary, skipping peptide bonds"
                )
            return

        trans_link = self.link_dict["TRANS"]
        pdb = self.model.pdb

        # Build peptide bonds using fast builder
        bond_result = InterResidueBondBuilder(verbose=self.verbose).build(
            pdb, trans_link, device, filter_atom_type="ATOM"
        )
        if bond_result:
            self.restraints["bond"]["peptide"] = bond_result
            if self.verbose > 0:
                print(
                    f"Built {bond_result['indices'].shape[0]} peptide bond restraints"
                )

        # Build peptide angles
        angle_result = InterResidueAngleBuilder(verbose=self.verbose).build(
            pdb, trans_link, device, filter_atom_type="ATOM"
        )
        if angle_result:
            self.restraints["angle"]["peptide"] = angle_result
            if self.verbose > 0:
                print(
                    f"Built {angle_result['indices'].shape[0]} peptide angle restraints"
                )

        # Build backbone torsions (phi, psi, omega)
        torsion_result = InterResidueTorsionBuilder(verbose=self.verbose).build(
            pdb, trans_link, device, filter_atom_type="ATOM"
        )
        if torsion_result:
            if "phi" in torsion_result:
                self.restraints["torsion"]["phi"] = torsion_result["phi"]
            if "psi" in torsion_result:
                self.restraints["torsion"]["psi"] = torsion_result["psi"]
            if "omega" in torsion_result:
                self.restraints["torsion"]["omega"] = torsion_result["omega"]

        # Build peptide planes
        plane_result = InterResiduePlaneBuilder(verbose=self.verbose).build(
            pdb, trans_link, device, filter_atom_type="ATOM"
        )
        if plane_result:
            n_planes = 0
            for key, data in plane_result.items():
                n_planes += data["indices"].shape[0]
                if key in self.restraints["plane"]:
                    # Append to existing planes of same atom count
                    existing = self.restraints["plane"][key]
                    self.restraints["plane"][key] = {
                        "indices": torch.cat(
                            [existing["indices"], data["indices"]], dim=0
                        ),
                        "sigmas": torch.cat(
                            [existing["sigmas"], data["sigmas"]], dim=0
                        ),
                    }
                else:
                    self.restraints["plane"][key] = data
            if self.verbose > 0:
                print(f"Built {n_planes} peptide plane restraints")

    def _build_disulfide_restraints(self, device: torch.device):
        """Build disulfide bond restraints."""
        if "disulf" not in self.link_dict:
            if self.verbose > 1:
                print(
                    "Warning: disulf link not found in link dictionary, skipping disulfide bonds"
                )
            return

        disulf_link = self.link_dict["disulf"]
        disulf_bonds = disulf_link.get("bonds")
        disulf_angles = disulf_link.get("angles")
        disulf_torsions = disulf_link.get("torsions")

        if disulf_bonds is None:
            return

        # Get SG-SG bond parameters
        sg_sg_bond = disulf_bonds[
            (disulf_bonds["atom1"] == "SG") & (disulf_bonds["atom2"] == "SG")
        ]

        if len(sg_sg_bond) == 0:
            return

        bond_length = float(sg_sg_bond["value"].values[0])
        bond_sigma = float(sg_sg_bond["sigma"].values[0])

        # Find all SG atoms
        pdb = self.model.pdb
        sg_atoms = pdb[(pdb["name"] == "SG") & (pdb["ATOM"] == "ATOM")]

        if len(sg_atoms) == 0:
            return

        # Get coordinates and find close pairs
        xyz = self.model.xyz()
        sg_indices = sg_atoms["index"].values
        sg_coords = xyz[sg_indices]
        sg_residues = (
            sg_atoms["chainid"].astype(str) + "_" + sg_atoms["resseq"].astype(str)
        ).values

        distances = torch.cdist(sg_coords, sg_coords)
        threshold = 4.0
        close_pairs = torch.where((distances < threshold) & (distances > 0.1))

        valid_pairs = []
        for i, j in zip(close_pairs[0].cpu().numpy(), close_pairs[1].cpu().numpy()):
            if i < j and sg_residues[i] != sg_residues[j]:
                valid_pairs.append((i, j))

        if len(valid_pairs) == 0:
            return

        # Create builders
        bond_builder = InterResidueBondBuilder(verbose=self.verbose)
        angle_builder = InterResidueAngleBuilder(verbose=self.verbose)
        torsion_builder = InterResidueTorsionBuilder(verbose=self.verbose)

        # Process each disulfide bond
        for i_local, j_local in valid_pairs:
            sg1_idx = int(sg_indices[i_local])
            sg2_idx = int(sg_indices[j_local])

            # Add bond
            bond_builder.process_disulfide_bond(
                sg1_idx, sg2_idx, bond_length, bond_sigma
            )

            # Get residues for angle/torsion restraints
            residue1 = pdb[pdb["index"] == sg1_idx].iloc[0]
            residue2 = pdb[pdb["index"] == sg2_idx].iloc[0]

            res1_atoms = pdb[
                (pdb["chainid"] == residue1["chainid"])
                & (pdb["resseq"] == residue1["resseq"])
            ]
            res2_atoms = pdb[
                (pdb["chainid"] == residue2["chainid"])
                & (pdb["resseq"] == residue2["resseq"])
            ]

            if disulf_angles is not None:
                angle_builder.process_disulfide_angles(
                    res1_atoms, res2_atoms, disulf_angles
                )

            if disulf_torsions is not None:
                torsion_builder.process_disulfide_torsions(
                    res1_atoms, res2_atoms, disulf_torsions
                )

        # Finalize
        bond_result = bond_builder.finalize(device)
        if bond_result:
            self.restraints["bond"]["disulfide"] = bond_result
            if self.verbose > 0:
                print(
                    f"Built {bond_result['indices'].shape[0]} disulfide bond restraints"
                )

        angle_result = angle_builder.finalize(device)
        if angle_result:
            self.restraints["angle"]["disulfide"] = angle_result
            if self.verbose > 0:
                print(
                    f"Built {angle_result['indices'].shape[0]} disulfide angle restraints"
                )

        torsion_result = torsion_builder.finalize_disulfide(device)
        if torsion_result:
            self.restraints["torsion"]["disulfide"] = torsion_result
            if self.verbose > 0:
                print(
                    f"Built {torsion_result['indices'].shape[0]} disulfide torsion restraints"
                )

    def _build_exclusion_set(self):
        """Build set of atom pairs to exclude from VDW calculations."""
        exclusions = set()

        # 1-2: Direct bonds
        for origin in self.restraints.get("bond", {}).keys():
            indices = self.restraints["bond"][origin].get("indices")
            if indices is not None and len(indices) > 0:
                idx_np = indices.cpu().numpy()
                for i1, i2 in idx_np:
                    exclusions.add((int(min(i1, i2)), int(max(i1, i2))))

        # 1-3: Angles
        for origin in self.restraints.get("angle", {}).keys():
            indices = self.restraints["angle"][origin].get("indices")
            if indices is not None and len(indices) > 0:
                idx_np = indices.cpu().numpy()
                for i1, i2, i3 in idx_np:
                    exclusions.add((int(min(i1, i3)), int(max(i1, i3))))

        # 1-4: Torsions
        for origin in self.restraints.get("torsion", {}).keys():
            indices = self.restraints["torsion"][origin].get("indices")
            if indices is not None and len(indices) > 0:
                idx_np = indices.cpu().numpy()
                for i1, i2, i3, i4 in idx_np:
                    exclusions.add((int(min(i1, i4)), int(max(i1, i4))))

        return exclusions

    def _find_nearby_pairs_spatial_hash(self, xyz, cutoff=6.0):
        """
        Find all atom pairs within cutoff distance.

        Uses torch.cdist for vectorized distance computation, which is
        faster than spatial hashing for typical protein structures where
        atoms are densely packed.

        Parameters
        ----------
        xyz : torch.Tensor
            Atom coordinates of shape (N, 3).
        cutoff : float
            Distance cutoff in Angstroms.

        Returns
        -------
        torch.Tensor
            Pairs of atom indices, shape (M, 2).
        """
        device = xyz.device
        n_atoms = xyz.shape[0]

        if n_atoms == 0:
            return torch.tensor([], dtype=torch.long, device=device).reshape(0, 2)

        # Use cdist for vectorized distance computation
        dist_matrix = torch.cdist(xyz, xyz)

        # Find close pairs (upper triangle only to avoid duplicates)
        mask = (dist_matrix < cutoff) & (dist_matrix > 0)
        mask = torch.triu(mask, diagonal=1)

        pairs = torch.stack(torch.where(mask), dim=1)
        return pairs

    def _build_vdw_restraints(
        self, cutoff=5.0, sigma=0.2, inter_residue_only=True, use_spatial_hash=True
    ):
        """Build van der Waals (non-bonded contact) restraints."""
        if self.verbose > 0:
            print("\nBuilding VDW (non-bonded) restraints...")

        exclusions = self._build_exclusion_set()
        vdw_radii = self.model.get_vdw_radii()
        xyz = self.model.xyz()
        device = xyz.device
        pdb = self.model.pdb

        # Find nearby pairs
        if use_spatial_hash:
            nearby_pairs = self._find_nearby_pairs_spatial_hash(xyz, cutoff)
        else:
            n_atoms = xyz.shape[0]
            pairs_list = []
            cutoff_sq = cutoff**2
            for i in range(n_atoms):
                for j in range(i + 1, n_atoms):
                    dist_sq = ((xyz[i] - xyz[j]) ** 2).sum()
                    if dist_sq < cutoff_sq:
                        pairs_list.append([i, j])
            nearby_pairs = (
                torch.tensor(pairs_list, dtype=torch.long, device=device)
                if pairs_list
                else torch.tensor([], dtype=torch.long, device=device).reshape(0, 2)
            )

        if len(nearby_pairs) == 0:
            self.restraints["vdw"] = {
                "indices": torch.tensor([], dtype=torch.long, device=device).reshape(
                    0, 2
                ),
                "min_distances": torch.tensor([], dtype=torch.float32, device=device),
                "sigmas": torch.tensor([], dtype=torch.float32, device=device),
            }
            return

        # Vectorized filtering
        pairs_np = nearby_pairs.cpu().numpy()
        i1_arr = pairs_np[:, 0]
        i2_arr = pairs_np[:, 1]

        # Create exclusion mask efficiently using numpy
        # Convert exclusion set to array for vectorized lookup
        if exclusions:
            exclusion_arr = np.array(list(exclusions), dtype=np.int64)
            # Create a hash for fast lookup
            max_idx = max(pdb["index"].max() + 1, i1_arr.max() + 1, i2_arr.max() + 1)
            pair_hash = i1_arr * max_idx + i2_arr
            excl_hash = exclusion_arr[:, 0] * max_idx + exclusion_arr[:, 1]
            exclusion_mask = np.isin(pair_hash, excl_hash)
        else:
            exclusion_mask = np.zeros(len(pairs_np), dtype=bool)

        # Inter-residue mask
        if inter_residue_only:
            chainid_array = pdb["chainid"].values
            resseq_array = pdb["resseq"].values
            same_residue = (chainid_array[i1_arr] == chainid_array[i2_arr]) & (
                resseq_array[i1_arr] == resseq_array[i2_arr]
            )
        else:
            same_residue = np.zeros(len(pairs_np), dtype=bool)

        # Combined mask: keep pairs that are not excluded and not same-residue
        keep_mask = ~exclusion_mask & ~same_residue

        # Filter pairs
        filtered_pairs = pairs_np[keep_mask]

        if len(filtered_pairs) == 0:
            self.restraints["vdw"] = {
                "indices": torch.tensor([], dtype=torch.long, device=device).reshape(
                    0, 2
                ),
                "min_distances": torch.tensor([], dtype=torch.float32, device=device),
                "sigmas": torch.tensor([], dtype=torch.float32, device=device),
            }
            return

        # Compute min distances vectorized
        vdw_np = vdw_radii.cpu().numpy()
        min_distances = vdw_np[filtered_pairs[:, 0]] + vdw_np[filtered_pairs[:, 1]]

        self.restraints["vdw"] = {
            "indices": torch.tensor(filtered_pairs, dtype=torch.long, device=device),
            "min_distances": torch.tensor(
                min_distances, dtype=torch.float32, device=device
            ),
            "sigmas": torch.full(
                (len(filtered_pairs),), sigma, dtype=torch.float32, device=device
            ),
        }

        if self.verbose > 0:
            scope = "inter-residue" if inter_residue_only else "all"
            print(f"  Built {len(filtered_pairs)} VDW restraints ({scope} contacts)")

    def _move_to_device(self, device):
        """Move all restraint tensors to the specified device."""
        for restraint_type in ["bond", "angle", "torsion", "plane"]:
            if restraint_type in self.restraints:
                for origin, properties in self.restraints[restraint_type].items():
                    if isinstance(properties, dict):
                        for prop_name, tensor in properties.items():
                            if tensor is not None and isinstance(tensor, torch.Tensor):
                                self.restraints[restraint_type][origin][prop_name] = (
                                    tensor.to(device)
                                )

        if "vdw" in self.restraints:
            for prop_name, tensor in self.restraints["vdw"].items():
                if tensor is not None and isinstance(tensor, torch.Tensor):
                    self.restraints["vdw"][prop_name] = tensor.to(device)

        if "chiral" in self.restraints:
            for prop_name, tensor in self.restraints["chiral"].items():
                if tensor is not None and isinstance(tensor, torch.Tensor):
                    self.restraints["chiral"][prop_name] = tensor.to(device)

    def cuda(self, device: Optional[int] = None):
        """Move all restraint tensors to CUDA device."""
        cuda_device = torch.device("cuda" if device is None else f"cuda:{device}")
        self._move_to_device(cuda_device)
        return self

    def cpu(self):
        """Move all restraint tensors to CPU."""
        self._move_to_device(torch.device("cpu"))
        return self

    def summary(self):
        """Print a detailed summary of all restraints."""
        print("=" * 80)
        print("Restraints Summary (New Implementation)")
        print("=" * 80)
        print(f"CIF file: {self.cif_path}")
        print(f"Residue types in dictionary: {len(self.cif_dict)}")
        print()

        def get_count(rtype, origin):
            indices = self.restraints.get(rtype, {}).get(origin, {}).get("indices")
            return 0 if indices is None else indices.shape[0]

        print("INTRA-RESIDUE RESTRAINTS:")
        print("-" * 80)
        print(f"  Bonds: {get_count('bond', 'intra')}")
        print(f"  Angles: {get_count('angle', 'intra')}")
        print(f"  Torsions: {get_count('torsion', 'intra')}")

        # Count planes
        n_planes = 0
        for key in self.restraints.get("plane", {}).keys():
            n_planes += get_count("plane", key)
        print(f"  Planes: {n_planes}")

        # Chiral
        chiral_count = 0
        if "chiral" in self.restraints:
            indices = self.restraints["chiral"].get("indices")
            chiral_count = 0 if indices is None else indices.shape[0]
        print(f"  Chirals: {chiral_count}")

        print()
        print("INTER-RESIDUE RESTRAINTS:")
        print("-" * 80)
        print(f"  Peptide bonds: {get_count('bond', 'peptide')}")
        print(f"  Peptide angles: {get_count('angle', 'peptide')}")
        print(f"  Disulfide bonds: {get_count('bond', 'disulfide')}")
        print(f"  Disulfide angles: {get_count('angle', 'disulfide')}")
        print(f"  Disulfide torsions: {get_count('torsion', 'disulfide')}")

        print()
        print("BACKBONE TORSIONS:")
        print("-" * 80)
        print(f"  Phi: {get_count('torsion', 'phi')}")
        print(f"  Psi: {get_count('torsion', 'psi')}")
        print(f"  Omega: {get_count('torsion', 'omega')}")

        print()
        print("VDW RESTRAINTS:")
        print("-" * 80)
        vdw_count = 0
        if "vdw" in self.restraints:
            indices = self.restraints["vdw"].get("indices")
            vdw_count = 0 if indices is None else indices.shape[0]
        print(f"  Non-bonded contacts: {vdw_count}")

        print("=" * 80)

    def __repr__(self):
        """Return string representation."""

        def get_count(rtype, origin):
            indices = self.restraints.get(rtype, {}).get(origin, {}).get("indices")
            return 0 if indices is None else indices.shape[0]

        n_bonds = get_count("bond", "intra")
        n_angles = get_count("angle", "intra")
        n_torsions = get_count("torsion", "intra")
        n_bonds_peptide = get_count("bond", "peptide")

        return (
            f"RestraintsNew(bonds={n_bonds}, angles={n_angles}, "
            f"torsions={n_torsions}, peptide_bonds={n_bonds_peptide})"
        )

    def _get_all_indices(self, restraint_type, keys_to_merge=None):
        """
        Gather all indices of a given restraint type across all origins.

        Parameters
        ----------
        restraint_type : str
            Type of restraint ('bond', 'angle', or 'torsion').
        keys_to_merge : list of str, optional
            Specific origins to include. If None, includes all origins.

        Returns
        -------
        torch.Tensor or None
            Concatenated tensor of all indices, or None if none exist.
        """
        indices_list = []
        for origin, data in self.restraints.get(restraint_type, {}).items():
            indices = data.get("indices")
            if indices is not None:
                if keys_to_merge is None:
                    indices_list.append(indices)
                elif origin in keys_to_merge:
                    indices_list.append(indices)

        if not indices_list:
            return None

        return torch.cat(indices_list, dim=0)

    def _get_all_property(self, restraint_type, property_name, keys_to_merge=None):
        """
        Gather all values of a given property across all origins.

        Parameters
        ----------
        restraint_type : str
            Type of restraint ('bond', 'angle', or 'torsion').
        property_name : str
            Property to gather ('references', 'sigmas', or 'periods').
        keys_to_merge : list of str, optional
            Specific origins to include. If None, includes all origins.

        Returns
        -------
        torch.Tensor or None
            Concatenated tensor of all property values, or None if none exist.
        """
        values_list = []
        for origin, data in self.restraints.get(restraint_type, {}).items():
            values = data.get(property_name)
            if values is not None:
                if keys_to_merge is None:
                    values_list.append(values)
                elif origin in keys_to_merge:
                    values_list.append(values)

        if not values_list:
            return None

        return torch.cat(values_list, dim=0)

    def bond_lengths(self, idx):
        """
        Compute current bond lengths from atomic coordinates.

        Parameters
        ----------
        idx : torch.Tensor
            Bond indices tensor of shape (N, 2).

        Returns
        -------
        torch.Tensor
            Tensor of bond lengths of shape (N,).
        """
        if idx is None:
            return torch.tensor([], device=self.model.xyz().device)
        xyz = self.model.xyz()
        pos1 = xyz[idx[:, 0], :]
        pos2 = xyz[idx[:, 1], :]
        return torch.linalg.norm(pos2 - pos1, dim=-1)

    def copy(self):
        """
        Create a deep copy of the Restraints object.

        Returns
        -------
        Restraints
            A deep copy of this Restraints instance.
        """
        import copy

        return copy.deepcopy(self)

    def bond_deviations(self):
        """
        Compute bond length deviations and sigmas.

        Returns
        -------
        deviations : torch.Tensor
            Calculated minus expected bond lengths in Angstroms.
        sigmas : torch.Tensor
            Standard deviations from CIF library in Angstroms.
        """
        if "all" not in self.restraints["bond"]:
            self.cat_dict()

        idx = self.restraints["bond"]["all"]["indices"]
        references = self.restraints["bond"]["all"]["references"]
        sigmas = self.restraints["bond"]["all"]["sigmas"]

        # Get current bond lengths
        bond_lengths = self.bond_lengths(idx)
        deviations = bond_lengths - references

        return deviations, sigmas

    def nll_bonds(self):
        """
        Compute negative log-likelihood for bond length restraints.

        For Gaussian distribution: NLL = -log(P(x|μ,σ))
        NLL = 0.5 * ((x - μ) / σ)^2 + log(σ) + 0.5 * log(2π)

        This is the true NLL where exp(-NLL) = probability density.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_bonds,) with negative log-likelihood values.
        """
        from torchref.refinement.targets import gaussian_nll

        deviations, sigmas = self.bond_deviations()
        return gaussian_nll(deviations, sigmas)

    def angles(self, idx):
        """
        Compute current angle values for all angle restraints.

        Parameters
        ----------
        idx : torch.Tensor
            Angle indices tensor of shape (N, 3).

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_angles,) with current angle values in degrees.
        """
        xyz = self.model.xyz()
        pos1 = xyz[idx[:, 0], :]
        pos2 = xyz[idx[:, 1], :]
        pos3 = xyz[idx[:, 2], :]

        # Compute vectors
        v1 = pos1 - pos2  # Vector from atom2 to atom1
        v2 = pos3 - pos2  # Vector from atom2 to atom3

        # Compute angle using dot product
        # cos(θ) = (v1 · v2) / (|v1| * |v2|)
        dot_product = torch.sum(v1 * v2, dim=-1)
        norm1 = torch.linalg.norm(v1, dim=-1)
        norm2 = torch.linalg.norm(v2, dim=-1)

        # Clamp to avoid numerical issues with arccos
        cos_angle = torch.clamp(dot_product / (norm1 * norm2), -1.0, 1.0)

        # Return angle in degrees
        angles_rad = torch.acos(cos_angle)
        angles_deg = torch.rad2deg(angles_rad)

        return angles_deg

    def angle_deviations(self):
        """
        Compute angle deviations and sigmas.

        Returns
        -------
        deviations : torch.Tensor
            Calculated minus expected angles in radians.
        sigmas : torch.Tensor
            Standard deviations in radians.
        """
        if "all" not in self.restraints["angle"]:
            self.cat_dict()

        idx = self.restraints["angle"]["all"]["indices"]
        references_rad = self.restraints["angle"]["all"]["references"] * (
            torch.pi / 180.0
        )
        sigmas_rad = self.restraints["angle"]["all"]["sigmas"] * (torch.pi / 180.0)

        calculated_rad = self.angles(idx) * (torch.pi / 180.0)
        deviations = calculated_rad - references_rad

        return deviations, sigmas_rad

    def nll_angles(self):
        """
        Compute negative log-likelihood for angle restraints.

        For Gaussian distribution: NLL = -log(P(x|μ,σ))
        NLL = 0.5 * ((x - μ) / σ)^2 + log(σ) + 0.5 * log(2π)

        This is the true NLL where exp(-NLL) = probability density.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_angles,) with negative log-likelihood values.
        """
        from torchref.refinement.targets import gaussian_nll

        deviations, sigmas = self.angle_deviations()
        return gaussian_nll(deviations, sigmas)

    def cat_dict(self):
        """
        Concatenate all restraint dictionaries into 'all' keys.

        Creates restraints['bond']['all'], restraints['angle']['all'],
        and restraints['torsion']['all'] by concatenating all origins.
        """
        self.restraints["bond"]["all"] = {
            "indices": self._get_all_indices("bond"),
            "references": self._get_all_property("bond", "references"),
            "sigmas": self._get_all_property("bond", "sigmas"),
        }
        self.restraints["angle"]["all"] = {
            "indices": self._get_all_indices("angle"),
            "references": self._get_all_property("angle", "references"),
            "sigmas": self._get_all_property("angle", "sigmas"),
        }
        self.restraints["torsion"]["all"] = {
            "indices": self._get_all_indices("torsion", ["intra", "disulfide"]),
            "references": self._get_all_property(
                "torsion", "references", ["intra", "disulfide"]
            ),
            "sigmas": self._get_all_property(
                "torsion", "sigmas", ["intra", "disulfide"]
            ),
            "periods": self._get_all_property(
                "torsion", "periods", ["intra", "disulfide"]
            ),
        }

    def torsions(self, idx):
        """
        Compute current torsion angle values for all torsion restraints.

        Parameters
        ----------
        idx : torch.Tensor
            Torsion indices tensor of shape (N, 4).

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_torsions,) with current torsion values in degrees.
        """
        xyz = self.model.xyz()

        pos1 = xyz[idx[:, 0], :]
        pos2 = xyz[idx[:, 1], :]
        pos3 = xyz[idx[:, 2], :]
        pos4 = xyz[idx[:, 3], :]

        # Compute torsion angles using vector math
        b1 = pos2 - pos1
        b2 = pos3 - pos2
        b3 = pos4 - pos3

        # Normalize b2 for projection
        b2_norm = torch.linalg.norm(b2, dim=-1, keepdim=True)
        b2_unit = b2 / b2_norm

        # Compute normals to planes
        n1 = torch.cross(b1, b2, dim=-1)
        n2 = torch.cross(b2, b3, dim=-1)

        # Normalize normals
        n1_unit = n1 / torch.linalg.norm(n1, dim=-1, keepdim=True)
        n2_unit = n2 / torch.linalg.norm(n2, dim=-1, keepdim=True)

        # Compute angle between normals
        m1 = torch.cross(n1_unit, b2_unit, dim=-1)

        x = torch.sum(n1_unit * n2_unit, dim=-1)
        y = torch.sum(m1 * n2_unit, dim=-1)

        torsions_rad = torch.atan2(y, x)
        torsions_deg = torch.rad2deg(torsions_rad)
        return torsions_deg

    def _wrap_torsion_periodicity(self, diff_rad, periods):
        """
        Find minimum angular deviation considering n-fold rotational symmetry.

        For period=n, angles differing by 360°/n are equivalent. This function
        finds the equivalent angle with the smallest absolute deviation.

        Parameters
        ----------
        diff_rad : torch.Tensor
            Tensor of angular deviations in radians (any shape).
        periods : torch.Tensor
            Tensor of periodicity values (same shape as diff_rad).
            Period=0 or 1 means no symmetry (simple wrapping).
            Period=n means n-fold rotational symmetry.

        Returns
        -------
        torch.Tensor
            Tensor of minimum wrapped deviations in radians (same shape as input).
            Values are wrapped to [-π, π] and account for rotational symmetry.

        Examples
        --------
        For period=6 (e.g., benzene), angles of 10°, 70°, 130°, 190°, 250°, 310°
        are all equivalent. The function returns the one closest to 0°.
        """
        # Clamp periods to minimum of 1 to avoid division by zero
        periods_safe = torch.clamp(periods, min=1)
        max_period = periods_safe.max().item()

        if max_period > 1:
            # Vectorized approach: generate all equivalent angles
            device = diff_rad.device
            original_shape = diff_rad.shape

            # Flatten input for processing
            diff_rad_flat = diff_rad.flatten()
            periods_flat = periods_safe.flatten()
            n_angles = len(diff_rad_flat)

            # Create offset matrix: k * (2π / period) for k in [0, 1, ..., period-1]
            # Shape: (n_angles, max_period)
            k_range = torch.arange(max_period, device=device).unsqueeze(
                0
            )  # (1, max_period)
            periods_expanded = periods_flat.unsqueeze(1).float()  # (n_angles, 1)

            # Offsets for each angle: k * 2π/period
            offsets = k_range * (
                2.0 * torch.pi / periods_expanded
            )  # (n_angles, max_period)

            # Apply offsets to differences: (n_angles, max_period)
            diff_rad_expanded = diff_rad_flat.unsqueeze(1)  # (n_angles, 1)
            equiv_diffs = diff_rad_expanded - offsets  # (n_angles, max_period)

            # Wrap all equivalent angles to [-pi, pi]
            equiv_diffs_wrapped = torch.atan2(
                torch.sin(equiv_diffs), torch.cos(equiv_diffs)
            )

            # Mask out invalid offsets (where k >= period for each angle)
            valid_mask = k_range < periods_expanded  # (n_angles, max_period)

            # Set invalid positions to large value so they won't be selected
            equiv_diffs_wrapped_masked = torch.where(
                valid_mask,
                torch.abs(equiv_diffs_wrapped),
                torch.tensor(float("inf"), device=device),
            )

            # Find minimum absolute difference for each angle
            min_indices = torch.argmin(equiv_diffs_wrapped_masked, dim=1)  # (n_angles,)

            # Gather the best wrapped difference for each angle
            diff_wrapped_best = equiv_diffs_wrapped[
                torch.arange(n_angles, device=device), min_indices
            ]

            # Reshape back to original shape
            return diff_wrapped_best.reshape(original_shape)
        else:
            # All periods are 0 or 1, simple wrapping
            return torch.atan2(torch.sin(diff_rad), torch.cos(diff_rad))

    def torsion_deviations(self, wrapped=True):
        """
        Compute deviations between calculated and expected torsion angles.

        Parameters
        ----------
        wrapped : bool, default True
            If True, wrap deviations accounting for periodicity.
            If False, return raw deviations (calculated - expected).

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_torsions,) with deviations in degrees.
            For wrapped=True, deviations are in range appropriate for the period.

        Notes
        -----
        Expected values from CIF library are discrete (typically -60°, 0°, 60°, 90°, 180°)
        while calculated values from structure are continuous. This is correct!
        Use wrapped=True for meaningful comparison and visualization.
        """
        if "all" not in self.restraints["torsion"]:
            self.cat_dict()

        idx = self.restraints["torsion"]["all"]["indices"]
        expected = self.restraints["torsion"]["all"]["references"]
        periods = self.restraints["torsion"]["all"]["periods"]
        calculated = self.torsions(idx)

        if not wrapped:
            # Simple difference
            return calculated - expected
        else:
            # Use the helper function for periodicity handling
            diff_rad = (calculated - expected) * torch.pi / 180.0
            diff_wrapped_rad = self._wrap_torsion_periodicity(diff_rad, periods)

            # Convert back to degrees
            return torch.rad2deg(diff_wrapped_rad)

    def torsion_deviations_with_sigmas(self):
        """
        Compute torsion deviations (wrapped for periodicity) and sigmas.

        Returns
        -------
        deviations_rad : torch.Tensor
            Wrapped deviations in radians.
        sigmas_deg : torch.Tensor
            Standard deviations in degrees (for von Mises NLL).
        """
        if "all" not in self.restraints["torsion"]:
            self.cat_dict()

        idx = self.restraints["torsion"]["all"]["indices"]
        expected = self.restraints["torsion"]["all"]["references"]
        sigmas_deg = self.restraints["torsion"]["all"]["sigmas"]
        periods = self.restraints["torsion"]["all"]["periods"]

        calculated = self.torsions(idx)

        # Wrap for periodicity
        diff_rad = (calculated - expected) * (torch.pi / 180.0)
        deviations_rad = self._wrap_torsion_periodicity(diff_rad, periods)

        return deviations_rad, sigmas_deg

    def nll_torsions(self):
        """
        Compute negative log-likelihood for torsion angle restraints.

        For von Mises distribution: NLL = -log(P(θ|μ,κ))
        NLL = -κ*cos(θ-μ) + log(I₀(κ)) + log(2π)

        where κ = 1/σ² is the concentration parameter and I₀ is the modified
        Bessel function of the first kind.

        Notes
        -----
        Period indicates n-fold rotational symmetry (e.g., period=6 for benzene).
        We handle this by finding the minimum angular distance considering periodicity.
        For period=n, angles differing by 360°/n are equivalent.

        This is the true NLL where exp(-NLL) = probability density.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_torsions,) with negative log-likelihood values.
        """
        from torchref.refinement.targets import von_mises_nll

        deviations_rad, sigmas_deg = self.torsion_deviations_with_sigmas()
        return von_mises_nll(deviations_rad, sigmas_deg)

    def nll_planes(self):
        """
        Compute negative log-likelihood for plane restraints.

        For each plane, computes the RMSD of atom deviations from the best-fit plane.
        Uses Gaussian NLL: NLL = 0.5 * (deviation / σ)² + log(σ) + 0.5 * log(2π)

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_planes,) with negative log-likelihood values.
        """
        from torchref.refinement.targets import gaussian_nll

        xyz = self.model.xyz()
        device = xyz.device

        all_nlls = []

        if "plane" in self.restraints:
            for key, plane_data in self.restraints["plane"].items():
                indices = plane_data.get("indices")
                sigmas = plane_data.get("sigmas")

                if indices is None or len(indices) == 0:
                    continue

                # indices shape: (n_planes, n_atoms_per_plane)
                # sigmas shape: (n_planes, n_atoms_per_plane)
                n_planes, n_atoms = indices.shape

                for i in range(n_planes):
                    plane_indices = indices[i]
                    plane_sigmas = sigmas[i]

                    # Get positions of atoms in this plane
                    positions = xyz[plane_indices]  # (n_atoms, 3)

                    # Compute centroid
                    centroid = positions.mean(dim=0)
                    centered = positions - centroid

                    # SVD to find best-fit plane normal
                    # The plane normal is the singular vector with smallest singular value
                    U, S, Vh = torch.linalg.svd(centered)
                    normal = Vh[-1]  # Normal to best-fit plane

                    # Compute deviations from plane (distance to plane)
                    deviations = torch.abs(centered @ normal)

                    # Compute NLL for each atom
                    nll = gaussian_nll(deviations, plane_sigmas)
                    all_nlls.append(nll)

        if all_nlls:
            return torch.cat(all_nlls)
        return torch.tensor([0.0], device=device)

    def nll_vdw(self):
        """
        Compute negative log-likelihood for VDW (non-bonded) restraints.

        Uses a soft-repulsive potential based on distance violations.
        NLL = 0.5 * (max(0, min_dist - actual_dist) / σ)² + log(σ) + 0.5 * log(2π)

        Only violations (distances shorter than minimum) contribute to the loss.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_pairs,) with negative log-likelihood values.
        """
        from torchref.refinement.targets import gaussian_nll

        xyz = self.model.xyz()
        device = xyz.device

        if "vdw" not in self.restraints:
            return torch.tensor([0.0], device=device)

        vdw_data = self.restraints["vdw"]
        indices = vdw_data.get("indices")

        if indices is None or len(indices) == 0:
            return torch.tensor([0.0], device=device)

        min_distances = vdw_data["min_distances"]
        sigmas = vdw_data["sigmas"]

        # Get current positions
        pos1 = xyz[indices[:, 0]]
        pos2 = xyz[indices[:, 1]]

        # Compute actual distances
        actual_distances = torch.norm(pos2 - pos1, dim=-1)

        # Violations: where actual distance is less than minimum
        # Deviation = max(0, min_dist - actual_dist)
        deviations = torch.clamp(min_distances - actual_distances, min=0.0)

        # Compute NLL (only non-zero for violations)
        nll = gaussian_nll(deviations, sigmas)

        return nll

    def adp_b_differences(self):
        """
        Compute B-factor differences between bonded atoms.

        Returns
        -------
        torch.Tensor
            Tensor of B-factor differences (B_i - B_j) for all bonds.
        """
        b_factors = self.model.b()

        diffs_list = []
        if "bond" in self.restraints:
            for origin, restraint_group in self.restraints["bond"].items():
                if origin == "all":
                    continue
                indices = restraint_group.get("indices")
                if indices is not None and len(indices) > 0:
                    b1 = b_factors[indices[:, 0]]
                    b2 = b_factors[indices[:, 1]]
                    diffs_list.append(b1 - b2)

        if diffs_list:
            return torch.cat(diffs_list, dim=0)
        return torch.tensor([], device=b_factors.device)

    def adp_similarity_loss(self, sigma: float = 2.0):
        """
        Compute ADP similarity loss (SIMU in Phenix/SHELX).

        This restrains the B-factors of bonded atoms to be similar.
        Loss = Σ ((B_i - B_j) / sigma)^2

        Parameters
        ----------
        sigma : float, default 2.0
            Target standard deviation for B-factor differences in Å².

        Returns
        -------
        torch.Tensor
            Mean similarity loss.
        """
        from torchref.refinement.targets import adp_similarity_nll

        b_diffs = self.adp_b_differences()
        if len(b_diffs) == 0:
            return torch.tensor(0.0, device=self.model.xyz().device)
        return adp_similarity_nll(b_diffs, sigma).mean()
