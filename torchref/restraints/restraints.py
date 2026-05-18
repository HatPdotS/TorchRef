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
- Decoupled from Model: accepts pdb DataFrame and callable functions for xyz/adp
"""

from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch.nn import Module

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
from torchref.utils.utils import TensorDict
from torchref.utils.device_mixin import DeviceMixin


class _RestraintsAccessor:
    """
    Provides backward-compatible dict-like access to restraints stored in TensorDict.

    This class mimics the old nested dict interface:
        restraints["bond"]["intra"]["indices"]

    While actually accessing the TensorDict with flattened keys:
        _tensor_storage["bond_intra_indices"]
    """

    # Types that don't have origin level (assigned directly as dicts)
    _FLAT_TYPES = {"vdw", "chiral"}

    def __init__(self, parent: "RestraintsNew"):
        self._parent = parent

    def __getitem__(self, rtype: str) -> "_RestraintTypeAccessor":
        return _RestraintTypeAccessor(self._parent, rtype)

    def __setitem__(self, rtype: str, value):
        """Handle direct assignment for flat types like vdw and chiral."""
        if rtype in self._FLAT_TYPES and isinstance(value, dict):
            # Store all tensors with empty origin
            self._parent._set_restraint_group(rtype, "", value)
        else:
            raise TypeError(
                f"Cannot assign directly to restraints['{rtype}']. "
                f"Use restraints['{rtype}'][origin] = data for nested types."
            )

    def __contains__(self, rtype: str) -> bool:
        return len(self._parent._restraint_groups.get(rtype, set())) > 0 or \
               rtype in self._FLAT_TYPES and self._parent._has_restraint(rtype, "")

    def get(self, rtype: str, default=None):
        if rtype in self:
            return self[rtype]
        return default

    def keys(self):
        """Return all restraint types that have data."""
        result = []
        for rtype in ["bond", "angle", "torsion", "plane"]:
            if len(self._parent._restraint_groups.get(rtype, set())) > 0:
                result.append(rtype)
        # Check for special types (vdw, chiral) which don't have origins
        for rtype in self._FLAT_TYPES:
            if self._parent._has_restraint(rtype, ""):
                result.append(rtype)
        return result


class _RestraintTypeAccessor:
    """
    Provides access to origins within a restraint type.

    For regular types (bond, angle, torsion, plane):
        restraints["bond"]["intra"] -> dict with indices, references, sigmas

    For special types (vdw, chiral), this class acts as the dict itself:
        restraints["vdw"]["indices"] -> tensor
        restraints["vdw"] = {"indices": ..., "sigmas": ...}
    """

    # Types that don't have origin level (accessed directly as dicts)
    _FLAT_TYPES = {"vdw", "chiral"}

    def __init__(self, parent: "RestraintsNew", rtype: str):
        self._parent = parent
        self._rtype = rtype

    def __getitem__(self, key: str):
        if self._rtype in self._FLAT_TYPES:
            # For vdw/chiral, key is a property name (indices, sigmas, etc.)
            tensor = self._parent._get_restraint_tensor(self._rtype, "", key)
            if tensor is None:
                raise KeyError(f"No {key} for {self._rtype}")
            return tensor
        else:
            # For bond/angle/torsion/plane, key is an origin name
            result = self._parent._get_restraint_group(self._rtype, key)
            if result is None:
                raise KeyError(f"No restraints for {self._rtype}/{key}")
            return result

    def __setitem__(self, key: str, value):
        if self._rtype in self._FLAT_TYPES:
            # For vdw/chiral, if value is a tensor, store it directly
            # If value is a dict, store all tensors
            if isinstance(value, torch.Tensor):
                self._parent._set_restraint_tensor(self._rtype, "", key, value)
            elif isinstance(value, dict):
                # This handles: restraints["vdw"] = {"indices": ..., "sigmas": ...}
                # But this is called as restraints["vdw"][key] = value, so it won't work
                # We need special handling in the parent accessor
                pass
        else:
            # For bond/angle/torsion/plane, key is origin, value is dict
            self._parent._set_restraint_group(self._rtype, key, value)

    def __contains__(self, key: str) -> bool:
        if self._rtype in self._FLAT_TYPES:
            return self._parent._get_restraint_tensor(self._rtype, "", key) is not None
        return self._parent._has_restraint(self._rtype, key)

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        if self._rtype in self._FLAT_TYPES:
            # Return property names for flat types
            result = []
            for prop in ["indices", "references", "sigmas", "periods", "min_distances",
                         "symop_indices", "cell_offsets"]:
                if self._parent._get_restraint_tensor(self._rtype, "", prop) is not None:
                    result.append(prop)
            return result
        return self._parent._get_origins_for_type(self._rtype)

    def items(self):
        if self._rtype in self._FLAT_TYPES:
            for prop in self.keys():
                yield prop, self._parent._get_restraint_tensor(self._rtype, "", prop)
        else:
            for origin in self.keys():
                yield origin, self._parent._get_restraint_group(self._rtype, origin)

    def __iter__(self):
        return iter(self.keys())


class RestraintsNew(DeviceMixin, DebugMixin, Module):
    """
    Refactored restraints handler for crystallographic model refinement.

    This class uses the builder pattern internally for efficient construction
    of restraint tensors. It is decoupled from Model and accepts a pdb DataFrame
    with callable functions for accessing coordinates and ADPs.

    Parameters
    ----------
    pdb : pd.DataFrame, optional
        DataFrame containing atomic structure data. If None, creates empty shell.
    cif_path : str or list of str, optional
        Path to the CIF restraints dictionary file(s).
    xyz_fn : callable, optional
        Function returning current xyz coordinates as torch.Tensor.
        Required for building and evaluation if pdb is provided.
    adp_fn : callable, optional
        Function returning current ADP values as torch.Tensor.
        Required for ADP-based restraints.
    vdw_radii_fn : callable, optional
        Function returning VDW radii as torch.Tensor.
        Required for VDW restraints.
    verbose : int, default 1
        Verbosity level (0=silent, 1=normal, 2=detailed).

    Attributes
    ----------
    pdb : pd.DataFrame
        DataFrame containing atomic structure data.
    xyz_fn : callable
        Function returning current xyz coordinates.
    adp_fn : callable
        Function returning current ADP values.
    vdw_radii_fn : callable
        Function returning VDW radii.
    cif_dict : dict
        Parsed CIF dictionary with restraints for each residue type.
    restraints : dict
        Hierarchical dictionary containing all restraints.
    """

    def __init__(
        self,
        pdb: pd.DataFrame = None,
        cif_path=None,
        xyz_fn: Callable[[], torch.Tensor] = None,
        adp_fn: Callable[[], torch.Tensor] = None,
        vdw_radii_fn: Callable[[], torch.Tensor] = None,
        cell=None,
        spacegroup=None,
        links: pd.DataFrame = None,
        verbose: int = 1,
    ):
        """Initialize the Restraints handler."""
        super().__init__()
        self.cif_path = cif_path
        self.verbose = verbose
        self.links = links

        # Store callable functions for coordinate/ADP access
        self._xyz_fn = xyz_fn
        self._adp_fn = adp_fn
        self._vdw_radii_fn = vdw_radii_fn

        # Store crystallographic info for symmetry VDW restraints
        self._cell = cell
        self._spacegroup = spacegroup

        # Initialize TensorDict for restraint storage (registered as submodule)
        self._tensor_storage = TensorDict()

        # Track which restraint groups exist (for iteration)
        self._restraint_groups = {"bond": set(), "angle": set(), "torsion": set(), "plane": set()}

        # Empty initialization
        if pdb is None:
            self.pdb = None
            self.cif_dict = {}
            self.unique_residues = []
            return

        # Full initialization with pdb
        self.pdb = pdb
        self.unique_residues = pdb.resname.unique()
        self.unique_residues = [
            residue
            for residue in self.unique_residues
            if self.pdb.loc[self.pdb["resname"] == residue, "name"].nunique() > 1
        ]

        # Parse CIF files
        self._load_cif_dictionaries(cif_path)

        # Load link definitions for inter-residue restraints
        if verbose > 1:
            print("Loading link definitions from monomer library...")
        self.link_dict, self.link_list = read_link_definitions()
        if verbose > 1:
            print(f"Loaded {len(self.link_dict)} link types")

        # Build restraints using the new builder pattern
        self.build_restraints()
        if self.verbose > 0:
            self.summary()

    def xyz(self, xyz: torch.Tensor = None) -> torch.Tensor:
        """
        Get current xyz coordinates.

        Parameters
        ----------
        xyz : torch.Tensor, optional
            If provided, returns this tensor directly.
            Otherwise calls the stored xyz_fn callable.

        Returns
        -------
        torch.Tensor
            Current xyz coordinates of shape (n_atoms, 3).
        """
        if xyz is not None:
            return xyz
        if self._xyz_fn is None:
            raise RuntimeError(
                "No xyz callable provided. Initialize with xyz_fn or pass xyz argument."
            )
        return self._xyz_fn()

    def adp(self, adp: torch.Tensor = None) -> torch.Tensor:
        """
        Get current ADP values.

        Parameters
        ----------
        adp : torch.Tensor, optional
            If provided, returns this tensor directly.
            Otherwise calls the stored adp_fn callable.

        Returns
        -------
        torch.Tensor
            Current ADP values of shape (n_atoms,).
        """
        if adp is not None:
            return adp
        if self._adp_fn is None:
            raise RuntimeError(
                "No adp callable provided. Initialize with adp_fn or pass adp argument."
            )
        return self._adp_fn()

    def get_vdw_radii(self) -> torch.Tensor:
        """
        Get VDW radii for all atoms.

        Returns
        -------
        torch.Tensor
            VDW radii of shape (n_atoms,).
        """
        if self._vdw_radii_fn is None:
            raise RuntimeError(
                "No vdw_radii callable provided. Initialize with vdw_radii_fn."
            )
        return self._vdw_radii_fn()

    # =========================================================================
    # TensorDict Helper Methods for Restraint Storage
    # =========================================================================

    def _make_key(self, rtype: str, origin: str, prop: str) -> str:
        """Create flattened key for TensorDict storage."""
        if origin:
            return f"{rtype}_{origin}_{prop}"
        else:
            # For flat types (vdw, chiral) with no origin
            return f"{rtype}_{prop}"

    def _set_restraint_tensor(
        self, rtype: str, origin: str, prop: str, tensor: torch.Tensor
    ):
        """Store a restraint tensor with flattened key."""
        key = self._make_key(rtype, origin, prop)
        self._tensor_storage[key] = tensor
        # Track that this origin exists for this restraint type
        if rtype in self._restraint_groups:
            self._restraint_groups[rtype].add(origin)

    def _get_restraint_tensor(
        self, rtype: str, origin: str, prop: str
    ) -> Optional[torch.Tensor]:
        """Get a restraint tensor by type, origin, and property."""
        key = self._make_key(rtype, origin, prop)
        if key in self._tensor_storage:
            return self._tensor_storage[key]
        return None

    def _has_restraint(self, rtype: str, origin: str) -> bool:
        """Check if a restraint group exists."""
        key = self._make_key(rtype, origin, "indices")
        return key in self._tensor_storage

    def _set_restraint_group(self, rtype: str, origin: str, data: dict):
        """Store all tensors from a restraint data dict."""
        for prop, tensor in data.items():
            if tensor is not None and isinstance(tensor, torch.Tensor):
                self._set_restraint_tensor(rtype, origin, prop, tensor)

    def _get_restraint_group(self, rtype: str, origin: str) -> Optional[dict]:
        """Get all tensors for a restraint group as a dict."""
        if not self._has_restraint(rtype, origin):
            return None
        result = {}
        # Common properties for different restraint types
        for prop in ["indices", "references", "sigmas", "periods", "min_distances",
                     "is_proline"]:
            tensor = self._get_restraint_tensor(rtype, origin, prop)
            if tensor is not None:
                result[prop] = tensor
        return result if result else None

    def _get_origins_for_type(self, rtype: str) -> list:
        """Get all origins (e.g., 'intra', 'peptide') for a restraint type."""
        return list(self._restraint_groups.get(rtype, set()))

    @property
    def restraints(self) -> "_RestraintsAccessor":
        """
        Provide dict-like access to restraints for backward compatibility.

        Returns an accessor object that mimics the old nested dict interface.
        """
        return _RestraintsAccessor(self)

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

    def _load_rama_surfaces(self, device: torch.device):
        """Load pre-computed Ramachandran NLL surfaces as a buffer."""
        from torchref.restraints.ramachandran import load_nll_surfaces

        surfaces = load_nll_surfaces(device)
        self.register_buffer("_rama_surfaces", surfaces)

    def build_restraints(self):
        """
        Build all restraints using the fast builder API.

        This method uses the optimized builders that handle all residues
        internally with Numba-accelerated matching (~10x faster).
        """
        try:
            device = self.xyz().device
            pdb = self.pdb

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
            self._build_link_restraints(device)

            # Build VDW restraints. Cutoff is held ~1 Å wider than the
            # maximum heavy-atom VDW sum (~3.6 Å) plus the expected inter-
            # build drift, so the maintenance-triggered rebuild can be
            # driven by a displacement threshold well inside the cutoff
            # margin without missing newly-formed contacts.
            self._build_vdw_restraints(
                cutoff=6.0, sigma=0.05, inter_residue_only=False, use_spatial_hash=True
            )

            # Pre-compute concatenated 'all' groups so every buffer is registered
            # on the correct device at build time.  This prevents register_buffer()
            # being called during a forward pass (which would break CUDA-graph
            # capture) and ensures model.to(device) moves ALL restraint tensors.
            self.cat_dict()

        except Exception as e:
            self.debug_on_error(e, context="RestraintsNew.build_restraints")
            raise

    def _build_peptide_restraints(self, device: torch.device):
        """Build peptide bond restraints using fast inter-residue builders.

        Uses TRANS/CIS links for standard peptide bonds and PTRANS/PCIS
        links for peptide bonds to proline.  The proline-specific links
        include the C(i-1)-N-CD angle that constrains the pyrrolidine
        ring orientation, and use proline-specific angle target values.
        """
        if "TRANS" not in self.link_dict:
            if self.verbose > 0:
                print(
                    "Warning: TRANS link not found in link dictionary, skipping peptide bonds"
                )
            return

        trans_link = self.link_dict["TRANS"]
        ptrans_link = self.link_dict.get("PTRANS")
        pdb = self.pdb

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

        # Build peptide angles.
        # If PTRANS is available, use it for proline pairs (excludes PRO
        # from TRANS to avoid duplicate/conflicting restraints) and TRANS
        # for non-proline pairs.  Otherwise fall back to TRANS for all.
        angle_builder = InterResidueAngleBuilder(verbose=self.verbose)
        if ptrans_link is not None:
            # Non-proline pairs: TRANS angles
            angle_result = angle_builder.build(
                pdb, trans_link, device, filter_atom_type="ATOM",
                exclude_next_resname="PRO",
            )
            # Proline pairs: PTRANS angles (includes C-N-CD)
            pro_angle_result = angle_builder.build(
                pdb, ptrans_link, device, filter_atom_type="ATOM",
                next_resname_filter="PRO",
            )
            # Merge results
            if angle_result and pro_angle_result:
                angle_result = {
                    "indices": torch.cat([angle_result["indices"], pro_angle_result["indices"]]),
                    "references": torch.cat([angle_result["references"], pro_angle_result["references"]]),
                    "sigmas": torch.cat([angle_result["sigmas"], pro_angle_result["sigmas"]]),
                }
            elif pro_angle_result:
                angle_result = pro_angle_result
        else:
            angle_result = angle_builder.build(
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
            if "ramachandran" in torsion_result:
                rama = torsion_result["ramachandran"]
                self.register_buffer("_rama_phi_indices", rama["phi_indices"])
                self.register_buffer("_rama_psi_indices", rama["psi_indices"])
                self.register_buffer("_rama_surface_type", rama["surface_type"])
                self._load_rama_surfaces(device)

        # Build peptide planes
        plane_result = InterResiduePlaneBuilder(verbose=self.verbose).build(
            pdb, trans_link, device, filter_atom_type="ATOM"
        )
        if plane_result:
            n_planes = 0
            for key, data in plane_result.items():
                n_planes += data["indices"].shape[0]
                if self._has_restraint("plane", key):
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
        pdb = self.pdb
        sg_atoms = pdb[(pdb["name"] == "SG") & (pdb["ATOM"] == "ATOM")]

        if len(sg_atoms) == 0:
            return

        # Get coordinates and find close pairs
        xyz = self.xyz()
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

    def _build_link_restraints(self, device: torch.device):
        """Build bond restraints from PDB LINK records.

        Each accepted LINK contributes one bond restraint between the two
        named atoms. The bond automatically becomes part of the VDW
        exclusion set (via ``_build_exclusion_set``), preventing the
        non-bonded term from pushing the linked atoms apart.

        Behaviour:
        - Distance/sigma source: ``length`` from the LINK record is used as
          target distance with sigma=0.02 Å. If the field was blank we fall
          back to a generic 1.5 Å bond.
        - Symmetry-mate links are filtered out earlier in
          ``extract_link_records``.
        - LINKs that duplicate an auto-detected disulfide (CYS SG-SG pair)
          are skipped, because the disulfide builder has already added a
          bond + angles + torsions for that pair.
        """
        if self.links is None or len(self.links) == 0:
            return

        pdb = self.pdb

        # Already-bonded SG-SG pairs from auto-disulfide detection.
        disulf = self.restraints.get("bond", {}).get("disulfide")
        existing_disulf_pairs = set()
        if disulf is not None and "indices" in disulf:
            for i, j in disulf["indices"].cpu().numpy():
                existing_disulf_pairs.add((int(min(i, j)), int(max(i, j))))

        bond_builder = InterResidueBondBuilder(verbose=self.verbose)
        n_skipped_unresolved = 0
        n_skipped_dedup = 0

        for _, link in self.links.iterrows():
            idx1 = self._lookup_link_atom(
                pdb,
                chainid=link["chainid1"],
                resseq=int(link["resseq1"]),
                icode=link["icode1"],
                resname=link["resname1"],
                name=link["name1"],
                altloc=link["altloc1"],
            )
            idx2 = self._lookup_link_atom(
                pdb,
                chainid=link["chainid2"],
                resseq=int(link["resseq2"]),
                icode=link["icode2"],
                resname=link["resname2"],
                name=link["name2"],
                altloc=link["altloc2"],
            )

            if idx1 is None or idx2 is None:
                n_skipped_unresolved += 1
                if self.verbose > 1:
                    print(
                        f"Warning: LINK atom not found "
                        f"({link['chainid1']}/{link['resname1']}{link['resseq1']}/"
                        f"{link['name1']} -- "
                        f"{link['chainid2']}/{link['resname2']}{link['resseq2']}/"
                        f"{link['name2']}); skipping."
                    )
                continue

            if idx1 == idx2:
                n_skipped_unresolved += 1
                continue

            pair = (min(idx1, idx2), max(idx1, idx2))
            if pair in existing_disulf_pairs:
                n_skipped_dedup += 1
                continue

            length = link["length"]
            if not (isinstance(length, (int, float)) and length == length and length > 0):
                length = 1.5
            bond_builder.process_disulfide_bond(idx1, idx2, float(length), 0.02)

        bond_result = bond_builder.finalize(device)
        if bond_result:
            self.restraints["bond"]["link"] = bond_result
            if self.verbose > 0:
                print(
                    f"Built {bond_result['indices'].shape[0]} LINK bond restraints"
                    + (
                        f" (skipped {n_skipped_dedup} disulfide-dup,"
                        f" {n_skipped_unresolved} unresolved)"
                        if (n_skipped_dedup or n_skipped_unresolved)
                        else ""
                    )
                )

    @staticmethod
    def _lookup_link_atom(
        pdb: pd.DataFrame,
        chainid: str,
        resseq: int,
        icode: str,
        resname: str,
        name: str,
        altloc: str,
    ):
        """Resolve a LINK atom record to a row index in the model pdb.

        Match on (chainid, resseq, icode, name); resname is used as a tie-
        breaker if present. Altloc preference: requested altloc first, then
        blank, then 'A', then any.
        """
        sel = pdb[
            (pdb["chainid"].astype(str) == str(chainid))
            & (pdb["resseq"].astype(int) == int(resseq))
            & (pdb["icode"].astype(str) == str(icode))
            & (pdb["name"].astype(str).str.strip() == str(name).strip())
        ]
        if len(sel) == 0:
            return None
        if resname:
            tied = sel[sel["resname"].astype(str).str.strip() == str(resname).strip()]
            if len(tied) > 0:
                sel = tied

        if altloc:
            for cand in (altloc, ""):
                hit = sel[sel["altloc"].astype(str) == cand]
                if len(hit) > 0:
                    return int(hit.iloc[0]["index"])
        for cand in ("", "A"):
            hit = sel[sel["altloc"].astype(str) == cand]
            if len(hit) > 0:
                return int(hit.iloc[0]["index"])
        return int(sel.iloc[0]["index"])

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
        Find all atom pairs within cutoff distance using spatial cell lists.

        Divides space into cubic cells of side length = cutoff and only checks
        atom pairs in the same or adjacent cells (14 unique offsets: self + 13
        forward neighbours).  This gives O(N) memory and O(N*k) time where k
        is the average number of neighbours, compared to O(N^2) for a full
        distance matrix.

        Parameters
        ----------
        xyz : torch.Tensor
            Atom coordinates of shape (N, 3).
        cutoff : float
            Distance cutoff in Angstroms.

        Returns
        -------
        torch.Tensor
            Pairs of atom indices, shape (M, 2), each row (i, j) with i < j.
        """
        device = xyz.device
        n_atoms = xyz.shape[0]

        if n_atoms == 0:
            return torch.tensor([], dtype=torch.long, device=device).reshape(0, 2)

        # Work on CPU to avoid per-iteration GPU kernel launch overhead
        coords = xyz.detach().cpu()
        cell_size = cutoff

        # Assign each atom to a cubic cell
        xyz_min = coords.min(dim=0).values
        cell_idx = ((coords - xyz_min) / cell_size).long()  # (N, 3)

        grid_dims = cell_idx.max(dim=0).values + 1
        gx, gy, gz = grid_dims[0].item(), grid_dims[1].item(), grid_dims[2].item()
        gyz = gy * gz

        # Flat cell index per atom
        flat = cell_idx[:, 0] * gyz + cell_idx[:, 1] * gz + cell_idx[:, 2]

        # Sort atoms by cell so each cell's atoms are contiguous
        order = flat.argsort()
        sorted_flat = flat[order]

        unique_cells, counts = torch.unique_consecutive(
            sorted_flat, return_counts=True
        )
        n_unique = len(unique_cells)
        starts = torch.zeros(n_unique + 1, dtype=torch.long)
        starts[1:] = counts.cumsum(0)

        # Lookup: flat_cell -> index in unique_cells (-1 if empty)
        n_grid = gx * gyz
        cell_lookup = torch.full((n_grid,), -1, dtype=torch.long)
        cell_lookup[unique_cells] = torch.arange(n_unique)

        # 14 unique neighbour offsets: self (0,0,0) + 13 forward neighbours.
        # "Forward" = first non-zero component is positive, avoiding double counting.
        offsets_list = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    if (
                        dx > 0
                        or (dx == 0 and dy > 0)
                        or (dx == 0 and dy == 0 and dz >= 0)
                    ):
                        offsets_list.append(
                            (dx, dy, dz, dx * gyz + dy * gz + dz)
                        )

        cutoff_sq = cutoff * cutoff
        pair_chunks = []

        # Move to numpy for tight loop (faster item access than torch on CPU)
        unique_np = unique_cells.numpy()
        starts_np = starts.numpy()
        order_np = order.numpy()
        coords_np = coords.numpy()

        for ci in range(n_unique):
            cell_flat = int(unique_np[ci])
            sa, ea = int(starts_np[ci]), int(starts_np[ci + 1])
            atoms_a = order_np[sa:ea]
            xyz_a = coords_np[atoms_a]  # (na, 3)

            cx = cell_flat // gyz
            cy = (cell_flat % gyz) // gz
            cz = cell_flat % gz

            for dx, dy, dz, off_flat in offsets_list:
                ncx, ncy, ncz = cx + dx, cy + dy, cz + dz
                if (
                    ncx < 0 or ncx >= gx
                    or ncy < 0 or ncy >= gy
                    or ncz < 0 or ncz >= gz
                ):
                    continue

                nb_flat = ncx * gyz + ncy * gz + ncz
                nb_ci = int(cell_lookup[nb_flat])
                if nb_ci < 0:
                    continue

                sb, eb = int(starts_np[nb_ci]), int(starts_np[nb_ci + 1])
                atoms_b = order_np[sb:eb]
                xyz_b = coords_np[atoms_b]  # (nb, 3)

                # Vectorised distance² via broadcasting: (na, nb, 3)
                diff = xyz_a[:, None, :] - xyz_b[None, :, :]
                dist_sq = (diff * diff).sum(axis=-1)  # (na, nb)

                if off_flat == 0:
                    # Self-cell: upper triangle only
                    na = len(atoms_a)
                    if na < 2:
                        continue
                    ii, jj = np.triu_indices(na, k=1)
                    mask = dist_sq[ii, jj] < cutoff_sq
                    if mask.any():
                        ai = atoms_a[ii[mask]]
                        aj = atoms_a[jj[mask]]
                        pairs = np.stack(
                            [np.minimum(ai, aj), np.maximum(ai, aj)], axis=1
                        )
                        pair_chunks.append(pairs)
                else:
                    # Inter-cell: all pairs
                    ii, jj = np.where(dist_sq < cutoff_sq)
                    if len(ii) > 0:
                        ai = atoms_a[ii]
                        bj = atoms_b[jj]
                        pairs = np.stack(
                            [np.minimum(ai, bj), np.maximum(ai, bj)], axis=1
                        )
                        pair_chunks.append(pairs)

        if pair_chunks:
            all_pairs = np.concatenate(pair_chunks, axis=0)
            return torch.from_numpy(all_pairs).to(dtype=torch.long, device=device)
        else:
            return torch.tensor([], dtype=torch.long, device=device).reshape(0, 2)

    def _expand_with_symmetry_mates(self, xyz, cutoff):
        """
        Expand ASU coordinates with symmetry mate positions for neighbor search.

        Generates Cartesian coordinates of symmetry-related copies that could
        potentially have contacts with the ASU, using centroid-based pre-filtering
        to skip distant mates.

        Parameters
        ----------
        xyz : torch.Tensor
            ASU Cartesian coordinates of shape (N, 3).
        cutoff : float
            Distance cutoff in Angstroms for contact search.

        Returns
        -------
        combined_xyz : torch.Tensor
            Concatenated coordinates (N_asu + N_mates, 3).
        provenance : dict
            Dictionary with arrays describing the origin of each atom:
            - 'asu_source_indices': (N_total,) int array, ASU atom index
            - 'symop_indices': (N_total,) int array, symmetry operation index
            - 'cell_offsets': (N_total, 3) int array, unit cell offset
        """
        from torchref.config import dtypes
        from torchref.symmetry import SpaceGroup

        cell = self._cell
        sg = self._spacegroup
        if not isinstance(sg, SpaceGroup):
            sg = SpaceGroup(sg)

        n_asu = xyz.shape[0]
        device = xyz.device
        fdtype = dtypes.float

        # Work on the model's device throughout
        xyz_det = xyz.detach().to(fdtype)
        xyz_frac = cell.cartesian_to_fractional(xyz_det)

        # Compute centroid and molecule radius for pre-filtering
        centroid_frac = xyz_frac.mean(dim=0)
        centroid_cart = xyz_det.mean(dim=0)
        molecule_radius = (xyz_det - centroid_cart).norm(dim=1).max().item()
        threshold = 2 * molecule_radius + cutoff

        B = cell.fractional_matrix.to(device=device, dtype=fdtype)
        I_mat = torch.eye(3, dtype=fdtype, device=device)

        # Phase 1: centroid pre-filter to find which (symop, offset) combos
        # can produce contacts.  This is a small loop over scalar ops.
        n_ops = sg.n_ops
        matrices = sg.matrices.to(device=device, dtype=fdtype)
        translations = sg.translations.to(device=device, dtype=fdtype)

        valid_ops = []  # list of (op_idx, dx, dy, dz)
        for op_idx in range(n_ops):
            R = matrices[op_idx]
            t = translations[op_idx]
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    for dz in range(-1, 2):
                        if op_idx == 0 and dx == 0 and dy == 0 and dz == 0:
                            continue
                        offset = torch.tensor([dx, dy, dz], dtype=fdtype,
                                              device=device)
                        d_frac = (R - I_mat) @ centroid_frac + t + offset
                        d_cart = B @ d_frac
                        if d_cart.norm().item() <= threshold:
                            valid_ops.append((op_idx, dx, dy, dz))

        if not valid_ops:
            provenance = {
                "asu_source_indices": np.arange(n_asu, dtype=np.int64),
                "symop_indices": np.zeros(n_asu, dtype=np.int64),
                "cell_offsets": np.zeros((n_asu, 3), dtype=np.int64),
            }
            if self.verbose > 0:
                print("  Symmetry expansion: 0 mate(s) within range "
                      f"({n_asu} total atoms for neighbor search)")
            return xyz_det, provenance

        # Phase 2: batch-generate all mate coordinates in one go
        n_valid = len(valid_ops)
        op_indices = [v[0] for v in valid_ops]
        cell_offs = torch.tensor(
            [[v[1], v[2], v[3]] for v in valid_ops], dtype=fdtype,
            device=device,
        )  # (n_valid, 3)

        # Gather rotation matrices and translations for valid ops
        R_batch = matrices[op_indices]          # (n_valid, 3, 3)
        t_batch = translations[op_indices]      # (n_valid, 3)

        # Batched transform: for each valid op, compute R @ xyz_frac.T + t + offset
        # xyz_frac: (N, 3), R_batch: (n_valid, 3, 3)
        # -> (n_valid, 3, N) via batched matmul, then transpose to (n_valid, N, 3)
        xyz_frac_T = xyz_frac.T.unsqueeze(0).expand(n_valid, -1, -1)  # (n_valid, 3, N)
        mate_frac_all = torch.bmm(R_batch, xyz_frac_T).permute(0, 2, 1)  # (n_valid, N, 3)
        mate_frac_all = mate_frac_all + t_batch.unsqueeze(1) + cell_offs.unsqueeze(1)

        # Convert all to Cartesian: (n_valid * N, 3)
        mate_frac_flat = mate_frac_all.reshape(-1, 3)
        mate_cart_flat = cell.fractional_to_cartesian(mate_frac_flat)

        # Build combined coordinate array: ASU + all mates
        combined_xyz = torch.cat(
            [xyz_det, mate_cart_flat], dim=0
        )

        # Build provenance arrays
        asu_source = np.arange(n_asu, dtype=np.int64)
        # ASU block
        all_asu_sources = [asu_source]
        all_symops = [np.zeros(n_asu, dtype=np.int64)]
        all_offsets = [np.zeros((n_asu, 3), dtype=np.int64)]
        # Mate blocks (each has n_asu atoms)
        for op_idx, dx, dy, dz in valid_ops:
            all_asu_sources.append(asu_source)
            all_symops.append(np.full(n_asu, op_idx, dtype=np.int64))
            all_offsets.append(np.tile([dx, dy, dz], (n_asu, 1)).astype(np.int64))

        provenance = {
            "asu_source_indices": np.concatenate(all_asu_sources),
            "symop_indices": np.concatenate(all_symops),
            "cell_offsets": np.concatenate(all_offsets),
        }

        if self.verbose > 0:
            print(f"  Symmetry expansion: {n_valid} mate(s) within range "
                  f"({combined_xyz.shape[0]} total atoms for neighbor search)")

        return combined_xyz, provenance

    @property
    def h_topo(self):
        """Access riding hydrogen topology (None if not built)."""
        return getattr(self, "_h_topo", None)

    @property
    def h_excl_hash(self):
        """Access H-specific exclusion hash tensor (None if not built)."""
        return getattr(self, "_h_excl_hash", None)

    def _build_h_exclusion_hash(self, h_topo, device):
        """Build sorted hash tensor for H-specific 1-2 and 1-3 exclusions.

        Exclusions are stored as ``min(i, j) * max_idx + max(i, j)`` hashes,
        sorted for O(log n) lookup via ``torch.searchsorted``.

        Parameters
        ----------
        h_topo : HydrogenTopology
        device : torch.device

        Returns
        -------
        torch.Tensor
            Sorted 1-D long tensor of exclusion hashes.
        """
        if h_topo is None or h_topo.n_hydrogens == 0:
            return torch.tensor([], dtype=torch.long, device=device)

        n_heavy = len(self.pdb)
        n_h = h_topo.n_hydrogens
        exclusions = set()

        parent_idx = h_topo.h_parent_idx.cpu().numpy()
        nb_idx = h_topo.parent_neighbor_idx.cpu().numpy()
        nb_count = h_topo.parent_neighbor_count.cpu().numpy()

        for hi in range(n_h):
            # H index in the combined array is n_heavy + hi
            h_combined = n_heavy + hi
            p = int(parent_idx[hi])

            # 1-2: H — parent
            exclusions.add((min(h_combined, p), max(h_combined, p)))

            # 1-3: H — parent's heavy neighbours
            for ni in range(int(nb_count[hi])):
                nb = int(nb_idx[hi, ni])
                if nb >= 0:
                    exclusions.add((min(h_combined, nb), max(h_combined, nb)))

        if not exclusions:
            return torch.tensor([], dtype=torch.long, device=device)

        arr = np.array(list(exclusions), dtype=np.int64)
        max_idx = max(n_heavy + n_h, int(arr.max()) + 1)
        hashes = arr[:, 0] * max_idx + arr[:, 1]
        hashes.sort()
        return torch.tensor(hashes, dtype=torch.long, device=device)

    def _build_vdw_restraints(
        self, cutoff=6.0, sigma=0.2, inter_residue_only=True, use_spatial_hash=True
    ):
        """Build van der Waals (non-bonded contact) restraints.

        When cell and spacegroup are available, also includes contacts between
        ASU atoms and symmetry-related copies in neighboring molecules.

        Uses GPU-native periodic grid search when crystallographic symmetry
        is available.  Falls back to the legacy spatial hash otherwise.

        Also builds the riding hydrogen topology for H-VDW evaluation.

        Caches the build kwargs and a detached snapshot of the ASU
        coordinates at build time in ``_vdw_build_kwargs`` and
        ``_last_vdw_build_xyz``. :meth:`rebuild_vdw_restraints` consults
        those to refresh the pair list with the same parameters, and
        ``NonBondedTarget.maintenance`` uses the snapshot to decide
        whether a rebuild is needed.
        """
        # Remember how we built so rebuild can call back with the same
        # parameters without re-plumbing them through every caller.
        self._vdw_build_kwargs = dict(
            cutoff=cutoff,
            sigma=sigma,
            inter_residue_only=inter_residue_only,
            use_spatial_hash=use_spatial_hash,
        )

        if self.verbose > 0:
            print("\nBuilding VDW (non-bonded) restraints...")

        has_symmetry = (
            self._cell is not None
            and self._spacegroup is not None
        )

        if has_symmetry:
            from torchref.restraints.neighbor_search import build_vdw_restraints_gpu

            exclusions = self._build_exclusion_set()
            self.restraints["vdw"] = build_vdw_restraints_gpu(
                xyz_fn=self.xyz,
                vdw_radii_fn=self.get_vdw_radii,
                cell=self._cell,
                sg=self._spacegroup,
                pdb=self.pdb,
                exclusion_set=exclusions,
                cutoff=cutoff,
                sigma=sigma,
                inter_residue_only=inter_residue_only,
                verbose=self.verbose,
            )
        else:
            self._build_vdw_restraints_legacy(
                cutoff=cutoff, sigma=sigma,
                inter_residue_only=inter_residue_only,
                use_spatial_hash=use_spatial_hash,
            )

        # Build riding hydrogen topology and precompute candidate pairs
        from torchref.restraints.hydrogen_topology import (
            build_hydrogen_topology,
            build_h_candidate_pairs,
        )

        device = self.xyz().device if self._xyz_fn is not None else torch.device("cpu")
        self._h_topo = build_hydrogen_topology(
            pdb=self.pdb,
            device=device,
            verbose=self.verbose,
        )
        self._h_excl_hash = self._build_h_exclusion_hash(self._h_topo, device)

        # Precompute H candidate pairs from heavy-atom VDW pair list
        vdw_data = self.restraints.get("vdw")
        if vdw_data is not None and self._h_topo.n_hydrogens > 0:
            build_h_candidate_pairs(
                h_topo=self._h_topo,
                vdw_data=vdw_data,
                pdb=self.pdb,
                h_excl_hash=self._h_excl_hash,
                device=device,
                verbose=self.verbose,
            )
            # Fill in VDW min distances using combined radii array
            if self._h_topo.has_candidates:
                heavy_radii = self.get_vdw_radii()         # (N_heavy,)
                h_radii = self._h_topo.h_vdw_radius        # (N_h,)
                all_radii = torch.cat([heavy_radii, h_radii])
                self._h_topo.cand_min_dist = (
                    all_radii[self._h_topo.cand_idx_i]
                    + all_radii[self._h_topo.cand_idx_j]
                )

        # Snapshot the ASU coordinates *at* build time so maintenance()
        # callers can diff current positions against it and decide if a
        # rebuild is needed. Detached clone lives on the same device as
        # the model, so the compare is a single GPU op.
        if self._xyz_fn is not None:
            self._last_vdw_build_xyz = self.xyz().detach().clone()

    def rebuild_vdw_restraints(self) -> None:
        """Refresh the VDW pair list using the cached build kwargs.

        Called by :meth:`NonBondedTarget.maintenance` after it detects
        that the maximum atomic displacement since the last build has
        exceeded the rebuild threshold. Uses the same ``cutoff``,
        ``sigma``, ``inter_residue_only`` and ``use_spatial_hash`` that
        the initial build was given, so behaviour is stable across the
        run.
        """
        if not hasattr(self, "_vdw_build_kwargs"):
            raise RuntimeError(
                "rebuild_vdw_restraints called before initial build "
                "— _vdw_build_kwargs is missing"
            )
        self._build_vdw_restraints(**self._vdw_build_kwargs)

    def _build_vdw_restraints_legacy(
        self, cutoff=5.0, sigma=0.2, inter_residue_only=True, use_spatial_hash=True
    ):
        """Legacy VDW restraint builder (no symmetry or CPU fallback)."""

        exclusions = self._build_exclusion_set()
        vdw_radii = self.get_vdw_radii()
        xyz = self.xyz()
        device = xyz.device
        pdb = self.pdb
        n_asu = xyz.shape[0]

        # Expand with symmetry mates if crystallographic info is available
        has_symmetry = (
            self._cell is not None
            and self._spacegroup is not None
        )
        if has_symmetry:
            combined_xyz, provenance = self._expand_with_symmetry_mates(xyz, cutoff)
        else:
            combined_xyz = xyz
            provenance = None

        # Find nearby pairs in the (potentially expanded) coordinate set
        if use_spatial_hash:
            nearby_pairs = self._find_nearby_pairs_spatial_hash(combined_xyz, cutoff)
        else:
            n_total = combined_xyz.shape[0]
            pairs_list = []
            cutoff_sq = cutoff**2
            for i in range(n_total):
                for j in range(i + 1, n_total):
                    dist_sq = ((combined_xyz[i] - combined_xyz[j]) ** 2).sum()
                    if dist_sq < cutoff_sq:
                        pairs_list.append([i, j])
            nearby_pairs = (
                torch.tensor(pairs_list, dtype=torch.long, device=device)
                if pairs_list
                else torch.tensor([], dtype=torch.long, device=device).reshape(0, 2)
            )

        empty_result = {
            "indices": torch.tensor([], dtype=torch.long, device=device).reshape(0, 2),
            "min_distances": torch.tensor([], dtype=torch.float32, device=device),
            "sigmas": torch.tensor([], dtype=torch.float32, device=device),
            "symop_indices": torch.tensor([], dtype=torch.long, device=device),
            "cell_offsets": torch.tensor([], dtype=torch.long, device=device).reshape(0, 3),
        }

        if len(nearby_pairs) == 0:
            self.restraints["vdw"] = empty_result
            return

        pairs_np = nearby_pairs.cpu().numpy()

        # Map indices through provenance to get ASU source atoms and symop info
        if provenance is not None:
            prov_asu = provenance["asu_source_indices"]
            prov_sym = provenance["symop_indices"]
            prov_off = provenance["cell_offsets"]

            # Get provenance for each atom in each pair
            idx0 = pairs_np[:, 0]
            idx1 = pairs_np[:, 1]

            asu_src_0 = prov_asu[idx0]
            asu_src_1 = prov_asu[idx1]
            sym_0 = prov_sym[idx0]
            sym_1 = prov_sym[idx1]
            off_0 = prov_off[idx0]
            off_1 = prov_off[idx1]

            is_asu_0 = (sym_0 == 0) & (off_0 == 0).all(axis=1)
            is_asu_1 = (sym_1 == 0) & (off_1 == 0).all(axis=1)

            # Keep only pairs where at least one atom is from the ASU
            has_asu = is_asu_0 | is_asu_1
            pairs_np = pairs_np[has_asu]
            asu_src_0 = asu_src_0[has_asu]
            asu_src_1 = asu_src_1[has_asu]
            sym_0 = sym_0[has_asu]
            sym_1 = sym_1[has_asu]
            off_0 = off_0[has_asu]
            off_1 = off_1[has_asu]
            is_asu_0 = is_asu_0[has_asu]
            is_asu_1 = is_asu_1[has_asu]

            # Normalize: put the ASU atom in position 0, mate in position 1
            # For intra-ASU pairs (both ASU), keep as-is (both are ASU anyway)
            # For symmetry pairs: swap so ASU is first
            swap = ~is_asu_0 & is_asu_1
            if swap.any():
                asu_src_0[swap], asu_src_1[swap] = asu_src_1[swap].copy(), asu_src_0[swap].copy()
                sym_0[swap], sym_1[swap] = sym_1[swap].copy(), sym_0[swap].copy()
                off_0[swap], off_1[swap] = off_1[swap].copy(), off_0[swap].copy()
                is_asu_0[swap] = True
                is_asu_1[swap] = False

            # Final indices: ASU atom indices for both atoms in each pair
            final_i1 = asu_src_0
            final_i2 = asu_src_1
            # Symmetry info comes from the mate atom (position 1)
            final_symop = sym_1
            final_offsets = off_1

            is_both_asu = is_asu_0 & is_asu_1
        else:
            # No symmetry: all pairs are intra-ASU
            final_i1 = pairs_np[:, 0]
            final_i2 = pairs_np[:, 1]
            final_symop = np.zeros(len(pairs_np), dtype=np.int64)
            final_offsets = np.zeros((len(pairs_np), 3), dtype=np.int64)
            is_both_asu = np.ones(len(pairs_np), dtype=bool)

        # --- Filtering ---
        # Bonded exclusions, same-residue, and altloc filters apply only to
        # intra-ASU pairs. Symmetry pairs cannot be bonded.

        # Start with all pairs kept
        keep_mask = np.ones(len(final_i1), dtype=bool)

        # Exclusion mask (bonded 1-2, 1-3, 1-4) -- intra-ASU only
        if exclusions and is_both_asu.any():
            exclusion_arr = np.array(list(exclusions), dtype=np.int64)
            max_idx = max(
                pdb["index"].max() + 1,
                final_i1[is_both_asu].max() + 1,
                final_i2[is_both_asu].max() + 1,
            )
            # Normalize pair order for comparison
            norm_i1 = np.minimum(final_i1, final_i2)
            norm_i2 = np.maximum(final_i1, final_i2)
            pair_hash = norm_i1 * max_idx + norm_i2
            excl_hash = exclusion_arr[:, 0] * max_idx + exclusion_arr[:, 1]
            is_excluded = np.isin(pair_hash, excl_hash)
            # Only apply to intra-ASU pairs
            keep_mask &= ~(is_excluded & is_both_asu)

        # Inter-residue mask -- intra-ASU only
        if inter_residue_only:
            chainid_array = pdb["chainid"].values
            resseq_array = pdb["resseq"].values
            same_residue = (
                (chainid_array[final_i1] == chainid_array[final_i2])
                & (resseq_array[final_i1] == resseq_array[final_i2])
            )
            keep_mask &= ~(same_residue & is_both_asu)

        # Altloc compatibility -- intra-ASU only
        if "altloc" in pdb.columns:
            altloc_array = pdb["altloc"].values.astype(str)
            altloc_array = np.where(
                np.isin(altloc_array, ["", " "]), " ", altloc_array
            )
            altloc_i = altloc_array[final_i1]
            altloc_j = altloc_array[final_i2]
            incompatible_altloc = (
                (altloc_i != " ") & (altloc_j != " ") & (altloc_i != altloc_j)
            )
            keep_mask &= ~(incompatible_altloc & is_both_asu)

        # Apply filter
        final_i1 = final_i1[keep_mask]
        final_i2 = final_i2[keep_mask]
        final_symop = final_symop[keep_mask]
        final_offsets = final_offsets[keep_mask]

        if len(final_i1) == 0:
            self.restraints["vdw"] = empty_result
            return

        # Compute min distances using VDW radii of ASU source atoms.
        vdw_np = vdw_radii.cpu().numpy()
        min_distances = vdw_np[final_i1] + vdw_np[final_i2]

        # Store results
        final_pairs = np.stack([final_i1, final_i2], axis=1)
        self.restraints["vdw"] = {
            "indices": torch.tensor(final_pairs, dtype=torch.long, device=device),
            "min_distances": torch.tensor(
                min_distances, dtype=torch.float32, device=device
            ),
            "sigmas": torch.full(
                (len(final_pairs),), sigma, dtype=torch.float32, device=device
            ),
            "symop_indices": torch.tensor(
                final_symop, dtype=torch.long, device=device
            ),
            "cell_offsets": torch.tensor(
                final_offsets, dtype=torch.long, device=device
            ),
        }

        if self.verbose > 0:
            scope = "inter-residue" if inter_residue_only else "all"
            msg = f"  Built {len(final_pairs)} VDW restraints ({scope} contacts)"
            if has_symmetry:
                is_sym_pair = (final_symop != 0) | (final_offsets != 0).any(axis=1)
                n_sym_count = int(is_sym_pair.sum())
                msg += f", {n_sym_count} symmetry contacts"
            print(msg)

    # Device movement is handled automatically by TensorDict (registered as _tensor_storage)
    # through PyTorch's Module.to(), cuda(), and cpu() methods

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
        print(f"  LINK bonds: {get_count('bond', 'link')}")

        print()
        print("BACKBONE TORSIONS:")
        print("-" * 80)
        print(f"  Phi: {get_count('torsion', 'phi')}")
        print(f"  Psi: {get_count('torsion', 'psi')}")
        print(f"  Omega: {get_count('torsion', 'omega')}")

        # Ramachandran
        rama_count = 0
        if hasattr(self, "_rama_phi_indices") and self._rama_phi_indices is not None:
            rama_count = self._rama_phi_indices.shape[0]
        if rama_count > 0:
            print(f"  Ramachandran: {rama_count}")

        print()
        print("VDW RESTRAINTS:")
        print("-" * 80)
        vdw_count = 0
        vdw_sym_count = 0
        if "vdw" in self.restraints:
            indices = self.restraints["vdw"].get("indices")
            vdw_count = 0 if indices is None else indices.shape[0]
            symop_indices = self.restraints["vdw"].get("symop_indices")
            cell_offsets = self.restraints["vdw"].get("cell_offsets")
            if symop_indices is not None and len(symop_indices) > 0:
                import torch as _torch
                is_sym = (symop_indices != 0) | (cell_offsets != 0).any(dim=-1)
                vdw_sym_count = int(is_sym.sum().item())
        vdw_asu_count = vdw_count - vdw_sym_count
        if vdw_sym_count > 0:
            print(f"  Non-bonded contacts: {vdw_count} ({vdw_asu_count} intra-ASU, {vdw_sym_count} symmetry)")
        else:
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

    def bond_lengths(self, idx, xyz: torch.Tensor = None):
        """
        Compute current bond lengths from atomic coordinates.

        Parameters
        ----------
        idx : torch.Tensor
            Bond indices tensor of shape (N, 2).
        xyz : torch.Tensor, optional
            Coordinates tensor of shape (n_atoms, 3).
            If None, uses the stored xyz_fn callable.

        Returns
        -------
        torch.Tensor
            Tensor of bond lengths of shape (N,).
        """
        xyz = self.xyz(xyz)
        if idx is None:
            return torch.tensor([], device=xyz.device)
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

    def bond_deviations(self, xyz: torch.Tensor = None):
        """
        Compute bond length deviations and sigmas.

        Parameters
        ----------
        xyz : torch.Tensor, optional
            Coordinates tensor. If None, uses the stored xyz_fn callable.

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
        bond_lengths = self.bond_lengths(idx, xyz)
        deviations = bond_lengths - references

        return deviations, sigmas

    def nll_bonds(self, xyz: torch.Tensor = None):
        """
        Compute negative log-likelihood for bond length restraints.

        For Gaussian distribution: NLL = -log(P(x|μ,σ))
        NLL = 0.5 * ((x - μ) / σ)^2 + log(σ) + 0.5 * log(2π)

        This is the true NLL where exp(-NLL) = probability density.

        Parameters
        ----------
        xyz : torch.Tensor, optional
            Coordinates tensor. If None, uses the stored xyz_fn callable.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_bonds,) with negative log-likelihood values.
        """
        from torchref.refinement.targets import gaussian_nll

        deviations, sigmas = self.bond_deviations(xyz)
        return gaussian_nll(deviations, sigmas)

    def angles(self, idx, xyz: torch.Tensor = None):
        """
        Compute current angle values for all angle restraints.

        Parameters
        ----------
        idx : torch.Tensor
            Angle indices tensor of shape (N, 3).
        xyz : torch.Tensor, optional
            Coordinates tensor. If None, uses the stored xyz_fn callable.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_angles,) with current angle values in degrees.
        """
        xyz = self.xyz(xyz)
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

    def angle_deviations(self, xyz: torch.Tensor = None):
        """
        Compute angle deviations and sigmas.

        Parameters
        ----------
        xyz : torch.Tensor, optional
            Coordinates tensor. If None, uses the stored xyz_fn callable.

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

        calculated_rad = self.angles(idx, xyz) * (torch.pi / 180.0)
        deviations = calculated_rad - references_rad

        return deviations, sigmas_rad

    def nll_angles(self, xyz: torch.Tensor = None):
        """
        Compute negative log-likelihood for angle restraints.

        For Gaussian distribution: NLL = -log(P(x|μ,σ))
        NLL = 0.5 * ((x - μ) / σ)^2 + log(σ) + 0.5 * log(2π)

        This is the true NLL where exp(-NLL) = probability density.

        Parameters
        ----------
        xyz : torch.Tensor, optional
            Coordinates tensor. If None, uses the stored xyz_fn callable.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_angles,) with negative log-likelihood values.
        """
        from torchref.refinement.targets import gaussian_nll

        deviations, sigmas = self.angle_deviations(xyz)
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
        # Note: phi/psi origins are excluded because they have no reference
        # values or sigmas (conformationally free). Omega is excluded here
        # because it is handled by a dedicated OmegaTarget that uses a
        # cis/trans von Mises mixture model.
        _torsion_origins = ["intra", "disulfide"]
        self.restraints["torsion"]["all"] = {
            "indices": self._get_all_indices("torsion", _torsion_origins),
            "references": self._get_all_property(
                "torsion", "references", _torsion_origins
            ),
            "sigmas": self._get_all_property(
                "torsion", "sigmas", _torsion_origins
            ),
            "periods": self._get_all_property(
                "torsion", "periods", _torsion_origins
            ),
        }
        # Cache max period to avoid .item() GPU sync every iteration
        periods = self.restraints["torsion"]["all"]["periods"]
        if periods is not None and periods.numel() > 0:
            self._torsion_max_period = int(periods.max().item())
        else:
            self._torsion_max_period = 1

    def torsions(self, idx, xyz: torch.Tensor = None):
        """
        Compute current torsion angle values for all torsion restraints.

        Parameters
        ----------
        idx : torch.Tensor
            Torsion indices tensor of shape (N, 4).
        xyz : torch.Tensor, optional
            Coordinates tensor. If None, uses the stored xyz_fn callable.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_torsions,) with current torsion values in degrees.
        """
        xyz = self.xyz(xyz)

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
        # Use cached max_period to avoid .item() GPU sync every iteration
        max_period = getattr(self, "_torsion_max_period", None)
        if max_period is None:
            max_period = int(periods_safe.max().item())

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
            equiv_diffs_wrapped = torch.remainder(
                equiv_diffs + torch.pi, 2.0 * torch.pi
            ) - torch.pi

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
            return torch.remainder(diff_rad + torch.pi, 2.0 * torch.pi) - torch.pi

    def torsion_deviations(self, xyz: torch.Tensor = None, wrapped=True):
        """
        Compute deviations between calculated and expected torsion angles.

        Parameters
        ----------
        xyz : torch.Tensor, optional
            Coordinates tensor. If None, uses the stored xyz_fn callable.
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
        calculated = self.torsions(idx, xyz)

        if not wrapped:
            # Simple difference
            return calculated - expected
        else:
            # Use the helper function for periodicity handling
            diff_rad = (calculated - expected) * torch.pi / 180.0
            diff_wrapped_rad = self._wrap_torsion_periodicity(diff_rad, periods)

            # Convert back to degrees
            return torch.rad2deg(diff_wrapped_rad)

    def torsion_deviations_with_sigmas(self, xyz: torch.Tensor = None):
        """
        Compute torsion deviations (wrapped for periodicity) and sigmas.

        Parameters
        ----------
        xyz : torch.Tensor, optional
            Coordinates tensor. If None, uses the stored xyz_fn callable.

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

        calculated = self.torsions(idx, xyz)

        # Wrap for periodicity
        diff_rad = (calculated - expected) * (torch.pi / 180.0)
        deviations_rad = self._wrap_torsion_periodicity(diff_rad, periods)

        return deviations_rad, sigmas_deg

    def nll_torsions(self, xyz: torch.Tensor = None):
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

        Parameters
        ----------
        xyz : torch.Tensor, optional
            Coordinates tensor. If None, uses the stored xyz_fn callable.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_torsions,) with negative log-likelihood values.
        """
        from torchref.refinement.targets import von_mises_nll

        deviations_rad, sigmas_deg = self.torsion_deviations_with_sigmas(xyz)
        return von_mises_nll(deviations_rad, sigmas_deg)

    def nll_planes(self, xyz: torch.Tensor = None):
        """
        Compute negative log-likelihood for plane restraints.

        For each plane, computes the RMSD of atom deviations from the best-fit plane.
        Uses Gaussian NLL: NLL = 0.5 * (deviation / σ)² + log(σ) + 0.5 * log(2π)

        Parameters
        ----------
        xyz : torch.Tensor, optional
            Coordinates tensor. If None, uses the stored xyz_fn callable.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_planes,) with negative log-likelihood values.
        """
        from torchref.refinement.targets import gaussian_nll

        xyz = self.xyz(xyz)
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

    def nll_vdw(self, xyz: torch.Tensor = None):
        """
        Compute negative log-likelihood for VDW (non-bonded) restraints.

        Uses a soft-repulsive potential based on distance violations.
        NLL = 0.5 * (max(0, min_dist - actual_dist) / σ)² + log(σ) + 0.5 * log(2π)

        Only violations (distances shorter than minimum) contribute to the loss.

        Parameters
        ----------
        xyz : torch.Tensor, optional
            Coordinates tensor. If None, uses the stored xyz_fn callable.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_pairs,) with negative log-likelihood values.
        """
        from torchref.refinement.targets import gaussian_nll

        xyz = self.xyz(xyz)
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

    def adp_b_differences(self, adp: torch.Tensor = None):
        """
        Compute B-factor differences between bonded atoms.

        Parameters
        ----------
        adp : torch.Tensor, optional
            ADP values. If None, uses the stored adp_fn callable.

        Returns
        -------
        torch.Tensor
            Tensor of B-factor differences (B_i - B_j) for all bonds.
        """
        b_factors = self.adp(adp)

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

    def adp_similarity_loss(self, adp: torch.Tensor = None, sigma: float = 2.0):
        """
        Compute ADP similarity loss (SIMU in Phenix/SHELX).

        This restrains the B-factors of bonded atoms to be similar.
        Loss = Σ ((B_i - B_j) / sigma)^2

        Parameters
        ----------
        adp : torch.Tensor, optional
            ADP values. If None, uses the stored adp_fn callable.
        sigma : float, default 2.0
            Target standard deviation for B-factor differences in Å².

        Returns
        -------
        torch.Tensor
            Mean similarity loss.
        """
        from torchref.refinement.targets import adp_similarity_nll

        b_diffs = self.adp_b_differences(adp)
        if len(b_diffs) == 0:
            return torch.tensor(0.0, device=self.xyz().device)
        return adp_similarity_nll(b_diffs, sigma).mean()
