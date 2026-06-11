"""
Clash score calculator for crystallographic alignment.

Provides clash-based scoring to complement Patterson alignment by detecting
steric clashes between symmetry-related molecules.
"""

from typing import TYPE_CHECKING, List, Optional

import torch
import torch.nn as nn

from torchref.base.coordinates import (
    cartesian_to_fractional_torch,
    fractional_to_cartesian_torch,
)
from torchref.config import get_default_device, get_float_dtype
from torchref.symmetry import Cell, SpaceGroup
from torchref.symmetry.spacegroup import SpaceGroupLike
from torchref.utils.device_mixin import DeviceMixin

from .transform import RigidTransform

if TYPE_CHECKING:
    from torchref.model.model import Model


class AtomSampler:
    """
    Select representative atoms for clash checking.

    Provides methods to create atom selection masks based on the type of
    molecular structure (protein vs. ligand-containing).
    """

    @staticmethod
    def from_model(
        model: "Model",
        mode: str = "auto",
    ) -> torch.Tensor:
        """
        Create atom selection mask from a Model.

        Parameters
        ----------
        model : Model
            The crystallographic model containing atomic data.
        mode : str, default 'auto'
            Selection mode:
            - 'auto': Use CA atoms if protein present (ATOM records),
                      else all atoms (for small molecules with only HETATM)
            - 'ca_only': Only CA atoms (alpha carbons)
            - 'all_atoms': All atoms in the structure

        Returns
        -------
        torch.Tensor
            Boolean mask of shape (n_atoms,) indicating which atoms to use.

        Examples
        --------
        ::

            from torchref.model import Model
            model = Model().load_pdb('protein.pdb')
            mask = AtomSampler.from_model(model, mode='auto')
            print(f"Selected {mask.sum()} atoms out of {len(mask)}")
        """
        pdb = model.pdb

        if mode == "auto":
            # Check if structure has normal ATOM records (protein/nucleic acid)
            has_atom = (pdb["ATOM"] == "ATOM").any()
            if has_atom:
                # Has protein/nucleic acid: use CA atoms for efficiency
                return torch.tensor((pdb["name"] == "CA").values, dtype=torch.bool)
            else:
                # Only HETATM (small molecule): use all atoms
                return torch.ones(len(pdb), dtype=torch.bool)
        elif mode == "ca_only":
            return torch.tensor((pdb["name"] == "CA").values, dtype=torch.bool)
        elif mode == "all_atoms":
            return torch.ones(len(pdb), dtype=torch.bool)
        else:
            raise ValueError(
                f"Unknown mode '{mode}'. Use 'auto', 'ca_only', or 'all_atoms'."
            )


class ClashScoreCalculator(DeviceMixin, nn.Module):
    """
    Calculate clash scores between symmetry-related molecules.

    Computes steric clash violations between an asymmetric unit (ASU) and
    its symmetry-related copies. Automatically filters symmetry mates based
    on the actual input coordinates to only consider those that can potentially
    clash.

    Uses a steep **4 penalty: (radius² - dist²)² which rises quickly as
    atoms get closer than the clash radius.

    Parameters
    ----------
    symmetry : str, int, gemmi.SpaceGroup, or SpaceGroup
        Space group specification for symmetry expansion.
    default_clash_radius : float, default 5.0
        Default minimum allowed distance between atoms (can be overridden in forward).
    dtype : torch.dtype, default torch.float32
        Data type for computations.
    device : torch.device, default 'cpu'
        Device for computations.

    Examples
    --------
    ::

        from torchref.alignment.clashscore import ClashScoreCalculator, AtomSampler
        from torchref.model import Model

        model = Model().load_pdb('structure.pdb')
        calc = ClashScoreCalculator(symmetry=model.spacegroup)
        mask = AtomSampler.from_model(model)
        score = calc(xyz=model.xyz(), cell=model.cell, atom_mask=mask)
        print(f"Clash score: {score.item():.4f}")
    """

    # Cell offsets for neighboring cells (7 cells: central + 6 face neighbors)
    _cell_offsets = [
        (0, 0, 0),  # Central
        (-1, 0, 0),
        (1, 0, 0),  # x neighbors
        (0, -1, 0),
        (0, 1, 0),  # y neighbors
        (0, 0, -1),
        (0, 0, 1),  # z neighbors
    ]

    def __init__(
        self,
        symmetry: SpaceGroupLike,
        default_clash_radius: float = 5.0,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ):
        super().__init__()
        if dtype is None:
            dtype = get_float_dtype()
        if device is None:
            device = get_default_device()
        self.default_clash_radius = default_clash_radius
        self.dtype = dtype
        self._device = device

        # Initialize symmetry handler. Use the user-configured dtype so the
        # SpaceGroup stays MPS-compatible; ``_get_valid_transforms`` casts to
        # CPU+float64 internally where high-precision symmetry math is needed.
        if isinstance(symmetry, SpaceGroup):
            self.symmetry = symmetry
        else:
            self.symmetry = SpaceGroup(symmetry, dtype=self.dtype, device=device)

    def _get_valid_transforms(
        self,
        cell: torch.Tensor,
        centroid_frac: torch.Tensor,
        molecule_radius: float,
        clash_radius: float,
    ) -> List[RigidTransform]:
        """
        Compute which symmetry mates can potentially clash with the ASU.

        Parameters
        ----------
        cell : torch.Tensor
            Unit cell parameters [a, b, c, alpha, beta, gamma].
        centroid_frac : torch.Tensor
            Fractional coordinates of molecule centroid.
        molecule_radius : float
            Approximate radius of the molecule in Angstroms.
        clash_radius : float
            Minimum allowed distance between atoms.

        Returns
        -------
        List[RigidTransform]
            List of symmetry transforms that could produce clashes.
        """
        # Compute fractionalization matrix using Cell. Pin to CPU because the
        # rest of this method does CPU-only float64 symmetry math.
        cell_obj = Cell(cell, dtype=torch.float64, device="cpu")
        B = cell_obj.fractional_matrix

        centroid_frac = centroid_frac.to(device="cpu", dtype=torch.float64)

        # Threshold distance for filtering
        # Two molecules can clash if centroid distance < 2*radius + clash_radius
        threshold = 2 * molecule_radius + clash_radius

        # Identity matrix for comparison with rotation matrices
        I = torch.eye(3, dtype=torch.float64)

        valid_transforms = []
        n_ops = self.symmetry.n_ops

        for op_idx in range(n_ops):
            R = self.symmetry.matrices[op_idx].cpu().to(torch.float64)
            t = self.symmetry.translations[op_idx].cpu().to(torch.float64)

            for offset in self._cell_offsets:
                # Skip identity operation in central cell (self-interaction)
                if op_idx == 0 and offset == (0, 0, 0):
                    continue

                offset_tensor = torch.tensor(offset, dtype=torch.float64)

                # Displacement between ASU centroid and this symmetry mate's centroid
                # Symmetry mate position: R @ x + t + offset
                # Displacement from ASU (identity, no offset): (R - I) @ centroid + t + offset
                d_frac = (R - I) @ centroid_frac + t + offset_tensor

                # Convert to Cartesian distance
                d_cart = B @ d_frac
                dist = d_cart.norm().item()

                if dist < threshold:
                    # Create RigidTransform for this symmetry operation in fractional space
                    t_total = t + offset_tensor
                    transform = RigidTransform.from_matrix(R, t_total)
                    valid_transforms.append(transform)

        return valid_transforms

    def forward(
        self,
        xyz: torch.Tensor,
        cell: torch.Tensor,
        atom_mask: Optional[torch.Tensor] = None,
        clash_radius: float = 5.0,
    ) -> torch.Tensor:
        """
        Compute clash score for given coordinates.

        Automatically determines which symmetry mates could clash based on
        the actual input coordinates. Uses squared distances for efficiency
        and a steep **4 penalty for clashes.

        Parameters
        ----------
        xyz : torch.Tensor
            Cartesian coordinates of shape (N, 3).
        cell : torch.Tensor
            Unit cell parameters [a, b, c, alpha, beta, gamma].
        atom_mask : torch.Tensor, optional
            Boolean mask of shape (N,) selecting atoms to use.
            If None, all atoms are used.
        clash_radius : float, default 5.0
            Minimum allowed distance between atoms in Angstroms.
            Atoms closer than this will contribute to the clash score.

        Returns
        -------
        torch.Tensor
            Scalar clash score. Lower values indicate fewer clashes.
            Zero indicates no clashes within the clash radius.
        """
        device = xyz.device
        dtype = xyz.dtype

        # Apply atom mask
        if atom_mask is not None:
            atom_mask = atom_mask.to(device)
            xyz_selected = xyz[atom_mask]
        else:
            xyz_selected = xyz

        n_atoms = xyz_selected.shape[0]

        if n_atoms == 0:
            return torch.tensor(0.0, device=device, dtype=dtype)

        # Compute molecule properties from actual coordinates
        # Use Cell object to get transformation matrices
        cell_obj = Cell(cell, dtype=dtype, device=device)
        B = cell_obj.fractional_matrix
        B_inv = cell_obj.inv_fractional_matrix

        centroid = xyz_selected.mean(dim=0)
        molecule_radius = (xyz_selected - centroid).norm(dim=1).max().item()
        centroid_frac = cartesian_to_fractional_torch(
            centroid.unsqueeze(0), cell_obj.data, B_inv
        ).squeeze(0)

        # Get valid transforms for these coordinates
        valid_transforms = self._get_valid_transforms(
            cell=cell,
            centroid_frac=centroid_frac,
            molecule_radius=molecule_radius,
            clash_radius=clash_radius,
        )

        # If no valid mates after filtering, no clashes possible
        if len(valid_transforms) == 0:
            return torch.tensor(0.0, device=device, dtype=dtype)

        # Convert ASU to fractional coordinates
        xyz_frac = cartesian_to_fractional_torch(xyz_selected, cell_obj.data, B_inv)

        # Precompute squared clash radius threshold
        clash_radius_sq = clash_radius**2

        # Accumulate score over valid symmetry mates
        total_score = torch.tensor(0.0, device=device, dtype=dtype)
        n_clashes = 0

        for transform in valid_transforms:
            # Apply symmetry operation in fractional space using RigidTransform
            xyz_mate_frac = transform.apply(xyz_frac)

            # Convert back to Cartesian
            xyz_mate_cart = fractional_to_cartesian_torch(
                xyz_mate_frac, cell_obj.data, B
            )

            # Compute squared pairwise distances (more efficient, no sqrt)
            diff = xyz_selected.unsqueeze(1) - xyz_mate_cart.unsqueeze(0)  # (N, N, 3)
            dists_sq = (diff**2).sum(dim=-1)  # (N, N)

            # Compute violations: (radius² - dist²), clipped to 0
            # This gives a steep penalty that increases rapidly as atoms get closer
            violations_sq = torch.clamp(clash_radius_sq - dists_sq, min=0.0)

            # Apply **2 to squared violations = **4 penalty on distance violation
            # (radius² - dist²)² rises steeply as dist approaches 0
            total_score = total_score + (violations_sq**2).sum()
            n_clashes += (violations_sq > 0).sum().item()

        # Normalize by number of atom pairs checked
        n_pairs = n_atoms * n_atoms * len(valid_transforms)
        if n_pairs > 0:
            total_score = total_score / n_pairs

        return total_score


def compute_clash_score(
    model: "Model",
    mode: str = "auto",
    clash_radius: float = 5.0,
) -> torch.Tensor:
    """
    Convenience function to compute clash score for a model.

    Parameters
    ----------
    model : Model
        Crystallographic model with coordinates and symmetry.
    mode : str, default 'auto'
        Atom selection mode ('auto', 'ca_only', 'all_atoms').
    clash_radius : float, default 5.0
        Minimum allowed distance between atoms in Angstroms.

    Returns
    -------
    torch.Tensor
        Scalar clash score.

    Examples
    --------
    ::

        from torchref.model import Model
        from torchref.alignment.clashscore import compute_clash_score
        model = Model().load_pdb('structure.pdb')
        score = compute_clash_score(model)
        print(f"Clash score: {score.item():.4f}")
    """
    calc = ClashScoreCalculator(
        symmetry=model.spacegroup,
        device=model.device,
    )

    atom_mask = AtomSampler.from_model(model, mode=mode)

    return calc(
        xyz=model.xyz(),
        cell=model.cell,
        atom_mask=atom_mask,
        clash_radius=clash_radius,
    )
