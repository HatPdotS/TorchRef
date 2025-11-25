"""
Quick Start Example for Restraints Class

This module provides a simple example of how to use the Restraints class
for crystallographic refinement.
"""

from torchref.model import Model
from torchref.restraints import Restraints
import torch
import numpy as np


def load_model_and_restraints(pdb_path, cif_path):
    """
    Load a model and create restraints.
    
    Args:
        pdb_path: Path to PDB file
        cif_path: Path to CIF restraints dictionary
        
    Returns:
        Tuple of (model, restraints)
    """
    # Load model
    model = Model()
    model.load_pdb_from_file(pdb_path)
    
    # Create restraints
    restraints = Restraints(model, cif_path)
    
    return model, restraints


def compute_bond_length_rmsd(model, restraints):
    """
    Compute RMSD of bond lengths from ideal values.
    
    Args:
        model: Model instance
        restraints: Restraints instance
        
    Returns:
        RMSD in Ångströms
    """
    if restraints.bond_indices is None:
        return None
    
    # Get coordinates
    xyz = model.xyz()
    
    # Extract atom positions
    xyz1 = xyz[restraints.bond_indices[:, 0]]
    xyz2 = xyz[restraints.bond_indices[:, 1]]
    
    # Compute bond lengths
    bond_lengths = torch.sqrt(torch.sum((xyz1 - xyz2) ** 2, dim=1))
    
    # Compute deviations
    deviations = bond_lengths - restraints.bond_references
    
    # Compute RMSD
    rmsd = torch.sqrt((deviations ** 2).mean())
    
    return rmsd.item()


def compute_angle_rmsd(model, restraints):
    """
    Compute RMSD of angles from ideal values.
    
    Args:
        model: Model instance
        restraints: Restraints instance
        
    Returns:
        RMSD in degrees
    """
    if restraints.angle_indices is None:
        return None
    
    # Get coordinates
    xyz = model.xyz()
    
    # Extract atom positions
    xyz1 = xyz[restraints.angle_indices[:, 0]]
    xyz2 = xyz[restraints.angle_indices[:, 1]]  # Vertex
    xyz3 = xyz[restraints.angle_indices[:, 2]]
    
    # Compute vectors
    v1 = xyz1 - xyz2
    v2 = xyz3 - xyz2
    
    # Normalize
    v1_norm = v1 / torch.sqrt(torch.sum(v1**2, dim=1, keepdim=True))
    v2_norm = v2 / torch.sqrt(torch.sum(v2**2, dim=1, keepdim=True))
    
    # Compute angles
    dot_product = torch.clamp(torch.sum(v1_norm * v2_norm, dim=1), -1.0, 1.0)
    angles_rad = torch.arccos(dot_product)
    angles_deg = angles_rad * 180.0 / np.pi
    
    # Compute deviations
    deviations = angles_deg - restraints.angle_references
    
    # Compute RMSD
    rmsd = torch.sqrt((deviations ** 2).mean())
    
    return rmsd.item()


def compute_restraint_loss(model, restraints, weight_bonds=1.0, weight_angles=1.0):
    """
    Compute total restraint loss (differentiable for optimization).
    
    Args:
        model: Model instance
        restraints: Restraints instance
        weight_bonds: Weight for bond length term
        weight_angles: Weight for angle term
        
    Returns:
        Total loss (torch.Tensor)
    """
    loss = 0.0
    
    # Get coordinates
    xyz = model.xyz()
    
    # Bond length loss
    if restraints.bond_indices is not None:
        xyz1 = xyz[restraints.bond_indices[:, 0]]
        xyz2 = xyz[restraints.bond_indices[:, 1]]
        bond_lengths = torch.sqrt(torch.sum((xyz1 - xyz2) ** 2, dim=1))
        bond_deviations = (bond_lengths - restraints.bond_references) / restraints.bond_sigmas
        bond_loss = (bond_deviations ** 2).mean()
        loss = loss + weight_bonds * bond_loss
    
    # Angle loss
    if restraints.angle_indices is not None:
        xyz1 = xyz[restraints.angle_indices[:, 0]]
        xyz2 = xyz[restraints.angle_indices[:, 1]]
        xyz3 = xyz[restraints.angle_indices[:, 2]]
        
        v1 = xyz1 - xyz2
        v2 = xyz3 - xyz2
        v1_norm = v1 / torch.sqrt(torch.sum(v1**2, dim=1, keepdim=True))
        v2_norm = v2 / torch.sqrt(torch.sum(v2**2, dim=1, keepdim=True))
        
        dot_product = torch.clamp(torch.sum(v1_norm * v2_norm, dim=1), -1.0, 1.0)
        angles_rad = torch.arccos(dot_product)
        angles_deg = angles_rad * 180.0 / np.pi
        
        angle_deviations = (angles_deg - restraints.angle_references) / restraints.angle_sigmas
        angle_loss = (angle_deviations ** 2).mean()
        loss = loss + weight_angles * angle_loss
    
    return loss


def example_usage():
    """
    Example of how to use the restraints in refinement.
    """
    print("Example: Using Restraints for Crystallographic Refinement")
    print("=" * 80)
    
    # Paths
    pdb_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_all.pdb'
    cif_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/Merged_restraints_all_opened.cif'
    
    # Load model and restraints
    print("\n1. Loading model and restraints...")
    model, restraints = load_model_and_restraints(pdb_path, cif_path)
    
    print(f"   Model: {len(model.pdb)} atoms")
    print(f"   Bonds: {restraints.bond_indices.shape[0] if restraints.bond_indices is not None else 0}")
    print(f"   Angles: {restraints.angle_indices.shape[0] if restraints.angle_indices is not None else 0}")
    
    # Compute initial geometry
    print("\n2. Computing initial geometry quality...")
    bond_rmsd = compute_bond_length_rmsd(model, restraints)
    angle_rmsd = compute_angle_rmsd(model, restraints)
    
    if bond_rmsd is not None:
        print(f"   Bond length RMSD: {bond_rmsd:.4f} Å")
    if angle_rmsd is not None:
        print(f"   Angle RMSD: {angle_rmsd:.4f}°")
    
    # Compute loss
    print("\n3. Computing restraint loss...")
    loss = compute_restraint_loss(model, restraints)
    print(f"   Total loss: {loss.item():.4f}")
    
    print("\n" + "=" * 80)
    print("Example completed!")


if __name__ == '__main__':
    example_usage()
