#!/usr/bin/env python3
"""
Test script for plane restraint NLL calculation.
"""

import torch
import numpy as np
from torchref.model import Model
from torchref.restraints import Restraints

# Load test structure
model = Model()
test_pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/dark.pdb'
model.load_pdb_from_file(test_pdb)

# Load restraints with custom CIF
cif_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/elbow.AZO.dark_pdb.001.cif'
restraints = Restraints(model, cif_path, verbose=1)

print("\n" + "=" * 80)
print("Testing Plane NLL Calculation")
print("=" * 80)

# Check plane restraints exist
if len(restraints.restraints['plane']) > 0:
    print(f"\n✓ Found {len(restraints.restraints['plane'])} plane atom count groups:")
    total_planes = 0
    for key in sorted(restraints.restraints['plane'].keys()):
        n_planes = restraints.restraints['plane'][key]['indices'].shape[0]
        total_planes += n_planes
        print(f"  {key}: {n_planes} planes")
    print(f"  Total planes: {total_planes}")
    
    # Test plane NLL calculation
    print("\n" + "-" * 80)
    print("Testing nll_planes() method:")
    print("-" * 80)
    
    try:
        nll_planes = restraints.nll_planes()
        print(f"✓ nll_planes() executed successfully")
        print(f"  Shape: {nll_planes.shape}")
        print(f"  Device: {nll_planes.device}")
        print(f"  Mean NLL: {nll_planes.mean().item():.6f}")
        print(f"  Min NLL: {nll_planes.min().item():.6f}")
        print(f"  Max NLL: {nll_planes.max().item():.6f}")
        print(f"  Std NLL: {nll_planes.std().item():.6f}")
        
        # Check for NaN or Inf values
        if torch.isnan(nll_planes).any():
            print("  ⚠ Warning: Found NaN values in NLL")
        if torch.isinf(nll_planes).any():
            print("  ⚠ Warning: Found Inf values in NLL")
        
        # Test with loss function
        print("\n" + "-" * 80)
        print("Testing loss() method with planes:")
        print("-" * 80)
        
        total_loss = restraints.loss()
        print(f"✓ Total loss computed: {total_loss.item():.6f}")
        
        # Test with custom weights
        custom_weights = {'plane': 1.0}
        total_loss_weighted = restraints.loss(weights=custom_weights)
        print(f"✓ Total loss with plane weight=1.0: {total_loss_weighted.item():.6f}")
        
        # Compute plane contribution
        print("\n" + "-" * 80)
        print("Plane restraint statistics:")
        print("-" * 80)
        
        # Calculate deviations manually for inspection
        xyz = model.xyz()
        for atom_count_key in sorted(restraints.restraints['plane'].keys()):
            plane_data = restraints.restraints['plane'][atom_count_key]
            indices = plane_data['indices']
            sigmas = plane_data['sigmas']
            
            if indices.shape[0] == 0:
                continue
            
            # Get coordinates
            plane_coords = xyz[indices]
            center = plane_coords.mean(dim=1, keepdim=True)
            centered = plane_coords - center
            
            # Fit plane
            try:
                U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
                normals = Vt[:, -1, :]
                normals_expanded = normals.unsqueeze(1)
                distances = torch.abs(torch.sum(centered * normals_expanded, dim=2))
                
                print(f"\n  {atom_count_key}:")
                print(f"    Number of planes: {indices.shape[0]}")
                print(f"    Mean deviation: {distances.mean().item():.6f} Å")
                print(f"    Max deviation: {distances.max().item():.6f} Å")
                print(f"    Mean sigma: {sigmas.mean().item():.6f} Å")
                print(f"    Mean deviation/sigma: {(distances/sigmas).mean().item():.3f}")
                
            except RuntimeError as e:
                print(f"  {atom_count_key}: SVD failed - {e}")
        
        print("\n" + "=" * 80)
        print("✓ All tests passed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        
else:
    print("\n✗ No plane restraints found in structure")

print()
