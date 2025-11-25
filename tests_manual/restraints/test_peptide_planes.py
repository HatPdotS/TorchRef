#!/usr/bin/env python3
"""Test peptide plane restraints from TRANS link definition."""

import sys
from pathlib import Path
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from torchref.model import Model
from torchref.restraints import Restraints

def test_peptide_plane_restraints():
    """Test that peptide plane restraints are properly built and geometrically accurate."""
    
    # Load test structure
    pdb_file = Path(__file__).parent.parent.parent / "test_data" / "dark.pdb"
    
    print(f"Loading structure: {pdb_file}")
    model = Model()
    model.load_pdb_from_file(str(pdb_file))
    
    # Build restraints with verbose output
    print("\n" + "="*80)
    print("Building restraints...")
    print("="*80)
    restraints = Restraints(model, verbose=2)
    
    # Check if peptide planes were created
    print("\n" + "="*80)
    print("Checking peptide plane restraints...")
    print("="*80)
    
    if '4_atoms' not in restraints.restraints['plane']:
        print("❌ ERROR: No 4-atom planes found!")
        print(f"Available plane keys: {list(restraints.restraints['plane'].keys())}")
        return False
    
    plane_data = restraints.restraints['plane']['4_atoms']
    plane_indices = plane_data['indices']
    plane_sigmas = plane_data['sigmas']
    
    print(f"\n✓ Found {len(plane_indices)} total 4-atom planes")
    
    # Get coordinates
    xyz = model.xyz()
    
    # Separate intra-residue planes from peptide planes
    # Peptide planes should have atoms from different residues
    pdb = model.pdb
    
    peptide_plane_count = 0
    peptide_plane_deviations = []
    
    for i, indices in enumerate(plane_indices):
        # Get residue info for each atom
        atom_residues = []
        for idx in indices:
            atom = pdb.iloc[idx.item()]
            atom_residues.append((atom['chainid'], atom['resseq']))
        
        # Check if atoms are from different residues (peptide plane)
        unique_residues = set(atom_residues)
        if len(unique_residues) > 1:
            peptide_plane_count += 1
            
            # Calculate plane deviation
            coords = xyz[indices].detach().cpu().numpy()
            
            # Center coordinates
            centroid = coords.mean(axis=0)
            centered = coords - centroid
            
            # SVD to find best-fit plane
            U, S, Vt = np.linalg.svd(centered)
            normal = Vt[-1]  # Normal vector (smallest singular value)
            
            # Calculate deviations from plane
            deviations = np.abs(centered @ normal)
            max_dev = deviations.max()
            mean_dev = deviations.mean()
            
            peptide_plane_deviations.append({
                'indices': indices.cpu().numpy(),
                'atoms': atom_residues,
                'max_deviation': max_dev,
                'mean_deviation': mean_dev,
                'sigma': plane_sigmas[i][0].item()
            })
    
    print(f"\n✓ Found {peptide_plane_count} peptide planes (inter-residue)")
    print(f"  (vs {len(plane_indices) - peptide_plane_count} intra-residue aromatic planes)")
    
    # Analyze peptide plane geometry
    if len(peptide_plane_deviations) > 0:
        max_devs = [p['max_deviation'] for p in peptide_plane_deviations]
        mean_devs = [p['mean_deviation'] for p in peptide_plane_deviations]
        sigmas = [p['sigma'] for p in peptide_plane_deviations]
        
        print(f"\n📊 Peptide Plane Geometry:")
        print(f"  Mean deviation: {np.mean(mean_devs):.4f} Å")
        print(f"  Max deviation:  {np.max(max_devs):.4f} Å")
        print(f"  Target sigma:   {sigmas[0]:.3f} Å")
        print(f"  Deviation/sigma ratio: {np.mean(max_devs) / sigmas[0]:.2f}")
        
        # Check if geometry is good (should be < 1σ)
        if np.max(max_devs) < sigmas[0]:
            print(f"\n✅ Excellent peptide planarity! All deviations < {sigmas[0]:.3f} Å")
        elif np.mean(max_devs) < sigmas[0]:
            print(f"\n✓ Good peptide planarity (mean < {sigmas[0]:.3f} Å)")
        else:
            print(f"\n⚠ Some peptide planes deviate > {sigmas[0]:.3f} Å")
        
        # Show worst cases
        worst_planes = sorted(peptide_plane_deviations, key=lambda x: x['max_deviation'], reverse=True)[:5]
        print(f"\n📋 Top 5 most non-planar peptide bonds:")
        for i, plane in enumerate(worst_planes, 1):
            residues_str = ' - '.join([f"{chain}:{resseq}" for chain, resseq in plane['atoms']])
            print(f"  {i}. Residues {residues_str}")
            print(f"     Max deviation: {plane['max_deviation']:.4f} Å ({plane['max_deviation']/plane['sigma']:.2f}σ)")
        
        # Calculate NLL for peptide planes
        print(f"\n🔬 Negative Log-Likelihood Analysis:")
        nll = restraints.nll_planes()
        print(f"  Total plane NLL: {nll.sum().item():.3f}")
        
        # Expected NLL for perfect planarity: -0.5*log(2π) - log(σ) ≈ -2.92 for σ=0.02
        expected_nll_perfect = len(plane_indices) * (-0.5 * np.log(2 * np.pi) - np.log(sigmas[0]))
        print(f"  Expected NLL (perfect): {expected_nll_perfect:.3f}")
        print(f"  Difference: {nll.sum().item() - expected_nll_perfect:.3f}")
        
        return True
    else:
        print("\n❌ ERROR: No peptide planes found!")
        return False

if __name__ == '__main__':
    success = test_peptide_plane_restraints()
    sys.exit(0 if success else 1)
