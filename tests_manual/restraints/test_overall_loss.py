#!/usr/bin/env python3
"""Test overall loss function after torsion fixes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from torchref.model import Model
from torchref.restraints import Restraints

def test_overall_loss():
    """Test that all NLL components are reasonable."""
    
    # Load test structure
    pdb_file = Path(__file__).parent.parent.parent / "test_data" / "dark.pdb"
    
    print(f"Loading structure: {pdb_file}")
    model = Model()
    model.load_pdb_from_file(str(pdb_file))
    
    print("\nBuilding restraints...")
    restraints = Restraints(model, verbose=0)
    
    print("\n" + "="*80)
    print("INDIVIDUAL NLL COMPONENTS")
    print("="*80)
    
    # Calculate each component
    nll_bonds = restraints.nll_bonds()
    nll_angles = restraints.nll_angles()
    nll_torsions = restraints.nll_torsions()
    nll_planes = restraints.nll_planes()
    
    print(f"\n📊 Bonds:")
    print(f"   Total: {nll_bonds.sum().item():.1f}")
    print(f"   Mean:  {nll_bonds.mean().item():.3f}")
    print(f"   Count: {len(nll_bonds)}")
    
    print(f"\n📊 Angles:")
    print(f"   Total: {nll_angles.sum().item():.1f}")
    print(f"   Mean:  {nll_angles.mean().item():.3f}")
    print(f"   Count: {len(nll_angles)}")
    
    print(f"\n📊 Torsions:")
    print(f"   Total: {nll_torsions.sum().item():.1f}")
    print(f"   Mean:  {nll_torsions.mean().item():.3f}")
    print(f"   Count: {len(nll_torsions)}")
    
    print(f"\n📊 Planes:")
    print(f"   Total: {nll_planes.sum().item():.1f}")
    print(f"   Mean:  {nll_planes.mean().item():.3f}")
    print(f"   Count: {len(nll_planes)}")
    
    # Total loss
    print("\n" + "="*80)
    print("TOTAL LOSS")
    print("="*80)
    
    total_loss = restraints.loss()
    print(f"\n📊 Overall Loss: {total_loss.item():.1f}")
    
    # Breakdown
    bond_contrib = nll_bonds.sum().item()
    angle_contrib = nll_angles.sum().item()
    torsion_contrib = 0.5 * nll_torsions.sum().item()  # weight = 0.5
    plane_contrib = 0.5 * nll_planes.sum().item()      # weight = 0.5
    
    total_contrib = bond_contrib + angle_contrib + torsion_contrib + plane_contrib
    
    print(f"\nWeighted contributions:")
    print(f"   Bonds:    {bond_contrib:10.1f} ({100*bond_contrib/total_contrib:5.1f}%)")
    print(f"   Angles:   {angle_contrib:10.1f} ({100*angle_contrib/total_contrib:5.1f}%)")
    print(f"   Torsions: {torsion_contrib:10.1f} ({100*torsion_contrib/total_contrib:5.1f}%) [weight=0.5]")
    print(f"   Planes:   {plane_contrib:10.1f} ({100*plane_contrib/total_contrib:5.1f}%) [weight=0.5]")
    print(f"   {'─'*60}")
    print(f"   Total:    {total_contrib:10.1f}")
    
    # Sanity checks
    print("\n" + "="*80)
    print("SANITY CHECKS")
    print("="*80)
    
    checks_passed = True
    
    # Check 1: No NaN or Inf
    if not all([
        nll_bonds.isfinite().all(),
        nll_angles.isfinite().all(),
        nll_torsions.isfinite().all(),
        nll_planes.isfinite().all()
    ]):
        print("❌ FAIL: Found NaN or Inf in NLL values")
        checks_passed = False
    else:
        print("✅ PASS: All NLL values are finite")
    
    # Check 2: Mean NLLs in reasonable range
    reasonable_ranges = {
        'bonds': (-5, 10),
        'angles': (-5, 10),
        'torsions': (-2, 20),
        'planes': (-5, 10)
    }
    
    mean_nlls = {
        'bonds': nll_bonds.mean().item(),
        'angles': nll_angles.mean().item(),
        'torsions': nll_torsions.mean().item(),
        'planes': nll_planes.mean().item()
    }
    
    for name, mean_val in mean_nlls.items():
        low, high = reasonable_ranges[name]
        if low <= mean_val <= high:
            print(f"✅ PASS: {name.capitalize()} mean NLL ({mean_val:.2f}) in range [{low}, {high}]")
        else:
            print(f"❌ FAIL: {name.capitalize()} mean NLL ({mean_val:.2f}) outside range [{low}, {high}]")
            checks_passed = False
    
    # Check 3: Total loss is reasonable
    n_atoms = len(model.pdb)
    loss_per_atom = total_loss.item() / n_atoms
    
    if 0 < loss_per_atom < 100:
        print(f"✅ PASS: Loss per atom ({loss_per_atom:.2f}) is reasonable")
    else:
        print(f"❌ FAIL: Loss per atom ({loss_per_atom:.2f}) is unreasonable")
        checks_passed = False
    
    print("\n" + "="*80)
    if checks_passed:
        print("✅ ALL CHECKS PASSED - Restraints are working correctly!")
    else:
        print("❌ SOME CHECKS FAILED - Review the issues above")
    print("="*80)
    
    return checks_passed

if __name__ == '__main__':
    success = test_overall_loss()
    sys.exit(0 if success else 1)
