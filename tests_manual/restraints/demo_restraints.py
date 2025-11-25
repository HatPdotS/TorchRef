"""
Demo script showing how to use the new Restraints class.

This script demonstrates the basic usage of the Restraints class for
crystallographic model refinement.
"""

import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from torchref.model import Model
from torchref.restraints import Restraints


def main():
    """Main demo function."""
    print("\n" + "="*80)
    print("Restraints Class Demo")
    print("="*80)
    
    # Load test model
    print("\n1. Loading model...")
    model = Model()
    test_pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_all.pdb'
    model.load_pdb_from_file(test_pdb)
    
    print(f"   Loaded model with {len(model.pdb)} atoms")
    print(f"   Unique residue types: {model.pdb['resname'].nunique()}")
    print(f"   Chains: {', '.join(model.pdb['chainid'].unique())}")
    
    # Create restraints
    print("\n2. Creating restraints from CIF file...")
    cif_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/Merged_restraints_all_opened.cif'
    restraints = Restraints(model, cif_path)
    
    print(f"   Restraints created successfully!")
    print(restraints)
    
    # Display summary
    print("\n3. Restraints summary:")
    restraints.summary()
    
    # Show some example data
    print("\n4. Example restraint data:")
    
    if restraints.bond_indices is not None:
        print(f"\n   First 5 bond restraints:")
        for i in range(min(5, restraints.bond_indices.shape[0])):
            idx1, idx2 = restraints.bond_indices[i]
            ref = restraints.bond_references[i]
            sigma = restraints.bond_sigmas[i]
            print(f"     Bond {i+1}: atoms {idx1}-{idx2}, "
                  f"expected = {ref:.3f} Å, σ = {sigma:.4f} Å")
    
    if restraints.angle_indices is not None:
        print(f"\n   First 5 angle restraints:")
        for i in range(min(5, restraints.angle_indices.shape[0])):
            idx1, idx2, idx3 = restraints.angle_indices[i]
            ref = restraints.angle_references[i]
            sigma = restraints.angle_sigmas[i]
            print(f"     Angle {i+1}: atoms {idx1}-{idx2}-{idx3}, "
                  f"expected = {ref:.2f}°, σ = {sigma:.3f}°")
    
    if restraints.torsion_indices is not None and restraints.torsion_indices.shape[0] > 0:
        print(f"\n   First 5 torsion restraints:")
        for i in range(min(5, restraints.torsion_indices.shape[0])):
            idx1, idx2, idx3, idx4 = restraints.torsion_indices[i]
            ref = restraints.torsion_references[i]
            sigma = restraints.torsion_sigmas[i]
            print(f"     Torsion {i+1}: atoms {idx1}-{idx2}-{idx3}-{idx4}, "
                  f"expected = {ref:.2f}°, σ = {sigma:.3f}°")
    
    print("\n" + "="*80)
    print("Demo completed successfully!")
    print("="*80 + "\n")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
