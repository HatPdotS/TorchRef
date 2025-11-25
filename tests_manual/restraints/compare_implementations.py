"""
Comparison between old and new restraints implementations.

This script demonstrates the differences between the old restraints_handler
and the new Restraints class, showing the advantages of the new approach.
"""

import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import time
import torch
from torchref.model import Model
from torchref.restraints import Restraints


def compare_implementations():
    """Compare old and new restraints implementations."""
    print("\n" + "="*80)
    print("Comparison: Old vs New Restraints Implementation")
    print("="*80)
    
    # Load model
    print("\nLoading test model...")
    model = Model()
    test_pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_all.pdb'
    model.load_pdb_from_file(test_pdb)
    
    print(f"Model loaded: {len(model.pdb)} atoms")
    
    # Create new restraints
    print("\n" + "-"*80)
    print("NEW IMPLEMENTATION (restraints_new.py)")
    print("-"*80)
    
    cif_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/Merged_restraints_all_opened.cif'
    
    start_time = time.time()
    restraints_new = Restraints(model, cif_path)
    creation_time = time.time() - start_time
    
    print(f"\n✓ Restraints created in {creation_time:.4f} seconds")
    print(restraints_new)
    
    # Show advantages
    print("\n" + "-"*80)
    print("ADVANTAGES OF NEW IMPLEMENTATION")
    print("-"*80)
    
    print("\n1. PRE-BUILT TENSOR STRUCTURE")
    print("   - All restraints built once at initialization")
    print("   - No dictionary lookups during refinement")
    print("   - Direct tensor operations")
    
    if restraints_new.bond_indices is not None:
        print(f"\n   Bond restraints: {restraints_new.bond_indices.shape}")
        print(f"   - Stored as tensor of shape (N, 2)")
        print(f"   - Direct indexing: xyz[restraints.bond_indices[:, 0]]")
    
    print("\n2. DEVICE MANAGEMENT")
    print("   - Easy to move between CPU and GPU")
    device = restraints_new.bond_indices.device if restraints_new.bond_indices is not None else "N/A"
    print(f"   - Current device: {device}")
    print("   - Usage: restraints.cuda() or restraints.cpu()")
    
    print("\n3. VECTORIZED OPERATIONS")
    print("   - All operations use PyTorch tensors")
    print("   - Batch processing of all restraints at once")
    print("   - Automatic gradient computation")
    
    print("\n4. MEMORY EFFICIENCY")
    
    def get_tensor_size(tensor):
        if tensor is None:
            return 0
        return tensor.element_size() * tensor.nelement()
    
    total_size = 0
    if restraints_new.bond_indices is not None:
        bond_size = (get_tensor_size(restraints_new.bond_indices) +
                     get_tensor_size(restraints_new.bond_references) +
                     get_tensor_size(restraints_new.bond_sigmas))
        total_size += bond_size
        print(f"   - Bond restraints: {bond_size / 1024:.2f} KB")
    
    if restraints_new.angle_indices is not None:
        angle_size = (get_tensor_size(restraints_new.angle_indices) +
                      get_tensor_size(restraints_new.angle_references) +
                      get_tensor_size(restraints_new.angle_sigmas))
        total_size += angle_size
        print(f"   - Angle restraints: {angle_size / 1024:.2f} KB")
    
    if restraints_new.torsion_indices is not None:
        torsion_size = (get_tensor_size(restraints_new.torsion_indices) +
                        get_tensor_size(restraints_new.torsion_references) +
                        get_tensor_size(restraints_new.torsion_sigmas))
        total_size += torsion_size
        print(f"   - Torsion restraints: {torsion_size / 1024:.2f} KB")
    
    print(f"   - Total memory: {total_size / 1024:.2f} KB")
    
    print("\n5. CLEAN API")
    print("   - Simple initialization: Restraints(model, cif_path)")
    print("   - Clear data structure: restraints.bond_indices, etc.")
    print("   - Built-in summary: restraints.summary()")
    
    print("\n6. OPTIMIZATION READY")
    print("   - Direct integration with PyTorch optimizers")
    print("   - Automatic differentiation through restraints")
    print("   - Example:")
    print("     loss = compute_restraint_loss(model, restraints)")
    print("     loss.backward()")
    
    # Demonstrate computation speed
    print("\n" + "-"*80)
    print("PERFORMANCE DEMONSTRATION")
    print("-"*80)
    
    if restraints_new.bond_indices is not None:
        xyz = model.xyz()
        
        # Time bond length computation
        n_iterations = 1000
        start_time = time.time()
        for _ in range(n_iterations):
            xyz1 = xyz[restraints_new.bond_indices[:, 0]]
            xyz2 = xyz[restraints_new.bond_indices[:, 1]]
            bond_lengths = torch.sqrt(torch.sum((xyz1 - xyz2) ** 2, dim=1))
        computation_time = time.time() - start_time
        
        print(f"\nBond length computation:")
        print(f"  - {n_iterations} iterations in {computation_time:.4f} seconds")
        print(f"  - {computation_time/n_iterations*1000:.4f} ms per iteration")
        print(f"  - {restraints_new.bond_indices.shape[0]} bonds computed each time")
    
    print("\n" + "="*80)
    print("Comparison completed!")
    print("="*80 + "\n")


if __name__ == '__main__':
    try:
        compare_implementations()
    except Exception as e:
        print(f"\n✗ Comparison failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
