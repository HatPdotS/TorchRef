"""
Example usage of the cleaned-up ModelFT class.

This demonstrates the purpose-built FT implementation for electron density
map calculation without needing scattering vectors.
"""

import torch
import numpy as np
from torchref.model_ft import ModelFT


def basic_usage_example():
    """Basic example of using ModelFT."""
    print("="*60)
    print("BASIC MODELFT USAGE")
    print("="*60)
    
    # 1. Create and load model
    mol_ft = ModelFT()
    mol_ft.load_pdb_from_file("your_structure.pdb", strip_H=True)
    # This automatically:
    #   - Calls parent class load_pdb_from_file()
    #   - Sets up real-space grids
    #   - Builds ITC92 parametrization
    
    # 2. Build FT-specific cache (optional but recommended)
    # This caches xyz, B-factors, and ITC92 A,B parameters
    # NO scattering vectors needed!
    mol_ft.build_ft_cache()
    
    # 3. Build electron density map
    density_map = mol_ft.build_density_map(radius=30)
    
    # 4. Save map
    mol_ft.save_map("output_density.ccp4")
    
    # 5. Get statistics
    stats = mol_ft.get_map_statistics()
    print("\nMap Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def advanced_caching_example():
    """Demonstrate the improved caching system."""
    print("\n" + "="*60)
    print("ADVANCED CACHING - FT-SPECIFIC")
    print("="*60)
    
    mol_ft = ModelFT()
    mol_ft.load_pdb_from_file("your_structure.pdb")
    
    # The FT cache stores:
    # - xyz coordinates
    # - B-factors (isotropic) or U tensors (anisotropic)
    # - ITC92 A parameters (amplitude coefficients)
    # - ITC92 B parameters (width coefficients)
    # - Occupancies
    
    # Build cache
    mol_ft.build_ft_cache()
    
    # Access cached data directly (for inspection or custom operations)
    if mol_ft._cached_iso_xyz_ft is not None:
        print(f"\nIsotropic atoms cached:")
        print(f"  XYZ shape: {mol_ft._cached_iso_xyz_ft.shape}")
        print(f"  B-factors shape: {mol_ft._cached_iso_b_ft.shape}")
        print(f"  ITC92 A shape: {mol_ft._cached_iso_A_ft.shape}")
        print(f"  ITC92 B shape: {mol_ft._cached_iso_B_ft.shape}")
        print(f"  Occupancies shape: {mol_ft._cached_iso_occ_ft.shape}")
    
    if mol_ft._cached_aniso_xyz_ft is not None:
        print(f"\nAnisotropic atoms cached:")
        print(f"  XYZ shape: {mol_ft._cached_aniso_xyz_ft.shape}")
        print(f"  U tensors shape: {mol_ft._cached_aniso_U_ft.shape}")
        print(f"  ITC92 A shape: {mol_ft._cached_aniso_A_ft.shape}")
        print(f"  ITC92 B shape: {mol_ft._cached_aniso_B_ft.shape}")
    
    # Build density map using cached data (fast!)
    mol_ft.build_density_map(radius=30)
    
    # Modify coordinates and rebuild
    print("\nModifying coordinates...")
    mol_ft._cached_iso_xyz_ft.data += torch.randn_like(mol_ft._cached_iso_xyz_ft.data) * 0.1
    
    # Rebuild map (cache is still valid, just recomputes density)
    mol_ft.build_density_map(radius=30)
    
    # Clear cache if needed
    mol_ft.clear_ft_cache()
    print("Cache cleared")


def parametrization_example():
    """Demonstrate the extended parametrization functions."""
    print("\n" + "="*60)
    print("EXTENDED PARAMETRIZATION")
    print("="*60)
    
    import torchref.get_scattering_factor_torch as gsf
    
    # Method 1: From DataFrame (automatic with load_pdb_from_file)
    mol_ft = ModelFT()
    mol_ft.load_pdb_from_file("your_structure.pdb")
    
    print("\nParametrization from PDB:")
    print(f"  Elements: {list(mol_ft.parametrization.keys())}")
    
    # Method 2: For specific elements
    print("\nGetting parametrization for specific elements:")
    elements = ['C', 'N', 'O']
    params = gsf.get_parametrization_for_elements(elements)
    
    for elem in elements:
        A, B, C = params[elem]
        print(f"\n  {elem}:")
        print(f"    A (amplitudes): {A.squeeze()}")
        print(f"    B (widths): {B.squeeze()}")
        print(f"    C (constant): {C.squeeze()}")
    
    # Method 3: With charges
    print("\nWith ionic charges:")
    elements = ['Fe', 'Fe']
    charges = [2, 3]  # Fe2+, Fe3+
    params_charged = gsf.get_parametrization_for_elements(elements, charges)


def comparison_with_base_model():
    """Compare ModelFT with base Model class."""
    print("\n" + "="*60)
    print("MODELFT vs BASE MODEL")
    print("="*60)
    
    from torchref.Model import model
    
    # Base Model: Needs scattering vectors
    print("\nBase Model approach:")
    print("  1. Load PDB")
    print("  2. Generate scattering vectors from hkl")
    print("  3. Cache with scattering factors")
    print("  4. Calculate structure factors in reciprocal space")
    
    # ModelFT: No scattering vectors needed
    print("\nModelFT approach:")
    print("  1. Load PDB")
    print("  2. Build ITC92 parametrization (no hkl needed)")
    print("  3. Cache xyz, B-factors, and ITC92 A,B parameters")
    print("  4. Build electron density in real space")
    print("  5. FFT to get structure factors")
    
    print("\nKey differences:")
    print("  ✓ ModelFT doesn't need scattering vectors")
    print("  ✓ ModelFT caches ITC92 parameters directly")
    print("  ✓ ModelFT works in real space first")
    print("  ✓ ModelFT is purpose-built for FFT-based refinement")


def gpu_acceleration_example():
    """Demonstrate GPU acceleration."""
    print("\n" + "="*60)
    print("GPU ACCELERATION")
    print("="*60)
    
    mol_ft = ModelFT()
    mol_ft.load_pdb_from_file("your_structure.pdb")
    mol_ft.build_ft_cache()
    
    # Move to GPU
    if torch.cuda.is_available():
        print("Moving model to GPU...")
        mol_ft.cuda()
        
        # Build map on GPU
        import time
        start = time.time()
        mol_ft.build_density_map(radius=30)
        gpu_time = time.time() - start
        print(f"GPU density calculation: {gpu_time:.4f} seconds")
        
        # Move back to CPU for saving
        mol_ft.cpu()
        mol_ft.save_map("gpu_density.ccp4")
    else:
        print("CUDA not available, skipping GPU example")


def refinement_workflow_example():
    """Example refinement workflow with ModelFT."""
    print("\n" + "="*60)
    print("REFINEMENT WORKFLOW")
    print("="*60)
    
    mol_ft = ModelFT()
    mol_ft.load_pdb_from_file("your_structure.pdb")
    mol_ft.build_ft_cache()
    
    # Set up optimizer on cached parameters
    params_to_refine = []
    
    # Refine coordinates
    if mol_ft._cached_iso_xyz_ft is not None:
        params_to_refine.append(mol_ft._cached_iso_xyz_ft)
    if mol_ft._cached_aniso_xyz_ft is not None:
        params_to_refine.append(mol_ft._cached_aniso_xyz_ft)
    
    # Optionally refine B-factors
    if mol_ft._cached_iso_b_ft is not None:
        params_to_refine.append(mol_ft._cached_iso_b_ft)
    
    print(f"\nRefining {len(params_to_refine)} parameter tensors")
    print(f"  Total parameters: {sum(p.numel() for p in params_to_refine)}")
    
    # Create optimizer
    optimizer = torch.optim.LBFGS(params_to_refine, lr=0.01)
    
    # Refinement loop (pseudo-code)
    def closure():
        optimizer.zero_grad()
        
        # Build density map
        mol_ft.build_density_map(radius=30)
        
        # Compute loss (would compare with observed data)
        # loss = compute_loss(mol_ft.map, observed_map)
        # loss.backward()
        
        # return loss
        return torch.tensor(0.0)  # Placeholder
    
    # optimizer.step(closure)
    
    print("\nRefinement workflow:")
    print("  1. Build FT cache")
    print("  2. Set up optimizer on cached tensors")
    print("  3. In each iteration:")
    print("     - Build density map from cached params")
    print("     - FFT to get structure factors")
    print("     - Compute loss vs observed")
    print("     - Backpropagate")
    print("     - Update parameters")


def complete_example():
    """Complete example from loading to saving."""
    print("\n" + "="*60)
    print("COMPLETE EXAMPLE")
    print("="*60)
    
    # 1. Load
    print("\n1. Loading PDB file...")
    mol_ft = ModelFT()
    mol_ft.load_pdb_from_file("your_structure.pdb", strip_H=True)
    
    # 2. Setup grids (already done in load_pdb_from_file)
    print("\n2. Grids already set up")
    print(f"   Grid shape: {mol_ft.map.shape}")
    print(f"   Voxel size: {mol_ft.voxel_size}")
    
    # 3. Build cache
    print("\n3. Building FT cache...")
    mol_ft.build_ft_cache()
    
    # 4. Build density
    print("\n4. Building electron density map...")
    mol_ft.build_density_map(radius=30)
    
    # 5. Statistics
    print("\n5. Map statistics:")
    stats = mol_ft.get_map_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")
    
    # 6. Save
    print("\n6. Saving map...")
    mol_ft.save_map("final_density.ccp4")
    
    print("\n✓ Complete!")


if __name__ == "__main__":
    print("ModelFT Usage Examples")
    print("=" * 60)
    print("\nNote: Replace 'your_structure.pdb' with an actual PDB file")
    print("      to run these examples.\n")
    
    try:
        # Run examples
        # basic_usage_example()
        # advanced_caching_example()
        parametrization_example()
        comparison_with_base_model()
        # gpu_acceleration_example()
        # refinement_workflow_example()
        # complete_example()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure to provide a valid PDB file path.")
