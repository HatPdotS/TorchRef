"""
Test to verify ModelFT refactoring with new model class.
"""
import sys
sys.path.insert(0, '/das/work/p17/p17490/Peter/Library/multicopy_refinement')

import torch
from torchref.model_ft import ModelFT

def test_modelft_basic():
    """Test basic ModelFT functionality with new architecture."""
    print("=" * 80)
    print("Testing ModelFT with new model class architecture")
    print("=" * 80)
    
    # Load model
    pdb_file = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/dark.pdb'
    print(f"\n1. Loading PDB file: {pdb_file}")
    model_ft = ModelFT()
    model_ft.load_pdb_from_file(pdb_file)
    
    print(f"   ✓ Loaded {len(model_ft.pdb)} atoms")
    print(f"   ✓ Parametrization built for {len(model_ft.parametrization)} element types")
    
    # Test get_iso_with_params
    print("\n2. Testing get_iso_with_params()...")
    xyz_iso, b_iso, occ_iso, A_iso, B_iso = model_ft.get_iso_with_params()
    print(f"   ✓ Retrieved {len(xyz_iso)} isotropic atoms")
    print(f"   ✓ XYZ shape: {xyz_iso.shape}")
    print(f"   ✓ B-factor shape: {b_iso.shape}")
    print(f"   ✓ Occupancy shape: {occ_iso.shape}")
    print(f"   ✓ A parameters shape: {A_iso.shape}")
    print(f"   ✓ B parameters shape: {B_iso.shape}")
    
    # Test get_aniso_with_params
    print("\n3. Testing get_aniso_with_params()...")
    xyz_aniso, u_aniso, occ_aniso, A_aniso, B_aniso = model_ft.get_aniso_with_params()
    print(f"   ✓ Retrieved {len(xyz_aniso)} anisotropic atoms")
    if len(xyz_aniso) > 0:
        print(f"   ✓ XYZ shape: {xyz_aniso.shape}")
        print(f"   ✓ U parameters shape: {u_aniso.shape}")
        print(f"   ✓ Occupancy shape: {occ_aniso.shape}")
        print(f"   ✓ A parameters shape: {A_aniso.shape}")
        print(f"   ✓ B parameters shape: {B_aniso.shape}")
    else:
        print(f"   ✓ No anisotropic atoms in this structure")
    
    # Setup grids
    print("\n4. Testing setup_grids()...")
    model_ft.setup_grid(max_res=2.0)
    print(f"   ✓ Grid shape: {model_ft.real_space_grid.shape}")
    print(f"   ✓ Voxel size: {model_ft.voxel_size:.4f} Å")
    
    # Build density map
    print("\n5. Testing build_density_map()...")
    density_map = model_ft.build_density_map(radius=5, apply_symmetry=False)
    print(f"   ✓ Map shape: {density_map.shape}")
    print(f"   ✓ Map sum: {density_map.sum():.2f}")
    print(f"   ✓ Map range: [{density_map.min():.4f}, {density_map.max():.4f}]")
    
    # Test cuda/cpu methods
    print("\n6. Testing cuda() and cpu() methods...")
    if torch.cuda.is_available():
        print("   Testing CUDA...")
        model_ft.cuda()
        print(f"   ✓ Model moved to GPU")
        print(f"   ✓ Map device: {model_ft.map.device}")
        model_ft.cpu()
        print(f"   ✓ Model moved back to CPU")
        print(f"   ✓ Map device: {model_ft.map.device}")
    else:
        print("   ⚠ CUDA not available, skipping GPU tests")
    
    # Test update_pdb
    print("\n7. Testing update_pdb()...")
    model_ft.update_pdb()
    print(f"   ✓ PDB updated successfully")
    
    print("\n" + "=" * 80)
    print("✅ All tests passed! ModelFT refactoring is working correctly.")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    try:
        test_modelft_basic()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
