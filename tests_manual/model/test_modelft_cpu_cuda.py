"""
Test cpu() and cuda() methods for ModelFT class.
"""
import sys
import torch

sys.path.insert(0, '/das/work/p17/p17490/Peter/Library/multicopy_refinement')

from torchref.model_ft import ModelFT

print("="*80)
print("Testing ModelFT cpu() and cuda() methods")
print("="*80)

# Load a model
print("\n1. Loading model...")
model = ModelFT()
model.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/dark.pdb')
print(f"   ✓ Model loaded with {len(model.pdb)} atoms")

# Check initial device
print("\n2. Checking initial device (CPU)...")
print(f"   xyz device: {model.xyz.device}")
if hasattr(model, 'A') and model.A is not None:
    print(f"   A device: {model.A.device}")
if hasattr(model, 'B') and model.B is not None:
    print(f"   B device: {model.B.device}")

# Test cuda() if available
if torch.cuda.is_available():
    print("\n3. Moving to CUDA...")
    result = model.cuda()
    print(f"   cuda() returned: {type(result)}")
    print(f"   Is same object: {result is model}")
    print(f"   xyz device: {model.xyz.device}")
    if hasattr(model, 'A') and model.A is not None:
        print(f"   A device: {model.A.device}")
        assert model.A.is_cuda, "A should be on CUDA"
    if hasattr(model, 'B') and model.B is not None:
        print(f"   B device: {model.B.device}")
        assert model.B.is_cuda, "B should be on CUDA"
    print("   ✓ All tensors moved to CUDA")
    
    # Test method chaining
    print("\n4. Testing method chaining...")
    model2 = model.cpu().cuda()
    print(f"   Method chaining worked: {model2 is model}")
    print(f"   xyz device: {model2.xyz.device}")
    assert model2.xyz.is_cuda, "xyz should be on CUDA after chaining"
    print("   ✓ Method chaining works")
    
    # Move back to CPU
    print("\n5. Moving back to CPU...")
    result = model.cpu()
    print(f"   cpu() returned: {type(result)}")
    print(f"   Is same object: {result is model}")
    print(f"   xyz device: {model.xyz.device}")
    if hasattr(model, 'A') and model.A is not None:
        print(f"   A device: {model.A.device}")
        assert not model.A.is_cuda, "A should be on CPU"
    if hasattr(model, 'B') and model.B is not None:
        print(f"   B device: {model.B.device}")
        assert not model.B.is_cuda, "B should be on CPU"
    print("   ✓ All tensors moved to CPU")
    
else:
    print("\n3. CUDA not available, skipping GPU tests")

# Test with grids
print("\n6. Testing with grids setup...")
model.setup_grid(max_res=2.0)
print(f"   Grid shape: {model.map.shape}")
print(f"   real_space_grid device: {model.real_space_grid.device}")
print(f"   map device: {model.map.device}")

if torch.cuda.is_available():
    print("\n7. Moving model with grids to CUDA...")
    model.cuda()
    print(f"   real_space_grid device: {model.real_space_grid.device}")
    print(f"   map device: {model.map.device}")
    print(f"   inv_frac_matrix device: {model.inv_frac_matrix.device}")
    assert model.real_space_grid.is_cuda, "Grid should be on CUDA"
    assert model.map.is_cuda, "Map should be on CUDA"
    print("   ✓ Grid tensors moved to CUDA")
    
    print("\n8. Moving back to CPU...")
    model.cpu()
    print(f"   real_space_grid device: {model.real_space_grid.device}")
    print(f"   map device: {model.map.device}")
    assert not model.real_space_grid.is_cuda, "Grid should be on CPU"
    assert not model.map.is_cuda, "Map should be on CPU"
    print("   ✓ Grid tensors moved to CPU")

print("\n" + "="*80)
print("✅ ALL TESTS PASSED!")
print("="*80)
