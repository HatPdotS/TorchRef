#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python
#SBATCH --job-name=test_scaler_basic
#SBATCH -o /das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/scaler_analytical/test_basic.out
#SBATCH -c 16

"""
Basic test for AnalyticalScaler class.

This tests the fundamental functionality of the analytical scaling implementation:
1. Loading data and model
2. Creating solvent mask
3. Computing analytical scales
4. Applying scales via forward()
"""

import torch
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, '/das/work/p17/p17490/Peter/Library/multicopy_refinement')

from torchref.model_ft import ModelFT
from torchref.Data import ReflectionData
from torchref.solvent import SolventModel
from torchref.scaler_analytical import AnalyticalScaler
from torchref.scaler import Scaler
from torchref.math_torch import rfactor


def test_basic_initialization():
    """Test basic initialization of AnalyticalScaler."""
    print("\n" + "="*80)
    print("TEST 1: Basic Initialization")
    print("="*80)
    
    # Load test data
    pdb_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/tubulin/dark.pdb'
    mtz_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/tubulin/dark.mtz'
    
    print(f"\nLoading PDB: {pdb_path}")
    print(f"Loading MTZ: {mtz_path}")
    
    # Create ModelFT
    model = ModelFT(max_res=1.5, verbose=1)
    model.load_pdb_from_file(pdb_path)
    print(f"✓ Model loaded: {len(model.pdb)} atoms")
    
    # Load reflection data
    data = ReflectionData(verbose=1)
    data.load_from_mtz(mtz_path)
    data = data.filter_by_resolution(d_min=1.5, d_max=50.0)
    print(f"✓ Data loaded: {len(data.hkl)} reflections")
    
    # Create solvent model
    solvent = SolventModel(model, verbose=1)
    print(f"✓ Solvent mask created")
    
    # Create analytical scaler
    print("\nInitializing AnalyticalScaler...")
    scaler = AnalyticalScaler(
        model_ft=model,
        reflection_data=data,
        solvent_model=solvent,
        n_bins=20,
        verbose=2
    )
    print(f"✓ AnalyticalScaler initialized")
    
    # Check that scales were computed
    assert scaler.kmask_per_ref is not None, "kmask_per_ref not computed"
    assert scaler.K_per_ref is not None, "K_per_ref not computed"
    assert len(scaler.kmask_per_ref) == len(data.hkl), "Wrong number of k_mask values"
    assert len(scaler.K_per_ref) == len(data.hkl), "Wrong number of K values"
    
    print(f"\n✓ All checks passed!")
    print(f"  k_mask computed: {len(scaler.kmask_per_ref)} values")
    print(f"  K computed: {len(scaler.K_per_ref)} values")
    
    return scaler, model, data, solvent


def test_forward_pass(scaler, model, data):
    """Test forward pass (applying scales)."""
    print("\n" + "="*80)
    print("TEST 2: Forward Pass (Applying Scales)")
    print("="*80)
    
    # Test forward with default hkl
    print("\nApplying scales to structure factors...")
    F_model = scaler.forward()
    
    print(f"✓ Forward pass successful")
    print(f"  Input reflections: {len(data.hkl)}")
    print(f"  Output F_model: {F_model.shape}")
    
    # Check output shape
    assert F_model.shape[0] == len(data.hkl), "Wrong output shape"
    assert torch.is_complex(F_model), "Output should be complex"
    
    # Check that values are reasonable
    F_model_amp = torch.abs(F_model)
    print(f"\n  F_model amplitude statistics:")
    print(f"    Mean: {F_model_amp.mean():.2f}")
    print(f"    Std:  {F_model_amp.std():.2f}")
    print(f"    Min:  {F_model_amp.min():.2f}")
    print(f"    Max:  {F_model_amp.max():.2f}")
    
    # Compare with F_obs
    _, F_obs, _, _ = data()
    print(f"\n  F_obs amplitude statistics:")
    print(f"    Mean: {F_obs.mean():.2f}")
    print(f"    Std:  {F_obs.std():.2f}")
    print(f"    Min:  {F_obs.min():.2f}")
    print(f"    Max:  {F_obs.max():.2f}")
    
    # Compute R-factor
    R_factor = torch.sum(torch.abs(F_model_amp - F_obs)) / torch.sum(F_obs)
    print(f"\n  R-factor: {R_factor:.4f}")
    
    print(f"\n✓ All checks passed!")
    
    return F_model


def test_forward_with_custom_hkl(scaler, model, data):
    """Test forward pass with custom hkl."""
    print("\n" + "="*80)
    print("TEST 3: Forward Pass with Custom HKL")
    print("="*80)
    
    # Select a subset of reflections
    subset_size = min(1000, len(data.hkl))
    indices = torch.randperm(len(data.hkl))[:subset_size]
    hkl_subset = data.hkl[indices]
    
    print(f"\nApplying scales to {subset_size} random reflections...")
    F_model_subset = scaler.forward(hkl=hkl_subset)
    
    print(f"✓ Forward pass with custom hkl successful")
    print(f"  Input reflections: {len(hkl_subset)}")
    print(f"  Output F_model: {F_model_subset.shape}")
    
    # Check output shape
    assert F_model_subset.shape[0] == len(hkl_subset), "Wrong output shape"
    
    print(f"\n✓ All checks passed!")
    
    return F_model_subset


def test_statistics(scaler):
    """Test statistics printing."""
    print("\n" + "="*80)
    print("TEST 4: Scaling Statistics")
    print("="*80)
    
    # Get statistics
    stats = scaler.get_scaling_statistics()
    
    print(f"\nStatistics dictionary keys:")
    for key in stats.keys():
        if key != 'bin_info':
            print(f"  {key}: {stats[key]}")
    
    # Print formatted statistics
    scaler.print_statistics()
    
    print(f"✓ Statistics test passed!")
    
    return stats


def test_bin_information(scaler):
    """Test bin-level information."""
    print("\n" + "="*80)
    print("TEST 5: Resolution Bin Information")
    print("="*80)
    
    bin_info = scaler.bin_info
    
    print(f"\nBin-by-bin breakdown:")
    print(f"{'Bin':>4} {'d_min':>8} {'d_max':>8} {'N_ref':>8} {'k_mask':>10} {'K':>10}")
    print("-" * 60)
    
    for i in range(len(bin_info['kmask_values'])):
        d_min = bin_info['resolution_min'][i]
        d_max = bin_info['resolution_max'][i]
        n_ref = bin_info['n_reflections'][i]
        kmask = bin_info['kmask_values'][i]
        K = bin_info['K_values'][i]
        
        print(f"{i+1:4d} {d_min:8.2f} {d_max:8.2f} {n_ref:8d} {kmask:10.4f} {K:10.4f}")
    
    print(f"\n✓ Bin information test passed!")


def run_all_tests():
    """Run all tests in sequence."""
    print("\n" + "#"*80)
    print("# ANALYTICAL SCALER TEST SUITE")
    print("#"*80)
    
    try:
        # Test 1: Initialization
        scaler, model, data, solvent = test_basic_initialization()
        
        # Test 2: Forward pass
        F_model = test_forward_pass(scaler, model, data)
        
        # Test 3: Forward with custom hkl
        F_model_subset = test_forward_with_custom_hkl(scaler, model, data)
        
        # Test 4: Statistics
        stats = test_statistics(scaler)
        
        
        # Test 5: Bin information
        test_bin_information(scaler)
        
        # Test 6: R-factor validation
        test_rfactor_validation(scaler)
        
        print("\n" + "#"*80)
        print("# ALL TESTS PASSED!")
        print("#"*80 + "\n")
        
    except Exception as e:
        print("\n" + "#"*80)
        print("# TEST FAILED!")
        print("#"*80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def test_rfactor_validation(scaler):
    """Test R-factor computation and comparison with old scaler."""
    print("\n" + "="*80)
    print("Test 6: R-factor Validation")
    print("="*80)
    
    # Get structure factors (complex)
    F_calc_complex = scaler.model_ft(scaler.hkl)
    F_mask = scaler.solvent_model.get_rec_solvent(scaler.hkl)
    _, F_obs, _, _ = scaler.reflection_data()
    
    # Apply analytical scaling
    print(f"\nTesting analytical scaler...")
    F_analytical = scaler.forward(F_calc_complex)
    F_analytical_amp = torch.abs(F_analytical)
    analytical_r = rfactor(F_obs, F_analytical_amp)
    
    print(f"  R-factor: {analytical_r:.4f}")
    print(f"  F_model mean: {F_analytical_amp.mean():.2f} (F_obs mean: {F_obs.mean():.2f})")
    
    # Compare with old Scaler class
    print(f"\nTesting old scaler for comparison...")
    old_scaler = Scaler(F_calc_complex, scaler.reflection_data, nbins=20, verbose=0)
    
    with torch.no_grad():
        F_old_scaled = old_scaler.forward(F_calc_complex)
        F_old_scaled_amp = torch.abs(F_old_scaled)
        old_scaler_r = rfactor(F_obs, F_old_scaled_amp)
    
    print(f"  R-factor: {old_scaler_r:.4f}")
    print(f"  F_model mean: {F_old_scaled_amp.mean():.2f}")
    
    # Calculate relative performance
    if analytical_r < old_scaler_r:
        improvement = (old_scaler_r - analytical_r) / old_scaler_r * 100
        print(f"\n✓ Analytical scaler is {improvement:.1f}% BETTER than old scaler")
    elif analytical_r > old_scaler_r:
        degradation = (analytical_r - old_scaler_r) / old_scaler_r * 100
        if degradation > 10:
            print(f"\n⚠ WARNING: Analytical scaler is {degradation:.1f}% WORSE than old scaler")
        else:
            print(f"\n✓ Analytical scaler performance is comparable (within {degradation:.1f}%)")
    else:
        print(f"\n✓ Analytical and old scaler have identical performance")
    
    # Print summary
    print(f"\n" + "-"*80)
    print(f"Summary:")
    print(f"  Analytical scaler: R = {analytical_r:.4f}")
    print(f"  Old scaler:        R = {old_scaler_r:.4f}")
    print(f"  Difference:        ΔR = {analytical_r - old_scaler_r:.4f}")
    print(f"-"*80)
    
    # Quality check
    if analytical_r < 0.3:
        print("\n✓ R-factor is in excellent range (< 0.3)")
    elif analytical_r < 0.5:
        print("\n✓ R-factor is in acceptable range (< 0.5)")
    else:
        print(f"\n⚠ WARNING: R-factor is high ({analytical_r:.4f})")
    
    print("\n✓ R-factor validation completed")


if __name__ == "__main__":
    run_all_tests()

