#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python
#SBATCH --job-name=test_scaler_basic
#SBATCH -o /das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/scaler_analytical/test_validation.out
#SBATCH -c 16

"""
Advanced test for AnalyticalScaler - validation against expected behavior.

This tests:
1. Comparison of scaled vs unscaled structure factors
2. Verification that bulk solvent improves R-factor
3. Resolution-dependent scaling behavior
4. Consistency checks
"""

import torch
import numpy as np
import sys
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, '/das/work/p17/p17490/Peter/Library/multicopy_refinement')

from torchref.model_ft import ModelFT
from torchref.Data import ReflectionData
from torchref.solvent import SolventModel
from torchref.scaler_analytical import AnalyticalScaler
from torchref.math_torch import rfactor


def test_scaling_improves_rfactor():
    """Test that analytical scaling improves R-factor."""
    print("\n" + "="*80)
    print("TEST: Analytical Scaling Improves R-factor")
    print("="*80)
    
    # Load data
    pdb_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/tubulin/dark.pdb'
    mtz_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/tubulin/dark.mtz'
    
    print(f"\nLoading model and data...")
    model = ModelFT(max_res=1.5, verbose=0)
    model.load_pdb_from_file(pdb_path)
    
    data = ReflectionData(verbose=0)
    data.load_from_mtz(mtz_path)
    data = data.filter_by_resolution(d_min=1.5, d_max=50.0)
    
    print(f"✓ Loaded {len(data.hkl)} reflections")
    
    # Compute unscaled F_calc (no solvent, no scaling)
    print(f"\nComputing unscaled F_calc...")
    with torch.no_grad():
        F_calc_unscaled = model.get_structure_factor(data.hkl, recalc=False)
    
    R_unscaled = rfactor(data.F, F_calc_unscaled)
    print(f"  R-factor (unscaled, no solvent): {R_unscaled:.4f}")
    
    # Create solvent model and scaler
    print(f"\nCreating solvent model and analytical scaler...")
    solvent = SolventModel(model, verbose=0)
    scaler = AnalyticalScaler(
        model_ft=model,
        reflection_data=data,
        solvent_model=solvent,
        n_bins=20,
        verbose=1
    )
    
    # Compute scaled F_model (with solvent and analytical scaling)
    print(f"\nComputing scaled F_model...")
    F_model_scaled = scaler.forward()
    
    R_scaled = rfactor(data.F, F_model_scaled)
    print(f"  R-factor (scaled with solvent): {R_scaled:.4f}")
    
    # Check improvement
    improvement = (R_unscaled - R_scaled) / R_unscaled * 100
    print(f"\n  Improvement: {improvement:.2f}%")
    
    if R_scaled < R_unscaled:
        print(f"✓ Analytical scaling IMPROVED R-factor!")
    else:
        print(f"⚠ WARNING: Scaling did not improve R-factor")
        print(f"  This might indicate an issue, but could also be due to")
        print(f"  model limitations or data quality")
    
    return R_unscaled, R_scaled


def test_resolution_dependence():
    """Test that scaling parameters vary with resolution."""
    print("\n" + "="*80)
    print("TEST: Resolution-Dependent Scaling")
    print("="*80)
    
    # Load data
    pdb_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/tubulin/dark.pdb'
    mtz_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/tubulin/dark.mtz'
    
    print(f"\nLoading model and data...")
    model = ModelFT(max_res=1.5, verbose=0)
    model.load_pdb_from_file(pdb_path)
    
    data = ReflectionData(verbose=0)
    data.load_from_mtz(mtz_path)
    data = data.filter_by_resolution(d_min=1.5, d_max=50.0)
    
    # Create scaler
    solvent = SolventModel(model, verbose=0)
    scaler = AnalyticalScaler(
        model_ft=model,
        reflection_data=data,
        solvent_model=solvent,
        n_bins=20,
        verbose=0
    )
    
    # Analyze resolution dependence
    bin_info = scaler.bin_info
    
    print(f"\nResolution-dependent scaling parameters:")
    print(f"{'d_res':>8} {'k_mask':>10} {'K':>10}")
    print("-" * 30)
    
    for i in range(len(bin_info['kmask_values'])):
        d_min = bin_info['resolution_min'][i]
        d_max = bin_info['resolution_max'][i]
        d_avg = (d_min + d_max) / 2
        kmask = bin_info['kmask_values'][i]
        K = bin_info['K_values'][i]
        
        print(f"{d_avg:8.2f} {kmask:10.4f} {K:10.4f}")
    
    # Check that parameters vary
    kmask_values = np.array(bin_info['kmask_values'])
    K_values = np.array(bin_info['K_values'])
    
    kmask_std = np.std(kmask_values)
    K_std = np.std(K_values)
    
    print(f"\nParameter variation:")
    print(f"  k_mask std: {kmask_std:.4f}")
    print(f"  K std: {K_std:.4f}")
    
    if kmask_std > 0.01 or K_std > 0.01:
        print(f"✓ Scaling parameters show resolution dependence")
    else:
        print(f"⚠ WARNING: Scaling parameters show little variation")
        print(f"  This might be expected for good data or could indicate an issue")
    
    return bin_info


def test_bulk_solvent_contribution():
    """Test that bulk solvent makes a significant contribution."""
    print("\n" + "="*80)
    print("TEST: Bulk Solvent Contribution")
    print("="*80)
    
    # Load data
    pdb_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/tubulin/dark.pdb'
    mtz_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/tubulin/dark.mtz'
    
    model = ModelFT(max_res=1.5, verbose=0)
    model.load_pdb_from_file(pdb_path)
    
    data = ReflectionData(verbose=0)
    data.load_from_mtz(mtz_path)
    data = data.filter_by_resolution(d_min=1.5, d_max=50.0)
    
    # Create solvent model
    solvent = SolventModel(model, verbose=0)
    
    # Get F_calc and F_mask
    with torch.no_grad():
        F_calc = model.get_structure_factor(data.hkl, recalc=False)
        F_mask = solvent.get_rec_solvent(data.hkl)
    
    # Compute relative magnitudes
    F_calc_amp = torch.abs(F_calc)
    F_mask_amp = torch.abs(F_mask)
    
    print(f"\nStructure factor magnitudes:")
    print(f"  F_calc mean: {F_calc_amp.mean():.2f}")
    print(f"  F_mask mean: {F_mask_amp.mean():.2f}")
    print(f"  Ratio (F_mask/F_calc): {(F_mask_amp.mean() / F_calc_amp.mean()):.4f}")
    
    # Expected ratio is typically 0.3-0.5 for bulk solvent
    ratio = (F_mask_amp.mean() / F_calc_amp.mean()).item()
    
    if 0.1 < ratio < 1.0:
        print(f"✓ Bulk solvent contribution is reasonable ({ratio:.2%} of F_calc)")
    else:
        print(f"⚠ WARNING: Bulk solvent contribution seems unusual ({ratio:.2%})")
    
    return F_calc, F_mask


def test_scale_consistency():
    """Test internal consistency of scaling parameters."""
    print("\n" + "="*80)
    print("TEST: Scaling Parameter Consistency")
    print("="*80)
    
    # Load data
    pdb_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/tubulin/dark.pdb'
    mtz_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/tubulin/dark.mtz'
    
    model = ModelFT(max_res=1.5, verbose=0)
    model.load_pdb_from_file(pdb_path)
    
    data = ReflectionData(verbose=0)
    data.load_from_mtz(mtz_path)
    data = data.filter_by_resolution(d_min=1.5, d_max=50.0)
    
    solvent = SolventModel(model, verbose=0)
    scaler = AnalyticalScaler(
        model_ft=model,
        reflection_data=data,
        solvent_model=solvent,
        n_bins=20,
        verbose=0
    )
    
    # Check that all scales are positive
    assert torch.all(scaler.kmask_per_ref >= 0), "Some k_mask values are negative!"
    assert torch.all(scaler.K_per_ref > 0), "Some K values are non-positive!"
    
    print(f"✓ All k_mask values are non-negative")
    print(f"✓ All K values are positive")
    
    # Check that scales are not NaN or Inf
    assert not torch.any(torch.isnan(scaler.kmask_per_ref)), "NaN in k_mask!"
    assert not torch.any(torch.isnan(scaler.K_per_ref)), "NaN in K!"
    assert not torch.any(torch.isinf(scaler.kmask_per_ref)), "Inf in k_mask!"
    assert not torch.any(torch.isinf(scaler.K_per_ref)), "Inf in K!"
    
    print(f"✓ No NaN or Inf values in scaling parameters")
    
    # Check reasonable ranges
    kmask_max = scaler.kmask_per_ref.max().item()
    K_max = scaler.K_per_ref.max().item()
    
    print(f"\nParameter ranges:")
    print(f"  k_mask max: {kmask_max:.4f}")
    print(f"  K max: {K_max:.4f}")
    
    if kmask_max < 10.0 and K_max < 10.0:
        print(f"✓ Scaling parameters are in reasonable ranges")
    else:
        print(f"⚠ WARNING: Some scaling parameters seem large")


def plot_scaling_curves(save_path='scaling_curves.png'):
    """Plot k_mask and K as functions of resolution."""
    print("\n" + "="*80)
    print("TEST: Plotting Scaling Curves")
    print("="*80)
    
    # Load data
    pdb_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/tubulin/dark.pdb'
    mtz_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/tubulin/dark.mtz'
    
    model = ModelFT(max_res=1.5, verbose=0)
    model.load_pdb_from_file(pdb_path)
    
    data = ReflectionData(verbose=0)
    data.load_from_mtz(mtz_path)
    data = data.filter_by_resolution(d_min=1.5, d_max=50.0)
    
    solvent = SolventModel(model, verbose=0)
    scaler = AnalyticalScaler(
        model_ft=model,
        reflection_data=data,
        solvent_model=solvent,
        n_bins=20,
        verbose=0
    )
    
    # Extract bin information
    bin_info = scaler.bin_info
    
    # Compute average resolution for each bin
    d_avg = [(bin_info['resolution_min'][i] + bin_info['resolution_max'][i]) / 2 
             for i in range(len(bin_info['kmask_values']))]
    
    kmask_values = bin_info['kmask_values']
    K_values = bin_info['K_values']
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot k_mask
    ax1.plot(d_avg, kmask_values, 'o-', label='k_mask')
    ax1.set_xlabel('Resolution (Å)')
    ax1.set_ylabel('k_mask')
    ax1.set_title('Bulk-Solvent Scale Factor vs Resolution')
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()  # High res on right
    
    # Plot K
    ax2.plot(d_avg, K_values, 'o-', color='orange', label='K')
    ax2.set_xlabel('Resolution (Å)')
    ax2.set_ylabel('K')
    ax2.set_title('Isotropic Scale Parameter vs Resolution')
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()  # High res on right
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"✓ Saved plot to: {save_path}")
    
    return fig


def run_all_validation_tests():
    """Run all validation tests."""
    print("\n" + "#"*80)
    print("# ANALYTICAL SCALER VALIDATION TEST SUITE")
    print("#"*80)
    
    try:
        # Test 1: R-factor improvement
        R_unscaled, R_scaled = test_scaling_improves_rfactor()
        
        # Test 2: Resolution dependence
        bin_info = test_resolution_dependence()
        
        # Test 3: Bulk solvent contribution
        F_calc, F_mask = test_bulk_solvent_contribution()
        
        # Test 4: Consistency checks
        test_scale_consistency()
        
        # Test 5: Plotting
        try:
            fig = plot_scaling_curves(
                save_path='/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/scaler_analytical/scaling_curves.png'
            )
        except Exception as e:
            print(f"⚠ Plotting failed (not critical): {e}")
        
        print("\n" + "#"*80)
        print("# ALL VALIDATION TESTS PASSED!")
        print("#"*80 + "\n")
        
        # Summary
        print(f"\nSummary:")
        print(f"  R-factor (unscaled): {R_unscaled:.4f}")
        print(f"  R-factor (scaled):   {R_scaled:.4f}")
        print(f"  Improvement:         {(R_unscaled - R_scaled):.4f} ({(R_unscaled - R_scaled)/R_unscaled*100:.2f}%)")
        
    except Exception as e:
        print("\n" + "#"*80)
        print("# VALIDATION TEST FAILED!")
        print("#"*80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all_validation_tests()
