"""
Compare density maps from ModelFT vs CCTBX using correlation.
"""
import torch
import numpy as np
from torchref.model_ft import ModelFT
import gemmi

print("=" * 80)
print("COMPARING ModelFT vs CCTBX MAPS")
print("=" * 80)

# Build map with ModelFT
print("\n" + "-" * 80)
print("Building map with ModelFT...")
print("-" * 80)
M = ModelFT()   
M.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_FT/dark.pdb')
M.max_res = 0.8  # Match CCTBX resolution
M.setup_grid()

M.build_density_map(apply_symmetry=False, radius=10)
map_modelft = M.map.clone()

print(f"ModelFT map:")
print(f"  Shape: {map_modelft.shape}")
print(f"  Sum: {map_modelft.sum():.2f}")
print(f"  Mean: {map_modelft.mean():.6f}")
print(f"  Std: {map_modelft.std():.6f}")
print(f"  Max: {map_modelft.max():.4f}")
print(f"  Min: {map_modelft.min():.4f}")

# Save ModelFT map
modelft_file = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_FT/dark_modelft.ccp4'
M.save_map(modelft_file)
print(f"\n✓ Saved ModelFT map to: {modelft_file}")

# Load CCTBX map (generated from structure factors)
print("\n" + "-" * 80)
print("Loading CCTBX-derived map...")
print("-" * 80)

# First, calculate map from structure factors using gemmi
mtz_file = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_FT/dark_fcalc.mtz'

try:
    # Read MTZ file
    mtz = gemmi.read_mtz_file(mtz_file)
    print(f"MTZ file loaded: {mtz_file}")
    print(f"  Cell: {mtz.cell.parameters}")
    print(f"  Space group: {mtz.spacegroup.hm}")
    print(f"  Columns: {[col.label for col in mtz.columns]}")
    
    # Get F-model column
    if 'F-model' in [col.label for col in mtz.columns]:
        # Calculate map from structure factors with standard sampling
        grid = mtz.transform_f_phi_to_map('F-model', 'PHIF-model', sample_rate=3)
        
        print(f"\nCCTBX map (from FFT):")
        print(f"  Shape: {(grid.nu, grid.nv, grid.nw)}")
        
        # Convert to numpy array
        map_cctbx = np.array(grid, copy=False)
        
        print(f"  Sum: {map_cctbx.sum():.2f}")
        print(f"  Mean: {map_cctbx.mean():.6f}")
        print(f"  Std: {map_cctbx.std():.6f}")
        print(f"  Max: {map_cctbx.max():.4f}")
        print(f"  Min: {map_cctbx.min():.4f}")
        
        # Save CCTBX map for visual inspection
        ccp4 = gemmi.Ccp4Map()
        ccp4.grid = grid
        ccp4.update_ccp4_header(2, True)
        cctbx_file = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_FT/dark_cctbx.ccp4'
        ccp4.write_ccp4_map(cctbx_file)
        print(f"\n✓ Saved CCTBX map to: {cctbx_file}")
        
        # Compare map shapes
        print("\n" + "-" * 80)
        print("Map comparison:")
        print("-" * 80)
        
        print(f"ModelFT shape: {map_modelft.shape}")
        print(f"CCTBX shape:   {map_cctbx.shape}")
        
        if map_modelft.shape != map_cctbx.shape:
            print("\n⚠ WARNING: Map shapes don't match!")
            print("  This is expected when using different sampling rates.")
            print(f"  ModelFT: {map_modelft.shape}")
            print(f"  CCTBX: {map_cctbx.shape}")
            print("\n  Interpolating CCTBX map to ModelFT grid using trilinear interpolation...")
            
            # Use PyTorch's grid_sample for interpolation
            import torch.nn.functional as F
            
            # Convert CCTBX map to torch
            map_cctbx_torch = torch.from_numpy(map_cctbx).unsqueeze(0).unsqueeze(0).float()  # (1, 1, nx, ny, nz)
            
            # Create sampling grid for ModelFT shape
            # Coordinates in [-1, 1] range for grid_sample
            nx_new, ny_new, nz_new = map_modelft.shape
            nx_old, ny_old, nz_old = map_cctbx.shape
            
            x = torch.linspace(-1, 1, nx_new)
            y = torch.linspace(-1, 1, ny_new)
            z = torch.linspace(-1, 1, nz_new)
            
            grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing='ij')
            sampling_grid = torch.stack([grid_x, grid_y, grid_z], dim=-1).unsqueeze(0)  # (1, nx, ny, nz, 3)
            
            # Interpolate
            map_cctbx_interp = F.grid_sample(
                map_cctbx_torch,
                sampling_grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True
            ).squeeze().numpy()
            
            print(f"  Interpolated CCTBX map shape: {map_cctbx_interp.shape}")
            print(f"  Interpolated CCTBX map sum: {map_cctbx_interp.sum():.2f}")
            
            # Use interpolated map for comparison
            map_cctbx = map_cctbx_interp
            
        else:
            print("\n✓ Map shapes match!")
        
        # Calculate correlation
        # Convert to numpy for correlation calculation
        map_modelft_np = map_modelft.detach().numpy()
        
        # Flatten arrays
        flat_modelft = map_modelft_np.flatten()
        flat_cctbx = map_cctbx.flatten()
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(flat_modelft, flat_cctbx)[0, 1]
        
        # Calculate R-factor (crystallographic R-value for maps)
        r_factor = np.sum(np.abs(flat_modelft - flat_cctbx)) / np.sum(np.abs(flat_cctbx))
        
        # Calculate normalized differences
        mean_modelft = flat_modelft.mean()
        mean_cctbx = flat_cctbx.mean()
        std_modelft = flat_modelft.std()
        std_cctbx = flat_cctbx.std()
        
        # Z-score normalized correlation
        flat_modelft_z = (flat_modelft - mean_modelft) / (std_modelft + 1e-10)
        flat_cctbx_z = (flat_cctbx - mean_cctbx) / (std_cctbx + 1e-10)
        correlation_z = np.corrcoef(flat_modelft_z, flat_cctbx_z)[0, 1]
        
        print("\n" + "=" * 80)
        print("CORRELATION ANALYSIS")
        print("=" * 80)
        
        print(f"\nPearson correlation coefficient: {correlation:.6f}")
        print(f"  Interpretation:")
        print(f"    > 0.99: Excellent match")
        print(f"    > 0.95: Very good match")
        print(f"    > 0.90: Good match")
        print(f"    > 0.80: Reasonable match")
        print(f"    < 0.80: Poor match")
        
        if correlation > 0.99:
            status = "✓ EXCELLENT"
        elif correlation > 0.95:
            status = "✓ VERY GOOD"
        elif correlation > 0.90:
            status = "✓ GOOD"
        elif correlation > 0.80:
            status = "○ REASONABLE"
        else:
            status = "✗ POOR"
        
        print(f"  Status: {status}")
        
        print(f"\nZ-score normalized correlation: {correlation_z:.6f}")
        print(f"  (Removes mean/std differences)")
        
        print(f"\nR-factor (map): {r_factor:.6f}")
        print(f"  (Lower is better, typical crystallographic R ~ 0.20-0.25)")
        
        # Calculate point-by-point statistics
        diff = flat_modelft - flat_cctbx
        print(f"\nPoint-by-point differences:")
        print(f"  Mean difference: {diff.mean():.6f}")
        print(f"  Std of differences: {diff.std():.6f}")
        print(f"  Max absolute difference: {np.abs(diff).max():.6f}")
        print(f"  RMS difference: {np.sqrt((diff**2).mean()):.6f}")
        
        # Check if there's a scale factor difference
        scale = np.sum(flat_modelft * flat_cctbx) / np.sum(flat_cctbx**2)
        print(f"\nScale factor (ModelFT/CCTBX): {scale:.6f}")
        print(f"  If close to 1.0, maps are on same scale")
        print(f"  If not, one map may need rescaling")
        
        # Calculate correlation after scaling
        flat_modelft_scaled = flat_modelft / scale
        correlation_scaled = np.corrcoef(flat_modelft_scaled, flat_cctbx)[0, 1]
        print(f"\nCorrelation after scaling: {correlation_scaled:.6f}")
        
        # Sample some points to compare
        print("\n" + "-" * 80)
        print("Sample point comparison (first 10 non-zero voxels):")
        print("-" * 80)
        
        nonzero_indices = np.where(flat_cctbx > 0.001)[0][:10]
        print(f"{'Index':<8} {'ModelFT':<12} {'CCTBX':<12} {'Ratio':<10} {'Diff':<10}")
        print("-" * 60)
        for idx in nonzero_indices:
            ratio = flat_modelft[idx] / (flat_cctbx[idx] + 1e-10)
            diff = flat_modelft[idx] - flat_cctbx[idx]
            print(f"{idx:<8} {flat_modelft[idx]:<12.6f} {flat_cctbx[idx]:<12.6f} {ratio:<10.4f} {diff:<10.6f}")
        
        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        
        if correlation > 0.95:
            print("✓ Maps show very good agreement!")
            print("  The ModelFT implementation produces results consistent with CCTBX.")
        elif correlation > 0.90:
            print("○ Maps show good agreement.")
            print("  Minor differences may be due to:")
            print("    - Different B-factor conventions")
            print("    - Numerical precision")
            print("    - Grid sampling differences")
        else:
            print("✗ Maps show significant differences!")
            print("  Possible issues:")
            print("    - B-factor scaling (factor of 2 error?)")
            print("    - Different normalization conventions")
            print("    - Incorrect scattering factors")
            
        if abs(scale - 1.0) > 0.1:
            print(f"\n⚠ Scale factor is {scale:.4f}, suggesting a systematic difference.")
            if abs(scale - 2.0) < 0.1:
                print("  → Likely factor of 2 error in B-factor formula")
            elif abs(scale - 0.5) < 0.1:
                print("  → Likely factor of 2 error in opposite direction")
            else:
                print(f"  → Unexpected scale factor: {scale:.4f}")
    
    else:
        print("✗ ERROR: F-model column not found in MTZ file")
        
except Exception as e:
    print(f"✗ ERROR loading CCTBX map: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Comparison complete!")
print("=" * 80)
