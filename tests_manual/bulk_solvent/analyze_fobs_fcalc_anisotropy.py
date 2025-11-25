"""
Analyze anisotropic differences between F_obs and F_calc.

This script will:
1. Calculate |F_obs| - |F_calc| for each reflection
2. Analyze how these differences vary with crystallographic direction
3. Plot difference patterns in reciprocal space
4. Identify systematic anisotropic discrepancies
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from torchref.base_refinement import Refinement
from torchref.math_functions.math_torch import get_scattering_vectors

def analyze_anisotropic_differences(instance, title="Anisotropic Difference Analysis"):
    """
    Analyze directional patterns in F_obs - F_calc differences.
    """
    print(f"Performing {title}...")
    
    with torch.no_grad():
        # Get reflection data
        hkl, F_obs, sigma_F_obs, rfree_flags = instance.reflection_data()
        
        # Calculate F_calc
        instance.get_Fcalc()
        F_calc = instance.F_calc
        
        # Calculate differences
        F_obs_mag = torch.abs(F_obs)
        F_calc_mag = torch.abs(F_calc)
        diff = F_obs_mag - F_calc_mag
        
        # Get scattering vectors in reciprocal space
        s_vectors = get_scattering_vectors(hkl, instance.model.cell)
        s_vectors = s_vectors.cpu().numpy()
        hkl = hkl.cpu().numpy()
        diff = diff.cpu().numpy()
        
        # Calculate resolution
        s_mag = np.linalg.norm(s_vectors, axis=1) / 2.0
        resolution = 1.0 / (2.0 * s_mag)
        
        # Use only work set for analysis
        work_mask = (rfree_flags == 0).cpu().numpy()
        hkl_work = hkl[work_mask]
        s_vectors_work = s_vectors[work_mask]
        diff_work = diff[work_mask]
        resolution_work = resolution[work_mask]
        
        print(f"Using {len(diff_work)} work reflections for analysis")
    
    # Create comprehensive anisotropy plots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 3D scatter plot of differences in reciprocal space
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    scatter = ax1.scatter(hkl_work[:, 0], hkl_work[:, 1], hkl_work[:, 2], 
                         c=diff_work, cmap='RdBu_r', s=1, alpha=0.6,
                         vmin=np.percentile(diff_work, 5), 
                         vmax=np.percentile(diff_work, 95))
    ax1.set_xlabel('h')
    ax1.set_ylabel('k') 
    ax1.set_zlabel('l')
    ax1.set_title('3D Difference Map\n(F_obs - F_calc)')
    plt.colorbar(scatter, ax=ax1, shrink=0.5)
    
    # 2. Directional analysis - project differences onto principal directions
    ax2 = fig.add_subplot(2, 3, 2)
    
    # Find principal directions of anisotropy by fitting ellipsoid to difference data
    # Weight by |difference| to emphasize systematic effects
    weights = np.abs(diff_work)
    weights = weights / weights.max()  # normalize
    
    # Weighted covariance of scattering vectors
    weighted_s = s_vectors_work * weights[:, np.newaxis]
    cov_matrix = np.cov(weighted_s.T)
    eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
    
    # Project differences onto principal eigenvectors
    projections = s_vectors_work @ eigenvecs
    
    # Plot difference vs projection onto most anisotropic direction
    most_aniso_dir = np.argmax(eigenvals)
    proj_main = projections[:, most_aniso_dir]
    
    ax2.scatter(proj_main, diff_work, alpha=0.3, s=1)
    ax2.set_xlabel(f'Projection on most anisotropic direction\n(eigenvalue: {eigenvals[most_aniso_dir]:.4f})')
    ax2.set_ylabel('F_obs - F_calc')
    ax2.set_title('Difference vs Principal Direction')
    ax2.grid(True)
    
    # Add trend line
    z = np.polyfit(proj_main, diff_work, 1)
    p = np.poly1d(z)
    ax2.plot(sorted(proj_main), p(sorted(proj_main)), "r--", alpha=0.8, 
             label=f'Slope: {z[0]:.3f}')
    ax2.legend()
    
    # 3. Angular analysis - differences vs angle from principal directions
    ax3 = fig.add_subplot(2, 3, 3)
    
    # Calculate angles between scattering vectors and principal eigenvector
    s_unit = s_vectors_work / np.linalg.norm(s_vectors_work, axis=1)[:, np.newaxis]
    main_eigenvec = eigenvecs[:, most_aniso_dir]
    angles = np.arccos(np.clip(np.abs(s_unit @ main_eigenvec), 0, 1)) * 180 / np.pi
    
    # Bin by angle and plot mean differences
    angle_bins = np.linspace(0, 90, 19)
    bin_centers = (angle_bins[:-1] + angle_bins[1:]) / 2
    bin_means = []
    bin_stds = []
    bin_counts = []
    
    for i in range(len(angle_bins)-1):
        mask = (angles >= angle_bins[i]) & (angles < angle_bins[i+1])
        if mask.sum() > 10:  # Need enough reflections
            bin_means.append(diff_work[mask].mean())
            bin_stds.append(diff_work[mask].std() / np.sqrt(mask.sum()))
            bin_counts.append(mask.sum())
        else:
            bin_means.append(0)
            bin_stds.append(0)
            bin_counts.append(0)
    
    ax3.errorbar(bin_centers, bin_means, yerr=bin_stds, marker='o', capsize=3)
    ax3.set_xlabel('Angle from principal anisotropic direction (°)')
    ax3.set_ylabel('Mean F_obs - F_calc')
    ax3.set_title('Angular Dependence of Differences')
    ax3.grid(True)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # 4. Resolution-dependent anisotropy
    ax4 = fig.add_subplot(2, 3, 4)
    
    # Color by projection onto main anisotropic direction
    scatter4 = ax4.scatter(resolution_work, diff_work, c=proj_main, 
                          cmap='viridis', alpha=0.3, s=1)
    ax4.set_xlabel('Resolution (Å)')
    ax4.set_ylabel('F_obs - F_calc')
    ax4.set_title('Resolution vs Difference\n(colored by anisotropic direction)')
    ax4.grid(True)
    plt.colorbar(scatter4, ax=ax4)
    
    # 5. Systematic bias analysis - mean difference in resolution shells
    ax5 = fig.add_subplot(2, 3, 5)
    
    res_bins = np.logspace(np.log10(resolution_work.min()), 
                          np.log10(resolution_work.max()), 20)
    res_bin_centers = np.sqrt(res_bins[:-1] * res_bins[1:])  # geometric mean
    res_means = []
    res_stds = []
    
    for i in range(len(res_bins)-1):
        mask = (resolution_work >= res_bins[i]) & (resolution_work < res_bins[i+1])
        if mask.sum() > 0:
            res_means.append(diff_work[mask].mean())
            res_stds.append(diff_work[mask].std() / np.sqrt(mask.sum()))
        else:
            res_means.append(0)
            res_stds.append(0)
    
    ax5.errorbar(res_bin_centers, res_means, yerr=res_stds, marker='o', capsize=3)
    ax5.set_xlabel('Resolution (Å)')
    ax5.set_ylabel('Mean F_obs - F_calc')
    ax5.set_title('Systematic Bias vs Resolution')
    ax5.grid(True)
    ax5.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # 6. HKL plane analysis
    ax6 = fig.add_subplot(2, 3, 6)
    
    # Plot h=0 plane (k vs l, colored by difference)
    h0_mask = hkl_work[:, 0] == 0
    if h0_mask.sum() > 100:
        scatter6 = ax6.scatter(hkl_work[h0_mask, 1], hkl_work[h0_mask, 2], 
                              c=diff_work[h0_mask], cmap='RdBu_r', s=2,
                              vmin=np.percentile(diff_work, 5), 
                              vmax=np.percentile(diff_work, 95))
        ax6.set_xlabel('k')
        ax6.set_ylabel('l')
        ax6.set_title('h=0 plane (k vs l)')
        plt.colorbar(scatter6, ax=ax6)
    else:
        ax6.text(0.5, 0.5, 'Not enough h=0 reflections', 
                transform=ax6.transAxes, ha='center', va='center')
        ax6.set_title('h=0 plane (insufficient data)')
    
    plt.tight_layout()
    plt.suptitle(f'{title}\nAnisotropic Difference Analysis', fontsize=16, y=0.98)
    plt.show()
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("ANISOTROPIC DIFFERENCE ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Mean difference: {diff_work.mean():.3f}")
    print(f"RMS difference: {np.sqrt(np.mean(diff_work**2)):.3f}")
    print(f"Principal anisotropy eigenvalues: {eigenvals}")
    print(f"Anisotropy ratio: {eigenvals.max()/eigenvals.min():.2f}")
    
    # Test for systematic anisotropy
    correlation = np.corrcoef(proj_main, diff_work)[0, 1]
    print(f"Correlation with main anisotropic direction: {correlation:.3f}")
    
    if abs(correlation) > 0.1:
        print("⚠️  SIGNIFICANT anisotropic pattern detected!")
        print("   Consider applying anisotropic B-factor correction")
    elif abs(correlation) > 0.05:
        print("⚠️  Moderate anisotropic pattern detected")
    else:
        print("✓ No strong anisotropic pattern detected")
    
    return eigenvals, eigenvecs, correlation

def compare_before_after_anisotropy():
    """Compare anisotropic differences before and after anisotropic correction."""
    pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/refinement/dark.pdb'
    mtz = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/refinement/dark.mtz'
    cif = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/elbow.AZO.dark_pdb.001.cif'
    
    print("Loading refinement instance...")
    instance = Refinement(mtz, pdb, cif=cif, verbose=0)
    
    # Analysis before anisotropic correction
    print("\n" + "="*60)
    print("BEFORE ANISOTROPIC CORRECTION")
    print("="*60)
    eigenvals_before, eigenvecs_before, corr_before = analyze_anisotropic_differences(
        instance, "Before Anisotropic Correction")
    
    # Apply anisotropic correction
    print("\nSetting up and optimizing anisotropic correction...")
    instance.scaler.setup_anisotropy_correction()
    
    # Optimize anisotropic parameters
    optimizer = torch.optim.LBFGS([instance.scaler.U], lr=0.1, max_iter=50)
    
    def closure():
        optimizer.zero_grad()
        hkl, F_obs, _, rfree_flags = instance.reflection_data()
        F_calc = instance.model(hkl)
        F_calc_scaled = instance.scaler(F_calc)
        
        work_mask = rfree_flags == 0
        F_obs_work = F_obs[work_mask]
        F_calc_work = F_calc_scaled[work_mask]
        
        loss = (torch.abs(torch.abs(F_obs_work) - torch.abs(F_calc_work))).sum() / torch.abs(F_obs_work).sum()
        loss.backward()
        return loss
    
    optimizer.step(closure)
    
    U_optimized = instance.scaler.U.detach().cpu().numpy()
    print(f"Optimized U parameters: {U_optimized}")
    
    # Analysis after anisotropic correction
    print("\n" + "="*60)
    print("AFTER ANISOTROPIC CORRECTION")
    print("="*60)
    eigenvals_after, eigenvecs_after, corr_after = analyze_anisotropic_differences(
        instance, "After Anisotropic Correction")
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"Anisotropy ratio before: {eigenvals_before.max()/eigenvals_before.min():.2f}")
    print(f"Anisotropy ratio after:  {eigenvals_after.max()/eigenvals_after.min():.2f}")
    print(f"Correlation before: {corr_before:+.3f}")
    print(f"Correlation after:  {corr_after:+.3f}")
    print(f"Improvement: {abs(corr_before) - abs(corr_after):+.3f}")
    
    if abs(corr_after) < abs(corr_before):
        print("✓ Anisotropic correction IMPROVED the systematic differences")
    else:
        print("✗ Anisotropic correction did NOT improve systematic differences")

if __name__ == '__main__':
    compare_before_after_anisotropy()