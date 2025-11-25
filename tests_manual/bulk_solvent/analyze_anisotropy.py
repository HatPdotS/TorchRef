"""
Visualize anisotropic B-factor tensors and their effects on diffraction.

This script will:
1. Extract the anisotropic U tensor from your refinement
2. Convert to B-factor tensor (B = 8π²U)
3. Plot the anisotropy ellipsoid
4. Show resolution-dependent anisotropic scaling effects
5. Compare before/after anisotropic correction
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from torchref.base_refinement import Refinement

def U_to_matrix(U_params):
    """Convert 6 anisotropic parameters to 3x3 symmetric matrix."""
    U11, U22, U33, U12, U13, U23 = U_params
    return np.array([
        [U11, U12, U13],
        [U12, U22, U23],
        [U13, U23, U33]
    ])

def plot_anisotropy_ellipsoid(U_matrix, title="Anisotropy Ellipsoid", ax=None):
    """Plot the anisotropy ellipsoid from U matrix."""
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
    
    # Convert U to B-factor matrix (B = 8π²U)
    B_matrix = 8 * np.pi**2 * U_matrix
    
    # Eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eigh(B_matrix)
    
    # Create ellipsoid
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    
    # Scale factor for visualization
    scale = 100  # Adjust as needed
    
    # Ellipsoid in principal axes
    a, b, c = np.sqrt(np.abs(eigenvals)) * scale
    x_sphere = a * np.outer(np.cos(u), np.sin(v))
    y_sphere = b * np.outer(np.sin(u), np.sin(v))
    z_sphere = c * np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Rotate to crystal axes
    for i in range(len(x_sphere)):
        for j in range(len(x_sphere[0])):
            point = np.array([x_sphere[i,j], y_sphere[i,j], z_sphere[i,j]])
            rotated = eigenvecs @ point
            x_sphere[i,j], y_sphere[i,j], z_sphere[i,j] = rotated
    
    # Plot ellipsoid
    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.6, cmap='viridis')
    
    # Plot principal axes
    origin = np.array([0, 0, 0])
    for i, (val, vec) in enumerate(zip(eigenvals, eigenvecs.T)):
        ax.quiver(*origin, *(vec * np.sqrt(abs(val)) * scale * 1.5), 
                 color=['red', 'green', 'blue'][i], linewidth=3, 
                 label=f'Axis {i+1}: B={val:.4f}')
    
    ax.set_xlabel('a* direction')
    ax.set_ylabel('b* direction')
    ax.set_zlabel('c* direction')
    ax.set_title(title)
    ax.legend()
    
    # Print eigenvalues
    print(f"\n{title}:")
    print(f"Principal B-factors: {eigenvals}")
    print(f"Anisotropy ratio: {eigenvals.max()/eigenvals.min():.2f}")
    
    return ax

def plot_resolution_dependent_anisotropy(hkl, cell, U_matrix, title="Anisotropic Scaling vs Resolution"):
    """Plot how anisotropic scaling varies with resolution."""
    from torchref.math_functions.math_torch import get_scattering_vectors
    
    # Get scattering vectors
    s_vectors = get_scattering_vectors(torch.tensor(hkl, dtype=torch.float32), cell)
    s_vectors = s_vectors.numpy()
    
    # Calculate resolution
    s_mag = np.linalg.norm(s_vectors, axis=1) / 2.0  # sin(θ)/λ
    resolution = 1.0 / (2.0 * s_mag)
    
    # Calculate anisotropic B-factor: B·s²
    B_matrix = 8 * np.pi**2 * U_matrix
    s_squared_aniso = np.sum((s_vectors @ B_matrix) * s_vectors, axis=1)
    
    # Anisotropic scaling factor
    aniso_scale = np.exp(-s_squared_aniso)
    
    # Plot scaling vs resolution
    plt.figure(figsize=(12, 8))
    
    # Main plot: scaling vs resolution
    plt.subplot(2, 2, 1)
    plt.scatter(resolution, aniso_scale, alpha=0.3, s=1)
    plt.xlabel('Resolution (Å)')
    plt.ylabel('Anisotropic scaling factor')
    plt.title('Anisotropic Scaling vs Resolution')
    plt.grid(True)
    
    # Histogram of scaling factors
    plt.subplot(2, 2, 2)
    plt.hist(aniso_scale, bins=50, alpha=0.7)
    plt.xlabel('Anisotropic scaling factor')
    plt.ylabel('Count')
    plt.title('Distribution of Scaling Factors')
    plt.grid(True)
    
    # Resolution bins analysis
    plt.subplot(2, 2, 3)
    res_bins = np.logspace(np.log10(resolution.min()), np.log10(resolution.max()), 20)
    bin_centers = (res_bins[:-1] + res_bins[1:]) / 2
    bin_means = []
    bin_stds = []
    
    for i in range(len(res_bins)-1):
        mask = (resolution >= res_bins[i]) & (resolution < res_bins[i+1])
        if mask.sum() > 0:
            bin_means.append(aniso_scale[mask].mean())
            bin_stds.append(aniso_scale[mask].std())
        else:
            bin_means.append(1.0)
            bin_stds.append(0.0)
    
    plt.errorbar(bin_centers, bin_means, yerr=bin_stds, marker='o')
    plt.xlabel('Resolution (Å)')
    plt.ylabel('Mean anisotropic scaling')
    plt.title('Binned Anisotropic Effects')
    plt.grid(True)
    
    # Directional analysis
    plt.subplot(2, 2, 4)
    # Project onto principal B-factor directions
    eigenvals, eigenvecs = np.linalg.eigh(B_matrix)
    projections = s_vectors @ eigenvecs
    
    # Color by dominant direction
    dominant_dir = np.argmax(np.abs(projections), axis=1)
    colors = ['red', 'green', 'blue']
    
    for i in range(3):
        mask = dominant_dir == i
        if mask.sum() > 0:
            plt.scatter(resolution[mask], aniso_scale[mask], 
                       c=colors[i], alpha=0.3, s=1, 
                       label=f'Dir {i+1} (B={eigenvals[i]:.4f})')
    
    plt.xlabel('Resolution (Å)')
    plt.ylabel('Anisotropic scaling factor')
    plt.title('Scaling by Principal Direction')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.suptitle(title, y=1.02, fontsize=14)
    plt.show()
    
    return bin_centers, bin_means, bin_stds

def analyze_anisotropy_effects():
    """Main analysis function."""
    pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/refinement/dark.pdb'
    mtz = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/refinement/dark.mtz'
    cif = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/elbow.AZO.dark_pdb.001.cif'
    
    print("Loading refinement instance...")
    instance = Refinement(mtz, pdb, cif=cif, verbose=0)
    
    # Get HKL data
    hkl, F_obs, sigma_F_obs, rfree_flags = instance.reflection_data()
    hkl = hkl.cpu().numpy()
    
    print("Setting up anisotropic correction...")
    instance.scaler.setup_anisotropy_correction()
    
    # Initial (zero) anisotropy
    U_initial = np.zeros(6)
    U_initial_matrix = U_to_matrix(U_initial)
    
    print("Optimizing anisotropic parameters...")
    # Optimize anisotropic parameters
    optimizer = torch.optim.LBFGS([instance.scaler.U], lr=0.1, max_iter=50)
    
    def closure():
        optimizer.zero_grad()
        F_calc = instance.model(torch.tensor(hkl, dtype=torch.float32))
        F_calc_scaled = instance.scaler(F_calc)
        
        work_mask = rfree_flags == 0
        F_obs_work = F_obs[work_mask]
        F_calc_work = F_calc_scaled[work_mask]
        
        loss = (torch.abs(torch.abs(F_obs_work) - torch.abs(F_calc_work))).sum() / torch.abs(F_obs_work).sum()
        loss.backward()
        return loss
    
    optimizer.step(closure)
    
    # Get optimized anisotropy
    U_optimized = instance.scaler.U.detach().cpu().numpy()
    U_optimized_matrix = U_to_matrix(U_optimized)
    
    print(f"\nOptimized U parameters: {U_optimized}")
    
    # Create comparison plots
    fig = plt.figure(figsize=(15, 6))
    
    # Plot ellipsoids
    ax1 = fig.add_subplot(121, projection='3d')
    plot_anisotropy_ellipsoid(U_initial_matrix, "Initial (Isotropic)", ax1)
    
    ax2 = fig.add_subplot(122, projection='3d')
    plot_anisotropy_ellipsoid(U_optimized_matrix, "Optimized (Anisotropic)", ax2)
    
    plt.tight_layout()
    plt.show()
    
    # Plot resolution-dependent effects
    plot_resolution_dependent_anisotropy(hkl, instance.model.cell, U_optimized_matrix, 
                                       "Optimized Anisotropic Scaling Effects")
    
    # Compare R-factors
    with torch.no_grad():
        # Without anisotropy
        instance.scaler.U.data = torch.zeros(6)
        r_work_iso, r_free_iso = instance.get_rfactor()
        
        # With anisotropy
        instance.scaler.U.data = torch.tensor(U_optimized)
        r_work_aniso, r_free_aniso = instance.get_rfactor()
    
    print(f"\n{'='*50}")
    print("R-FACTOR COMPARISON")
    print(f"{'='*50}")
    print(f"Without anisotropy: Rwork={r_work_iso:.4f}, Rfree={r_free_iso:.4f}")
    print(f"With anisotropy:    Rwork={r_work_aniso:.4f}, Rfree={r_free_aniso:.4f}")
    print(f"Improvement:        ΔRwork={r_work_iso-r_work_aniso:+.4f}, ΔRfree={r_free_iso-r_free_aniso:+.4f}")
    
    return U_optimized_matrix, instance

if __name__ == '__main__':
    U_matrix, refinement_instance = analyze_anisotropy_effects()