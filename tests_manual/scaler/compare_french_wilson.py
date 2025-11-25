#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u
#SBATCH -c 16
#SBATCH -o /das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/scaler/compare_french_wilson.out

from torchref.model_ft import ModelFT
from torchref.scaler import Scaler
from torchref.Data import ReflectionData
import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from torchref.math_functions.math_torch import french_wilson_conversion
import torch
import reciprocalspaceship as rs

mtz = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/scaler/dark.mtz'


import torch
import torch.nn.functional as F


# Acentric lookup tables from French-Wilson supplement (1978)
AC_ZJ = torch.tensor([
    0.226, 0.230, 0.235, 0.240, 0.246, 0.251, 0.257, 0.263, 0.270,
    0.276, 0.283, 0.290, 0.298, 0.306, 0.314, 0.323, 0.332, 0.341, 0.351,
    0.362, 0.373, 0.385, 0.397, 0.410, 0.424, 0.439, 0.454, 0.470, 0.487,
    0.505, 0.525, 0.545, 0.567, 0.590, 0.615, 0.641, 0.668, 0.698, 0.729,
    0.762, 0.798, 0.835, 0.875, 0.917, 0.962, 1.009, 1.059, 1.112, 1.167,
    1.226, 1.287, 1.352, 1.419, 1.490, 1.563, 1.639, 1.717, 1.798, 1.882,
    1.967, 2.055, 2.145, 2.236, 2.329, 2.422, 2.518, 2.614, 2.710, 2.808,
    2.906, 3.004
], dtype=torch.float32)

AC_ZJ_SD = torch.tensor([
    0.217, 0.221, 0.226, 0.230, 0.235, 0.240, 0.245, 0.250, 0.255,
    0.261, 0.267, 0.273, 0.279, 0.286, 0.292, 0.299, 0.307, 0.314, 0.322,
    0.330, 0.339, 0.348, 0.357, 0.367, 0.377, 0.387, 0.398, 0.409, 0.421,
    0.433, 0.446, 0.459, 0.473, 0.488, 0.503, 0.518, 0.535, 0.551, 0.568,
    0.586, 0.604, 0.622, 0.641, 0.660, 0.679, 0.698, 0.718, 0.737, 0.757,
    0.776, 0.795, 0.813, 0.831, 0.848, 0.865, 0.881, 0.895, 0.909, 0.921,
    0.933, 0.943, 0.953, 0.961, 0.968, 0.974, 0.980, 0.984, 0.988, 0.991,
    0.994, 0.996
], dtype=torch.float32)

AC_ZF = torch.tensor([
    0.423, 0.428, 0.432, 0.437, 0.442, 0.447, 0.453, 0.458, 0.464,
    0.469, 0.475, 0.482, 0.488, 0.495, 0.502, 0.509, 0.516, 0.524, 0.532,
    0.540, 0.549, 0.557, 0.567, 0.576, 0.586, 0.597, 0.608, 0.619, 0.631,
    0.643, 0.656, 0.670, 0.684, 0.699, 0.714, 0.730, 0.747, 0.765, 0.783,
    0.802, 0.822, 0.843, 0.865, 0.887, 0.911, 0.935, 0.960, 0.987, 1.014,
    1.042, 1.070, 1.100, 1.130, 1.161, 1.192, 1.224, 1.257, 1.289, 1.322,
    1.355, 1.388, 1.421, 1.454, 1.487, 1.519, 1.551, 1.583, 1.615, 1.646,
    1.676, 1.706
], dtype=torch.float32)

AC_ZF_SD = torch.tensor([
    0.216, 0.218, 0.220, 0.222, 0.224, 0.226, 0.229, 0.231, 0.234,
    0.236, 0.239, 0.241, 0.244, 0.247, 0.250, 0.253, 0.256, 0.259, 0.262,
    0.266, 0.269, 0.272, 0.276, 0.279, 0.283, 0.287, 0.291, 0.295, 0.298,
    0.302, 0.307, 0.311, 0.315, 0.319, 0.324, 0.328, 0.332, 0.337, 0.341,
    0.345, 0.349, 0.353, 0.357, 0.360, 0.364, 0.367, 0.369, 0.372, 0.374,
    0.375, 0.376, 0.377, 0.377, 0.377, 0.376, 0.374, 0.372, 0.369, 0.366,
    0.362, 0.358, 0.353, 0.348, 0.343, 0.338, 0.332, 0.327, 0.321, 0.315,
    0.310, 0.304
], dtype=torch.float32)

# Centric lookup tables from French-Wilson supplement (1978)
C_ZJ = torch.tensor([
    0.114, 0.116, 0.119, 0.122, 0.124, 0.127, 0.130, 0.134, 0.137,
    0.141, 0.145, 0.148, 0.153, 0.157, 0.162, 0.166, 0.172, 0.177, 0.183,
    0.189, 0.195, 0.202, 0.209, 0.217, 0.225, 0.234, 0.243, 0.253, 0.263,
    0.275, 0.287, 0.300, 0.314, 0.329, 0.345, 0.363, 0.382, 0.402, 0.425,
    0.449, 0.475, 0.503, 0.534, 0.567, 0.603, 0.642, 0.684, 0.730, 0.779,
    0.833, 0.890, 0.952, 1.018, 1.089, 1.164, 1.244, 1.327, 1.416, 1.508,
    1.603, 1.703, 1.805, 1.909, 2.015, 2.123, 2.233, 2.343, 2.453, 2.564,
    2.674, 2.784, 2.894, 3.003, 3.112, 3.220, 3.328, 3.435, 3.541, 3.647,
    3.753, 3.962
], dtype=torch.float32)

C_ZJ_SD = torch.tensor([
    0.158, 0.161, 0.165, 0.168, 0.172, 0.176, 0.179, 0.184, 0.188,
    0.192, 0.197, 0.202, 0.207, 0.212, 0.218, 0.224, 0.230, 0.236, 0.243,
    0.250, 0.257, 0.265, 0.273, 0.282, 0.291, 0.300, 0.310, 0.321, 0.332,
    0.343, 0.355, 0.368, 0.382, 0.397, 0.412, 0.428, 0.445, 0.463, 0.481,
    0.501, 0.521, 0.543, 0.565, 0.589, 0.613, 0.638, 0.664, 0.691, 0.718,
    0.745, 0.773, 0.801, 0.828, 0.855, 0.881, 0.906, 0.929, 0.951, 0.971,
    0.989, 1.004, 1.018, 1.029, 1.038, 1.044, 1.049, 1.052, 1.054, 1.054,
    1.053, 1.051, 1.049, 1.047, 1.044, 1.041, 1.039, 1.036, 1.034, 1.031,
    1.029, 1.028
], dtype=torch.float32)

C_ZF = torch.tensor([
    0.269, 0.272, 0.276, 0.279, 0.282, 0.286, 0.289, 0.293, 0.297,
    0.301, 0.305, 0.309, 0.314, 0.318, 0.323, 0.328, 0.333, 0.339, 0.344,
    0.350, 0.356, 0.363, 0.370, 0.377, 0.384, 0.392, 0.400, 0.409, 0.418,
    0.427, 0.438, 0.448, 0.460, 0.471, 0.484, 0.498, 0.512, 0.527, 0.543,
    0.560, 0.578, 0.597, 0.618, 0.639, 0.662, 0.687, 0.713, 0.740, 0.769,
    0.800, 0.832, 0.866, 0.901, 0.938, 0.976, 1.016, 1.057, 1.098, 1.140,
    1.183, 1.227, 1.270, 1.313, 1.356, 1.398, 1.439, 1.480, 1.519, 1.558,
    1.595, 1.632, 1.667, 1.701, 1.735, 1.767, 1.799, 1.829, 1.859, 1.889,
    1.917, 1.945
], dtype=torch.float32)

C_ZF_SD = torch.tensor([
    0.203, 0.205, 0.207, 0.209, 0.211, 0.214, 0.216, 0.219, 0.222,
    0.224, 0.227, 0.230, 0.233, 0.236, 0.239, 0.243, 0.246, 0.250, 0.253,
    0.257, 0.261, 0.265, 0.269, 0.273, 0.278, 0.283, 0.288, 0.293, 0.298,
    0.303, 0.309, 0.314, 0.320, 0.327, 0.333, 0.340, 0.346, 0.353, 0.361,
    0.368, 0.375, 0.383, 0.390, 0.398, 0.405, 0.413, 0.420, 0.427, 0.433,
    0.440, 0.445, 0.450, 0.454, 0.457, 0.459, 0.460, 0.460, 0.458, 0.455,
    0.451, 0.445, 0.438, 0.431, 0.422, 0.412, 0.402, 0.392, 0.381, 0.370,
    0.360, 0.349, 0.339, 0.330, 0.321, 0.312, 0.304, 0.297, 0.290, 0.284,
    0.278, 0.272
], dtype=torch.float32)


def interpolate_table(h: torch.Tensor, table: torch.Tensor, h_min: float = -4.0) -> torch.Tensor:
    """
    Interpolate values from French-Wilson lookup table.
    
    Args:
        h: Normalized parameter (tensor of any shape)
        table: Lookup table tensor (1D)
        h_min: Minimum h value (default -4.0)
    
    Returns:
        Interpolated values (same shape as h)
    """
    # Map h to table index: point = 10.0 * (h + 4.0)
    point = 10.0 * (h + h_min)
    point = torch.clamp(point, 0.0, len(table) - 1.001)  # Clamp to valid range
    
    # Linear interpolation
    pt_1 = point.long()
    pt_2 = torch.clamp(pt_1 + 1, max=len(table) - 1)
    delta = point - pt_1.float()
    
    # Interpolate: (1-delta)*table[pt_1] + delta*table[pt_2]
    val_1 = table[pt_1]
    val_2 = table[pt_2]
    result = (1.0 - delta) * val_1 + delta * val_2
    
    return result


def french_wilson_acentric(
    I: torch.Tensor,
    sigma_I: torch.Tensor,
    mean_intensity: torch.Tensor,
    h_min: float = -4.0,
    i_sig_min: float = -3.7
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    French-Wilson conversion for acentric reflections.
    
    Args:
        I: Measured intensities (any shape)
        sigma_I: Standard deviations of intensities (same shape as I)
        mean_intensity: Mean intensity for each reflection's resolution bin (same shape as I)
        h_min: Minimum h value for rejection (default -4.0)
        i_sig_min: Minimum I/sigma_I for rejection (default -3.7 = h_min + 0.3)
    
    Returns:
        F: Structure factor amplitudes (same shape as I)
        sigma_F: Standard deviations of F (same shape as I)
        valid_mask: Boolean mask indicating valid (not rejected) reflections
    """
    device = I.device
    dtype = I.dtype
    
    # Move lookup tables to same device and dtype
    ac_zj = AC_ZJ.to(device=device, dtype=dtype)
    ac_zj_sd = AC_ZJ_SD.to(device=device, dtype=dtype)
    ac_zf = AC_ZF.to(device=device, dtype=dtype)
    ac_zf_sd = AC_ZF_SD.to(device=device, dtype=dtype)
    
    # Compute normalized parameter h
    h = (I / sigma_I) - (sigma_I / mean_intensity)
    
    # Rejection criterion
    i_over_sig = I / sigma_I
    valid_mask = (i_over_sig >= i_sig_min) & (h >= h_min)
    
    # Initialize outputs
    F = torch.zeros_like(I)
    sigma_F = torch.zeros_like(I)
    
    # Case 1: Small h (h < 3.0) - use lookup tables
    small_h_mask = valid_mask & (h < 3.0)
    if small_h_mask.any():
        h_small = h[small_h_mask]
        sigma_I_small = sigma_I[small_h_mask]
        
        # Interpolate from tables
        zf = interpolate_table(h_small, ac_zf)
        zf_sd = interpolate_table(h_small, ac_zf_sd)
        
        F[small_h_mask] = zf * torch.sqrt(sigma_I_small)
        sigma_F[small_h_mask] = zf_sd * torch.sqrt(sigma_I_small)
    
    # Case 2: Large h (h >= 3.0) - use asymptotic formula
    large_h_mask = valid_mask & (h >= 3.0)
    if large_h_mask.any():
        h_large = h[large_h_mask]
        sigma_I_large = sigma_I[large_h_mask]
        
        J = h_large * sigma_I_large
        F_large = torch.sqrt(J)
        sigma_F_large = 0.5 * (sigma_I_large / F_large)
        
        F[large_h_mask] = F_large
        sigma_F[large_h_mask] = sigma_F_large
    
    return F, sigma_F, valid_mask


def french_wilson_centric(
    I: torch.Tensor,
    sigma_I: torch.Tensor,
    mean_intensity: torch.Tensor,
    h_min: float = -4.0,
    i_sig_min: float = -3.7
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    French-Wilson conversion for centric reflections.
    
    Args:
        I: Measured intensities (any shape)
        sigma_I: Standard deviations of intensities (same shape as I)
        mean_intensity: Mean intensity for each reflection's resolution bin (same shape as I)
        h_min: Minimum h value for rejection (default -4.0)
        i_sig_min: Minimum I/sigma_I for rejection (default -3.7 = h_min + 0.3)
    
    Returns:
        F: Structure factor amplitudes (same shape as I)
        sigma_F: Standard deviations of F (same shape as I)
        valid_mask: Boolean mask indicating valid (not rejected) reflections
    """
    device = I.device
    dtype = I.dtype
    
    # Move lookup tables to same device and dtype
    c_zj = C_ZJ.to(device=device, dtype=dtype)
    c_zj_sd = C_ZJ_SD.to(device=device, dtype=dtype)
    c_zf = C_ZF.to(device=device, dtype=dtype)
    c_zf_sd = C_ZF_SD.to(device=device, dtype=dtype)
    
    # Compute normalized parameter h (note factor of 2 for centric!)
    h = (I / sigma_I) - (sigma_I / (2.0 * mean_intensity))
    
    # Rejection criterion
    i_over_sig = I / sigma_I
    valid_mask = (i_over_sig >= i_sig_min) & (h >= h_min)
    
    # Initialize outputs
    F = torch.zeros_like(I)
    sigma_F = torch.zeros_like(I)
    
    # Case 1: Small h (h < 4.0) - use lookup tables
    small_h_mask = valid_mask & (h < 4.0)
    if small_h_mask.any():
        h_small = h[small_h_mask]
        sigma_I_small = sigma_I[small_h_mask]
        
        # Interpolate from tables
        zf = interpolate_table(h_small, c_zf)
        zf_sd = interpolate_table(h_small, c_zf_sd)
        
        F[small_h_mask] = zf * torch.sqrt(sigma_I_small)
        sigma_F[small_h_mask] = zf_sd * torch.sqrt(sigma_I_small)
    
    # Case 2: Large h (h >= 4.0) - use extended asymptotic formula
    large_h_mask = valid_mask & (h >= 4.0)
    if large_h_mask.any():
        h_large = h[large_h_mask]
        sigma_I_large = sigma_I[large_h_mask]
        
        # Extended asymptotic expansion (Phenix extension)
        h_2 = 1.0 / (h_large * h_large)
        h_4 = h_2 * h_2
        h_6 = h_2 * h_4
        
        # Posterior mean of F
        post_F = torch.sqrt(h_large) * (
            1.0 - (3.0/8.0) * h_2 - (87.0/128.0) * h_4 - (2889.0/1024.0) * h_6
        )
        
        # Posterior standard deviation of F
        post_sig_F = torch.sqrt(
            h_large * ((1.0/4.0) * h_2 + (15.0/32.0) * h_4 + (273.0/128.0) * h_6)
        )
        
        F[large_h_mask] = post_F * torch.sqrt(sigma_I_large)
        sigma_F[large_h_mask] = post_sig_F * torch.sqrt(sigma_I_large)
    
    return F, sigma_F, valid_mask


def french_wilson(
    I: torch.Tensor,
    sigma_I: torch.Tensor,
    mean_intensity: torch.Tensor,
    is_centric: torch.Tensor = None,
    h_min: float = -4.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    French-Wilson conversion from intensities to structure factors.
    
    Automatically handles both centric and acentric reflections.
    
    Args:
        I: Measured intensities, shape (...,)
        sigma_I: Standard deviations of intensities, shape (...,)
        mean_intensity: Mean intensity for each reflection's resolution bin, shape (...,)
        is_centric: Boolean mask indicating centric reflections, shape (...,)
                    If None, assumes all reflections are acentric.
        h_min: Minimum h value for rejection (default -4.0)
    
    Returns:
        F: Structure factor amplitudes, shape (...,)
        sigma_F: Standard deviations of F, shape (...,)
        valid_mask: Boolean mask indicating valid (not rejected) reflections, shape (...,)
    
    Example:
        >>> I = torch.tensor([100.0, 5.0, -15.0, 200.0])
        >>> sigma_I = torch.tensor([10.0, 10.0, 10.0, 15.0])
        >>> mean_I = torch.tensor([80.0, 80.0, 80.0, 150.0])
        >>> F, sigma_F, valid = french_wilson(I, sigma_I, mean_I)
        >>> print(f"F = {F}")
        >>> print(f"sigma_F = {sigma_F}")
        >>> print(f"Valid: {valid}")
    """
    i_sig_min = h_min + 0.3
    
    # Initialize outputs
    F = torch.zeros_like(I)
    sigma_F = torch.zeros_like(I)
    valid_mask = torch.zeros_like(I, dtype=torch.bool)
    
    if is_centric is None:
        # All acentric
        F, sigma_F, valid_mask = french_wilson_acentric(
            I, sigma_I, mean_intensity, h_min, i_sig_min
        )
    else:
        # Process acentric reflections
        acentric_mask = ~is_centric
        if acentric_mask.any():
            F_acen, sigma_F_acen, valid_acen = french_wilson_acentric(
                I[acentric_mask],
                sigma_I[acentric_mask],
                mean_intensity[acentric_mask],
                h_min,
                i_sig_min
            )
            F[acentric_mask] = F_acen
            sigma_F[acentric_mask] = sigma_F_acen
            valid_mask[acentric_mask] = valid_acen
        
        # Process centric reflections
        if is_centric.any():
            F_cen, sigma_F_cen, valid_cen = french_wilson_centric(
                I[is_centric],
                sigma_I[is_centric],
                mean_intensity[is_centric],
                h_min,
                i_sig_min
            )
            F[is_centric] = F_cen
            sigma_F[is_centric] = sigma_F_cen
            valid_mask[is_centric] = valid_cen
    
    return F, sigma_F, valid_mask


def estimate_mean_intensity_by_resolution(
    I: torch.Tensor,
    d_spacings: torch.Tensor,
    n_bins: int = 60,
    min_per_bin: int = 10
) -> torch.Tensor:
    """
    Estimate mean intensity for each reflection based on resolution binning.
    
    Args:
        I: Measured intensities, shape (n_reflections,)
        d_spacings: Resolution (d-spacing) for each reflection, shape (n_reflections,)
        n_bins: Number of resolution bins (default 60)
        min_per_bin: Minimum reflections per bin (default 10)
    
    Returns:
        mean_intensity: Estimated mean intensity for each reflection, shape (n_reflections,)
    """
    # Sort by resolution
    sort_idx = torch.argsort(d_spacings, descending=True)
    I_sorted = I[sort_idx]
    d_sorted = d_spacings[sort_idx]
    
    n_reflections = len(I)
    reflections_per_bin = max(min_per_bin, n_reflections // n_bins)
    actual_n_bins = max(1, n_reflections // reflections_per_bin)
    
    # Compute mean intensity for each bin
    mean_I = torch.zeros(n_reflections, dtype=I.dtype, device=I.device)
    
    for i in range(actual_n_bins):
        start_idx = i * reflections_per_bin
        end_idx = min((i + 1) * reflections_per_bin, n_reflections)
        
        bin_mean = I_sorted[start_idx:end_idx].mean()
        mean_I[sort_idx[start_idx:end_idx]] = bin_mean
    
    return mean_I


data = rs.read_mtz(mtz)
data = data.dropna()


I = data['I-obs'].values.astype(float)
SigI = data['SIGI-obs'].values.astype(float)

I_torch = torch.tensor(I, dtype=torch.float32)
SigI_torch = torch.tensor(SigI, dtype=torch.float32)

F, F_sig = french_wilson(I_torch, SigI_torch)

F = F.detach().cpu().numpy()

f = data['F-model'].values.astype(float)


plt.plot(F, f, 'o', alpha=0.5)
plt.xlabel('Fobs my french wilson (multicopy_refinement)')
plt.ylabel('Fobs (phenix)')
plt.savefig('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/scaler/compare_french_wilson.png', dpi=300)
plt.close()

ideal_scale = np.mean(np.log(f) - np.log(F))

F_scaled = F * np.exp(ideal_scale)

plt.plot(F_scaled, f, 'o', alpha=0.5)
plt.xlabel('Fobs my french wilson scaled (multicopy_refinement)')
plt.ylabel('Fobs (phenix)')
plt.savefig('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/scaler/compare_french_wilson_scaled.png', dpi=300)
print(f"Ideal scale factor (log space): {ideal_scale:.4f}")



rfactor = np.sum(np.abs(f - F_scaled)) / np.sum(np.abs(f))
print(f"R-factor after ideal scaling: {rfactor:.4f}")