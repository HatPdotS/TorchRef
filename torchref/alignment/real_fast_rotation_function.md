This is a great question about implementing the Crowther fast rotation function. Let me walk through the key steps and then provide a PyTorch implementation.

## Conceptual Overview

The Crowther approach decomposes the rotation function into:

1. **Spherical harmonic expansion** of both Patterson functions (observed and calculated)
2. **Radial integration** to get coefficients $c_{\ell mm'}$
3. **2D FFT** over Euler angles $(\alpha, \gamma)$ for each $\beta$ section

The key insight is that equation (12) from the paper shows R(α,β,γ) is a Fourier series in α and γ, allowing FFT computation.

## Implementation Strategy

```python
import torch
import torch.fft
from torch import Tensor
import math
from typing import Tuple, Optional

class FastRotationFunction:
    """
    Crowther-style fast rotation function using spherical harmonic decomposition.
    
    The rotation function R(Ω) = ∫ P(r)Q(Ω⁻¹r)dr is computed as:
    1. Expand P and Q in spherical harmonics
    2. Compute radial overlap coefficients c_ℓmm'
    3. Evaluate via 2D FFT for each β section
    """
    
    def __init__(
        self,
        l_max: int = 20,
        n_beta: int = 180,
        n_alpha: int = 360,
        n_gamma: int = 360,
        r_max: float = 30.0,  # Patterson integration radius in Å
        n_radial: int = 100,  # radial integration points
        device: torch.device = None,
    ):
        self.l_max = l_max
        self.n_beta = n_beta
        self.n_alpha = n_alpha
        self.n_gamma = n_gamma
        self.r_max = r_max
        self.n_radial = n_radial
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Precompute Wigner d-matrices for all β values
        self.beta_values = torch.linspace(0, math.pi, n_beta, device=self.device)
        self.d_matrices = self._precompute_wigner_d()
        
    def _precompute_wigner_d(self) -> dict:
        """Precompute Wigner small-d matrices d^ℓ_{m'm}(β) for all β values."""
        d_matrices = {}
        for l in range(self.l_max + 1):
            # Shape: (n_beta, 2l+1, 2l+1) for indices m', m from -l to l
            d_matrices[l] = self._compute_wigner_d_matrix(l, self.beta_values)
        return d_matrices
    
    def _compute_wigner_d_matrix(self, l: int, beta: Tensor) -> Tensor:
        """
        Compute Wigner d-matrix d^l_{m'm}(β) using recurrence relations.
        
        Returns tensor of shape (n_beta, 2l+1, 2l+1)
        """
        n_beta = beta.shape[0]
        size = 2 * l + 1
        d = torch.zeros(n_beta, size, size, device=self.device, dtype=torch.float64)
        
        cos_beta = torch.cos(beta)
        sin_beta = torch.sin(beta)
        cos_half = torch.cos(beta / 2)
        sin_half = torch.sin(beta / 2)
        
        # Use the explicit formula for d^l_{m'm}(β)
        for mp_idx, mp in enumerate(range(-l, l + 1)):
            for m_idx, m in enumerate(range(-l, l + 1)):
                d[:, mp_idx, m_idx] = self._wigner_d_element(l, mp, m, cos_half, sin_half)
        
        return d
    
    def _wigner_d_element(
        self, l: int, mp: int, m: int, 
        cos_half: Tensor, sin_half: Tensor
    ) -> Tensor:
        """
        Compute single element d^l_{m'm}(β) using the formula:
        
        d^l_{m'm}(β) = Σ_s (-1)^{m'-m+s} * 
                       sqrt((l+m')!(l-m')!(l+m)!(l-m)!) /
                       ((l+m-s)!(l-m'-s)!(m'-m+s)!s!) *
                       cos(β/2)^{2l+m-m'-2s} * sin(β/2)^{m'-m+2s}
        """
        result = torch.zeros_like(cos_half)
        
        # Sum limits
        s_min = max(0, m - mp)
        s_max = min(l + m, l - mp)
        
        prefactor = math.sqrt(
            math.factorial(l + mp) * math.factorial(l - mp) *
            math.factorial(l + m) * math.factorial(l - m)
        )
        
        for s in range(s_min, s_max + 1):
            denom = (
                math.factorial(l + m - s) *
                math.factorial(l - mp - s) *
                math.factorial(mp - m + s) *
                math.factorial(s)
            )
            sign = (-1) ** (mp - m + s)
            
            cos_power = 2 * l + m - mp - 2 * s
            sin_power = mp - m + 2 * s
            
            term = sign * prefactor / denom
            term = term * (cos_half ** cos_power) * (sin_half ** sin_power)
            result = result + term
            
        return result

    def compute_spherical_harmonic_coefficients(
        self,
        F_squared: Tensor,  # |F|² or |G|² - shape (N,)
        hkl: Tensor,        # Miller indices - shape (N, 3)
        cell: Tensor,       # Unit cell parameters [a, b, c, α, β, γ]
    ) -> dict:
        """
        Compute spherical harmonic coefficients a_ℓm(r) or b_ℓm(r) from structure factors.
        
        From equation (8): a_ℓm(r) = (4πi^ℓ/V) Σ_s |F(s)|² j_ℓ(2πsr) Y_ℓm*(θ_s, φ_s)
        
        Returns dict mapping ℓ -> tensor of shape (n_radial, 2ℓ+1) for m = -ℓ to ℓ
        """
        # Convert HKL to reciprocal space vectors and spherical coordinates
        s_vectors = self._hkl_to_reciprocal(hkl, cell)  # Shape (N, 3)
        s_mag = torch.norm(s_vectors, dim=1)  # Shape (N,)
        theta_s, phi_s = self._cartesian_to_spherical_angles(s_vectors)  # Each (N,)
        
        # Radial sampling points
        r_values = torch.linspace(0.1, self.r_max, self.n_radial, device=self.device)
        
        # Volume (for normalization)
        V = self._cell_volume(cell)
        
        coefficients = {}
        
        for l in range(self.l_max + 1):
            coeff_l = torch.zeros(self.n_radial, 2 * l + 1, 
                                  device=self.device, dtype=torch.complex128)
            
            # Compute Y_ℓm*(θ_s, φ_s) for all m and all reflections
            # Shape: (N, 2l+1)
            Ylm_conj = self._spherical_harmonics_conj(l, theta_s, phi_s)
            
            # Phase factor i^ℓ
            phase = (1j) ** l
            
            for r_idx, r in enumerate(r_values):
                # Spherical Bessel function j_ℓ(2πsr) for all reflections
                # Shape: (N,)
                j_l = self._spherical_bessel(l, 2 * math.pi * s_mag * r)
                
                # Sum over reflections: Σ_s |F(s)|² j_ℓ(2πsr) Y_ℓm*(θ_s, φ_s)
                # F_squared: (N,), j_l: (N,), Ylm_conj: (N, 2l+1)
                # Result: (2l+1,)
                weighted = F_squared[:, None] * j_l[:, None] * Ylm_conj  # (N, 2l+1)
                coeff_l[r_idx, :] = (4 * math.pi * phase / V) * weighted.sum(dim=0)
            
            coefficients[l] = coeff_l
            
        return coefficients, r_values
    
    def compute_radial_overlap(
        self,
        a_coeffs: dict,  # From observed data
        b_coeffs: dict,  # From model
        r_values: Tensor,
    ) -> dict:
        """
        Compute c_ℓmm' = ∫ a_ℓm(r) b_ℓm'(r) r² dr
        
        Returns dict mapping ℓ -> tensor of shape (2ℓ+1, 2ℓ+1)
        """
        dr = r_values[1] - r_values[0]
        r_squared = r_values ** 2
        
        c_coeffs = {}
        
        for l in range(self.l_max + 1):
            # a_coeffs[l]: (n_radial, 2l+1) 
            # b_coeffs[l]: (n_radial, 2l+1)
            a_l = a_coeffs[l]  # (n_radial, 2l+1)
            b_l = b_coeffs[l]  # (n_radial, 2l+1)
            
            # Integrate: c_ℓmm' = Σ_r a_ℓm(r) * b_ℓm'(r) * r² * dr
            # Shape: (n_radial, 2l+1, 1) * (n_radial, 1, 2l+1) * (n_radial, 1, 1)
            integrand = (a_l[:, :, None] * b_l[:, None, :].conj() * 
                        r_squared[:, None, None])
            
            c_coeffs[l] = integrand.sum(dim=0) * dr  # (2l+1, 2l+1)
            
        return c_coeffs
    
    def compute_rotation_function(
        self,
        F_obs: Tensor,      # Observed |F| - shape (N_obs,)
        hkl_obs: Tensor,    # Observed HKL - shape (N_obs, 3)
        F_calc: Tensor,     # Calculated |F| for model - shape (N_calc,)
        hkl_calc: Tensor,   # Calculated HKL for model - shape (N_calc, 3)
        cell: Tensor,       # Unit cell
    ) -> Tensor:
        """
        Compute the full rotation function R(α, β, γ).
        
        Returns tensor of shape (n_alpha, n_beta, n_gamma)
        """
        # Step 1: Compute spherical harmonic coefficients
        F_obs_sq = F_obs.abs() ** 2
        F_calc_sq = F_calc.abs() ** 2
        
        a_coeffs, r_values = self.compute_spherical_harmonic_coefficients(
            F_obs_sq, hkl_obs, cell
        )
        b_coeffs, _ = self.compute_spherical_harmonic_coefficients(
            F_calc_sq, hkl_calc, cell
        )
        
        # Step 2: Compute radial overlap coefficients
        c_coeffs = self.compute_radial_overlap(a_coeffs, b_coeffs, r_values)
        
        # Step 3: For each β, compute Σ_ℓ c_ℓmm' d^ℓ_mm'(β)
        # Then FFT over (α, γ)
        R = self._evaluate_via_fft(c_coeffs)
        
        return R
    
    def _evaluate_via_fft(self, c_coeffs: dict) -> Tensor:
        """
        Evaluate R(α, β, γ) = Σ_m Σ_m' {Σ_ℓ c_ℓmm' d^ℓ_mm'(β)} exp(-i(mα + m'γ))
        
        This is a 2D Fourier series in α and γ for each β.
        """
        # Build Fourier coefficient matrix for each β
        # Index convention: m ranges from -l_max to l_max
        m_size = 2 * self.l_max + 1
        
        # Fourier coefficients: shape (n_beta, m_size, m_size)
        # F[β, m+l_max, m'+l_max] = Σ_ℓ c_ℓmm' d^ℓ_mm'(β)
        fourier_coeffs = torch.zeros(
            self.n_beta, m_size, m_size,
            device=self.device, dtype=torch.complex128
        )
        
        for l in range(self.l_max + 1):
            c_l = c_coeffs[l]  # (2l+1, 2l+1)
            d_l = self.d_matrices[l]  # (n_beta, 2l+1, 2l+1)
            
            # Map local indices to global m indices
            # Local index 0 corresponds to m = -l
            for local_mp, mp in enumerate(range(-l, l + 1)):
                for local_m, m in enumerate(range(-l, l + 1)):
                    global_mp = mp + self.l_max
                    global_m = m + self.l_max
                    
                    fourier_coeffs[:, global_mp, global_m] += (
                        c_l[local_mp, local_m] * d_l[:, local_mp, local_m]
                    )
        
        # Now compute R via inverse FFT
        # R(α, β, γ) = Σ_m Σ_m' F[β, m, m'] exp(-2πi(m·α/2π + m'·γ/2π))
        # We need to handle the frequency mapping carefully
        
        R = torch.zeros(
            self.n_alpha, self.n_beta, self.n_gamma,
            device=self.device, dtype=torch.float64
        )
        
        # For each β section, compute 2D inverse FFT
        for beta_idx in range(self.n_beta):
            # Zero-pad to desired output size
            padded = torch.zeros(
                self.n_alpha, self.n_gamma,
                device=self.device, dtype=torch.complex128
            )
            
            # Place coefficients accounting for negative frequencies
            for local_mp, mp in enumerate(range(-self.l_max, self.l_max + 1)):
                for local_m, m in enumerate(range(-self.l_max, self.l_max + 1)):
                    # Map to FFT frequency indices
                    freq_mp = mp % self.n_alpha
                    freq_m = m % self.n_gamma
                    padded[freq_mp, freq_m] = fourier_coeffs[beta_idx, local_mp, local_m]
            
            # Inverse FFT gives R(α, γ) for this β
            R[:, beta_idx, :] = torch.fft.ifft2(padded).real * self.n_alpha * self.n_gamma
        
        return R
    
    # Helper functions
    def _hkl_to_reciprocal(self, hkl: Tensor, cell: Tensor) -> Tensor:
        """Convert Miller indices to reciprocal space vectors in Å⁻¹."""
        a, b, c, alpha, beta, gamma = cell
        
        # Convert angles to radians
        alpha_r = alpha * math.pi / 180
        beta_r = beta * math.pi / 180
        gamma_r = gamma * math.pi / 180
        
        # Compute reciprocal cell
        V = self._cell_volume(cell)
        
        cos_alpha = torch.cos(alpha_r)
        cos_beta = torch.cos(beta_r)
        cos_gamma = torch.cos(gamma_r)
        sin_gamma = torch.sin(gamma_r)
        
        # Reciprocal lattice vectors (in Å⁻¹)
        a_star = b * c * torch.sin(alpha_r) / V
        b_star = a * c * torch.sin(beta_r) / V
        c_star = a * b * torch.sin(gamma_r) / V
        
        cos_alpha_star = (cos_beta * cos_gamma - cos_alpha) / (torch.sin(beta_r) * sin_gamma)
        cos_beta_star = (cos_alpha * cos_gamma - cos_beta) / (torch.sin(alpha_r) * sin_gamma)
        cos_gamma_star = (cos_alpha * cos_beta - cos_gamma) / (torch.sin(alpha_r) * torch.sin(beta_r))
        
        sin_alpha_star = torch.sqrt(1 - cos_alpha_star**2)
        sin_beta_star = torch.sqrt(1 - cos_beta_star**2)
        
        # Build reciprocal space matrix (columns are a*, b*, c* in Cartesian)
        # Standard convention: a* along x, b* in xy plane
        B = torch.zeros(3, 3, device=self.device, dtype=cell.dtype)
        B[0, 0] = a_star
        B[0, 1] = b_star * cos_gamma_star
        B[0, 2] = c_star * cos_beta_star
        B[1, 1] = b_star * sin_gamma_star
        B[1, 2] = -c_star * sin_beta_star * cos_alpha
        B[2, 2] = c_star * sin_beta_star * sin_alpha_star
        
        # s = B @ hkl.T -> (3, N)
        s_vectors = (B @ hkl.T.to(cell.dtype)).T  # (N, 3)
        
        return s_vectors
    
    def _cell_volume(self, cell: Tensor) -> Tensor:
        """Compute unit cell volume."""
        a, b, c, alpha, beta, gamma = cell
        alpha_r = alpha * math.pi / 180
        beta_r = beta * math.pi / 180
        gamma_r = gamma * math.pi / 180
        
        V = a * b * c * torch.sqrt(
            1 - torch.cos(alpha_r)**2 - torch.cos(beta_r)**2 - torch.cos(gamma_r)**2 +
            2 * torch.cos(alpha_r) * torch.cos(beta_r) * torch.cos(gamma_r)
        )
        return V
    
    def _cartesian_to_spherical_angles(self, xyz: Tensor) -> Tuple[Tensor, Tensor]:
        """Convert Cartesian to spherical (θ, φ) - polar angle from z-axis."""
        r = torch.norm(xyz, dim=1)
        r = torch.clamp(r, min=1e-10)  # Avoid division by zero
        
        theta = torch.acos(torch.clamp(xyz[:, 2] / r, -1, 1))  # 0 to π
        phi = torch.atan2(xyz[:, 1], xyz[:, 0])  # -π to π
        phi = phi % (2 * math.pi)  # 0 to 2π
        
        return theta, phi
    
    def _spherical_harmonics_conj(self, l: int, theta: Tensor, phi: Tensor) -> Tensor:
        """
        Compute Y_ℓm*(θ, φ) for m = -l to l.
        
        Y_ℓm(θ,φ) = sqrt((2l+1)/(4π) * (l-m)!/(l+m)!) * P_l^m(cos θ) * exp(imφ)
        
        Returns tensor of shape (N, 2l+1)
        """
        N = theta.shape[0]
        result = torch.zeros(N, 2 * l + 1, device=self.device, dtype=torch.complex128)
        
        cos_theta = torch.cos(theta)
        
        for m_idx, m in enumerate(range(-l, l + 1)):
            # Compute P_l^m(cos θ)
            P_lm = self._associated_legendre(l, abs(m), cos_theta)
            
            # Normalization
            norm = math.sqrt(
                (2 * l + 1) / (4 * math.pi) *
                math.factorial(l - abs(m)) / math.factorial(l + abs(m))
            )
            
            # Handle negative m: Y_l^{-m} = (-1)^m (Y_l^m)*
            if m < 0:
                phase = (-1) ** abs(m)
                Y_lm = phase * norm * P_lm * torch.exp(-1j * abs(m) * phi)
            else:
                Y_lm = norm * P_lm * torch.exp(1j * m * phi)
            
            # Conjugate
            result[:, m_idx] = Y_lm.conj()
            
        return result
    
    def _associated_legendre(self, l: int, m: int, x: Tensor) -> Tensor:
        """
        Compute associated Legendre polynomial P_l^m(x).
        Uses stable recurrence relation.
        """
        if m > l:
            return torch.zeros_like(x)
        
        # Start with P_m^m
        pmm = torch.ones_like(x)
        if m > 0:
            somx2 = torch.sqrt((1 - x) * (1 + x))
            fact = 1.0
            for i in range(1, m + 1):
                pmm = pmm * (-fact) * somx2
                fact += 2.0
        
        if l == m:
            return pmm
        
        # P_{m+1}^m
        pmmp1 = x * (2 * m + 1) * pmm
        
        if l == m + 1:
            return pmmp1
        
        # Use recurrence for l > m + 1
        pll = torch.zeros_like(x)
        for ll in range(m + 2, l + 1):
            pll = (x * (2 * ll - 1) * pmmp1 - (ll + m - 1) * pmm) / (ll - m)
            pmm = pmmp1
            pmmp1 = pll
        
        return pll
    
    def _spherical_bessel(self, l: int, x: Tensor) -> Tensor:
        """
        Compute spherical Bessel function j_l(x).
        
        j_0(x) = sin(x)/x
        j_1(x) = sin(x)/x² - cos(x)/x
        j_l(x) via recurrence for l > 1
        """
        # Handle x near zero
        x_safe = torch.where(x.abs() < 1e-10, torch.ones_like(x) * 1e-10, x)
        
        if l == 0:
            result = torch.sin(x_safe) / x_safe
        elif l == 1:
            result = torch.sin(x_safe) / x_safe**2 - torch.cos(x_safe) / x_safe
        else:
            # Use downward recurrence for stability
            # Start from large l and work down
            l_max_recur = max(l + 15, int(x.max().item()) + 15)
            
            j_curr = torch.zeros_like(x_safe)
            j_next = torch.ones_like(x_safe) * 1e-30
            
            for ll in range(l_max_recur, l - 1, -1):
                j_prev = (2 * ll + 1) / x_safe * j_next - j_curr
                j_curr = j_next
                j_next = j_prev
                
                if ll == l:
                    j_l_unnorm = j_next.clone()
            
            # Normalize using j_0
            j_0_computed = j_next
            j_0_exact = torch.sin(x_safe) / x_safe
            result = j_l_unnorm * j_0_exact / j_0_computed
        
        # Fix x = 0 cases
        if l == 0:
            result = torch.where(x.abs() < 1e-10, torch.ones_like(x), result)
        else:
            result = torch.where(x.abs() < 1e-10, torch.zeros_like(x), result)
        
        return result


def fast_rotation_function(
    F_obs: Tensor,
    hkl_obs: Tensor,
    F_calc: Tensor,
    hkl_calc: Tensor,
    cell: Tensor,
    l_max: int = 15,
    n_beta: int = 90,
    n_alpha: int = 180,
    n_gamma: int = 180,
    r_max: float = 25.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Convenience function to compute fast rotation function.
    
    Args:
        F_obs: Observed structure factor magnitudes (N_obs,)
        hkl_obs: Observed Miller indices (N_obs, 3)
        F_calc: Calculated structure factor magnitudes (N_calc,)
        hkl_calc: Calculated Miller indices (N_calc, 3)
        cell: Unit cell parameters [a, b, c, α, β, γ]
        l_max: Maximum spherical harmonic order
        n_beta, n_alpha, n_gamma: Grid sizes for Euler angles
        r_max: Patterson integration radius
        
    Returns:
        R: Rotation function values (n_alpha, n_beta, n_gamma)
        alpha: Alpha angle values
        beta: Beta angle values  
        gamma: Gamma angle values
    """
    frf = FastRotationFunction(
        l_max=l_max,
        n_beta=n_beta,
        n_alpha=n_alpha,
        n_gamma=n_gamma,
        r_max=r_max,
        device=F_obs.device,
    )
    
    R = frf.compute_rotation_function(F_obs, hkl_obs, F_calc, hkl_calc, cell)
    
    alpha = torch.linspace(0, 2*math.pi, n_alpha, device=F_obs.device)
    beta = torch.linspace(0, math.pi, n_beta, device=F_obs.device)
    gamma = torch.linspace(0, 2*math.pi, n_gamma, device=F_obs.device)
    
    return R, alpha, beta, gamma
```

## Key Implementation Notes

1. **Spherical harmonic coefficients** (Eq. 8): The summation over reflections with Bessel functions and spherical harmonics. This is the most computationally intensive part for large reflection counts.

2. **Wigner d-matrices**: These encode how spherical harmonics transform under rotation. I use the explicit formula, but for production code you'd want optimized recurrence relations.

3. **FFT trick**: The Fourier series structure (Eq. 12) means the final evaluation is just 2D inverse FFTs for each β section.

4. **Practical considerations for your MR package**:
   - For efficiency, you might want to bin reflections by resolution shell
   - The Patterson radius `r_max` should cover the expected interatomic vectors
   - `l_max` depends on resolution; higher resolution needs higher `l_max`

Would you like me to elaborate on any specific part, or discuss optimizations like using `torch.vmap` for the batched Bessel/spherical harmonic calculations?