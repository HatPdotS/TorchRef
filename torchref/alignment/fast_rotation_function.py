"""
Fast Rotation Function for Molecular Replacement.

Implements the Crowther-style fast rotation function using spherical harmonic
decomposition and FFT-based evaluation. This approach decomposes Patterson
functions into spherical harmonics, computes radial overlap coefficients,
and evaluates the rotation function via 2D FFT for each beta section.

Reference: Crowther (1972) "The fast rotation function"
"""

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .transform import euler_zyz_to_quaternion, quaternion_to_matrix


# =============================================================================
# E-value Normalization
# =============================================================================


def fit_anisotropic_wilson(
    F: torch.Tensor,
    s_vectors: torch.Tensor,
    epsilon: torch.Tensor = None,
) -> torch.Tensor:
    """
    Fit anisotropic Wilson model to structure factors.

    Models: log(<F²>/epsilon) = coeffs[0] - s^T * U * s

    where U is a symmetric 3x3 tensor encoded as coeffs[1:7].

    Parameters
    ----------
    F : torch.Tensor
        Structure factor magnitudes |F|, shape (N,).
    s_vectors : torch.Tensor
        Reciprocal space vectors in Å⁻¹, shape (N, 3).
    epsilon : torch.Tensor, optional
        Symmetry multiplicity factors, shape (N,). Default: all ones.

    Returns
    -------
    torch.Tensor
        Fitted coefficients, shape (7,).
        [log(scale), U_11, U_22, U_33, U_12, U_13, U_23]
    """
    device = F.device

    F_sq = F.abs() ** 2

    if epsilon is None:
        epsilon = torch.ones_like(F)

    # Ensure s_vectors is on the same device
    s_vectors = s_vectors.to(device=device, dtype=torch.float64)

    sx, sy, sz = s_vectors[:, 0], s_vectors[:, 1], s_vectors[:, 2]

    # Design matrix for symmetric tensor fit
    A = torch.stack(
        [
            torch.ones_like(sx),  # log(scale)
            sx * sx,  # U_11
            sy * sy,  # U_22
            sz * sz,  # U_33
            2 * sx * sy,  # U_12
            2 * sx * sz,  # U_13
            2 * sy * sz,  # U_23
        ],
        dim=1,
    ).to(device=device, dtype=torch.float64)

    # Target: log(F²/epsilon)
    F_sq_norm = F_sq / epsilon
    valid = F_sq_norm > 1e-10
    y = torch.zeros(len(F), device=device, dtype=torch.float64)
    y[valid] = torch.log(F_sq_norm[valid].to(torch.float64))

    # Weighted least squares (weight by F² to emphasize strong reflections)
    weights = torch.sqrt(F_sq_norm.to(torch.float64).clamp(min=1e-10))
    weights[~valid] = 0

    Aw = A * weights[:, None]
    yw = y * weights
    AtWA = Aw.T @ Aw + 1e-6 * torch.eye(7, device=device, dtype=torch.float64)
    AtWy = Aw.T @ yw

    try:
        coeffs = torch.linalg.solve(AtWA, AtWy)
    except RuntimeError:
        coeffs = torch.linalg.lstsq(Aw, yw).solution

    return coeffs


def apply_anisotropic_normalization(
    F: torch.Tensor,
    s_vectors: torch.Tensor,
    wilson_coeffs: torch.Tensor,
    epsilon: torch.Tensor = None,
) -> torch.Tensor:
    """
    Normalize structure factors using pre-fitted Wilson coefficients.

    E = F / sqrt(<F²>) where <F²> is computed from wilson_coeffs.

    Parameters
    ----------
    F : torch.Tensor
        Structure factor magnitudes |F|, shape (N,).
    s_vectors : torch.Tensor
        Reciprocal space vectors in Å⁻¹, shape (N, 3).
    wilson_coeffs : torch.Tensor
        Pre-fitted Wilson coefficients from fit_anisotropic_wilson, shape (7,).
    epsilon : torch.Tensor, optional
        Symmetry multiplicity factors, shape (N,). Default: all ones.

    Returns
    -------
    torch.Tensor
        Normalized E-values, shape (N,).
    """
    device = F.device
    dtype = F.dtype

    F_sq = F.abs() ** 2

    if epsilon is None:
        epsilon = torch.ones_like(F)

    # Ensure tensors are on same device
    s_vectors = s_vectors.to(device=device, dtype=torch.float64)
    wilson_coeffs = wilson_coeffs.to(device=device, dtype=torch.float64)

    sx, sy, sz = s_vectors[:, 0], s_vectors[:, 1], s_vectors[:, 2]

    # Build design matrix
    A = torch.stack(
        [
            torch.ones_like(sx),
            sx * sx,
            sy * sy,
            sz * sz,
            2 * sx * sy,
            2 * sx * sz,
            2 * sy * sz,
        ],
        dim=1,
    ).to(device=device, dtype=torch.float64)

    # Compute expected <F²> from coefficients
    log_expected = A @ wilson_coeffs
    expected_F_sq = torch.exp(log_expected) * epsilon.to(torch.float64)

    # Compute E = F / sqrt(<F²>)
    E_sq = F_sq.to(torch.float64) / expected_F_sq.clamp(min=1e-10)
    E = torch.sqrt(E_sq.clamp(min=0))

    return E.to(dtype)


def compute_e_values(
    F: torch.Tensor,
    s_vectors: torch.Tensor,
    epsilon: torch.Tensor = None,
    centric: torch.Tensor = None,
    wilson_coeffs: torch.Tensor = None,
) -> torch.Tensor:
    """
    Convert structure factors F to normalized E-values with anisotropic correction.

    E² = F² / <F²>

    where <F²> is modeled with an anisotropic Gaussian:
    <F²> = scale * exp(-s^T * U * s)

    Parameters
    ----------
    F : torch.Tensor
        Structure factor magnitudes |F|, shape (N,).
    s_vectors : torch.Tensor
        Reciprocal space vectors in Å⁻¹, shape (N, 3).
    epsilon : torch.Tensor, optional
        Symmetry multiplicity factors, shape (N,). Default: all ones.
    centric : torch.Tensor, optional
        Boolean mask for centric reflections, shape (N,). Default: all False (unused).
    wilson_coeffs : torch.Tensor, optional
        Pre-fitted Wilson coefficients. If provided, uses these instead of
        fitting to F. This enables shared normalization between datasets.

    Returns
    -------
    torch.Tensor
        Normalized E-values, shape (N,).
    """
    if wilson_coeffs is not None:
        # Use pre-fitted coefficients (shared normalization)
        return apply_anisotropic_normalization(F, s_vectors, wilson_coeffs, epsilon)

    # Fit and apply in one step (original behavior)
    coeffs = fit_anisotropic_wilson(F, s_vectors, epsilon)
    return apply_anisotropic_normalization(F, s_vectors, coeffs, epsilon)


def apply_patterson_sharpening(
    F: torch.Tensor,
    s_vectors: torch.Tensor,
    sharpening_b: float,
) -> torch.Tensor:
    """
    Apply Patterson sharpening (negative B-factor) to structure factors.

    Sharpening multiplies F by exp(-B_sharp * s² / 4) where B_sharp is typically
    negative (-40 to -100 Å²). This enhances high-resolution Patterson peaks
    and can improve discrimination in the rotation function.

    Parameters
    ----------
    F : torch.Tensor
        Structure factor magnitudes |F|, shape (N,).
    s_vectors : torch.Tensor
        Reciprocal space vectors in Å⁻¹, shape (N, 3).
    sharpening_b : float
        Sharpening B-factor in Å². Negative values sharpen (enhance high-res),
        positive values blur (dampen high-res). Typical values: -40 to -100 Å².

    Returns
    -------
    torch.Tensor
        Sharpened structure factors, shape (N,).
    """
    device = F.device
    dtype = F.dtype

    # s² = |s|² where s is in Å⁻¹
    s_vectors = s_vectors.to(device=device, dtype=torch.float64)
    s_sq = (s_vectors**2).sum(dim=-1)

    # Sharpening factor: exp(-B * s² / 4)
    # For negative B, this increases high-resolution terms
    sharpening_factor = torch.exp(-sharpening_b * s_sq / 4.0)

    F_sharpened = F.to(torch.float64) * sharpening_factor

    return F_sharpened.to(dtype)


def compute_anisotropic_scale(
    F: torch.Tensor,
    s_vectors: torch.Tensor,
    epsilon: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fit anisotropic scaling model and return scale factors.

    Models <F²> = scale * exp(-s^T * U * s)

    Parameters
    ----------
    F : torch.Tensor
        Structure factor magnitudes |F|, shape (N,).
    s_vectors : torch.Tensor
        Reciprocal space vectors in Å⁻¹, shape (N, 3).
    epsilon : torch.Tensor, optional
        Symmetry multiplicity factors, shape (N,).

    Returns
    -------
    scale_factors : torch.Tensor
        Anisotropic scale factors for each reflection, shape (N,).
    U_tensor : torch.Tensor
        Fitted anisotropic U tensor, shape (3, 3).
    """
    device = F.device
    dtype = F.dtype

    F_sq = F.abs() ** 2

    if epsilon is None:
        epsilon = torch.ones_like(F)

    # Ensure s_vectors is on the same device
    s_vectors = s_vectors.to(device=device, dtype=torch.float64)

    sx, sy, sz = s_vectors[:, 0], s_vectors[:, 1], s_vectors[:, 2]

    # Design matrix
    A = torch.stack(
        [
            torch.ones_like(sx),
            sx * sx,
            sy * sy,
            sz * sz,
            2 * sx * sy,
            2 * sx * sz,
            2 * sy * sz,
        ],
        dim=1,
    ).to(device=device, dtype=torch.float64)

    F_sq_norm = F_sq / epsilon
    valid = F_sq_norm > 1e-10
    y = torch.zeros(len(F), device=device, dtype=torch.float64)
    y[valid] = torch.log(F_sq_norm[valid].to(torch.float64))

    weights = torch.sqrt(F_sq_norm.to(torch.float64).clamp(min=1e-10))
    weights[~valid] = 0

    # Efficient weighted least squares
    Aw = A * weights[:, None]
    yw = y * weights
    AtWA = Aw.T @ Aw + 1e-6 * torch.eye(7, device=device, dtype=torch.float64)
    AtWy = Aw.T @ yw

    coeffs = torch.linalg.solve(AtWA, AtWy)

    # Extract U tensor
    U_tensor = torch.stack(
        [
            torch.stack([coeffs[1], coeffs[4], coeffs[5]]),
            torch.stack([coeffs[4], coeffs[2], coeffs[6]]),
            torch.stack([coeffs[5], coeffs[6], coeffs[3]]),
        ]
    )

    # Compute scale factors
    log_scale = A @ coeffs
    scale_factors = torch.exp(log_scale)

    return scale_factors.to(dtype), U_tensor

if TYPE_CHECKING:
    from torchref.io.datasets.reflection_data import ReflectionData
    from torchref.model.model_ft import ModelFT


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class FRFPeak:
    """
    A peak in the rotation function.

    Attributes
    ----------
    euler_angles : torch.Tensor
        ZYZ Euler angles [alpha, beta, gamma] in radians, shape (3,).
    rotation_matrix : torch.Tensor
        Corresponding 3x3 rotation matrix.
    score : float
        Raw rotation function score at this peak.
    sigma : float
        Score in sigma units above the mean.
    """

    euler_angles: torch.Tensor  # (3,) [alpha, beta, gamma] radians
    rotation_matrix: torch.Tensor  # (3, 3)
    score: float
    sigma: float


@dataclass
class FRFResult:
    """
    Result of a fast rotation function search.

    Attributes
    ----------
    peaks : List[FRFPeak]
        List of peaks sorted by score (descending).
    rotation_function : torch.Tensor
        Full rotation function R(alpha, beta, gamma), shape (n_alpha, n_beta, n_gamma).
    alpha_values : torch.Tensor
        Alpha angle values in radians.
    beta_values : torch.Tensor
        Beta angle values in radians.
    gamma_values : torch.Tensor
        Gamma angle values in radians.
    runtime_seconds : float
        Total computation time in seconds.
    """

    peaks: List[FRFPeak] = field(default_factory=list)
    rotation_function: Optional[torch.Tensor] = None
    alpha_values: Optional[torch.Tensor] = None
    beta_values: Optional[torch.Tensor] = None
    gamma_values: Optional[torch.Tensor] = None
    runtime_seconds: float = 0.0


# =============================================================================
# Fast Rotation Function
# =============================================================================


class FastRotationFunction(nn.Module):
    """
    Crowther-style fast rotation function using spherical harmonic decomposition.

    The rotation function R(omega) = integral P(r)Q(omega^{-1}r)dr is computed as:
    1. Expand Patterson functions P and Q in spherical harmonics
    2. Compute radial overlap coefficients c_lmm'
    3. Evaluate via 2D FFT for each beta section

    Parameters
    ----------
    l_max : int, optional
        Maximum spherical harmonic order. Default is 20.
    n_beta : int, optional
        Number of beta angle samples. Default is 90 (2 degree resolution).
    n_alpha : int, optional
        Number of alpha angle samples. Default is 180.
    n_gamma : int, optional
        Number of gamma angle samples. Default is 180.
    r_max : float, optional
        Patterson integration radius in Angstroms. Default is 30.0.
    n_radial : int, optional
        Number of radial integration points. Default is 100.
    resolution_limit : float, optional
        High resolution cutoff in Angstroms. Default is 4.0.
    device : torch.device, optional
        Device for computation. Default is None (auto-detect).

    Examples
    --------
    ::

        frf = FastRotationFunction(l_max=15, resolution_limit=4.0)
        result = frf.search(model, data)
        print(f"Top peak at {result.peaks[0].euler_angles} with sigma={result.peaks[0].sigma:.2f}")
    """

    def __init__(
        self,
        l_max: int = 20,
        n_beta: int = 90,
        n_alpha: int = 180,
        n_gamma: int = 180,
        r_max: float = 30.0,
        n_radial: int = 100,
        resolution_limit: float = 4.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.l_max = l_max
        self.n_beta = n_beta
        self.n_alpha = n_alpha
        self.n_gamma = n_gamma
        self.r_max = r_max
        self.n_radial = n_radial
        self.resolution_limit = resolution_limit
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Precompute beta values and Wigner d-matrices
        self.register_buffer(
            "beta_values",
            torch.linspace(0, math.pi, n_beta, device=self.device, dtype=torch.float64),
        )
        self._d_matrices: Optional[Dict[int, torch.Tensor]] = None

    def _ensure_wigner_d_matrices(self) -> None:
        """Lazy initialization of Wigner d-matrices."""
        if self._d_matrices is None:
            self._d_matrices = self._precompute_wigner_d()

    def _precompute_wigner_d(self) -> Dict[int, torch.Tensor]:
        """
        Precompute Wigner small-d matrices d^l_{m'm}(beta) for all beta values.

        Returns
        -------
        Dict[int, torch.Tensor]
            Dictionary mapping l to tensor of shape (n_beta, 2l+1, 2l+1).
        """
        d_matrices = {}
        for l in range(self.l_max + 1):
            d_matrices[l] = self._compute_wigner_d_matrix(l, self.beta_values)
        return d_matrices

    def _compute_wigner_d_matrix(self, l: int, beta: torch.Tensor) -> torch.Tensor:
        """
        Compute Wigner d-matrix d^l_{m'm}(beta) using explicit formula.

        Parameters
        ----------
        l : int
            Angular momentum quantum number.
        beta : torch.Tensor
            Beta angles of shape (n_beta,).

        Returns
        -------
        torch.Tensor
            Wigner d-matrices of shape (n_beta, 2l+1, 2l+1).
        """
        n_beta = beta.shape[0]
        size = 2 * l + 1
        d = torch.zeros(n_beta, size, size, device=self.device, dtype=torch.float64)

        cos_half = torch.cos(beta / 2)
        sin_half = torch.sin(beta / 2)

        for mp_idx, mp in enumerate(range(-l, l + 1)):
            for m_idx, m in enumerate(range(-l, l + 1)):
                d[:, mp_idx, m_idx] = self._wigner_d_element(
                    l, mp, m, cos_half, sin_half
                )

        return d

    def _wigner_d_element(
        self,
        l: int,
        mp: int,
        m: int,
        cos_half: torch.Tensor,
        sin_half: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute single element d^l_{m'm}(beta) using explicit formula.

        d^l_{m'm}(beta) = sum_s (-1)^{m'-m+s} *
                         sqrt((l+m')!(l-m')!(l+m)!(l-m)!) /
                         ((l+m-s)!(l-m'-s)!(m'-m+s)!s!) *
                         cos(beta/2)^{2l+m-m'-2s} * sin(beta/2)^{m'-m+2s}
        """
        result = torch.zeros_like(cos_half)

        # Sum limits
        s_min = max(0, m - mp)
        s_max = min(l + m, l - mp)

        if s_max < s_min:
            return result

        prefactor = math.sqrt(
            math.factorial(l + mp)
            * math.factorial(l - mp)
            * math.factorial(l + m)
            * math.factorial(l - m)
        )

        for s in range(s_min, s_max + 1):
            denom = (
                math.factorial(l + m - s)
                * math.factorial(l - mp - s)
                * math.factorial(mp - m + s)
                * math.factorial(s)
            )
            sign = (-1) ** (mp - m + s)

            cos_power = 2 * l + m - mp - 2 * s
            sin_power = mp - m + 2 * s

            term = sign * prefactor / denom

            # Handle powers carefully to avoid 0^0 issues
            if cos_power == 0:
                cos_term = torch.ones_like(cos_half)
            else:
                cos_term = cos_half**cos_power

            if sin_power == 0:
                sin_term = torch.ones_like(sin_half)
            else:
                sin_term = sin_half**sin_power

            result = result + term * cos_term * sin_term

        return result

    # =========================================================================
    # Spherical Bessel Functions
    # =========================================================================

    def _spherical_bessel_batched(self, l: int, x: torch.Tensor) -> torch.Tensor:
        """
        Compute spherical Bessel function j_l(x) for all x values.

        Uses upward recurrence from j_0 and j_1 for stability.

        Parameters
        ----------
        l : int
            Order of spherical Bessel function.
        x : torch.Tensor
            Argument values of shape (N,).

        Returns
        -------
        torch.Tensor
            j_l(x) values of shape (N,).
        """
        # Small x threshold for using series approximation
        small_x_threshold = 1e-6

        # Compute small-x approximation: j_l(x) ≈ x^l / (2l+1)!!
        double_factorial = 1.0
        for k in range(1, 2 * l + 2, 2):
            double_factorial *= k
        small_x_approx = (x.abs() ** l) / double_factorial

        # Safe x for avoiding division by zero
        x_safe = torch.where(x.abs() < small_x_threshold, torch.ones_like(x), x)

        if l == 0:
            # j_0(x) = sin(x)/x, with j_0(0) = 1
            result = torch.sin(x_safe) / x_safe
            result = torch.where(x.abs() < small_x_threshold, torch.ones_like(x), result)
            return result

        if l == 1:
            # j_1(x) = sin(x)/x^2 - cos(x)/x, with j_1(0) = 0
            result = torch.sin(x_safe) / x_safe**2 - torch.cos(x_safe) / x_safe
            result = torch.where(x.abs() < small_x_threshold, small_x_approx, result)
            return result

        # For l >= 2, use upward recurrence: j_{l+1} = (2l+1)/x * j_l - j_{l-1}
        # This is stable for x > l, but we need to be careful for x < l

        # Start with j_0 and j_1
        j_prev = torch.sin(x_safe) / x_safe  # j_0
        j_curr = torch.sin(x_safe) / x_safe**2 - torch.cos(x_safe) / x_safe  # j_1

        # Upward recurrence
        for ll in range(1, l):
            j_next = (2 * ll + 1) / x_safe * j_curr - j_prev
            j_prev = j_curr
            j_curr = j_next

        result = j_curr

        # For small x, upward recurrence can be unstable, use approximation
        # The threshold depends on l: roughly when x < l/2
        unstable_mask = x.abs() < max(l / 2.0, 0.5)
        result = torch.where(unstable_mask, small_x_approx, result)

        # Also fix any NaN/Inf that might have occurred
        result = torch.where(torch.isfinite(result), result, small_x_approx)

        return result

    # =========================================================================
    # Associated Legendre Polynomials
    # =========================================================================

    def _associated_legendre_batched(
        self, l_max: int, cos_theta: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute associated Legendre polynomials P_l^m(cos(theta)) for all l, m.

        Uses stable 3-term recurrence relations.

        Parameters
        ----------
        l_max : int
            Maximum l value.
        cos_theta : torch.Tensor
            cos(theta) values of shape (N,).

        Returns
        -------
        torch.Tensor
            P_l^m values of shape (N, l_max+1, l_max+1).
            Index [n, l, m] gives P_l^m for the nth sample.
        """
        N = cos_theta.shape[0]
        P = torch.zeros(
            N, l_max + 1, l_max + 1, device=self.device, dtype=torch.float64
        )

        # sin(theta) from cos(theta)
        sin_theta = torch.sqrt(1.0 - cos_theta**2).clamp(min=1e-12)

        # P_0^0 = 1
        P[:, 0, 0] = 1.0

        # Compute P_m^m using (l-1, m-1) -> (l, m) formula
        # P_m^m = -(2m-1) * sin(theta) * P_{m-1}^{m-1}
        for m in range(1, l_max + 1):
            P[:, m, m] = -(2 * m - 1) * sin_theta * P[:, m - 1, m - 1]

        # Compute P_{m+1}^m
        # P_{m+1}^m = cos(theta) * (2m+1) * P_m^m
        for m in range(l_max):
            P[:, m + 1, m] = cos_theta * (2 * m + 1) * P[:, m, m]

        # Use recurrence for l > m + 1:
        # P_l^m = ((2l-1)*cos(theta)*P_{l-1}^m - (l+m-1)*P_{l-2}^m) / (l-m)
        for m in range(l_max + 1):
            for l in range(m + 2, l_max + 1):
                P[:, l, m] = (
                    (2 * l - 1) * cos_theta * P[:, l - 1, m] - (l + m - 1) * P[:, l - 2, m]
                ) / (l - m)

        return P

    # =========================================================================
    # Spherical Harmonics
    # =========================================================================

    def _spherical_harmonics_conj(
        self, l: int, theta: torch.Tensor, phi: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute complex conjugate of spherical harmonics Y_lm*(theta, phi).

        Y_lm(theta, phi) = sqrt((2l+1)/(4*pi) * (l-m)!/(l+m)!) * P_l^m(cos(theta)) * exp(i*m*phi)

        Parameters
        ----------
        l : int
            Angular momentum quantum number.
        theta : torch.Tensor
            Polar angle (from z-axis) of shape (N,).
        phi : torch.Tensor
            Azimuthal angle of shape (N,).

        Returns
        -------
        torch.Tensor
            Y_lm* values of shape (N, 2l+1) for m = -l to l.
        """
        N = theta.shape[0]
        result = torch.zeros(N, 2 * l + 1, device=self.device, dtype=torch.complex128)

        cos_theta = torch.cos(theta)

        for m_idx, m in enumerate(range(-l, l + 1)):
            # Compute P_l^|m|(cos(theta))
            P_lm = self._associated_legendre_single(l, abs(m), cos_theta)

            # Normalization factor
            norm = math.sqrt(
                (2 * l + 1)
                / (4 * math.pi)
                * math.factorial(l - abs(m))
                / math.factorial(l + abs(m))
            )

            # Handle negative m: Y_l^{-m} = (-1)^m * (Y_l^m)*
            if m < 0:
                phase = (-1) ** abs(m)
                Y_lm = phase * norm * P_lm * torch.exp(-1j * abs(m) * phi)
            else:
                Y_lm = norm * P_lm * torch.exp(1j * m * phi)

            # Conjugate
            result[:, m_idx] = Y_lm.conj()

        return result

    def _associated_legendre_single(
        self, l: int, m: int, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute associated Legendre polynomial P_l^m(x) for single (l, m).

        Parameters
        ----------
        l : int
            Degree.
        m : int
            Order (must be >= 0).
        x : torch.Tensor
            Argument values of shape (N,).

        Returns
        -------
        torch.Tensor
            P_l^m(x) values of shape (N,).
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

    # =========================================================================
    # Reciprocal Space Utilities
    # =========================================================================

    def _hkl_to_reciprocal(
        self, hkl: torch.Tensor, cell: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert Miller indices to reciprocal space vectors in Angstrom^{-1}.

        Parameters
        ----------
        hkl : torch.Tensor
            Miller indices of shape (N, 3).
        cell : torch.Tensor
            Unit cell parameters [a, b, c, alpha, beta, gamma].

        Returns
        -------
        torch.Tensor
            Reciprocal space vectors of shape (N, 3).
        """
        a, b, c, alpha, beta, gamma = cell
        dtype = cell.dtype

        # Convert angles to radians
        alpha_r = alpha * math.pi / 180
        beta_r = beta * math.pi / 180
        gamma_r = gamma * math.pi / 180

        # Cell volume
        V = self._cell_volume(cell)

        cos_alpha = torch.cos(alpha_r)
        cos_beta = torch.cos(beta_r)
        cos_gamma = torch.cos(gamma_r)
        sin_alpha = torch.sin(alpha_r)
        sin_beta = torch.sin(beta_r)
        sin_gamma = torch.sin(gamma_r)

        # Reciprocal cell parameters
        a_star = b * c * sin_alpha / V
        b_star = a * c * sin_beta / V
        c_star = a * b * sin_gamma / V

        cos_alpha_star = (cos_beta * cos_gamma - cos_alpha) / (sin_beta * sin_gamma)
        cos_beta_star = (cos_alpha * cos_gamma - cos_beta) / (sin_alpha * sin_gamma)
        cos_gamma_star = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta)

        sin_gamma_star = torch.sqrt(1 - cos_gamma_star**2)
        sin_beta_star = torch.sqrt(1 - cos_beta_star**2)

        # Build B matrix (reciprocal space metric)
        B = torch.zeros(3, 3, device=self.device, dtype=dtype)
        B[0, 0] = a_star
        B[0, 1] = b_star * cos_gamma_star
        B[0, 2] = c_star * cos_beta_star
        B[1, 1] = b_star * sin_gamma_star
        B[1, 2] = -c_star * sin_beta_star * cos_alpha
        B[2, 2] = c_star * sin_beta_star * torch.sqrt(1 - cos_alpha**2)

        # s = B @ hkl.T
        s_vectors = (B @ hkl.T.to(dtype)).T

        return s_vectors

    def _cell_volume(self, cell: torch.Tensor) -> torch.Tensor:
        """Compute unit cell volume in Angstrom^3."""
        a, b, c, alpha, beta, gamma = cell
        alpha_r = alpha * math.pi / 180
        beta_r = beta * math.pi / 180
        gamma_r = gamma * math.pi / 180

        V = (
            a
            * b
            * c
            * torch.sqrt(
                1
                - torch.cos(alpha_r) ** 2
                - torch.cos(beta_r) ** 2
                - torch.cos(gamma_r) ** 2
                + 2 * torch.cos(alpha_r) * torch.cos(beta_r) * torch.cos(gamma_r)
            )
        )
        return V

    def _cartesian_to_spherical_angles(
        self, xyz: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert Cartesian coordinates to spherical angles (theta, phi).

        Parameters
        ----------
        xyz : torch.Tensor
            Cartesian coordinates of shape (N, 3).

        Returns
        -------
        theta : torch.Tensor
            Polar angle from z-axis, range [0, pi], shape (N,).
        phi : torch.Tensor
            Azimuthal angle, range [0, 2*pi], shape (N,).
        """
        r = torch.norm(xyz, dim=1)
        r = torch.clamp(r, min=1e-10)

        theta = torch.acos(torch.clamp(xyz[:, 2] / r, -1, 1))
        phi = torch.atan2(xyz[:, 1], xyz[:, 0])
        phi = phi % (2 * math.pi)

        return theta, phi

    # =========================================================================
    # Spherical Harmonic Coefficients
    # =========================================================================

    def compute_spherical_harmonic_coefficients(
        self,
        F_squared: torch.Tensor,
        hkl: torch.Tensor,
        cell: torch.Tensor,
    ) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
        """
        Compute spherical harmonic expansion coefficients from structure factors.

        From equation (8): a_lm(r) = (4*pi*i^l/V) * sum_s |F(s)|^2 * j_l(2*pi*s*r) * Y_lm*(theta_s, phi_s)

        Parameters
        ----------
        F_squared : torch.Tensor
            |F|^2 values of shape (N,).
        hkl : torch.Tensor
            Miller indices of shape (N, 3).
        cell : torch.Tensor
            Unit cell parameters.

        Returns
        -------
        coefficients : Dict[int, torch.Tensor]
            Dictionary mapping l to tensor of shape (n_radial, 2l+1).
        r_values : torch.Tensor
            Radial sampling points.
        """
        # Ensure all inputs are on the correct device and dtype
        hkl = hkl.to(device=self.device, dtype=torch.float64)
        cell = cell.to(device=self.device, dtype=torch.float64)
        F_squared = F_squared.to(device=self.device, dtype=torch.float64)

        # Convert HKL to reciprocal space vectors
        s_vectors = self._hkl_to_reciprocal(hkl, cell)
        s_mag = torch.norm(s_vectors, dim=1)
        theta_s, phi_s = self._cartesian_to_spherical_angles(s_vectors)

        # Radial sampling points
        r_values = torch.linspace(
            0.1, self.r_max, self.n_radial, device=self.device, dtype=torch.float64
        )

        # Volume normalization
        V = self._cell_volume(cell)

        coefficients: Dict[int, torch.Tensor] = {}

        # Precompute x = 2*pi*s*r for all (r, s) pairs
        # Shape: (n_radial, N)
        x_all = 2 * math.pi * s_mag[None, :] * r_values[:, None]

        for l in range(self.l_max + 1):
            # Y_lm*(theta_s, phi_s) for all m and all reflections
            # Shape: (N, 2l+1)
            Ylm_conj = self._spherical_harmonics_conj(l, theta_s, phi_s)

            # Phase factor i^l
            phase = (1j) ** l

            # Compute spherical Bessel j_l for all (r, s) pairs at once
            # Shape: (n_radial, N)
            j_l_all = self._spherical_bessel_batched(l, x_all.flatten()).reshape(
                self.n_radial, -1
            )

            # Vectorized weighted sum over all radial points
            # weighted shape: (n_radial, N, 2l+1)
            # = F_squared[None, :, None] * j_l_all[:, :, None] * Ylm_conj[None, :, :]
            weighted = F_squared[None, :, None] * j_l_all[:, :, None] * Ylm_conj[None, :, :]

            # Sum over reflections (dim=1): (n_radial, 2l+1)
            coeff_l = (4 * math.pi * phase / V) * weighted.sum(dim=1)

            coefficients[l] = coeff_l

        return coefficients, r_values

    def compute_radial_overlap(
        self,
        a_coeffs: Dict[int, torch.Tensor],
        b_coeffs: Dict[int, torch.Tensor],
        r_values: torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        """
        Compute radial overlap coefficients c_lmm'.

        c_lmm' = integral a_lm(r) * b_lm'(r)* * r^2 dr

        Parameters
        ----------
        a_coeffs : Dict[int, torch.Tensor]
            Coefficients from observed data, mapping l to (n_radial, 2l+1).
        b_coeffs : Dict[int, torch.Tensor]
            Coefficients from model, mapping l to (n_radial, 2l+1).
        r_values : torch.Tensor
            Radial sampling points.

        Returns
        -------
        Dict[int, torch.Tensor]
            Overlap coefficients mapping l to (2l+1, 2l+1) tensor.
        """
        dr = r_values[1] - r_values[0]
        r_squared = r_values**2

        c_coeffs: Dict[int, torch.Tensor] = {}

        for l in range(self.l_max + 1):
            a_l = a_coeffs[l]  # (n_radial, 2l+1)
            b_l = b_coeffs[l]  # (n_radial, 2l+1)

            # Integrate: c_lmm' = sum_r a_lm(r) * b_lm'(r)* * r^2 * dr
            # Shape: (n_radial, 2l+1, 1) * (n_radial, 1, 2l+1) * (n_radial, 1, 1)
            integrand = a_l[:, :, None] * b_l[:, None, :].conj() * r_squared[:, None, None]

            c_coeffs[l] = integrand.sum(dim=0) * dr  # (2l+1, 2l+1)

        return c_coeffs

    # =========================================================================
    # FFT Evaluation
    # =========================================================================

    def _evaluate_via_fft(self, c_coeffs: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        Evaluate R(alpha, beta, gamma) via 2D FFT.

        R(alpha, beta, gamma) = sum_m sum_m' {sum_l c_lmm' * d^l_mm'(beta)} * exp(-i(m*alpha + m'*gamma))

        This is a 2D Fourier series in alpha and gamma for each beta.

        Parameters
        ----------
        c_coeffs : Dict[int, torch.Tensor]
            Radial overlap coefficients.

        Returns
        -------
        torch.Tensor
            Rotation function of shape (n_alpha, n_beta, n_gamma).
        """
        self._ensure_wigner_d_matrices()

        # Index convention: m ranges from -l_max to l_max
        m_size = 2 * self.l_max + 1

        # Build Fourier coefficient matrix for each beta - VECTORIZED
        # F[beta, m+l_max, m'+l_max] = sum_l c_lmm' * d^l_mm'(beta)
        fourier_coeffs = torch.zeros(
            self.n_beta, m_size, m_size, device=self.device, dtype=torch.complex128
        )

        for l in range(self.l_max + 1):
            c_l = c_coeffs[l]  # (2l+1, 2l+1)
            d_l = self._d_matrices[l]  # (n_beta, 2l+1, 2l+1)

            # Vectorized: add c_l * d_l to the correct slice of fourier_coeffs
            # The slice corresponds to m, m' from -l to +l, offset by l_max
            start_idx = self.l_max - l
            end_idx = self.l_max + l + 1
            fourier_coeffs[:, start_idx:end_idx, start_idx:end_idx] += c_l[None, :, :] * d_l

        # Build padded FFT input for all beta values at once - BATCHED
        # Shape: (n_beta, n_alpha, n_gamma)
        padded = torch.zeros(
            self.n_beta, self.n_alpha, self.n_gamma, device=self.device, dtype=torch.complex128
        )

        # Create frequency index mapping (vectorized)
        m_vals = torch.arange(-self.l_max, self.l_max + 1, device=self.device)
        freq_m_alpha = m_vals % self.n_alpha  # (m_size,)
        freq_m_gamma = m_vals % self.n_gamma  # (m_size,)

        # Use advanced indexing to place all coefficients at once
        # Create meshgrid for indexing
        mp_idx, m_idx = torch.meshgrid(
            torch.arange(m_size, device=self.device),
            torch.arange(m_size, device=self.device),
            indexing="ij",
        )
        freq_mp_flat = freq_m_alpha[mp_idx.flatten()]  # (m_size*m_size,)
        freq_m_flat = freq_m_gamma[m_idx.flatten()]  # (m_size*m_size,)

        # Place coefficients for all beta at once using index_put
        for beta_idx in range(self.n_beta):
            padded[beta_idx, freq_mp_flat, freq_m_flat] = fourier_coeffs[
                beta_idx
            ].flatten()

        # Batched inverse FFT over all beta values at once
        R_batched = torch.fft.ifft2(padded).real * self.n_alpha * self.n_gamma

        # Transpose to get (n_alpha, n_beta, n_gamma)
        R = R_batched.permute(1, 0, 2)

        return R

    # =========================================================================
    # Peak Finding
    # =========================================================================

    def find_peaks(
        self,
        R: torch.Tensor,
        n_peaks: int = 100,
        sigma_cutoff: float = 3.0,
        cluster_radius_deg: float = 5.0,
    ) -> List[FRFPeak]:
        """
        Extract and cluster top peaks from the rotation function.

        Parameters
        ----------
        R : torch.Tensor
            Rotation function of shape (n_alpha, n_beta, n_gamma).
        n_peaks : int, optional
            Maximum number of peaks to return. Default is 100.
        sigma_cutoff : float, optional
            Minimum sigma above mean to consider as peak. Default is 3.0.
        cluster_radius_deg : float, optional
            Clustering radius in degrees. Default is 5.0.

        Returns
        -------
        List[FRFPeak]
            Sorted list of peaks (highest score first).
        """
        # Compute statistics
        R_mean = R.mean()
        R_std = R.std()

        # Find values above threshold
        threshold = R_mean + sigma_cutoff * R_std
        above_threshold = R > threshold

        if not above_threshold.any():
            return []

        # Get indices and values of points above threshold
        indices = torch.nonzero(above_threshold, as_tuple=False)
        values = R[above_threshold]

        # Sort by value (descending) and limit candidates to avoid O(N²) issues
        sorted_indices = torch.argsort(values, descending=True)

        # Limit to top candidates for efficiency (10x requested peaks is usually enough)
        max_candidates = min(len(sorted_indices), n_peaks * 20)
        sorted_indices = sorted_indices[:max_candidates]

        indices = indices[sorted_indices]
        values = values[sorted_indices]

        # Precompute all angles on GPU (vectorized)
        alpha_step = 2 * math.pi / self.n_alpha
        gamma_step = 2 * math.pi / self.n_gamma

        alpha_angles = indices[:, 0].float() * alpha_step
        beta_angles = self.beta_values[indices[:, 1]]
        gamma_angles = indices[:, 2].float() * gamma_step

        # Cluster peaks using vectorized operations
        cluster_radius_rad = cluster_radius_deg * math.pi / 180
        peaks: List[FRFPeak] = []
        used = torch.zeros(len(indices), dtype=torch.bool, device=self.device)

        # Pre-convert values to CPU for peak creation
        R_mean_val = R_mean.item()
        R_std_val = R_std.item()

        for i in range(len(indices)):
            if used[i] or len(peaks) >= n_peaks:
                continue

            alpha = alpha_angles[i]
            beta = beta_angles[i]
            gamma = gamma_angles[i]

            # Vectorized distance computation for clustering
            if i + 1 < len(indices):
                # Compute distances to all remaining candidates at once
                remaining_mask = ~used[i + 1:]
                if remaining_mask.any():
                    remaining_idx = torch.arange(i + 1, len(indices), device=self.device)[
                        remaining_mask
                    ]

                    d_alpha = torch.abs(alpha_angles[remaining_idx] - alpha)
                    d_alpha = torch.minimum(d_alpha, 2 * math.pi - d_alpha)
                    d_beta = torch.abs(beta_angles[remaining_idx] - beta)
                    d_gamma = torch.abs(gamma_angles[remaining_idx] - gamma)
                    d_gamma = torch.minimum(d_gamma, 2 * math.pi - d_gamma)

                    dist = torch.sqrt(d_alpha**2 + d_beta**2 + d_gamma**2)
                    close_mask = dist < cluster_radius_rad
                    used[remaining_idx[close_mask]] = True

            used[i] = True

            # Create peak (move to CPU only at the end)
            euler_angles = torch.stack([alpha, beta, gamma]).to(torch.float64)
            q = euler_zyz_to_quaternion(euler_angles)
            rotation_matrix = quaternion_to_matrix(q)

            sigma_val = (values[i].item() - R_mean_val) / R_std_val

            peak = FRFPeak(
                euler_angles=euler_angles,
                rotation_matrix=rotation_matrix,
                score=values[i].item(),
                sigma=sigma_val,
            )
            peaks.append(peak)

        return peaks

    # =========================================================================
    # Main Search Methods
    # =========================================================================

    def compute_rotation_function(
        self,
        F_obs: torch.Tensor,
        hkl_obs: torch.Tensor,
        F_calc: torch.Tensor,
        hkl_calc: torch.Tensor,
        cell: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the full rotation function R(alpha, beta, gamma).

        Parameters
        ----------
        F_obs : torch.Tensor
            Observed structure factor magnitudes |F|, shape (N_obs,).
        hkl_obs : torch.Tensor
            Miller indices for observed data, shape (N_obs, 3).
        F_calc : torch.Tensor
            Calculated structure factor magnitudes |F|, shape (N_calc,).
        hkl_calc : torch.Tensor
            Miller indices for calculated data, shape (N_calc, 3).
        cell : torch.Tensor
            Unit cell parameters [a, b, c, alpha, beta, gamma].

        Returns
        -------
        torch.Tensor
            Rotation function of shape (n_alpha, n_beta, n_gamma).
        """
        # Step 1: Compute spherical harmonic coefficients for Patterson functions
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

        # Step 3: Evaluate via FFT
        R = self._evaluate_via_fft(c_coeffs)

        return R

    def search(
        self,
        model: "ModelFT",
        data: "ReflectionData",
        n_peaks: int = 100,
        sigma_cutoff: float = 3.0,
        use_e_values: bool = False,
        sharpening_b: float = None,
    ) -> FRFResult:
        """
        Run fast rotation function search.

        Parameters
        ----------
        model : ModelFT
            Model with structure factors to search for.
        data : ReflectionData
            Observed reflection data.
        n_peaks : int, optional
            Maximum number of peaks to return. Default is 100.
        sigma_cutoff : float, optional
            Minimum sigma for peak detection. Default is 3.0.
        use_e_values : bool, optional
            If True, normalize F to E-values with anisotropic correction
            before computing Patterson overlap. This improves discrimination
            by removing resolution-dependent intensity fall-off. Default is False.
        sharpening_b : float, optional
            Patterson sharpening B-factor in Å². Use negative values (e.g., -40 to -100)
            to sharpen Patterson by enhancing high-resolution terms. Applied after
            E-value normalization if both are enabled. Default is None (no sharpening).

        Returns
        -------
        FRFResult
            Search result with peaks and rotation function.
        """
        start_time = time.time()

        # Get observed structure factors
        cell = data.cell.to(self.device, dtype=torch.float64)
        hkl_obs = data.hkl.to(self.device)
        F_obs = data.F.to(self.device, dtype=torch.float64)

        # Apply resolution cutoff
        if self.resolution_limit > 0:
            mask = data.resolution >= self.resolution_limit
            hkl_obs = hkl_obs[mask]
            F_obs = F_obs[mask]

        # Compute model structure factors
        F_calc = model.forward(hkl_obs).abs().to(torch.float64)
        hkl_calc = hkl_obs  # Same HKL set

        # Compute s-vectors once (needed for E-values and/or sharpening)
        s_vectors = self._hkl_to_reciprocal(hkl_obs, cell)

        # Convert to E-values if requested (with shared normalization)
        if use_e_values:
            # Fit Wilson model to OBSERVED data only
            wilson_coeffs = fit_anisotropic_wilson(F_obs, s_vectors)
            # Apply same normalization to both - ensures same scale
            F_obs = apply_anisotropic_normalization(F_obs, s_vectors, wilson_coeffs)
            F_calc = apply_anisotropic_normalization(F_calc, s_vectors, wilson_coeffs)

        # Apply Patterson sharpening if requested
        if sharpening_b is not None:
            F_obs = apply_patterson_sharpening(F_obs, s_vectors, sharpening_b)
            F_calc = apply_patterson_sharpening(F_calc, s_vectors, sharpening_b)

        # Compute rotation function
        R = self.compute_rotation_function(F_obs, hkl_obs, F_calc, hkl_calc, cell)

        # Find peaks
        peaks = self.find_peaks(R, n_peaks=n_peaks, sigma_cutoff=sigma_cutoff)

        runtime = time.time() - start_time

        # Create angle arrays
        alpha_values = torch.linspace(
            0, 2 * math.pi, self.n_alpha, device=self.device, dtype=torch.float64
        )
        gamma_values = torch.linspace(
            0, 2 * math.pi, self.n_gamma, device=self.device, dtype=torch.float64
        )

        return FRFResult(
            peaks=peaks,
            rotation_function=R,
            alpha_values=alpha_values,
            beta_values=self.beta_values,
            gamma_values=gamma_values,
            runtime_seconds=runtime,
        )


# =============================================================================
# Convenience Function
# =============================================================================


def fast_rotation_function(
    F_obs: torch.Tensor,
    hkl_obs: torch.Tensor,
    F_calc: torch.Tensor,
    hkl_calc: torch.Tensor,
    cell: torch.Tensor,
    l_max: int = 15,
    n_beta: int = 90,
    n_alpha: int = 180,
    n_gamma: int = 180,
    r_max: float = 25.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convenience function to compute fast rotation function.

    Parameters
    ----------
    F_obs : torch.Tensor
        Observed structure factor magnitudes (N_obs,).
    hkl_obs : torch.Tensor
        Observed Miller indices (N_obs, 3).
    F_calc : torch.Tensor
        Calculated structure factor magnitudes (N_calc,).
    hkl_calc : torch.Tensor
        Calculated Miller indices (N_calc, 3).
    cell : torch.Tensor
        Unit cell parameters [a, b, c, alpha, beta, gamma].
    l_max : int, optional
        Maximum spherical harmonic order. Default is 15.
    n_beta, n_alpha, n_gamma : int, optional
        Grid sizes for Euler angles.
    r_max : float, optional
        Patterson integration radius. Default is 25.0.

    Returns
    -------
    R : torch.Tensor
        Rotation function values (n_alpha, n_beta, n_gamma).
    alpha : torch.Tensor
        Alpha angle values.
    beta : torch.Tensor
        Beta angle values.
    gamma : torch.Tensor
        Gamma angle values.
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

    alpha = torch.linspace(0, 2 * math.pi, n_alpha, device=F_obs.device)
    beta = torch.linspace(0, math.pi, n_beta, device=F_obs.device)
    gamma = torch.linspace(0, 2 * math.pi, n_gamma, device=F_obs.device)

    return R, alpha, beta, gamma
