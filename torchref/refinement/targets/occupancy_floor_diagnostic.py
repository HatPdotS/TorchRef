"""
Occupancy Floor Diagnostic for Time-Resolved Crystallography.

This module provides tools to estimate a lower bound on the activation fraction
by analyzing electron density. The key insight is that negative electron density
is unphysical - you cannot remove more electrons than were present.

For atoms that move in the excited state (e.g., waters, ligands):
- The dark state has density ρ_dark at the original position
- If the atom completely leaves, the light state has ρ_light ≈ 0 there
- The observed difference is: Δρ = α × (ρ_light - ρ_dark) = -α × ρ_dark
- The depth of the negative peak gives: α = |Δρ| / ρ_dark

If α is underestimated, the model must predict ρ_light < 0 to fit the data,
which is unphysical. This provides a floor on α.
"""

import torch
from torch import nn
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from torchref.model import ModelFT, MixedModel


class OccupancyFloorDiagnostic:
    """
    Diagnostic tool to estimate activation fraction floor from electron density.

    Analyzes the electron density of the light/refined model and checks for
    unphysical negative density, which indicates the activation fraction is
    too small.

    Parameters
    ----------
    model_dark : ModelFT
        The dark/ground state model.
    model_light : ModelFT
        The light/excited state model (the refined one, not MixedModel).
    grid_spacing : float, optional
        Grid spacing in Angstroms for density calculation. Default is 0.5.
    negative_threshold : float, optional
        Threshold below which density is considered "significantly negative".
        Default is -0.5 (in sigma units after normalization).

    Examples
    --------
    Basic usage::

        diagnostic = OccupancyFloorDiagnostic(model_dark, model_light_refine)
        result = diagnostic.analyze()
        print(f"Estimated alpha floor: {result['alpha_floor']:.3f}")
    """

    def __init__(
        self,
        model_dark: "ModelFT",
        model_light: "ModelFT",
        grid_spacing: float = 0.5,
        negative_threshold: float = -0.5,
    ):
        self.model_dark = model_dark
        self.model_light = model_light
        self.grid_spacing = grid_spacing
        self.negative_threshold = negative_threshold

    def compute_density_at_positions(
        self,
        model: "ModelFT",
        positions: torch.Tensor,
        hkl: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute electron density at specific positions using Fourier summation.

        This is a simplified calculation that sums F_calc * exp(2πi * h·r).

        Parameters
        ----------
        model : ModelFT
            Model to compute density from.
        positions : torch.Tensor
            Positions in fractional coordinates, shape (N, 3).
        hkl : torch.Tensor
            Miller indices, shape (M, 3).

        Returns
        -------
        torch.Tensor
            Electron density values at each position, shape (N,).
        """
        with torch.no_grad():
            fcalc = model(hkl, recalc=True)

            # Compute h·r for all position-reflection pairs
            # positions: (N, 3), hkl: (M, 3)
            # h_dot_r: (N, M)
            h_dot_r = torch.matmul(positions, hkl.T.float())

            # Fourier sum: ρ(r) = Σ_h F(h) * exp(2πi * h·r)
            # For real density, this is: Σ_h |F(h)| * cos(2π*h·r + φ(h))
            phase = torch.angle(fcalc)  # (M,)
            amplitude = torch.abs(fcalc)  # (M,)

            # ρ(r) = Σ_h |F(h)| * cos(2π*h·r + φ(h))
            density = (amplitude.unsqueeze(0) * torch.cos(2 * torch.pi * h_dot_r + phase.unsqueeze(0))).sum(dim=1)

            # Normalize by number of reflections (approximate)
            density = density / len(hkl)

        return density

    def analyze_at_dark_positions(
        self,
        hkl: torch.Tensor,
        atom_mask: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Analyze light model density at dark atom positions.

        Parameters
        ----------
        hkl : torch.Tensor
            Miller indices for Fourier calculation.
        atom_mask : torch.Tensor, optional
            Boolean mask selecting which atoms to analyze (e.g., waters only).

        Returns
        -------
        dict
            Dictionary with analysis results including:
            - 'rho_dark': Dark model density at atom positions
            - 'rho_light': Light model density at atom positions
            - 'rho_ratio': ρ_light / ρ_dark (should be ≥ 0)
            - 'negative_mask': Boolean mask of atoms with negative light density
            - 'alpha_floor': Estimated lower bound on activation fraction
            - 'worst_atoms': Indices of atoms with most negative density
        """
        # Get atom positions in fractional coordinates
        xyz_dark = self.model_dark.xyz()
        cell = self.model_dark.cell

        # Convert to fractional coordinates
        frac_dark = cell.cartesian_to_fractional(xyz_dark)

        if atom_mask is not None:
            frac_dark = frac_dark[atom_mask]

        # Compute density at dark positions for both models
        rho_dark = self.compute_density_at_positions(self.model_dark, frac_dark, hkl)
        rho_light = self.compute_density_at_positions(self.model_light, frac_dark, hkl)

        # Find atoms where light density is negative
        negative_mask = rho_light < 0

        # Compute density ratio (avoiding division by zero)
        rho_ratio = rho_light / (rho_dark + 1e-6)

        # Estimate alpha floor from the most negative cases
        # If ρ_light < 0 and we need ρ_light ≥ 0, then:
        # The model is predicting: ρ_light_model = ρ_dark + (1/α) * Δρ_obs
        # For this to be ≥ 0: α ≥ |Δρ_obs| / ρ_dark
        #
        # From the current (wrong) model: ρ_light_wrong < 0
        # This means the model's α is too small
        #
        # The minimum valid α would make ρ_light = 0 at these positions
        # α_min = Δρ_obs / ρ_dark where Δρ_obs comes from the data
        #
        # As a proxy, if ρ_light_model < 0, the ratio |ρ_light_model|/ρ_dark
        # tells us roughly how much α needs to increase

        if negative_mask.any():
            # For atoms with negative light density
            rho_light_neg = rho_light[negative_mask]
            rho_dark_at_neg = rho_dark[negative_mask]

            # The "missing" density that would need to be added
            # to make light density non-negative
            missing = -rho_light_neg

            # Rough estimate: if current α gives negative density,
            # we need α to be larger by factor of roughly (1 + missing/rho_dark)
            # This is a heuristic, not exact
            correction_factor = (missing / (rho_dark_at_neg + 1e-6)).max()

            # Find worst atoms
            worst_idx = torch.argsort(rho_light)[:5]  # 5 most negative
        else:
            correction_factor = torch.tensor(0.0)
            worst_idx = torch.tensor([])

        return {
            'rho_dark': rho_dark,
            'rho_light': rho_light,
            'rho_ratio': rho_ratio,
            'negative_mask': negative_mask,
            'n_negative': negative_mask.sum().item(),
            'n_total': len(rho_light),
            'fraction_negative': negative_mask.float().mean().item(),
            'min_rho_light': rho_light.min().item(),
            'correction_factor': correction_factor.item(),
            'worst_atoms': worst_idx,
        }

    def estimate_alpha_floor_from_difference_map(
        self,
        hkl: torch.Tensor,
        delta_F_obs: torch.Tensor,
        sigma_diff: torch.Tensor,
        n_peaks: int = 10,
        sigma_cutoff: float = 3.0,
    ) -> Dict:
        """
        Estimate alpha floor from significant negative peaks in difference map.

        For each significant negative peak in the difference map, estimate
        the minimum α that could produce that peak without requiring negative
        density in the light state.

        Parameters
        ----------
        hkl : torch.Tensor
            Miller indices.
        delta_F_obs : torch.Tensor
            Observed difference amplitudes (can be negative).
        sigma_diff : torch.Tensor
            Uncertainties on difference amplitudes.
        n_peaks : int, optional
            Number of peaks to analyze. Default is 10.
        sigma_cutoff : float, optional
            Minimum significance (|ΔF|/σ) for peaks. Default is 3.0.

        Returns
        -------
        dict
            Dictionary with alpha floor estimates.
        """
        # Find significant negative differences
        significance = delta_F_obs / sigma_diff
        negative_sig = significance < -sigma_cutoff

        if not negative_sig.any():
            return {
                'alpha_floor': 0.0,
                'message': 'No significant negative peaks found',
                'n_negative_peaks': 0,
            }

        # Get the most negative peaks
        neg_indices = torch.where(negative_sig)[0]
        neg_values = delta_F_obs[neg_indices]
        sorted_idx = torch.argsort(neg_values)[:n_peaks]
        peak_indices = neg_indices[sorted_idx]

        # For each peak, estimate required α
        # The negative ΔF comes from atoms leaving their dark positions
        # |ΔF_neg| ≈ α × F_dark_contribution
        # So α ≈ |ΔF_neg| / F_dark_contribution

        # Compute F_dark at these reflections
        with torch.no_grad():
            fcalc_dark = self.model_dark(hkl, recalc=True)
            F_dark = torch.abs(fcalc_dark)

        # Estimate alpha from each peak
        neg_dF = torch.abs(delta_F_obs[peak_indices])
        F_dark_at_peaks = F_dark[peak_indices]

        # α ≈ |ΔF| / F_dark (rough estimate)
        alpha_estimates = neg_dF / (F_dark_at_peaks + 1e-6)

        return {
            'alpha_floor': alpha_estimates.max().item(),
            'alpha_estimates': alpha_estimates.tolist(),
            'peak_indices': peak_indices.tolist(),
            'peak_dF': neg_dF.tolist(),
            'n_negative_peaks': len(peak_indices),
        }


class NegativeDensityPenalty(nn.Module):
    """
    Loss term that penalizes negative electron density in the MIXED model.

    This provides a soft constraint that prevents the activation fraction
    from being too small (which would require unphysical negative density).

    The key insight: the MIXED state (not pure light) should have non-negative
    density everywhere. If α is too small and atoms have moved, the mixed
    model might predict negative density at some positions, which is unphysical.

    Parameters
    ----------
    mixed_model : MixedModel
        The mixed model (combines dark and light states with fractions).
    model_dark : ModelFT
        The dark/ground state model (provides reference positions to check).
    hkl : torch.Tensor
        Miller indices for density calculation.
    atom_mask : torch.Tensor, optional
        Mask selecting which atoms to monitor.
    check_grid : bool, optional
        If True, also check density on a grid (more thorough but slower).
        Default is False.
    """

    def __init__(
        self,
        mixed_model: "MixedModel",
        model_dark: "ModelFT",
        hkl: torch.Tensor,
        atom_mask: Optional[torch.Tensor] = None,
        check_grid: bool = False,
    ):
        super().__init__()
        self.mixed_model = mixed_model
        self.model_dark = model_dark
        self.register_buffer('hkl', hkl)
        self.atom_mask = atom_mask
        self.check_grid = check_grid

        # Pre-compute dark positions in fractional coordinates
        with torch.no_grad():
            xyz_dark = model_dark.xyz()
            cell = model_dark.cell
            frac_dark = cell.cartesian_to_fractional(xyz_dark)
            if atom_mask is not None:
                frac_dark = frac_dark[atom_mask]
            self.register_buffer('frac_positions', frac_dark)

    def forward(self) -> torch.Tensor:
        """
        Compute penalty for negative density in mixed model.

        Returns
        -------
        torch.Tensor
            Scalar penalty value (0 if no negative density).
        """
        # Get mixed model structure factors (includes the α weighting)
        fcalc_mixed = self.mixed_model(self.hkl, recalc=True)

        # Compute density at dark atom positions
        h_dot_r = torch.matmul(self.frac_positions, self.hkl.T.float())
        phase = torch.angle(fcalc_mixed)
        amplitude = torch.abs(fcalc_mixed)

        density = (amplitude.unsqueeze(0) * torch.cos(2 * torch.pi * h_dot_r + phase.unsqueeze(0))).sum(dim=1)
        density = density / len(self.hkl)

        # Penalize negative density (ReLU-like penalty)
        # Use a soft margin to avoid penalizing small numerical fluctuations
        margin = 0.1  # Small positive margin
        negative_density = torch.relu(-(density - margin))

        # Return mean squared negative density
        return (negative_density ** 2).mean()


class DisplacementRegularizer(nn.Module):
    """
    Regularizer that penalizes large atomic displacements from reference structure.

    This directly breaks the α-δF degeneracy by favoring solutions where atoms
    haven't moved too far from the dark structure, which implies larger α.

    The loss is: mean((xyz_light - xyz_dark)²)

    Parameters
    ----------
    model_light : ModelFT
        The light model being refined.
    model_dark : ModelFT
        The dark reference model (frozen).
    atom_mask : torch.Tensor, optional
        Boolean mask selecting which atoms to include.
    max_displacement : float, optional
        Maximum allowed displacement in Angstroms. Displacements beyond this
        are penalized quadratically. Default is 2.0 Å.
    """

    def __init__(
        self,
        model_light: "ModelFT",
        model_dark: "ModelFT",
        atom_mask: Optional[torch.Tensor] = None,
        max_displacement: float = 2.0,
    ):
        super().__init__()
        self.model_light = model_light
        self.model_dark = model_dark
        self.atom_mask = atom_mask
        self.max_displacement = max_displacement

        # Store reference dark positions
        with torch.no_grad():
            xyz_dark = model_dark.xyz()
            if atom_mask is not None:
                xyz_dark = xyz_dark[atom_mask]
            self.register_buffer('xyz_dark_ref', xyz_dark.clone())

    def forward(self) -> torch.Tensor:
        """
        Compute displacement penalty.

        Returns
        -------
        torch.Tensor
            Mean squared displacement penalty.
        """
        xyz_light = self.model_light.xyz()

        if self.atom_mask is not None:
            xyz_light = xyz_light[self.atom_mask]

        # Compute per-atom displacement
        displacement = xyz_light - self.xyz_dark_ref
        dist_sq = (displacement ** 2).sum(dim=1)

        # Penalize ALL movement proportionally (mean squared displacement)
        # This directly favors smaller movements → larger α
        return dist_sq.mean()


class DifferenceAmplitudeRegularizer(nn.Module):
    """
    Regularizer that encourages consistency between α and difference amplitudes.

    The key insight: the ratio of calculated to observed difference amplitudes
    should be consistent. If α is too small, the model compensates by making
    larger structural changes, which changes this ratio in a detectable way.

    This regularizer penalizes deviations from the expected relationship:
        |ΔF_calc| ≈ |ΔF_obs|

    When α is correct and the structure is correct, these should match.
    When α is too small and structure has moved too far, the pattern of
    |ΔF_calc| vs |ΔF_obs| will be distorted.

    Parameters
    ----------
    dataset_collection : DatasetCollection
        Collection with 'dark' and 'light' datasets.
    mixed_model : MixedModel
        The mixed model being refined.
    model_dark : ModelFT
        The dark reference model.
    """

    def __init__(
        self,
        dataset_collection,
        mixed_model: "MixedModel",
        model_dark: "ModelFT",
    ):
        super().__init__()
        self.mixed_model = mixed_model
        self.model_dark = model_dark
        self._dataset_collection = dataset_collection

        # Get observed amplitudes
        _, F_light, _, _ = dataset_collection['light']()
        _, F_dark, _, _ = dataset_collection['dark']()

        if hasattr(F_light, "get_data"):
            F_light = F_light.get_data()
        if hasattr(F_dark, "get_data"):
            F_dark = F_dark.get_data()

        self.register_buffer('_dF_obs', F_light - F_dark)
        self.register_buffer('_F_obs_dark', F_dark)

    @property
    def hkl(self):
        return self._dataset_collection.hkl

    def forward(self) -> torch.Tensor:
        """
        Compute regularization loss.

        Penalizes the variance in the ratio |ΔF_calc|/|ΔF_obs|.
        If α and structure are correct, this ratio should be ~1 everywhere.
        If α is wrong, this ratio will have high variance.
        """
        hkl = self.hkl

        fcalc_mixed = self.mixed_model(hkl, recalc=True)
        fcalc_dark = self.model_dark(hkl, recalc=True)

        dF_calc = torch.abs(fcalc_mixed) - torch.abs(fcalc_dark)

        # Only consider reflections with significant observed difference
        significant = torch.abs(self._dF_obs) > 0.1 * self._F_obs_dark

        if significant.sum() < 100:
            return torch.tensor(0.0, device=hkl.device)

        dF_obs_sig = self._dF_obs[significant]
        dF_calc_sig = dF_calc[significant]

        # The residual between calculated and observed differences
        # Should be small when both α and structure are correct
        residual = dF_calc_sig - dF_obs_sig

        # Normalized by observed amplitude to make it scale-invariant
        normalized_residual = residual / (torch.abs(dF_obs_sig) + 1e-6)

        # Penalize large residuals
        return (normalized_residual ** 2).mean()
