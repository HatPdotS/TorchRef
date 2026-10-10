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

from typing import TYPE_CHECKING, Dict, Optional

import torch

if TYPE_CHECKING:
    from torchref.model import ModelFT


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
        Stored but currently unused. Density is evaluated only at atom
        positions via Fourier summation, not on a grid. Default is 0.5.
    negative_threshold : float, optional
        Stored but currently unused. Negative-density detection uses a
        hardcoded ``rho_light < 0`` test rather than this threshold.
        Default is -0.5.

    Examples
    --------
    Basic usage::

        diagnostic = OccupancyFloorDiagnostic(model_dark, model_light_refine)
        result = diagnostic.estimate_alpha_floor_from_difference_map(
            hkl, delta_F_obs, sigma_diff
        )
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

        Sum ``Re[F_calc(h) exp(-2πi h·r)]`` over the given reflections, the
        synthesis that inverts TorchRef's ``F = Σ f exp(+2πi h·x)``.

        Parameters
        ----------
        model : ModelFT
            Model to compute density from.
        positions : torch.Tensor
            Positions in fractional coordinates, shape (N, 3).
        hkl : torch.Tensor
            P1 Miller indices, shape (M, 3), one per Friedel pair (e.g.
            ``spacegroup.expand_hkl(hkl, include_friedel=False)[0]``); the sum
            is not symmetry-expanded, so ASU indices give a filtered density
            that is negative at some atom sites even for identical models.

        Returns
        -------
        torch.Tensor
            Electron density values at each position, shape (N,).
        """
        with torch.no_grad():
            fcalc = model(hkl, recalc=True)

            # Match hkl to the positions' (configured) dtype so the matmul
            # does not raise under a float64 config.
            h_dot_r = torch.matmul(positions, hkl.T.to(dtype=positions.dtype))

            # ρ(r) = Σ_h |F(h)| cos(2π h·r - φ(h)); a + sign gives ρ(-r).
            phase = torch.angle(fcalc)  # (M,)
            amplitude = torch.abs(fcalc)  # (M,)

            density = (
                amplitude.unsqueeze(0)
                * torch.cos(2 * torch.pi * h_dot_r - phase.unsqueeze(0))
            ).sum(dim=1)

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
            P1 Miller indices, shape (M, 3); see ``compute_density_at_positions``.
        atom_mask : torch.Tensor, optional
            Boolean mask selecting which atoms to analyze (e.g., waters only).

        Returns
        -------
        dict
            ``rho_dark``, ``rho_light``, ``rho_ratio`` (ρ_light / ρ_dark) and
            ``negative_mask`` (ρ_light < 0) per analysed atom; the counts
            ``n_negative``, ``n_total``, ``fraction_negative``; ``min_rho_light``;
            ``correction_factor`` (max -ρ_light / ρ_dark over negative atoms, else 0)
            and ``worst_atoms`` (indices of the 5 lowest ρ_light, empty if none < 0).
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
        Estimate alpha floor from significantly negative difference amplitudes.

        For the most negative reflections with ΔF/σ below ``-sigma_cutoff``,
        estimate α as ``|ΔF_obs| / |F_calc,dark|`` and report the largest.

        Parameters
        ----------
        hkl : torch.Tensor
            Miller indices.
        delta_F_obs : torch.Tensor
            Observed difference amplitudes (can be negative).
        sigma_diff : torch.Tensor
            Uncertainties on difference amplitudes.
        n_peaks : int, optional
            Number of most negative reflections to analyze. Default is 10.
        sigma_cutoff : float, optional
            Minimum significance (-ΔF/σ) for a reflection. Default is 3.0.

        Returns
        -------
        dict
            ``alpha_floor`` and ``n_negative_peaks``; when reflections pass the
            cutoff also ``alpha_estimates``, ``peak_indices`` and ``peak_dF``,
            otherwise ``message``.
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
