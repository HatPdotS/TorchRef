"""
Real-Space Targets for Crystallographic Refinement.

This module provides target (loss) functions that compare electron density
maps in real space rather than reciprocal space. Two targets are provided:

1. RealSpaceCorrelationTarget: Maximizes RSCC between 2mFo-DFc and Fcalc density
2. RealSpaceDifferenceTarget: Minimizes mean squared Fo-Fc difference density

Both targets use a molecular mask (inverse of solvent mask) to restrict
comparison to the protein region, and follow the phase detachment pattern
from PhaseInformedDifferenceTarget to ensure correct gradient flow.
"""

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch

from torchref.base.reciprocal.grid_operations import place_on_grid
from torchref.symmetry.grid_utils import calculate_optimal_grid_size
from torchref.utils.stats import (
    VERBOSITY_DEBUG,
    VERBOSITY_DETAILED,
    VERBOSITY_STANDARD,
    StatEntry,
    stat,
)

from .targets import DataTarget

if TYPE_CHECKING:
    from torchref.io.datasets import ReflectionData
    from torchref.model.model_ft import ModelFT
    from torchref.scaling.scaler_base import Scaler


class RealSpaceTarget(DataTarget):
    """
    Base class for real-space electron density targets.

    Inherits from DataTarget to get model, data, and scaler references.
    Provides common infrastructure for computing observed maps, model density,
    and molecular masks used by the concrete subclasses.

    Gradient Flow Design
    --------------------
    - Model density: gradients flow through Fcalc -> grid -> IFFT -> density
    - Observed map (2mFo-DFc): phases and |Fcalc| detached, no gradients
    - Observed map (Fo-Fc): |Fcalc| retains gradients, phases detached
    - Molecular mask: boolean, no gradients

    Parameters
    ----------
    data : ReflectionData
        Observed reflection data.
    model : ModelFT
        Model for computing Fcalc.
    scaler : Scaler, optional
        Scaler for Fcalc (applied before map coefficient computation).
    map_type : str
        ``"2mFo-DFc"`` or ``"Fo-Fc"``.
    mask_solvent : bool
        Whether to apply molecular mask. Default True.
    solvent_radius : float
        Probe radius for mask dilation in Angstroms. Default 1.1.
    erosion_radius : float
        Radius for mask erosion in Angstroms. Default 0.9.
    verbose : int
        Verbosity level. Default 0.
    target_value : float
        Target value for loss. Default 0.0.
    sigma : float
        Sigma for weighting. Default 0.5.
    """

    VALID_MAP_TYPES = ("2mFo-DFc", "Fo-Fc")

    def __init__(
        self,
        data: "ReflectionData" = None,
        model: "ModelFT" = None,
        scaler: "Scaler" = None,
        map_type: str = "2mFo-DFc",
        mask_solvent: bool = True,
        solvent_radius: float = 1.1,
        erosion_radius: float = 0.9,
        verbose: int = 0,
        target_value: float = 0.0,
        sigma: float = 0.5,
    ):
        super().__init__(
            data=data, model=model, scaler=scaler,
            verbose=verbose, target_value=target_value, sigma=sigma,
        )
        if map_type not in self.VALID_MAP_TYPES:
            raise ValueError(
                f"map_type must be one of {self.VALID_MAP_TYPES}, got '{map_type}'"
            )
        self.map_type = map_type
        self._mask_solvent = mask_solvent
        self._solvent_radius = solvent_radius
        self._erosion_radius = erosion_radius

        # Caches (not registered as buffers since they're lazily computed)
        self._data_p1 = None
        self._molecular_mask = None
        self._gridsize = None

    def _ensure_grid(self):
        """Ensure model's SfFFT grid is set up."""
        if self._model is None:
            raise RuntimeError("No model set for RealSpaceTarget")
        if self._model.real_space_grid is None:
            self._model.setup_grid()

    def _get_data_p1(self) -> "ReflectionData":
        """Return P1-expanded ReflectionData, cached after first call."""
        if self._data_p1 is None:
            self._data_p1 = self._data.expand_to_p1()
        return self._data_p1

    def _get_gridsize(self) -> Tuple[int, int, int]:
        """
        Get grid size for map computation.

        Uses the model's FFT grid size to ensure compatibility with
        the molecular mask (which is built on the model's grid).
        """
        if self._gridsize is not None:
            return self._gridsize

        self._ensure_grid()
        gs = self._model.fft.gridsize
        self._gridsize = tuple(int(x) for x in gs)
        return self._gridsize

    def _compute_observed_map(self) -> torch.Tensor:
        """
        Compute observed electron density map.

        For ``"2mFo-DFc"``: ``(2*Fobs - |Fcalc|) * exp(i * phi_calc)``
        with both |Fcalc| and phases detached (no gradients on observed side).

        For ``"Fo-Fc"``: ``(Fobs - |Fcalc|) * exp(i * phi_calc)``
        with |Fcalc| retaining gradients and phases detached.

        Returns
        -------
        torch.Tensor
            3D real-space density map.
        """
        data_p1 = self._get_data_p1()
        hkl_p1, fobs_p1, _, _ = data_p1.data_indexed()

        # Compute Fcalc at P1 hkl (with optional scaling)
        fcalc_p1 = self.get_fcalc_scaled(hkl=hkl_p1)

        # Detach phases (following PhaseInformedDifferenceTarget pattern)
        phi_calc = torch.angle(fcalc_p1).detach()

        if self.map_type == "2mFo-DFc":
            # Fully detached observed side
            fcalc_amp = fcalc_p1.abs().detach()
            coefficients = (2.0 * fobs_p1 - fcalc_amp) * torch.exp(1j * phi_calc)
        elif self.map_type == "Fo-Fc":
            # |Fcalc| retains gradients, phases detached
            fcalc_amp = fcalc_p1.abs()
            coefficients = (fobs_p1 - fcalc_amp) * torch.exp(1j * phi_calc)
        else:
            raise ValueError(f"Unknown map_type: {self.map_type}")

        gridsize = self._get_gridsize()
        grid = place_on_grid(hkl_p1, coefficients, gridsize, enforce_hermitian=False)
        return torch.fft.ifftn(grid, dim=(0, 1, 2), norm="forward").real

    def _compute_model_density(self) -> torch.Tensor:
        """
        Compute model electron density via Fcalc -> grid -> IFFT.

        Retains full autograd graph for gradient flow through model parameters.

        Returns
        -------
        torch.Tensor
            3D real-space model density map.
        """
        data_p1 = self._get_data_p1()
        hkl_p1, _, _, _ = data_p1.data_indexed()

        # Compute Fcalc with optional scaling (retains gradients)
        fcalc_p1 = self.get_fcalc_scaled(hkl=hkl_p1)

        gridsize = self._get_gridsize()
        grid = place_on_grid(hkl_p1, fcalc_p1, gridsize, enforce_hermitian=False)
        return torch.fft.ifftn(grid, dim=(0, 1, 2), norm="forward").real

    def _build_molecular_mask(self):
        """
        Build molecular mask using SolventModel.

        The molecular mask is the inverse of the solvent mask:
        True = protein region, False = solvent region.
        """
        from torchref.scaling.solvent import SolventModel

        self._ensure_grid()

        with torch.no_grad():
            solvent = SolventModel(
                model=self._model,
                radius=self._solvent_radius,
                erosion_radius=self._erosion_radius,
                optimize_phase=False,
                verbose=0,
            )
            solvent_mask = solvent.get_solvent_mask()  # True = solvent
            self._molecular_mask = ~solvent_mask  # True = protein

    def _get_molecular_mask(self) -> torch.Tensor:
        """Get molecular mask, building on first call."""
        if self._molecular_mask is None:
            self._build_molecular_mask()
        return self._molecular_mask

    def update_mask(self):
        """Explicitly recompute the molecular mask."""
        self._molecular_mask = None
        self._build_molecular_mask()


class RealSpaceCorrelationTarget(RealSpaceTarget):
    """
    Real-space correlation coefficient (RSCC) target.

    Computes RSCC between a 2mFo-DFc observed map and Fcalc model density
    within the molecular mask. The loss is ``1 - RSCC``.

    The observed map uses detached model phases and amplitudes, so
    gradients flow only through the model density side.

    Parameters
    ----------
    data : ReflectionData
        Observed reflection data.
    model : ModelFT
        Model for computing Fcalc.
    scaler : Scaler, optional
        Scaler for Fcalc.
    mask_solvent : bool
        Whether to apply molecular mask. Default True.
    solvent_radius : float
        Probe radius for mask in Angstroms. Default 1.1.
    erosion_radius : float
        Radius for mask erosion in Angstroms. Default 0.9.
    verbose : int
        Verbosity level. Default 0.
    """

    name: str = "realspace/correlation"

    def __init__(
        self,
        data: "ReflectionData" = None,
        model: "ModelFT" = None,
        scaler: "Scaler" = None,
        mask_solvent: bool = True,
        solvent_radius: float = 1.1,
        erosion_radius: float = 0.9,
        verbose: int = 0,
    ):
        super().__init__(
            data=data,
            model=model,
            scaler=scaler,
            map_type="2mFo-DFc",
            mask_solvent=mask_solvent,
            solvent_radius=solvent_radius,
            erosion_radius=erosion_radius,
            verbose=verbose,
            target_value=0.0,
            sigma=0.5,
        )

    def forward(self) -> torch.Tensor:
        """
        Compute 1 - RSCC loss.

        Returns
        -------
        torch.Tensor
            Scalar loss value (1 - RSCC).
        """
        obs_map = self._compute_observed_map()
        model_density = self._compute_model_density()

        if self._mask_solvent:
            mask = self._get_molecular_mask()
            obs_vals = obs_map[mask]
            calc_vals = model_density[mask]
        else:
            obs_vals = obs_map.flatten()
            calc_vals = model_density.flatten()

        # RSCC = cov(obs, calc) / (std(obs) * std(calc) + eps)
        obs_centered = obs_vals - obs_vals.mean()
        calc_centered = calc_vals - calc_vals.mean()

        eps = 1e-8
        cov = (obs_centered * calc_centered).mean()
        std_obs = torch.sqrt((obs_centered**2).mean() + eps)
        std_calc = torch.sqrt((calc_centered**2).mean() + eps)

        rscc = cov / (std_obs * std_calc)

        return 1.0 - rscc

    def stats(self) -> Dict[str, StatEntry]:
        """
        Get statistics for the correlation target.

        Returns
        -------
        dict
            Dictionary with loss, rscc, and n_voxels.
        """
        with torch.no_grad():
            loss = self.forward()
            rscc = 1.0 - loss.item()

            if self._mask_solvent:
                mask = self._get_molecular_mask()
                n_voxels = int(mask.sum().item())
            else:
                n_voxels = int(self._compute_model_density().numel())

        return {
            "loss": stat(loss.item(), VERBOSITY_STANDARD),
            "rscc": stat(rscc, VERBOSITY_STANDARD),
            "n_voxels": stat(n_voxels, VERBOSITY_DETAILED),
        }


class RealSpaceDifferenceTarget(RealSpaceTarget):
    """
    Real-space Fo-Fc difference density target.

    Computes the mean squared Fo-Fc difference density within the
    molecular mask. This penalizes unexplained features in the
    difference map.

    The |Fcalc| component retains gradients while phases are detached,
    providing direct gradient signal for model refinement.

    Parameters
    ----------
    data : ReflectionData
        Observed reflection data.
    model : ModelFT
        Model for computing Fcalc.
    scaler : Scaler, optional
        Scaler for Fcalc.
    mask_solvent : bool
        Whether to apply molecular mask. Default True.
    solvent_radius : float
        Probe radius for mask in Angstroms. Default 1.1.
    erosion_radius : float
        Radius for mask erosion in Angstroms. Default 0.9.
    verbose : int
        Verbosity level. Default 0.
    """

    name: str = "realspace/difference"

    def __init__(
        self,
        data: "ReflectionData" = None,
        model: "ModelFT" = None,
        scaler: "Scaler" = None,
        mask_solvent: bool = True,
        solvent_radius: float = 1.1,
        erosion_radius: float = 0.9,
        verbose: int = 0,
    ):
        super().__init__(
            data=data,
            model=model,
            scaler=scaler,
            map_type="Fo-Fc",
            mask_solvent=mask_solvent,
            solvent_radius=solvent_radius,
            erosion_radius=erosion_radius,
            verbose=verbose,
            target_value=0.0,
            sigma=0.5,
        )

    def forward(self) -> torch.Tensor:
        """
        Compute mean squared Fo-Fc difference density.

        Returns
        -------
        torch.Tensor
            Scalar loss value (mean squared difference density).
        """
        diff_map = self._compute_observed_map()

        if self._mask_solvent:
            mask = self._get_molecular_mask()
            diff_vals = diff_map[mask]
        else:
            diff_vals = diff_map.flatten()

        return (diff_vals**2).mean()

    def stats(self) -> Dict[str, StatEntry]:
        """
        Get statistics for the difference target.

        Returns
        -------
        dict
            Dictionary with loss, rms_diff, mean_abs_diff, peak values, and n_voxels.
        """
        with torch.no_grad():
            diff_map = self._compute_observed_map()

            if self._mask_solvent:
                mask = self._get_molecular_mask()
                diff_vals = diff_map[mask]
                n_voxels = int(mask.sum().item())
            else:
                diff_vals = diff_map.flatten()
                n_voxels = int(diff_vals.numel())

            loss = (diff_vals**2).mean()
            rms_diff = torch.sqrt(loss)
            mean_abs_diff = diff_vals.abs().mean()
            max_pos_peak = diff_vals.max()
            max_neg_peak = diff_vals.min()

        return {
            "loss": stat(loss.item(), VERBOSITY_STANDARD),
            "rms_diff": stat(rms_diff.item(), VERBOSITY_STANDARD),
            "mean_abs_diff": stat(mean_abs_diff.item(), VERBOSITY_DETAILED),
            "max_pos_peak": stat(max_pos_peak.item(), VERBOSITY_DETAILED),
            "max_neg_peak": stat(max_neg_peak.item(), VERBOSITY_DETAILED),
            "n_voxels": stat(n_voxels, VERBOSITY_DETAILED),
        }
