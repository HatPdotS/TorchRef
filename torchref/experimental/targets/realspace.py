"""
Real-Space Targets for Crystallographic Refinement.

This module provides target (loss) functions that compare electron density
maps in real space rather than reciprocal space. Two targets are provided:

1. RealSpaceCorrelationTarget: Maximizes RSCC between the observed map and
   Fcalc density. The observed map labelled ``"2mFo-DFc"`` is in fact the
   unweighted 2Fo-Fc approximation (m=D=1); see ``RealSpaceTarget``.
2. RealSpaceDifferenceTarget: Minimizes mean squared Fo-Fc difference density

Both targets use a molecular mask (inverse of solvent mask) to restrict
comparison to the protein region, and detach the model phases in the
observed map so its gradient flows only through the amplitudes.
"""

from typing import TYPE_CHECKING, Dict, Tuple

import torch

from torchref.base.fourier.fft import fft
from torchref.base.reciprocal.grid_operations import place_on_grid
from torchref.symmetry import SpaceGroup
from torchref.utils.stats import (
    VERBOSITY_DETAILED,
    VERBOSITY_STANDARD,
    StatEntry,
    stat,
)

from torchref.refinement.targets.base import DataTarget

if TYPE_CHECKING:
    from torchref.io.datasets import ReflectionData
    from torchref.model.model_ft import ModelFT
    from torchref.scaling.scaler_base import Scaler


class RealSpaceTarget(DataTarget):
    """
    Base class for real-space electron density targets.

    Inherits from DataTarget to get model, data, and scaler references.
    Provides common infrastructure for computing observed maps, model density,
    and molecular masks used by the concrete subclasses. Both maps are
    synthesised from the valid work reflections only; the free set is left out.

    Gradient Flow Design
    --------------------
    - Model density: gradients flow through Fcalc -> grid -> FFT -> density
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
        ``"2mFo-DFc"`` or ``"Fo-Fc"``. Note the ``"2mFo-DFc"`` option is the
        *unweighted* 2Fo-Fc approximation (figure-of-merit ``m=1``, sigma_a
        weight ``D=1``: ``(2*Fobs - |Fcalc|) * exp(i*phi_calc)``), not a true
        likelihood-weighted 2mFo-DFc map. The string value is kept for
        backward compatibility.
    mask_solvent : bool
        Whether to apply molecular mask. Default True.
    solvent_radius : float
        Probe radius for mask dilation in Angstroms. Default 1.1.
    erosion_radius : float
        Radius for mask erosion in Angstroms. Default 0.9.
    verbose : int
        Verbosity level. Default 0.
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
    ):
        super().__init__(data=data, model=model, scaler=scaler, verbose=verbose)
        if map_type not in self.VALID_MAP_TYPES:
            raise ValueError(
                f"map_type must be one of {self.VALID_MAP_TYPES}, got '{map_type}'"
            )
        self.map_type = map_type
        self._mask_solvent = mask_solvent
        self._solvent_radius = solvent_radius
        self._erosion_radius = erosion_radius

        # Caches (not registered as buffers since they're lazily computed)
        self._molecular_mask = None

        # P1 expansion cache (ASU → P1 mapping)
        self._hkl_p1 = None
        self._p1_indices = None
        self._p1_phase_shifts = None

    def _ensure_p1_expansion(self):
        """Compute and cache the ASU → P1 expansion mapping."""
        if self._hkl_p1 is not None:
            return
        sg = self._data.spacegroup or SpaceGroup("P1")
        # One row per valid work reflection, so the free set stays unbiased;
        # anomalous F_obs is Bijvoet-averaged below.
        rows = self._data.bijvoet_representatives(self._data.work.mask)
        hkl_p1, indices, phase_shifts = sg.expand_hkl(
            self._data.hkl[rows],
            include_friedel=False,
            remove_absences=True,
            device=self._data.hkl.device,
        )
        self._hkl_p1 = hkl_p1
        self._p1_indices = rows[indices]
        self._p1_phase_shifts = phase_shifts

    def _expand_to_p1(self, fcalc: torch.Tensor) -> torch.Tensor:
        """Expand ASU complex structure factors to P1 using cached mapping.

        Only the symmetry copies are returned; ``place_on_grid`` with
        ``enforce_hermitian=True`` adds the Friedel half as ``conj(F)``.
        """
        self._ensure_p1_expansion()
        fcalc_p1 = fcalc[self._p1_indices]
        return fcalc_p1 * torch.exp(1j * self._p1_phase_shifts)

    def _get_gridsize(self) -> Tuple[int, int, int]:
        """Grid size for map computation: the model's, so it matches the
        molecular mask built on the model's grid."""
        if self._model is None:
            raise RuntimeError("No model set for RealSpaceTarget")
        return self._model.fft.grid_shape

    def _compute_observed_map(self) -> torch.Tensor:
        """
        Compute observed electron density map.

        For ``"2mFo-DFc"``: ``(2*Fobs - |Fcalc|) * exp(i * phi_calc)``
        with both |Fcalc| and phases detached (no gradients on observed side).

        For ``"Fo-Fc"``: ``(Fobs - |Fcalc|) * exp(i * phi_calc)``
        with |Fcalc| retaining gradients and phases detached.

        Scaling is applied at ASU level before P1 expansion.

        Returns
        -------
        torch.Tensor
            3D real-space density map.
        """
        self._ensure_p1_expansion()

        # Expand Fobs to P1 using the same index mapping as Fcalc
        # (amplitudes are invariant under symmetry, no phase shift needed)
        work = self._data.work.mask
        fobs_p1 = self._data.bijvoet_mean(self._data.F, work)[self._p1_indices]

        # Compute and scale Fcalc at ASU level, then expand to P1
        fcalc_asu = self.get_fcalc_scaled()
        fcalc_p1 = self._expand_to_p1(fcalc_asu)

        # Detached phases: the observed map's gradient flows only through |Fcalc|.
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
        grid = place_on_grid(
            self._hkl_p1, coefficients, gridsize, enforce_hermitian=True
        )
        return fft(grid)

    def _compute_model_density(self) -> torch.Tensor:
        """
        Compute model electron density via Fcalc -> grid -> FFT.

        Scaling is applied at ASU level before P1 expansion.
        Retains full autograd graph for gradient flow through model parameters.

        Returns
        -------
        torch.Tensor
            3D real-space model density map.
        """
        self._ensure_p1_expansion()

        # Compute and scale Fcalc at ASU level, then expand to P1
        fcalc_asu = self.get_fcalc_scaled()
        fcalc_p1 = self._expand_to_p1(fcalc_asu)

        gridsize = self._get_gridsize()
        grid = place_on_grid(self._hkl_p1, fcalc_p1, gridsize, enforce_hermitian=True)
        return fft(grid)

    def _build_molecular_mask(self):
        """
        Build molecular mask using SolventModel.

        The molecular mask is the inverse of the solvent mask:
        True = protein region, False = solvent region.
        """
        from torchref.scaling.solvent import SolventModel


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

    Computes RSCC between the observed map (the ``"2mFo-DFc"`` option is the
    unweighted 2Fo-Fc approximation, m=D=1) and Fcalc model density
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
