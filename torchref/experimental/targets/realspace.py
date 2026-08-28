"""
Real-Space Targets for Crystallographic Refinement.

This module provides target (loss) functions that compare electron density
maps in real space rather than reciprocal space. Two targets are provided:

1. RealSpaceCorrelationTarget: Maximizes RSCC between the observed map and
   Fcalc density. The observed map labelled ``"2mFo-DFc"`` is in fact the
   unweighted 2Fo-Fc approximation (m=D=1); see ``RealSpaceTarget``.
2. RealSpaceDifferenceTarget: Minimizes mean squared Fo-Fc difference density

Both targets use a molecular mask (inverse of solvent mask) to restrict
comparison to the protein region, and follow the phase detachment pattern
from PhaseInformedDifferenceTarget to ensure correct gradient flow.
"""

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch

from torchref.base.reciprocal.grid_operations import place_on_grid
from torchref.symmetry import SpaceGroup
from torchref.utils.stats import (
    VERBOSITY_DEBUG,
    VERBOSITY_DETAILED,
    VERBOSITY_STANDARD,
    StatEntry,
    stat,
)

from torchref.refinement.targets.base import DataTarget

if TYPE_CHECKING:
    from torchref.io.datasets import ReflectionData, DatasetCollection
    from torchref.model import MixedModel
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

        # P1 expansion cache (ASU → P1 mapping)
        self._hkl_p1 = None
        self._p1_indices = None
        self._p1_phase_shifts = None

    def _ensure_grid(self):
        """Ensure model's SfFFT grid is set up."""
        if self._model is None:
            raise RuntimeError("No model set for RealSpaceTarget")
        if self._model.gridsize is None:
            self._model.setup_grid()

    def _get_data_p1(self) -> "ReflectionData":
        """Return P1-expanded ReflectionData, cached after first call."""
        if self._data_p1 is None:
            self._data_p1 = self._data.expand_to_p1()
        return self._data_p1

    def _ensure_p1_expansion(self):
        """Compute and cache the ASU → P1 expansion mapping."""
        if self._hkl_p1 is not None:
            return
        sg = self._data.spacegroup or SpaceGroup("P1")
        hkl_p1, indices, phase_shifts = sg.expand_hkl(
            self._data.hkl,
            include_friedel=True,
            remove_absences=True,
            device=self._data.hkl.device,
        )
        self._hkl_p1 = hkl_p1
        self._p1_indices = indices
        self._p1_phase_shifts = phase_shifts

    def _expand_to_p1(self, fcalc: torch.Tensor) -> torch.Tensor:
        """Expand ASU complex structure factors to P1 using cached mapping."""
        self._ensure_p1_expansion()
        fcalc_p1 = fcalc[self._p1_indices]
        return fcalc_p1 * torch.exp(1j * self._p1_phase_shifts)

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

        Scaling is applied at ASU level before P1 expansion.

        Returns
        -------
        torch.Tensor
            3D real-space density map.
        """
        self._ensure_p1_expansion()

        # Expand Fobs to P1 using the same index mapping as Fcalc
        # (amplitudes are invariant under symmetry, no phase shift needed)
        fobs_p1 = self._data.F[self._p1_indices]

        # Compute and scale Fcalc at ASU level, then expand to P1
        fcalc_asu = self.get_fcalc_scaled()
        fcalc_p1 = self._expand_to_p1(fcalc_asu)

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
        grid = place_on_grid(self._hkl_p1, coefficients, gridsize, enforce_hermitian=False)
        return torch.fft.ifftn(grid, dim=(0, 1, 2), norm="forward").real

    def _compute_model_density(self) -> torch.Tensor:
        """
        Compute model electron density via Fcalc -> grid -> IFFT.

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
        grid = place_on_grid(self._hkl_p1, fcalc_p1, gridsize, enforce_hermitian=False)
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


class RealSpaceExtrapolatedTarget(RealSpaceTarget):
    """
    Real-space correlation target using extrapolated pure-light density.

    Computes the RSCC between an extrapolated pure-light electron density map
    and the light model's Fcalc density within the molecular mask.  The loss
    is ``1 - RSCC``.

    The extrapolation combines observed dark/light amplitudes with
    model-derived phases:

        F_extra = (F_light * exp(i*phi_mixed) - w_dark * F_dark * exp(i*phi_dark)) / w_light

    where w_dark, w_light are population fractions from the mixed model.

    Parameters
    ----------
    dataset_collection : DatasetCollection
        Collection containing 'dark' and 'light' datasets (aligned HKL).
    model_dark : ModelFT
        Dark-state model (for dark phases).
    model_light : ModelFT
        Light-state model (gradients flow through this model's density).
    model_mixed : MixedModel
        Mixed model (for mixed-state phases and population fractions).
    scaler_dark : Scaler, optional
        Scaler for dark Fcalc.
    scaler_mixed : Scaler, optional
        Scaler for mixed Fcalc.
    scaler_light : Scaler, optional
        Scaler for light model Fcalc (model density side).
    mask_solvent : bool, optional
        Whether to apply molecular mask. Default True.
    solvent_radius : float, optional
        Probe radius for mask in Angstroms. Default 1.1.
    erosion_radius : float, optional
        Radius for mask erosion in Angstroms. Default 0.9.
    verbose : int, optional
        Verbosity level. Default 0.

    Notes
    -----
    The ``name`` attribute is ``"realspace_extrapolated"`` (underscore),
    whereas the sibling targets use the slash convention
    (``"realspace/correlation"``, ``"realspace/difference"``).  This
    inconsistency is intentional for now but may be normalised in a future
    release if ``name`` becomes part of the logging API.
    """

    name: str = "realspace_extrapolated"

    def __init__(
        self,
        dataset_collection: "DatasetCollection",
        model_dark: "ModelFT" = None,
        model_light: "ModelFT" = None,
        model_mixed: "MixedModel" = None,
        scaler_dark: "Scaler" = None,
        scaler_mixed: "Scaler" = None,
        scaler_light: "Scaler" = None,
        mask_solvent: bool = True,
        solvent_radius: float = 1.1,
        erosion_radius: float = 0.9,
        verbose: int = 0,
    ):
        if "dark" not in dataset_collection:
            raise ValueError("DatasetCollection must contain a 'dark' dataset")
        if "light" not in dataset_collection:
            raise ValueError("DatasetCollection must contain a 'light' dataset")

        # Parent uses model_light for mask/model density, light data for P1 grid
        super().__init__(
            data=dataset_collection["light"],
            model=model_light,
            scaler=scaler_light,
            map_type="2mFo-DFc",  # overridden by _compute_observed_map
            mask_solvent=mask_solvent,
            solvent_radius=solvent_radius,
            erosion_radius=erosion_radius,
            verbose=verbose,
            target_value=0.0,
            sigma=0.5,
        )

        # Additional models and scalers for phase computation
        self.add_module("_model_dark", model_dark)
        self.add_module("_model_mixed", model_mixed)
        self.add_module("_scaler_dark", scaler_dark)
        self.add_module("_scaler_mixed", scaler_mixed)

        # Store references to datasets
        self._data_dark = dataset_collection["dark"]
        self._data_light = dataset_collection["light"]

        # Precompute observed data as buffers
        self._setup_data()

        # P1 expansion cache
        self._hkl_p1 = None
        self._p1_indices = None
        self._p1_phase_shifts = None

    def _setup_data(self):
        """Extract and store observed data from datasets as buffers."""
        hkl = self._data_light.hkl
        F_light, sigma_light = self._data_light.get_corrected_data()
        rfree_light = self._data_light.rfree_flags
        valid_light = self._data_light.masks().to(torch.bool)
        F_dark, sigma_dark = self._data_dark.get_corrected_data()
        valid_dark = self._data_dark.masks().to(torch.bool)

        # Combined validity mask
        valid_mask = torch.ones_like(F_light, dtype=torch.bool)
        if valid_light is not None:
            valid_mask = valid_mask & valid_light
        if valid_dark is not None:
            valid_mask = valid_mask & valid_dark

        # Zero invalid values to prevent NaN propagation
        F_light = torch.where(valid_mask, F_light, torch.zeros_like(F_light))
        F_dark = torch.where(valid_mask, F_dark, torch.zeros_like(F_dark))

        self.register_buffer("_hkl", hkl)
        self.register_buffer("_F_obs_light", F_light)
        self.register_buffer("_F_obs_dark", F_dark)
        self.register_buffer("_valid_mask", valid_mask)

    def _ensure_p1_expansion(self):
        """Expand HKL to P1, caching the result."""
        if self._hkl_p1 is not None:
            return

        spacegroup = self._data_light.spacegroup
        hkl_p1, indices, phase_shifts = spacegroup.expand_hkl(
            self._hkl,
            include_friedel=True,
            remove_absences=True,
            device=self._hkl.device,
        )
        self._hkl_p1 = hkl_p1
        self._p1_indices = indices
        self._p1_phase_shifts = phase_shifts

    def _compute_observed_map(self) -> torch.Tensor:
        """
        Compute extrapolated pure-light electron density map.

        Phases are detached to prevent circular gradients where atoms can
        minimise loss by rotating model phases rather than matching the true
        structure.  Population fractions retain gradients, providing a strong
        self-consistency signal for fraction refinement: the fractions that
        produce an extrapolated map most consistent with the model density.

        Returns
        -------
        torch.Tensor
            3D real-space density map of the extrapolated pure-light state.
        """
        self._ensure_p1_expansion()

        # Compute model phases at ASU HKL — detached to prevent circular
        # gradients where atoms minimise loss by rotating phases rather than
        # matching the true structure.  Phases still update each step from
        # the current model; they just don't contribute to this target's gradient.
        fcalc_dark = self._model_dark(self._hkl)
        if self._scaler_dark is not None:
            fcalc_dark = self._scaler_dark(fcalc_dark)
        phi_dark = torch.angle(fcalc_dark).detach()

        fcalc_mixed = self._model_mixed(self._hkl)
        if self._scaler_mixed is not None:
            fcalc_mixed = self._scaler_mixed(fcalc_mixed)
        phi_mixed = torch.angle(fcalc_mixed).detach()

        # Population fractions — gradients retained for fraction refinement
        fractions = self._model_mixed.fractions
        w_dark = fractions[0]
        w_light = fractions[1]

        # Phase observed amplitudes and extrapolate
        F_obs_dark_phased = self._F_obs_dark * torch.exp(1j * phi_dark)
        F_obs_light_phased = self._F_obs_light * torch.exp(1j * phi_mixed)
        F_extra = (F_obs_light_phased - w_dark * F_obs_dark_phased) / w_light

        # Zero out invalid reflections
        F_extra = torch.where(self._valid_mask, F_extra, torch.zeros_like(F_extra))

        # Expand to P1
        F_extra_p1 = F_extra[self._p1_indices]
        # Apply phase shifts from symmetry translations
        F_extra_p1 = F_extra_p1 * torch.exp(1j * self._p1_phase_shifts)

        # FFT to real space
        gridsize = self._get_gridsize()
        grid = place_on_grid(
            self._hkl_p1, F_extra_p1, gridsize, enforce_hermitian=False
        )
        return torch.fft.ifftn(grid, dim=(0, 1, 2), norm="forward").real

    def _compute_model_density(self) -> torch.Tensor:
        """
        Compute model density from light model Fcalc.

        Uses the P1 expansion computed by this target (not the parent's
        data_p1 cache) for consistency.

        Returns
        -------
        torch.Tensor
            3D real-space model density map.
        """
        self._ensure_p1_expansion()

        # Compute light Fcalc at ASU HKL
        fcalc = self._model(self._hkl)
        if self._scaler is not None:
            fcalc = self._scaler(fcalc)

        # Expand to P1
        fcalc_p1 = fcalc[self._p1_indices]
        fcalc_p1 = fcalc_p1 * torch.exp(1j * self._p1_phase_shifts)

        gridsize = self._get_gridsize()
        grid = place_on_grid(
            self._hkl_p1, fcalc_p1, gridsize, enforce_hermitian=False
        )
        return torch.fft.ifftn(grid, dim=(0, 1, 2), norm="forward").real

    def forward(self) -> torch.Tensor:
        """
        Compute 1 - RSCC between extrapolated map and model density.

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

        # RSCC via Pearson correlation
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
        Get statistics for the extrapolated real-space target.

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
