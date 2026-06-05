"""
Base class for crystallographic refinement.
"""

from typing import Any, Dict, Optional

import torch
from torch.nn import Module as nnModule

from torchref.config import get_default_device
from torchref.io import ReflectionData
from torchref.model.model_ft import ModelFT
from torchref.refinement.logger import Logger
from torchref.refinement.loss_state import LossState
from torchref.refinement.targets.adp.scaler_log_scale import (
    ScalerLogScaleTrendTarget,
)
from torchref.refinement.targets.adp.scaler_u import ScalerURegularizationTarget
from torchref.refinement.targets.combined import (
    TotalADPTarget,
    TotalGeometryTarget,
)

# Target system imports
from torchref.refinement.targets.xray import create_xray_target
from torchref.scaling.scaler import Scaler
from torchref.utils.debug_utils import DebugMixin
from torchref.utils.device_mixin import DeviceMixin
from torchref.utils.device_resolution import resolve_device


class Refinement(DeviceMixin, DebugMixin, nnModule):
    """
    Refinement class to handle the overall crystallographic refinement process.

    Supports two initialization patterns:

    1. Empty initialization (for state_dict loading)::

        refinement = Refinement()  # Creates empty shell with submodules
        refinement.load_state_dict(torch.load('refinement.pt'))

    2. Full initialization with file paths::

        refinement = Refinement(data_file='data.mtz', pdb='model.pdb')

    Parameters
    ----------
    data_file : str, optional
        Path to MTZ or CIF file containing reflection data.
    pdb : str, optional
        Path to PDB or CIF file containing initial model.
    cif : str, optional
        Path to CIF file for restraints.
    verbose : int, optional
        Verbosity level. Default is 1.
    max_res : float, optional
        Maximum resolution for reflections.
    device : torch.device, optional
        Computation device. Defaults to the configured device.current.
    weighter : LossWeightingModule, optional
        Loss weighting module. Creates default if None.
    nbins : int, optional
        Number of resolution bins. Default is 10.

    Attributes
    ----------
    device : torch.device
        Computation device.
    verbose : int
        Verbosity level.
    reflection_data : ReflectionData
        Reflection data container.
    model : ModelFT
        Structure factor model (includes lazy restraints via model.restraints).
    scaler : Scaler
        Scale factor calculator.
    weighter : LossWeightingModule
        Loss weighting module.
    """

    def __init__(
        self,
        data_file: str = None,
        pdb: str = None,
        cif=None,
        verbose: int = 1,
        max_res: float = None,
        device: Optional[torch.device] = None,
        nbins: int = 10,
        manual_weights: Dict[str, float] = None,
        component_weights: Dict[str, float] = None,
        column_names: Optional[Dict[str, str]] = None,
        wavelength: Optional[float] = 1.0,
        anomalous_threshold: float = 0.5,
        french_wilson: bool = True,
    ):
        """
        Initialize Refinement.

        If data_file and pdb are provided, fully initializes the refinement.
        If not provided (empty init), creates a shell with empty submodules
        ready for load_state_dict().

        Parameters
        ----------
        data_file : str, optional
            Path to MTZ or CIF file containing reflection data.
        pdb : str, optional
            Path to PDB or CIF file containing initial model.
        cif : str, optional
            Path to CIF file for restraints.
        verbose : int, optional
            Verbosity level. Default is 1.
        max_res : float, optional
            Maximum resolution for reflections.
        device : torch.device, optional
            Computation device. Defaults to the configured device.current.
        weighter : LossWeightingModule, optional
            Loss weighting module. Creates default if None.
        nbins : int, optional
            Number of resolution bins. Default is 10.
        french_wilson : bool, optional
            Whether to derive amplitudes from intensities via French-Wilson.
            Default True. Set False to use existing French-Wilson-corrected
            ``F``/``SIGF`` columns directly when the MTZ also carries
            intensities.
        """
        super().__init__()
        # Refinement constructs its own submodules from file paths, so
        # there is nothing to reconcile yet — ``resolve_device`` with no
        # modules just normalises ``device`` (or returns the default).
        self.device = resolve_device(device=device)
        self.verbose = verbose
        self.data_file = data_file
        self.pdb = pdb
        self.history = dict()
        self.max_res = max_res
        self.nbins = nbins
        self.lr = 1e-3
        # Wavelength drives f'/f'' anomalous scattering corrections in ModelFT.
        # Default 1.0 preserves prior behavior; set to the experimental wavelength
        # for anomalous (Bijvoet) refinement, or None to disable entirely.
        self.wavelength = wavelength
        self.anomalous_threshold = anomalous_threshold
        self.french_wilson = french_wilson

        # Persistent state and logger (created lazily)
        self._loss_state: Optional[LossState] = None
        self._logger: Optional[Logger] = None

        # Empty initialization - create empty submodules for state_dict loading
        if data_file is None and pdb is None:
            # Create empty submodules so state_dict keys exist
            self.reflection_data = ReflectionData(
                verbose=self.verbose, device=self.device
            )
            self.model = ModelFT(
                verbose=self.verbose,
                device=self.device,
                wavelength=self.wavelength,
                anomalous_threshold=self.anomalous_threshold,
            )
            self.scaler = Scaler(
                verbose=self.verbose, device=self.device, nbins=self.nbins
            )
            # Restraints are now lazy-loaded via model.restraints property
            self.weighter = None
            self.manual_weights = manual_weights if manual_weights is not None else {}
            self.component_weights = (
                component_weights if component_weights is not None else {}
            )
            return

        # Full initialization with file paths
        try:
            self.to(self.device)
            if isinstance(data_file, str):
                self.reflection_data = ReflectionData(
                    verbose=self.verbose, device=self.device
                )
                if data_file.endswith(".mtz"):
                    self.reflection_data.load_mtz(
                        data_file,
                        column_names=column_names,
                        french_wilson=self.french_wilson,
                    )
                elif data_file.endswith(".cif"):
                    self.reflection_data.load_cif(data_file)
                else:
                    raise ValueError(
                        f"Unsupported data file format: {data_file}. Supported formats are .mtz and .cif"
                    )
            if max_res is not None:
                try:
                    max_res_val = float(max_res)
                except (TypeError, ValueError):
                    raise ValueError(f"max_res must be a float > 0, got {max_res!r}")
                if max_res_val <= 0:
                    raise ValueError(f"max_res must be > 0, got {max_res_val}")
                self.reflection_data = self.reflection_data.cut_res(max_res_val)
                self.max_res = max_res_val
            else:
                self.max_res = self.reflection_data.get_max_res()
            self.model = ModelFT(
                verbose=self.verbose,
                max_res=self.max_res,
                device=self.device,
                wavelength=self.wavelength,
                anomalous_threshold=self.anomalous_threshold,
                # Apply the f'' (Bijvoet) term only when the data were loaded as
                # explicit Friedel pairs; merged data gate it off.
                apply_bijvoet=not self.reflection_data.friedel_merged,
            )
            if pdb.endswith(".cif"):
                self.model.load_cif(pdb)
            elif pdb.endswith(".pdb"):
                self.model.load_pdb(pdb)
            else:
                raise ValueError(
                    f"Unsupported model file format: {pdb}. Supported formats are .pdb and .cif"
                )

            self.setup_scaler()
            # Configure CIF path for lazy restraint building (restraints built on first access)
            self.model.set_restraints_cif(cif)
            self.model._build_restraints()

            self.manual_weights = manual_weights if manual_weights is not None else {}
            self.component_weights = (
                component_weights if component_weights is not None else {}
            )

            # Initialize target functions (instantiated once, evaluated each iteration)
            self._init_targets()

        except Exception as e:
            if self.verbose > 1:
                self.debug_on_error(e)
            raise e

    def _init_targets(self, xray_mode: str = "ml_sigmaa"):
        """
        Initialize target functions.

        Parameters
        ----------
        xray_mode : str, optional
            X-ray target mode. Options are 'gaussian', 'ls', 'ml',
            'ml_sigmaa', or 'bhattacharyya'. Default is 'ml_sigmaa'.
        """
        # X-ray targets (now accept model, data, scaler directly)
        self.xray_target_work = create_xray_target(
            model=self.model,
            data=self.reflection_data,
            scaler=self.scaler,
            mode=xray_mode,
            use_work_set=True,
            verbose=self.verbose,
        )
        self.xray_target_test = create_xray_target(
            model=self.model,
            data=self.reflection_data,
            scaler=self.scaler,
            mode=xray_mode,
            use_work_set=False,
            verbose=self.verbose,
        )

        # Total geometry target (handles bond, angle, torsion internally)
        # Geometry targets now accept model directly instead of refinement
        self.geometry_target = TotalGeometryTarget(self.model, verbose=self.verbose)

        self.adp_target = TotalADPTarget(self.model, verbose=self.verbose)

        self.setup_component_weighting()

        if self.verbose > 0:
            print(f"Initialized targets with xray_mode='{xray_mode}'")

    def set_xray_target_mode(self, mode: str):
        """
        Change the X-ray target mode.

        Parameters
        ----------
        mode : str
            X-ray target mode. Options: 'gaussian', 'ls', 'ml', 'ml_sigmaa', 'bhattacharyya'.
        """
        sigma_m_scale = getattr(self, "sigma_m_scale", 1.0)
        self.xray_target_work = create_xray_target(
            model=self.model,
            data=self.reflection_data,
            scaler=self.scaler,
            mode=mode,
            use_work_set=True,
            sigma_m_scale=sigma_m_scale,
            verbose=self.verbose,
        )
        self.xray_target_test = create_xray_target(
            model=self.model,
            data=self.reflection_data,
            scaler=self.scaler,
            mode=mode,
            use_work_set=False,
            sigma_m_scale=sigma_m_scale,
            verbose=self.verbose,
        )
        # Reset loss state since targets changed
        self.reset_loss_state()
        if self.verbose > 0:
            print(f"Changed X-ray target mode to '{mode}'")

    @property
    def data(self):
        """
        Expose reflection_data as 'data' for weighting module compatibility.

        Returns
        -------
        ReflectionData
            The reflection data container.
        """
        return self.reflection_data

    @property
    def loss_state(self) -> LossState:
        """
        Get or create the persistent LossState.

        The LossState is created once and reused across refinement cycles.
        Targets are registered once; weights are updated each cycle.

        Returns
        -------
        LossState
            The persistent loss state with targets registered.
        """
        if self._loss_state is None:
            self._loss_state = self._create_loss_state()
        return self._loss_state

    @property
    def logger(self) -> Logger:
        """
        Get or create the Logger for this refinement.

        Returns
        -------
        Logger
            Logger instance linked to the persistent LossState.
        """
        if self._logger is None:
            self._logger = Logger(
                state=self.loss_state,
                verbose=self.verbose,
            )
        return self._logger

    def reset_loss_state(self) -> None:
        """
        Reset the persistent LossState and Logger.

        Call this if targets need to be re-registered (e.g., after changing
        target modes or reinitializing targets).
        """
        self._loss_state = None
        self._logger = None

    def get_scales(self):
        if not hasattr(self, "scaler"):
            self.setup_scaler()
        self.scaler.initialize()
        self.reflection_data.find_outliers(self.model, self.scaler, z_threshold=5.0)
        self.scaler.refine_lbfgs()
        self.reflection_data.find_outliers(self.model, self.scaler, z_threshold=5.0)

    def setup_scaler(self):
        cls = getattr(self, "_scaler_class", None) or Scaler
        self.scaler = cls(
            self.model,
            self.reflection_data,
            nbins=self.nbins,
            verbose=self.verbose,
            device=self.device,
        )

    def parameters(self, recurse: bool = True):
        """
        Return unique parameters from this module and all submodules.

        Uses the default Module.parameters() to gather parameters, then removes
        duplicates while preserving order to avoid passing the same tensor
        multiple times to the optimizer.

        Parameters
        ----------
        recurse : bool, optional
            If True, yields parameters of this module and all submodules.
            Default is True.

        Returns
        -------
        list
            List of unique parameter tensors.
        """
        params = list[Any](super().parameters(recurse))
        seen = set()
        unique_params = []
        for p in params:
            pid = id(p)
            if pid not in seen:
                seen.add(pid)
                unique_params.append(p)
        return unique_params

    def get_fcalc(self, hkl=None, recalc=False):
        if hkl is None:
            # Signed HKL so Bijvoet mates (which share a canonical ASU index)
            # are evaluated at +h/-h and get distinct |F_calc| under anomalous
            # scattering. Falls back to canonical hkl when unavailable.
            hkl = self.reflection_data.hkl_for_sf()
        return self.model(hkl, recalc=recalc)

    def get_fcalc_scaled(self, hkl=None, recalc=False):
        fcalc = self.get_fcalc(hkl, recalc=recalc)
        fcalc_scaled = self.scaler(fcalc)
        return fcalc_scaled

    def adp_loss(self):
        """
        Compute total ADP loss using TotalADPTarget.

        This combines:

        - Bond-based similarity (SIMU-like)
        - Spread control (tighter than KL)
        - Bounds penalty

        Returns
        -------
        torch.Tensor
            Total ADP loss value.
        """
        return self.adp_target()

    def get_F_calc(self, hkl=None, recalc=False):
        return torch.abs(self.get_fcalc(hkl, recalc=recalc))

    def get_F_calc_scaled(self, hkl=None, recalc=False):
        return torch.abs(self.get_fcalc_scaled(hkl, recalc=recalc))

    def nll_xray(self):
        """
        Compute X-ray negative log-likelihood for work and test sets.

        Returns
        -------
        tuple of torch.Tensor
            Tuple of (work_nll, test_nll) tensors.
        """
        return self.xray_target_work(), self.xray_target_test()

    def xray_loss_work(self) -> torch.Tensor:
        """
        Compute X-ray loss on work set using instantiated target.

        Returns
        -------
        torch.Tensor
            X-ray loss on work set.
        """
        return self.xray_target_work()

    def xray_loss_test(self) -> torch.Tensor:
        """
        Compute X-ray loss on test set using instantiated target.

        Returns
        -------
        torch.Tensor
            X-ray loss on test set.
        """
        return self.xray_target_test()

    def bond_loss(self) -> torch.Tensor:
        """
        Compute bond length NLL via geometry_target.

        Returns
        -------
        torch.Tensor
            Bond length NLL loss.

        """
        return self.geometry_target.target_losses()["bond_target"]

    def angle_loss(self) -> torch.Tensor:
        """
        Compute angle NLL via geometry_target.

        Returns
        -------
        torch.Tensor
            Angle NLL loss.
        """
        return self.geometry_target.target_losses()["angle_target"]

    def torsion_loss(self) -> torch.Tensor:
        """
        Compute torsion angle NLL via geometry_target.

        Returns
        -------
        torch.Tensor
            Torsion angle NLL loss.
        """
        return self.geometry_target.target_losses()["torsion_target"]

    def geometry_loss(self) -> torch.Tensor:
        """
        Compute total geometry NLL using TotalGeometryTarget.

        Returns
        -------
        torch.Tensor
            Total geometry NLL loss.
        """
        return self.geometry_target()

    def loss(self):
        """
        Compute total loss using LossState pipeline.

        Creates a LossState, populates meta, caches losses, updates weights,
        and returns the aggregated weighted loss.

        Returns
        -------
        torch.Tensor
            Total weighted loss.
        """
        state = LossState(device=self.device)

        # Register targets
        state.register_target("xray_work", lambda: self.xray_target_work())
        state.register_targets(self.geometry_target)
        state.register_targets(self.adp_target)
        n_ref = int(self.reflection_data.hkl.shape[0])
        state.register_target(
            "adp/scaler_U",
            ScalerURegularizationTarget(self.scaler, n_reflections=n_ref),
        )
        state.register_target(
            "adp/scaler_log_scale",
            ScalerLogScaleTrendTarget(self.scaler, n_reflections=n_ref),
        )

        # Populate meta and update weights
        state = self.populate_state_meta(state)
        state = self.update_weights(state)

        return state.aggregate()

    def setup_component_weighting(self):
        """
        Set up component weighting with ResolutionWeighting + OverfittingWeighting.
        """
        from torchref.refinement.weighting.component_weighting import ComponentWeighting

        self.get_scales()

        self.component_weighting = ComponentWeighting(
            device=self.device,
            weights=self.manual_weights,
            component_weights=self.component_weights,
        )

    def populate_state_meta(self, state: "LossState") -> "LossState":
        """
        Populate LossState.meta with all model-level data.

        Called once per macro cycle before weighting schemes are applied.
        This is the single location where refinement data is extracted into state.

        Parameters
        ----------
        state : LossState
            State to populate with meta data.

        Returns
        -------
        LossState
            State with meta populated.
        """
        with torch.no_grad():
            # R-factors
            rwork, rfree = self.get_rfactor()
            rwork = float(rwork) if isinstance(rwork, torch.Tensor) else rwork
            rfree = float(rfree) if isinstance(rfree, torch.Tensor) else rfree

            # X-ray losses
            xray_work = self.xray_target_work().detach().item()
            xray_test = self.xray_target_test().detach().item()

            # ADP statistics
            adp_values = self.model.adp()
            mean_adp = float(adp_values.mean())
            adp_std = float(adp_values.std())

            # Geometry deviations (restraints accessed via model.restraints)
            bond_rmsd = 0.0
            angle_rmsd = 0.0
            if self.model.initialized and self.model._restraints is not None:
                restraints = self.model.restraints
                if hasattr(restraints, "bond_deviations"):
                    bond_devs, _ = restraints.bond_deviations()
                    bond_rmsd = float(torch.sqrt((bond_devs**2).mean()))
                if hasattr(restraints, "angle_deviations"):
                    angle_devs, _ = restraints.angle_deviations()
                    angle_rmsd = float(torch.sqrt((angle_devs**2).mean()))

        state.update_meta(
            {
                # Device
                "device": self.device,
                # Static structure/data properties
                "n_atoms": len(self.model.pdb),
                "n_hkl": self.reflection_data.hkl.shape[0],
                "resolution_min": float(self.reflection_data.resolution.min()),
                "wilson_b": (
                    float(self.reflection_data.wilson_b)
                    if self.reflection_data.wilson_b is not None
                    else 45.0
                ),
                # Dynamic refinement state
                "rwork": rwork,
                "rfree": rfree,
                "rfree_gap": rfree - rwork,
                "xray_loss_work": xray_work,
                "xray_loss_test": xray_test,
                "mean_adp": mean_adp,
                "adp_std": adp_std,
                "bond_rmsd": bond_rmsd,
                "angle_rmsd": angle_rmsd,
            }
        )

        return state

    def update_weights(self, state: "LossState", multiply=False) -> "LossState":
        """
        Compute weights from component_weighting and update state.
        Weights are clipped to [0.01, 100.0] to avoid extreme values.

        Parameters
        ----------
        state : LossState
            State with meta populated.
        multiply : bool, optional
            If True, multiply existing weights by computed weights.
            If False, replace existing weights with computed weights.

        Returns
        -------
        LossState
            State with weights updated.
        """
        weights = self.component_weighting(state)

        for name, weight in weights.items():
            current = state.get_weight(name, default=1.0)
            if multiply:
                weight_effective = min(max(current * weight, 0.01), 100.0)
            else:
                weight_effective = min(max(weight, 0.01), 100.0)
            state.set_weight(name, weight_effective)

        return state

    def _create_loss_state(self) -> LossState:
        """
        Create a configured LossState for optimization (internal).

        Sets up a LossState with all targets registered as callables with
        hierarchical naming (e.g., 'geometry/bond', 'adp/simu').

        Returns
        -------
        LossState
            Configured LossState with targets registered.
        """
        state = LossState(device=self.device)

        # Register X-ray target
        state.register_target("xray", self.xray_target_work)

        # Register geometry targets
        state.register_targets(self.geometry_target)

        # Register ADP targets
        state.register_targets(self.adp_target)
        n_ref = int(self.reflection_data.hkl.shape[0])
        state.register_target(
            "adp/scaler_U",
            ScalerURegularizationTarget(self.scaler, n_reflections=n_ref),
        )
        state.register_target(
            "adp/scaler_log_scale",
            ScalerLogScaleTrendTarget(self.scaler, n_reflections=n_ref),
        )

        return state

    def create_loss_state(self) -> LossState:
        """
        Create a configured LossState for optimization.

        .. deprecated::
            Use the `loss_state` property instead for the persistent state.
            This method is kept for backwards compatibility.

        Sets up a LossState with all targets registered as callables with
        hierarchical naming (e.g., 'geometry/bond', 'adp/simu'). Weights are
        applied from component_weighting.

        Usage:
            from torchref.utils import validate_loss

            state = refinement.create_loss_state()
            params = list(refinement.parameters())

            # Log initial state
            state.aggregate(log_values=True)

            # In an LBFGS closure, wrap with validate_loss so non-finite
            # losses warn + reject the step instead of poisoning the run.
            def closure():
                optimizer.zero_grad()
                loss = state.aggregate()
                loss.backward()
                ok = validate_loss(
                    loss, state=state, parameters=params,
                    context="my_refinement", raise_on_fail=False,
                )
                if not ok:
                    for p in params:
                        if p.grad is not None:
                            p.grad.zero_()
                    return torch.full_like(loss.detach(), float("inf"))
                return loss

            optimizer.step(closure)

            # Log final state
            state.new_entry()
            state.aggregate(log_values=True)

        Returns
        -------
        LossState
            Configured LossState with targets and weights.
        """
        return self._create_loss_state()

    def complete_loss_state(self) -> "LossState":
        """
        Update and return the persistent LossState.

        Updates the persistent LossState with current meta, target info,
        cached losses, and weights. The state is reused across cycles.

        The cached active-parameter leaf set is *not* refreshed here. Stale
        leaves are not a correctness hazard: a leaf that's in the set but
        whose Parameter object was replaced externally (e.g. by
        ``Model.freeze``) just gets ignored by ``_freeze_graph_extras``,
        which costs a marginal amount of wasted backward work but never
        produces wrong answers. If you do call ``Model.freeze`` /
        ``Model.unfreeze`` between LossState creation and a refinement
        step, call ``state.refresh_loss_leaves()`` explicitly.

        Returns
        -------
        LossState
            Complete LossState with targets, meta, losses, and weights.
        """
        state = self.loss_state
        state = self.populate_state_meta(state)
        state.cache_losses()
        state = self.update_weights(state)
        return state

    def xray_loss(self):
        """
        Compute X-ray loss on work set.

        Returns
        -------
        torch.Tensor
            X-ray loss on work set.
        """
        return self.xray_loss_work()

    def restraints_loss(self):
        """
        Compute total geometry restraints loss.

        Returns
        -------
        torch.Tensor
            Total geometry restraints loss.
        """
        return self.geometry_loss()

    def collect_metrics(self) -> Dict[str, Any]:
        """
        Collect all metrics from component_weighting.stats().

        This is the standard method for gathering refinement metrics for logging.
        Uses the centralized component_weighting module for all statistics.
        Returns full unfiltered stats - filtering is done at display time.

        Returns
        -------
        dict
            Dictionary with all metrics (unfiltered, with StatEntry objects).
        """
        metrics = {}

        with torch.no_grad():
            # R-factors (always essential)
            rwork, rfree = self.get_rfactor()
            metrics["rwork"] = (
                rwork
                if isinstance(rwork, float)
                else rwork.item() if hasattr(rwork, "item") else float(rwork)
            )
            metrics["rfree"] = (
                rfree
                if isinstance(rfree, float)
                else rfree.item() if hasattr(rfree, "item") else float(rfree)
            )
            metrics["rfree_gap"] = metrics["rfree"] - metrics["rwork"]

            if hasattr(self, "geometry_target"):
                metrics["geometry"] = self.geometry_target.stats()
            if hasattr(self, "adp_target"):
                metrics["adp"] = self.adp_target.stats()

            # Add X-ray NLL stats for component_weighting
            if hasattr(self, "component_weighting"):
                xray_work = self.xray_loss_work()
                xray_test = self.xray_loss_test()
                metrics["component_weighting"] = {
                    "xray": {
                        "work_nll": (
                            xray_work.item()
                            if hasattr(xray_work, "item")
                            else float(xray_work)
                        ),
                        "test_nll": (
                            xray_test.item()
                            if hasattr(xray_test, "item")
                            else float(xray_test)
                        ),
                    }
                }

        return metrics

    def add_target_info_to_state(self, state: "LossState") -> "LossState":
        """
        Add target information from geometry and ADP targets to LossState.meta.

        .. deprecated::
            This method is no longer needed. Use :meth:`complete_loss_state`
            instead, which handles all state setup in one call.

        Parameters
        ----------
        state : LossState
            Current loss state. Meta will be updated with target info.
        Returns
        -------
        LossState
            Updated loss state (unchanged).
        """
        import warnings

        warnings.warn(
            "add_target_info_to_state is deprecated and is a no-op. "
            "Use complete_loss_state() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return state

    def get_rfactor(self):
        return self.scaler.rfactor()

    def update_outliers(self, z_threshold=4.0):
        with torch.no_grad():
            self.reflection_data = self.reflection_data.update_outliers(
                self.model, self.scaler, z_threshold=z_threshold
            )
            self.setup_scaler()

    def plot_fcalc_vs_fobs(self, outpath="fcalc_vs_fobs.png"):
        import matplotlib.pyplot as plt

        with torch.no_grad():
            hkl, F_obs, sigma_F_obs, self.rfree_flags = self.reflection_data()
            self.get_F_calc()
            F_calc = self.F_calc
            F_obs_amp = torch.abs(F_obs).cpu().numpy()
            F_calc_amp = torch.abs(F_calc).cpu().numpy()
            plt.figure(figsize=(8, 8))
            plt.scatter(F_obs_amp, F_calc_amp, alpha=0.5)
            plt.plot(
                [0, max(F_obs_amp)], [0, max(F_obs_amp)], color="red", linestyle="--"
            )
            plt.xlabel("Observed |F|")
            plt.ylabel("Calculated |F|")
            plt.title("F_calc vs F_obs")
            plt.grid()
            plt.savefig(outpath)

    def write_out_mtz(self, out_mtz_path="refined_output.mtz", anomalous=None):
        """Write refined map coefficients to an MTZ file.

        Parameters
        ----------
        out_mtz_path : str
            Output MTZ path.
        anomalous : bool, optional
            If True, emit a phenix-style anomalous MTZ: display maps and merged
            columns in the canonical ASU (Friedel mates merged by mean
            amplitude) plus unstacked ``F-obs(+/-)`` / ``F-model(+/-)`` columns
            on the same ASU index. If False, the legacy per-row layout is
            written. If None (default), chosen automatically from the data:
            anomalous output when the data were loaded as Bijvoet pairs
            (``reflection_data.friedel_merged`` is False), legacy otherwise.
        """
        with torch.no_grad():
            # Signed HKL so the per-row fcalc carries the anomalous (Bijvoet)
            # difference; write_mtz consumes it row-aligned with reflection_data.
            hkl = self.reflection_data.hkl_for_sf()
            fcalc = self.scaler(self.get_fcalc(hkl), use_mask=False)
            self.reflection_data.write_mtz(out_mtz_path, fcalc, anomalous=anomalous)

    def collect_deposition_metadata(self, metadata=None):
        """Collect refinement statistics into a RefinementMetadata object.

        Reuses existing statistics from ``collect_metrics()``,
        ``get_rfactor()``, and reflection data attributes.

        Parameters
        ----------
        metadata : RefinementMetadata, optional
            Existing metadata to merge with (e.g. from input file pass-through).
            Refinement statistics take precedence over pass-through values.

        Returns
        -------
        RefinementMetadata
            Metadata populated with final refinement statistics.
        """
        from torchref.io.metadata import RefinementMetadata

        refinement_meta = RefinementMetadata.from_refinement(self)

        # Merge with input file metadata if available
        if metadata is not None:
            return metadata.merge(refinement_meta)

        # Merge with pass-through headers from input file
        if hasattr(self.model, "_input_file") and self.model._input_file:
            input_file = self.model._input_file
            if input_file.endswith(".pdb"):
                input_meta = RefinementMetadata.from_pdb_file(input_file)
            elif input_file.endswith((".cif", ".mmcif")):
                input_meta = RefinementMetadata.from_cif_file(input_file)
            else:
                input_meta = None
            if input_meta is not None:
                return input_meta.merge(refinement_meta)

        return refinement_meta

    def write_out_pdb(self, out_pdb_path="refined_output.pdb", metadata=None):
        """Write refined PDB with optional metadata header.

        Parameters
        ----------
        out_pdb_path : str
            Output PDB file path.
        metadata : RefinementMetadata, optional
            Metadata for PDB header. If None, auto-collected from refinement.
        """
        if metadata is None:
            metadata = self.collect_deposition_metadata()
        self.model.write_pdb(out_pdb_path, metadata=metadata)

    def write_out_cif(self, out_cif_path="refined_output.cif", metadata=None):
        """Write refined coordinates as mmCIF with metadata.

        Parameters
        ----------
        out_cif_path : str
            Output mmCIF file path.
        metadata : RefinementMetadata, optional
            Metadata for mmCIF categories. If None, auto-collected from refinement.
        """
        if metadata is None:
            metadata = self.collect_deposition_metadata()
        self.model.write_cif(out_cif_path, metadata=metadata)

    def save_state(self, path: str):
        """
        Save the complete state of the refinement to a file.

        Parameters
        ----------
        path : str
            Path to save the state dictionary to.
        """
        torch.save(self.state_dict(), path)
        if self.verbose > 0:
            print(f"Saved refinement state to {path}")

    def load_state(self, path: str, strict: bool = True):
        """
        Load the complete state of the refinement from a file.

        Parameters
        ----------
        path : str
            Path to load the state dictionary from.
        strict : bool, optional
            Whether to strictly enforce that keys match. Default is True.
        """
        state_dict = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(state_dict, strict=strict)
        if self.verbose > 0:
            print(f"Loaded refinement state from {path}")

    @classmethod
    def create_from_state_dict(
        cls,
        state_dict: dict,
        device: torch.device = get_default_device(),
        verbose: int = 1,
    ) -> "Refinement":
        """
        Create a fully initialized Refinement from a state dictionary.

        This is the recommended way to restore a Refinement from a saved state.
        It creates the proper submodules using their respective create_from_state_dict
        methods, then calls PyTorch's default load_state_dict.

        Parameters
        ----------
        state_dict : dict
            State dictionary from torch.save(refinement.state_dict(), ...)
            or from loading a checkpoint file.
        device : torch.device, optional
            Device to place tensors on. Defaults to the configured device.current.
        verbose : int, optional
            Verbosity level. Default is 1.

        Returns
        -------
        Refinement
            Fully initialized instance with restored state.

        Examples
        --------
        Save and load refinement state::

            # Save
            torch.save(refinement.state_dict(), 'refinement.pt')

            # Load
            state = torch.load('refinement.pt')
            refinement = Refinement.create_from_state_dict(state)

            # Continue refinement
            rwork, rfree = refinement.get_rfactor()
            print(f"Restored at R-work={rwork:.4f}, R-free={rfree:.4f}")
        """

        # Helper to extract submodule state from flattened state_dict
        def extract_submodule_state(state_dict: dict, prefix: str) -> dict:
            """Extract keys starting with prefix and strip the prefix."""
            result = {}
            prefix_with_dot = prefix + "."
            for key, value in state_dict.items():
                if key.startswith(prefix_with_dot):
                    result[key[len(prefix_with_dot) :]] = value
            return result

        # Extract submodule states from flattened keys
        model_state = extract_submodule_state(state_dict, "model")
        reflection_data_state = extract_submodule_state(state_dict, "reflection_data")
        scaler_state = extract_submodule_state(state_dict, "scaler")
        restraints_state = extract_submodule_state(state_dict, "restraints")
        weighter_state = extract_submodule_state(state_dict, "weighter")

        if verbose > 0:
            print(
                f"Extracted state dict sizes: model={len(model_state)}, data={len(reflection_data_state)}, "
                f"scaler={len(scaler_state)}, restraints={len(restraints_state)}"
            )

        # Create submodules using their factory methods
        # These properly set up structure before loading values
        # ReflectionData is now a dataclass with _from_state() method
        reflection_data = ReflectionData._from_state(
            reflection_data_state, device=str(device)
        )

        model = ModelFT.create_from_state_dict(
            model_state, device=device, verbose=verbose
        )

        # Create Scaler with model and data (required for proper setup)
        scaler = Scaler(model, reflection_data, verbose=verbose, device=device)

        # Create Restraints with model (required for proper setup)
        restraints = Restraints(model, verbose=verbose)

        # Create empty instance
        instance = cls.__new__(cls)
        nnModule.__init__(instance)

        # Set basic attributes
        instance.device = device
        instance.verbose = verbose
        instance.data_file = None
        instance.pdb = None
        instance.history = {}
        instance.max_res = model_state.get("_metadata_max_res", None)
        instance.nbins = 10
        instance.lr = 1e-3
        instance.effective_weights = {}

        # Register the properly created submodules
        instance.reflection_data = reflection_data
        instance.model = model
        instance.scaler = scaler
        instance.restraints = restraints
        instance.weighter = None

        # Now load the state dict - PyTorch's default will fill in values
        # Use strict=False since we may have metadata keys and properly created submodules
        instance.load_state_dict(state_dict, strict=False)

        # Reconnect model and data to scaler after loading
        instance.scaler.set_model_and_data(instance.model, instance.reflection_data)

        # Initialize targets if model is available
        if instance.model is not None and instance.model.initialized:
            try:
                instance._init_targets()
            except Exception as e:
                if verbose > 0:
                    print(f"Note: Could not initialize targets: {e}")

        if verbose > 0:
            n_atoms = len(instance.model.pdb) if instance.model.pdb is not None else 0
            n_refl = (
                instance.reflection_data.hkl.shape[0]
                if instance.reflection_data.hkl is not None
                else 0
            )
            print(
                f"Created Refinement from state_dict: {n_atoms} atoms, {n_refl} reflections"
            )

        return instance
