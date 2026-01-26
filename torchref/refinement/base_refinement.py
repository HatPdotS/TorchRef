from typing import Any, Dict

import torch
from torch.nn import Module as nnModule

from torchref.io import ReflectionData
from torchref.model.model_ft import ModelFT
from torchref.refinement.loss_state import LossState
from torchref.refinement.targets.combined_targets import (
    TotalADPTarget,
    TotalGeometryTarget,
)

# Target system imports
from torchref.refinement.targets.targets import create_xray_target
from torchref.scaling.scaler import Scaler
from torchref.utils.debug_utils import DebugMixin


class Refinement(DebugMixin, nnModule):
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
        Computation device. Default is cpu.
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
        device: torch.device = torch.device("cpu"),
        nbins: int = 10,
        manual_weights: Dict[str, float] = None,
        component_weights: Dict[str, float] = None,
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
            Computation device. Default is cpu.
        weighter : LossWeightingModule, optional
            Loss weighting module. Creates default if None.
        nbins : int, optional
            Number of resolution bins. Default is 10.
        """
        super().__init__()
        self.device = device
        self.verbose = verbose
        self.data_file = data_file
        self.pdb = pdb
        self.history = dict()
        self.max_res = max_res
        self.nbins = nbins
        self.lr = 1e-3

        # Empty initialization - create empty submodules for state_dict loading
        if data_file is None and pdb is None:
            # Create empty submodules so state_dict keys exist
            self.reflection_data = ReflectionData(
                verbose=self.verbose, device=self.device
            )
            self.model = ModelFT(verbose=self.verbose, device=self.device)
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
                self.reflection_data = ReflectionData(verbose=self.verbose)
                if data_file.endswith(".mtz"):
                    self.reflection_data.load_mtz(data_file)
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
                verbose=self.verbose, max_res=self.max_res, device=self.device
            )
            if pdb.endswith(".cif"):
                self.model.load_cif(pdb)
            elif pdb.endswith(".pdb"):
                self.model.load_pdb(pdb)
            else:
                raise ValueError(
                    f"Unsupported model file format: {pdb}. Supported formats are .pdb and .cif"
                )

            self.scaler = Scaler(
                self.model,
                self.reflection_data,
                verbose=self.verbose,
                device=self.device,
                nbins=self.nbins,
            )
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

    def _init_targets(self, xray_mode: str = "ml"):
        """
        Initialize target functions.

        Parameters
        ----------
        xray_mode : str, optional
            X-ray target mode. Options are 'gaussian', 'ls', or 'ml'.
            Default is 'ml'.
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
            X-ray target mode. Options are 'gaussian', 'ls', or 'ml'.
        """
        self.xray_target_work = create_xray_target(
            model=self.model,
            data=self.reflection_data,
            scaler=self.scaler,
            mode=mode,
            use_work_set=True,
            verbose=self.verbose,
        )
        self.xray_target_test = create_xray_target(
            model=self.model,
            data=self.reflection_data,
            scaler=self.scaler,
            mode=mode,
            use_work_set=False,
            verbose=self.verbose,
        )
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

    def get_scales(self):
        if not hasattr(self, "scaler"):
            self.setup_scaler()
        self.scaler.initialize()
        self.reflection_data.find_outliers(self.model, self.scaler, z_threshold=5.0)
        self.scaler.refine_lbfgs()
        self.reflection_data.find_outliers(self.model, self.scaler, z_threshold=5.0)

    def setup_scaler(self):
        self.scaler = Scaler(
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
            hkl, _, _, _ = self.reflection_data()
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

        # Populate meta and update weights
        state = self.populate_state_meta(state)
        state = self.update_weights(state)

        return state.aggregate()

    def setup_component_weighting(self):
        """

        Set up component weighting.
        Loads default hyperparameters and initializes ComponentWeighting.

        """
        from torchref.refinement.weighting.component_weighting import ComponentWeighting

        self.get_scales()

        # Get initial xray loss for XrayScaleWeighting
        with torch.no_grad():
            initial_xray_loss = self.xray_target_work().detach().item()

        self.component_weighting = ComponentWeighting(
            device=self.device,
            weights=self.manual_weights,
            component_weights=self.component_weights,
            initial_xray_loss=initial_xray_loss,
        )

        # Load default hyperparameters
        from torchref import PATH_TORCHREF_DATA
        import os
        from torchref.utils.utils import json_to_state_dicts_separate

        hyperparams_path = os.path.join(
            PATH_TORCHREF_DATA, "default_hyperparameters.json"
        )
        (
            component_weighting_state,
            geometry_target_state,
            adp_target_state,
            unassigned_keys,
        ) = json_to_state_dicts_separate(hyperparams_path)

        self.component_weighting.load_state_dict(
            component_weighting_state, strict=False
        )
        self.geometry_target.load_state_dict(geometry_target_state, strict=False)
        self.adp_target.load_state_dict(adp_target_state, strict=False)

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

    def update_weights(self, state: "LossState") -> "LossState":
        """
        Compute weights from component_weighting and update state.

        Parameters
        ----------
        state : LossState
            State with meta populated.

        Returns
        -------
        LossState
            State with weights updated.
        """
        weights = self.component_weighting(state)

        for name, weight in weights.items():
            current = state.get_weight(name, default=1.0)
            state.set_weight(name, current * weight)

        return state

    def create_loss_state(self) -> LossState:
        """
        Create a configured LossState for optimization.

        Sets up a LossState with all targets registered as callables with
        hierarchical naming (e.g., 'geometry/bond', 'adp/simu'). Weights are
        applied from component_weighting.

        Usage:
            state = refinement.create_loss_state()

            # Log initial state
            state.aggregate(log_values=True)

            # In LBFGS closure:
            def closure():
                optimizer.zero_grad()
                loss = state.aggregate()
                loss.backward()
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
        state = LossState(device=self.device)

        # Register X-ray target
        state.register_target("xray", self.xray_target_work)

        state.register_targets(self.geometry_target)

        # Get weights from component_weighting and apply
        state.register_targets(self.adp_target)

        return state

    def complete_loss_state(self) -> "LossState":
        """
        Create and populate a complete LossState.

        Creates a LossState with all targets registered, meta populated,
        losses cached, and weights applied. Useful for logging or analysis
        outside of optimization.

        Returns
        -------
        LossState
            Complete LossState with targets, meta, losses, and weights.
        """
        state = self.create_loss_state()
        state = self.populate_state_meta(state)
        state = self.add_target_info_to_state(state)
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

        return metrics

    def log_refinement(self, phase: str, before: Dict[str, Any], after: Dict[str, Any]):
        """
        Log refinement comparison showing metrics before/after.

        Uses the refinement's verbosity level to determine detail level.
        Filters the full stats dict based on verbosity at display time.

        Parameters
        ----------
        phase : str
            Refinement phase name ('XYZ', 'ADP', 'scaling', etc.).
        before : dict
            Metrics dict from collect_metrics() before refinement.
        after : dict
            Metrics dict from collect_metrics() after refinement.
        """
        if self.verbose < 1:
            return

        from torchref.utils.stats import filter_stats, StatEntry

        # Filter stats based on verbosity level for display
        before_filtered = filter_stats(before, self.verbose)
        after_filtered = filter_stats(after, self.verbose)

        # Get weights from after stats
        weights = after_filtered.get("component_weighting", {}).get("weights", {})

        # Get overfitting weight from after stats (need to look in unfiltered to find it)
        overfitting_weight = 0.0
        after_cw = after.get("component_weighting", {})
        if "overfitting_weighting" in after_cw:
            ow_stats = after_cw["overfitting_weighting"]
            if isinstance(ow_stats, dict):
                ow_val = ow_stats.get("overfitting_weight")
                if isinstance(ow_val, StatEntry):
                    overfitting_weight = ow_val.value
                elif ow_val is not None:
                    overfitting_weight = ow_val

        # Get X-ray scale weight from after stats
        xray_scale_weight = 1.0
        if "xray_scale_weighting" in after_cw:
            xsw_stats = after_cw["xray_scale_weighting"]
            if isinstance(xsw_stats, dict):
                xsw_val = xsw_stats.get("xray_weight")
                if isinstance(xsw_val, StatEntry):
                    xray_scale_weight = xsw_val.value
                elif xsw_val is not None:
                    xray_scale_weight = xsw_val

        print(f"\n{'─'*80}")
        print(f"  {phase} Refinement Summary")
        print(f"  X-ray scale weight: {xray_scale_weight:.4f}", end="")
        if overfitting_weight > 0:
            print(f"  |  Overfitting penalty: {overfitting_weight:.3f}", end="")
        print()  # End the line
        print(f"{'─'*80}")

        # Header with weight column
        print(
            f"\n  {'Metric':<30} {'Before':>12} {'After':>12} {'Change':>12} {'Weight':>10}"
        )
        print(f"  {'-'*76}")

        def format_row(label, b_val, a_val, weight=None, fmt="12.4f"):
            delta = a_val - b_val
            weight_str = f"{weight:>10.4f}" if weight is not None else f"{'':>10}"
            return f"  {label:<30} {b_val:>{fmt}} {a_val:>{fmt}} {delta:>+{fmt}} {weight_str}"

        # R-factor metrics (always included)
        for key, label in [
            ("rwork", "Rwork"),
            ("rfree", "Rfree"),
            ("rfree_gap", "Rfree-Rwork gap"),
        ]:
            b_val = before_filtered.get(key, 0)
            a_val = after_filtered.get(key, 0)
            print(format_row(label, b_val, a_val))

        # X-ray NLL with weight
        print()
        before_xray = before_filtered.get("component_weighting", {}).get("xray", {})
        after_xray = after_filtered.get("component_weighting", {}).get("xray", {})
        xray_weight = weights.get("xray")
        for key, label in [
            ("work_nll", "X-ray NLL (work)"),
            ("test_nll", "X-ray NLL (test)"),
        ]:
            b_val = before_xray.get(key, 0)
            a_val = after_xray.get(key, 0)
            # Only show weight on first X-ray row
            w = xray_weight if key == "work_nll" else None
            print(format_row(label, b_val, a_val, w))

        # Geometry loss with weight (verbosity >= 1)
        if self.verbose >= 1:
            geom_weight = weights.get("geometry")
            before_geom = before_filtered.get("geometry", {})
            after_geom = after_filtered.get("geometry", {})
            # Get total geometry loss if available
            b_geom_total = 0.0
            a_geom_total = 0.0
            for target_name, target_stats in after_geom.items():
                if isinstance(target_stats, dict):
                    a_geom_total += target_stats.get("loss", 0)
                    b_stats = before_geom.get(target_name, {})
                    if isinstance(b_stats, dict):
                        b_geom_total += b_stats.get("loss", 0)
            if a_geom_total > 0 or b_geom_total > 0:
                print(
                    format_row(
                        "Geometry (total)", b_geom_total, a_geom_total, geom_weight
                    )
                )

            # ADP loss with weight
            adp_weight = weights.get("adp")
            before_adp = before_filtered.get("adp", {})
            after_adp = after_filtered.get("adp", {})
            # Get total ADP loss if available
            b_adp_total = 0.0
            a_adp_total = 0.0
            for target_name, target_stats in after_adp.items():
                if isinstance(target_stats, dict):
                    a_adp_total += target_stats.get("loss", 0)
                    b_stats = before_adp.get(target_name, {})
                    if isinstance(b_stats, dict):
                        b_adp_total += b_stats.get("loss", 0)
            if a_adp_total > 0 or b_adp_total > 0:
                print(format_row("ADP (total)", b_adp_total, a_adp_total, adp_weight))

        # Detailed component losses (verbosity >= 2)
        if self.verbose >= 2:
            # Geometry targets breakdown
            before_geom = before_filtered.get("geometry", {})
            after_geom = after_filtered.get("geometry", {})
            if after_geom:
                print("\n  Geometry breakdown:")
                for target_name, target_stats in after_geom.items():
                    if isinstance(target_stats, dict):
                        loss = target_stats.get("loss", 0)
                        b_loss = (
                            before_geom.get(target_name, {}).get("loss", 0)
                            if isinstance(before_geom.get(target_name), dict)
                            else 0
                        )
                        label = (
                            "    "
                            + target_name.replace("_target", "")
                            .replace("_", " ")
                            .title()
                        )
                        print(format_row(label, b_loss, loss))

                        # Detailed stats at verbosity >= 3
                        if self.verbose >= 3:
                            for stat_key, stat_val in target_stats.items():
                                if stat_key != "loss" and isinstance(
                                    stat_val, (int, float)
                                ):
                                    b_stat = (
                                        before_geom.get(target_name, {}).get(
                                            stat_key, 0
                                        )
                                        if isinstance(
                                            before_geom.get(target_name), dict
                                        )
                                        else 0
                                    )
                                    print(
                                        format_row(
                                            f"      {stat_key}", b_stat, stat_val
                                        )
                                    )

            # ADP targets breakdown
            before_adp = before_filtered.get("adp", {})
            after_adp = after_filtered.get("adp", {})
            if after_adp:
                print("\n  ADP breakdown:")
                for target_name, target_stats in after_adp.items():
                    if isinstance(target_stats, dict):
                        loss = target_stats.get("loss", 0)
                        b_loss = (
                            before_adp.get(target_name, {}).get("loss", 0)
                            if isinstance(before_adp.get(target_name), dict)
                            else 0
                        )
                        label = (
                            "    "
                            + target_name.replace("_target", "")
                            .replace("_", " ")
                            .title()
                        )
                        print(format_row(label, b_loss, loss))

                        # Detailed stats at verbosity >= 3
                        if self.verbose >= 3:
                            for stat_key, stat_val in target_stats.items():
                                if stat_key != "loss" and isinstance(
                                    stat_val, (int, float)
                                ):
                                    b_stat = (
                                        before_adp.get(target_name, {}).get(stat_key, 0)
                                        if isinstance(before_adp.get(target_name), dict)
                                        else 0
                                    )
                                    print(
                                        format_row(
                                            f"      {stat_key}", b_stat, stat_val
                                        )
                                    )

        print(f"{'─'*80}\n")

    def log_xyz_comparison(
        self, before: Dict[str, float], after: Dict[str, float], weight: float = None
    ):
        """
        Log XYZ refinement comparison.

        Deprecated: Use log_refinement() instead.

        Parameters
        ----------
        before : dict
            Metrics dict from collect_metrics() before XYZ refinement.
        after : dict
            Metrics dict from collect_metrics() after XYZ refinement.
        weight : float, optional
            Restraint weight (ignored, for backwards compatibility).
        """
        self.log_refinement("XYZ", before, after)

    def log_adp_comparison(
        self, before: Dict[str, float], after: Dict[str, float], weight: float = None
    ):
        """
        Log ADP refinement comparison.

        Deprecated: Use log_refinement() instead.

        Parameters
        ----------
        before : dict
            Metrics dict from collect_metrics() before ADP refinement.
        after : dict
            Metrics dict from collect_metrics() after ADP refinement.
        weight : float, optional
            ADP weight (ignored, for backwards compatibility).
        """
        self.log_refinement("ADP", before, after)

    def add_target_info_to_state(self, state: "LossState") -> "LossState":
        """
        Add target information from geometry and ADP targets to LossState.meta.
        Parameters
        ----------
        state : LossState
            Current loss state. Meta will be updated with target info.
        Returns
        -------
        LossState
            Updated loss state with target info in meta.
        """

        target_info = {}

        for loss_object in self.geometry_target.values():
            target_val, sigma = loss_object._target_value, loss_object._sigma
            target_info[loss_object.name] = {
                "target_value": target_val,
                "sigma": sigma,
            }

        for loss_object in self.adp_target.values():
            target_val, sigma = loss_object._target_value, loss_object._sigma
            target_info[loss_object.name] = {
                "target_value": target_val,
                "sigma": sigma,
            }

        state.update_meta({"target_info": target_info})

        return state

    def cuda(self):
        super().cuda()
        self.model.cuda()  # Explicitly call cuda on model (restraints moved via model)
        self.reflection_data.cuda()
        if hasattr(self, "scaler") and self.scaler is not None:
            self.scaler.cuda()
        # Note: restraints are now managed by model and moved via model.cuda()
        if (
            hasattr(self, "component_weighting")
            and self.component_weighting is not None
        ):
            self.component_weighting.cuda()
            self.component_weighting.device = torch.device("cuda")
        self.device = torch.device("cuda")
        return self

    def cpu(self):
        super().cpu()
        self.model.cpu()  # Explicitly call cpu on model (restraints moved via model)
        self.reflection_data.cpu()
        if hasattr(self, "scaler") and self.scaler is not None:
            self.scaler.cpu()
        # Note: restraints are now managed by model and moved via model.cpu()
        if (
            hasattr(self, "component_weighting")
            and self.component_weighting is not None
        ):
            self.component_weighting.cpu()
            self.component_weighting.device = torch.device("cpu")
        self.device = torch.device("cpu")
        return self

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

    def write_out_mtz(self, out_mtz_path="refined_output.mtz"):
        with torch.no_grad():
            hkl, _, _, _ = self.reflection_data(mask=False)
            fcalc = self.scaler(self.get_fcalc(hkl), use_mask=False)
            self.reflection_data.write_mtz(out_mtz_path, fcalc)

    def write_out_pdb(self, out_pdb_path="refined_output.pdb"):
        self.model.write_pdb(out_pdb_path)

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
        device: torch.device = torch.device("cpu"),
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
            Device to place tensors on. Default is cpu.
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
