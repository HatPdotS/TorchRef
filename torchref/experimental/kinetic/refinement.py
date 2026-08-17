"""
Kinetic refinement orchestrator.

Uses LossState + ModelCollection + DatasetCollection to refine multiple
structural models against multiple time-resolved datasets with kinetics-
constrained occupancy fractions.

Example
-------
::

    from torchref import ModelFT, ReflectionData, DatasetCollection
    from torchref.experimental.kinetic import ModelCollection, KineticRefinement

    # Base models
    model_dark = ModelFT(max_res=1.5).load_pdb("dark.pdb")
    model_light = ModelFT(max_res=1.5).load_pdb("light.pdb")

    # Collections
    models = ModelCollection([model_dark, model_light])
    models.add_dark()
    models.add_timepoint("1ps", [0.9, 0.1])

    datasets = DatasetCollection()
    datasets.add_dataset("dark", ReflectionData().load_mtz("dark.mtz"))
    datasets.add_dataset("1ps", ReflectionData().load_mtz("1ps.mtz"))

    ref = KineticRefinement(datasets, models)
    ref.setup()
    ref.refine(macro_cycles=5)
"""

from typing import TYPE_CHECKING, Dict, List, Optional

import torch
from torch import nn

from torchref.refinement.loss_state import LossState, create_loss_state
from torchref.experimental.kinetic.targets import (
    CollectionDifferenceTarget,
    CollectionRiceTarget,
    MultiModelGeometryTarget,
    MultiModelADPTarget,
    KineticPriorTarget,
)
from torchref.utils.device_mixin import DeviceMixin

if TYPE_CHECKING:
    from torchref.io.datasets.collection import DatasetCollection
    from torchref.model.model_collection import ModelCollection
    from torchref.scaling import Scaler


class KineticRefinement(DeviceMixin, nn.Module):
    """
    Orchestrator for kinetic refinement of time-resolved data.

    Manages LossState registration, scaler initialization, and
    optimization loops for alternating structure / fraction refinement.

    Parameters
    ----------
    dataset_collection : DatasetCollection
        Collection of reflection datasets keyed by timepoint name.
    model_collection : ModelCollection
        Collection of mixed models keyed by timepoint name.
    xray_weight_difference : float
        Weight for the difference X-ray target.
    xray_weight_rice : float
        Weight for the Rice amplitude target.
    geometry_weight : float
        Weight for geometry restraints.
    adp_weight : float
        Weight for ADP restraints.
    kinetic_prior_weight : float
        Weight for kinetic prior regularization (0 to disable).
    device : torch.device, optional
        Computation device.  If None, inferred from model collection.
    verbose : int
        Verbosity level.
    """

    def __init__(
        self,
        dataset_collection: "DatasetCollection",
        model_collection: "ModelCollection",
        xray_weight_difference: float = 2.0,
        xray_weight_rice: float = 1.0,
        geometry_weight: float = 10.0,
        adp_weight: float = 3.0,
        kinetic_prior_weight: float = 0.0,
        device: Optional[torch.device] = None,
        verbose: int = 1,
    ):
        super().__init__()

        self.dataset_collection = dataset_collection
        self.model_collection = model_collection
        self.verbose = verbose

        if device is None:
            device = model_collection.device
        self._device = device

        # Default weights
        self._weights = {
            "xray/difference": xray_weight_difference,
            "xray/rice": xray_weight_rice,
            "geometry": geometry_weight,
            "adp": adp_weight,
            "kinetic_prior": kinetic_prior_weight,
        }

        # Placeholders (populated by setup())
        self.scaler: Optional["Scaler"] = None
        self.loss_state: Optional[LossState] = None
        self.kinetic_prior_target: Optional[KineticPriorTarget] = None
        self._diff_target: Optional[CollectionDifferenceTarget] = None
        self._rice_target: Optional[CollectionRiceTarget] = None
        self._kinetic_model = None
        self._timepoints_map: Optional[Dict[str, int]] = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(
        self,
        cif_paths: Optional[List[str]] = None,
        kinetic_model=None,
        timepoints_map: Optional[Dict[str, int]] = None,
    ):
        """
        One-shot initialization: scalers, restraints, targets, LossState.

        Parameters
        ----------
        cif_paths : List[str], optional
            Paths to CIF restraint dictionaries for ligands.
        kinetic_model : occupancies_kinetics, optional
            Kinetic model for prior regularization.  If None, the kinetic
            prior target is not created.
        timepoints_map : Dict[str, int], optional
            Maps timepoint names to indices into the kinetic model's time
            axis.  Required when *kinetic_model* is provided.
        """
        dc = self.dataset_collection
        mc = self.model_collection
        device = self._device

        # ---- CIF restraints on base models ----
        if cif_paths:
            for model in mc.base_models:
                if hasattr(model, "set_restraints_cif"):
                    model.set_restraints_cif(cif_paths)

        # ---- Scalers ----
        self._setup_scalers()

        # ---- Targets ----
        diff_target = CollectionDifferenceTarget(
            dc, mc,
            scaler=self.scaler,
            verbose=self.verbose,
        )
        rice_target = CollectionRiceTarget(
            dc, mc,
            scaler=self.scaler,
            verbose=self.verbose,
        )
        # Store direct references for refine_kinetics()
        self._diff_target = diff_target
        self._rice_target = rice_target
        geom_target = MultiModelGeometryTarget(mc, verbose=self.verbose)
        adp_target = MultiModelADPTarget(mc, verbose=self.verbose)

        # ---- LossState ----
        self.loss_state = create_loss_state(device=device)

        self.loss_state.register_target("xray/difference", diff_target)
        self.loss_state.register_target("xray/rice", rice_target)
        self.loss_state.register_target("geometry", geom_target)
        self.loss_state.register_target("adp", adp_target)

        # ---- Kinetic prior (optional) ----
        if kinetic_model is not None and timepoints_map is not None:
            self._kinetic_model = kinetic_model
            self._timepoints_map = timepoints_map
            self.kinetic_prior_target = KineticPriorTarget(
                mc, kinetic_model, timepoints_map, verbose=self.verbose
            )
            self.loss_state.register_target(
                "kinetic_prior", self.kinetic_prior_target
            )

        # ---- Set weights ----
        self.loss_state.set_weights(self._weights)

        if self.verbose > 0:
            print("KineticRefinement setup complete.")
            print(f"  Targets: {list(self.loss_state.targets.keys())}")
            print(f"  Weights: {self._weights}")

    def _setup_scalers(self):
        """Initialize a joint scaler for all datasets.

        Uses ``CollectionScaler`` which shares scale parameters (log_scale,
        U, k_sol, B_sol) across **all** data–model pairs and creates
        per-component solvent models for each base structure.  The solvent
        contribution for a mixed model is the fraction-weighted sum of
        individual component solvent SFs.
        """
        from torchref.scaling.collection_scaler import CollectionScaler

        self.scaler = CollectionScaler(
            dataset_collection=self.dataset_collection,
            model_collection=self.model_collection,
            device=self._device,
            verbose=max(0, self.verbose - 1),
        )
        self.scaler.initialize()
        self.scaler.refine_lbfgs_joint()

        if self.verbose > 0:
            print("  Joint scaler initialized and refined (all datasets + component solvents).")

    # ------------------------------------------------------------------
    # Weight management
    # ------------------------------------------------------------------

    def set_weights(self, **kwargs):
        """
        Set target weights by name.

        Parameters
        ----------
        **kwargs
            Keyword arguments mapping target paths to weights.
            E.g. ``set_weights(geometry=5.0, adp=2.0)``.
        """
        # Map short names to full paths
        mapping = {
            "difference": "xray/difference",
            "rice": "xray/rice",
            "geometry": "geometry",
            "adp": "adp",
            "kinetic_prior": "kinetic_prior",
        }
        for key, val in kwargs.items():
            full_key = mapping.get(key, key)
            self._weights[full_key] = val
            if self.loss_state is not None:
                self.loss_state.set_weight(full_key, val)

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def get_loss(self, log_values: bool = False) -> torch.Tensor:
        """Evaluate all targets and return weighted total loss."""
        return self.loss_state.aggregate(log_values=log_values)

    def print_loss_summary(self):
        """Print breakdown of current losses."""
        self.loss_state.aggregate(log_values=True)
        self.loss_state.summary()

    def report_rfactors(self) -> Dict[str, dict]:
        """Report per-target work/free R-factors across the collection.

        Routes through each collection X-ray target's shared ``get_rfactor``
        (``rfactor_work_free`` per dataset — validity masked, validation excluded
        from both work and free), so the reported numbers use exactly the same
        reflection partition the loss does. R-factors form a distribution over the
        datasets; the median is the headline and the 10-90% spread is shown.

        Returns
        -------
        dict
            ``{target_name: get_rfactor() result}`` for each X-ray target.
        """
        from torchref.refinement.targets.collection import CollectionXrayTarget

        out: Dict[str, dict] = {}
        with torch.no_grad():
            for name, target in self.loss_state.targets.items():
                if not isinstance(target, CollectionXrayTarget):
                    continue
                rf = target.get_rfactor()
                out[name] = rf
                pw, pf = rf["rwork_pct"], rf["rfree_pct"]
                if not pw or self.verbose <= 0:
                    continue
                print(f"  [{name}] R-factors (median [10-90%]):")
                print(
                    f"    R_work = {pw['p50']:.4f} "
                    f"[{pw['p10']:.4f}-{pw['p90']:.4f}]"
                )
                print(
                    f"    R_free = {pf['p50']:.4f} "
                    f"[{pf['p10']:.4f}-{pf['p90']:.4f}]"
                )
                if self.verbose > 1:
                    for k, (rw, rfr) in rf["per_dataset"].items():
                        print(f"      {k}: R_work={rw:.4f} R_free={rfr:.4f}")
        return out

    # ------------------------------------------------------------------
    # Optimization loops
    # ------------------------------------------------------------------

    def _collect_parameters(self, structures=True, fractions=True):
        """Collect parameters for optimization."""
        params = []
        mc = self.model_collection

        if structures:
            for model in mc.base_models:
                params.extend(
                    p for p in model.parameters() if p.requires_grad
                )

        if fractions:
            for name in mc.timepoint_names:
                p = mc[name].fraction_params
                if p.requires_grad:
                    params.append(p)

        # Scaler parameters (if not frozen)
        if self.scaler is not None:
            params.extend(
                p for p in self.scaler.parameters() if p.requires_grad
            )

        return params

    def _lbfgs_loop(self, params, niter=10, max_iter=50, lr=1.0):
        """Run L-BFGS optimization loop via :meth:`LossState.run`.

        ``LossState.run`` handles the closure, NaN validation, and
        automatically disables ``requires_grad`` on loss-relevant leaves
        outside ``params`` for the duration of the run.
        """
        if not params:
            return

        optimizer = torch.optim.LBFGS(
            params, lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe"
        )
        self.loss_state.run(
            optimizer,
            nsteps=niter,
            log=False,
            context="kinetic.refinement._lbfgs_loop",
        )

        if self.verbose > 0:
            final_loss = self.get_loss(log_values=True)
            print(f"  Optimization complete: loss = {final_loss.item():.6f}")

    def refine(self, macro_cycles: int = 5, niter: int = 10, max_iter: int = 50):
        """
        Full refinement: structures + fractions jointly.

        Parameters
        ----------
        macro_cycles : int
            Number of macro-cycles.
        niter : int
            LBFGS outer iterations per macro-cycle.
        max_iter : int
            LBFGS inner iterations per step.
        """
        for cycle in range(macro_cycles):
            if self.verbose > 0:
                print(f"\n--- Macro-cycle {cycle+1}/{macro_cycles} ---")

            params = self._collect_parameters(structures=True, fractions=True)
            self._lbfgs_loop(params, niter=niter, max_iter=max_iter)

            # Update solvent masks after structure changes
            self.scaler.update_solvent()

            # Report R-factors through the shared source of truth.
            if self.verbose > 0:
                self.report_rfactors()

    def refine_structures(self, niter: int = 10, max_iter: int = 50):
        """
        Refine base model structures only (xyz, adp).

        Freezes fractions during optimization.
        """
        mc = self.model_collection
        mc.freeze_all_fractions()

        if self.verbose > 0:
            print("\n--- Refining structures (fractions frozen) ---")

        params = self._collect_parameters(structures=True, fractions=False)
        self._lbfgs_loop(params, niter=niter, max_iter=max_iter)

        mc.unfreeze_all_fractions()

    def refine_fractions(self, niter: int = 10, max_iter: int = 50):
        """
        Refine per-timepoint fractions only.

        Freezes structures during optimization.
        """
        mc = self.model_collection
        mc.freeze_structures()

        if self.verbose > 0:
            print("\n--- Refining fractions (structures frozen) ---")

        params = self._collect_parameters(structures=False, fractions=True)
        self._lbfgs_loop(params, niter=niter, max_iter=max_iter)

        mc.unfreeze_structures()

    def refine_alternating(
        self,
        n_cycles: int = 5,
        niter_structures: int = 10,
        niter_fractions: int = 5,
        refit_prior_every: int = 2,
        max_iter: int = 50,
    ):
        """
        Alternating refinement: structures → fractions → refit prior.

        Parameters
        ----------
        n_cycles : int
            Number of alternating cycles.
        niter_structures : int
            LBFGS iterations for structure refinement.
        niter_fractions : int
            LBFGS iterations for fraction refinement.
        refit_prior_every : int
            Refit kinetic prior every N cycles (0 to disable).
        max_iter : int
            LBFGS inner iterations per step.
        """
        for cycle in range(n_cycles):
            if self.verbose > 0:
                print(f"\n{'='*50}")
                print(f"Alternating cycle {cycle+1}/{n_cycles}")
                print(f"{'='*50}")

            # 1. Refine structures
            self.refine_structures(niter=niter_structures, max_iter=max_iter)

            # Update solvent masks after structure changes
            self.scaler.update_solvent()

            # 2. Refine fractions
            self.refine_fractions(niter=niter_fractions, max_iter=max_iter)

            # 3. Refit kinetic prior (if applicable)
            if (
                refit_prior_every > 0
                and self.kinetic_prior_target is not None
                and (cycle + 1) % refit_prior_every == 0
            ):
                if self.verbose > 0:
                    print("\n--- Refitting kinetic prior ---")
                self.refit_kinetic_prior()

            # Print fractions
            if self.verbose > 0:
                fracs = self.model_collection.get_all_fractions()
                for name, f in fracs.items():
                    frac_str = ", ".join(f"{v:.3f}" for v in f.detach().tolist())
                    print(f"  {name}: [{frac_str}]")

                # Report R-factors through the shared source of truth.
                self.report_rfactors()

    def refit_kinetic_prior(self, niter: int = 50, lr: float = 1e-2):
        """
        Refit the kinetic model to match current free fractions.

        This is the M-step in the EM-style alternation.
        """
        if self.kinetic_prior_target is not None:
            self.kinetic_prior_target.refit_prior(niter=niter, lr=lr)

    def refine_kinetics(self, niter: int = 200, lr: float = 1e-2):
        """
        Optimize kinetic model parameters directly against the X-ray targets.

        Component structure factors (base ModelFT) are frozen.  The kinetic
        model predictions are injected as fraction overrides into each
        timepoint's mixed model, so the gradient path is::

            kinetic params → occupancies → fractions → F_calc → X-ray loss

        After optimization, the free fraction parameters are updated to
        match the final kinetic predictions.

        Parameters
        ----------
        niter : int
            Number of Adam optimizer steps.
        lr : float
            Learning rate.
        """
        if self._kinetic_model is None:
            raise RuntimeError("No kinetic model configured. Call setup() with kinetic_model first.")

        kinetic_model = self._kinetic_model
        timepoints_map = self._timepoints_map
        mc = self.model_collection

        # Freeze component structure factors
        mc.freeze_structures()

        # Collect only kinetic model parameters
        params = [p for p in kinetic_model.parameters() if p.requires_grad]
        if not params:
            if self.verbose > 0:
                print("  No refinable kinetic parameters found.")
            mc.unfreeze_structures()
            return

        optimizer = torch.optim.Adam(params, lr=lr)

        w_diff = self._weights.get("xray/difference", 1.0)
        w_rice = self._weights.get("xray/rice", 1.0)

        # Collect all model keys that have kinetic indices (including dark)
        all_overrides = {}
        for tp_name in mc.keys():
            if tp_name in timepoints_map:
                all_overrides[tp_name] = timepoints_map[tp_name]

        for step in range(niter):
            optimizer.zero_grad()

            # Kinetic model predictions: [n_states, n_timepoints]
            kinetic_occ = kinetic_model()

            # Set fraction overrides on ALL models (including dark)
            for tp_name, t_idx in all_overrides.items():
                mc[tp_name].set_fraction_override(kinetic_occ[:, t_idx])

            # Compute X-ray loss only (difference + Rice)
            loss = w_diff * self._diff_target() + w_rice * self._rice_target()

            loss.backward()

            # Clear overrides after backward pass
            for tp_name in all_overrides:
                mc[tp_name].clear_fraction_override()

            optimizer.step()

            if self.verbose > 1 and (step + 1) % 10 == 0:
                print(f"    Kinetic opt step {step+1}/{niter}: loss = {loss.item():.6f}")

        # Update free fraction parameters to match final kinetic predictions
        with torch.no_grad():
            kinetic_occ = kinetic_model()
            for tp_name, t_idx in all_overrides.items():
                if tp_name == mc.dark_key:
                    continue  # dark fractions stay frozen at [1,0,...,0]
                predicted = kinetic_occ[:, t_idx]
                mc[tp_name].fraction_params.data = torch.log(
                    predicted.clamp(min=1e-6)
                )

        mc.unfreeze_structures()

        if self.verbose > 0:
            print(f"  Kinetic refinement: {niter} steps, final X-ray loss = {loss.item():.6f}")

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def write_pdbs(self, outdir: str):
        """Write base model PDBs to *outdir*."""
        self.model_collection.write_pdbs(outdir)

    def __repr__(self):
        n_tp = len(self.model_collection)
        n_ds = self.dataset_collection.n_datasets
        n_base = self.model_collection.n_base_models
        return (
            f"KineticRefinement({n_base} base models, "
            f"{n_tp} timepoints, {n_ds} datasets)"
        )
