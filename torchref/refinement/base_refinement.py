from typing import Any, Dict

from torchref.io.Data import ReflectionData
from torchref.model.model_ft import ModelFT
from torch.nn import Module as nnModule
import torch
from torchref.restraints.restraints import Restraints
from torchref.scaling.scaler import Scaler
from torchref.utils.debug_utils import DebugMixin

# Target system imports
from torchref.refinement.targets import (
    Target,create_xray_target
)
from torchref.refinement.combined_targets import (
    TotalGeometryTarget,
    TotalADPTarget,)

class Refinement(DebugMixin, nnModule):
    """
    Refinement class to handle the overall crystallographic refinement process.

    Supports two initialization patterns:

    1. Empty initialization (for state_dict loading):
        >>> refinement = Refinement()  # Creates empty shell with submodules
        >>> refinement.load_state_dict(torch.load('refinement.pt'))

    2. Full initialization with file paths:
        >>> refinement = Refinement(data_file='data.mtz', pdb='model.pdb')

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
        Structure factor model.
    scaler : Scaler
        Scale factor calculator.
    restraints : Restraints
        Geometry restraints.
    weighter : LossWeightingModule
        Loss weighting module.
    """

    def __init__(self, data_file: str = None, pdb: str = None, cif=None, verbose: int = 1, 
                 max_res: float = None, device: torch.device = torch.device('cpu'), 
                 weighter: 'LossWeightingModule' = None, nbins: int = 10):
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
            self.reflection_data = ReflectionData(verbose=self.verbose, device=self.device)
            self.model = ModelFT(verbose=self.verbose, device=self.device)
            self.scaler = Scaler(verbose=self.verbose, device=self.device, nbins=self.nbins)
            self.restraints = None  # Restraints needs model, created during load_state_dict
            self.weighter = None
            self.effective_weights = {}
            return

        # Full initialization with file paths
        try:
            self.to(self.device)
            self.reflection_data = ReflectionData(verbose=self.verbose)
            if data_file.endswith('.mtz'):
                self.reflection_data.load_mtz(data_file)
            elif data_file.endswith('.cif'):
                self.reflection_data.load_cif(data_file)
            else:
                raise ValueError(f"Unsupported data file format: {data_file}. Supported formats are .mtz and .cif")
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
            self.model = ModelFT(verbose=self.verbose, max_res=self.max_res, device=self.device)
            
            if pdb.endswith('.cif'):
                self.model.load_cif(pdb)
            elif pdb.endswith('.pdb'):
                self.model.load_pdb(pdb)
            else:
                raise ValueError(f"Unsupported model file format: {pdb}. Supported formats are .pdb and .cif")
            
            self.scaler = Scaler(self.model, self.reflection_data, verbose=self.verbose, device=self.device, nbins=self.nbins)
            self.restraints = Restraints(self.model, cif, self.verbose)
            
            # Loss weighting module
            if weighter is None:
                # Create default weighter
                from torchref.refinement.loss_weighting import ResolutionDependentWeighting
                self.weighter = ResolutionDependentWeighting()
                if self.verbose > 0:
                    print("No weighter provided, using default ResolutionDependentWeighting")
            else:
                self.weighter = weighter
                if self.verbose > 0:
                    print(f"Using loss weighting module: {type(self.weighter).__name__}")
            
            # Initialize target functions (instantiated once, evaluated each iteration)
            self._init_targets()
            
            # Initialize weights
            self.update_effective_weights(phase='all', cycle=0)
        except Exception as e:
            if self.verbose > 1:
                self.debug_on_error(e)
            raise e
    
    def _init_targets(self, xray_mode: str = 'ml'):
        """
        Initialize target functions.

        Parameters
        ----------
        xray_mode : str, optional
            X-ray target mode. Options are 'gaussian', 'ls', or 'ml'.
            Default is 'ml'.
        """
        # X-ray targets
        self.xray_target_work = create_xray_target(self, xray_mode, use_work_set=True, verbose=self.verbose)
        self.xray_target_test = create_xray_target(self, xray_mode, use_work_set=False, verbose=self.verbose)
        
        # Total geometry target (handles bond, angle, torsion internally)
        self.geometry_target = TotalGeometryTarget(self, verbose=self.verbose)
        
        self.adp_target = TotalADPTarget(
            self,
            verbose=self.verbose
        )
        
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
        self.xray_target_work = create_xray_target(self, mode, use_work_set=True, verbose=self.verbose)
        self.xray_target_test = create_xray_target(self, mode, use_work_set=False, verbose=self.verbose)
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

    def setup_grad_weighting(self):
        """
        Setup gradient-informed weighting module.

        Creates a GradientInformedWeighting instance and assigns it to
        the weighter attribute.
        """
        from torchref.refinement.loss_weighting import GradientInformedWeighting
        self.get_scales()
        self.weighter = GradientInformedWeighting(self,target_weights=self.effective_weights)
        if self.verbose > 0:
            print("Using GradientInformedWeighting for loss weighting")

    def set_up_smart_weighting(
        self,
        base_restraint: float = 8.0,  # Good default - conservative
        base_adp: float = 3.0,  # Increased - ADP can overfit significantly
        nll_modulation_strength: float = 0.5,
        recompute_gradnorms_every: int = 5,
        target_rfree_gap: float = 0.04
    ):
        """
        Set up smart hybrid weighting combining gradient norm balancing with ML modulation.

        This is the RECOMMENDED weighting strategy as it provides:

        - Balanced gradients across loss terms (via gradient norms)
        - Physical correctness (via ML target values)
        - Overfitting prevention (via Rfree gap monitoring)
        - Stable refinement that converges without oscillation

        The weighting strategy:

        1. Computes gradient norms to balance optimization landscape
        2. Modulates weights based on ML target values to respect physical meaning
        3. Monitors Rfree gap and increases BOTH restraints AND ADP weights if overfitting
        4. Uses smooth transitions to avoid weight oscillations

        Parameters
        ----------
        base_restraint : float, optional
            Base restraint weight. Default is 8.0 for stability.
        base_adp : float, optional
            Base ADP weight. Default is 3.0 (higher because ADP can overfit
            significantly).
        nll_modulation_strength : float, optional
            How strongly ML target values modulate weights (0-1).
            0 = pure gradient norm balancing, 1 = strong ML target influence.
            Default is 0.5.
        recompute_gradnorms_every : int, optional
            Recompute gradient norms every N cycles. Default is 5.
        target_rfree_gap : float, optional
            Target Rfree-Rwork gap. Default is 0.04 (~4%).

        Examples
        --------
        >>> refinement = Refinement(data_file="data.mtz", pdb="model.pdb")
        >>> refinement.set_up_smart_weighting(
        ...     base_restraint=1.5,
        ...     base_adp=1.5,
        ...     nll_modulation_strength=0.3
        ... )
        >>> refinement.refine(n_cycles=10)
        """
        from torchref.refinement.loss_weighting import create_hybrid_gradnorm_ML_weighting
        
        # Initialize scaler first (needed for gradient computation)
        self.get_scales()
        
        # Create hybrid weighting
        self.weighter = create_hybrid_gradnorm_ML_weighting(
            refinement=self,
            base_restraint=base_restraint,
            base_adp=base_adp,
            nll_modulation_strength=nll_modulation_strength,
            recompute_gradnorms_every=recompute_gradnorms_every,
            target_rfree_gap=target_rfree_gap,
            verbose=self.verbose
        )
        
        if self.verbose > 0:
            print(f"Using HybridGradNormMLWeighting for smart loss weighting")
            print(f"  Base weights: restraints={base_restraint}, adp={base_adp}")
            print(f"  ML target modulation strength: {nll_modulation_strength}")
            print(f"  Target Rfree gap: {target_rfree_gap}")
            print(f"  Recompute gradient norms every {recompute_gradnorms_every} cycles")

    def get_scales(self):
        if not hasattr(self, 'scaler'):
            self.setup_scaler()
        self.scaler.initialize()
        self.reflection_data.find_outliers(self.model, self.scaler, z_threshold=5.0)
        self.scaler.refine_lbfgs()
        self.reflection_data.find_outliers(self.model, self.scaler, z_threshold=5.0)

    def setup_scaler(self):
        self.scaler = Scaler(self.model, self.reflection_data, nbins=self.nbins, verbose=self.verbose, device=self.device)

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
    
    def get_fcalc(self,hkl=None, recalc=False):
        if hkl is None:
            hkl, _,_, _ = self.reflection_data()
        return self.model(hkl,recalc=recalc)

    def get_fcalc_scaled(self,hkl=None, recalc=False):
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

    def get_Fcalc(self,hkl=None, recalc=False):
        return torch.abs(self.get_fcalc(hkl, recalc=recalc))

    def get_F_calc_scaled(self,hkl=None, recalc=False):
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
        return self.geometry_target.target_losses()['bond_target']
    
    def angle_loss(self) -> torch.Tensor:
        """
        Compute angle NLL via geometry_target.

        Returns
        -------
        torch.Tensor
            Angle NLL loss.
        """
        return self.geometry_target.target_losses()['angle_target']
    
    def torsion_loss(self) -> torch.Tensor:
        """
        Compute torsion angle NLL via geometry_target.

        Returns
        -------
        torch.Tensor
            Torsion angle NLL loss.
        """
        return self.geometry_target.target_losses()['torsion_target']
    
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
        Compute total loss using instantiated targets.

        Returns
        -------
        tuple of torch.Tensor
            Tuple of (total_loss, xray_work, restraints, xray_test).
        """
        xray_work = self.xray_loss_work()
        xray_test = self.xray_loss_test()
        restraints = self.geometry_loss()
        total_loss = self.effective_weights['xray'] * xray_work + self.effective_weights['restraints'] * restraints
        return total_loss, xray_work, restraints, xray_test

    def setup_component_weighting(self):
        from torchref.refinement.component_weighting import ComponentWeighting
        self.component_weighting = ComponentWeighting(self)

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
    
    def collect_metrics(self) -> Dict[str, float]:
        """
        Collect all metrics from targets into a single flat dictionary.

        This is the standard method for gathering refinement metrics for logging.
        Merges metrics from:

        - R-factors (rwork, rfree, gap)
        - X-ray targets (nll_work, nll_test)
        - Geometry targets (via get_metrics())
        - ADP targets (via get_metrics())
        - Current weights

        Returns
        -------
        dict
            Dictionary with all metrics as Python floats (not tensors).
        """
        metrics = {}
        
        with torch.no_grad():
            # R-factors
            rwork, rfree = self.get_rfactor()
            metrics['rwork'] = rwork if isinstance(rwork, float) else rwork.item() if hasattr(rwork, 'item') else float(rwork)
            metrics['rfree'] = rfree if isinstance(rfree, float) else rfree.item() if hasattr(rfree, 'item') else float(rfree)
            metrics['rfree_gap'] = metrics['rfree'] - metrics['rwork']
            
            # X-ray NLL
            nll_work = self.xray_loss_work()
            nll_test = self.xray_loss_test()
            metrics['nll_xray_work'] = nll_work.item() if torch.is_tensor(nll_work) else nll_work
            metrics['nll_xray_test'] = nll_test.item() if torch.is_tensor(nll_test) else nll_test
            
            # Geometry metrics (if target exists)
            if hasattr(self, 'geometry_target') and self.geometry_target is not None:
                geom_metrics = self.geometry_target.get_metrics()
                metrics.update(geom_metrics)
            
            # ADP metrics (if target exists)
            if hasattr(self, 'adp_target') and self.adp_target is not None:
                adp_metrics = self.adp_target.get_metrics()
                metrics.update(adp_metrics)
            
            # Current weights
            if hasattr(self, 'effective_weights') and self.effective_weights:
                for key, val in self.effective_weights.items():
                    weight_val = val.item() if torch.is_tensor(val) else val
                    metrics[f'weight_{key}'] = weight_val
        
        return metrics
    
    def log_xyz_comparison(self, before: Dict[str, float], after: Dict[str, float], 
                           weight: float = None):
        """
        Log XYZ refinement comparison showing geometry metrics before/after.

        Parameters
        ----------
        before : dict
            Metrics dict from collect_metrics() before XYZ refinement.
        after : dict
            Metrics dict from collect_metrics() after XYZ refinement.
        weight : float, optional
            Restraint weight used.
        """
        print(f"\n{'─'*70}")
        print(f"  XYZ Refinement Summary")
        if weight is not None:
            print(f"  Restraint weight: {weight:.3f}")
        print(f"{'─'*70}")
        
        # R-factors
        print(f"\n  {'Metric':<25} {'Before':>12} {'After':>12} {'Change':>12}")
        print(f"  {'-'*61}")
        
        def format_row(label, b_val, a_val, fmt):
            delta = a_val - b_val
            # Use +/- sign format for change column
            return f"  {label:<25} {b_val:>{fmt}} {a_val:>{fmt}} {delta:>+{fmt}}"
        
        # R-factor metrics
        for key, label, fmt in [
            ('rwork', 'Rwork', '12.4f'),
            ('rfree', 'Rfree', '12.4f'),
            ('rfree_gap', 'Rfree-Rwork gap', '12.4f'),
        ]:
            b_val = before.get(key, 0)
            a_val = after.get(key, 0)
            print(format_row(label, b_val, a_val, fmt))
        
        print()
        
        # X-ray loss
        for key, label, fmt in [
            ('nll_xray_work', 'X-ray NLL (work)', '12.4f'),
            ('nll_xray_test', 'X-ray NLL (test)', '12.4f'),
        ]:
            b_val = before.get(key, 0)
            a_val = after.get(key, 0)
            print(format_row(label, b_val, a_val, fmt))
        
        print()
        
        # Geometry metrics - dynamically extract from before/after dicts
        geom_keys = [k for k in set(before.keys()) | set(after.keys()) if k.startswith('geom_')]
        # Sort keys for consistent output: total_loss first, then by target name
        geom_keys_sorted = sorted(geom_keys, key=lambda k: (
            0 if 'total_loss' in k else 1,
            k
        ))
        
        for key in geom_keys_sorted:
            if key in before or key in after:
                # Generate readable label from key
                label = key.replace('geom_', '').replace('_target', '').replace('_', ' ').title()
                # Use appropriate format based on key content
                if 'loss' in key or 'rms_z' in key or 'rms_delta' in key:
                    fmt = '12.4f'
                elif 'n_' in key or key.endswith('_n'):
                    fmt = '12.0f'
                else:
                    fmt = '12.4f'
                b_val = before.get(key, 0)
                a_val = after.get(key, 0)
                print(format_row(label, b_val, a_val, fmt))
        
        print(f"{'─'*70}\n")
    
    def log_adp_comparison(self, before: Dict[str, float], after: Dict[str, float],
                           weight: float = None):
        """
        Log ADP refinement comparison showing B-factor metrics before/after.

        Parameters
        ----------
        before : dict
            Metrics dict from collect_metrics() before ADP refinement.
        after : dict
            Metrics dict from collect_metrics() after ADP refinement.
        weight : float, optional
            ADP weight used.
        """
        print(f"\n{'─'*70}")
        print(f"  ADP Refinement Summary")
        if weight is not None:
            print(f"  ADP weight: {weight:.3f}")
        print(f"{'─'*70}")
        
        print(f"\n  {'Metric':<25} {'Before':>12} {'After':>12} {'Change':>12}")
        print(f"  {'-'*61}")
        
        def format_row(label, b_val, a_val, fmt):
            delta = a_val - b_val
            # Use +/- sign format for change column
            return f"  {label:<25} {b_val:>{fmt}} {a_val:>{fmt}} {delta:>+{fmt}}"
        
        # R-factor metrics
        for key, label, fmt in [
            ('rwork', 'Rwork', '12.4f'),
            ('rfree', 'Rfree', '12.4f'),
            ('rfree_gap', 'Rfree-Rwork gap', '12.4f'),
        ]:
            b_val = before.get(key, 0)
            a_val = after.get(key, 0)
            print(format_row(label, b_val, a_val, fmt))
        
        print()
        
        # X-ray loss
        for key, label, fmt in [
            ('nll_xray_work', 'X-ray NLL (work)', '12.4f'),
            ('nll_xray_test', 'X-ray NLL (test)', '12.4f'),
        ]:
            b_val = before.get(key, 0)
            a_val = after.get(key, 0)
            print(format_row(label, b_val, a_val, fmt))
        
        print()
        
        # ADP metrics - dynamically extract from before/after dicts
        adp_keys = [k for k in set(before.keys()) | set(after.keys()) if k.startswith('adp_')]
        # Sort keys for consistent output: total_loss first, then by target name
        adp_keys_sorted = sorted(adp_keys, key=lambda k: (
            0 if 'total_loss' in k else 1,
            k
        ))
        
        for key in adp_keys_sorted:
            if key in before or key in after:
                # Generate readable label from key
                label = key.replace('adp_', '').replace('_target', '').replace('_', ' ').title()
                # Use appropriate format based on key content
                if 'loss' in key:
                    fmt = '12.4f'
                elif 'mean_b' in key or 'rms' in key:
                    fmt = '12.2f'
                elif 'sigma' in key:
                    fmt = '12.4f'
                else:
                    fmt = '12.4f'
                b_val = before.get(key, 0)
                a_val = after.get(key, 0)
                print(format_row(label, b_val, a_val, fmt))
        
        print(f"{'─'*70}\n")
    
    def update_effective_weights(self, phase='all', cycle=0,recompute=False):
        """
        Update effective weights using the weighting module.

        Parameters
        ----------
        phase : str, optional
            Refinement phase - 'xyz', 'b', or 'all'. Default is 'all'.
        cycle : int, optional
            Current refinement cycle number. Default is 0.
        recompute : bool, optional
            Whether to recompute weights. Default is False.
        """
        self.effective_weights = self.weighter(refinement_obj=self,phase=phase, cycle=cycle, recompute=recompute)
        if self.verbose > 2:
            print(f"Updated weights via {type(self.weighter).__name__}: {self.effective_weights}")

    def setup_optimizer(self, **kwargs):
        from torch.optim import Adam
        self.optimizer = Adam(self.parameters(), **kwargs)

    def run_refinement(self, macro_cycles=5, n_steps=10, lr=[1e-2,5e-4,1e-3, 5e-4, 1e-4]):
        """
        Run refinement cycles.

        Parameters
        ----------
        macro_cycles : int, optional
            Number of macro cycles. Default is 5.
        n_steps : int, optional
            Steps per learning rate. Default is 10.
        lr : list of float, optional
            Learning rate schedule. Default is [1e-2, 5e-4, 1e-3, 5e-4, 1e-4].
        """
        for cycle in range(macro_cycles):
            self.scaler.unfreeze()
            self.get_scales()
            self.scaler.freeze()
            
            # Update weights for this cycle
            self.update_effective_weights(phase='all', cycle=cycle)
            
            self.setup_optimizer(lr=lr[0])
            if self.verbose > 0:
                print(f"Starting macro cycle {cycle+1}/{macro_cycles} with learning rate {self.lr if isinstance(self.lr, float) else self.lr[cycle]}")
            for _lr in lr:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = _lr
                for step in range(n_steps):
                    self.optimizer.zero_grad()
                    total_loss, xray_work, restraints, xray_test = self.loss()
                    adp_loss = self.model.adp_loss()
                    total_loss = total_loss + adp_loss * self.effective_weights.get('adp', 0.0)
                    if torch.isnan(total_loss):
                        raise ValueError("NaN encountered in total loss during refinement.")
                    total_loss.backward()
                    self.optimizer.step()
                    if self.verbose > 2:
                        print(f"  Step {step+1}/{n_steps}, Total Loss: {total_loss.item():.4f}, XRay Work NLL: {xray_work.item():.4f}, Restraints Loss: {restraints.item():.4f}, XRay Test NLL: {xray_test.item():.4f}")
                
                # Update weights after each learning rate step
                self.update_effective_weights(phase='all', cycle=cycle)
                
                if self.verbose > 1:
                    print(f"  Ran for {_lr}, Total Loss: {total_loss.item():.4f}, XRay Work NLL: {xray_work.item():.4f}, Restraints Loss: {restraints.item():.4f}, XRay Test NLL: {xray_test.item():.4f}")
            if self.verbose > 0:
                rwork, rfree = self.get_rfactor()
                print(f'Nll work: {xray_work.item():.4f}, Nll test: {xray_test.item():.4f}, Nll: Restraints: {restraints.item():.4f}')
                print('Weights:', self.effective_weights)
                print(f"  R-work: {rwork:.4f}, R-free: {rfree:.4f}")

    def cuda(self):
        super().cuda()
        self.model.cuda()  # Explicitly call cuda on model to update its device attributes
        self.reflection_data.cuda()
        self.scaler.cuda() if hasattr(self.scaler, 'cuda') else None  # Also update scaler if it has cuda method
        self.restraints.cuda() if hasattr(self.restraints, 'cuda') else None  # Also update restraints if it has cuda method
        self.device = torch.device('cuda')
        return self
    
    def cpu(self):
        super().cpu()
        self.model.cpu()  # Explicitly call cpu on model to update its device attribute
        self.scaler.cpu() if hasattr(self.scaler, 'cpu') else None  # Also update scaler if it has cpu method
        self.restraints.cpu() if hasattr(self.restraints, 'cpu') else None  # Also update restraints if it has cpu method
        self.device = torch.device('cpu')
        return self

    def get_rfactor(self):
        return self.scaler.rfactor()

    def update_outliers(self, z_threshold=4.0):
        with torch.no_grad():
            self.reflection_data = self.reflection_data.update_outliers(self.model, self.scaler, z_threshold=z_threshold)
            self.register_buffer('hkl', self.reflection_data.get_hkl())
            self.setup_scaler()

    def plot_fcalc_vs_fobs(self,outpath='fcalc_vs_fobs.png'):
        import matplotlib.pyplot as plt
        with torch.no_grad():
            hkl, F_obs, sigma_F_obs, self.rfree_flags = self.reflection_data()
            self.get_Fcalc()
            F_calc = self.F_calc
            F_obs_amp = torch.abs(F_obs).cpu().numpy()
            F_calc_amp = torch.abs(F_calc).cpu().numpy()
            plt.figure(figsize=(8,8))
            plt.scatter(F_obs_amp, F_calc_amp, alpha=0.5)
            plt.plot([0, max(F_obs_amp)], [0, max(F_obs_amp)], color='red', linestyle='--')
            plt.xlabel('Observed |F|')
            plt.ylabel('Calculated |F|')
            plt.title('Fcalc vs Fobs')
            plt.grid()
            plt.savefig(outpath)
    
    def write_out_mtz(self, out_mtz_path='refined_output.mtz'):
        with torch.no_grad():
            hkl, _, _, _ = self.reflection_data(mask=False)
            fcalc = self.scaler(self.get_fcalc(hkl), use_mask=False)
            self.reflection_data.write_mtz(out_mtz_path, fcalc)
    
    def write_out_pdb(self, out_pdb_path='refined_output.pdb'):
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
    def create_from_state_dict(cls, state_dict: dict, device: torch.device = torch.device('cpu'),
                               verbose: int = 1) -> 'Refinement':
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
        >>> # Save
        >>> torch.save(refinement.state_dict(), 'refinement.pt')
        >>>
        >>> # Load
        >>> state = torch.load('refinement.pt')
        >>> refinement = Refinement.create_from_state_dict(state)
        >>>
        >>> # Continue refinement
        >>> rwork, rfree = refinement.get_rfactor()
        >>> print(f"Restored at R-work={rwork:.4f}, R-free={rfree:.4f}")
        """
        # Helper to extract submodule state from flattened state_dict
        def extract_submodule_state(state_dict: dict, prefix: str) -> dict:
            """Extract keys starting with prefix and strip the prefix."""
            result = {}
            prefix_with_dot = prefix + '.'
            for key, value in state_dict.items():
                if key.startswith(prefix_with_dot):
                    result[key[len(prefix_with_dot):]] = value
            return result
        
        # Extract submodule states from flattened keys
        model_state = extract_submodule_state(state_dict, 'model')
        reflection_data_state = extract_submodule_state(state_dict, 'reflection_data')
        scaler_state = extract_submodule_state(state_dict, 'scaler')
        restraints_state = extract_submodule_state(state_dict, 'restraints')
        weighter_state = extract_submodule_state(state_dict, 'weighter')
        
        if verbose > 0:
            print(f"Extracted state dict sizes: model={len(model_state)}, data={len(reflection_data_state)}, "
                  f"scaler={len(scaler_state)}, restraints={len(restraints_state)}")
        
        # Create submodules using their factory methods
        # These properly set up structure before loading values
        reflection_data = ReflectionData.create_from_state_dict(
            reflection_data_state, device=device, verbose=verbose
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
        instance.max_res = model_state.get('_metadata_max_res', None)
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
            n_refl = instance.reflection_data.hkl.shape[0] if instance.reflection_data.hkl is not None else 0
            print(f"Created Refinement from state_dict: {n_atoms} atoms, {n_refl} reflections")
        
        return instance
    

