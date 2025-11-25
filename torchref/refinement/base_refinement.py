from typing import Any, Optional, Dict


from torchref.io.Data import ReflectionData
from torchref.model.model_ft import ModelFT
from torch.nn import Module as nnModule
from torch.nn.modules.module import _IncompatibleKeys
import torch
from torchref.restraints.restraints import Restraints
from torchref.scaling.scaler import Scaler
from torchref.utils.debug_utils import DebugMixin

# Target system imports
from torchref.refinement.targets import (
    Target,
    GaussianXrayTarget,
    LeastSquaresXrayTarget,
    MaximumLikelihoodXrayTarget,
    BondTarget,
    AngleTarget,
    TorsionTarget,
    TotalGeometryTarget,
    ADPSimilarityTarget,
    ADPEntropyTarget,
    CompositeTarget,
    create_xray_target,
    create_geometry_target,
    create_default_targets,
)

class Refinement(DebugMixin, nnModule):
    def __init__(self, data_file:str = None, pdb:str = None,  cif = None, verbose: int = 1, max_res: float = None, device: torch.device = torch.device('cpu'), 
                 weighter: 'LossWeightingModule' = None, nbins: int = 20):
        """
        Refinement class to handle the overall refinement process.
        
        Args:
            data_file (str): Path to the MTZ or CIF file containing reflection data.
            pdb (str): Path to the PDB or CIF file containing the initial model.
            cif (str, optional): Path to the CIF file for restraints. Defaults to None.
            verbose (int, optional): Verbosity level. Defaults to 1.
            max_res (float, optional): Maximum resolution for reflections. Defaults to None.
            device (torch.device): Computation device. Defaults to CPU.
            weighter (LossWeightingModule): Loss weighting module for managing loss weights.
                    If None, will create a default ResolutionDependentWeighting instance.
        """
        super().__init__()
        self.device = device
        self.verbose = verbose
        self.data_file = data_file
        self.pdb = pdb
        self.history = dict()
        self.max_res = max_res
        self.nbins = nbins
        
        # Allow empty initialization for loading from state_dict
        if data_file is None and pdb is None:
            return

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
            self.model = ModelFT(verbose=self.verbose,max_res=self.max_res,device=self.device)
            
            if pdb.endswith('.cif'):
                self.model.load_cif(pdb)
            elif pdb.endswith('.pdb'):
                self.model.load_pdb(pdb)
            else:
                raise ValueError(f"Unsupported model file format: {pdb}. Supported formats are .pdb and .cif")
            
            self.scaler = Scaler(self.model, self.reflection_data, verbose=self.verbose, device=self.device)
            self.lr = 1e-3
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
            self.debug_on_error(e)
            raise e
    
    def _init_targets(self, xray_mode: str = 'ml'):
        """
        Initialize target functions.
        
        Args:
            xray_mode: X-ray target mode ('gaussian', 'ls', 'ml')
        """
        # X-ray targets
        self.xray_target_work = create_xray_target(self, xray_mode, use_work_set=True, verbose=self.verbose)
        self.xray_target_test = create_xray_target(self, xray_mode, use_work_set=False, verbose=self.verbose)
        
        # Geometry targets
        self.bond_target = BondTarget(self, verbose=self.verbose)
        self.angle_target = AngleTarget(self, verbose=self.verbose)
        self.torsion_target = TorsionTarget(self, verbose=self.verbose)
        self.geometry_target = TotalGeometryTarget(self, verbose=self.verbose)
        
        # ADP targets
        self.adp_similarity_target = ADPSimilarityTarget(self, sigma=2.0, verbose=self.verbose)
        self.adp_entropy_target = ADPEntropyTarget(self, verbose=self.verbose)
        
        if self.verbose > 0:
            print(f"Initialized targets with xray_mode='{xray_mode}'")
    
    def set_xray_target_mode(self, mode: str):
        """
        Change the X-ray target mode.
        
        Args:
            mode: 'gaussian', 'ls', or 'ml'
        """
        self.xray_target_work = create_xray_target(self, mode, use_work_set=True, verbose=self.verbose)
        self.xray_target_test = create_xray_target(self, mode, use_work_set=False, verbose=self.verbose)
        if self.verbose > 0:
            print(f"Changed X-ray target mode to '{mode}'")

    @property
    def data(self):
        """Expose reflection_data as 'data' for weighting module compatibility."""
        return self.reflection_data

    def setup_grad_weighting(self):
        """
        Setup gradient-informed weighting module.
        """
        from torchref.refinement.loss_weighting import GradientInformedWeighting
        self.get_scales()
        self.weighter = GradientInformedWeighting(self,target_weights=self.effective_weights)
        if self.verbose > 0:
            print("Using GradientInformedWeighting for loss weighting")

    def set_up_smart_weighting(
        self,
        base_restraint: float = 1.0,
        base_adp: float = 1.0,
        nll_modulation_strength: float = 0.5,
        recompute_gradnorms_every: int = 5
    ):
        """
        Set up smart hybrid weighting that combines gradient norm balancing with ML target-aware modulation.
        
        This is the RECOMMENDED weighting strategy as it provides:
        - Balanced gradients across loss terms (via gradient norms)
        - Physical correctness (via ML target values)
        - Avoids over-restraining structures that are already geometrically good
        
        The weighting strategy:
        1. Computes gradient norms to balance optimization landscape
        2. Modulates weights based on ML target values to respect physical meaning
        3. High geometry loss → increase restraint weight
        4. Low geometry loss → reduce restraint weight (avoid over-restraining)
        
        Args:
            base_restraint: Base restraint weight (default: 1.0)
            base_adp: Base ADP weight (default: 1.0)
            nll_modulation_strength: How strongly ML target values modulate weights (0-1)
                                    0 = pure gradient norm balancing
                                    1 = strong ML target influence
                                    default: 0.5 (balanced)
            recompute_gradnorms_every: Recompute gradient norms every N cycles (default: 5)
        
        Example:
            >>> refinement = Refinement(data_file="data.mtz", pdb="model.pdb")
            >>> refinement.set_up_smart_weighting(
            ...     base_restraint=1.0,
            ...     base_adp=1.0,
            ...     nll_modulation_strength=0.5
            ... )
            >>> # Now refine with smart weighting
            >>> refinement.refine(n_cycles=100)
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
            verbose=self.verbose
        )
        
        if self.verbose > 0:
            print(f"Using HybridGradNormMLWeighting for smart loss weighting")
            print(f"  Base weights: restraints={base_restraint}, adp={base_adp}")
            print(f"  ML target modulation strength: {nll_modulation_strength}")
            print(f"  Recompute gradient norms every {recompute_gradnorms_every} cycles")

    def get_scales(self):
        if not hasattr(self, 'scaler'):
            self.setup_scaler()
        self.scaler.initialize()
        self.reflection_data.find_outliers(self.model, self.scaler, z_threshold=4.0)
        self.scaler.refine_lbfgs()
        self.reflection_data.find_outliers(self.model, self.scaler, z_threshold=4.0)

    def setup_scaler(self):
        self.scaler = Scaler(self.model, self.reflection_data, nbins=self.nbins, verbose=self.verbose, device=self.device)

    def parameters(self, recurse: bool = True):
        """
        Return unique parameters from this module and all submodules.

        Uses the default Module.parameters() to gather parameters, then removes
        duplicates while preserving order to avoid passing the same tensor
        multiple times to the optimizer.
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
        return self.model.adp_kl_divergence_loss()

    def get_Fcalc(self,hkl=None, recalc=False):
        return torch.abs(self.get_fcalc(hkl, recalc=recalc))

    def get_F_calc_scaled(self,hkl=None, recalc=False):
        return torch.abs(self.get_fcalc_scaled(hkl, recalc=recalc))

    def nll_xray(self):
        """
        Compute X-ray negative log-likelihood for work and test sets.
        
        Returns:
            Tuple of (work_nll, test_nll) torch.Tensors
        """
        return self.xray_target_work(), self.xray_target_test()

    # =========================================================================
    # Target-based Loss Methods (new pattern - instantiate once, evaluate each iteration)
    # =========================================================================
    
    def xray_loss_work(self) -> torch.Tensor:
        """Compute X-ray loss on work set using instantiated target."""
        return self.xray_target_work()
    
    def xray_loss_test(self) -> torch.Tensor:
        """Compute X-ray loss on test set using instantiated target."""
        return self.xray_target_test()
    
    def bond_loss(self) -> torch.Tensor:
        """Compute bond length NLL using instantiated target."""
        return self.bond_target()
    
    def angle_loss(self) -> torch.Tensor:
        """Compute angle NLL using instantiated target."""
        return self.angle_target()
    
    def torsion_loss(self) -> torch.Tensor:
        """Compute torsion angle NLL using instantiated target."""
        return self.torsion_target()
    
    def geometry_loss(self) -> torch.Tensor:
        """Compute total geometry NLL using instantiated target."""
        return self.geometry_target()
    
    def adp_simu_loss(self, sigma: float = 2.0) -> torch.Tensor:
        """
        Compute ADP similarity loss (SIMU restraint).
        
        Args:
            sigma: Target sigma for B-factor differences (default: 2.0 Å²)
        """
        self.adp_similarity_target.sigma = sigma
        return self.adp_similarity_target()
    
    def adp_entropy_loss(self) -> torch.Tensor:
        """Compute ADP entropy loss using instantiated target."""
        return self.adp_entropy_target()

    def loss(self):
        """
        Compute total loss using instantiated targets.
        
        Returns:
            Tuple of (total_loss, xray_work, restraints, xray_test)
        """
        xray_work = self.xray_loss_work()
        xray_test = self.xray_loss_test()
        restraints = self.geometry_loss()
        total_loss = self.effective_weights['xray'] * xray_work + self.effective_weights['restraints'] * restraints
        return total_loss, xray_work, restraints, xray_test

    def xray_loss(self):
        """Compute X-ray loss on work set."""
        return self.xray_loss_work()

    def restraints_loss(self):
        """Compute total geometry restraints loss."""
        return self.geometry_loss()
    
    def update_effective_weights(self, phase='all', cycle=0,recompute=False):
        """
        Update effective weights using the weighting module.
        
        Args:
            phase (str): Refinement phase - 'xyz', 'b', or 'all'
            cycle (int): Current refinement cycle number
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
        
        Args:
            macro_cycles (int): Number of macro cycles
            n_steps (int): Steps per learning rate
            lr (list): Learning rate schedule
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

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """
        Returns a dictionary containing the complete state of the Refinement.
        
        This includes:
        - All submodules (model, scaler, restraints, reflection_data, weighter)
        - Refinement-specific state (history, effective_weights, lr, etc.)
        - File paths and configuration
        
        Args:
            destination: Optional dict to populate
            prefix: Prefix for parameter names
            keep_vars: Whether to keep variables in computational graph
            
        Returns:
            dict: Complete state dictionary
            
        Example:
            >>> # Save refinement state
            >>> state = refinement.state_dict()
            >>> torch.save(state, 'refinement_checkpoint.pt')
            >>> 
            >>> # Later, restore state
            >>> refinement2 = Refinement(data_file, pdb_file)
            >>> refinement2.load_state_dict(torch.load('refinement_checkpoint.pt'))
        """
        # Get parent class state_dict
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        
        # Add submodule states
        state[prefix + 'model'] = self.model.state_dict()
        state[prefix + 'scaler'] = self.scaler.state_dict()
        state[prefix + 'restraints'] = self.restraints.state_dict()
        state[prefix + 'reflection_data'] = self.reflection_data.state_dict()
        
        if hasattr(self, 'weighter') and self.weighter is not None:
            state[prefix + 'weighter'] = self.weighter.state_dict() if hasattr(self.weighter, 'state_dict') else None
            state[prefix + 'weighter_type'] = type(self.weighter).__name__
        else:
            state[prefix + 'weighter'] = None
            state[prefix + 'weighter_type'] = None
        
        # Add Refinement-specific state
        state[prefix + 'history'] = self.history
        state[prefix + 'effective_weights'] = self.effective_weights if hasattr(self, 'effective_weights') else {}
        state[prefix + 'lr'] = self.lr
        state[prefix + 'data_file'] = self.data_file
        state[prefix + 'pdb'] = self.pdb
        state[prefix + 'max_res'] = self.max_res
        state[prefix + 'verbose'] = self.verbose
        
        return state

    def load_state_dict(self, state_dict, strict=True):
        """
        Loads the Refinement state from a dictionary.
        
        This restores all submodules and refinement-specific state.
        
        Args:
            state_dict: Dictionary containing refinement state
            strict: Whether to strictly enforce that keys match
            
        Example:
            >>> refinement = Refinement(data_file, pdb_file)
            >>> state = torch.load('refinement_checkpoint.pt')
            >>> refinement.load_state_dict(state)
        """
        # Extract submodule states
        model_state = state_dict.pop('model', None)
        scaler_state = state_dict.pop('scaler', None)
        restraints_state = state_dict.pop('restraints', None)
        reflection_data_state = state_dict.pop('reflection_data', None)
        weighter_state = state_dict.pop('weighter', None)
        weighter_type = state_dict.pop('weighter_type', None)
        
        # Extract Refinement-specific state
        self.history = state_dict.pop('history', {})
        self.effective_weights = state_dict.pop('effective_weights', {})
        self.lr = state_dict.pop('lr', 1e-3)
        self.data_file = state_dict.pop('data_file', self.data_file)
        self.pdb = state_dict.pop('pdb', self.pdb)
        self.max_res = state_dict.pop('max_res', self.max_res)
        self.verbose = state_dict.pop('verbose', self.verbose)
        
        # Instantiate submodules if they don't exist (e.g. empty init)
        if not hasattr(self, 'model') or self.model is None:
            # Assume ModelFT since Refinement uses it
            self.model = ModelFT(verbose=self.verbose, device=self.device)
            
        if not hasattr(self, 'reflection_data') or self.reflection_data is None:
            self.reflection_data = ReflectionData(verbose=self.verbose, device=self.device)
            
        # Load model and data first so they are populated for Scaler/Restraints
        if model_state is not None:
            self.model.load_state_dict(model_state, strict=strict)
        if reflection_data_state is not None:
            self.reflection_data.load_state_dict(reflection_data_state, strict=strict)
            
        # Now create Scaler and Restraints which depend on model/data
        if not hasattr(self, 'scaler') or self.scaler is None:
            self.scaler = Scaler(self.model, self.reflection_data, verbose=self.verbose, device=self.device)
            
        if not hasattr(self, 'restraints') or self.restraints is None:
            self.restraints = Restraints(self.model, verbose=self.verbose)
        
        # Load their states
        if scaler_state is not None:
            self.scaler.load_state_dict(scaler_state, strict=strict)
        if restraints_state is not None:
            self.restraints.load_state_dict(restraints_state, strict=strict)
            
        # Instantiate and load weighter
        if weighter_state is not None:
            # If weighter_type is known, try to instantiate it
            if weighter_type is not None:
                import torchref.refinement.loss_weighting as lw
                if hasattr(lw, weighter_type):
                    WeighterClass = getattr(lw, weighter_type)
                    
                    # If weighter already exists and is correct type, use it
                    if hasattr(self, 'weighter') and self.weighter is not None and type(self.weighter).__name__ == weighter_type:
                        pass
                    else:
                        # Try to instantiate
                        try:
                            # Some weighters need refinement object
                            if weighter_type in ['GradientInformedWeighting', 'HybridGradNormNLLWeighting']:
                                self.weighter = WeighterClass(self)
                            else:
                                self.weighter = WeighterClass()
                        except Exception as e:
                            if self.verbose > 0:
                                print(f"Warning: Could not instantiate weighter {weighter_type}: {e}")
                                print("Using existing weighter or default.")
            
            if hasattr(self, 'weighter') and self.weighter is not None and hasattr(self.weighter, 'load_state_dict'):
                self.weighter.load_state_dict(weighter_state, strict=strict)
        
        # Since we manually loaded all submodules using the nested dictionaries,
        # we don't need to call super().load_state_dict() which would try to load
        # the flattened keys and cause double loading or "unexpected key" errors.
        # Refinement itself has no parameters/buffers, only submodules and python attributes.
        
        return _IncompatibleKeys(missing_keys=[], unexpected_keys=[])

    def save_state(self, path: str):
        """
        Save the complete state of the refinement to a file.
        
        Args:
            path (str): Path to save the state dictionary to.
        """
        torch.save(self.state_dict(), path)
        if self.verbose > 0:
            print(f"Saved refinement state to {path}")

    def load_state(self, path: str, strict: bool = True):
        """
        Load the complete state of the refinement from a file.
        
        Args:
            path (str): Path to load the state dictionary from.
            strict (bool): Whether to strictly enforce that keys match.
        """
        state_dict = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(state_dict, strict=strict)
        if self.verbose > 0:
            print(f"Loaded refinement state from {path}")
    

