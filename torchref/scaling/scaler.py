"""

A class for scaling and post corrections of scattering factors.

Currently implements:
- Overall scale per resolution bin
- B-factor per resolution bin
- Anisotropy correction
- Solvent model correction

"""



import torch
import torch.nn as nn
from torchref.io.Data import ReflectionData
from torchref.math_functions.math_torch import get_scattering_vectors, U_to_matrix, nll_xray, get_rfactors, bin_wise_rfactors, nll_xray_lognormal
from torchref.scaling.solvent_new import SolventModel
from torchref.utils.debug_utils import DebugMixin
from torchref.utils.utils import ModuleReference

class Scaler(DebugMixin, nn.Module):
    def __init__(self, model, data: ReflectionData, nbins: int = 20, verbose: int = 1,device=torch.device('cpu')):
        """
        Scaler class to apply scaling and corrections to calculated structure factors.
        self.device = fcalc.device
        Args:
            fcalc (torch.Tensor): Calculated structure factors.
            fobs (torch.Tensor): Observed structure factors.
            hkl (torch.Tensor): Miller indices corresponding to the structure factors.
        """
        super(Scaler, self).__init__()
        self.device = device
        self.to(self.device)
        # Wrap model in ModuleReference to prevent registration as submodule
        self._model = ModuleReference(model)
        self._data = ModuleReference(data)

        self.nbins = nbins
        self.verbose = verbose
        self.cell = data.cell
        # Don't store hkl directly - always access it from data to avoid device mismatch
        # self.hkl will be a property that accesses data.hkl
        self.register_buffer('s',get_scattering_vectors(data.hkl, self.cell))
        bins, self.nbins = self._data.get_bins(self.nbins)
        self.register_buffer('bins', bins)
        if self.verbose > 0:
            print(f"Initialized Scaler with {self.nbins} bins.")
        self.frozen = False

    def initialize(self):
        self.calc_initial_scale()
        self.setup_solvent()
        self.setup_anisotropy_correction()

    @property
    def hkl(self):
        return self._data.hkl
    
    def freeze(self):
        self.frozen = True

    def unfreeze(self):
        self.frozen = False

    def calc_initial_scale(self):
        """
        Calculate the initial scale factor based on the ratio of observed to calculated structure factors.
        Excludes reflections with negative intensities to avoid bias from French-Wilson conversion.
        """
        hkl, fobs, sigma, rfree = self._data(mask=False)
        fcalc = self._model(hkl)
        if self.verbose > 0:
            print(f"Calculating initial scale factors using {self.nbins} bins.")
        assert torch.all(torch.isfinite(fcalc)), "Non-finite values found in fcalc during initial scale calculation."
        
        scales = torch.zeros(self.nbins, device=self.device, dtype=fobs.dtype)
        counts = torch.zeros(self.nbins, device=self.device, dtype=fobs.dtype)
        fcalc_amp = torch.abs(fcalc).to(fobs.dtype)
        # Exclude reflections with negative intensities from scale calculation
        # These have biased F values from French-Wilson conversion
        if hasattr(self._data, 'I') and self._data.I is not None:
            positive_mask = self._data.I > 0
            if self.verbose > 1:
                n_excluded = (~positive_mask).sum().item()
                print(f"Excluding {n_excluded} negative intensity reflections from scale calculation")
        else:
            positive_mask = torch.ones_like(fobs, dtype=torch.bool)
        
        # Calculate ratios only for positive intensity reflections

        mask = (self._data.masks() & rfree & positive_mask).to(torch.bool)
        bins = self.bins[mask].to(torch.int64)

        fobs = fobs.clamp(min=1e-3)[mask]
        fcalc_amp = fcalc_amp.clamp(min=1e-3)[mask]

        log_ratios = torch.log(fobs) - torch.log(fcalc_amp)
        assert torch.all(torch.isfinite(log_ratios)), f"Non-finite log ratios encountered in initial scale calculation {torch.sum(~torch.isfinite(log_ratios)).item()}"
    
        # Ensure all tensors are on the same device for scatter_add
        log_ratios = log_ratios.to(self.device)
        bins = bins.to(self.device)
        counts_vals = torch.ones_like(self.bins, device=self.device, dtype=fobs.dtype)
        sum_log_scales = torch.scatter_add(scales, 0, bins, log_ratios)
        counts = torch.scatter_add(counts, 0, bins, counts_vals)
        log_scale = sum_log_scales / (counts + 1e-6)
        initial_log_scale = log_scale
        if self.verbose > 1:
            print("Initial scale factors per bin:", initial_log_scale.detach().cpu().numpy())
        self.log_scale = nn.Parameter(initial_log_scale.detach().to(self.device))    
        return self.log_scale
    
    def setup_anisotropy_correction(self):
        self.U = nn.Parameter(torch.normal(0, 0.001, (6,), dtype=torch.float32, device=self.device))

    def anisotropy_correction(self):
        U = U_to_matrix(self.U)
        exp = -2 * torch.pi ** 2 * torch.einsum('ij,jk,ik->i', self.s, U, self.s)
        return torch.exp(exp.clamp(max=10.0,min=-10.0))

    def fit_anisotropy(self,fcalc: torch.Tensor):
        if not hasattr(self, 'U'):
            self.U = nn.Parameter(torch.normal(0, 0.01, (6,), dtype=torch.float32, device=self.device))
        hkl, fobs, sigma, rfree = self._data()

        fobs = fobs.to(torch.float32).detach()

        fcalc = torch.abs(fcalc).to(torch.float32).detach()
        optimizer = torch.optim.Adam([self.U, self.log_scale], lr=1e-1)
        for i in range(100):
            optimizer.zero_grad()
            scaled_fcalc = self.forward(fcalc)
            loss = nll_xray(fobs[rfree], scaled_fcalc[rfree], sigma[rfree])

            loss.backward()
            optimizer.step()
            if self.verbose > 0 and (i % 10 == 0 or i == 99):
                print(f"Anisotropy fit iteration {i+1}/100, Loss: {loss.item():.4f}")

    def setup_solvent(self):
        self.solvent = SolventModel(self._model, device=self.device, radius=1.1, k_solvent=0.35, b_solvent=46.0, verbose=self.verbose)
        self.solvent.update_solvent()

    def fit_all_scales(self):
        hkl, fobs, sigma, rfree = self._data()
        fobs = fobs.to(torch.float32).detach()
        fcalc = self._model(hkl).detach()
        for lr in [1e-1, 5e-2, 1e-2]:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
            for i in range(20):
                optimizer.zero_grad()
                scaled_fcalc = self.forward(fcalc)
                nll_loss = nll_xray(fobs[rfree], scaled_fcalc[rfree], sigma[rfree])
                if torch.isnan(nll_loss):
                    raise ValueError("NaN encountered in NLL loss during scale fitting.")
                nll_log_loss_xray = nll_xray_lognormal(fobs[rfree], scaled_fcalc[rfree], sigma[rfree])
                loss = nll_loss 
                loss.backward()
                optimizer.step()
            if self.verbose > 1: print(f"Solvent fit after step, Loss: {loss.item():.4f}, NLL: {nll_loss.item():.4f}, LogLoss: {nll_log_loss_xray.item():.4f}")
        
    def cuda(self, device=None):
        """
        Move the Scaler module to GPU.

        Args:
            device (torch.device, optional): The target device. If None, uses the default CUDA device.
        """
        super().cuda(device)
        if hasattr(self, 'solvent'):
            self.solvent.cuda(device)
        self.device = torch.device('cuda' if device is None else device)
        if self.verbose > 1:
            print(f"Scaler moved to device: {self.device}")

    def cpu(self):
        """
        Move the Scaler module to CPU.
        """
        super().cpu()
        if hasattr(self, 'solvent'):
            self.solvent.cpu()
        self.device = next(self.parameters()).device
        if self.verbose > 1:
            print("Scaler moved to CPU")

    def rfactor(self):
        """
        Calculate the R-factor between observed and calculated structure factors.

        Args:
            fcalc (torch.Tensor): Calculated structure factors.
        Returns:
            float: R-factor value.
        """
        hkl, fobs, _, rfree = self._data()
        fcalc = self._model(hkl)
        fcalc_scaled = self.forward(fcalc)
        return get_rfactors(torch.abs(fobs), torch.abs(fcalc_scaled), rfree)

    def bin_wise_rfactor(self, fcalc=None):
        """
        Calculate the bin-wise R-factor between observed and calculated structure factors.

        Args:
            fcalc (torch.Tensor): Calculated structure factors.

        Returns:
            tuple: Mean resolution per bin, R work per bin, R free per bin.
        """
        hkl, fobs, _, rfree = self._data()
        if fcalc is None:
            fcalc = self._model(hkl)
        fcalc_scaled = self.forward(fcalc)
        mean_res_per_bin = self._data.mean_res_per_bin()
        return mean_res_per_bin, *bin_wise_rfactors(torch.abs(fobs), torch.abs(fcalc_scaled), rfree, self.bins[self._data.masks()])
    
    def setup_bin_wise_bfactor(self):
        self.bin_wise_bfactor = nn.Parameter(torch.zeros(self.nbins, dtype=torch.float32, device=self.device))
        
    def bin_wise_bfactor_correction(self):
        b_expanded = self.bin_wise_bfactor[self.bins]
        s = torch.norm(self.s, dim=1) 
        s_squared = s ** 2  # Now s² is correct for B-factor formula
        exp = -b_expanded * s_squared / 4
        return torch.exp(exp.clamp(max=10.0,min=-10.0))
    
    def get_binwise_mean_intensity(self):
        hkl, fobs, _, rfree = self._data()
        Fcalc = torch.abs(self(self._model(hkl)))
        intensities = torch.abs(fobs) ** 2
        calc_intensities = torch.abs(Fcalc) ** 2
        mean_obs_intensity = torch.zeros(self.nbins, device=self.device)
        mean_calc_intensity = torch.zeros(self.nbins, device=self.device)
        counts = torch.zeros(self.nbins, device=self.device)
        counts_vals = torch.ones_like(Fcalc, device=self.device, dtype=fobs.dtype)
        mask = self._data.get_mask()
        mean_obs_intensity = torch.scatter_add(mean_obs_intensity, 0, self.bins.to(torch.int64)[mask][rfree], intensities[rfree])
        mean_calc_intensity = torch.scatter_add(mean_calc_intensity, 0, self.bins.to(torch.int64)[mask][rfree], calc_intensities[rfree])
        counts = torch.scatter_add(counts, 0, self.bins.to(torch.int64)[mask][rfree], counts_vals[rfree])
        mean_obs_intensity = mean_obs_intensity / (counts + 1e-6)
        mean_calc_intensity = mean_calc_intensity / (counts + 1e-6)
        return mean_obs_intensity, mean_calc_intensity, self._data.mean_res_per_bin()

    def screen_solvent_params(self,steps=15):
        hkl, fobs, sigma, rfree = self._data()
        fobs = fobs.to(torch.float32).detach()
        fcalc = self._model(hkl).detach()
        best_log_k_solvent = self.solvent.log_k_solvent.clone()
        best_b_solvent = self.solvent.b_solvent.clone()
        best_loss = float('inf')
        ksol_start, ksol_end = torch.log(torch.tensor(0.1, device=self.device)), torch.log(torch.tensor(1.4, device=self.device))
        for log_k_solvent in torch.linspace(ksol_start, ksol_end, steps=steps, device=self.device):
            for b_solvent in torch.linspace(20.0, 120.0, steps=steps, device=self.device):
                self.solvent.log_k_solvent.data = log_k_solvent.to(dtype=self.solvent.log_k_solvent.dtype)
                self.solvent.b_solvent.data = b_solvent.to(dtype=self.solvent.b_solvent.dtype)
                scaled_fcalc = self.forward(fcalc)
                nll_loss = nll_xray(fobs[rfree], scaled_fcalc[rfree], sigma[rfree])
                if nll_loss.item() < best_loss:
                    best_loss = nll_loss.item()
                    best_log_k_solvent = log_k_solvent.clone()
                    best_b_solvent = b_solvent.clone()
        self.solvent.log_k_solvent.data = best_log_k_solvent.to(dtype=self.solvent.log_k_solvent.dtype)
        self.solvent.b_solvent.data = best_b_solvent.to(dtype=self.solvent.b_solvent.dtype)
        if self.verbose > 0:
            print(f"Optimal solvent parameters found: log_k_solvent={best_log_k_solvent.item()}, b_solvent={best_b_solvent.item()}, NLL Loss={best_loss:.4f}")

    def refine_lbfgs(self,
                     nsteps: int = 3,
                     lr: float = 1.0,
                     max_iter: int = 200,
                     history_size: int = 10,
                     verbose: bool = True):
        """
        Refine scale parameters using LBFGS optimizer.
        
        This method optimizes the anisotropic scaling and B-factor parameters
        that relate calculated structure factors to observed structure factors.
        Uses the L-BFGS quasi-Newton optimization method for fast convergence.
        
        Args:
            nsteps: Number of LBFGS steps
            lr: Learning rate (typically 1.0 for LBFGS)
            max_iter: Maximum iterations per line search
            history_size: Number of previous gradients to store for Hessian approximation
            verbose: Print progress information
            
        Returns:
            Dictionary with refinement metrics including steps, xray_work, xray_test, rwork, rfree
            
        Example:
            >>> scaler.unfreeze()
            >>> metrics = scaler.refine_lbfgs(nsteps=5, verbose=True)
            >>> scaler.freeze()
        """
        # Ensure scaler is unfrozen
        was_frozen = self.frozen
        if was_frozen:
            self.unfreeze()

        self.screen_solvent_params(steps=14)
        
        # Create LBFGS optimizer for scaler parameters only
        optimizer = torch.optim.LBFGS(
            self.parameters(),
            lr=lr,
            max_iter=max_iter,
            history_size=history_size
        )
        
        def closure():
            optimizer.zero_grad()
            fcalc_scaled = self.forward(fcalc)
            loss = nll_xray(fobs[rfree], fcalc_scaled[rfree], sigma[rfree])
            
            # Handle non-finite loss gracefully for LBFGS line search
            if not torch.all(torch.isfinite(loss)):
                if self.verbose > 0:
                    import warnings
                    warnings.warn("Non-finite loss encountered during scale optimization. "
                                "LBFGS line search will reject this step and try a smaller step size.",
                                RuntimeWarning)
                # Return a large finite penalty to reject this step in line search
                # Don't call backward() as gradients would be invalid
                return torch.tensor(1e10, device=loss.device, dtype=loss.dtype, requires_grad=True)
            
            loss.backward(retain_graph=True)
            return loss
        
        # Track metrics
        metrics = {
            'target': 'scales',
            'steps': [],
            'xray_work': [],
            'xray_test': [],
            'rwork': [],
            'rfree': []
        }
        
        if verbose and self.verbose > 0:
            print("Refining scales with LBFGS...")

        hkl, fobs, sigma, rfree = self._data()
        fcalc = self._model(hkl).detach()

        if self.verbose > 2:
            assert torch.all(torch.isfinite(fcalc)), "Non-finite values found in fcalc during scale optimization."
        
        # Run optimization
        for step in range(nsteps):
            optimizer.step(closure)
            
            # Evaluate metrics
            with torch.no_grad():
                hkl, fobs, sigma, rfree = self._data()
                fcalc_scaled = self.forward(fcalc)
                
                xray_work = nll_xray(fobs[rfree], fcalc_scaled[rfree], sigma[rfree])
                xray_test = nll_xray(fobs[~rfree], fcalc_scaled[~rfree], sigma[~rfree])
                rwork, rfree_val = get_rfactors(torch.abs(fobs), torch.abs(fcalc_scaled), rfree)
                
                metrics['steps'].append(step + 1)
                metrics['xray_work'].append(xray_work.item())
                metrics['xray_test'].append(xray_test.item())
                metrics['rwork'].append(rwork)
                metrics['rfree'].append(rfree_val)
                
                if verbose and self.verbose > 2:
                    print(f"  Step {step+1}/{nsteps}: "
                          f"Rwork={rwork:.4f}, Rfree={rfree_val:.4f}, "
                          f"NLL_work={xray_work.item():.2f}, NLL_test={xray_test.item():.2f}")
        
        # Restore frozen state
        if was_frozen:
            self.freeze()
        
        if verbose and self.verbose > 0:
            with torch.no_grad():
                print(f"Scale refinement complete. rwork: {rwork:.4f}, rfree: {rfree_val:.4f}\n")
                print("Final Scale Parameters: ")
                for name, param in self.named_parameters():
                    if param.requires_grad:
                        print(f"  {name}: {param.data}")
        
        return metrics

    def parameters(self, recurse = True):
        if self.frozen:
            return []
        return super().parameters(recurse)

    def forward(self, fcalc, use_mask=True):
        """
        Forward pass for the Scaler module.

        Args:
            fcalc (torch.Tensor): Calculated structure factors. Expected shape (N,), an additional dimension ofr batch is possible.
            use_mask (bool): Whether to apply the data mask.

        Returns:
            torch.Tensor: Scaled structure factors.
        """
        batched = True

        if fcalc.ndim == 1:
            fcalc = fcalc.unsqueeze(0)  # Add batch dimension if missing
            batched = False

        if use_mask:
            mask = self._data.masks().to(torch.bool)

        else:
            mask = torch.ones(fcalc.shape[1], dtype=torch.bool, device=self.device)

        if hasattr(self, 'U'):
            anisotropy_factors = self.anisotropy_correction()
            aniso_correction = anisotropy_factors[mask]
        else:
            aniso_correction = torch.tensor(1.0, device=self.device, dtype=fcalc.dtype)

        if hasattr(self, 'solvent'):
            f_sol = self.solvent(self.hkl)
            f_sol = f_sol[mask]
        else:
            f_sol = torch.tensor(0.0, device=self.device, dtype=fcalc.dtype)

        if hasattr(self, 'log_scale'):
            K_overall = torch.exp(self.log_scale[self.bins[mask]])
        else:
            K_overall = torch.tensor(1.0, device=self.device, dtype=fcalc.dtype)
        
        if hasattr(self, 'bin_wise_bfactor'):
            bfactor_factors = self.bin_wise_bfactor_correction()
            b_overall = bfactor_factors[mask]
        else:
            b_overall = torch.tensor(1.0, device=self.device, dtype=fcalc.dtype)
        
        fcalc = K_overall.unsqueeze(0) * b_overall.unsqueeze(0) * (aniso_correction.unsqueeze(0) * fcalc + f_sol.unsqueeze(0))

        if not batched:
            fcalc = fcalc.squeeze(0)  # Remove batch dimension if it was added

        return fcalc

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """
        Returns a dictionary containing the complete state of the Scaler.
        
        This includes:
        - All registered buffers and parameters (via parent class)
        - Scaler-specific metadata (nbins, frozen state, etc.)
        - Solvent model state (if initialized)
        
        Note: Model and data references are NOT saved (managed separately)
        
        Args:
            destination: Optional dict to populate
            prefix: Prefix for parameter names
            keep_vars: Whether to keep variables in computational graph
            
        Returns:
            dict: Complete state dictionary
        """
        # Get parent class state_dict (includes all registered buffers and parameters)
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        
        # Add Scaler-specific metadata
        state[prefix + 'nbins'] = self.nbins
        state[prefix + 'verbose'] = self.verbose
        state[prefix + 'frozen'] = self.frozen
        
        # Save solvent model state if it exists
        if hasattr(self, 'solvent'):
            state[prefix + 'solvent'] = self.solvent.state_dict()
        
        return state

    def load_state_dict(self, state_dict, strict=True):
        """
        Loads the Scaler state from a dictionary.
        
        Note: This assumes model and data are already set via __init__ or assignment
        
        Args:
            state_dict: Dictionary containing scaler state
            strict: Whether to strictly enforce that keys match
        """
        # Extract Scaler-specific metadata
        self.nbins = state_dict.pop('nbins', 20)
        self.verbose = state_dict.pop('verbose', 1)
        self.frozen = state_dict.pop('frozen', False)
        
        # Extract and load solvent model state if it exists
        solvent_state = state_dict.pop('solvent', None)
        
        # If solvent state exists but module doesn't, instantiate it
        if solvent_state is not None and not hasattr(self, 'solvent'):
            from torchref.scaling.solvent_new import SolventModel
            # We need to instantiate SolventModel.
            # It requires: model, radius, k_solvent, b_solvent, etc.
            # We can use default values or try to extract from state_dict if they were saved there?
            # SolventModel state_dict contains parameters (k_solvent, b_solvent) but not config (radius).
            # However, we can instantiate with defaults and load_state_dict will overwrite parameters.
            # The critical part is 'model' which we have as self._model.module
            
            if hasattr(self, '_model') and self._model is not None:
                # Use defaults for config parameters as they are not in state_dict usually
                # Unless we modify SolventModel.state_dict to save them?
                # For now, assume defaults or that they don't matter for loading parameters
                # (except radius which affects grid size/mask, but mask is computed in forward?)
                # SolventModel computes masks in forward/update_mask?
                # Actually SolventModel has buffers like 'mask'.
                
                self.solvent = SolventModel(
                    model=self._model.module,
                    device=self.device,
                    verbose=self.verbose
                )
        
        # Load parent class state_dict (buffers and parameters)
        result = super().load_state_dict(state_dict, strict=strict)
        
        # Restore solvent model if state was saved
        if solvent_state is not None and hasattr(self, 'solvent'):
            self.solvent.load_state_dict(solvent_state)
        
        return result

    def save_state(self, path: str):
        """
        Save the complete state of the scaler to a file.
        
        Args:
            path (str): Path to save the state dictionary to.
        """
        torch.save(self.state_dict(), path)
        if self.verbose > 0:
            print(f"Saved scaler state to {path}")

    def load_state(self, path: str, strict: bool = True):
        """
        Load the complete state of the scaler from a file.
        
        Args:
            path (str): Path to load the state dictionary from.
            strict (bool): Whether to strictly enforce that keys match.
        """
        state_dict = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(state_dict, strict=strict)
        if self.verbose > 0:
            print(f"Loaded scaler state from {path}")
