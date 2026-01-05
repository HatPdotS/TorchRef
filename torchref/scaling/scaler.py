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
    """
    Scaler class to apply scaling and corrections to calculated structure factors.

    Supports two initialization patterns:

    1. Empty initialization (for state_dict loading)::

        >>> scaler = Scaler()  # Creates empty shell
        >>> scaler.load_state_dict(torch.load('scaler.pt'))

    2. Full initialization with model and data::

        >>> scaler = Scaler(model, reflection_data, nbins=20)
        >>> scaler.initialize()

    Parameters
    ----------
    model : Model, optional
        Model object for structure factor calculation.
    data : ReflectionData, optional
        ReflectionData object with observed data.
    nbins : int, default 20
        Number of resolution bins.
    verbose : int, default 1
        Verbosity level.
    device : torch.device, default torch.device('cpu')
        Computation device.

    Attributes
    ----------
    device : torch.device
        Current computation device.
    nbins : int
        Number of resolution bins.
    frozen : bool
        Whether the scaler parameters are frozen.
    """
    
    def __init__(self, model=None, data: ReflectionData = None, nbins: int = 20, verbose: int = 1, device=torch.device('cpu')):
        """
        Initialize Scaler.

        If model and data are provided, fully initializes the scaler.
        If not provided (empty init), creates a shell ready for load_state_dict().

        Parameters
        ----------
        model : Model, optional
            Model object for structure factor calculation.
        data : ReflectionData, optional
            ReflectionData object with observed data.
        nbins : int, default 20
            Number of resolution bins.
        verbose : int, default 1
            Verbosity level.
        device : torch.device, default torch.device('cpu')
            Computation device.
        """
        super(Scaler, self).__init__()
        self.device = device
        self.verbose = verbose
        self.nbins = nbins
        self.frozen = False
        
        # Empty initialization - just set up configuration
        if model is None or data is None:
            self._model = None
            self._data = None
            self.cell = None
            self.register_buffer('s', None)
            self.register_buffer('bins', None)
            return
        
        # Full initialization with model and data
        self.to(self.device)
        # Wrap model in ModuleReference to prevent registration as submodule
        self._model = ModuleReference(model)
        self._data = ModuleReference(data)

        self.cell = data.cell
        # Don't store hkl directly - always access it from data to avoid device mismatch
        self.register_buffer('s', get_scattering_vectors(data.hkl, self.cell))
        bins, self.nbins = self._data.get_bins(self.nbins)
        self.register_buffer('bins', bins)
        if self.verbose > 0:
            print(f"Initialized Scaler with {self.nbins} bins.")
    
    def set_model_and_data(self, model, data: ReflectionData):
        """
        Set model and data references after empty initialization.

        This is useful when loading from state_dict and then needing
        to reconnect to model/data objects.

        Parameters
        ----------
        model : Model
            Model object for structure factor calculation.
        data : ReflectionData
            ReflectionData object with observed data.
        """
        self._model = ModuleReference(model)
        self._data = ModuleReference(data)
        # Use data.cell for calculations
        if data.cell is not None:
            self.cell = data.cell
        if self.s is None and data.hkl is not None and data.cell is not None:
            self.register_buffer('s', get_scattering_vectors(data.hkl, data.cell))
        if self.bins is None and data.hkl is not None:
            bins, self.nbins = self._data.get_bins(self.nbins)
            self.register_buffer('bins', bins)

    def initialize(self):
        self.calc_initial_scale()
        self.setup_solvent()
        # self.setup_binwise_solvent_scale()
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

        Returns
        -------
        torch.nn.Parameter
            The log scale parameter for each resolution bin.
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
        
    def setup_binwise_solvent_scale(self):
        """
        Setup bin-wise solvent scaling (Phenix-style kmask per bin).

        This allows finer control over solvent contribution per resolution bin,
        which is more flexible than a single global B_sol parameter.
        """
        # Initialize k_mask per bin - starts at 0.35 for low res, decreases to 0 for high res
        # Phenix typically shows kmask ranging from ~0.35 at low res to 0 at high res
        # Initialize with a smooth decrease
        mean_res = self._data.mean_res_per_bin()
        
        # Initialize with exponential decay: k_mask = k_sol * exp(-B * s^2)
        # Use B=46 as starting point (Phenix-like)
        s_per_bin = 1.0 / (2.0 * mean_res + 1e-6)  # sin(theta)/lambda
        initial_kmask = 0.35 * torch.exp(-46.0 * s_per_bin**2)
        
        # Set high-res bins to 0 (where kmask < 0.05)
        initial_kmask = torch.where(initial_kmask < 0.05, 
                                    torch.zeros_like(initial_kmask), 
                                    initial_kmask)
        
        self.log_kmask = nn.Parameter(torch.log(initial_kmask.clamp(min=1e-6) + 1e-6).to(self.device))

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

        Parameters
        ----------
        device : torch.device, optional
            The target device. If None, uses the default CUDA device.
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

        Returns
        -------
        tuple
            R-work and R-free values.
        """
        hkl, fobs, _, rfree = self._data()
        fcalc = self._model(hkl)
        fcalc_scaled = self.forward(fcalc)
        return get_rfactors(torch.abs(fobs), torch.abs(fcalc_scaled), rfree)

    def bin_wise_rfactor(self, fcalc=None):
        """
        Calculate the bin-wise R-factor between observed and calculated structure factors.

        Parameters
        ----------
        fcalc : torch.Tensor, optional
            Calculated structure factors. If None, computed from model.

        Returns
        -------
        mean_res_per_bin : torch.Tensor
            Mean resolution per bin.
        rwork_per_bin : torch.Tensor
            R-work per bin.
        rfree_per_bin : torch.Tensor
            R-free per bin.
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

    def screen_solvent_params(self, steps=15, use_low_res_weighting=True, low_res_cutoff=5.0,
                               fit_on_low_res_only=True, low_res_limit=3.5):
        """
        Screen solvent parameters (k_sol, B_sol) using grid search.

        The bulk solvent contributes primarily at low resolution. Fitting on low-resolution
        reflections only (fit_on_low_res_only=True) prevents high-resolution reflections
        from dominating the optimization and pushing B_sol too low.

        Parameters
        ----------
        steps : int, default 15
            Number of grid points for each parameter.
        use_low_res_weighting : bool, default True
            If True, weight low-resolution reflections more heavily
            since solvent primarily contributes at low resolution.
        low_res_cutoff : float, default 5.0
            Resolution cutoff for weighting in Angstroms. Only used if
            use_low_res_weighting=True.
        fit_on_low_res_only : bool, default True
            If True, fit using only low-resolution reflections.
        low_res_limit : float, default 3.5
            Resolution limit for low-res only fitting in Angstroms.
        """
        hkl, fobs, sigma, rfree = self._data()
        fobs = fobs.to(torch.float32).detach()
        fcalc = self._model(hkl).detach()
        
        # Calculate resolution for weighting/filtering
        s = torch.norm(get_scattering_vectors(hkl, self.cell), dim=1)
        resolution = 1.0 / (s + 1e-6)  # in Angstroms
        
        # Create mask for low-resolution reflections
        if fit_on_low_res_only:
            low_res_mask = (resolution > low_res_limit) & rfree
            n_low_res = low_res_mask.sum().item()
            if self.verbose > 1:
                print(f"Solvent screening using {n_low_res} low-res reflections (>{low_res_limit}Å)")
            
            if n_low_res < 100:
                print(f"Warning: Only {n_low_res} low-res reflections, using all reflections instead")
                fit_on_low_res_only = False
        
        if not fit_on_low_res_only:
            low_res_mask = rfree  # Use all work reflections
        
        # Create weights for low-resolution preference (within the selected reflections)
        if use_low_res_weighting:
            # Smooth weighting: higher weight for low resolution
            weights = torch.exp(-s * low_res_cutoff).detach()
            weights = weights / weights[low_res_mask].sum()  # Normalize over selected reflections
            if self.verbose > 1:
                low_res_frac = (resolution > low_res_cutoff).float().mean()
                print(f"Low-resolution weighting: {low_res_frac*100:.1f}% reflections above {low_res_cutoff}Å")
        else:
            weights = torch.ones_like(fobs)
            weights = weights / weights[low_res_mask].sum()
        
        best_log_k_solvent = self.solvent.log_k_solvent.clone()
        best_b_solvent = self.solvent.b_solvent.clone()
        best_loss = float('inf')
        
        # Grid search ranges - k_sol from 0.1 to 0.6, B_sol from 30 to 100
        # Phenix typically finds k_sol ~0.35, B_sol ~46
        ksol_start = torch.log(torch.tensor(0.1, device=self.device))
        ksol_end = torch.log(torch.tensor(0.6, device=self.device))
        
        for log_k_solvent in torch.linspace(ksol_start, ksol_end, steps=steps, device=self.device):
            for b_solvent in torch.linspace(30.0, 100.0, steps=steps, device=self.device):
                self.solvent.log_k_solvent.data = log_k_solvent.to(dtype=self.solvent.log_k_solvent.dtype)
                self.solvent.b_solvent.data = b_solvent.to(dtype=self.solvent.b_solvent.dtype)
                
                scaled_fcalc = self.forward(fcalc)
                
                # Compute loss only on selected reflections
                diff = fobs[low_res_mask] - torch.abs(scaled_fcalc[low_res_mask])
                # Handle MaskedTensor: extract valid data for median calculation
                sigma_subset = sigma[low_res_mask]
                if hasattr(sigma_subset, 'get_mask'):
                    sigma_data = sigma_subset.get_data()[sigma_subset.get_mask()]
                    eps = torch.median(sigma_data).item() * 1e-1
                else:
                    eps = torch.median(sigma_subset).item() * 1e-1
                sigma_safe = torch.clamp(sigma_subset, min=eps)
                nll_per_refl = 0.5 * (diff**2) / (sigma_safe**2)
                
                if use_low_res_weighting:
                    # Weighted mean: emphasize lower resolution within the selection
                    nll_loss = (nll_per_refl * weights[low_res_mask]).sum()
                else:
                    nll_loss = nll_per_refl.mean()
                
                if nll_loss.item() < best_loss:
                    best_loss = nll_loss.item()
                    best_log_k_solvent = log_k_solvent.clone()
                    best_b_solvent = b_solvent.clone()
        
        self.solvent.log_k_solvent.data = best_log_k_solvent.to(dtype=self.solvent.log_k_solvent.dtype)
        self.solvent.b_solvent.data = best_b_solvent.to(dtype=self.solvent.b_solvent.dtype)
        
        if self.verbose > 0:
            k_sol = torch.exp(best_log_k_solvent).item()
            print(f"Optimal solvent parameters found: k_sol={k_sol:.4f}, B_sol={best_b_solvent.item():.1f}, "
                  f"NLL Loss={best_loss:.4f}")

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

        Parameters
        ----------
        nsteps : int, default 3
            Number of LBFGS steps.
        lr : float, default 1.0
            Learning rate (typically 1.0 for LBFGS).
        max_iter : int, default 200
            Maximum iterations per line search.
        history_size : int, default 10
            Number of previous gradients to store for Hessian approximation.
        verbose : bool, default True
            Print progress information.

        Returns
        -------
        dict
            Dictionary with refinement metrics including steps, xray_work,
            xray_test, rwork, rfree.

        Examples
        --------
        >>> scaler.unfreeze()
        >>> metrics = scaler.refine_lbfgs(nsteps=5, verbose=True)
        >>> scaler.freeze()
        """
        # Ensure scaler is unfrozen
        was_frozen = self.frozen
        if was_frozen:
            self.unfreeze()
        
        # Create LBFGS optimizer for scaler parameters only
        optimizer = torch.optim.LBFGS(
            self.parameters(),
            lr=lr,
            max_iter=max_iter,
            history_size=history_size,
            line_search_fn='strong_wolfe'

        )
        
        def closure():
            optimizer.zero_grad()
            try: fcalc_scaled = self.forward(fcalc)
            except RuntimeError as e:
                if self.verbose > 0:
                    import warnings
                    warnings.warn("Non-finite loss encountered during scale optimization. "
                                "LBFGS line search will reject this step and try a smaller step size.",
                                RuntimeWarning)
                return torch.tensor(1e10, device=loss.device, dtype=loss.dtype, requires_grad=True)
            
            U_penalty = torch.sum(self.U ** 2)
            loss = nll_xray(fobs, fcalc_scaled, sigma) + U_penalty
            
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

        Parameters
        ----------
        fcalc : torch.Tensor
            Calculated structure factors. Expected shape (N,), an additional
            dimension for batch is possible. N should match the full HKL size.
        use_mask : bool, default True
            Deprecated parameter, kept for backward compatibility. When using
            MaskedTensors, masking is handled in loss functions, not here.

        Returns
        -------
        torch.Tensor
            Scaled structure factors of same shape as input.
        """
        batched = True

        if fcalc.ndim == 1:
            fcalc = fcalc.unsqueeze(0)  # Add batch dimension if missing
            batched = False

        # Determine if we should mask internally or work with full arrays
        # When fcalc matches full HKL size, work with full arrays (MaskedTensor mode)
        # When fcalc is already filtered, apply mask to internal arrays
        n_full = len(self.bins)
        n_fcalc = fcalc.shape[1]
        
        if n_fcalc == n_full:
            # Full-size mode (MaskedTensor): don't apply mask to internal arrays
            apply_internal_mask = False
        else:
            # Filtered mode: apply mask to internal arrays to match fcalc size
            apply_internal_mask = True
            mask = self._data.masks().to(torch.bool)

        if hasattr(self, 'U'):
            anisotropy_factors = self.anisotropy_correction()
            aniso_correction = anisotropy_factors[mask] if apply_internal_mask else anisotropy_factors
        else:
            aniso_correction = torch.tensor(1.0, device=self.device, dtype=fcalc.dtype)

        if hasattr(self, 'solvent'):
            # Check if using bin-wise kmask (Phenix-style) or global k_sol/B_sol
            if hasattr(self, 'log_kmask'):
                # Bin-wise scaling: get raw F_mask and apply per-bin kmask
                f_mask = self.solvent.get_rec_solvent(self.hkl)
                f_mask = f_mask[mask] if apply_internal_mask else f_mask
                
                # Apply bin-wise kmask (like Phenix kmask)
                kmask = torch.exp(self.log_kmask.clamp(min=-10.0, max=10.0))
                # Clamp kmask to reasonable values
                kmask = torch.clamp(kmask, min=0.0, max=10.0)
                # Expand to per-reflection using bin indices
                bins_to_use = self.bins[mask] if apply_internal_mask else self.bins
                kmask_per_refl = kmask[bins_to_use]
                f_sol = kmask_per_refl * f_mask
            else:
                # Original: global k_sol and B_sol applied in solvent model
                f_sol = self.solvent(self.hkl)
                f_sol = f_sol[mask] if apply_internal_mask else f_sol
        else:
            f_sol = torch.tensor(0.0, device=self.device, dtype=fcalc.dtype)

        if hasattr(self, 'log_scale'):
            bins_to_use = self.bins[mask] if apply_internal_mask else self.bins
            K_overall = torch.exp(self.log_scale[bins_to_use].clamp(min=-10.0, max=10.0))
        else:
            K_overall = torch.tensor(1.0, device=self.device, dtype=fcalc.dtype)
        
        if hasattr(self, 'bin_wise_bfactor'):
            bfactor_factors = self.bin_wise_bfactor_correction()
            b_overall = bfactor_factors[mask] if apply_internal_mask else bfactor_factors
        else:
            b_overall = torch.tensor(1.0, device=self.device, dtype=fcalc.dtype)
        
        fcalc = K_overall.unsqueeze(0) * b_overall.unsqueeze(0) * (aniso_correction.unsqueeze(0) * fcalc + f_sol.unsqueeze(0))

        if not batched:
            fcalc = fcalc.squeeze(0)  # Remove batch dimension if it was added

        return fcalc

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """
        Return a dictionary containing the complete state of the Scaler.

        This includes:

        - All registered buffers and parameters (via parent class)
        - Scaler-specific metadata (nbins, frozen state, etc.)
        - Solvent model state (if initialized)

        Note: Model and data references are NOT saved (managed separately).

        Parameters
        ----------
        destination : dict, optional
            Optional dict to populate.
        prefix : str, default ''
            Prefix for parameter names.
        keep_vars : bool, default False
            Whether to keep variables in computational graph.

        Returns
        -------
        dict
            Complete state dictionary.
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
        Load the Scaler state from a dictionary.

        Note: This assumes model and data are already set via __init__ or assignment.

        Parameters
        ----------
        state_dict : dict
            Dictionary containing scaler state.
        strict : bool, default True
            Whether to strictly enforce that keys match.
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

        Parameters
        ----------
        path : str
            Path to save the state dictionary to.
        """
        torch.save(self.state_dict(), path)
        if self.verbose > 0:
            print(f"Saved scaler state to {path}")

    def load_state(self, path: str, strict: bool = True):
        """
        Load the complete state of the scaler from a file.

        Parameters
        ----------
        path : str
            Path to load the state dictionary from.
        strict : bool, default True
            Whether to strictly enforce that keys match.
        """
        state_dict = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(state_dict, strict=strict)
        if self.verbose > 0:
            print(f"Loaded scaler state from {path}")
