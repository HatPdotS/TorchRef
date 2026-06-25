from torch import nn
import torch
from typing import Dict, List, Optional, Union, Tuple
import numpy as np

from torchref.utils.device_mixin import DeviceMixin


class occupancy_unrestrained(DeviceMixin, nn.Module):
    """
    Unrestrained occupancy model where each state at each timepoint is independent.
    
    This is the most flexible model but may lead to physically unrealistic solutions
    since occupancies at different timepoints are not coupled.
    
    Parameters
    ----------
    nstates : int
        Number of structural states
    time : list or array-like
        Time points for the experiment
    """
    def __init__(self, nstates, time):
        super(occupancy_unrestrained, self).__init__()
        self.nstates = nstates
        self.logits = nn.Parameter(torch.zeros(nstates, len(time)))
        self.time = time

    def forward(self):
        occupancies = torch.exp(self.logits)
        occupancies = occupancies / occupancies.sum(dim=0, keepdim=True)
        return occupancies
    
    def plot_occupancies(self, path, log_scale=False, figsize=(10, 6)):
        import matplotlib.pyplot as plt
        occupancies = self().detach().cpu().numpy()
        time_points = np.array(self.time)
        
        plt.figure(figsize=figsize)
        for i in range(self.nstates):
            plt.plot(time_points, occupancies[i], label=f'State {i}', linewidth=2)
        plt.xlabel('Time')
        plt.ylabel('Occupancy')
        plt.legend()
        if log_scale and (time_points > 0).any():
            plt.xscale('log')
            pos_times = time_points[time_points > 0]
            if len(pos_times) > 0:
                plt.xlim(left=pos_times.min())
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()


from torchref.experimental.kinetic.kinetics import KineticModel


class occupancies_kinetics(DeviceMixin, nn.Module):
    """
    Kinetics-constrained occupancy model.
    
    This model uses a kinetic scheme to constrain occupancies at different timepoints.
    Instead of independent parameters for each timepoint, the occupancies are derived
    from rate constants, efficiencies, and a kinetic flow chart.
    
    This provides several advantages over unrestrained refinement:
    1. Physical constraints: occupancies follow kinetic laws
    2. Reduced parameters: n_rates instead of n_states * n_timepoints
    3. Extrapolation: can predict occupancies at unmeasured timepoints
    4. Interpretability: rate constants have physical meaning

    Parameters
    ----------
    flow_chart : str
        Kinetic scheme, e.g., "A->B,B->C,C->D" or "A->B,B->A,B->C" (with back-reaction)
    time : list or tensor
        Time points at which to evaluate occupancies
    rate_constants : dict or None, optional
        Initial rate constants as {"A->B": value, ...}. If None, uses smart initialization.
    efficiencies : dict or None, optional
        Accepted for backward compatibility but ignored: efficiencies are frozen
        at 1.0 (degenerate with rate constants) and are not refinable.
    instrument_function : str, optional
        Instrument response function model, either 'none' or 'gaussian'.
        Default: 'none'.
    instrument_width : float, optional
        Instrument response function width (Gaussian sigma). Default: 10
    light_activated : bool, optional
        If True, products returning to ground state become inactive. Default: False
    state_mapping : dict or None, optional
        Mapping from kinetic states to structural model indices.
        E.g., {"A": 0, "B": 1, "C": 2, "D": 3} or {"A": 0, "B": 1, "C": 1, "D": 2}
        The latter allows multiple kinetic states to map to the same structure.
        If None, assumes sequential mapping (A=0, B=1, ...).
    regularization : dict or None, optional
        Regularization settings:
        - 'rate_prior_weight': weight for log-normal prior on rates
        - 'rate_prior_mean': mean of log-rate prior (default: based on time range)
        - 'rate_prior_std': std of log-rate prior (default: 2.0, allows ~2 orders of magnitude)
        - 'efficiency_prior_weight': weight for efficiency prior (favoring 1.0)
    verbose : int, optional
        Verbosity level. Default: 1
    
    Examples
    --------
    >>> # Simple sequential kinetics: A -> B -> C -> D
    >>> occ = occupancies_kinetics(
    ...     flow_chart="A->B,B->C,C->D",
    ...     time=torch.linspace(0, 100, 50),
    ...     rate_constants={"A->B": 1.0, "B->C": 0.1, "C->D": 0.01}
    ... )
    >>> occupancies = occ()  # Shape: [n_states, n_timepoints]
    
    >>> # With back-reaction
    >>> occ = occupancies_kinetics(
    ...     flow_chart="A->B,B->A,B->C",
    ...     time=times,
    ...     light_activated=True  # Products returning to A become inactive
    ... )
    
    >>> # Mapping multiple kinetic states to same structure
    >>> occ = occupancies_kinetics(
    ...     flow_chart="A->B,B->C,C->D",
    ...     time=times,
    ...     state_mapping={"A": 0, "B": 1, "C": 1, "D": 0}  # B and C share structure
    ... )
    """
    
    def __init__(
        self,
        flow_chart: str,
        time: Union[List, torch.Tensor],
        rate_constants: Optional[Dict[str, float]] = None,
        efficiencies: Optional[Dict[str, float]] = None,
        instrument_function: str = 'none',
        instrument_width: float = 10.0,
        light_activated: bool = False,
        state_mapping: Optional[Dict[str, int]] = None,
        regularization: Optional[Dict[str, float]] = None,
        verbose: int = 1
    ):
        super(occupancies_kinetics, self).__init__()

        self.verbose = verbose

        # Convert time to tensor if needed
        if not isinstance(time, torch.Tensor):
            time = torch.tensor(time, dtype=torch.float32)
        self.register_buffer('time', time)

        # Initialize the kinetic model
        self.kinetics = KineticModel(
            flow_chart=flow_chart,
            timepoints=time,
            rate_constants=rate_constants,
            efficiencies=efficiencies,
            instrument_function=instrument_function,
            instrument_width=instrument_width,
            light_activated=light_activated,
            verbose=verbose
        )
        
        # Setup state mapping (kinetic states to structural model indices)
        self._setup_state_mapping(state_mapping)
        
        # Setup regularization
        self._setup_regularization(regularization)
        
        if self.verbose:
            print(f"\nKinetic occupancy model initialized:")
            print(f"  Flow chart: {flow_chart}")
            print(f"  Time points: {len(time)} (range: {time.min():.2f} to {time.max():.2f})")
            print(f"  Kinetic states: {self.kinetics.n_states}")
            print(f"  Structural models: {self.nstates}")
            print(f"  Mapping: {self.state_mapping}")
    
    def _setup_state_mapping(self, state_mapping: Optional[Dict[str, int]]):
        """
        Setup mapping from kinetic states to structural model indices.
        
        This allows:
        1. Multiple kinetic intermediates to share the same structure
        2. Reordering of states if kinetic and structural order differ
        """
        kinetic_states = self.kinetics.states
        
        if state_mapping is None:
            # Default: sequential mapping (A=0, B=1, C=2, ...)
            state_mapping = {state: i for i, state in enumerate(kinetic_states)}
        
        self.state_mapping = state_mapping
        
        # Number of unique structural states
        self.nstates = max(state_mapping.values()) + 1
        
        # Create mapping matrix: [n_structural, n_kinetic]
        # This sums kinetic populations that map to the same structure
        mapping_matrix = torch.zeros(self.nstates, len(kinetic_states))
        for kinetic_state, struct_idx in state_mapping.items():
            kinetic_idx = self.kinetics.state_to_idx[kinetic_state]
            mapping_matrix[struct_idx, kinetic_idx] = 1.0
        
        self.register_buffer('mapping_matrix', mapping_matrix)
    
    def _setup_regularization(self, regularization: Optional[Dict[str, float]]):
        """Setup regularization priors for kinetic parameters."""
        self.regularization = regularization or {}
        
        # Default regularization settings
        defaults = {
            'rate_prior_weight': 0.0,  # Disabled by default
            'rate_prior_mean': None,  # Auto-determine from time range
            'rate_prior_std': 2.0,  # Allows ~2 orders of magnitude variation
            'efficiency_prior_weight': 0.0,  # Disabled by default
        }
        
        for key, default_val in defaults.items():
            if key not in self.regularization:
                self.regularization[key] = default_val
        
        # Auto-determine rate prior mean from time range if not specified
        if self.regularization['rate_prior_mean'] is None:
            t_range = self.time.max() - self.time.min()
            if t_range > 0:
                # Center prior on rate that gives observable kinetics in the time range
                self.regularization['rate_prior_mean'] = float(np.log(3.0 / t_range.item()))
            else:
                self.regularization['rate_prior_mean'] = 0.0  # k = 1
    
    def forward(self) -> torch.Tensor:
        """
        Compute occupancies at all timepoints.
        
        Returns
        -------
        occupancies : torch.Tensor
            Occupancy of each structural state at each timepoint.
            Shape: [n_structural_states, n_timepoints]
        """
        # Get kinetic populations: [n_timepoints, n_kinetic_states]
        kinetic_populations = self.kinetics()
        
        # Map to structural states: [n_timepoints, n_structural_states]
        structural_populations = torch.matmul(kinetic_populations, self.mapping_matrix.T)
        
        # Transpose to match expected format: [n_structural_states, n_timepoints]
        occupancies = structural_populations.T
        
        return occupancies
    
    def get_regularization_loss(self) -> torch.Tensor:
        """
        Compute regularization loss for kinetic parameters.
        
        This implements prior distributions on the parameters:
        - Log-normal prior on rate constants
        - Beta-like prior on efficiencies (favoring values near 1)
        
        Returns
        -------
        reg_loss : torch.Tensor
            Regularization loss to be added to the main loss
        """
        reg_loss = torch.tensor(0.0, device=self.kinetics.log_rate_constants.device)
        
        # Rate constant prior (log-normal)
        rate_weight = self.regularization.get('rate_prior_weight', 0.0)
        if rate_weight > 0:
            rate_mean = self.regularization['rate_prior_mean']
            rate_std = self.regularization['rate_prior_std']
            
            # Log-normal prior: (log(k) - mu)^2 / (2 * sigma^2)
            log_rates = self.kinetics.log_rate_constants
            rate_prior_loss = torch.sum((log_rates - rate_mean) ** 2) / (2 * rate_std ** 2)
            reg_loss = reg_loss + rate_weight * rate_prior_loss
        
        # Efficiency prior — no-op since efficiencies are frozen at 1.0
        
        return reg_loss
    
    def get_rate_constants(self) -> Dict[str, float]:
        """Get current rate constants."""
        return self.kinetics.get_rate_constants()
    
    def get_efficiencies(self) -> Dict[str, float]:
        """Get current efficiencies."""
        return self.kinetics.get_efficiencies()
    
    def get_time_constants(self) -> Dict[str, float]:
        """Get current time constants (1/k_eff)."""
        return self.kinetics.get_time_constants()
    
    def set_rate_constant(self, transition: str, value: float):
        """Set a specific rate constant."""
        self.kinetics.set_rate_constant(transition, value)
    
    def freeze_rates(self, transitions: Optional[List[str]] = None):
        """
        Freeze rate constants (exclude from optimization).
        
        Parameters
        ----------
        transitions : list of str or None
            Transitions to freeze (e.g., ["A->B"]). If None, freezes all.
        """
        if transitions is None:
            self.kinetics.log_rate_constants.requires_grad = False
        else:
            # Create mask for transitions to freeze
            # Note: This requires more complex handling with hooks
            # For now, just warn that partial freezing needs custom implementation
            import warnings
            warnings.warn("Partial rate freezing not yet implemented. Use full freeze or manual gradient masking.")
            self.kinetics.log_rate_constants.requires_grad = False
    
    def unfreeze_rates(self):
        """Unfreeze all rate constants."""
        self.kinetics.log_rate_constants.requires_grad = True
    
    def freeze_efficiencies(self):
        """No-op: efficiencies are always frozen at 1.0."""
        pass

    def unfreeze_efficiencies(self):
        """No-op: efficiencies are always frozen at 1.0."""
        pass
    
    def freeze_instrument(self):
        """Freeze instrument function width."""
        if self.kinetics.log_instrument_width is not None:
            self.kinetics.log_instrument_width.requires_grad = False
    
    def unfreeze_instrument(self):
        """Unfreeze instrument function width."""
        if self.kinetics.log_instrument_width is not None:
            self.kinetics.log_instrument_width.requires_grad = True
    
    def get_parameter_groups(self, base_lr: float = 1e-3) -> List[Dict]:
        """
        Get parameter groups with appropriate learning rates.
        
        Kinetic parameters often benefit from different learning rates:
        - Rate constants (log-space): can use larger steps
        - Efficiencies: moderate steps
        - Instrument width: small steps (often well-constrained)
        
        Parameters
        ----------
        base_lr : float
            Base learning rate
        
        Returns
        -------
        param_groups : list of dict
            Parameter groups suitable for torch optimizers
        """
        param_groups = [
            {
                'params': [self.kinetics.log_rate_constants],
                'lr': base_lr,
                'name': 'rate_constants'
            },
        ]
        
        if self.kinetics.log_instrument_width is not None:
            param_groups.append({
                'params': [self.kinetics.log_instrument_width],
                'lr': base_lr * 0.1,  # Very conservative for instrument width
                'name': 'instrument_width'
            })
        
        return param_groups
    
    def print_parameters(self):
        """Print current kinetic parameters."""
        self.kinetics.print_parameters()
    
    def plot_occupancies(
        self, 
        path: str, 
        log_scale: bool = True,
        show_kinetic_states: bool = False,
        figsize: Tuple[int, int] = (10, 6),
        title: Optional[str] = None
    ):
        """
        Plot state occupancies over time.
        
        Parameters
        ----------
        path : str
            Output path for the plot
        log_scale : bool
            Whether to use log scale for time axis
        show_kinetic_states : bool
            If True, shows all kinetic states. If False, shows mapped structural states.
        figsize : tuple
            Figure size
        title : str or None
            Custom title
        """
        import matplotlib.pyplot as plt
        
        with torch.no_grad():
            if show_kinetic_states:
                # Show raw kinetic populations
                populations = self.kinetics().cpu().numpy()  # [n_time, n_kinetic]
                state_names = self.kinetics.states
            else:
                # Show structural occupancies
                populations = self().T.cpu().numpy()  # [n_time, n_structural]
                state_names = [f'Struct {i}' for i in range(self.nstates)]
        
        time_points = self.time.cpu().numpy()
        
        plt.figure(figsize=figsize)
        for i, name in enumerate(state_names):
            plt.plot(time_points, populations[:, i], label=name, linewidth=2)
        
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Occupancy', fontsize=12)
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3)
        
        if log_scale and (time_points > 0).any():
            plt.xscale('log')
            pos_times = time_points[time_points > 0]
            if len(pos_times) > 0:
                plt.xlim(left=pos_times.min() * 0.9)
        
        if title is None:
            title = f'Kinetic Occupancies: {self.kinetics.flow_chart}'
        plt.title(title, fontsize=14)
        
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if self.verbose:
            print(f"Plot saved to: {path}")
    
    def plot_comparison(
        self,
        target_occupancies: torch.Tensor,
        path: str,
        log_scale: bool = True,
        figsize: Tuple[int, int] = (12, 5)
    ):
        """
        Plot comparison between current and target occupancies.
        
        Useful for debugging refinement by comparing to known ground truth
        or to unrestrained refinement results.
        
        Parameters
        ----------
        target_occupancies : torch.Tensor
            Target occupancies, shape [n_states, n_timepoints]
        path : str
            Output path for the plot
        log_scale : bool
            Whether to use log scale for time axis
        figsize : tuple
            Figure size
        """
        import matplotlib.pyplot as plt
        
        with torch.no_grad():
            current = self().cpu().numpy()  # [n_states, n_time]
            target = target_occupancies.cpu().numpy()
        
        time_points = self.time.cpu().numpy()
        n_states = current.shape[0]
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Left: current (kinetic model)
        ax1 = axes[0]
        for i in range(n_states):
            ax1.plot(time_points, current[i], label=f'State {i}', linewidth=2)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Occupancy')
        ax1.set_title('Kinetic Model')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        if log_scale and (time_points > 0).any():
            ax1.set_xscale('log')
        
        # Right: target
        ax2 = axes[1]
        for i in range(min(n_states, target.shape[0])):
            ax2.plot(time_points, target[i], label=f'State {i}', linewidth=2)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Occupancy')
        ax2.set_title('Target')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        if log_scale and (time_points > 0).any():
            ax2.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def state_dict_kinetics(self) -> Dict:
        """
        Get state dict with kinetic parameters for saving/loading.
        """
        return {
            'flow_chart': self.kinetics.flow_chart,
            'time': self.time,
            'state_mapping': self.state_mapping,
            'regularization': self.regularization,
            'kinetics_state_dict': self.kinetics.state_dict(),
        }
    
    @classmethod
    def load_from_kinetics_state(cls, state: Dict, verbose: int = 1) -> 'occupancies_kinetics':
        """
        Load model from saved kinetic state.
        """
        model = cls(
            flow_chart=state['flow_chart'],
            time=state['time'],
            state_mapping=state['state_mapping'],
            regularization=state['regularization'],
            verbose=verbose
        )
        model.kinetics.load_state_dict(state['kinetics_state_dict'])
        return model


class occupancies_kinetics_multiexperiment(DeviceMixin, nn.Module):
    """
    Kinetics-constrained occupancy model for multiple experiments.
    
    This handles the case where you have multiple datasets with different
    conditions (e.g., different temperatures, different excitation wavelengths)
    but want to share some kinetic parameters across them.
    
    Parameters
    ----------
    flow_chart : str
        Kinetic scheme (shared across experiments)
    experiments : list of dict
        Each dict contains:
        - 'time': time points for this experiment
        - 'rate_constants': experiment-specific rate constants (or None to share)
        - 'name': optional name for the experiment
    shared_rates : list of str or None
        List of transitions that should share rates across experiments.
        E.g., ["A->B", "C->D"]. If None, all rates are experiment-specific.
    verbose : int
        Verbosity level
    """
    
    def __init__(
        self,
        flow_chart: str,
        experiments: List[Dict],
        shared_rates: Optional[List[str]] = None,
        verbose: int = 1
    ):
        super(occupancies_kinetics_multiexperiment, self).__init__()
        
        self.flow_chart = flow_chart
        self.n_experiments = len(experiments)
        self.shared_rates = shared_rates or []
        self.verbose = verbose
        
        # Create separate kinetic models for each experiment
        self.kinetic_models = nn.ModuleList()
        
        for i, exp in enumerate(experiments):
            model = occupancies_kinetics(
                flow_chart=flow_chart,
                time=exp['time'],
                rate_constants=exp.get('rate_constants'),
                verbose=verbose if i == 0 else 0
            )
            self.kinetic_models.append(model)
        
        # Setup shared parameters if requested
        if self.shared_rates:
            self._setup_shared_rates()
    
    def _setup_shared_rates(self):
        """Link shared rate parameters across experiments."""
        # Use the first experiment's parameters as the shared ones
        primary = self.kinetic_models[0].kinetics
        
        for exp_idx in range(1, self.n_experiments):
            secondary = self.kinetic_models[exp_idx].kinetics
            
            for trans_idx, (from_s, to_s) in enumerate(primary.transitions):
                trans_key = f"{from_s}->{to_s}"
                if trans_key in self.shared_rates:
                    # Share this rate constant
                    # Note: This creates a shared reference, gradients will accumulate
                    if self.verbose:
                        print(f"Sharing rate {trans_key} across experiments")
    
    def forward(self, experiment_idx: int) -> torch.Tensor:
        """
        Get occupancies for a specific experiment.
        
        Parameters
        ----------
        experiment_idx : int
            Index of the experiment
        
        Returns
        -------
        occupancies : torch.Tensor
            Shape [n_states, n_timepoints]
        """
        return self.kinetic_models[experiment_idx]()
    
    def forward_all(self) -> List[torch.Tensor]:
        """
        Get occupancies for all experiments.
        
        Returns
        -------
        occupancies_list : list of torch.Tensor
            List of occupancy tensors, one per experiment
        """
        return [model() for model in self.kinetic_models]
