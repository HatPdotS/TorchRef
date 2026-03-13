import torch
from torch.nn import Module as nnModule
from torch.nn import Parameter
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
import numpy as np


class KineticModel(nnModule):
    """
    Configurable PyTorch module for fitting kinetic behavior.
    
    Supports arbitrary kinetic schemes defined by relational strings:
    - "A->B,B->C" (sequential)
    - "A->B,B->A,B->C" (with back reactions)
    - "A->B,A->C,B->D,C->D" (parallel pathways)
    - "A->B,B->C,D" (D is non-reactive state)
    
    Each transition has TWO parameters:
    - Reactivity constant k (rate)
    - Reaction efficiency η (0 to 1, controls maximum conversion)
    
    States can have baseline occupancy offsets (default: 0, not refined).
    
    The initial transfer is driven by photoabsorption with quasi-instant
    conversion around zero, with spread accounted for by an instrument function.
    
    Parameters
    ----------
    flow_chart : str
        Relational string describing the kinetic scheme using comma-separated transitions.
        Standalone states (non-reactive) can be included without transitions.
        Example: "A->B,B->C" or "A->B,B->A,B->C,D" (D is non-reactive)
    timepoints : torch.Tensor or array-like
        Time points at which to evaluate the kinetics
    rate_constants : dict or list, optional
        Initial rate constants. Can be:
        - Dict mapping "A->B" to float value
        - List of floats (same order as transitions in flow_chart)
        - None (random initialization)
    efficiencies : dict or list, optional
        Initial reaction efficiencies (0-1). Same format as rate_constants.
        Default: all 1.0 (100% efficient)
    instrument_function : str, optional
        Type of instrument response function. Options: 'gaussian', 'none'
        Default: 'gaussian'
    instrument_width : float, optional
        Width parameter for the instrument function (e.g., sigma for gaussian)
        Default: 0.1
    initial_state : str, optional
        Which state starts with population 1. Default: first state in flow chart
    light_activated : bool, optional
        If True, treats this as a light-activated reaction where the initial
        photoexcitation can only happen once. Products returning to the initial
        state become inactive (A*) and cannot undergo photoactivation again.
        Default: False
    verbose : int, optional
        Verbosity level. Default: 1
    """
    
    def __init__(
        self, 
        flow_chart: str,
        timepoints,
        rate_constants: Optional[Union[Dict[str, float], List[float]]] = None,
        efficiencies: Optional[Union[Dict[str, float], List[float]]] = None,
        instrument_function: str = 'gaussian',
        instrument_width: float = 10,
        initial_state: Optional[str] = None,
        light_activated: bool = False,
        activation_level: Optional[float] = 0.5,
        verbose: int = 1,

    ):
        super(KineticModel, self).__init__()
        
        self.flow_chart = flow_chart
        self.verbose = verbose
        
        # Convert timepoints to tensor
        if not isinstance(timepoints, torch.Tensor):
            timepoints = torch.tensor(timepoints, dtype=torch.float32)
        self.register_buffer('timepoints', timepoints)
        
        # Parse flow chart to extract states and transitions
        self.states, self.transitions = self._parse_flow_chart(flow_chart)
        self.n_states = len(self.states)
        self.state_to_idx = {state: idx for idx, state in enumerate(self.states)}
        self.n_transitions = len(self.transitions)
        
        # Handle light-activated reactions
        self.light_activated = light_activated
        if light_activated:
            # Identify initial state
            if initial_state is None:
                initial_state = self.states[0]
            
            # Create an inactive version of the initial state (e.g., A -> A*)
            inactive_state = initial_state + '*'
            
            # Add A* as a new state
            self.states.append(inactive_state)
            self.n_states += 1
            self.state_to_idx[inactive_state] = len(self.states) - 1
            
            # Modify transitions: anything that returns to initial_state now goes to inactive_state
            modified_transitions = []
            self._light_activated_remapping = {}  # Track redirected transitions
            for from_state, to_state in self.transitions:
                if to_state == initial_state and from_state != initial_state:
                    # Redirect back-reactions to A* instead of A
                    modified_transitions.append((from_state, inactive_state))
                    # Remember the mapping for rate constant initialization
                    old_key = f"{from_state}->{initial_state}"
                    new_key = f"{from_state}->{inactive_state}"
                    self._light_activated_remapping[new_key] = old_key
                else:
                    modified_transitions.append((from_state, to_state))
            
            self.transitions = modified_transitions
            self.n_transitions = len(self.transitions)
            
            if self.verbose:
                print(f"Light-activated mode: {initial_state} products → {inactive_state} (cannot re-photoactivate)")
        else:
            self._light_activated_remapping = {}
        
        if self.verbose:
            print(f"Identified {self.n_states} states: {self.states}")
            print(f"Identified {self.n_transitions} transitions: {self.transitions}")
        
        # Store instrument width for smart initialization
        self._instrument_width = instrument_width
        
        # Initialize rate constants (k) as learnable parameters with smart defaults
        init_log_k = self._initialize_rate_constants(rate_constants, instrument_width, timepoints)
        self.log_rate_constants = Parameter(init_log_k)
        
        # Efficiencies are frozen at 1.0 (not refinable).
        # They are degenerate with rate constants for sequential models.
        self.register_buffer(
            'efficiencies',
            torch.ones(self.n_transitions),
        )
        
        # Initial population
        if initial_state is None:
            initial_state = self.states[0]
        self.initial_state = initial_state
        initial_populations = torch.zeros(self.n_states)
        initial_populations[self.state_to_idx[initial_state]] = 1.0
        self.register_buffer('initial_populations', initial_populations)
        
        # Instrument function parameters (now refinable)
        self.instrument_function = instrument_function
        if instrument_function == 'gaussian':
            # Store log of width to ensure positivity (refinable)
            self.log_instrument_width = Parameter(
                torch.tensor(np.log(instrument_width), dtype=torch.float32)
            )
        elif instrument_function == 'none':
            self.log_instrument_width = None
        else:
            raise ValueError(f"Unknown instrument function: {instrument_function}")
        
        # Baseline occupancy offsets (default: all zeros, not refined)
        # These are constant offsets added to the populations
        # Smart initialization: initial state (A) has 50% baseline (unreactive fraction)
        self.baseline_occupancies = torch.zeros(self.n_states)
        initial_state_idx = self.state_to_idx[initial_state]
        self.baseline_occupancies[initial_state_idx] = 1-activation_level
        self.register_buffer('_baseline_occupancies', self.baseline_occupancies)
        self._baseline_refinable = {}  # Track which baselines are refinable
        
        if self.verbose:
            print(f"Baseline initialization: State {initial_state} = {1-activation_level} ({activation_level*100}% reactive)")
    
    def _initialize_parameter(
        self, 
        values: Optional[Union[Dict[str, float], List[float]]], 
        default_value: float = 1.0,
        transform: str = 'log'
    ) -> torch.Tensor:
        """
        Initialize a parameter from various input formats.
        
        Parameters
        ----------
        values : dict, list, or None
            Initial values
        default_value : float
            Default value if values is None
        transform : str
            'log' for log-transformation, 'none' for no transformation
        
        Returns
        -------
        tensor : torch.Tensor
            Initialized parameter tensor
        """
        init_values = torch.ones(self.n_transitions) * default_value
        
        if values is not None:
            if isinstance(values, dict):
                # Dictionary mapping "A->B" to value
                for idx, (from_state, to_state) in enumerate(self.transitions):
                    key = f"{from_state}->{to_state}"
                    if key in values:
                        init_values[idx] = values[key]
                    elif key in self._light_activated_remapping:
                        # Check if this transition was redirected (e.g., O->A* was O->A)
                        original_key = self._light_activated_remapping[key]
                        if original_key in values:
                            init_values[idx] = values[original_key]
            elif isinstance(values, (list, tuple)):
                # List of values in order
                if len(values) != self.n_transitions:
                    raise ValueError(f"Expected {self.n_transitions} values, got {len(values)}")
                init_values = torch.tensor(values, dtype=torch.float32)
            else:
                raise ValueError("values must be dict, list, or None")
        
        # Apply transformation
        if transform == 'log':
            return torch.log(init_values)
        else:
            return init_values
    
    def _initialize_rate_constants(
        self,
        rate_constants: Optional[Union[Dict[str, float], List[float]]],
        instrument_width: float,
        timepoints: torch.Tensor
    ) -> torch.Tensor:
        """
        Initialize rate constants with smart defaults based on observability constraints.
        
        Rules:
        1. First transition (photoabsorption): quasi-instant, limited by instrument function
           τ_1 = σ/3, so k_1 = 3/σ
        2. For observable states: 2*k_in ≈ k_out (state reaches ~50% occupancy)
        3. Scale rates based on timeframe to ensure observability
        
        Parameters
        ----------
        rate_constants : dict, list, or None
            User-provided rate constants (override smart defaults)
        instrument_width : float
            Instrument function width (σ)
        timepoints : torch.Tensor
            Time points for the experiment
        
        Returns
        -------
        log_k : torch.Tensor
            Log-transformed rate constants
        """
        if rate_constants is not None:
            # User provided values - use the standard initialization
            return self._initialize_parameter(rate_constants, default_value=1.0, transform='log')
        
        # Smart initialization based on observability
        init_k = torch.ones(self.n_transitions)
        
        # Determine timeframe
        t_max = timepoints.max().item()
        t_min = timepoints[timepoints > 0].min().item() if (timepoints > 0).any() else 1e-3
        time_range = t_max - t_min
        if time_range < 1e-6:
            # Handle case of single or very closely spaced timepoints
            time_range = max(t_max, 1.0)
        
        # Build a state connectivity map
        # For each state, track: incoming rates, outgoing rates
        state_in_indices = {state: [] for state in self.states}
        state_out_indices = {state: [] for state in self.states}
        
        for idx, (from_state, to_state) in enumerate(self.transitions):
            state_out_indices[from_state].append(idx)
            state_in_indices[to_state].append(idx)
        
        # Identify the first transition (from initial state)
        initial_state = self.states[0]  # Will be set properly later
        first_transition_indices = state_out_indices[initial_state]
        
        # Initialize first transition(s): quasi-instant, limited by instrument function
        # τ = σ/3, so k = 3/σ
        if instrument_width > 0:
            k_first = 3.0 / instrument_width
        else:
            k_first = 10.0  # Fallback if no instrument function
        
        for idx in first_transition_indices:
            init_k[idx] = k_first
            if self.verbose > 1:
                from_s, to_s = self.transitions[idx]
                print(f"  First transition {from_s}->{to_s}: k = {k_first:.3f} (τ = {1/k_first:.3f})")
        
        # For remaining transitions: apply observability constraint
        # Work through the chain, ensuring 2*k_in ≈ k_out
        processed = set(first_transition_indices)
        
        # Process states in order of connectivity
        for iteration in range(self.n_transitions):
            made_progress = False
            
            for state in self.states:
                in_indices = state_in_indices[state]
                out_indices = state_out_indices[state]
                
                # Skip if no outgoing transitions
                if not out_indices:
                    continue
                
                # Check if we have incoming rates already set
                incoming_set = [idx for idx in in_indices if idx in processed]
                outgoing_unset = [idx for idx in out_indices if idx not in processed]
                
                if incoming_set and outgoing_unset:
                    # Calculate average incoming rate
                    avg_k_in = torch.mean(init_k[incoming_set]).item()
                    
                    # Observability: 2*k_in ≈ k_out for state to reach ~50% occupancy
                    # This ensures the state is observable
                    k_out = avg_k_in / 3.0
                    
                    # Also consider timeframe - states should be observable within the time range
                    # Ensure τ_out is within the observable window
                    tau_out = 1.0 / k_out
                    if tau_out > time_range:
                        # Too slow, speed it up to be observable
                        k_out = 2.0 / time_range
                    elif tau_out < t_min:
                        # Too fast, slow it down
                        k_out = 1.0 / t_min
                    
                    for idx in outgoing_unset:
                        init_k[idx] = k_out
                        processed.add(idx)
                        made_progress = True
                        
                        if self.verbose > 1:
                            from_s, to_s = self.transitions[idx]
                            print(f"  Transition {from_s}->{to_s}: k = {k_out:.3f} (τ = {1/k_out:.3f})")
            
            if not made_progress:
                break
        
        # Handle any remaining unprocessed transitions (disconnected or cyclic)
        for idx in range(self.n_transitions):
            if idx not in processed:
                # Use time-range based default
                k_default = 1.0 / (time_range / 3.0)  # Observable in middle third of range
                init_k[idx] = k_default
                
                if self.verbose > 1:
                    from_s, to_s = self.transitions[idx]
                    print(f"  Transition {from_s}->{to_s}: k = {k_default:.3f} (τ = {1/k_default:.3f}) [default]")
        
        if self.verbose:
            print(f"Smart initialization:")
            print(f"  Time range: {t_min:.3f} to {t_max:.3f} (Δt = {time_range:.3f})")
            print(f"  Instrument width σ = {instrument_width:.3f}")
            print(f"  First transition k = {k_first:.3f} (τ = {1/k_first:.3f})")
        
        return torch.log(init_k)
    
    def _parse_flow_chart(self, flow_chart: str) -> Tuple[List[str], List[Tuple[str, str]]]:
        """
        Parse relational flow chart string to extract states and transitions.
        
        New format: "A->B,B->C,C->D,C->A"
        Comma-separated list of transitions.
        Standalone states (non-reactive) can be included: "A->B,B->C,D"
        
        Parameters
        ----------
        flow_chart : str
            Flow chart string like "A->B,B->C,C->D" or "A->B,B->C,D"
            where D is a non-reactive state
        
        Returns
        -------
        states : List[str]
            Ordered list of unique states
        transitions : List[Tuple[str, str]]
            List of (from_state, to_state) tuples
        """
        # Split by comma to get individual transitions or standalone states
        transition_strings = [t.strip() for t in flow_chart.split(',')]
        
        states_set = set()
        transitions = []
        
        for trans_str in transition_strings:
            if '->' in trans_str:
                # It's a transition
                parts = trans_str.split('->')
                if len(parts) != 2:
                    raise ValueError(f"Invalid transition format: '{trans_str}'. Expected exactly one '->'")
                
                from_state = parts[0].strip()
                to_state = parts[1].strip()
                
                if not from_state or not to_state:
                    raise ValueError(f"Empty state name in transition: '{trans_str}'")
                
                states_set.add(from_state)
                states_set.add(to_state)
                transitions.append((from_state, to_state))
            else:
                # It's a standalone (non-reactive) state
                state_name = trans_str.strip()
                if not state_name:
                    raise ValueError("Empty state name in flow chart")
                states_set.add(state_name)
        
        # Sort states to ensure consistent ordering
        states = sorted(states_set)
        
        return states, transitions
    
    def _build_rate_matrix(self, rate_constants: torch.Tensor, efficiencies: torch.Tensor) -> torch.Tensor:
        """
        Build rate matrix K from rate constants and efficiencies.
        
        The rate matrix K is defined such that:
        dP/dt = K @ P
        
        where P is the population vector.
        
        Effective rate = k * η (rate constant * efficiency)
        
        K[i, j] = effective rate from state j to state i (for i != j)
        K[i, i] = -sum of rates leaving state i
        
        Parameters
        ----------
        rate_constants : torch.Tensor
            Rate constants k for each transition (must be positive)
        efficiencies : torch.Tensor
            Reaction efficiencies η for each transition (0 to 1)
        
        Returns
        -------
        K : torch.Tensor
            Rate matrix of shape (n_states, n_states)
        """
        K = torch.zeros(self.n_states, self.n_states, device=rate_constants.device)
        
        # Effective rates = k * η
        effective_rates = rate_constants * efficiencies
        
        # Fill off-diagonal elements
        for rate_idx, (from_state, to_state) in enumerate(self.transitions):
            from_idx = self.state_to_idx[from_state]
            to_idx = self.state_to_idx[to_state]
            # K[to, from] = effective rate (gain to 'to' from 'from')
            K[to_idx, from_idx] += effective_rates[rate_idx]
        
        # Fill diagonal elements (conservation of probability)
        for i in range(self.n_states):
            K[i, i] = -torch.sum(K[:, i])
        
        return K
    
    def _solve_kinetics(self, rate_matrix: torch.Tensor) -> torch.Tensor:
        """
        Solve kinetic equations using matrix exponential.

        P(t) = exp(K * t) @ P(0)  for t >= 0
        P(t) = P(0)                for t < 0

        torch.matrix_exp uses Padé approximation with scaling-and-squaring,
        which handles large ||K*t|| safely. No element-wise clipping is needed
        (clipping would destroy the row-sum-to-zero structure of the rate matrix).

        Parameters
        ----------
        rate_matrix : torch.Tensor
            Rate matrix K of shape (n_states, n_states)

        Returns
        -------
        populations : torch.Tensor
            Population of each state at each timepoint
            Shape: (n_timepoints, n_states)
        """
        populations = []
        orig_dtype = rate_matrix.dtype

        # Use float64 for matrix exponential to maintain precision
        # when ||K*t|| is large (e.g. fast rates × long times).
        K64 = rate_matrix.double()
        P0_64 = self.initial_populations.double()

        for t in self.timepoints:
            t_val = t.item() if torch.is_tensor(t) else t

            if t_val < 0:
                P_t = P0_64.clone()
            else:
                Kt = K64 * t_val
                exp_Kt = torch.matrix_exp(Kt)
                P_t = exp_Kt @ P0_64
            populations.append(P_t)

        populations = torch.stack(populations, dim=0).to(orig_dtype)
        return populations
    
    def _apply_instrument_function(self, populations: torch.Tensor) -> torch.Tensor:
        """
        Apply instrument response function to account for time resolution.

        Performs convolution in real time space using a kernel matrix that
        accounts for the actual time differences between measurement points.
        This is essential for non-uniformly spaced time grids (e.g. logarithmic).

        For Gaussian IRF:
            S(t_i) = Σ_j P(t_j) * G(t_i - t_j) * w_j  /  Σ_j G(t_i - t_j) * w_j

        where G(Δt) = exp(-Δt²/(2σ²)) and w_j are trapezoidal quadrature weights.

        Parameters
        ----------
        populations : torch.Tensor
            Raw populations, shape (n_timepoints, n_states)

        Returns
        -------
        populations_conv : torch.Tensor
            Populations after convolution, shape (n_timepoints, n_states)
        """
        if self.instrument_function == 'none':
            return populations

        elif self.instrument_function == 'gaussian':
            sigma = torch.exp(self.log_instrument_width)
            t = self.timepoints  # (n_timepoints,)

            # Time difference matrix: dt[i,j] = t_i - t_j
            dt = t.unsqueeze(0) - t.unsqueeze(1)  # (n_t, n_t)

            # Gaussian kernel evaluated at actual time differences
            K = torch.exp(-0.5 * (dt / sigma) ** 2)  # (n_t, n_t)

            # Trapezoidal quadrature weights for non-uniform grid
            n_t = len(t)
            w = torch.zeros(n_t, device=t.device, dtype=t.dtype)
            if n_t > 1:
                w[0] = (t[1] - t[0]) / 2
                w[-1] = (t[-1] - t[-2]) / 2
                if n_t > 2:
                    w[1:-1] = (t[2:] - t[:-2]) / 2
            else:
                w[0] = 1.0

            # Weight kernel by quadrature weights
            K = K * w.unsqueeze(0)  # (n_t, n_t)

            # Normalize each row so weights sum to 1
            K = K / K.sum(dim=1, keepdim=True)

            # Apply convolution: populations_conv = K @ populations
            # Ensure matching dtypes (timepoints may differ from populations)
            populations_conv = torch.matmul(K.to(populations.dtype), populations)

            return populations_conv

        else:
            raise ValueError(f"Unknown instrument function: {self.instrument_function}")
    
    def forward(self) -> torch.Tensor:
        """
        Forward pass: compute populations at all timepoints.
        
        Returns
        -------
        populations : torch.Tensor
            Population of each state at each timepoint
            Shape: (n_timepoints, n_states)
        """
        # Get rate constants (ensure positivity via exp)
        rate_constants = torch.exp(self.log_rate_constants)

        # Build rate matrix (efficiencies frozen at 1.0)
        rate_matrix = self._build_rate_matrix(rate_constants, self.efficiencies)
        
        # Solve kinetics (dynamic populations, sum to 1)
        populations = self._solve_kinetics(rate_matrix)
        
        # Apply instrument function
        populations = self._apply_instrument_function(populations)
        
        # Rescale and add baseline occupancies to maintain total population = 1
        if hasattr(self, '_baseline_occupancies'):
            # Build baseline tensor that includes refinable parameters in the
            # computation graph (so gradients flow through them).
            baseline = self._baseline_occupancies.clone()
            for state, param_name in self._baseline_refinable.items():
                idx = self.state_to_idx[state]
                param = getattr(self, param_name)
                baseline[idx] = torch.sigmoid(param)

            # Calculate total baseline occupancy
            total_baseline = baseline.sum()

            # Rescale dynamic populations to (1 - total_baseline)
            # This ensures that dynamic + baseline = 1
            reactive_fraction = 1.0 - total_baseline
            populations = populations * reactive_fraction

            # Add baseline occupancies
            populations = populations + baseline.unsqueeze(0)
        
        return populations
    
    def set_baseline(
        self, 
        state: str, 
        occupancy: float, 
        refinable: bool = False
    ):
        """
        Set baseline occupancy offset for a state.
        
        Baseline occupancies are constant offsets added to the population
        of a state. This is useful for non-reactive background states.
        
        Parameters
        ----------
        state : str
            Name of the state
        occupancy : float
            Baseline occupancy value (offset)
        refinable : bool, optional
            If True, this baseline becomes a refinable parameter.
            If False (default), it remains constant.
        
        Examples
        --------
        >>> model.set_baseline('D', 0.1, refinable=False)  # Constant 10% background
        >>> model.set_baseline('E', 0.05, refinable=True)  # Refinable baseline
        """
        if state not in self.state_to_idx:
            raise ValueError(f"State '{state}' not found in model. Available states: {self.states}")
        
        idx = self.state_to_idx[state]
        
        if refinable:
            # Create a refinable parameter if it doesn't exist
            param_name = f'_baseline_{state}'
            if not hasattr(self, param_name):
                # Use logit transformation to keep baseline in (0, 1)
                # baseline = sigmoid(logit_baseline)
                init_val = torch.clamp(torch.tensor(occupancy, dtype=torch.float32), 0.01, 0.99)
                logit_val = torch.log(init_val / (1 - init_val))
                setattr(self, param_name, Parameter(logit_val))
                self._baseline_refinable[state] = param_name
            else:
                # Update existing parameter
                param = getattr(self, param_name)
                init_val = torch.clamp(torch.tensor(occupancy, dtype=torch.float32), 0.01, 0.99)
                with torch.no_grad():
                    param.data = torch.log(init_val / (1 - init_val))
        else:
            # Set as constant (non-refinable)
            self._baseline_occupancies[idx] = occupancy
            # Remove from refinable dict if it was there
            if state in self._baseline_refinable:
                param_name = self._baseline_refinable.pop(state)
                if hasattr(self, param_name):
                    delattr(self, param_name)
    
    def get_baselines(self) -> Dict[str, float]:
        """
        Get current baseline occupancies for all states.
        
        Returns
        -------
        baselines : Dict[str, float]
            Dictionary mapping state names to baseline occupancies
        """
        baselines = {}
        for state, idx in self.state_to_idx.items():
            # Check if it's refinable
            if state in self._baseline_refinable:
                param_name = self._baseline_refinable[state]
                param = getattr(self, param_name)
                baselines[state] = float(torch.sigmoid(param).detach().cpu().numpy())
            else:
                # Use the constant value
                baselines[state] = float(self._baseline_occupancies[idx].cpu().numpy())
        return baselines
    
    def _update_baselines_from_refinable(self):
        """
        Update baseline occupancies tensor from refinable parameters.
        Called internally during forward pass if needed.
        """
        for state, param_name in self._baseline_refinable.items():
            idx = self.state_to_idx[state]
            param = getattr(self, param_name)
            self._baseline_occupancies[idx] = torch.sigmoid(param)
    
    def get_rate_constants(self) -> Dict[str, float]:
        """
        Get current rate constants (k) as a dictionary.
        
        Returns
        -------
        rate_dict : Dict[str, float]
            Dictionary mapping transition strings to rate constants
        """
        rate_constants = torch.exp(self.log_rate_constants).detach().cpu().numpy()
        rate_dict = {}
        for idx, (from_state, to_state) in enumerate(self.transitions):
            key = f"{from_state}->{to_state}"
            rate_dict[key] = float(rate_constants[idx])
        return rate_dict
    
    def set_rate_constant(self, transition: str, value: float):
        """
        Set rate constant for a specific transition.
        
        Parameters
        ----------
        transition : str
            Transition string in the format "A->B"
        value : float
            New rate constant value (must be positive)
        """
        if '->' not in transition:
            raise ValueError(f"Invalid transition format: '{transition}'. Expected 'A->B'")
        
        parts = transition.split('->')
        if len(parts) != 2:
            raise ValueError(f"Invalid transition format: '{transition}'. Expected exactly one '->'")
        
        from_state = parts[0].strip()
        to_state = parts[1].strip()
        
        # Find index of the transition
        for idx, (f_state, t_state) in enumerate(self.transitions):
            if f_state == from_state and t_state == to_state:
                with torch.no_grad():
                    self.log_rate_constants[idx] = torch.log(torch.tensor(value, dtype=torch.float32))
                return
        
        raise ValueError(f"Transition '{transition}' not found in model.")
    
    def get_efficiencies(self) -> Dict[str, float]:
        """
        Get current reaction efficiencies (η) as a dictionary.
        
        Returns
        -------
        eff_dict : Dict[str, float]
            Dictionary mapping transition strings to efficiencies (0-1)
        """
        eff_dict = {}
        for idx, (from_state, to_state) in enumerate(self.transitions):
            key = f"{from_state}->{to_state}"
            eff_dict[key] = float(self.efficiencies[idx].item())
        return eff_dict
    
    def get_effective_rates(self) -> Dict[str, float]:
        """
        Get effective rates (k * η) as a dictionary.
        
        Returns
        -------
        eff_rate_dict : Dict[str, float]
            Dictionary mapping transition strings to effective rates
        """
        rate_dict = self.get_rate_constants()
        eff_dict = self.get_efficiencies()
        eff_rate_dict = {key: rate_dict[key] * eff_dict[key] for key in rate_dict}
        return eff_rate_dict
    
    def get_time_constants(self) -> Dict[str, float]:
        """
        Get time constants (1/k_eff) for each transition.
        
        Returns
        -------
        time_dict : Dict[str, float]
            Dictionary mapping transition strings to time constants
        """
        eff_rate_dict = self.get_effective_rates()
        time_dict = {key: 1.0/rate if rate > 1e-10 else float('inf') 
                     for key, rate in eff_rate_dict.items()}
        return time_dict
    
    def parameters(self) -> Dict[str, torch.Tensor]:
        """
        Get all flexible (learnable) parameters as a dictionary.
        
        Returns
        -------
        params : Dict[str, torch.Tensor]
            Dictionary mapping parameter names to their tensors:
            - 'log_rate_constants': log-transformed rate constants
            - 'log_instrument_width': log-transformed instrument width (if refinable)
            - 'baseline_{state}': refinable baseline for specific states (if any)
        
        Examples
        --------
        >>> model = KineticModel(...)
        >>> params = model.parameters()
        >>> print(params.keys())
        >>> # Use with optimizer: optimizer = torch.optim.Adam(params.values(), lr=0.01)
        """
        params = {
            'log_rate_constants': self.log_rate_constants,
        }
        
        if self.log_instrument_width is not None:
            params['log_instrument_width'] = self.log_instrument_width
        
        # Add refinable baselines
        if hasattr(self, '_baseline_refinable'):
            for state in sorted(self._baseline_refinable.keys()):
                param_name = self._baseline_refinable[state]
                params[f'baseline_{state}'] = getattr(self, param_name)
        
        return params
    
    def cuda(self, device: Optional[Union[int, str]] = None):
        """
        Move model to CUDA device.
        
        Parameters
        ----------
        device : int, str, or None, optional
            CUDA device index or name (e.g., 0, 'cuda:0', 'cuda:1').
            If None, uses the default CUDA device.
        
        Returns
        -------
        self : KineticModel
            Returns self for method chaining
        
        Examples
        --------
        >>> model.cuda()  # Move to default CUDA device
        >>> model.cuda(0)  # Move to cuda:0
        >>> model.cuda('cuda:1')  # Move to cuda:1
        """
        if device is None:
            device = 'cuda'
        elif isinstance(device, int):
            device = f'cuda:{device}'
        
        # Move all parameters and buffers
        super().cuda(device)
        
        return self
    
    def cpu(self):
        """
        Move model to CPU.
        
        Returns
        -------
        self : KineticModel
            Returns self for method chaining
        
        Examples
        --------
        >>> model.cpu()  # Move to CPU
        >>> model.cuda().cpu()  # Move to CUDA and back to CPU
        """
        # Move all parameters and buffers
        super().cpu()
        
        return self
    
    def to(self, device: Union[str, torch.device]):
        """
        Move model to specified device.
        
        Parameters
        ----------
        device : str or torch.device
            Target device (e.g., 'cuda', 'cpu', 'cuda:0', torch.device('cuda:1'))
        
        Returns
        -------
        self : KineticModel
            Returns self for method chaining
        
        Examples
        --------
        >>> model.to('cuda')
        >>> model.to('cpu')
        >>> model.to(torch.device('cuda:1'))
        """
        super().to(device)
        
        return self
    
    def print_parameters(self):
        """Print current model parameters."""
        print("\n" + "="*50)
        print(f"Kinetic Model: {self.flow_chart}")
        print("="*50)
        print("\nRate Constants (k):")
        for key, val in self.get_rate_constants().items():
            print(f"  {key}: {val:.6f}")
        print("\nEfficiencies (η):")
        for key, val in self.get_efficiencies().items():
            print(f"  {key}: {val:.4f}")
        print("\nEffective Rates (k*η):")
        for key, val in self.get_effective_rates().items():
            print(f"  {key}: {val:.6f}")
        print("\nTime Constants (1/k_eff):")
        for key, val in self.get_time_constants().items():
            if val == float('inf'):
                print(f"  {key}: ∞")
            else:
                print(f"  {key}: {val:.6f}")
        if self.instrument_function == 'gaussian':
            sigma = torch.exp(self.log_instrument_width).item()
            print(f"\nInstrument Function: Gaussian (σ = {sigma:.6f})")
        
        # Print baselines if any are non-zero
        baselines = self.get_baselines()
        if any(val != 0.0 for val in baselines.values()):
            print("\nBaseline Occupancies:")
            for state, val in baselines.items():
                if val != 0.0:
                    refinable_marker = " (refinable)" if state in self._baseline_refinable else ""
                    print(f"  {state}: {val:.6f}{refinable_marker}")
        
        print("="*50 + "\n")
    
    def plot_occupancies(
        self, 
        outpath: str,
        times: Optional[torch.Tensor] = None,
        log: bool = False,
        figsize: Tuple[int, int] = (10, 6),
        dpi: int = 150,
        title: Optional[str] = None
    ):
        """
        Plot state occupancies over time and save to file.
        
        Parameters
        ----------
        outpath : str
            Path to save the plot (e.g., 'kinetics.png')
        log : bool, optional
            If True, use log scale for x-axis. Default: False
        figsize : Tuple[int, int], optional
            Figure size (width, height). Default: (10, 6)
        dpi : int, optional
            DPI for saving figure. Default: 150
        title : str, optional
            Custom title for the plot. If None, uses flow chart string
        """
        # Compute populations
        if times is not None:
            # Temporarily override timepoints
            original_timepoints = self.timepoints
            self.timepoints = times

        with torch.no_grad():
            populations = self().detach().cpu().numpy()

        t = self.timepoints.cpu().numpy()

        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot each state (combine A and A* if in light-activated mode)
        plotted_states = []
        plotted_populations = []
        
        if self.light_activated:
            # Find initial state and its inactive version
            initial_state = self.states[0]
            inactive_state = initial_state + '*'
            
            # Combine A and A* populations
            if inactive_state in self.states:
                idx_active = self.state_to_idx[initial_state]
                idx_inactive = self.state_to_idx[inactive_state]
                combined_pop = populations[:, idx_active] + populations[:, idx_inactive]
                
                plotted_states.append(initial_state)
                plotted_populations.append(combined_pop)
                
                # Add other states (excluding A and A*)
                for i, state in enumerate(self.states):
                    if state not in [initial_state, inactive_state]:
                        plotted_states.append(state)
                        plotted_populations.append(populations[:, i])
            else:
                # Fallback if something went wrong
                plotted_states = self.states
                plotted_populations = [populations[:, i] for i in range(len(self.states))]
        else:
            # Normal mode: plot all states separately
            plotted_states = self.states
            plotted_populations = [populations[:, i] for i in range(len(self.states))]
        
        # Plot
        for state, pop in zip(plotted_states, plotted_populations):
            plt.plot(t, pop, label=f'State {state}', linewidth=2)
        
        # Formatting
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Occupancy', fontsize=12)
        
        if title is None:
            title = f'State Occupancies: {self.flow_chart}'
        plt.title(title, fontsize=14)
        
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
        
        # Set log scale if requested
        if log:
            # Only use log scale for positive time values
            if (t > 0).any():
                plt.xscale('log')
                # Adjust x-limits to show only positive times
                pos_times = t[t > 0]
                if len(pos_times) > 0:
                    plt.xlim(left=pos_times.min())
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(outpath, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to: {outpath}")
        plt.close()
    
    def visualize(self, outpath: str, **kwargs):
        """
        Alias for plot_occupancies for convenience.
        
        Parameters
        ----------
        outpath : str
            Path to save the plot
        **kwargs
            Additional arguments passed to plot_occupancies
        """
        self.plot_occupancies(outpath, **kwargs)

