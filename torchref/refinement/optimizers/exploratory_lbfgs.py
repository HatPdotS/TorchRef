"""
ExploratoryLBFGS: LBFGS optimizer with automatic landscape exploration.

After convergence, uses Lanczos eigenanalysis to find flat/negative-curvature
directions in the Hessian, scans along them for alternative basins, and hops
to better ones. Useful for escaping local minima in crystallographic refinement
where parameter degeneracies are common.

State machine: OPTIMIZING -> CONVERGED -> EXPLORING -> HOPPING -> OPTIMIZING
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, List, Optional, Tuple

import torch


class OptimizerPhase(Enum):
    """State machine phases for ExploratoryLBFGS."""

    OPTIMIZING = auto()
    CONVERGED = auto()
    EXPLORING = auto()
    HOPPING = auto()


@dataclass
class Mode:
    """A Hessian eigenmode from Lanczos analysis."""

    eigenvalue: float
    direction: torch.Tensor  # flat parameter-space direction vector
    index: int
    is_negative: bool


@dataclass
class ScanPoint:
    """A single evaluation point along a scan direction."""

    t: float  # displacement parameter
    loss: float


@dataclass
class Basin:
    """A detected basin along a scan direction."""

    t: float  # displacement at basin minimum
    loss: float
    direction: torch.Tensor
    loss_improvement: float  # negative means better than current


@dataclass
class ParameterGroup:
    """Group of parameters with significant participation in a mode."""

    indices: List[int]
    direction: torch.Tensor  # full-space direction
    eigenvalue: float
    mode_index: int


@dataclass
class ConvergenceTracker:
    """Tracks convergence criteria over multiple steps."""

    grad_threshold: float = 1e-5
    loss_threshold: float = 1e-7
    param_threshold: float = 1e-6
    n_stable: int = 3

    grad_norms: List[float] = field(default_factory=list)
    loss_changes: List[float] = field(default_factory=list)
    param_changes: List[float] = field(default_factory=list)
    prev_loss: Optional[float] = None
    prev_params: Optional[torch.Tensor] = None

    def update(self, grad_norm: float, loss: float, params: torch.Tensor):
        """Record a new step's metrics."""
        self.grad_norms.append(grad_norm)

        if self.prev_loss is not None:
            self.loss_changes.append(abs(loss - self.prev_loss))
        else:
            self.loss_changes.append(float("inf"))

        if self.prev_params is not None:
            self.param_changes.append(
                (params - self.prev_params).norm().item()
            )
        else:
            self.param_changes.append(float("inf"))

        self.prev_loss = loss
        self.prev_params = params.clone()

    @property
    def is_converged(self) -> bool:
        """Check if all criteria are met for n_stable consecutive steps."""
        if len(self.grad_norms) < self.n_stable:
            return False

        for i in range(-self.n_stable, 0):
            if self.grad_norms[i] > self.grad_threshold:
                return False
            if self.loss_changes[i] > self.loss_threshold:
                return False
            if self.param_changes[i] > self.param_threshold:
                return False
        return True

    def reset(self):
        """Reset all tracking state."""
        self.grad_norms.clear()
        self.loss_changes.clear()
        self.param_changes.clear()
        self.prev_loss = None
        self.prev_params = None


@dataclass
class ExplorationResult:
    """Diagnostics from one exploration cycle."""

    cycle: int
    n_modes_found: int
    n_negative_modes: int
    n_degenerate_modes: int
    n_basins_found: int
    best_basin: Optional[Basin]
    hopped: bool
    eigenvalues: Optional[List[float]] = None


class LanczosError(Exception):
    """Raised when Lanczos iteration fails."""

    pass


class ExploratoryLBFGS(torch.optim.Optimizer):
    """
    LBFGS optimizer with automatic landscape exploration via Lanczos analysis.

    Composes with (rather than subclasses) torch.optim.LBFGS. After the internal
    LBFGS converges, performs eigenanalysis of the Hessian to find degenerate/flat
    directions, scans along them, and hops to better basins if found.

    Parameters
    ----------
    params : iterable
        Parameters to optimize.
    lr : float
        LBFGS learning rate. Default: 1.0.
    max_iter : int
        LBFGS max line search iterations per step. Default: 20.
    history_size : int
        LBFGS Hessian approximation memory. Default: 100.
    m_modes : int
        Number of lowest eigenmodes to compute. Default: 10.
    m_lanczos_iter : int, optional
        Lanczos iterations. Default: 2*m_modes + 10.
    eigenvalue_threshold : float
        Mode is degenerate if eigenvalue < threshold * median(positive). Default: 0.01.
    participation_threshold : float
        Parameter participates if |component| > threshold * ||mode||. Default: 0.05.
    scan_points : int
        Evaluation points per scan direction. Default: 20.
    scan_step_size : float
        Step size in parameter space units. Default: 0.1.
    max_exploration_cycles : int
        Cap on explore-hop cycles. Default: 5.
    hvp_epsilon : float
        Finite-difference epsilon for Hessian-vector products. Default: 1e-4.
    convergence_grad_threshold : float
        Gradient norm convergence threshold. Default: 1e-5.
    convergence_loss_threshold : float
        Loss change convergence threshold. Default: 1e-7.
    convergence_param_threshold : float
        Parameter change convergence threshold. Default: 1e-6.
    n_stable : int
        Consecutive converged steps required. Default: 3.
    verbose : int
        Verbosity level: 0=silent, 1=summary, 2=detailed. Default: 1.
    """

    def __init__(
        self,
        params,
        lr: float = 1.0,
        max_iter: int = 20,
        history_size: int = 100,
        m_modes: int = 10,
        m_lanczos_iter: Optional[int] = None,
        eigenvalue_threshold: float = 0.01,
        participation_threshold: float = 0.05,
        scan_points: int = 20,
        scan_step_size: float = 0.1,
        max_exploration_cycles: int = 5,
        hvp_epsilon: float = 1e-4,
        convergence_grad_threshold: float = 1e-5,
        convergence_loss_threshold: float = 1e-7,
        convergence_param_threshold: float = 1e-6,
        n_stable: int = 3,
        verbose: int = 1,
    ):
        # Store param list before super().__init__ consumes it
        param_list = list(params)

        defaults = dict(lr=lr)
        super().__init__(param_list, defaults)

        # LBFGS config
        self._lr = lr
        self._max_iter = max_iter
        self._history_size = history_size

        # Exploration config
        self._m_modes = m_modes
        self._m_lanczos_iter = (
            m_lanczos_iter if m_lanczos_iter is not None else 2 * m_modes + 10
        )
        self._eigenvalue_threshold = eigenvalue_threshold
        self._participation_threshold = participation_threshold
        self._scan_points = scan_points
        self._scan_step_size = scan_step_size
        self._max_exploration_cycles = max_exploration_cycles
        self._hvp_epsilon = hvp_epsilon
        self._verbose = verbose

        # Compute parameter shapes/sizes for flatten/unflatten
        self._param_shapes = []
        self._param_sizes = []
        for group in self.param_groups:
            for p in group["params"]:
                self._param_shapes.append(p.shape)
                self._param_sizes.append(p.numel())
        self._total_params = sum(self._param_sizes)

        # State machine
        self._phase = OptimizerPhase.OPTIMIZING
        self._exploration_cycle = 0
        self._convergence_tracker = ConvergenceTracker(
            grad_threshold=convergence_grad_threshold,
            loss_threshold=convergence_loss_threshold,
            param_threshold=convergence_param_threshold,
            n_stable=n_stable,
        )
        self._best_basin: Optional[Basin] = None

        # History / diagnostics
        self.exploration_history: List[ExplorationResult] = []

        # Create internal LBFGS
        self._lbfgs = self._create_lbfgs()

    # =========================================================================
    # Parameter flatten / unflatten utilities
    # =========================================================================

    def _get_params(self) -> List[torch.nn.Parameter]:
        """Get flat list of all parameters across groups."""
        params = []
        for group in self.param_groups:
            for p in group["params"]:
                params.append(p)
        return params

    def _gather_flat_params(self) -> torch.Tensor:
        """Flatten all parameters into a single 1D tensor."""
        params = self._get_params()
        return torch.cat([p.data.view(-1) for p in params])

    def _set_flat_params(self, flat: torch.Tensor):
        """Set parameters from a flat 1D tensor."""
        params = self._get_params()
        offset = 0
        for p, size, shape in zip(
            params, self._param_sizes, self._param_shapes
        ):
            p.data.copy_(flat[offset : offset + size].view(shape))
            offset += size

    def _gather_flat_grad(self) -> torch.Tensor:
        """Flatten all gradients into a single 1D tensor (None -> zeros)."""
        params = self._get_params()
        grads = []
        for p in params:
            if p.grad is not None:
                grads.append(p.grad.data.view(-1))
            else:
                grads.append(torch.zeros(p.numel(), device=p.device))
        return torch.cat(grads)

    # =========================================================================
    # Internal LBFGS management
    # =========================================================================

    def _create_lbfgs(self) -> torch.optim.LBFGS:
        """Create (or recreate) the internal LBFGS optimizer."""
        params = self._get_params()
        return torch.optim.LBFGS(
            params,
            lr=self._lr,
            max_iter=self._max_iter,
            history_size=self._history_size,
            line_search_fn="strong_wolfe",
        )

    # =========================================================================
    # Hessian-vector product via finite differences
    # =========================================================================

    def _hvp_finite_difference(
        self, closure: Callable, v: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Hessian-vector product using central finite differences.

        H @ v ~ (grad(x + eps*v) - grad(x - eps*v)) / (2 * eps)

        Parameters
        ----------
        closure : callable
            Evaluates loss and computes gradients. Must call zero_grad + backward.
        v : torch.Tensor
            Direction vector (flat, same size as all params).

        Returns
        -------
        torch.Tensor
            Hessian-vector product (flat).
        """
        eps = self._hvp_epsilon
        x0 = self._gather_flat_params()

        # Forward perturbation: x + eps*v
        self._set_flat_params(x0 + eps * v)
        closure()
        grad_plus = self._gather_flat_grad().clone()

        # Backward perturbation: x - eps*v
        self._set_flat_params(x0 - eps * v)
        closure()
        grad_minus = self._gather_flat_grad().clone()

        # Restore original parameters
        self._set_flat_params(x0)

        return (grad_plus - grad_minus) / (2.0 * eps)

    # =========================================================================
    # Lanczos eigenanalysis
    # =========================================================================

    def _lanczos(
        self, closure: Callable
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Lanczos algorithm with full reorthogonalization.

        Builds a Krylov subspace and tridiagonal matrix T, then extracts
        Ritz values/vectors as approximate eigenpairs of the Hessian.

        Parameters
        ----------
        closure : callable
            Evaluates loss and computes gradients.

        Returns
        -------
        eigenvalues : torch.Tensor
            Shape (k,) — lowest eigenvalues, sorted ascending.
        eigenvectors : torch.Tensor
            Shape (k, n_params) — corresponding eigenvectors.

        Raises
        ------
        LanczosError
            If the algorithm fails to produce valid results.
        """
        n = self._total_params
        m = min(self._m_lanczos_iter, n)
        k = min(self._m_modes, m)

        device = self._get_params()[0].device

        # Lanczos vectors
        Q = torch.zeros(m + 1, n, device=device)
        alphas = torch.zeros(m, device=device)
        betas = torch.zeros(m, device=device)

        # Initial random vector
        q = torch.randn(n, device=device)
        q = q / q.norm()
        Q[0] = q

        for j in range(m):
            # Hessian-vector product
            w = self._hvp_finite_difference(closure, Q[j])

            # Compute alpha_j = q_j^T H q_j
            alphas[j] = torch.dot(Q[j], w)

            # Subtract projections
            w = w - alphas[j] * Q[j]
            if j > 0:
                w = w - betas[j - 1] * Q[j - 1]

            # Full reorthogonalization
            for i in range(j + 1):
                coeff = torch.dot(w, Q[i])
                w = w - coeff * Q[i]

            beta = w.norm()

            if beta < 1e-12:
                # Invariant subspace found — truncate
                m_actual = j + 1
                alphas = alphas[:m_actual]
                betas = betas[:m_actual]
                Q = Q[: m_actual + 1]
                break
            else:
                betas[j] = beta
                Q[j + 1] = w / beta
                m_actual = j + 1

        # Build tridiagonal matrix T
        T = torch.zeros(m_actual, m_actual, device=device)
        for i in range(m_actual):
            T[i, i] = alphas[i]
        for i in range(m_actual - 1):
            T[i, i + 1] = betas[i]
            T[i + 1, i] = betas[i]

        # Eigendecompose T
        try:
            evals, evecs_T = torch.linalg.eigh(T)
        except Exception as e:
            raise LanczosError(f"Eigendecomposition of T failed: {e}")

        # Compute Ritz vectors: V = Q[:m_actual]^T @ evecs_T
        Q_basis = Q[:m_actual]  # (m_actual, n)
        ritz_vectors = evecs_T.T @ Q_basis  # (m_actual, n)

        # Normalize Ritz vectors
        norms = ritz_vectors.norm(dim=1, keepdim=True)
        valid = norms.squeeze() > 1e-10
        if not valid.any():
            raise LanczosError("All Ritz vectors have zero norm")
        ritz_vectors[valid] = ritz_vectors[valid] / norms[valid]

        # Return k lowest eigenvalues/vectors (already sorted by eigh)
        k = min(k, m_actual)
        return evals[:k], ritz_vectors[:k]

    # =========================================================================
    # Mode identification
    # =========================================================================

    def _identify_degenerate_modes(
        self, eigenvalues: torch.Tensor, eigenvectors: torch.Tensor
    ) -> List[Mode]:
        """
        Classify eigenmodes as negative (saddle) or degenerate (flat).

        Parameters
        ----------
        eigenvalues : torch.Tensor
            Eigenvalues from Lanczos.
        eigenvectors : torch.Tensor
            Corresponding eigenvectors.

        Returns
        -------
        list of Mode
            Modes classified as interesting for exploration.
        """
        modes = []

        # Compute threshold from positive eigenvalues
        positive_mask = eigenvalues > 0
        if positive_mask.any():
            median_positive = torch.median(eigenvalues[positive_mask]).item()
            threshold = self._eigenvalue_threshold * median_positive
        else:
            # All non-positive — explore everything
            threshold = float("inf")

        for i, (ev, vec) in enumerate(zip(eigenvalues, eigenvectors)):
            ev_val = ev.item()
            is_negative = ev_val < 0

            if is_negative or ev_val < threshold:
                modes.append(
                    Mode(
                        eigenvalue=ev_val,
                        direction=vec,
                        index=i,
                        is_negative=is_negative,
                    )
                )

        return modes

    # =========================================================================
    # Parameter grouping
    # =========================================================================

    def _discover_groups(self, modes: List[Mode]) -> List[ParameterGroup]:
        """
        For each mode, group parameters with significant participation.

        Parameters
        ----------
        modes : list of Mode
            Degenerate/negative modes to analyze.

        Returns
        -------
        list of ParameterGroup
        """
        groups = []
        for mode in modes:
            direction = mode.direction
            norm = direction.norm().item()
            if norm < 1e-12:
                continue

            threshold = self._participation_threshold * norm
            significant = (direction.abs() > threshold).nonzero(as_tuple=True)[
                0
            ]

            if len(significant) > 0:
                groups.append(
                    ParameterGroup(
                        indices=significant.tolist(),
                        direction=direction,
                        eigenvalue=mode.eigenvalue,
                        mode_index=mode.index,
                    )
                )

        return groups

    # =========================================================================
    # Scanning along directions
    # =========================================================================

    def _scan_group(
        self, group: ParameterGroup, eval_fn: Callable
    ) -> List[ScanPoint]:
        """
        Scan along a mode direction, evaluating loss at each point.

        Scans both positive and negative directions from current position.
        Restores parameters after scanning.

        Parameters
        ----------
        group : ParameterGroup
            The parameter group with direction to scan.
        eval_fn : callable
            Forward-only loss evaluation (no backward). Called inside
            torch.no_grad(). Must return a scalar loss value.

        Returns
        -------
        list of ScanPoint
            Loss evaluations along the scan direction.
        """
        x0 = self._gather_flat_params()
        direction = group.direction
        # Normalize direction
        d_norm = direction.norm()
        if d_norm < 1e-12:
            return []
        d = direction / d_norm

        n_points = self._scan_points
        step = self._scan_step_size

        scan_results = []

        # Scan from -n_points to +n_points
        for i in range(-n_points, n_points + 1):
            t = i * step
            self._set_flat_params(x0 + t * d)
            loss = eval_fn()
            loss_val = (
                loss.item() if isinstance(loss, torch.Tensor) else loss
            )
            scan_results.append(ScanPoint(t=t, loss=loss_val))

        # Restore original parameters
        self._set_flat_params(x0)

        return scan_results

    # =========================================================================
    # Basin detection
    # =========================================================================

    def _detect_basins(
        self,
        scan_results: List[ScanPoint],
        direction: torch.Tensor,
        current_loss: float,
    ) -> List[Basin]:
        """
        Find local minima in a 1D loss profile by neighbor comparison.

        Parameters
        ----------
        scan_results : list of ScanPoint
            Loss values along the scan.
        direction : torch.Tensor
            The scan direction.
        current_loss : float
            Loss at the starting point (t=0).

        Returns
        -------
        list of Basin
            Detected basins, sorted by loss (best first).
        """
        if len(scan_results) < 3:
            return []

        basins = []
        for i in range(1, len(scan_results) - 1):
            prev_loss = scan_results[i - 1].loss
            curr_loss = scan_results[i].loss
            next_loss = scan_results[i + 1].loss

            # Local minimum: lower than both neighbors
            if curr_loss < prev_loss and curr_loss < next_loss:
                # Skip the origin basin (t ~ 0)
                if abs(scan_results[i].t) < self._scan_step_size * 0.5:
                    continue

                improvement = curr_loss - current_loss
                basins.append(
                    Basin(
                        t=scan_results[i].t,
                        loss=curr_loss,
                        direction=direction,
                        loss_improvement=improvement,
                    )
                )

        # Sort by loss (best first)
        basins.sort(key=lambda b: b.loss)
        return basins

    # =========================================================================
    # Saddle mode handling
    # =========================================================================

    def _handle_saddle_modes(
        self, saddle_modes: List[Mode], eval_fn: Callable, current_loss: float
    ) -> List[Basin]:
        """
        Scan along negative-eigenvalue directions and find basins.

        Parameters
        ----------
        saddle_modes : list of Mode
            Modes with negative eigenvalues.
        eval_fn : callable
            Forward-only loss evaluation (no backward).
        current_loss : float
            Current loss value.

        Returns
        -------
        list of Basin
            Best basins found along saddle directions.
        """
        all_basins = []
        for mode in saddle_modes:
            group = ParameterGroup(
                indices=list(range(self._total_params)),
                direction=mode.direction,
                eigenvalue=mode.eigenvalue,
                mode_index=mode.index,
            )
            scan_results = self._scan_group(group, eval_fn)
            basins = self._detect_basins(
                scan_results, mode.direction, current_loss
            )
            all_basins.extend(basins)

        all_basins.sort(key=lambda b: b.loss)
        return all_basins

    # =========================================================================
    # State machine: step()
    # =========================================================================

    @property
    def phase(self) -> OptimizerPhase:
        """Current optimizer phase."""
        return self._phase

    def step(self, closure: Callable) -> Optional[float]:
        """
        Perform one step of the state machine.

        Parameters
        ----------
        closure : callable
            A closure that re-evaluates the model loss. Should call
            ``optimizer.zero_grad()``, compute the loss, call ``loss.backward()``,
            and return the loss.

        Returns
        -------
        float or None
            The loss value.
        """
        if self._phase == OptimizerPhase.OPTIMIZING:
            return self._step_optimizing(closure)
        elif self._phase == OptimizerPhase.CONVERGED:
            return self._step_converged(closure)
        elif self._phase == OptimizerPhase.EXPLORING:
            return self._step_exploring(closure)
        elif self._phase == OptimizerPhase.HOPPING:
            return self._step_hopping(closure)

    def _step_optimizing(self, closure: Callable) -> Optional[float]:
        """OPTIMIZING phase: delegate to internal LBFGS, track convergence."""
        loss = self._lbfgs.step(closure)

        # Track convergence
        with torch.no_grad():
            flat_params = self._gather_flat_params()
            grad_norm = self._gather_flat_grad().norm().item()
            loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss

        self._convergence_tracker.update(grad_norm, loss_val, flat_params)

        if self._convergence_tracker.is_converged:
            if self._exploration_cycle < self._max_exploration_cycles:
                self._phase = OptimizerPhase.CONVERGED
                if self._verbose >= 1:
                    print(
                        f"[ExploratoryLBFGS] Converged (cycle {self._exploration_cycle + 1}/"
                        f"{self._max_exploration_cycles}). "
                        f"Loss={loss_val:.6f}, |grad|={grad_norm:.2e}"
                    )
            else:
                if self._verbose >= 1:
                    print(
                        f"[ExploratoryLBFGS] Converged. Max exploration cycles "
                        f"({self._max_exploration_cycles}) reached."
                    )

        return loss

    def _step_converged(self, closure: Callable) -> Optional[float]:
        """CONVERGED phase: immediately transition to EXPLORING."""
        self._phase = OptimizerPhase.EXPLORING
        return self._step_exploring(closure)

    def _step_exploring(self, closure: Callable) -> Optional[float]:
        """EXPLORING phase: Lanczos -> modes -> scan -> detect basins."""
        self._exploration_cycle += 1

        # Get current loss — closure already does zero_grad + forward + backward
        loss = closure()
        current_loss = (
            loss.item() if isinstance(loss, torch.Tensor) else loss
        )

        if self._verbose >= 1:
            print(
                f"[ExploratoryLBFGS] Exploring (cycle {self._exploration_cycle})..."
            )

        # Lanczos eigenanalysis — pass closure directly (it does zero_grad + backward)
        try:
            eigenvalues, eigenvectors = self._lanczos(closure)
        except LanczosError as e:
            if self._verbose >= 1:
                print(f"[ExploratoryLBFGS] Lanczos failed: {e}. Resuming optimization.")
            self._record_exploration(0, 0, 0, 0, None, False, None)
            self._phase = OptimizerPhase.OPTIMIZING
            return loss

        if self._verbose >= 2:
            print(
                f"  Eigenvalues: {eigenvalues.tolist()}"
            )

        # Identify interesting modes
        modes = self._identify_degenerate_modes(eigenvalues, eigenvectors)
        saddle_modes = [m for m in modes if m.is_negative]
        degenerate_modes = [m for m in modes if not m.is_negative]

        if self._verbose >= 1:
            print(
                f"  Found {len(saddle_modes)} saddle modes, "
                f"{len(degenerate_modes)} degenerate modes"
            )

        if not modes:
            if self._verbose >= 1:
                print(
                    "  No degenerate/saddle modes. Landscape is well-determined."
                )
            self._record_exploration(
                len(eigenvalues),
                0,
                0,
                0,
                None,
                False,
                eigenvalues.tolist(),
            )
            self._phase = OptimizerPhase.OPTIMIZING
            return loss

        # Scan along modes and detect basins
        # Use the closure directly — it does zero_grad + forward + backward
        all_basins = []

        # Handle saddle modes first (priority — negative curvature)
        if saddle_modes:
            saddle_basins = self._handle_saddle_modes(
                saddle_modes, closure, current_loss
            )
            all_basins.extend(saddle_basins)

        # Handle degenerate modes
        if degenerate_modes:
            groups = self._discover_groups(degenerate_modes)
            for group in groups:
                scan_results = self._scan_group(group, closure)
                basins = self._detect_basins(
                    scan_results, group.direction, current_loss
                )
                all_basins.extend(basins)

        if self._verbose >= 1:
            print(f"  Found {len(all_basins)} basins total")

        # Find best basin (lowest loss, must improve)
        improving_basins = [b for b in all_basins if b.loss_improvement < 0]
        if improving_basins:
            self._best_basin = min(improving_basins, key=lambda b: b.loss)
            if self._verbose >= 1:
                print(
                    f"  Best basin: loss={self._best_basin.loss:.6f} "
                    f"(improvement={self._best_basin.loss_improvement:.6f})"
                )
            self._record_exploration(
                len(eigenvalues),
                len(saddle_modes),
                len(degenerate_modes),
                len(all_basins),
                self._best_basin,
                True,
                eigenvalues.tolist(),
            )
            self._phase = OptimizerPhase.HOPPING
        else:
            if self._verbose >= 1:
                print("  No improving basins found. Stopping exploration.")
            self._record_exploration(
                len(eigenvalues),
                len(saddle_modes),
                len(degenerate_modes),
                len(all_basins),
                None,
                False,
                eigenvalues.tolist(),
            )
            self._phase = OptimizerPhase.OPTIMIZING

        return loss

    def _step_hopping(self, closure: Callable) -> Optional[float]:
        """HOPPING phase: move to best basin, recreate LBFGS, resume optimizing."""
        basin = self._best_basin
        if basin is None:
            self._phase = OptimizerPhase.OPTIMIZING
            return None

        # Move parameters to basin
        x0 = self._gather_flat_params()
        d = basin.direction
        d_norm = d.norm()
        if d_norm > 1e-12:
            d = d / d_norm
        self._set_flat_params(x0 + basin.t * d)

        if self._verbose >= 1:
            print(
                f"[ExploratoryLBFGS] Hopped to basin at t={basin.t:.4f}, "
                f"expected loss={basin.loss:.6f}"
            )

        # Recreate LBFGS (reset curvature history)
        self._lbfgs = self._create_lbfgs()

        # Reset convergence tracker
        self._convergence_tracker.reset()

        # Clear basin
        self._best_basin = None

        # Return to optimizing
        self._phase = OptimizerPhase.OPTIMIZING

        # Do one optimization step immediately
        return self._step_optimizing(closure)

    # =========================================================================
    # Diagnostics recording
    # =========================================================================

    def _record_exploration(
        self,
        n_modes: int,
        n_negative: int,
        n_degenerate: int,
        n_basins: int,
        best_basin: Optional[Basin],
        hopped: bool,
        eigenvalues: Optional[List[float]],
    ):
        """Record exploration diagnostics."""
        self.exploration_history.append(
            ExplorationResult(
                cycle=self._exploration_cycle,
                n_modes_found=n_modes,
                n_negative_modes=n_negative,
                n_degenerate_modes=n_degenerate,
                n_basins_found=n_basins,
                best_basin=best_basin,
                hopped=hopped,
                eigenvalues=eigenvalues,
            )
        )
