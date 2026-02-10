"""
Internal Coordinate Simulated Annealing optimizer.

Implements batch Metropolis SA for internal coordinates with:
- Auto-calibrated temperature from energy landscape sampling
- Single scale parameter controlling perturbation magnitude
- Best structure tracking throughout optimization
- Torsion wrapping to [-pi, pi]

Usage:
    model.use_internal_coordinates(n_aa_per_segment=3)
    ic = model.xyz

    # Torsion-only refinement (recommended for conformational exploration):
    state, best_loss = optimize_internal_coord_sa(
        state=state,
        params=[ic.torsions],
        scale=1.0,
        ...
    )

    # All internal coordinates:
    state, best_loss = optimize_internal_coord_sa(
        state=state,
        params=[ic.torsions, ic.bond_lengths, ic.angles,
                ic.segment_positions, ic.segment_orientations],
        scale=1.0,
        ...
    )
"""

import math
from typing import Callable, Dict, List, Optional, Tuple, Any

import torch

from torchref.refinement.loss_state import LossState


def _is_angle_like(param: torch.Tensor) -> bool:
    """
    Heuristic to detect if a parameter tensor contains angle-like values.

    Angles (torsions, bond angles) typically have values in [-pi, pi] or [0, pi].
    Position-like parameters (segment_positions) have larger magnitudes.
    """
    if param.numel() == 0:
        return False
    with torch.no_grad():
        abs_max = param.abs().max().item()
        # Angles are bounded by pi (~3.14), positions are typically > 10 Angstrom
        return abs_max < 10.0


def _wrap_torsions(param: torch.Tensor) -> torch.Tensor:
    """Wrap angle values to [-pi, pi]."""
    return torch.atan2(torch.sin(param), torch.cos(param))


def _calibrate_temperature(
    state: LossState,
    params: List[torch.Tensor],
    scale: float,
    n_samples: int = 20,
    target_acceptance: float = 0.5,
) -> float:
    """
    Auto-calibrate initial temperature based on energy landscape.

    Samples typical energy changes from random perturbations and sets
    T_initial so that the median delta_E gives target_acceptance probability.

    For Metropolis: P(accept) = exp(-delta_E / T)
    => T = -delta_E / ln(target_acceptance)
    """
    deltas = []

    with torch.no_grad():
        # Save original state
        saved_values = [p.data.clone() for p in params]
        current_loss = state.aggregate().item()

        for _ in range(n_samples):
            # Apply random perturbations
            for param in params:
                if _is_angle_like(param):
                    # Angle-like: scale * 0.1 rad
                    noise = torch.randn_like(param) * (scale * 0.1)
                else:
                    # Position-like: scale * 0.01 A
                    noise = torch.randn_like(param) * (scale * 0.01)
                param.data.add_(noise)

            # Compute energy change
            new_loss = state.aggregate().item()
            delta = new_loss - current_loss

            # Only consider positive deltas for calibration
            if delta > 0:
                deltas.append(delta)

            # Restore
            for param, saved in zip(params, saved_values):
                param.data.copy_(saved)

    if len(deltas) == 0:
        # All moves were downhill - use a small default
        return 1.0

    # Use median positive delta for robust calibration
    median_delta = sorted(deltas)[len(deltas) // 2]

    # T such that exp(-median_delta / T) = target_acceptance
    # => T = -median_delta / ln(target_acceptance)
    T = -median_delta / math.log(target_acceptance)

    return max(T, 0.01)  # Ensure positive temperature


def optimize_internal_coord_sa(
    state: LossState,
    params: List[torch.Tensor],
    scale: float = 1.0,
    T_initial: Optional[float] = None,
    T_final: float = 0.01,
    n_steps: int = 5000,
    cooling_schedule: str = "exponential",
    verbose: int = 0,
    callback: Optional[Callable] = None,
) -> Tuple[LossState, float]:
    """
    Universal Simulated Annealing for internal coordinates.

    Perturbs the provided parameter tensors using batch Metropolis:
    all parameters are perturbed together, single accept/reject decision.

    Parameters
    ----------
    state : LossState
        Configured loss state with targets.
    params : List[torch.Tensor]
        Parameter tensors to optimize. For internal coordinates:
        - Torsion-only: [ic.torsions]
        - All coords: [ic.torsions, ic.bond_lengths, ic.angles,
                       ic.segment_positions, ic.segment_orientations]
    scale : float
        Master scale for perturbations. Default 1.0.
        - Angle-like params: scale * 0.1 rad (~6 degrees at scale=1.0)
        - Position-like params: scale * 0.01 A
    T_initial : float, optional
        Initial temperature. If None, auto-calibrated from energy landscape.
    T_final : float
        Final temperature. Default 0.01.
    n_steps : int
        Number of SA steps. Default 5000.
    cooling_schedule : str
        'exponential' or 'linear'. Default 'exponential'.
    verbose : int
        Verbosity level.
    callback : callable, optional
        Called after each step: callback(step, T, loss, best_loss).

    Returns
    -------
    state : LossState
        Updated state.
    best_loss : float
        Best (lowest) loss found during optimization.
    """
    if len(params) == 0:
        raise ValueError("No parameters provided to optimize")

    # Ensure all params have requires_grad (for consistency)
    for p in params:
        if not isinstance(p, torch.Tensor):
            raise TypeError(f"params must be torch.Tensor, got {type(p)}")

    # Identify param types (angle-like vs position-like)
    is_angle = [_is_angle_like(p) for p in params]

    if verbose > 0:
        print("Internal Coord SA Parameter Summary:")
        total_params = 0
        for i, p in enumerate(params):
            n = p.numel()
            total_params += n
            ptype = "angle" if is_angle[i] else "position"
            pert = scale * 0.1 if is_angle[i] else scale * 0.01
            print(f"  Param {i}: {n:6d} params ({ptype:8s}), perturbation={pert:.4f}")
        print(f"  Total: {total_params} parameters")

    # Auto-calibrate temperature if not provided
    if T_initial is None:
        T_initial = _calibrate_temperature(state, params, scale)
        if verbose > 0:
            print(f"  Auto-calibrated T_initial: {T_initial:.4f}")

    # Cooling rate
    if cooling_schedule == "exponential":
        cooling_rate = (T_final / T_initial) ** (1.0 / n_steps)
    else:
        cooling_rate = None

    # Initialize
    with torch.no_grad():
        current_loss = state.aggregate().item()

    best_loss = current_loss
    best_params = [p.data.clone() for p in params]

    T = T_initial
    n_accepted = 0
    n_proposed = 0

    for step in range(n_steps):
        # Update temperature
        if cooling_schedule == "exponential":
            T = T_initial * (cooling_rate ** step)
        else:
            T = T_initial - (T_initial - T_final) * (step / n_steps)

        n_proposed += 1

        # Save current values
        with torch.no_grad():
            saved_values = [p.data.clone() for p in params]

            # Apply perturbations
            for i, param in enumerate(params):
                if is_angle[i]:
                    noise = torch.randn_like(param) * (scale * 0.1)
                else:
                    noise = torch.randn_like(param) * (scale * 0.01)
                param.data.add_(noise)

                # Wrap torsions to [-pi, pi]
                if is_angle[i]:
                    param.data.copy_(_wrap_torsions(param.data))

            # Evaluate new loss
            new_loss = state.aggregate().item()

        delta_E = new_loss - current_loss

        # Metropolis criterion
        accept = False
        if delta_E < 0:
            accept = True
        elif T > 0 and torch.rand(1).item() < math.exp(-delta_E / T):
            accept = True

        if accept:
            current_loss = new_loss
            n_accepted += 1

            # Track best
            if current_loss < best_loss:
                best_loss = current_loss
                with torch.no_grad():
                    best_params = [p.data.clone() for p in params]
        else:
            # Reject - restore
            with torch.no_grad():
                for param, saved in zip(params, saved_values):
                    param.data.copy_(saved)

        # Progress logging
        if verbose > 0 and (step + 1) % max(1, n_steps // 10) == 0:
            accept_rate = n_accepted / max(1, n_proposed)
            print(f"Step {step+1}/{n_steps}, T={T:.4f}, Loss={current_loss:.4f}, "
                  f"Best={best_loss:.4f}, Accept={accept_rate:.1%}")

        # Callback
        if callback is not None:
            callback(step, T, current_loss, best_loss)

    # Restore best parameters
    with torch.no_grad():
        for param, best in zip(params, best_params):
            param.data.copy_(best)

    if verbose > 0:
        final_accept = n_accepted / max(1, n_proposed)
        print(f"\nInternal Coord SA Complete:")
        print(f"  Final loss: {best_loss:.4f}")
        print(f"  Acceptance rate: {final_accept:.1%}")

    return state, best_loss


def _calibrate_temperature_gradient(
    loss_fn: Callable[[], torch.Tensor],
    params: List[torch.Tensor],
    scale: float,
    n_samples: int = 20,
    target_acceptance: float = 0.5,
) -> float:
    """
    Auto-calibrate temperature for gradient-based SA.

    Samples typical |gradient · perturbation| values and sets T_initial
    so that the median gives target_acceptance probability.
    """
    deltas = []

    for _ in range(n_samples):
        # Compute gradient
        for p in params:
            if p.grad is not None:
                p.grad.zero_()

        loss = loss_fn()
        loss.backward()

        # Generate random perturbations and compute g · δ
        total_delta = 0.0
        for p in params:
            if p.grad is not None:
                if _is_angle_like(p):
                    delta = torch.randn_like(p) * (scale * 0.1)
                else:
                    delta = torch.randn_like(p) * (scale * 0.01)
                # g · δ for this parameter
                total_delta += (p.grad * delta).sum().item()

        if total_delta > 0:
            deltas.append(total_delta)

    if len(deltas) == 0:
        return 1.0

    median_delta = sorted(deltas)[len(deltas) // 2]
    T = -median_delta / math.log(target_acceptance)
    return max(T, 0.01)


def optimize_gradient_sa(
    loss_fn: Callable[[], torch.Tensor],
    params: List[torch.Tensor],
    scale: float = 1.0,
    T_initial: Optional[float] = None,
    T_final: float = 0.01,
    n_steps: int = 5000,
    cooling_schedule: str = "exponential",
    per_parameter: bool = True,
    verbose: int = 0,
    callback: Optional[Callable] = None,
) -> Tuple[float, float]:
    """
    Gradient-based Simulated Annealing.

    Uses gradient information for Metropolis acceptance criterion:
    - ΔE ≈ gradient · perturbation (first-order Taylor approximation)
    - Accept if g·δ < 0 (moved downhill)
    - Accept with probability exp(-g·δ / T) if g·δ > 0

    Advantages over loss-based SA:
    - One gradient computation per step (vs one loss eval per perturbation)
    - Per-parameter acceptance allows partial moves
    - Gradient guides acceptance decisions

    Parameters
    ----------
    loss_fn : callable
        Loss function returning a differentiable scalar tensor.
    params : List[torch.Tensor]
        Parameter tensors to optimize (must have requires_grad=True).
    scale : float
        Perturbation scale. Default 1.0.
        - Angle-like: scale * 0.1 rad
        - Position-like: scale * 0.01 A
    T_initial : float, optional
        Initial temperature. Auto-calibrated if None.
    T_final : float
        Final temperature. Default 0.01.
    n_steps : int
        Number of SA steps. Default 5000.
    cooling_schedule : str
        'exponential' or 'linear'. Default 'exponential'.
    per_parameter : bool
        If True, accept/reject each parameter independently.
        If False, use batch acceptance (sum of g·δ for all params).
        Default True.
    verbose : int
        Verbosity level.
    callback : callable, optional
        Called after each step: callback(step, T, loss, accept_rate).

    Returns
    -------
    best_loss : float
        Best loss found during optimization.
    final_loss : float
        Loss at the end of optimization.
    """
    if len(params) == 0:
        raise ValueError("No parameters provided")

    # Ensure requires_grad
    for p in params:
        if not p.requires_grad:
            p.requires_grad_(True)

    # Identify param types
    is_angle = [_is_angle_like(p) for p in params]

    if verbose > 0:
        print("Gradient-based SA Parameter Summary:")
        total_params = 0
        for i, p in enumerate(params):
            n = p.numel()
            total_params += n
            ptype = "angle" if is_angle[i] else "position"
            pert = scale * 0.1 if is_angle[i] else scale * 0.01
            print(f"  Param {i}: {n:6d} params ({ptype:8s}), perturbation={pert:.4f}")
        print(f"  Total: {total_params} parameters")
        print(f"  Mode: {'per-parameter' if per_parameter else 'batch'}")

    # Auto-calibrate temperature
    if T_initial is None:
        T_initial = _calibrate_temperature_gradient(loss_fn, params, scale)
        if verbose > 0:
            print(f"  Auto-calibrated T_initial: {T_initial:.4f}")

    # Cooling rate
    if cooling_schedule == "exponential":
        cooling_rate = (T_final / T_initial) ** (1.0 / n_steps)

    # Initial loss
    with torch.no_grad():
        current_loss = loss_fn().item()

    best_loss = current_loss
    best_params = [p.data.clone() for p in params]

    T = T_initial
    total_accepted = 0
    total_proposed = 0

    for step in range(n_steps):
        # Update temperature
        if cooling_schedule == "exponential":
            T = T_initial * (cooling_rate ** step)
        else:
            T = T_initial - (T_initial - T_final) * (step / n_steps)

        # Zero gradients and compute loss
        for p in params:
            if p.grad is not None:
                p.grad.zero_()

        loss = loss_fn()
        loss.backward()

        # Generate perturbations and apply based on gradient
        step_accepted = 0
        step_proposed = 0

        if per_parameter:
            # Per-parameter acceptance
            with torch.no_grad():
                for i, param in enumerate(params):
                    if param.grad is None:
                        continue

                    # Generate perturbation
                    if is_angle[i]:
                        delta = torch.randn_like(param) * (scale * 0.1)
                    else:
                        delta = torch.randn_like(param) * (scale * 0.01)

                    # Compute g · δ for each element
                    g_dot_delta = param.grad * delta

                    # Metropolis criterion per element
                    # Accept if g·δ < 0 (downhill) or with prob exp(-g·δ/T)
                    accept_prob = torch.where(
                        g_dot_delta < 0,
                        torch.ones_like(g_dot_delta),
                        torch.exp(-g_dot_delta / T)
                    )
                    accept_mask = torch.rand_like(accept_prob) < accept_prob

                    # Apply accepted perturbations
                    param.data.add_(delta * accept_mask.float())

                    # Wrap angles
                    if is_angle[i]:
                        param.data.copy_(_wrap_torsions(param.data))

                    step_accepted += accept_mask.sum().item()
                    step_proposed += param.numel()
        else:
            # Batch acceptance: sum of g·δ for all params
            with torch.no_grad():
                deltas = []
                total_g_dot_delta = 0.0

                for i, param in enumerate(params):
                    if param.grad is None:
                        deltas.append(None)
                        continue

                    if is_angle[i]:
                        delta = torch.randn_like(param) * (scale * 0.1)
                    else:
                        delta = torch.randn_like(param) * (scale * 0.01)

                    deltas.append(delta)
                    total_g_dot_delta += (param.grad * delta).sum().item()

                # Single Metropolis decision
                step_proposed = 1
                accept = False
                if total_g_dot_delta < 0:
                    accept = True
                elif T > 0 and torch.rand(1).item() < math.exp(-total_g_dot_delta / T):
                    accept = True

                if accept:
                    step_accepted = 1
                    for i, (param, delta) in enumerate(zip(params, deltas)):
                        if delta is not None:
                            param.data.add_(delta)
                            if is_angle[i]:
                                param.data.copy_(_wrap_torsions(param.data))

        total_accepted += step_accepted
        total_proposed += step_proposed

        # Evaluate actual loss periodically (for tracking, not for acceptance)
        if (step + 1) % max(1, n_steps // 20) == 0 or step == n_steps - 1:
            with torch.no_grad():
                current_loss = loss_fn().item()

            if current_loss < best_loss:
                best_loss = current_loss
                best_params = [p.data.clone() for p in params]

        # Progress logging
        if verbose > 0 and (step + 1) % max(1, n_steps // 10) == 0:
            accept_rate = total_accepted / max(1, total_proposed)
            with torch.no_grad():
                current_loss = loss_fn().item()
            print(f"Step {step+1}/{n_steps}, T={T:.4f}, Loss={current_loss:.4f}, "
                  f"Best={best_loss:.4f}, Accept={accept_rate:.1%}")

        # Callback
        if callback is not None:
            accept_rate = total_accepted / max(1, total_proposed)
            callback(step, T, current_loss, accept_rate)

    # Restore best parameters
    with torch.no_grad():
        for param, best in zip(params, best_params):
            param.data.copy_(best)
        final_loss = loss_fn().item()

    if verbose > 0:
        final_accept = total_accepted / max(1, total_proposed)
        print(f"\nGradient SA Complete:")
        print(f"  Best loss: {best_loss:.4f}")
        print(f"  Final loss: {final_loss:.4f}")
        print(f"  Overall acceptance: {final_accept:.1%}")

    return best_loss, final_loss


def refine_sa_lbfgs(
    model,
    loss_fn: Callable[[], torch.Tensor],
    n_sa_steps: int = 5000,
    sa_scale: float = 1.0,
    sa_T_initial: Optional[float] = None,
    sa_T_final: float = 0.01,
    sa_params: Optional[List[str]] = None,
    n_lbfgs_cycles: int = 20,
    lbfgs_lr: float = 1.0,
    lbfgs_max_iter: int = 20,
    verbose: int = 1,
    sa_callback: Optional[Callable] = None,
    lbfgs_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Combined SA + LBFGS pipeline for internal coordinate refinement.

    Pipeline:
    1. SA phase: Global exploration using internal coordinates (torsions by default)
    2. LBFGS phase: Local optimization to refine best structure

    Parameters
    ----------
    model : ModelFT
        Model with internal coordinates enabled (model.use_internal_coordinates()).
    loss_fn : callable
        Loss function that returns a scalar tensor: loss = loss_fn()
    n_sa_steps : int
        Number of SA steps. Default 5000.
    sa_scale : float
        SA perturbation scale. Default 1.0.
    sa_T_initial : float, optional
        Initial SA temperature. Auto-calibrated if None.
    sa_T_final : float
        Final SA temperature. Default 0.01.
    sa_params : List[str], optional
        Which internal coord params to use for SA. Options:
        'torsions', 'bond_lengths', 'angles', 'segment_positions', 'segment_orientations'
        Default: ['torsions'] (torsion-only exploration)
    n_lbfgs_cycles : int
        Number of LBFGS optimization cycles. Default 20.
    lbfgs_lr : float
        LBFGS learning rate. Default 1.0.
    lbfgs_max_iter : int
        Max iterations per LBFGS cycle. Default 20.
    verbose : int
        Verbosity level. Default 1.
    sa_callback : callable, optional
        Callback for SA phase: callback(step, T, loss, best_loss)
    lbfgs_callback : callable, optional
        Callback for LBFGS phase: callback(cycle, loss)

    Returns
    -------
    dict with keys:
        'initial_loss': Loss before any optimization
        'sa_loss': Loss after SA phase
        'final_loss': Loss after LBFGS phase
        'sa_history': SA history if callback provided
        'lbfgs_history': LBFGS history if callback provided
    """
    # Check model has internal coordinates
    ic = model.xyz
    if not hasattr(ic, 'torsions'):
        raise ValueError(
            "Model must have internal coordinates enabled. "
            "Call model.use_internal_coordinates() first."
        )

    # Default: torsion-only SA
    if sa_params is None:
        sa_params = ['torsions']

    # Build parameter list
    param_map = {
        'torsions': ic.torsions,
        'bond_lengths': ic.bond_lengths,
        'angles': ic.angles,
        'segment_positions': ic.segment_positions,
        'segment_orientations': ic.segment_orientations,
    }

    params = []
    for name in sa_params:
        if name not in param_map:
            raise ValueError(f"Unknown param type: {name}. "
                           f"Options: {list(param_map.keys())}")
        p = param_map[name]
        if p.numel() > 0:
            params.append(p)

    if len(params) == 0:
        raise ValueError("No parameters selected for SA optimization")

    # Setup LossState for SA
    state = LossState(device=next(model.parameters()).device)
    state.register_target("combined", loss_fn)

    # Initial loss
    with torch.no_grad():
        initial_loss = loss_fn().item()

    if verbose > 0:
        print("=" * 60)
        print("SA + LBFGS Refinement Pipeline")
        print("=" * 60)
        print(f"Initial loss: {initial_loss:.4f}")
        print(f"\nSA Phase: {n_sa_steps} steps, scale={sa_scale}")
        print(f"  Optimizing: {sa_params}")

    # ============================================
    # Phase 1: Simulated Annealing
    # ============================================
    sa_history = []

    def _sa_callback(step, T, loss, best_loss):
        sa_history.append({'step': step, 'T': T, 'loss': loss, 'best': best_loss})
        if sa_callback is not None:
            sa_callback(step, T, loss, best_loss)

    state, sa_best_loss = optimize_internal_coord_sa(
        state=state,
        params=params,
        scale=sa_scale,
        T_initial=sa_T_initial,
        T_final=sa_T_final,
        n_steps=n_sa_steps,
        verbose=verbose,
        callback=_sa_callback,
    )

    if verbose > 0:
        print(f"\nSA Complete: {initial_loss:.4f} -> {sa_best_loss:.4f}")

    # ============================================
    # Phase 2: LBFGS Local Optimization
    # ============================================
    if verbose > 0:
        print(f"\nLBFGS Phase: {n_lbfgs_cycles} cycles")

    # Use all internal coord parameters for LBFGS
    all_params = [p for p in param_map.values() if p.numel() > 0]

    optimizer = torch.optim.LBFGS(
        all_params,
        lr=lbfgs_lr,
        max_iter=lbfgs_max_iter,
        history_size=5,
        line_search_fn="strong_wolfe",
    )

    lbfgs_history = []

    def closure():
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        return loss

    for cycle in range(n_lbfgs_cycles):
        optimizer.step(closure)

        with torch.no_grad():
            current_loss = loss_fn().item()

        lbfgs_history.append({'cycle': cycle, 'loss': current_loss})

        if lbfgs_callback is not None:
            lbfgs_callback(cycle, current_loss)

        if verbose > 1:
            print(f"  LBFGS Cycle {cycle+1}/{n_lbfgs_cycles}: Loss={current_loss:.4f}")

    with torch.no_grad():
        final_loss = loss_fn().item()

    if verbose > 0:
        print(f"LBFGS Complete: {sa_best_loss:.4f} -> {final_loss:.4f}")
        print(f"\n{'='*60}")
        print(f"Total improvement: {initial_loss:.4f} -> {final_loss:.4f}")
        improvement = (initial_loss - final_loss) / initial_loss * 100
        print(f"  ({improvement:.1f}% reduction)")
        print("=" * 60)

    return {
        'initial_loss': initial_loss,
        'sa_loss': sa_best_loss,
        'final_loss': final_loss,
        'sa_history': sa_history,
        'lbfgs_history': lbfgs_history,
    }
