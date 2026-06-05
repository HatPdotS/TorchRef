"""
LossState - Hierarchical loss computation with lazy evaluation.

Design:
- Targets stored as callables, evaluated only on aggregation
- Hierarchical naming via '/' separator (e.g., 'geometry/bond', 'adp/simu')
- Weights computed at initialization, not recalculated each pass
- Internal history logging for debugging/analysis
- Aggregation handled directly in this class
- Loss functions with zeroed weight are not evaluated at all.
- Implements optimization closures and data handling
- Automatically detects parameter root tensors and disables requires_grad on any that the loss touches but the optimizer wasn't built to update, enabling dynamic caching.



Example:
    state = LossState(device)
    state.register_target('xray/work', xray_work_target)
    state.register_target('geometry/bond', bond_target)
    state.set_weight('xray', 1.0)
    state.set_weight('geometry', 0.5)
    state.set_weight('geometry/bond', 1.0)

    total = state.aggregate()  # Evaluates targets, applies hierarchical weights

    state.step(optimizer)  # Runs optimizer step with closure that validates loss and auto-freezes parameters depending on the loss graph.
"""

import warnings
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

import torch
from torch import nn

from torchref.config import get_default_device
from torchref.utils.autograd_introspection import collect_loss_leaves, _iter_roots
from torchref.utils.device_mixin import DeviceMovementMixin
from torchref.utils.loss_validation import validate_loss


class LossStateWarning(UserWarning):
    """Performance hints emitted by :class:`LossState`.

    Subclassed from ``UserWarning`` so it shows up by default, but exposed
    as a distinct category so callers can silence/escalate it independently.
    """


@dataclass
class LossState(DeviceMovementMixin):
    """
    Hierarchical loss state with lazy evaluation.

    Attributes
    ----------
    device : torch.device
        Computation device.
    targets : Dict[str, Callable]
        Target functions keyed by hierarchical name (e.g., 'geometry/bond').
    weights : Dict[str, float]
        Weights keyed by name. Can be group weights ('geometry') or
        component weights ('geometry/bond').
    history : List[Dict]
        Log of computed values per aggregation call.
    meta : Dict[str, Any]
        Model-level data (rwork, rfree, n_atoms, etc.) populated by refinement.
    """

    device: torch.device = field(default_factory=get_default_device)

    # Targets as callables - only evaluated on aggregate()
    targets: Dict[str, Callable] = field(default_factory=dict)

    # Weights - computed at init, hierarchical via naming
    weights: Dict[str, float] = field(default_factory=dict)

    # History log
    history: List[Dict[str, Any]] = field(default_factory=list)

    # Cache for computed losses (cleared on each aggregate)
    _losses: Dict[str, torch.Tensor] = field(default_factory=dict, repr=False)

    # Set of target keys marked as compilable
    _compilable: Set[str] = field(default_factory=set, repr=False)

    # Cached compiled callable; None until compile_aggregate() is called
    _compiled_aggregate: Optional[Callable] = field(default=None, repr=False)

    # Union of leaf nn.Parameters that registered targets' backward will
    # accumulate into. Populated incrementally during register_target via a
    # one-shot probe forward + autograd graph walk. Used by step()/run() to
    # diff against the optimizer's intent and disable requires_grad on the
    # leaves the loss touches but the optimizer wasn't built to update.
    _loss_leaves: Set[nn.Parameter] = field(default_factory=set, repr=False)

    # Submodules attached to registered targets that expose a reset_cache
    # method (e.g. ModelFT and its CachedForwardMixin wrappers). Collected
    # once at registration time and reset after every step() so that
    # validate_loss-rejected closures or stale forward-cache entries can't
    # silently poison the next forward.
    _resettable_modules: List[nn.Module] = field(default_factory=list, repr=False)

    # Model-level data for weighting schemes
    meta: Dict[str, Any] = field(default_factory=dict)

    # =========================================================================
    # Item Access (meta and _losses)
    # =========================================================================

    def __getitem__(self, key: str) -> Any:
        """
        Get value from meta or _losses by key.

        Parameters
        ----------
        key : str
            Key to look up. Checks meta first, then _losses.

        Returns
        -------
        Any
            Value from meta or _losses.

        Raises
        ------
        KeyError
            If key not found in either dict.
        """
        if key in self.meta:
            return self.meta[key]
        if key in self._losses:
            return self._losses[key]
        raise KeyError(f"Key '{key}' not found in meta or _losses")

    def __contains__(self, key: str) -> bool:
        """Check if key exists in meta or _losses."""
        return key in self.meta or key in self._losses

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value with default fallback.

        Parameters
        ----------
        key : str
            Key to look up.
        default : Any
            Value to return if key not found.

        Returns
        -------
        Any
            Value from meta, _losses, or default.
        """
        if key in self.meta:
            return self.meta[key]
        if key in self._losses:
            return self._losses[key]
        return default

    def cache_losses(self, force: bool = False) -> "LossState":
        """
        Cache all target losses.

        Evaluates all registered targets and stores results in _losses.

        Parameters
        ----------
        force : bool
            If True, re-evaluate all targets even if already cached.

        Returns
        -------
        LossState
            Self for chaining.
        """
        if force:
            self._losses.clear()

        for name, target in self.targets.items():
            if name not in self._losses:
                self._losses[name] = target()

        return self

    def update_meta(self, data: Dict[str, Any]) -> "LossState":
        """
        Update meta dict with model-level data.

        Parameters
        ----------
        data : Dict[str, Any]
            Data to add to meta.

        Returns
        -------
        LossState
            Self for chaining.
        """
        self.meta.update(data)
        return self

    # =========================================================================
    # Target Registration
    # =========================================================================

    def register_target(
        self,
        name: str,
        target: Callable,
        prefix: str = None,
        compile: bool = False,
        probe: bool = True,
    ) -> "LossState":
        """
        Register a target function.

        Automatically detects combined targets (like TotalGeometryTarget,
        TotalADPTarget) and expands them into their component targets.

        Parameters
        ----------
        name : str
            Hierarchical name (e.g., 'geometry/bond', 'adp/simu').
        target : Callable
            Function that returns a loss tensor when called. Can also be a
            combined target with .items() method, which will be auto-expanded.
        prefix : str, optional
            Prefix to prepend to the name (e.g., 'model1' -> 'model1/geometry/bond').
            Useful for registering targets from multiple models in the same state.
        compile : bool
            If True, mark this target (or all its sub-targets if combined) as
            eligible for the compiled aggregate closure built by compile_aggregate().
        probe : bool
            If True (default), run the target's forward once, walk the
            autograd graph, and merge the resulting leaf set into
            ``self._loss_leaves``. The target's dependencies (model loaded,
            data attached, etc.) must therefore be in place before
            registration. Set ``probe=False`` to skip — the leaf-set entry
            for this target will be empty, so :meth:`step`/:meth:`run` will
            not auto-disable any leaves on its account. Useful only for
            targets whose forward genuinely cannot be called at registration
            time.

        Returns
        -------
        LossState
            Self for chaining.
        """
        self._compiled_aggregate = None  # invalidate stale compiled closure

        # Check if target is a combined/dictionary-like target with .items()
        # This handles CombinedTargets, TotalGeometryTarget, TotalADPTarget, etc.
        if hasattr(target, "items") and callable(getattr(target, "items", None)):
            # Use name as prefix to maintain hierarchy (e.g., "geometry" -> "geometry/bond")
            combined_prefix = f"{prefix}/{name}" if prefix else name
            return self.register_targets(
                target, prefix=combined_prefix, compile=compile, probe=probe
            )

        # Normal single target registration
        key = f"{prefix}/{name}" if prefix else name
        self.targets[key] = target
        if compile:
            self._compilable.add(key)
        if probe:
            self._probe_and_merge_leaves(target)
        self._collect_resettable_modules(target)
        return self

    def _collect_resettable_modules(self, target: Callable) -> None:
        """Walk ``target``'s submodules and collect any that expose a
        ``reset_cache`` method, deduplicating against modules already
        captured from earlier registrations.

        These modules are reset after every :meth:`step` call so that a
        ``validate_loss``-rejected closure or a stale forward cache cannot
        silently poison the next aggregate.
        """
        if not isinstance(target, nn.Module):
            return
        # Use object identity for dedup; nn.Modules aren't hashable by
        # default in a way that matches identity, so iterate.
        seen_ids = {id(m) for m in self._resettable_modules}
        for module in target.modules():
            method = getattr(module, "reset_cache", None)
            if callable(method) and id(module) not in seen_ids:
                self._resettable_modules.append(module)
                seen_ids.add(id(module))

    def _probe_and_merge_leaves(self, target: Callable) -> None:
        """Run ``target()`` once with grad enabled, walk the autograd graph,
        and union the resulting leaves into ``self._loss_leaves``.

        ``target()`` may return a Tensor, a tuple/list of tensors, or a dict
        of tensors — :func:`collect_loss_leaves` handles all three.

        Convention: targets are expected to be probed while every parameter
        the loss should track has ``requires_grad=True``. If the probe walks
        the full graph and finds zero leaves — meaning every root is either
        constant (``grad_fn is None``) or only depends on tensors with
        ``requires_grad=False`` — emit a warning. This is almost never what
        the caller intended (a target with no trainable parameters
        contributes nothing to gradient-based optimization). It is *not*
        an error: missing leaves only cost a bit of extra backward work
        inside the optimization loop, never correctness.
        """
        with torch.enable_grad():
            roots = target()
        new_leaves = collect_loss_leaves(roots)
        self._loss_leaves |= new_leaves

    def register_targets(
        self,
        targets,
        prefix: str = None,
        compile: bool = False,
        probe: bool = True,
    ) -> "LossState":
        """Register multiple targets from a component target or dict.

        For targets with a .name attribute, uses target.name as the key.
        For plain callables, uses the dict key.

        Parameters
        ----------
        targets : dict
            Dictionary of name -> target mappings.
        prefix : str, optional
            Prefix to prepend to all target names.
        compile : bool
            If True, propagate the compile flag to all sub-targets.
        probe : bool
            Forwarded to :meth:`register_target`.
        """
        for name, target in targets.items():
            target_name = getattr(target, "name", name)
            self.register_target(
                target_name, target, prefix=prefix, compile=compile, probe=probe
            )
        return self

    # =========================================================================
    # Weight Management
    # =========================================================================

    def set_weight(self, name: str, weight: float) -> "LossState":
        """
        Set a weight value.

        Parameters
        ----------
        name : str
            Weight name. Can be a group ('geometry') or component ('geometry/bond').
        weight : float
            Weight value.

        Returns
        -------
        LossState
            Self for chaining.
        """
        self.weights[name] = weight
        self._compiled_aggregate = (
            None  # invalidate stale compiled closure (weights baked in)
        )
        return self

    def set_weights(self, weights: Dict[str, float]) -> "LossState":
        """Set multiple weights."""
        for name, weight in weights.items():
            self.set_weight(name, weight)
        return self

    def get_weight(self, name: str, default: float = 1.0) -> float:
        """Get a weight value."""
        return self.weights.get(name, default)

    def get_effective_weight(self, name: str) -> float:
        """
        Get effective weight for a target, including group weights.

        For 'geometry/bond', returns: weights['geometry'] * weights['geometry/bond']
        Missing weights default to 1.0.

        Parameters
        ----------
        name : str
            Target name (e.g., 'geometry/bond').

        Returns
        -------
        float
            Product of all hierarchical weights.
        """
        parts = name.split("/")
        effective = 1.0

        # Apply weights at each level
        path = ""
        for part in parts:
            path = f"{path}/{part}" if path else part
            effective *= self.weights.get(path, 1.0)

        return effective

    # =========================================================================
    # Compiled Aggregate
    # =========================================================================

    def mark_compilable(self, names: List[str]) -> "LossState":
        """
        Mark already-registered targets as eligible for the compiled aggregate.

        Parameters
        ----------
        names : List[str]
            Target keys to mark (must already be registered).

        Returns
        -------
        LossState
            Self for chaining.
        """
        for name in names:
            if name in self.targets:
                self._compilable.add(name)
        self._compiled_aggregate = None
        return self

    def compile_aggregate(self, **compile_kwargs) -> "LossState":
        """
        Build and cache a torch.compile'd closure over all compilable targets.

        Must be called after all targets and weights have been registered.
        Re-call if weights or compilable targets change (or call reset_compiled_aggregate()).

        Parameters
        ----------
        **compile_kwargs
            Keyword arguments forwarded to torch.compile. Defaults to
            fullgraph=False so partial-graph fallback is allowed.

        Returns
        -------
        LossState
            Self for chaining.
        """
        compile_kwargs.setdefault("fullgraph", False)

        active = [
            (self.targets[n], self.get_effective_weight(n))
            for n in self.targets
            if n in self._compilable and self.get_effective_weight(n) != 0.0
        ]
        if not active:
            self._compiled_aggregate = None
            return self

        fns, weights = zip(*active)
        fns, weights = list(fns), list(weights)
        device = self.device

        def _compiled_fn():
            total = torch.tensor(0.0, device=device)
            for fn, w in zip(fns, weights):
                total = total + w * fn()
            return total

        self._compiled_aggregate = torch.compile(_compiled_fn, **compile_kwargs)
        return self

    def reset_compiled_aggregate(self) -> "LossState":
        """Clear the cached compiled closure (e.g. after changing weights)."""
        self._compiled_aggregate = None
        return self

    # =========================================================================
    # History Logging
    # =========================================================================

    def log(self, name: str, value: Any) -> None:
        """
        Log a value to the current history entry.

        Creates a new history entry if needed.

        Parameters
        ----------
        name : str
            Key for the logged value.
        value : Any
            Value to log. Tensors are converted to Python floats.
        """
        # Ensure we have a current entry
        if not self.history:
            self.history.append({})

        # Convert tensor to float
        if isinstance(value, torch.Tensor):
            value = value.detach().item()

        self.history[-1][name] = value

    def new_entry(self) -> None:
        """Start a new history entry."""
        self.history.append({})

    def get_history(self, name: str) -> List[Any]:
        """Get all logged values for a key across history."""
        return [entry.get(name) for entry in self.history if name in entry]

    # =========================================================================
    # Aggregation
    # =========================================================================

    def aggregate(self, log_values: bool = False) -> torch.Tensor:
        """
        Evaluate all targets and compute weighted sum.

        When compile_aggregate() has been called and log_values=False, the
        compilable targets are evaluated through a single torch.compile'd closure
        for improved performance.  With log_values=True all targets run eagerly
        so per-target losses are available in _losses.

        Parameters
        ----------
        log_values : bool
            If True, log all losses, weights, and total to history.

        Returns
        -------
        torch.Tensor
            Total weighted loss.
        """
        if log_values:
            self.new_entry()

        self._losses.clear()
        total = torch.tensor(0.0, device=self.device)

        # --- compiled group ---
        # Skipped when log_values=True: the fused closure does not expose
        # per-target losses needed for logging.
        if self._compiled_aggregate is not None and not log_values:
            total = total + self._compiled_aggregate()
        else:
            # Run compilable targets eagerly (log_values path or no compiled fn)
            for name in self._compilable:
                if name not in self.targets:
                    continue
                weight = self.get_effective_weight(name)
                if weight == 0.0:
                    continue
                loss = self.targets[name]()
                self._losses[name] = loss
                weighted = weight * loss
                total = total + weighted
                if log_values:
                    self.log(f"loss/{name}", loss)
                    self.log(f"weight/{name}", weight)
                    self.log(f"weighted/{name}", weighted)

        # --- eager group (non-compilable) ---
        for name, target in self.targets.items():
            if name in self._compilable:
                continue  # already handled above
            weight = self.get_effective_weight(name)
            if weight == 0.0:
                continue
            loss = target()
            self._losses[name] = loss
            weighted = weight * loss
            total = total + weighted
            if log_values:
                self.log(f"loss/{name}", loss)
                self.log(f"weight/{name}", weight)
                self.log(f"weighted/{name}", weighted)

        if log_values:
            self.log("total", total)

        return total

    def get_loss(self, name: str) -> Optional[torch.Tensor]:
        """
        Get a cached loss value (after aggregate() was called).

        Parameters
        ----------
        name : str
            Target name.

        Returns
        -------
        torch.Tensor or None
            Cached loss, or None if not computed.
        """
        return self._losses.get(name)

    # =========================================================================
    # Optimization (step / run / active-parameter introspection)
    # =========================================================================

    def active_parameters(self) -> Set[nn.Parameter]:
        """Return the set of leaf ``nn.Parameter``s that registered targets'
        backward passes will accumulate gradient into.

        Populated incrementally by :meth:`register_target` via a one-shot
        probe forward + autograd graph walk — calling this method does not
        run any forward, walk any graph, or evaluate any target. The result
        is conservative: a target whose weight is later set to 0 still
        contributes its leaves here, which is harmless for the freezing
        logic in :meth:`step` (it can only over-freeze, never under-freeze).
        """
        return self._loss_leaves

    def refresh_loss_leaves(self) -> "LossState":
        """Re-probe every registered target and rebuild ``_loss_leaves``
        and the resettable-modules cache.

        Use this after external code has replaced parameter identity on the
        underlying model — for example after :meth:`Model.freeze` /
        :meth:`Model.unfreeze` (which rebuild ``refinable_params`` tensors).
        Under normal :meth:`step`/:meth:`run` usage no parameter identity
        ever changes, so this method is rarely needed.
        """
        self._loss_leaves = set()
        self._resettable_modules = []
        for target in self.targets.values():
            self._probe_and_merge_leaves(target)
            self._collect_resettable_modules(target)
        return self

    def reset_caches(self) -> None:
        """Call ``reset_cache()`` on every registered target's submodules
        that expose one. Invoked automatically at the end of :meth:`step`.
        """
        for module in self._resettable_modules:
            module.reset_cache()

    def restore_loss_leaf_grads(self) -> None:
        """Unconditionally re-enable ``requires_grad`` on every leaf in
        ``self._loss_leaves``. Called at the end of :meth:`step` so the
        next call sees a clean, fully-differentiable model regardless of
        what state the previous step (or external code) left things in.
        """
        for p in self._loss_leaves:
            if not p.requires_grad:
                p.requires_grad_(True)

    def run(
        self,
        optimizer: torch.optim.Optimizer,
        log=False,
        nsteps: int = 1,
        *,
        context: str = "loss_state.step",
    ) -> Optional[torch.Tensor]:
        """Run a single ``optimizer.step(closure)``.

        Builds the closure, validates each loss for finiteness via
        :func:`torchref.utils.validate_loss`, and on failure zeros the
        gradients and returns ``+inf`` so the strong-Wolfe line search
        backtracks. Automatically disables ``requires_grad`` on any leaf
        that the loss touches but the optimizer was not constructed with —
        autograd then prunes those subgraphs from the backward pass.

        Technically this should work with all optimzers in pytorch that support closures but it has only been tested for LBFGS so far. The closure is built to be as general as possible, so if you have a custom optimizer that supports closures it should "just work" with this method.

        Every collected ``reset_cache``-bearing submodule is reset
        **before** the optimizer step so the closure's first forward sees
        a clean cache (a previous rejected closure may have stored a
        NaN/inf forward result that the fingerprint would happily serve
        again if parameter values haven't changed).

        After the the run we call maintenance on all targets.

        On exit, ``requires_grad=True`` is unconditionally re-enabled on
        every leaf in ``self._loss_leaves`` — defending against state
        bleeding between successive refinement methods.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer to step. Its ``param_groups`` define the *intent* —
            the leaves the caller actually wants to update.
        log : bool
            If True, calls ``aggregate(log_values=True)`` before and after the optimization loop
        nsteps : int
            Number of steps to run (default 1). Only the first step's closure caching is enabled between multiple steps.
            If you want to run truly independent steps, call this method multiple times with nsteps=1. This adds overhead but might be desirable if the overhead is negligible anyway.
        context : str
            Diagnostic label forwarded to ``validate_loss``.

        Returns
        -------
        torch.Tensor or None
            The loss tensor from the last accepted closure call, or ``None``
            if no closure call succeeded (every call produced non-finite
            loss).
        """

        params = list(_optimizer_param_set(optimizer))
        last_loss: Dict[str, Optional[torch.Tensor]] = {"val": None}
        # Device/dtype for the +inf sentinel returned when a trial step is
        # rejected before a loss value exists (linalg op raised on non-finite
        # input — see below).
        _ref = params[0] if params else None
        _inf_dev = _ref.device if _ref is not None else self.device
        _inf_dtype = _ref.dtype if _ref is not None else torch.get_default_dtype()
        _warned_linalg = {"done": False}

        def _reject():
            """Zero grads and return +inf so strong-Wolfe backtracks the step."""
            for p in params:
                if p.grad is not None:
                    p.grad.zero_()
            return torch.full((), float("inf"), device=_inf_dev, dtype=_inf_dtype)

        def closure():
            optimizer.zero_grad()
            try:
                loss = self.aggregate()
                loss.backward()
            except (torch._C._LinAlgError, RuntimeError) as exc:
                # The value-based gate below only sees losses that are
                # *returned*; a few linalg ops (svd/eig/cholesky/inv) instead
                # *raise* on non-finite input. When strong-Wolfe probes an
                # overshooting trial point that sends parameters to inf, such an
                # op throws here, bypassing validate_loss. Treat it exactly like
                # a non-finite loss: reject the step (+inf) so the line search
                # backtracks, instead of letting the exception kill refinement.
                if not _warned_linalg["done"]:
                    warnings.warn(
                        f"LossState.run({context!r}): a linear-algebra op raised "
                        f"during a trial step ({type(exc).__name__}: {exc}); the "
                        "parameters likely diverged to non-finite values. "
                        "Rejecting the step (+inf) so the optimizer backtracks. "
                        "Further occurrences this step are suppressed.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    _warned_linalg["done"] = True
                return _reject()
            ok = validate_loss(
                loss,
                state=self,
                parameters=params,
                context=context,
                raise_on_fail=False,
            )
            if not ok:
                for p in params:
                    if p.grad is not None:
                        p.grad.zero_()
                return torch.full_like(loss.detach(), float("inf"))
            last_loss["val"] = loss
            return loss

        if log:
            self.aggregate(log_values=True)

        # Clear forward caches BEFORE the step so the closure's first
        # forward starts from a known-clean state — a previous rejected
        # closure may have left a NaN/inf cached fcalc that the fingerprint
        # would otherwise serve again unchanged. This helps with robustness but "should" not be necessary.
        self.reset_caches()
        try:
            with _freeze_graph_extras(self, optimizer):
                for i in range(nsteps):
                    optimizer.step(closure)
        finally:
            # Re-enable grads on every loss leaf regardless of how the
            # step exited. Defends against state bleeding between
            # successive refinement methods.
            self.restore_loss_leaf_grads()

        # Post-step maintenance hook: each target decides whether its
        # internal state is stale (e.g. NonBondedTarget rebuilds the VDW
        # pair list when atoms have drifted too far since the last
        # build). Targets that don't care inherit the no-op default
        # from ``Target.maintenance``.
        for target in self.targets.values():
            maint = getattr(target, "maintenance", None)
            if callable(maint):
                maint()

        if log:
            self.aggregate(log_values=True)

        return last_loss["val"]

    def step(self, optimizer: torch.optim.Optimizer, *args, **kwargs) -> "LossState":
        """Convenience method that calls :meth:`run` with 1 step.


        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer to run.
        *args, **kwargs
            Forwarded to :meth:`run`.
        """
        return self.run(optimizer, *args, nsteps=1, **kwargs)

    # =========================================================================
    # Breakdown / Analysis
    # =========================================================================

    def get_breakdown(self) -> Dict[str, Dict[str, Any]]:
        """
        Get breakdown of losses by group.

        Returns
        -------
        Dict
            Nested dict: {group: {component: {'loss': ..., 'weight': ..., 'weighted': ...}}}
        """
        breakdown = defaultdict(dict)

        for name, loss in self._losses.items():
            parts = name.split("/")
            group = parts[0] if len(parts) > 1 else "root"
            component = "/".join(parts[1:]) if len(parts) > 1 else parts[0]

            weight = self.get_effective_weight(name)

            breakdown[group][component] = {
                "loss": loss.item() if isinstance(loss, torch.Tensor) else loss,
                "weight": weight,
                "weighted": (
                    (weight * loss).item()
                    if isinstance(loss, torch.Tensor)
                    else weight * loss
                ),
            }

        return dict(breakdown)

    def get_group_totals(self) -> Dict[str, float]:
        """
        Get total weighted loss per group.

        Returns
        -------
        Dict[str, float]
            {group_name: total_weighted_loss}
        """
        totals = defaultdict(float)

        for name, loss in self._losses.items():
            parts = name.split("/")
            group = parts[0]
            weight = self.get_effective_weight(name)
            weighted = (
                (weight * loss).item()
                if isinstance(loss, torch.Tensor)
                else weight * loss
            )
            totals[group] += weighted

        return dict(totals)

    def format_breakdown(self) -> str:
        """Return per-target loss / weight / weighted / finite as a string.

        One row per target currently in ``self._losses`` (populated by the
        most recent eager ``aggregate()`` call). Used by both ``summary()``
        and :func:`torchref.utils.validate_loss` so the diagnostic format
        does not drift.
        """
        lines = []
        for name, loss in self._losses.items():
            weight = self.get_effective_weight(name)
            try:
                loss_val = loss.item() if torch.is_tensor(loss) else float(loss)
            except Exception:
                loss_val = float("nan")
            weighted_val = weight * loss_val
            is_finite = loss_val == loss_val and abs(loss_val) != float("inf")
            finite_flag = "yes" if is_finite else "NO "
            lines.append(
                f"  {name:<32} w={weight:>9.4g}  "
                f"loss={loss_val:>14.6g}  "
                f"weighted={weighted_val:>14.6g}  {finite_flag}"
            )
        return "\n".join(lines)

    def summary(self) -> None:
        """Print a per-target loss breakdown to stdout."""
        print("LossState Summary:")
        print(self.format_breakdown())

    # =========================================================================
    # Device Management
    # =========================================================================

    def to(self, *args, **kwargs):
        """Move via :class:`DeviceMixin`; honour an explicit device when no tensors exist yet."""
        result = super().to(*args, **kwargs)
        # If no tensor was found to refresh ``self.device``, fall back to the
        # explicit device argument so subsequent allocations land correctly.
        if not isinstance(result.device, torch.device):
            from torchref.utils.device_mixin import _parse_to_args

            device, _ = _parse_to_args(args, kwargs)
            if device is not None:
                result.device = (
                    torch.device(device)
                    if not isinstance(device, torch.device)
                    else device
                )
        return result

    # =========================================================================
    # Utility
    # =========================================================================

    def clear(self) -> "LossState":
        """Clear cached losses (not targets or weights)."""
        self._losses.clear()
        return self

    def clear_history(self) -> "LossState":
        """Clear history log."""
        self.history.clear()
        return self

    def __repr__(self) -> str:
        n_targets = len(self.targets)
        n_weights = len(self.weights)
        n_history = len(self.history)
        n_meta = len(self.meta)
        return f"LossState(device={self.device}, targets={n_targets}, weights={n_weights}, meta={n_meta}, history={n_history})"


def _optimizer_param_set(optimizer: torch.optim.Optimizer) -> Set[nn.Parameter]:
    """Flatten an optimizer's param_groups into a set."""
    return {p for g in optimizer.param_groups for p in g["params"]}


@contextmanager
def _freeze_graph_extras(state: "LossState", optimizer: torch.optim.Optimizer):
    """Disable ``requires_grad`` on leaves that ``state`` touches but
    ``optimizer`` was not constructed with.

    Reads the cached leaf union via :meth:`LossState.active_parameters` —
    no probe forward is run here. Restoration is handled by the enclosing
    :meth:`LossState.step`'s ``finally`` clause, which unconditionally
    re-enables ``requires_grad`` on every leaf in ``self._loss_leaves``
    (not just the ones we disabled here). That avoids subtle bugs where a
    pre-frozen leaf or a leaf disabled by an unrelated code path leaks
    into the next step.
    """
    intended = _optimizer_param_set(optimizer)
    for p in state.active_parameters():
        if p not in intended and p.requires_grad:
            p.requires_grad_(False)
    yield


def create_loss_state(
    device: torch.device,
    targets: Dict[str, Callable] = None,
    weights: Dict[str, float] = None,
) -> LossState:
    """
    Factory function to create a LossState.

    Parameters
    ----------
    device : torch.device
        Computation device.
    targets : Dict[str, Callable], optional
        Initial targets to register.
    weights : Dict[str, float], optional
        Initial weights to set.

    Returns
    -------
    LossState
        Configured LossState instance.
    """
    state = LossState(device=device)

    if targets:
        state.register_targets(targets)
    if weights:
        state.set_weights(weights)

    return state


__all__ = ["LossState", "create_loss_state"]
