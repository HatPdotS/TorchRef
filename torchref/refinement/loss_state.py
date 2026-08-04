"""LossState -- hierarchical loss computation with lazy evaluation.

Targets are stored as callables and evaluated only on aggregation, keyed by a
hierarchical '/' name (e.g. 'geometry/bond'). Weights apply per group or component;
targets with a zeroed effective weight are skipped. Also provides the optimization
closure used by :meth:`LossState.run`, which auto-freezes leaves the loss touches but
the optimizer was not built to update::

    state.register_target('geometry/bond', bond_target)
    state.set_weight('geometry', 0.5)
    total = state.aggregate()   # evaluate, apply hierarchical weights
    state.step(optimizer)       # step with a loss-validating closure
"""

import warnings
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

import torch
from torch import nn

from torchref.config import canonical_device, get_default_device
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
    """Hierarchical loss state with lazy evaluation.

    Several underscored fields are load-bearing: ``_losses`` caches the per-target values
    that :meth:`__getitem__`/:meth:`get`/:meth:`get_loss` read, and ``_loss_leaves`` /
    ``_resettable_modules`` drive the parameter auto-freezing and cache resets in
    :meth:`run`.

    Attributes
    ----------
    device : torch.device
        Computation device.
    targets : Dict[str, Callable]
        Target functions keyed by hierarchical name (e.g. 'geometry/bond').
    weights : Dict[str, float]
        Group ('geometry') or component ('geometry/bond') weights.
    history : List[Dict]
        Log of computed values per aggregation call.
    meta : Dict[str, Any]
        Model-level data (rwork, rfree, n_atoms, ...) populated by refinement.
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
        """Look ``key`` up in ``meta`` first, then ``_losses``; ``KeyError`` if in
        neither."""
        if key in self.meta:
            return self.meta[key]
        if key in self._losses:
            return self._losses[key]
        raise KeyError(f"Key '{key}' not found in meta or _losses")

    def __contains__(self, key: str) -> bool:
        """Check if key exists in meta or _losses."""
        return key in self.meta or key in self._losses

    def get(self, key: str, default: Any = None) -> Any:
        """As :meth:`__getitem__` but returning ``default`` instead of raising."""
        if key in self.meta:
            return self.meta[key]
        if key in self._losses:
            return self._losses[key]
        return default

    def cache_losses(self, force: bool = False) -> "LossState":
        """Evaluate registered targets into ``_losses`` and return self.

        ``force=False`` fills only missing keys and leaves stale entries for de-registered
        targets in place; ``force=True`` clears the cache first.
        """
        if force:
            self._losses.clear()

        for name, target in self.targets.items():
            if name not in self._losses:
                self._losses[name] = target()

        return self

    def update_meta(self, data: Dict[str, Any]) -> "LossState":
        """Merge ``data`` into ``meta``; returns self for chaining."""
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
        """Register one target, or auto-expand a combined target into its components.

        Parameters
        ----------
        name : str
            Hierarchical name (e.g. 'geometry/bond', 'adp/simu').
        target : Callable
            Returns a loss tensor when called. A combined target (one with ``.items()``) is
            expanded into its components.
        prefix : str, optional
            Prepended to the name, for registering several models into one state.
        compile : bool
            Mark this target (and any sub-targets) eligible for the compiled aggregate closure
            built by :meth:`compile_aggregate`.
        probe : bool
            If True (default), run the target's forward once and merge the autograd graph's
            leaves into ``self._loss_leaves`` -- so **the target's dependencies (model loaded,
            data attached) must already be in place**. ``probe=False`` skips it, leaving an
            empty leaf set so :meth:`run` auto-disables nothing on this target's account.

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
        self._warn_on_device_mismatch(key, target)
        return self

    def _warn_on_device_mismatch(self, key: str, target: Callable) -> None:
        """Warn if ``target`` disagrees with this state's device.

        Deliberately a warning and not a move: ``target.to()`` would drag the model,
        data and
        scaler the target *borrows* along with it, so registering a loss term could silently
        relocate an entire ``ReflectionData``.
        """
        target_device = getattr(target, "device", None)
        if target_device is None or not isinstance(self.device, torch.device):
            return
        if canonical_device(target_device) == canonical_device(self.device):
            return
        warnings.warn(
            f"LossState: target {key!r} is on {target_device} but the state is "
            f"on {self.device}. Not moving it -- a target borrows its model and "
            "data, so relocating it here would move those too. Construct the "
            "target on the right device instead.",
            stacklevel=3,
        )

    def _collect_resettable_modules(self, target: Callable) -> None:
        """Collect ``target``'s submodules exposing ``reset_cache``, deduplicated.

        Reset after every :meth:`step` so a rejected closure's stale forward cache cannot
        poison the next aggregate.
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
        """Run ``target()`` once with grad on and union its autograd leaves into
        ``self._loss_leaves``; the return may be a tensor, sequence or dict.

        Probe while every parameter the loss should track has ``requires_grad=True``. Zero
        leaves is harmless, only costing extra backward work.
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
        """Register many targets from a component target or dict.

        Keys come from ``target.name`` where present, EXCEPT when the dict key is itself
        hierarchical (contains ``/``) -- that encodes structure the leaf ``.name``
        cannot (e.g.
        ``"model_0/bond"`` from the MultiModel targets), and without honouring it every base
        model's leaf targets collapse onto one key and all but the last are dropped.
        ``prefix``, ``compile`` and ``probe`` are forwarded to :meth:`register_target`.
        """
        for name, target in targets.items():
            # Honor hierarchical dict keys (from MultiModel expansion); they
            # carry the per-model index that the leaf target's fixed .name
            # would otherwise discard, causing model-to-model key collisions.
            target_name = name if "/" in name else getattr(target, "name", name)
            self.register_target(
                target_name, target, prefix=prefix, compile=compile, probe=probe
            )
        return self

    # =========================================================================
    # Weight Management
    # =========================================================================

    def set_weight(self, name: str, weight: float) -> "LossState":
        """Set a group ('geometry') or component ('geometry/bond') weight; returns self."""
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
        """Product of the hierarchical weights for ``name``, missing ones defaulting to 1.

        For 'geometry/bond' that is ``weights['geometry'] * weights['geometry/bond']``.
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
        """Mark already-registered ``names`` eligible for the compiled aggregate."""
        for name in names:
            if name in self.targets:
                self._compilable.add(name)
        self._compiled_aggregate = None
        return self

    def compile_aggregate(self, **compile_kwargs) -> "LossState":
        """Build and cache a ``torch.compile``'d closure over all compilable targets.

        Call after every target and weight is registered, and re-call (or
        :meth:`reset_compiled_aggregate`) if either changes. ``**compile_kwargs`` go to
        ``torch.compile``; ``fullgraph=False`` by default so partial-graph fallback is
        allowed.
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
        """Log ``value`` under ``name`` in the current history entry, creating one if
        needed.

        Tensors are converted with ``.detach().item()``, so a non-scalar tensor raises.
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
        """Evaluate all targets and return the weighted sum.

        With :meth:`compile_aggregate` called and ``log_values=False``, compilable
        targets run
        through the single compiled closure; ``log_values=True`` forces every target
        eager so
        per-target losses land in ``_losses`` and history.
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
        """The cached loss for ``name`` after :meth:`aggregate`, or None."""
        return self._losses.get(name)

    # =========================================================================
    # Optimization (step / run / active-parameter introspection)
    # =========================================================================

    def active_parameters(self) -> Set[nn.Parameter]:
        """The leaf ``nn.Parameter`` set that registered targets' backwards touch.

        Populated incrementally by :meth:`register_target`'s one-shot probe -- this
        method runs
        no forward and walks no graph. Conservative: a target later weighted 0 still
        contributes its leaves, which can only over-freeze, never under-freeze.
        """
        return self._loss_leaves

    def refresh_loss_leaves(self) -> "LossState":
        """Re-probe every target, rebuilding ``_loss_leaves`` and the resettable-module
        cache.

        Needed only after external code replaced parameter identity -- e.g.
        :meth:`Model.freeze`/:meth:`unfreeze`, which rebuild ``refinable_params``. Normal
        :meth:`run` usage never changes identity.
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
        """Run ``nsteps`` optimizer steps, each an ``optimizer.step(closure)``.

        The closure validates each loss for finiteness via
        :func:`torchref.utils.validate_loss` and on failure zeros the gradients and returns
        ``+inf``, so a strong-Wolfe line search backtracks. Works with any closure-taking
        optimizer, though it is exercised mainly with LBFGS.

        Leaves the loss touches but the optimizer was not constructed with get
        ``requires_grad`` disabled, so autograd prunes those subgraphs; on exit it is
        unconditionally re-enabled on every leaf in ``self._loss_leaves``, which stops state
        bleeding between refinement methods. Every ``reset_cache``-bearing submodule is
        reset
        **before** the step loop, so the first forward cannot be served a NaN result cached by
        a previously rejected closure. ``maintenance()`` is called on every target
        afterwards.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Its ``param_groups`` define the *intent* -- the leaves the caller wants updated.
        log : bool
            Call ``aggregate(log_values=True)`` before and after the loop.
        nsteps : int
            Number of ``optimizer.step(closure)`` calls. Forward caches are reset once before
            the loop, not between steps; call repeatedly with ``nsteps=1`` for independent ones.
        context : str
            Diagnostic label forwarded to ``validate_loss``.

        Returns
        -------
        torch.Tensor or None
            The loss from the last accepted closure call, or None if every call was non-finite.
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
        """:meth:`run` with ``nsteps=1``; extra arguments are forwarded."""
        return self.run(optimizer, *args, nsteps=1, **kwargs)

    # =========================================================================
    # Breakdown / Analysis
    # =========================================================================

    def get_breakdown(self) -> Dict[str, Dict[str, Any]]:
        """Nested ``{group: {component: {'loss', 'weight', 'weighted'}}}``."""
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
        """``{group_name: total_weighted_loss}``."""
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
        """Per-target loss / weight / weighted / finite as a printable string.

        One row per target in ``self._losses`` (from the last eager :meth:`aggregate`).
        Shared
        by :meth:`summary` and :func:`torchref.utils.validate_loss` so the format cannot
        drift.
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
    #
    # ``LossState`` normally owns no tensors (targets are callables, weights are
    # floats), so its ``device`` tracker used to need a bespoke ``to()``
    # override to survive a move. ``DeviceMixin`` now carries the parsed request
    # down the traversal and updates tensor-free objects itself, so the override
    # is gone; ``self.device`` follows ``.to()`` via the shared machinery.

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
    """Disable ``requires_grad`` on leaves ``state`` touches but ``optimizer`` lacks.

    Reads the cached leaf union; no probe forward runs here. The enclosing
    :meth:`LossState.step` re-enables every leaf in ``_loss_leaves``, not just these, so a
    pre-frozen leaf cannot leak into the next step.
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
    """Build a :class:`LossState` on ``device`` with optional initial
    ``targets``/``weights``."""
    state = LossState(device=device)

    if targets:
        state.register_targets(targets)
    if weights:
        state.set_weights(weights)

    return state


__all__ = ["LossState", "create_loss_state"]
