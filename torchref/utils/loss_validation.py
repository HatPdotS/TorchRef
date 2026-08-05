"""
Centralized loss finiteness validator for refinement closures.

:func:`validate_loss` checks a loss tensor (and optionally grads and parameters) and, on
failure, dumps a per-target breakdown via ``LossState.format_breakdown``. With
``raise_on_fail=True`` it raises :class:`NonFiniteLossError`; with ``False`` it warns and
returns ``False``, and **the caller must then reject the step itself** -- inside an LBFGS
closure that means zeroing grads and returning ``+inf`` so strong-Wolfe backtracks.

The happy path costs one GPU->CPU sync, plus one more when ``check_grads=True``; the
diagnostic path runs only on failure.
"""

from collections import defaultdict
from typing import TYPE_CHECKING, Iterable, List, Optional

import torch

if TYPE_CHECKING:
    from torchref.refinement.loss_state import LossState


class NonFiniteLossError(RuntimeError):
    """Raised when a refinement step produces non-finite loss, grads, or params."""


# Per-context diagnostic budget so a pathological run doesn't flood logs.
_DIAGNOSTIC_COUNTS: "defaultdict[str, int]" = defaultdict(int)


def reset_diagnostic_budget(context: Optional[str] = None) -> None:
    """Reset the failure counter used to stride full diagnostics.

    Parameters
    ----------
    context : str, optional
        Reset a single context's counter. If omitted, reset all.
    """
    if context is None:
        _DIAGNOSTIC_COUNTS.clear()
    else:
        _DIAGNOSTIC_COUNTS.pop(context, None)


def _is_finite_scalar(t: torch.Tensor) -> bool:
    """Sync a zero-dim finiteness check to a Python bool."""
    return bool(torch.isfinite(t).all().item())


def _any_nonfinite_grads(parameters: List[torch.Tensor]) -> bool:
    """Whether any parameter gradient contains a non-finite entry.

    One reduction per gradient tensor, OR-ed into a device bool, with a single sync at the
    end. Parameters without a gradient are skipped, so no grads at all returns ``False``.
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return False
    # Single-sync reduction: any non-finite across all grads.
    bad = torch.zeros((), dtype=torch.bool, device=grads[0].device)
    for g in grads:
        bad = bad | (~torch.isfinite(g).all())
    return bool(bad.item())


def _nonfinite_counts(parameters: List[torch.Tensor]) -> List[tuple]:
    """``(label, non_finite_count)`` entries for diagnostics only.

    One entry per parameter, plus a second for its gradient when it has one -- so the list
    is longer than ``parameters``.
    """
    out = []
    for i, p in enumerate(parameters):
        label = f"param[{i}] shape={tuple(p.shape)}"
        n_bad = int((~torch.isfinite(p)).sum().item())
        out.append((label, n_bad))
        if p.grad is not None:
            n_bad_g = int((~torch.isfinite(p.grad)).sum().item())
            out.append((f"  .grad shape={tuple(p.grad.shape)}", n_bad_g))
    return out


def validate_loss(
    loss: torch.Tensor,
    *,
    state: Optional["LossState"] = None,
    parameters: Optional[Iterable[torch.Tensor]] = None,
    check_grads: bool = True,
    context: str = "",
    raise_on_fail: bool = True,
    max_full_diagnostics: int = 3,
) -> bool:
    """Check that ``loss`` (and optionally grads / parameters) are finite.

    Parameters
    ----------
    loss : torch.Tensor
        The scalar loss returned by the closure; zero-dim or one-element.
    state : LossState, optional
        The diagnostic path re-runs ``state.aggregate(log_values=True)``, so it **calls
        the targets again**. Omit for closures with no LossState.
    parameters : iterable of torch.Tensor, optional
        Parameters to inspect; their grads are checked when ``check_grads=True``, and both
        are reported with non-finite counts on failure.
    check_grads : bool, default True
        Check gradients after backward(). The usual pathology is a NaN grad from a finite
        forward, so leave it on.
    context : str, default ""
        Label for the diagnostic header and messages; also keys the diagnostic budget.
    raise_on_fail : bool, default True
        True raises :class:`NonFiniteLossError`; False warns and returns ``False``, leaving
        the caller to reject the step.
    max_full_diagnostics : int, default 3
        Per-context budget for the full breakdown; later failures print one compact line.
        ``0`` always prints compact.

    Returns
    -------
    bool
        ``True`` if everything is finite. ``False`` only reaches the caller when
        ``raise_on_fail=False``.

    Raises
    ------
    NonFiniteLossError
        If ``raise_on_fail=True`` and any of loss / grads / params is non-finite.
    """
    if not torch.is_tensor(loss):
        raise TypeError(f"validate_loss: expected tensor, got {type(loss)!r}")

    params_list: List[torch.Tensor] = (
        list(parameters) if parameters is not None else []
    )

    # ---- Fast path: scalar finiteness check(s) ----
    loss_finite = _is_finite_scalar(loss)
    grads_finite = True
    if loss_finite and check_grads and params_list:
        grads_finite = not _any_nonfinite_grads(params_list)

    if loss_finite and grads_finite:
        return True  # happy path

    # ---- Diagnostic path ----
    budget_key = context or "<unnamed>"
    _DIAGNOSTIC_COUNTS[budget_key] += 1
    count = _DIAGNOSTIC_COUNTS[budget_key]
    full = count <= max_full_diagnostics

    _emit_diagnostic(
        loss=loss,
        state=state,
        parameters=params_list,
        context=context,
        loss_finite=loss_finite,
        grads_finite=grads_finite,
        full=full,
        count=count,
    )

    short = _short_message(context, loss_finite, grads_finite)

    if raise_on_fail:
        raise NonFiniteLossError(short)

    return False


def _short_message(context: str, loss_finite: bool, grads_finite: bool) -> str:
    msg = "Non-finite loss detected"
    if context:
        msg += f" in '{context}'"
    msg += (
        f" (loss_finite={loss_finite}, grads_finite={grads_finite})."
        " See diagnostic block above for per-target breakdown."
    )
    return msg


def _emit_diagnostic(
    *,
    loss: torch.Tensor,
    state: Optional["LossState"],
    parameters: List[torch.Tensor],
    context: str,
    loss_finite: bool,
    grads_finite: bool,
    full: bool,
    count: int,
) -> None:
    """Print a diagnostic block without raising; ``full=False`` emits one WARN line."""
    try:
        loss_val = loss.item()
    except Exception:
        loss_val = float("nan")

    if not full:
        print(
            f"WARN[{count:>4}] non-finite loss in '{context}' "
            f"(loss={loss_val!r}, loss_finite={loss_finite}, "
            f"grads_finite={grads_finite})",
            flush=True,
        )
        return

    sep = "=" * 78
    lines: List[str] = []
    lines.append("")
    lines.append(sep)
    header = f"NON-FINITE LOSS DETECTED  (occurrence #{count})"
    if context:
        header += f"  context='{context}'"
    lines.append(header)
    lines.append(sep)
    lines.append(f"total loss        : {loss_val!r}")
    lines.append(f"loss finite       : {loss_finite}")
    lines.append(f"grads finite      : {grads_finite}")

    # Per-target breakdown via the LossState formatter.
    if state is not None:
        try:
            with torch.no_grad():
                state.aggregate(log_values=True)
            lines.append("")
            lines.append("per-target breakdown (eager re-aggregation):")
            try:
                lines.append(state.format_breakdown())
            except AttributeError:
                lines.append("  (LossState.format_breakdown unavailable)")
                for name, lt in getattr(state, "_losses", {}).items():
                    w = state.get_effective_weight(name)
                    try:
                        v = lt.item()
                    except Exception:
                        v = float("nan")
                    finite = "yes" if (v == v and abs(v) != float("inf")) else "NO"
                    lines.append(
                        f"  {name:<32} w={w:>8.4g}  loss={v:>14.6g}  {finite}"
                    )
        except Exception as exc:
            lines.append(f"  (state.aggregate raised during diagnostic: {exc!r})")

    # Parameter / gradient non-finite entry counts.
    if parameters:
        lines.append("")
        lines.append("parameter / gradient inspection:")
        for label, n_bad in _nonfinite_counts(parameters):
            marker = " <-- NaN/Inf" if n_bad else ""
            lines.append(f"  {label}: non_finite={n_bad}{marker}")

    lines.append(sep)
    print("\n".join(lines), flush=True)
