"""L-BFGS with one initial-Hessian scale per parameter block.

Stock :class:`torch.optim.LBFGS` seeds its two-loop recursion with a scalar
``H0 = gamma * I``, ``gamma = s.y / y.y`` over the whole parameter vector. In a
joint refinement that vector concatenates coordinates (Å), log-B, Cholesky-encoded
anisotropic U and occupancy logits, whose curvatures differ by orders of magnitude.
Outside the span of the stored (s, y) pairs the search direction is ``-gamma * g``,
so one block's share of the step is mis-scaled against another's and the line
search, which only picks a length along that direction, cannot repair it. Because
:class:`~torchref.refinement.lbfgs_refinement.LBFGSRefinement` starts a fresh
optimizer every call with ``max_iter=20``, the history never spans more than ~20 of
the tens of thousands of directions, and H0 sets most of every step.

:class:`BlockLBFGS` replaces the scalar with a block-constant diagonal: block ``b``
gets ``gamma_b = s_b.y_b / y_b.y_b`` from the latest pair (a Barzilai-Borwein scale
per block, the same per-block normalisation Muon applies to each weight matrix).
It needs no extra loss or gradient evaluations and no second derivatives, so it
runs with every electron-density backend.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch
from torch.optim.lbfgs import _strong_wolfe

from torchref.config import get_int_dtype


class BlockLBFGS(torch.optim.LBFGS):
    """L-BFGS whose initial inverse Hessian is a per-block multiple of the identity.

    Constructed like :class:`torch.optim.LBFGS` and driven the same way
    (``optimizer.step(closure)``); only the scaling of the two-loop recursion
    differs.

    Parameters
    ----------
    params : iterable of torch.nn.Parameter
        Parameters to optimize. A single parameter group, as for
        :class:`torch.optim.LBFGS`.
    blocks : sequence of sequence of torch.nn.Parameter, optional
        Partition of ``params`` into blocks that share one scale. Every parameter
        must appear in exactly one block. Default: each parameter is its own block,
        which in TorchRef means one block per parametrization (``xyz``, ``adp``,
        ``u``, ``occupancy``, each scaler term).
    **kwargs
        Forwarded to :class:`torch.optim.LBFGS` (``lr``, ``max_iter``,
        ``history_size``, ``line_search_fn``, ...).

    Notes
    -----
    A block whose latest pair has ``s_b.y_b <= 0`` (curvature along the step is not
    positive inside that block, possible when blocks are coupled even though the
    strong-Wolfe step guarantees ``s.y > 0`` overall) or ``y_b = 0`` (the block did
    not move or does not enter the loss) falls back to the global scalar ``gamma``.
    With a single block the trajectory equals :class:`torch.optim.LBFGS` up to
    floating-point summation order.

    Not used by :class:`~torchref.refinement.lbfgs_refinement.LBFGSRefinement`. In a
    joint xyz + ADP step with the scaler held fixed, the ADP block's larger scale also
    speeds motion along the weakly determined overall-B direction, which the ML
    target absorbs through σ_A but R-work does not: total loss falls faster early
    while R-work and R-free end worse than with the scalar scale.

    Raises
    ------
    ValueError
        If ``blocks`` is not a partition of ``params``.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        blocks: Optional[Sequence[Sequence[torch.nn.Parameter]]] = None,
        **kwargs,
    ):
        super().__init__(params, **kwargs)
        if blocks is None:
            block_of = {id(p): i for i, p in enumerate(self._params)}
            n_blocks = len(self._params)
        else:
            block_of = {}
            for i, block in enumerate(blocks):
                for p in block:
                    if id(p) in block_of:
                        raise ValueError("a parameter appears in more than one block")
                    block_of[id(p)] = i
            if set(block_of) != {id(p) for p in self._params}:
                raise ValueError("blocks must partition the optimizer's parameters")
            n_blocks = len(blocks)
        self._n_blocks = n_blocks
        # Flat block label per element, in _gather_flat_grad order.
        self._block_index = torch.cat(
            [
                torch.full(
                    (p.numel(),),
                    block_of[id(p)],
                    dtype=get_int_dtype(),
                    device=p.device,
                )
                for p in self._params
            ]
        )

    def _block_scale(self, s: torch.Tensor, y: torch.Tensor, ys: torch.Tensor):
        """Return the flat diagonal of H0 from the latest pair ``(s, y)``."""
        idx = self._block_index.to(s.device)
        sy_b = torch.zeros(self._n_blocks, device=s.device, dtype=s.dtype)
        yy_b = torch.zeros_like(sy_b)
        sy_b.index_add_(0, idx, s * y)
        yy_b.index_add_(0, idx, y * y)
        gamma = ys / y.dot(y)
        ok = (sy_b > 0) & (yy_b > 0)
        gamma_b = torch.where(ok, sy_b / torch.where(ok, yy_b, 1.0), gamma)
        return gamma_b[idx]

    @torch.no_grad()
    def step(self, closure):  # type: ignore[override]
        """Perform one optimization step (up to ``max_iter`` inner iterations).

        Identical to :meth:`torch.optim.LBFGS.step` except that the initial inverse
        Hessian of the two-loop recursion is the block-constant diagonal described
        on the class.

        Parameters
        ----------
        closure : callable
            Re-evaluates the model and returns the loss; called with grad enabled.

        Returns
        -------
        torch.Tensor
            The loss from the first closure call of this step.
        """
        assert len(self.param_groups) == 1

        closure = torch.enable_grad()(closure)

        group = self.param_groups[0]
        lr = float(group["lr"])
        max_iter = group["max_iter"]
        max_eval = group["max_eval"]
        tolerance_grad = group["tolerance_grad"]
        tolerance_change = group["tolerance_change"]
        line_search_fn = group["line_search_fn"]
        history_size = group["history_size"]

        # LBFGS has only global state, registered on the first param as in torch.
        state = self.state[self._params[0]]
        state.setdefault("func_evals", 0)
        state.setdefault("n_iter", 0)

        orig_loss = closure()
        loss = float(orig_loss)
        current_evals = 1
        state["func_evals"] += 1

        flat_grad = self._gather_flat_grad()
        if flat_grad.abs().max() <= tolerance_grad:
            return orig_loss

        d = state.get("d")
        t = state.get("t")
        old_dirs = state.get("old_dirs")
        old_stps = state.get("old_stps")
        ro = state.get("ro")
        H_diag = state.get("H_diag")
        prev_flat_grad = state.get("prev_flat_grad")
        prev_loss = state.get("prev_loss")

        n_iter = 0
        while n_iter < max_iter:
            n_iter += 1
            state["n_iter"] += 1

            if state["n_iter"] == 1:
                d = flat_grad.neg()
                old_dirs = []
                old_stps = []
                ro = []
                H_diag = 1
            else:
                y = flat_grad.sub(prev_flat_grad)
                s = d.mul(t)
                ys = y.dot(s)
                if ys > 1e-10:
                    if len(old_dirs) == history_size:
                        old_dirs.pop(0)
                        old_stps.pop(0)
                        ro.pop(0)
                    old_dirs.append(y)
                    old_stps.append(s)
                    ro.append(1.0 / ys)
                    H_diag = self._block_scale(s, y, ys)

                num_old = len(old_dirs)
                if "al" not in state:
                    state["al"] = [None] * history_size
                al = state["al"]

                q = flat_grad.neg()
                for i in range(num_old - 1, -1, -1):
                    al[i] = old_stps[i].dot(q) * ro[i]
                    q.add_(old_dirs[i], alpha=-al[i])

                d = r = torch.mul(q, H_diag)
                for i in range(num_old):
                    be_i = old_dirs[i].dot(r) * ro[i]
                    r.add_(old_stps[i], alpha=al[i] - be_i)

            if prev_flat_grad is None:
                prev_flat_grad = flat_grad.clone(memory_format=torch.contiguous_format)
            else:
                prev_flat_grad.copy_(flat_grad)
            prev_loss = loss

            if state["n_iter"] == 1:
                t = min(1.0, 1.0 / flat_grad.abs().sum()) * lr
            else:
                t = lr

            gtd = flat_grad.dot(d)
            if gtd > -tolerance_change:
                break

            ls_func_evals = 0
            if line_search_fn is not None:
                if line_search_fn != "strong_wolfe":
                    raise RuntimeError("only 'strong_wolfe' is supported")
                x_init = self._clone_param()

                def obj_func(x, t, d):
                    return self._directional_evaluate(closure, x, t, d)

                loss, flat_grad, t, ls_func_evals = _strong_wolfe(
                    obj_func, x_init, t, d, loss, flat_grad, gtd
                )
                self._add_grad(t, d)
                opt_cond = flat_grad.abs().max() <= tolerance_grad
            else:
                self._add_grad(t, d)
                if n_iter != max_iter:
                    with torch.enable_grad():
                        loss = float(closure())
                    flat_grad = self._gather_flat_grad()
                    opt_cond = flat_grad.abs().max() <= tolerance_grad
                    ls_func_evals = 1

            current_evals += ls_func_evals
            state["func_evals"] += ls_func_evals

            if n_iter == max_iter:
                break
            if current_evals >= max_eval:
                break
            if opt_cond:
                break
            if d.mul(t).abs().max() <= tolerance_change:
                break
            if abs(loss - prev_loss) < tolerance_change:
                break

        state["d"] = d
        state["t"] = t
        state["old_dirs"] = old_dirs
        state["old_stps"] = old_stps
        state["ro"] = ro
        state["H_diag"] = H_diag
        state["prev_flat_grad"] = prev_flat_grad
        state["prev_loss"] = prev_loss

        return orig_loss
