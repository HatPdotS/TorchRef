"""SeededLBFGS: L-BFGS whose first step is a diagonal-Newton step.

Stock :class:`torch.optim.LBFGS` initialises its inverse-Hessian approximation with a
*scalar*, so its first search direction is plain steepest descent -- badly conditioned
when parameter blocks have very different curvature scales (coordinates in Angstrom, ADPs
in Angstrom^2/log, occupancies in sigmoid-logit, in one call). ``SeededLBFGS`` takes a
per-parameter inverse-curvature preconditioner ``1 / (|H_ii| + lambda)`` and uses it for
the **first** inner iteration only::

    d_0 = -(precond .* g)     # diagonal-Newton direction
    t_0 = lr                  # Newton scale -> start the line search at t = 1

Every later iteration is stock L-BFGS. The seed fires exactly while the curvature history
is empty, so re-seeding a fresh block is ``optimizer.state.clear()`` +
:meth:`set_init_hess_diag`. The diagonal comes from
:mod:`torchref.refinement.optimizers.curvature`, but this optimizer is agnostic to that.

With ``init_hess_diag=None`` the trajectory is **bit-identical** to stock L-BFGS. With an
all-ones preconditioner it is deliberately *not*: the direction matches steepest descent
but the first trial step is ``lr`` rather than the ``min(1, 1/||g||_1)`` heuristic,
because the diagonal supplies Newton scale.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.optim.lbfgs import _strong_wolfe, _to_scalar


class SeededLBFGS(torch.optim.LBFGS):
    """L-BFGS that seeds its first step from the diagonal of the Hessian.

    Constructed exactly like :class:`torch.optim.LBFGS`. Before running, hand it
    the inverse-curvature preconditioner via :meth:`set_init_hess_diag` (a flat
    tensor in ``_gather_flat_grad`` order over the optimizer's parameters). Pass
    ``None`` (the default) to behave exactly like stock L-BFGS.
    """

    def __init__(self, *args, init_hess_diag: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_hess_diag: Optional[torch.Tensor] = None
        if init_hess_diag is not None:
            self.set_init_hess_diag(init_hess_diag)

    def set_init_hess_diag(self, precond: Optional[torch.Tensor]) -> None:
        """Set the flat inverse-curvature preconditioner ``1/(|diag|+lambda)``.

        ``precond`` must be flat and aligned to ``cat(p.view(-1) for p in params)``, the
        :meth:`_gather_flat_grad` layout; ``None`` disables seeding. It applies to the
        first inner iteration of the next :meth:`step` whose curvature history is empty.
        Raises ``ValueError`` if the element count does not match the parameters.
        """
        if precond is None:
            self._init_hess_diag = None
            return
        if precond.dim() != 1:
            precond = precond.reshape(-1)
        expected = self._numel()
        if precond.numel() != expected:
            raise ValueError(
                f"init_hess_diag numel {precond.numel()} != param numel {expected}"
            )
        self._init_hess_diag = precond

    @torch.no_grad()
    def step(self, closure):  # type: ignore[override]
        """One optimization step, ``closure`` returning the loss.

        Identical to :meth:`torch.optim.LBFGS.step` except that, on the first inner
        iteration with a seed set, the direction is ``-(precond .* g)`` and the initial
        trial step is ``lr``.
        """
        assert len(self.param_groups) == 1

        # Make sure the closure is always called with grad enabled
        closure = torch.enable_grad()(closure)

        group = self.param_groups[0]
        lr = _to_scalar(group["lr"])
        max_iter = group["max_iter"]
        max_eval = group["max_eval"]
        tolerance_grad = group["tolerance_grad"]
        tolerance_change = group["tolerance_change"]
        line_search_fn = group["line_search_fn"]
        history_size = group["history_size"]

        # NOTE: LBFGS has only global state, but we register it as state for
        # the first param, because this helps with casting in load_state_dict
        state = self.state[self._params[0]]
        state.setdefault("func_evals", 0)
        state.setdefault("n_iter", 0)

        # evaluate initial f(x) and df/dx
        orig_loss = closure()
        loss = float(orig_loss)
        current_evals = 1
        state["func_evals"] += 1

        flat_grad = self._gather_flat_grad()
        opt_cond = flat_grad.abs().max() <= tolerance_grad

        # optimal condition
        if opt_cond:
            return orig_loss

        # tensors cached in state (for tracing)
        d = state.get("d")
        t = state.get("t")
        old_dirs = state.get("old_dirs")
        old_stps = state.get("old_stps")
        ro = state.get("ro")
        H_diag = state.get("H_diag")
        prev_flat_grad = state.get("prev_flat_grad")
        prev_loss = state.get("prev_loss")

        # Resolve the diagonal seed once for this .step() call. Only consulted on
        # the first inner iteration (empty history); cast to the grad's layout.
        seed = self._init_hess_diag
        if seed is not None:
            seed = seed.to(device=flat_grad.device, dtype=flat_grad.dtype)

        n_iter = 0
        # optimize for a max of max_iter iterations
        while n_iter < max_iter:
            # keep track of nb of iterations
            n_iter += 1
            state["n_iter"] += 1

            ############################################################
            # compute gradient descent direction
            ############################################################
            if state["n_iter"] == 1:
                if seed is not None:
                    # ---- seeded first step: diagonal-Newton direction ----
                    d = flat_grad.neg().mul_(seed)
                else:
                    d = flat_grad.neg()
                old_dirs = []
                old_stps = []
                ro = []
                H_diag = 1
            else:
                # do lbfgs update (update memory)
                y = flat_grad.sub(prev_flat_grad)
                s = d.mul(t)
                ys = y.dot(s)  # y*s
                if ys > 1e-10:
                    # updating memory
                    if len(old_dirs) == history_size:
                        # shift history by one (limited-memory)
                        old_dirs.pop(0)
                        old_stps.pop(0)
                        ro.pop(0)

                    # store new direction/step
                    old_dirs.append(y)
                    old_stps.append(s)
                    ro.append(1.0 / ys)

                    # update scale of initial Hessian approximation
                    H_diag = ys / y.dot(y)  # (y*y)

                # compute the approximate (L-BFGS) inverse Hessian
                # multiplied by the gradient
                num_old = len(old_dirs)

                if "al" not in state:
                    state["al"] = [None] * history_size
                al = state["al"]

                # iteration in L-BFGS loop collapsed to use just one buffer
                q = flat_grad.neg()
                for i in range(num_old - 1, -1, -1):
                    al[i] = old_stps[i].dot(q) * ro[i]
                    q.add_(old_dirs[i], alpha=-al[i])

                # multiply by initial Hessian
                # r/d is the final direction
                d = r = torch.mul(q, H_diag)
                for i in range(num_old):
                    be_i = old_dirs[i].dot(r) * ro[i]
                    r.add_(old_stps[i], alpha=al[i] - be_i)

            if prev_flat_grad is None:
                prev_flat_grad = flat_grad.clone(memory_format=torch.contiguous_format)
            else:
                prev_flat_grad.copy_(flat_grad)
            prev_loss = loss

            ############################################################
            # compute step length
            ############################################################
            # reset initial guess for step size
            if state["n_iter"] == 1:
                if seed is not None:
                    # diagonal seed already carries Newton scale -> t = 1 (lr)
                    t = lr
                else:
                    t = min(1.0, 1.0 / flat_grad.abs().sum()) * lr
            else:
                t = lr

            # directional derivative
            gtd = flat_grad.dot(d)  # g * d

            # directional derivative is below tolerance
            if gtd > -tolerance_change:
                break

            # optional line search: user function
            ls_func_evals = 0
            if line_search_fn is not None:
                # perform line search, using user function
                if line_search_fn != "strong_wolfe":
                    raise RuntimeError("only 'strong_wolfe' is supported")
                else:
                    x_init = self._clone_param()

                    def obj_func(x, t, d):
                        return self._directional_evaluate(closure, x, t, d)

                    loss, flat_grad, t, ls_func_evals = _strong_wolfe(
                        obj_func, x_init, t, d, loss, flat_grad, gtd
                    )
                self._add_grad(t, d)
                opt_cond = flat_grad.abs().max() <= tolerance_grad
            else:
                # no line search, simply move with fixed-step
                self._add_grad(t, d)
                if n_iter != max_iter:
                    # re-evaluate function only if not in last iteration
                    # the reason we do this: in a stochastic setting,
                    # no use to re-evaluate that function here
                    with torch.enable_grad():
                        loss = closure()
                    loss = float(loss)
                    flat_grad = self._gather_flat_grad()
                    opt_cond = flat_grad.abs().max() <= tolerance_grad
                    ls_func_evals = 1

            # update func eval
            current_evals += ls_func_evals
            state["func_evals"] += ls_func_evals

            ############################################################
            # check conditions
            ############################################################
            if n_iter == max_iter:
                break

            if current_evals >= max_eval:
                break

            # optimal condition
            if opt_cond:
                break

            # lack of progress
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
