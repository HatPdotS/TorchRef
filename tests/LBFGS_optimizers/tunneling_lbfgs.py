"""
Tunneling LBFGS Optimizer
=========================

An LBFGS optimizer where the line search "over-explores" along the search
direction — sampling many points instead of stopping at the nearest Wolfe-
satisfying step. This lets the optimizer "tunnel" through barriers between
local minima, analogous to quantum tunneling.

Key idea:
  Standard Strong Wolfe line search → finds the *nearest* acceptable step
  Tunneling line search → scans *broadly*, picks the *best* step

Hybrid strategy:
  1. Every step starts with a coarse bidirectional scan along the LBFGS
     direction (the "tunneling" phase).
  2. If the scan finds a much better point (relative improvement > threshold),
     take the tunneling step.
  3. Otherwise, fall back to a standard Strong Wolfe line search for fast
     local convergence.

This gives the best of both worlds: global exploration when there are better
minima reachable along the search direction, and quadratic local convergence
when we're already in a good basin.
"""

import torch
from torch.optim import Optimizer


def _cubic_interpolate(x1, f1, g1, x2, f2, g2, bounds=None):
    """Attempt cubic interpolation; fall back to bisection."""
    # Cubic interpolation from Numerical Optimization (Nocedal & Wright)
    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    d2_sq = d1 * d1 - g1 * g2
    if d2_sq >= 0:
        d2 = d2_sq.sqrt()
        if isinstance(x1, torch.Tensor):
            min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
            min_pos = float(min_pos)
        else:
            import math
            min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
    else:
        min_pos = (x1 + x2) / 2.0  # fallback: bisection

    if bounds is not None:
        min_pos = max(bounds[0], min(bounds[1], min_pos))
    return min_pos


def _strong_wolfe(closure_with_grad, x0_flat, set_flat, gather_grad,
                  direction, f0, g0, lr=1.0, c1=1e-4, c2=0.9, max_ls=25):
    """Strong Wolfe line search (zoom variant).

    Returns (step_size, f_new, g_new_dot_d).
    """
    d = direction
    gtd_0 = float(torch.dot(g0, d))  # directional derivative at α=0

    if gtd_0 >= 0:
        # not a descent direction
        return 0.0, f0, gtd_0

    alpha_prev = 0.0
    f_prev = f0
    gtd_prev = gtd_0
    alpha = lr
    done = False

    for i in range(max_ls):
        # evaluate at trial point
        set_flat(x0_flat + alpha * d)
        f_alpha = float(closure_with_grad())
        g_alpha = gather_grad().clone()
        gtd_alpha = float(torch.dot(g_alpha, d))

        # Armijo condition violated or function increased
        if f_alpha > f0 + c1 * alpha * gtd_0 or (f_alpha >= f_prev and i > 0):
            alpha, f_alpha, gtd_alpha = _zoom(
                closure_with_grad, x0_flat, set_flat, gather_grad, d,
                alpha_prev, alpha, f_prev, f_alpha, gtd_prev, gtd_alpha,
                f0, gtd_0, c1, c2)
            done = True
            break

        # Strong Wolfe conditions satisfied
        if abs(gtd_alpha) <= -c2 * gtd_0:
            done = True
            break

        # Positive slope — bracket is (alpha, alpha_prev)
        if gtd_alpha >= 0:
            alpha, f_alpha, gtd_alpha = _zoom(
                closure_with_grad, x0_flat, set_flat, gather_grad, d,
                alpha, alpha_prev, f_alpha, f_prev, gtd_alpha, gtd_prev,
                f0, gtd_0, c1, c2)
            done = True
            break

        alpha_prev = alpha
        f_prev = f_alpha
        gtd_prev = gtd_alpha
        alpha = min(alpha * 2.0, 1e6)  # expand bracket

    if not done:
        # max iterations — use last point
        set_flat(x0_flat + alpha * d)

    return alpha, f_alpha, gtd_alpha


def _zoom(closure_with_grad, x0_flat, set_flat, gather_grad, d,
          lo, hi, f_lo, f_hi, gtd_lo, gtd_hi,
          f0, gtd_0, c1, c2, max_iter=10):
    """Zoom phase of Strong Wolfe line search."""
    for _ in range(max_iter):
        alpha = _cubic_interpolate(lo, f_lo, gtd_lo, hi, f_hi, gtd_hi,
                                   bounds=(min(lo, hi), max(lo, hi)))

        set_flat(x0_flat + alpha * d)
        f_alpha = float(closure_with_grad())
        g_alpha = gather_grad().clone()
        gtd_alpha = float(torch.dot(g_alpha, d))

        if f_alpha > f0 + c1 * alpha * gtd_0 or f_alpha >= f_lo:
            hi = alpha
            f_hi = f_alpha
            gtd_hi = gtd_alpha
        else:
            if abs(gtd_alpha) <= -c2 * gtd_0:
                return alpha, f_alpha, gtd_alpha
            if gtd_alpha * (hi - lo) >= 0:
                hi = lo
                f_hi = f_lo
                gtd_hi = gtd_lo
            lo = alpha
            f_lo = f_alpha
            gtd_lo = gtd_alpha

    return alpha, f_alpha, gtd_alpha


class TunnelingLBFGS(Optimizer):
    """LBFGS with a tunneling (over-exploring) line search.

    Parameters
    ----------
    params : iterable
        Parameters to optimize.
    lr : float
        Base learning rate (scaling for the search direction). Default: 1.0.
    history_size : int
        Number of (s, y) pairs to keep for the L-BFGS approximation.
    n_scan : int
        Number of coarse scan points PER direction (total = 2 * n_scan).
    n_scan_fine : int
        Number of fine-refinement points around the best coarse point.
    max_step : float
        Maximum distance to scan in parameter space units.
    tunnel_threshold : float
        Relative improvement threshold: if the best scan point improves
        the loss by more than this fraction, take the tunnel step.
        Otherwise fall back to Strong Wolfe for local convergence.
    """

    def __init__(
        self,
        params,
        lr=1.0,
        history_size=10,
        n_scan=128,
        n_scan_fine=32,
        max_step=10.0,
        tunnel_threshold=0.01,
    ):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

        self.history_size = history_size
        self.n_scan = n_scan
        self.n_scan_fine = n_scan_fine
        self.max_step = max_step
        self.tunnel_threshold = tunnel_threshold

        # L-BFGS state
        self._s_history = []
        self._y_history = []
        self._rho_history = []

        # stats
        self.last_mode = None  # "tunnel" or "wolfe"

    # ------------------------------------------------------------------
    # Flat parameter helpers
    # ------------------------------------------------------------------
    def _gather_flat_params(self):
        views = []
        for group in self.param_groups:
            for p in group["params"]:
                views.append(p.data.view(-1))
        return torch.cat(views, 0)

    def _set_flat_params(self, flat):
        offset = 0
        for group in self.param_groups:
            for p in group["params"]:
                numel = p.numel()
                p.data.copy_(flat[offset : offset + numel].view_as(p))
                offset += numel

    def _gather_flat_grad(self):
        views = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    views.append(p.new(p.numel()).zero_())
                else:
                    views.append(p.grad.data.view(-1))
        return torch.cat(views, 0)

    # ------------------------------------------------------------------
    # L-BFGS two-loop recursion  →  search direction
    # ------------------------------------------------------------------
    def _lbfgs_direction(self, grad):
        s_hist = self._s_history
        y_hist = self._y_history
        rho_hist = self._rho_history
        m = len(s_hist)

        if m == 0:
            return -grad  # steepest descent on first step

        q = grad.clone()
        alphas = [None] * m

        for i in range(m - 1, -1, -1):
            alphas[i] = rho_hist[i] * torch.dot(s_hist[i], q)
            q.add_(y_hist[i], alpha=-alphas[i])

        gamma = torch.dot(s_hist[-1], y_hist[-1]) / torch.dot(
            y_hist[-1], y_hist[-1]
        )
        r = gamma * q

        for i in range(m):
            beta = rho_hist[i] * torch.dot(y_hist[i], r)
            r.add_(s_hist[i], alpha=(alphas[i] - beta))

        return -r

    # ------------------------------------------------------------------
    # Tunneling scan (bidirectional)
    # ------------------------------------------------------------------
    def _tunneling_scan(self, closure, x0, direction, f0):
        """Bidirectional coarse + fine scan; return (best_displacement, best_f).

        Returns the displacement vector (not alpha), so the caller just does
        x_new = x0 + displacement.
        """
        max_alpha = self.max_step

        # Linear spacing, bidirectional: scan both +d and -d
        alphas_pos = torch.linspace(
            max_alpha / self.n_scan, max_alpha, self.n_scan
        )
        alphas_all = torch.cat([-alphas_pos.flip(0), alphas_pos])

        best_alpha = 0.0
        best_f = f0

        for alpha in alphas_all:
            self._set_flat_params(x0 + alpha * direction)
            with torch.no_grad():
                f = float(closure())
            if f < best_f:
                best_f = f
                best_alpha = float(alpha)

        # --- fine scan around best ---
        if best_alpha != 0 and self.n_scan_fine > 0:
            half_width = max_alpha / self.n_scan  # one coarse step
            lo = best_alpha - half_width
            hi = best_alpha + half_width
            alphas_fine = torch.linspace(lo, hi, self.n_scan_fine)
            for alpha in alphas_fine:
                self._set_flat_params(x0 + alpha * direction)
                with torch.no_grad():
                    f = float(closure())
                if f < best_f:
                    best_f = f
                    best_alpha = float(alpha)

        return best_alpha * direction, best_f

    # ------------------------------------------------------------------
    # Optimizer step
    # ------------------------------------------------------------------
    def step(self, closure):
        """Perform one L-BFGS step with tunneling line search.

        Strategy:
          1. Compute L-BFGS direction
          2. Do a broad bidirectional scan for tunneling opportunities
          3. If scan finds a significantly better point → tunnel there
          4. Otherwise → use Strong Wolfe for precise local convergence
        """
        assert closure is not None, "TunnelingLBFGS requires a closure"
        closure = torch.enable_grad()(closure)

        # --- evaluate at current point ---
        loss = float(closure())
        flat_grad = self._gather_flat_grad().clone()
        x0 = self._gather_flat_params().clone()

        # --- search direction via L-BFGS ---
        direction = self._lbfgs_direction(flat_grad)

        # ensure descent direction
        if torch.dot(direction, flat_grad) > 0:
            direction = -flat_grad

        # normalize so that max_step is in parameter-space units
        dir_norm = direction.norm()
        if dir_norm < 1e-12:
            # gradient ≈ 0 → at a stationary point, use random direction
            direction = torch.randn_like(direction)
            dir_norm = direction.norm()
        direction = direction / dir_norm

        # --- Phase 1: tunneling scan ---
        tunnel_disp, tunnel_f = self._tunneling_scan(
            closure, x0, direction, loss
        )

        # decide: tunnel or local Wolfe?
        relative_improvement = (loss - tunnel_f) / (abs(loss) + 1e-12)

        if relative_improvement > self.tunnel_threshold:
            # ---- TUNNEL: take the big step ----
            self.last_mode = "tunnel"
            new_x = x0 + tunnel_disp
            self._set_flat_params(new_x)

            # re-evaluate with gradients
            closure()
            new_grad = self._gather_flat_grad().clone()

            # update LBFGS history
            s = new_x - x0
            y = new_grad - flat_grad
            ys = float(torch.dot(y, s))

            if ys > 1e-10:
                if len(self._s_history) >= self.history_size:
                    self._s_history.pop(0)
                    self._y_history.pop(0)
                    self._rho_history.pop(0)
                self._s_history.append(s)
                self._y_history.append(y)
                self._rho_history.append(1.0 / ys)
            else:
                # big jump broke curvature — reset history for clean restart
                self._s_history.clear()
                self._y_history.clear()
                self._rho_history.clear()

            return tunnel_f

        else:
            # ---- WOLFE: fast local convergence ----
            self.last_mode = "wolfe"

            # restore starting point
            self._set_flat_params(x0)

            # use the (un-normalized) LBFGS direction for Wolfe
            raw_direction = self._lbfgs_direction(flat_grad)
            if torch.dot(raw_direction, flat_grad) > 0:
                raw_direction = -flat_grad

            alpha, f_new, _ = _strong_wolfe(
                closure, x0, self._set_flat_params, self._gather_flat_grad,
                raw_direction, loss, flat_grad, lr=self.param_groups[0]["lr"],
            )

            if alpha == 0:
                self._set_flat_params(x0)
                return loss

            new_x = x0 + alpha * raw_direction
            self._set_flat_params(new_x)

            # re-evaluate for gradient
            closure()
            new_grad = self._gather_flat_grad().clone()

            # update LBFGS history
            s = new_x - x0
            y = new_grad - flat_grad
            ys = float(torch.dot(y, s))

            if ys > 1e-10:
                if len(self._s_history) >= self.history_size:
                    self._s_history.pop(0)
                    self._y_history.pop(0)
                    self._rho_history.pop(0)
                self._s_history.append(s)
                self._y_history.append(y)
                self._rho_history.append(1.0 / ys)

            return f_new
