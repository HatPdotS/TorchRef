"""Unit tests for SeededLBFGS and the Hessian-diagonal curvature helper.

Covers:
1. A diagonal quadratic bowl: an exact inverse-diagonal seed makes the first
   (single) inner iteration land on the minimum, where plain steepest-descent
   L-BFGS with the same one-iteration budget does not.
2. ``init_hess_diag=None`` reproduces ``torch.optim.LBFGS`` bit-for-bit; an
   all-ones seed reproduces the first *direction* but not the first step length.
3. The Hutchinson diagonal helper matches the true Hessian diagonal (exact
   ``probe="basis"``; statistical ``probe="rademacher"``).

All tests are pure-CPU (the analytic double-backward HVP runs under
``use_portable()`` on plain torch), so no GPU marker is needed.
"""

import pytest
import torch

from torchref.refinement.optimizers import (
    SeededLBFGS,
    hessian_diagonal,
    preconditioner_from_diagonal,
)

pytestmark = pytest.mark.unit


# =============================================================================
# 1. Diagonal quadratic bowl — one seeded step reaches the minimum
# =============================================================================
def test_seeded_step_solves_diagonal_quadratic_in_one_iter():
    """f(x) = 0.5 * sum(h_i x_i^2); seed = 1/h => d0 = -x, t = 1 => x -> 0."""
    h = torch.tensor([1.0, 4.0, 9.0, 16.0], dtype=torch.float64)
    x0 = torch.tensor([3.0, -2.0, 1.0, 0.5], dtype=torch.float64)

    # --- seeded ---
    x = torch.nn.Parameter(x0.clone())
    opt = SeededLBFGS(
        [x], lr=1.0, max_iter=1, history_size=100, line_search_fn="strong_wolfe"
    )
    opt.set_init_hess_diag(1.0 / h)  # exact inverse-diagonal (lambda = 0)

    def closure():
        opt.zero_grad()
        loss = 0.5 * (h * x * x).sum()
        loss.backward()
        return loss

    opt.step(closure)
    assert x.detach().abs().max().item() < 1e-8

    # --- stock steepest descent, same one-iteration budget: does NOT solve ---
    x2 = torch.nn.Parameter(x0.clone())
    stock = torch.optim.LBFGS(
        [x2], lr=1.0, max_iter=1, history_size=100, line_search_fn="strong_wolfe"
    )

    def closure2():
        stock.zero_grad()
        loss = 0.5 * (h * x2 * x2).sum()
        loss.backward()
        return loss

    stock.step(closure2)
    # one steepest-descent line search along -g cannot zero all four coords
    assert x2.detach().abs().max().item() > 1e-2


# =============================================================================
# 2. Equivalence / divergence vs stock torch.optim.LBFGS
# =============================================================================
def _quartic_closure(opt, x, a):
    def closure():
        opt.zero_grad()
        loss = ((x - a) ** 4).sum()
        loss.backward()
        return loss

    return closure


def test_none_seed_is_bit_identical_to_stock_lbfgs():
    """With init_hess_diag=None the whole trajectory matches torch.optim.LBFGS."""
    a = torch.tensor([1.0, -2.0, 0.5, 3.0], dtype=torch.float64)
    x0 = torch.zeros(4, dtype=torch.float64)

    kw = dict(lr=1.0, max_iter=20, history_size=10, line_search_fn="strong_wolfe")

    x_seed = torch.nn.Parameter(x0.clone())
    opt_seed = SeededLBFGS([x_seed], **kw)  # no seed set -> None
    c_seed = _quartic_closure(opt_seed, x_seed, a)

    x_stock = torch.nn.Parameter(x0.clone())
    opt_stock = torch.optim.LBFGS([x_stock], **kw)
    c_stock = _quartic_closure(opt_stock, x_stock, a)

    for _ in range(5):
        l_seed = float(opt_seed.step(c_seed))
        l_stock = float(opt_stock.step(c_stock))
        assert l_seed == l_stock
        assert torch.equal(x_seed.detach(), x_stock.detach())


def test_ones_seed_matches_stock_direction_but_not_step_length():
    """An all-ones seed gives the stock first direction (-g) but t = lr, not the
    ``min(1, 1/||g||_1)`` heuristic."""
    a = torch.tensor([1.0, -2.0, 0.5, 3.0], dtype=torch.float64)
    x0 = torch.zeros(4, dtype=torch.float64)
    kw = dict(lr=1.0, max_iter=1, history_size=10, line_search_fn="strong_wolfe")

    x_seed = torch.nn.Parameter(x0.clone())
    opt_seed = SeededLBFGS([x_seed], **kw)
    opt_seed.set_init_hess_diag(torch.ones(4, dtype=torch.float64))
    opt_seed.step(_quartic_closure(opt_seed, x_seed, a))

    x_stock = torch.nn.Parameter(x0.clone())
    opt_stock = torch.optim.LBFGS([x_stock], **kw)
    opt_stock.step(_quartic_closure(opt_stock, x_stock, a))

    d_seed = opt_seed.state[opt_seed._params[0]]["d"]
    d_stock = opt_stock.state[opt_stock._params[0]]["d"]
    # same first direction (steepest descent, since precond = ones)
    assert torch.allclose(d_seed, d_stock)
    # but the initial trial step length differs by design
    t_seed = opt_seed.state[opt_seed._params[0]]["t"]
    t_stock = opt_stock.state[opt_stock._params[0]]["t"]
    # (post-line-search t may coincide only by accident; assert the *first-guess*
    # rule differs by checking the resulting parameter points are not identical)
    assert not torch.equal(x_seed.detach(), x_stock.detach()) or t_seed != t_stock


# =============================================================================
# 3. Hessian-diagonal helper vs the true diagonal
# =============================================================================
def test_hessian_diagonal_basis_matches_true_diagonal():
    """Exact (basis-probe) diagonal matches torch.autograd.functional.hessian."""
    g = torch.Generator().manual_seed(0)
    n = 6
    M = torch.randn(n, n, generator=g, dtype=torch.float64)
    M = 0.5 * (M + M.t())  # symmetric (off-diagonal curvature present)
    c = torch.rand(n, generator=g, dtype=torch.float64) + 0.1
    x_val = torch.randn(n, generator=g, dtype=torch.float64)

    def f(v):
        return 0.5 * (v @ M @ v) + (c * v**4).sum()

    x = torch.nn.Parameter(x_val.clone())

    diag_est = hessian_diagonal(lambda: f(x), [x], probe="basis")

    H = torch.autograd.functional.hessian(f, x_val.clone())
    true_diag = torch.diagonal(H)
    assert torch.allclose(diag_est, true_diag, atol=1e-8)


def test_per_group_floor_does_not_crush_small_group():
    """A two-group diagonal with a 1000:1 magnitude gap: the global floor crushes
    the small group's preconditioner, but the per-group floor gives each group its
    own Newton scale."""
    # group A: large curvature (~1e3), group B: small curvature (~1.0)
    diag = torch.tensor([1000.0, 800.0, 1.0, 0.8], dtype=torch.float64)
    sizes = [2, 2]
    lam = 1e-2

    glob = preconditioner_from_diagonal(diag, lam=lam)              # global floor
    per = preconditioner_from_diagonal(diag, lam=lam, group_sizes=sizes)

    # Global floor = lam*max|diag| = 10 -> small group clamped to 1/10 = 0.1,
    # i.e. its true 1/1.0=1.0 preconditioner is crushed ~10x.
    assert torch.allclose(glob[2:], torch.tensor([0.1, 0.1], dtype=torch.float64))
    # Per-group: small group floored by its OWN max (1.0) -> ~1/|diag| preserved.
    assert per[2] > 0.9  # ~1/1.0, not crushed
    assert per[3] > 0.9  # 1/0.8 clamped by its own floor 1e-2*1.0 -> ~1/0.8=1.25
    # large group identical either way (it sets the global max)
    assert torch.allclose(per[:2], glob[:2])


def test_hessian_diagonal_rademacher_approximates_true_diagonal():
    """Stochastic (Rademacher) diagonal converges to the truth within tolerance."""
    g = torch.Generator().manual_seed(1)
    n = 6
    M = torch.randn(n, n, generator=g, dtype=torch.float64)
    M = 0.5 * (M + M.t())
    c = torch.rand(n, generator=g, dtype=torch.float64) + 0.1
    x_val = torch.randn(n, generator=g, dtype=torch.float64)

    def f(v):
        return 0.5 * (v @ M @ v) + (c * v**4).sum()

    x = torch.nn.Parameter(x_val.clone())

    probe_gen = torch.Generator().manual_seed(2)
    diag_hutch = hessian_diagonal(
        lambda: f(x), [x], n_probes=4000, generator=probe_gen, probe="rademacher"
    )

    true_diag = torch.diagonal(torch.autograd.functional.hessian(f, x_val.clone()))
    rel = (diag_hutch - true_diag).norm() / true_diag.norm()
    assert rel < 0.1
