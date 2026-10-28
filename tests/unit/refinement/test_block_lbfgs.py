"""Unit tests for :class:`BlockLBFGS`, L-BFGS with a per-block initial Hessian scale.

Pins:
1. With a single block the trajectory matches :class:`torch.optim.LBFGS`.
2. On a quadratic whose two blocks differ in curvature by 10^3, the per-block scale
   converges in an iteration budget where the scalar scale does not.
3. A block that does not enter the loss falls back to the global scale and stays put.
4. ``blocks`` must partition the parameters.
"""

import pytest
import torch

from torchref.config import get_float_dtype
from torchref.refinement.optimizers import BlockLBFGS

pytestmark = pytest.mark.unit

LBFGS_KW = dict(lr=1.0, max_iter=20, history_size=100, line_search_fn="strong_wolfe")


def _spd(n, lo, hi, gen, dtype):
    q, _ = torch.linalg.qr(torch.randn(n, n, generator=gen, dtype=dtype))
    return (q * torch.logspace(lo, hi, n, dtype=dtype)) @ q.T


def _two_block_problem(dtype):
    gen = torch.Generator().manual_seed(0)
    ha = _spd(40, 1.5, 2.5, gen, dtype)  # stiff block, curvature 30..300
    hb = _spd(20, -1.5, -0.5, gen, dtype)  # soft block, curvature 0.03..0.3
    xa = torch.randn(40, generator=gen, dtype=dtype)
    xb = torch.randn(20, generator=gen, dtype=dtype)
    return ha, hb, xa, xb


def _minimise(opt_cls, ha, hb, xa_star, xb_star, **kw):
    a = torch.nn.Parameter(torch.zeros_like(xa_star))
    b = torch.nn.Parameter(torch.zeros_like(xb_star))
    opt = opt_cls([a, b], **{**LBFGS_KW, **kw})

    def closure():
        opt.zero_grad()
        da, db = a - xa_star, b - xb_star
        loss = 0.5 * (da @ ha @ da + db @ hb @ db)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        da, db = a - xa_star, b - xb_star
        return float(0.5 * (da @ ha @ da + db @ hb @ db)), a.detach(), b.detach()


def test_single_block_matches_stock_lbfgs():
    ha, hb, xa, xb = _two_block_problem(get_float_dtype())
    _, a_stock, b_stock = _minimise(torch.optim.LBFGS, ha, hb, xa, xb, max_iter=8)

    a = torch.nn.Parameter(torch.zeros_like(xa))
    b = torch.nn.Parameter(torch.zeros_like(xb))
    opt = BlockLBFGS([a, b], blocks=[[a, b]], **{**LBFGS_KW, "max_iter": 8})

    def closure():
        opt.zero_grad()
        da, db = a - xa, b - xb
        loss = 0.5 * (da @ ha @ da + db @ hb @ db)
        loss.backward()
        return loss

    opt.step(closure)
    torch.testing.assert_close(a.detach(), a_stock, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(b.detach(), b_stock, rtol=1e-4, atol=1e-5)


def test_block_scale_beats_scalar_scale_on_mismatched_blocks():
    ha, hb, xa, xb = _two_block_problem(get_float_dtype())
    f0 = float(0.5 * (xa @ ha @ xa + xb @ hb @ xb))
    f_stock, _, _ = _minimise(torch.optim.LBFGS, ha, hb, xa, xb)
    f_block, _, _ = _minimise(BlockLBFGS, ha, hb, xa, xb)
    assert f_block < 1e-3 * f0
    assert f_block < 1e-2 * f_stock


def test_block_outside_loss_falls_back_and_stays_put():
    dtype = get_float_dtype()
    x = torch.nn.Parameter(torch.tensor([3.0, -2.0], dtype=dtype))
    unused = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0], dtype=dtype))
    opt = BlockLBFGS([x, unused], **LBFGS_KW)

    def closure():
        opt.zero_grad()
        loss = (x * x).sum() + 0.0 * unused.sum()
        loss.backward()
        return loss

    opt.step(closure)
    assert torch.isfinite(x).all()
    assert float(x.detach().abs().max()) < 1e-3
    torch.testing.assert_close(
        unused.detach(), torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
    )


def test_blocks_must_partition_params():
    a = torch.nn.Parameter(torch.zeros(2))
    b = torch.nn.Parameter(torch.zeros(2))
    with pytest.raises(ValueError):
        BlockLBFGS([a, b], blocks=[[a]])
    with pytest.raises(ValueError):
        BlockLBFGS([a, b], blocks=[[a, b], [b]])
