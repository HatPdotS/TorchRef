"""The memoised small-d blocks must not change what the contraction returns.

The blocks ``d^l(β)`` depend only on the bandwidth and the β grid, never on the
data, so hoisting them out of the per-call loop is pure reuse. That makes the
cache a correctness risk in exactly one way: a stale entry served for the wrong
``(L, betas)``. These tests pin the key, the identity of the result, and the
one-entry footprint bound.
"""

import math

import pytest
import torch

from torchref.experimental.alignment.frf.wigner_d import (
    _WIGNER_D_CACHE,
    _wigner_d_blocks,
    clear_wigner_d_cache,
    wigner_contraction_per_beta,
)


def _betas(n, step_deg=3.0):
    return torch.arange(n, dtype=torch.float64) * step_deg * (math.pi / 180.0)


def _xi(L, seed=0):
    g = torch.Generator().manual_seed(seed)
    dim = 2 * L - 1
    return torch.randn(L, dim, dim, generator=g, dtype=torch.float64).to(
        torch.complex128
    )


@pytest.mark.unit
def test_the_cached_call_is_bit_identical():
    """A cache hit must give exactly the first call's answer, not merely close."""
    clear_wigner_d_cache()
    L, betas = 9, _betas(12)
    xi = _xi(L)
    first = wigner_contraction_per_beta(xi, betas)
    second = wigner_contraction_per_beta(xi, betas)
    assert torch.equal(first, second)


@pytest.mark.unit
def test_different_data_at_the_same_bandwidth_still_differs():
    """Guard the premise: the cache holds β geometry, not the data.

    Without this, a cache keyed too loosely -- or one that memoised the whole
    result -- would pass the identity test above by returning a stale answer.
    """
    clear_wigner_d_cache()
    L, betas = 9, _betas(12)
    a = wigner_contraction_per_beta(_xi(L, seed=0), betas)
    b = wigner_contraction_per_beta(_xi(L, seed=1), betas)
    assert not torch.allclose(a, b)


@pytest.mark.unit
@pytest.mark.parametrize(
    "L2, n_beta2", [(9, 15), (11, 12)], ids=["other-beta-grid", "other-bandwidth"]
)
def test_a_different_key_is_not_served_the_cached_blocks(L2, n_beta2):
    """Changing either the bandwidth or the β grid must rebuild."""
    clear_wigner_d_cache()
    ref_blocks = _wigner_d_blocks(9, _betas(12), torch.device("cpu"), torch.float64)
    got = _wigner_d_blocks(L2, _betas(n_beta2), torch.device("cpu"), torch.float64)
    assert len(got) == L2 - 1
    assert got[0].shape[0] == n_beta2
    assert got is not ref_blocks


@pytest.mark.unit
def test_the_cache_holds_one_entry():
    """The blocks are hundreds of MB at production bandwidths, so they are not
    allowed to accumulate across keys."""
    clear_wigner_d_cache()
    for L in (7, 9, 11):
        _wigner_d_blocks(L, _betas(12), torch.device("cpu"), torch.float64)
        assert len(_WIGNER_D_CACHE) == 1
    clear_wigner_d_cache()
    assert len(_WIGNER_D_CACHE) == 0


@pytest.mark.unit
def test_the_blocks_are_the_wigner_small_d_matrices():
    """Anchor the cached quantity against d^l(β) computed from the definition.

    ``d^l(0) = I`` and ``d^l(β)`` is orthogonal for every β; both follow from
    ``d^l(β) = exp(-i β J_y)`` and neither holds for a mis-shaped or
    mis-transposed block.
    """
    clear_wigner_d_cache()
    L = 7
    betas = torch.tensor([0.0, 0.4, 1.7, 3.0], dtype=torch.float64)
    blocks = _wigner_d_blocks(L, betas, torch.device("cpu"), torch.float64)
    for l, d in enumerate(blocks, start=1):
        sz = 2 * l + 1
        assert d.shape == (betas.numel(), sz, sz)
        torch.testing.assert_close(d[0], torch.eye(sz, dtype=d.dtype))
        for k in range(betas.numel()):
            torch.testing.assert_close(
                d[k] @ d[k].transpose(-1, -2), torch.eye(sz, dtype=d.dtype)
            )
