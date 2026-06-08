"""Unit tests for torchref.base.metrics.binwise_scale."""

import torch

from torchref.base.metrics import binwise_scale


def _reference_loop(F_calc, F_obs, bins, valid, nbins, min_count=1):
    """Slow, explicit per-bin LS scale for cross-checking the vectorized form."""
    Fc = F_calc.abs()
    Fo = F_obs.abs()
    c = torch.ones(nbins, dtype=Fc.dtype)
    for b in range(nbins):
        m = (bins == b) & valid
        if int(m.sum()) < min_count:
            continue
        num = (Fo[m] * Fc[m]).sum()
        den = (Fc[m] ** 2).sum().clamp(min=1e-12)
        c[b] = num / den
    return c


def test_matches_reference_loop():
    torch.manual_seed(0)
    n, nbins = 500, 8
    Fc = torch.rand(n) + 0.1
    Fo = Fc * 1.7 + 0.1 * torch.randn(n)  # roughly scale 1.7
    bins = torch.randint(0, nbins, (n,))
    valid = torch.rand(n) > 0.3
    got = binwise_scale(Fc, Fo, bins, valid=valid, nbins=nbins)
    ref = _reference_loop(Fc, Fo, bins, valid, nbins)
    assert torch.allclose(got, ref, atol=1e-5)


def test_recovers_known_scale():
    torch.manual_seed(1)
    n, nbins = 2000, 5
    Fc = torch.rand(n) + 0.5
    true_c = torch.tensor([0.5, 1.0, 1.5, 2.0, 3.0])
    bins = torch.randint(0, nbins, (n,))
    Fo = Fc * true_c[bins]  # exact, no noise
    c = binwise_scale(Fc, Fo, bins, nbins=nbins)
    assert torch.allclose(c, true_c, atol=1e-4)


def test_empty_and_sparse_bins_default_to_one():
    Fc = torch.tensor([1.0, 2.0, 3.0])
    Fo = torch.tensor([2.0, 4.0, 6.0])
    bins = torch.tensor([0, 0, 0])  # bins 1,2 empty
    c = binwise_scale(Fc, Fo, bins, nbins=3)
    assert torch.isclose(c[0], torch.tensor(2.0), atol=1e-5)
    assert c[1] == 1.0 and c[2] == 1.0

    # min_count guard: a bin with one reflection stays at 1 when min_count=2
    c2 = binwise_scale(Fc, Fo, torch.tensor([0, 0, 1]), nbins=2, min_count=2)
    assert c2[1] == 1.0


def test_valid_mask_excludes_from_fit():
    Fc = torch.tensor([1.0, 1.0, 1.0, 1.0])
    Fo = torch.tensor([2.0, 2.0, 2.0, 100.0])  # last is an outlier
    bins = torch.zeros(4, dtype=torch.long)
    valid = torch.tensor([True, True, True, False])
    c = binwise_scale(Fc, Fo, bins, valid=valid, nbins=1)
    assert torch.isclose(c[0], torch.tensor(2.0), atol=1e-5)


def test_complex_input():
    Fc = torch.tensor([3.0 + 4.0j, 6.0 + 8.0j])  # |Fc| = 5, 10
    Fo = torch.tensor([10.0, 20.0])
    bins = torch.zeros(2, dtype=torch.long)
    c = binwise_scale(Fc, Fo, bins, nbins=1)
    assert torch.isclose(c[0], torch.tensor(2.0), atol=1e-5)
