"""The fused Legendre/shell kernel against the portable reference.

Two things need pinning. The kernel must agree with the torch reference -- it is
selected automatically wherever it builds, so a divergence would silently change
every rotation search on that host. And it must **refuse** float64 rather than
accept it: it reads every array through a raw ``float*``, so a float64 buffer
would be reinterpreted as twice as many float32s, not converted.

Agreement is to a float32 tolerance, not bit-exact, and deliberately so: the
kernel accumulates cluster-by-cluster within a shell while ``index_add_``
accumulates over the whole chunk, and in single precision a different summation
order is a different answer. Measured 4e-7 to 1e-5 relative, which sits between
the grouping's own error and Phaser's cos(theta) bucketing at ~2e-5.
"""

import pytest
import torch

from torchref.experimental.alignment.frf.kernels import portable
from torchref.experimental.alignment.frf.kernels.cpu import legendre_shell as fused
from torchref.experimental.alignment.sh import legendre_recurrence_coefficients

pytestmark = pytest.mark.unit

#: Summation order in single precision, nothing more. Set above the measured
#: 1e-5 worst case with margin; a real divergence is orders larger, because a
#: dropped term changes whole rows rather than their last digits.
_TOL = 1e-4


def _case(L, n_clusters, n_shells, seed):
    """Random input in the layout the kernel requires: shells sorted."""
    g = torch.Generator().manual_seed(seed)
    cos_t = 2 * torch.rand(n_clusters, generator=g, dtype=torch.float32) - 1
    sin_t = (1 - cos_t * cos_t).clamp(min=0).sqrt()
    Dr = torch.randn(n_clusters, L, generator=g, dtype=torch.float32)
    Di = torch.randn(n_clusters, L, generator=g, dtype=torch.float32)
    shell = torch.sort(
        torch.randint(0, n_shells, (n_clusters,), generator=g))[0]
    a, b, sect = legendre_recurrence_coefficients(
        L, torch.float32, torch.device("cpu"))
    n_even = (L - 1 if (L - 1) % 2 == 0 else L - 2) // 2
    return dict(shape=(n_even, n_shells, L), args=(cos_t, sin_t, Dr, Di, shell,
                                                   a, b, sect))


def _run(fn, case):
    Tr = torch.zeros(case["shape"], dtype=torch.float32)
    Ti = torch.zeros_like(Tr)
    fn(Tr, Ti, *case["args"])
    return Tr, Ti


@pytest.mark.parametrize("L,n_clusters,n_shells", [(13, 500, 40),
                                                   (65, 4000, 300),
                                                   (101, 3000, 250)])
def test_fused_agrees_with_portable(L, n_clusters, n_shells):
    if not fused.available():
        pytest.skip(f"fused kernel unavailable: {fused.why_unavailable()}")
    case = _case(L, n_clusters, n_shells, seed=4)
    ref_r, ref_i = _run(portable.legendre_shell_accumulate, case)
    got_r, got_i = _run(fused.legendre_shell_accumulate, case)
    for name, ref, got in (("real", ref_r, got_r), ("imag", ref_i, got_i)):
        rel = (got - ref).abs().max().item() / max(ref.abs().max().item(), 1e-30)
        assert rel < _TOL, f"L={L} {name} part differs by {rel:.2e}"


def test_float64_is_refused_not_reinterpreted():
    """A float64 caller must raise, naming the dtype."""
    if not fused.available():
        pytest.skip(f"fused kernel unavailable: {fused.why_unavailable()}")
    L, n_clusters, n_shells = 9, 20, 5
    n_even = (L - 1) // 2
    d = torch.float64
    with pytest.raises(RuntimeError, match="float32 only"):
        fused.legendre_shell_accumulate(
            torch.zeros(n_even, n_shells, L, dtype=d),
            torch.zeros(n_even, n_shells, L, dtype=d),
            torch.zeros(n_clusters, dtype=d), torch.zeros(n_clusters, dtype=d),
            torch.zeros(n_clusters, L, dtype=d),
            torch.zeros(n_clusters, L, dtype=d),
            torch.zeros(n_clusters, dtype=torch.long),
            torch.zeros(L, L, dtype=d), torch.zeros(L, L, dtype=d),
            torch.zeros(L, dtype=d))


def test_shell_offsets_partition_the_clusters():
    """The kernel's work split: contiguous, complete, and in shell order."""
    shell = torch.tensor([0, 0, 2, 2, 2, 5], dtype=torch.long)
    off = fused.shell_offsets(shell, 6)
    assert off.tolist() == [0, 2, 2, 5, 5, 5, 6]
    assert int(off[-1]) == shell.numel()


def test_the_dispatch_prefers_the_fused_kernel_when_it_builds():
    """Whatever the table selects is what the expansion runs."""
    from torchref.experimental.alignment.frf._backends import LEGENDRE_BACKENDS
    from torchref.utils.backends import select

    probe = [torch.zeros(1, 1, 9, dtype=torch.float32)] * 6
    chosen = select(LEGENDRE_BACKENDS, probe).name
    expected = "cpu_fused" if fused.available() else "portable"
    assert chosen == expected, (
        f"table chose {chosen!r} but the fused kernel is "
        f"{'available' if fused.available() else 'unavailable'}"
    )
