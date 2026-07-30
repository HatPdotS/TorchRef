"""Direct-summation dispatch *mechanics*, as distinct from its accuracy.

CPU-only. What is left here after the oracle package absorbed the numerics: that
reflection-chunking is exact, that an empty atom set returns zeros of the right shape, and
that an integer ``hkl`` -- the production dtype -- is accepted losslessly.

Two tests were deleted rather than migrated when the ``Engine`` enum went away. Both had
forcing as their *subject*: one asserted that a forced Triton engine refuses CPU inputs, and
one that a forced Metal engine runs eager instead of raising. With no forcing engine there is
nothing to refuse, and the second would have degenerated into ``assert torch.equal(F, F)``.
Selection is now asserted directly against the tables in
``tests/unit/utils/test_backend_tables.py``.

The accuracy questions this file used to own -- checkpointed-vs-eager parity and
``gradcheck`` -- moved to ``test_gradients.py`` in this package, which ties the shipping
``_checkpointed_*`` path to the ``_eager_*`` oracle and then to finite differences. They
were duplicated in ``tests/unit/test_gradient_correctness.py`` as well, at a different
strictness, which is why they were consolidated.
"""

import pytest
import torch

from torchref.base.direct_summation import ds_iso
from torchref.base.direct_summation import dispatch as D
from torchref.config import dtypes

pytestmark = pytest.mark.unit

# Ambient configured dtype, used only to build inputs. The eager-parity comparisons
# that needed a dtype-dependent tolerance here now live in ``test_gradients.py``, where
# the whole package is pinned to float64 by a fixture rather than reading the global at
# import time -- an import-time read made this file's strictness depend silently on
# ``TORCHREF_DTYPE_FLOAT``.
_F = dtypes.float


def _inputs(N=4, R=7, seed=0, dtype=None):
    dtype = dtype or _F
    torch.manual_seed(seed)
    hkl = torch.randint(-3, 4, (R, 3)).to(dtype)
    s = torch.rand(R, dtype=dtype) * 0.5
    svec = torch.randn(R, 3, dtype=dtype) * 0.3
    A = torch.rand(N, 5, dtype=dtype)
    B = torch.rand(N, 5, dtype=dtype) + 0.5
    return hkl, s, svec, A, B


def _leaves(N=4, seed=1, dtype=None):
    dtype = dtype or _F
    torch.manual_seed(seed)
    xyz = torch.rand(N, 3, dtype=dtype, requires_grad=True)
    occ = (torch.rand(N, dtype=dtype) * 0.4 + 0.6).requires_grad_()
    adp = (torch.rand(N, dtype=dtype) * 10 + 5).requires_grad_()
    U = (torch.rand(N, 6, dtype=dtype) * 0.04 + 0.01).requires_grad_()
    return xyz, occ, adp, U




def test_checkpointed_chunking_is_exact():
    # Chunking slices the reflection dimension, so each reflection's atom sum is
    # computed within a single chunk -- the result is mathematically identical
    # regardless of chunk size. It is not, however, guaranteed bit-exact: the
    # phase matmul dispatches to different BLAS kernels for (R,3)x(3,N) vs
    # (1,3)x(3,N), and last-ULP rounding there is CPU/BLAS-dependent. Assert
    # numerical equivalence, not bitwise equality.
    hkl, s, _, A, B = _inputs(R=11, dtype=torch.float64)
    xyz, occ, adp, _ = _leaves(dtype=torch.float64)
    F_full = D._checkpointed_iso(hkl, s, xyz, occ, adp, A, B, max_memory_gb=None)
    F_chunk = D._checkpointed_iso(hkl, s, xyz, occ, adp, A, B, max_memory_gb=1e-7)
    assert torch.allclose(F_full, F_chunk, rtol=1e-10, atol=1e-12)




def test_empty_atoms_returns_zeros():
    hkl, s, _, _, _ = _inputs()
    empty = torch.zeros(0, 3, dtype=torch.float64)
    z = torch.zeros(0, dtype=torch.float64)
    z5 = torch.zeros(0, 5, dtype=torch.float64)
    F = ds_iso(hkl, s, empty, z, z, z5, z5)
    assert F.shape == (hkl.shape[0],)
    assert F.abs().sum().item() == 0.0


def test_integer_hkl_is_accepted_and_lossless():
    """An ``int32`` ``hkl`` must keep working, and must give the float32 answer exactly.

    Miller indices arrive as ``int32`` straight from the MTZ reader
    (``torchref/io/mtz.py``), and every DS path casts ``hkl`` itself -- ``_cols_f32`` in
    the Triton kernel, ``hkl_c.to(xyz_frac.dtype)`` in the chunk math. So integer input
    is not merely tolerated, it is the production case.

    This matters for how the dtype gate may be tightened. The gate's test is
    ``t.dtype is torch.float32``, an identity check, so probing ``hkl`` with that rule
    would reject the production dtype and disable the Triton path outright. The rule has to
    exempt integer tensors, and this test is what fails if it does not.

    Worth recording alongside: casting ``hkl`` is not merely tolerable, it is *free*.
    Miller indices are integer-valued in every dtype that holds them (symmetry maps
    ``h -> h.R`` with integer ``R``), so the f64->f32 round-trip is bit-exact for any
    \\|h\\| < 2**24 and downcasting ``hkl`` alone was measured to change ``F`` by exactly
    0.0. An earlier version of this comment claimed the cast truncated the phase; it does
    not, and the gate correspondingly does not probe ``hkl``.
    """
    _, s, _, A, B = _inputs(dtype=torch.float64)
    xyz, occ, adp, _ = _leaves(dtype=torch.float64)
    torch.manual_seed(0)
    hkl_int = torch.randint(-3, 4, (7, 3), dtype=torch.int32)
    s = s[: hkl_int.shape[0]]

    F_int = ds_iso(hkl_int, s, xyz, occ, adp, A, B)
    F_float = ds_iso(hkl_int.to(torch.float64), s, xyz, occ, adp, A, B)
    assert torch.equal(F_int, F_float), "casting integer Miller indices must be exact"


def test_eager_none_scattering_factors_and_no_batching():
    """``max_memory_gb=None`` with ``scattering_factors=None`` must compute from A/B.

    Two independently optional arguments meeting: with no batching *and* no precomputed
    scattering factors, the eager path takes a branch that derives ``f(s)`` from the ITC92
    A/B coefficients inline. Chunking must not change the answer.

    Moved from ``tests/unit/test_kernel_fixes.py``, where it sat among dtype-switch and
    NaN-safety tests. It is the only structure-factor test in that file, and this package is
    where structure-factor coverage lives.
    """
    from torchref.base.direct_summation.dispatch import _eager_aniso, _eager_iso

    g = torch.Generator().manual_seed(0)
    N, R = 8, 12
    d = torch.float64
    hkl = torch.randint(-3, 4, (R, 3), generator=g).to(d)
    s = torch.rand(R, generator=g, dtype=d) * 0.4
    svec = torch.randn(R, 3, generator=g, dtype=d) * 0.3
    A = torch.rand(N, 5, generator=g, dtype=d)
    B = torch.rand(N, 5, generator=g, dtype=d) + 0.5
    xyz = torch.rand(N, 3, generator=g, dtype=d)
    occ = torch.rand(N, generator=g, dtype=d) * 0.4 + 0.6
    adp = torch.rand(N, generator=g, dtype=d) * 10 + 5
    U = torch.rand(N, 6, generator=g, dtype=d) * 0.04 + 0.01

    for tag, fn, geom, third in (
        ("iso", _eager_iso, s, adp),
        ("aniso", _eager_aniso, svec, U),
    ):
        f_none = fn(hkl, geom, xyz, occ, third, A, B, None)
        f_batch = fn(hkl, geom, xyz, occ, third, A, B, 2.0)
        assert torch.isfinite(f_none).all(), f"{tag}: non-finite F with no batching"
        torch.testing.assert_close(
            f_none, f_batch, rtol=1e-12, atol=1e-12,
            msg=f"{tag}: chunking changed the answer",
        )
