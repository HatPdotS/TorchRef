"""The radial contraction has to *materialise* its conjugate.

``torch.conj`` does not conjugate anything: it returns a view carrying a
conjugate *bit*, and every consumer is expected to honour it. MPS's batched
complex matmul does not -- and the contraction in
:func:`~torchref.experimental.alignment.frf.data_mr.cross_correlate_xi` is an
``einsum`` that lowers to exactly that. The failure is silent and total: the
unconjugated values are contracted instead, which on 1DAW moved ``xi`` by 173%
of its own peak magnitude, reordered the entire rotation-function peak list, and
pushed the true orientation from rank 0 out of the top 200 -- while the top
score moved by only 0.1%, so nothing looked wrong.

Elementwise ops, ``where``, ``index_add`` and 2-D ``matmul`` all honour the bit;
only the batched path drops it. That is narrow enough that the guard has to be
the value of the contraction itself, checked on whatever device this host has,
against a reference computed on the host in double.
"""

import pytest
import torch

from torchref.config import get_complex_dtype, get_default_device
from torchref.experimental.alignment.frf.data_mr import cross_correlate_xi
from torchref.experimental.alignment.frf.types import BesselSHCoefficients

pytestmark = pytest.mark.unit

L, N_RADIAL = 6, 8


def _coeffs(seed, device, dtype):
    """A filled ``(N_radial, L, 2L-1)`` coefficient block."""
    g = torch.Generator().manual_seed(seed)
    real = torch.randn(N_RADIAL, L, 2 * L - 1, generator=g, dtype=torch.float64)
    imag = torch.randn(N_RADIAL, L, 2 * L - 1, generator=g, dtype=torch.float64)
    c = torch.complex(real, imag)
    return BesselSHCoefficients(
        coeffs=c.to(device=device, dtype=dtype), L=L, bessel_h_scale=40.0,
    )


def test_the_contraction_conjugates_the_calc_side():
    """On this host's device, against the same sum done on the host in double.

    A dropped conjugation is not a small error -- it changes the sign of every
    imaginary part in one operand -- so the bar can be tight without being
    brittle about float32 rounding.
    """
    device, cplx = get_default_device(), get_complex_dtype()
    obs, calc = _coeffs(0, device, cplx), _coeffs(1, device, cplx)
    got = cross_correlate_xi(obs, calc)

    host_obs = BesselSHCoefficients(
        coeffs=obs.coeffs.cpu().to(torch.complex128), L=L, bessel_h_scale=40.0)
    host_calc = BesselSHCoefficients(
        coeffs=calc.coeffs.cpu().to(torch.complex128), L=L, bessel_h_scale=40.0)
    ref = torch.einsum(
        "rln,rlm->lmn",
        host_obs.coeffs,
        torch.conj(host_calc.coeffs).resolve_conj(),
    )

    err = (got.cpu().to(torch.complex128) - ref).abs().max()
    scale = ref.abs().max()
    assert float(err / scale) < 1e-5, (
        f"contraction is {float(err / scale):.2e} away from the host double "
        f"reference on {device}; a dropped conjugation shows up here as O(1)"
    )


def test_dropping_the_conjugation_would_be_caught():
    """The guard above has to be able to see the failure it exists for.

    Contracting the unconjugated coefficients is what a lost conjugate bit
    produces, so that has to land far outside the tolerance -- otherwise the
    test would pass on the broken path too.
    """
    device, cplx = get_default_device(), get_complex_dtype()
    obs, calc = _coeffs(0, device, cplx), _coeffs(1, device, cplx)
    ref = torch.einsum(
        "rln,rlm->lmn",
        obs.coeffs.cpu().to(torch.complex128),
        torch.conj(calc.coeffs.cpu().to(torch.complex128)).resolve_conj(),
    )
    unconjugated = torch.einsum(
        "rln,rlm->lmn",
        obs.coeffs.cpu().to(torch.complex128),
        calc.coeffs.cpu().to(torch.complex128),
    )
    rel = float((unconjugated - ref).abs().max() / ref.abs().max())
    assert rel > 0.1, f"the two differ by only {rel:.2e}; this guard is blind"
