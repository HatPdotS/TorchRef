"""Shared Triton helpers for dihedral-angle computation and gradients.

:func:`dihedral_and_grad` is the Triton counterpart of
:func:`torchref.base.targets._common.torsions_from_xyz` and computes the angle with the
same formula, ``atan2(m1·n2, n1·n2)`` with ``m1 = (b2/|b2|) x n1``, so it carries the
same IUPAC sign as ``gemmi.calculate_dihedral`` and the monomer-library references.
The forces are the Blondel-Karplus / Bekker forms for that sign.

Given four atoms p1, p2, p3, p4 with bonds b1 = p2-p1, b2 = p3-p2, b3 = p4-p3:

    n1 = b1 x b2,  n2 = b2 x b3,  b2_len = |b2|

    ∂ω/∂p1 = -(b2_len / |n1|²) · n1     (call F1)
    ∂ω/∂p4 =  (b2_len / |n2|²) · n2     (call F4)
    ∂ω/∂p2 = -((b1·b2) / |b2|² + 1) · F1 + ((b3·b2) / |b2|²) · F4   (call F2)
    ∂ω/∂p3 = −F1 − F2 − F4              (returned as F3)

The angle and the forces change sign together: F2 and F3 are linear in F1 and F4,
so negating ω means negating F1 and F4 and nothing else.
"""

from __future__ import annotations

import triton
import triton.language as tl
from triton.language.extra import libdevice

from .._common import EPS

# Safe-divide floor, the eager one. Keeps the 1/|b2|, b2_len/|n1|², b2_len/|n2|²
# and 1/c22 terms finite at degenerate (collinear / zero-length-bond) dihedrals so
# the gradient is finite rather than NaN — mirroring the eager guards in
# ``torsions_from_xyz``.
_EPS = tl.constexpr(EPS)


@triton.jit
def dihedral_and_grad(
    p1x,
    p1y,
    p1z,
    p2x,
    p2y,
    p2z,
    p3x,
    p3y,
    p3z,
    p4x,
    p4y,
    p4z,
):
    """Return (omega_rad, F1.., F2.., F3.., F4..) — 1 angle + 12 gradient comps.

    ``omega_rad`` is the IUPAC dihedral in radians and ``Fi = ∂omega/∂pi``. All
    inputs are SIMD lanes of float32 from a Triton block.
    """
    b1x = p2x - p1x
    b1y = p2y - p1y
    b1z = p2z - p1z
    b2x = p3x - p2x
    b2y = p3y - p2y
    b2z = p3z - p2z
    b3x = p4x - p3x
    b3y = p4y - p3y
    b3z = p4z - p3z

    n1x = b1y * b2z - b1z * b2y
    n1y = b1z * b2x - b1x * b2z
    n1z = b1x * b2y - b1y * b2x

    n2x = b2y * b3z - b2z * b3y
    n2y = b2z * b3x - b2x * b3z
    n2z = b2x * b3y - b2y * b3x

    c22 = b2x * b2x + b2y * b2y + b2z * b2z
    c22_safe = c22 + _EPS
    b2_len = tl.sqrt(c22)

    # m1 = (b2 / |b2|) x n1, the operand order of torsions_from_xyz: it is what
    # makes the sign IUPAC.
    inv_b2 = 1.0 / tl.sqrt(c22_safe)
    m1x = (b2y * n1z - b2z * n1y) * inv_b2
    m1y = (b2z * n1x - b2x * n1z) * inv_b2
    m1z = (b2x * n1y - b2y * n1x) * inv_b2
    y = m1x * n2x + m1y * n2y + m1z * n2z
    x = n1x * n2x + n1y * n2y + n1z * n2z
    omega = libdevice.atan2(y, x)

    N1 = n1x * n1x + n1y * n1y + n1z * n1z
    N2 = n2x * n2x + n2y * n2y + n2z * n2z

    # Floor |n1|², |n2|² so collinear b1∥b2 / b2∥b3 give finite forces.
    f1c = -b2_len / (N1 + _EPS)
    F1x = f1c * n1x
    F1y = f1c * n1y
    F1z = f1c * n1z
    f4c = b2_len / (N2 + _EPS)
    F4x = f4c * n2x
    F4y = f4c * n2y
    F4z = f4c * n2z

    c12 = b1x * b2x + b1y * b2y + b1z * b2z
    c23 = b2x * b3x + b2y * b3y + b2z * b3z

    a = -(c12 / c22_safe + 1.0)
    b = c23 / c22_safe

    F2x = a * F1x + b * F4x
    F2y = a * F1y + b * F4y
    F2z = a * F1z + b * F4z

    F3x = -F1x - F2x - F4x
    F3y = -F1y - F2y - F4y
    F3z = -F1z - F2z - F4z

    return (omega, F1x, F1y, F1z, F2x, F2y, F2z, F3x, F3y, F3z, F4x, F4y, F4z)
