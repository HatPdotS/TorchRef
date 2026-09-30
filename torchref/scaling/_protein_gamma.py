"""
Empirical protein correction to the Wilson curve, vendored from cctbx.

Mean intensities of real protein crystals deviate from the ideal random-atom
curve ``sum f^2 exp(-B d*^2 / 2)`` by a resolution-dependent factor
``1 + gamma(d*^2)`` -- the dip near 6 Å and the secondary-structure bump near
4.5 Å. Dividing it out makes the Wilson plot linear from about 11 Å to 1.2 Å,
which is what lets :func:`torchref.scaling.wilson.fit_wilson_b` fit low
resolution data at all.

``gamma`` was obtained from experimental data by Zwart & Lamzin, Acta Cryst.
(2004) D60, 220-226. The coefficients below are the 45-term Chebyshev fit
``coefs_mean`` of ``gamma_protein`` in cctbx ``mmtbx/scaling/absolute_scaling.py``,
copied verbatim, with its range 0.008 <= d*^2 <= 0.69 Å^-2 from
``mmtbx/scaling/scaling.h``. They are used under the cctbx licence, whose
notice redistribution must retain:

    cctbx Copyright (c) 2006 - 2026, The Regents of the University of
    California, through Lawrence Berkeley National Laboratory (subject to
    receipt of any required approvals from the U.S. Dept. of Energy).  All
    rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    (1) Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

    (2) Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.

    (3) Neither the name of the University of California, Lawrence Berkeley
    National Laboratory, U.S. Dept. of Energy nor the names of its
    contributors may be used to endorse or promote products derived from
    this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
    OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    You are under no obligation whatsoever to provide any bug fixes,
    patches, or upgrades to the features, functionality or performance of
    the source code ("Enhancements") to anyone; however, if you choose to
    make your Enhancements available either publicly, or directly to
    Lawrence Berkeley National Laboratory, without imposing a separate
    written license agreement for such Enhancements, then you hereby grant
    the following license: a  non-exclusive, royalty-free perpetual license
    to install, use, modify, prepare derivative works, incorporate into
    other computer software, distribute, and sublicense such enhancements or
    derivative works thereof, in binary and source code form.
"""

import torch

from torchref.scaling.basis import chebyshev_design

__all__ = ["protein_gamma", "D_STAR_SQ_LOW", "D_STAR_SQ_HIGH"]

#: Range of the fit, in Å^-2. Outside it the curve is held at its end values.
D_STAR_SQ_LOW = 0.008
D_STAR_SQ_HIGH = 0.69

_COEFFS = (
    -0.24994838652402987,
    0.15287426147680838,
    0.068108692925184011,
    0.15780196907582875,
    -0.07811375753346686,
    0.043211175909300889,
    -0.043407219965134192,
    0.024613271516995903,
    0.0035146404613345932,
    -0.064118486637211411,
    0.10521875419321854,
    -0.10153928782775833,
    0.0335706778430487,
    -0.0066629477818811282,
    -0.0058221659481290031,
    0.0136026246654981,
    -0.013385834361135244,
    0.022526368996167032,
    -0.019843844247892727,
    0.018128145323325774,
    -0.0091740188657759101,
    0.0068283902389141915,
    -0.0060880807366142566,
    0.0004002124110802677,
    -0.00065686973991185187,
    -0.0039358839200389316,
    0.0056185833386634149,
    -0.0075257168326962913,
    -0.0015215201587884459,
    -0.0036383549957990221,
    -0.0064289154284325831,
    0.0059080442658917334,
    -0.0089851215734611887,
    0.0036488156067441039,
    -0.0047375008148055706,
    -0.00090999496111171302,
    0.00096986728652170276,
    -0.0051006830761911011,
    0.0046838536228956777,
    -0.0031683076118337885,
    0.0037866523617167236,
    0.0015810274077361975,
    0.0011030841357086191,
    0.0015715596895281762,
    -0.0041354783162507788,
)


def protein_gamma(d_star_sq: torch.Tensor) -> torch.Tensor:
    """Fractional deviation of mean protein intensity from the Wilson curve.

    Parameters
    ----------
    d_star_sq : torch.Tensor
        ``1/d^2`` in Å^-2, any shape. Values outside
        [:data:`D_STAR_SQ_LOW`, :data:`D_STAR_SQ_HIGH`] are clamped to the range,
        as cctbx does.

    Returns
    -------
    torch.Tensor
        ``gamma``, same shape and dtype as ``d_star_sq``; the mean intensity is
        ``(1 + gamma)`` times the random-atom value.
    """
    x = d_star_sq.reshape(-1).clamp(D_STAR_SQ_LOW, D_STAR_SQ_HIGH)
    design = chebyshev_design(x, len(_COEFFS), D_STAR_SQ_LOW, D_STAR_SQ_HIGH)
    coeffs = torch.tensor(_COEFFS, dtype=x.dtype, device=x.device)
    # scitbx's Chebyshev series halves the zeroth coefficient.
    coeffs[0] = 0.5 * coeffs[0]
    return (design @ coeffs).reshape(d_star_sq.shape)
