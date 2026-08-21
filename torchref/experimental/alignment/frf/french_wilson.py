"""French–Wilson posterior + Luzzati DFAC chain.

Pure ports of Phaser's ``lib/math_FrenchWilson.cc`` (centric/acentric
posterior moments via Parabolic-cylinder ratios) and the Halley-iteration
``getDfactor`` in ``lib/math_RiceLLG.cc``. The public entry point
:func:`french_wilson_preprocess` returns ``(eEobs, DFAC, sqrt_mean_F2)``
from raw ``(F, σF, |s|, centric)``.

Everything except ``french_wilson_preprocess`` is module-private; expose
the public name through :mod:`torchref.experimental.alignment.frf.preprocessing`.

References (paths under
``…/reverse_engineering/phenix/.../phaser/src/``):
- ``lib/math_FrenchWilson.cc:8-178``  posterior ``<E>`` / ``<E²>``
- ``lib/math_RiceLLG.cc:12-250``      Rice-moment effective σA + Halley
- ``Dfactor.cc:87-93``                eEobs assembly + clamp
"""
from __future__ import annotations

import torch


__all__ = ["french_wilson_preprocess"]


# -----------------------------------------------------------------------------
# French-Wilson posterior expected values  (math_FrenchWilson.cc)
# -----------------------------------------------------------------------------


def _expectE_FW_acen(eosq, sigesq):
    """
    Acentric posterior expected E from normalised observed intensity (eosq)
    and its standard deviation (sigesq). Translates verbatim from Phaser's
    `lib/math_FrenchWilson.cc:expectEFWacen` (lines 8-44). Vectorised NumPy.

    `eosq = Iobs / <I>`, `sigesq = σIobs / <I>`.
    """
    import numpy as np
    from scipy.special import erfc, pbdv
    CROSS1, CROSS2 = -12.5, 18.0
    SQRT2 = np.sqrt(2.0)
    x = (eosq - sigesq ** 2) / sigesq
    xsqr = x * x
    ee = np.empty_like(eosq)
    m_neg = x < CROSS1
    if m_neg.any():
        xs = xsqr[m_neg]
        num = (-916620705. + xs *
               (91891800. + xs *
                (-11531520. + xs *
                 (1935360. + xs *
                  (-491520. + xs * 262144.)))))
        den = (-495452160. + xs *
               (55050240. + xs *
                (-7864320. + xs *
                 (1572864. + xs *
                  (-524288. + xs * 524288.)))))
        ee[m_neg] = np.sqrt(-np.pi * sigesq[m_neg] / x[m_neg]) * num / den
    m_pos = x > CROSS2
    if m_pos.any():
        xs = xsqr[m_pos]
        num = (-45045. + 32. * xs *
               (-315. + 8. * xs *
                (-15. - 16. * xs + 128. * xs * xs)))
        ee[m_pos] = (np.sqrt(sigesq[m_pos]) * num /
                     (32768. * x[m_pos] ** 7.5))
    m_mid = ~(m_neg | m_pos)
    if m_mid.any():
        xm = x[m_mid]
        pcd, _ = pbdv(-1.5, -xm)
        ee[m_mid] = (np.sqrt(sigesq[m_mid] / 2.0) * np.exp(-xm * xm / 4.0) *
                     pcd / erfc(-xm / SQRT2))
    return ee


def _expectEsq_FW_acen(eosq, sigesq):
    """Acentric posterior <E²>. From `expectEsqFWacen` (lines 46-78)."""
    import numpy as np
    from scipy.special import erfc
    CROSS1, CROSS2 = -8.9, 5.7
    SQRT2_BY_PI = np.sqrt(2.0 / np.pi)
    SQRT2 = np.sqrt(2.0)
    eesq_base = eosq - sigesq ** 2
    x = eesq_base / (SQRT2 * sigesq)
    xsqr = x * x
    eesq = eesq_base.copy()
    m_neg = x < CROSS1
    if m_neg.any():
        xs = xsqr[m_neg]
        num = (-135135. + xs * (20790. + xs * (-3780. + xs *
                (840. + xs * (-240. + xs * (96. - xs * 64.))))))
        den = (-135135. + xs * (20790. + xs * (-3780. + xs *
                (840. + xs * (-240. + xs * (96. + xs *
                 (-64. + xs * 128.)))))))
        eesq[m_neg] = eesq_base[m_neg] * num / den
    m_mid = (x >= CROSS1) & (x <= CROSS2)
    if m_mid.any():
        xm = x[m_mid]
        eesq[m_mid] = (eesq_base[m_mid] +
                       SQRT2_BY_PI * sigesq[m_mid] /
                       (np.exp(xm * xm) * erfc(-xm)))
    return eesq


def _expectE_FW_cen(eosq, sigesq):
    """Centric posterior <E>. From `expectEFWcen` (lines 80-113)."""
    import numpy as np
    from scipy.special import pbdv
    CROSS1, CROSS2 = -17.5, 17.5
    SQRTPI = np.sqrt(np.pi)
    x = sigesq / 2.0 - eosq / sigesq
    xsqr = x * x
    pcdratio = np.empty_like(x)
    m_neg = x < CROSS1
    if m_neg.any():
        xn, xs = x[m_neg], xsqr[m_neg]
        pcdratio[m_neg] = ((1024. * SQRTPI * (-xn) ** 6.5) /
                           (3465. + xs *
                            (840. + xs *
                             (384. + xs * 1024.))))
    m_pos = x > CROSS2
    if m_pos.any():
        xp, xs = x[m_pos], xsqr[m_pos]
        num = (3440640. + xs *
               (-491520. + xs *
                (98304. + xs *
                 (-32768. + xs * 32768.))))
        den = (675675. + xs *
               (-110880. + xs *
                (26880. + xs *
                 (-12288. + xs * 32768.))))
        pcdratio[m_pos] = num / (den * np.sqrt(xp))
    m_mid = ~(m_neg | m_pos)
    if m_mid.any():
        xm = x[m_mid]
        d_neg1, _ = pbdv(-1.0, xm)
        d_neghalf, _ = pbdv(-0.5, xm)
        pcdratio[m_mid] = d_neg1 / d_neghalf
    return np.sqrt(sigesq / np.pi) * pcdratio


def _expectEsq_FW_cen(eosq, sigesq):
    """Centric posterior <E²>. From `expectEsqFWcen` (lines 115-152)."""
    import numpy as np
    from scipy.special import pbdv
    CROSS1, CROSS2 = -17.5, 17.5
    x = sigesq / 2.0 - eosq / sigesq
    xsqr = x * x
    pcdratio = np.empty_like(x)
    m_neg = x < CROSS1
    if m_neg.any():
        xn, xs = x[m_neg], xsqr[m_neg]
        num = (45045. + xs *
               (10080. + xs *
                (3840. + xs *
                 (4096. - xs * 32768.))))
        den = xn * (55440. + xs *
                    (13440. + xs *
                     (6144. + xs * 16384.)))
        pcdratio[m_neg] = num / den
    m_pos = x > CROSS2
    if m_pos.any():
        xp, xs = x[m_pos], xsqr[m_pos]
        num = (11486475. + xs *
               (-1441440. + xs *
                (241920. + xs *
                 (-61440. + xs * 32768.))))
        den = xp * (675675. + xs *
                    (-110880. + xs *
                     (26880. + xs *
                      (-12288. + xs * 32768.))))
        pcdratio[m_pos] = num / den
    m_mid = ~(m_neg | m_pos)
    if m_mid.any():
        xm = x[m_mid]
        d_neg15, _ = pbdv(-1.5, xm)
        d_neghalf, _ = pbdv(-0.5, xm)
        pcdratio[m_mid] = d_neg15 / d_neghalf
    return sigesq * pcdratio / 2.0


def _french_wilson_posterior(eosq, sigesq, centric_mask):
    """Wrap centric/acentric branches.

    Phaser `expectEFW` / `expectEsqFW` (lines 154-178): if sigesq <= 0 the
    measurement is treated as exact and (eEFW, eEsqFW) = (sqrt(eosq), eosq).
    """
    import numpy as np
    eEFW = np.empty_like(eosq)
    eEsqFW = np.empty_like(eosq)
    zero_sig = sigesq <= 0.0
    if zero_sig.any():
        eEFW[zero_sig] = np.sqrt(np.maximum(eosq[zero_sig], 0.0))
        eEsqFW[zero_sig] = np.maximum(eosq[zero_sig], 0.0)
    valid = ~zero_sig
    if valid.any():
        cen = centric_mask & valid
        acen = (~centric_mask) & valid
        if cen.any():
            eEFW[cen] = _expectE_FW_cen(eosq[cen], sigesq[cen])
            eEsqFW[cen] = _expectEsq_FW_cen(eosq[cen], sigesq[cen])
        if acen.any():
            eEFW[acen] = _expectE_FW_acen(eosq[acen], sigesq[acen])
            eEsqFW[acen] = _expectEsq_FW_acen(eosq[acen], sigesq[acen])
    return eEFW, eEsqFW


# -----------------------------------------------------------------------------
# DFAC via Halley iteration  (math_RiceLLG.cc:getDfactor)
# -----------------------------------------------------------------------------


def _i0e_full(x):
    """Phaser's `eBesselI0(x) = I0(x)·exp(-|x|)`. Symmetric in x."""
    import numpy as np
    from scipy.special import i0e
    return i0e(np.abs(x))


def _i1e_full(x):
    """Phaser's `eBesselI1(x) = I1(x)·exp(-|x|)`. Antisymmetric in x."""
    import numpy as np
    from scipy.special import i1e
    return np.sign(x) * i1e(np.abs(x))


def _effSigaRoot_acen(ee, eesq, sa):
    """`effSigaRootAcen` (math_RiceLLG.cc:12-34)."""
    import numpy as np
    sigbsqr = 1.0 - sa * sa
    x = 0.5 * (eesq - sigbsqr) / sigbsqr
    return (np.sqrt(np.pi * sigbsqr) / (2.0 * sigbsqr) *
            (eesq * _i0e_full(x) + (eesq - sigbsqr) * _i1e_full(x)) - ee)


def _deffSigaRoot_acen(eesq, sa):
    """`deffSigaRootAcen_by_dsa` (lines 36-52)."""
    import numpy as np
    sigbsqr = 1.0 - sa * sa
    x = 0.5 * (eesq - sigbsqr) / sigbsqr
    return np.sqrt(np.pi / sigbsqr) * (sa / 2.0) * _i1e_full(x)


def _d2effSigaRoot_acen(eesq, sa):
    """`d2effSigaRootAcen_by_dsa2` (lines 54-81)."""
    import numpy as np
    sigasqr = sa * sa
    sigapow4 = sigasqr * sigasqr
    sigbsqr = 1.0 - sigasqr
    xnum = eesq - sigbsqr
    x = 0.5 * xnum / sigbsqr
    out = np.empty_like(eesq)
    big = xnum > 1e-10
    if big.any():
        I0 = _i0e_full(x[big])
        I1 = _i1e_full(x[big])
        out[big] = (np.sqrt(np.pi / sigbsqr[big]) / (2.0 * sigbsqr[big] ** 2) *
                    (eesq[big] * sigasqr[big] * I0 +
                     (eesq[big] - 1.0 - (-2.0 + eesq[big] * (2.0 + eesq[big])) * sigasqr[big] +
                      (eesq[big] - 1.0) * sigapow4[big]) * I1 / xnum[big]))
    small = ~big
    if small.any():
        samin = np.sqrt(np.maximum(1.0 - eesq[small], 0.0))
        out[small] = (np.sqrt(np.pi) * samin *
                      ((3.0 + samin * samin) * sa[small] -
                       2.0 * (samin + samin ** 3)) /
                      (4.0 * eesq[small] ** 2.5))
    return out


def _effSigaRoot_cen(ee, eesq, sa):
    """`effSigaRootCen` (lines 83-105)."""
    import numpy as np
    from scipy.special import erf
    sigbsqr = 1.0 - sa * sa
    x = 0.5 * (eesq - sigbsqr) / sigbsqr
    x_safe = np.maximum(x, 0.0)
    return (np.exp(-x) * np.sqrt(2.0 * sigbsqr / np.pi) +
            np.sqrt(np.maximum(eesq - sigbsqr, 0.0)) * erf(np.sqrt(x_safe)) - ee)


def _deffSigaRoot_cen(eesq, sa):
    """`deffSigaRootCen_by_dsa` (lines 107-130)."""
    import numpy as np
    from scipy.special import erf
    sigbsqr = 1.0 - sa * sa
    xnum = eesq - sigbsqr
    x = 0.5 * xnum / sigbsqr
    out = np.empty_like(eesq)
    big = np.abs(xnum) > 1e-10
    if big.any():
        x_safe = np.maximum(x[big], 0.0)
        out[big] = (sa[big] * erf(np.sqrt(x_safe)) /
                    np.sqrt(np.maximum(xnum[big], 1e-30)) -
                    np.exp(-x[big]) * np.sqrt(2.0 * sigbsqr[big] / np.pi) *
                    sa[big] / sigbsqr[big])
    small = ~big
    if small.any():
        out[small] = (xnum[small] * np.sqrt(2.0 / np.pi) * sa[small] /
                      (3.0 * sigbsqr[small] ** 1.5))
    return out


def _d2effSigaRoot_cen(eesq, sa):
    """`d2effSigaRootCen_by_dsa2` (lines 132-159)."""
    import numpy as np
    from scipy.special import erf
    sigasqr = sa * sa
    sigapow4 = sigasqr * sigasqr
    sigbsqr = 1.0 - sigasqr
    xnum = eesq - sigbsqr
    x = 0.5 * xnum / sigbsqr
    sigbsqrtpi = np.sqrt(np.pi * sigbsqr)
    out = np.empty_like(eesq)
    big = np.abs(xnum) > 1e-10
    if big.any():
        x_safe = np.maximum(x[big], 0.0)
        d2num = ((eesq[big] - 1.0) * sigbsqr[big] ** 2 * sigbsqrtpi[big] *
                 erf(np.sqrt(x_safe)))
        exp_part = np.where(
            x[big] < 20.0,
            np.sqrt(np.maximum(2.0 * xnum[big], 0.0)) * np.exp(-x[big]) *
            (1.0 - eesq[big] + sigasqr[big] *
             (eesq[big] + eesq[big] ** 2 - 2.0) + sigapow4[big]),
            np.zeros_like(x[big]),
        )
        d2num = d2num + exp_part
        out[big] = d2num / (sigbsqr[big] ** 2 *
                            np.maximum(xnum[big], 1e-30) ** 1.5 *
                            sigbsqrtpi[big])
    small = ~big
    if small.any():
        out[small] = (np.sqrt(2.0 / np.pi) * sa[small] /
                      (1.5 * sigbsqr[small] ** 1.5))
    return out


def _get_dfactor_vectorised(ee_np, eesq_np, centric_np):
    """Vectorised port of Phaser's ``math_RiceLLG.cc:getDfactor`` (lines 191-250).

    Halley's method with bisection fallback, run over all reflections in
    parallel. Each reflection has its own bracket ``[dflo, dfhi]``. Returns a
    ``(N,)`` numpy float64 array of DFAC values in ``(0, 1)``.
    """
    import numpy as np

    EPS1 = 1e-7
    EPS2 = 1e-10
    MAXDFAC = 1.0 - EPS1

    ee = np.asarray(ee_np, dtype=np.float64)
    eesq = np.asarray(eesq_np, dtype=np.float64)
    cen = np.asarray(centric_np, dtype=bool)
    N = ee.shape[0]

    out = np.ones(N, dtype=np.float64)
    has_err = (eesq - ee * ee) > 0.0

    if not has_err.any():
        return out

    ee_a, eesq_a, cen_a = ee[has_err], eesq[has_err], cen[has_err]
    dflo = np.maximum(np.sqrt(np.maximum(1.0 - np.minimum(eesq_a, 1.0), 0.0)) + EPS1, EPS1)
    dfhi = np.full_like(dflo, MAXDFAC)

    early = dflo >= MAXDFAC
    if early.any():
        pass

    dfmid = 0.5 * (dflo + dfhi)
    fmid = np.empty_like(dfmid)
    if cen_a.any():
        fmid[cen_a] = _effSigaRoot_cen(ee_a[cen_a], eesq_a[cen_a], dfmid[cen_a])
    if (~cen_a).any():
        fmid[~cen_a] = _effSigaRoot_acen(ee_a[~cen_a], eesq_a[~cen_a], dfmid[~cen_a])

    active = ~early
    for _ in range(50):
        if not active.any():
            break
        conv = (dfhi - dflo) <= EPS1
        conv |= np.abs(fmid) <= EPS2
        active = active & ~conv
        if not active.any():
            break

        slope = np.empty_like(dfmid)
        curve = np.empty_like(dfmid)
        cen_act = cen_a & active
        acen_act = (~cen_a) & active
        if cen_act.any():
            slope[cen_act] = _deffSigaRoot_cen(eesq_a[cen_act], dfmid[cen_act])
            curve[cen_act] = _d2effSigaRoot_cen(eesq_a[cen_act], dfmid[cen_act])
        if acen_act.any():
            slope[acen_act] = _deffSigaRoot_acen(eesq_a[acen_act], dfmid[acen_act])
            curve[acen_act] = _d2effSigaRoot_acen(eesq_a[acen_act], dfmid[acen_act])

        denom_halley = 2.0 * (slope ** 2 - fmid * curve)
        use_halley = (curve > 0.0) & (np.abs(denom_halley) > 1e-30)
        step = np.where(
            use_halley,
            2.0 * fmid * slope / np.where(use_halley, denom_halley, 1.0),
            fmid * slope,
        )
        dfnew = dfmid - step
        in_bracket = (dfnew > dflo) & (dfnew < dfhi)
        dfmid_new = np.where(in_bracket, dfnew, 0.5 * (dflo + dfhi))

        dfmid = np.where(active, dfmid_new, dfmid)

        if cen_act.any():
            fmid[cen_act] = _effSigaRoot_cen(ee_a[cen_act], eesq_a[cen_act],
                                              dfmid[cen_act])
        if acen_act.any():
            fmid[acen_act] = _effSigaRoot_acen(ee_a[acen_act], eesq_a[acen_act],
                                                  dfmid[acen_act])

        below = (fmid < 0.0) & active
        above = (fmid >= 0.0) & active
        dflo = np.where(below, dfmid, dflo)
        dfhi = np.where(above, dfmid, dfhi)

    out[has_err] = dfmid
    return np.clip(out, EPS1, MAXDFAC)


def french_wilson_preprocess(
    F: torch.Tensor,
    sig_F: torch.Tensor,
    s_mag: torch.Tensor,
    centric: torch.Tensor,
    *,
    n_wilson_shells: int = 20,
) -> dict:
    """Phaser-style preprocessing from raw ``(F, σF, centric)`` to ``(eEobs, DFAC)``.

    Implements the chain:

    1. equal-count Wilson shells over ``s_mag``
    2. per-shell ``<F²>_p`` (Phaser's ``SIGMAN.BINS``)
    3. per-reflection normalised intensity ``eosq = F² / <F²>`` and σ
       ``sigesq = σI / <I> ≈ 2·F·σF / <F²>``
    4. French-Wilson posterior ``eEFW, eEsqFW`` (``math_FrenchWilson.cc``)
    5. DFAC via Halley iteration on Rice moments (``math_RiceLLG.cc``)
    6. ``eEobs = sqrt(eEsqFW + (DFAC²−1)/DFAC²)``, clamped to ≤10
       (Phaser ``Dfactor.cc:87-93``).

    Returns a dict with torch tensors back on the input device:
      eEobs: (N,) effective normalised amplitude
      DFAC : (N,) per-reflection D-factor ∈ [1e-7, 1−1e-7]
      sqrt_mean_F2: (N,) per-reflection √<F²>_p
    """
    import numpy as np

    device = F.device
    F_np = F.detach().to("cpu").to(torch.float64).numpy()
    sigF_np = sig_F.detach().to("cpu").to(torch.float64).numpy()
    s_np = s_mag.detach().to("cpu").to(torch.float64).numpy()
    cen_np = centric.detach().to("cpu").bool().numpy()

    sorted_idx = np.argsort(s_np)
    edges_idx = np.linspace(0, len(s_np) - 1, n_wilson_shells + 1).round().astype(np.int64)
    s_edges = s_np[sorted_idx][edges_idx]
    s_edges[0] -= 1e-6
    s_edges[-1] += 1e-6
    shell_idx = np.clip(
        np.searchsorted(s_edges, s_np, side="right") - 1, 0, n_wilson_shells - 1,
    )
    F2 = F_np * F_np
    mean_F2 = np.zeros(n_wilson_shells, dtype=np.float64)
    counts = np.zeros(n_wilson_shells, dtype=np.int64)
    np.add.at(mean_F2, shell_idx, F2)
    np.add.at(counts, shell_idx, 1)
    mean_F2 = mean_F2 / np.maximum(counts, 1)
    mean_F2 = np.maximum(mean_F2, 1e-12)
    mean_I_per_h = mean_F2[shell_idx]
    sqrt_mean_F2 = np.sqrt(mean_I_per_h)

    eosq = F2 / mean_I_per_h
    sigesq = 2.0 * F_np * sigF_np / mean_I_per_h
    sigesq = np.maximum(sigesq, 0.0)

    eEFW, eEsqFW = _french_wilson_posterior(eosq, sigesq, cen_np)
    bad = eEsqFW < eEFW * eEFW
    if bad.any():
        eEsqFW[bad] = eEFW[bad] ** 2 + 1e-12

    DFAC = _get_dfactor_vectorised(eEFW, eEsqFW, cen_np)

    dfsqr = DFAC * DFAC
    eEobs_sqr = eEsqFW + (dfsqr - 1.0) / np.maximum(dfsqr, 1e-30)
    eEobs_sqr = np.maximum(eEobs_sqr, 0.0)
    eEobs = np.sqrt(eEobs_sqr)
    clamp_mask = (eEobs > 10.0) & (eEsqFW > 1.0)
    if clamp_mask.any():
        eEobs[clamp_mask] = 10.0
        DFAC[clamp_mask] = 1.0 / np.sqrt(np.maximum(eEsqFW[clamp_mask] - 99.0, 1e-30))
        DFAC[clamp_mask] = np.clip(DFAC[clamp_mask], 1e-7, 1.0 - 1e-7)

    return {
        "eEobs": torch.from_numpy(eEobs).to(device=device, dtype=F.dtype),
        "DFAC": torch.from_numpy(DFAC).to(device=device, dtype=F.dtype),
        "sqrt_mean_F2": torch.from_numpy(sqrt_mean_F2).to(device=device, dtype=F.dtype),
    }
