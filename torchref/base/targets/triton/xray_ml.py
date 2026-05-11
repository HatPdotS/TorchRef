"""Triton kernels for the Maximum-Likelihood X-ray target.

Forward computes log(I0e(arg)) using Abramowitz & Stegun 9.8.1/9.8.3
polynomial approximations split at x=3.75:

  x < 3.75:  I0(x) = Poly_small(t²),    t = x/3.75
             log(I0e(x)) = log(I0(x)) - x

  x ≥ 3.75:  sqrt(x)·exp(-x)·I0(x) = Poly_large(t),    t = 3.75/x
             log(I0e(x)) = log(Poly_large(t)) - 0.5·log(x)

Backward needs I1(x)/I0(x), which is computed from the same polynomial
pair (Abramowitz 9.8.3-9.8.4). This avoids overflow at the clamped
arg_bessel = 1e6 ceiling.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


_LOG_PI = float(math.log(math.pi))


# Polynomials below are Abramowitz & Stegun 9.8.1-9.8.4. Each `_p<n>`
# constant becomes a kernel constexpr so the compiler can fold them.

# --- I0/I1 for x < 3.75, in t² = (x/3.75)² ---
# I0 polynomial: a0 + a1*y + a2*y² + ... where y = t²
_I0S = (
    1.0, 3.5156229, 3.0899424, 1.2067492,
    0.2659732, 0.0360768, 0.0045813,
)
# I1(x)/x polynomial in y = t²
_I1S = (
    0.5, 0.87890594, 0.51498869, 0.15084934,
    0.02658733, 0.00301532, 0.00032411,
)

# --- sqrt(x)*exp(-x)*I0(x) for x ≥ 3.75, in t = 3.75/x ---
_I0L = (
    0.39894228,  0.01328592, 0.00225319, -0.00157565,
    0.00916281, -0.02057706, 0.02635537, -0.01647633,
    0.00392377,
)
# sqrt(x)*exp(-x)*I1(x)
_I1L = (
    0.39894228, -0.03988024, -0.00362018,  0.00163801,
   -0.01031555,  0.02282967, -0.02895312,  0.01787654,
   -0.00420059,
)


@triton.jit
def _log_i0e_and_ratio(x):
    """Return (log_I0e(x), I1(x)/I0(x)) for x ≥ 0 via A&S 9.8.

    Branchless: both branches are computed, then ``tl.where`` picks the
    correct one. Avoids divergence and keeps the kernel simple.
    """
    SMALL_BOUND = 3.75

    # Small-argument branch (x < 3.75):
    t = x / SMALL_BOUND
    y = t * t
    i0_small = (
        1.0
        + y * (3.5156229
        + y * (3.0899424
        + y * (1.2067492
        + y * (0.2659732
        + y * (0.0360768
        + y * 0.0045813)))))
    )
    i1_over_x_small = (
        0.5
        + y * (0.87890594
        + y * (0.51498869
        + y * (0.15084934
        + y * (0.02658733
        + y * (0.00301532
        + y * 0.00032411)))))
    )
    log_i0e_small = tl.log(i0_small) - x
    ratio_small = x * i1_over_x_small / i0_small

    # Large-argument branch (x ≥ 3.75):
    x_safe = tl.where(x > 1e-6, x, 1e-6)  # protect /x in else branch only
    u = SMALL_BOUND / x_safe
    p_i0_large = (
        0.39894228
        + u * (0.01328592
        + u * (0.00225319
        + u * (-0.00157565
        + u * (0.00916281
        + u * (-0.02057706
        + u * (0.02635537
        + u * (-0.01647633
        + u * 0.00392377)))))))
    )
    p_i1_large = (
        0.39894228
        + u * (-0.03988024
        + u * (-0.00362018
        + u * (0.00163801
        + u * (-0.01031555
        + u * (0.02282967
        + u * (-0.02895312
        + u * (0.01787654
        + u * -0.00420059)))))))
    )
    # log(I0e) = log(sqrt(x)*exp(-x)*I0(x)) - 0.5*log(x)
    log_i0e_large = tl.log(p_i0_large) - 0.5 * tl.log(x_safe)
    ratio_large = p_i1_large / p_i0_large

    use_small = x < SMALL_BOUND
    log_i0e = tl.where(use_small, log_i0e_small, log_i0e_large)
    ratio = tl.where(use_small, ratio_small, ratio_large)
    return log_i0e, ratio


@triton.jit
def _ml_fwd_kernel(
    F_obs_ptr,
    F_calc_ptr,
    sigma_ptr,
    centric_ptr,    # uint8 (0 = acentric, 1 = centric)
    mask_ptr,       # uint8
    out_ptr,
    N: tl.constexpr,
    LOG_PI: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < N

    F_obs = tl.load(F_obs_ptr + offs, mask=valid, other=0.0)
    F_calc = tl.load(F_calc_ptr + offs, mask=valid, other=0.0)
    sig = tl.load(sigma_ptr + offs, mask=valid, other=1.0)
    centric = tl.load(centric_ptr + offs, mask=valid, other=0).to(tl.int1)
    m = tl.load(mask_ptr + offs, mask=valid, other=0).to(tl.float32)

    # alpha = 1, epsilon = 1
    eb = sig * sig
    eb = tl.where(eb < 1e-6, 1e-6, eb)
    inv_eb = 1.0 / eb

    F_obs_sq = F_obs * F_obs
    F_calc_sq = F_calc * F_calc

    # ---- acentric ----
    term1 = -tl.log(2.0 * F_obs * inv_eb + 1e-12)
    term2 = F_obs_sq * inv_eb
    term3 = F_calc_sq * inv_eb
    arg_b = 2.0 * F_obs * F_calc * inv_eb
    arg_b = tl.where(arg_b > 1e6, 1e6, arg_b)
    log_i0e, _ratio = _log_i0e_and_ratio(arg_b)
    # term4 = -(log(I0e(arg) + 1e-12) + arg); for arg ≥ 0, I0e ≥ ~1e-4 even
    # at arg=1e6, so the 1e-12 floor is negligible.
    term4 = -(log_i0e + arg_b)
    loss_a = term1 + term2 + term3 + term4

    # ---- centric ----
    # term1_c = -0.5 * log(2/(pi*eb) + 1e-12) = -0.5 * log(2 - pi*eb*1e-12 ... ) ; just compute
    term1_c = -0.5 * tl.log(2.0 * inv_eb / 3.141592653589793 + 1e-12)
    term2_c = 0.5 * F_obs_sq * inv_eb
    term3_c = 0.5 * F_calc_sq * inv_eb
    term4_c = -F_obs * F_calc * inv_eb
    arg_e = -2.0 * F_obs * F_calc * inv_eb
    arg_e = tl.where(arg_e < -80.0, -80.0, tl.where(arg_e > 80.0, 80.0, arg_e))
    term5_c = -tl.log((1.0 + tl.exp(arg_e)) * 0.5 + 1e-12)
    loss_c = term1_c + term2_c + term3_c + term4_c + term5_c

    loss = tl.where(centric, loss_c, loss_a)
    # NaN/Inf scrub — match torch.where(isfinite, loss, 1e6)
    finite = (loss == loss) & (loss < 1e30) & (loss > -1e30)
    loss = tl.where(finite, loss, 1e6)

    tl.store(out_ptr + offs, loss * m, mask=valid)


@triton.jit
def _ml_bwd_kernel(
    F_obs_ptr,
    F_calc_ptr,
    sigma_ptr,
    centric_ptr,
    mask_ptr,
    grad_out,
    dF_calc_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < N

    F_obs = tl.load(F_obs_ptr + offs, mask=valid, other=0.0)
    F_calc = tl.load(F_calc_ptr + offs, mask=valid, other=0.0)
    sig = tl.load(sigma_ptr + offs, mask=valid, other=1.0)
    centric = tl.load(centric_ptr + offs, mask=valid, other=0).to(tl.int1)
    m = tl.load(mask_ptr + offs, mask=valid, other=0).to(tl.float32)

    eb = sig * sig
    eb = tl.where(eb < 1e-6, 1e-6, eb)
    inv_eb = 1.0 / eb

    # ---- acentric gradient ----
    # term3 = F_calc^2 / eb  ->  2*F_calc/eb
    # term4 = -log(I0e(arg)) - arg  ->  d/dF_calc =
    #   -[d log(I0e)/d_arg + 1] * d_arg/dF_calc
    #   d log(I0e(x))/dx = I1(x)/I0(x) - 1
    #   so d term4 / dF_calc = -[(I1/I0 - 1) + 1] * (2 F_obs/eb) = -(I1/I0)*(2 F_obs/eb)
    # Inside the kernel we additionally check whether arg_bessel was clamped:
    # when arg ≥ 1e6 the gradient w.r.t. F_calc is zero through that branch.
    arg_b_raw = 2.0 * F_obs * F_calc * inv_eb
    clamped_b = arg_b_raw >= 1e6
    arg_b = tl.where(clamped_b, 1e6, arg_b_raw)
    _log_i0e, ratio = _log_i0e_and_ratio(arg_b)
    d_arg_dFcalc = tl.where(clamped_b, 0.0, 2.0 * F_obs * inv_eb)
    dL_a = 2.0 * F_calc * inv_eb - ratio * d_arg_dFcalc

    # ---- centric gradient ----
    # term3_c = 0.5*F_calc^2/eb -> F_calc/eb
    # term4_c = -F_obs*F_calc/eb -> -F_obs/eb
    # term5_c = -log((1 + exp(u))/2 + 1e-12), u = -2*F_obs*F_calc/eb (clamped)
    #   d/du log((1+exp(u))/2 + eps) = (0.5*exp(u)) / ((1+exp(u))/2 + eps)
    #                                 = exp(u) / (1 + exp(u) + 2*eps)
    #   d term5_c / du = - sigmoid_like = -exp(u) / (1 + exp(u) + 2e-12)
    #   d u / dF_calc = -2*F_obs/eb       (clamped to 0 outside [-80, 80])
    u_raw = -2.0 * F_obs * F_calc * inv_eb
    clamped_lo = u_raw < -80.0
    clamped_hi = u_raw > 80.0
    u = tl.where(clamped_lo, -80.0, tl.where(clamped_hi, 80.0, u_raw))
    eu = tl.exp(u)
    d_term5_du = -eu / (1.0 + eu + 2e-12)
    d_u_dFcalc = tl.where(clamped_lo | clamped_hi, 0.0, -2.0 * F_obs * inv_eb)
    dL_c = F_calc * inv_eb - F_obs * inv_eb + d_term5_du * d_u_dFcalc

    dL = tl.where(centric, dL_c, dL_a)

    # NaN/Inf scrub: if loss was non-finite, the eager code replaces with 1e6
    # constant, whose derivative wrt F_calc is 0. Zero out non-finite grads here.
    finite = (dL == dL) & (dL < 1e30) & (dL > -1e30)
    dL = tl.where(finite, dL, 0.0)

    g = grad_out * dL * m
    tl.store(dF_calc_ptr + offs, g, mask=valid)


class _MLXrayMathTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, F_obs, F_calc, sigma, centric_flags, mask):
        assert F_calc.is_cuda and F_calc.dtype == torch.float32
        N = F_calc.shape[0]
        F_obs = F_obs.contiguous()
        F_calc = F_calc.contiguous()
        sigma = sigma.contiguous()
        if centric_flags is None:
            centric_u8 = torch.zeros(N, dtype=torch.uint8, device=F_calc.device)
        else:
            centric_u8 = centric_flags.to(torch.uint8).contiguous()
        mask_u8 = mask.to(torch.uint8).contiguous()

        out = torch.empty(N, dtype=F_calc.dtype, device=F_calc.device)
        BLOCK = 512
        grid = (triton.cdiv(N, BLOCK),)
        _ml_fwd_kernel[grid](
            F_obs, F_calc, sigma, centric_u8, mask_u8, out,
            N=N, LOG_PI=_LOG_PI, BLOCK=BLOCK,
        )
        ctx.save_for_backward(F_obs, F_calc, sigma, centric_u8, mask_u8)
        return out.sum()

    @staticmethod
    def backward(ctx, grad_out):
        F_obs, F_calc, sigma, centric_u8, mask_u8 = ctx.saved_tensors
        N = F_calc.shape[0]
        dF_calc = torch.empty_like(F_calc)
        BLOCK = 512
        grid = (triton.cdiv(N, BLOCK),)
        _ml_bwd_kernel[grid](
            F_obs, F_calc, sigma, centric_u8, mask_u8,
            float(grad_out.item()), dF_calc,
            N=N, BLOCK=BLOCK,
        )
        return None, dF_calc, None, None, None


def ml_xray_loss_math_triton(
    F_obs: torch.Tensor,
    F_calc: torch.Tensor,
    sigma: torch.Tensor,
    centric_flags,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Triton-backed maximum-likelihood X-ray loss.

    Drop-in replacement for
    :func:`torchref.base.targets.xray_ml.ml_xray_loss_math`.
    """
    return _MLXrayMathTriton.apply(F_obs, F_calc, sigma, centric_flags, mask)
