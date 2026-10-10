//! The scalar abstraction the kernels are generic over (`f32`, `f64`).
//!
//! Constants go through [`Real::c`] from an `f64` literal, which rounds exactly like the
//! C-style `(scalar_t)0.25` casts of the kernels' reference formulation.

use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

pub trait Real:
    Copy
    + Send
    + Sync
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + 'static
{
    const ZERO: Self;
    const ONE: Self;
    /// Whether [`Real::kexp`] is branch-free inline code the compiler can vectorise.
    /// Kernels skip out-of-sphere voxels with a branch when it is not, since there every
    /// avoided `exp` is a real saving, and with a 0/1 multiply when it is.
    const VECTOR_EXP: bool;
    fn c(x: f64) -> Self;
    fn from_i32(i: i32) -> Self;
    fn sqrt(self) -> Self;
    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    /// Truncate toward zero, as a C `(int)` cast does for in-range values.
    fn to_i32(self) -> i32;
    /// Round half to even, as `std::nearbyint` does in the default rounding mode.
    fn round_even_i32(self) -> i32;
    /// Fused `self * b + c`; only fast where the target has FMA (see [`madd`]).
    fn fused_mul_add(self, b: Self, c: Self) -> Self;
    /// The kernels' exponential: [`fast_exp`] for `f32`, libm `exp` for `f64`.
    fn kexp<const F: bool>(self) -> Self;
}

/// `a * b + c`, fused when `F`.
///
/// Kernels pass `F` = "this CPU variant has FMA" (`multiversion`'s `target_cfg_f!`), so
/// aarch64 and x86-64-v3/v4 get one rounding and one instruction, while the baseline x86
/// build keeps the plain form instead of falling back to a slow software `fma` call.
#[inline(always)]
pub fn madd<T: Real, const F: bool>(a: T, b: T, c: T) -> T {
    if F { a.fused_mul_add(b, c) } else { a * b + c }
}

/// Branchless `exp` for `f32`.
///
/// `exp(x) = 2^(x log2 e)`: the integer part goes straight into the IEEE exponent field, the
/// fractional part through a degree-5 minimax polynomial for `2^f` on `[0, 1)` with
/// `p(0) = 1`. Its relative error (8.5e-8, 1.7e-7 in float32 Horner) alternates in sign. The
/// Taylor coefficients `ln2^k / k!` would all err low, by up to 8.5e-5, and bias every map,
/// so the minimax coefficients are load-bearing. No branches, so the voxel loop vectorises.
#[inline(always)]
#[allow(clippy::excessive_precision)]
pub fn fast_exp<const F: bool>(x: f32) -> f32 {
    let x = if x < -87.0_f32 { -87.0_f32 } else { x }; // below this, exp underflows
    let t = x * 1.442_695_040_888_963_41_f32;
    let n = t.floor();
    let f = t - n;
    let mut p = madd::<f32, F>(f, 0.001_867_13_f32, 0.009_017_03_f32);
    p = madd::<f32, F>(f, p, 0.055_799_91_f32);
    p = madd::<f32, F>(f, p, 0.240_164_45_f32);
    p = madd::<f32, F>(f, p, 0.693_151_31_f32);
    p = madd::<f32, F>(f, p, 1.0_f32);
    let bits = (((n + 127.0_f32) * 8_388_608.0_f32) as i32) & 0x7f80_0000;
    p * f32::from_bits(bits as u32)
}

impl Real for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const VECTOR_EXP: bool = true;
    #[inline(always)]
    fn c(x: f64) -> Self {
        x as f32
    }
    #[inline(always)]
    fn from_i32(i: i32) -> Self {
        i as f32
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
    #[inline(always)]
    fn floor(self) -> Self {
        f32::floor(self)
    }
    #[inline(always)]
    fn ceil(self) -> Self {
        f32::ceil(self)
    }
    #[inline(always)]
    fn to_i32(self) -> i32 {
        self as i32
    }
    #[inline(always)]
    fn round_even_i32(self) -> i32 {
        f32::round_ties_even(self) as i32
    }
    #[inline(always)]
    fn fused_mul_add(self, b: Self, c: Self) -> Self {
        f32::mul_add(self, b, c)
    }
    #[inline(always)]
    fn kexp<const F: bool>(self) -> Self {
        fast_exp::<F>(self)
    }
}

impl Real for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const VECTOR_EXP: bool = false;
    #[inline(always)]
    fn c(x: f64) -> Self {
        x
    }
    #[inline(always)]
    fn from_i32(i: i32) -> Self {
        i as f64
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    #[inline(always)]
    fn floor(self) -> Self {
        f64::floor(self)
    }
    #[inline(always)]
    fn ceil(self) -> Self {
        f64::ceil(self)
    }
    #[inline(always)]
    fn to_i32(self) -> i32 {
        self as i32
    }
    #[inline(always)]
    fn round_even_i32(self) -> i32 {
        f64::round_ties_even(self) as i32
    }
    // A float64 caller is precision-motivated by definition, so it gets libm.
    #[inline(always)]
    fn fused_mul_add(self, b: Self, c: Self) -> Self {
        f64::mul_add(self, b, c)
    }
    #[inline(always)]
    fn kexp<const F: bool>(self) -> Self {
        f64::exp(self)
    }
}

#[cfg(test)]
mod tests {
    use super::fast_exp;

    fn fast_exp_both(x: f32) -> (f32, f32) {
        (fast_exp::<false>(x), fast_exp::<true>(x))
    }

    fn worst_relative_error(lo: f32, hi: f32) -> f64 {
        let mut worst = 0.0_f64;
        let mut x = lo;
        while x <= hi {
            let (a, b) = fast_exp_both(x);
            for v in [a, b] {
                let rel = ((v as f64) - (x as f64).exp()).abs() / (x as f64).exp();
                worst = worst.max(rel);
            }
            x += 1.0e-4;
        }
        worst
    }

    #[test]
    fn fast_exp_polynomial_error_is_below_2e7() {
        // Near zero the argument reduction is exact, so this isolates the polynomial.
        let worst = worst_relative_error(-1.0, 0.0);
        assert!(worst < 2.0e-7, "worst relative error {worst}");
    }

    #[test]
    fn fast_exp_error_is_bounded_by_argument_rounding() {
        // Rounding x * log2(e) to float32 costs up to ln2 * ulp(|t|) / 2 relative error,
        // about 3e-6 at |t| = 125; the polynomial adds nothing comparable.
        let worst = worst_relative_error(-87.0, 0.0);
        assert!(worst < 5.0e-6, "worst relative error {worst}");
    }

    #[test]
    fn fast_exp_clamps_instead_of_producing_garbage() {
        for v in [fast_exp::<false>(-1.0e4), fast_exp::<true>(-1.0e4)] {
            assert!((0.0..1.0e-37).contains(&v));
        }
    }
}
