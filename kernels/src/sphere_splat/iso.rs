//! Isotropic splat: five Gaussians per atom with widths `max((B_k + B) / 4, 0.1)`.

use super::common::{
    Atoms, Cell, PI_1P5, PI_SQ, TWO_PI_SQ, Tangent, anchor, for_each_segment, voxel,
};
use crate::real::{Real, madd};
use crate::simd::fma_variants;

/// Widths and amplitudes of atom `i`'s five Gaussians; `occ = 1` gives the occupancy-free
/// amplitudes the backward pass needs.
#[inline(always)]
fn gaussians<T: Real>(at: &Atoms<'_, T>, i: usize, occ: T) -> ([T; 5], [T; 5], [T; 5]) {
    let (mut bt, mut an, mut clampf) = ([T::ZERO; 5], [T::ZERO; 5], [T::ZERO; 5]);
    let floor = T::c(0.1);
    for k in 0..5 {
        let raw = (at.b[5 * i + k] + at.adp[i]) * T::c(0.25);
        let w = if raw > floor { raw } else { floor };
        bt[k] = w;
        an[k] = at.a[5 * i + k] * occ * T::c(PI_1P5) / (w * w.sqrt());
        // In the clamp region d(Bt)/d(adp) = 0.
        clampf[k] = if raw > floor { T::ONE } else { T::ZERO };
    }
    (bt, an, clampf)
}

fma_variants! {
    /// Accumulate every atom's density into the x-planes `[xlo, xhi)`, stored in `chunk`.
    pub fn fwd_planes<T: Real>(
        chunk: &mut [T], xlo: i32, xhi: i32, c: &Cell<T>, at: &Atoms<'_, T>, idx: &[u32]
    )
        => fwd_planes_body
}

#[inline(always)]
fn fwd_planes_body<T: Real, const F: bool>(
    chunk: &mut [T],
    xlo: i32,
    xhi: i32,
    c: &Cell<T>,
    at: &Atoms<'_, T>,
    idx: &[u32],
) {
    let neg_pi_sq = -T::c(PI_SQ);
    for &i in idx {
        let i = i as usize;
        let g = anchor(c, at, i);
        let (bt, an, _) = gaussians(at, i, at.occ[i]);
        let inv_bt = bt.map(|b| neg_pi_sq / b);
        for_each_segment::<T, F>(
            c,
            &g,
            xlo,
            xhi,
            #[inline(always)]
            |start, len, oz, q| {
                for (j, v) in chunk[start..start + len].iter_mut().enumerate() {
                    let (_, r2) = voxel::<T, F>(c, q, oz + j as i32);
                    if !T::VECTOR_EXP && r2 > g.rc2 {
                        continue;
                    }
                    let keep = if r2 <= g.rc2 { T::ONE } else { T::ZERO };
                    let mut dens = T::ZERO;
                    for k in 0..5 {
                        dens = madd::<T, F>(an[k], (r2 * inv_bt[k]).kexp::<F>(), dens);
                    }
                    *v = madd::<T, F>(keep, dens, *v);
                }
            },
        );
    }
}

fma_variants! {
    /// Gradients of `sum(grad_out * splat)` for atom `i`: `([d xyz; 3], [d adp], d occ)`.
    pub fn bwd_atom<T: Real>(i: usize, grad: &[T], c: &Cell<T>, at: &Atoms<'_, T>)
        -> ([T; 3], [T; 1], T) => bwd_atom_body
}

#[inline(always)]
fn bwd_atom_body<T: Real, const F: bool>(
    i: usize,
    grad: &[T],
    c: &Cell<T>,
    at: &Atoms<'_, T>,
) -> ([T; 3], [T; 1], T) {
    let neg_pi_sq = -T::c(PI_SQ);
    let g = anchor(c, at, i);
    // The density is linear in occ, so its gradient is the occ = 1 density, and occ only
    // scales the other gradients.
    let (bt, an, clampf) = gaussians(at, i, T::ONE);
    let inv_bt = bt.map(|b| neg_pi_sq / b);
    let rcp_bt = bt.map(|b| T::ONE / b);
    // d(density)/d(adp) per Gaussian is Ae * (-1.5 / Bt + pi^2 r^2 / Bt^2) * clamp.
    let lin = core::array::from_fn::<T, 5, _>(|k| -T::c(1.5) * rcp_bt[k] * clampf[k]);
    let quad = core::array::from_fn::<T, 5, _>(|k| T::c(PI_SQ) * rcp_bt[k] * rcp_bt[k] * clampf[k]);
    let oc = at.occ[i];
    // `[d xyz; 3], d adp, d occ` before the occ scale.
    let mut acc = [T::ZERO; 5];
    for_each_segment::<T, F>(
        c,
        &g,
        0,
        c.nx,
        #[inline(always)]
        |start, len, oz, q| {
            // A branch, not a 0/1 factor: the per-atom reduction does not vectorise without
            // reassociation, so skipping a voxel is a pure saving. The sums run in a local so
            // they stay in registers across the segment.
            let mut a = acc;
            for (j, &gout) in grad[start..start + len].iter().enumerate() {
                let (w, r2) = voxel::<T, F>(c, q, oz + j as i32);
                if r2 > g.rc2 {
                    continue;
                }
                let (mut dens, mut coeff, mut dbs) = (T::ZERO, T::ZERO, T::ZERO);
                for k in 0..5 {
                    let ae = an[k] * (r2 * inv_bt[k]).kexp::<F>();
                    dens += ae;
                    coeff = madd::<T, F>(ae, rcp_bt[k], coeff);
                    dbs = madd::<T, F>(ae, madd::<T, F>(r2, quad[k], lin[k]), dbs);
                }
                let s = gout * T::c(2.0 * PI_SQ) * coeff;
                let d = [s * w[0], s * w[1], s * w[2], gout * dbs, gout * dens];
                for t in 0..5 {
                    a[t] += d[t];
                }
            }
            acc = a;
        },
    );
    (
        [oc * acc[0], oc * acc[1], oc * acc[2]],
        [oc * T::c(0.25) * acc[3]],
        acc[4],
    )
}

/// Per-Gaussian sums the second-order passes need at one voxel, occupancy-free, with
/// `s = 1 / Bt`, `kap = d(Bt)/d(adp)` and `L = d ln(rho_k)/d(Bt)`: the density `D`,
/// `X1 = sum s rho_k` and `X2 = sum s^2 rho_k` (the xyz derivatives), `Bb1 = d rho/d adp`,
/// and the xyz-adp and adp-adp curvatures `XB`, `BB`.
#[inline(always)]
fn curvature<T: Real, const F: bool>(
    r2: T,
    an: &[T; 5],
    inv_bt: &[T; 5],
    rcp_bt: &[T; 5],
    kap: &[T; 5],
) -> [T; 6] {
    let mut acc = [T::ZERO; 6];
    for k in 0..5 {
        let s = rcp_bt[k];
        let d = an[k] * (r2 * inv_bt[k]).kexp::<F>();
        let l = s * madd::<T, F>(T::c(PI_SQ) * r2, s, -T::c(1.5));
        let sd = s * d;
        let kd = kap[k] * d;
        acc[0] += d;
        acc[1] += sd;
        acc[2] = madd::<T, F>(s, sd, acc[2]);
        acc[3] = madd::<T, F>(kd, l, acc[3]);
        acc[4] = madd::<T, F>(kd * s, l - s, acc[4]);
        let bb = madd::<T, F>(
            l,
            l,
            s * s * madd::<T, F>(-T::c(TWO_PI_SQ) * r2, s, T::c(1.5)),
        );
        acc[5] = madd::<T, F>(kap[k] * kd, bb, acc[5]);
    }
    acc
}

/// Atom `i`'s Gaussians and its cotangent for the second-order passes:
/// `(inv_bt = -pi^2 / Bt, 1 / Bt, kap, occupancy-free amplitudes, t_xyz, t_adp, t_occ)`.
#[allow(clippy::type_complexity)]
#[inline(always)]
fn second_order_setup<T: Real>(
    at: &Atoms<'_, T>,
    t: &Tangent<'_, T>,
    i: usize,
) -> ([T; 5], [T; 5], [T; 5], [T; 5], [T; 3], T, T) {
    let (bt, an, clampf) = gaussians(at, i, T::ONE);
    let neg_pi_sq = -T::c(PI_SQ);
    (
        bt.map(|b| neg_pi_sq / b),
        bt.map(|b| T::ONE / b),
        clampf.map(|f| T::c(0.25) * f),
        an,
        [t.xyz[3 * i], t.xyz[3 * i + 1], t.xyz[3 * i + 2]],
        t.adp[i],
        t.occ[i],
    )
}

fma_variants! {
    /// Accumulate every atom's tangent map `sum_p t_p d(density)/dp` into the x-planes
    /// `[xlo, xhi)`, stored in `chunk`.
    pub fn tan_planes<T: Real>(
        chunk: &mut [T], xlo: i32, xhi: i32, c: &Cell<T>, at: &Atoms<'_, T>, t: &Tangent<'_, T>,
        idx: &[u32]
    ) => tan_planes_body
}

#[inline(always)]
fn tan_planes_body<T: Real, const F: bool>(
    chunk: &mut [T],
    xlo: i32,
    xhi: i32,
    c: &Cell<T>,
    at: &Atoms<'_, T>,
    t: &Tangent<'_, T>,
    idx: &[u32],
) {
    let tp = T::c(TWO_PI_SQ);
    for &i in idx {
        let i = i as usize;
        let g = anchor(c, at, i);
        let (inv_bt, rcp_bt, kap, an, tx, tb, to) = second_order_setup(at, t, i);
        let oc = at.occ[i];
        for_each_segment::<T, F>(
            c,
            &g,
            xlo,
            xhi,
            #[inline(always)]
            |start, len, oz, q| {
                for (j, v) in chunk[start..start + len].iter_mut().enumerate() {
                    let (w, r2) = voxel::<T, F>(c, q, oz + j as i32);
                    if !T::VECTOR_EXP && r2 > g.rc2 {
                        continue;
                    }
                    let keep = if r2 <= g.rc2 { T::ONE } else { T::ZERO };
                    let [d, x1, _, bb1, _, _] = curvature::<T, F>(r2, &an, &inv_bt, &rcp_bt, &kap);
                    let wt = madd::<T, F>(w[0], tx[0], madd::<T, F>(w[1], tx[1], w[2] * tx[2]));
                    let jv = madd::<T, F>(tp * x1, wt, bb1 * tb);
                    *v = madd::<T, F>(keep, madd::<T, F>(oc, jv, to * d), *v);
                }
            },
        );
    }
}

fma_variants! {
    /// Atom `i`'s Hessian-vector product, the gradient of `sum(grad_out * T)` with `T` the
    /// tangent map of [`tan_planes`]: `([d xyz; 3], [d adp], d occ)`.
    pub fn hvp_atom<T: Real>(i: usize, grad: &[T], c: &Cell<T>, at: &Atoms<'_, T>, t: &Tangent<'_, T>)
        -> ([T; 3], [T; 1], T) => hvp_atom_body
}

#[inline(always)]
fn hvp_atom_body<T: Real, const F: bool>(
    i: usize,
    grad: &[T],
    c: &Cell<T>,
    at: &Atoms<'_, T>,
    t: &Tangent<'_, T>,
) -> ([T; 3], [T; 1], T) {
    let tp = T::c(TWO_PI_SQ);
    let g = anchor(c, at, i);
    let (inv_bt, rcp_bt, kap, an, tx, tb, to) = second_order_setup(at, t, i);
    let oc = at.occ[i];
    // `[w-weighted xyz; 3], X1, d adp, d occ`, with the -2 pi^2 X1 I part of d2/dxyz2
    // applied after the sum.
    let mut acc = [T::ZERO; 6];
    for_each_segment::<T, F>(
        c,
        &g,
        0,
        c.nx,
        #[inline(always)]
        |start, len, oz, q| {
            // Local sums stay in registers across the segment, as in the backward.
            let mut a = acc;
            for (j, &gout) in grad[start..start + len].iter().enumerate() {
                let (w, r2) = voxel::<T, F>(c, q, oz + j as i32);
                if r2 > g.rc2 {
                    continue;
                }
                let [_, x1, x2, bb1, xb, bb] = curvature::<T, F>(r2, &an, &inv_bt, &rcp_bt, &kap);
                let wt = madd::<T, F>(w[0], tx[0], madd::<T, F>(w[1], tx[1], w[2] * tx[2]));
                let alpha = madd::<T, F>(
                    oc,
                    madd::<T, F>(tp * tp * x2, wt, tp * xb * tb),
                    to * tp * x1,
                );
                let ga = gout * alpha;
                let d = [
                    ga * w[0],
                    ga * w[1],
                    ga * w[2],
                    gout * x1,
                    gout * madd::<T, F>(oc, madd::<T, F>(tp * xb, wt, bb * tb), to * bb1),
                    gout * madd::<T, F>(tp * x1, wt, bb1 * tb),
                ];
                for k in 0..6 {
                    a[k] += d[k];
                }
            }
            acc = a;
        },
    );
    let diag = oc * tp * acc[3];
    (
        [
            acc[0] - diag * tx[0],
            acc[1] - diag * tx[1],
            acc[2] - diag * tx[2],
        ],
        [acc[4]],
        acc[5],
    )
}
