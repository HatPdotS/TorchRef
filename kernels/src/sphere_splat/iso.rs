//! Isotropic splat: five Gaussians per atom with widths `max((B_k + B) / 4, 0.1)`.

use super::common::{Atoms, Cell, PI_1P5, PI_SQ, anchor, for_each_segment, touches_x, voxel};
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
    pub fn fwd_planes<T: Real>(chunk: &mut [T], xlo: i32, xhi: i32, c: &Cell<T>, at: &Atoms<'_, T>)
        => fwd_planes_body
}

#[inline(always)]
fn fwd_planes_body<T: Real, const F: bool>(
    chunk: &mut [T],
    xlo: i32,
    xhi: i32,
    c: &Cell<T>,
    at: &Atoms<'_, T>,
) {
    let neg_pi_sq = -T::c(PI_SQ);
    for i in 0..at.n_atoms {
        let g = anchor(c, at, i);
        if !touches_x(g.cix, g.bhx, c.nx, xlo, xhi) {
            continue;
        }
        let (bt, an, _) = gaussians(at, i, at.occ[i]);
        let inv_bt = bt.map(|b| neg_pi_sq / b);
        for_each_segment::<T, F>(c, &g, xlo, xhi, |start, len, oz, q| {
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
        });
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
    for_each_segment::<T, F>(c, &g, 0, c.nx, |start, len, oz, q| {
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
    });
    (
        [oc * acc[0], oc * acc[1], oc * acc[2]],
        [oc * T::c(0.25) * acc[3]],
        acc[4],
    )
}
