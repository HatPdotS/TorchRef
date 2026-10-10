//! Isotropic splat: five Gaussians per atom with widths `max((B_k + B) / 4, 0.1)`.

use super::common::{Atoms, Cell, PI_1P5, PI_SQ, anchor, oz_run, row_offset, touches_x, wrap};
use crate::real::{Real, madd};
use multiversion::multiversion;
use multiversion::target::target_cfg_f;

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

/// Accumulate every atom's density into the x-planes `[xlo, xhi)`, stored in `chunk`.
#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl+avx2+fma+bmi1+bmi2+lzcnt+movbe+f16c",
    "x86_64+avx2+fma+bmi1+bmi2+lzcnt+movbe+f16c",
))]
pub fn fwd_planes<T: Real>(chunk: &mut [T], xlo: i32, xhi: i32, c: &Cell<T>, at: &Atoms<'_, T>) {
    if target_cfg_f!(any(target_feature = "fma", target_arch = "aarch64")) {
        fwd_planes_body::<T, true>(chunk, xlo, xhi, c, at)
    } else {
        fwd_planes_body::<T, false>(chunk, xlo, xhi, c, at)
    }
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
        for ox in -g.bhx..=g.bhx {
            let vix = wrap(g.cix + ox, c.nx);
            if vix < xlo || vix >= xhi {
                continue;
            }
            let fox = T::from_i32(ox);
            let p = [
                fox * c.ua[0] - g.w0[0],
                fox * c.ua[1] - g.w0[1],
                fox * c.ua[2] - g.w0[2],
            ];
            for oy in -g.bhy..=g.bhy {
                let foy = T::from_i32(oy);
                let q = [
                    madd::<T, F>(foy, c.ub[0], p[0]),
                    madd::<T, F>(foy, c.ub[1], p[1]),
                    madd::<T, F>(foy, c.ub[2], p[2]),
                ];
                let Some((zlo, zhi)) = oz_run(c, q, g.rc2, g.bhz) else {
                    continue;
                };
                let base = row_offset(vix, xlo, wrap(g.ciy + oy, c.ny), c.ny, c.nz);
                let row = &mut chunk[base..base + c.nz as usize];
                // Walk the run as wrap-free contiguous segments (at most two unless the
                // sphere is wider than the cell), so the voxel loop below has no index
                // arithmetic and vectorises across voxels.
                let mut oz = zlo;
                let mut iz = wrap(g.ciz + zlo, c.nz) as usize;
                while oz <= zhi {
                    let seg = ((zhi - oz + 1) as usize).min(row.len() - iz);
                    for (j, v) in row[iz..iz + seg].iter_mut().enumerate() {
                        let foz = T::from_i32(oz + j as i32);
                        let wx = madd::<T, F>(foz, c.uc[0], q[0]);
                        let wy = madd::<T, F>(foz, c.uc[1], q[1]);
                        let wz = madd::<T, F>(foz, c.uc[2], q[2]);
                        let r2 = madd::<T, F>(wx, wx, madd::<T, F>(wy, wy, wz * wz));
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
                    oz += seg as i32;
                    iz = 0;
                }
            }
        }
    }
}

/// Gradients of `sum(grad_out * splat)` for atom `i`: `([d xyz; 3], d adp, d occ)`.
#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl+avx2+fma+bmi1+bmi2+lzcnt+movbe+f16c",
    "x86_64+avx2+fma+bmi1+bmi2+lzcnt+movbe+f16c",
))]
pub fn bwd_atom<T: Real>(i: usize, grad: &[T], c: &Cell<T>, at: &Atoms<'_, T>) -> ([T; 3], T, T) {
    if target_cfg_f!(any(target_feature = "fma", target_arch = "aarch64")) {
        bwd_atom_body::<T, true>(i, grad, c, at)
    } else {
        bwd_atom_body::<T, false>(i, grad, c, at)
    }
}

#[inline(always)]
fn bwd_atom_body<T: Real, const F: bool>(
    i: usize,
    grad: &[T],
    c: &Cell<T>,
    at: &Atoms<'_, T>,
) -> ([T; 3], T, T) {
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
    let mut acc = [T::ZERO; 5];
    for ox in -g.bhx..=g.bhx {
        let vix = wrap(g.cix + ox, c.nx);
        let fox = T::from_i32(ox);
        let p = [
            fox * c.ua[0] - g.w0[0],
            fox * c.ua[1] - g.w0[1],
            fox * c.ua[2] - g.w0[2],
        ];
        for oy in -g.bhy..=g.bhy {
            let foy = T::from_i32(oy);
            let q = [
                madd::<T, F>(foy, c.ub[0], p[0]),
                madd::<T, F>(foy, c.ub[1], p[1]),
                madd::<T, F>(foy, c.ub[2], p[2]),
            ];
            let Some((zlo, zhi)) = oz_run(c, q, g.rc2, g.bhz) else {
                continue;
            };
            let base = row_offset(vix, 0, wrap(g.ciy + oy, c.ny), c.ny, c.nz);
            let row = &grad[base..base + c.nz as usize];
            // One in-sphere voxel's contribution `[d xyz; 3], d adp, d occ` before the
            // occ scale.
            let voxel = |foz: T, gout: T| -> [T; 5] {
                let wx = madd::<T, F>(foz, c.uc[0], q[0]);
                let wy = madd::<T, F>(foz, c.uc[1], q[1]);
                let wz = madd::<T, F>(foz, c.uc[2], q[2]);
                let r2 = madd::<T, F>(wx, wx, madd::<T, F>(wy, wy, wz * wz));
                let (mut dens, mut coeff, mut dbs) = (T::ZERO, T::ZERO, T::ZERO);
                for k in 0..5 {
                    let ae = an[k] * (r2 * inv_bt[k]).kexp::<F>();
                    dens += ae;
                    coeff = madd::<T, F>(ae, rcp_bt[k], coeff);
                    dbs = madd::<T, F>(ae, madd::<T, F>(r2, quad[k], lin[k]), dbs);
                }
                let s = gout * T::c(2.0 * PI_SQ) * coeff;
                [s * wx, s * wy, s * wz, gout * dbs, gout * dens]
            };
            let mut oz = zlo;
            let mut iz = wrap(g.ciz + zlo, c.nz) as usize;
            while oz <= zhi {
                let seg = ((zhi - oz + 1) as usize).min(row.len() - iz);
                let gseg = &row[iz..iz + seg];
                // A branch, not a 0/1 factor: the per-atom reduction does not
                // vectorise without reassociation, so skipping a voxel is a pure saving.
                for (j, &gout) in gseg.iter().enumerate() {
                    let foz = T::from_i32(oz + j as i32);
                    let wx = madd::<T, F>(foz, c.uc[0], q[0]);
                    let wy = madd::<T, F>(foz, c.uc[1], q[1]);
                    let wz = madd::<T, F>(foz, c.uc[2], q[2]);
                    if madd::<T, F>(wx, wx, madd::<T, F>(wy, wy, wz * wz)) > g.rc2 {
                        continue;
                    }
                    let d = voxel(foz, gout);
                    for t in 0..5 {
                        acc[t] += d[t];
                    }
                }
                oz += seg as i32;
                iz = 0;
            }
        }
    }
    (
        [oc * acc[0], oc * acc[1], oc * acc[2]],
        oc * T::c(0.25) * acc[3],
        acc[4],
    )
}
