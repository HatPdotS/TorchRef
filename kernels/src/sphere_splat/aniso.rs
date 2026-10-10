//! Anisotropic splat. `M_g = (B_g I + 8 pi^2 U) / 4` is inverted analytically; the density
//! uses the Mahalanobis form `q = w^T M^-1 w`, while the cutoff stays the Euclidean sphere.

use super::common::{
    Atoms, Cell, PI_1P5, PI_SQ, TWO_PI_SQ, anchor, oz_run, row_offset, touches_x, wrap,
};
use crate::real::{Real, madd};
use multiversion::multiversion;
use multiversion::target::target_cfg_f;

/// Upper triangle of `M_g^-1` and the amplitude, for each of atom `i`'s five Gaussians.
struct Minv<T> {
    p00: [T; 5],
    p11: [T; 5],
    p22: [T; 5],
    p01: [T; 5],
    p02: [T; 5],
    p12: [T; 5],
    an: [T; 5],
}

#[inline(always)]
fn minv<T: Real>(at: &Atoms<'_, T>, i: usize, occ: T) -> Minv<T> {
    let t = T::c(TWO_PI_SQ);
    let u = &at.adp[6 * i..6 * i + 6];
    let (md, me, mf) = (t * u[3], t * u[4], t * u[5]);
    let mut m = Minv {
        p00: [T::ZERO; 5],
        p11: [T::ZERO; 5],
        p22: [T::ZERO; 5],
        p01: [T::ZERO; 5],
        p02: [T::ZERO; 5],
        p12: [T::ZERO; 5],
        an: [T::ZERO; 5],
    };
    for k in 0..5 {
        let bg = at.b[5 * i + k];
        let ma = T::c(0.25) * bg + t * u[0];
        let mb = T::c(0.25) * bg + t * u[1];
        let mc = T::c(0.25) * bg + t * u[2];
        let det = ma * (mb * mc - mf * mf) - md * (md * mc - me * mf) + me * (md * mf - me * mb);
        let inv = T::ONE / det;
        m.p00[k] = (mb * mc - mf * mf) * inv;
        m.p11[k] = (ma * mc - me * me) * inv;
        m.p22[k] = (ma * mb - md * md) * inv;
        m.p01[k] = (me * mf - md * mc) * inv;
        m.p02[k] = (md * mf - me * mb) * inv;
        m.p12[k] = (md * me - ma * mf) * inv;
        let floor = T::c(1e-10);
        let d = if det > floor { det } else { floor };
        m.an[k] = at.a[5 * i + k] * occ * T::c(PI_1P5) / d.sqrt();
    }
    m
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
        let m = minv(at, i, at.occ[i]);
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
                // Wrap-free contiguous segments and a 0/1 keep factor instead of a
                // branch, so the voxel loop vectorises (as in the isotropic kernel).
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
                        let (xx, yy, zz) = (wx * wx, wy * wy, wz * wz);
                        let (xy, xz, yz) = (wx * wy, wx * wz, wy * wz);
                        let mut dens = T::ZERO;
                        for k in 0..5 {
                            let off = madd::<T, F>(
                                m.p01[k],
                                xy,
                                madd::<T, F>(m.p02[k], xz, m.p12[k] * yz),
                            );
                            let diag = madd::<T, F>(
                                m.p00[k],
                                xx,
                                madd::<T, F>(m.p11[k], yy, m.p22[k] * zz),
                            );
                            let qf = madd::<T, F>(T::c(2.0), off, diag);
                            dens = madd::<T, F>(m.an[k], (neg_pi_sq * qf).kexp::<F>(), dens);
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

/// Gradients of `sum(grad_out * splat)` for atom `i`: `([d xyz; 3], [d U; 6], d occ)`.
#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl+avx2+fma+bmi1+bmi2+lzcnt+movbe+f16c",
    "x86_64+avx2+fma+bmi1+bmi2+lzcnt+movbe+f16c",
))]
pub fn bwd_atom<T: Real>(
    i: usize,
    grad: &[T],
    c: &Cell<T>,
    at: &Atoms<'_, T>,
) -> ([T; 3], [T; 6], T) {
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
) -> ([T; 3], [T; 6], T) {
    let neg_pi_sq = -T::c(PI_SQ);
    let p_sq = T::c(PI_SQ);
    let half = T::c(0.5);
    let g = anchor(c, at, i);
    let oc = at.occ[i];
    // Occupancy-free amplitudes, as in the isotropic backward.
    let m = minv(at, i, T::ONE);
    let mut gxyz = [T::ZERO; 3];
    let mut gu = [T::ZERO; 6];
    let mut gocc = T::ZERO;
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
            let mut oz0 = zlo;
            let mut iz = wrap(g.ciz + zlo, c.nz) as usize;
            while oz0 <= zhi {
                let seg = ((zhi - oz0 + 1) as usize).min(row.len() - iz);
                for (j, &gout) in row[iz..iz + seg].iter().enumerate() {
                    let foz = T::from_i32(oz0 + j as i32);
                    let wx = madd::<T, F>(foz, c.uc[0], q[0]);
                    let wy = madd::<T, F>(foz, c.uc[1], q[1]);
                    let wz = madd::<T, F>(foz, c.uc[2], q[2]);
                    if madd::<T, F>(wx, wx, madd::<T, F>(wy, wy, wz * wz)) > g.rc2 {
                        continue;
                    }
                    let mut dens = T::ZERO;
                    let mut sv = [T::ZERO; 3];
                    let mut s = [T::ZERO; 6];
                    for k in 0..5 {
                        let vx =
                            madd::<T, F>(m.p00[k], wx, madd::<T, F>(m.p01[k], wy, m.p02[k] * wz));
                        let vy =
                            madd::<T, F>(m.p01[k], wx, madd::<T, F>(m.p11[k], wy, m.p12[k] * wz));
                        let vz =
                            madd::<T, F>(m.p02[k], wx, madd::<T, F>(m.p12[k], wy, m.p22[k] * wz));
                        let qf = madd::<T, F>(wx, vx, madd::<T, F>(wy, vy, wz * vz));
                        let dg = m.an[k] * (neg_pi_sq * qf).kexp::<F>();
                        dens += dg;
                        sv[0] = madd::<T, F>(dg, vx, sv[0]);
                        sv[1] = madd::<T, F>(dg, vy, sv[1]);
                        sv[2] = madd::<T, F>(dg, vz, sv[2]);
                        // pi^2 v_i v_j - p_ij / 2, as one fused step on the hot path.
                        let (px, py, pz) = (p_sq * vx, p_sq * vy, p_sq * vz);
                        s[0] = madd::<T, F>(dg, madd::<T, F>(px, vx, -half * m.p00[k]), s[0]);
                        s[1] = madd::<T, F>(dg, madd::<T, F>(py, vy, -half * m.p11[k]), s[1]);
                        s[2] = madd::<T, F>(dg, madd::<T, F>(pz, vz, -half * m.p22[k]), s[2]);
                        s[3] = madd::<T, F>(dg, madd::<T, F>(px, vy, -half * m.p01[k]), s[3]);
                        s[4] = madd::<T, F>(dg, madd::<T, F>(px, vz, -half * m.p02[k]), s[4]);
                        s[5] = madd::<T, F>(dg, madd::<T, F>(py, vz, -half * m.p12[k]), s[5]);
                    }
                    let s2pi = gout * T::c(2.0) * p_sq;
                    let s4pi = gout * T::c(4.0) * p_sq;
                    for d in 0..3 {
                        gxyz[d] = madd::<T, F>(s2pi, sv[d], gxyz[d]);
                        gu[d] = madd::<T, F>(s2pi, s[d], gu[d]); // diagonal U
                        gu[d + 3] = madd::<T, F>(s4pi, s[d + 3], gu[d + 3]); // off-diagonal U
                    }
                    gocc = madd::<T, F>(gout, dens, gocc);
                }
                oz0 += seg as i32;
                iz = 0;
            }
        }
    }
    (gxyz.map(|v| oc * v), gu.map(|v| oc * v), gocc)
}
