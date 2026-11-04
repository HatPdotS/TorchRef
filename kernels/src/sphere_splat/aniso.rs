//! Anisotropic splat. `M_g = (B_g I + 8 pi^2 U) / 4` is inverted analytically; the density
//! uses the Mahalanobis form `q = w^T M^-1 w`, while the cutoff stays the Euclidean sphere.

use super::common::{
    Atoms, Cell, PI_1P5, PI_SQ, TWO_PI_SQ, Tangent, anchor, for_each_segment, touches_x, voxel,
};
use crate::real::{Real, madd};
use crate::simd::fma_variants;

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
        let m = minv(at, i, at.occ[i]);
        // A 0/1 keep factor instead of a branch, so the voxel loop vectorises (as in the
        // isotropic kernel).
        for_each_segment::<T, F>(c, &g, xlo, xhi, |start, len, oz, q| {
            for (j, v) in chunk[start..start + len].iter_mut().enumerate() {
                let ([wx, wy, wz], r2) = voxel::<T, F>(c, q, oz + j as i32);
                if !T::VECTOR_EXP && r2 > g.rc2 {
                    continue;
                }
                let keep = if r2 <= g.rc2 { T::ONE } else { T::ZERO };
                let (xx, yy, zz) = (wx * wx, wy * wy, wz * wz);
                let (xy, xz, yz) = (wx * wy, wx * wz, wy * wz);
                let mut dens = T::ZERO;
                for k in 0..5 {
                    let off = madd::<T, F>(m.p01[k], xy, madd::<T, F>(m.p02[k], xz, m.p12[k] * yz));
                    let diag =
                        madd::<T, F>(m.p00[k], xx, madd::<T, F>(m.p11[k], yy, m.p22[k] * zz));
                    let qf = madd::<T, F>(T::c(2.0), off, diag);
                    dens = madd::<T, F>(m.an[k], (neg_pi_sq * qf).kexp::<F>(), dens);
                }
                *v = madd::<T, F>(keep, dens, *v);
            }
        });
    }
}

fma_variants! {
    /// Gradients of `sum(grad_out * splat)` for atom `i`: `([d xyz; 3], [d U; 6], d occ)`.
    pub fn bwd_atom<T: Real>(i: usize, grad: &[T], c: &Cell<T>, at: &Atoms<'_, T>)
        -> ([T; 3], [T; 6], T) => bwd_atom_body
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
    let mut sums = ([T::ZERO; 3], [T::ZERO; 6], T::ZERO);
    for_each_segment::<T, F>(c, &g, 0, c.nx, |start, len, oz, q| {
        // Local sums stay in registers across the segment, as in the isotropic backward.
        let (mut gxyz, mut gu, mut gocc) = sums;
        for (j, &gout) in grad[start..start + len].iter().enumerate() {
            let ([wx, wy, wz], r2) = voxel::<T, F>(c, q, oz + j as i32);
            if r2 > g.rc2 {
                continue;
            }
            let mut dens = T::ZERO;
            let mut sv = [T::ZERO; 3];
            let mut s = [T::ZERO; 6];
            for k in 0..5 {
                let vx = madd::<T, F>(m.p00[k], wx, madd::<T, F>(m.p01[k], wy, m.p02[k] * wz));
                let vy = madd::<T, F>(m.p01[k], wx, madd::<T, F>(m.p11[k], wy, m.p12[k] * wz));
                let vz = madd::<T, F>(m.p02[k], wx, madd::<T, F>(m.p12[k], wy, m.p22[k] * wz));
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
        sums = (gxyz, gu, gocc);
    });
    let (gxyz, gu, gocc) = sums;
    (gxyz.map(|v| oc * v), gu.map(|v| oc * v), gocc)
}

/// The U cotangent as the matching step of `M`, `tM = 2 pi^2 t_U`, upper triangle
/// `[00, 11, 22, 01, 02, 12]`, and per Gaussian `tr(M^-1 tM)` and the upper triangle of
/// `M^-1 tM M^-1`.
struct Step<T> {
    tm: [T; 6],
    tau: [T; 5],
    q: [[T; 6]; 5],
}

#[inline(always)]
fn step<T: Real>(m: &Minv<T>, t: &Tangent<'_, T>, i: usize) -> Step<T> {
    let tp = T::c(TWO_PI_SQ);
    let tu = &t.adp[6 * i..6 * i + 6];
    let tm = [
        tp * tu[0],
        tp * tu[1],
        tp * tu[2],
        tp * tu[3],
        tp * tu[4],
        tp * tu[5],
    ];
    let [m00, m11, m22, m01, m02, m12] = tm;
    let mut st = Step {
        tm,
        tau: [T::ZERO; 5],
        q: [[T::ZERO; 6]; 5],
    };
    for k in 0..5 {
        let (p00, p11, p22) = (m.p00[k], m.p11[k], m.p22[k]);
        let (p01, p02, p12) = (m.p01[k], m.p02[k], m.p12[k]);
        // C = M^-1 tM, then Q = C M^-1.
        let c00 = p00 * m00 + p01 * m01 + p02 * m02;
        let c01 = p00 * m01 + p01 * m11 + p02 * m12;
        let c02 = p00 * m02 + p01 * m12 + p02 * m22;
        let c10 = p01 * m00 + p11 * m01 + p12 * m02;
        let c11 = p01 * m01 + p11 * m11 + p12 * m12;
        let c12 = p01 * m02 + p11 * m12 + p12 * m22;
        let c20 = p02 * m00 + p12 * m01 + p22 * m02;
        let c21 = p02 * m01 + p12 * m11 + p22 * m12;
        let c22 = p02 * m02 + p12 * m12 + p22 * m22;
        st.tau[k] = c00 + c11 + c22;
        st.q[k] = [
            c00 * p00 + c01 * p01 + c02 * p02,
            c10 * p01 + c11 * p11 + c12 * p12,
            c20 * p02 + c21 * p12 + c22 * p22,
            c00 * p01 + c01 * p11 + c02 * p12,
            c00 * p02 + c01 * p12 + c02 * p22,
            c10 * p02 + c11 * p12 + c12 * p22,
        ];
    }
    st
}

/// Gaussian `k` at voxel `w`: the occupancy-free density, `v = M^-1 w`, `tM v` and
/// `lam = d ln(rho_k) . t`.
#[inline(always)]
fn gaussian_step<T: Real, const F: bool>(
    m: &Minv<T>,
    st: &Step<T>,
    k: usize,
    [wx, wy, wz]: [T; 3],
    tx: [T; 3],
) -> (T, [T; 3], [T; 3], T) {
    let [m00, m11, m22, m01, m02, m12] = st.tm;
    let vx = madd::<T, F>(m.p00[k], wx, madd::<T, F>(m.p01[k], wy, m.p02[k] * wz));
    let vy = madd::<T, F>(m.p01[k], wx, madd::<T, F>(m.p11[k], wy, m.p12[k] * wz));
    let vz = madd::<T, F>(m.p02[k], wx, madd::<T, F>(m.p12[k], wy, m.p22[k] * wz));
    let qf = madd::<T, F>(wx, vx, madd::<T, F>(wy, vy, wz * vz));
    let d = m.an[k] * (-T::c(PI_SQ) * qf).kexp::<F>();
    let mv = [
        madd::<T, F>(m00, vx, madd::<T, F>(m01, vy, m02 * vz)),
        madd::<T, F>(m01, vx, madd::<T, F>(m11, vy, m12 * vz)),
        madd::<T, F>(m02, vx, madd::<T, F>(m12, vy, m22 * vz)),
    ];
    let vt = madd::<T, F>(vx, tx[0], madd::<T, F>(vy, tx[1], vz * tx[2]));
    let vmv = madd::<T, F>(vx, mv[0], madd::<T, F>(vy, mv[1], vz * mv[2]));
    let lam = madd::<T, F>(
        T::c(TWO_PI_SQ),
        vt,
        madd::<T, F>(T::c(PI_SQ), vmv, -T::c(0.5) * st.tau[k]),
    );
    (d, [vx, vy, vz], mv, lam)
}

fma_variants! {
    /// Accumulate every atom's tangent map `sum_p t_p d(density)/dp` into the x-planes
    /// `[xlo, xhi)`, stored in `chunk`.
    pub fn tan_planes<T: Real>(
        chunk: &mut [T], xlo: i32, xhi: i32, c: &Cell<T>, at: &Atoms<'_, T>, t: &Tangent<'_, T>
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
) {
    for i in 0..at.n_atoms {
        let g = anchor(c, at, i);
        if !touches_x(g.cix, g.bhx, c.nx, xlo, xhi) {
            continue;
        }
        let m = minv(at, i, T::ONE);
        let st = step(&m, t, i);
        let (oc, to) = (at.occ[i], t.occ[i]);
        let tx = [t.xyz[3 * i], t.xyz[3 * i + 1], t.xyz[3 * i + 2]];
        for_each_segment::<T, F>(c, &g, xlo, xhi, |start, len, oz, q| {
            for (j, v) in chunk[start..start + len].iter_mut().enumerate() {
                let (w, r2) = voxel::<T, F>(c, q, oz + j as i32);
                if !T::VECTOR_EXP && r2 > g.rc2 {
                    continue;
                }
                let keep = if r2 <= g.rc2 { T::ONE } else { T::ZERO };
                let mut tang = T::ZERO;
                for k in 0..5 {
                    let (d, _, _, lam) = gaussian_step::<T, F>(&m, &st, k, w, tx);
                    tang = madd::<T, F>(d, madd::<T, F>(oc, lam, to), tang);
                }
                *v = madd::<T, F>(keep, tang, *v);
            }
        });
    }
}

fma_variants! {
    /// Atom `i`'s Hessian-vector product, the gradient of `sum(grad_out * T)` with `T` the
    /// tangent map of [`tan_planes`]: `([d xyz; 3], [d U; 6], d occ)`.
    pub fn hvp_atom<T: Real>(i: usize, grad: &[T], c: &Cell<T>, at: &Atoms<'_, T>, t: &Tangent<'_, T>)
        -> ([T; 3], [T; 6], T) => hvp_atom_body
}

/// Per Gaussian, with `v = M^-1 w`, `m = M^-1 (t_xyz + tM v)` and `c1 = occ lam + t_occ`:
///
/// - `dT/dxyz = 2 pi^2 sum rho_k (c1 v - occ m)`
/// - `dT/dM = sum rho_k (c1 S - 2 pi^2 occ sym(m v^T) + occ/2 M^-1 tM M^-1)`, with
///   `S = -M^-1 / 2 + pi^2 v v^T`, mapped to U as the backward maps `S`.
#[inline(always)]
fn hvp_atom_body<T: Real, const F: bool>(
    i: usize,
    grad: &[T],
    c: &Cell<T>,
    at: &Atoms<'_, T>,
    t: &Tangent<'_, T>,
) -> ([T; 3], [T; 6], T) {
    let p_sq = T::c(PI_SQ);
    let tp = T::c(TWO_PI_SQ);
    let half = T::c(0.5);
    let g = anchor(c, at, i);
    let m = minv(at, i, T::ONE);
    let st = step(&m, t, i);
    let (oc, to) = (at.occ[i], t.occ[i]);
    let tx = [t.xyz[3 * i], t.xyz[3 * i + 1], t.xyz[3 * i + 2]];
    let mut sums = ([T::ZERO; 3], [T::ZERO; 6], T::ZERO);
    for_each_segment::<T, F>(c, &g, 0, c.nx, |start, len, oz, q| {
        let (mut hxyz, mut hm, mut hocc) = sums;
        for (j, &gout) in grad[start..start + len].iter().enumerate() {
            let (w, r2) = voxel::<T, F>(c, q, oz + j as i32);
            if r2 > g.rc2 {
                continue;
            }
            let mut sx = [T::ZERO; 3];
            let mut sm = [T::ZERO; 6];
            let mut so = T::ZERO;
            for k in 0..5 {
                let (d, v, mv, lam) = gaussian_step::<T, F>(&m, &st, k, w, tx);
                let r = [tx[0] + mv[0], tx[1] + mv[1], tx[2] + mv[2]];
                let mm = [
                    madd::<T, F>(
                        m.p00[k],
                        r[0],
                        madd::<T, F>(m.p01[k], r[1], m.p02[k] * r[2]),
                    ),
                    madd::<T, F>(
                        m.p01[k],
                        r[0],
                        madd::<T, F>(m.p11[k], r[1], m.p12[k] * r[2]),
                    ),
                    madd::<T, F>(
                        m.p02[k],
                        r[0],
                        madd::<T, F>(m.p12[k], r[1], m.p22[k] * r[2]),
                    ),
                ];
                let dc = d * madd::<T, F>(oc, lam, to);
                let d_occ = d * oc;
                so = madd::<T, F>(d, lam, so);
                for e in 0..3 {
                    sx[e] = madd::<T, F>(dc, v[e], madd::<T, F>(-d_occ, mm[e], sx[e]));
                }
                let p = [m.p00[k], m.p11[k], m.p22[k], m.p01[k], m.p02[k], m.p12[k]];
                // (row, column) of each upper-triangle slot.
                let ij = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)];
                for (e, &(a, b)) in ij.iter().enumerate() {
                    let s_ab = madd::<T, F>(p_sq * v[a], v[b], -half * p[e]);
                    let mv_ab = if a == b {
                        tp * mm[a] * v[a]
                    } else {
                        p_sq * madd::<T, F>(mm[a], v[b], mm[b] * v[a])
                    };
                    let extra = madd::<T, F>(half, st.q[k][e], -mv_ab);
                    sm[e] = madd::<T, F>(dc, s_ab, madd::<T, F>(d_occ, extra, sm[e]));
                }
            }
            for e in 0..3 {
                hxyz[e] = madd::<T, F>(gout, sx[e], hxyz[e]);
            }
            for e in 0..6 {
                hm[e] = madd::<T, F>(gout, sm[e], hm[e]);
            }
            hocc = madd::<T, F>(gout, so, hocc);
        }
        sums = (hxyz, hm, hocc);
    });
    let (hxyz, hm, hocc) = sums;
    let four_pi_sq = T::c(2.0) * tp;
    (
        hxyz.map(|v| tp * v),
        [
            tp * hm[0],
            tp * hm[1],
            tp * hm[2],
            four_pi_sq * hm[3],
            four_pi_sq * hm[4],
            four_pi_sq * hm[5],
        ],
        hocc,
    )
}
