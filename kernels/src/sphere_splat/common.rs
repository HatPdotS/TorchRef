//! Cell geometry, per-atom anchors and the in-sphere run solver shared by the iso and
//! aniso kernels.

use crate::real::Real;

pub const PI_1P5: f64 = 5.568_327_996_831_708; // pi^1.5
pub const PI_SQ: f64 = 9.869_604_401_089_358; // pi^2
pub const TWO_PI_SQ: f64 = 19.739_208_802_178_716; // 2 pi^2 = 8 pi^2 / 4

/// The per-call inputs every splat kernel reads. `adp` holds `(n,)` B-factors for the
/// isotropic kernels and `(n, 6)` U components for the anisotropic ones.
pub struct Atoms<'a, T> {
    pub xyz: &'a [T],
    pub adp: &'a [T],
    pub occ: &'a [T],
    pub a: &'a [T],
    pub b: &'a [T],
    pub r2cut: &'a [T],
    pub inv_frac: &'a [T],
    pub frac: &'a [T],
    pub n_atoms: usize,
}

/// Voxel steps and index-space scale factors of one grid.
pub struct Cell<T> {
    pub ua: [T; 3],
    pub ub: [T; 3],
    pub uc: [T; 3],
    pub inva: T,
    pub invb: T,
    pub invc: T,
    /// `|u_c|^2`, the leading coefficient of the per-column quadratic in [`oz_run`].
    pub auc: T,
    pub fnx: T,
    pub fny: T,
    pub fnz: T,
    pub nx: i32,
    pub ny: i32,
    pub nz: i32,
}

impl<T: Real> Cell<T> {
    pub fn new(fm: &[T], im: &[T], nx: i32, ny: i32, nz: i32) -> Self {
        let (fnx, fny, fnz) = (T::from_i32(nx), T::from_i32(ny), T::from_i32(nz));
        // frac columns are the cell vectors a, b, c; divided by n, the voxel steps.
        let ua = [fm[0] / fnx, fm[3] / fnx, fm[6] / fnx];
        let ub = [fm[1] / fny, fm[4] / fny, fm[7] / fny];
        let uc = [fm[2] / fnz, fm[5] / fnz, fm[8] / fnz];
        Cell {
            ua,
            ub,
            uc,
            inva: (im[0] * im[0] + im[1] * im[1] + im[2] * im[2]).sqrt(),
            invb: (im[3] * im[3] + im[4] * im[4] + im[5] * im[5]).sqrt(),
            invc: (im[6] * im[6] + im[7] * im[7] + im[8] * im[8]).sqrt(),
            auc: uc[0] * uc[0] + uc[1] * uc[1] + uc[2] * uc[2],
            fnx,
            fny,
            fnz,
            nx,
            ny,
            nz,
        }
    }
}

/// One atom's anchor node, index-space half-widths and sub-voxel offset.
pub struct Anchor<T> {
    pub cix: i32,
    pub ciy: i32,
    pub ciz: i32,
    pub bhx: i32,
    pub bhy: i32,
    pub bhz: i32,
    pub w0: [T; 3],
    pub rc2: T,
}

#[inline(always)]
pub fn anchor<T: Real>(c: &Cell<T>, at: &Atoms<'_, T>, i: usize) -> Anchor<T> {
    let (im, fm) = (at.inv_frac, at.frac);
    let rc2 = at.r2cut[i];
    let r = rc2.sqrt();
    // Per-axis bounding box of the Cartesian r-sphere in index space; the sphere test
    // culls the corners. Same formula as the Triton and Metal kernels.
    let bhx = (r * c.fnx * c.inva).ceil().to_i32();
    let bhy = (r * c.fny * c.invb).ceil().to_i32();
    let bhz = (r * c.fnz * c.invc).ceil().to_i32();
    let (ax, ay, az) = (at.xyz[3 * i], at.xyz[3 * i + 1], at.xyz[3 * i + 2]);
    let mut fx = ax * im[0] + ay * im[1] + az * im[2];
    let mut fy = ax * im[3] + ay * im[4] + az * im[5];
    let mut fz = ax * im[6] + ay * im[7] + az * im[8];
    fx = fx - fx.floor();
    fy = fy - fy.floor();
    fz = fz - fz.floor();
    let cix = (fx * c.fnx).round_even_i32();
    let ciy = (fy * c.fny).round_even_i32();
    let ciz = (fz * c.fnz).round_even_i32();
    // Sub-voxel residual, taken to Cartesian: the sphere is centred on the atom, not on
    // the anchor node.
    let sx = fx - T::from_i32(cix) / c.fnx;
    let sy = fy - T::from_i32(ciy) / c.fny;
    let sz = fz - T::from_i32(ciz) / c.fnz;
    Anchor {
        cix,
        ciy,
        ciz,
        bhx,
        bhy,
        bhz,
        w0: [
            fm[0] * sx + fm[1] * sy + fm[2] * sz,
            fm[3] * sx + fm[4] * sy + fm[5] * sz,
            fm[6] * sx + fm[7] * sy + fm[8] * sz,
        ],
        rc2,
    }
}

#[inline(always)]
pub fn wrap(i: i32, n: i32) -> i32 {
    i.rem_euclid(n)
}

/// Does any x-plane the atom touches fall in `[xlo, xhi)`? The plane set is the wrapped
/// interval `[cix - bhx, cix + bhx] mod nx`, so at most two pieces.
#[inline(always)]
pub fn touches_x(cix: i32, bhx: i32, nx: i32, xlo: i32, xhi: i32) -> bool {
    if 2 * bhx + 1 >= nx {
        return true;
    }
    let lo = wrap(cix - bhx, nx);
    let hi = lo + 2 * bhx;
    if hi < nx {
        return lo < xhi && hi >= xlo;
    }
    lo < xhi || (hi - nx) >= xlo
}

/// In-sphere `oz` run for one `(ox, oy)` column, or `None` if the column misses.
///
/// `r^2(oz)` is a convex quadratic in `oz`, so the in-sphere set is one contiguous run.
/// The run is widened by one voxel each way and callers keep the exact `r^2 <= rc2` test,
/// so the accepted voxel set is identical to a straight comparison over the full box.
#[inline(always)]
pub fn oz_run<T: Real>(c: &Cell<T>, q: [T; 3], rc2: T, bhz: i32) -> Option<(i32, i32)> {
    let bq = T::c(2.0) * (q[0] * c.uc[0] + q[1] * c.uc[1] + q[2] * c.uc[2]);
    let cq = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] - rc2;
    let disc = bq * bq - T::c(4.0) * c.auc * cq;
    if disc < T::ZERO {
        return None;
    }
    let sq = disc.sqrt();
    let inv2a = T::c(0.5) / c.auc;
    let zlo = ((-bq - sq) * inv2a).ceil().to_i32() - 1;
    let zhi = ((-bq + sq) * inv2a).floor().to_i32() + 1;
    let zlo = zlo.max(-bhz);
    let zhi = zhi.min(bhz);
    (zlo <= zhi).then_some((zlo, zhi))
}

/// Index of `(ix, iy)`'s z-row in a C-contiguous `(nx, ny, nz)` grid whose x range starts
/// at `x0`.
#[inline(always)]
pub fn row_offset(ix: i32, x0: i32, iy: i32, ny: i32, nz: i32) -> usize {
    ((ix - x0) as usize * ny as usize + iy as usize) * nz as usize
}
