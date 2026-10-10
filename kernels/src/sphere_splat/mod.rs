//! Fused per-atom spherical-cutoff density splat, forward and backward.
//!
//! Implements the truncation contract shared with TorchRef's CUDA and Metal kernels: voxel
//! `v` receives atom `i`'s full five-Gaussian density iff `||w||^2 <= r_i^2`, where `w` is
//! the minimum-image Cartesian atom-to-voxel vector and `r_i` the raw per-atom radius,
//! enumerated over the triclinic-correct per-axis half-width
//! `ceil(r_i * n_axis * ||inv_frac row_axis||)`.
//!
//! Forward partitions the **output** by x-plane and backward partitions over **atoms**, so
//! neither needs atomics; both are deterministic and independent of the thread count.
//! Gradients cover `xyz`, `adp`/`U` and `occ`; `A`, `B` and the cell matrices get none.

mod aniso;
mod common;
mod iso;

use crate::ffi::{Buf, expect_len, input, no_aliasing, output};
use crate::pool;
use crate::real::Real;
use common::{Atoms, Cell};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

fn grid(nx: i64, ny: i64, nz: i64) -> PyResult<(i32, i32, i32, usize)> {
    let ok = |n: i64| n > 0 && n <= i64::from(i32::MAX);
    if !(ok(nx) && ok(ny) && ok(nz)) {
        return Err(PyValueError::new_err(format!(
            "grid dimensions must be positive, got ({nx}, {ny}, {nz})"
        )));
    }
    let len = (nx as usize)
        .checked_mul(ny as usize)
        .and_then(|v| v.checked_mul(nz as usize))
        .ok_or_else(|| PyValueError::new_err("grid too large"))?;
    Ok((nx as i32, ny as i32, nz as i32, len))
}

/// Borrow and shape-check the atom inputs. `adp_width` is 1 (B) or 6 (U).
///
/// # Safety
/// Every `Buf` must satisfy [`crate::ffi::input`]'s contract.
#[allow(clippy::too_many_arguments)]
unsafe fn atoms<'a, T>(
    xyz: Buf,
    adp: Buf,
    occ: Buf,
    a: Buf,
    b: Buf,
    r2cut: Buf,
    inv_frac: Buf,
    frac: Buf,
    adp_width: usize,
) -> PyResult<Atoms<'a, T>> {
    let n = occ.1;
    expect_len(xyz.1, 3 * n, "xyz")?;
    expect_len(
        adp.1,
        adp_width * n,
        if adp_width == 1 { "adp" } else { "u" },
    )?;
    expect_len(a.1, 5 * n, "A")?;
    expect_len(b.1, 5 * n, "B")?;
    expect_len(r2cut.1, n, "r2cut")?;
    expect_len(inv_frac.1, 9, "inv_frac")?;
    expect_len(frac.1, 9, "frac")?;
    unsafe {
        Ok(Atoms {
            xyz: input(xyz, "xyz")?,
            adp: input(adp, "adp")?,
            occ: input(occ, "occ")?,
            a: input(a, "A")?,
            b: input(b, "B")?,
            r2cut: input(r2cut, "r2cut")?,
            inv_frac: input(inv_frac, "inv_frac")?,
            frac: input(frac, "frac")?,
            n_atoms: n,
        })
    }
}

/// Run `planes` over x-plane blocks of `out`, one block per worker.
fn forward<T: Real>(
    out: &mut [T],
    c: &Cell<T>,
    at: &Atoms<'_, T>,
    n_threads: usize,
    planes: fn(&mut [T], i32, i32, &Cell<T>, &Atoms<'_, T>),
) {
    if at.n_atoms == 0 {
        return;
    }
    let pool = pool::get(n_threads);
    let plane = c.ny as usize * c.nz as usize;
    let per = (c.nx as usize).div_ceil(pool.current_num_threads()).max(1);
    pool.install(|| {
        out.par_chunks_mut(per * plane)
            .enumerate()
            .for_each(|(ci, chunk)| {
                let xlo = ci * per;
                let xhi = xlo + chunk.len() / plane;
                planes(chunk, xlo as i32, xhi as i32, c, at);
            });
    });
}

macro_rules! splat_bindings {
    ($t:ty, $iso_fwd:ident, $iso_bwd:ident, $aniso_fwd:ident, $aniso_bwd:ident) => {
        /// Isotropic forward: accumulate the splat into `out` in place.
        #[pyfunction]
        #[allow(clippy::too_many_arguments)]
        fn $iso_fwd(
            py: Python<'_>,
            out: Buf,
            xyz: Buf,
            adp: Buf,
            occ: Buf,
            a: Buf,
            b: Buf,
            r2cut: Buf,
            inv_frac: Buf,
            frac: Buf,
            nx: i64,
            ny: i64,
            nz: i64,
            n_threads: usize,
        ) -> PyResult<()> {
            let (nx, ny, nz, len) = grid(nx, ny, nz)?;
            expect_len(out.1, len, "out")?;
            no_aliasing(
                &[(out, "out")],
                &[xyz, adp, occ, a, b, r2cut, inv_frac, frac],
                std::mem::size_of::<$t>(),
            )?;
            let at = unsafe { atoms::<$t>(xyz, adp, occ, a, b, r2cut, inv_frac, frac, 1)? };
            let out = unsafe { output::<$t>(out, "out")? };
            let c = Cell::new(at.frac, at.inv_frac, nx, ny, nz);
            py.detach(|| forward(out, &c, &at, n_threads, iso::fwd_planes::<$t>));
            Ok(())
        }

        /// Isotropic backward: write per-atom gradients of `sum(grad * splat)`.
        #[pyfunction]
        #[allow(clippy::too_many_arguments)]
        fn $iso_bwd(
            py: Python<'_>,
            g_xyz: Buf,
            g_adp: Buf,
            g_occ: Buf,
            grad: Buf,
            xyz: Buf,
            adp: Buf,
            occ: Buf,
            a: Buf,
            b: Buf,
            r2cut: Buf,
            inv_frac: Buf,
            frac: Buf,
            nx: i64,
            ny: i64,
            nz: i64,
            n_threads: usize,
        ) -> PyResult<()> {
            let (nx, ny, nz, len) = grid(nx, ny, nz)?;
            expect_len(grad.1, len, "grad")?;
            let n = occ.1;
            expect_len(g_xyz.1, 3 * n, "g_xyz")?;
            expect_len(g_adp.1, n, "g_adp")?;
            expect_len(g_occ.1, n, "g_occ")?;
            no_aliasing(
                &[(g_xyz, "g_xyz"), (g_adp, "g_adp"), (g_occ, "g_occ")],
                &[grad, xyz, adp, occ, a, b, r2cut, inv_frac, frac],
                std::mem::size_of::<$t>(),
            )?;
            let at = unsafe { atoms::<$t>(xyz, adp, occ, a, b, r2cut, inv_frac, frac, 1)? };
            let grad = unsafe { input::<$t>(grad, "grad")? };
            let gx = unsafe { output::<$t>(g_xyz, "g_xyz")? };
            let gb = unsafe { output::<$t>(g_adp, "g_adp")? };
            let go = unsafe { output::<$t>(g_occ, "g_occ")? };
            let c = Cell::new(at.frac, at.inv_frac, nx, ny, nz);
            py.detach(|| {
                pool::get(n_threads).install(|| {
                    gx.par_chunks_mut(3)
                        .zip(gb.par_iter_mut())
                        .zip(go.par_iter_mut())
                        .enumerate()
                        .for_each(|(i, ((x, b), o))| {
                            let (dx, db, dov) = iso::bwd_atom(i, grad, &c, &at);
                            x.copy_from_slice(&dx);
                            *b = db;
                            *o = dov;
                        });
                })
            });
            Ok(())
        }

        /// Anisotropic forward: accumulate the splat into `out` in place.
        #[pyfunction]
        #[allow(clippy::too_many_arguments)]
        fn $aniso_fwd(
            py: Python<'_>,
            out: Buf,
            xyz: Buf,
            u: Buf,
            occ: Buf,
            a: Buf,
            b: Buf,
            r2cut: Buf,
            inv_frac: Buf,
            frac: Buf,
            nx: i64,
            ny: i64,
            nz: i64,
            n_threads: usize,
        ) -> PyResult<()> {
            let (nx, ny, nz, len) = grid(nx, ny, nz)?;
            expect_len(out.1, len, "out")?;
            no_aliasing(
                &[(out, "out")],
                &[xyz, u, occ, a, b, r2cut, inv_frac, frac],
                std::mem::size_of::<$t>(),
            )?;
            let at = unsafe { atoms::<$t>(xyz, u, occ, a, b, r2cut, inv_frac, frac, 6)? };
            let out = unsafe { output::<$t>(out, "out")? };
            let c = Cell::new(at.frac, at.inv_frac, nx, ny, nz);
            py.detach(|| forward(out, &c, &at, n_threads, aniso::fwd_planes::<$t>));
            Ok(())
        }

        /// Anisotropic backward: write per-atom gradients of `sum(grad * splat)`.
        #[pyfunction]
        #[allow(clippy::too_many_arguments)]
        fn $aniso_bwd(
            py: Python<'_>,
            g_xyz: Buf,
            g_u: Buf,
            g_occ: Buf,
            grad: Buf,
            xyz: Buf,
            u: Buf,
            occ: Buf,
            a: Buf,
            b: Buf,
            r2cut: Buf,
            inv_frac: Buf,
            frac: Buf,
            nx: i64,
            ny: i64,
            nz: i64,
            n_threads: usize,
        ) -> PyResult<()> {
            let (nx, ny, nz, len) = grid(nx, ny, nz)?;
            expect_len(grad.1, len, "grad")?;
            let n = occ.1;
            expect_len(g_xyz.1, 3 * n, "g_xyz")?;
            expect_len(g_u.1, 6 * n, "g_u")?;
            expect_len(g_occ.1, n, "g_occ")?;
            no_aliasing(
                &[(g_xyz, "g_xyz"), (g_u, "g_u"), (g_occ, "g_occ")],
                &[grad, xyz, u, occ, a, b, r2cut, inv_frac, frac],
                std::mem::size_of::<$t>(),
            )?;
            let at = unsafe { atoms::<$t>(xyz, u, occ, a, b, r2cut, inv_frac, frac, 6)? };
            let grad = unsafe { input::<$t>(grad, "grad")? };
            let gx = unsafe { output::<$t>(g_xyz, "g_xyz")? };
            let gu = unsafe { output::<$t>(g_u, "g_u")? };
            let go = unsafe { output::<$t>(g_occ, "g_occ")? };
            let c = Cell::new(at.frac, at.inv_frac, nx, ny, nz);
            py.detach(|| {
                pool::get(n_threads).install(|| {
                    gx.par_chunks_mut(3)
                        .zip(gu.par_chunks_mut(6))
                        .zip(go.par_iter_mut())
                        .enumerate()
                        .for_each(|(i, ((x, uu), o))| {
                            let (dx, du, dov) = aniso::bwd_atom(i, grad, &c, &at);
                            x.copy_from_slice(&dx);
                            uu.copy_from_slice(&du);
                            *o = dov;
                        });
                })
            });
            Ok(())
        }
    };
}

splat_bindings!(
    f32,
    sphere_iso_fwd_f32,
    sphere_iso_bwd_f32,
    sphere_aniso_fwd_f32,
    sphere_aniso_bwd_f32
);
splat_bindings!(
    f64,
    sphere_iso_fwd_f64,
    sphere_iso_bwd_f64,
    sphere_aniso_fwd_f64,
    sphere_aniso_bwd_f64
);

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sphere_iso_fwd_f32, m)?)?;
    m.add_function(wrap_pyfunction!(sphere_iso_bwd_f32, m)?)?;
    m.add_function(wrap_pyfunction!(sphere_aniso_fwd_f32, m)?)?;
    m.add_function(wrap_pyfunction!(sphere_aniso_bwd_f32, m)?)?;
    m.add_function(wrap_pyfunction!(sphere_iso_fwd_f64, m)?)?;
    m.add_function(wrap_pyfunction!(sphere_iso_bwd_f64, m)?)?;
    m.add_function(wrap_pyfunction!(sphere_aniso_fwd_f64, m)?)?;
    m.add_function(wrap_pyfunction!(sphere_aniso_bwd_f64, m)?)?;
    Ok(())
}
