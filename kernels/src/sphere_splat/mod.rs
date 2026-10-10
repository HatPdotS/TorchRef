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

//!
//! Every entry point takes its arguments positionally, as `(outputs..., inputs...,
//! nx, ny, nz, n_threads)` with `(data_ptr, numel)` pairs for the buffers:
//!
//! - forward: `out`, then the [`INPUTS`]; accumulates the splat into `out` in place;
//! - backward: `g_xyz`, `g_adp`, `g_occ`, `grad`, then the [`INPUTS`]; writes the per-atom
//!   gradients of `sum(grad * splat)`.
//!
//! `adp` is `(n,)` B-factors for `sphere_iso_*` and `(n, 6)` U components for
//! `sphere_aniso_*`.

mod aniso;
mod common;
mod iso;

use crate::ffi::{Buf, expect_len, input, no_aliasing, output};
use crate::pool;
use crate::real::Real;
use common::{Atoms, Cell};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use rayon::prelude::*;

/// The atom and cell inputs every entry point takes, in order.
const INPUTS: [&str; 8] = ["xyz", "adp", "occ", "A", "B", "r2cut", "inv_frac", "frac"];

/// `N` buffers, the checked grid shape `(nx, ny, nz, voxels)`, and the thread count.
type Args<const N: usize> = ([Buf; N], (i32, i32, i32, usize), usize);

/// Split `(*bufs, nx, ny, nz, n_threads)` as passed from Python.
fn parse<const N: usize>(args: &Bound<'_, PyTuple>) -> PyResult<Args<N>> {
    if args.len() != N + 4 {
        return Err(PyTypeError::new_err(format!(
            "expected {} arguments, got {}",
            N + 4,
            args.len()
        )));
    }
    let mut bufs = [(0, 0); N];
    for (i, b) in bufs.iter_mut().enumerate() {
        *b = args.get_item(i)?.extract()?;
    }
    let dim = |i: usize| -> PyResult<i64> { args.get_item(N + i)?.extract() };
    let (nx, ny, nz) = (dim(0)?, dim(1)?, dim(2)?);
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
    let grid = (nx as i32, ny as i32, nz as i32, len);
    Ok((bufs, grid, args.get_item(N + 3)?.extract()?))
}

/// Borrow and shape-check the [`INPUTS`]; `W` is the per-atom `adp` width.
///
/// # Safety
/// Every `Buf` must satisfy [`crate::ffi::input`]'s contract.
unsafe fn atoms<'a, T, const W: usize>(ins: &[Buf; 8]) -> PyResult<Atoms<'a, T>> {
    let n = ins[2].1;
    for (buf, (name, len)) in
        ins.iter()
            .zip(INPUTS.iter().zip([3 * n, W * n, n, 5 * n, 5 * n, n, 9, 9]))
    {
        expect_len(buf.1, len, name)?;
    }
    let s = |i: usize| unsafe { input::<T>(ins[i], INPUTS[i]) };
    Ok(Atoms {
        xyz: s(0)?,
        adp: s(1)?,
        occ: s(2)?,
        a: s(3)?,
        b: s(4)?,
        r2cut: s(5)?,
        inv_frac: s(6)?,
        frac: s(7)?,
        n_atoms: n,
    })
}

type Planes<T> = fn(&mut [T], i32, i32, &Cell<T>, &Atoms<'_, T>);
type AtomGrad<T, const W: usize> = fn(usize, &[T], &Cell<T>, &Atoms<'_, T>) -> ([T; 3], [T; W], T);

/// Forward: run `planes` over x-plane blocks of `out`, one block per worker.
fn fwd<T: Real, const W: usize>(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    planes: Planes<T>,
) -> PyResult<()> {
    let ([out, ins @ ..], (nx, ny, nz, len), n_threads) = parse::<9>(args)?;
    expect_len(out.1, len, "out")?;
    no_aliasing(&[(out, "out")], &ins, size_of::<T>())?;
    let at = unsafe { atoms::<T, W>(&ins)? };
    let out = unsafe { output::<T>(out, "out")? };
    let c = Cell::new(at.frac, at.inv_frac, nx, ny, nz);
    if at.n_atoms == 0 {
        return Ok(());
    }
    py.detach(|| {
        let pool = pool::get(n_threads);
        let plane = c.ny as usize * c.nz as usize;
        let per = (c.nx as usize).div_ceil(pool.current_num_threads()).max(1);
        pool.install(|| {
            out.par_chunks_mut(per * plane)
                .enumerate()
                .for_each(|(ci, chunk)| {
                    let xlo = ci * per;
                    let xhi = xlo + chunk.len() / plane;
                    planes(chunk, xlo as i32, xhi as i32, &c, &at);
                });
        });
    });
    Ok(())
}

/// Backward: one `atom` call per atom, each writing only that atom's gradient slots.
fn bwd<T: Real, const W: usize>(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    atom: AtomGrad<T, W>,
) -> PyResult<()> {
    let ([g_xyz, g_adp, g_occ, grad, ins @ ..], (nx, ny, nz, len), n_threads) = parse::<12>(args)?;
    expect_len(grad.1, len, "grad")?;
    let n = ins[2].1;
    expect_len(g_xyz.1, 3 * n, "g_xyz")?;
    expect_len(g_adp.1, W * n, "g_adp")?;
    expect_len(g_occ.1, n, "g_occ")?;
    no_aliasing(
        &[(g_xyz, "g_xyz"), (g_adp, "g_adp"), (g_occ, "g_occ")],
        &[&[grad][..], &ins[..]].concat(),
        size_of::<T>(),
    )?;
    let at = unsafe { atoms::<T, W>(&ins)? };
    let grad = unsafe { input::<T>(grad, "grad")? };
    let gx = unsafe { output::<T>(g_xyz, "g_xyz")? };
    let ga = unsafe { output::<T>(g_adp, "g_adp")? };
    let go = unsafe { output::<T>(g_occ, "g_occ")? };
    let c = Cell::new(at.frac, at.inv_frac, nx, ny, nz);
    py.detach(|| {
        pool::get(n_threads).install(|| {
            gx.par_chunks_mut(3)
                .zip(ga.par_chunks_mut(W))
                .zip(go.par_iter_mut())
                .enumerate()
                .for_each(|(i, ((x, a), o))| {
                    let (dx, da, dov) = atom(i, grad, &c, &at);
                    x.copy_from_slice(&dx);
                    a.copy_from_slice(&da);
                    *o = dov;
                });
        })
    });
    Ok(())
}

macro_rules! entry_points {
    ($($name:ident => $run:ident::<$t:ty, $w:literal>($kernel:path);)*) => {
        $(
            #[pyfunction]
            #[pyo3(signature = (*args))]
            fn $name(py: Python<'_>, args: &Bound<'_, PyTuple>) -> PyResult<()> {
                $run::<$t, $w>(py, args, $kernel)
            }
        )*

        pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
            $(m.add_function(wrap_pyfunction!($name, m)?)?;)*
            Ok(())
        }
    };
}

entry_points! {
    sphere_iso_fwd_f32 => fwd::<f32, 1>(iso::fwd_planes::<f32>);
    sphere_iso_bwd_f32 => bwd::<f32, 1>(iso::bwd_atom::<f32>);
    sphere_aniso_fwd_f32 => fwd::<f32, 6>(aniso::fwd_planes::<f32>);
    sphere_aniso_bwd_f32 => bwd::<f32, 6>(aniso::bwd_atom::<f32>);
    sphere_iso_fwd_f64 => fwd::<f64, 1>(iso::fwd_planes::<f64>);
    sphere_iso_bwd_f64 => bwd::<f64, 1>(iso::bwd_atom::<f64>);
    sphere_aniso_fwd_f64 => fwd::<f64, 6>(aniso::fwd_planes::<f64>);
    sphere_aniso_bwd_f64 => bwd::<f64, 6>(aniso::bwd_atom::<f64>);
}
