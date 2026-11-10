//! Fused per-atom spherical-cutoff density splat, forward and backward.
//!
//! Implements the truncation contract shared with TorchRef's CUDA and Metal kernels: voxel
//! `v` receives atom `i`'s full five-Gaussian density iff `||w||^2 <= r_i^2`, where `w` is
//! the minimum-image Cartesian atom-to-voxel vector and `r_i` the raw per-atom radius,
//! enumerated over the triclinic-correct per-axis half-width
//! `ceil(r_i * n_axis * ||inv_frac row_axis||)`.
//!
//! Forward partitions the **output** into blocks of x-planes, several per worker, each
//! visiting only the atoms that reach it; backward partitions over **atoms**. Neither needs
//! atomics, and both are deterministic and independent of the thread count.
//! Gradients cover `xyz`, `adp`/`U` and `occ`; `A`, `B` and the cell matrices get none.
//!
//! The second order is two more passes, split the same way. Given cotangents `t` on the
//! backward's outputs, the tangent pass accumulates the map `T = sum_p t_p d(splat)/dp` by
//! x-plane, and the Hessian-vector pass writes each atom's gradient of `sum(grad * T)`.

//!
//! Every entry point takes its arguments positionally, as `(outputs..., inputs...,
//! offsets, nx, ny, nz, n_threads)` with `(data_ptr, numel)` pairs for the buffers:
//!
//! - forward: `out`, then the [`INPUTS`]; accumulates the splat into `out` in place;
//! - backward: `g_xyz`, `g_adp`, `g_occ`, `grad`, then the [`INPUTS`]; writes the per-atom
//!   gradients of `sum(grad * splat)`;
//! - tangent: `out`, `t_xyz`, `t_adp`, `t_occ`, then the [`INPUTS`]; accumulates `T` into
//!   `out` in place;
//! - Hessian-vector: `h_xyz`, `h_adp`, `h_occ`, `grad`, `t_xyz`, `t_adp`, `t_occ`, then the
//!   [`INPUTS`]; writes the per-atom gradients of `sum(grad * T)`.
//!
//! `adp` is `(n,)` B-factors for `sphere_iso_*` and `(n, 6)` U components for
//! `sphere_aniso_*`.
//!
//! `offsets` is an `i64` buffer of `B + 1` member offsets: member `b` owns atoms
//! `[offsets[b], offsets[b + 1])` and grid slab `b` of `out` / `grad`, which hold `B` grids
//! back to back. An unbatched call is one member, `[0, n]`. Members run the unchanged
//! per-member kernels, so a batched result is bitwise the stack of single-member results.

mod aniso;
mod common;
mod iso;

use crate::ffi::{Buf, expect_len, input, no_aliasing, output};
use crate::pool;
use crate::real::Real;
use common::{Atoms, Cell, Tangent, bin_atoms};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use rayon::prelude::*;

/// The atom and cell inputs every entry point takes, in order.
const INPUTS: [&str; 8] = ["xyz", "adp", "occ", "A", "B", "r2cut", "inv_frac", "frac"];

/// `N` buffers, the checked grid shape `(nx, ny, nz, voxels)`, and the thread count.
type Args<const N: usize> = ([Buf; N], (i32, i32, i32, usize), usize);

/// Split `(*bufs, nx, ny, nz, n_threads)` as passed from Python; `offsets` is the last
/// buffer.
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

/// Borrow and check the member `offsets` against `n_atoms`.
///
/// # Safety
/// `buf` must satisfy [`crate::ffi::input`]'s contract.
unsafe fn offsets<'a>(buf: Buf, n_atoms: usize) -> PyResult<&'a [i64]> {
    let off = unsafe { input::<i64>(buf, "offsets")? };
    let ok = off.len() >= 2
        && off[0] == 0
        && off[off.len() - 1] == n_atoms as i64
        && off.windows(2).all(|w| w[0] <= w[1]);
    if !ok {
        return Err(PyValueError::new_err(format!(
            "offsets must be B + 1 >= 2 non-decreasing values from 0 to {n_atoms}"
        )));
    }
    Ok(off)
}

/// Borrow and shape-check the cotangents `t_xyz`, `t_adp`, `t_occ` of `n` atoms.
///
/// # Safety
/// Every `Buf` must satisfy [`crate::ffi::input`]'s contract.
unsafe fn tangent<'a, T, const W: usize>(bufs: [Buf; 3], n: usize) -> PyResult<Tangent<'a, T>> {
    let names = ["t_xyz", "t_adp", "t_occ"];
    for (buf, (name, len)) in bufs.iter().zip(names.iter().zip([3 * n, W * n, n])) {
        expect_len(buf.1, len, name)?;
    }
    let s = |i: usize| unsafe { input::<T>(bufs[i], names[i]) };
    Ok(Tangent {
        xyz: s(0)?,
        adp: s(1)?,
        occ: s(2)?,
    })
}

/// Each member's atoms, in member order.
fn members<'a, T>(at: &Atoms<'a, T>, off: &[i64], w: usize) -> Vec<Atoms<'a, T>> {
    off.windows(2)
        .map(|m| at.member(m[0] as usize, m[1] as usize, w))
        .collect()
}

/// Each member's cotangents, in member order.
fn member_tangents<'a, T>(t: &Tangent<'a, T>, off: &[i64], w: usize) -> Vec<Tangent<'a, T>> {
    off.windows(2)
        .map(|m| t.member(m[0] as usize, m[1] as usize, w))
        .collect()
}

type Planes<T> = fn(&mut [T], i32, i32, &Cell<T>, &Atoms<'_, T>, &[u32]);
type TanPlanes<T> = fn(&mut [T], i32, i32, &Cell<T>, &Atoms<'_, T>, &Tangent<'_, T>, &[u32]);
type AtomGrad<T, const W: usize> = fn(usize, &[T], &Cell<T>, &Atoms<'_, T>) -> ([T; 3], [T; W], T);
type AtomHvp<T, const W: usize> =
    fn(usize, &[T], &Cell<T>, &Atoms<'_, T>, &Tangent<'_, T>) -> ([T; 3], [T; W], T);

/// Blocks per worker in the plane-parallel passes. Model atoms are one asymmetric unit
/// and fill only part of the cell along x, so one slab per worker leaves most workers
/// idle; many thin blocks let work stealing even the load.
const BLOCKS_PER_THREAD: usize = 8;

/// x-planes per block for a pool of `n_threads` workers.
fn block_planes(nx: i32, n_threads: usize) -> usize {
    (nx as usize)
        .div_ceil(BLOCKS_PER_THREAD * n_threads.max(1))
        .max(1)
}

/// Run `planes(chunk, xlo, xhi, b, k)` over the `per`-plane blocks `k` of member `b`'s
/// slab of `out`, for every member with atoms.
fn over_planes<T: Real>(
    py: Python<'_>,
    out: &mut [T],
    len: usize,
    c: &Cell<T>,
    sizes: &[usize],
    n_threads: usize,
    per: usize,
    planes: impl Fn(&mut [T], i32, i32, usize, usize) + Sync,
) {
    py.detach(|| {
        let pool = pool::get(n_threads);
        let plane = c.ny as usize * c.nz as usize;
        pool.install(|| {
            // Members and their x-plane blocks are one flat set of tasks, so a batch of
            // small members still fills the pool. Every block writes only its own planes.
            out.par_chunks_mut(len)
                .zip(sizes.par_iter())
                .enumerate()
                .filter(|(_, (_, n))| **n > 0)
                .for_each(|(b, (slab, _))| {
                    slab.par_chunks_mut(per * plane)
                        .enumerate()
                        .for_each(|(k, chunk)| {
                            let xlo = k * per;
                            let xhi = xlo + chunk.len() / plane;
                            planes(chunk, xlo as i32, xhi as i32, b, k);
                        });
                });
        });
    });
}

/// Each member's atoms sorted into `per`-plane blocks ([`bin_atoms`]), in parallel.
fn member_bins<T: Real>(
    py: Python<'_>,
    c: &Cell<T>,
    members: &[Atoms<'_, T>],
    n_threads: usize,
    per: usize,
) -> Vec<Vec<Vec<u32>>> {
    py.detach(|| {
        pool::get(n_threads).install(|| members.par_iter().map(|m| bin_atoms(c, m, per)).collect())
    })
}

/// One `atom(i, b, slab)` call per atom, with `i` its index within member `b` and `slab`
/// that member's grid of `grad`; each call writes only its atom's slots of the outputs.
fn over_atoms<T: Real, const W: usize>(
    py: Python<'_>,
    (gx, ga, go): (&mut [T], &mut [T], &mut [T]),
    grad: &[T],
    len: usize,
    off: &[i64],
    n_threads: usize,
    atom: impl Fn(usize, usize, &[T]) -> ([T; 3], [T; W], T) + Sync,
) {
    py.detach(|| {
        pool::get(n_threads).install(|| {
            gx.par_chunks_mut(3)
                .zip(ga.par_chunks_mut(W))
                .zip(go.par_iter_mut())
                .enumerate()
                .for_each(|(i, ((x, a), o))| {
                    // The last member starting at or before atom i; empty members share
                    // their start with the next one, so they are skipped over.
                    let b = off.partition_point(|&s| s <= i as i64) - 1;
                    let slab = &grad[b * len..(b + 1) * len];
                    let (dx, da, dov) = atom(i - off[b] as usize, b, slab);
                    x.copy_from_slice(&dx);
                    a.copy_from_slice(&da);
                    *o = dov;
                });
        })
    });
}

/// Forward: run `planes` over x-plane blocks of each member's slab of `out`.
fn fwd<T: Real, const W: usize>(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    planes: Planes<T>,
) -> PyResult<()> {
    let ([out, ins @ .., off], (nx, ny, nz, len), n_threads) = parse::<10>(args)?;
    no_aliasing(&[(out, "out")], &ins, size_of::<T>())?;
    let at = unsafe { atoms::<T, W>(&ins)? };
    let off = unsafe { offsets(off, at.n_atoms)? };
    expect_len(out.1, (off.len() - 1) * len, "out")?;
    let out = unsafe { output::<T>(out, "out")? };
    let c = Cell::new(at.frac, at.inv_frac, nx, ny, nz);
    if at.n_atoms == 0 {
        return Ok(());
    }
    let members = members(&at, off, W);
    let sizes: Vec<usize> = members.iter().map(|m| m.n_atoms).collect();
    let per = block_planes(c.nx, pool::get(n_threads).current_num_threads());
    let bins = member_bins(py, &c, &members, n_threads, per);
    over_planes(
        py,
        out,
        len,
        &c,
        &sizes,
        n_threads,
        per,
        |chunk, xlo, xhi, b, k| planes(chunk, xlo, xhi, &c, &members[b], &bins[b][k]),
    );
    Ok(())
}

/// Tangent: run `planes` over x-plane blocks of each member's slab of `out`, as [`fwd`].
fn tan<T: Real, const W: usize>(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    planes: TanPlanes<T>,
) -> PyResult<()> {
    let ([out, tx, ta, to, ins @ .., off], (nx, ny, nz, len), n_threads) = parse::<13>(args)?;
    let n = ins[2].1;
    no_aliasing(
        &[(out, "out")],
        &[&[tx, ta, to][..], &ins[..]].concat(),
        size_of::<T>(),
    )?;
    let at = unsafe { atoms::<T, W>(&ins)? };
    let tg = unsafe { tangent::<T, W>([tx, ta, to], n)? };
    let off = unsafe { offsets(off, n)? };
    expect_len(out.1, (off.len() - 1) * len, "out")?;
    let out = unsafe { output::<T>(out, "out")? };
    let c = Cell::new(at.frac, at.inv_frac, nx, ny, nz);
    if n == 0 {
        return Ok(());
    }
    let members = members(&at, off, W);
    let tangents = member_tangents(&tg, off, W);
    let sizes: Vec<usize> = members.iter().map(|m| m.n_atoms).collect();
    let per = block_planes(c.nx, pool::get(n_threads).current_num_threads());
    let bins = member_bins(py, &c, &members, n_threads, per);
    over_planes(
        py,
        out,
        len,
        &c,
        &sizes,
        n_threads,
        per,
        |chunk, xlo, xhi, b, k| planes(chunk, xlo, xhi, &c, &members[b], &tangents[b], &bins[b][k]),
    );
    Ok(())
}

/// Backward: one `atom` call per atom, each writing only that atom's gradient slots.
fn bwd<T: Real, const W: usize>(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    atom: AtomGrad<T, W>,
) -> PyResult<()> {
    let ([g_xyz, g_adp, g_occ, grad, ins @ .., off], (nx, ny, nz, len), n_threads) =
        parse::<13>(args)?;
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
    let off = unsafe { offsets(off, n)? };
    expect_len(grad.1, (off.len() - 1) * len, "grad")?;
    let grad = unsafe { input::<T>(grad, "grad")? };
    let gx = unsafe { output::<T>(g_xyz, "g_xyz")? };
    let ga = unsafe { output::<T>(g_adp, "g_adp")? };
    let go = unsafe { output::<T>(g_occ, "g_occ")? };
    let c = Cell::new(at.frac, at.inv_frac, nx, ny, nz);
    let members = members(&at, off, W);
    over_atoms(py, (gx, ga, go), grad, len, off, n_threads, |i, b, slab| {
        atom(i, slab, &c, &members[b])
    });
    Ok(())
}

/// Hessian-vector: one `atom` call per atom, as [`bwd`].
fn hvp<T: Real, const W: usize>(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    atom: AtomHvp<T, W>,
) -> PyResult<()> {
    let ([h_xyz, h_adp, h_occ, grad, tx, ta, to, ins @ .., off], (nx, ny, nz, len), n_threads) =
        parse::<16>(args)?;
    let n = ins[2].1;
    expect_len(h_xyz.1, 3 * n, "h_xyz")?;
    expect_len(h_adp.1, W * n, "h_adp")?;
    expect_len(h_occ.1, n, "h_occ")?;
    no_aliasing(
        &[(h_xyz, "h_xyz"), (h_adp, "h_adp"), (h_occ, "h_occ")],
        &[&[grad, tx, ta, to][..], &ins[..]].concat(),
        size_of::<T>(),
    )?;
    let at = unsafe { atoms::<T, W>(&ins)? };
    let tg = unsafe { tangent::<T, W>([tx, ta, to], n)? };
    let off = unsafe { offsets(off, n)? };
    expect_len(grad.1, (off.len() - 1) * len, "grad")?;
    let grad = unsafe { input::<T>(grad, "grad")? };
    let hx = unsafe { output::<T>(h_xyz, "h_xyz")? };
    let ha = unsafe { output::<T>(h_adp, "h_adp")? };
    let ho = unsafe { output::<T>(h_occ, "h_occ")? };
    let c = Cell::new(at.frac, at.inv_frac, nx, ny, nz);
    let members = members(&at, off, W);
    let tangents = member_tangents(&tg, off, W);
    over_atoms(py, (hx, ha, ho), grad, len, off, n_threads, |i, b, slab| {
        atom(i, slab, &c, &members[b], &tangents[b])
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
    sphere_iso_tan_f32 => tan::<f32, 1>(iso::tan_planes::<f32>);
    sphere_iso_hvp_f32 => hvp::<f32, 1>(iso::hvp_atom::<f32>);
    sphere_aniso_fwd_f32 => fwd::<f32, 6>(aniso::fwd_planes::<f32>);
    sphere_aniso_bwd_f32 => bwd::<f32, 6>(aniso::bwd_atom::<f32>);
    sphere_aniso_tan_f32 => tan::<f32, 6>(aniso::tan_planes::<f32>);
    sphere_aniso_hvp_f32 => hvp::<f32, 6>(aniso::hvp_atom::<f32>);
    sphere_iso_fwd_f64 => fwd::<f64, 1>(iso::fwd_planes::<f64>);
    sphere_iso_bwd_f64 => bwd::<f64, 1>(iso::bwd_atom::<f64>);
    sphere_iso_tan_f64 => tan::<f64, 1>(iso::tan_planes::<f64>);
    sphere_iso_hvp_f64 => hvp::<f64, 1>(iso::hvp_atom::<f64>);
    sphere_aniso_fwd_f64 => fwd::<f64, 6>(aniso::fwd_planes::<f64>);
    sphere_aniso_bwd_f64 => bwd::<f64, 6>(aniso::bwd_atom::<f64>);
    sphere_aniso_tan_f64 => tan::<f64, 6>(aniso::tan_planes::<f64>);
    sphere_aniso_hvp_f64 => hvp::<f64, 6>(aniso::hvp_atom::<f64>);
}
