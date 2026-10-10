//! Fused Legendre recurrence and shell accumulation for the FRF rotation function.
//!
//! For every cluster `c` (sorted by shell), the normalised associated Legendre values
//! `barP(l, m)` at `cos(theta_c)` are built by the vertical recurrence in a three-row stack
//! buffer and immediately contracted:
//!
//! `T[pos][shell[c]][m] += barP(l, m) * D[c][m]` for even `l >= 2`, `m <= l`,
//! with `pos = (l - 2) / 2`.
//!
//! Fusing removes the per-row round trip to memory that dominates the torch formulation.
//! Work is partitioned by shell, so a worker owns every write into its shells' rows and no
//! atomics are needed. float32 only.

use crate::ffi::{Buf, expect_len, input, no_aliasing, output};
use crate::pool;
use crate::real::{Real, madd};
use multiversion::multiversion;
use multiversion::target::target_cfg_f;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

struct Inputs<'a> {
    off: &'a [i64],
    rep_cos: &'a [f32],
    rep_sin: &'a [f32],
    dr: &'a [f32],
    di: &'a [f32],
    a: &'a [f32],
    b: &'a [f32],
    sect: &'a [f32],
    l: usize,
    nb: usize,
    seed: f32,
}

/// Raw view of the two accumulators, shared across workers.
///
/// Sound because [`check_partition`] proves every cluster in shell `s`'s range carries
/// label `s`, so the worker for `s` writes only rows `(pos, s, ..)`, disjoint from every
/// other worker's.
#[derive(Clone, Copy)]
struct Acc {
    tr: *mut f32,
    ti: *mut f32,
}
unsafe impl Send for Acc {}
unsafe impl Sync for Acc {}

/// Every shell range `[off[s], off[s + 1])` is in bounds, non-decreasing, and labelled `s`.
fn check_partition(off: &[i64], shell: &[i64], nb: usize) -> PyResult<()> {
    let n = shell.len() as i64;
    if off[0] != 0 || off[nb] != n {
        return Err(PyValueError::new_err(
            "offsets must start at 0 and end at the number of clusters",
        ));
    }
    for s in 0..nb {
        let (lo, hi) = (off[s], off[s + 1]);
        if lo > hi {
            return Err(PyValueError::new_err("offsets must be non-decreasing"));
        }
        if shell[lo as usize..hi as usize]
            .iter()
            .any(|&v| v != s as i64)
        {
            return Err(PyValueError::new_err(
                "shell must be sorted and consistent with offsets",
            ));
        }
    }
    Ok(())
}

#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl+avx2+fma+bmi1+bmi2+lzcnt+movbe+f16c",
    "x86_64+avx2+fma+bmi1+bmi2+lzcnt+movbe+f16c",
))]
fn one_shell(s: usize, buf: &mut [f32], inp: &Inputs<'_>, acc: Acc) {
    if target_cfg_f!(any(target_feature = "fma", target_arch = "aarch64")) {
        one_shell_body::<true>(s, buf, inp, acc)
    } else {
        one_shell_body::<false>(s, buf, inp, acc)
    }
}

#[inline(always)]
fn one_shell_body<const F: bool>(s: usize, buf: &mut [f32], inp: &Inputs<'_>, acc: Acc) {
    let l_max = inp.l;
    let (mut prev2, rest) = buf.split_at_mut(l_max);
    let (mut prev1, mut cur) = rest.split_at_mut(l_max);
    for c in inp.off[s] as usize..inp.off[s + 1] as usize {
        let (co, si) = (inp.rep_cos[c], inp.rep_sin[c]);
        let dr = &inp.dr[c * l_max..(c + 1) * l_max];
        let di = &inp.di[c * l_max..(c + 1) * l_max];
        prev1.fill(0.0);
        prev2.fill(0.0);
        prev1[0] = inp.seed; // barP_0^0
        for l in 1..l_max {
            let a = &inp.a[l * l_max..l * l_max + l];
            let b = &inp.b[l * l_max..l * l_max + l];
            // Vertical recurrence, only where the row can be non-zero.
            // Equal-length slices and iterators, so the loop has no bounds checks and
            // vectorises.
            for (((c_m, &p1), &p2), (&a_m, &b_m)) in cur[..l]
                .iter_mut()
                .zip(&prev1[..l])
                .zip(&prev2[..l])
                .zip(a.iter().zip(b))
            {
                *c_m = madd::<f32, F>(a_m * co, p1, -(b_m * p2));
            }
            // The sectoral m == l entry is this row's diagonal and must be in place before
            // the contraction below.
            cur[l] = inp.sect[l] * si * prev1[l - 1];
            if l >= 2 && l % 2 == 0 {
                let pos = (l - 2) / 2;
                let start = (pos * inp.nb + s) * l_max;
                // SAFETY: rows (pos, s, ..) belong to this worker alone (see `Acc`), and
                // start + l < n_even * nb * L by the shape checks in the binding.
                let (tr, ti) = unsafe {
                    (
                        std::slice::from_raw_parts_mut(acc.tr.add(start), l + 1),
                        std::slice::from_raw_parts_mut(acc.ti.add(start), l + 1),
                    )
                };
                for ((((t_r, t_i), &c_m), &d_r), &d_i) in tr
                    .iter_mut()
                    .zip(ti.iter_mut())
                    .zip(&cur[..=l])
                    .zip(&dr[..=l])
                    .zip(&di[..=l])
                {
                    *t_r = madd::<f32, F>(c_m, d_r, *t_r);
                    *t_i = madd::<f32, F>(c_m, d_i, *t_i);
                }
            }
            // Rotate the three rows; nothing is copied.
            let t = prev2;
            prev2 = prev1;
            prev1 = cur;
            cur = t;
        }
    }
}

/// Accumulate the fused recurrence into `Tr`/`Ti` (shape `(n_even, n_shells, L)`) in place.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn legendre_shell_accumulate_f32(
    py: Python<'_>,
    tr: Buf,
    ti: Buf,
    rep_cos: Buf,
    rep_sin: Buf,
    dr: Buf,
    di: Buf,
    shell: Buf,
    offsets: Buf,
    a_coef: Buf,
    b_coef: Buf,
    sect: Buf,
    n_even: usize,
    n_shells: usize,
    l: usize,
    seed: f64,
    n_threads: usize,
) -> PyResult<()> {
    let n = shell.1;
    let even_rows = if l >= 3 { (l - 1) / 2 } else { 0 };
    if n_even < even_rows {
        return Err(PyValueError::new_err(format!(
            "Tr has {n_even} even-l rows but L={l} needs {even_rows}"
        )));
    }
    let t_len = n_even
        .checked_mul(n_shells)
        .and_then(|v| v.checked_mul(l))
        .ok_or_else(|| PyValueError::new_err("accumulator too large"))?;
    expect_len(tr.1, t_len, "Tr")?;
    expect_len(ti.1, t_len, "Ti")?;
    expect_len(rep_cos.1, n, "rep_cos")?;
    expect_len(rep_sin.1, n, "rep_sin")?;
    expect_len(dr.1, n * l, "Dr")?;
    expect_len(di.1, n * l, "Di")?;
    expect_len(offsets.1, n_shells + 1, "offsets")?;
    expect_len(a_coef.1, l * l, "a_coef")?;
    expect_len(b_coef.1, l * l, "b_coef")?;
    expect_len(sect.1, l, "sect")?;
    no_aliasing(
        &[(tr, "Tr"), (ti, "Ti")],
        &[rep_cos, rep_sin, dr, di, a_coef, b_coef, sect],
        std::mem::size_of::<f32>(),
    )?;
    if l == 0 || n == 0 {
        return Ok(());
    }
    let off = unsafe { input::<i64>(offsets, "offsets")? };
    let shell_v = unsafe { input::<i64>(shell, "shell")? };
    check_partition(off, shell_v, n_shells)?;
    let inp = unsafe {
        Inputs {
            off,
            rep_cos: input(rep_cos, "rep_cos")?,
            rep_sin: input(rep_sin, "rep_sin")?,
            dr: input(dr, "Dr")?,
            di: input(di, "Di")?,
            a: input(a_coef, "a_coef")?,
            b: input(b_coef, "b_coef")?,
            sect: input(sect, "sect")?,
            l,
            nb: n_shells,
            seed: f32::c(seed),
        }
    };
    let acc = unsafe {
        Acc {
            tr: output::<f32>(tr, "Tr")?.as_mut_ptr(),
            ti: output::<f32>(ti, "Ti")?.as_mut_ptr(),
        }
    };
    py.detach(|| {
        pool::get(n_threads).install(|| {
            // Dynamic scheduling: clusters per shell vary by an order of magnitude, so
            // equal shell counts are not equal work.
            (0..n_shells).into_par_iter().for_each_init(
                || vec![0.0_f32; 3 * l],
                |buf, s| one_shell(s, buf, &inp, acc),
            );
        })
    });
    Ok(())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(legendre_shell_accumulate_f32, m)?)?;
    Ok(())
}
