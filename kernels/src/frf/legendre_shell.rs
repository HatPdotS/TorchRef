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
//!
//! Rows are short (`l < L`), so a loop's scalar tail would be a large share of each one.
//! Instead row `l` is computed over its support `0..=l` rounded up to whole [`LANES`]-wide
//! blocks: the three row buffers and the kernel's own copies of `a` and `b` have a row
//! stride of `L` rounded up to [`LANES`], and the coefficient copies are zero from column
//! `l` on. The buffers start each cluster zeroed, so a padding lane computes `0 * x - 0 * y`
//! and every entry right of the diagonal stays zero, as the next rows need; contracting it
//! adds zero. Valid lanes do exactly the arithmetic they would unpadded, so results are
//! unchanged up to the sign of an exact zero.

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
    /// `a_coef` and `b_coef` with row stride `lp`, zero from column `l` of row `l` on.
    a: &'a [f32],
    b: &'a [f32],
    sect: &'a [f32],
    l: usize,
    /// `l` rounded up to a whole number of [`LANES`]-wide blocks.
    lp: usize,
    nb: usize,
    seed: f32,
}

/// Block width the rows are padded to: one AVX2 register, two NEON registers.
const LANES: usize = 8;

/// `n` rounded up to a multiple of [`LANES`].
fn padded(n: usize) -> usize {
    n.div_ceil(LANES) * LANES
}

/// `coef` (`l` x `l`, row-major) with row stride `lp` and every entry at or right of the
/// diagonal zero, which is what makes the padding lanes compute zero.
fn pad_coefficients(coef: &[f32], l: usize, lp: usize) -> Aligned {
    let mut out = Aligned::zeroed(l * lp);
    let t = out.as_mut_slice();
    for r in 0..l {
        t[r * lp..r * lp + r].copy_from_slice(&coef[r * l..r * l + r]);
    }
    out
}

/// A zeroed buffer whose first element sits on a [`LANES`]-block boundary, so every block
/// of a padded row is one aligned load instead of half of them straddling a cache line.
struct Aligned {
    v: Vec<f32>,
    off: usize,
    n: usize,
}

impl Aligned {
    fn zeroed(n: usize) -> Self {
        let v = vec![0.0_f32; n + LANES - 1];
        // `align_offset` may decline; then the buffer is merely unaligned, never wrong.
        let off = v.as_ptr().align_offset(LANES * size_of::<f32>());
        let off = if off < LANES { off } else { 0 };
        Self { v, off, n }
    }

    fn as_slice(&self) -> &[f32] {
        &self.v[self.off..self.off + self.n]
    }

    fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.v[self.off..self.off + self.n]
    }
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
    let (l_max, lp) = (inp.l, inp.lp);
    let (mut prev2, rest) = buf.split_at_mut(lp);
    let (mut prev1, mut cur) = rest.split_at_mut(lp);
    for c in inp.off[s] as usize..inp.off[s + 1] as usize {
        let (co, si) = (inp.rep_cos[c], inp.rep_sin[c]);
        let dr = &inp.dr[c * l_max..(c + 1) * l_max];
        let di = &inp.di[c * l_max..(c + 1) * l_max];
        prev2.fill(0.0);
        prev1.fill(0.0);
        cur.fill(0.0);
        prev1[0] = inp.seed; // barP_0^0
        for l in 1..l_max {
            // Vertical recurrence over the row's support, padded to whole blocks; this also
            // covers every column the contraction below reads.
            let w = padded(l + 1);
            let a = &inp.a[l * lp..l * lp + w];
            let b = &inp.b[l * lp..l * lp + w];
            recur::<LANES, F>(&mut cur[..w], &prev1[..w], &prev2[..w], a, b, co);
            // The sectoral m == l entry is this row's diagonal and must be in place before
            // the contraction below.
            cur[l] = inp.sect[l] * si * prev1[l - 1];
            if l >= 2 && l % 2 == 0 {
                let pos = (l - 2) / 2;
                let start = (pos * inp.nb + s) * l_max;
                // Padded too, but only to the end of the accumulator row: past it lies the
                // next shell's row, which another worker owns.
                let w = padded(l + 1).min(l_max);
                // SAFETY: rows (pos, s, ..) belong to this worker alone (see `Acc`), and
                // start + w <= n_even * nb * L by the shape checks in the binding.
                let (tr, ti) = unsafe {
                    (
                        std::slice::from_raw_parts_mut(acc.tr.add(start), w),
                        std::slice::from_raw_parts_mut(acc.ti.add(start), w),
                    )
                };
                let (e8, e4) = (w & !7, w & !3);
                contract::<8, F>(
                    &mut tr[..e8],
                    &mut ti[..e8],
                    &cur[..e8],
                    &dr[..e8],
                    &di[..e8],
                );
                contract::<4, F>(
                    &mut tr[e8..e4],
                    &mut ti[e8..e4],
                    &cur[e8..e4],
                    &dr[e8..e4],
                    &di[e8..e4],
                );
                contract::<1, F>(
                    &mut tr[e4..],
                    &mut ti[e4..],
                    &cur[e4..w],
                    &dr[e4..w],
                    &di[e4..w],
                );
            }
            // Rotate the three rows; nothing is copied.
            let t = prev2;
            prev2 = prev1;
            prev1 = cur;
            cur = t;
        }
    }
}

/// `cur = a * co * p1 - b * p2` in blocks of `W` lanes; every slice is a multiple of `W` long.
#[inline(always)]
fn recur<const W: usize, const F: bool>(
    cur: &mut [f32],
    p1: &[f32],
    p2: &[f32],
    a: &[f32],
    b: &[f32],
    co: f32,
) {
    debug_assert!(cur.len().is_multiple_of(W));
    let (p1, p2, a, b) = (
        blocks::<W>(p1),
        blocks::<W>(p2),
        blocks::<W>(a),
        blocks::<W>(b),
    );
    for ((((c, p1), p2), a), b) in cur
        .as_chunks_mut::<W>()
        .0
        .iter_mut()
        .zip(p1)
        .zip(p2)
        .zip(a)
        .zip(b)
    {
        // Computed into a local and stored once: stores interleaved with the next lane's
        // loads keep the compiler from vectorising the block.
        let mut v = [0.0_f32; W];
        for k in 0..W {
            v[k] = madd::<f32, F>(a[k] * co, p1[k], -(b[k] * p2[k]));
        }
        *c = v;
    }
}

/// `tr += p * dr` and `ti += p * di` in blocks of `W` lanes; every slice is a multiple of
/// `W` long. The contraction runs as blocks of 8, then at most one of 4, then at most three
/// of 1, for the rows whose padding the accumulator's row end cuts short.
#[inline(always)]
fn contract<const W: usize, const F: bool>(
    tr: &mut [f32],
    ti: &mut [f32],
    p: &[f32],
    dr: &[f32],
    di: &[f32],
) {
    debug_assert!(tr.len().is_multiple_of(W) && ti.len().is_multiple_of(W));
    let (tr, ti) = (tr.as_chunks_mut::<W>().0, ti.as_chunks_mut::<W>().0);
    let (p, dr, di) = (blocks::<W>(p), blocks::<W>(dr), blocks::<W>(di));
    for ((((t_r, t_i), p), d_r), d_i) in tr.iter_mut().zip(ti).zip(p).zip(dr).zip(di) {
        let (mut vr, mut vi) = (*t_r, *t_i);
        for k in 0..W {
            vr[k] = madd::<f32, F>(p[k], d_r[k], vr[k]);
            vi[k] = madd::<f32, F>(p[k], d_i[k], vi[k]);
        }
        (*t_r, *t_i) = (vr, vi);
    }
}

/// `x` as whole `W`-lane blocks; callers pass multiples of `W`.
#[inline(always)]
fn blocks<const W: usize>(x: &[f32]) -> &[[f32; W]] {
    debug_assert!(x.len().is_multiple_of(W));
    x.as_chunks::<W>().0
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
    let lp = padded(l);
    let (a_pad, b_pad) = unsafe {
        (
            pad_coefficients(input(a_coef, "a_coef")?, l, lp),
            pad_coefficients(input(b_coef, "b_coef")?, l, lp),
        )
    };
    let inp = unsafe {
        Inputs {
            off,
            rep_cos: input(rep_cos, "rep_cos")?,
            rep_sin: input(rep_sin, "rep_sin")?,
            dr: input(dr, "Dr")?,
            di: input(di, "Di")?,
            a: a_pad.as_slice(),
            b: b_pad.as_slice(),
            sect: input(sect, "sect")?,
            l,
            lp,
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
                || Aligned::zeroed(3 * lp),
                |buf, s| one_shell(s, buf.as_mut_slice(), &inp, acc),
            );
        })
    });
    Ok(())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(legendre_shell_accumulate_f32, m)?)?;
    Ok(())
}
