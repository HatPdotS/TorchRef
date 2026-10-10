//! The one unsafe boundary: raw `(address, length)` pairs from Python become slices.
//!
//! Python passes `tensor.data_ptr()` and `tensor.numel()` for CPU, contiguous tensors of the
//! kernel's dtype (`torchref.utils.native.buf` checks all three). Everything past this
//! module works on safe slices, and every kernel checks slice lengths against each other
//! before touching memory, so a wrong shape is a `ValueError`, not a segfault.

use pyo3::PyResult;
use pyo3::exceptions::PyValueError;
use std::ptr::NonNull;

/// `(data_ptr, numel)` of a CPU tensor, as passed from Python.
pub type Buf = (usize, usize);

fn checked<T>(buf: Buf, what: &str) -> PyResult<*mut T> {
    let (addr, len) = buf;
    if len == 0 {
        // An empty tensor may report address 0, which is not a valid slice pointer.
        return Ok(NonNull::<T>::dangling().as_ptr());
    }
    if addr == 0 {
        return Err(PyValueError::new_err(format!("{what}: null data pointer")));
    }
    if addr % std::mem::align_of::<T>() != 0 {
        return Err(PyValueError::new_err(format!(
            "{what}: data pointer is not aligned for its dtype"
        )));
    }
    Ok(addr as *mut T)
}

/// Borrow an input tensor's storage.
///
/// # Safety
/// `buf` must describe `buf.1` initialised elements of `T` that stay alive and are not
/// written by anything else while the returned slice is in use.
pub unsafe fn input<'a, T>(buf: Buf, what: &str) -> PyResult<&'a [T]> {
    let ptr = checked::<T>(buf, what)?;
    Ok(unsafe { std::slice::from_raw_parts(ptr, buf.1) })
}

/// Borrow an output tensor's storage for writing.
///
/// # Safety
/// As [`input`], and additionally no other slice (input or output) may overlap it.
pub unsafe fn output<'a, T>(buf: Buf, what: &str) -> PyResult<&'a mut [T]> {
    let ptr = checked::<T>(buf, what)?;
    Ok(unsafe { std::slice::from_raw_parts_mut(ptr, buf.1) })
}

/// Fail with a `ValueError` unless `len == expected`.
pub fn expect_len(len: usize, expected: usize, what: &str) -> PyResult<()> {
    if len != expected {
        return Err(PyValueError::new_err(format!(
            "{what}: expected {expected} elements, got {len}"
        )));
    }
    Ok(())
}

/// Address ranges of two buffers overlap (empty buffers never do).
fn overlaps(a: Buf, b: Buf, elem: usize) -> bool {
    a.1 != 0 && b.1 != 0 && a.0 < b.0 + b.1 * elem && b.0 < a.0 + a.1 * elem
}

/// Refuse outputs that alias each other or any input; the kernels assume exclusive access.
pub fn no_aliasing(outputs: &[(Buf, &str)], inputs: &[Buf], elem: usize) -> PyResult<()> {
    for (i, (o, name)) in outputs.iter().enumerate() {
        for (p, _) in &outputs[i + 1..] {
            if overlaps(*o, *p, elem) {
                return Err(PyValueError::new_err(format!(
                    "{name} aliases another output"
                )));
            }
        }
        for p in inputs {
            if overlaps(*o, *p, elem) {
                return Err(PyValueError::new_err(format!("{name} aliases an input")));
            }
        }
    }
    Ok(())
}
