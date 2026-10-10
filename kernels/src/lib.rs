//! Prebuilt CPU kernels for TorchRef.
//!
//! The library links neither libtorch nor a specific CPython: kernels take raw
//! `(data_ptr, numel)` pairs (see [`ffi`]), so one abi3 wheel per platform serves every
//! supported torch and Python version. Autograd, validation and dispatch stay in TorchRef.
//!
//! Adding a kernel: a module with its `#[pyfunction]`s and a `register` function, plus one
//! line in [`_native`]. Bump [`ABI_VERSION`] whenever an existing entry point changes.

mod ffi;
mod frf;
mod pool;
mod real;
mod simd;
mod sphere_splat;

use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Version of the Python-facing calling convention. TorchRef refuses a build whose value
/// differs from the one it was written against.
pub const ABI_VERSION: u32 = 1;

/// How this binary was built, and which CPU variant its kernels will run.
#[pyfunction]
fn build_info(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("version", env!("CARGO_PKG_VERSION"))?;
    d.set_item("abi_version", ABI_VERSION)?;
    d.set_item("profile", env!("TORCHREF_KERNELS_PROFILE"))?;
    d.set_item("opt_level", env!("TORCHREF_KERNELS_OPT_LEVEL"))?;
    d.set_item("source_hash", env!("TORCHREF_KERNELS_SOURCE_HASH"))?;
    d.set_item("cpu_variant", simd::cpu_variant())?;
    d.set_item("x86_targets", simd::X86_TARGETS.to_vec())?;
    d.set_item("target_arch", std::env::consts::ARCH)?;
    d.set_item("target_os", std::env::consts::OS)?;
    Ok(d)
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("ABI_VERSION", ABI_VERSION)?;
    m.add_function(wrap_pyfunction!(build_info, m)?)?;
    sphere_splat::register(m)?;
    frf::legendre_shell::register(m)?;
    Ok(())
}
