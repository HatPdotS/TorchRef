//! CPU-variant selection shared by every kernel.
//!
//! A wheel cannot be built with `-march=native`, so each hot function is compiled once per
//! x86-64 microarchitecture level and the best one the running CPU supports is picked at
//! first call. aarch64 needs no variants: NEON and FMA are baseline there.
//!
//! Annotate a hot function with `#[multiversion(targets(...))]` using [`X86_TARGETS`]'s
//! levels (copied literally: the attribute needs string literals), and report the level
//! with [`cpu_variant`].

/// The levels every multiversioned kernel is compiled for, best first. Keep in step with
/// the `targets(...)` lists on the kernels.
pub const X86_TARGETS: [&str; 2] = ["x86-64-v4", "x86-64-v3"];

/// The variant the kernels will run on this CPU.
pub fn cpu_variant() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512cd")
            && std::arch::is_x86_feature_detected!("avx512dq")
            && std::arch::is_x86_feature_detected!("avx512vl")
            && std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
        {
            return "x86-64-v4";
        }
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
            && std::arch::is_x86_feature_detected!("bmi1")
            && std::arch::is_x86_feature_detected!("bmi2")
            && std::arch::is_x86_feature_detected!("lzcnt")
            && std::arch::is_x86_feature_detected!("movbe")
            && std::arch::is_x86_feature_detected!("f16c")
        {
            return "x86-64-v3";
        }
        "x86-64"
    }
    #[cfg(target_arch = "aarch64")]
    {
        "aarch64"
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        "generic"
    }
}
