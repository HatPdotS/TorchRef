//! CPU-variant selection shared by every kernel.
//!
//! A wheel cannot be built with `-march=native`, so each hot function is compiled once per
//! x86-64 microarchitecture level and the best one the running CPU supports is picked at
//! first call. aarch64 needs no variants: NEON and FMA are baseline there.
//!
//! Wrap a hot function's body in [`fma_variants!`], and report the level with
//! [`cpu_variant`].

/// The levels every kernel is compiled for, best first. Keep in step with the `targets(...)`
/// list in [`fma_variants!`] (the attribute needs string literals).
pub const X86_TARGETS: [&str; 2] = ["x86-64-v4", "x86-64-v3"];

/// Define `fn $name` compiled for every [`X86_TARGETS`] level, forwarding its arguments to
/// `$body::<.., F>` with `F` = "this variant has FMA" (see [`crate::real::madd`]).
macro_rules! fma_variants {
    (
        $(#[$meta:meta])*
        $vis:vis fn $name:ident $(<$g:ident: $bound:path>)? ($($arg:ident: $ty:ty),* $(,)?)
            $(-> $ret:ty)? => $body:ident
    ) => {
        $(#[$meta])*
        #[multiversion::multiversion(targets(
            "x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl+avx2+fma+bmi1+bmi2+lzcnt+movbe+f16c",
            "x86_64+avx2+fma+bmi1+bmi2+lzcnt+movbe+f16c",
        ))]
        $vis fn $name $(<$g: $bound>)? ($($arg: $ty),*) $(-> $ret)? {
            if multiversion::target::target_cfg_f!(any(
                target_feature = "fma",
                target_arch = "aarch64"
            )) {
                $body::<$($g,)? true>($($arg),*)
            } else {
                $body::<$($g,)? false>($($arg),*)
            }
        }
    };
}
pub(crate) use fma_variants;

/// The variant the kernels will run on this CPU.
pub fn cpu_variant() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        macro_rules! has {
            ($($f:tt),*) => { true $(&& std::arch::is_x86_feature_detected!($f))* };
        }
        if has!(
            "avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl", "avx2", "fma"
        ) {
            return "x86-64-v4";
        }
        if has!("avx2", "fma", "bmi1", "bmi2", "lzcnt", "movbe", "f16c") {
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
