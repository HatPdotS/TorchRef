//! Embed the build profile, optimisation level and a hash of the kernel sources, so the
//! Python side can refuse a debug build and detect a stale one.

use std::fs;
use std::path::{Path, PathBuf};

/// FNV-1a 64; `torchref.utils.native.source_hash` computes the same digest in Python.
fn fnv1a(hash: &mut u64, bytes: &[u8]) {
    for b in bytes {
        *hash ^= u64::from(*b);
        *hash = hash.wrapping_mul(0x100_0000_01b3);
    }
}

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("readable source dir") {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            collect(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

fn main() {
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let mut files = Vec::new();
    collect(&root.join("src"), &mut files);
    // Relative, '/'-separated and sorted, so the digest is the same on every OS.
    let mut rel: Vec<(String, PathBuf)> = files
        .into_iter()
        .map(|p| {
            let r = p.strip_prefix(&root).unwrap().to_string_lossy().replace('\\', "/");
            (r, p)
        })
        .collect();
    rel.sort();
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for (name, path) in &rel {
        fnv1a(&mut hash, name.as_bytes());
        fnv1a(&mut hash, &[0]);
        fnv1a(&mut hash, &fs::read(path).expect("readable source"));
        fnv1a(&mut hash, &[0]);
        println!("cargo:rerun-if-changed={}", path.display());
    }
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rustc-env=TORCHREF_KERNELS_SOURCE_HASH={hash:016x}");
    println!(
        "cargo:rustc-env=TORCHREF_KERNELS_PROFILE={}",
        std::env::var("PROFILE").unwrap()
    );
    println!(
        "cargo:rustc-env=TORCHREF_KERNELS_OPT_LEVEL={}",
        std::env::var("OPT_LEVEL").unwrap()
    );
}
