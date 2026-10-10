# torchref-kernels

Prebuilt CPU kernels for [TorchRef](https://github.com/HatPdotS/TorchRef), written in Rust.

The wheels link neither libtorch nor a specific CPython version (abi3), so one wheel per
platform works with every torch and Python version TorchRef supports, and nothing is
compiled when TorchRef is imported. On x86-64 each kernel is built for several
microarchitecture levels and the best one for the running CPU is chosen on first use.

This package is an implementation detail of TorchRef and is installed with it; its entry
points take raw tensor pointers and are not a public API.

## Development

```bash
pip install -e ./kernels          # or: maturin develop --release -m kernels/Cargo.toml
cargo test --manifest-path kernels/Cargo.toml
```

Rust changes are picked up only after rebuilding. `torchref_kernels.build_info()` reports
the build profile, the selected CPU variant and a hash of the sources it was built from.
