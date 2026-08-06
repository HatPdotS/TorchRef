<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/torchref-banner-dark.svg">
    <img src="assets/torchref-banner-light.svg" alt="TorchRef" width="380">
  </picture>
</p>

<h1 align="center">A Py<ins>Torch</ins>-based crystallographic <ins>Ref</ins>inement library</h1>

<div align="center">

[![Tests](https://github.com/HatPdotS/TorchRef/actions/workflows/tests.yml/badge.svg)](https://github.com/HatPdotS/TorchRef/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://readthedocs.org/projects/torchref/badge/?version=latest)](https://torchref.readthedocs.io/)
[![CUDA](https://img.shields.io/badge/CUDA-supported-76b900.svg)](https://developer.nvidia.com/cuda-zone)
[![Apple Silicon MPS](https://img.shields.io/badge/Apple%20Silicon-MPS-000000.svg?logo=apple)](https://developer.apple.com/metal/pytorch/)

</div>

TorchRef is a crystallographic refinement package built entirely on PyTorch. Autograd and GPU acceleration make it composable with machine-learning workflows and cheap to extend with new targets.

> **Scope.** TorchRef is mainly a library/framework to build and experiment with. It is not intended to replace mainline refinement programs for standard problems.

## Benchmark

![TorchRef AlphaFold-start refinement benchmark](paper/figure2_alphafold_start/figures/figure_af_benchmark.png)

*Refinement of Phaser-placed AlphaFold models against experimental data, on a conserved set of ~720 PDB structures (1.4–3.0 Å). All engines start from the same placed models and are scored by one common validator (PHENIX).*

**(A)** TorchRef reaches essentially the same R-factors as the established programs — median R-free 0.3167 vs 0.3166 (PHENIX) and 0.3161 (REFMAC5). **(B)** Geometry is valid (bond/angle/chiral/main-chain-B RMS Z against REFMAC restraints), though the restraints run slightly looser than PHENIX/REFMAC (bond RMSZ ≈ 1.3). **(C)** Median runtime per structure on 4 CPU cores is 1.65 min, between REFMAC (0.53) and PHENIX (4.63). **(D)** Fraction of the total R-free improvement reached per macrocycle — the programs differ in convergence behaviour.

## Key Features

- **Native PyTorch Integration**: Built on PyTorch's `nn.Module` architecture, so TorchRef composes with PyTorch models, optimizers, and devices.

- **Automatic Differentiation**: No hand-written gradients. Define a new refinement target's forward pass and PyTorch supplies the derivatives.

- **Modular Architecture**: Custom targets, restraints, and optimizers plug in without modifying core code.

- **GPU Acceleration**: CUDA for structure factors, scaling, and optimization. Apple Silicon works through PyTorch's MPS backend — unsupported ops fall back to CPU automatically via `PYTORCH_ENABLE_MPS_FALLBACK=1`, which TorchRef sets on import.

- **FFT-based Structure Factors**: F_calc via FFT, so large unit cells stay tractable.

## Getting Started

| Notebook | Description |
|----------|-------------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HatPdotS/TorchRef/blob/main/example_notebooks/quickstart.ipynb) | Quickstart — MTZ + PDB to refined structure, refined MTZ and CCP4 map; selection- and parameter-type-based refinement |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HatPdotS/TorchRef/blob/main/example_notebooks/structure_factors.ipynb) | Structure factors — one-liner, `FFT` class, and manual voxel pipeline; standalone scaling; autograd |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HatPdotS/TorchRef/blob/main/example_notebooks/targets_and_weighting.ipynb) | Targets and weighting — standard targets, target-offset weighting, X-ray mode comparison, custom targets, driving an optimizer from a `LossState` |

### Installation

```bash
pip install torchref
```

For development:

```bash
git clone https://github.com/HatPdotS/TorchRef.git
cd TorchRef
pip install -e ".[dev]"
```

### Dependencies

Python ≥ 3.10, PyTorch ≥ 2.4, NumPy ≥ 2.0, Pandas ≥ 2.0, SciPy ≥ 1.10, Gemmi ≥ 0.5, reciprocalspaceship ≥ 0.9.18, Numba ≥ 0.59, Matplotlib ≥ 3.7. `pyproject.toml` carries the authoritative pinned ranges — upper bounds are set one minor version above the tested maximum, so a newer dependency will refuse to install rather than fail at runtime.

### Testing

```bash
pytest tests/                      # all tests
pytest tests/ --cov=torchref       # with coverage
pytest tests/unit/                 # fast unit tests only
```

Slow tests need `--run-slow`. Accelerator tests are not opt-in: they run wherever CUDA or MPS is available and are skipped when it is not.

### Contributing

Contributions are welcome. Please use [NumPy docstring style](https://numpydoc.readthedocs.io/en/latest/format.html), add tests for new functionality, and make sure the suite passes before submitting.

### License

MIT — see [LICENSE](LICENSE).
