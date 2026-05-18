# TorchRef

**A PyTorch-based crystallographic refinement library**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://readthedocs.org/projects/torchref/badge/?version=latest)](https://torchref.readthedocs.io/)

TorchRef is a crystallographic refinement package built entirely on PyTorch. By leveraging PyTorch's automatic differentiation and GPU acceleration, TorchRef enables seamless integration with machine learning workflows and provides a flexible, extensible framework for crystallographic structure refinement.

# Key Features

- **Native PyTorch Integration**: Built on PyTorch's `nn.Module` architecture, TorchRef integrates naturally with the PyTorch ecosystem, including machine learning models, optimizers, and GPU acceleration.

- **Automatic Differentiation**: Dynamic computational graphs eliminate the need for manually implemented gradient calculations. Define new refinement targets directly—PyTorch handles the derivatives automatically.

- **Modular Architecture**: Following PyTorch's module pattern, components are easily composable and extensible. Add custom targets, restraints, or optimizers without modifying core code.

- **GPU Acceleration**: Leverage CUDA for structure factor calculations, scaling, and optimization—achieving significant speedups for large structures.

- **FFT-based Structure Factors**: Efficient structure factor calculation using Fast Fourier Transform (FFT) methods, enabling rapid F_calc computation even for large unit cells.

# Getting Started

| Notebook | Description |
|----------|-------------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HatPdotS/TorchRef/blob/main/example_notebooks/quickstart.ipynb) | Quickstart — MTZ + PDB to refined structure, refined MTZ and CCP4 map; selection- and parameter-type-based refinement |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HatPdotS/TorchRef/blob/main/example_notebooks/structure_factors.ipynb) | Structure factors — one-liner, `FFT` class, and manual voxel pipeline; standalone scaling; autograd |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HatPdotS/TorchRef/blob/main/example_notebooks/targets_and_weighting.ipynb) | Targets and weighting — standard targets, target-offset weighting, X-ray mode comparison, custom targets, driving an optimizer from a `LossState` |

## Installation

```bash

pip install torchref

```

### Local installation for development

### clone the repository
git clone https://github.com/HatPdotS/TorchRef.git
cd torchref

### Install with pip
pip install -e .

### Or install with development dependencies
pip install -e ".[dev]"

## Dependencies

- Python ≥ 3.10
- PyTorch ≥ 2.40
- NumPy ≥ 2.0
- Gemmi ≥ 0.5
- reciprocalspaceship ≥ 0.9
- SciPy ≥ 1.7

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=torchref

# Run specific test categories
pytest tests/unit/           # Fast unit tests
pytest tests/integration/    # Integration tests
pytest tests/functional/     # Full workflow tests
```

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Follow the [NumPy docstring style](https://numpydoc.readthedocs.io/en/latest/format.html)
2. Add tests for new functionality
3. Ensure all tests pass before submitting

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.






