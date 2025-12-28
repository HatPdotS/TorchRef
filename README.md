# TorchRef

**A PyTorch-based crystallographic refinement library**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

TorchRef is a modern crystallographic refinement package built entirely on PyTorch. By leveraging PyTorch's automatic differentiation and GPU acceleration, TorchRef enables seamless integration with machine learning workflows and provides a flexible, extensible framework for crystallographic structure refinement.

## Key Features

- **🔥 Native PyTorch Integration**: Built on PyTorch's `nn.Module` architecture, TorchRef integrates naturally with the PyTorch ecosystem, including machine learning models, optimizers, and GPU acceleration.

- **📈 Automatic Differentiation**: Dynamic computational graphs eliminate the need for manually implemented gradient calculations. Define new refinement targets directly—PyTorch handles the derivatives automatically.

- **🧩 Modular Architecture**: Following PyTorch's module pattern, components are easily composable and extensible. Add custom targets, restraints, or optimizers without modifying core code.

- **⚡ GPU Acceleration**: Leverage CUDA for structure factor calculations, scaling, and optimization—achieving significant speedups for large structures.

- **🔬 FFT-based Structure Factors**: Efficient structure factor calculation using Fast Fourier Transform (FFT) methods, enabling rapid F_calc computation even for large unit cells.

- **💾 State Management**: Full `state_dict` support enables saving and loading complete refinement states, including model parameters, scaler settings, and restraints.

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/torchref.git
cd torchref

# Install with pip
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Dependencies

- Python ≥ 3.8
- PyTorch ≥ 1.9
- NumPy ≥ 1.20
- Gemmi ≥ 0.5
- reciprocalspaceship ≥ 0.9
- SciPy ≥ 1.7

## Quick Start

### Basic Refinement

```python
from torchref.refinement.base_refinement import Refinement

# Initialize refinement with data and model
refinement = Refinement(
    data_file="reflections.mtz",
    pdb="structure.pdb",
    device=torch.device("cuda")  # Use GPU
)

# Run refinement
refinement.run_refinement(macro_cycles=10)

# Write refined structure
refinement.model.write_pdb("refined.pdb")
```

### Custom Target Functions

One of TorchRef's key strengths is the ease of defining custom refinement targets. Thanks to PyTorch's automatic differentiation, you simply define the forward computation:

```python
from torchref.refinement.targets import Target
import torch

class CustomTarget(Target):
    """Custom refinement target with automatic gradient computation."""
    
    def __init__(self, refinement, weight=1.0):
        super().__init__(refinement)
        self.weight = weight
    
    def forward(self):
        # Define your target function - gradients computed automatically!
        F_calc = self.refinement.model.get_F_calc()
        F_obs = self.data.F
        
        # Custom loss computation
        loss = torch.mean((F_calc - F_obs) ** 2)
        return self.weight * loss
```

### Integration with Machine Learning

TorchRef's PyTorch foundation enables seamless integration with neural networks:

```python
import torch
import torch.nn as nn
from torchref.refinement.base_refinement import Refinement

# Combine crystallographic refinement with a neural network
class HybridModel(nn.Module):
    def __init__(self, refinement):
        super().__init__()
        self.refinement = refinement
        self.neural_prior = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # Coordinate corrections
        )
    
    def forward(self, features):
        # Neural network predictions
        corrections = self.neural_prior(features)
        
        # Apply to crystallographic model
        self.refinement.model.xyz.data += corrections
        
        # Compute crystallographic loss
        return self.refinement.compute_loss()
```

### Saving and Loading State

```python
import torch
from torchref.refinement.base_refinement import Refinement

# Save complete refinement state
refinement = Refinement(data_file="data.mtz", pdb="model.pdb")
torch.save(refinement.state_dict(), "checkpoint.pt")

# Load state into new refinement object
new_refinement = Refinement()
new_refinement.load_state_dict(torch.load("checkpoint.pt"))
```

## Architecture

TorchRef follows PyTorch's module architecture:

```
torchref/
├── io/                    # File I/O (MTZ, PDB, CIF)
│   ├── Data.py           # ReflectionData container
│   ├── cif_readers.py    # CIF/mmCIF parsing
│   └── file_writers.py   # Output writers
├── model/                 # Atomic structure models
│   ├── model.py          # Base Model class (nn.Module)
│   ├── model_ft.py       # Fourier Transform model
│   └── parameter_wrappers.py  # MixedTensor, OccupancyTensor
├── refinement/            # Refinement framework
│   ├── base_refinement.py    # Core Refinement class
│   ├── targets.py        # Loss functions (LS, ML, etc.)
│   ├── optimizers.py     # Custom optimizers
│   └── loss_weighting.py # Adaptive loss balancing
├── restraints/            # Geometry restraints
│   └── restraints.py     # Bonds, angles, torsions, planes
├── scaling/               # Structure factor scaling
│   ├── scaler.py         # Overall/anisotropic scaling
│   └── solvent_new.py    # Bulk solvent model
├── symmetrie/             # Crystallographic symmetry
│   └── symmetrie.py      # Space group operations
├── math_functions/        # Mathematical utilities
│   ├── math_torch.py     # PyTorch implementations
│   └── french_wilson.py  # French-Wilson conversion
└── cli/                   # Command-line interface
    └── refine.py         # CLI entry point
```

## Why PyTorch for Crystallography?

### Automatic Differentiation

Traditional refinement programs require explicit implementation of gradients for each target function and parameter. TorchRef eliminates this burden:

```python
# Traditional approach: implement gradients manually
def compute_loss_and_gradient(params):
    loss = complex_crystallographic_function(params)
    grad = manually_derived_gradient(params)  # Error-prone!
    return loss, grad

# TorchRef approach: let PyTorch handle it
def compute_loss(params):
    loss = complex_crystallographic_function(params)
    loss.backward()  # Automatic gradient computation
    return loss
```

### GPU Acceleration

Structure factor calculations parallelize naturally on GPUs:

```python
# Move everything to GPU
refinement = Refinement(
    data_file="data.mtz",
    pdb="model.pdb", 
    device=torch.device("cuda")
)
# All computations now run on GPU
```

### Ecosystem Integration

Being built on PyTorch opens possibilities for:

- **Neural network potentials** for improved geometry
- **Variational inference** for uncertainty quantification
- **Differentiable physics** combining with simulations
- **End-to-end learning** from diffraction to structure

## Command Line Interface

TorchRef provides CLI tools for common refinement tasks:

```bash
# Basic refinement
torchref-refine --data reflections.mtz --model structure.pdb --output refined.pdb

# With options
torchref-refine --data reflections.mtz --model structure.pdb \
    --cycles 10 --resolution 2.0 --device cuda
```

## Documentation

Comprehensive documentation is available in the `documentation/` directory:

- [Package Overview](documentation/README.md)
- [Model Documentation](documentation/torchref/model/)
- [Refinement Guide](documentation/torchref/refinement/)
- [API Reference](documentation/torchref/)

## Examples

See the `examples/` directory for detailed usage examples:

- [`state_dict_example.py`](examples/state_dict_example.py) - Saving and loading refinement state
- [`selection_freezing_examples.py`](examples/selection_freezing_examples.py) - Parameter selection and freezing
- [`adp_entropy_regularization_example.py`](examples/adp_entropy_regularization_example.py) - ADP regularization

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=torchref

# Run specific test categories
pytest tests/unit/           # Fast unit tests
pytest tests/integration/    # Integration tests
```

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Follow the [NumPy docstring style](https://numpydoc.readthedocs.io/en/latest/format.html)
2. Add tests for new functionality
3. Ensure all tests pass before submitting

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.




