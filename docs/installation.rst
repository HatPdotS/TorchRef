Installation
============

Requirements
------------

- Python ≥ 3.8
- PyTorch ≥ 1.9
- NumPy ≥ 1.20
- Gemmi ≥ 0.5
- reciprocalspaceship ≥ 0.9
- SciPy ≥ 1.7

Installing from Source
----------------------

Clone the repository and install with pip:

.. code-block:: bash

   git clone https://github.com/your-org/torchref.git
   cd torchref
   pip install -e .

For development, install with additional dependencies:

.. code-block:: bash

   pip install -e ".[dev]"

This includes testing tools (pytest), code formatting (black, isort), and linting (flake8).

Verifying Installation
----------------------

After installation, verify that TorchRef is correctly installed:

.. code-block:: python

   import torchref
   print(torchref.__version__)

GPU Support
-----------

TorchRef automatically uses GPU acceleration if PyTorch is installed with CUDA support. 
To verify GPU availability:

.. code-block:: python

   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
