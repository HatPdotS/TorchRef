Installation
============

Requirements
------------

- Python ≥ 3.10
- PyTorch ≥ 2.4
- NumPy ≥ 2.0
- Gemmi ≥ 0.5
- reciprocalspaceship ≥ 0.9.18
- SciPy ≥ 1.10
- Pandas ≥ 2.0
- Numba ≥ 0.59
- Matplotlib ≥ 3.7


Installing via pip
----------------------

.. code-block:: bash

   pip install torchref

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

TorchRef supports GPU acceleration if PyTorch is installed with CUDA support. 
To verify GPU availability:

.. code-block:: python

   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
