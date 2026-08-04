Installation
============

Requirements
------------

Python ≥ 3.10, PyTorch ≥ 2.4, NumPy ≥ 2.0, Pandas ≥ 2.0, SciPy ≥ 1.10,
Gemmi ≥ 0.5, reciprocalspaceship ≥ 0.9.18, Numba ≥ 0.59, Matplotlib ≥ 3.7.

``pyproject.toml`` carries the authoritative pinned ranges. Upper bounds are set
one minor version above the tested maximum, so an untested dependency version
fails at install time rather than surfacing as a runtime error.

Installing via pip
------------------

.. code-block:: bash

   pip install torchref

Installing from Source
----------------------

.. code-block:: bash

   git clone https://github.com/HatPdotS/TorchRef.git
   cd TorchRef
   pip install -e .

For development, add the extras (pytest, black, isort, flake8):

.. code-block:: bash

   pip install -e ".[dev]"

The optional ``[amber]`` extra pulls in OpenMM for the Amber target; see
:doc:`user_guide/testing` for what it gates.

Verifying Installation
----------------------

.. code-block:: python

   import torchref
   print(torchref.__version__)

GPU Support
-----------

Auto-selection is stricter than ``torch.cuda.is_available()``: TorchRef also
requires ≥ 10 GB VRAM and a compute capability your PyTorch build was compiled
for, and falls back to CPU with a warning otherwise. A machine that reports CUDA
as available may therefore still run on CPU by design — set
``TORCHREF_DEVICE=cuda`` to force it.

Apple Silicon runs through the MPS backend. Unsupported ops fall back to CPU
via ``PYTORCH_ENABLE_MPS_FALLBACK=1``, which TorchRef sets on import.

.. code-block:: python

   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"MPS available: {torch.backends.mps.is_available()}")
