.. TorchRef documentation master file

TorchRef Documentation
======================

**A PyTorch-based crystallographic refinement library**

TorchRef is a modern crystallographic refinement package built entirely on PyTorch. 
By leveraging PyTorch's automatic differentiation and GPU acceleration, TorchRef 
enables seamless integration with machine learning workflows and provides a flexible, 
extensible framework for crystallographic structure refinement.

Key Features
------------

- **Native PyTorch Integration**: Built on PyTorch's ``nn.Module`` architecture
- **Automatic Differentiation**: No manual gradient implementations required
- **Modular Architecture**: Easily composable and extensible components
- **GPU Acceleration**: CUDA support for fast structure factor calculations
- **State Management**: Full ``state_dict`` support for checkpointing

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/refinement
   user_guide/targets
   user_guide/restraints
   user_guide/scaling
   user_guide/naming_conventions

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/torchref.io
   api/torchref.model
   api/torchref.refinement
   api/torchref.restraints
   api/torchref.scaling
   api/torchref.symmetry
   api/torchref.math_functions
   api/torchref.utils
   api/torchref.alignment

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
