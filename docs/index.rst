.. TorchRef documentation master file

TorchRef Documentation
======================

**A PyTorch-based crystallographic refinement library**

TorchRef is a crystallographic refinement package built entirely on PyTorch.
Autograd and GPU acceleration make it composable with machine-learning
workflows and cheap to extend with new targets, restraints, and optimizers.

It is mainly a library to build and experiment with, not a replacement for
mainline refinement programs on standard problems.

Key Features
------------

- **Native PyTorch Integration**: built on ``nn.Module``
- **Automatic Differentiation**: define a forward pass, get gradients
- **Modular Architecture**: composable targets, restraints, optimizers
- **GPU Acceleration**: CUDA and Apple Silicon (MPS)
- **State Management**: full ``state_dict`` support for checkpointing

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
   user_guide/cli
   user_guide/testing

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api

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
