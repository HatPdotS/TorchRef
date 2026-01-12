"""
TorchRef Refinement Package.

A package for crystallographic refinement using PyTorch.
"""

from pathlib import Path

__version__ = '0.1.0'

# Project root path for referencing package files
ROOT_TORCHREF = Path(__file__).parent.parent.resolve()

# Package path for referencing internal files
PATH_TORCHREF = Path(__file__).parent.resolve()
