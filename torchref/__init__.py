"""
TorchRef Refinement Package.

A package for crystallographic refinement using PyTorch.
"""


from torchref._bootstrap import detect_available_cpus, configure_threading

import os 
import warnings

if 'TORCHREF_NUM_THREADS' in os.environ:
    N_CPUS = int(os.environ['TORCHREF_NUM_THREADS'])
    warnings.warn(f"TorchRef using user-specified {N_CPUS} threads from TORCHREF_NUM_THREADS.", stacklevel=2)
else:
    N_CPUS = detect_available_cpus()
    os.environ['TORCHREF_NUM_THREADS'] = str(N_CPUS)
    warnings.warn(f"TorchRef auto-configured {N_CPUS} threads. Set TORCHREF_NUM_THREADS to override.", stacklevel=2)


configure_threading(N_CPUS)


import torch

torch.set_num_threads(N_CPUS)

from pathlib import Path

__version__ = '0.1.0'

# Project root path for referencing package files
ROOT_TORCHREF = Path(__file__).parent.parent.resolve()

# Package path for referencing internal files
PATH_TORCHREF = Path(__file__).parent.resolve()





