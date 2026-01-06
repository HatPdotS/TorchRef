"""
Base classes for crystallographic datasets.

This module defines the abstract base class for all crystallographic datasets,
providing common functionality for device management, masking, and metadata.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, TYPE_CHECKING
import torch
import torch.nn as nn

from torchref.utils.utils import TensorMasks
from torchref.utils.debug_utils import DebugMixin

if TYPE_CHECKING:
    from torch.masked import MaskedTensor


class CrystalDataset(DebugMixin, nn.Module, ABC):
    """
    Abstract base class for crystallographic datasets.

    Provides common functionality shared across all dataset types:
    - Device management (cuda/cpu movement)
    - Masking system via TensorMasks
    - Unit cell and space group handling
    - Resolution calculation

    Subclasses must implement:
    - forward(): Return dataset tensors
    - __len__(): Return number of reflections
    - _calculate_resolution(): Compute resolution from cell and HKL

    Parameters
    ----------
    verbose : int, optional
        Verbosity level (0=silent, 1=normal, 2=debug). Default is 1.
    device : str, optional
        Device for tensors ('cpu', 'cuda', etc.). Default is 'cpu'.

    Attributes
    ----------
    device : torch.device
        Current device for all tensors.
    masks : TensorMasks
        Dictionary-like container for boolean masks.
    cell : torch.Tensor
        Unit cell parameters [a, b, c, alpha, beta, gamma].
    spacegroup : str
        Space group symbol.
    resolution : torch.Tensor
        Resolution per reflection in Angstroms.
    """

    def __init__(self, verbose: int = 1, device: str = 'cpu'):
        """
        Initialize base dataset.

        Parameters
        ----------
        verbose : int, optional
            Verbosity level. Default is 1.
        device : str, optional
            Device for tensors. Default is 'cpu'.
        """
        super().__init__()

        self.verbose: int = verbose
        self.device = torch.device(device)

        # Masking system for filtering reflections
        self.masks = TensorMasks()

        # Unit cell and symmetry
        self.register_buffer('_cell', None)
        self._spacegroup: Optional[str] = None

        # Resolution
        self.register_buffer('_resolution', None)

    @property
    def cell(self) -> Optional[torch.Tensor]:
        """Unit cell parameters [a, b, c, alpha, beta, gamma]."""
        return self._cell

    @cell.setter
    def cell(self, value: torch.Tensor):
        """Set unit cell and recalculate resolution."""
        if value is not None:
            value = value.to(self.device)
        self.register_buffer('_cell', value)
        self._calculate_resolution()

    @property
    def spacegroup(self) -> Optional[str]:
        """Space group symbol."""
        return self._spacegroup

    @spacegroup.setter
    def spacegroup(self, value: str):
        """Set space group."""
        self._spacegroup = value

    @property
    def resolution(self) -> Optional[torch.Tensor]:
        """Resolution per reflection in Angstroms."""
        return self._resolution

    def cuda(self, device=None):
        """
        Move dataset to CUDA device.

        Parameters
        ----------
        device : torch.device or int, optional
            Target CUDA device. If None, uses default.

        Returns
        -------
        CrystalDataset
            Self, for method chaining.
        """
        super().cuda(device)
        self.device = torch.device('cuda') if device is None else torch.device(device)
        if hasattr(self, 'masks'):
            self.masks.cuda(device)
        if self.verbose > 1:
            print(f"{self.__class__.__name__} moved to device: {self.device}")
        return self

    def cpu(self):
        """
        Move dataset to CPU.

        Returns
        -------
        CrystalDataset
            Self, for method chaining.
        """
        super().cpu()
        self.device = torch.device('cpu')
        if hasattr(self, 'masks'):
            self.masks.cpu()
        if self.verbose > 1:
            print(f"{self.__class__.__name__} moved to cpu")
        return self

    def get_valid_mask(self) -> torch.Tensor:
        """
        Get combined validity mask from all registered masks.

        Returns
        -------
        torch.Tensor
            Boolean mask where True indicates valid reflections.
        """
        return self.masks()

    @abstractmethod
    def _calculate_resolution(self) -> None:
        """
        Calculate resolution from cell parameters and HKL indices.

        Must be implemented by subclasses to compute resolution
        based on their specific HKL storage.
        """
        pass

    @abstractmethod
    def forward(self, mask: bool = True) -> Tuple:
        """
        Return dataset tensors.

        Parameters
        ----------
        mask : bool, optional
            Whether to apply masking. Default is True.

        Returns
        -------
        Tuple
            Dataset tensors (implementation-specific).
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return number of reflections in dataset."""
        pass

    def __repr__(self) -> str:
        """String representation of dataset."""
        n_refl = len(self)
        sg = self.spacegroup or "unknown"
        return f"{self.__class__.__name__}(n_reflections={n_refl}, spacegroup='{sg}', device={self.device})"
