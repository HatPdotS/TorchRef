"""
Force Field Target for Molecular Dynamics Energy Calculations.

This module provides a target that computes molecular energy using
TorchMD-Net neural network potentials for crystallographic refinement.
"""

from typing import TYPE_CHECKING, Dict, Optional
import torch
from torch import nn

from .base import ModelTarget
from torchref.utils.stats import (
    VERBOSITY_DEBUG,
    VERBOSITY_DETAILED,
    VERBOSITY_STANDARD,
    StatEntry,
    stat,
)

if TYPE_CHECKING:
    from torchref.model.model import Model


class ForceFieldTarget(ModelTarget):
    """
    Force field energy target using TorchMD-Net ML potentials.

    Computes molecular energy from atomic coordinates using a pre-trained
    neural network potential. Returns energy as a differentiable tensor
    suitable for gradient-based refinement.

    Parameters
    ----------
    model : Model, optional
        Reference to the Model object. Must have hydrogens (load with strip_H=False).
    model_path : str, optional
        Path to TorchMD-Net checkpoint file (.ckpt).
    cutoff : float, optional
        Interaction cutoff distance in Angstroms. Default is 5.0.
    normalize_by_atoms : bool, optional
        If True, return energy per atom. Default is True.
    verbose : int, optional
        Verbosity level. Default is 0.

    Examples
    --------
    >>> from torchref.model import Model
    >>> from torchref.refinement.targets import ForceFieldTarget
    >>>
    >>> # Load model WITH hydrogens
    >>> model = Model(strip_H=False)
    >>> model.load_pdb('structure_with_H.pdb')
    >>>
    >>> # Create force field target
    >>> ff_target = ForceFieldTarget(
    ...     model=model,
    ...     model_path='path/to/torchmdnet.ckpt',
    ... )
    >>>
    >>> # Get energy
    >>> energy = ff_target()

    Notes
    -----
    Requires torchmd-net package: pip install torchmd-net

    Pre-trained models available at:
    https://github.com/torchmd/torchmd-net/tree/main/examples
    """

    name: str = "forcefield"

    def __init__(
        self,
        model: "Model" = None,
        model_path: str = None,
        cutoff: float = 5.0,
        normalize_by_atoms: bool = True,
        verbose: int = 0,
    ):
        super().__init__(model=model, verbose=verbose)

        # Store configuration
        self._model_path = model_path
        self._normalize_by_atoms = normalize_by_atoms

        # Register cutoff as buffer for state_dict compatibility
        self.register_buffer("_cutoff", torch.tensor(cutoff))

        # Neural network potential (lazy initialization)
        self._nn_potential = None

    def _ensure_nn_potential(self) -> None:
        """Initialize the TorchMD-Net model on first use."""
        if self._nn_potential is not None:
            return

        # Import TorchMD-Net
        try:
            from torchmdnet.models.model import load_model
        except ImportError:
            raise ImportError(
                "ForceFieldTarget requires torchmd-net package.\n"
                "Install with: pip install torchref[forcefield]\n"
                "Pre-trained models: https://github.com/torchmd/torchmd-net/tree/main/examples"
            ) from None

        # Validate model path
        if self._model_path is None:
            raise ValueError(
                "model_path is required. Provide path to a TorchMD-Net checkpoint (.ckpt).\n"
                "Pre-trained models: https://github.com/torchmd/torchmd-net/tree/main/examples"
            )

        # Load model
        self._nn_potential = load_model(self._model_path)

        # Move to same device as atomic model
        if self.model is not None:
            try:
                device = next(self.model.parameters()).device
                self._nn_potential = self._nn_potential.to(device)
            except StopIteration:
                pass  # Model has no parameters yet

    def _validate_hydrogens(self) -> None:
        """Warn if model appears to lack hydrogens."""
        if self.model is None:
            return

        # Check if any hydrogen atoms present
        Z = self.model.Z
        has_hydrogens = (Z == 1).any().item()

        if not has_hydrogens and self.verbose > 0:
            import warnings
            warnings.warn(
                "Model appears to have no hydrogen atoms. "
                "TorchMD-Net typically requires all-atom structures. "
                "Load with Model(strip_H=False) if hydrogens are needed.",
                UserWarning
            )

    def forward(self) -> torch.Tensor:
        """
        Compute force field energy for current model coordinates.

        Returns
        -------
        torch.Tensor
            Scalar energy tensor with gradient support.
        """
        # Ensure neural network is loaded
        self._ensure_nn_potential()

        # Validate on first call
        if not hasattr(self, '_validated'):
            self._validate_hydrogens()
            self._validated = True

        # Get coordinates and atomic numbers
        xyz = self.model.xyz()  # Shape: (n_atoms, 3)
        Z = self.model.Z        # Shape: (n_atoms,)

        # Ensure Z is long tensor
        if Z.dtype != torch.long:
            Z = Z.long()

        # Create batch tensor (single structure = all zeros)
        batch = torch.zeros(len(Z), dtype=torch.long, device=xyz.device)

        # Compute energy via TorchMD-Net
        # Returns (energy, forces) or just energy depending on model config
        result = self._nn_potential(Z, xyz, batch)

        if isinstance(result, tuple):
            energy = result[0]
        else:
            energy = result

        # Ensure scalar
        if energy.dim() > 0:
            energy = energy.sum()

        # Normalize by number of atoms if requested
        if self._normalize_by_atoms:
            energy = energy / len(Z)

        return energy

    def stats(self) -> Dict[str, StatEntry]:
        """
        Get statistics for this target.

        Returns
        -------
        dict
            Dictionary with StatEntry values.
        """
        with torch.no_grad():
            energy = self.forward().item()

        n_atoms = len(self.model.Z) if self.model is not None else 0

        return {
            "loss": stat(energy, VERBOSITY_STANDARD),
            "n_atoms": stat(n_atoms, VERBOSITY_DEBUG),
            "model_path": stat(str(self._model_path), VERBOSITY_DETAILED),
        }
