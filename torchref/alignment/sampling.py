"""
Atom pair sampling for efficient Patterson vector generation.

Supports weighted sampling to prioritize informative pairs
(heavy atoms, close distances).
"""

from typing import Optional

import torch


class VectorSampler:
    """
    Samples atom pairs for Patterson vector matching.

    Supports weighted sampling to prioritize informative pairs
    (heavy atoms via Z-weighting). Samples pairs from the asymmetric
    unit (ASU) only - symmetry is already encoded in the Patterson map.

    Parameters
    ----------
    model : Model
        TorchRef Model object. The caller is responsible for filtering
        atoms (e.g., excluding waters) before passing to this class.
    weighting : str, optional
        Weighting scheme: 'uniform' or 'Z2' (weight by atomic number squared).
        Default is 'Z2'.
    seed : int, optional
        Random seed for reproducibility. Default is None.

    Attributes
    ----------
    model : Model
        The model used for sampling.
    n_atoms : int
        Number of atoms in the model.
    weighting : str
        Weighting scheme used.
    weights : torch.Tensor
        Sampling weights for each atom (n_atoms,).
    rng : torch.Generator
        Random number generator.
    """

    def __init__(self, model, weighting: str = "Z2", seed: int = None):
        """
        Initialize the VectorSampler.

        Parameters
        ----------
        model : Model
            TorchRef Model object. The caller is responsible for filtering
            atoms (e.g., excluding waters) before passing to this class.
        weighting : str
            Weighting scheme for sampling.
        seed : int, optional
            Random seed for reproducibility.
        """
        self.model = model
        self.n_atoms = len(model.pdb)
        self.weighting = weighting
        self.rng = (
            torch.Generator().manual_seed(seed)
            if seed is not None
            else torch.Generator()
        )
        self.weights = self._compute_weights()

    def _compute_weights(
        self,
    ) -> torch.Tensor:
        """
        Compute sampling probability for each atom based on atomic number.

        Returns
        -------
        torch.Tensor
            Weight for each atom with shape (n_atoms,).
        """
        from torchref.utils.pse import PERIODIC_TABLE

        elements = self.model.pdb.element.values

        Zs = torch.tensor(
            [PERIODIC_TABLE[el]["number"] for el in elements],
            dtype=torch.float32,
            device=self.model.device,
        )

        if self.weighting == "Z2":
            weights = Zs**2
        else:  # uniform
            weights = torch.ones_like(Zs)

        weights = weights / weights.sum()  # Normalize to probabilities
        return weights

    def sample(
        self, n_vectors: int, weights: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample atom pairs according to weighting scheme.

        Parameters
        ----------
        n_vectors : int
            Number of atom pairs to sample.
        weights : torch.Tensor, optional
            Override weights for sampling. If None, uses self.weights.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Two tensors of shape (n_vectors,) containing
            the indices of the sampled atom pairs.
        """
        w = weights if weights is not None else self.weights

        # Sample first indices according to weights
        idx1 = torch.multinomial(w, n_vectors, replacement=True, generator=self.rng)

        # Sample second indices according to weights
        idx2 = torch.multinomial(w, n_vectors, replacement=True, generator=self.rng)

        # Redraw idx2 where it equals idx1
        same_mask = idx1 == idx2
        max_attempts = 100  # Prevent infinite loop
        attempt = 0
        while same_mask.any() and attempt < max_attempts:
            n_resample = same_mask.sum().item()
            idx2[same_mask] = torch.multinomial(
                w, n_resample, replacement=True, generator=self.rng
            )
            same_mask = idx1 == idx2
            attempt += 1

        return idx1, idx2
