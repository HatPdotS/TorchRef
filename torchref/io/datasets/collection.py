"""
Dataset collection for handling multiple crystallographic datasets.

This module provides the DatasetCollection class for managing multiple
related ReflectionData objects, useful for joint refinement, MAD phasing,
and time-series crystallography.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import torch

from .base import CrystalDataset
from .reflection_data import ReflectionData


@dataclass
class DatasetCollection(CrystalDataset):
    """
    Container for multiple related crystal datasets.

    All datasets share a common HKL set for efficient computation.
    Datasets are aligned using the first dataset as a reference, with
    missing reflections in subsequent datasets masked out.

    Parameters
    ----------
    verbose : int, optional
        Verbosity level (0=silent, 1=normal, 2=debug). Default is 1.
    device : str, optional
        Device for tensors ('cpu', 'cuda', etc.). Defaults to the configured device.current.

    Attributes
    ----------
    hkl : torch.Tensor
        Common HKL set for all datasets.
    n_datasets : int
        Number of datasets in collection.

    Examples
    --------
    ::

        from torchref.io import DatasetCollection, ReflectionData

        collection = DatasetCollection(device='cuda')

        native = ReflectionData().load_mtz('native.mtz')
        derivative = ReflectionData().load_mtz('derivative.mtz')

        collection.add_dataset('native', native, set_as_reference=True)
        collection.add_dataset('derivative', derivative)

        for name, dataset in collection:
            print(f"{name}: {len(dataset)} reflections")

        # Access by name
        native_F = collection['native'].F
    """

    # Collection-specific fields (not inherited from CrystalDataset)
    _datasets: Dict[str, ReflectionData] = field(default_factory=dict, repr=False)
    _dataset_order: List[str] = field(default_factory=list, repr=False)
    _reference_dataset: Optional[str] = field(default=None, repr=False)
    _common_hkl: Optional[torch.Tensor] = field(default=None, repr=False)
    _cell: Optional[torch.Tensor] = field(default=None, repr=False)
    _spacegroup: Optional[str] = field(default=None, repr=False)
    _resolution: Optional[torch.Tensor] = field(default=None, repr=False)
    _scale_factors: Dict[str, torch.Tensor] = field(default_factory=dict, repr=False)

    def add_dataset(
        self, name: str, dataset: ReflectionData, set_as_reference: bool = False
    ) -> "DatasetCollection":
        """
        Add a dataset to the collection.

        Parameters
        ----------
        name : str
            Identifier for this dataset.
        dataset : ReflectionData
            The dataset to add.
        set_as_reference : bool, optional
            If True, use this dataset's HKL as the reference.
            Default is False, but the first dataset added automatically
            becomes the reference.

        Returns
        -------
        DatasetCollection
            Self, for method chaining.

        Raises
        ------
        ValueError
            If a dataset with the same name already exists.

        Examples
        --------
        ::

            collection = DatasetCollection()
            collection.add_dataset('native', native_data, set_as_reference=True)
            collection.add_dataset('derivative', derivative_data)
        """
        if name in self._datasets:
            raise ValueError(f"Dataset '{name}' already exists in collection")

        # First dataset or explicit reference becomes the reference
        if len(self._datasets) == 0 or set_as_reference:
            self._reference_dataset = name
            self._common_hkl = dataset.hkl.clone()
            if dataset.cell is not None:
                self._cell = dataset.cell.clone()
            self._spacegroup = dataset.spacegroup

        # Validate HKL against reference
        if self._common_hkl is not None and dataset.hkl is not None:
            dataset.validate_hkl(self._common_hkl)

        # Move dataset to same device as collection. ``.to`` is a no-op when
        # the dataset is already on this device (handles cuda/mps/cpu uniformly
        # and avoids the ``mps`` vs ``mps:0`` index-mismatch edge case).
        dataset.to(self.device)

        self._datasets[name] = dataset
        self._dataset_order.append(name)

        if self.verbose > 0:
            print(f"Added dataset '{name}' ({len(dataset)} reflections)")

        return self

    @property
    def hkl(self) -> Optional[torch.Tensor]:
        """Common HKL set for all datasets."""
        return self._common_hkl

    @hkl.setter
    def hkl(self, value: Optional[torch.Tensor]) -> None:
        """Set common HKL (redirects to _common_hkl)."""
        self._common_hkl = value

    @property
    def datasets(self) -> Dict[str, ReflectionData]:
        """Access all datasets as a dictionary."""
        return self._datasets

    @property
    def n_datasets(self) -> int:
        """Number of datasets in collection."""
        return len(self._datasets)

    @property
    def reference_dataset(self) -> Optional[str]:
        """Name of the reference dataset."""
        return self._reference_dataset

    @property
    def spacegroup(self) -> Optional[str]:
        """Space group of the reference dataset."""
        return self._spacegroup

    @spacegroup.setter
    def spacegroup(self, value: Optional[str]) -> None:
        """Set space group (redirects to _spacegroup)."""
        self._spacegroup = value

    def __getitem__(self, name: str) -> ReflectionData:
        """
        Get dataset by name.

        Parameters
        ----------
        name : str
            Name of the dataset.

        Returns
        -------
        ReflectionData
            The requested dataset.

        Raises
        ------
        KeyError
            If dataset name not found.
        """
        return self._datasets[name]

    def __iter__(self) -> Iterator[Tuple[str, ReflectionData]]:
        """
        Iterate over (name, dataset) pairs in order of addition.

        Yields
        ------
        tuple of (str, ReflectionData)
            Name and dataset for each dataset in collection.
        """
        for name in self._dataset_order:
            yield name, self._datasets[name]

    def __len__(self) -> int:
        """Number of reflections in common HKL set."""
        return len(self._common_hkl) if self._common_hkl is not None else 0

    def __contains__(self, name: str) -> bool:
        """Check if dataset exists in collection."""
        return name in self._datasets

    def _calculate_resolution(self) -> None:
        """Calculate resolution for common HKL."""
        from torchref.base import math_torch

        if self._common_hkl is None or self._cell is None:
            return

        s = math_torch.get_scattering_vectors(self._common_hkl, self._cell)
        resolution = 1.0 / torch.linalg.norm(s, axis=1)
        self._resolution = resolution

    def __call__(self, mask: bool = True) -> Dict[str, Tuple]:
        """
        Return all datasets' data scaled if scale factors are set.

        Parameters
        ----------
        mask : bool, optional
            Whether to apply masking. Default is True.

        Returns
        -------
        dict
            Dictionary mapping name to (hkl, F, F_sigma, rfree) tuples.
        """
        return {name: ds(mask=mask, scale=True) for name, ds in self}

    def scale(self):
        """
        Scale all non-reference datasets onto the reference dataset.

        Optimizes the scaling parameters of every non-reference dataset to
        minimize the summed squared error between their corrected structure
        factors and those of the reference dataset, correcting for both
        overall scale and anisotropy. Uses an L-BFGS optimizer with strong
        Wolfe line search over a fixed number of optimization steps.

        Raises
        ------
        ValueError
            If no reference dataset has been set, or if the collection
            contains only the reference dataset (nothing to scale).

        Notes
        -----
        Scaling parameters of the reference dataset are left untouched; only
        the non-reference datasets are optimized.
        """
        if self._reference_dataset is None:
            raise ValueError("No reference dataset set for scaling")


        ref_ds = self._datasets[self._reference_dataset]
        to_scale = [ds for name, ds in self if name != self._reference_dataset]

        if not to_scale:
            raise ValueError("No datasets to scale against reference")
        
        parameters = [p for data in to_scale for p in data.parameters()]
        [p.requires_grad_(True) for p in parameters]
        optimizer = torch.optim.LBFGS(parameters, max_iter=100, line_search_fn='strong_wolfe')

        # Get masks once (they don't change during optimization)
        ref_mask = ref_ds.masks()
        ds_masks = [ds.masks() for ds in to_scale]

        def closure():
            optimizer.zero_grad()
            loss = 0.0
            # Get scaled reference data (bypassing MaskedTensor which doesn't support autograd)
            ref_F_scaled, _ = ref_ds.get_corrected_data()

            for ds, ds_mask in zip(to_scale, ds_masks):
                F_scaled, _ = ds.get_corrected_data()
                # Combine masks
                combined_mask = ds_mask & ref_mask
                # Compute loss on valid reflections only
                F_data = F_scaled[combined_mask]
                ref_F_data = ref_F_scaled[combined_mask]
                loss = loss + torch.sum((F_data - ref_F_data) ** 2)
            loss.backward()
            return loss

        for i in range(10):
            optimizer.step(closure)
        [p.requires_grad_(False) for p in parameters]


    def keys(self) -> List[str]:
        """Return list of dataset names."""
        return list(self._dataset_order)

    def values(self) -> List[ReflectionData]:
        """Return list of datasets."""
        return [self._datasets[name] for name in self._dataset_order]

    def items(self) -> List[Tuple[str, ReflectionData]]:
        """Return list of (name, dataset) tuples."""
        return [(name, self._datasets[name]) for name in self._dataset_order]

    def get(self, name: str, default=None) -> Optional[ReflectionData]:
        """Get dataset by name with default fallback."""
        return self._datasets.get(name, default)

    def __repr__(self) -> str:
        """String representation of collection."""
        n_datasets = self.n_datasets
        n_refl = len(self)
        sg = self.spacegroup or "unknown"
        names = ", ".join(self._dataset_order[:3])
        if n_datasets > 3:
            names += f", ... ({n_datasets} total)"
        return (
            f"DatasetCollection(datasets=[{names}], "
            f"n_reflections={n_refl}, spacegroup='{sg}', device={self.device})"
        )
