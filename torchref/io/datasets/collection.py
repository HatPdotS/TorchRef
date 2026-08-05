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
    Container for multiple related crystal datasets on a common HKL set.

    Members are expanded in place onto the reference dataset's HKL grid
    (:meth:`ReflectionData.validate_hkl`) and moved to the collection's device,
    so adding a dataset MUTATES it. Dict-like access via ``[]``, ``keys()``,
    ``values()``, ``items()``, ``get()``, and iteration yields
    ``(name, dataset)`` in insertion order.

    Parameters
    ----------
    verbose : int, optional
        Verbosity level (0=silent, 1=normal, 2=debug). Default is 1.
    device : str, optional
        Device for tensors. Defaults to ``get_default_device()``.

    Attributes
    ----------
    hkl : torch.Tensor
        Common HKL set for all datasets.
    n_datasets : int
        Number of datasets in the collection.
    datasets : Dict[str, ReflectionData]
        All member datasets keyed by name.
    reference_dataset : str or None
        Name of the reference dataset (drives HKL alignment).
    spacegroup : str or None
        Space group of the reference dataset.
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
        Add a dataset, expanding it onto the reference HKL grid **in place**.

        Parameters
        ----------
        name : str
            Identifier for this dataset.
        dataset : ReflectionData
            The dataset to add.
        set_as_reference : bool, optional
            If True, this dataset's HKL becomes the reference. The first dataset
            added becomes the reference regardless.

        Returns
        -------
        DatasetCollection
            Self, for method chaining.

        Raises
        ------
        ValueError
            If a dataset with the same name already exists.
        """
        if name in self._datasets:
            raise ValueError(f"Dataset '{name}' already exists in collection")

        if len(self._datasets) == 0 or set_as_reference:
            self._reference_dataset = name
            self._common_hkl = dataset.hkl.clone()
            if dataset.cell is not None:
                self._cell = dataset.cell.clone()
            self._spacegroup = dataset.spacegroup

        if self._common_hkl is not None and dataset.hkl is not None:
            dataset.validate_hkl(self._common_hkl)

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
        """Get a member dataset by name; ``KeyError`` if absent."""
        return self._datasets[name]

    def __iter__(self) -> Iterator[Tuple[str, ReflectionData]]:
        """Iterate over ``(name, dataset)`` pairs in order of addition."""
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

    def harmonize_partition(
        self,
        val_fraction_of_free: Optional[float] = None,
        seed: Optional[int] = None,
        source: Optional[str] = None,
    ) -> "DatasetCollection":
        """Make the work/free (and validation) partition identical across members.

        Overwrites every non-source member's ``rfree_flags`` /
        ``validation_flags`` with the source's (row-aligned clones), because
        per-dataset free sets would let a reflection that is free in one member
        leak into another's work set and bias the cross-dataset R-free.

        Parameters
        ----------
        val_fraction_of_free : float, optional
            Fraction of the free reflections to reassign as a held-out validation
            set, shared across all datasets. If None, no validation set is created
            (existing ``validation_flags`` on the source, if any, are still
            broadcast).
        seed : int, optional
            Seed for the validation split (reproducibility).
        source : str, optional
            Name of the member whose partition is canonical. Defaults to the
            reference dataset (or the first added dataset).

        Returns
        -------
        DatasetCollection
            Self, for chaining.
        """
        if not self._datasets:
            raise RuntimeError("Cannot harmonize an empty collection.")

        src_name = source or self._reference_dataset or self._dataset_order[0]
        if src_name not in self._datasets:
            raise KeyError(f"Source dataset {src_name!r} not in collection.")
        src = self._datasets[src_name]

        if src.rfree_flags is None:
            raise ValueError(
                f"Source dataset {src_name!r} has no rfree_flags to harmonize on."
            )

        if val_fraction_of_free is not None:
            src.generate_validation_set(
                val_fraction_of_free=val_fraction_of_free, seed=seed
            )

        # Broadcast the canonical partition to every member (row-aligned clones).
        canonical_rfree = src.rfree_flags
        canonical_val = src.validation_flags
        for name, ds in self._datasets.items():
            if name == src_name:
                continue
            ds.rfree_flags = canonical_rfree.clone().to(ds.device)
            if canonical_val is not None:
                ds.validation_flags = canonical_val.clone().to(ds.device)

        if self.verbose > 0:
            n_work = int(canonical_rfree.to(torch.bool).sum().item())
            total = len(canonical_rfree)
            n_val = (
                int(canonical_val.to(torch.bool).sum().item())
                if canonical_val is not None
                else 0
            )
            n_free = total - n_work - n_val
            print(
                f"Harmonized partition from {src_name!r} across "
                f"{self.n_datasets} datasets: work={n_work}, free={n_free}, "
                f"val={n_val}."
            )
        return self

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
            Name -> ``(hkl, F, F_sigma, rfree)``. Routed through the deprecated
            ``ReflectionData.__call__``, so F/F_sigma are MaskedTensors and each
            member emits a DeprecationWarning.
        """
        return {name: ds(mask=mask, scale=True) for name, ds in self}

    def scale(self):
        """
        Least-squares fit every non-reference dataset's scale and anisotropy
        onto the reference, whose own parameters are left untouched.

        L-BFGS with strong-Wolfe line search, 10 outer steps of ``max_iter=100``.
        Members' ``log_scale``/``U_aniso`` are mutated, and ``requires_grad`` is
        turned on and back off around the fit.

        Raises
        ------
        ValueError
            If no reference dataset is set, or there is nothing else to scale.
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
            # get_corrected_data, not __call__: MaskedTensor has no autograd.
            ref_F_scaled, _ = ref_ds.get_corrected_data()

            for ds, ds_mask in zip(to_scale, ds_masks):
                F_scaled, _ = ds.get_corrected_data()
                combined_mask = ds_mask & ref_mask
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
