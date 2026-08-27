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

#: Objectives for :meth:`DatasetCollection.scale`, the **data-to-data** fit. Least
#: squares only, and not for want of alternatives: there is no model in that fit, so
#: there is no model error for a sigma_A or Rice likelihood to account for. ``ls_sigma``
#: weights by the propagated error on the difference, which is the correct weight
#: precisely because both sides are measurements.
DATA_SCALE_OBJECTIVES = ("ls", "ls_sigma")

#: Default for :meth:`DatasetCollection.scale`. Unit-weight least squares, matching
#: :data:`~torchref.scaling.scaler_base.DEFAULT_SCALE_TARGET` for the model-to-data fit.
DEFAULT_DATA_SCALE_OBJECTIVE = "ls"


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

    def scale(self, objective: str = DEFAULT_DATA_SCALE_OBJECTIVE):
        """
        Fit every non-reference dataset's scale and anisotropy onto the reference,
        whose own parameters are left untouched.

        **This is the data-to-data fit**, and it is the only one in the library: there
        is no model here, so there is no model error to account for and nothing for a
        sigma_A or Rice likelihood to do. Both sides are measurements of the same
        quantity, which is why the objectives are least squares --
        :data:`DATA_SCALE_OBJECTIVES`:

        ``ls``
            ``sum (F - F_ref)**2``, unit weights. The default, matching
            :data:`~torchref.scaling.scaler_base.DEFAULT_SCALE_TARGET` for the
            model-to-data fit.
        ``ls_sigma``
            ``sum (F - F_ref)**2 / (sigma**2 + sigma_ref**2)``. Both sides being
            measured is exactly the condition under which inverse-variance weighting is
            the correct weight rather than a modelling choice: the denominator is the
            propagated error on the difference being minimised, with no model-error term
            in it.

        The ``ls_sigma`` weights are computed **once, detached**, from the starting
        sigmas. They must not be re-derived inside the closure: ``sigma`` carries the
        same ``log_scale`` as ``F``, so a live denominator rewards inflating the scale to
        inflate the variance, and without the ``+log(sigma)`` term of a full Gaussian
        there is nothing to oppose it. Fixed weights are what "weighted least squares"
        means; see :mod:`torchref.scaling.scaler_base` on the related hazard of fitting a
        scale against a likelihood that carries the scale in its variance.

        Fitted on the **work set** of both datasets. L-BFGS with strong-Wolfe line
        search, 10 outer steps of ``max_iter=100``, on an objective normalised to O(1)
        because those tolerances are absolute. Members' ``log_scale``/``U_aniso`` are
        mutated, and ``requires_grad`` is turned on and back off around the fit.

        Parameters
        ----------
        objective : str, optional
            One of :data:`DATA_SCALE_OBJECTIVES`.

        Raises
        ------
        ValueError
            If no reference dataset is set, there is nothing else to scale, or
            ``objective`` is not recognised.
        """
        if objective not in DATA_SCALE_OBJECTIVES:
            raise ValueError(
                f"objective must be one of {DATA_SCALE_OBJECTIVES}, got {objective!r}"
            )
        if self._reference_dataset is None:
            raise ValueError("No reference dataset set for scaling")

        ref_ds = self._datasets[self._reference_dataset]
        to_scale = [ds for name, ds in self if name != self._reference_dataset]

        if not to_scale:
            raise ValueError("No datasets to scale against reference")

        parameters = [p for data in to_scale for p in data.parameters()]
        [p.requires_grad_(True) for p in parameters]
        optimizer = torch.optim.LBFGS(parameters, max_iter=100, line_search_fn='strong_wolfe')

        # Masks once (they do not change during the fit). The WORK subset, not
        # `masks()`: the latter is validity only -- `TensorMasks.__call__` ANDs the
        # validity masks and carries no work/free notion at all -- so fitting against it
        # puts the free reflections into the scale parameters, upstream of every target,
        # and compromises any free-set number the pipeline later reports. Degrades to
        # all-valid on a dataset with no R-free flags, which is the pre-existing
        # behaviour for that case.
        ref_mask = ref_ds.work.mask
        combined = [ds.work.mask & ref_mask for ds in to_scale]

        # Weights and the normaliser: once, detached, outside the closure.
        with torch.no_grad():
            ref_F0, ref_sig0 = ref_ds.get_corrected_data()
            weights = None
            if objective == "ls_sigma":
                # Local import: `torchref.base.targets` is not otherwise reachable from
                # `torchref.io`, and hoisting it would couple the two packages.
                from torchref.base.targets.xray_likelihoods import floor_sigma_obs

                weights = []
                for ds, cm in zip(to_scale, combined):
                    _, sig0 = ds.get_corrected_data()
                    var = (
                        floor_sigma_obs(sig0[cm]) ** 2
                        + floor_sigma_obs(ref_sig0[cm]) ** 2
                    )
                    weights.append(1.0 / var)
                n_fitted = sum(int(cm.sum()) for cm in combined)
                norm = 1.0 / max(n_fitted, 1)
            else:
                ssq = sum(float(ref_F0[cm].pow(2).sum()) for cm in combined)
                norm = 1.0 / max(ssq, 1e-30)

        def closure():
            optimizer.zero_grad()
            loss = 0.0
            # get_corrected_data, not __call__: MaskedTensor has no autograd.
            ref_F_scaled, _ = ref_ds.get_corrected_data()

            for i, (ds, cm) in enumerate(zip(to_scale, combined)):
                F_scaled, _ = ds.get_corrected_data()
                resid_sq = (F_scaled[cm] - ref_F_scaled[cm]) ** 2
                if weights is not None:
                    resid_sq = resid_sq * weights[i]
                loss = loss + torch.sum(resid_sq)
            loss = loss * norm
            loss.backward()
            return loss

        for i in range(10):
            optimizer.step(closure)
        [p.requires_grad_(False) for p in parameters]


    # ------------------------------------------------------------------
    # Batched observation accessors
    # ------------------------------------------------------------------
    #
    # Every member is expanded onto the common HKL grid by ``add_dataset``, so these
    # stack cleanly on a leading dataset axis. All of them return the **scaled**
    # observations -- the per-dataset ``log_scale``/``U_aniso`` that ``scale()`` fits
    # exists only in the corrected accessors, and a target reading the raw tensors
    # would silently ignore the inter-dataset scaling.

    def _keys_or_all(self, keys: Optional[List[str]]) -> List[str]:
        if keys is None:
            return list(self._dataset_order)
        missing = [k for k in keys if k not in self._datasets]
        if missing:
            raise KeyError(f"Unknown dataset keys: {missing}")
        return list(keys)

    def stack_F_obs(self, keys: Optional[List[str]] = None) -> torch.Tensor:
        """Scaled observed amplitudes, shape ``(n_datasets, n_reflections)``."""
        return torch.stack(
            [self._datasets[k]._corrected_or_raw()[0] for k in self._keys_or_all(keys)],
            dim=0,
        )

    def stack_F_sigma(self, keys: Optional[List[str]] = None) -> torch.Tensor:
        """Scaled amplitude sigmas, shape ``(n_datasets, n_reflections)``."""
        return torch.stack(
            [self._datasets[k]._corrected_or_raw()[1] for k in self._keys_or_all(keys)],
            dim=0,
        )

    def stack_I_obs(self, keys: Optional[List[str]] = None) -> torch.Tensor:
        """Scaled observed intensities, shape ``(n_datasets, n_reflections)``.

        Raises
        ------
        ValueError
            If any selected dataset carries no intensities.
        """
        return torch.stack(
            [
                self._require_intensities(k)[0]
                for k in self._keys_or_all(keys)
            ],
            dim=0,
        )

    def stack_I_sigma(self, keys: Optional[List[str]] = None) -> torch.Tensor:
        """Scaled intensity sigmas, shape ``(n_datasets, n_reflections)``.

        Raises
        ------
        ValueError
            If any selected dataset carries no intensities.
        """
        return torch.stack(
            [
                self._require_intensities(k)[1]
                for k in self._keys_or_all(keys)
            ],
            dim=0,
        )

    def _require_intensities(self, key: str):
        """``(I, I_sigma)`` scaled, with the dataset named in the error."""
        data = self._datasets[key]
        if data.I is None:
            raise ValueError(
                f"Dataset {key!r} carries no intensities; its reflection file had no "
                f"I/SIGI columns. An intensity-space target needs them on every member."
            )
        return data._corrected_or_raw_intensities()

    def stack_masks(
        self, keys: Optional[List[str]] = None, use_set: str = "work"
    ) -> torch.Tensor:
        """Per-dataset boolean subset masks, shape ``(n_datasets, n_reflections)``.

        Uses the 3-way ``work``/``free``/``validation`` accessors, so validation
        reflections are excluded from both work and free -- matching what the
        collection targets fit. The 2-way ``rfree_flags`` cannot express that.

        Parameters
        ----------
        keys : list of str, optional
            Datasets to stack; all of them in insertion order by default.
        use_set : {"work", "free", "val"}, optional
            Which subset to select. Default ``"work"``.
        """
        if use_set not in ("work", "free", "val"):
            raise ValueError(
                f"use_set must be 'work', 'free' or 'val'; got {use_set!r}"
            )
        attr = {"work": "work", "free": "free", "val": "validation"}[use_set]
        return torch.stack(
            [
                getattr(self._datasets[k], attr).mask
                for k in self._keys_or_all(keys)
            ],
            dim=0,
        )

    def get_centric_flags(self) -> Optional[torch.Tensor]:
        """Centric flags on the common HKL, from the reference dataset.

        A pure function of ``(hkl, spacegroup)``, so it is shared by every member and
        needs no dataset axis.
        """
        if self._reference_dataset is None:
            return None
        return self._datasets[self._reference_dataset].centric

    def component_structure_factors(
        self, model_collection, recalc: bool = False
    ) -> torch.Tensor:
        """Per-base-model ``F_calc`` on the common HKL, in the canonical convention.

        The batched counterpart of :meth:`ReflectionData.structure_factors`: models are
        evaluated at the **signed** indices so Bijvoet mates get distinct ``|F_calc|``,
        and the result is returned on the canonical ASU index that :attr:`hkl` holds.
        Use this rather than calling
        :meth:`~torchref.model.model_collection.ModelCollection.compute_component_fcalcs`
        on :attr:`hkl` directly, which would skip both halves of that convention.

        The convention is taken from the reference dataset. Members are all expanded onto
        one HKL grid, but a member with different completeness can still carry different
        ``friedel_flags`` (absent rows are filled ``False``); where they differ, the
        returned **phases** follow the reference. Amplitudes are unaffected, so a target
        working in moduli or intensities is insensitive to this.

        Parameters
        ----------
        model_collection : ModelCollection
            Supplies the shared base models.
        recalc : bool, optional
            Force recomputation rather than reusing each model's cached SF.

        Returns
        -------
        torch.Tensor
            Complex SFs of shape ``(n_base_models, n_reflections)``, row-aligned with
            :attr:`hkl` on the reflection axis.

        Raises
        ------
        ValueError
            If the collection has no reference dataset.
        """
        if self._reference_dataset is None:
            raise ValueError(
                "No reference dataset set; add a dataset before computing "
                "component structure factors."
            )
        ref = self._datasets[self._reference_dataset]
        stacked = model_collection.compute_component_fcalcs(
            ref._hkl_for_sf(), recalc=recalc
        )
        return ref.conjugate_friedel(stacked)

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
