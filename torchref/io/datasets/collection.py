"""
Dataset collection for handling multiple crystallographic datasets.

This module provides the DatasetCollection class for managing multiple
related ReflectionData objects, useful for joint refinement, MAD phasing,
and time-series crystallography.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Tuple

import torch

from .base import CrystalDataset
from .reflection_data import ReflectionData
from .scaled_dataset import ScaledDataset

if TYPE_CHECKING:
    from torchref.scaling import DatasetScaler


@dataclass
class DatasetCollection(CrystalDataset):
    """
    Container for multiple related crystal datasets on a common HKL set.

    Members are copied onto the union HKL grid without changing input datasets.
    The reference supplies cell and space-group metadata, not a fixed scale.
    ``scale()`` installs ScaledDataset members backed by one shared scaler. Dict-like access via ``[]``, ``keys()``,
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
    scaler: Optional["DatasetScaler"] = field(default=None, repr=False)
    scaling_metrics: dict = field(default_factory=dict, repr=False)

    def add_dataset(
        self, name: str, dataset: ReflectionData, set_as_reference: bool = False
    ) -> "DatasetCollection":
        """Add a copied dataset and rebuild the union reflection grid.

        Parameters
        ----------
        name : str
            Unique member name.
        dataset : ReflectionData
            Raw or scaled observations; scaled inputs contribute their raw values.
        set_as_reference : bool
            Use this dataset's cell and symmetry as collection metadata.

        Returns
        -------
        DatasetCollection
            Self. Membership changes discard the fitted joint scaler; call scale()
            again to fit all members. Existing raw inputs are never mutated.
        """
        if name in self._datasets:
            raise ValueError(f"Dataset '{name}' already exists in collection")
        members = {
            k: d.raw_data() if isinstance(d, ScaledDataset) else d
            for k, d in self._datasets.items()
        }
        raw = dataset.raw_data() if isinstance(dataset, ScaledDataset) else dataset
        members[name] = raw.__select__(torch.arange(len(raw), device=raw.device))
        members[name].source = None
        members[name].spacegroup = raw.spacegroup.copy()
        if (
            len({d.spacegroup.xhm for d in members.values()}) != 1
            or len({d.friedel_merged for d in members.values()}) != 1
        ):
            raise ValueError(
                "Datasets require compatible symmetry and Friedel conventions"
            )
        if not self._dataset_order or set_as_reference:
            self._reference_dataset = name
            self._cell = raw.cell.clone() if raw.cell is not None else None
            self._spacegroup = members[name].spacegroup
        self._dataset_order.append(name)
        union_hkl = torch.unique(
            torch.cat(
                [
                    (d.hkl if d.friedel_merged else d._hkl_for_sf()).to(self.device)
                    for d in members.values()
                ]
            ),
            dim=0,
        )
        identity_hkl = None
        if raw.friedel_merged:
            self._common_hkl = union_hkl
        else:
            canonical, _, _, order = raw.spacegroup.canonicalize_hkl(union_hkl)
            self._common_hkl = canonical
            identity_hkl = union_hkl[order]
        for data in members.values():
            data.to(self.device)
            data.validate_hkl(self._common_hkl, identity_hkl=identity_hkl)
        self._datasets = members
        self.scaler = None
        self.scaling_metrics = {}
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
        """Return full observation arrays for every collection member.

        Parameters
        ----------
        mask : bool, optional
            Wrap amplitudes and uncertainties in detached MaskedTensors carrying
            validity masks. If False, return live observation tensors.

        Returns
        -------
        dict
            Name to (HKL, amplitudes, sigmas, work flags). HKL has shape (H, 3);
            other arrays have shape (H,) in each dataset's observation units.
        """
        from torch.masked import MaskedTensor

        result = {}
        for name, data in self:
            amplitudes, sigmas = data.F, data.F_sigma
            if mask:
                valid = data.masks()
                if not bool(valid.any()):
                    raise ValueError(f"Dataset {name!r} has no valid observations")
                amplitudes = MaskedTensor(amplitudes.detach().clone(), valid)
                if sigmas is not None:
                    sigmas = MaskedTensor(sigmas.detach().clone(), valid)
            result[name] = data.hkl, amplitudes, sigmas, data.rfree_flags
        return result

    def scale(self, nsteps: int = 10, max_iter: int = 100) -> "DatasetCollection":
        """Jointly scale observations and expose live ScaledDataset members.

        Parameters
        ----------
        nsteps, max_iter : int
            Outer steps and per-step iteration limit for the dedicated scaler.

        Returns
        -------
        DatasetCollection
            Self. Retrieve scaled observations from this collection; references
            to original inputs remain raw. Repeated calls reuse the parameter owner.
        """
        from torchref.scaling.dataset_scaler import DatasetScaler

        raw = {
            k: d.raw_data() if isinstance(d, ScaledDataset) else d
            for k, d in self._datasets.items()
        }
        if self.scaler is None:
            scaler = DatasetScaler(raw, device=self.device)
            metrics = scaler.fit(nsteps=nsteps, max_iter=max_iter)
            self._datasets = {k: ScaledDataset(d, scaler, k) for k, d in raw.items()}
            self.scaler = scaler
        else:
            self.scaler.datasets = raw
            metrics = self.scaler.fit(nsteps=nsteps, max_iter=max_iter)
        self.scaling_metrics = metrics
        return self

    def _get_state(self) -> dict:
        raw = {
            k: (d.raw_data() if isinstance(d, ScaledDataset) else d)._get_state()
            for k, d in self._datasets.items()
        }
        scaler_state = None if self.scaler is None else self.scaler.get_state()
        if scaler_state is not None:
            scaler_state.pop("datasets")
        return {
            "datasets": raw,
            "reference": self._reference_dataset,
            "scaler": scaler_state,
            "scaling_metrics": self.scaling_metrics,
        }

    @classmethod
    def _from_state(cls, state: dict, device=None) -> "DatasetCollection":
        from torchref.scaling.dataset_scaler import DatasetScaler

        result = cls(device=device) if device is not None else cls()
        for key, raw in state["datasets"].items():
            result.add_dataset(
                key,
                ReflectionData._from_state(dict(raw), device),
                set_as_reference=key == state["reference"],
            )
        if state["scaler"] is not None:
            result.scaler = DatasetScaler.from_state(
                {**state["scaler"], "datasets": state["datasets"]}, device
            )
            result._datasets = {
                k: ScaledDataset(d, result.scaler, k)
                for k, d in result._datasets.items()
            }
        result.scaling_metrics = state.get("scaling_metrics", {})
        return result

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
            [self._require_intensities(k)[0] for k in self._keys_or_all(keys)],
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
            [self._require_intensities(k)[1] for k in self._keys_or_all(keys)],
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
            [getattr(self._datasets[k], attr).mask for k in self._keys_or_all(keys)],
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
