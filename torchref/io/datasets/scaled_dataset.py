"""ReflectionData subclass exposing live scaler-owned observation corrections."""

from dataclasses import fields
from typing import TYPE_CHECKING

import torch

from .reflection_data import ReflectionData

if TYPE_CHECKING:
    from torchref.scaling.dataset_scaler import DatasetScaler


class _ScaledObservation:
    """Keep dataclass initialization in raw storage and scale only public reads."""

    def __init__(self, name, power):
        self.name, self.power = name, power

    def __get__(self, obj, owner=None):
        if obj is None:
            return None
        raw = obj.__dict__.get("_raw_" + self.name)
        scaler = obj.__dict__.get("scaler")
        if raw is None or scaler is None:
            return raw
        correction = scaler(obj.scale_key, obj.hkl).to(raw)
        return raw * correction.pow(self.power)

    def __set__(self, obj, value):
        if obj.__dict__.get("scaler") is not None:
            raise AttributeError(
                "Scaled observations are read-only; edit the raw dataset before scaling"
            )
        obj.__dict__["_raw_" + self.name] = value


class ScaledDataset(ReflectionData):
    """Expose a raw dataset through one live row of a shared DatasetScaler.

    Parameters
    ----------
    data : ReflectionData
        Measurements and metadata, copied without changing the source. Passing
        a scaled dataset uses its raw observations, never a second correction.
    scaler : DatasetScaler
        Shared parameter owner, retained strongly even outside a collection.
    key : str
        Stable dataset key in the scaler.

    Notes
    -----
    F/F_sigma (amplitude units) and I/I_sigma (intensity units), shape (N,),
    are corrected read-only expressions. Explicit *_raw properties expose the
    stored measurements. Selection/copy preserves the shared scaler; moving a
    view also moves the shared scaler. Raw source datasets remain unchanged.
    """

    F = _ScaledObservation("F", 1)
    F_sigma = _ScaledObservation("F_sigma", 1)
    I = _ScaledObservation("I", 2)
    I_sigma = _ScaledObservation("I_sigma", 2)

    def __init__(self, data: ReflectionData, scaler: "DatasetScaler", key: str) -> None:
        if key not in scaler.keys:
            raise KeyError(key)
        raw = data.raw_data() if isinstance(data, ScaledDataset) else data
        raw = raw.__select__(torch.arange(len(raw), device=raw.device))
        raw.spacegroup = raw.spacegroup.copy()
        self._install_raw(raw)
        self.scaler = scaler
        self.scale_key = key
        self.to(scaler.device)

    def _install_raw(self, raw):
        self.scaler = None
        super().__init__(
            **{f.name: getattr(raw, f.name) for f in fields(ReflectionData)}
        )
        self.masks = raw.masks
        self.source = None

    @property
    def F_raw(self) -> torch.Tensor | None:
        """Uncorrected amplitudes, shape (N,), in input amplitude units."""
        return self._raw_F

    @property
    def F_sigma_raw(self) -> torch.Tensor | None:
        """Uncorrected amplitude sigmas, shape (N,), in input amplitude units."""
        return self._raw_F_sigma

    @property
    def I_raw(self) -> torch.Tensor | None:
        """Uncorrected intensities, shape (N,), in input intensity units."""
        return self._raw_I

    @property
    def I_sigma_raw(self) -> torch.Tensor | None:
        """Uncorrected intensity sigmas, shape (N,), in input intensity units."""
        return self._raw_I_sigma

    def raw_data(self) -> ReflectionData:
        """Return an independent parameter-free copy of the raw observations."""
        values = {
            f.name: getattr(self, f.name)
            for f in fields(ReflectionData)
            if f.name not in ("F", "F_sigma", "I", "I_sigma", "source")
        }
        values.update(
            {
                name: getattr(self, name + "_raw")
                for name in ("F", "F_sigma", "I", "I_sigma")
            }
        )
        raw = ReflectionData(**values)
        raw.masks = self.masks
        result = raw.__select__(torch.arange(len(raw), device=raw.device))
        result.source = None
        return result

    def __select__(self, indices: torch.Tensor, op=None) -> "ScaledDataset":
        """Select reflection indices or a boolean mask, preserving live scaling."""
        raw = self.raw_data().__select__(indices, op=op)
        return ScaledDataset(raw, self.scaler, self.scale_key)

    def copy(self) -> "ScaledDataset":
        """Copy observations and metadata while sharing the scaler parameters."""
        return ScaledDataset(self.raw_data(), self.scaler, self.scale_key)

    def __deepcopy__(self, memo: dict) -> "ScaledDataset":
        """Copy this view without duplicating the shared parameter owner."""
        result = self.copy()
        memo[id(self)] = result
        return result

    def validate_hkl(
        self, hkl_ref: torch.Tensor, *, identity_hkl: torch.Tensor | None = None
    ) -> "ScaledDataset":
        """Align this view to HKL (H, 3), without modifying its source or scaler."""
        raw = self.raw_data().validate_hkl(hkl_ref, identity_hkl=identity_hkl)
        scaler, key = self.scaler, self.scale_key
        self._install_raw(raw)
        self.scaler, self.scale_key = scaler, key
        return self

    def _get_state(self) -> dict:
        return {
            "raw": self.raw_data()._get_state(),
            "scaler": self.scaler.get_state(),
            "key": self.scale_key,
        }

    @classmethod
    def _from_state(cls, state: dict, device=None) -> "ScaledDataset":
        from torchref.scaling.dataset_scaler import DatasetScaler

        scaler = DatasetScaler.from_state(state["scaler"], device)
        raw = ReflectionData._from_state(dict(state["raw"]), device)
        return cls(raw, scaler, state["key"])
