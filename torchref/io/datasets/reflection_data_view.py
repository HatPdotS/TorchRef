"""
Live sub-set view over a parent ``ReflectionData``.

A ``ReflectionDataView`` exposes the same attribute interface as
``ReflectionData`` (``hkl``, ``F``, ``F_sigma``, ``resolution``, ...) but
returns the parent's tensors index-selected to a particular flag class
(work / free / validation). Two key properties:

1. **Object identity is stable.** ``data.work`` always returns the same
   ``ReflectionDataView`` instance for the lifetime of the parent.
2. **Live propagation.** When the parent's ``rfree_flags`` changes, the
   view's cached tensors are invalidated and recomputed on next access.
   References the user is holding remain valid and show fresh data.

Cached tensors are keyed by a monotonic ``_flags_version`` counter on the
parent. Bumping that counter (via the ``rfree_flags`` setter or
``_bump_flags_version()``) invalidates all dependent views.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

import torch

if TYPE_CHECKING:
    from torchref.io.datasets.reflection_data import ReflectionData


# Attributes that are per-reflection tensors and must be index-selected
# from the parent to expose only this view's subset.
_PER_REFLECTION_ATTRS = frozenset(
    {
        "hkl",
        "F",
        "F_sigma",
        "I",
        "I_sigma",
        "rfree_flags",
        "resolution",
        "phase",
        "fom",
        "outlier_flags",
        "_centric_flags",
        "E",
        "E_squared",
        "F_squared_corrected",
        "radial_shell_indices",
    }
)

# Attributes that are dataset-wide scalars / Cell / SpaceGroup / etc.
# These are delegated to the parent unchanged.
_PARENT_PASSTHROUGH = frozenset(
    {
        "cell",
        "spacegroup",
        "device",
        "verbose",
        "wilson_b",
        "wilson_b_structure",
        "wilson_b_solvent",
        "wilson_k_sol",
        "U_aniso",
        "log_scale",
        "rfree_source",
        "amplitude_source",
        "intensity_source",
        "phase_source",
        "outlier_detection_params",
    }
)


class ReflectionDataView:
    """
    A live, cached view onto a flag-class subset of a ``ReflectionData``.

    Parameters
    ----------
    parent : ReflectionData
        The parent dataset.
    set_name : {'work', 'free', 'val'}
        Which flag class this view exposes.

    Notes
    -----
    Tensor attribute reads are cached and tagged with the parent's
    ``_flags_version``. When the version changes, the cache is cleared
    and tensors are rebuilt on next access via ``index_select``.

    The view is *not* a drop-in for every ``ReflectionData`` method —
    only the attributes used by downstream consumers (X-ray targets,
    Wilson prior, Scaler queries) are exposed explicitly. Other
    attributes fall through to the parent via ``__getattr__``.
    """

    __slots__ = (
        "_parent",
        "_set_name",
        "_cache",
        "_cache_version",
        "_bin_indices",
        "_bin_indices_version",
        "_n_bins",
    )

    def __init__(self, parent: "ReflectionData", set_name: str):
        if set_name not in ("work", "free", "val"):
            raise ValueError(f"set_name must be 'work', 'free', or 'val'; got {set_name!r}")
        object.__setattr__(self, "_parent", parent)
        object.__setattr__(self, "_set_name", set_name)
        object.__setattr__(self, "_cache", {})
        object.__setattr__(self, "_cache_version", -1)
        object.__setattr__(self, "_bin_indices", None)
        object.__setattr__(self, "_bin_indices_version", -1)
        object.__setattr__(self, "_n_bins", None)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _check_cache(self) -> None:
        """Invalidate the cache if the parent's flags have changed."""
        parent_version = self._parent._flags_version
        if self._cache_version != parent_version:
            self._cache.clear()
            object.__setattr__(self, "_cache_version", parent_version)

    @property
    def _idx(self) -> torch.Tensor:
        """Integer indices into the parent for this view's reflections."""
        return getattr(self._parent, f"{self._set_name}_idx")

    @property
    def _mask(self) -> torch.Tensor:
        """Boolean mask into the parent for this view's reflections."""
        return getattr(self._parent, f"{self._set_name}_mask")

    def _select(self, attr: str) -> Optional[torch.Tensor]:
        """
        Return ``parent.<attr>`` index-selected to this view, cached.

        Returns ``None`` if the parent attribute is ``None``. Non-tensor
        attributes and tensors whose leading dim does not match the
        parent's reflection count are returned as-is from the parent.
        """
        self._check_cache()
        cache: Dict[str, Any] = self._cache
        if attr in cache:
            return cache[attr]
        val = getattr(self._parent, attr, None)
        if val is None:
            cache[attr] = None
            return None
        if isinstance(val, torch.Tensor):
            parent_n = len(self._parent)
            if val.dim() >= 1 and val.shape[0] == parent_n:
                result = val.index_select(0, self._idx)
            else:
                result = val
        else:
            result = val
        cache[attr] = result
        return result

    # ------------------------------------------------------------------
    # Per-reflection tensor properties
    # ------------------------------------------------------------------

    @property
    def hkl(self) -> Optional[torch.Tensor]:
        return self._select("hkl")

    @property
    def F(self) -> Optional[torch.Tensor]:
        return self._select("F")

    @property
    def F_sigma(self) -> Optional[torch.Tensor]:
        return self._select("F_sigma")

    @property
    def I(self) -> Optional[torch.Tensor]:  # noqa: E743
        return self._select("I")

    @property
    def I_sigma(self) -> Optional[torch.Tensor]:
        return self._select("I_sigma")

    @property
    def rfree_flags(self) -> Optional[torch.Tensor]:
        return self._select("rfree_flags")

    @property
    def resolution(self) -> Optional[torch.Tensor]:
        return self._select("resolution")

    @property
    def phase(self) -> Optional[torch.Tensor]:
        return self._select("phase")

    @property
    def fom(self) -> Optional[torch.Tensor]:
        return self._select("fom")

    @property
    def outlier_flags(self) -> Optional[torch.Tensor]:
        return self._select("outlier_flags")

    @property
    def _centric_flags(self) -> Optional[torch.Tensor]:
        return self._select("_centric_flags")

    @property
    def E(self) -> Optional[torch.Tensor]:
        return self._select("E")

    @property
    def E_squared(self) -> Optional[torch.Tensor]:
        return self._select("E_squared")

    @property
    def F_squared_corrected(self) -> Optional[torch.Tensor]:
        return self._select("F_squared_corrected")

    @property
    def radial_shell_indices(self) -> Optional[torch.Tensor]:
        return self._select("radial_shell_indices")

    # ------------------------------------------------------------------
    # Bin indices: recomputed on the view's own resolution
    # ------------------------------------------------------------------

    @property
    def bin_indices(self) -> Optional[torch.Tensor]:
        """
        Resolution-bin assignment computed on the view's own reflections.

        Recomputed (lazily) whenever the parent's flags change.
        """
        if self._bin_indices_version != self._parent._flags_version:
            object.__setattr__(self, "_bin_indices", None)
            object.__setattr__(self, "_n_bins", None)
            object.__setattr__(self, "_bin_indices_version", self._parent._flags_version)
        return self._bin_indices

    @bin_indices.setter
    def bin_indices(self, value: Optional[torch.Tensor]) -> None:
        object.__setattr__(self, "_bin_indices", value)
        object.__setattr__(self, "_bin_indices_version", self._parent._flags_version)

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return int(self._idx.shape[0]) if self._idx is not None else 0

    def __repr__(self) -> str:
        return (
            f"ReflectionDataView(set={self._set_name}, n={len(self)}, "
            f"parent_id=0x{id(self._parent):x})"
        )

    def __getattr__(self, name: str) -> Any:
        """
        Fallback: delegate any unknown attribute to the parent.

        This makes the view duck-type compatible with most
        ``ReflectionData`` consumers without enumerating every accessor.
        """
        # Avoid infinite recursion: __slots__ attrs accessed before init
        # go through normal attribute lookup, not here.
        if name in _PARENT_PASSTHROUGH:
            return getattr(self._parent, name)
        if name in _PER_REFLECTION_ATTRS:
            return self._select(name)
        # Methods and anything else: ask the parent.
        return getattr(self._parent, name)
