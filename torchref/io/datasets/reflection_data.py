"""
Reflection data container for crystallographic datasets.

This module provides the ReflectionData class for handling single-crystal
reflection data including Miller indices, structure factor amplitudes,
intensities, and R-free flags.
"""

import warnings
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Tuple, Union

import numpy as np
import torch

from torchref.base import math_torch
from torchref.base.french_wilson import french_wilson_auto
from torchref.config import dtypes, normalize_device
from torchref.io import cif, mtz
from torchref.io.datasets.base import CrystalDataset
from torchref.symmetry import Cell, SpaceGroup
from torchref.utils.debug_utils import DebugMixin
from torchref.utils.utils import TensorMasks

if TYPE_CHECKING:
    from torchref.model.model_ft import ModelFT

class _ReflectionSubset:
    """
    Lightweight view of one reflection subset (``work`` / ``free`` /
    ``validation`` / ``all``) of a :class:`ReflectionData`.

    The first three apply the validity masks; ``all`` deliberately does not (see
    :attr:`ReflectionData.all`).

    Returns the **compact** (indexed) subset for any per-reflection field and
    caches the integer index map on the parent (rebuilt when the masks or the
    set selection change). Use :meth:`select` to align a full-size,
    model-computed array (e.g. ``F_calc``) to the same subset::

        F_obs  = data.work.F     # corrected amplitudes, work set only
        F_calc = data.work.select(scaler(data.structure_factors(model)))
    """

    __slots__ = ("_parent", "_kind")

    def __init__(self, parent: "ReflectionData", kind: str):
        self._parent = parent
        self._kind = kind

    # -- index / mask -----------------------------------------------------
    @property
    def kind(self) -> str:
        """Which subset this view is: ``work``/``free``/``validation``/``all``.

        Public so a caller keying a cache on "which reflections is this" has a
        stable label. Length alone does not distinguish the views, and
        ``indices.data_ptr()`` can be recycled after a rebuild.
        """
        return self._kind

    @property
    def indices(self) -> torch.Tensor:
        """Cached ``LongTensor`` of positions (into the full array) in this subset."""
        return self._parent._subset_indices(self._kind)

    @property
    def mask(self) -> torch.Tensor:
        """Full-size boolean mask for this subset (``valid & selection``)."""
        n = len(self._parent.hkl)
        m = torch.zeros(n, dtype=torch.bool, device=self._parent.device)
        m[self.indices] = True
        return m

    def select(self, t: torch.Tensor) -> torch.Tensor:
        """Index a full-size (per-reflection) tensor down to this subset."""
        return t.index_select(0, self.indices)

    def __len__(self) -> int:
        return int(self.indices.numel())

    @property
    def n(self) -> int:
        return int(self.indices.numel())

    # Subset reads dispatch through the parent observation attributes.
    @property
    def F(self) -> torch.Tensor:
        return self._parent.F.index_select(0, self.indices)

    @property
    def sigF(self) -> torch.Tensor:
        return self._parent.F_sigma.index_select(0, self.indices)

    # -- raw (uncorrected) amplitudes -------------------------------------
    @property
    def F_raw(self) -> torch.Tensor:
        return self._parent.F_raw.index_select(0, self.indices)

    @property
    def sigF_raw(self) -> torch.Tensor:
        return self._parent.F_sigma_raw.index_select(0, self.indices)

    # -- common aliases ---------------------------------------------------
    @property
    def hkl(self) -> torch.Tensor:
        return self._parent.hkl.index_select(0, self.indices)

    @property
    def rfree(self) -> torch.Tensor:
        return self._parent.rfree_flags.index_select(0, self.indices)

    # -- intensities, corrected to match F/sigF above -----------------------
    @property
    def I(self) -> torch.Tensor:  # noqa: E743 - crystallographic name
        """Scaled intensities, or None when this dataset carries no intensities.

        Corrected, like :attr:`F` -- both the anisotropy factor and the overall scale
        enter squared. Use :attr:`I_raw` for the unscaled values.
        """
        I_corr = self._parent.I
        return I_corr.index_select(0, self.indices) if I_corr is not None else None

    @property
    def sigI(self):
        """Scaled intensity sigmas, or None. See :attr:`I`."""
        sig_corr = self._parent.I_sigma
        return sig_corr.index_select(0, self.indices) if sig_corr is not None else None

    @property
    def I_raw(self):
        """Unscaled intensities, or None."""
        i = self._parent.I_raw
        return i.index_select(0, self.indices) if i is not None else None

    @property
    def sigI_raw(self):
        """Unscaled intensity sigmas, or None."""
        si = self._parent.I_sigma_raw
        return si.index_select(0, self.indices) if si is not None else None

    @property
    def centric(self):
        c = self._parent.centric
        return c.index_select(0, self.indices) if c is not None else None

    # -- generic per-reflection field forwarding --------------------------
    def __getattr__(self, name: str):
        # Only reached for names not found via __slots__/properties above.
        parent = object.__getattribute__(self, "_parent")
        val = getattr(parent, name, None)
        if (
            isinstance(val, torch.Tensor)
            and val.dim() >= 1
            and val.shape[0] == len(parent.hkl)
        ):
            idx = parent._subset_indices(object.__getattribute__(self, "_kind"))
            return val.index_select(0, idx)
        raise AttributeError(
            f"{type(parent).__name__} subset has no per-reflection field {name!r}"
        )

    def __repr__(self) -> str:
        return f"_ReflectionSubset(kind={self._kind!r}, n={self.n})"


@dataclass
class ReflectionData(CrystalDataset, DebugMixin):
    """
    Container for crystallographic reflection data.

    Loads and holds Miller indices, amplitudes, intensities and R-free flags as
    PyTorch tensors, all on one device.

    Parameters
    ----------
    verbose : int, optional
        Verbosity level for logging (0=silent, 1=normal, 2=debug). Default is 1.
    device : str, optional
        Device to store tensors on. Defaults to ``get_default_device()``.

    Attributes
    ----------
    hkl : torch.Tensor
        Miller indices of shape (N, 3), dtype int32.
    F, F_sigma : torch.Tensor
        Amplitudes and their uncertainties, shape (N,), dtype float32.
    I, I_sigma : torch.Tensor
        Intensities and their uncertainties, shape (N,), dtype float32.
    rfree_flags : torch.Tensor
        Test-set flags of shape (N,), convention **1=work, 0=free**. Dtype is
        int32 when generated but bool when read from an MTZ FreeR column, so
        never assume one; internal accessors coerce to bool.
    cell : torch.Tensor
        Unit cell parameters [a, b, c, alpha, beta, gamma].
    spacegroup : str
        Annotated ``str``, but ``load`` / ``from_tensors`` store a
        ``torchref.symmetry.SpaceGroup`` object here.
    resolution : torch.Tensor
        Resolution per reflection in Ångströms of shape (N,).
    """

    # Additional fields specific to ReflectionData (beyond CrystalDataset)
    # Note: Most fields are inherited from CrystalDataset dataclass

    # Provenance: the dataset this one was derived from, and the operation.
    source: Optional["ReflectionData"] = field(default=None, repr=False)
    last_op: Optional[str] = field(default=None, repr=False)

    def __post_init__(self):
        """
        Initialize non-dataclass attributes after dataclass init.

        This is called automatically after the dataclass __init__.
        """
        # Call parent __post_init__ to initialize masks
        super().__post_init__()
        # Subset membership is cached independently of observation values.
        self._subset_cache = {
            "work": None,
            "free": None,
            "validation": None,
            "all": None,
        }
        self._subset_fp = None

    # ===================== work / free / validation =====================

    @property
    def work(self) -> "_ReflectionSubset":
        """Working-set view (``rfree_flags != 0``, excluding validation)."""
        return _ReflectionSubset(self, "work")

    @property
    def free(self) -> "_ReflectionSubset":
        """Free/test-set view (``rfree_flags == 0``, excluding validation)."""
        return _ReflectionSubset(self, "free")

    @property
    def validation(self) -> "_ReflectionSubset":
        """Validation-set view (``validation_flags``). Empty unless populated."""
        return _ReflectionSubset(self, "validation")

    @property
    def all(self) -> "_ReflectionSubset":
        """Every reflection, in storage order, **ignoring the masks entirely**.

        The odd one out: ``work``/``free``/``validation`` are all intersected with
        ``masks()``, this one is not. It exists for diagnostics that need a value for
        every reflection -- per-reflection residuals above all -- and for anything
        aligned to ``hkl``, where storage order is what makes the result addressable
        at all.

        So it is **not** a drop-in for the other three: reflections here may have
        been zeroed by ``sanitize_F`` or excluded by any other mask, and feeding them
        to a loss would undo the masking. ``sub.select`` is a full copy rather than
        the identity, so the view behaves identically to the others.
        """
        return _ReflectionSubset(self, "all")

    def _subset_fingerprint(self):
        """Fingerprint of everything the subset index maps depend on:
        reflection count, the rfree/validation selection, the combined
        validity masks, and device. Rebuilds the index cache when any change.
        """

        def _tv(t):
            return (t.data_ptr(), t._version) if isinstance(t, torch.Tensor) else None

        masks = getattr(self, "masks", None)
        mask_fp = (
            tuple(sorted((k, _tv(v)) for k, v in masks.items()))
            if masks is not None
            else None
        )
        n = 0 if self.hkl is None else len(self.hkl)
        return (
            n,
            _tv(self.rfree_flags),
            _tv(self.validation_flags),
            mask_fp,
            str(self.device),
        )

    def _subset_indices(self, kind: str) -> torch.Tensor:
        """Return cached integer indices for ``kind`` in {work, free, validation, all}.

        The first three are disjoint: validation is carved out of both work
        and free, and each is intersected with the validity masks. ``all`` is
        every reflection, masks included -- see the :attr:`all` docstring for why
        that asymmetry is deliberate. Rebuilt whenever
        :meth:`_subset_fingerprint` changes.
        """
        fp = self._subset_fingerprint()
        if self._subset_fp != fp or self._subset_cache.get(kind) is None:
            n = 0 if self.hkl is None else len(self.hkl)
            device = self.device
            if n == 0:
                empty = torch.empty(0, dtype=torch.long, device=device)  # dtype-ok: empty index tensor; PyTorch requires int64 for indexing
                self._subset_cache = {
                    "work": empty,
                    "free": empty,
                    "validation": empty,
                    "all": empty,
                }
                self._subset_fp = fp
                return self._subset_cache[kind]

            valid = self.masks().to(torch.bool)
            if self.validation_flags is not None:
                val_sel = valid & self.validation_flags.to(torch.bool)
            else:
                val_sel = torch.zeros(n, dtype=torch.bool, device=device)
            not_val = ~val_sel
            if self.rfree_flags is not None:
                rwork = self.rfree_flags.to(torch.bool)
            else:
                rwork = torch.ones(n, dtype=torch.bool, device=device)
            work_sel = valid & rwork & not_val
            free_sel = valid & (~rwork) & not_val
            self._subset_cache = {
                "work": torch.nonzero(work_sel, as_tuple=False).squeeze(-1),
                "free": torch.nonzero(free_sel, as_tuple=False).squeeze(-1),
                "validation": torch.nonzero(val_sel, as_tuple=False).squeeze(-1),
                # Mask-independent by construction; rebuilt with the rest only
                # because the four share one fingerprint.
                "all": torch.arange(n, device=device),
            }
            self._subset_fp = fp
        return self._subset_cache[kind]

    @property
    def F_raw(self) -> Optional[torch.Tensor]:
        """Measured amplitudes, shape (N,), in the input amplitude units."""
        return self.F

    @property
    def F_sigma_raw(self) -> Optional[torch.Tensor]:
        """Measured amplitude uncertainties, shape (N,), in amplitude units."""
        return self.F_sigma

    @property
    def I_raw(self) -> Optional[torch.Tensor]:
        """Measured intensities, shape (N,), in the input intensity units."""
        return self.I

    @property
    def I_sigma_raw(self) -> Optional[torch.Tensor]:
        """Measured intensity uncertainties, shape (N,), in intensity units."""
        return self.I_sigma

    # ===================== per-reflection field reindexing =====================
    #
    # Any routine that changes the HKL set (expand onto a reference grid, remap,
    # symmetry reduction) must carry EVERY per-reflection field along, not a
    # hand-maintained subset. The single source of truth below is shared by all
    # of them.

    # Fill value used for MISSING output rows (index == -1) per field. Missing
    # rows are always masked out downstream (``hkl_present`` / ``missing``), so
    # these fills only need to be shape-correct and non-poisonous. Fields not
    # listed default to 0. NOTE: ``hkl_anomalous`` is special-cased (filled with
    # the reference HKL row, never 0 which would be a spurious Miller index) and
    # the derived-from-HKL fields below are recomputed, never gathered.
    _REINDEX_FILL = {
        "F": 0.0,
        "I": 0.0,
        "phase": 0.0,
        "fom": 0.0,
        "F_sigma": 1.0,
        "I_sigma": 1.0,
        "rfree_flags": 1,  # missing reflections default to the work set
        "friedel_flags": False,
        "validation_flags": False,
    }

    # Per-reflection fields that are pure functions of (hkl, cell, spacegroup):
    # never gathered/aggregated, always recomputed or invalidated after an HKL
    # change (``resolution`` recomputed; ``_centric_flags`` lazily rebuilt by
    # the ``centric`` property).
    _REINDEX_DERIVED = ("resolution", "_centric_flags")

    def _per_row_fields(self):
        """Yield ``(name, tensor)`` for each per-reflection dataclass field.

        Enumerated generically (``shape[0] == len(hkl)``) so a new per-reflection
        field is carried by every reindexing operation without being listed.
        """
        n = len(self.hkl) if self.hkl is not None else 0
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, torch.Tensor) and val.shape and val.shape[0] == n:
                yield f.name, val

    @staticmethod
    def _gather_rows(val: torch.Tensor, index: torch.Tensor, fill) -> torch.Tensor:
        """``val[index]``, with rows where ``index == -1`` set to ``fill``."""
        present = index >= 0
        if bool(present.all()):
            return val[index]
        shape = (len(index),) + tuple(val.shape[1:])
        out = torch.full(shape, fill, dtype=val.dtype, device=val.device)
        out[present] = val[index[present]]
        return out

    def _gathered_masks(self, index: torch.Tensor) -> TensorMasks:
        """Every mask gathered by ``index``; rows with ``index == -1`` are masked out.

        Call before ``hkl`` changes length: masks of any other length are dropped.
        """
        n = len(self.hkl) if self.hkl is not None else 0
        out = TensorMasks(device=self.device)
        for name, mask in self.masks.items():
            if mask is not None and len(mask) == n:
                out[name] = self._gather_rows(mask, index, False)
        return out

    def _replace_masks(self, new: TensorMasks) -> None:
        """Swap in ``new``'s masks, keeping the existing ``TensorMasks`` object."""
        self.masks.clear()
        for name, mask in new.items():
            self.masks[name] = mask

    def _reindex_per_reflection(
        self,
        index_map: torch.Tensor,
        new_hkl: torch.Tensor,
        target: Optional["ReflectionData"] = None,
    ) -> torch.Tensor:
        """Gather every per-reflection field from ``self`` onto ``new_hkl``.

        Enumerates dataclass fields generically (same ``shape[0] == n`` guard as
        :meth:`__select__`) so future per-reflection fields are carried too.

        Parameters
        ----------
        index_map : torch.Tensor
            LongTensor of length ``len(new_hkl)`` mapping each output row to a
            source row of ``self``, or ``-1`` when absent (filled per
            :attr:`_REINDEX_FILL`).
        new_hkl : torch.Tensor
            Miller indices for the reindexed dataset, shape ``(M, 3)``.
        target : ReflectionData, optional
            Where to write; defaults to ``self`` (in-place). A fresh instance
            must already have ``cell``/``spacegroup`` set so resolution can be
            recomputed.

        Returns
        -------
        torch.Tensor
            Boolean presence mask (``index_map >= 0``), for building the
            caller's ``hkl_present`` / ``missing`` masks.
        """
        if target is None:
            target = self

        new_hkl = new_hkl.to(dtype=dtypes.int, device=self.device)
        index_map = index_map.to(device=self.device, dtype=torch.long)  # dtype-ok: index map used for indexing/gather; PyTorch requires int64
        present = index_map >= 0

        skip = {"hkl", *self._REINDEX_DERIVED}
        # Collected first: writing into self (the in-place case) changes the
        # row count _per_row_fields keys on.
        gathered = {
            name: self._gather_rows(val, index_map, self._REINDEX_FILL.get(name, 0))
            for name, val in self._per_row_fields()
            if name not in skip
        }
        if "hkl_anomalous" in gathered:
            # Missing rows fall back to the reference HKL, never a 0,0,0 row.
            gathered["hkl_anomalous"][~present] = new_hkl[~present]
        for name, val in gathered.items():
            setattr(target, name, val)

        # Install the new HKL and recompute / invalidate derived-from-HKL fields.
        target.hkl = new_hkl
        target._centric_flags = None
        if target.cell is not None:
            target._calculate_resolution()
        else:
            target.resolution = None
        return present

    def _assert_per_reflection_consistent(self) -> None:
        """Invariant: every per-reflection tensor matches ``len(self.hkl)``.

        Post-condition for the reindex routines; raises rather than letting a
        stale-length field surface as a downstream shape mismatch.
        """
        n = len(self.hkl) if self.hkl is not None else 0
        bad = []
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, torch.Tensor) and val.ndim >= 1 and val.shape[0] != n:
                bad.append((f.name, tuple(val.shape)))
        if bad:
            raise RuntimeError(
                f"ReflectionData per-reflection length mismatch (n_hkl={n}): {bad}"
            )

    def _canonicalize_in_place(self) -> None:
        """Remap HKL to canonical CCP4 ASU form and reorder all data in-place."""
        if self.hkl is None or self.spacegroup is None:
            return

        canonical_hkl, phase_shifts, friedel_flags, sort_indices = (
            self.spacegroup.canonicalize_hkl(
                self.hkl, include_friedel=True, device=self.device
            )
        )

        masks = self._gathered_masks(sort_indices)
        for name, val in list(self._per_row_fields()):
            setattr(self, name, val[sort_indices])
        self._replace_masks(masks)
        self.hkl = canonical_hkl

        if self.phase is not None:
            self.phase = (
                torch.where(friedel_flags, -self.phase, self.phase) + phase_shifts
            )

        if self.cell is not None:
            self._calculate_resolution()

        # friedel_flags comes back already in sorted (canonical) order, matching
        # self.hkl. hkl_anomalous carries the SIGNED index used for
        # structure-factor evaluation -- canonical for the (+) member, negated
        # for the conjugated (-) mate -- so the model yields a genuine Bijvoet
        # difference. See ReflectionData.structure_factors.
        self.friedel_flags = friedel_flags
        self.hkl_anomalous = torch.where(
            friedel_flags.unsqueeze(-1), -self.hkl, self.hkl
        )

        n_flipped = int(friedel_flags.sum())
        if n_flipped:
            print(
                f"  Reindexed {n_flipped}/{len(self.hkl)} reflections to the "
                f"CCP4 ASU (output is written on that index, not the input one)."
            )
            # A row needing conjugation to reach the ASU does NOT by itself mean
            # the data are Bijvoet-unmerged: a merged dataset indexed in another
            # convention flags rows while carrying no mate at all. Real mates
            # show up as one canonical index holding both a flagged and an
            # unflagged row. Either this or the reader's (+)/(-) columns
            # declaring unmerged is enough.
            inverse, n_groups = self.asu_group_indices()
            has_mates = bool(
                (
                    self._group_any(friedel_flags, inverse, n_groups)
                    & self._group_any(~friedel_flags, inverse, n_groups)
                ).any()
            )
            if has_mates:
                self.friedel_merged = False

    def _hkl_for_sf(self) -> torch.Tensor:
        """Signed Miller indices for structure-factor evaluation.

        Returns ``hkl_anomalous`` when present so the two members of a Bijvoet
        pair (which share a canonical ASU index in :attr:`hkl`) are evaluated at
        their true ``+h``/``-h`` positions and therefore get distinct
        ``|F_calc|`` under anomalous scattering. Falls back to the canonical
        :attr:`hkl` when anomalous bookkeeping is unavailable (e.g. empty init).

        These indices are a model *input* only. A structure factor evaluated
        here is in the signed convention, and pairing its phase with :attr:`hkl`
        negates that phase wherever :attr:`friedel_flags` is set. Convert with
        :meth:`conjugate_friedel` first, or use :meth:`structure_factors`, which
        does both and is the supported entry point.

        Returns
        -------
        torch.Tensor
            Miller indices of shape (N, 3), row-aligned with :attr:`hkl`.
        """
        if self.hkl_anomalous is not None:
            return self.hkl_anomalous
        return self.hkl

    def conjugate_friedel(self, fcalc: torch.Tensor) -> torch.Tensor:
        """Move complex structure factors between the signed and canonical index.

        Rows flagged in :attr:`friedel_flags` are evaluated at ``-h`` by
        :meth:`_hkl_for_sf` while :attr:`hkl` holds ``+h``; ``F(-h)`` is the
        conjugate of ``F(h)`` up to the anomalous ``f''`` term. Conjugating
        exactly those rows re-expresses the array on the other index. The
        operation is its own inverse, so it converts in both directions.

        Amplitudes are unaffected -- only phases move.

        Parameters
        ----------
        fcalc : torch.Tensor
            Complex structure factors of shape (N,).

        Returns
        -------
        torch.Tensor
            ``fcalc`` with the Friedel-flagged rows conjugated, or unchanged
            when no Friedel bookkeeping is present.
        """
        if self.friedel_flags is None:
            return fcalc
        return torch.where(self.friedel_flags, fcalc.conj(), fcalc)

    def structure_factors(
        self, model, recalc: bool = False, cached: bool = True
    ) -> torch.Tensor:
        """Complex ``F_calc`` from ``model``, on the canonical ASU index.

        Evaluates the model at the signed indices so Bijvoet mates get distinct
        ``|F_calc|``, then returns the result on :attr:`hkl` -- the index this
        dataset writes as ``H,K,L``. Structure factors are in this convention
        everywhere in TorchRef; the signed index does not escape this method.

        Parameters
        ----------
        model : ModelFT
            Model to evaluate.
        recalc : bool, optional
            Force recomputation rather than reusing the cached SF. Default False.
        cached : bool, optional
            True (default) goes through the model's forward cache. False calls
            ``model.forward`` directly, leaving the cache untouched.

        Returns
        -------
        torch.Tensor
            Complex structure factors of shape (N,), row-aligned with
            :attr:`hkl`.
        """
        hkl = self._hkl_for_sf()
        fcalc = model(hkl, recalc=recalc) if cached else model.forward(hkl)
        return self.conjugate_friedel(fcalc)

    def asu_group_indices(self) -> Tuple[torch.Tensor, int]:
        """Group rows that describe the same unique reflection.

        After canonicalization :attr:`hkl` holds CCP4-ASU indices and may
        contain duplicate rows: the two members of a Bijvoet pair share one
        canonical index and are distinguished only by :attr:`friedel_flags`
        (and the signed :attr:`hkl_anomalous`). Symmetry-equivalent rows that
        survive merging collapse the same way.

        Anything that must treat such rows as a *single* observation has to
        group by canonical index rather than by row -- the work/free partition
        above all, since splitting a Bijvoet pair across the two sets leaks the
        held-out reflection into the work set.

        Returns
        -------
        group_id : torch.Tensor
            Shape (N,), int64, on :attr:`device`. Rows sharing a canonical ASU
            index share a value in ``[0, n_groups)``.
        n_groups : int
            Number of distinct canonical reflections.

        Raises
        ------
        RuntimeError
            If :attr:`hkl` is missing, or the data have not been canonicalized.
            Grouping raw indices would silently fail to unite ``+h`` with
            ``-h``, which is precisely the case this exists to handle.
        """
        if self.hkl is None:
            raise RuntimeError("No hkl present; cannot group reflections.")
        if self.friedel_flags is None:
            raise RuntimeError(
                "asu_group_indices requires canonicalized data (friedel_flags "
                "is None). Load via load_mtz/from_tensors, or run "
                "_canonicalize_in_place first -- grouping raw Miller indices "
                "would not unite Bijvoet mates."
            )
        # On CPU: torch.unique(dim=0) is not reliably supported across
        # accelerator backends, and this runs once per load on integer data.
        uniq, inverse = torch.unique(self.hkl.cpu(), dim=0, return_inverse=True)
        return inverse.to(self.device), int(uniq.shape[0])

    def bijvoet_mean(
        self, values: torch.Tensor, valid: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Replace each row's value by the mean over the valid rows of its Bijvoet pair.

        For consumers that want one value per reflection -- a Hermitian map
        puts each amplitude at ``h`` and its conjugate at ``-h``, so feeding it
        both mates would count every measured pair twice. Merged data are
        returned unchanged.

        Parameters
        ----------
        values : torch.Tensor
            Per-row real values of shape (N,), e.g. amplitudes or differences.
        valid : torch.Tensor, optional
            Boolean (N,), rows allowed to contribute. Defaults to ``masks()``.

        Returns
        -------
        torch.Tensor
            Shape (N,). Rows of a pair with no valid member keep their own value.
        """
        if self.friedel_merged:
            return values
        if valid is None:
            valid = self.masks()
        if valid is None:
            valid = torch.ones_like(values, dtype=torch.bool)
        group_id, n_groups = self.asu_group_indices()
        w = valid.to(values.dtype)
        total = torch.zeros(n_groups, dtype=values.dtype, device=values.device)
        total = total.index_add(0, group_id, torch.where(valid, values, 0.0))
        count = torch.zeros_like(total).index_add(0, group_id, w)
        mean = (total / count.clamp(min=1))[group_id]
        return torch.where(count[group_id] > 0, mean, values)

    def bijvoet_representatives(
        self, valid: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """One row index per unique reflection, in row order.

        Pairs with :meth:`bijvoet_mean` to build a Friedel-averaged reflection
        list from anomalous data. For merged data every row is its own
        representative.

        Parameters
        ----------
        valid : torch.Tensor, optional
            Boolean (N,). If given, only reflections with at least one valid row
            are represented.

        Returns
        -------
        torch.Tensor
            Row indices, int64, ascending.
        """
        if self.friedel_merged:
            if valid is None:
                return torch.arange(len(self.hkl), device=self.device)
            return torch.nonzero(valid).squeeze(-1)
        group_id, n_groups = self.asu_group_indices()
        rows = self._group_representative_rows(group_id, n_groups)
        if valid is not None:
            rows = rows[self._group_any(valid, group_id, n_groups)]
        return torch.sort(rows).values

    @staticmethod
    def _group_any(
        mask: torch.Tensor, group_id: torch.Tensor, n_groups: int
    ) -> torch.Tensor:
        """True for each ASU group with at least one row set in ``mask``.

        Uses ``index_add_`` on float rather than ``scatter_reduce_(amax)``: the
        latter raises "not supported for torch.int64" on the MPS backend.
        """
        # dtype-ok: float32 count accumulator for the int64-scatter MPS workaround
        # above; the result is reduced to bool (> 0), so precision is irrelevant.
        counts = torch.zeros(n_groups, dtype=torch.float32, device=mask.device)
        counts.index_add_(0, group_id, mask.to(torch.float32))  # dtype-ok: float32 counter for the MPS workaround above; reduced to bool
        return counts > 0

    @staticmethod
    def _group_representative_rows(
        group_id: torch.Tensor, n_groups: int
    ) -> torch.Tensor:
        """One row index per ASU group, ordered by group id.

        The lowest-numbered row of each group, via a stable sort. Used for
        per-group quantities that are constant within a group -- resolution and
        therefore the resolution bin, since every row in a group shares a
        canonical Miller index. Taking a single representative also pins a group
        that straddles a bin edge (``get_bins`` cuts on sorted position, so rows
        at identical resolution can fall either side) into exactly one bin.
        """
        order = torch.argsort(group_id, stable=True)
        sorted_gid = group_id[order]
        first = torch.ones_like(sorted_gid, dtype=torch.bool)
        first[1:] = sorted_gid[1:] != sorted_gid[:-1]
        return order[first]

    def load(self, reader, french_wilson: bool = True):
        """
        Load reflection data using a data reader.

        Parameters
        ----------
        reader : callable
            Data reader object that returns (data_dict, cell, spacegroup) when called.
            Can be MTZ, ReflectionCIFReader, or other compatible reader.
        french_wilson : bool, optional
            Whether to derive amplitudes from intensities via French-Wilson.
            Default True. When False, existing amplitude columns (``F``/``SIGF``)
            are used directly and the French-Wilson step is skipped -- use this
            when the input amplitudes are already French-Wilson corrected. If the
            data contain only intensities, French-Wilson is applied regardless.

        Returns
        -------
        ReflectionData
            Self, for method chaining.

        Raises
        ------
        ValueError
            If unit cell parameters are missing or no amplitude/intensity data found.
        """

        data_dict, cell, spacegroup = reader()

        # Merge state from the reader: False when anomalous F(+)/F(-) (or I(+)/I(-))
        # were loaded as explicit signed-HKL Bijvoet pairs. Canonicalization below
        # then sets friedel_flags / hkl_anomalous accordingly.
        self.friedel_merged = bool(data_dict.get("friedel_merged", True))

        hkl = torch.tensor(
            data_dict["HKL"], dtype=dtypes.int, device=self.device, requires_grad=False
        )

        self.hkl = hkl

        if cell is not None:
            self.cell = Cell(cell, dtype=dtypes.float, device=self.device)
        else:
            raise ValueError(
                "Unit cell parameters are required in the data and could not be read."
            )

        if spacegroup is not None:
            # On the dataset's device, not the process default -- otherwise the
            # symmetry matrices can land elsewhere than every other tensor here
            # and nothing notices until an op mixes the two.
            self.spacegroup = SpaceGroup(spacegroup, device=self.device)
        self._calculate_resolution()

        use_intensities = "I" in data_dict and (french_wilson or "F" not in data_dict)
        if "I" in data_dict and not use_intensities and self.verbose:
            print(
                "French-Wilson disabled; using existing F/SIGF columns directly "
                "(ignoring intensity columns).",
                flush=True,
            )

        if use_intensities:
            self.I = torch.tensor(
                data_dict["I"],
                dtype=dtypes.float,
                device=self.device,
                requires_grad=False,
            )
            if "SIGI" in data_dict:
                self.I_sigma = torch.tensor(
                    data_dict["SIGI"],
                    dtype=dtypes.float,
                    device=self.device,
                    requires_grad=False,
                )
            self.intensity_source = data_dict.get("I_col", "Unknown")
            self.F, self.F_sigma, fw_keep = french_wilson_auto(
                self.I,
                self.I_sigma,
                self.hkl,
                self.resolution,
                self.spacegroup or "P1",
            )
            # Record French-Wilson's own input criterion, evaluated on the true
            # intensities. This is strictly better than anything reconstructible
            # from the amplitudes afterwards: F is a positive posterior mean, so
            # it no longer knows which intensities were inexplicably negative.
            # Set here rather than recomputed in _post_load_cleanup so the mask
            # is exactly the one French-Wilson applied; _canonicalize_in_place
            # reorders masks along with everything else. Kept separate from the
            # outlier mask -- this one guards the posterior integral against
            # unphysical input, which is a different question from whether an
            # observation is an outlier.
            self._set_french_wilson_mask(fw_keep)
        elif "F" in data_dict:
            self.F = torch.tensor(
                data_dict["F"],
                dtype=dtypes.float,
                device=self.device,
                requires_grad=False,
            )
            if "SIGF" in data_dict:
                if data_dict["SIGF"] is not None:
                    self.F_sigma = torch.tensor(
                        data_dict["SIGF"],
                        dtype=dtypes.float,
                        device=self.device,
                        requires_grad=False,
                    )
                else:
                    sigF = math_torch.estimate_sigma_F(self.F)
                    self.F_sigma = sigF
            else:
                sigF = math_torch.estimate_sigma_F(self.F)
                self.F_sigma = sigF
            self.amplitude_source = data_dict.get("F_col", "Unknown")

        else:
            raise ValueError("No amplitude or intensity data found in MTZ file")

        if "R-free-flags" in data_dict:
            rfree = torch.tensor(
                data_dict["R-free-flags"], device=self.device, requires_grad=False
            )
            flagged = rfree < 0
            rfree = rfree.clip(min=0, max=1).to(torch.bool)
            self.rfree_flags = rfree
            self.masks["flagged_initial"] = ~flagged
            # Record the provenance for every file-sourced set, not only the
            # ones that also carry a validation column: a header reporting
            # R-free has to be able to say which test set produced it. Named
            # after the reader rather than hardcoded "MTZ", since `load` also
            # takes ReflectionCIFReader and any other compatible reader.
            reader_name = type(reader).__name__
            self.rfree_source = f"{reader_name} FreeR"
            # A third (validation) column goes into the separate boolean
            # ``validation_flags``; ``rfree_flags`` stays binary work/free.
            if "Validation-flags" in data_dict:
                self.validation_flags = torch.tensor(
                    data_dict["Validation-flags"],
                    device=self.device,
                    requires_grad=False,
                ).to(torch.bool)
                self.rfree_source = f"{reader_name} FreeR+Validation"

        self._post_load_cleanup()

        # Generate only after canonicalization: the free set must be drawn on
        # unique ASU reflections, and the Bijvoet grouping that requires does
        # not exist until _canonicalize_in_place has run. See
        # asu_group_indices / generate_rfree_flags.
        if self.rfree_flags is None:
            self.generate_rfree_flags()

        return self

    def _post_load_cleanup(self) -> "ReflectionData":
        """Resolution, all-valid mask, ASU canonicalization, F sanitation and
        Wilson-probability outlier flagging; returns ``self``. Run by ``load`` /
        ``from_tensors``.
        """
        if self.resolution is None:
            self._calculate_resolution()

        if "flagged_initial" not in self.masks:
            self.masks["flagged_initial"] = torch.ones(
                len(self.hkl), dtype=torch.bool, device=self.device
            )

        self._canonicalize_in_place()
        self.sanitize_F()
        # No-op when ``load`` already installed French-Wilson's own mask from the
        # true intensities; this covers the amplitude-only path.
        self.flag_wilson_outliers()
        return self

    @classmethod
    def from_tensors(
        cls,
        hkl: torch.Tensor,
        F: torch.Tensor,
        F_sigma: torch.Tensor,
        cell: "Cell",
        spacegroup: "SpaceGroup",
        rfree_flags: Optional[torch.Tensor] = None,
        device=None,
        verbose: int = 1,
        friedel_merged: Optional[bool] = None,
        detach: bool = True,
        I: Optional[torch.Tensor] = None,
        I_sigma: Optional[torch.Tensor] = None,
        validation_flags: Optional[torch.Tensor] = None,
    ) -> "ReflectionData":
        """
        Construct ReflectionData directly from tensors.

        Every per-reflection tensor must be row-aligned with ``hkl`` as passed.
        Canonicalization then reorders all of them together, so pass them here
        rather than assigning them to the returned object.

        Parameters
        ----------
        hkl : torch.Tensor
            Miller indices of shape (N, 3).
        F : torch.Tensor
            Structure factor amplitudes of shape (N,).
        F_sigma : torch.Tensor
            Amplitude uncertainties of shape (N,).
        cell : Cell
            Unit cell parameters.
        spacegroup : SpaceGroup
            Space group.
        rfree_flags : torch.Tensor, optional
            Flags of shape (N,), convention 1=work, 0=free. If None, generated
            (2% free) as int32; the stored dtype is not guaranteed bool.
        device : str, optional
            Device for tensors. Defaults to the device of ``hkl``.
        verbose : int, optional
            Verbosity level. Default is 1.
        friedel_merged : bool, optional
            False means ``hkl`` holds explicit Bijvoet pairs (both ``+h`` and
            ``-h``), which enables the model's f'' term downstream. Default
            True. Canonicalization overrides it to False when it finds real
            mates, so passing True is a statement about the input rather than
            the last word.
        detach : bool, optional
            True (default) stores constant observations, dropping the caller's
            autograd graph; False keeps it so gradients reach whatever produced
            ``F``/``F_sigma``.
        I, I_sigma : torch.Tensor, optional
            Intensities and their uncertainties of shape (N,). Stored as given;
            ``F`` is not derived from them.
        validation_flags : torch.Tensor, optional
            Boolean validation-set flags of shape (N,).

        Returns
        -------
        ReflectionData
            Fully initialized reflection data with all cleanup applied.
        """
        # Follow the incoming tensors when no device is given, rather than
        # the global default -- otherwise building from accelerator-resident
        # arrays on a CPU-default host silently round-trips every one of them
        # through host memory. ``hkl`` is a bare tensor, so read its device
        # directly; ``resolve_device`` is for objects it can move in place.
        if device is None and isinstance(hkl, torch.Tensor):
            device = hkl.device
        data = cls(device=normalize_device(device), verbose=verbose)

        def _prep(t: torch.Tensor) -> torch.Tensor:
            # Detach (constant data) unless the caller wants the graph preserved.
            return t.detach() if detach else t

        data.hkl = _prep(hkl).to(device=data.device)
        data.F = _prep(F).to(device=data.device)
        data.F_sigma = _prep(F_sigma).to(device=data.device)
        data.cell = (
            cell.to(device=data.device)
            if hasattr(cell, "to")
            else Cell(cell, device=data.device)
        )
        data.spacegroup = (
            spacegroup
            if isinstance(spacegroup, SpaceGroup)
            else SpaceGroup(spacegroup, device=data.device)
        )

        if rfree_flags is not None:
            data.rfree_flags = _prep(rfree_flags).to(
                device=data.device, dtype=torch.bool
            )
        if I is not None:
            data.I = _prep(I).to(device=data.device)
        if I_sigma is not None:
            data.I_sigma = _prep(I_sigma).to(device=data.device)
        if validation_flags is not None:
            data.validation_flags = _prep(validation_flags).to(
                device=data.device, dtype=torch.bool
            )

        # Set before canonicalization, which is what detects real Bijvoet mates
        # and downgrades this to False -- the same seam load() goes through.
        data.friedel_merged = True if friedel_merged is None else bool(friedel_merged)

        data._post_load_cleanup()

        # As in load(): generate after canonicalization so the draw can group
        # Bijvoet mates onto a shared canonical index.
        if data.rfree_flags is None:
            data.generate_rfree_flags()

        return data

    def load_mtz(
        self,
        path: Union[str, Path],
        column_names: Optional[dict] = None,
        french_wilson: bool = True,
        anomalous: Optional[bool] = None,
    ) -> "ReflectionData":
        """
        Load reflection data from MTZ file.

        Parameters
        ----------
        path : str
            Path to MTZ file.
        column_names : dict, optional
            Explicit column name mapping to override automatic detection.
            Supported keys: ``"F"``, ``"SIGF"``, ``"I"``, ``"SIGI"``.
            Example: ``{"F": "dFo", "SIGF": "sig_dFo"}``.
        french_wilson : bool, optional
            Whether to derive amplitudes from intensities via French-Wilson.
            Default True. Set False to use existing French-Wilson-corrected
            ``F``/``SIGF`` columns directly when the file also carries
            intensities. See :meth:`load`.
        anomalous : bool, optional
            Anomalous (Bijvoet) handling. If None (default), ``F(+)/F(-)`` (or
            ``I(+)/I(-)``) columns are auto-detected and loaded as explicit
            Friedel pairs when present (anomalous preferred). True forces this;
            False forces a merged load even when anomalous columns are present.

        Returns
        -------
        ReflectionData
            Self, for method chaining.
        """
        # gemmi's readers take a str, so coerce here rather than fail deep
        # inside gemmi with an argument-type error.
        reader = mtz.MTZReader(
            verbose=self.verbose, column_names=column_names, anomalous=anomalous
        ).read(str(path))
        return self.load(reader, french_wilson=french_wilson)

    def load_crystfel_hkl(
        self, path: str, cell, spacegroup,
    ) -> "ReflectionData":
        """
        Load a CrystFEL ``partialator`` ``.hkl`` reflection list.

        Unlike MTZ, the CrystFEL format carries no cell or space-group metadata, so both
        must be supplied by the caller -- they usually live in a ``.cell`` file alongside.

        The format is intensity-native, so amplitudes are derived by French-Wilson on
        load exactly as they are for an MTZ carrying I/SIGI columns.

        Parameters
        ----------
        path : str
            Path to the ``.hkl`` file.
        cell : list | tuple | np.ndarray | Cell | torch.Tensor
            Unit cell (a, b, c, alpha, beta, gamma).
        spacegroup : str | gemmi.SpaceGroup | SpaceGroup
            Space group identifier.

        Returns
        -------
        ReflectionData
            Self, for method chaining.
        """
        from torchref.io import hkl as _hkl

        reader = _hkl.HKLReader(verbose=self.verbose).read(path, cell, spacegroup)
        return self.load(reader)

    def load_cif(
        self,
        path: Union[str, Path],
        data_block: Optional[str] = None,
        anomalous: Optional[bool] = None,
    ) -> "ReflectionData":
        """
        Load reflection data from CIF file.

        Parameters
        ----------
        path : str
            Path to CIF file.
        data_block : str, optional
            Specific data block name to read (e.g., 'r1vlmsf'). If None, reads
            the first data block. Useful for multi-dataset CIF files.
        anomalous : bool, optional
            Anomalous (Bijvoet) handling. If None (default), ``pdbx_F_plus/minus``
            (or ``I``) columns are auto-detected and loaded as explicit Friedel
            pairs when present (anomalous preferred). True forces this; False
            forces a merged load (Bijvoet mates averaged) even when present.

        Returns
        -------
        ReflectionData
            Self, for method chaining.
        """
        reader = cif.ReflectionCIFReader(
            str(path), verbose=self.verbose, data_block=data_block, anomalous=anomalous
        )
        return self.load(reader)

    def generate_rfree_flags(
        self,
        free_fraction: float = 0.02,
        n_bins: int = 10,
        min_per_bin: int = 1000,
        min_free_per_bin: int = 50,
        seed: Optional[int] = None,
        force: bool = False,
    ) -> None:
        """
        Generate R-free flags with resolution-stratified sampling.

        Sets ``rfree_flags`` (int32, 1=work/0=free) and ``rfree_source``.
        ``load`` and ``from_tensors`` call this when the input carries no flags.

        The draw is over *unique ASU reflections*, not rows: the two members of
        a Bijvoet pair share a canonical index (see :meth:`asu_group_indices`)
        and differ only by the anomalous signal, so splitting them across
        work/free would leak the held-out reflection into the work set and bias
        R-free downwards. Both members always land in the same set. For merged
        data every group is a singleton and this is a no-op.

        Only reflections passing the validity masks are drawn from, so the
        counts below describe usable reflections rather than raw rows.

        Parameters
        ----------
        free_fraction : float, optional
            Fraction to mark free. Per bin the count is
            ``max(min_free_per_bin, free_fraction*bin)``, so the realised
            fraction can exceed this on small bins.
        n_bins : int, optional
            Target number of resolution bins. Default is 10.
        min_per_bin : int, optional
            Minimum reflections per bin (default 1000); bins are coarsened below
            ``n_bins`` on small datasets to honour it.
        min_free_per_bin : int, optional
            Minimum free unique reflections per bin, clamped to the number the
            bin holds.
        seed : int, optional
            Seeds the **global** torch and numpy RNGs before the draw, so the
            same seed on the same data reproduces the same set.
        force : bool, optional
            Overwrite existing flags. Default False: existing flags are kept and
            the call only warns.

        Raises
        ------
        ValueError
            If resolution information is not available.
        RuntimeError
            If the data have not been canonicalized (via
            :meth:`asu_group_indices`).
        """
        if self.rfree_flags is not None and not force:
            warnings.warn(
                f"R-free flags already exist ({self.rfree_source}); "
                "pass force=True to overwrite them."
            )
            return
        if self.resolution is None:
            raise ValueError("Resolution information required to generate R-free flags")
        if self.verbose > 0:
            if self.rfree_flags is not None:
                print(f"Overwriting existing R-free flags ({self.rfree_source})")
            print(
                f"Generating R-free flags: {free_fraction*100:.1f}% free, "
                f"{n_bins} bins of >= {min_per_bin}, >= {min_free_per_bin} free per bin"
            )

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        bin_indices, actual_n_bins = self.get_bins(
            n_bins=n_bins, min_per_bin=min_per_bin
        )
        group_id, n_groups = self.asu_group_indices()

        # A group is eligible if any of its rows survives the validity masks;
        # spending the free quota on masked-out rows would silently shrink the
        # usable free set below min_free_per_bin.
        group_valid = self._group_any(self.masks().to(torch.bool), group_id, n_groups)
        if not bool(group_valid.any()):
            warnings.warn(
                "No reflections pass the validity masks; drawing R-free flags "
                "from all reflections instead. The input data are likely bad."
            )
            group_valid = torch.ones_like(group_valid)

        group_bin = bin_indices[self._group_representative_rows(group_id, n_groups)]
        group_free = self._stratified_group_draw(
            group_valid,
            group_bin,
            actual_n_bins,
            lambda n: min(n, max(min_free_per_bin, int(n * free_fraction))),
        )

        flags = torch.ones(len(self.resolution), dtype=dtypes.int, device=self.device)
        flags[group_free[group_id]] = 0
        self.rfree_flags = flags
        # The seed belongs in the provenance string: without it "generated"
        # names a draw nobody can reproduce.
        self.rfree_source = (
            "Generated (resolution-binned, ASU-grouped"
            + (f", seed {seed}" if seed is not None else "")
            + ")"
        )

        if self.verbose > 0:
            n_free = int((flags == 0).sum())
            print(
                f"  {n_free} free ({100.0 * n_free / len(flags):.1f}%) in "
                f"{actual_n_bins} bins, drawn over {int(group_valid.sum())} unique "
                "ASU reflections; Bijvoet mates share a flag"
            )

    @staticmethod
    def _stratified_group_draw(
        eligible: torch.Tensor,
        group_bin: torch.Tensor,
        n_bins: int,
        n_to_draw: Callable[[int], int],
    ) -> torch.Tensor:
        """Draw ``n_to_draw(n)`` of the ``n`` eligible groups in each resolution bin.

        Shared by R-free and validation-set generation so both split whole ASU
        groups the same way. Uses the global torch RNG, one ``randperm`` per
        non-empty bin in bin order, so a seeded caller is reproducible.

        Returns
        -------
        torch.Tensor
            Boolean mask of shape ``(n_groups,)``, True for drawn groups.
        """
        drawn = torch.zeros_like(eligible, dtype=torch.bool)
        for b in range(n_bins):
            members = torch.where((group_bin == b) & eligible)[0]
            n = int(members.numel())
            if n == 0:
                continue
            perm = torch.randperm(n, device=members.device)[: n_to_draw(n)]
            drawn[members[perm]] = True
        return drawn

    def get_bins(
        self, n_bins: int = 20, min_per_bin: int = 100
    ) -> Tuple[torch.Tensor, int]:
        """
        Create resolution bins with approximately equal counts of valid reflections.

        Pure: nothing is stored on the dataset, so callers that need the same
        bins later (e.g. :meth:`mean_res_per_bin`) must keep the returned tensor.

        Parameters
        ----------
        n_bins : int, optional
            Target number of resolution bins. Default is 20.
        min_per_bin : int, optional
            Minimum reflections per bin. Default is 100.

        Returns
        -------
        bin_indices : torch.Tensor
            Tensor of shape (N,) with bin index for each reflection.
        n_bins : int
            Actual number of bins created (may be less than target for small datasets).
        """
        n_refl = len(self.resolution)
        valid_mask = self.masks()
        total_valid = valid_mask.sum().item()

        # Calculate how many bins we can actually create given min_per_bin constraint
        max_possible_bins = max(1, total_valid // min_per_bin)
        actual_n_bins = min(n_bins, max_possible_bins)

        if actual_n_bins < n_bins and self.verbose > 0:
            print(
                f"  Note: Adjusted bins from {n_bins} to {actual_n_bins} (min {min_per_bin} refl/bin)"
            )

        # Sort reflections by resolution
        _, sort_indices = torch.sort(self.resolution)

        # Create bins with approximately equal number of VALID reflections
        bin_indices = torch.zeros(n_refl, dtype=dtypes.int, device=self.device)
        reflections_per_bin = total_valid // actual_n_bins

        # Get the valid mask in sorted order
        valid_mask_sorted = valid_mask[sort_indices]

        # Cumulative sum of valid reflections in sorted order
        cumsum_valid = torch.cumsum(valid_mask_sorted.to(dtypes.int), dim=0)

        # Create bin edges based on cumulative count of valid reflections
        # Each bin should contain approximately reflections_per_bin valid reflections
        bin_edges = [0]
        for bin_idx in range(1, actual_n_bins):
            target_count = bin_idx * reflections_per_bin
            # Find first index where cumsum >= target_count
            edge_candidates = torch.where(cumsum_valid >= target_count)[0]
            if len(edge_candidates) > 0:
                bin_edges.append(edge_candidates[0].item())
        bin_edges.append(n_refl)

        # Assign bin indices to sorted reflections, then map back to original order
        for bin_idx in range(len(bin_edges) - 1):
            start, end = bin_edges[bin_idx], bin_edges[bin_idx + 1]
            bin_indices[sort_indices[start:end]] = bin_idx

        if self.verbose > 1:
            # Print bin statistics
            print("  Resolution bins:")
            for bin_idx in range(min(actual_n_bins, 20)):  # Show all bins (up to 20)
                bin_mask = bin_indices == bin_idx
                if bin_mask.sum() > 0:
                    valid_reflexes = bin_mask & valid_mask
                    bin_res = self.resolution[bin_mask]
                    print(
                        f"    Bin {bin_idx:2d}: {valid_reflexes.sum():6d} valid refl, "
                        f"resolution {bin_res.min():.2f}-{bin_res.max():.2f} Å"
                    )
            if actual_n_bins > 20:
                print(f"    ... ({actual_n_bins - 20} more bins)")
        return bin_indices, actual_n_bins

    def mean_res_per_bin(self, bin_indices: torch.Tensor, n_bins: int) -> torch.Tensor:
        """
        Mean resolution of the valid reflections in each bin.

        Parameters
        ----------
        bin_indices : torch.Tensor
            Bin of each reflection, shape (N,), as returned by :meth:`get_bins`.
        n_bins : int
            Number of bins, as returned by :meth:`get_bins`.

        Returns
        -------
        torch.Tensor
            Mean resolution per bin in Ångströms, shape (n_bins,); 0 for an
            empty bin.
        """
        if self.resolution is None:
            self._calculate_resolution()
        mask = self.masks()
        idx = bin_indices[mask].to(torch.int64)  # dtype-ok: index_add_ requires int64 indices
        res = self.resolution[mask]
        total = torch.zeros(n_bins, dtype=res.dtype, device=res.device).index_add_(
            0, idx, res
        )
        count = torch.zeros_like(total).index_add_(0, idx, torch.ones_like(res))
        return total / count.clamp(min=1)

    def _calculate_resolution(self) -> None:
        """Set ``self.resolution`` to per-reflection d-spacing in Ångströms.

        Raises ``ValueError`` if ``hkl`` or ``cell`` is missing.
        """
        if self.hkl is None:
            raise ValueError(
                "Miller indices (hkl) are required to calculate resolution"
            )
        if self.cell is None:
            raise ValueError(
                "Unit cell parameters are required to calculate resolution"
            )
        s = math_torch.get_scattering_vectors(self.hkl, self.cell.data)
        resolution = 1.0 / torch.linalg.norm(s, axis=1)
        self.resolution = resolution

    def filter_by_resolution(
        self, d_min: Optional[float] = None, d_max: Optional[float] = None
    ) -> "ReflectionData":
        """
        Filter reflections by resolution range.

        Adds a boolean mask to self.masks for the specified resolution range.

        Parameters
        ----------
        d_min : float, optional
            Minimum resolution / high resolution cutoff (e.g., 1.5 Å).
        d_max : float, optional
            Maximum resolution / low resolution cutoff (e.g., 50.0 Å).

        Returns
        -------
        ReflectionData
            Self, for method chaining.
        """
        if self.resolution is None:
            self._calculate_resolution()

        mask = torch.ones(len(self.hkl), dtype=torch.bool, device=self.device)

        if d_min is not None:
            mask &= self.resolution >= d_min
        if d_max is not None:
            mask &= self.resolution <= d_max

        self.masks["resolution"] = mask

        if self.verbose > 0:
            valid = self.masks().sum().item()
            print(
                f"Filtering: {mask.sum()}/{len(mask)} reflections in range "
                f"[{d_max if d_max else 'inf'} - {d_min if d_min else 'inf'}] "
                f"\u00c5 ({valid} valid after all masks)"
            )

        return self

    def __len__(self) -> int:
        """Number of reflections (full array, ignoring masks)."""
        return len(self.hkl) if self.hkl is not None else 0

    @property
    def d_min(self) -> Optional[float]:
        """High-resolution limit: smallest d-spacing of the valid reflections, in Å."""
        if self.resolution is None:
            self._calculate_resolution()
        return float(self.resolution[self.masks()].min().item())

    def __repr__(self) -> str:
        """Count, data sources, resolution range and space group."""
        if self.hkl is None:
            return "ReflectionData(empty)"

        parts = [f"ReflectionData(n={len(self.hkl)}"]
        if self.amplitude_source:
            parts.append(f"F={self.amplitude_source}")
        if self.phase_source:
            parts.append(f"φ={self.phase_source}")
        if self.resolution is not None:
            parts.append(f"d={self.resolution.min():.2f}-{self.resolution.max():.2f}Å")
        parts.append(f"sg={self.spacegroup}")

        return ", ".join(parts) + ")"

    def data_indexed(
        self,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]
    ]:
        """
        Return reflection data as compact (valid-only) tensors.

        ScaledDataset returns its live corrected observations.

        Returns
        -------
        hkl : torch.Tensor
            Miller indices of shape (M, 3), M = number of valid reflections.
        F : torch.Tensor
            Structure factor amplitudes of shape (M,).
        F_sigma : torch.Tensor or None
            Uncertainties of shape (M,) or None.
        rfree_flags : torch.Tensor or None
            Flags of shape (M,) or None, coerced to bool (True=work).
        """
        to_mask = self.masks()

        hkl = self.hkl[to_mask]
        F = self.F[to_mask]
        F_sigma = self.F_sigma[to_mask] if self.F_sigma is not None else None
        rfree_flags = (
            self.rfree_flags[to_mask].to(torch.bool)
            if self.rfree_flags is not None
            else None
        )

        return hkl, F, F_sigma, rfree_flags

    def __getitem__(self, key):
        """
        Index into the reflection dataset.

        Parameters
        ----------
        key : torch.Tensor
            Boolean mask or integer indices for selection.

        Returns
        -------
        ReflectionData
            New ReflectionData object with selected reflections.
        """
        if isinstance(key, torch.Tensor):
            return self.__select__(key)
        raise TypeError(f"Unsupported index type: {type(key)}")

    def __select__(self, indices: torch.Tensor, op=None) -> "ReflectionData":
        """
        Select reflections by boolean mask or integer indices.

        Iterates over all dataclass fields generically, so new tensor fields
        are handled automatically.

        Parameters
        ----------
        indices : torch.Tensor
            Boolean mask of shape (N,) or integer indices for selection.
        op : str, optional
            Operation name for tracking purposes.

        Returns
        -------
        ReflectionData
            New ReflectionData object with selected reflections.
        """
        if indices.dtype == torch.bool:
            indices = torch.nonzero(indices).squeeze(-1)
        selected = ReflectionData(verbose=self.verbose, device=self.device)

        per_row = dict(self._per_row_fields())
        for f in fields(self):
            val = getattr(self, f.name)
            if f.name in per_row:
                setattr(selected, f.name, val[indices])
            elif isinstance(val, (torch.Tensor, Cell)):
                setattr(selected, f.name, val.clone())
            elif val is not None:
                setattr(selected, f.name, val)
        selected.masks = self._gathered_masks(indices)

        selected.source = self
        selected.last_op = op
        return selected

    def sanitize_F(self):
        """
        Remove invalid values from structure factors.

        Adds a mask to filter out NaN, Inf, and non-positive values
        from F and F_sigma.
        """
        mask = torch.zeros(len(self.F), dtype=torch.bool, device=self.device)
        if self.F is not None:
            # ~isfinite catches NaN AND +/-Inf (isnan alone let Inf through).
            nonfinite = ~torch.isfinite(self.F)
            if self.verbose > 0:
                print(
                    "found non-finite F values (NaN/Inf): ",
                    nonfinite.sum().item(),
                )
            mask |= nonfinite
        if self.F_sigma is not None:
            nonfinite_sigma = ~torch.isfinite(self.F_sigma)
            if self.verbose > 0:
                print(
                    "found non-finite F_sigma values (NaN/Inf): ",
                    nonfinite_sigma.sum().item(),
                )
            mask |= nonfinite_sigma
            # A non-positive sigma is not a measurement: it claims either
            # infinite precision or nonsense, and it is a division by zero in
            # any weighting scheme (including the Wilson h below). Caught
            # explicitly rather than relying on F == 0 happening to coincide,
            # which is what makes it invisible today.
            nonpositive_sigma = self.F_sigma <= 0
            if torch.any(nonpositive_sigma) and self.verbose > 0:
                print(
                    "found non-positive F_sigma values: ",
                    nonpositive_sigma.sum().item(),
                )
            mask |= nonpositive_sigma
        neg_mask = self.F <= 0
        if torch.any(neg_mask):
            warnings.warn(
                f"Found {neg_mask.sum().item()} non-positive F values, masking them out. This really should not happen!"
            )
            mask |= neg_mask
        self.masks["sanity_F"] = ~mask
        # Zero out invalid values so they can't leak NaN through autograd
        # (masked indexing produces 0 gradients, but 0 * NaN = NaN in IEEE 754)
        if mask.any():
            self.F[mask] = 0.0
            if self.F_sigma is not None:
                self.F_sigma[mask] = 0.0
        return self

    def validate_hkl(
        self, hkl_ref: torch.Tensor, *, identity_hkl: Optional[torch.Tensor] = None
    ) -> "ReflectionData":
        """
        Expand this dataset **in place** onto a reference HKL set.

        Every per-reflection array is reordered/expanded so ``self.hkl`` equals
        ``hkl_ref`` exactly; reflections absent here are given placeholder fills
        and excluded via the new ``hkl_present`` mask. Datasets aligned to the
        same reference then share a shape, instead of intersecting away data.
        ``rfree_flags`` keeps its stored dtype (int32 or bool).

        Parameters
        ----------
        hkl_ref : torch.Tensor
            Reference Miller indices of shape (N, 3), dtype int32; defines the
            canonical ordering for all aligned datasets.
        identity_hkl : torch.Tensor, optional
            Signed anomalous indices of shape (N, 3), distinguishing Bijvoet
            observations that share a canonical HKL. When supplied, match these
            against the dataset's signed indices and preserve their identities.

        Returns
        -------
        ReflectionData
            Self, mutated.
        """
        if self.hkl is None:
            raise ValueError("No Miller indices loaded in ReflectionData")

        if not isinstance(hkl_ref, torch.Tensor):
            raise TypeError(f"hkl_ref must be a torch.Tensor, got {type(hkl_ref)}")

        if hkl_ref.shape[-1] != 3:
            raise ValueError(f"hkl_ref must have shape (N, 3), got {hkl_ref.shape}")

        # Ensure hkl_ref is 2D and int32
        if hkl_ref.dim() == 1:
            hkl_ref = hkl_ref.unsqueeze(0)
        hkl_ref = hkl_ref.to(dtype=dtypes.int, device=self.device)

        n_ref = len(hkl_ref)
        n_data = len(self.hkl)

        # Build lookup from data HKL to index
        # Use a dictionary with tuple keys for fast lookup
        source_hkl = self.hkl if identity_hkl is None else self._hkl_for_sf()
        hkl_data_np = source_hkl.cpu().numpy()
        data_hkl_to_idx = {tuple(hkl): idx for idx, hkl in enumerate(hkl_data_np)}

        # For each reference HKL, find the corresponding data index (or -1 if missing)
        lookup_hkl = hkl_ref if identity_hkl is None else identity_hkl
        if lookup_hkl.shape != hkl_ref.shape:
            raise ValueError("identity_hkl must match the reference HKL shape")
        hkl_ref_np = lookup_hkl.cpu().numpy()
        ref_to_data_idx = np.array(
            [data_hkl_to_idx.get(tuple(hkl), -1) for hkl in hkl_ref_np], dtype=np.int64
        )

        # Index map into this dataset for each reference row (-1 where missing).
        valid_indices = torch.from_numpy(ref_to_data_idx).to(device=self.device)

        # Reindex EVERY per-reflection field via the shared primitive. Masks are
        # handled separately below because they are not dataclass fields.
        masks = self._gathered_masks(valid_indices)
        presence_mask = self._reindex_per_reflection(valid_indices, hkl_ref)
        if identity_hkl is not None:
            self.hkl_anomalous = identity_hkl.to(self.hkl).clone()
            self.friedel_flags = (self.hkl_anomalous != self.hkl).any(dim=-1)
        self._replace_masks(masks)
        # The mask that tells real reflections from placeholder rows.
        self.masks["hkl_present"] = presence_mask

        n_present = presence_mask.sum().item()
        n_missing = n_ref - n_present

        if self.verbose > 0:
            print("HKL validation (expand mode):")
            print(f"  Original dataset: {n_data} reflections")
            print(f"  Reference set: {n_ref} reflections")
            print(f"  Present in data: {n_present} ({100*n_present/n_ref:.1f}%)")
            print(f"  Missing (masked): {n_missing} ({100*n_missing/n_ref:.1f}%)")

        self._assert_per_reflection_consistent()
        return self

    WILSON_MASK_KEY = "wilson_valid"
    FRENCH_WILSON_MASK_KEY = "french_wilson_valid"

    def _set_french_wilson_mask(self, keep: Optional[torch.Tensor]) -> None:
        """Install French-Wilson's own input criterion as a keep-mask.

        This is the ``h >= -4`` guard on intensities too negative to be a noisy
        measurement of any Wilson-distributed reflection -- it protects the
        French-Wilson posterior integral. It is not outlier rejection; see
        :meth:`flag_wilson_outliers` for that.

        Raises rather than falling back when nothing survives: an all-False mask
        means every intensity is unphysical, which is a broken dataset. Silently
        widening it back to all-True would admit exactly the observations the
        test just identified as garbage.
        """
        if keep is None:
            return
        keep = keep.to(device=self.device, dtype=torch.bool)
        n_rejected = int((~keep).sum())
        if n_rejected == len(keep):
            raise ValueError(
                f"French-Wilson rejected all {len(keep)} reflections as "
                "unphysical. Every intensity is too negative to be a noisy "
                "measurement of a Wilson-distributed reflection, which usually "
                "means the intensities/sigmas are corrupt or the "
                "resolution-shell mean could not be estimated. Refusing to "
                "guess; inspect the input data."
            )
        if self.verbose > 0 and n_rejected:
            pct = 100.0 * n_rejected / max(len(keep), 1)
            # Deliberately not called an outlier count: this lumps together
            # absent measurements (non-finite in the file) and intensities too
            # negative to be noise, because French-Wilson rejects both for the
            # same reason -- it cannot integrate them.
            print(
                f"French-Wilson input guard: {n_rejected}/{len(keep)} "
                f"({pct:.4f}%) reflections unusable as intensities "
                "(absent, or too negative to be a noisy measurement)"
            )
        self.masks[self.FRENCH_WILSON_MASK_KEY] = keep

    def flag_wilson_outliers(
        self, alpha: float = 0.01, d_max: float = 4.0
    ) -> None:
        """
        Flag observations Wilson statistics cannot explain, in either tail.

        Model-free: nothing here has seen an ``F_calc``. Each reflection is
        compared against what Wilson statistics predict for its resolution
        shell, multiplicity and centricity, and rejected when the observation
        sits further into either tail than the size of the dataset can account
        for. See :mod:`torchref.base.wilson_outliers` for the criterion.

        Runs during loading, and the resulting mask joins the combined
        :attr:`masks`, so rejected reflections are excluded from scaling,
        refinement and R-factors. Call again with a different ``alpha`` to
        loosen it, or ``del data.masks[ReflectionData.WILSON_MASK_KEY]`` to drop
        it entirely.

        Reflections with no usable measurement are not outliers and are left to
        ``sanitize_F``; the count reported here excludes them.

        Tests :attr:`I` when the file carried intensities, since that is the
        measurement; otherwise the intensities are reconstructed from :attr:`F`
        via :func:`~torchref.base.french_wilson.intensities_from_amplitudes`,
        which is **not** an inverse of French-Wilson: ``F`` is a positive
        posterior mean, so an inexplicably negative intensity leaves no trace.
        The *negative* tail is therefore blind on amplitude-only data -- the
        strong tail, where zingers and mis-integrated spots live, is unaffected.

        Parameters
        ----------
        alpha : float, optional
            Family-wise error rate over the whole dataset, default 0.01. The
            per-reflection threshold is ``alpha/N``, so it tightens as the
            dataset grows rather than rejecting a fixed fraction.
        d_max : float, optional
            Only reflections finer than this are tested, in Å. Default 4.0.
            Below it bulk solvent dominates and the shells are sparse, so
            observations depart from Wilson statistics for reasons that have
            nothing to do with being outliers.
        """
        from torchref.base.french_wilson import intensities_from_amplitudes
        from torchref.base.wilson_outliers import wilson_outlier_mask
        from torchref.refinement.model_error_estimation.sigma_a import (
            epsilon_from_hkl,
        )

        if self.F is None or self.F_sigma is None or self.resolution is None:
            return
        if self.hkl is None or self.cell is None:
            return

        if self.I is not None and self.I_sigma is not None:
            I, sigma_I = self.I, self.I_sigma
        else:
            I, sigma_I = intensities_from_amplitudes(self.F, self.F_sigma)

        usable = torch.isfinite(I) & torch.isfinite(sigma_I) & (sigma_I > 0)
        if self.masks.get("sanity_F") is not None:
            usable = usable & self.masks["sanity_F"]
        if int(usable.sum()) == 0:
            return

        keep, info = wilson_outlier_mask(
            I,
            sigma_I,
            self.hkl,
            self.resolution,
            self.cell.data,
            epsilon=epsilon_from_hkl(self.hkl, self.spacegroup),
            is_centric=self.centric,
            usable=usable,
            alpha=alpha,
            d_max=d_max,
        )

        n_rejected = int((~keep).sum())
        if n_rejected == int(usable.sum()):
            raise ValueError(
                f"Wilson outlier rejection would reject all {n_rejected} "
                "measured reflections. That is a failed Sigma estimate or a "
                "dataset whose intensities are not Wilson-distributed (severe "
                "pseudo-translation, twinning, corrupt sigmas) -- not an "
                "outlier population. Refusing to guess; inspect the input data."
            )
        if self.verbose > 0 and info["n_tested"]:
            pct = 100.0 * n_rejected / info["n_tested"]
            print(
                f"Wilson outlier rejection: {n_rejected}/{info['n_tested']} "
                f"({pct:.4f}%) of tested reflections flagged "
                f"({info['n_strong']} too strong, {info['n_weak']} too weak)"
            )
        self.masks[self.WILSON_MASK_KEY] = keep.to(
            device=self.device, dtype=torch.bool
        )

    def write_mtz(
        self,
        fname: str,
        fcalc: Optional[torch.Tensor] = None,
        model_ft: Optional["ModelFT"] = None,
        anomalous: Optional[bool] = None,
    ) -> None:
        """Write this dataset, and optionally a model's map coefficients, to MTZ.

        A thin wrapper over :func:`torchref.io.mtz.write_reflections`, which
        documents the layouts and on-disk labels.

        Parameters
        ----------
        fname : str
            Output MTZ filename.
        fcalc : torch.Tensor, optional
            Complex structure factors of shape (N,), row-aligned with
            :attr:`hkl` in the canonical-ASU convention (as returned by
            :meth:`structure_factors`) and on the scale of ``F``. Adds model
            and 2Fo-Fc / Fo-Fc columns.
        model_ft : ModelFT, optional
            Used to compute ``fcalc`` when it is not given.
        anomalous : bool, optional
            Phenix-style anomalous layout; default when the data hold Bijvoet
            pairs (``friedel_merged`` False).

        Raises
        ------
        ValueError
            If ``fcalc`` is not row-aligned with :attr:`hkl`.
        """
        # One fallback for both layouts, so ``fcalc`` means the same thing
        # whether the caller supplied it or it was derived here. cached=False
        # keeps a no-grad write from leaving a detached tensor in the model's
        # forward cache.
        if fcalc is None and model_ft is not None:
            fcalc = self.structure_factors(model_ft, cached=False)
        if fcalc is not None and fcalc.shape[0] != len(self.hkl):
            raise ValueError(
                f"fcalc has {fcalc.shape[0]} rows but this dataset has "
                f"{len(self.hkl)}; it must be row-aligned with hkl."
            )
        mtz.write_reflections(
            self, fname, fcalc=fcalc, anomalous=anomalous, verbose=self.verbose
        )

    @property
    def centric(self):
        """Centric-reflection flags, shape (N,) full size and unfiltered.

        Computed on first access and cached on ``_centric_flags``; ``None`` when
        no HKL is loaded. Defaults to P1 if no space group is set.
        """
        if self.hkl is None:
            return None

        # Cached on the _centric_flags dataclass field, so it survives
        # serialization.
        if not hasattr(self, "_centric_flags") or self._centric_flags is None:
            sg = self.spacegroup or SpaceGroup("P1", device=self.hkl.device)

            self._centric_flags = sg.is_centric(self.hkl)

        return self._centric_flags

    def remap(
        self,
        new_hkl: torch.Tensor,
        index_mapping: torch.Tensor,
        phase_shifts: Optional[torch.Tensor] = None,
        spacegroup=None,
        op_name: str = "remap",
    ) -> "ReflectionData":
        """
        Create new ReflectionData with remapped HKL set and data.

        The core index-based transformation: every per-reflection field is
        gathered through ``index_mapping``, with ``-1`` marking reflections
        absent from the source.

        Parameters
        ----------
        new_hkl : torch.Tensor, shape (M, 3)
            New Miller indices.
        index_mapping : torch.Tensor, shape (M,), dtype int64
            Maps new indices to original: ``new[i] = old[index_mapping[i]]``
            Values of -1 indicate missing reflections (filled with defaults).
        phase_shifts : torch.Tensor, optional, shape (M,)
            Phase offsets to apply (e.g., from symmetry translations).
        spacegroup : str, int, gemmi.SpaceGroup, or None
            New spacegroup. If None, keeps original.
        op_name : str
            Operation name for provenance tracking.

        Returns
        -------
        ReflectionData
            New object with remapped data. Missing reflections get:
            - 0.0 for F, I, phase, fom
            - 1.0 for F_sigma, I_sigma (conservative uncertainty)
            - True for masks['missing']
        """
        from torchref.symmetry.spacegroup import SpaceGroup

        # Create new ReflectionData; set cell/spacegroup first so the shared
        # reindexer can recompute resolution on the new grid.
        remapped = ReflectionData(verbose=self.verbose, device=self.device)
        remapped.cell = self.cell.clone() if self.cell is not None else None
        if spacegroup is not None:
            remapped.spacegroup = SpaceGroup(spacegroup, device=remapped.device)
        else:
            remapped.spacegroup = self.spacegroup

        # Reindex ALL per-reflection dataclass fields onto new_hkl: sets hkl /
        # resolution, invalidates _centric_flags, and fills
        # missing rows (index -1) per _REINDEX_FILL.
        self._reindex_per_reflection(index_mapping, new_hkl, target=remapped)

        # Apply optional phase shifts (e.g. from symmetry translations).
        if remapped.phase is not None and phase_shifts is not None:
            remapped.phase = remapped.phase + phase_shifts.to(device=self.device)

        # Carry forward prior combined mask if available.
        prior_mask = self.masks()
        if prior_mask is not None:
            remapped.masks["prior_flagged"] = self._gather_rows(
                prior_mask, index_mapping.to(self.device), False
            )

        # Copy metadata sources
        remapped.amplitude_source = self.amplitude_source
        remapped.intensity_source = self.intensity_source
        remapped.phase_source = self.phase_source
        remapped.rfree_source = self.rfree_source

        # Track provenance
        remapped.source = self
        remapped.last_op = op_name

        # Add missing mask
        missing_mask = index_mapping < 0
        if missing_mask.any():
            remapped.masks["missing"] = ~missing_mask.to(device=self.device)

        remapped._assert_per_reflection_consistent()
        return remapped

    def expand_to_p1(
        self, include_friedel: bool = True, remove_absences: bool = True
    ) -> "ReflectionData":
        """
        Expand reflection data from asymmetric unit to P1.

        Applies all symmetry operations from the current space group to generate
        all symmetry-equivalent reflections. Returns a NEW ReflectionData object
        with expanded reflections; does not modify self.

        Anomalous data (``friedel_merged`` False) expand from their signed
        indices, so ``F(+)`` and ``F(-)`` each keep their own P1 reflections
        (``h`` and ``-h``) and the result holds both halves of reciprocal space
        whatever ``include_friedel`` says; with it, a Friedel copy fills in only
        where a mate was not measured. Consumers that want one value per
        reflection pair (a Hermitian map) must merge the mates first, e.g. with
        ``merge_to_spacegroup(data, data.spacegroup, anomalous=False)``.

        Parameters
        ----------
        include_friedel : bool, default True
            Include Friedel mates (-h, -k, -l). For normal (non-anomalous)
            scattering, Friedel pairs have identical amplitudes.
        remove_absences : bool, default True
            Remove systematically absent reflections from output.

        Returns
        -------
        ReflectionData
            New object at ``spacegroup="P1"`` holding every symmetry-equivalent
            reflection. Per-reflection fields are indexed from the original,
            ``phase`` additionally gets the translation phase shift, and
            ``resolution`` is recomputed. ``hkl_anomalous`` equals ``hkl``: each
            P1 row is its own index. ``source``/``last_op`` record the provenance.

        Raises
        ------
        ValueError
            If the data hold symmetry-equivalent rows (unmerged observations),
            which expansion would otherwise silently drop.
        """
        if self.hkl is None:
            raise ValueError("ReflectionData has no Miller indices loaded")

        anomalous = not self.friedel_merged and self.hkl_anomalous is not None
        sg = self.spacegroup or SpaceGroup("P1", device=self.device)
        hkl_p1, indices, phase_shifts = sg.expand_hkl(
            self.hkl_anomalous if anomalous else self.hkl,
            include_friedel=include_friedel,
            remove_absences=remove_absences,
            device=self.device,
        )

        p1 = self.remap(
            new_hkl=hkl_p1,
            index_mapping=indices,
            spacegroup="P1",
            op_name=f"expand_to_p1(include_friedel={include_friedel})",
        )
        if p1.phase is not None:
            phase = p1.phase
            if anomalous:
                # A conjugated mate stores the phase of its canonical index, the
                # negative of the phase at its own signed index.
                phase = torch.where(self.friedel_flags[indices], -phase, phase)
            p1.phase = phase + phase_shifts
        p1.hkl_anomalous = p1.hkl.clone()
        p1.friedel_flags = torch.zeros_like(p1.hkl[:, 0], dtype=torch.bool)
        p1.friedel_merged = self.friedel_merged
        return p1

    # ========== E-VALUE AND ANISOTROPY CORRECTION METHODS ==========

    def get_scattering_vectors(self) -> torch.Tensor:
        """
        Get scattering vectors (s-vectors) from hkl and cell.

        The s-vector for a reflection hkl is defined as:
            s = B* @ hkl
        where B* is the reciprocal basis matrix.

        Returns
        -------
        s_vectors : torch.Tensor
            Reciprocal space vectors in Angstroms^-1, shape (N, 3).

        Raises
        ------
        ValueError
            If hkl or cell is not available.
        """
        if self.hkl is None:
            raise ValueError("No Miller indices loaded")
        if self.cell is None:
            raise ValueError("No unit cell defined")

        return math_torch.get_scattering_vectors(self.hkl, self.cell.data)

    def get_corrected_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return amplitudes and sigmas, shape (N,), in this dataset's units.

        Raw datasets return their measurements; ScaledDataset exposes the live
        scale correction through the same observation attributes.
        """
        return self.F, self.F_sigma

    def get_corrected_intensities(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return intensities and sigmas, shape (N,), in this dataset's units.

        Raises
        ------
        ValueError
            If no intensity observations are available.
        """
        if self.I is None:
            raise ValueError("No intensities on this dataset (I/SIGI required)")
        return self.I, self.I_sigma

    def generate_validation_set(
        self,
        val_fraction_of_free: float = 0.5,
        seed: Optional[int] = None,
    ) -> None:
        """
        Carve a validation set out of the existing free-set reflections.

        Used when an MTZ file has only the standard ``FreeR_flag`` (work/free)
        but downstream code (e.g. ensemble refinement) needs a third held-out
        set for hyperparameter tuning. Free reflections are split
        resolution-stratified; ``val_fraction_of_free`` of them are marked in
        the separate boolean :attr:`validation_flags`, leaving
        :attr:`rfree_flags` untouched. The work/free/validation subsets are
        disjoint (validation is carved out of free) -- see
        :meth:`_subset_indices` and the ``work``/``free``/``validation``
        accessors. Like :meth:`generate_rfree_flags`, the split is over whole
        ASU groups so Bijvoet mates stay together (see
        :meth:`asu_group_indices`).

        Parameters
        ----------
        val_fraction_of_free : float, optional
            Fraction of *existing free* reflections to reassign as
            validation. Default 0.5.
        seed : int, optional
            Random seed for reproducibility.
        """
        if self.rfree_flags is None:
            raise ValueError("No rfree_flags present; cannot split into validation.")
        if not 0.0 < val_fraction_of_free < 1.0:
            raise ValueError(
                f"val_fraction_of_free must be in (0, 1); got {val_fraction_of_free}"
            )
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Free reflections are those with rfree_flags == 0 (0=free, nonzero=work).
        rwork = self.rfree_flags.to(torch.bool)
        free_mask = ~rwork

        # Split whole ASU groups, exactly as generate_rfree_flags does -- a
        # per-row draw here would re-open the Friedel leak at the free/validation
        # boundary. The free set is already group-consistent, so a group is
        # wholly free or wholly work.
        group_id, n_groups = self.asu_group_indices()
        group_free = self._group_any(free_mask, group_id, n_groups)

        bin_indices, n_bins = self.get_bins(n_bins=20, min_per_bin=20)
        group_bin = bin_indices[self._group_representative_rows(group_id, n_groups)]
        group_val = self._stratified_group_draw(
            group_free,
            group_bin,
            n_bins,
            lambda n: max(1, int(n * val_fraction_of_free)),
        )

        # Broadcast to rows, staying within the free set.
        val_flags = group_val[group_id] & free_mask

        self.validation_flags = val_flags
        self.rfree_source = (self.rfree_source or "") + "+val_split"

        if self.verbose > 0:
            n_work = int(rwork.sum().item())
            n_val = int(val_flags.sum().item())
            total = len(val_flags)
            n_free = total - n_work - n_val
            print(
                f"  Generated validation set: "
                f"work={n_work} ({100*n_work/total:.1f}%), "
                f"free={n_free} ({100*n_free/total:.1f}%), "
                f"val={n_val} ({100*n_val/total:.1f}%)"
            )
