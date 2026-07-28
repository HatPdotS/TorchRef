"""
Model collection for time-resolved kinetic refinement.

Provides ModelCollection — a named dictionary of MixedModel instances at
different timepoints that share the same base structural models (ModelFT).
Keys match DatasetCollection keys so targets can automatically pair them.
"""

from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Tuple

import torch
from torch import nn

from torchref.utils.device_mixin import DeviceMovementMixin
from torchref.utils.device_resolution import resolve_device

if TYPE_CHECKING:
    from torchref.model.model_ft import ModelFT
    from torchref.model.mixed_model import MixedModel


class _SharedMixedModel(DeviceMovementMixin, nn.Module):
    """
    MixedModel variant that references shared base models without re-registering them.

    Standard MixedModel wraps models in nn.ModuleList, which causes
    double-registration when the same ModelFT objects appear in multiple
    timepoints.  This class stores the shared models as a plain list
    (no ownership) and only owns its own fraction parameters.

    An external fraction override (via ``set_fraction_override``) can replace
    the softmax-derived fractions; while active, ``fractions`` and ``forward``
    use the override tensor instead of ``softmax(fraction_params)``.

    Parameters
    ----------
    base_models : List[ModelFT]
        Shared structural models (not re-registered as submodules here).
    initial_fractions : List[float]
        Initial population fractions (must sum to 1).
    frozen_fractions : bool
        If True, fractions are excluded from optimization.
    device : torch.device, optional
        Device for fraction parameters.
    """

    def __init__(
        self,
        base_models: List["ModelFT"],
        initial_fractions: List[float],
        frozen_fractions: bool = False,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        # Store as plain list — the parent ModelCollection owns the ModuleList
        self._base_models = base_models

        n = len(base_models)
        if len(initial_fractions) != n:
            raise ValueError(
                f"Number of fractions ({len(initial_fractions)}) must match "
                f"number of models ({n})."
            )
        total = sum(initial_fractions)
        if abs(total - 1.0) > 1e-3:
            raise ValueError(f"Initial fractions must sum to 1.0, got {total:.6f}.")

        # Normalize to handle floating point drift
        initial_fractions = [f / total for f in initial_fractions]

        # Reconcile across *all* base models, not just the first: reading
        # ``base_models[0].device`` left a mixed-device list unreconciled, so
        # ``fractions_tensor`` below could land on a device the later models
        # were not on.
        device = resolve_device(*base_models, device=device)

        # Match base models' float dtype (consistent under a float64 config).
        fractions_tensor = torch.tensor(
            initial_fractions, dtype=base_models[0].dtype_float, device=device
        )
        theta = torch.log(fractions_tensor.clamp(min=1e-6))
        self.fraction_params = nn.Parameter(theta, requires_grad=not frozen_fractions)

        # Optional override: when set, fractions property returns this tensor
        # instead of softmax(fraction_params). Used by refine_kinetics() to
        # route kinetic model predictions directly into the F_calc computation.
        self._fraction_override: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def fractions(self) -> torch.Tensor:
        """Normalized population fractions (sum to 1).

        When a fraction override is active (set by ``set_fraction_override``),
        returns the override tensor instead of softmax(fraction_params).
        This allows kinetic model predictions to flow directly into F_calc.
        """
        if self._fraction_override is not None:
            return self._fraction_override
        return torch.softmax(self.fraction_params, dim=0)

    @property
    def models(self) -> List["ModelFT"]:
        """The shared base models (read-only reference)."""
        return self._base_models

    @property
    def cell(self):
        return self._base_models[0].cell

    @property
    def spacegroup(self):
        return self._base_models[0].spacegroup

    @property
    def device(self):
        return self._base_models[0].device

    @property
    def dtype_float(self):
        return self._base_models[0].dtype_float

    @property
    def real_space_grid(self):
        return self._base_models[0].real_space_grid

    @property
    def fft(self):
        return self._base_models[0].fft

    @property
    def gridsize(self):
        return self._base_models[0].gridsize

    @property
    def map_symmetry(self):
        return self._base_models[0].map_symmetry

    @property
    def inv_fractional_matrix(self):
        return self.cell.inv_fractional_matrix.to(dtype=self.dtype_float)

    @property
    def fractional_matrix(self):
        return self.cell.fractional_matrix.to(dtype=self.dtype_float)

    # ------------------------------------------------------------------
    # Grid / density helpers (delegate to base models)
    # ------------------------------------------------------------------

    def setup_grid(self, max_res=None, gridsize=None):
        for model in self._base_models:
            model.setup_grid(max_res=max_res, gridsize=gridsize)

    def get_radius(self, min_radius_Angstrom: float = 4.0) -> int:
        return self._base_models[0].get_radius(min_radius_Angstrom)

    def build_complete_map(self) -> torch.Tensor:
        """Mixed electron density: sum_i w_i * density_i."""
        fractions = self.fractions
        density = None
        for i, model in enumerate(self._base_models):
            weighted = fractions[i] * model.build_complete_map()
            density = weighted if density is None else density + weighted
        return density

    # ------------------------------------------------------------------
    # Forward: weighted structure factors
    # ------------------------------------------------------------------

    def forward(self, hkl: torch.Tensor, recalc: bool = False) -> torch.Tensor:
        """
        Compute f_mixed = sum_i w_i * f_i.

        Parameters
        ----------
        hkl : torch.Tensor
            Miller indices, shape (n_reflections, 3).
        recalc : bool
            Force recalculation of structure factors.

        Returns
        -------
        torch.Tensor
            Mixed complex structure factors.
        """
        fractions = self.fractions
        f_mixed = None
        for i, model in enumerate(self._base_models):
            f_i = model(hkl, recalc=recalc)
            weighted_f = fractions[i] * f_i
            f_mixed = weighted_f if f_mixed is None else f_mixed + weighted_f
        return f_mixed

    # ------------------------------------------------------------------
    # Freeze / unfreeze
    # ------------------------------------------------------------------

    def freeze_fractions(self):
        self.fraction_params.requires_grad = False

    def unfreeze_fractions(self):
        self.fraction_params.requires_grad = True

    def set_fraction_override(self, fractions: torch.Tensor):
        """Override fractions with an external tensor (e.g. from kinetic model).

        While active, ``self.fractions`` returns this tensor instead of
        ``softmax(fraction_params)``, allowing gradients to flow through
        the external source.
        """
        self._fraction_override = fractions

    def clear_fraction_override(self):
        """Remove fraction override, reverting to softmax(fraction_params)."""
        self._fraction_override = None

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_vdw_radii(self):
        return self._base_models[0].get_vdw_radii()

    def xyz(self):
        """Cartesian coordinates of the first shared base model."""
        return self._base_models[0].xyz()

    def get_individual_fcalc(self, hkl, recalc=True):
        """Per-model (unweighted) structure factors, one tensor per base model."""
        return [m(hkl, recalc=recalc) for m in self._base_models]

    def __repr__(self):
        fracs = self.fractions.detach().tolist()
        frac_str = ", ".join(f"{f:.3f}" for f in fracs)
        frozen_str = "frozen" if not self.fraction_params.requires_grad else "learnable"
        return f"_SharedMixedModel({len(self._base_models)} models, fractions=[{frac_str}], {frozen_str})"


class ModelCollection(DeviceMovementMixin, nn.Module):
    """
    Named dictionary of MixedModel instances at different timepoints.

    All timepoint models share the same base structural models (ModelFT
    objects stored once in an nn.ModuleList). Each timepoint gets its own
    independent fraction parameters via _SharedMixedModel.

    Keys should match DatasetCollection keys so that collection-aware
    targets can automatically pair datasets with models.

    Parameters
    ----------
    base_models : List[ModelFT]
        The K shared structural models (e.g., ground state + intermediates).
    dark_key : str
        Key for the dark / reference entry. Default ``"dark"``.
    verbose : int
        Verbosity level.

    Examples
    --------
    ::

        models = ModelCollection([model_dark, model_light])
        models.add_dark()                             # fractions=[1, 0]
        models.add_timepoint("1ps", [0.9, 0.1])
        models.add_timepoint("5ps", [0.7, 0.3])

        # Access
        mixed = models["1ps"]
        fcalc = mixed(hkl)
        print(mixed.fractions)
    """

    def __init__(
        self,
        base_models: List["ModelFT"],
        dark_key: str = "dark",
        verbose: int = 0,
    ):
        super().__init__()

        if not base_models:
            raise ValueError("At least one base model is required.")

        self._dark_key = dark_key
        self.verbose = verbose

        # Register base models as owned submodules (single source of truth)
        self._base_models = nn.ModuleList(base_models)

        # Per-timepoint mixed models (own only fraction params)
        self._timepoints = nn.ModuleDict()
        self._order: List[str] = []

        if self.verbose > 0:
            print(
                f"ModelCollection initialized with {len(base_models)} base models"
            )

    # ------------------------------------------------------------------
    # Add timepoints
    # ------------------------------------------------------------------

    def add_timepoint(
        self,
        name: str,
        fractions: Optional[List[float]] = None,
        frozen_fractions: bool = False,
    ) -> "ModelCollection":
        """
        Add a timepoint with given initial fractions.

        Parameters
        ----------
        name : str
            Timepoint identifier (should match DatasetCollection key).
        fractions : List[float], optional
            Initial population fractions. If None, uses equal fractions.
        frozen_fractions : bool
            If True, fractions are not updated during optimization.

        Returns
        -------
        ModelCollection
            Self, for method chaining.
        """
        if name in self._timepoints:
            raise ValueError(f"Timepoint '{name}' already exists.")

        n = len(self._base_models)
        if fractions is None:
            fractions = [1.0 / n] * n

        mixed = _SharedMixedModel(
            base_models=list(self._base_models),
            initial_fractions=fractions,
            frozen_fractions=frozen_fractions,
        )
        self._timepoints[name] = mixed
        self._order.append(name)

        if self.verbose > 0:
            frac_str = ", ".join(f"{f:.3f}" for f in fractions)
            print(f"  Added timepoint '{name}': fractions=[{frac_str}]")

        return self

    def add_dark(
        self, fractions: Optional[List[float]] = None
    ) -> "ModelCollection":
        """
        Add the dark / reference entry.

        Default fractions: [1, 0, 0, ...] (100 % ground state).

        Parameters
        ----------
        fractions : List[float], optional
            Override dark fractions. Default is pure ground state.

        Returns
        -------
        ModelCollection
            Self, for method chaining.
        """
        if fractions is None:
            n = len(self._base_models)
            fractions = [0.0] * n
            fractions[0] = 1.0
        return self.add_timepoint(self._dark_key, fractions, frozen_fractions=True)

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_kinetics(
        cls,
        base_models: List["ModelFT"],
        occ_model,
        timepoint_names: List[str],
        dark_key: str = "dark",
        verbose: int = 0,
    ) -> "ModelCollection":
        """
        Create a ModelCollection from a kinetics occupancy model.

        Parameters
        ----------
        base_models : List[ModelFT]
            Shared structural models.
        occ_model : occupancies_kinetics
            Kinetic occupancy model whose forward() returns
            shape [n_states, n_timepoints].
        timepoint_names : List[str]
            Names for each timepoint column (excluding dark).
        dark_key : str
            Key for the dark entry.
        verbose : int
            Verbosity level.

        Returns
        -------
        ModelCollection
        """
        collection = cls(base_models, dark_key=dark_key, verbose=verbose)
        collection.add_dark()

        with torch.no_grad():
            occ = occ_model()  # [n_states, n_timepoints]

        for t_idx, name in enumerate(timepoint_names):
            # +1 because index 0 in occ is the dark timepoint
            fracs = occ[:, t_idx + 1].tolist()
            collection.add_timepoint(name, fracs)

        return collection

    # ------------------------------------------------------------------
    # IHM I/O
    # ------------------------------------------------------------------

    @classmethod
    def from_ihm(
        cls,
        filepath: str,
        max_res: float = 1.5,
        device=None,
        verbose: int = 0,
    ) -> tuple:
        """
        Load a ModelCollection from an IHM mmCIF file.

        Requires the optional ``python-ihm`` dependency.

        Parameters
        ----------
        filepath : str
            Path to IHM mmCIF file.
        max_res : float
            Maximum resolution for FFT grid setup.
        device : torch.device, optional
            Device for model tensors.
        verbose : int
            Verbosity level.

        Returns
        -------
        tuple of (ModelCollection, IHMEnsembleMapping)
        """
        from torchref.io.ihm import IHMReader

        reader = IHMReader(filepath, verbose=verbose)
        return reader(
            max_res=max_res,
            device=device,
        )

    def write_ihm(self, filepath: str, mapping=None, datasets=None) -> None:
        """
        Write this ModelCollection to IHM mmCIF format.

        Requires the optional ``python-ihm`` dependency.

        Parameters
        ----------
        filepath : str
            Output file path.
        mapping : IHMEnsembleMapping, optional
            Mapping with metadata for round-tripping. If ``None``,
            a minimal mapping is created from the collection structure.
        datasets : dict of str -> ReflectionData, optional
            Per-timepoint reflection data to embed in the CIF.
            Each key should match a timepoint name.
        """
        from torchref.io.ihm import IHMWriter

        writer = IHMWriter(
            self, mapping=mapping, datasets=datasets, verbose=self.verbose,
        )
        writer.write(filepath)

    # ------------------------------------------------------------------
    # Dict-like access
    # ------------------------------------------------------------------

    def __getitem__(self, name: str) -> "_SharedMixedModel":
        return self._timepoints[name]

    def __contains__(self, name: str) -> bool:
        return name in self._timepoints

    def __iter__(self) -> Iterator[Tuple[str, "_SharedMixedModel"]]:
        for name in self._order:
            yield name, self._timepoints[name]

    def __len__(self) -> int:
        return len(self._timepoints)

    def keys(self) -> List[str]:
        return list(self._order)

    def values(self) -> List["_SharedMixedModel"]:
        return [self._timepoints[n] for n in self._order]

    def items(self) -> List[Tuple[str, "_SharedMixedModel"]]:
        return [(n, self._timepoints[n]) for n in self._order]

    def get(self, name: str, default=None):
        return self._timepoints.get(name, default)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def dark_key(self) -> str:
        return self._dark_key

    @property
    def dark_model(self) -> "_SharedMixedModel":
        """Shortcut for ``self[dark_key]``."""
        return self._timepoints[self._dark_key]

    @property
    def base_models(self) -> nn.ModuleList:
        """The shared structural models (owned by this collection)."""
        return self._base_models

    @property
    def n_base_models(self) -> int:
        return len(self._base_models)

    @property
    def timepoint_names(self) -> List[str]:
        """All keys except the dark key."""
        return [n for n in self._order if n != self._dark_key]

    @property
    def cell(self):
        return self._base_models[0].cell

    @property
    def spacegroup(self):
        return self._base_models[0].spacegroup

    @property
    def device(self):
        return self._base_models[0].device

    # ------------------------------------------------------------------
    # Fractions inspection
    # ------------------------------------------------------------------

    def get_all_fractions(self) -> Dict[str, torch.Tensor]:
        """Current fractions for each timepoint (including dark)."""
        return {name: self._timepoints[name].fractions for name in self._order}

    def get_fractions_matrix(self) -> torch.Tensor:
        """
        All fractions as a matrix [n_timepoints, n_models].

        Rows are ordered by ``self._order`` (i.e. insertion order).
        """
        return torch.stack(
            [self._timepoints[n].fractions for n in self._order], dim=0
        )

    # ------------------------------------------------------------------
    # Freeze / unfreeze helpers
    # ------------------------------------------------------------------

    def freeze_all_fractions(self):
        """Freeze fractions at all timepoints."""
        for _, mixed in self:
            mixed.freeze_fractions()

    def unfreeze_all_fractions(self):
        """Unfreeze fractions at all timepoints (except dark)."""
        for name, mixed in self:
            if name != self._dark_key:
                mixed.unfreeze_fractions()

    def freeze_structures(self):
        """Freeze xyz and adp on all base models."""
        for model in self._base_models:
            model.freeze("xyz")
            model.freeze("b")

    def unfreeze_structures(self):
        """Unfreeze xyz and adp on all base models."""
        for model in self._base_models:
            model.unfreeze("xyz")
            model.unfreeze("b")

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def write_pdbs(self, outdir: str):
        """
        Write each base model to a PDB file in *outdir*.

        Files are named ``base_model_0.pdb``, ``base_model_1.pdb``, etc.

        Parameters
        ----------
        outdir : str
            Directory to write PDB files into (must exist).
        """
        import os

        for i, model in enumerate(self._base_models):
            path = os.path.join(outdir, f"base_model_{i}.pdb")
            model.write_pdb(path)
            if self.verbose > 0:
                print(f"  Wrote {path}")

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self):
        tp_names = ", ".join(self._order[:4])
        if len(self._order) > 4:
            tp_names += f", ... ({len(self._order)} total)"
        return (
            f"ModelCollection({self.n_base_models} base models, "
            f"timepoints=[{tp_names}])"
        )
