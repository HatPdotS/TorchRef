"""
Model collection for time-resolved kinetic refinement.

Provides ModelCollection — a named dictionary of mixed models at different timepoints
that share the same base structural models (ModelFT). Keys match DatasetCollection keys
so targets can automatically pair them.

Populations are stored **factorised**, not as a free vector per timepoint::

    w(t) = (1 - alpha) * e_ref  +  alpha * q(t)

with one mean activation ``alpha`` shared across every timepoint and a per-timepoint
branching ``q(t)`` over the non-reference components. This is the statement that only
the overall degree of activation varies from crystal to crystal, while the branching
among excited states is conserved -- and it makes the mixture exactly linear in
``alpha``, so :meth:`ModelCollection.activation_jacobian` is constant and a second
moment of the activation distribution costs no extra structure-factor evaluation.

``ModelCollection`` owns the population parameters; each timepoint is a view onto one
row (:class:`_SharedMixedModel`). One consequence worth knowing: freezing or unfreezing
fractions is collection-wide, because a single activation cannot be frozen for one
timepoint alone. Timepoints that genuinely need independent populations are driven
through ``set_fraction_override`` instead.
"""

from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Tuple

import torch
from torch import nn

from torchref.utils.device_mixin import DeviceMovementMixin
from torchref.utils.device_resolution import resolve_device
from torchref.utils.utils import ModuleReference

if TYPE_CHECKING:
    from torchref.model.model_ft import ModelFT
    from torchref.model.mixed_model import MixedModel

#: Activation fractions are clamped away from 0 and 1 before taking a logit, which
#: would otherwise be infinite. 1e-6 is the same floor the fraction storage has always
#: applied.
_FRACTION_EPS = 1e-6


def _logit(p: float) -> float:
    """Inverse sigmoid, clamped away from the infinities at 0 and 1."""
    p = min(max(float(p), _FRACTION_EPS), 1.0 - _FRACTION_EPS)
    return float(torch.log(torch.tensor(p / (1.0 - p))))


class _SharedMixedModel(DeviceMovementMixin, nn.Module):
    """
    One timepoint's view of a :class:`ModelCollection`.

    Owns nothing. The shared base models are held as a plain list and the population
    parameters live on the parent collection, so neither is re-registered here --
    the same ownership pattern in both cases, and what keeps a base model's
    parameters from appearing once per timepoint in ``parameters()``.

    An external fraction override (via ``set_fraction_override``) replaces the
    derived fractions; while active, ``fractions`` and ``forward`` use the override
    tensor, and gradients flow to whatever produced it.

    Parameters
    ----------
    base_models : List[ModelFT]
        Shared structural models (not re-registered as submodules here).
    collection : ModelCollection
        Owner of the activation, branching and dispersion parameters. Referenced
        without registration.
    index : int
        This timepoint's row in the collection's insertion order.
    device : torch.device, optional
        Device to reconcile the base models onto.
    """

    def __init__(
        self,
        base_models: List["ModelFT"],
        collection: "ModelCollection",
        index: int,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        # Store as plain list — the parent ModelCollection owns the ModuleList
        self._base_models = base_models

        # Parent reference, deliberately not a submodule: the population parameters
        # are the collection's, shared across every timepoint.
        self._collection_ref = ModuleReference(collection)
        self._index = index

        # Reconcile across *all* base models, not just the first, so a mixed-device
        # list does not stay unreconciled.
        resolve_device(*base_models, device=device)

        # Optional override: when set, fractions property returns this tensor
        # instead of the collection's derived row. Used by refine_kinetics() to
        # route kinetic model predictions directly into the F_calc computation.
        self._fraction_override: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def collection(self) -> "ModelCollection":
        """The owning collection."""
        return self._collection_ref.module

    @property
    def fractions(self) -> torch.Tensor:
        """Normalized population fractions -- the override tensor while one is
        set (see ``set_fraction_override``), else this timepoint's row of the
        parent's :meth:`ModelCollection.fractions_matrix`.
        """
        if self._fraction_override is not None:
            return self._fraction_override
        return self.collection.fractions_matrix()[self._index]

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
        """Freeze the population parameters.

        **Collection-wide.** The mean activation is a single parameter shared by every
        timepoint, so it cannot be frozen for one timepoint alone; this delegates to
        :meth:`ModelCollection.freeze_all_fractions`.
        """
        self.collection.freeze_all_fractions()

    def unfreeze_fractions(self):
        """Unfreeze the population parameters. Collection-wide; see
        :meth:`freeze_fractions`."""
        self.collection.unfreeze_all_fractions()

    def set_fraction_override(self, fractions: torch.Tensor):
        """Override fractions with an external tensor (e.g. from kinetic model).

        While active, ``self.fractions`` returns this tensor instead of the
        collection's derived row, allowing gradients to flow through the external
        source.
        """
        self._fraction_override = fractions

    def clear_fraction_override(self):
        """Remove the fraction override, reverting to the collection's derived row."""
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
        learnable = self.collection._activation_logit.requires_grad
        frozen_str = "learnable" if learnable else "frozen"
        return (
            f"_SharedMixedModel({len(self._base_models)} models, "
            f"fractions=[{frac_str}], {frozen_str})"
        )


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
        fcalc = models["1ps"](hkl)
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

        # Per-timepoint views (own nothing; see _SharedMixedModel)
        self._timepoints = nn.ModuleDict()
        self._order: List[str] = []

        device = resolve_device(*base_models)
        dtype = base_models[0].dtype_float

        # --- population parameters -------------------------------------
        #
        # Only the *overall* activation varies from crystal to crystal; the branching
        # among excited components is conserved. So the populations factorise as
        #
        #     w(t) = (1 - alpha) * e_ref  +  alpha * q(t)
        #
        # with one activation shared across timepoints and a per-timepoint branching
        # distribution over the K-1 non-reference components. Both are frozen by
        # default: population refinement is opt-in.
        self._activation_logit = nn.Parameter(
            torch.tensor(_logit(1e-6), dtype=dtype, device=device),
            requires_grad=False,
        )
        self._branching_logits = nn.ParameterList()
        self._branching_rows: Dict[str, int] = {}

        # Dispersion of the activation across crystals, as the fraction of its
        # maximum: sigma_alpha^2 = alpha (1 - alpha) * lambda, so 0 <= lambda <= 1
        # holds by construction. Stored as a plain float while fixed, because
        # sigmoid can never return exactly 0 and lambda = 0 is what reproduces the
        # single-moment (coherent) model.
        self._lambda_logit = nn.Parameter(
            torch.zeros((), dtype=dtype, device=device), requires_grad=False
        )
        self._lambda_fixed: Optional[float] = 0.0

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
        if len(fractions) != n:
            raise ValueError(
                f"Number of fractions ({len(fractions)}) must match "
                f"number of models ({n})."
            )
        total = sum(fractions)
        if abs(total - 1.0) > 1e-3:
            raise ValueError(f"Initial fractions must sum to 1.0, got {total:.6f}.")
        fractions = [f / total for f in fractions]

        index = len(self._order)
        self._install_populations(name, fractions)

        mixed = _SharedMixedModel(
            base_models=list(self._base_models),
            collection=self,
            index=index,
        )
        self._timepoints[name] = mixed
        self._order.append(name)

        if frozen_fractions:
            self.freeze_all_fractions()

        if self.verbose > 0:
            frac_str = ", ".join(f"{f:.3f}" for f in fractions)
            print(f"  Added timepoint '{name}': fractions=[{frac_str}]")

        return self

    def _install_populations(self, name: str, fractions: List[float]) -> None:
        """Invert requested fractions into the (activation, branching) factorisation.

        The reference component's weight is ``1 - alpha`` by construction, so a
        timepoint that is pure reference carries no branching row and leaves the
        activation alone. Every other timepoint pins the shared activation; a second
        one asking for a different value cannot be represented and is rejected rather
        than silently projected.

        Raises
        ------
        ValueError
            If ``fractions`` implies an activation incompatible with one already set
            by an earlier timepoint.
        """
        alpha = 1.0 - fractions[0]

        if alpha <= _FRACTION_EPS:
            # Pure reference: this is the dark / ground state, i.e. the alpha = 0
            # evaluation of the same parametrisation. No branching row.
            return

        current = float(self.alpha_mean)
        if self._branching_rows:
            if abs(alpha - current) > 1e-3:
                established = ", ".join(sorted(self._branching_rows))
                raise ValueError(
                    f"Timepoint {name!r} asks for activation {alpha:.4f}, but "
                    f"{established} already set it to {current:.4f}. One activation "
                    f"fraction is shared across all timepoints -- only the branching "
                    f"among excited components varies with time. To drive timepoints "
                    f"with independent populations, use set_fraction_override() on "
                    f"each one instead of passing fractions here."
                )
        else:
            with torch.no_grad():
                self._activation_logit.fill_(_logit(alpha))

        # Branching over the K-1 non-reference components, renormalised within alpha.
        excited = torch.tensor(
            [f / alpha for f in fractions[1:]],
            dtype=self._activation_logit.dtype,
            device=self._activation_logit.device,
        )
        logits = torch.log(excited.clamp(min=_FRACTION_EPS))
        self._branching_rows[name] = len(self._branching_logits)
        self._branching_logits.append(nn.Parameter(logits, requires_grad=False))

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
        # No frozen_fractions here: the reference is the alpha = 0 evaluation of the
        # shared parametrisation, so it owns nothing that could be frozen.
        return self.add_timepoint(self._dark_key, fractions)

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
        """All fractions as a matrix ``[n_timepoints, n_models]``, in insertion order.

        Alias of :meth:`fractions_matrix`, kept because it is the established name.
        """
        return self.fractions_matrix()

    # ------------------------------------------------------------------
    # Population factorisation
    # ------------------------------------------------------------------

    @property
    def alpha_mean(self) -> torch.Tensor:
        """Mean activation fraction, shared across all timepoints."""
        return torch.sigmoid(self._activation_logit)

    @property
    def lambda_twin(self) -> torch.Tensor:
        """Activation dispersion as a fraction of its maximum, in ``[0, 1]``.

        Zero is the coherent single-moment model. One means every crystal is either
        fully activated or fully dark. Exactly representable while fixed; once
        refinement is enabled it is ``sigmoid`` of a parameter and therefore strictly
        interior.
        """
        if self._lambda_fixed is not None:
            return torch.tensor(
                self._lambda_fixed,
                dtype=self._lambda_logit.dtype,
                device=self._lambda_logit.device,
            )
        return torch.sigmoid(self._lambda_logit)

    @property
    def sigma_alpha_sq(self) -> torch.Tensor:
        """Variance of the activation across crystals.

        ``alpha (1 - alpha) * lambda``, so ``0 <= sigma_alpha_sq <= alpha (1 - alpha)``
        holds by construction -- the upper bound being the Bernoulli case.
        """
        alpha = self.alpha_mean
        return alpha * (1.0 - alpha) * self.lambda_twin

    def branching(self) -> torch.Tensor:
        """Per-timepoint distribution over the non-reference components.

        Returns
        -------
        torch.Tensor
            Shape ``(n_branching_rows, n_base_models - 1)``, rows summing to 1.
            Empty when no non-reference timepoint has been added.
        """
        if not len(self._branching_logits):
            return torch.zeros(
                (0, max(len(self._base_models) - 1, 0)),
                dtype=self._activation_logit.dtype,
                device=self._activation_logit.device,
            )
        return torch.stack(
            [torch.softmax(row, dim=0) for row in self._branching_logits], dim=0
        )

    def activation_jacobian(self) -> torch.Tensor:
        """``d(fractions) / d(alpha)`` for every timepoint.

        Shape ``(n_timepoints, n_base_models)``. Reference-only rows are exactly zero,
        so the reference dataset carries no activation gradient. Every other row is
        ``q(t) - e_ref``, whose entries sum to zero because the fractions stay on the
        simplex.

        This is what makes a second moment computable through the same machinery as the
        first: the mixture is exactly linear in ``alpha``, so this Jacobian is constant
        in ``alpha`` and can be scaled by the same affine scaler as the mixture itself.
        """
        n_models = len(self._base_models)
        dtype = self._activation_logit.dtype
        device = self._activation_logit.device

        q_all = self.branching()
        rows = []
        for name in self._order:
            row = torch.zeros(n_models, dtype=dtype, device=device)
            if name in self._branching_rows:
                q = q_all[self._branching_rows[name]]
                row = torch.cat(
                    [torch.full((1,), -1.0, dtype=dtype, device=device), q]
                )
            rows.append(row)
        if not rows:
            return torch.zeros((0, n_models), dtype=dtype, device=device)
        return torch.stack(rows, dim=0)

    def fractions_matrix(self) -> torch.Tensor:
        """Population fractions for every timepoint, ``[n_timepoints, n_models]``.

        ``e_ref + alpha * activation_jacobian``. Reference-only rows come out as exactly
        ``e_ref`` with no gradient path to the activation.
        """
        n_models = len(self._base_models)
        dtype = self._activation_logit.dtype
        device = self._activation_logit.device

        e_ref = torch.zeros(n_models, dtype=dtype, device=device)
        e_ref[0] = 1.0
        return e_ref.unsqueeze(0) + self.alpha_mean * self.activation_jacobian()

    def fraction_parameters(self) -> List[nn.Parameter]:
        """The population parameters, for handing to an optimizer.

        The shared activation and every branching row, plus the dispersion when it is
        refinable. Replaces reaching into a per-timepoint parameter.
        """
        params: List[nn.Parameter] = [self._activation_logit]
        params.extend(self._branching_logits)
        if self._lambda_fixed is None:
            params.append(self._lambda_logit)
        return params

    def set_activation(self, alpha: float) -> "ModelCollection":
        """Set the shared mean activation fraction, in place and without gradient."""
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError(f"alpha must lie in [0, 1]; got {alpha}")
        with torch.no_grad():
            self._activation_logit.fill_(_logit(alpha))
        return self

    def set_branching(self, name: str, q: torch.Tensor) -> "ModelCollection":
        """Set one timepoint's branching distribution, in place and without gradient.

        Parameters
        ----------
        name : str
            Timepoint key. Must be a non-reference timepoint.
        q : torch.Tensor
            Weights over the ``n_base_models - 1`` non-reference components. Normalised
            internally; need not sum to 1.
        """
        if name not in self._branching_rows:
            raise KeyError(
                f"{name!r} has no branching row -- it is the reference timepoint, "
                f"whose fractions are fixed at the alpha = 0 evaluation."
            )
        q = torch.as_tensor(
            q, dtype=self._activation_logit.dtype, device=self._activation_logit.device
        )
        q = q / q.sum()
        with torch.no_grad():
            self._branching_logits[self._branching_rows[name]].copy_(
                torch.log(q.clamp(min=_FRACTION_EPS))
            )
        return self

    def set_lambda_twin(
        self, value: Optional[float], refinable: bool = False
    ) -> "ModelCollection":
        """Set the activation dispersion, fixed or refinable.

        Parameters
        ----------
        value : float or None
            Dispersion in ``[0, 1]``. ``None`` keeps the current value and only changes
            refinability.
        refinable : bool, optional
            If True, ``lambda_twin`` becomes ``sigmoid`` of a live parameter and joins
            :meth:`fraction_parameters`. Default False, which stores an exact float --
            the only way ``lambda_twin`` can be exactly 0.
        """
        if value is not None:
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"lambda_twin must lie in [0, 1]; got {value}")
            with torch.no_grad():
                self._lambda_logit.fill_(_logit(value))
        if refinable:
            self._lambda_fixed = None
            self._lambda_logit.requires_grad_(True)
        else:
            self._lambda_fixed = (
                float(value) if value is not None else float(self.lambda_twin)
            )
            self._lambda_logit.requires_grad_(False)
        return self

    # ------------------------------------------------------------------
    # Batched structure factors
    # ------------------------------------------------------------------

    def compute_component_fcalcs(
        self, hkl: torch.Tensor, recalc: bool = False
    ) -> torch.Tensor:
        """Per-base-model structure factors, stacked.

        Each base model is evaluated once, so a caller that needs several fraction
        mixtures of the same models pays for the structure factors once rather than
        once per mixture.

        Parameters
        ----------
        hkl : torch.Tensor
            Miller indices of shape (n_reflections, 3). These reach the models
            unchanged, so pass the *signed* indices when Bijvoet mates must be
            distinguished -- or go through
            :meth:`~torchref.io.datasets.collection.DatasetCollection.component_structure_factors`,
            which handles the convention.
        recalc : bool, optional
            Force recomputation rather than reusing each model's cached SF.

        Returns
        -------
        torch.Tensor
            Complex structure factors of shape ``(n_base_models, n_reflections)``.
        """
        return torch.stack(
            [m(hkl, recalc=recalc) for m in self._base_models], dim=0
        )

    def mix_component_fcalcs(
        self, component_fcalcs: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """Contract stacked per-component SFs with a weight matrix.

        ``weights [T, K] @ component_fcalcs [K, R] -> [T, R]``. Separated from
        :meth:`compute_component_fcalcs` because the same component stack is contracted
        with more than one weight matrix -- the fractions themselves, and any derivative
        of them with respect to a shared parameter.

        Parameters
        ----------
        component_fcalcs : torch.Tensor
            Complex SFs of shape ``(K, n_reflections)``.
        weights : torch.Tensor
            Real weights of shape ``(T, K)``.

        Returns
        -------
        torch.Tensor
            Complex SFs of shape ``(T, n_reflections)``.
        """
        return torch.einsum(
            "tk,kr->tr", weights.to(component_fcalcs.dtype), component_fcalcs
        )

    def compute_all_fcalc(
        self, hkl: torch.Tensor, recalc: bool = False
    ) -> torch.Tensor:
        """Mixed ``F_calc`` for every timepoint at once.

        Equivalent to calling each timepoint's ``forward`` in turn, but evaluates each
        shared base model once instead of once per timepoint. Rows follow
        :meth:`get_fractions_matrix`, i.e. insertion order.

        Parameters
        ----------
        hkl : torch.Tensor
            Miller indices of shape (n_reflections, 3); see
            :meth:`compute_component_fcalcs` on the index convention.
        recalc : bool, optional
            Force recomputation rather than reusing each model's cached SF.

        Returns
        -------
        torch.Tensor
            Complex SFs of shape ``(n_timepoints, n_reflections)``.
        """
        component_fcalcs = self.compute_component_fcalcs(hkl, recalc=recalc)
        return self.mix_component_fcalcs(
            component_fcalcs, self.get_fractions_matrix()
        )

    # ------------------------------------------------------------------
    # Freeze / unfreeze helpers
    # ------------------------------------------------------------------

    def freeze_all_fractions(self):
        """Exclude the population parameters from optimization.

        Acts on the shared activation and every branching row. There is nothing
        per-timepoint to freeze: one activation serves all of them, and the reference
        timepoint has no parameters at all.
        """
        self._activation_logit.requires_grad_(False)
        for row in self._branching_logits:
            row.requires_grad_(False)

    def unfreeze_all_fractions(self):
        """Include the population parameters in optimization.

        The dispersion ``lambda_twin`` is *not* affected; enable it explicitly with
        :meth:`set_lambda_twin` so it can never be refined by accident.
        """
        self._activation_logit.requires_grad_(True)
        for row in self._branching_logits:
            row.requires_grad_(True)

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
