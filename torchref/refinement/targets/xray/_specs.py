"""The X-ray target taxonomy, as data: one row per selectable ``--xray-mode``.

:data:`XRAY_TARGETS` is the single source of truth:
:func:`~torchref.refinement.targets.xray.factory.create_xray_target` dispatches from it and
``torchref.refine`` takes its ``--xray-mode`` choices from it. Frozen dataclass rows
plus a table that checks its own invariants at import, following
:mod:`torchref.utils.backends`. Each row's ``observable`` names the column it fits,
``F_obs`` or (``nll_i``) ``I_obs``; see
:mod:`torchref.refinement.targets.xray.observable`.

The amplitude likelihoods, as (distribution) x (mean) x (variance):

==============  ========================  ============  ================================
mode            distribution              mean          variance
==============  ========================  ============  ================================
``nll``         Gaussian on ``|F|``       ``|F_c|``     ``sigma_obs**2``
``nll_beta``    Gaussian on ``|F|``       ``|F_c|``     ``eps*beta`` -> amplitude var.
``ml``          Rice / folded normal      ``a*|F_c|``   ``eps*beta``
``ml_noalpha``  Rice / folded normal      ``|F_c|``     ``eps*beta``
``ml_full``     Rice (x) Gaussian, marg.  ``a*|F_c|``   ``eps*beta_model`` + sigma_obs
==============  ========================  ============  ================================

``mean`` is where the likelihood centres: the estimator behind ``beta`` fits ``alpha``
for every row that has one, as :mod:`torchref.refinement.model_error_estimation.sigma_a`
explains. :mod:`torchref.base.targets.xray_likelihoods` says why no row pairs Rice with
``sigma_obs``.

A row is not thereby a scale-fit objective:
:meth:`~torchref.scaling.scaler_base.ScalerBase.refine_lbfgs` builds its objective from
this table but accepts only :data:`~torchref.scaling.scaler_base.SCALE_TARGETS`
(``nll``, ``ml_noalpha`` and ``ls``), and says why no ``alpha``-centred row may fit a
scale.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple
import warnings

from .base import XrayTarget
from .least_squares import LeastSquaresXrayTarget, UnitWeightK1XrayTarget
from .ml import MLXrayTarget
from .ml_full import MLFullXrayTarget
from .ml_noalpha import MLNoAlphaXrayTarget
from .nll import NLLXrayTarget
from .nll_beta import NLLBetaXrayTarget
from .observable import IntensityObservableMixin, NLLIntensityXrayTarget  # noqa: F401

#: The mode built when none is given.
DEFAULT_XRAY_MODE = "ml"


@dataclass(frozen=True)
class XrayTargetSpec:
    """One selectable x-ray target mode: a name, and the class that implements it.

    Attributes
    ----------
    name
        The ``--xray-mode`` value.
    target_cls
        The class. **One class per row**, checked by :class:`XrayTargetTable` below, so
        dispatch is just ``spec.target_cls(**kwargs)`` with nothing to branch on.
    doc
        One line, surfaced in ``--help``.
    aliases
        Retired spellings kept working; resolving one emits a ``DeprecationWarning``. No row
        carries one at present, so the tests exercise this with their own table.
    observable
        Which measured column the row fits: ``"amplitude"`` or ``"intensity"``. Declarative
        rather than a constructor flag, because it is a property of the row -- see
        :mod:`torchref.refinement.targets.xray.observable`. Checked here against the class,
        so a spec and its implementation cannot disagree.
    """

    name: str
    target_cls: type
    doc: str
    aliases: Tuple[str, ...] = ()
    observable: str = "amplitude"

    def __post_init__(self):
        if not (isinstance(self.target_cls, type) and issubclass(self.target_cls, XrayTarget)):
            raise TypeError(
                f"{self.name}: target_cls {self.target_cls!r} is not an XrayTarget subclass"
            )
        if self.observable not in ("amplitude", "intensity"):
            raise ValueError(
                f"{self.name}: observable must be 'amplitude' or 'intensity', "
                f"got {self.observable!r}"
            )
        # The class declares its own observable (the mixin sets it); the spec must agree.
        # Otherwise a row could advertise intensities while reading `sub.F`, which no test
        # downstream of here would notice -- the loss would simply be wrong by 2|F|.
        declared = getattr(self.target_cls, "observable", "amplitude")
        if declared != self.observable:
            raise ValueError(
                f"{self.name}: spec says observable={self.observable!r} but "
                f"{self.target_cls.__name__} says {declared!r}"
            )


@dataclass(frozen=True)
class XrayTargetTable:
    """The taxonomy, with uniqueness checked at import."""

    specs: Tuple[XrayTargetSpec, ...]
    _by_name: Dict[str, XrayTargetSpec] = field(init=False, repr=False, default=None)

    def __post_init__(self):
        lookup: Dict[str, XrayTargetSpec] = {}
        for spec in self.specs:
            for key in (spec.name,) + tuple(spec.aliases):
                if key in lookup:
                    raise ValueError(
                        f"duplicate x-ray target name/alias {key!r} "
                        f"({lookup[key].name} and {spec.name})"
                    )
                lookup[key] = spec
        by_cls: Dict[type, XrayTargetSpec] = {}
        for spec in self.specs:
            if spec.target_cls in by_cls:
                raise ValueError(
                    f"{spec.name} and {by_cls[spec.target_cls].name} both map to "
                    f"{spec.target_cls.__name__}. One class per mode is the invariant this "
                    f"table exists to enforce: a class serving two modes has to branch on "
                    f"something at runtime."
                )
            by_cls[spec.target_cls] = spec
        object.__setattr__(self, "_by_name", lookup)

    @property
    def names(self) -> Tuple[str, ...]:
        """Canonical names, in table order. Drives the CLI's ``choices=``."""
        return tuple(s.name for s in self.specs)

    def by_name(self, name: str) -> XrayTargetSpec:
        """Resolve a mode name or alias, warning on retired spellings."""
        spec = self._by_name.get(name)
        if spec is None:
            raise ValueError(
                f"Unknown X-ray target mode: {name!r}. "
                f"Available: {', '.join(self.names)}"
            )
        if name != spec.name:
            warnings.warn(
                f"X-ray target mode {name!r} is deprecated; use {spec.name!r}.",
                DeprecationWarning,
                stacklevel=3,
            )
        return spec


XRAY_TARGETS = XrayTargetTable(
    specs=(
        XrayTargetSpec(
            name="ml",
            target_cls=MLXrayTarget,
            doc="Read MLF, variance epsilon*beta, conditional mean alpha*|F_calc| "
            "(default).",
        ),
        XrayTargetSpec(
            name="ml_noalpha",
            target_cls=MLNoAlphaXrayTarget,
            doc="As 'ml' with the Luzzati mean coupling fixed at 1.",
        ),
        XrayTargetSpec(
            name="ml_full",
            target_cls=MLFullXrayTarget,
            doc="Full-form MLF: marginalises the unknown error-free amplitude, so the "
            "observation error enters as an amplitude-only Gaussian. ~4x the cost.",
        ),
        XrayTargetSpec(
            name="nll_beta",
            target_cls=NLLBetaXrayTarget,
            doc="Gaussian amplitude NLL on ml's model-error variance -- the large-signal "
            "limit of 'ml'. Diagnostic: isolates the variance model from the shape.",
        ),
        XrayTargetSpec(
            name="nll",
            target_cls=NLLXrayTarget,
            doc="Gaussian amplitude NLL weighted by the experimental sigma only. No "
            "model-error term, so it does not control overfitting.",
        ),
        XrayTargetSpec(
            name="nll_i",
            target_cls=NLLIntensityXrayTarget,
            observable="intensity",
            doc="Gaussian NLL on the observed INTENSITIES weighted by sigma(I). As 'nll' "
            "but skips the French-Wilson conversion, which reshapes the weak tail.",
        ),
        XrayTargetSpec(
            name="ls",
            target_cls=LeastSquaresXrayTarget,
            doc="Least squares with unit weights; the scaler owns the overall scale.",
        ),
        XrayTargetSpec(
            name="ls_wunit_k1",
            target_cls=UnitWeightK1XrayTarget,
            doc="Phenix-style least squares: unit weights and a single global scale "
            "recomputed every gradient call (bypasses the scaler).",
        ),
    )
)
