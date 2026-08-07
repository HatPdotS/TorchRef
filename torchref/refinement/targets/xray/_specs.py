"""The X-ray target taxonomy, as data.

One row per selectable ``--xray-mode``. This table is the **single source of truth**:
:func:`~torchref.refinement.targets.xray.factory.create_xray_target` dispatches from it and
the CLI derives its ``choices=`` from it, so a mode added in one place cannot go missing
from the other. Frozen dataclass rows plus a table that checks its own invariants at import,
following :mod:`torchref.utils.backends`; string literals validated against a table, not
enums, is the house convention.

## The sigma_A family

Each is a choice of (distribution) x (variance) x (mean):

=============  ========================  ============  ================================
mode           distribution              mean          variance
=============  ========================  ============  ================================
``nll``        Gaussian on |F|           ``|F_c|``     ``sigma_obs**2``
``nll_beta``   Gaussian on |F|           ``|F_c|``     ``eps*beta`` -> amplitude var.
``ml``         Rice / folded normal      ``a*|F_c|``   ``eps*beta``
``ml_noalpha`` Rice / folded normal      ``|F_c|``     ``eps*beta``
``ml_full``    Rice (x) Gaussian, marg.  ``a*|F_c|``   ``eps*beta_model`` + sigma_obs
=============  ========================  ============  ================================

``ml`` is the default and centres on ``alpha*|F_calc|``; ``ml_noalpha`` is the same
likelihood with the coupling pinned at 1, which is the correct choice for a **scale** fit,
where ``alpha`` is degenerate with the per-bin scale being optimised -- an ``alpha``-centred
row drives the scale to absorb ``1/alpha`` and inflates every R-factor computed from
``k*|F_calc|``. That constraint is enforced, not advised:
:data:`~torchref.scaling.scaler_base.SCALE_TARGETS` admits only ``nll`` and ``ml_noalpha``,
and :meth:`ScalerBase.refine_lbfgs` builds its objective from *this* table. ``ml_full`` is
the most principled row (it treats the two error kinds as the different objects they are) but
costs a 32-node quadrature per loss evaluation, so it is not the default.

**The ``mean`` column describes the LIKELIHOOD, never the estimator.** The estimator behind
``beta`` fits ``alpha`` and ``beta`` *jointly* for every row, including rows whose mean is
``|F_c|``, because pinning the mean during the fit biases ``sigma_A`` high. Read
``mean = |F_c|`` as "this row does not *centre* on alpha", not "alpha is absent here".

Nothing else varies by row: same estimator, same ``sigma_obs``, same shrinkage, so a
comparison between two rows measures the likelihood and nothing else.

There is deliberately no Rice-with-``sigma_obs`` row; see
:mod:`torchref.base.targets.xray_likelihoods` for why the pairing is never correct.
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
    """

    name: str
    target_cls: type
    doc: str
    aliases: Tuple[str, ...] = ()

    def __post_init__(self):
        if not (isinstance(self.target_cls, type) and issubclass(self.target_cls, XrayTarget)):
            raise TypeError(
                f"{self.name}: target_cls {self.target_cls!r} is not an XrayTarget subclass"
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
                    f"something at runtime, which is what the 2026-08 refactor removed."
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
