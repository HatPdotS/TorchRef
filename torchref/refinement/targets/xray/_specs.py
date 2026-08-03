"""The X-ray target taxonomy, as data.

One row per selectable ``--xray-mode``. This table is the **single source of truth**:
:func:`~torchref.refinement.targets.xray.factory.create_xray_target`
dispatches from it and the CLI derives its ``choices=`` from it. Before this existed the
mode list was maintained by hand in two places -- an ``if/elif`` chain in the factory and
a literal list in ``cli/refine.py`` -- which drifted (``nll_beta`` reached the factory but
never the CLI, and ``bhattacharyya_ensemble`` was reachable only from Python).

Follows the house pattern of :mod:`torchref.utils.backends`: frozen dataclass rows plus a
table object that checks its own invariants at import time, so a malformed row fails the
build rather than a refinement. Enums are deliberately not used -- string literals
validated against a table are the established convention here.

## The sigma_A family

Four targets, each a choice of (distribution) x (variance) x (mean):

=============  ========================  ============  ================================
mode           distribution              mean          variance
=============  ========================  ============  ================================
``nll``        Gaussian on |F|           ``|F_c|``     ``sigma_obs**2``
``nll_beta``   Gaussian on |F|           ``|F_c|``     ``eps*beta`` -> amplitude var.
``ml``         Rice / folded normal      ``a*|F_c|``   ``eps*beta``
``ml_noalpha`` Rice / folded normal      ``|F_c|``     ``eps*beta``
``ml_full``    Rice (x) Gaussian, marg.  ``a*|F_c|``   ``eps*beta_model`` + sigma_obs
=============  ========================  ============  ================================

``ml`` is the default and **centres on** ``alpha*|F_calc|``.

**On the full 767-structure AF-start benchmark, alpha is a small net LOSS.** Paired against
``ml_noalpha`` (n=766, against a clean same-config null at p=0.74):

===============  ==============  =========
quantity         alpha - no alpha  p
===============  ==============  =========
R_work           **+0.00200**     0.0000
R_free           **+0.00020**     0.0000
work-free gap    -0.00180         0.0000
===============  ==============  =========

The narrower gap is **not** evidence of better generalisation: R_work degrades ten times
more than R_free does, so the gap closes because the fit to the work data gets worse, not
because prediction improves. R_free -- the metric that matters -- is *significantly worse*
with alpha, by a small but consistent margin (357 losses to 312 wins, and the losing
magnitudes are systematically larger).

An earlier 50-structure screen read this the other way (R_free +0.00030 at p=0.47, gap
-0.00105 at p=0.008) and was written up here as "less overfitting for equal prediction".
That conclusion did not survive 15x the data: at n=50 the R_free cost was real but below
the resolution floor, and the R_work channel -- which is what identifies the mechanism --
was not examined. **Do not re-derive the optimistic reading from the small-n numbers.**

``ml_noalpha`` is that same likelihood with the coupling fixed at 1. It exists because a
**scale** fit is a nuisance-magnitude fit in which ``alpha`` is degenerate with the very
per-bin scale being optimised, so pinning it there is the correct choice rather than an
approximation. It is also, exactly, the likelihood
:meth:`ScalerBase.refine_lbfgs`'s ``scale_target='sigmaa'`` evaluates.

**The ``mean`` column describes the LIKELIHOOD, never the estimator.** The estimator behind
``beta`` fits ``alpha`` and ``beta`` *jointly* for every row, including the rows whose mean
is ``|F_c|``, because pinning the mean during the fit biases ``sigma_A`` +0.035 high. Read
``mean = |F_c|`` as "this row does not *centre* on alpha", not as "alpha is absent here".

Nothing else in the table varies by row: every target is fitted by the same estimator, with
the same ``sigma_obs`` and the same shrinkage, so a comparison between two rows measures the
likelihood and nothing else. That was not true before -- ``ml`` and ``ml_full`` used to get
differently configured estimators, which confounded every comparison between them.

## What the full benchmark settled about ``ml_full``

The ``sigma_obs`` marginalisation -- ``ml_full``'s distinguishing ingredient -- **is worth
nothing on R_free**: ``ml_full`` vs ``ml`` gives ``dR_free = +0.00000`` (p=0.055) on 766
structures, and on 50 structures it gave ``+0.00005`` at *both* alpha settings (p=0.914 and
p=0.417). It shows the same signature alpha does, one order smaller: ``dR_work = +0.00030``,
``dgap = -0.00030``, i.e. it damps the work fit rather than improving prediction.

So ranked by R_free on 766 structures, ``ml_noalpha`` is best, and both ``ml`` (+0.00020,
p=0.0000) and ``ml_full`` (+0.00015, p=0.0001) are significantly behind it. The margins are
small -- 2e-4 -- but they are significant against a null that returns p=0.74, and they are
consistent in sign.

``ml_full`` is kept as the most principled statement of the model available (the two error
kinds are genuinely different objects and it treats them so), and because the null is a null
*in this regime*: ``S2/beta_model`` is ~0.087 on AF-start data against ~0.247 on deposited
coordinates, so the term it adds is near its weakest here. It is not the default, and it
costs a 32-node quadrature inside every loss evaluation.

## Why there is no Rice-with-sigma_obs row

A ``rice`` mode used to exist and was removed. The Rice distribution arises from a 2-D
isotropic Gaussian in the complex plane marginalised over phase, so supplying
``sigma_obs`` as its ``Sigma`` asserts an isotropic *complex* error. ``sigma_obs`` is a
1-DOF amplitude error carrying no phase information, so there is no regime in which the
pairing is correct -- and empirically it was the worst-behaved target measured, destroying
geometry (bond RMSZ 28.0 where every other target sat near 1.3). Model error, which
*does* have a phase component, is what belongs in a Rice ``Sigma``; that is ``beta``.
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
        The class. **One class per row**, checked by :class:`XrayTargetTable` below.

        Until 2026-08 this table also carried ``family`` / ``distribution`` / ``variance`` /
        ``mean`` / ``needs_estimator``, and four rows shared one class that read those fields
        at runtime to pick a likelihood, a variance field and a mean. Each row is now its own
        class, so dispatch is ``spec.target_cls(**kwargs)`` and those columns had no readers
        left. Keeping them would have left import-time invariants asserting *documentation*,
        which looks load-bearing and is not. The taxonomy they described is in this module's
        docstring, where a description belongs.
    doc
        One line, surfaced in ``--help``.
    aliases
        Retired spellings kept working. Resolving one emits a ``DeprecationWarning``. No row
        carries one at present -- ``ml_sigmaa`` and ``gaussian`` were removed -- so the
        machinery is exercised by a purpose-built table in the tests rather than by a live
        deprecated name.
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
            doc="As 'ml' with the Luzzati mean coupling fixed at 1. Best R_free of the "
            "five on the AF-start benchmark.",
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
