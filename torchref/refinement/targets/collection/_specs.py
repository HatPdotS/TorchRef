"""The collection X-ray target taxonomy, as data.

The multi-dataset mirror of :mod:`torchref.refinement.targets.xray._specs`, with the same
invariants checked the same way at import: unique names, and **one class per row**, so
dispatch is ``spec.target_cls(**kwargs)`` with nothing to branch on.

Four rows over two axes -- what the loss compares (a difference from the collection mean,
or each dataset absolutely) and in which observable:

====================  ===========  ==============================================
row                   observable   compares
====================  ===========  ==============================================
``difference``        amplitude    ``F_i - F_mean`` against the model's own spread
``difference_i``      intensity    the same, in intensities
``two_moment``        intensity    ``|F(alpha)|^2 + sigma_alpha^2 |dF|^2``
``ml``                amplitude    each dataset absolutely, at a shared Luzzati beta
====================  ===========  ==============================================

The two difference rows are both offered rather than one being chosen. Which is better is a
property of a dataset's signal-to-noise -- amplitudes keep the loss in the same space as the
output DED coefficients, intensities avoid the French-Wilson posterior reshaping the weak
tail the signal lives in -- and that is not something to settle once in a library.

``ml`` is the absolute channel, for the scenarios that are not difference refinement: with
K free base models a purely relative loss leaves the overall level unconstrained. It
replaces a hand-rolled Rice target that set ``beta = sigma_obs**2``, i.e. exactly the
sigma_obs-in-a-Rice-Sigma pairing that
:mod:`torchref.base.targets.xray_likelihoods` documents as never correct and that the
single-dataset table deliberately does not offer.

There is no intensity ``ml`` row, for the same reason the single-dataset table has none:
Rice and the folded normal are distributions *of an amplitude*, and the intensity analogue
is the exponential / chi-square_1 Wilson distribution -- a different primitive rather than a
different variance.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple

from .base import CollectionXrayTarget
from .intensity import CollectionTwoMomentIntensityTarget
from .xray import (
    CollectionDifferenceIntensityTarget,
    CollectionDifferenceTarget,
    CollectionMLTarget,
)


@dataclass(frozen=True)
class CollectionXrayTargetSpec:
    """One selectable collection x-ray target: a name, and the class implementing it.

    Attributes
    ----------
    name
        The row name, as used in ``LossState`` keys (``xray/<name>``).
    target_cls
        The class. **One class per row**, checked by :class:`CollectionXrayTargetTable`.
    doc
        One line, for ``--help`` and the loss breakdown.
    observable
        ``"amplitude"`` or ``"intensity"``. Checked against the class, so a spec and its
        implementation cannot disagree -- a row advertising intensities while reading
        amplitudes would be wrong by ``2|F|``, which is resolution-dependent and so reads
        as a scale or B error rather than as a bug.
    """

    name: str
    target_cls: type
    doc: str
    observable: str = "amplitude"

    def __post_init__(self):
        if not (
            isinstance(self.target_cls, type)
            and issubclass(self.target_cls, CollectionXrayTarget)
        ):
            raise TypeError(
                f"{self.name}: target_cls {self.target_cls!r} is not a "
                f"CollectionXrayTarget subclass"
            )
        if self.observable not in ("amplitude", "intensity"):
            raise ValueError(
                f"{self.name}: observable must be 'amplitude' or 'intensity', "
                f"got {self.observable!r}"
            )
        declared = getattr(self.target_cls, "observable", "amplitude")
        if declared != self.observable:
            raise ValueError(
                f"{self.name}: spec says observable={self.observable!r} but "
                f"{self.target_cls.__name__} says {declared!r}"
            )


@dataclass(frozen=True)
class CollectionXrayTargetTable:
    """The taxonomy, with uniqueness checked at import."""

    specs: Tuple[CollectionXrayTargetSpec, ...]
    _by_name: Dict[str, CollectionXrayTargetSpec] = field(
        init=False, repr=False, default=None
    )

    def __post_init__(self):
        lookup: Dict[str, CollectionXrayTargetSpec] = {}
        for spec in self.specs:
            if spec.name in lookup:
                raise ValueError(f"duplicate collection x-ray target name {spec.name!r}")
            lookup[spec.name] = spec
        by_cls: Dict[type, CollectionXrayTargetSpec] = {}
        for spec in self.specs:
            if spec.target_cls in by_cls:
                raise ValueError(
                    f"{spec.name} and {by_cls[spec.target_cls].name} both map to "
                    f"{spec.target_cls.__name__}. One class per row is the invariant this "
                    f"table exists to enforce: a class serving two rows has to branch on "
                    f"something at runtime."
                )
            by_cls[spec.target_cls] = spec
        object.__setattr__(self, "_by_name", lookup)

    @property
    def names(self) -> Tuple[str, ...]:
        """Canonical names, in table order."""
        return tuple(s.name for s in self.specs)

    def by_name(self, name: str) -> CollectionXrayTargetSpec:
        spec = self._by_name.get(name)
        if spec is None:
            raise ValueError(
                f"Unknown collection X-ray target: {name!r}. "
                f"Available: {', '.join(self.names)}"
            )
        return spec


COLLECTION_XRAY_TARGETS = CollectionXrayTargetTable(
    specs=(
        CollectionXrayTargetSpec(
            name="difference",
            target_cls=CollectionDifferenceTarget,
            doc="Gaussian on each dataset's amplitude difference from the collection "
            "mean, with the dataset/mean covariance propagated.",
        ),
        CollectionXrayTargetSpec(
            name="difference_i",
            target_cls=CollectionDifferenceIntensityTarget,
            observable="intensity",
            doc="As 'difference' but on intensities, skipping the French-Wilson "
            "conversion that reshapes the weak tail.",
        ),
        CollectionXrayTargetSpec(
            name="two_moment",
            target_cls=CollectionTwoMomentIntensityTarget,
            observable="intensity",
            doc="Merged intensities as |F(alpha)|^2 + sigma_alpha^2 |dF|^2, accounting "
            "for crystal-to-crystal spread in activation.",
        ),
        CollectionXrayTargetSpec(
            name="ml",
            target_cls=CollectionMLTarget,
            doc="Read MLF per dataset at one shared Luzzati beta fitted on the pooled "
            "free reflections. The absolute channel.",
        ),
    )
)
