"""
Verbosity-aware statistics reporting across refinement components.

Wrap a number with :func:`stat` at one of ``VERBOSITY_ESSENTIAL`` (0: major weights,
R-factors), ``_STANDARD`` (1: component weights and losses), ``_DETAILED`` (2: RMSDs,
per-restraint) or ``_DEBUG`` (3: internals), then :func:`filter_stats` to a level::

    filter_stats({'rwork': stat(0.20, VERBOSITY_ESSENTIAL)}, VERBOSITY_ESSENTIAL)
    # {'rwork': 0.20}

**Import side effect:** this module replaces the stdlib ``json.dumps``/``json.dump``
process-wide with wrappers defaulting ``cls`` to :class:`StatEntryEncoder`, so any call
without an explicit ``cls`` uses the custom encoder once this module is imported anywhere.
"""

import json
from dataclasses import dataclass
from typing import Any, Dict

# Verbosity levels
VERBOSITY_ESSENTIAL = 0  # Major weights (ADP, GEOM, Xray), R-factors
VERBOSITY_STANDARD = 1  # Component weights / component losses
VERBOSITY_DETAILED = 2  # Detailed stats / RMSDs / per-restraint info
VERBOSITY_DEBUG = 3  # All internal parameters for debugging


@dataclass
class StatEntry:
    """
    A statistics entry with value and verbosity level.

    JSON serializable - when serialized, only the value is written.

    Attributes
    ----------
    value : Any
        The statistic value.
    verbosity : int
        Verbosity level required to show this stat.
    """

    value: Any
    verbosity: int = VERBOSITY_STANDARD

    def __repr__(self):
        # repr shows only the value (not the dataclass form) for log readability.
        return f"{self.value}"

    def __json__(self):
        """The value, for libraries that look for a ``__json__`` hook.

        Dead weight for the stdlib, which never calls it -- :class:`StatEntryEncoder` does
        the real work in its ``default``.
        """
        return self.value


class StatEntryEncoder(json.JSONEncoder):
    """JSON encoder for ``StatEntry``, torch tensors and numpy scalars/arrays.

    Installed as the default by this module's import patch, so ``cls=`` is rarely needed.
    """

    def default(self, obj):
        if isinstance(obj, StatEntry):
            return obj.value
        try:
            import torch

            if isinstance(obj, torch.Tensor):
                return obj.tolist() if obj.numel() > 1 else obj.item()
        except ImportError:
            pass
        try:
            import numpy as np

            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
        except ImportError:
            pass
        return super().default(obj)


# Monkey-patch json module to use our encoder by default
_original_dumps = json.dumps
_original_dump = json.dump


def _patched_dumps(obj, *, cls=None, **kwargs):
    """json.dumps that automatically handles StatEntry objects."""
    if cls is None:
        cls = StatEntryEncoder
    return _original_dumps(obj, cls=cls, **kwargs)


def _patched_dump(obj, fp, *, cls=None, **kwargs):
    """json.dump that automatically handles StatEntry objects."""
    if cls is None:
        cls = StatEntryEncoder
    return _original_dump(obj, fp, cls=cls, **kwargs)


# Apply patches
json.dumps = _patched_dumps
json.dump = _patched_dump


def stat(value: Any, verbosity: int = VERBOSITY_STANDARD) -> StatEntry:
    """Tag ``value`` with the verbosity level at which it should be reported."""
    return StatEntry(value=value, verbosity=verbosity)


def filter_stats(stats: Dict, max_verbosity: int) -> Dict:
    """
    Keep entries at or below ``max_verbosity``, unwrapping :class:`StatEntry`.

    Recurses into nested dicts and drops any that end up empty. A value that is *not* a
    ``StatEntry`` has no level to test, so it is treated as ``VERBOSITY_STANDARD`` and kept
    at level 1 and above.

    Parameters
    ----------
    stats : dict
        Stats dictionary with StatEntry values or nested dicts.
    max_verbosity : int
        Maximum verbosity level to include.

    Returns
    -------
    dict
        Filtered stats holding raw values.
    """
    filtered = {}
    for key, val in stats.items():
        if isinstance(val, StatEntry):
            if val.verbosity <= max_verbosity:
                filtered[key] = val.value
        elif isinstance(val, dict):
            nested = filter_stats(val, max_verbosity)
            if nested:  # Only include non-empty dicts
                filtered[key] = nested
        else:
            # Raw values without StatEntry wrapper - include at STANDARD level
            if max_verbosity >= VERBOSITY_STANDARD:
                filtered[key] = val
    return filtered


def flatten_stats(stats: Dict, prefix: str = "") -> Dict[str, Any]:
    """
    Flatten a nested stats dict to dotted keys, unwrapping :class:`StatEntry`.

    Unlike :func:`filter_stats` this keeps every entry regardless of verbosity.

    Parameters
    ----------
    stats : dict
        Nested stats dictionary.
    prefix : str, optional
        Prefix for keys. Default is ''.
    """
    flat = {}
    for key, val in stats.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(val, StatEntry):
            flat[full_key] = val.value
        elif isinstance(val, dict):
            flat.update(flatten_stats(val, full_key))
        else:
            flat[full_key] = val
    return flat


def format_stats_table(stats: Dict, title: str = "", indent: int = 2) -> str:
    """
    Format a stats dictionary as a printable table.

    Expects raw values -- run :func:`filter_stats` first. Nesting is handled to three
    levels only; anything deeper is rendered with ``str()``.

    Parameters
    ----------
    stats : dict
        Stats dictionary (already filtered by verbosity).
    title : str, optional
        Title for the table.
    indent : int, optional
        Indentation spaces. Default is 2.
    """
    lines = []
    ind = " " * indent

    if title:
        lines.append(f"{ind}{title}")
        lines.append(f"{ind}{'-' * len(title)}")

    def format_value(val):
        if isinstance(val, float):
            if abs(val) < 0.001 and val != 0:
                return f"{val:.2e}"
            elif abs(val) >= 1000:
                return f"{val:.1f}"
            else:
                return f"{val:.4f}"
        return str(val)

    for key, val in stats.items():
        if isinstance(val, dict):
            lines.append(f"\n{ind}{key}:")
            for subkey, subval in val.items():
                if isinstance(subval, dict):
                    lines.append(f"{ind}  {subkey}:")
                    for k, v in subval.items():
                        lines.append(f"{ind}    {k}: {format_value(v)}")
                else:
                    lines.append(f"{ind}  {subkey}: {format_value(subval)}")
        else:
            lines.append(f"{ind}{key}: {format_value(val)}")

    return "\n".join(lines)


__all__ = [
    # Verbosity levels
    "VERBOSITY_ESSENTIAL",
    "VERBOSITY_STANDARD",
    "VERBOSITY_DETAILED",
    "VERBOSITY_DEBUG",
    # Stats classes and functions
    "StatEntry",
    "stat",
    "filter_stats",
    "flatten_stats",
    "format_stats_table",
]
