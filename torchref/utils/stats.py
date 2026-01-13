"""
Statistics Utilities for torchref.

Provides verbosity-aware statistics reporting across all refinement components.

Verbosity Levels
----------------
- VERBOSITY_ESSENTIAL (0): Major weights (ADP, GEOM, Xray targets), R-factors
- VERBOSITY_STANDARD (1): Component weights and component losses
- VERBOSITY_DETAILED (2): Detailed stats (RMSDs, per-restraint statistics)
- VERBOSITY_DEBUG (3): All internal parameters for debugging

Usage
-----
>>> from torchref.utils.stats import stat, filter_stats, VERBOSITY_STANDARD
>>> stats = {
...     'rwork': stat(0.20, VERBOSITY_ESSENTIAL),
...     'bond_rmsd': stat(0.015, VERBOSITY_DETAILED),
... }
>>> filter_stats(stats, VERBOSITY_ESSENTIAL)
{'rwork': 0.20}

# StatEntry is JSON serializable - just use json.dumps directly:
>>> import json
>>> json.dumps(stats, cls=StatEntryEncoder)
'{"rwork": 0.2, "bond_rmsd": 0.015}'
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
        return f"{self.value}"

    def __json__(self):
        """Return JSON-serializable representation (just the value)."""
        return self.value


class StatEntryEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles StatEntry and torch tensors.

    Usage:
        json.dumps(data, cls=StatEntryEncoder)
    """

    def default(self, obj):
        if isinstance(obj, StatEntry):
            return obj.value
        # Handle torch tensors
        try:
            import torch

            if isinstance(obj, torch.Tensor):
                return obj.tolist() if obj.numel() > 1 else obj.item()
        except ImportError:
            pass
        # Handle numpy arrays
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
    """
    Create a StatEntry with given value and verbosity.

    Parameters
    ----------
    value : Any
        The statistic value.
    verbosity : int, optional
        Verbosity level. Default is VERBOSITY_STANDARD.

    Returns
    -------
    StatEntry
        A statistics entry object.
    """
    return StatEntry(value=value, verbosity=verbosity)


def filter_stats(stats: Dict, max_verbosity: int) -> Dict:
    """
    Filter stats dictionary to only include entries at or below max_verbosity.

    Parameters
    ----------
    stats : dict
        Stats dictionary with StatEntry values or nested dicts.
    max_verbosity : int
        Maximum verbosity level to include.

    Returns
    -------
    dict
        Filtered stats with raw values (StatEntry unwrapped).
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
    Flatten nested stats dict into flat dict with dotted keys.

    Parameters
    ----------
    stats : dict
        Nested stats dictionary.
    prefix : str, optional
        Prefix for keys. Default is ''.

    Returns
    -------
    dict
        Flattened dictionary with dotted keys.
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
    Format stats dictionary as a printable table.

    Parameters
    ----------
    stats : dict
        Stats dictionary (already filtered by verbosity).
    title : str, optional
        Title for the table.
    indent : int, optional
        Indentation spaces. Default is 2.

    Returns
    -------
    str
        Formatted table string.
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
