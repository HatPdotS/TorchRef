"""
Refinement Logger - Separate logging concerns from refinement logic.

Provides verbosity-aware statistics recording and comparison for refinement workflows.
Integrates with LossState to capture and display refinement progress.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Pattern

import torch

from torchref.refinement.loss_state import LossState
from torchref.utils.stats import (
    VERBOSITY_STANDARD,
    filter_stats,
)


@dataclass
class Logger:
    """Verbosity-aware recording, comparison and display of a LossState's stats.

    Parameters
    ----------
    state : LossState
        The loss state to monitor.
    verbose : int
        0=essential, 1=standard, 2=detailed, 3=debug.
    pattern : str
        Regex filtering target names (``"xray.*"``, ``"geometry/bond"``); ``".*"``
        matches all. Per-call ``pattern=`` arguments override it.

    Examples
    --------
    ::

        logger = Logger(state, verbose=1)
        logger.record(label="before_xyz")
        ...                                   # run refinement
        logger.record(label="after_xyz")
        logger.compare(title="XYZ Refinement")
    """

    state: LossState
    verbose: int = VERBOSITY_STANDARD
    pattern: str = ".*"
    _records: List[Dict[str, Any]] = field(default_factory=list)
    _labels: Dict[str, int] = field(default_factory=dict)
    _compiled_pattern: Pattern = field(init=False, repr=False)

    def __post_init__(self):
        self._compiled_pattern = re.compile(self.pattern)

    def record(self, label: str = None) -> Dict[str, Any]:
        """Record every matching target's stats into history under ``label``.

        Aggregates the LossState first, so this forces a loss evaluation. Returns the
        recorded dict.
        """
        stats = {}

        with torch.no_grad():
            # Aggregate to ensure losses are computed
            self.state.aggregate()

            # Collect stats from each target (filtered by pattern)
            for name, target in self.state.targets.items():
                if not self._matches_pattern(name):
                    continue
                if hasattr(target, "stats"):
                    target_stats = target.stats()
                    if target_stats:
                        stats[name] = target_stats

            # Add loss values (filtered by pattern)
            stats["losses"] = {
                name: loss.item()
                for name, loss in self.state._losses.items()
                if self._matches_pattern(name)
            }

            # Add weights (filtered by pattern)
            stats["weights"] = {
                name: weight
                for name, weight in self.state.weights.items()
                if self._matches_pattern(name)
            }

            # Add group totals
            stats["group_totals"] = self.state.get_group_totals()

        # Store with optional label
        record_entry = {"stats": stats, "label": label}
        self._records.append(record_entry)
        if label:
            self._labels[label] = len(self._records) - 1

        return stats

    def compare(
        self,
        label_before: str = None,
        label_after: str = None,
        pattern: str = None,
        title: str = "Refinement Comparison",
    ) -> None:
        """Print a before/after table for two recorded states.

        Unlabelled arguments fall back to the last two records (``label_before`` to
        second-to-last, ``label_after`` to last); a missing label is silently treated
        as absent rather than raising. ``pattern`` overrides the instance filter.
        """
        # Get before record
        if label_before and label_before in self._labels:
            before = self._records[self._labels[label_before]]["stats"]
        elif len(self._records) >= 2:
            before = self._records[-2]["stats"]
        else:
            before = {}

        # Get after record
        if label_after and label_after in self._labels:
            after = self._records[self._labels[label_after]]["stats"]
        elif self._records:
            after = self._records[-1]["stats"]
        else:
            after = {}

        # Filter by verbosity
        before_filtered = filter_stats(before, self.verbose)
        after_filtered = filter_stats(after, self.verbose)

        # Apply regex pattern filter
        before_filtered = self._filter_by_pattern(before_filtered, pattern)
        after_filtered = self._filter_by_pattern(after_filtered, pattern)

        # Print formatted comparison table
        self._print_comparison(before_filtered, after_filtered, title)

    def current(self, pattern: str = None, title: str = "Current State") -> None:
        """Print the latest recorded state, recording one first if none exist."""
        if not self._records:
            self.record()

        stats = self._records[-1]["stats"]
        filtered = filter_stats(stats, self.verbose)
        filtered = self._filter_by_pattern(filtered, pattern)
        self._print_current(filtered, title)

    def get_record(self, label: str) -> Optional[Dict[str, Any]]:
        """The stats dict recorded under ``label``, or None if there is none."""
        if label in self._labels:
            return self._records[self._labels[label]]["stats"]
        return None

    def clear(self) -> None:
        """Clear all recorded history."""
        self._records.clear()
        self._labels.clear()

    @property
    def history(self) -> List[Dict[str, Any]]:
        """Access full recording history."""
        return self._records

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _matches_pattern(self, name: str, pattern: str = None) -> bool:
        """Whether ``name`` matches ``pattern``, or the instance pattern if None."""
        if pattern is None:
            return self._compiled_pattern.search(name) is not None
        return re.search(pattern, name) is not None

    def _filter_by_pattern(self, stats: Dict, pattern: str = None) -> Dict:
        """``stats`` keyed only by names matching the pattern, recursing into
        nested dicts and dropping any that end up empty."""
        filtered = {}
        for key, val in stats.items():
            if self._matches_pattern(key, pattern):
                filtered[key] = val
            elif isinstance(val, dict):
                # Recurse into nested dicts
                nested = self._filter_by_pattern(val, pattern)
                if nested:
                    filtered[key] = nested
        return filtered

    def _group_by_hierarchy(self, keys_and_values: Dict) -> Dict[str, Dict]:
        """Split flat keys on the first ``/`` into ``{group: {component: value}}``;
        keys without one land under ``"other"``."""
        groups = {}
        for key, val in keys_and_values.items():
            if "/" in key:
                group, component = key.split("/", 1)
            else:
                group, component = "other", key

            if group not in groups:
                groups[group] = {}
            groups[group][component] = val

        return groups

    def _print_comparison(self, before: Dict, after: Dict, title: str) -> None:
        """Format and print the before/after comparison table."""
        width = 68
        separator = "─" * width

        print(f"\n{separator}")
        print(f"  {title}")
        print(separator)

        # Get losses and weights
        before_losses = before.get("losses", {})
        after_losses = after.get("losses", {})
        before_weights = before.get("weights", {})
        after_weights = after.get("weights", {})
        before_totals = before.get("group_totals", {})
        after_totals = after.get("group_totals", {})

        # Group losses by hierarchy
        before_grouped = self._group_by_hierarchy(before_losses)
        after_grouped = self._group_by_hierarchy(after_losses)

        # Get all groups
        all_groups = sorted(set(before_grouped.keys()) | set(after_grouped.keys()))

        for group in all_groups:
            # Get group weight
            weight = after_weights.get(group, before_weights.get(group, 1.0))
            print(f"\n  {group} (weight: {weight:.4f})")
            print(f"  {'─' * (width - 4)}")
            print(f"    {'Metric':<24} {'Before':>10} {'After':>10} {'Change':>10}")

            before_group = before_grouped.get(group, {})
            after_group = after_grouped.get(group, {})
            all_components = sorted(set(before_group.keys()) | set(after_group.keys()))

            for comp in all_components:
                bval = before_group.get(comp)
                aval = after_group.get(comp)
                self._print_comparison_row(comp, bval, aval)

        # Print group totals
        if before_totals or after_totals:
            print(f"\n  Group Totals")
            print(f"  {'─' * (width - 4)}")
            print(f"    {'Group':<24} {'Before':>10} {'After':>10} {'Change':>10}")

            all_total_groups = sorted(
                set(before_totals.keys()) | set(after_totals.keys())
            )
            for group in all_total_groups:
                bval = before_totals.get(group)
                aval = after_totals.get(group)
                self._print_comparison_row(group, bval, aval)

        print(separator)

    def _print_current(self, stats: Dict, title: str) -> None:
        """Format and print a single state's table."""
        width = 68
        separator = "─" * width

        print(f"\n{separator}")
        print(f"  {title}")
        print(separator)

        # Get losses, weights, and totals
        losses = stats.get("losses", {})
        weights = stats.get("weights", {})
        totals = stats.get("group_totals", {})

        # Group losses by hierarchy
        grouped = self._group_by_hierarchy(losses)

        for group in sorted(grouped.keys()):
            # Get group weight
            weight = weights.get(group, 1.0)
            print(f"\n  {group} (weight: {weight:.4f})")
            print(f"  {'─' * (width - 4)}")

            for comp, val in sorted(grouped[group].items()):
                formatted = self._format_value(val)
                print(f"    {comp:<32} {formatted:>10}")

        # Print group totals
        if totals:
            print(f"\n  Group Totals")
            print(f"  {'─' * (width - 4)}")

            for group, val in sorted(totals.items()):
                formatted = self._format_value(val)
                print(f"    {group:<32} {formatted:>10}")

        print(separator)

    def _print_comparison_row(
        self, label: str, before: float, after: float
    ) -> None:
        """Print one row; a ``None`` on either side prints "-" and no change."""
        bstr = self._format_value(before) if before is not None else "-"
        astr = self._format_value(after) if after is not None else "-"

        if before is not None and after is not None:
            change = after - before
            cstr = self._format_value(change, show_sign=True)
        else:
            cstr = "-"

        print(f"    {label:<24} {bstr:>10} {astr:>10} {cstr:>10}")

    def _format_value(self, val: float, show_sign: bool = False) -> str:
        """Fixed-width numeric string, switching to exponential below 1e-4;
        ``show_sign`` prefixes positives with ``+``. ``None`` renders as "-"."""
        if val is None:
            return "-"

        # Handle very small values
        if abs(val) < 0.0001 and val != 0:
            if show_sign and val > 0:
                return f"+{val:.2e}"
            return f"{val:.2e}"
        # Handle large values
        elif abs(val) >= 1000:
            if show_sign and val > 0:
                return f"+{val:.1f}"
            return f"{val:.1f}"
        # Standard precision
        else:
            if show_sign and val > 0:
                return f"+{val:.4f}"
            return f"{val:.4f}"


__all__ = ["Logger"]
