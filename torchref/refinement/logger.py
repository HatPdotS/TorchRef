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
    """
    Refinement logging with verbosity-aware stat reporting.

    Integrates with LossState to record, compare, and display refinement stats.
    Supports regex-based filtering of target names.

    Parameters
    ----------
    state : LossState
        The loss state to monitor.
    verbose : int
        Verbosity level (0=essential, 1=standard, 2=detailed, 3=debug).
    pattern : str
        Regex pattern for filtering target names. Default ".*" matches all.
        Examples: "xray.*" for X-ray targets only, "geometry/bond" for specific target.

    Examples
    --------
    Basic usage::

        from torchref.refinement import LossState, Logger

        state = LossState(device=device)
        state.register_target("xray/work", xray_target)
        state.register_target("geometry/bond", bond_target)

        logger = Logger(state, verbose=1)

        # Record before refinement
        logger.record(label="before_xyz")

        # ... run refinement ...

        # Record after and compare
        logger.record(label="after_xyz")
        logger.compare(title="XYZ Refinement")

    Filtering by pattern::

        # Show only X-ray targets
        logger.current(pattern="xray.*")

        # Create a logger that only tracks geometry
        geom_logger = Logger(state, pattern="geometry.*")
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
        """
        Record current refinement state.

        Gathers stats from all targets in LossState, stores in history.
        Uses the instance's pattern filter.

        Parameters
        ----------
        label : str, optional
            Label for this record (e.g., "before_xyz", "after_xyz").

        Returns
        -------
        dict
            The recorded stats dictionary.
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
        """
        Print comparison between two recorded states.

        If labels not provided, compares last two records.

        Parameters
        ----------
        label_before : str, optional
            Label of "before" state. Default: second-to-last record.
        label_after : str, optional
            Label of "after" state. Default: last record.
        pattern : str, optional
            Regex to filter which targets to display. Default: use instance pattern.
        title : str, optional
            Title for the comparison output. Default: "Refinement Comparison".
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
        """
        Print the current refinement state.

        Uses latest recorded state, or records new one if none exist.

        Parameters
        ----------
        pattern : str, optional
            Regex to filter which targets to display. Default: use instance pattern.
        title : str, optional
            Title for the output. Default: "Current State".
        """
        if not self._records:
            self.record()

        stats = self._records[-1]["stats"]
        filtered = filter_stats(stats, self.verbose)
        filtered = self._filter_by_pattern(filtered, pattern)
        self._print_current(filtered, title)

    def get_record(self, label: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific recorded state by label.

        Parameters
        ----------
        label : str
            The label to look up.

        Returns
        -------
        dict or None
            The recorded stats dictionary, or None if label not found.
        """
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
        """
        Check if target name matches the filter pattern.

        Parameters
        ----------
        name : str
            The name to check.
        pattern : str, optional
            Pattern to use. If None, uses instance's compiled pattern.

        Returns
        -------
        bool
            True if name matches the pattern.
        """
        if pattern is None:
            return self._compiled_pattern.search(name) is not None
        return re.search(pattern, name) is not None

    def _filter_by_pattern(self, stats: Dict, pattern: str = None) -> Dict:
        """
        Filter stats dict by regex pattern on keys.

        Parameters
        ----------
        stats : dict
            Stats dictionary to filter.
        pattern : str, optional
            Pattern to use. If None, uses instance pattern.

        Returns
        -------
        dict
            Filtered dictionary.
        """
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
        """
        Group flat keys by hierarchical prefix.

        Parameters
        ----------
        keys_and_values : dict
            Flat dictionary with potentially hierarchical keys.

        Returns
        -------
        dict
            Grouped dictionary: {"xray": {"work": 3.2, "test": 3.4}, ...}

        Examples
        --------
        Input: {"xray/work": 3.2, "xray/test": 3.4, "geometry/bond": 0.02}
        Output: {"xray": {"work": 3.2, "test": 3.4}, "geometry": {"bond": 0.02}}
        """
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
        """
        Format and print before/after comparison table.

        Parameters
        ----------
        before : dict
            Stats from before state.
        after : dict
            Stats from after state.
        title : str
            Title for the comparison.
        """
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
        """
        Format and print current state.

        Parameters
        ----------
        stats : dict
            Stats dictionary.
        title : str
            Title for the output.
        """
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
        """
        Print a single comparison row.

        Parameters
        ----------
        label : str
            Row label.
        before : float or None
            Value before.
        after : float or None
            Value after.
        """
        bstr = self._format_value(before) if before is not None else "-"
        astr = self._format_value(after) if after is not None else "-"

        if before is not None and after is not None:
            change = after - before
            cstr = self._format_value(change, show_sign=True)
        else:
            cstr = "-"

        print(f"    {label:<24} {bstr:>10} {astr:>10} {cstr:>10}")

    def _format_value(self, val: float, show_sign: bool = False) -> str:
        """
        Format a numeric value for display.

        Parameters
        ----------
        val : float
            Value to format.
        show_sign : bool
            If True, show + sign for positive values.

        Returns
        -------
        str
            Formatted string.
        """
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
