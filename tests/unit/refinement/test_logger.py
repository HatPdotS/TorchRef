"""
Unit tests for torchref.refinement.logger

Tests the Logger class for verbosity-aware refinement logging.
"""

import pytest
import torch


class TestLoggerCreation:
    """Tests for Logger creation and initialization."""

    @pytest.mark.unit
    def test_logger_creation(self):
        """Test Logger can be created with a LossState."""
        from torchref.refinement.logger import Logger
        from torchref.refinement.loss_state import LossState

        state = LossState()
        logger = Logger(state=state)

        assert logger.state is state
        assert logger.verbose == 1  # VERBOSITY_STANDARD
        assert logger.pattern == ".*"
        assert logger.history == []

    @pytest.mark.unit
    def test_logger_with_custom_verbosity(self):
        """Test Logger creation with custom verbosity."""
        from torchref.refinement.logger import Logger
        from torchref.refinement.loss_state import LossState
        from torchref.utils.stats import VERBOSITY_DETAILED

        state = LossState()
        logger = Logger(state=state, verbose=VERBOSITY_DETAILED)

        assert logger.verbose == VERBOSITY_DETAILED

    @pytest.mark.unit
    def test_logger_with_custom_pattern(self):
        """Test Logger creation with custom filter pattern."""
        from torchref.refinement.logger import Logger
        from torchref.refinement.loss_state import LossState

        state = LossState()
        logger = Logger(state=state, pattern="xray.*")

        assert logger.pattern == "xray.*"


class TestPatternMatching:
    """Tests for regex pattern matching."""

    @pytest.mark.unit
    def test_matches_pattern_default(self):
        """Test default pattern matches everything."""
        from torchref.refinement.logger import Logger
        from torchref.refinement.loss_state import LossState

        state = LossState()
        logger = Logger(state=state)

        assert logger._matches_pattern("xray/work")
        assert logger._matches_pattern("geometry/bond")
        assert logger._matches_pattern("anything")

    @pytest.mark.unit
    def test_matches_pattern_xray(self):
        """Test xray pattern matches xray targets."""
        from torchref.refinement.logger import Logger
        from torchref.refinement.loss_state import LossState

        state = LossState()
        logger = Logger(state=state, pattern="xray.*")

        assert logger._matches_pattern("xray/work")
        assert logger._matches_pattern("xray/test")
        assert not logger._matches_pattern("geometry/bond")

    @pytest.mark.unit
    def test_matches_pattern_custom(self):
        """Test custom pattern matching."""
        from torchref.refinement.logger import Logger
        from torchref.refinement.loss_state import LossState

        state = LossState()
        logger = Logger(state=state)

        # Test with explicit pattern override
        assert logger._matches_pattern("geometry/bond", pattern="geometry.*")
        assert logger._matches_pattern("geometry/angle", pattern="geometry.*")
        assert not logger._matches_pattern("xray/work", pattern="geometry.*")


class TestRecording:
    """Tests for recording refinement state."""

    @pytest.mark.unit
    def test_record_basic(self):
        """Test basic recording without label."""
        from torchref.refinement.logger import Logger
        from torchref.refinement.loss_state import LossState

        state = LossState()
        state.register_target("xray/work", lambda: torch.tensor(0.20))
        state.register_target("geometry/bond", lambda: torch.tensor(0.02))
        state.set_weight("xray", 1.0)
        state.set_weight("geometry", 0.5)

        logger = Logger(state=state)
        stats = logger.record()

        assert len(logger.history) == 1
        assert "losses" in stats
        assert "xray/work" in stats["losses"]
        assert "geometry/bond" in stats["losses"]

    @pytest.mark.unit
    def test_record_with_label(self):
        """Test recording with a label."""
        from torchref.refinement.logger import Logger
        from torchref.refinement.loss_state import LossState

        state = LossState()
        state.register_target("xray/work", lambda: torch.tensor(0.20))

        logger = Logger(state=state)
        logger.record(label="before_xyz")
        logger.record(label="after_xyz")

        assert len(logger.history) == 2
        assert "before_xyz" in logger._labels
        assert "after_xyz" in logger._labels
        assert logger._labels["before_xyz"] == 0
        assert logger._labels["after_xyz"] == 1

    @pytest.mark.unit
    def test_record_filters_by_pattern(self):
        """Test that recording respects the instance pattern filter."""
        from torchref.refinement.logger import Logger
        from torchref.refinement.loss_state import LossState

        state = LossState()
        state.register_target("xray/work", lambda: torch.tensor(0.20))
        state.register_target("xray/test", lambda: torch.tensor(0.22))
        state.register_target("geometry/bond", lambda: torch.tensor(0.02))

        logger = Logger(state=state, pattern="xray.*")
        stats = logger.record()

        assert "xray/work" in stats["losses"]
        assert "xray/test" in stats["losses"]
        assert "geometry/bond" not in stats["losses"]

    @pytest.mark.unit
    def test_record_includes_weights(self):
        """Test that recording includes weight information."""
        from torchref.refinement.logger import Logger
        from torchref.refinement.loss_state import LossState

        state = LossState()
        state.register_target("xray/work", lambda: torch.tensor(0.20))
        state.set_weight("xray", 1.0)
        state.set_weight("xray/work", 2.0)

        logger = Logger(state=state)
        stats = logger.record()

        assert "weights" in stats
        assert stats["weights"]["xray"] == 1.0
        assert stats["weights"]["xray/work"] == 2.0

    @pytest.mark.unit
    def test_record_includes_group_totals(self):
        """Test that recording includes group totals."""
        from torchref.refinement.logger import Logger
        from torchref.refinement.loss_state import LossState

        state = LossState()
        state.register_target("xray/work", lambda: torch.tensor(0.20))
        state.register_target("geometry/bond", lambda: torch.tensor(0.02))
        state.set_weight("xray", 1.0)
        state.set_weight("geometry", 0.5)

        logger = Logger(state=state)
        stats = logger.record()

        assert "group_totals" in stats
        assert "xray" in stats["group_totals"]
        assert "geometry" in stats["group_totals"]


class TestGetRecord:
    """Tests for retrieving recorded states."""

    @pytest.mark.unit
    def test_get_record_by_label(self):
        """Test retrieving a record by label."""
        from torchref.refinement.logger import Logger
        from torchref.refinement.loss_state import LossState

        state = LossState()
        state.register_target("xray/work", lambda: torch.tensor(0.20))

        logger = Logger(state=state)
        logger.record(label="test_record")

        record = logger.get_record("test_record")
        assert record is not None
        assert "losses" in record

    @pytest.mark.unit
    def test_get_record_missing_label(self):
        """Test retrieving a record with non-existent label returns None."""
        from torchref.refinement.logger import Logger
        from torchref.refinement.loss_state import LossState

        state = LossState()
        logger = Logger(state=state)

        record = logger.get_record("nonexistent")
        assert record is None


class TestClear:
    """Tests for clearing recorded history."""

    @pytest.mark.unit
    def test_clear(self):
        """Test clearing all recorded history."""
        from torchref.refinement.logger import Logger
        from torchref.refinement.loss_state import LossState

        state = LossState()
        state.register_target("xray/work", lambda: torch.tensor(0.20))

        logger = Logger(state=state)
        logger.record(label="test1")
        logger.record(label="test2")

        assert len(logger.history) == 2
        assert len(logger._labels) == 2

        logger.clear()

        assert len(logger.history) == 0
        assert len(logger._labels) == 0


class TestGroupByHierarchy:
    """Tests for hierarchical grouping of stats."""

    @pytest.mark.unit
    def test_group_by_hierarchy_basic(self):
        """Test basic hierarchical grouping."""
        from torchref.refinement.logger import Logger
        from torchref.refinement.loss_state import LossState

        state = LossState()
        logger = Logger(state=state)

        data = {
            "xray/work": 3.2,
            "xray/test": 3.4,
            "geometry/bond": 0.02,
        }

        grouped = logger._group_by_hierarchy(data)

        assert "xray" in grouped
        assert "geometry" in grouped
        assert grouped["xray"]["work"] == 3.2
        assert grouped["xray"]["test"] == 3.4
        assert grouped["geometry"]["bond"] == 0.02

    @pytest.mark.unit
    def test_group_by_hierarchy_flat_keys(self):
        """Test grouping with flat keys (no hierarchy)."""
        from torchref.refinement.logger import Logger
        from torchref.refinement.loss_state import LossState

        state = LossState()
        logger = Logger(state=state)

        data = {
            "simple_key": 1.0,
            "another_key": 2.0,
        }

        grouped = logger._group_by_hierarchy(data)

        assert "other" in grouped
        assert grouped["other"]["simple_key"] == 1.0
        assert grouped["other"]["another_key"] == 2.0


class TestFilterByPattern:
    """Tests for pattern-based filtering."""

    @pytest.mark.unit
    def test_filter_by_pattern_basic(self):
        """Test basic pattern filtering."""
        from torchref.refinement.logger import Logger
        from torchref.refinement.loss_state import LossState

        state = LossState()
        logger = Logger(state=state)

        stats = {
            "xray/work": 0.20,
            "xray/test": 0.22,
            "geometry/bond": 0.02,
        }

        filtered = logger._filter_by_pattern(stats, pattern="xray.*")

        assert "xray/work" in filtered
        assert "xray/test" in filtered
        assert "geometry/bond" not in filtered

    @pytest.mark.unit
    def test_filter_by_pattern_nested(self):
        """Test pattern filtering with nested dictionaries."""
        from torchref.refinement.logger import Logger
        from torchref.refinement.loss_state import LossState

        state = LossState()
        logger = Logger(state=state)

        stats = {
            "losses": {
                "xray/work": 0.20,
                "geometry/bond": 0.02,
            },
            "weights": {
                "xray": 1.0,
                "geometry": 0.5,
            },
        }

        filtered = logger._filter_by_pattern(stats, pattern="xray.*")

        assert "losses" in filtered
        assert "xray/work" in filtered["losses"]
        assert "geometry/bond" not in filtered["losses"]
        assert "weights" in filtered
        assert "xray" in filtered["weights"]


class TestFormatValue:
    """Tests for value formatting."""

    @pytest.mark.unit
    def test_format_value_normal(self):
        """Test formatting normal values."""
        from torchref.refinement.logger import Logger
        from torchref.refinement.loss_state import LossState

        state = LossState()
        logger = Logger(state=state)

        assert logger._format_value(0.2345) == "0.2345"
        assert logger._format_value(1.0) == "1.0000"

    @pytest.mark.unit
    def test_format_value_small(self):
        """Test formatting very small values."""
        from torchref.refinement.logger import Logger
        from torchref.refinement.loss_state import LossState

        state = LossState()
        logger = Logger(state=state)

        formatted = logger._format_value(0.00001)
        assert "e" in formatted  # Scientific notation

    @pytest.mark.unit
    def test_format_value_large(self):
        """Test formatting large values."""
        from torchref.refinement.logger import Logger
        from torchref.refinement.loss_state import LossState

        state = LossState()
        logger = Logger(state=state)

        formatted = logger._format_value(1234.5)
        assert formatted == "1234.5"

    @pytest.mark.unit
    def test_format_value_with_sign(self):
        """Test formatting with sign for positive values."""
        from torchref.refinement.logger import Logger
        from torchref.refinement.loss_state import LossState

        state = LossState()
        logger = Logger(state=state)

        formatted = logger._format_value(0.1234, show_sign=True)
        assert formatted.startswith("+")

    @pytest.mark.unit
    def test_format_value_none(self):
        """Test formatting None values."""
        from torchref.refinement.logger import Logger
        from torchref.refinement.loss_state import LossState

        state = LossState()
        logger = Logger(state=state)

        assert logger._format_value(None) == "-"


class TestCompare:
    """Tests for comparison functionality."""

    @pytest.mark.unit
    def test_compare_last_two_records(self, capsys):
        """Test comparing last two records without labels."""
        from torchref.refinement.logger import Logger
        from torchref.refinement.loss_state import LossState

        state = LossState()
        counter = [0]

        def xray_target():
            counter[0] += 1
            # First call returns 0.22, second returns 0.20
            return torch.tensor(0.22 if counter[0] == 1 else 0.20)

        state.register_target("xray/work", xray_target)
        state.set_weight("xray", 1.0)

        logger = Logger(state=state)
        logger.record(label="before")
        logger.record(label="after")
        logger.compare(title="Test Comparison")

        captured = capsys.readouterr()
        assert "Test Comparison" in captured.out
        assert "xray" in captured.out

    @pytest.mark.unit
    def test_compare_with_labels(self, capsys):
        """Test comparing specific labeled records."""
        from torchref.refinement.logger import Logger
        from torchref.refinement.loss_state import LossState

        state = LossState()
        counter = [0]

        def xray_target():
            counter[0] += 1
            return torch.tensor(0.22 if counter[0] == 1 else 0.20)

        state.register_target("xray/work", xray_target)

        logger = Logger(state=state)
        logger.record(label="initial")
        logger.record(label="final")
        logger.compare(label_before="initial", label_after="final")

        captured = capsys.readouterr()
        assert "Refinement Comparison" in captured.out


class TestCurrent:
    """Tests for current state display."""

    @pytest.mark.unit
    def test_current_uses_last_record(self, capsys):
        """Test current() uses the last recorded state."""
        from torchref.refinement.logger import Logger
        from torchref.refinement.loss_state import LossState

        state = LossState()
        state.register_target("xray/work", lambda: torch.tensor(0.20))
        state.set_weight("xray", 1.0)

        logger = Logger(state=state)
        logger.record()
        logger.current(title="Test Current State")

        captured = capsys.readouterr()
        assert "Test Current State" in captured.out
        assert "xray" in captured.out

    @pytest.mark.unit
    def test_current_records_if_empty(self, capsys):
        """Test current() records if no history exists."""
        from torchref.refinement.logger import Logger
        from torchref.refinement.loss_state import LossState

        state = LossState()
        state.register_target("xray/work", lambda: torch.tensor(0.20))

        logger = Logger(state=state)
        assert len(logger.history) == 0

        logger.current()

        assert len(logger.history) == 1


class TestImport:
    """Tests for module imports."""

    @pytest.mark.unit
    def test_import_from_refinement_module(self):
        """Test Logger can be imported from refinement module."""
        from torchref.refinement import Logger

        assert Logger is not None

    @pytest.mark.unit
    def test_import_from_logger_module(self):
        """Test Logger can be imported from logger module."""
        from torchref.refinement.logger import Logger

        assert Logger is not None
