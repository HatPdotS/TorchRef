"""Tests for torchref._logging: levels, quiet(), the default handler and CLI routing."""

import io
import logging
import subprocess
import sys
import threading
from contextlib import redirect_stderr, redirect_stdout

import pytest

from torchref import _logging
from torchref._logging import (
    TRACE,
    configure_cli,
    enabled,
    get_logger,
    quiet,
    resolve_level,
    set_verbosity,
    verbosity,
)


@pytest.fixture(autouse=True)
def _restore_torchref_logger():
    root = logging.getLogger("torchref")
    sub = logging.getLogger("torchref.test_sub")
    state = (root.level, list(root.handlers), root.propagate, sub.level)
    warnings_log = logging.getLogger("py.warnings")
    w_state = (list(warnings_log.handlers), warnings_log.propagate)
    yield
    root.setLevel(state[0])
    root.handlers[:] = state[1]
    root.propagate = state[2]
    sub.setLevel(state[3])
    warnings_log.handlers[:] = w_state[0]
    warnings_log.propagate = w_state[1]
    logging.captureWarnings(False)


@pytest.mark.parametrize(
    "level, expected",
    [
        (0, logging.WARNING),
        (1, logging.INFO),
        (2, logging.DEBUG),
        (3, TRACE),
        (4, TRACE),
        (-1, logging.WARNING),
        (logging.ERROR, logging.ERROR),
        ("debug", logging.DEBUG),
        ("TRACE", TRACE),
        ("2", logging.DEBUG),
    ],
)
def test_resolve_level(level, expected):
    assert resolve_level(level) == expected


def test_resolve_level_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown log level"):
        resolve_level("loud")


def test_set_verbosity_scopes_to_module():
    log = get_logger("torchref.test_sub")
    set_verbosity(0)
    set_verbosity(2, "torchref.test_sub")
    assert log.isEnabledFor(logging.DEBUG)
    assert not logging.getLogger("torchref.other").isEnabledFor(logging.INFO)


def test_verbosity_restores_previous_level():
    root = logging.getLogger("torchref")
    root.setLevel(logging.WARNING)
    with verbosity(2):
        assert root.level == logging.DEBUG
    assert root.level == logging.WARNING


def test_set_verbosity_handler_is_idempotent():
    set_verbosity(1)
    set_verbosity(2)
    root = logging.getLogger("torchref")
    assert root.handlers.count(_logging._DEFAULT_HANDLER) == 1


def test_quiet_drops_info_keeps_warning(caplog):
    log = get_logger("torchref.test_sub")
    caplog.set_level(logging.DEBUG, logger="torchref")
    with quiet():
        log.info("hidden")
        log.warning("shown")
        assert not enabled(log, logging.INFO)
        assert enabled(log, logging.WARNING)
    log.info("visible again")
    messages = [r.getMessage() for r in caplog.records]
    assert messages == ["shown", "visible again"]


def test_quiet_nests():
    log = get_logger("torchref.test_sub")
    set_verbosity(1)
    with quiet():
        with quiet():
            assert not enabled(log, logging.INFO)
        assert not enabled(log, logging.INFO)
    assert enabled(log, logging.INFO)


def test_quiet_is_per_thread():
    log = get_logger("torchref.test_sub")
    set_verbosity(1)
    seen = []
    with quiet():
        thread = threading.Thread(target=lambda: seen.append(enabled(log, logging.INFO)))
        thread.start()
        thread.join()
    assert seen == [True]


def test_default_handler_formats_and_stands_down(monkeypatch):
    """INFO is bare, other levels are tagged; silent once root logging is configured."""
    log = get_logger("torchref.test_sub")
    set_verbosity(1)
    monkeypatch.setattr(logging.getLogger(), "handlers", [])
    buf = io.StringIO()
    with redirect_stderr(buf):
        log.info("progress line")
        log.warning("careful")
    assert buf.getvalue().splitlines() == [
        "progress line",
        "WARNING [torchref.test_sub] careful",
    ]
    monkeypatch.setattr(logging.getLogger(), "handlers", [logging.NullHandler()])
    buf = io.StringIO()
    with redirect_stderr(buf):
        log.warning("not duplicated")
    assert buf.getvalue() == ""


def test_configure_cli_splits_streams(tmp_path):
    log = get_logger("torchref.test_sub")
    out, err = io.StringIO(), io.StringIO()
    log_file = tmp_path / "run.log"
    with redirect_stdout(out), redirect_stderr(err):
        configure_cli(1, log_file=str(log_file))
        log.debug("too detailed")
        log.info("progress")
        log.warning("careful")
    assert out.getvalue() == "progress\n"
    assert err.getvalue() == "WARNING [torchref.test_sub] careful\n"
    assert log_file.read_text().splitlines() == [
        "progress",
        "WARNING [torchref.test_sub] careful",
    ]


def test_configure_cli_is_idempotent():
    out = io.StringIO()
    with redirect_stdout(out), redirect_stderr(io.StringIO()):
        configure_cli(1)
        configure_cli(1)
        get_logger("torchref.test_sub").info("once")
    assert out.getvalue() == "once\n"


def test_env_level_applied_at_import():
    code = "import logging, torchref; print(logging.getLogger('torchref').level)"
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**__import__("os").environ, "TORCHREF_LOG_LEVEL": "2"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == str(logging.DEBUG)
