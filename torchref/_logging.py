"""Logging for TorchRef: one stdlib logger per module under ``torchref``.

Every module gets its logger with :func:`get_logger` (``log = get_logger(__name__)``) and
reports through it; nothing in the library prints diagnostics. What is shown is decided
by whoever runs the program:

- As a library, the ``torchref`` logger is left at ``NOTSET``, so it follows the root
  logger (WARNING by default): warnings are shown, progress is not.
- :func:`set_verbosity` / :func:`verbosity` raise or lower it, for the whole package or
  one subpackage (``set_verbosity(2, "torchref.scaling")``). ``TORCHREF_LOG_LEVEL`` sets
  it at import.
- The CLI calls :func:`configure_cli` and defaults to INFO.

Verbosity levels, as taken by :func:`set_verbosity` and the CLI ``-v``: 0 = WARNING,
1 = INFO, 2 = DEBUG, 3 = :data:`TRACE`.

The ``torchref`` logger carries one default stderr handler, active only while the root
logger has no handlers: an application that configures logging itself (``basicConfig``,
pytest) receives the records by propagation instead, without duplicates.

:func:`quiet` silences everything below WARNING for the current context (thread / task)
without touching levels; it replaces the old internal ``verbose=0`` arguments used to
keep throwaway objects quiet. A message whose arguments need a host sync (``.item()``,
``float(tensor)``) or extra computation is built only under :func:`enabled`.

Stdlib only: imported by ``torchref/__init__`` before torch.
"""

import contextvars
import logging
import os
import sys
from contextlib import contextmanager
from typing import Iterator, Optional, Union

TRACE = 5
logging.addLevelName(TRACE, "TRACE")

_ROOT = "torchref"
_VERBOSITY_LEVELS = (logging.WARNING, logging.INFO, logging.DEBUG, TRACE)

_quiet = contextvars.ContextVar("torchref_quiet", default=False)


class _QuietFilter(logging.Filter):
    """Drop records below WARNING while :func:`quiet` is active."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.WARNING or not _quiet.get()


_QUIET_FILTER = _QuietFilter()


class _Formatter(logging.Formatter):
    """Bare message at INFO, so progress reads as plain text; level and logger otherwise."""

    _plain = logging.Formatter("%(message)s")
    _tagged = logging.Formatter("%(levelname)s [%(name)s] %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        fmt = self._plain if record.levelno == logging.INFO else self._tagged
        return fmt.format(record)


class _DefaultHandler(logging.StreamHandler):
    """stderr handler that stands down once the application configures root logging."""

    def __init__(self):
        super().__init__()
        self.setFormatter(_Formatter())

    @property
    def stream(self):
        # Resolved per record, so pytest's capsys and redirect_stderr see the output.
        return sys.stderr

    @stream.setter
    def stream(self, value):
        pass

    def emit(self, record: logging.LogRecord) -> None:
        if not logging.getLogger().handlers:
            super().emit(record)


_DEFAULT_HANDLER = _DefaultHandler()


def _root_logger() -> logging.Logger:
    root = logging.getLogger(_ROOT)
    if _DEFAULT_HANDLER not in root.handlers:
        root.addHandler(_DEFAULT_HANDLER)
    if _QUIET_FILTER not in root.filters:
        root.addFilter(_QUIET_FILTER)
    return root


_root_logger()


def get_logger(name: str) -> logging.Logger:
    """The logger for module ``name`` (pass ``__name__``), honouring :func:`quiet`."""
    log = logging.getLogger(name)
    if _QUIET_FILTER not in log.filters:
        log.addFilter(_QUIET_FILTER)
    return log


def enabled(log: logging.Logger, level: int) -> bool:
    """Whether a ``level`` record on ``log`` would be emitted.

    Guard messages whose arguments are costly with it::

        if enabled(log, logging.DEBUG):
            log.debug("Rwork %.4f", rwork.item())
    """
    if level < logging.WARNING and _quiet.get():
        return False
    return log.isEnabledFor(level)


def resolve_level(level: Union[int, str]) -> int:
    """A logging level from a verbosity (0-3), a logging level (>= 5) or a level name."""
    if isinstance(level, str):
        name = level.strip().upper()
        if name.lstrip("-").isdigit():
            return resolve_level(int(name))
        value = logging.getLevelName(name)
        if not isinstance(value, int):
            raise ValueError(
                f"Unknown log level {level!r}; use 0-3 or one of "
                "TRACE, DEBUG, INFO, WARNING, ERROR"
            )
        return value
    if level < TRACE:
        return _VERBOSITY_LEVELS[min(max(level, 0), len(_VERBOSITY_LEVELS) - 1)]
    return level


def set_verbosity(level: Union[int, str], module: str = _ROOT) -> None:
    """Show TorchRef messages at ``level`` and above, for ``module`` and its children.

    ``level`` is a verbosity (0 = warnings only, 1 = progress, 2 = detail, 3 = trace), a
    logging level or a level name.
    """
    _root_logger()
    logging.getLogger(module).setLevel(resolve_level(level))


@contextmanager
def verbosity(level: Union[int, str], module: str = _ROOT) -> Iterator[None]:
    """:func:`set_verbosity` for the duration of a ``with`` block."""
    log = logging.getLogger(module)
    previous = log.level
    set_verbosity(level, module)
    try:
        yield
    finally:
        log.setLevel(previous)


@contextmanager
def quiet() -> Iterator[None]:
    """Suppress TorchRef messages below WARNING in the current context."""
    token = _quiet.set(True)
    try:
        yield
    finally:
        _quiet.reset(token)


class _BelowWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno < logging.WARNING


def configure_cli(level: Union[int, str] = 1, log_file: Optional[str] = None) -> None:
    """Route TorchRef logging for a command-line run.

    Records below WARNING go to stdout, interleaved with the command's results; warnings
    and errors go to stderr. ``log_file`` additionally receives every record at
    ``level`` and above. Only the ``torchref`` logger is configured, so other libraries'
    logging is left alone. Python warnings are routed through logging too.
    """
    root = _root_logger()
    for handler in list(root.handlers):
        if getattr(handler, "_torchref_cli", False):
            root.removeHandler(handler)
            handler.close()

    out = logging.StreamHandler(sys.stdout)
    out.addFilter(_BelowWarning())
    err = logging.StreamHandler(sys.stderr)
    err.setLevel(logging.WARNING)
    handlers = [out, err]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file))
    for handler in handlers:
        handler._torchref_cli = True
        handler.setFormatter(_Formatter())
        root.addHandler(handler)

    root.removeHandler(_DEFAULT_HANDLER)
    root.propagate = False
    root.setLevel(resolve_level(level))

    logging.captureWarnings(True)
    warnings_log = logging.getLogger("py.warnings")
    for handler in handlers[1:]:
        if handler not in warnings_log.handlers:
            warnings_log.addHandler(handler)
    warnings_log.propagate = False


_env_level = os.environ.get("TORCHREF_LOG_LEVEL")
if _env_level:
    set_verbosity(_env_level)
