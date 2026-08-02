"""
SRAG Logging Setup (FD-07).

Implements:
    FD-03: File logging (JSON lines format) — JSON lines at log file path.
    FD-04: Log levels (console=INFO, file=DEBUG).
    FD-08: Structured logging (no secrets in logs; redacted references OK).

Usage example:
    >>> from log_config import get_logger
    >>> logger = get_logger("ingest")
    >>> logger.info("Processed %d files", 42)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOG_FORMAT_JSON = "json"
LOG_FORMAT_TEXT = "text"

DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_DIR = os.path.join(os.getcwd(), "logs")
DEFAULT_LOG_FILE = os.path.join(DEFAULT_LOG_DIR, "srag.log")


# ---------------------------------------------------------------------------
# JSON formatter for structured logging (FD-03)
# ---------------------------------------------------------------------------

class JsonFormatter(logging.Formatter):
    """Format log records as JSON lines (one JSON object per line)."""

    def format(self, record: logging.LogRecord) -> str:
        """Return a single JSON line for the given log record."""
        # Build structured log entry
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present (redacted — no secrets)
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields if provided via logging.bindLogger / extra dict
        if hasattr(record, "extra_data"):
            log_entry["data"] = record.extra_data

        return json.dumps(log_entry, default=str, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Text formatter for console output (FD-04: human-readable)
# ---------------------------------------------------------------------------

class ConsoleColorizer:
    """ANSI color codes for console log levels."""

    COLORS = {
        "DEBUG": "\033[36m",      # cyan
        "INFO": "\033[32m",       # green
        "WARNING": "\033[33m",    # yellow
        "ERROR": "\033[31m",      # red
        "CRITICAL": "\033[35m",   # magenta
    }
    RESET = "\033[0m"

    @classmethod
    def colorize(cls, levelname: str, message: str) -> str:
        color = cls.COLORS.get(levelname, "")
        if color:
            return f"{color}[{levelname:>8}]{cls.RESET} {message}"
        return f"[{levelname:>8}] {message}"


class ColoredFormatter(logging.Formatter):
    """Format log records with ANSI colors for console output."""

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        levelname = record.levelname
        colored = ConsoleColorizer.colorize(levelname, msg)
        return f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}  {colored}"


# ---------------------------------------------------------------------------
# Redaction helper (FD-08: no secrets in logs)
# ---------------------------------------------------------------------------

_SECRET_PATTERNS = [
    # API keys, tokens — redact values that look like secrets
    (r"(api[_-]?key\s*[:=]\s*)\S+", r"\1[REDACTED]"),
    (r"(token\s*[:=]\s*)\S+", r"\1[REDACTED]"),
    (r"(password\s*[:=]\s*)\S+", r"\1[REDACTED]"),
]


def redact_secrets(message: str) -> str:
    """Redact potential secrets from log messages."""
    for pattern, replacement in _SECRET_PATTERNS:
        message = __import__("re").sub(pattern, replacement, flags=re.IGNORECASE)
    return message


# ---------------------------------------------------------------------------
# get_logger — main public API (FD-07: console=INFO, file=DEBUG)
# ---------------------------------------------------------------------------

_LOGGERS_SETUP = set()  # track which logger names have been configured


def get_logger(
    name: str,
    log_file: str | None = None,
    log_level: str | None = None,
) -> logging.Logger:
    """Get or create a named logger with console + optional file handler.

    Args:
        name: Logger name (e.g., "ingest", "search"). Convention: dotted prefix
              for hierarchy (e.g., "srag.ingest").
        log_file: Path to log file. Defaults to DEFAULT_LOG_FILE. Set to None to skip file logging.
        log_level: Logging level. Defaults to DEFAULT_LOG_LEVEL ("INFO").

    Returns:
        Configured logging.Logger instance.

    Note:
        - Console handler: level=INFO (human-readable, colored).
        - File handler:   level=DEBUG (JSON lines, full detail).
        - Logger setup is idempotent per logger name (no duplicate handlers).
        - If LOG_FILE=none env var is set, file logging is disabled.
    """
    if name in _LOGGERS_SETUP:
        return logging.getLogger(name)

    log_file = log_file or os.environ.get("SRAG_LOG_FILE", DEFAULT_LOG_FILE)
    log_level = (log_level or os.environ.get("SRAG_LOG_LEVEL", DEFAULT_LOG_LEVEL)).upper()

    logger = logging.getLogger(name)
    logger.setLevel(log_level)  # type: ignore[arg-type]

    # Console handler — INFO level, colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = ColoredFormatter()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler — DEBUG level, JSON lines (if log_file is enabled)
    if os.environ.get("SRAG_LOG_FILE", "default") != "none":
        try:
            log_dir = os.path.dirname(log_file) or DEFAULT_LOG_DIR
            Path(log_dir).mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_formatter = JsonFormatter()
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        except (OSError, PermissionError) as exc:
            # File logging failure should not break the application
            print(f"Warning: Could not create log file at {log_file}: {exc}", file=sys.stderr)

    _LOGGERS_SETUP.add(name)
    return logger


# ---------------------------------------------------------------------------
# Convenience: get root logger with standard setup
# ---------------------------------------------------------------------------

def setup_logging(
    level: str = DEFAULT_LOG_LEVEL,
    log_file: str | None = None,
    enable_console: bool = True,
    enable_file: bool = True,
) -> logging.Logger:
    """Set up the root logger with standard handlers.

    This is a convenience function for scripts that need a quick setup
    (e.g., run.py, init_db.py). For web apps and libraries, use get_logger() instead.

    Args:
        level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
        log_file: Path to log file. None = no file logging.
        enable_console: Whether to add a console handler.
        enable_file: Whether to add a file handler.

    Returns:
        The root logger (named "srag").
    """
    root_logger = logging.getLogger("srag")
    root_logger.setLevel(level)  # type: ignore[arg-type]

    if enable_console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(getattr(logging, level.upper(), logging.INFO))
        ch.setFormatter(ColoredFormatter())
        root_logger.addHandler(ch)

    if enable_file and log_file:
        try:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(JsonFormatter())
            root_logger.addHandler(fh)
        except (OSError, PermissionError):
            pass  # Silently skip file logging on failure

    return root_logger