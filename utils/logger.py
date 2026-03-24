"""
Logger Setup
============
Centralized logging configuration with color-coded console output.
"""

import logging
import sys
from typing import Optional

_loggers = {}


class ColorFormatter(logging.Formatter):
    """Color-coded log output for terminal."""

    COLORS = {
        "DEBUG": "\033[36m",  # cyan
        "INFO": "\033[32m",  # green
        "WARNING": "\033[33m",  # yellow
        "ERROR": "\033[31m",  # red
        "CRITICAL": "\033[35m",  # magenta
    }
    RESET = "\033[0m"
    GRAY = "\033[90m"

    def format(self, record):
        level_color = self.COLORS.get(record.levelname, "")
        time_str = self.formatTime(record, "%H:%M:%S")
        name = f"{record.name:<12}"
        msg = record.getMessage()
        return (
            f"{self.GRAY}{time_str}{self.RESET} "
            f"{level_color}[{record.levelname[0]}]{self.RESET} "
            f"{self.GRAY}{name}{self.RESET} {msg}"
        )


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get or create a named logger."""
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ColorFormatter())
        logger.addHandler(handler)
        logger.propagate = False

    _loggers[name] = logger
    return logger
