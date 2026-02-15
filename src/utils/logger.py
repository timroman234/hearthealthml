"""Logging utilities."""

import logging
import sys
from pathlib import Path

# Global logger cache
_loggers: dict[str, logging.Logger] = {}

# Default format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str = "hearthealthml",
    level: int = logging.INFO,
    log_file: Path | None = None,
    format_string: str = DEFAULT_FORMAT,
) -> logging.Logger:
    """Setup and configure a logger.

    Args:
        name: Logger name.
        level: Logging level.
        log_file: Optional file path for logging.
        format_string: Log message format.

    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(format_string, datefmt=DEFAULT_DATE_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    _loggers[name] = logger
    return logger


def get_logger(name: str = "hearthealthml") -> logging.Logger:
    """Get a logger by name.

    Creates a new logger if it doesn't exist.

    Args:
        name: Logger name.

    Returns:
        Logger instance.
    """
    if name in _loggers:
        return _loggers[name]

    # Create child logger under main logger
    if "." in name:
        parent_name = name.rsplit(".", 1)[0]
        if parent_name not in _loggers:
            setup_logger(parent_name)

    logger = logging.getLogger(name)

    # Setup with defaults if not configured
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(DEFAULT_FORMAT, datefmt=DEFAULT_DATE_FORMAT)
        )
        logger.addHandler(handler)
        logger.propagate = False

    _loggers[name] = logger
    return logger


def set_log_level(level: int | str, name: str = "hearthealthml") -> None:
    """Set logging level for a logger.

    Args:
        level: Logging level (int or string like 'DEBUG').
        name: Logger name.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    logger = get_logger(name)
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)
