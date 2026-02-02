"""Logging setup using Python's logging module with Rich console output."""

import logging
from pathlib import Path
from typing import Optional

from rich.logging import RichHandler


def setup_logging(
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure Python logging with Rich console and optional file output.

    Args:
        log_dir: Directory for log file output. If None, console-only.
        level: Logging level (default: INFO).

    Returns:
        Configured root logger for the tff package.
    """
    logger = logging.getLogger("tff")
    logger.setLevel(level)
    logger.propagate = False  # Don't duplicate into Hydra's root logger

    # Clear any existing handlers (avoid duplicates on re-init)
    logger.handlers.clear()

    # Rich console handler
    console_handler = RichHandler(
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "training.log")
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger
