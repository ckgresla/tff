"""
Logging setup and console utilities
"""

import logging
import os
from datetime import datetime

from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text
from rich.traceback import install as init_rich_tracebacks

# Consts et al
console = Console()
init_rich_tracebacks(console=console)


class CustomRichHandler(RichHandler):
    """Custom handler that colors only specific parts of the log message."""

    # see- https://rich.readthedocs.io/en/latest/appendix/colors.html#appendix-colors
    LEVEL_COLORS = {
        "DEBUG": "black on light_steel_blue3",
        "INFO": "dim white",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold red",
    }

    def render_message(self, record, message):
        """Override to customize the message rendering."""
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        level = record.levelname
        filename = record.filename
        lineno = record.lineno
        msg = record.getMessage()

        # Build the colored text
        text = Text()
        text.append(f"[{timestamp} - ", style="")  # No color for timestamp
        text.append(level, style=self.LEVEL_COLORS.get(level, "white"))  # Colored level
        text.append(f" - {filename}:", style="")  # No color for filename
        text.append(
            str(lineno), style=self.LEVEL_COLORS.get(level, "white")
        )  # Colored line number
        text.append(f"] {msg}", style="")  # No color for message

        return text


def get_log_level_from_env() -> int:
    """
    Parse LOG environment variable and return the appropriate logging level.
    Supports: DEBUG, INFO, WARNING, ERROR, CRITICAL
    Default: INFO

    i.e: `export LOG=DEBUG` sets the log level to debug (which will log ~everything)
    """
    log_level_str = os.getenv("LOG", "INFO").upper()

    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "WARN": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    level = level_map.get(log_level_str, logging.INFO)

    # Optional: print what level was set
    if log_level_str not in level_map:
        print(f"Warning: Invalid LOG level '{log_level_str}', defaulting to INFO")

    return level


# Module-level logger with custom Rich handler
log = logging.getLogger(__name__)
log_level = get_log_level_from_env()
log.setLevel(log_level)

if not log.handlers:
    rich_handler = CustomRichHandler(
        console=console,
        show_time=False,
        show_level=False,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
    )
    log.addHandler(rich_handler)
    log.propagate = False


# Quick Utils for [Errors, Warnings, Logs/Info, Good Things]
def printerr(*args, **kwargs):
    """Prints a message to the console with red style."""
    console.print(*args, **kwargs, style="red", highlight=False)


def printwar(*args, **kwargs):
    """Prints a message to the console with yellow style."""
    console.print(*args, **kwargs, style="yellow", highlight=False)


def printlog(*args, **kwargs):
    """Prints a message to the console with blue style."""
    console.print(*args, **kwargs, style="blue", highlight=False)


def printok(*args, **kwargs):
    """Prints a message to the console with green style."""
    console.print(*args, **kwargs, style="green", highlight=False)
