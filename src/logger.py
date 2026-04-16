"""
logger.py
---------
Sets up the application logger.

We use Python's built-in `logging` module.
- Log messages go to BOTH the terminal AND a log file.
- The log file is written to logs/run.log.
- This module provides a single `get_logger()` function.

Usage:
    from src.logger import get_logger
    log = get_logger(__name__)
    log.info("Starting scout workflow")
    log.warning("No website found for Acme Corp")
    log.error("OpenAI call failed: ...")
"""

import logging
import sys
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger for the given module name.

    Call this at the top of each module:
        log = get_logger(__name__)

    The logger writes to:
    - Terminal (INFO level and above)
    - logs/run.log (DEBUG level and above — more verbose)

    Args:
        name: Typically __name__ from the calling module.

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Only configure handlers once, even if get_logger is called multiple times.
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)  # Capture everything at the logger level

    # ------------------------------------------------------------------ #
    # Formatter: show time, logger name, level, and message
    # ------------------------------------------------------------------ #
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ------------------------------------------------------------------ #
    # Console handler — INFO and above, clean output
    # ------------------------------------------------------------------ #
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ------------------------------------------------------------------ #
    # File handler — DEBUG and above, written to logs/run.log
    # ------------------------------------------------------------------ #
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "run.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
