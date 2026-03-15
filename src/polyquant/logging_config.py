"""Project-level logging configuration."""

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configure the polyquant logger with a stderr StreamHandler.

    Args:
        level: Logging level name (e.g. "DEBUG", "INFO", "WARNING").
    """
    logger = logging.getLogger("polyquant")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid adding duplicate handlers on repeated calls
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
