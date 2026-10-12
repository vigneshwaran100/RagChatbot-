import logging
import sys
from app.config import get_settings


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name."""
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    return logger
