"""Shared utility functions."""

import functools
import logging
import time

logger = logging.getLogger(__name__)


def retry_with_backoff(max_retries=3, base_delay=1.0, exceptions=(Exception,)):
    """Decorator that retries a function with exponential backoff."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        logger.error("All %d retries exhausted for %s: %s", max_retries, func.__name__, e)
                        raise
                    delay = base_delay * (2 ** attempt)
                    logger.warning("Retry %d/%d for %s after %.1fs: %s", attempt + 1, max_retries, func.__name__, delay, e)
                    time.sleep(delay)
        return wrapper
    return decorator
