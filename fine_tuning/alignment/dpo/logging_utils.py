import logging
import time


class _UTCFormatter(logging.Formatter):
    converter = time.gmtime  # type: ignore[attr-defined]


def setup_logging(level: int = logging.INFO) -> None:
    """
    Initialize root logger with consistent UTC ISO-like timestamps if not already configured.
    Safe to call multiple times (no duplicate handlers added).
    """
    root = logging.getLogger()
    if root.handlers:
        return
    fmt = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
    datefmt = "%Y-%m-%dT%H:%M:%S"
    handler = logging.StreamHandler()
    handler.setFormatter(_UTCFormatter(fmt, datefmt))
    root.setLevel(level)
    root.addHandler(handler)

__all__ = ["setup_logging"]
