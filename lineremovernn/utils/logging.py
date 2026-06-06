import logging
import threading
from datetime import datetime
from pathlib import Path

from lineremovernn.utils.consts import DEFAULT_LOGS


class _ColorFormatter(logging.Formatter):
    _COLORS = {
        logging.DEBUG: "\x1b[90m",
        logging.INFO: "\x1b[96m",
        logging.WARNING: "\x1b[93m",
        logging.ERROR: "\x1b[91m",
        logging.CRITICAL: "\x1b[31;1m",
    }
    _RESET = "\x1b[0m"
    _FMT = "%(asctime)s [LineRemoverNN] [%(shortname)s] [%(levelname)s] : %(message)s"

    def format(self, record: logging.LogRecord) -> str:
        color = self._COLORS.get(record.levelno, self._RESET)
        formatter = logging.Formatter(
            f"{color}{self._FMT}{self._RESET}",
            datefmt="%H:%M:%S",
        )
        return formatter.format(record)


class _StripPrefixFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.shortname = record.name.removeprefix("LineRemoverNN.")
        return True


class _PlainFormatter(logging.Formatter):
    _FMT = "%(asctime)s [LineRemoverNN] [%(shortname)s] [%(levelname)s] : %(message)s"

    def __init__(self):
        super().__init__(self._FMT, datefmt="%H:%M:%S")


_setup_lock = threading.Lock()
_root_configured = False


def _configure_root(log_dir: Path) -> None:
    global _root_configured
    with _setup_lock:
        if _root_configured:
            return

        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        log_file = log_dir / f"{timestamp}-logs.log"

        root = logging.getLogger("LineRemoverNN")
        root.setLevel(logging.DEBUG)
        root.propagate = False

        console = logging.StreamHandler()
        console.addFilter(_StripPrefixFilter())
        console.setFormatter(_ColorFormatter())

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.addFilter(_StripPrefixFilter())
        file_handler.setFormatter(_PlainFormatter())

        root.addHandler(console)
        root.addHandler(file_handler)

        _root_configured = True


def get_logger(
    name: str,
    log_dir: Path | None = None,
) -> logging.Logger:
    if log_dir is None:
        log_dir = DEFAULT_LOGS

    _configure_root(log_dir)
    return logging.getLogger(f"LineRemoverNN.{name}")
