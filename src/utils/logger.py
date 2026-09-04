"""
utils/logger.py — 統一 loguru 日誌設定
"""

import sys
from pathlib import Path
from loguru import logger as _logger


def setup_logger(level: str = "INFO") -> None:
    _logger.remove()
    _logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # Ensure outputs directory exists before creating file sink
    out_dir = Path("outputs")
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        _logger.add(
            str(out_dir / "lpr_{time:YYYY-MM-DD}.log"),
            level="DEBUG",
            rotation="00:00",
            retention="7 days",
            encoding="utf-8",
        )
    except PermissionError:
        _logger.warning("Cannot create outputs directory for file logging; continuing without file sink.")
    except Exception as exc:  # pragma: no cover - defensive fallback
        _logger.warning("Failed to add file sink: {}", exc)


logger = _logger
