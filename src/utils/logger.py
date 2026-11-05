"""
utils/logger.py — 統一 loguru 日誌設定

整個專案的輸出規則：
    - 過程訊息（載入、進度、警告、錯誤）一律使用這裡的 `logger`
    - 結果資料（JSON、摘要表、準確率）使用 `print` 輸出到 stdout
"""

import sys
from pathlib import Path
from typing import Optional, Union

from loguru import logger as _logger

# console 與 log 檔共用同一個格式（寫檔時會自動去掉顏色標籤）
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
    "<level>{message}</level>"
)


def setup_logger(level: str = "INFO", log_file: Optional[Union[str, Path]] = None) -> None:
    """設定 logger。

    Args:
        level: console 顯示的最低等級。
        log_file: 同時寫入的 log 檔（不指定則只輸出到 console）。
            檔名含 {time...} 時（例如 lpr_{time:YYYY-MM-DD}.log）會每天換檔、保留 7 天。
    """
    _logger.remove()
    _logger.add(sys.stderr, level=level, format=LOG_FORMAT, colorize=True)
    if log_file is None:
        return

    log_path = Path(log_file)
    rotate = "{time" in log_path.name
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _logger.add(
            str(log_path),
            level="DEBUG",
            format=LOG_FORMAT,
            encoding="utf-8",
            rotation="00:00" if rotate else None,
            retention="7 days" if rotate else None,
        )
    except PermissionError:
        _logger.warning(f"沒有權限寫入 {log_path.parent}，log 只輸出到 console")


logger = _logger
