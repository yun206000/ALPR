"""
utils/tracking.py — 用 trackio 紀錄訓練 / 微調過程。

紀錄存在本機 SQLite（<trackio_dir>/<project>.db），用儀表板看折線圖：
    TRACKIO_DIR=logs/trackio trackio show --project ALPR

使用方式：
    run_name = make_run_name("parseq")          # parseq_20261105_013000，trackio 上的 run 名稱
    with ExperimentTracker(cfg.tracking, run_name, run_config) as tracker:
        tracker.log({"train/loss": 0.12}, step=10)
"""

from __future__ import annotations

import math
import os
from datetime import datetime
from typing import Any, Dict, Mapping, Optional

from ..config import TrackingConfig
from .logger import logger


def make_run_name(run_type: str) -> str:
    """每次訓練的名稱：<類型>_<時間>，同時用於權重資料夾與 trackio run。"""
    return f"{run_type}_{datetime.now():%Y%m%d_%H%M%S}"


class ExperimentTracker:
    def __init__(self, cfg: TrackingConfig, run_name: str, run_config: Optional[Dict[str, Any]] = None) -> None:
        self.enabled = cfg.enabled
        self.run_name = run_name
        self._trackio = None
        if not self.enabled:
            logger.info("tracking.enabled 為 false，不紀錄實驗")
            return

        # trackio 在 import 時讀取 TRACKIO_DIR，必須先設好
        os.environ["TRACKIO_DIR"] = str(cfg.trackio_dir.resolve())
        import trackio

        trackio.init(project=cfg.project, name=self.run_name, config=run_config or {}, capture_console=False)
        self._trackio = trackio
        logger.info(f"實驗紀錄 | run={self.run_name} | {cfg.trackio_dir}/{cfg.project}.db")

    def log(self, metrics: Mapping[str, Any], step: int) -> None:
        """紀錄一組數值指標（非數值或 NaN 會略過）。"""
        if not self.enabled:
            return
        scalars = {}
        for key, value in metrics.items():
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                scalars[key] = value
        if scalars:
            self._trackio.log(scalars, step=step)

    def close(self) -> None:
        if not self.enabled:
            return
        self._trackio.finish()
        logger.info(f"實驗紀錄完成：{self.run_name}")

    def __enter__(self) -> "ExperimentTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
