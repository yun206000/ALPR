"""
detector/yolo_detector.py — YOLO（Ultralytics）車牌偵測模組。
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from ultralytics import YOLO

from ..config import DetectorConfig
from ..models import BoundingBox, DetectionResult
from ..utils import logger


class YOLODetector:
    """
    YOLO 車牌偵測器。

    使用方式：
        detector = YOLODetector(config)
        results = detector.detect(image_bgr)
    """

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        self.device = config.device

        weights = config.weights_path
        if not weights.exists():
            raise FileNotFoundError(f"找不到 YOLO 權重：{weights}（請檢查 config.yaml 的 detector.weights_path）")
        logger.info(f"載入 YOLO 權重：{weights}")
        self._model = YOLO(str(weights))

    @torch.no_grad()
    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: Optional[float] = None,
    ) -> List[DetectionResult]:
        """
        偵測影像中的車牌。

        Args:
            image: BGR ndarray（cv2 格式），shape (H, W, 3)。Ultralytics 把 numpy 輸入視為 BGR，
                   傳 RGB 會讓紅藍通道對調、信心分數下降
            confidence_threshold: 覆蓋 config 的 threshold

        Returns:
            依 detector_score 由高到低排序的 DetectionResult
        """
        threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else self.config.confidence_threshold
        )

        predictions = self._model.predict(
            source=image,
            conf=threshold,
            device=self.device,
            verbose=False,
        )
        if not predictions or predictions[0].boxes is None:
            return []

        h, w = image.shape[:2]
        results: List[DetectionResult] = []
        for box in predictions[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()

            # Clamp bbox 到影像範圍內
            x1, x2 = max(0.0, min(x1, w)), max(0.0, min(x2, w))
            y1, y2 = max(0.0, min(y1, h)), max(0.0, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                continue

            results.append(
                DetectionResult(
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    detector_score=float(box.conf[0].cpu().item()),
                )
            )

        results.sort(key=lambda r: r.detector_score, reverse=True)
        logger.debug(f"偵測到 {len(results)} 個車牌（threshold={threshold}）")
        return results

    def warmup(self) -> None:
        """用空白影像跑一次推論，讓第一個真正的請求不用等初始化。"""
        self.detect(np.zeros((640, 640, 3), dtype=np.uint8))
        logger.debug("YOLO warmup 完成")
