"""
detector/yolo_detector.py

YOLOv8 車牌偵測模組。
使用 Ultralytics YOLOv8 作為車牌 detector。

支援：
    - yolov8n
    - yolov8s
    - yolov8m
    - yolov8l
    - yolov8x
    - 自訂 fine-tuned 權重

安裝：
    pip install ultralytics
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from ultralytics import YOLO

from ..config import DetectorConfig
from ..models import BoundingBox, DetectionResult
from ..utils import logger


class YOLODetector:
    """
    YOLOv8 車牌偵測器。

    使用方式：
        detector = YOLODetector(config)
        results = detector.detect(image)
    """

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        self.device = config.device
        self._model = None

        self._load_model()

    def _load_model(self) -> None:
        """
        載入 YOLOv8 模型。
        """

        logger.info(f"載入 YOLO 模型：{self.config.model_variant}")

        weights: Path = self.config.weights_path

        try:
            if weights.exists():
                logger.info(f"使用本地權重：{weights}")
                self._model = YOLO(str(weights))

            else:
                logger.warning(
                    f"找不到本地權重 {weights}，使用預設模型"
                )

                # 例如：
                # yolov8n.pt
                # yolov8s.pt
                model_name = f"{self.config.model_variant}.pt"

                self._model = YOLO(model_name)

            logger.info("YOLO 模型載入完成")

        except Exception as e:
            logger.exception("YOLO 模型載入失敗")
            raise e

    @torch.no_grad()
    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: Optional[float] = None,
    ) -> List[DetectionResult]:
        """
        偵測影像中的車牌。

        Args:
            image:
                RGB ndarray, shape (H, W, 3)

            confidence_threshold:
                覆蓋 config 的 threshold

        Returns:
            List[DetectionResult]
        """

        threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else self.config.confidence_threshold
        )

        results: List[DetectionResult] = []

        h, w = image.shape[:2]

        # YOLO inference
        predictions = self._model.predict(
            source=image,
            conf=threshold,
            device=self.device,
            verbose=False,
        )

        if len(predictions) == 0:
            return results

        pred = predictions[0]

        if pred.boxes is None:
            return results

        for box in pred.boxes:

            xyxy = box.xyxy[0].cpu().numpy().tolist()

            x1, y1, x2, y2 = xyxy

            score = float(box.conf[0].cpu().item())

            class_id = int(box.cls[0].cpu().item())

            # Clamp bbox
            x1 = max(0.0, min(float(x1), w))
            y1 = max(0.0, min(float(y1), h))
            x2 = max(0.0, min(float(x2), w))
            y2 = max(0.0, min(float(y2), h))

            if x2 <= x1 or y2 <= y1:
                continue

            results.append(
                DetectionResult(
                    bbox=BoundingBox(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                    ),
                    detector_score=score,
                    class_id=class_id,
                    class_name="license_plate",
                )
            )

        # 依 confidence 排序
        results.sort(
            key=lambda r: r.detector_score,
            reverse=True,
        )

        logger.debug(
            f"偵測到 {len(results)} 個車牌 "
            f"(threshold={threshold})"
        )

        return results

    def warmup(self) -> None:
        """
        使用 dummy image warmup YOLO model。
        """

        dummy = np.zeros((640, 640, 3), dtype=np.uint8)

        _ = self.detect(dummy)

        logger.debug("YOLO warmup 完成")