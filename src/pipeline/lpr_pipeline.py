"""
pipeline/lpr_pipeline.py — 串接 YOLO 偵測與 PARSeq 辨識的車牌辨識流程。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from ..config import AppConfig, get_config
from ..detector import YOLODetector
from ..models import BatchOutput, PlateResult, PredictionOutput
from ..recognizer import PARSeqRecognizer
from ..utils import (
    crop_bbox,
    list_images,
    load_image_cv2,
    logger,
    bgr_to_rgb,
    setup_logger,
)


class LPRPipeline:
    """
    車牌辨識 Pipeline。

    功能：
    - 接收單張影像路徑 / numpy array
    - 執行 YOLO 車牌偵測
    - 裁切車牌區域
    - 執行 PARSeq 字元辨識
    - 回傳結構化 PredictionOutput（可序列化為 JSON）

    使用範例：
        pipeline = LPRPipeline()
        result = pipeline.predict("car.jpg")
        print(result.model_dump_json(indent=2))
    """

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or get_config()
        setup_logger(self.config.log_level, self.config.log_file)

        logger.info("初始化 LPR Pipeline...")
        self.detector = YOLODetector(self.config.detector)
        self.recognizer = PARSeqRecognizer(self.config.recognizer)

        # 確保輸出目錄存在
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("LPR Pipeline 初始化完成")

    def warmup(self) -> None:
        """預熱模型（建議在 server 啟動後呼叫）"""
        logger.info("模型 warm up...")
        self.detector.warmup()
        self.recognizer.warmup()
        logger.info("Warm up 完成")

    # ------------------------------------------------------------------
    # 核心辨識流程
    # ------------------------------------------------------------------

    def predict(
        self,
        source: Union[str, Path, np.ndarray],
        source_name: Optional[str] = None,
        save_json: bool = False,
    ) -> PredictionOutput:
        """
        對單張影像執行車牌辨識。

        Args:
            source: 影像路徑 (str/Path) 或 RGB ndarray
            source_name: 顯示用檔名（當 source 為 ndarray 時使用）
            save_json: 是否將結果存為 JSON 檔案

        Returns:
            PredictionOutput — 結構化辨識結果
        """
        t_start = time.perf_counter()

        # 載入影像
        if isinstance(source, (str, Path)):
            path = Path(source)
            filename = path.name
            try:
                img_bgr = load_image_cv2(path)
            except ValueError as e:
                return PredictionOutput(source_file=filename, error=str(e))
        else:
            img_bgr = source  # 假設傳入 BGR（cv2 格式）
            filename = source_name or "unknown.jpg"

        img_rgb = bgr_to_rgb(img_bgr)
        h, w = img_rgb.shape[:2]

        # 偵測車牌
        detections = self.detector.detect(img_rgb)

        # 裁切每個車牌並辨識字元
        plate_results: List[PlateResult] = []

        if detections:
            crops = [
                crop_bbox(
                    img_rgb,
                    d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2,
                    padding=self.config.recognizer.crop_padding,
                )
                for d in detections
            ]

            rec_results = self.recognizer.recognize_batch(crops)

            for det, rec in zip(detections, rec_results):
                combined_confidence = det.detector_score * rec.recognizer_score
                plate_results.append(
                    PlateResult(
                        plate_text=rec.plate_text,
                        confidence=round(combined_confidence, 4),
                        detector_score=round(det.detector_score, 4),
                        recognizer_score=round(rec.recognizer_score, 4),
                        bbox=det.bbox.to_list(),
                    )
                )

            # 依綜合可信度排序
            plate_results.sort(key=lambda p: p.confidence, reverse=True)

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        output = PredictionOutput(
            source_file=filename,
            image_width=w,
            image_height=h,
            plates=plate_results,
            processing_time_ms=round(elapsed_ms, 2),
        )

        logger.debug(
            f"[{filename}] 偵測到 {len(plate_results)} 個車牌 | "
            f"耗時 {elapsed_ms:.1f}ms"
        )

        if save_json:
            self._save_result(output)

        return output

    def predict_batch(
        self,
        sources: List[Union[str, Path]],
        save_json: bool = True,
        batch_output_path: Optional[Path] = None,
    ) -> BatchOutput:
        """
        批次對多張影像執行車牌辨識。

        Args:
            sources: 影像路徑列表
            save_json: 是否儲存批次整合 JSON（每張影像不會各自存檔）
            batch_output_path: 批次 JSON 的輸出路徑（預設 outputs/batch_<id>.json）

        Returns:
            BatchOutput — 批次結果
        """
        from tqdm import tqdm

        results: List[PredictionOutput] = []

        for src in tqdm(sources, desc="辨識進度", unit="img"):
            results.append(self.predict(src))

        batch = BatchOutput(
            total_images=len(sources),
            results=results,
        )

        if save_json:
            self.save_batch(batch, batch_output_path)
        return batch

    def save_batch(self, batch: BatchOutput, out_path: Optional[Path] = None) -> Path:
        """將整批結果存成一份 JSON（預設 <output_dir>/batch_<id>.json）"""
        out_path = out_path or (self.config.output_dir / f"batch_{batch.batch_id[:8]}.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(batch.model_dump_json(indent=2), encoding="utf-8")
        logger.info(f"批次結果已儲存：{out_path}")
        return out_path

    def predict_folder(
        self,
        folder: Union[str, Path],
        save_json: bool = True,
        batch_output_path: Optional[Path] = None,
    ) -> BatchOutput:
        """
        對資料夾內所有影像執行批次辨識。
        """
        folder = Path(folder)
        images = list_images(folder)
        if not images:
            raise ValueError(f"資料夾 {folder} 內找不到支援的影像檔案")
        logger.info(f"找到 {len(images)} 張影像於 {folder}")
        return self.predict_batch(images, save_json=save_json, batch_output_path=batch_output_path)

    def _save_result(self, output: PredictionOutput) -> Path:
        """將單張辨識結果存為 JSON"""
        stem = Path(output.source_file).stem
        out_path = self.config.output_dir / f"{stem}_{output.job_id[:8]}.json"
        out_path.write_text(
            output.model_dump_json(indent=2),
            encoding="utf-8",
        )
        logger.debug(f"結果已儲存：{out_path}")
        return out_path
