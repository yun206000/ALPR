"""
models.py — 共用 Pydantic schema，定義所有輸入輸出結構。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Detection / Recognition（模組內部使用）
# ---------------------------------------------------------------------------

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    def to_list(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]


class DetectionResult(BaseModel):
    """單一車牌的偵測結果（YOLO 輸出）"""
    bbox: BoundingBox
    detector_score: float = Field(..., ge=0.0, le=1.0)


class RecognitionResult(BaseModel):
    """單一裁切影像的 PARSeq 辨識結果"""
    plate_text: str
    recognizer_score: float = Field(..., ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# 對外輸出
# ---------------------------------------------------------------------------

class PlateResult(BaseModel):
    """一張車牌的完整辨識結果（偵測 + 辨識）"""
    plate_text: str
    confidence: float = Field(..., description="綜合可信度 = detector × recognizer")
    detector_score: float
    recognizer_score: float
    bbox: List[float] = Field(..., description="[x1, y1, x2, y2]")


_PREDICTION_EXAMPLE = {
    "job_id": "3f1c2a9e-8d4b-4c41-9a51-6c2f0d7b1e20",
    "timestamp": "2026-11-04T12:00:00.000000+00:00",
    "source_file": "car.jpg",
    "image_width": 1920,
    "image_height": 1080,
    "plates": [
        {
            "plate_text": "ABC-1234",
            "confidence": 0.9312,
            "detector_score": 0.9521,
            "recognizer_score": 0.9781,
            "bbox": [812.4, 640.1, 1015.7, 702.9],
        }
    ],
    "processing_time_ms": 85.31,
    "error": None,
}


class PredictionOutput(BaseModel):
    """單一影像的完整辨識輸出"""
    model_config = ConfigDict(json_schema_extra={"examples": [_PREDICTION_EXAMPLE]})

    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=_now)
    source_file: str = Field(..., description="原始檔名")
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    plates: List[PlateResult] = Field(default_factory=list)
    processing_time_ms: float = 0.0
    error: Optional[str] = Field(None, description="影像讀取 / 解碼失敗時的錯誤訊息")

    def best_plate(self) -> Optional[PlateResult]:
        """回傳可信度最高的車牌"""
        if not self.plates:
            return None
        return max(self.plates, key=lambda p: p.confidence)


class BatchOutput(BaseModel):
    """批次辨識結果"""
    model_config = ConfigDict(json_schema_extra={"examples": [{
        "batch_id": "a7d0e3b2-5f6c-4e8a-b1d9-2c3e4f5a6b7c",
        "timestamp": "2026-11-04T12:00:00.000000+00:00",
        "total_images": 2,
        "results": [
            _PREDICTION_EXAMPLE,
            {**_PREDICTION_EXAMPLE, "job_id": "c1d2e3f4-0000-4000-8000-000000000000", "source_file": "broken.jpg",
             "image_width": None, "image_height": None, "plates": [], "processing_time_ms": 0.0,
             "error": "無法解碼影像：broken.jpg"},
        ],
    }]})

    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=_now)
    total_images: int
    results: List[PredictionOutput]


class ErrorResponse(BaseModel):
    """API 錯誤回應（所有 4xx / 5xx 都是這個格式）"""
    detail: str = Field(..., examples=["缺少或無效的 API token"])


# ---------------------------------------------------------------------------
# 評估
# ---------------------------------------------------------------------------

class EvalSample(BaseModel):
    """單張影像的評估結果"""
    source_file: str
    gt_text: str
    pred_text: Optional[str] = Field(None, description="可信度最高的車牌；未偵測到為 null")
    correct: bool
    edit_distance: int = Field(..., description="預測與 GT 的字元編輯距離")


class EvalReport(BaseModel):
    """整份 ground truth 的評估結果"""
    timestamp: str = Field(default_factory=_now)
    gt_csv: str
    total: int
    correct: int
    accuracy: float = Field(..., description="整串完全正確的比例")
    char_accuracy: float = Field(..., description="1 - 總編輯距離 / GT 總字元數")
    no_plate: int = Field(..., description="未偵測到車牌的張數")
    samples: List[EvalSample]

    def mismatches(self) -> List[EvalSample]:
        return [s for s in self.samples if not s.correct]
