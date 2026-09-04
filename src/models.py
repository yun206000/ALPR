"""
models.py — 共用 Pydantic schema，定義所有輸入輸出結構。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def to_list(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]


class DetectionResult(BaseModel):
    """單一物件的偵測結果（RF-DETR 輸出）"""
    bbox: BoundingBox
    detector_score: float = Field(..., ge=0.0, le=1.0)
    class_id: int = 0
    class_name: str = "licence_plate"


# ---------------------------------------------------------------------------
# Recognition
# ---------------------------------------------------------------------------

class RecognitionResult(BaseModel):
    """單一裁切影像的 PARSeq 辨識結果"""
    plate_text: str
    recognizer_score: float = Field(..., ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Combined plate result
# ---------------------------------------------------------------------------

class PlateResult(BaseModel):
    """一張車牌的完整辨識結果（偵測 + 辨識）"""
    plate_text: str
    confidence: float = Field(..., description="綜合可信度 = detector × recognizer")
    detector_score: float
    recognizer_score: float
    bbox: List[float] = Field(..., description="[x1, y1, x2, y2]")


# ---------------------------------------------------------------------------
# Job / file-level output
# ---------------------------------------------------------------------------

class PredictionOutput(BaseModel):
    """單一影像的完整辨識輸出，對應 JSON 結構化輸出"""
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    source_file: str = Field(..., description="原始檔名")
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    plates: List[PlateResult] = Field(default_factory=list)
    processing_time_ms: float = 0.0
    # error: Optional[str] = None

    def best_plate(self) -> Optional[PlateResult]:
        """回傳可信度最高的車牌"""
        if not self.plates:
            return None
        return max(self.plates, key=lambda p: p.confidence)


class BatchOutput(BaseModel):
    """批次辨識結果"""
    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    total_images: int
    results: List[PredictionOutput]
