"""
tests/test_pipeline.py

單元測試與整合測試。
使用 mock 模擬 RF-DETR 與 PARSeq 模型，不需要實際模型權重。
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.models import (
    BatchOutput,
    BoundingBox,
    DetectionResult,
    PlateResult,
    PredictionOutput,
    RecognitionResult,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_image() -> np.ndarray:
    """建立假的 RGB 影像 (100×200)"""
    return np.zeros((100, 200, 3), dtype=np.uint8)


@pytest.fixture
def sample_detection() -> DetectionResult:
    return DetectionResult(
        bbox=BoundingBox(x1=10, y1=20, x2=80, y2=50),
        detector_score=0.95,
    )


@pytest.fixture
def sample_recognition() -> RecognitionResult:
    return RecognitionResult(plate_text="ABC-1234", recognizer_score=0.92)


# ---------------------------------------------------------------------------
# 資料模型測試
# ---------------------------------------------------------------------------

class TestPredictionOutput:
    def test_best_plate_returns_highest_confidence(self):
        output = PredictionOutput(
            source_file="test.jpg",
            plates=[
                PlateResult(plate_text="AAA-0001", confidence=0.8,
                            detector_score=0.9, recognizer_score=0.89, bbox=[0,0,1,1]),
                PlateResult(plate_text="BBB-9999", confidence=0.95,
                            detector_score=0.97, recognizer_score=0.98, bbox=[0,0,1,1]),
            ],
        )
        assert output.best_plate().plate_text == "BBB-9999"

    def test_best_plate_empty(self):
        output = PredictionOutput(source_file="empty.jpg")
        assert output.best_plate() is None

    def test_json_serializable(self):
        output = PredictionOutput(
            source_file="car.jpg",
            plates=[
                PlateResult(plate_text="XYZ-5678", confidence=0.9,
                            detector_score=0.95, recognizer_score=0.95,
                            bbox=[10, 20, 80, 50]),
            ],
            processing_time_ms=42.5,
        )
        json_str = output.model_dump_json()
        data = json.loads(json_str)
        assert data["source_file"] == "car.jpg"
        assert data["plates"][0]["plate_text"] == "XYZ-5678"
        assert "job_id" in data
        assert "timestamp" in data


class TestBoundingBox:
    def test_width_height(self):
        bbox = BoundingBox(x1=10, y1=20, x2=90, y2=60)
        assert bbox.width == 80
        assert bbox.height == 40

    def test_to_list(self):
        bbox = BoundingBox(x1=1, y1=2, x2=3, y2=4)
        assert bbox.to_list() == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Utils 測試
# ---------------------------------------------------------------------------

class TestImageUtils:
    def test_crop_bbox_basic(self, dummy_image):
        from src.utils.image import crop_bbox
        crop = crop_bbox(dummy_image, 10, 20, 80, 50, padding=0.0)
        assert crop.shape == (30, 70, 3)

    def test_crop_bbox_with_padding(self, dummy_image):
        from src.utils.image import crop_bbox
        crop = crop_bbox(dummy_image, 10, 20, 80, 50, padding=0.1)
        # 加了 padding，裁切範圍應更大
        assert crop.shape[0] >= 30
        assert crop.shape[1] >= 70

    def test_crop_bbox_clamp_boundary(self, dummy_image):
        from src.utils.image import crop_bbox
        # bbox 超出影像邊界時不應報錯
        crop = crop_bbox(dummy_image, -10, -10, 250, 150, padding=0.0)
        h, w = dummy_image.shape[:2]
        assert crop.shape[0] <= h
        assert crop.shape[1] <= w

    def test_list_images(self, tmp_path):
        from src.utils.image import list_images
        (tmp_path / "a.jpg").touch()
        (tmp_path / "b.png").touch()
        (tmp_path / "c.txt").touch()  # 不支援
        imgs = list_images(tmp_path)
        names = {p.name for p in imgs}
        assert "a.jpg" in names
        assert "b.png" in names
        assert "c.txt" not in names


# ---------------------------------------------------------------------------
# Pipeline 整合測試（mock 模型）
# ---------------------------------------------------------------------------

class TestLPRPipeline:
    @pytest.fixture(autouse=True)
    def mock_models(self, sample_detection, sample_recognition):
        """Mock RF-DETR 與 PARSeq，避免載入真實模型"""
        with (
            patch("src.detector.rfdetr_detector.RFDETRDetector._load_model"),
            patch("src.recognizer.parseq_recognizer.PARSeqRecognizer._load_model"),
        ):
            from src.detector import RFDETRDetector
            from src.recognizer import PARSeqRecognizer

            RFDETRDetector.detect = MagicMock(return_value=[sample_detection])
            PARSeqRecognizer.recognize_batch = MagicMock(return_value=[sample_recognition])
            yield

    def test_predict_returns_prediction_output(self, tmp_path, dummy_image):
        import cv2
        from src.pipeline import LPRPipeline
        from src.config import PipelineConfig, DetectorConfig, RecognizerConfig

        # 儲存假影像
        img_path = tmp_path / "test.jpg"
        cv2.imwrite(str(img_path), dummy_image)

        cfg = PipelineConfig(output_dir=tmp_path)
        pipeline = LPRPipeline(config=cfg)
        result = pipeline.predict(img_path)

        assert isinstance(result, PredictionOutput)
        assert result.source_file == "test.jpg"
        assert len(result.plates) == 1
        assert result.plates[0].plate_text == "ABC-1234"
        assert 0 < result.plates[0].confidence <= 1.0

    def test_predict_saves_json(self, tmp_path, dummy_image):
        import cv2
        from src.pipeline import LPRPipeline
        from src.config import PipelineConfig

        img_path = tmp_path / "car.jpg"
        cv2.imwrite(str(img_path), dummy_image)

        cfg = PipelineConfig(output_dir=tmp_path)
        pipeline = LPRPipeline(config=cfg)
        pipeline.predict(img_path, save_json=True)

        json_files = list(tmp_path.glob("*.json"))
        assert len(json_files) == 1
        data = json.loads(json_files[0].read_text())
        assert data["plates"][0]["plate_text"] == "ABC-1234"

    def test_predict_folder(self, tmp_path, dummy_image):
        import cv2
        from src.pipeline import LPRPipeline
        from src.config import PipelineConfig

        img_dir = tmp_path / "imgs"
        img_dir.mkdir()
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        for i in range(3):
            cv2.imwrite(str(img_dir / f"car{i}.jpg"), dummy_image)

        cfg = PipelineConfig(output_dir=out_dir)
        pipeline = LPRPipeline(config=cfg)
        batch = pipeline.predict_folder(img_dir, save_json=False)

        assert batch.total_images == 3
        assert batch.success_count == 3
        assert batch.fail_count == 0


# ---------------------------------------------------------------------------
# API 測試
# ---------------------------------------------------------------------------

class TestAPI:
    @pytest.fixture(autouse=True)
    def mock_pipeline(self, sample_detection, sample_recognition):
        with (
            patch("src.detector.rfdetr_detector.RFDETRDetector._load_model"),
            patch("src.recognizer.parseq_recognizer.PARSeqRecognizer._load_model"),
        ):
            from src.detector import RFDETRDetector
            from src.recognizer import PARSeqRecognizer
            RFDETRDetector.detect = MagicMock(return_value=[sample_detection])
            RFDETRDetector.warmup = MagicMock()
            PARSeqRecognizer.recognize_batch = MagicMock(return_value=[sample_recognition])
            PARSeqRecognizer.warmup = MagicMock()
            yield

    @pytest.fixture
    def client(self):
        from httpx import AsyncClient, ASGITransport
        from src.api import app
        return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")

    @pytest.mark.asyncio
    async def test_health_endpoint(self, client):
        async with client as c:
            resp = await c.get("/health")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_predict_endpoint(self, client, dummy_image):
        import cv2
        import io

        _, buf = cv2.imencode(".jpg", dummy_image)
        img_bytes = buf.tobytes()

        async with client as c:
            resp = await c.post(
                "/predict",
                files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert "plates" in data
        assert data["plates"][0]["plate_text"] == "ABC-1234"
