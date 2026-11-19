"""
真實模型的端到端測試：models/ 的 YOLO + PARSeq 權重、data/ 的測試集（CPU）。

缺權重或資料集時自動略過；只想跑快速測試時：python -m pytest -m "not integration"
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api import server
from src.detector import YOLODetector
from src.pipeline import LPRPipeline
from src.recognizer import PARSeqRecognizer
from src.utils import bgr_to_rgb, load_image_cv2, load_plate_csv

from conftest import AUTH, DATASET_DIR, PARSEQ_WEIGHTS, YOLO_WEIGHTS, requires

pytestmark = [pytest.mark.integration, requires(YOLO_WEIGHTS, PARSEQ_WEIGHTS)]

TEST_CSV = DATASET_DIR / "test" / "test_gt.csv"


@pytest.fixture(scope="module")
def samples():
    if not TEST_CSV.exists():
        pytest.skip(f"缺少 {TEST_CSV}")
    return load_plate_csv(TEST_CSV)[:5]


@pytest.fixture
def pipeline(app_config):
    return LPRPipeline(app_config)


def test_pipeline_reads_test_plates(pipeline, samples):
    correct = 0
    for path, gt in samples:
        best = pipeline.predict(path).best_plate()
        correct += best is not None and best.plate_text == gt
    assert correct == len(samples)


def test_detector_expects_bgr(app_config, samples):
    """YOLO 要吃 BGR：給 RGB 時分數會下降（之前 pipeline 的通道錯誤）。"""
    detector = YOLODetector(app_config.detector)
    bgr = load_image_cv2(samples[0][0])
    bgr_score = detector.detect(bgr, confidence_threshold=0.01)[0].detector_score
    rgb_dets = detector.detect(np.ascontiguousarray(bgr_to_rgb(bgr)), confidence_threshold=0.01)
    assert bgr_score >= app_config.detector.confidence_threshold
    assert not rgb_dets or rgb_dets[0].detector_score <= bgr_score + 1e-6


def test_recognizer_on_crop(app_config, samples):
    detector = YOLODetector(app_config.detector)
    recognizer = PARSeqRecognizer(app_config.recognizer)
    path, gt = samples[0]
    bgr = load_image_cv2(path)
    box = detector.detect(bgr)[0].bbox
    rgb = bgr_to_rgb(bgr)
    crop = rgb[int(box.y1):int(box.y2), int(box.x1):int(box.x2)]
    result = recognizer.recognize_batch([crop])[0]
    assert result.plate_text == gt and 0 < result.recognizer_score <= 1


def test_api_end_to_end(app_config, samples, monkeypatch):
    monkeypatch.setattr(server, "_pipeline", None)
    path, gt = samples[0]
    with TestClient(server.app) as client:
        r = client.post("/predict", files={"file": (path.name, path.read_bytes())}, headers=AUTH)
        assert r.status_code == 200
        body = r.json()
        assert body["plates"][0]["plate_text"] == gt
        assert client.get(f"/results/{body['job_id']}", headers=AUTH).json() == body
