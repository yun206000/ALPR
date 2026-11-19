"""
測試共用的 fixture 與假物件。

- 單元測試用假的 YOLO / PARSeq，不需要模型權重，幾秒內跑完
- 標記 integration 的測試載入真實權重與資料集，缺檔案時自動略過
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src import config as config_module  # noqa: E402
from src.models import DetectionResult, BoundingBox, RecognitionResult  # noqa: E402

TEST_TOKEN = "t" * 32
AUTH = {"Authorization": f"Bearer {TEST_TOKEN}"}

DATASET_DIR = REPO_ROOT / "data" / "taiwan_plate_dataset"
YOLO_WEIGHTS = REPO_ROOT / "models" / "yolo" / "finetuned" / "best.pt"
PARSEQ_WEIGHTS = REPO_ROOT / "models" / "parseq" / "finetuned" / "best.ckpt"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """每個測試都在專案根目錄執行，結束後還原環境變數與全域設定。"""
    monkeypatch.chdir(REPO_ROOT)
    saved_env = dict(os.environ)
    monkeypatch.setattr(config_module, "_config", None)
    yield
    os.environ.clear()
    os.environ.update(saved_env)


@pytest.fixture
def app_config(tmp_path, monkeypatch):
    """以專案的 config.yaml 為基礎，輸出 / 實驗紀錄改到暫存資料夾，並設定測試用 token。"""
    for key in [k for k in os.environ if k.startswith("ALPR_")]:
        monkeypatch.delenv(key)
    monkeypatch.setenv("ALPR_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("ALPR_TRACKING__ENABLED", "false")
    monkeypatch.setenv("ALPR_TRACKING__TRACKIO_DIR", str(tmp_path / "trackio"))
    monkeypatch.setenv("ALPR_API__TOKENS", f'{{"test": "{TEST_TOKEN}"}}')
    monkeypatch.setenv("ALPR_DETECTOR__DEVICE", "cpu")
    monkeypatch.setenv("ALPR_RECOGNIZER__DEVICE", "cpu")
    return config_module.load_config(REPO_ROOT / "config.yaml")


def make_image(w: int = 320, h: int = 240, color=(40, 80, 160)) -> np.ndarray:
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = color
    return img


def jpg_bytes(w: int = 320, h: int = 240) -> bytes:
    ok, buf = cv2.imencode(".jpg", make_image(w, h))
    assert ok
    return buf.tobytes()


def write_jpg(path: Path, w: int = 320, h: int = 240) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), make_image(w, h))
    return path


class FakeDetector:
    """取代 YOLODetector：回傳固定的偵測框，並記錄收到的影像。"""

    def __init__(self, config=None, detections: List[DetectionResult] | None = None):
        self.config = config
        self.calls: List[np.ndarray] = []
        self.warmed = False
        self.detections = detections if detections is not None else [
            DetectionResult(bbox=BoundingBox(x1=10, y1=20, x2=110, y2=60), detector_score=0.9),
            DetectionResult(bbox=BoundingBox(x1=150, y1=100, x2=250, y2=140), detector_score=0.6),
        ]

    def detect(self, image, confidence_threshold=None):
        self.calls.append(image)
        return list(self.detections)

    def warmup(self):
        self.warmed = True


class FakeRecognizer:
    """取代 PARSeqRecognizer：依序回傳固定文字。"""

    def __init__(self, config=None, texts=("ABC-1234", "XYZ-9999"), scores=(0.9, 0.95)):
        self.config = config
        self.texts, self.scores = texts, scores
        self.crops: List[np.ndarray] = []
        self.warmed = False

    def recognize_batch(self, crops):
        self.crops.extend(crops)
        return [
            RecognitionResult(plate_text=self.texts[i % len(self.texts)], recognizer_score=self.scores[i % len(self.scores)])
            for i in range(len(crops))
        ]

    def warmup(self):
        self.warmed = True


@pytest.fixture
def fake_pipeline(app_config, monkeypatch):
    """用假偵測器 / 辨識器建立真正的 LPRPipeline（其餘流程都是真的）。"""
    from src.pipeline import lpr_pipeline

    monkeypatch.setattr(lpr_pipeline, "YOLODetector", FakeDetector)
    monkeypatch.setattr(lpr_pipeline, "PARSeqRecognizer", FakeRecognizer)
    return lpr_pipeline.LPRPipeline(app_config)


def requires(*paths: Path):
    missing = [str(p.relative_to(REPO_ROOT)) for p in paths if not p.exists()]
    return pytest.mark.skipif(bool(missing), reason=f"缺少 {', '.join(missing)}")


def ns(**kwargs):
    return SimpleNamespace(**kwargs)
