"""YOLODetector 與 PARSeqRecognizer（用假的 Ultralytics / torch.hub 模型）。"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.config import DetectorConfig, RecognizerConfig
from src.detector import yolo_detector
from src.detector.yolo_detector import YOLODetector
from src.recognizer import parseq_recognizer
from src.recognizer.parseq_recognizer import PARSeqRecognizer, normalize_plate_text, postprocess_plate
from src.recognizer.transforms import get_transform

from conftest import make_image


# ---------------------------------------------------------------------------
# YOLODetector
# ---------------------------------------------------------------------------

def _box(xyxy, conf):
    return SimpleNamespace(xyxy=torch.tensor([xyxy], dtype=torch.float32), conf=torch.tensor([conf]))


class FakeYOLO:
    predictions = []
    last_kwargs = None

    def __init__(self, weights):
        self.weights = weights

    def predict(self, **kwargs):
        FakeYOLO.last_kwargs = kwargs
        return FakeYOLO.predictions


@pytest.fixture
def detector(tmp_path, monkeypatch):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"x")
    monkeypatch.setattr(yolo_detector, "YOLO", FakeYOLO)
    return YOLODetector(DetectorConfig(weights_path=weights, confidence_threshold=0.4, device="cpu"))


def test_detector_missing_weights(tmp_path):
    with pytest.raises(FileNotFoundError, match="找不到 YOLO 權重"):
        YOLODetector(DetectorConfig(weights_path=tmp_path / "no.pt", confidence_threshold=0.4, device="cpu"))


def test_detector_sorts_clamps_and_drops_invalid(detector):
    FakeYOLO.predictions = [SimpleNamespace(boxes=[
        _box([10, 10, 50, 30], 0.5),
        _box([-5, 5, 400, 30], 0.9),     # 超出影像 → clamp 到 [0, 320]
        _box([330, 10, 360, 30], 0.8),   # 完全在影像外 → clamp 後寬度為 0，丟掉
    ])]
    results = detector.detect(make_image(320, 240))
    assert [r.detector_score for r in results] == pytest.approx([0.9, 0.5])
    assert results[0].bbox.to_list() == [0, 5, 320, 30]
    assert FakeYOLO.last_kwargs["conf"] == 0.4 and FakeYOLO.last_kwargs["device"] == "cpu"

    detector.detect(make_image(), confidence_threshold=0.1)
    assert FakeYOLO.last_kwargs["conf"] == 0.1


@pytest.mark.parametrize("predictions", [[], [SimpleNamespace(boxes=None)]])
def test_detector_no_predictions(detector, predictions):
    FakeYOLO.predictions = predictions
    assert detector.detect(make_image()) == []
    detector.warmup()


# ---------------------------------------------------------------------------
# 後處理 / 前處理
# ---------------------------------------------------------------------------

def test_postprocess_and_normalize():
    assert postprocess_plate("abo-1i23") == "AB0-1L23"
    assert normalize_plate_text("  ab*c-12o\n", "ABC-12O") == "ABC-120"
    assert normalize_plate_text(" ab#", "") == "AB#"  # charset 為空時不過濾


@pytest.mark.parametrize("augment", [False, True])
def test_transform_output_shape(augment):
    from PIL import Image

    tensor = get_transform((32, 128), augment=augment)(Image.new("RGB", (200, 60), (255, 0, 0)))
    assert tensor.shape == (3, 32, 128)
    assert tensor.min() >= -1 and tensor.max() <= 1


# ---------------------------------------------------------------------------
# PARSeqRecognizer
# ---------------------------------------------------------------------------

class FakeParseq(torch.nn.Module):
    def __init__(self, texts=("abc-1234",)):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.hparams = SimpleNamespace(img_size=(32, 128))
        self.texts = texts
        self.tokenizer = SimpleNamespace(decode=self._decode)
        self.batch_sizes = []

    def forward(self, x):
        self.batch_sizes.append(x.shape[0])
        assert x.shape[1:] == (3, 32, 128)
        return torch.zeros(x.shape[0], 8, 4)

    def _decode(self, probs):
        n = probs.shape[0]
        return [self.texts[i % len(self.texts)] for i in range(n)], [torch.tensor([0.5, 1.0])] * n


def _rec_cfg(weights: Path):
    return RecognizerConfig(weights_path=weights, model_variant="parseq_tiny", charset="0123456789ABCDEFGHJKLMNPQRSTUVWXYZ-OI",
                            device="cpu", crop_padding=0.05)


def test_recognizer_falls_back_to_hub_pretrained(tmp_path, monkeypatch):
    calls = []

    def fake_load(repo, variant, **kw):
        calls.append((repo, variant, kw))
        return FakeParseq(texts=("ab o-1i2*",))

    monkeypatch.setattr(parseq_recognizer.torch.hub, "load", fake_load)
    rec = PARSeqRecognizer(_rec_cfg(tmp_path / "missing.ckpt"))
    assert calls[0][1] == "parseq_tiny" and calls[0][2]["pretrained"] is True

    results = rec.recognize_batch([make_image(100, 30), make_image(80, 20)])
    assert [r.plate_text for r in results] == ["AB0-1L2", "AB0-1L2"]  # 過濾 charset 外的字元 + 後處理
    assert results[0].recognizer_score == pytest.approx(0.75)
    assert rec.recognize_batch([]) == []
    rec.warmup()
    assert rec._model.batch_sizes == [2, 1]


def test_recognizer_loads_checkpoint(tmp_path, monkeypatch):
    model = FakeParseq()
    ckpt_path = tmp_path / "best.ckpt"
    torch.save({"model_state_dict": model.state_dict(), "hyper_parameters": {"max_label_length": 9}}, ckpt_path)
    calls = []

    def fake_load(repo, variant, **kw):
        calls.append((variant, kw))
        return FakeParseq()

    monkeypatch.setattr(parseq_recognizer.torch.hub, "load", fake_load)
    PARSeqRecognizer(_rec_cfg(ckpt_path))
    assert calls == [("parseq", {"pretrained": False, "trust_repo": True, "verbose": False, "max_label_length": 9})]


def test_recognizer_checkpoint_refuses_pickled_objects(tmp_path, monkeypatch):
    """weights_only=True：checkpoint 裡有任意 Python 物件時拒絕載入。"""
    ckpt_path = tmp_path / "evil.ckpt"
    torch.save({"model_state_dict": {}, "hyper_parameters": {}, "obj": SimpleNamespace(a=1)}, ckpt_path)
    monkeypatch.setattr(parseq_recognizer.torch.hub, "load", lambda *a, **k: FakeParseq())
    with pytest.raises(Exception, match="Weights only load failed"):
        PARSeqRecognizer(_rec_cfg(ckpt_path))
