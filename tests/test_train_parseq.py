"""src/train/train.py：資料集、驗證裁切、評估、loss 曲線，以及完整訓練流程（1 epoch、合成資料）。"""

import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

import src.detector
from src.models import BoundingBox, DetectionResult
from src.train import train as train_mod

from conftest import FakeDetector, PARSEQ_WEIGHTS, YOLO_WEIGHTS, requires

HUB_CACHE = Path(torch.hub.get_dir())
PARSEQ_HUB_CODE = HUB_CACHE / "baudm_parseq_main"
PARSEQ_HUB_WEIGHTS = HUB_CACHE / "checkpoints" / "parseq-bb5792a6.pt"


def _plate_image(path: Path, text: str, size=(128, 32)) -> Path:
    """畫一張白底黑字的「車牌」。"""
    w, h = size
    img = np.full((h, w, 3), 255, np.uint8)
    scale = h / 40
    cv2.putText(img, text, (int(w * 0.05), int(h * 0.75)), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), max(1, int(scale * 2)))
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return path


# ---------------------------------------------------------------------------
# 小元件
# ---------------------------------------------------------------------------

def test_plate_dataset_accepts_path_or_pil(tmp_path):
    path = _plate_image(tmp_path / "a.jpg", "ABC")
    ds = train_mod.PlateDataset([(path, "ABC"), (Image.new("RGB", (50, 20)), "XYZ")], (32, 128))
    assert len(ds) == 2
    x0, y0 = ds[0]
    x1, y1 = ds[1]
    assert x0.shape == x1.shape == (3, 32, 128) and (y0, y1) == ("ABC", "XYZ")
    images, labels = train_mod.collate_fn([ds[0], ds[1]])
    assert images.shape == (2, 3, 32, 128) and labels == ["ABC", "XYZ"]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_crop_plates(tmp_path, app_config, monkeypatch, device):
    a = _plate_image(tmp_path / "a.jpg", "ABC", size=(200, 100))
    b = _plate_image(tmp_path / "b.jpg", "XYZ", size=(200, 100))
    detectors = []

    class Detector(FakeDetector):
        def __init__(self, config):
            super().__init__(config, detections=[
                DetectionResult(bbox=BoundingBox(x1=20, y1=10, x2=120, y2=50), detector_score=0.9),
                DetectionResult(bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10), detector_score=0.5),
            ])
            detectors.append(self)

        def detect(self, image, confidence_threshold=None):
            self.calls.append(image)
            return list(self.detections) if len(self.calls) == 1 else []  # 第二張偵測不到

    monkeypatch.setattr(src.detector, "YOLODetector", Detector)
    emptied = []
    monkeypatch.setattr(train_mod.torch.cuda, "empty_cache", lambda: emptied.append(True))

    out = train_mod.crop_plates([(a, "ABC"), (b, "XYZ")], app_config, torch.device(device))
    det = detectors[0]
    assert det.config.device == device
    # 送進 YOLO 的是 BGR：白底黑字看不出來，所以比對和原檔（cv2 讀進來就是 BGR）相同
    assert np.array_equal(det.calls[0], cv2.imread(str(a)))
    assert out[0][0].size == (110, 44) and out[0][1] == "ABC"   # 分數最高的框 + 5% padding
    assert out[1][0].size == (200, 100) and out[1][1] == "XYZ"  # 偵測不到 → 整張
    assert emptied == ([True] if device == "cuda" else [])


def test_crop_plates_all_detected(tmp_path, app_config, monkeypatch):
    a = _plate_image(tmp_path / "a.jpg", "ABC", size=(200, 100))
    monkeypatch.setattr(src.detector, "YOLODetector", FakeDetector)
    out = train_mod.crop_plates([(a, "ABC")], app_config, torch.device("cpu"))
    assert out[0][0].size == (110, 44)


class _FakeParseq:
    def __init__(self, preds):
        self.preds = preds
        self.mode = []
        self.tokenizer = SimpleNamespace(decode=lambda probs: (self.preds[: probs.shape[0]], None))

    def __call__(self, images):
        return torch.zeros(images.shape[0], 3, 5)

    def eval(self):
        self.mode.append("eval")

    def train(self):
        self.mode.append("train")


def test_evaluate_uses_inference_postprocessing():
    model = _FakeParseq(["abc-1o23", "XYZ-0000", "ab"])
    loader = [(torch.zeros(3, 3, 32, 128), ["ABC-1023", "XYZ-0001", "AB "])]
    charset = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ-OI"
    assert train_mod.evaluate(model, loader, torch.device("cpu"), charset) == pytest.approx(2 / 3)
    assert model.mode == ["eval", "train"]
    assert train_mod.evaluate(model, [], torch.device("cpu"), charset) == 0


def test_draw_loss_curve(tmp_path):
    out = tmp_path / "loss.png"
    train_mod.draw_loss_curve([1.0, 0.5], [0.8, 0.6], out)
    assert Image.open(out).size[0] > 100


# ---------------------------------------------------------------------------
# 完整訓練流程（真實 PARSeq，CPU、合成資料、1 epoch）
# ---------------------------------------------------------------------------

def _dataset(tmp_path):
    texts = ["ABC-1234", "XYZ-5678", "KLM-0099", "RTS-4321"]
    train_rows = [f"{_plate_image(tmp_path / 'train' / f'{i}.jpg', t)},{t}" for i, t in enumerate(texts)]
    val_rows = [f"{_plate_image(tmp_path / 'val' / f'{i}.jpg', t, size=(320, 240))},{t}" for i, t in enumerate(texts[:2])]
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    train_csv.write_text("image_path,plate_text\n" + "\n".join(train_rows) + "\n", encoding="utf-8")
    val_csv.write_text("image_path,plate_text\n" + "\n".join(val_rows) + "\n", encoding="utf-8")
    return train_csv, val_csv


def _run_main(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["train.py", "--epochs", "1", "--batch-size", "2", "--num-workers", "0",
                                      "--device", "cpu", *map(str, args)])
    train_mod.main()


@pytest.mark.integration
@requires(PARSEQ_HUB_CODE, PARSEQ_WEIGHTS, YOLO_WEIGHTS)
def test_main_finetune_from_checkpoint_replaces_output(tmp_path, app_config, monkeypatch):
    train_csv, val_csv = _dataset(tmp_path)
    out_dir = tmp_path / "parseq_out"
    out_dir.mkdir()
    (out_dir / "old.txt").write_text("old")

    # 讓 epoch 0 的準確率一定比較差，確保會產生 best.ckpt（其餘照常用真實模型計算）
    real_evaluate = train_mod.evaluate
    calls = []

    def evaluate(*a, **k):
        acc = real_evaluate(*a, **k)
        calls.append(acc)
        return -1.0 if len(calls) == 1 else acc

    monkeypatch.setattr(train_mod, "evaluate", evaluate)
    _run_main(monkeypatch, "--train-csv", train_csv, "--val-csv", val_csv, "--crop-val",
              "--finetune-from", PARSEQ_WEIGHTS, "--output-dir", out_dir)

    assert sorted(p.name for p in out_dir.iterdir()) == ["best.ckpt", "last.ckpt", "loss_curve.png", "train.log"]
    ckpt = torch.load(out_dir / "best.ckpt", weights_only=True)
    assert ckpt["epoch"] == 1 and set(ckpt) == {"epoch", "model_state_dict", "hyper_parameters", "val_acc"}
    log = (out_dir / "train.log").read_text(encoding="utf-8")
    assert "驗證影像已用 YOLO 裁出車牌" in log and "已取代" in log
    assert "要用這個權重推論" in log
    assert stat.S_IMODE(out_dir.stat().st_mode) != 0o700  # 不是暫存資料夾的 700，其他 UID（容器）讀得到


@pytest.mark.integration
@requires(PARSEQ_HUB_CODE, PARSEQ_HUB_WEIGHTS)
def test_main_keeps_old_weights_when_not_better(tmp_path, app_config, monkeypatch):
    train_csv, _ = _dataset(tmp_path)
    out_dir = tmp_path / "parseq_out"
    out_dir.mkdir()
    (out_dir / "best.ckpt").write_text("original")

    monkeypatch.setattr(train_mod, "evaluate", lambda *a, **k: 1.0)
    monkeypatch.setattr(train_mod, "evaluate_loss", lambda *a, **k: 0.0)
    # val_csv=null：從訓練集切驗證集（已是裁切好的車牌，不再用 YOLO 裁）；torch.hub 預訓練權重；凍結 encoder
    _run_main(monkeypatch, "--train-csv", train_csv, "--val-csv", "null", "--val-frac", "0.25",
              "--crop-val", "--freeze-encoder", "--finetune-from", "null", "--output-dir", out_dir)
    assert (out_dir / "best.ckpt").read_text() == "original"
    assert sorted(p.name for p in out_dir.iterdir()) == ["best.ckpt"]


@pytest.mark.integration
@requires(PARSEQ_HUB_CODE, PARSEQ_WEIGHTS)
def test_main_reports_when_weights_path_points_to_output(tmp_path, app_config, monkeypatch):
    train_csv, val_csv = _dataset(tmp_path)
    out_dir = tmp_path / "parseq_out"
    monkeypatch.setenv("ALPR_RECOGNIZER__WEIGHTS_PATH", str(out_dir / "best.ckpt"))
    accs = iter([-1.0, 0.5])  # epoch 0、epoch 1 的準確率
    monkeypatch.setattr(train_mod, "evaluate", lambda *a, **k: next(accs))
    _run_main(monkeypatch, "--train-csv", train_csv, "--val-csv", val_csv, "--no-crop-val",
              "--finetune-from", PARSEQ_WEIGHTS, "--output-dir", out_dir)
    assert "API Server 重新啟動後才會載入新權重" in (out_dir / "train.log").read_text(encoding="utf-8")
