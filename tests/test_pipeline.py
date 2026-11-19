"""LPRPipeline：偵測 → 裁切 → 辨識 → 輸出 JSON（假模型）。"""

import json

import pytest

from src.models import BatchOutput, PredictionOutput

from conftest import make_image, write_jpg


def test_predict_ndarray_flow(fake_pipeline):
    img_bgr = make_image(320, 240, color=(10, 20, 30))
    out = fake_pipeline.predict(img_bgr, source_name="car.jpg")

    # YOLO 收到原本的 BGR 影像；PARSeq 收到的是 RGB 裁切
    assert fake_pipeline.detector.calls[0] is img_bgr
    assert fake_pipeline.recognizer.crops[0][0, 0].tolist() == [30, 20, 10]
    assert fake_pipeline.recognizer.crops[0].shape == (44, 110, 3)  # bbox 100×40 外擴 5%

    assert (out.source_file, out.image_width, out.image_height, out.error) == ("car.jpg", 320, 240, None)
    # 綜合可信度 = detector × recognizer，由高到低排序
    assert [p.confidence for p in out.plates] == [pytest.approx(0.81), pytest.approx(0.57)]
    assert out.best_plate().plate_text == "ABC-1234"
    assert out.plates[0].bbox == [10, 20, 110, 60]
    assert fake_pipeline.predict(img_bgr).source_file == "unknown.jpg"


def test_predict_no_detection(fake_pipeline):
    fake_pipeline.detector.detections = []
    out = fake_pipeline.predict(make_image())
    assert out.plates == [] and fake_pipeline.recognizer.crops == []


def test_predict_path_and_save_json(fake_pipeline, tmp_path):
    path = write_jpg(tmp_path / "car.jpg")
    out = fake_pipeline.predict(path, save_json=True)
    saved = fake_pipeline.config.output_dir / f"car_{out.job_id[:8]}.json"
    assert PredictionOutput.model_validate_json(saved.read_text(encoding="utf-8")) == out
    assert not list(fake_pipeline.config.output_dir.glob("*.tmp"))


def test_predict_unreadable_file(fake_pipeline, tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    out = fake_pipeline.predict(bad)
    assert out.error.startswith("無法載入影像") and out.plates == []
    assert fake_pipeline.detector.calls == []


def test_predict_folder_and_batch_json(fake_pipeline, tmp_path):
    folder = tmp_path / "imgs"
    for name in ("a.jpg", "b.png"):
        write_jpg(folder / name)
    (folder / "notes.txt").write_text("x")

    batch = fake_pipeline.predict_folder(folder)
    assert [r.source_file for r in batch.results] == ["a.jpg", "b.png"] and batch.total_images == 2
    saved = fake_pipeline.config.output_dir / f"batch_{batch.batch_id[:8]}.json"
    assert json.loads(saved.read_text(encoding="utf-8"))["batch_id"] == batch.batch_id

    custom = tmp_path / "out" / "custom.json"
    fake_pipeline.predict_folder(folder, batch_output_path=custom)
    assert BatchOutput.model_validate_json(custom.read_text(encoding="utf-8")).total_images == 2

    before = set(fake_pipeline.config.output_dir.iterdir())
    fake_pipeline.predict_batch([folder / "a.jpg"], save_json=False)
    assert set(fake_pipeline.config.output_dir.iterdir()) == before


def test_predict_folder_empty(fake_pipeline, tmp_path):
    with pytest.raises(ValueError, match="找不到支援的影像"):
        fake_pipeline.predict_folder(tmp_path)


def test_warmup_and_default_config(app_config, monkeypatch):
    from conftest import FakeDetector, FakeRecognizer
    from src.pipeline import lpr_pipeline

    monkeypatch.setattr(lpr_pipeline, "YOLODetector", FakeDetector)
    monkeypatch.setattr(lpr_pipeline, "PARSeqRecognizer", FakeRecognizer)
    pipeline = lpr_pipeline.LPRPipeline()  # 沒給 config → 用 get_config()
    assert pipeline.config is app_config
    assert app_config.output_dir.is_dir()
    pipeline.warmup()
    assert pipeline.detector.warmed and pipeline.recognizer.warmed
