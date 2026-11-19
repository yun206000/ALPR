"""main.py 的 CLI：--image / --folder / --eval / --serve。"""

import json
import os
import sys

import pytest

import main
from src.pipeline import lpr_pipeline

from conftest import FakeDetector, FakeRecognizer, REPO_ROOT, write_jpg


@pytest.fixture
def run(app_config, monkeypatch):
    """用假模型執行 main.run_cli()，回傳 stdout。"""
    monkeypatch.setattr(lpr_pipeline, "YOLODetector", FakeDetector)
    monkeypatch.setattr(lpr_pipeline, "PARSeqRecognizer", FakeRecognizer)

    def _run(capsys, *args):
        monkeypatch.setattr(sys, "argv", ["main.py", "--config", str(REPO_ROOT / "config.yaml"), *map(str, args)])
        main.run_cli()
        return capsys.readouterr().out

    return _run


def test_image(run, capsys, tmp_path, app_config):
    out = run(capsys, "--image", write_jpg(tmp_path / "car.jpg"), "--device", "cpu", "--log-level", "WARNING")
    assert json.loads(out)["plates"][0]["plate_text"] == "ABC-1234"
    assert os.environ["ALPR_DETECTOR__DEVICE"] == os.environ["ALPR_RECOGNIZER__DEVICE"] == "cpu"
    assert list(app_config.output_dir.glob("car_*.json"))


def test_image_unreadable_and_no_plate(run, capsys, tmp_path, monkeypatch):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"x")
    assert json.loads(run(capsys, "--image", bad, "--no-save"))["error"].startswith("無法載入影像")

    monkeypatch.setattr(FakeDetector, "detect", lambda self, image, confidence_threshold=None: [])
    assert json.loads(run(capsys, "--image", write_jpg(tmp_path / "empty.jpg"), "--no-save"))["plates"] == []


def test_folder(run, capsys, tmp_path, monkeypatch):
    folder = tmp_path / "imgs"
    write_jpg(folder / "a.jpg")
    write_jpg(folder / "b.jpg")
    calls = iter([[], None])  # 第一張沒偵測到 → 表格顯示 "-"
    real_detect = FakeDetector.detect

    def detect(self, image, confidence_threshold=None):
        result = next(calls, None)
        return result if result is not None else real_detect(self, image)

    monkeypatch.setattr(FakeDetector, "detect", detect)
    output = tmp_path / "batch.json"
    out = run(capsys, "--folder", folder, "-o", output)
    lines = out.splitlines()
    assert lines[0].split() == ["檔名", "車牌", "耗時(ms)"]
    assert lines[2].split()[:2] == ["a.jpg", "-"] and lines[3].split()[:2] == ["b.jpg", "ABC-1234"]
    assert json.loads(output.read_text(encoding="utf-8"))["total_images"] == 2


def test_eval(run, capsys, tmp_path, app_config):
    write_jpg(tmp_path / "a.jpg")
    write_jpg(tmp_path / "b.jpg")
    gt = tmp_path / "gt.csv"
    gt.write_text("image_path,plate_text\na.jpg,ABC-1234\nb.jpg,ZZZ-0000\n", encoding="utf-8")

    out = run(capsys, "--eval", gt)
    assert "完全正確    1（50.00%）" in out
    assert "錯誤樣本（2）" not in out and "錯誤樣本（1）" in out
    saved = list(app_config.output_dir.glob("eval_*.json"))
    assert len(saved) == 1 and json.loads(saved[0].read_text(encoding="utf-8"))["correct"] == 1

    run(capsys, "--eval", gt, "--no-save")
    assert len(list(app_config.output_dir.glob("eval_*.json"))) == 1

    out_path = tmp_path / "r" / "eval.json"
    run(capsys, "--eval", gt, "--image-root", tmp_path, "-o", out_path)
    assert out_path.exists()


def test_eval_report_no_plate_row(capsys):
    from src.models import EvalReport, EvalSample

    report = EvalReport(gt_csv="x", total=1, correct=0, accuracy=0, char_accuracy=0, no_plate=1, samples=[
        EvalSample(source_file="車牌.jpg", gt_text="ABC", pred_text=None, correct=False, edit_distance=3)])
    main._print_eval_report(report)
    assert "<未偵測到>" in capsys.readouterr().out

    report.samples[0].correct = True
    main._print_eval_report(report)
    assert "錯誤樣本" not in capsys.readouterr().out


def test_display_width_and_table_alignment(capsys):
    assert main._display_width("AB車牌") == 6
    main._print_table(("名稱", "值"), [("車牌", "1"), ("abcdef", "22")])
    lines = capsys.readouterr().out.splitlines()
    assert lines[1] == "------  --"
    assert lines[2] == "車牌    1"


def test_serve(run, capsys, monkeypatch):
    called = {}
    monkeypatch.setattr(main.uvicorn, "run", lambda app, **kw: called.update(app=app, **kw))
    run(capsys, "--serve", "--port", "9999")
    assert called == {"app": "src.api:app", "host": "0.0.0.0", "port": 9999, "workers": 1, "reload": False}


def test_mode_is_required(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py"])
    with pytest.raises(SystemExit):
        main.parse_args()
