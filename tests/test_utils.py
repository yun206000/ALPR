"""src/utils：影像、資料集 CSV、檔案取代、logger、實驗紀錄。"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.config import TrackingConfig
from src.utils import bgr_to_rgb, crop_bbox, list_images, load_image_cv2, load_plate_csv, logger, setup_logger
from src.utils.files import replace_dir
from src.utils.tracking import ExperimentTracker, make_run_name

from conftest import make_image, write_jpg


# ---------------------------------------------------------------------------
# image
# ---------------------------------------------------------------------------

def test_load_image_cv2_ok_and_missing(tmp_path):
    path = write_jpg(tmp_path / "a.jpg", 64, 32)
    assert load_image_cv2(path).shape == (32, 64, 3)
    with pytest.raises(ValueError, match="無法載入影像"):
        load_image_cv2(tmp_path / "missing.jpg")


def test_bgr_to_rgb_swaps_channels():
    img = make_image(4, 4, color=(1, 2, 3))
    assert bgr_to_rgb(img)[0, 0].tolist() == [3, 2, 1]


def test_crop_bbox_padding_and_clamp():
    img = np.arange(100 * 200 * 3, dtype=np.uint32).reshape(100, 200, 3)
    crop = crop_bbox(img, 50, 20, 150, 60, padding=0.1)  # 外擴 10 px / 4 px
    assert crop.shape[:2] == (48, 120)
    # 超出影像邊界的部分被截掉
    crop = crop_bbox(img, -10, -10, 250, 150, padding=0.0)
    assert crop.shape[:2] == (100, 200)


def test_list_images_filters_extensions(tmp_path):
    for name in ("b.PNG", "a.jpg", "c.txt", "d.webp"):
        (tmp_path / name).write_bytes(b"x")
    assert [p.name for p in list_images(tmp_path)] == ["a.jpg", "b.PNG", "d.webp"]


# ---------------------------------------------------------------------------
# dataset CSV
# ---------------------------------------------------------------------------

def _csv(path: Path, rows, header="image_path,plate_text"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n" + "".join(f"{a},{b}\n" for a, b in rows), encoding="utf-8")
    return path


def test_load_plate_csv_resolves_every_location(tmp_path):
    root = tmp_path / "ds"
    csv_dir = root / "split"
    write_jpg(csv_dir / "same.jpg")                    # 相對於 CSV 資料夾
    write_jpg(root / "split" / "images" / "img.jpg")   # CSV 資料夾底下的 images/
    write_jpg(root / "parent.jpg")                     # CSV 資料夾的上一層
    absolute = write_jpg(tmp_path / "abs.jpg")         # 絕對路徑
    csv_path = _csv(csv_dir / "gt.csv", [
        ("same.jpg", " ABC-1234 "), ("img.jpg", "B"), ("parent.jpg", "C"), (absolute, "D"),
    ])
    samples = load_plate_csv(csv_path)
    assert [s[1] for s in samples] == ["ABC-1234", "B", "C", "D"]
    assert all(p.is_absolute() and p.exists() for p, _ in samples)


def test_load_plate_csv_image_root(tmp_path):
    write_jpg(tmp_path / "imgs" / "x.jpg")
    csv_path = _csv(tmp_path / "csv" / "gt.csv", [("x.jpg", "AAA")])
    assert load_plate_csv(csv_path, image_root=tmp_path / "imgs")[0][1] == "AAA"


def test_load_plate_csv_errors(tmp_path):
    with pytest.raises(ValueError, match="缺少欄位"):
        load_plate_csv(_csv(tmp_path / "bad.csv", [("a", "b")], header="path,text"))
    with pytest.raises(FileNotFoundError, match="2 張影像找不到"):
        load_plate_csv(_csv(tmp_path / "missing.csv", [("nope.jpg", "A"), (tmp_path / "abs_missing.jpg", "B")]))


# ---------------------------------------------------------------------------
# replace_dir
# ---------------------------------------------------------------------------

def test_replace_dir_replaces_existing_and_creates_new(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "new.txt").write_text("new")
    dst = tmp_path / "dst"
    dst.mkdir()
    (dst / "old.txt").write_text("old")
    (tmp_path / "dst.new").mkdir()  # 上次失敗留下的殘骸會先被清掉

    replace_dir(src, dst)
    assert sorted(p.name for p in dst.iterdir()) == ["new.txt"]
    assert not (tmp_path / "dst.new").exists() and not (tmp_path / "dst.old").exists()

    replace_dir(src, tmp_path / "fresh")
    assert (tmp_path / "fresh" / "new.txt").read_text() == "new"


# ---------------------------------------------------------------------------
# logger
# ---------------------------------------------------------------------------

def test_setup_logger_writes_file(tmp_path):
    log_file = tmp_path / "logs" / "a.log"
    setup_logger("INFO", log_file)
    logger.debug("debug 也寫進檔案")
    assert "debug 也寫進檔案" in log_file.read_text(encoding="utf-8")
    setup_logger("INFO", tmp_path / "lpr_{time:YYYY-MM-DD}.log")  # 每日換檔
    logger.info("rotate")
    assert list(tmp_path.glob("lpr_*.log"))
    setup_logger("INFO")


def test_setup_logger_permission_error(tmp_path, monkeypatch, capsys):
    def deny(self, *a, **k):
        raise PermissionError

    monkeypatch.setattr(Path, "mkdir", deny)
    setup_logger("INFO", tmp_path / "x" / "a.log")
    monkeypatch.undo()
    assert "沒有權限寫入" in capsys.readouterr().err
    setup_logger("INFO")


# ---------------------------------------------------------------------------
# tracking
# ---------------------------------------------------------------------------

def test_make_run_name():
    assert make_run_name("yolo").startswith("yolo_2")


def test_tracker_disabled_is_noop(tmp_path):
    cfg = TrackingConfig(enabled=False, project="p", trackio_dir=tmp_path)
    with ExperimentTracker(cfg, "run") as tracker:
        tracker.log({"a": 1}, step=1)
    assert tracker._trackio is None


def test_tracker_logs_only_finite_numbers(tmp_path, monkeypatch):
    calls = []
    fake = SimpleNamespace(
        init=lambda **kw: calls.append(("init", kw)),
        log=lambda metrics, step: calls.append(("log", metrics, step)),
        finish=lambda: calls.append(("finish",)),
    )
    monkeypatch.setitem(sys.modules, "trackio", fake)
    cfg = TrackingConfig(enabled=True, project="p", trackio_dir=tmp_path)
    with ExperimentTracker(cfg, "run", {"lr": 1}) as tracker:
        tracker.log({"loss": "0.5", "nan": float("nan"), "text": "abc", "none": None}, step=3)
        tracker.log({"nan": float("inf")}, step=4)  # 沒有可紀錄的值就不呼叫 trackio
    assert calls[0][0] == "init" and calls[0][1]["config"] == {"lr": 1}
    assert calls[1] == ("log", {"loss": 0.5}, 3)
    assert calls[2] == ("finish",)
    assert len(calls) == 3
