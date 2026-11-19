"""src/finetune/finetune_yolo.py：資料準備、trackio callback，以及完整流程（假的 Ultralytics YOLO）。"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.finetune import finetune_yolo as fy

from conftest import write_jpg


def _split(root: Path, names, missing_label=()):
    for name in names:
        write_jpg(root / "images" / f"{name}.jpg", 32, 32)
        if name not in missing_label:
            (root / "labels").mkdir(parents=True, exist_ok=True)
            (root / "labels" / f"{name}.txt").write_text("0 0.5 0.5 0.2 0.1\n")
    (root / "images" / "notes.txt").write_text("不是影像")
    return root


def test_list_labeled_images(tmp_path):
    split = _split(tmp_path / "train", ["a", "b", "c"], missing_label={"b"})
    assert [img.name for img, _ in fy.list_labeled_images(split)] == ["a.jpg", "c.jpg"]
    with pytest.raises(AssertionError, match="找不到"):
        fy.list_labeled_images(tmp_path / "nope")


@pytest.mark.parametrize("symlink_ok", [True, False])
def test_link_pairs(tmp_path, monkeypatch, symlink_ok):
    pairs = fy.list_labeled_images(_split(tmp_path / "train", ["a"]))
    if not symlink_ok:  # 不支援 symlink 的檔案系統 → 改用複製
        monkeypatch.setattr(Path, "symlink_to", lambda *a, **k: (_ for _ in ()).throw(OSError()))
    fy.link_pairs(pairs, tmp_path / "out" / "images", tmp_path / "out" / "labels")
    fy.link_pairs(pairs, tmp_path / "out" / "images", tmp_path / "out" / "labels")  # 已存在 → 略過
    img = tmp_path / "out" / "images" / "a.jpg"
    assert img.is_symlink() is symlink_ok and img.read_bytes() == pairs[0][0].read_bytes()
    assert (tmp_path / "out" / "labels" / "a.txt").exists()


def test_prepare_dataset_with_val_dir(tmp_path):
    train = _split(tmp_path / "train", ["a", "b"])
    val = _split(tmp_path / "valid", ["c"])
    assert fy.prepare_dataset(train, val, 0.1, tmp_path / "work", 42) == (train / "images", val / "images")
    assert not (tmp_path / "work").exists()  # 有 val 目錄時不建立任何東西


def test_prepare_dataset_auto_split(tmp_path):
    train = _split(tmp_path / "train", [f"img{i}" for i in range(10)])
    train_imgs, val_imgs = fy.prepare_dataset(train, None, 0.2, tmp_path / "work", 42)
    n_train, n_val = len(list(train_imgs.iterdir())), len(list(val_imgs.iterdir()))
    assert (n_train, n_val) == (8, 2)
    again = fy.prepare_dataset(train, None, 0.2, tmp_path / "work2", 42)[1]
    assert sorted(p.name for p in again.iterdir()) == sorted(p.name for p in val_imgs.iterdir())  # seed 固定 → 切法相同


def test_write_data_yaml(tmp_path):
    import yaml

    path = fy.write_data_yaml(tmp_path, tmp_path / "t", tmp_path / "v")
    assert yaml.safe_load(path.read_text()) == {"train": str(tmp_path / "t"), "val": str(tmp_path / "v"), "names": {0: "plate"}}


def test_tracking_callbacks():
    callbacks, logged = {}, []
    model = SimpleNamespace(add_callback=lambda name, fn: callbacks.__setitem__(name, fn))
    tracker = SimpleNamespace(log=lambda metrics, step: logged.append((metrics, step)))
    fy.add_tracking_callbacks(model, tracker)

    trainer = SimpleNamespace(
        epoch=2, tloss=[0.1, 0.2], lr={"lr/pg0": 0.001}, metrics={"metrics/mAP50(B)": 0.9},
        label_loss_items=lambda loss, prefix: {f"{prefix}/box_loss": loss[0]},
    )
    callbacks["on_train_epoch_end"](trainer)
    callbacks["on_fit_epoch_end"](trainer)
    assert logged == [({"train/box_loss": 0.1}, 3), ({"lr/pg0": 0.001}, 3), ({"metrics/mAP50(B)": 0.9}, 3)]


class FakeYOLO:
    """模擬 Ultralytics：train() 在 project/name/weights/ 寫出 best.pt、last.pt。"""

    instances = []

    def __init__(self, weights):
        self.weights = weights
        self.callbacks = {}
        self.train_kwargs = None
        FakeYOLO.instances.append(self)

    def add_callback(self, name, fn):
        self.callbacks[name] = fn

    def train(self, **kwargs):
        self.train_kwargs = kwargs
        weights = Path(kwargs["project"]) / kwargs["name"] / "weights"
        weights.mkdir(parents=True)
        (weights / "best.pt").write_text("best")
        (weights / "last.pt").write_text("last")
        (weights.parent / "results.csv").write_text("epoch\n1\n")

    def val(self, **kwargs):
        return SimpleNamespace(box=SimpleNamespace(map50=0.99, map=0.78))


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "ds"
    _split(root / "train", ["a", "b", "c", "d"])
    return root


def _run_main(monkeypatch, *args):
    FakeYOLO.instances.clear()
    monkeypatch.setattr(fy, "YOLO", FakeYOLO)
    monkeypatch.setattr(sys, "argv", ["finetune_yolo.py", *map(str, args)])
    fy.main()
    return FakeYOLO.instances[0]


def test_main_finetune_from_weights(tmp_path, app_config, monkeypatch, dataset):
    (dataset / "valid").mkdir()
    _split(dataset / "valid", ["e"])
    out_dir = tmp_path / "yolo_out"
    out_dir.mkdir()
    (out_dir / "stale.pt").write_text("old")

    model = _run_main(monkeypatch, "--data-root", dataset, "--output-dir", out_dir, "--epochs", "1",
                      "--finetune-from", "some/pretrained.pt", "--device", "cpu", "--freeze", "10")

    assert model.weights == "some/pretrained.pt"
    kw = model.train_kwargs
    assert "pretrained" not in kw  # 不傳 pretrained：沿用 YOLO(weights) 載入的權重
    assert (kw["optimizer"], kw["epochs"], kw["device"], kw["freeze"], kw["plots"]) == (
        app_config.finetune_yolo.optimizer, 1, "cpu", 10, False)
    assert set(model.callbacks) == {"on_train_epoch_end", "on_fit_epoch_end"}
    assert sorted(p.name for p in out_dir.iterdir()) == ["best.pt", "finetune_yolo.log", "last.pt", "results.csv"]
    log = (out_dir / "finetune_yolo.log").read_text(encoding="utf-8")
    assert "mAP50=0.9900" in log and "請把 config.yaml 的 detector.weights_path 改成" in log


def test_main_auto_split_and_model_fallback(tmp_path, app_config, monkeypatch, dataset):
    out_dir = tmp_path / "yolo_out"
    monkeypatch.setenv("ALPR_DETECTOR__WEIGHTS_PATH", str(out_dir / "best.pt"))
    model = _run_main(monkeypatch, "--data-root", dataset, "--output-dir", out_dir,
                      "--finetune-from", "null", "--model", "yolov8n.pt", "--freeze", "0")
    assert model.weights == "yolov8n.pt"
    assert "freeze" not in model.train_kwargs and "device" not in model.train_kwargs
    data_yaml = Path(model.train_kwargs["data"]).name
    assert data_yaml == "data.yaml"
    assert "API Server 重新啟動後才會載入新權重" in (out_dir / "finetune_yolo.log").read_text(encoding="utf-8")
