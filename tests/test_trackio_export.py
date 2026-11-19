"""src/utils/trackio_export.py：trackio run → trackio 相容的 CSV（含實際用 trackio.import_csv 匯入的往返測試）。"""

import csv
import json
import os
import sqlite3
import subprocess
import sys

import pytest

from src.utils import trackio_export as te

from conftest import REPO_ROOT


def _make_db(path, runs):
    """建立和 trackio 相同結構的 metrics / configs 資料表。"""
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE metrics (id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, timestamp TEXT, "
                "run_name TEXT, step INTEGER, metrics TEXT)")
    con.execute("CREATE TABLE configs (id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, run_name TEXT, "
                "config TEXT, created_at TEXT)")
    for run, (rows, config) in runs.items():
        for step, ts, metrics in rows:
            con.execute("INSERT INTO metrics (run_id, timestamp, run_name, step, metrics) VALUES (?, ?, ?, ?, ?)",
                        (run, ts, run, step, json.dumps(metrics)))
        if config is not None:
            con.execute("INSERT INTO configs (run_id, run_name, config, created_at) VALUES (?, ?, ?, ?)",
                        (run, run, json.dumps(config), "t"))
    con.commit()
    con.close()
    return path


PARSEQ_ROWS = [
    (0, "2026-01-01T00:00:00+00:00", {"val/loss": 0.5, "val/acc": 0.9, "epoch": 0.0}),
    (1, "2026-01-01T00:00:01+00:00", {"train/loss": 0.4, "train/lr": 5e-5}),
    (2, "2026-01-01T00:00:02+00:00", {"train/loss": 0.3, "train/lr": 4e-5}),
    (2, "2026-01-01T00:00:03+00:00", {"train/epoch_loss": 0.35, "val/loss": 0.1, "val/acc": 1.0, "val/best_acc": 1.0,
                                      "epoch": 1.0}),
]


@pytest.fixture
def db(tmp_path):
    return _make_db(tmp_path / "ALPR.db", {
        "parseq_20260101_000000": (PARSEQ_ROWS, {"lr": 5e-5, "epochs": 1}),
        "yolo_20260102_000000": ([(1, "2026-01-02T00:00:00+00:00", {"metrics/mAP50-95(B)": 0.7, "lr/pg0": 1e-3})], None),
        "parseq_20260103_000000": ([(0, "2026-01-03T00:00:00+00:00", {"custom_metric": 1})], None),
    })


def _read(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_export_merges_steps_and_renames(db, tmp_path):
    out = tmp_path / "out" / "parseq.csv"
    assert te.export_run(db, "parseq_20260101_000000", out) == 3
    rows = _read(out)
    assert list(rows[0]) == ["step", "timestamp", "epoch", "train/loss", "train/lr", "epoch/loss", "val/loss", "val/acc", "val/best"]
    assert rows[0]["val/acc"] == "0.9" and rows[0]["train/loss"] == ""
    # 同一個 step 的兩次紀錄合併成一列（時間取第一次）
    assert rows[2]["train/loss"] == "0.3" and rows[2]["val/best"] == "1.0"
    assert rows[2]["timestamp"] == "2026-01-01T00:00:02+00:00"
    assert all(te.trackio_column_name(c) == c for c in rows[0])


def test_export_keep_names_and_unknown_metric(db, tmp_path):
    out = tmp_path / "raw.csv"
    te.export_run(db, "parseq_20260101_000000", out, rename=False)
    assert "train/epoch_loss" in _read(out)[0]
    with pytest.raises(ValueError, match="custommetr"):  # 匯入後會被縮短的名稱 → 要求補上別名
        te.export_run(db, "parseq_20260103_000000", tmp_path / "x.csv")
    te.export_run(db, "parseq_20260103_000000", tmp_path / "x.csv", rename=False)
    assert list(_read(tmp_path / "x.csv")[0]) == ["step", "timestamp", "custom_metric"]


def test_list_latest_config_and_errors(db, tmp_path):
    assert [r["run"] for r in te.list_runs(db)] == [
        "parseq_20260101_000000", "yolo_20260102_000000", "parseq_20260103_000000"]
    assert te.latest_run(db, "parseq") == "parseq_20260103_000000"
    assert te.run_config(db, "parseq_20260101_000000") == {"lr": 5e-5, "epochs": 1}
    assert te.run_config(db, "yolo_20260102_000000") is None
    with pytest.raises(ValueError, match="沒有以 nope_ 開頭"):
        te.latest_run(db, "nope")
    with pytest.raises(ValueError, match="沒有 run"):
        te.export_run(db, "missing", tmp_path / "m.csv")
    with pytest.raises(FileNotFoundError):
        te.list_runs(tmp_path / "none.db")


def test_cli(db, tmp_path, capsys, monkeypatch):
    te.main(["--db", str(db), "--list"])
    assert "yolo_20260102_000000" in capsys.readouterr().out

    out, cfg = tmp_path / "y.csv", tmp_path / "cfg" / "c.json"
    te.main(["--db", str(db), "--latest", "yolo", "-o", str(out), "--config-output", str(cfg)])
    assert list(_read(out)[0])[2:] == ["mAP50to95", "lr/pg0"]
    assert json.loads(cfg.read_text(encoding="utf-8")) is None

    monkeypatch.chdir(tmp_path)
    te.main(["--db", str(db), "--run", "parseq_20260101_000000", "--keep-names"])
    assert (tmp_path / "parseq_20260101_000000.csv").exists()


def test_trackio_import_round_trip(db, tmp_path):
    """用 trackio.import_csv 匯入匯出的 CSV：欄位名稱不變、每個數值與 step 都相同。"""
    out = tmp_path / "parseq.csv"
    te.export_run(db, "parseq_20260101_000000", out)
    trackio_dir = tmp_path / "trackio"
    code = (
        "import json, sqlite3, sys, trackio\n"
        "trackio.import_csv(sys.argv[1], project='rt', name='r')\n"
        "con = sqlite3.connect(sys.argv[2])\n"
        "print(json.dumps([[s, t, json.loads(m)] for s, t, m in con.execute("
        "'select step, timestamp, metrics from metrics order by step')]))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(out), str(trackio_dir / "rt.db")],
        capture_output=True, text=True, cwd=REPO_ROOT, env={**os.environ, "TRACKIO_DIR": str(trackio_dir)}, check=True,
    )
    imported = json.loads(result.stdout.strip().splitlines()[-1])
    expected = [
        [int(r["step"]), r["timestamp"], {k: float(v) for k, v in r.items() if k not in ("step", "timestamp") and v}]
        for r in _read(out)
    ]
    assert imported == expected
