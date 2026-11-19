"""
utils/trackio_export.py — 把 trackio 紀錄的一個 run 匯出成 trackio 相容的 CSV。

CSV 格式就是 `trackio.import_csv` 接受的格式：
    - 第一列是欄位名稱；`step` 欄必填，`timestamp` 欄選填，其餘欄位都是數值指標
    - 同一個 step 記錄過多次（例如每個 step 的 loss 與每個 epoch 結束時的驗證指標）會合併成同一列
    - 該 step 沒有記錄的指標留空

trackio 匯入時會把欄位名稱縮短成「前 10 個英數字或 / 字元」，所以預設會把指標改成
符合這個規則的短名稱（見 METRIC_ALIASES），匯入後的名稱就和 CSV 完全相同。

使用方式：
    python -m src.utils.trackio_export --run parseq_20261113_180518 -o parseq_train.csv
    python -m src.utils.trackio_export --latest parseq -o parseq_train.csv   # 最新的 PARSeq 訓練
    python -m src.utils.trackio_export --latest yolo -o yolo_finetune.csv     # 最新的 YOLO fine-tune
    python -m src.utils.trackio_export --list                                 # 列出所有 run
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

# 原始指標名稱 → 匯出的欄位名稱（trackio 匯入後不會再被縮短）
METRIC_ALIASES: Dict[str, str] = {
    # PARSeq（src/train/train.py）
    "epoch": "epoch",
    "train/loss": "train/loss",
    "train/lr": "train/lr",
    "train/epoch_loss": "epoch/loss",
    "val/loss": "val/loss",
    "val/acc": "val/acc",
    "val/best_acc": "val/best",
    # YOLO（src/finetune/finetune_yolo.py，Ultralytics 的指標名稱）
    "train/box_loss": "train/box",
    "train/cls_loss": "train/cls",
    "train/dfl_loss": "train/dfl",
    "val/box_loss": "val/box",
    "val/cls_loss": "val/cls",
    "val/dfl_loss": "val/dfl",
    "metrics/precision(B)": "precision",
    "metrics/recall(B)": "recall",
    "metrics/mAP50(B)": "mAP50",
    "metrics/mAP50-95(B)": "mAP50to95",
    "lr/pg0": "lr/pg0",
    "lr/pg1": "lr/pg1",
    "lr/pg2": "lr/pg2",
}


def trackio_column_name(name: str) -> str:
    """trackio.import_csv 對欄位名稱的處理（utils.simplify_column_names，不含重複名稱的編號）。"""
    return re.sub(r"[^a-zA-Z0-9/]", "", name)[:10]


def _connect(db_path: Path) -> sqlite3.Connection:
    if not db_path.is_file():
        raise FileNotFoundError(f"找不到 trackio 資料庫：{db_path}")
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)  # 唯讀，不影響正在訓練的程式


def list_runs(db_path: Path) -> List[dict]:
    """所有 run：名稱、筆數、最大 step、開始 / 結束時間（依開始時間排序）。"""
    with _connect(db_path) as con:
        rows = con.execute(
            "SELECT run_name, COUNT(*), MAX(step), MIN(timestamp), MAX(timestamp) "
            "FROM metrics GROUP BY run_name ORDER BY MIN(timestamp)"
        ).fetchall()
    return [dict(zip(("run", "rows", "max_step", "start", "end"), r)) for r in rows]


def latest_run(db_path: Path, prefix: str) -> str:
    runs = [r for r in list_runs(db_path) if r["run"].startswith(prefix + "_")]
    if not runs:
        raise ValueError(f"{db_path} 中沒有以 {prefix}_ 開頭的 run")
    return runs[-1]["run"]


def run_config(db_path: Path, run: str) -> Optional[dict]:
    """訓練時傳給 trackio.init 的設定（超參數）。"""
    with _connect(db_path) as con:
        row = con.execute("SELECT config FROM configs WHERE run_name = ?", (run,)).fetchone()
    return json.loads(row[0]) if row else None


def export_run(db_path: Path, run: str, out_csv: Path, rename: bool = True) -> int:
    """把一個 run 的指標寫成 CSV，回傳列數。"""
    with _connect(db_path) as con:
        rows = con.execute(
            "SELECT step, timestamp, metrics FROM metrics WHERE run_name = ? ORDER BY step, id", (run,)
        ).fetchall()
    if not rows:
        raise ValueError(f"{db_path} 中沒有 run：{run}")

    merged: Dict[int, dict] = {}
    columns: List[str] = []
    for step, timestamp, metrics in rows:
        row = merged.setdefault(step, {"step": step, "timestamp": timestamp})
        for key, value in json.loads(metrics).items():
            column = METRIC_ALIASES.get(key, key) if rename else key
            if column not in columns:
                columns.append(column)
            row[column] = value

    if rename:
        clashes = {c: trackio_column_name(c) for c in columns if trackio_column_name(c) != c}
        if clashes:
            raise ValueError(f"這些欄位匯入 trackio 後名稱會被改掉，請加進 METRIC_ALIASES：{clashes}")

    # 欄位順序：已知指標依 METRIC_ALIASES 的順序，其餘依第一次出現的順序
    known = list(dict.fromkeys(METRIC_ALIASES.values() if rename else METRIC_ALIASES))
    first_seen = {c: i for i, c in enumerate(columns)}
    columns.sort(key=lambda c: (known.index(c) if c in known else len(known), first_seen[c]))

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "timestamp", *columns], restval="")
        writer.writeheader()
        writer.writerows(merged[s] for s in sorted(merged))
    return len(merged)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=Path("logs/trackio/ALPR.db"), help="trackio 資料庫（預設 logs/trackio/ALPR.db）")
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--run", help="run 名稱，例如 parseq_20261113_180518")
    which.add_argument("--latest", choices=["parseq", "yolo"], help="該類型最新的一個 run")
    which.add_argument("--list", action="store_true", help="列出所有 run")
    ap.add_argument("-o", "--output", type=Path, help="輸出 CSV（預設 <run>.csv）")
    ap.add_argument("--config-output", type=Path, help="另外把這個 run 的設定（超參數）存成 JSON")
    ap.add_argument("--keep-names", action="store_true", help="保留原始指標名稱（trackio 匯入時會被縮短）")
    args = ap.parse_args(argv)

    if args.list:
        for r in list_runs(args.db):
            print(f"{r['run']:28s} rows={r['rows']:<6d} max_step={r['max_step']:<6d} {r['start']} ~ {r['end']}")
        return

    run = args.run or latest_run(args.db, args.latest)
    out = args.output or Path(f"{run}.csv")
    n = export_run(args.db, run, out, rename=not args.keep_names)
    print(f"{run} → {out}（{n} 列）")
    if args.config_output:
        args.config_output.parent.mkdir(parents=True, exist_ok=True)
        args.config_output.write_text(
            json.dumps(run_config(args.db, run), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"設定 → {args.config_output}")


if __name__ == "__main__":
    main()
