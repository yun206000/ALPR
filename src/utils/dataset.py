"""
utils/dataset.py — 讀取車牌 ground truth CSV（訓練與評估共用）。

CSV 欄位：image_path, plate_text
image_path 可以是絕對路徑，或以下任一種相對路徑（依序尋找第一個存在的檔案）：
    1. 相對於 image_root（有指定時）
    2. 相對於 CSV 所在資料夾            例：valid/valid_gt.csv + xxx.jpg（同層）
    3. 相對於 CSV 所在資料夾的上一層    例：test/test_gt.csv + test/images/xxx.jpg
    4. CSV 所在資料夾底下的 images/     例：train/train_gt.csv + xxx.jpg → train/images/xxx.jpg
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional, Tuple


def _resolve_image_path(raw: str, csv_dir: Path, image_root: Optional[Path]) -> Optional[Path]:
    path = Path(raw)
    if path.is_absolute():
        return path if path.exists() else None

    candidates = []
    if image_root is not None:
        candidates.append(image_root / path)
    candidates += [csv_dir / path, csv_dir.parent / path, csv_dir / "images" / path]

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_plate_csv(
    csv_path: str | Path,
    image_root: Optional[str | Path] = None,
) -> List[Tuple[Path, str]]:
    """讀取 ground truth CSV，回傳 [(影像絕對路徑, 車牌文字), ...]。

    找不到任何一張影像時直接拋出 FileNotFoundError，避免默默少算樣本。
    """
    csv_path = Path(csv_path)
    csv_dir = csv_path.resolve().parent
    root = Path(image_root).resolve() if image_root is not None else None

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing_cols = {"image_path", "plate_text"} - set(reader.fieldnames or [])
        if missing_cols:
            raise ValueError(f"{csv_path} 缺少欄位：{sorted(missing_cols)}")
        rows = list(reader)

    samples: List[Tuple[Path, str]] = []
    missing: List[str] = []
    for row in rows:
        resolved = _resolve_image_path(row["image_path"], csv_dir, root)
        if resolved is None:
            missing.append(row["image_path"])
            continue
        samples.append((resolved.resolve(), str(row["plate_text"]).strip()))

    if missing:
        raise FileNotFoundError(
            f"{csv_path} 中有 {len(missing)} 張影像找不到（例如：{missing[0]}）。"
            "請確認路徑，或用 image_root 指定影像所在資料夾。"
        )
    return samples
