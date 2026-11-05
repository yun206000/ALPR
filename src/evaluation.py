"""
evaluation.py — 用 ground truth CSV 評估整個 pipeline（偵測 + 辨識）的準確率。

使用方式：
    python main.py --eval data/taiwan_plate_dataset/test/test_gt.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from tqdm import tqdm

from .models import EvalReport, EvalSample
from .pipeline import LPRPipeline
from .utils import load_plate_csv, logger


def edit_distance(a: str, b: str) -> int:
    """Levenshtein 編輯距離（插入 / 刪除 / 替換各算 1）。"""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def evaluate(
    pipeline: LPRPipeline,
    gt_csv: str | Path,
    image_root: Optional[str | Path] = None,
) -> EvalReport:
    """對 GT CSV 中的每張影像跑 pipeline，與 plate_text 比對。

    每張影像取可信度最高的車牌當作預測；比對為大小寫、符號都要完全相同。
    """
    samples = load_plate_csv(gt_csv, image_root)
    logger.info(f"評估 {len(samples)} 張影像：{gt_csv}")

    results = []
    for image_path, gt_text in tqdm(samples, desc="評估進度", unit="img"):
        output = pipeline.predict(image_path)
        best = output.best_plate()
        pred_text = best.plate_text if best else None
        results.append(
            EvalSample(
                source_file=image_path.name,
                gt_text=gt_text,
                pred_text=pred_text,
                correct=pred_text == gt_text,
                edit_distance=edit_distance(gt_text, pred_text or ""),
            )
        )

    total = len(results)
    correct = sum(r.correct for r in results)
    total_chars = sum(len(r.gt_text) for r in results)
    total_edits = sum(r.edit_distance for r in results)

    report = EvalReport(
        gt_csv=str(gt_csv),
        total=total,
        correct=correct,
        accuracy=correct / total if total else 0.0,
        char_accuracy=max(0.0, 1 - total_edits / total_chars) if total_chars else 0.0,
        no_plate=sum(r.pred_text is None for r in results),
        samples=results,
    )
    logger.info(f"評估完成 | 準確率={report.accuracy:.2%} | 字元準確率={report.char_accuracy:.2%}")
    return report
