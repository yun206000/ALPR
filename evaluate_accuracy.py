from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_ground_truth(csv_path: Path) -> List[Tuple[str, str]]:
    """讀取 test_gt.csv，回傳 (檔名, 文字) 清單。"""
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    items: List[Tuple[str, str]] = []
    for row in rows:
        image_path = row.get("image_path", "")
        plate_text = row.get("plate_text", "")
        if image_path:
            items.append((Path(image_path).name, plate_text))

    return items


def load_predictions(json_path: Path) -> Dict[str, str]:
    """讀取 outputs/results.json，回傳 {檔名: 文字} 的對照表。"""
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    predictions: Dict[str, str] = {}
    for item in data.get("results", []):
        if item.get("error"):
            continue

        source_file = item.get("source_file", "")
        plates = item.get("plates", [])
        if not source_file or not plates:
            continue

        pred_text = plates[0].get("plate_text", "")
        predictions[Path(source_file).name] = pred_text

    return predictions


def evaluate_accuracy(
    gt_csv: Path,
    results_json: Path,
) -> Tuple[int, int, float, List[Tuple[str, str, str]]]:
    """計算準確率，回傳 (正確數, 總數, 準確率, 錯誤清單)。"""
    gt_items = load_ground_truth(gt_csv)
    predictions = load_predictions(results_json)

    correct = 0
    mismatches: List[Tuple[str, str, str]] = []

    for filename, gt_text in gt_items:
        pred_text = predictions.get(filename)
        if pred_text is None:
            mismatches.append((filename, gt_text, "<missing>"))
            continue

        if pred_text == gt_text:
            correct += 1
        else:
            mismatches.append((filename, gt_text, pred_text))

    total = len(gt_items)
    accuracy = correct / total if total else 0.0
    return correct, total, accuracy, mismatches


def main() -> None:
    parser = argparse.ArgumentParser(description="計算 OCR 準確率")
    parser.add_argument(
        "--gt",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "datasets" / "test_gt.csv",
        help="ground truth CSV 路徑",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "results.json",
        help="推論結果 JSON 路徑",
    )
    args = parser.parse_args()

    correct, total, accuracy, mismatches = evaluate_accuracy(args.gt, args.results)

    print(f"總樣本數: {total}")
    print(f"正確數: {correct}")
    print(f"準確率: {accuracy:.2%}")

    if mismatches:
        print("\n錯誤樣本：")
        for filename, gt_text, pred_text in mismatches:
            print(f"- {filename}: GT={gt_text}, Pred={pred_text}")


if __name__ == "__main__":
    main()
