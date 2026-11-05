#!/usr/bin/env python3
"""
main.py — CLI 入口點

使用方式：
    # 辨識單張影像
    python main.py --image data/car.jpg

    # 辨識資料夾
    python main.py --folder data/ --output outputs/batch.json

    # 用 ground truth CSV 評估準確率
    python main.py --eval data/taiwan_plate_dataset/test/test_gt.csv

    # 啟動 API Server（host / port 讀 config.yaml 的 api 區塊）
    python main.py --serve

所有設定（權重、裝置、輸出目錄……）都在 config.yaml，CLI 參數只用來臨時覆寫。
"""

from __future__ import annotations

import argparse
import os
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import List, Sequence

import uvicorn

from src.config import DEFAULT_CONFIG_PATH, load_config
from src.models import EvalReport
from src.pipeline import LPRPipeline
from src.utils import logger, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ALPR — YOLO + PARSeq 車牌辨識系統",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image", type=Path, help="辨識單張影像")
    mode.add_argument("--folder", type=Path, help="批次辨識資料夾內所有影像")
    mode.add_argument("--eval", type=Path, metavar="GT_CSV",
                      help="用 ground truth CSV（image_path,plate_text）評估準確率")
    mode.add_argument("--serve", action="store_true", help="啟動 FastAPI REST API Server")

    parser.add_argument(
        "--output", "-o", type=Path,
        help="--folder / --eval 的 JSON 輸出路徑\n"
             "（預設：outputs/batch_<id>.json、outputs/eval_<時間>.json）",
    )
    parser.add_argument("--no-save", action="store_true", help="不儲存 JSON，只輸出到 stdout")
    parser.add_argument("--image-root", type=Path, default=None,
                        help="--eval 時，CSV 中 image_path 的相對基準資料夾（預設自動尋找）")

    # 以下都是覆寫 config.yaml 用，不指定就用設定檔的值
    parser.add_argument("--config", type=Path, default=None, help=f"設定檔路徑（預設 {DEFAULT_CONFIG_PATH}）")
    parser.add_argument("--device", default=None, help="推論裝置 cuda | mps | cpu（同時覆寫 detector / recognizer）")
    parser.add_argument("--log-level", default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--host", default=None, help="API server host")
    parser.add_argument("--port", type=int, default=None, help="API server port")
    parser.add_argument("--workers", type=int, default=None, help="Uvicorn workers 數量")
    return parser.parse_args()


def run_cli() -> None:
    args = parse_args()

    # CLI 覆寫用環境變數傳遞，API Server 的 worker 行程也會讀到同樣的設定
    if args.device:
        os.environ["ALPR_DETECTOR__DEVICE"] = args.device
        os.environ["ALPR_RECOGNIZER__DEVICE"] = args.device
    if args.log_level:
        os.environ["ALPR_LOG_LEVEL"] = args.log_level

    cfg = load_config(args.config)
    setup_logger(cfg.log_level, cfg.log_file)

    if args.serve:
        host = args.host or cfg.api.host
        port = args.port or cfg.api.port
        workers = args.workers or cfg.api.workers
        logger.info(f"LPR API Server 啟動中：http://{host}:{port}")
        logger.info(f"API 文件：http://{host}:{port}/docs")
        uvicorn.run("src.api:app", host=host, port=port, workers=workers, reload=False)
        return

    pipeline = LPRPipeline(cfg)
    save = not args.no_save

    if args.image:
        result = pipeline.predict(args.image, save_json=save)
        if result.error:
            logger.error(result.error)
        elif result.best_plate() is None:
            logger.warning(f"未偵測到車牌：{result.source_file}")
        print(result.model_dump_json(indent=2))

    elif args.folder:
        batch = pipeline.predict_folder(args.folder, save_json=save, batch_output_path=args.output)
        logger.info(f"批次辨識完成，共 {batch.total_images} 張")
        rows = []
        for r in batch.results:
            best = r.best_plate()
            rows.append((r.source_file, best.plate_text if best else "-", f"{r.processing_time_ms:.1f}"))
        _print_table(("檔名", "車牌", "耗時(ms)"), rows)

    elif args.eval:
        from src.evaluation import evaluate

        report = evaluate(pipeline, args.eval, args.image_root)
        _print_eval_report(report)
        if save:
            out_path = args.output or cfg.output_dir / f"eval_{datetime.now():%Y%m%d_%H%M%S}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
            logger.info(f"評估結果已儲存：{out_path}")


# ---------------------------------------------------------------------------
# 結果輸出（stdout）
# ---------------------------------------------------------------------------

def _print_eval_report(report: EvalReport) -> None:
    print(f"樣本數      {report.total}")
    print(f"完全正確    {report.correct}（{report.accuracy:.2%}）")
    print(f"字元準確率  {report.char_accuracy:.2%}")
    print(f"未偵測到    {report.no_plate}")

    mismatches = report.mismatches()
    if mismatches:
        print(f"\n錯誤樣本（{len(mismatches)}）")
        _print_table(
            ("檔名", "GT", "預測"),
            [(s.source_file, s.gt_text, s.pred_text or "<未偵測到>") for s in mismatches],
        )


def _display_width(text: str) -> int:
    """終端機顯示寬度（中文等全形字元算 2 格）。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def _print_table(headers: Sequence[str], rows: List[Sequence[str]]) -> None:
    """以純文字表格輸出（支援中英文混排對齊）。"""
    table = [tuple(headers), *rows]
    widths = [max(_display_width(row[i]) for row in table) for i in range(len(headers))]
    for i, row in enumerate(table):
        print("  ".join(cell + " " * (w - _display_width(cell)) for cell, w in zip(row, widths)).rstrip())
        if i == 0:
            print("  ".join("-" * w for w in widths))


if __name__ == "__main__":
    run_cli()
