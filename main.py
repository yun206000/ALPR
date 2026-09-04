#!/usr/bin/env python3
"""
main.py — CLI 入口點

使用方式：
    # 辨識單張影像
    python main.py --image data/samples/car.jpg

    # 辨識資料夾
    python main.py --folder data/samples/ --output outputs/batch.json

    # 啟動 API Server
    python main.py --serve --host 0.0.0.0 --port 8000

    # 顯示結果 JSON（不儲存）
    python main.py --image car.jpg --no-save
"""

from __future__ import annotations

import os
import argparse
import json
import sys
from pathlib import Path
import uvicorn
from src.pipeline import LPRPipeline
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.config import get_config
from src.utils import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LPR Pipeline — YOLOv8 + PARSeq 車牌辨識系統",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image", type=Path, help="辨識單張影像")
    mode.add_argument("--folder", type=Path, help="批次辨識資料夾內所有影像")
    mode.add_argument("--serve", action="store_true", help="啟動 FastAPI REST API Server")

    parser.add_argument(
        "--output", "-o", type=Path,
        help="批次結果輸出路徑（預設：outputs/batch_<id>.json）",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="不儲存 JSON 結果（僅輸出到 stdout）",
    )
    parser.add_argument("--host", default="0.0.0.0", help="API server host（預設：0.0.0.0）")
    parser.add_argument("--port", type=int, default=8000, help="API server port（預設：8000）")
    parser.add_argument("--workers", type=int, default=1, help="Uvicorn workers 數量")
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--device", default=None,
        help="推論裝置：cuda | mps | cpu（預設由環境決定）",
    )
    return parser.parse_args()


def run_cli() -> None:
    args = parse_args()

    # 動態設定環境變數（傳遞給 config）
    if args.device:
        os.environ["DETECTOR_DEVICE"] = args.device
        os.environ["RECOGNIZER_DEVICE"] = args.device

    cfg = get_config()
    cfg.log_level = args.log_level
    setup_logger(args.log_level)

    # API Server mode
    if args.serve:
        print(f"\nLPR API Server 啟動中 → http://{args.host}:{args.port}")
        print(f"    Docs: http://{args.host}:{args.port}/docs\n")
        uvicorn.run(
            "src.api:app",
            host=args.host,
            port=args.port,
            workers=args.workers,
            reload=False,
        )
        return

    console = Console()
    pipeline = LPRPipeline(cfg)
    save = not args.no_save
    
    # Single image mode
    if args.image:
        result = pipeline.predict(args.image, save_json=save)
        _print_result(result, console)
        print(result.model_dump_json(indent=2))
        return

    # Batch mode
    if args.folder:
        batch = pipeline.predict_folder(
            args.folder,
            save_json=save,
            batch_output_path=args.output,
        )
        console.print(f"\n[bold green]✓ 批次辨識完成[/bold green]")
        console.print(f"  總計：{batch.total_images} 張")
        # console.print(f"  成功：{batch.success_count} | 失敗：{batch.fail_count}")

        table = Table(title="辨識摘要", show_lines=True)
        table.add_column("檔名", style="cyan")
        table.add_column("車牌", style="green")
        # table.add_column("可信度", justify="right")
        table.add_column("耗時(ms)", justify="right")

        for r in batch.results:
            best = r.best_plate()
            table.add_row(
                r.source_file,
                best.plate_text if best else "—",
                # f"{best.confidence:.3f}" if best else "—",
                f"{r.processing_time_ms:.1f}",
            )
        console.print(table)


def _print_result(result, console) -> None:
    best = result.best_plate()
    if best:
        console.print(Panel(
            f"辨識結果：[bold green]{best.plate_text}[/bold green]",
            # f"可信度：{best.confidence:.3f}  "
            # f"(偵測={best.detector_score:.3f} / 辨識={best.recognizer_score:.3f})",
            title=f"{result.source_file}",
        ))
    else:
        console.print(f"[yellow]⚠  未偵測到車牌：{result.source_file}[/yellow]")


if __name__ == "__main__":
    run_cli()
