"""
api/server.py

FastAPI REST API 服務。
提供影像上傳辨識、批次辨識、結果查詢等端點。
"""

from __future__ import annotations

import io
import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..models import BatchOutput, PredictionOutput
from ..pipeline import LPRPipeline
from ..utils import logger

# ---------------------------------------------------------------------------
# 建立 FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LPR Pipeline API",
    description="RF-DETR + PARSeq 車牌辨識服務",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# 全域 pipeline 實例（lazy init on startup）
# ---------------------------------------------------------------------------

_pipeline: Optional[LPRPipeline] = None
# 簡易記憶體結果快取 {job_id: PredictionOutput}
_result_store: Dict[str, PredictionOutput] = {}


@app.on_event("startup")
async def startup_event() -> None:
    global _pipeline
    logger.info("API server 啟動，初始化 LPR Pipeline...")
    _pipeline = LPRPipeline()
    _pipeline.warmup()
    logger.info("API server 就緒")


def _get_pipeline() -> LPRPipeline:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline 尚未初始化")
    return _pipeline


# ---------------------------------------------------------------------------
# 端點
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health_check():
    """健康檢查端點"""
    return {"status": "ok", "pipeline_ready": _pipeline is not None}


@app.post(
    "/predict",
    response_model=PredictionOutput,
    tags=["Prediction"],
    summary="上傳單張影像進行車牌辨識",
)
async def predict_single(
    file: UploadFile = File(..., description="影像檔案（jpg/png/bmp）"),
):
    """
    上傳一張影像，回傳結構化辨識結果（JSON）。

    - **file**: 影像檔案

    回傳欄位：
    - `job_id`: 唯一辨識 ID
    - `source_file`: 原始檔名
    - `plates`: 辨識到的車牌列表（含 plate_text、confidence、bbox）
    - `processing_time_ms`: 推論耗時
    """
    pipeline = _get_pipeline()

    # 讀取上傳影像
    contents = await file.read()
    img_array = _decode_image_bytes(contents, file.filename)

    result = pipeline.predict(
        source=img_array,
        source_name=file.filename or "upload.jpg",
        save_json=True,
    )

    # 存入快取
    _result_store[result.job_id] = result
    return result


@app.post(
    "/predict/batch",
    response_model=BatchOutput,
    tags=["Prediction"],
    summary="批次上傳多張影像",
)
async def predict_batch(
    files: List[UploadFile] = File(..., description="多張影像檔案"),
):
    """
    批次上傳影像，回傳批次辨識結果。
    """
    pipeline = _get_pipeline()
    results: List[PredictionOutput] = []

    for file in files:
        contents = await file.read()
        try:
            img_array = _decode_image_bytes(contents, file.filename)
            result = pipeline.predict(
                source=img_array,
                source_name=file.filename or "upload.jpg",
                save_json=True,
            )
        except Exception as e:
            result = PredictionOutput(
                source_file=file.filename or "unknown",
                error=str(e),
            )

        _result_store[result.job_id] = result
        results.append(result)

    batch = BatchOutput(
        total_images=len(files),
        results=results,
    )
    return batch


@app.get(
    "/results/{job_id}",
    response_model=PredictionOutput,
    tags=["Results"],
    summary="查詢辨識結果",
)
async def get_result(job_id: str):
    """
    依 job_id 查詢先前的辨識結果。
    結果存於記憶體，重啟後清除。
    """
    result = _result_store.get(job_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"找不到 job_id={job_id} 的結果",
        )
    return result


@app.get(
    "/results",
    response_model=List[PredictionOutput],
    tags=["Results"],
    summary="列出所有辨識結果",
)
async def list_results(limit: int = 50, offset: int = 0):
    """列出所有快取中的辨識結果"""
    all_results = list(_result_store.values())
    return all_results[offset: offset + limit]


# ---------------------------------------------------------------------------
# 工具函式
# ---------------------------------------------------------------------------

def _decode_image_bytes(data: bytes, filename: Optional[str] = None) -> np.ndarray:
    """將上傳的 bytes 解碼為 BGR ndarray"""
    import cv2

    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        name = filename or "file"
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"無法解碼影像：{name}",
        )
    return img
