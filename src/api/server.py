"""
api/server.py — FastAPI REST API 服務。

除了 /health 與 API 文件之外，所有端點都要帶 token：
    Authorization: Bearer <token>
token 設定在 config.yaml 的 api.tokens（或環境變數 ALPR_API__TOKENS）。

每個辨識請求在 output_dir 存「一份」JSON：
    POST /predict        → <檔名>_<job_id 前 8 碼>.json（PredictionOutput）
    POST /predict/batch  → batch_<batch_id 前 8 碼>.json（BatchOutput，整批一份）

所有錯誤回應都是 {"detail": "<訊息>"}。
"""

from __future__ import annotations

import secrets
import threading
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..config import get_config
from ..models import BatchOutput, ErrorResponse, PredictionOutput
from ..pipeline import LPRPipeline
from ..utils import logger

# ---------------------------------------------------------------------------
# 全域狀態
# ---------------------------------------------------------------------------

_pipeline: Optional[LPRPipeline] = None
# 簡易記憶體結果快取 {job_id: PredictionOutput}，server 重啟後清除
_result_store: Dict[str, PredictionOutput] = {}
# 模型推論不是 thread-safe（Ultralytics predictor），同一時間只允許一個推論
_inference_lock = threading.Lock()
# {token: 用戶端名稱}
_tokens: Dict[str, str] = {}


def _load_tokens() -> Dict[str, str]:
    tokens = {secret.get_secret_value(): name for name, secret in get_config().api.tokens.items()}
    if not tokens:
        raise RuntimeError(
            "沒有設定任何 API token，拒絕啟動。請設定 config.yaml 的 api.tokens，"
            """或環境變數 ALPR_API__TOKENS='{"web": "<token>"}'"""
        )
    if any(len(t) < 16 for t in tokens):
        raise RuntimeError("API token 至少要 16 個字元")
    return tokens


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _pipeline, _tokens
    _tokens = _load_tokens()
    logger.info(f"API server 啟動，允許的用戶端：{', '.join(sorted(_tokens.values()))}")
    _pipeline = LPRPipeline()
    _pipeline.warmup()
    logger.info("API server 就緒")
    yield


# ---------------------------------------------------------------------------
# 建立 FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    lifespan=lifespan,
    title="ALPR API",
    description=(
        "YOLO + PARSeq 車牌辨識服務。\n\n"
        "除了 `/health` 之外的端點都需要 token：`Authorization: Bearer <token>`"
        "（在本頁右上角的 **Authorize** 輸入）。所有錯誤回應都是 `{\"detail\": \"<訊息>\"}`。"
    ),
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


def _openapi_with_file_pickers() -> dict:
    """
    FastAPI 用 OpenAPI 3.1 的 contentMediaType 描述上傳檔案，但 Swagger UI 對「檔案陣列」
    （/predict/batch 的 files）只認 format: binary，否則 /docs 只會顯示文字框、不能選檔案。
    """
    if app.openapi_schema is None:
        from fastapi.openapi.utils import get_openapi

        schema = get_openapi(title=app.title, version=app.version, description=app.description, routes=app.routes)

        def patch(node):
            if isinstance(node, dict):
                if node.get("contentMediaType") == "application/octet-stream":
                    node["format"] = "binary"
                for value in node.values():
                    patch(value)
            elif isinstance(node, list):
                for value in node:
                    patch(value)

        patch(schema)
        app.openapi_schema = schema
    return app.openapi_schema


app.openapi = _openapi_with_file_pickers


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    """把 FastAPI 預設的錯誤列表轉成單一字串，讓所有錯誤回應格式一致。"""
    messages = []
    for err in exc.errors():
        loc = ".".join(str(x) for x in err.get("loc", ()) if x not in ("body", "query", "path"))
        messages.append(f"{loc}: {err.get('msg')}" if loc else str(err.get("msg")))
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content={"detail": "; ".join(messages)})


# ---------------------------------------------------------------------------
# 驗證
# ---------------------------------------------------------------------------

_bearer = HTTPBearer(auto_error=False, description="config.yaml 的 api.tokens 其中一個 token")


def require_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer)) -> str:
    """驗證 Bearer token，回傳用戶端名稱。"""
    if credentials is not None:
        given = credentials.credentials.encode()
        for token, client in _tokens.items():
            if secrets.compare_digest(given, token.encode()):
                return client
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="缺少或無效的 API token",
        headers={"WWW-Authenticate": "Bearer"},
    )


# 需要 token 的端點共用的錯誤回應說明（顯示在 /docs）
_ERRORS = {
    401: {"model": ErrorResponse, "description": "缺少或無效的 token"},
    503: {"model": ErrorResponse, "description": "模型尚未載入完成"},
}


# ---------------------------------------------------------------------------
# 工具函式
# ---------------------------------------------------------------------------

def _get_pipeline() -> LPRPipeline:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline 尚未初始化")
    return _pipeline


async def _predict(pipeline: LPRPipeline, img: np.ndarray, name: str, save_json: bool) -> PredictionOutput:
    """在 worker thread 執行推論，避免卡住事件迴圈（推論期間 /health 等請求仍可回應）。"""

    def run() -> PredictionOutput:
        with _inference_lock:
            return pipeline.predict(source=img, source_name=name, save_json=save_json)

    return await run_in_threadpool(run)


def _decode_image_bytes(data: bytes, filename: Optional[str] = None) -> np.ndarray:
    """將上傳的 bytes 解碼為 BGR ndarray"""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
    if img is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"無法解碼影像：{filename or 'file'}")
    return img


# ---------------------------------------------------------------------------
# 端點
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"], summary="健康檢查（不需要 token）")
async def health_check():
    return {"status": "ok", "pipeline_ready": _pipeline is not None}


@app.post(
    "/predict",
    response_model=PredictionOutput,
    responses={400: {"model": ErrorResponse, "description": "檔案無法解碼成影像"}, **_ERRORS},
    tags=["Prediction"],
    summary="上傳單張影像進行車牌辨識",
)
async def predict_single(
    file: UploadFile = File(..., description="影像檔案（jpg / png / bmp ...）"),
    client: str = Depends(require_token),
):
    """
    以 `multipart/form-data` 上傳一張影像（欄位名稱 `file`），回傳辨識結果。

    結果會存成一份 JSON 到 `output_dir`，之後也可以用 `job_id` 查詢。
    """
    pipeline = _get_pipeline()
    img = _decode_image_bytes(await file.read(), file.filename)
    result = await _predict(pipeline, img, file.filename or "upload.jpg", save_json=True)
    logger.info(f"[{client}] /predict {result.source_file} → {len(result.plates)} 個車牌")

    _result_store[result.job_id] = result
    return result


@app.post(
    "/predict/batch",
    response_model=BatchOutput,
    responses=_ERRORS,
    tags=["Prediction"],
    summary="批次上傳多張影像",
)
async def predict_batch(
    files: List[UploadFile] = File(..., description="多張影像檔案"),
    client: str = Depends(require_token),
):
    """
    以 `multipart/form-data` 上傳多張影像（欄位名稱都是 `files`），回傳整批結果。

    - `results` 的順序與上傳順序相同。
    - 某張無法解碼時不會讓整批失敗，該張的 `plates` 為空、`error` 有錯誤訊息。
    - 整批結果存成**一份** JSON 到 `output_dir`。每張的 `job_id` 都可以用 `/results/{job_id}` 查詢。
    """
    pipeline = _get_pipeline()
    results: List[PredictionOutput] = []

    for file in files:
        name = file.filename or "upload.jpg"
        try:
            img = _decode_image_bytes(await file.read(), name)
            result = await _predict(pipeline, img, name, save_json=False)
        except HTTPException as e:
            result = PredictionOutput(source_file=name, error=str(e.detail))
        except Exception as e:
            logger.exception(f"辨識失敗：{name}")
            result = PredictionOutput(source_file=name, error=str(e))

        _result_store[result.job_id] = result
        results.append(result)

    batch = BatchOutput(total_images=len(files), results=results)
    await run_in_threadpool(pipeline.save_batch, batch)
    logger.info(f"[{client}] /predict/batch {len(files)} 張 → batch_id={batch.batch_id}")
    return batch


@app.get(
    "/results/{job_id}",
    response_model=PredictionOutput,
    responses={404: {"model": ErrorResponse, "description": "找不到該 job_id"}, 401: _ERRORS[401]},
    tags=["Results"],
    summary="查詢辨識結果",
)
async def get_result(job_id: str, _client: str = Depends(require_token)):
    """依 `job_id` 查詢先前的單張辨識結果（批次中的每一張也有自己的 `job_id`）。結果存於記憶體，server 重啟後清除。"""
    result = _result_store.get(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"找不到 job_id={job_id} 的結果")
    return result


@app.get(
    "/results",
    response_model=List[PredictionOutput],
    responses={401: _ERRORS[401]},
    tags=["Results"],
    summary="列出辨識結果",
)
async def list_results(
    limit: int = Query(50, ge=1, le=1000, description="最多回傳幾筆"),
    offset: int = Query(0, ge=0, description="略過前幾筆"),
    _client: str = Depends(require_token),
):
    """依辨識時間由舊到新列出記憶體中的結果。"""
    return list(_result_store.values())[offset: offset + limit]
