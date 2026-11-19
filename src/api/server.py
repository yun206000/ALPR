"""
api/server.py — FastAPI REST API 服務。

除了 /health 與 API 文件之外，所有端點都要帶 token：
    Authorization: Bearer <token>
token 設定在 config.yaml 的 api.tokens（或環境變數 ALPR_API__TOKENS）。

每個辨識請求在 output_dir 存「一份」JSON，/results 直接讀這些檔案（server 重啟後仍查得到）：
    POST /predict        → <檔名>_<job_id 前 8 碼>.json（PredictionOutput）
    POST /predict/batch  → batch_<batch_id 前 8 碼>.json（BatchOutput，整批一份）

上傳限制（config.yaml 的 api 區塊，超過時回 413）：
    - 整個請求：max_upload_mb × max_batch_files + 1 MB，依 Content-Length 在讀取內容之前判斷
    - 單一檔案：max_upload_mb；批次張數：max_batch_files；影像像素：max_image_pixels

所有錯誤回應都是 {"detail": "<訊息>"}。
"""

from __future__ import annotations

import io
import itertools
import re
import secrets
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image
from pydantic import ValidationError
from starlette.types import ASGIApp, Receive, Scope, Send

from ..config import get_config
from ..models import BatchOutput, ErrorResponse, PredictionOutput
from ..pipeline import LPRPipeline
from ..utils import logger

# ---------------------------------------------------------------------------
# 全域狀態
# ---------------------------------------------------------------------------

_pipeline: Optional[LPRPipeline] = None
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
    logger.info(f"辨識結果目錄：{_pipeline.config.output_dir.resolve()}")
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

_MB = 1024 * 1024


def _max_file_bytes() -> int:
    return int(get_config().api.max_upload_mb * _MB)


def _max_request_bytes() -> int:
    """整個請求的上限：每個檔案都到上限、張數也到上限，再加 1 MB 給 multipart 的欄位標頭。"""
    return _max_file_bytes() * get_config().api.max_batch_files + _MB


class RequestSizeLimitMiddleware:
    """在讀取 body 之前，依 Content-Length 擋掉過大的上傳。

    - 沒有 Content-Length（chunked 上傳）→ 411，否則無法事先知道大小
    - Content-Length 不是非負整數 → 400
    - 超過上限 → 413
    Uvicorn 會確保實際收到的 body 不超過 Content-Length，所以這裡判斷就足夠。
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["method"] in ("POST", "PUT", "PATCH"):
            error = self._check(dict(scope["headers"]).get(b"content-length"))
            if error is not None:
                status_code, detail = error
                await JSONResponse(status_code=status_code, content={"detail": detail})(scope, receive, send)
                return
        await self.app(scope, receive, send)

    @staticmethod
    def _check(content_length: Optional[bytes]):
        if content_length is None:
            return status.HTTP_411_LENGTH_REQUIRED, "需要 Content-Length 標頭（不接受 chunked 上傳）"
        if not content_length.isdigit():
            return status.HTTP_400_BAD_REQUEST, "Content-Length 不正確"
        limit = _max_request_bytes()
        if int(content_length) > limit:
            return (
                status.HTTP_413_CONTENT_TOO_LARGE,
                f"請求太大：{int(content_length) / _MB:.1f} MB，上限 {limit / _MB:.0f} MB",
            )
        return None


app.add_middleware(RequestSizeLimitMiddleware)
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
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, content={"detail": "; ".join(messages)})


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
    411: {"model": ErrorResponse, "description": "缺少 Content-Length（不接受 chunked 上傳）"},
    413: {"model": ErrorResponse, "description": "超過上傳限制（請求大小、檔案大小、張數或影像像素）"},
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


def _too_large(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail=detail)


async def _read_upload(file: UploadFile) -> bytes:
    """讀取上傳檔案；超過 max_upload_mb 時回 413（最多只讀上限 + 1 byte 進記憶體）。"""
    max_bytes = _max_file_bytes()
    name = file.filename or "file"
    if file.size is not None and file.size > max_bytes:
        raise _too_large(f"檔案太大：{name}（{file.size / _MB:.1f} MB，上限 {max_bytes / _MB:g} MB）")
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise _too_large(f"檔案太大：{name}（上限 {max_bytes / _MB:g} MB）")
    return data


def _check_pixels(width: int, height: int, filename: str) -> None:
    max_pixels = get_config().api.max_image_pixels
    if width * height > max_pixels:
        raise _too_large(f"影像太大：{filename}（{width}×{height}，上限 {max_pixels:,} 像素）")


def _decode_image_bytes(data: bytes, filename: Optional[str] = None) -> np.ndarray:
    """將上傳的 bytes 解碼為 BGR ndarray。

    解碼前先只讀影像標頭檢查寬高，避免「檔案很小、解碼後佔用大量記憶體」的影像（解壓縮炸彈）。
    """
    name = filename or "file"
    if not data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"無法解碼影像：{name}")
    try:
        with Image.open(io.BytesIO(data)) as header:  # 只解析標頭，不解碼像素
            width, height = header.size
    except Image.DecompressionBombError:
        raise _too_large(f"影像太大：{name}")
    except Exception:
        width = height = None  # Pillow 不認得的格式交給 OpenCV，解碼後再檢查
    if width is not None:
        _check_pixels(width, height, name)

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"無法解碼影像：{name}")
    _check_pixels(img.shape[1], img.shape[0], name)
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

    檔案大小上限為 `api.max_upload_mb`、影像寬 × 高上限為 `api.max_image_pixels`，超過時回 413。

    結果會存成一份 JSON 到 `output_dir`，之後可以用 `job_id` 查詢。
    """
    pipeline = _get_pipeline()
    img = _decode_image_bytes(await _read_upload(file), file.filename)
    result = await _predict(pipeline, img, file.filename or "upload.jpg", save_json=True)
    logger.info(f"[{client}] /predict {result.source_file} → {len(result.plates)} 個車牌")
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
    - 整批結果存成**一份** JSON 到 `output_dir`，之後可以用 `batch_id` 查詢。
    - 一次最多 `api.max_batch_files` 張，超過時整批回 413。
    - 單張超過檔案大小或像素上限時，該張的 `error` 有錯誤訊息，其他張照常辨識。
    """
    pipeline = _get_pipeline()
    max_files = get_config().api.max_batch_files
    if len(files) > max_files:
        raise _too_large(f"一次最多上傳 {max_files} 張，這次有 {len(files)} 張")
    results: List[PredictionOutput] = []

    for file in files:
        name = file.filename or "upload.jpg"
        try:
            img = _decode_image_bytes(await _read_upload(file), name)
            result = await _predict(pipeline, img, name, save_json=False)
        except HTTPException as e:
            result = PredictionOutput(source_file=name, error=str(e.detail))
        except Exception as e:
            logger.exception(f"辨識失敗：{name}")
            result = PredictionOutput(source_file=name, error=str(e))
        results.append(result)

    batch = BatchOutput(total_images=len(files), results=results)
    await run_in_threadpool(pipeline.save_batch, batch)
    logger.info(f"[{client}] /predict/batch {len(files)} 張 → batch_id={batch.batch_id}")
    return batch


# ---------------------------------------------------------------------------
# 結果查詢：直接讀 output_dir 裡的 JSON
# ---------------------------------------------------------------------------

Record = Union[BatchOutput, PredictionOutput]

# 辨識結果的檔名：<檔名>_<id 前 8 碼>.json、batch_<id 前 8 碼>.json（eval_*.json 等其他檔案不算）
_RECORD_NAME = re.compile(r"^.+_[0-9a-f]{8}\.json$")


def _output_dir() -> Path:
    return _get_pipeline().config.output_dir


def _load_record(path: Path) -> Optional[Record]:
    """讀一個結果檔；不是辨識結果（或讀取失敗）就回傳 None。"""
    try:
        text = path.read_text(encoding="utf-8")
        if path.name.startswith("batch_"):
            return BatchOutput.model_validate_json(text)
        return PredictionOutput.model_validate_json(text)
    except (OSError, ValidationError):
        return None


def _record_id(record: Record) -> str:
    return record.batch_id if isinstance(record, BatchOutput) else record.job_id


def _list_record_files() -> List[Path]:
    """所有結果檔，新的在前。"""
    files = []
    for path in _output_dir().glob("*.json"):
        if _RECORD_NAME.match(path.name):
            try:
                files.append((path.stat().st_mtime, path))
            except OSError:  # 剛好被刪掉
                continue
    return [path for _, path in sorted(files, reverse=True)]


@app.get(
    "/results/{result_id}",
    response_model=Record,
    responses={404: {"model": ErrorResponse, "description": "找不到該筆結果"}, **_ERRORS},
    tags=["Results"],
    summary="查詢辨識結果",
)
async def get_result(result_id: str, _client: str = Depends(require_token)):
    """
    用 `/predict` 回傳的 `job_id`，或 `/predict/batch` 回傳的 `batch_id` 查詢先前的結果。

    - `job_id` → 回傳 `PredictionOutput`（單張）
    - `batch_id` → 回傳 `BatchOutput`（整批，有 `batch_id` 與 `results` 欄位）
    """
    try:
        result_id = str(uuid.UUID(result_id))
    except ValueError:
        raise HTTPException(status_code=404, detail=f"找不到 id={result_id} 的結果")

    def find() -> Optional[Record]:
        for path in _output_dir().glob(f"*_{result_id[:8]}.json"):
            record = _load_record(path)
            if record is not None and _record_id(record) == result_id:
                return record
        return None

    record = await run_in_threadpool(find)
    if record is None:
        raise HTTPException(status_code=404, detail=f"找不到 id={result_id} 的結果")
    return record


@app.get(
    "/results",
    response_model=List[Record],
    responses=_ERRORS,
    tags=["Results"],
    summary="列出辨識結果",
)
async def list_results(
    limit: int = Query(50, ge=1, le=1000, description="最多回傳幾筆"),
    offset: int = Query(0, ge=0, description="略過前幾筆"),
    _client: str = Depends(require_token),
):
    """
    列出 `output_dir` 裡的辨識結果，**新的在前**。

    每一筆是一次請求：單張是 `PredictionOutput`，批次是 `BatchOutput`（有 `batch_id` 欄位）。
    """

    def page() -> List[Record]:
        # 先略過讀不出來的檔案再分頁：壞掉的檔案不佔名額，每頁都是完整的 limit 筆
        records = (r for r in map(_load_record, _list_record_files()) if r is not None)
        return list(itertools.islice(records, offset, offset + limit))

    return await run_in_threadpool(page)
