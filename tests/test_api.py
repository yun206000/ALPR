"""FastAPI server：token 驗證、上傳限制、辨識端點、結果查詢（假模型）。"""

import io
import os
import uuid

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api import server
from src.models import BatchOutput, PredictionOutput

from conftest import AUTH, TEST_TOKEN, jpg_bytes


@pytest.fixture
def client(fake_pipeline, monkeypatch):
    """lifespan 照常執行（讀 token、建立 pipeline、warmup），模型是假的。"""
    monkeypatch.setattr(server, "_pipeline", None)
    monkeypatch.setattr(server, "_tokens", {})
    monkeypatch.setattr(server, "LPRPipeline", lambda: fake_pipeline)  # lifespan 建立的就是同一個 pipeline
    with TestClient(server.app) as c:
        yield c


def _png_bomb(w, h) -> bytes:
    buf = io.BytesIO()
    Image.new("1", (w, h)).save(buf, "PNG", optimize=True)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 啟動 / 驗證
# ---------------------------------------------------------------------------

def test_health_and_lifespan(client):
    assert client.get("/health").json() == {"status": "ok", "pipeline_ready": True}
    assert server._tokens == {TEST_TOKEN: "test"}
    assert server._pipeline.detector.warmed


def test_health_before_ready(app_config, monkeypatch):
    monkeypatch.setattr(server, "_pipeline", None)
    c = TestClient(server.app)  # 不進 with → 不執行 lifespan
    assert c.get("/health").json()["pipeline_ready"] is False
    monkeypatch.setattr(server, "_tokens", {TEST_TOKEN: "test"})
    r = c.post("/predict", files={"file": ("a.jpg", jpg_bytes())}, headers=AUTH)
    assert r.status_code == 503


@pytest.mark.parametrize("tokens,match", [("{}", "沒有設定任何 API token"), ('{"a": "short"}', "至少要 16 個字元")])
def test_startup_rejects_bad_tokens(app_config, monkeypatch, tokens, match):
    monkeypatch.setenv("ALPR_API__TOKENS", tokens)
    from src.config import load_config

    load_config()
    with pytest.raises(RuntimeError, match=match):
        server._load_tokens()


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong-token-wrong-token"}, {"Authorization": "Basic abc"}])
def test_requires_valid_token(client, headers):
    r = client.get("/results", headers=headers)
    assert r.status_code == 401
    assert r.json() == {"detail": "缺少或無效的 API token"}
    assert r.headers["www-authenticate"] == "Bearer"


# ---------------------------------------------------------------------------
# 辨識
# ---------------------------------------------------------------------------

def test_predict_single(client, fake_pipeline):
    r = client.post("/predict", files={"file": ("car.jpg", jpg_bytes(320, 240))}, headers=AUTH)
    assert r.status_code == 200
    body = PredictionOutput.model_validate(r.json())
    assert (body.source_file, body.image_width, body.image_height) == ("car.jpg", 320, 240)
    assert body.best_plate().plate_text == "ABC-1234"
    assert (fake_pipeline.config.output_dir / f"car_{body.job_id[:8]}.json").exists()


@pytest.mark.parametrize("data", [b"", b"not an image"])
def test_predict_undecodable(client, data):
    r = client.post("/predict", files={"file": ("x.jpg", data)}, headers=AUTH)
    assert r.status_code == 400 and r.json()["detail"] == "無法解碼影像：x.jpg"


def test_predict_missing_file_field(client):
    r = client.post("/predict", data={"other": "1"}, headers=AUTH)
    assert r.status_code == 422 and r.json()["detail"] == "file: Field required"


def test_predict_batch(client, fake_pipeline):
    files = [("files", ("a.jpg", jpg_bytes())), ("files", ("bad.jpg", b"xx")), ("files", ("b.jpg", jpg_bytes()))]
    r = client.post("/predict/batch", files=files, headers=AUTH)
    assert r.status_code == 200
    batch = BatchOutput.model_validate(r.json())
    assert [x.source_file for x in batch.results] == ["a.jpg", "bad.jpg", "b.jpg"]
    assert [x.error for x in batch.results] == [None, "無法解碼影像：bad.jpg", None]
    assert (fake_pipeline.config.output_dir / f"batch_{batch.batch_id[:8]}.json").exists()


def test_predict_batch_unexpected_error_is_per_file(client, fake_pipeline, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("GPU 爆了")

    monkeypatch.setattr(fake_pipeline, "predict", boom)
    r = client.post("/predict/batch", files=[("files", ("a.jpg", jpg_bytes()))], headers=AUTH)
    assert r.status_code == 200 and r.json()["results"][0]["error"] == "GPU 爆了"


# ---------------------------------------------------------------------------
# 上傳限制
# ---------------------------------------------------------------------------

def test_file_too_large(client, app_config):
    big = jpg_bytes() + b"\0" * int(app_config.api.max_upload_mb * 1024 * 1024)
    r = client.post("/predict", files={"file": ("big.jpg", big)}, headers=AUTH)
    assert r.status_code == 413 and r.json()["detail"].startswith("檔案太大：big.jpg")
    r = client.post("/predict/batch", files=[("files", ("big.jpg", big)), ("files", ("ok.jpg", jpg_bytes()))], headers=AUTH)
    assert [x["error"] is None for x in r.json()["results"]] == [False, True]


def test_read_upload_stops_at_limit(app_config):
    """UploadFile.size 不可信或沒有時，仍只讀上限 + 1 byte。"""
    import asyncio
    from starlette.datastructures import UploadFile

    limit = server._max_file_bytes()
    upload = UploadFile(io.BytesIO(b"x" * (limit + 100)), filename=None, size=None)
    with pytest.raises(server.HTTPException) as exc:
        asyncio.run(server._read_upload(upload))
    assert exc.value.status_code == 413 and "file" in exc.value.detail
    assert upload.file.tell() == limit + 1


def test_pixel_limits(client, app_config, monkeypatch):
    bomb = _png_bomb(10000, 10000)  # 檔案只有幾十 KB，解碼後 1 億像素
    r = client.post("/predict", files={"file": ("bomb.png", bomb)}, headers=AUTH)
    assert r.status_code == 413 and "10000×10000" in r.json()["detail"]

    # Pillow 判定為解壓縮炸彈（超過 Image.MAX_IMAGE_PIXELS × 2）
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)
    r = client.post("/predict", files={"file": ("a.png", _png_bomb(100, 100))}, headers=AUTH)
    assert r.status_code == 413 and r.json()["detail"] == "影像太大：a.png"


def test_pixel_limit_after_decode_for_formats_pillow_cannot_read(app_config, monkeypatch):
    """Pillow 讀不了標頭的格式，交給 OpenCV 解碼後再檢查像素。"""
    monkeypatch.setattr(server.Image, "open", lambda *_: (_ for _ in ()).throw(OSError("unknown format")))
    img = server._decode_image_bytes(jpg_bytes(64, 32), "x.jpg")
    assert img.shape == (32, 64, 3)
    monkeypatch.setenv("ALPR_API__MAX_IMAGE_PIXELS", "100")
    from src.config import load_config

    load_config()
    with pytest.raises(server.HTTPException) as exc:
        server._decode_image_bytes(jpg_bytes(64, 32))
    assert exc.value.status_code == 413 and "file" in exc.value.detail


def test_too_many_batch_files(client, app_config):
    n = app_config.api.max_batch_files + 1
    r = client.post("/predict/batch", files=[("files", (f"{i}.jpg", b"x")) for i in range(n)], headers=AUTH)
    assert r.status_code == 413 and f"這次有 {n} 張" in r.json()["detail"]


def test_request_size_middleware(client):
    def chunked():
        yield b"--b\r\n"

    multipart = {"Content-Type": "multipart/form-data; boundary=b"}
    r = client.post("/predict", content=chunked(), headers={**AUTH, **multipart})
    assert r.status_code == 411
    r = client.post("/predict", content=b"x", headers={**AUTH, **multipart, "Content-Length": "-1"})
    assert r.status_code == 400 and r.json()["detail"] == "Content-Length 不正確"
    r = client.post("/predict", content=b"x", headers={**AUTH, **multipart, "Content-Length": str(10**10),
                                                         "Origin": "http://example.com"})
    assert r.status_code == 413 and r.json()["detail"].startswith("請求太大")
    assert r.headers["access-control-allow-origin"] == "*"  # 413 也帶 CORS 標頭，瀏覽器才讀得到錯誤訊息
    assert client.get("/health").status_code == 200  # GET 不檢查


# ---------------------------------------------------------------------------
# 結果查詢
# ---------------------------------------------------------------------------

def test_results_lookup_and_listing(client, fake_pipeline):
    single = client.post("/predict", files={"file": ("car.jpg", jpg_bytes())}, headers=AUTH).json()
    batch = client.post("/predict/batch", files=[("files", ("a.jpg", jpg_bytes()))], headers=AUTH).json()
    out_dir = fake_pipeline.config.output_dir
    os.utime(out_dir / f"car_{single['job_id'][:8]}.json", (1, 1))  # 單張比較舊
    (out_dir / "eval_20260101_120000.json").write_text("{}")         # 評估結果，不是辨識結果，不列出
    (out_dir / "broken_0123abcd.json").write_text("not json")        # 最新的檔案但讀取失敗，略過且不佔分頁名額

    assert client.get(f"/results/{single['job_id']}", headers=AUTH).json()["job_id"] == single["job_id"]
    assert client.get(f"/results/{batch['batch_id']}", headers=AUTH).json()["batch_id"] == batch["batch_id"]

    listing = client.get("/results", headers=AUTH).json()
    assert [x.get("batch_id") or x["job_id"] for x in listing] == [batch["batch_id"], single["job_id"]]
    for offset, expected in ((0, batch["batch_id"]), (1, single["job_id"])):
        page = client.get(f"/results?limit=1&offset={offset}", headers=AUTH).json()
        assert [x.get("batch_id") or x["job_id"] for x in page] == [expected]
    assert client.get("/results?offset=2", headers=AUTH).json() == []
    assert client.get("/results?limit=0", headers=AUTH).status_code == 422


@pytest.mark.parametrize("result_id", ["not-a-uuid", str(uuid.uuid4())])
def test_result_not_found(client, fake_pipeline, result_id):
    # 檔名前 8 碼相同、但內容讀不出來的檔案不算
    (fake_pipeline.config.output_dir / f"x_{result_id[:8]}.json").write_text("not json")
    r = client.get(f"/results/{result_id}", headers=AUTH)
    assert r.status_code == 404 and result_id in r.json()["detail"]


def test_list_skips_files_deleted_while_listing(client, fake_pipeline, monkeypatch):
    path = fake_pipeline.config.output_dir / "gone_0123abcd.json"
    path.write_text("{}")
    real_stat = type(path).stat

    def flaky_stat(self, *a, **k):
        if self.name == path.name:
            raise FileNotFoundError
        return real_stat(self, *a, **k)

    monkeypatch.setattr(type(path), "stat", flaky_stat)
    assert server._list_record_files() == []


def test_openapi_has_file_pickers(client):
    schema = client.get("/openapi.json").json()
    assert '"format": "binary"' in str(schema).replace("'", '"')
    assert client.get("/openapi.json").json() == schema  # 第二次用快取
