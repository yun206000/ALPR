# LPR Pipeline — RF-DETR + PARSeq

車牌辨識系統，使用 RF-DETR 進行車牌偵測，PARSeq 進行字元辨識。

## 專案結構

```
lpr_pipeline/
├── src/
│   ├── detector/        # RF-DETR 車牌偵測模組
│   ├── recognizer/      # PARSeq 字元辨識模組
│   ├── pipeline/        # 整合 pipeline 邏輯
│   ├── utils/           # 共用工具（影像處理、logger）
│   └── api/             # FastAPI REST 服務
├── models/              # 模型權重存放目錄
├── data/samples/        # 測試影像
├── outputs/             # 辨識結果 JSON
├── tests/               # 單元 & 整合測試
├── docker/              # Dockerfile & compose
├── requirements.txt
└── main.py              # CLI 入口
```

## 快速開始

### 本機執行

```bash
pip install -r requirements.txt

# 單張影像
python main.py --image data/samples/car.jpg

# 批次資料夾
python main.py --folder data/samples/ --output outputs/results.json

# 啟動 API Server
python main.py --serve --host 0.0.0.0 --port 8000
```

### Docker

```bash
# 建置映像
docker build -f docker/Dockerfile -t lpr-pipeline:latest .

# 執行 CLI
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs \
  lpr-pipeline:latest --folder /app/data/samples

# 執行 API Server
docker compose -f docker/docker-compose.yml up
```

## API 端點

| Method | Path | 說明 |
|--------|------|------|
| POST | `/predict` | 上傳影像進行辨識 |
| POST | `/predict/batch` | 批次上傳多張影像 |
| GET  | `/results/{job_id}` | 查詢辨識結果 |
| GET  | `/health` | 健康檢查 |

## 輸出格式

```json
{
  "job_id": "uuid-...",
  "timestamp": "2026-05-19T12:00:00Z",
  "source_file": "car.jpg",
  "plates": [
    {
      "plate_text": "ABC-1234",
      "confidence": 0.97,
      "detector_score": 0.98,
      "recognizer_score": 0.95,
      "bbox": [x1, y1, x2, y2]
    }
  ],
  "processing_time_ms": 123.4
}
```
