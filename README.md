# ALPR — 車牌辨識系統（YOLO + PARSeq）

兩階段的台灣車牌辨識系統：

1. **偵測**：YOLO（Ultralytics）在影像中找出車牌位置。
2. **辨識**：把車牌區域裁切出來，交給 PARSeq 辨識車牌字元。

提供 CLI（單張 / 批次辨識、準確率評估）與 FastAPI 服務兩種使用方式。

## 專案結構

```
ALPR/
├── config.yaml                 # 所有設定（權重、裝置、訓練參數……）
├── main.py                     # CLI 入口：辨識、評估、啟動 API Server
├── environment.yml             # conda「ALPR」環境（Docker 也用這份建置）
├── src/
│   ├── config.py               # 讀取 config.yaml
│   ├── models.py               # 輸出資料格式（pydantic）
│   ├── evaluation.py           # 用 ground truth CSV 評估準確率
│   ├── detector/               # YOLO 車牌偵測
│   ├── recognizer/             # PARSeq 字元辨識
│   │   └── transforms.py       #   訓練與推論共用的影像前處理
│   ├── pipeline/               # 偵測 + 辨識整合流程
│   ├── api/                    # FastAPI REST 服務
│   ├── utils/                  # logger、影像處理、讀取 GT CSV、實驗紀錄
│   ├── train/train.py          # 訓練 PARSeq 字元辨識模型
│   └── finetune/finetune_yolo.py  # Fine-tune YOLO 車牌偵測模型
├── models/                     # 所有權重（見「模型權重」）
│   ├── yolo/                   #   YOLO 車牌偵測
│   └── parseq/                 #   PARSeq 字元辨識
├── data/                       # 資料集與推論用影像
├── outputs/                    # 辨識 / 評估結果 JSON、log
├── logs/                       # 實驗紀錄（trackio 資料庫）
└── docker/                     # Dockerfile、docker-compose.yml
```

## 環境安裝

```bash
conda env create -f environment.yml
conda activate ALPR
```

`environment.yml` 是從開發用的 conda 環境直接匯出的。環境有更新時請重新匯出：

```bash
conda env export -n ALPR --no-builds | sed '/^prefix:/d' > environment.yml
```

## 設定（config.yaml）

**所有設定都寫在專案根目錄的 `config.yaml`**，程式只從這份檔案讀取設定，程式碼裡不藏任何預設值。

| 區塊 | 用途 |
|------|------|
| `detector` / `recognizer` | 推論用的權重、裝置、信心門檻等 |
| `output_dir` / `log_level` | 結果 JSON 與 log 的位置、log 等級 |
| `api` | API Server 的 host / port / workers / token |
| `tracking` | 訓練 / 微調的實驗紀錄（trackio） |
| `train_parseq` | `src/train/train.py` 的資料與超參數 |
| `finetune_yolo` | `src/finetune/finetune_yolo.py` 的資料與超參數 |

- 相對路徑以專案根目錄（執行指令的位置）為基準。
- `config.yaml` 少了任何欄位或拼錯，程式一啟動就會報錯，不會默默套用其他值。
- 可以準備多份設定檔，用 `--config other.yaml`（或環境變數 `ALPR_CONFIG`）切換。

需要臨時改某個值、又不想動檔案時，可以覆寫（優先順序：CLI 參數 > 環境變數 > `config.yaml`）：

| 方式 | 範例 |
|------|------|
| `main.py` 的 CLI 參數 | `--device cuda`（同時覆寫 detector / recognizer）、`--log-level DEBUG`、`--port 9000` |
| 訓練腳本的 CLI 參數 | 每個欄位都有對應參數，例如 `--epochs 1`、`--val-csv null`（`--help` 可看全部） |
| 環境變數 | `ALPR_<區塊>__<欄位>`，例如 `ALPR_DETECTOR__DEVICE=cuda`（Docker 用這個方式） |

## 模型權重

所有權重都放在 `models/` 底下，偵測和辨識各一個子資料夾。每次訓練 / fine-tune 成功後，`finetuned/` 整個被新結果**取代**，推論設定不用改：

```
models/
├── yolo/
│   ├── pretrained/license_plate_detector.pt   # 別人訓練好的車牌偵測權重（fine-tune 的起點）
│   └── finetuned/best.pt                      # 最近一次 fine-tune 的結果（每次 fine-tune 會取代）
└── parseq/
    ├── legacy/best.ckpt                       # 整理前就有的訓練結果
    └── finetuned/best.ckpt                    # 最近一次訓練的結果（每次訓練會取代）
```

推論要用哪一組權重，由 `config.yaml` 決定。預設**直接指向訓練輸出的檔案**，不需要另外複製或搬移：

| 用途 | 設定 | 目前的值 |
|------|------|----------|
| YOLO 車牌偵測 | `detector.weights_path` | `models/yolo/finetuned/best.pt` |
| PARSeq 字元辨識 | `recognizer.weights_path` | `models/parseq/finetuned/best.ckpt` |

- **直接換**：照 `config.yaml` 訓練，成功後自動取代 `finetuned/`。API Server 或 Docker 容器要重啟才會載入新權重。
- **先比較再決定**：用 `--output-dir` 輸出到別的資料夾，`finetuned/` 不會被動到；再用環境變數暫時指向它，在測試集上比較準確率：
  ```bash
  python src/train/train.py --output-dir models/parseq/trial
  ALPR_RECOGNIZER__WEIGHTS_PATH=models/parseq/trial/best.ckpt \
      python main.py --eval data/taiwan_plate_dataset/test/test_gt.csv
  ```
  確定比較好，再把 `trial/` 的檔案搬進 `finetuned/`（或把 `config.yaml` 的路徑改過去）。

每次訓練在 trackio 儀表板上是一個 run（`parseq_<時間>`、`yolo_<時間>`），權重被取代後，舊的訓練曲線仍然留著。

在測試集（84 張）上的比較：fine-tune 過的 `best.pt` 整串準確率 98.81%，`license_plate_detector.pt` 是 96.43%。

- 找不到 YOLO 權重會直接報錯。
- 找不到 PARSeq 權重時，會改用 torch.hub 的官方預訓練模型。這個模型沒針對車牌訓練，準確率會差很多。
- 不要的訓練結果，直接刪掉整個資料夾即可。

## 資料集

`data/taiwan_plate_dataset/` 是 Roboflow 的
[High-precision Taiwan license plate number recognition](https://universe.roboflow.com/jackresearch3/high-precision-taiwan-license-plate-number-recognition)（CC BY 4.0），
內容是車牌裁切圖：

```
taiwan_plate_dataset/
├── train/images/ + train_gt.csv    # 5204 張，PARSeq 訓練用
├── valid/images/ + valid_gt.csv    # 76 張，PARSeq 驗證用
├── test/images/  + test_gt.csv     # 84 張，評估準確率用
└── change.py                       # 把逐字元的 YOLO label 轉成 *_gt.csv
```

Ground truth CSV 的欄位是 `image_path,plate_text`。`image_path` 可以是絕對路徑，也可以是相對路徑，程式會依序找第一個存在的檔案：

1. `--image-root` 指定的資料夾（有給的話）
2. CSV 所在的資料夾，例如 `valid/valid_gt.csv` 裡寫 `xxx.jpg`
3. CSV 上一層的資料夾，例如 `test/test_gt.csv` 裡寫 `test/images/xxx.jpg`
4. CSV 所在資料夾底下的 `images/`，例如 `train/train_gt.csv` 裡寫 `xxx.jpg`，會找 `train/images/xxx.jpg`

只要有任何一張影像找不到，就會直接報錯，不會默默少算樣本。

## 推論

```bash
# 單張影像：結果 JSON 印到 stdout，並存到 output_dir
python main.py --image data/car.jpg

# 整個資料夾：結果整合成一份 JSON（預設 <output_dir>/batch_<id>.json）
python main.py --folder data/cars/ --output outputs/results.json

# 不儲存任何 JSON
python main.py --folder data/cars/ --no-save

# 臨時改用 GPU（不改 config.yaml）
python main.py --folder data/cars/ --device cuda
```

## 評估準確率

對 ground truth CSV 中的每張影像跑完整的 pipeline（偵測 + 辨識），再和 `plate_text` 比對：

```bash
python main.py --eval data/taiwan_plate_dataset/test/test_gt.csv --device cuda
```

```
樣本數      84
完全正確    81（96.43%）
字元準確率  98.63%
未偵測到    1

錯誤樣本（3）
檔名                                                GT        預測
--------------------------------------------------  --------  ----------
000560_jpg.rf.ec2e4f245fa240c2ebcfd947829c4f70.jpg  303-CZL   303-CZL-
000570_jpg.rf.2d3b19183bbb122518619b59e5c8d78c.jpg  NCQ-3555  NC0-3555
000633_jpg.rf.8e66e1a3df83aeb5cbcbba5ae89b8164.jpg  258-QEN   <未偵測到>
```

| 指標 | 說明 |
|------|------|
| 完全正確 | 整串車牌和 GT 完全相同，大小寫、`-` 都要一樣 |
| 字元準確率 | `1 - 總編輯距離 / GT 總字元數`，可以看出錯誤是差一個字還是整串錯 |
| 未偵測到 | YOLO 沒有偵測到任何車牌的張數，這些樣本一律算錯 |

- 每張影像取可信度最高的那個車牌當預測。
- 完整結果（含每張影像的預測）存在 `<output_dir>/eval_<時間>.json`，可以用 `--output` 指定路徑，或用 `--no-save` 不存。

## 模型訓練

### 訓練 PARSeq（字元辨識）

以 PARSeq 官方預訓練權重為起點，在車牌裁切圖上訓練。設定在 `config.yaml` 的 `train_parseq` 區塊：

```bash
python src/train/train.py                      # 完全照 config.yaml
python src/train/train.py --epochs 1 --output-dir /tmp/debug   # 臨時覆寫
```

| 設定 | 說明 |
|------|------|
| `train_csv` / `val_csv` | 訓練 / 驗證集的 GT CSV。`val_csv: null` 時，從訓練集隨機切出 `val_frac` 當驗證集（`seed` 固定，每次切法相同） |
| `freeze_encoder` | `true`：凍結 ViT encoder，只訓練 decoder 與 head |
| `finetune_from` | `null`：從 torch.hub 的預訓練權重開始；填 checkpoint 路徑（例如 `models/parseq/finetuned/best.ckpt`）則以它為起點，只載入模型權重，optimizer 和學習率排程從頭開始 |
| `device` | `null` = 有 GPU 就用 cuda |

輸出在 `output_dir`（預設 `models/parseq/finetuned/`）。訓練過程都寫在系統暫存資料夾，**成功跑完才整個取代** `output_dir` 原本的內容；中途失敗或按 Ctrl+C 中斷，原本的權重不會被動到。如果沒有任何 epoch 的驗證準確率超過訓練前，就不會產生 `best.ckpt`，這時也會保留原本的內容、不取代。

> 舊的權重會被覆蓋。想保留的話，訓練前先把 `finetuned/` 複製一份，或用 `--output-dir` 輸出到別的資料夾。

| 檔案 | 說明 |
|------|------|
| `best.ckpt` | 驗證集完全正確率最高的權重 |
| `last.ckpt` | 最後一個 epoch 的權重 |
| `loss_curve.png` | train / val loss 曲線 |
| `train.log` | 訓練過程的完整 log |

`recognizer.weights_path` 預設就指向 `finetuned/best.ckpt`，API Server 要重新啟動才會載入新權重。

### Fine-tune YOLO（車牌偵測）

需要**車牌層級**的偵測資料集，也就是 YOLO 格式、只有一個類別 `plate`。`taiwan_plate_dataset` 的標註是逐字元的，不能拿來訓練偵測器。

```
<資料集>/
├── train/images/ + labels/
└── valid/images/ + labels/     # 沒有的話會從 train 自動切
```

設定在 `config.yaml` 的 `finetune_yolo` 區塊：

```bash
python src/finetune/finetune_yolo.py                     # 完全照 config.yaml
python src/finetune/finetune_yolo.py --epochs 1 --batch-size 8   # 臨時覆寫
```

| 設定 | 說明 |
|------|------|
| `data_root` | 偵測資料集，底下要有 `train/`、`valid/` |
| `val_frac` | 沒有 `valid/` 時，從 `train/` 切出的比例。切分用的 symlink 和 Ultralytics 需要的 `data.yaml` 都放在系統暫存資料夾，訓練完自動刪除，不會動到原始資料 |
| `finetune_from` | 要 fine-tune 的權重；`null` 時改用 `model`（例如 `yolov8n.pt`）的預訓練權重 |
| `freeze` | 凍結前 N 層 |

輸出在 `output_dir`（預設 `models/yolo/finetuned/`），不會產生 `runs/` 資料夾。訓練過程都在系統暫存資料夾進行，**成功跑完才整個取代** `output_dir` 原本的內容；中途失敗或按 Ctrl+C 中斷，原本的權重不會被動到。

> 舊的權重會被覆蓋。想保留的話，fine-tune 前先把 `finetuned/` 複製一份，或用 `--output-dir models/yolo/<別的名稱>` 輸出到別的資料夾。

| 路徑 | 說明 |
|------|------|
| `best.pt` | 最佳權重，`detector.weights_path` 預設就指向這裡；API Server 要重新啟動才會載入新權重 |
| `last.pt` | 最後一個 epoch 的權重 |
| `results.csv` | 每個 epoch 的 loss / mAP |
| `args.yaml` | 本次訓練的完整參數（Ultralytics 產生） |
| `finetune_yolo.log` | 本腳本的 log。Ultralytics 自己的逐 epoch 輸出不會寫進來，請看 `results.csv` |

> Ultralytics 在 GPU 上訓練時會做 AMP 檢查，並在目前目錄下載 `yolo26n.pt`。這個檔案已經加進 `.gitignore`，可以直接刪掉。

### 實驗紀錄（trackio）

訓練 PARSeq 和 fine-tune YOLO 時，指標會寫到 trackio（本機的 `logs/trackio/ALPR.db`）。開啟儀表板看折線圖：

```bash
TRACKIO_DIR=logs/trackio trackio show --project ALPR
```

每次執行是一個 run，名稱為 `parseq_<時間>` 或 `yolo_<時間>`；該次的完整設定也會一起存進紀錄。權重每次都會被取代，但舊的訓練曲線仍然留在 trackio 裡。

| 訓練 | 指標 | step |
|------|------|------|
| PARSeq | `train/loss`、`train/lr` | 每個 batch（global step） |
| | `val/loss`、`val/acc`、`val/best_acc`、`train/epoch_loss`、`epoch` | 每個 epoch 結束（該時的 global step；step 0 是訓練前的基準） |
| YOLO | `train/box_loss`、`train/cls_loss`、`train/dfl_loss`、`lr/pg0~2` | 每個 epoch |
| | `metrics/precision(B)`、`metrics/recall(B)`、`metrics/mAP50(B)`、`metrics/mAP50-95(B)`、`val/*_loss` | 每個 epoch；最後一個點（step = epochs + 1）是 `best.pt` 的最終驗證結果 |

設定在 `config.yaml` 的 `tracking` 區塊。`enabled: false` 可以關閉紀錄，例如臨時除錯時用 `ALPR_TRACKING__ENABLED=false python src/train/train.py --epochs 1`。

## Docker

映像用 `environment.yml`（conda ALPR 環境）建置，CPU / GPU 共用同一個映像。`config.yaml`、權重、資料、輸出都用 volume 掛載：在主機上改 `config.yaml` 後重啟容器就會生效，不用重新 build。

容器的推論裝置由 compose 用環境變數 `ALPR_DETECTOR__DEVICE` / `ALPR_RECOGNIZER__DEVICE` 覆寫（GPU 版是 `cuda`、CPU 版是 `cpu`），其他設定都照 `config.yaml`。

### 事前準備

1. 安裝 Docker。要用 GPU 的話，還要安裝 [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)。
2. 確認 `docker/.env` 的設定：

   ```bash
   LOCAL_UID=1007              # `id -u` 的結果，容器會用這個身分執行
   LOCAL_GID=1007              # `id -g` 的結果
   DATASETS_DIR=../../datasets # YOLO 偵測資料集位置（相對於 docker/），掛到容器內的 /datasets
   API_PORT=8001               # API 對外 port
   ALPR_API_TOKENS='{"web": "<token>"}'  # API token，見下方「API → 驗證」
   ```

   `docker/.env` 含有 token，已加進 `.gitignore`，不要 commit。`ALPR_API_TOKENS` 沒設定時 compose 會直接報錯。

3. 掛載的資料夾必須**先存在，而且是你自己的**。compose 設了 `create_host_path: false`，資料夾不存在時會直接報錯，不會讓 Docker 用 root 自動建立。
   ```bash
   mkdir -p models data outputs logs
   ```

### 掛載對照

| 主機 | 容器 |
|------|------|
| `config.yaml` | `/app/config.yaml`（唯讀） |
| `models/` | `/app/models` |
| `data/` | `/app/data` |
| `outputs/` | `/app/outputs` |
| `logs/` | `/app/logs` |
| `$DATASETS_DIR` | `/datasets` |

### Build 映像

以下指令都在專案根目錄執行。

```bash
docker compose -f docker/docker-compose.yml build
```

- build 時會依 `environment.yml` 建 conda 環境，並預先下載 PARSeq 的 torch.hub 程式碼。第一次要下載 PyTorch + CUDA，會花比較久，映像也有好幾 GB。
- 只改 `src/`、`main.py` 時，重新 build 只會重跑最後幾層，很快。改了 `environment.yml` 才會重建 conda 環境。
- 權重、資料、`config.yaml` 都不會打包進映像（見 `.dockerignore`），改權重或設定不用重新 build。
- CPU / GPU 共用同一個映像（`alpr:latest`）。
- 也可以不透過 compose：`docker build -f docker/Dockerfile -t alpr:latest .`

### 啟動 API Server

```bash
# GPU 版（需要 nvidia-container-toolkit）
docker compose -f docker/docker-compose.yml up -d

# 沒有 GPU 的機器
docker compose -f docker/docker-compose.yml --profile cpu up -d alpr-cpu

# 確認狀態（STATUS 會從 health: starting 變成 healthy）
docker compose -f docker/docker-compose.yml ps

# 查看 log / 停止
docker compose -f docker/docker-compose.yml logs -f
docker compose -f docker/docker-compose.yml down
```

`up` 加上 `--build` 可以一次完成 build + 啟動。啟動後打開 <http://localhost:8001/docs> 就能看到互動式 API 文件。

如果出現 `permission denied while trying to connect to the Docker daemon socket`，代表目前的使用者沒有 Docker 權限：請管理員把你加進 `docker` 群組（`sudo usermod -aG docker $USER`，重新登入後生效）。

### 在容器裡執行其他指令

`docker compose run` 會用同樣的映像與掛載，執行一次性的指令：

```bash
# 評估準確率
docker compose -f docker/docker-compose.yml run --rm alpr \
    python main.py --eval data/taiwan_plate_dataset/test/test_gt.csv

# 訓練 PARSeq
docker compose -f docker/docker-compose.yml run --rm alpr \
    python src/train/train.py

# Fine-tune YOLO
docker compose -f docker/docker-compose.yml run --rm alpr \
    python src/finetune/finetune_yolo.py --data-root /datasets
```

## API

啟動：`python main.py --serve`（host / port 讀 `config.yaml` 的 `api` 區塊）。啟動後 <http://localhost:8000/docs> 有互動式文件，可以直接試打。

| Method | Path | 說明 | 需要 token |
|--------|------|------|:---:|
| GET | `/health` | 健康檢查，`pipeline_ready` 表示模型是否載入完成 | |
| POST | `/predict` | 上傳一張影像，回傳辨識結果 | ✅ |
| POST | `/predict/batch` | 一次上傳多張影像，回傳整批結果 | ✅ |
| GET | `/results/{id}` | 用 `job_id` 或 `batch_id` 查詢先前的結果 | ✅ |
| GET | `/results` | 列出辨識結果（新的在前，分頁） | ✅ |
| GET | `/docs`、`/redoc`、`/openapi.json` | API 文件 | |

### 驗證

除了 `/health` 和文件之外，每個請求都要帶 header：

```
Authorization: Bearer <token>
```

token 設定在 `config.yaml` 的 `api.tokens`，格式是「用戶端名稱: token」，可以設定多組，例如網頁和攝影機各用一組，要停用某個用戶端時刪掉那一組就好。用戶端名稱會出現在 log 裡。

token 是機密，`config.yaml` 會進 git，所以建議用環境變數設定：

```bash
# 產生 token
python -c "import secrets; print(secrets.token_urlsafe(32))"

# 啟動
ALPR_API__TOKENS='{"web": "<token>"}' python main.py --serve
```

- Docker 從 `docker/.env` 的 `ALPR_API_TOKENS` 讀取。
- 沒有設定任何 token，或 token 少於 16 個字元時，API Server 會拒絕啟動，避免在沒有保護的情況下對外開放。
- 在 `/docs` 頁面右上角的 **Authorize** 輸入 token 後，就可以直接在頁面上試打。

### Request / Response

所有回應都是 JSON。**所有錯誤回應的格式都一樣**：

```json
{ "detail": "錯誤訊息" }
```

| 狀態碼 | 情況 |
|--------|------|
| 400 | `/predict` 上傳的檔案無法解碼成影像 |
| 401 | 沒帶 token 或 token 錯誤（回應 header 會有 `WWW-Authenticate: Bearer`） |
| 404 | `/results/{id}` 找不到該筆結果 |
| 422 | 請求格式錯誤，例如沒帶檔案、`limit` 超出範圍 |
| 503 | 模型還沒載入完成 |

#### `POST /predict`

Request：`multipart/form-data`，欄位 `file` 放一張影像。

```bash
curl -X POST http://localhost:8000/predict \
     -H "Authorization: Bearer $TOKEN" \
     -F "file=@car.jpg"
```

```js
// 網頁前端
const form = new FormData();
form.append("file", fileInput.files[0]);
const res = await fetch("/predict", {
  method: "POST",
  headers: { Authorization: `Bearer ${token}` },
  body: form,
});
const result = await res.json(); // 失敗時 result.detail 是錯誤訊息
```

Response `200`（`PredictionOutput`）：

```json
{
  "job_id": "3f1c2a9e-8d4b-4c41-9a51-6c2f0d7b1e20",
  "timestamp": "2026-11-04T12:00:00.000000+00:00",
  "source_file": "car.jpg",
  "image_width": 1920,
  "image_height": 1080,
  "plates": [
    {
      "plate_text": "ABC-1234",
      "confidence": 0.9312,
      "detector_score": 0.9521,
      "recognizer_score": 0.9781,
      "bbox": [812.4, 640.1, 1015.7, 702.9]
    }
  ],
  "processing_time_ms": 85.31,
  "error": null
}
```

| 欄位 | 說明 |
|------|------|
| `job_id` | 這次辨識的 ID，可用 `/results/{job_id}` 再查詢 |
| `timestamp` | 辨識時間（UTC，ISO 8601） |
| `image_width` / `image_height` | 原圖尺寸（px） |
| `plates` | 偵測到的車牌，依 `confidence` 由高到低排序；沒有車牌時是空陣列 |
| `plates[].plate_text` | 車牌號碼 |
| `plates[].confidence` | 綜合可信度 = `detector_score` × `recognizer_score` |
| `plates[].bbox` | 車牌在**原圖**上的位置 `[x1, y1, x2, y2]`（px），網頁可以直接拿來畫框 |
| `processing_time_ms` | 推論耗時 |
| `error` | 只有批次中某張影像失敗時才有值，否則為 `null` |

#### `POST /predict/batch`

Request：`multipart/form-data`，每張影像都放在欄位 `files`。

```bash
curl -X POST http://localhost:8000/predict/batch \
     -H "Authorization: Bearer $TOKEN" \
     -F "files=@a.jpg" -F "files=@b.jpg"
```

Response `200`（`BatchOutput`）：

```json
{
  "batch_id": "a7d0e3b2-5f6c-4e8a-b1d9-2c3e4f5a6b7c",
  "timestamp": "2026-11-04T12:00:00.000000+00:00",
  "total_images": 2,
  "results": [
    { "job_id": "...", "source_file": "a.jpg", "plates": [ ... ], "error": null, ... },
    { "job_id": "...", "source_file": "b.jpg", "plates": [], "error": "無法解碼影像：b.jpg", ... }
  ]
}
```

- 用 `batch_id` 可以再查詢整批結果。
- `results` 的每一筆都是上面的 `PredictionOutput`，順序與上傳順序相同。
- 某張無法解碼時不會讓整批失敗，該張的 `plates` 為空、`error` 有錯誤訊息。

#### `GET /results/{id}`

一次請求就是一筆結果：

- `/predict` 回傳的 `job_id` → 回傳 `PredictionOutput`
- `/predict/batch` 回傳的 `batch_id` → 回傳 `BatchOutput`（整批）

批次裡個別影像的 `job_id` 不能單獨查詢，請用 `batch_id` 查整批。

#### `GET /results?limit=50&offset=0`

回傳結果陣列，**新的在前**。單張是 `PredictionOutput`，批次是 `BatchOutput`，前端可以用有沒有 `batch_id` 欄位來區分。`limit` 範圍是 1–1000（預設 50），`offset` 預設 0。

#### `GET /health`

```json
{ "status": "ok", "pipeline_ready": true }
```

### 其他行為

- 每個辨識**請求**在 `output_dir` 存一份 JSON：`/predict` 存 `<檔名>_<job_id 前 8 碼>.json`，`/predict/batch` 整批存成一份 `batch_<batch_id 前 8 碼>.json`。
- `/results` 直接讀 `output_dir` 裡的這些 JSON，server 重啟後仍查得到；刪掉檔案就等於刪掉那筆結果。CLI 辨識（`--image`、`--folder`）存到 `output_dir` 的結果也會列出來，評估報告（`eval_*.json`）不會。
- 推論在背景執行緒中進行，辨識期間 `/health` 等其他請求仍能立即回應。模型同一時間只跑一個推論，多個請求會依序處理。

## 輸出格式

```json
{
  "job_id": "uuid-...",
  "timestamp": "2026-11-04T12:00:00+00:00",
  "source_file": "car.jpg",
  "image_width": 1920,
  "image_height": 1080,
  "plates": [
    {
      "plate_text": "ABC-1234",
      "confidence": 0.97,
      "detector_score": 0.98,
      "recognizer_score": 0.99,
      "bbox": [x1, y1, x2, y2]
    }
  ],
  "processing_time_ms": 123.4,
  "error": null
}
```

- `plates` 依 `confidence`（= `detector_score` × `recognizer_score`）由高到低排序。
- 影像讀取或解碼失敗時，`plates` 為空，`error` 會填入錯誤訊息。
- 辨識結果會經過台灣車牌的後處理：`O` 轉成 `0`、`I` 轉成 `L`。

## 輸出與 log

- **過程訊息**（載入模型、進度、警告）統一用 `src/utils/logger.py` 的 `logger` 輸出到 stderr，格式是：
  `時間 | 等級 | 模組:函式:行號 — 訊息`
- **結果資料**（JSON、摘要表、評估報告）用 `print` 輸出到 stdout，可以直接導到檔案。

Log 檔和 console 的格式相同：

| 情境 | Log 檔 |
|------|--------|
| 推論 / 評估 / API | `<output_dir>/lpr_<日期>.log`（每天換一個檔，保留 7 天） |
| 訓練 PARSeq | `<train_parseq.output_dir>/train.log` |
| Fine-tune YOLO | `<finetune_yolo.output_dir>/finetune_yolo.log` |

`log_level: DEBUG`（或 `--log-level DEBUG`）會額外印出每張影像的偵測數量與耗時。

訓練腳本啟動時，會把這次實際使用的完整設定印到 log 的第一行，方便事後追查。
