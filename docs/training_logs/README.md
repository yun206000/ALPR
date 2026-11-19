# 訓練紀錄（trackio 相容 CSV）

目前推論使用的兩個權重，訓練時 trackio 紀錄的完整指標，以 `src/utils/trackio_export.py` 從 `logs/trackio/ALPR.db` 匯出。

| 檔案 | 內容 | trackio run | 產生的權重 |
|------|------|-------------|------------|
| `parseq_train_trackio.csv` | **PARSeq 字元辨識模型訓練**：每個 step 的 loss / 學習率，每個 epoch 的驗證結果（2,461 列：step 0–2460） | `parseq_20261113_180518` | `models/parseq/finetuned/best.ckpt` |
| `parseq_train_config.json` | 上面那次訓練的設定（`config.yaml` 的 `train_parseq` 區塊 + 執行指令） | 同上 | |
| `yolo_finetune_trackio.csv` | YOLO 車牌偵測模型 fine-tune：每個 epoch 的 loss、驗證指標、學習率（51 列：step 1–51） | `yolo_20261113_180000` | `models/yolo/finetuned/best.pt` |
| `yolo_finetune_config.json` | 上面那次 fine-tune 的設定 | 同上 | |
| `experiments/parseq_lr5e-6_trackio.csv` | PARSeq 超參數實驗：學習率 5e-6（其餘同上） | `parseq_20261120_014036` | 不取代 models/（見 MODEL_CARD.md 4.3） |
| `experiments/parseq_lr5e-4_trackio.csv` | PARSeq 超參數實驗：學習率 5e-4 | `parseq_20261120_014343` | 同上 |
| `experiments/parseq_freeze_encoder_trackio.csv` | PARSeq 超參數實驗：凍結 ViT encoder（lr 5e-5） | `parseq_20261120_014654` | 同上（見 MODEL_CARD.md 4.4） |

## 格式

就是 `trackio.import_csv` 接受的格式：第一列是欄位名稱，`step` 欄必填，`timestamp`（UTC，ISO 8601）選填，其餘欄位都是數值指標；該 step 沒有紀錄的指標留空。

```python
import trackio

trackio.import_csv("docs/training_logs/parseq_train_trackio.csv", project="ALPR-parseq")   # project 必須是新的
# 開啟儀表板：trackio show --project ALPR-parseq
```

- trackio 匯入時會把欄位名稱縮短成「前 10 個英數字或 `/`」，所以匯出時已經把指標改成符合這個規則的名稱（對照見下表）；匯入後的名稱與 CSV 相同。
- 已實際用 trackio 0.45 匯入驗證：每個 step、時間、數值都與原始紀錄完全相同（PARSeq 5,073 個數值、YOLO 654 個數值）。`tests/test_trackio_export.py` 也有往返測試。
- 重新匯出：

```bash
python -m src.utils.trackio_export --list                    # 列出所有 run
python -m src.utils.trackio_export --latest parseq -o docs/training_logs/parseq_train_trackio.csv \
    --config-output docs/training_logs/parseq_train_config.json
```

## 欄位：PARSeq（`parseq_train_trackio.csv`）

每個 epoch 82 個 step（5,204 張 ÷ batch 64），共 30 epochs；step 0 是訓練前的驗證結果。

| 欄位 | trackio 原始名稱 | 紀錄時機 | 意義 |
|------|------------------|----------|------|
| `step` | — | | 訓練 step（第幾個 batch） |
| `timestamp` | — | | 紀錄時間（UTC） |
| `epoch` | `epoch` | step 0 與每個 epoch 結束 | 第幾個 epoch（0 = 訓練前） |
| `train/loss` | `train/loss` | 每個 step | 該 batch 的訓練 loss（permutation language modeling 的 cross-entropy） |
| `train/lr` | `train/lr` | 每個 step | 該 step 使用的學習率（cosine 衰減，5e-5 → 0） |
| `epoch/loss` | `train/epoch_loss` | 每個 epoch 結束 | 該 epoch 所有 batch 訓練 loss 的平均 |
| `val/loss` | `val/loss` | step 0 與每個 epoch 結束 | 驗證集 loss |
| `val/acc` | `val/acc` | step 0 與每個 epoch 結束 | 驗證集完全正確率（0–1；先用 YOLO 裁出車牌，經過與推論相同的後處理） |
| `val/best` | `val/best_acc` | 每個 epoch 結束 | 到目前為止最好的驗證準確率 |

## 欄位：YOLO（`yolo_finetune_trackio.csv`）

`step` = epoch（1–50）；step 51 是訓練結束後用 `best.pt` 再驗證一次的結果（只有 precision / recall / mAP）。

| 欄位 | trackio 原始名稱（Ultralytics） | 意義 |
|------|------------------|------|
| `train/box` / `val/box` | `train/box_loss` / `val/box_loss` | 框位置的 loss（CIoU） |
| `train/cls` / `val/cls` | `train/cls_loss` / `val/cls_loss` | 類別的 loss（BCE） |
| `train/dfl` / `val/dfl` | `train/dfl_loss` / `val/dfl_loss` | Distribution Focal Loss（框邊界分佈） |
| `precision` | `metrics/precision(B)` | 驗證集 precision |
| `recall` | `metrics/recall(B)` | 驗證集 recall |
| `mAP50` | `metrics/mAP50(B)` | IoU 0.5 的 mAP |
| `mAP50to95` | `metrics/mAP50-95(B)` | IoU 0.5–0.95 的平均 mAP |
| `lr/pg0`、`lr/pg1`、`lr/pg2` | 同左 | 三個參數群組的學習率（權重、BatchNorm、bias；前 3 個 epoch 是 warmup） |
