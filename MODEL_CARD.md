# ALPR 模型卡（Model Card）— 台灣車牌辨識：YOLOv8n 車牌偵測 + PARSeq 字元辨識

| 項目 | 內容 |
|------|------|
| 系統名稱 | ALPR（Automatic License Plate Recognition）台灣車牌辨識系統 |
| 模型版本 | 1.0.0（權重訓練日期 2026-11-13） |
| 文件日期 | 2026-11-20 |
| 任務 | 從車輛影像找出台灣車牌位置，並辨識車牌號碼（例如 `ABC-1234`） |
| 模型組成 | ① YOLOv8n 車牌偵測（Ultralytics）② PARSeq 字元辨識（ViT + Transformer decoder） |
| 框架 | PyTorch 2.11、Ultralytics 8.4.56、PARSeq（`baudm/parseq`，torch.hub） |
| 交付物 | Docker 映像 `alpr-bundle:1.0.0`（內含權重與環境）、PARSeq 訓練紀錄（trackio 相容 CSV，`docs/training_logs/`）、本文件 |
| 量測環境 | Intel Core i7-14700F（28 執行緒）、32 GB RAM、NVIDIA GeForce RTX 4090（24 GB）、Ubuntu（Linux 6.8） |

**權重檔**

| 模組 | 檔案 | 大小 | SHA-256 |
|------|------|------|---------|
| 車牌偵測 | `models/yolo/finetuned/best.pt` | 6.1 MB | `f9b1226c3e2aa242424c50e0136d771c9f55888a22e867eaa523470c348fa4c9` |
| 字元辨識 | `models/parseq/finetuned/best.ckpt` | 95.4 MB | `290dbc4cad704bd56d322985fd2f5fc918c0939edc5f0a73dc5058f2b301c7b4` |
| （偵測的 fine-tune 起點） | `models/yolo/pretrained/license_plate_detector.pt` | 6.2 MB | `8ec3b254a6c87610f037a90957462cafa11a9c03224e33a28c6a1d1ac2ac51b0` |
| （辨識的訓練起點，torch.hub 官方權重） | `parseq-bb5792a6.pt` | 95.4 MB | `bb5792a68e367476abca029cbf8699abc805f3d3dc7e57aae45c8ec4f7b7cd00` |

---

## 目錄

1. [各模組 AI 模型簡介與具體用途](#1-各模組-ai-模型簡介與具體用途)
2. [模型效能指標與綜合效能分數](#2-模型效能指標與綜合效能分數)
3. [模型參數數量與重要超參數之意義](#3-模型參數數量與重要超參數之意義)
4. [重要超參數對效能的影響](#4-重要超參數對效能的影響)
5. [預訓練資料集](#5-預訓練資料集)
6. [訓練細節、測試細節與最終效能數據](#6-訓練細節測試細節與最終效能數據)
7. [訓練與推論所需參考時間](#7-訓練與推論所需參考時間)
8. [最低軟硬體需求](#8-最低軟硬體需求)
9. [使用手冊](#9-使用手冊)
10. [限制與注意事項](#10-限制與注意事項)

---

## 1. 各模組 AI 模型簡介與具體用途

### 1.1 系統流程

```
車輛影像（BGR）
   │
   ├─①─ YOLOv8n 車牌偵測 ──→ 車牌框 (x1, y1, x2, y2) + 偵測分數
   │                              │（分數 ≥ 0.4 的框，外擴 5% 後裁切）
   │                              ▼
   └─────── 轉成 RGB ──────→ 車牌裁切影像
                                  │
                       ②─ PARSeq 字元辨識 ──→ 車牌字串 + 辨識分數
                                  │（轉大寫、只保留車牌字元、O→0、I→L）
                                  ▼
               輸出：車牌號碼、綜合可信度（偵測分數 × 辨識分數）、框座標
```

兩個模型各自訓練、各自可替換：偵測模型只需學會「車牌在哪裡」（1 個類別），辨識模型只看車牌區域，學習目標都很單純。

### 1.2 模型 ①：YOLOv8n 車牌偵測

| 項目 | 內容 |
|------|------|
| 模型 | YOLOv8n（Ultralytics，You Only Look Once v8，nano 尺寸） |
| 類型 | 單階段（one-stage）物件偵測，anchor-free 偵測頭 |
| 架構 | CSPDarknet backbone（C2f 模組）+ PAN-FPN neck + 解耦（decoupled）偵測頭，3 個尺度（stride 8 / 16 / 32） |
| 輸入 | 整張影像，letterbox 縮放到 640×640 |
| 輸出 | 每個車牌的框座標（原圖像素座標）與信心分數；類別只有 `plate` |
| 具體用途 | 在影像中找出所有車牌的位置，交給 PARSeq 辨識；同時提供「偵測分數」作為結果可信度的一部分 |
| 訓練方式 | 從外部取得的車牌偵測權重 `license_plate_detector.pt` 開始，在台灣車牌偵測資料集上 fine-tune |

### 1.3 模型 ②：PARSeq 字元辨識

| 項目 | 內容 |
|------|------|
| 模型 | PARSeq（Permuted Autoregressive Sequence，Bautista & Atienza，ECCV 2022） |
| 類型 | 場景文字辨識（Scene Text Recognition, STR） |
| 架構 | Vision Transformer encoder（12 層）+ Transformer decoder（1 層），以 position query 對影像特徵做 cross-attention |
| 訓練方式 | Permutation Language Modeling：訓練時隨機採樣多種字元輸出順序，同一個模型同時學會自迴歸解碼、並行解碼與迭代精修 |
| 輸入 | 裁切後的車牌影像（RGB），縮放成 32×128（高×寬） |
| 輸出 | 字元序列（最長 25 字）與每個字元的機率 |
| 具體用途 | 讀出車牌號碼；字元機率的平均值作為「辨識分數」 |
| 訓練方式 | 從 torch.hub 官方預訓練權重開始，在台灣車牌字元資料集上訓練（所有層都更新） |

### 1.4 適用與不適用範圍

| | 說明 |
|---|---|
| 適用 | 台灣一般車牌（白底黑字等常見樣式），格式如 `AAA-9999`、`999-AAA`、`AA9-999`；車牌在影像中清楚可見、大致正面 |
| 未驗證 | 夜間、強逆光、大角度傾斜、遮擋、較小的車牌（測試資料中車牌寬 151–416 px）、雙行車牌、外國車牌 |
| 不適用 | 個人身分識別、執法裁罰等需要高可靠度且無人工覆核的用途（見第 10 章） |

---

## 2. 模型效能指標與綜合效能分數

### 2.1 指標定義

| 指標 | 用於 | 定義 |
|------|------|------|
| Precision | YOLO | 偵測出的框中，正確框（與標註 IoU ≥ 0.5）的比例 |
| Recall | YOLO | 標註的車牌中，被正確偵測到的比例 |
| F1 | YOLO | `2 × P × R / (P + R)` |
| mAP@0.5 | YOLO | IoU 門檻 0.5 的平均精確度（Average Precision） |
| mAP@0.5:0.95 | YOLO | IoU 門檻 0.50、0.55、…、0.95 共 10 個門檻的 AP 平均，反映框的位置精準度 |
| 完全正確率（accuracy） | PARSeq、整體系統 | 預測字串與標準答案**完全相同**（含 `-`）的比例 |
| 字元準確率（char accuracy） | PARSeq、整體系統 | `1 − Σ編輯距離 / Σ標準答案字元數`（Levenshtein 距離；沒偵測到車牌時編輯距離 = 答案長度），即 1 − NED |

### 2.2 綜合效能分數的定義

| 分數 | 公式 | 說明 |
|------|------|------|
| **YOLO 綜合分數（fitness）** | `0.1 × mAP@0.5 + 0.9 × mAP@0.5:0.95` | Ultralytics 官方用來挑選最佳 epoch（`best.pt`）的綜合指標，重視框的精準度 |
| **PARSeq 綜合分數** | `(完全正確率 + 字元準確率) / 2` | 同時反映「整串讀對」與「讀錯時差多少」 |
| **系統綜合效能分數** | `(端到端完全正確率 + 端到端字元準確率) / 2`，在 **test 集**上計算 | 整條流程（偵測 + 辨識）對使用者的實際效果；分數 0–100 |

### 2.3 效能總表

**① YOLOv8n 車牌偵測**（`finetune_dataset` valid 400 張，Ultralytics `model.val`，imgsz 640）

| 指標 | 起點 `license_plate_detector.pt` | **最終 `best.pt`** |
|------|------|------|
| Precision | 0.934 | **1.000** |
| Recall | 0.925 | **0.997** |
| F1 | 0.929 | **0.998** |
| mAP@0.5 | 0.898 | **0.995** |
| mAP@0.5:0.95 | 0.289 | **0.783** |
| **綜合分數（fitness）** | 0.350 | **0.804** |

> 訓練過程中 epoch 40 的驗證結果為 fitness 0.805（mAP@0.5:0.95 0.784，見 6.1）；上表是訓練結束後以 `model.val` 對 `best.pt` 重新驗證的結果，兩者的後處理設定略有不同。

**② PARSeq 字元辨識**（`taiwan_plate_dataset` valid 76 張；輸入為 YOLO 裁切的車牌，與推論相同）

| 指標 | 起點（官方預訓練權重） | **最終 `best.ckpt`** |
|------|------|------|
| 完全正確率 | 90.79%（69 / 76） | **100.00%（76 / 76）** |
| 字元準確率 | 98.82% | **100.00%** |
| val loss | 0.0627 | **0.0038** |
| **綜合分數** | 94.81 | **100.00** |

**整體系統（端到端：偵測 + 辨識）**

| 資料 | 指標 | fine-tune 前（兩個起點權重） | **最終（兩個最終權重）** |
|------|------|------|------|
| test 84 張 | 完全正確率 | 94.05%（79） | **100.00%（84）** |
| test 84 張 | 字元準確率 | 98.17% | **100.00%** |
| test 84 張 | 未偵測到 | 1 | **0** |
| valid 76 張 | 完全正確率 | 96.05%（73） | **100.00%（76）** |
| valid 76 張 | 字元準確率 | 98.32% | **100.00%** |
| **test** | **系統綜合效能分數** | **96.11** | **100.00** |

推論速度（每張影像，暖機後）見第 7 章：GPU 約 18.5 ms、CPU 約 49 ms。

---

## 3. 模型參數數量與重要超參數之意義

### 3.1 參數數量

| 模型 | 參數量 | 組成 | 權重大小（FP32） | 運算量 |
|------|--------|------|------------------|--------|
| YOLOv8n 車牌偵測 | **3,011,043**（約 3.0 M） | backbone + neck + 偵測頭（1 類別） | 6.1 MB | 8.2 GFLOPs（640×640，130 層） |
| PARSeq 字元辨識 | **23,832,671**（約 23.8 M） | encoder 21.38 M、decoder 2.37 M、輸出層 0.04 M、文字嵌入 0.04 M、position query 0.01 M | 95.4 MB | 約 6.4 GFLOPs（每個車牌，32×128；其中 encoder 5.45） |
| 合計 | **26,843,714**（約 26.8 M） | | 101.5 MB | 單一車牌的影像約 14.6 GFLOPs |

### 3.2 YOLO 超參數

**架構 / 推論**

| 超參數 | 值 | 意義 |
|--------|----|------|
| 模型尺寸 | `n`（nano） | YOLOv8 的寬度 / 深度倍率（n < s < m < l < x）；越大越準但越慢。車牌只有 1 類、外觀單純，nano 已足夠 |
| `imgsz` | 640 | 輸入影像縮放後的邊長（letterbox，保持長寬比、補灰邊）；越大越容易抓到小車牌，運算量約與邊長平方成正比 |
| `confidence_threshold` | 0.4 | 偵測信心門檻；分數低於門檻的框直接丟掉。調高：誤判框變少但可能漏抓；調低：反之（影響見 4.2） |
| NMS IoU | 0.7（Ultralytics 預設） | 非極大值抑制：兩個框重疊 IoU 超過此值時只保留分數高的，避免同一車牌出現多個框 |

**訓練（fine-tune）**

| 超參數 | 值 | 意義 |
|--------|----|------|
| 起點權重（`finetune_from` / `pretrained`） | `license_plate_detector.pt` | 訓練開始時的權重；沿用已學過車牌的權重可大幅加快收斂（影響見 4.1） |
| `epochs` | 50 | 完整看過訓練集的次數 |
| `batch_size` | 16 | 每次前向傳播的影像數，決定 GPU 記憶體用量；Ultralytics 會累積梯度到 64 張（`nbs`）才更新一次參數，所以等效 batch 是 64 |
| `optimizer` | AdamW | 參數更新演算法；AdamW 是帶有解耦權重衰減的 Adam，對 fine-tune 較穩定 |
| `lr`（lr0） | 5e-4 | 初始學習率：每次更新的步伐大小；太大會破壞預訓練權重、太小收斂慢 |
| `lrf` | 0.01 | 最終學習率比例：學習率從 lr0 線性下降到 lr0 × lrf |
| warmup | 3 epochs | 前 3 個 epoch 學習率從小慢慢升到 lr0，避免一開始的大梯度破壞權重 |
| weight decay | 0.0005 | 權重衰減（L2 正則化），抑制過擬合 |
| `patience` | 20 | early stopping：驗證指標連續 20 個 epoch 沒有進步就停止 |
| `freeze` | 0 | 凍結前 N 層不訓練；0 = 全部層都訓練 |
| mosaic | 1.0（最後 10 epochs 關閉） | 資料增強：把 4 張影像拼成 1 張，增加小物件與背景變化 |
| 其他資料增強 | 水平翻轉 0.5、HSV（h 0.015 / s 0.7 / v 0.4）、縮放 ±50%、平移 ±10% | 增加影像多樣性，提升泛化能力 |
| `seed` | 42 | 隨機種子，讓結果可重現 |

### 3.3 PARSeq 超參數

**架構（沿用官方預訓練模型）**

| 超參數 | 值 | 意義 |
|--------|----|------|
| `img_size` | 32×128 | 輸入影像大小（高×寬）；車牌裁切影像不保持長寬比直接縮放 |
| `patch_size` | 4×8 | ViT 把影像切成 4×8 的小塊，共 8×16 = 128 個 patch |
| `embed_dim` | 384 | 每個 patch / 字元的特徵向量維度 |
| `enc_depth` / `enc_num_heads` | 12 / 6 | encoder 層數 / attention head 數 |
| `dec_depth` / `dec_num_heads` | 1 / 12 | decoder 層數 / attention head 數 |
| `max_label_length` | 25 | 最長可輸出的字元數 |
| `perm_num` | 6 | 訓練時每個 batch 採樣的字元順序（排列）數；越多越能學到不同解碼順序 |
| `decode_ar` / `refine_iters` | True / 1 | 推論時用自迴歸解碼，再用雙向上下文精修 1 次 |
| charset（tokenizer） | 94 個可列印 ASCII 字元 | 模型能輸出的字元集合 |

**訓練（`config.yaml` 的 `train_parseq`）**

| 超參數 | 值 | 意義 |
|--------|----|------|
| 起點 | torch.hub 官方預訓練權重 | 已在大量場景文字上學過「讀字」，只需學台灣車牌的字型與格式 |
| `epochs` | 30 | |
| `batch_size` | 64 | |
| `lr` | 5e-5 | 初始學習率（AdamW）；fine-tune 預訓練 Transformer 通常要比從頭訓練小 1–2 個數量級（影響見 4.3） |
| 學習率排程 | Cosine annealing（每個 step 更新） | 學習率依餘弦曲線從 lr 平滑降到 0，後期小步微調 |
| `weight_decay` | 0.01 | AdamW 的權重衰減，抑制過擬合 |
| gradient clipping | max norm 5.0 | 梯度範數超過 5 時等比例縮小，避免單一 batch 造成過大更新 |
| `freeze_encoder` | false | true 時凍結 ViT encoder，只訓練 decoder 與輸出層（影響見 4.4） |
| 資料增強 | 透視 / 仿射變換、模糊、色彩抖動、自動對比 | 模擬拍攝角度、失焦、光線變化（只用於訓練集） |
| `crop_val` | true | 驗證影像先用 YOLO 裁出車牌，讓驗證時的輸入與實際推論相同 |
| 模型選擇 | 驗證準確率較高，或準確率相同但 val loss 較低 | 決定 `best.ckpt` 是哪個 epoch |
| `seed` | 42 | |

**推論**

| 超參數 | 值 | 意義 |
|--------|----|------|
| `crop_padding` | 0.05 | 裁切車牌時，框的每一邊向外擴張框寬 / 高的 5%，避免切掉邊緣字元（影響見 4.5） |
| `charset` | `0123456789ABCDEFGHJKLMNPQRSTUVWXYZ-OI` | 輸出只保留這些字元（之後 `O→0`、`I→L`） |

---

## 4. 重要超參數對效能的影響

### 4.1 YOLO：起點權重（`pretrained`：沿用預訓練權重 vs 隨機初始化）

兩次 fine-tune 只差在起點（其餘設定、資料、seed 完全相同；Ultralytics `pretrained` 參數）：

| 指標（valid 400 張） | 隨機初始化 | **從 `license_plate_detector.pt` 開始（採用）** |
|------|------|------|
| epoch 1 的 mAP@0.5 | 0.022 | **0.995** |
| epoch 1 的 mAP@0.5:0.95 | 0.006 | **0.538** |
| mAP@0.5 第一次 ≥ 0.95 | epoch 5 | **epoch 1** |
| epoch 10 的 mAP@0.5:0.95 | 0.664 | **0.734** |
| 最佳 mAP@0.5:0.95（50 epochs 內） | 0.767（epoch 50，仍在上升） | **0.784**（epoch 40） |
| 最佳 fitness | 0.790 | **0.805** |

- 預訓練權重已經會找車牌，第 1 個 epoch 就幾乎全部框到（mAP@0.5 0.995），之後的訓練都用在讓框更貼合（mAP@0.5:0.95 提升）。
- 隨機初始化要先花約 5 個 epoch 學會「什麼是車牌」，50 epochs 結束時框的精準度仍落後 0.017，且還沒收斂。
- **結論**：在資料量有限（1,400 張）時，沿用同領域的預訓練權重能同時提升收斂速度與最終精準度。

### 4.2 YOLO：偵測信心門檻 `confidence_threshold`（推論）

以 `best.pt` 在 `conf=0.01` 下偵測，統計各門檻的正確框（TP，與標註 IoU ≥ 0.5）與誤判框（FP）：

| 門檻 | finetune valid（400 個車牌）TP / FP | finetune test（200 個車牌）TP / FP | taiwan test 84 張：漏抓 / 多出第二個框 | 端到端完全正確率（taiwan test） |
|------|------|------|------|------|
| 0.05 | 399 / 59 | 200 / 37 | 0 / 64 | — |
| 0.10 | 399 / 20 | 200 / 18 | 0 / 59 | — |
| 0.20 | 399 / 9 | 200 / 5 | 0 / 29 | — |
| 0.25 | 398 / 5 | 200 / 4 | 0 / 4 | 100% |
| 0.30 | 398 / 5 | 200 / 3 | 0 / 1 | — |
| **0.40（採用）** | **398 / 1** | **200 / 1** | **0 / 1** | **100%** |
| 0.50 | 398 / 1 | 200 / 1 | 0 / 1 | — |

- 門檻主要影響**誤判框數量**：0.05 → 0.40，valid 的誤判框從 59 個降到 1 個，真車牌只少 1 個（該車牌分數 0.246，任何合理門檻都抓不到）。
- 真車牌的分數幾乎都 ≥ 0.65、誤判框幾乎都 ≤ 0.40，兩者之間有明顯間隔；0.4 抓到的真車牌與 0.25 相同，且距離真車牌分數保留較大餘裕。
- 端到端評估只取可信度最高的車牌，所以 0.25 與 0.4 的正確率相同；差別在 API 回應的 `plates` 中誤判框的數量。

### 4.3 PARSeq：學習率 `lr`

其餘設定相同（30 epochs、batch 64、AdamW、cosine 排程、seed 42），只改初始學習率（5e-5 是目前權重的訓練紀錄；5e-6、5e-4 於 2026-11-20 在同一台機器上訓練）：

| 指標 | lr 5e-6 | **lr 5e-5（採用）** | lr 5e-4 |
|------|---------|---------------------|---------|
| epoch 1 驗證準確率 | 98.68% | **100%** | 100% |
| 驗證準確率第一次達到 100% | epoch 5 | **epoch 1** | epoch 1 |
| 達到 100% 後是否又下降 | 否 | **否** | **是**：epoch 2 降到 97.37%、epoch 5 與 20 降到 98.68% |
| 最低 val loss（epoch） | 0.0080（25） | **0.0038（28）** | 0.0036（8） |
| 第 30 epoch 的 val loss | 0.0088 | **0.0043** | 0.0054 |
| 第 30 epoch 的平均 train loss | 0.0571 | 0.0019 | 0.0002 |
| `best.ckpt` 的端到端準確率（test / valid） | 100% / 100% | **100% / 100%** | 100% / 100% |

- **太小（5e-6）**：權重更新太慢，要 5 個 epoch 才全部讀對；30 epochs 結束時 train loss 仍有 0.057，模型還沒學完（underfitting），val loss 是採用設定的 2 倍。
- **太大（5e-4）**：一開始收斂很快，但更新步伐太大，驗證準確率在訓練中反覆掉下來（不穩定）；後期 train loss 幾乎為 0 而 val loss 回升到 0.0054，開始過擬合（overfitting）。最終選到的權重也剛好在 test / valid 上全對，但結果較依賴「在哪個 epoch 停下來」。
- **5e-5**：第 1 個 epoch 就全部讀對、之後不再掉下來，val loss 持續下降到 0.0038，是穩定度與最終 loss 的最佳平衡。
- 三者最終的端到端準確率相同（這份測試集對已收斂的模型沒有鑑別度），差異表現在**收斂速度、訓練穩定度與 val loss**（val loss 越低代表對正確答案越有把握）。

### 4.4 PARSeq：`freeze_encoder`（凍結 ViT encoder）

| 指標 | **false：全部訓練（採用）** | true：只訓練 decoder 與輸出層 |
|------|---------------------|---------------------|
| 可訓練參數 | 23.8 M | 2.5 M |
| 驗證準確率第一次達到 100% | **epoch 1** | epoch 2 |
| 最低 val loss | **0.0038** | 0.0058 |
| 第 30 epoch 的平均 train loss | **0.0019** | 0.0343 |
| 訓練時間（30 epochs） | 約 190 秒 | **141 秒** |
| GPU 記憶體峰值 | 3.75 GB | **1.22 GB** |
| 端到端準確率（test / valid） | 100% / 100% | 100% / 100% |

- 凍結 encoder 時，影像特徵沿用一般場景文字學到的表示，只能調整「怎麼讀」；val loss 高 50%，train loss 也降不下來，代表台灣車牌的字型與版面需要 encoder 一起調整才學得完整。
- 凍結的好處是省 GPU 記憶體（約 1/3）與時間（少 25%）：GPU 記憶體只有 2 GB 左右時可改用 `freeze_encoder: true`。

### 4.5 PARSeq：裁切外擴比例 `crop_padding`（推論）

同一組權重，只改推論時裁切車牌的外擴比例（GPU，偵測門檻 0.4）：

| crop_padding | test 完全正確率（84 張） | test 字元準確率 | valid 完全正確率（76 張） | valid 字元準確率 |
|--------------|------|------|------|------|
| 0（不外擴） | 100% | 100% | 100% | 100% |
| 0.02 | 98.81% | 99.85% | 97.37% | 99.66% |
| **0.05（採用）** | **100%** | **100%** | **100%** | **100%** |
| 0.10 | 96.43% | 99.54% | 97.37% | 99.66% |
| 0.15 | 98.81% | 99.85% | 100% | 100% |
| 0.20 | 98.81% | 99.85% | 100% | 100% |
| 0.30 | 98.81% | 99.85% | 97.37% | 99.66% |

- 外擴比例改變的是 PARSeq 看到的車牌範圍與縮放比例（32×128 固定大小），會讓少數邊界樣本的讀法改變；錯誤幾乎都只差 1 個字元。
- 出錯的主要是同一張影像 `000110`（`EWE-3750` 讀成 `EME-3750`，W / M 形近），另外 0.10 時有 `EPK-2956 → EPK-7956` 與多讀一個 `-`。
- 0.05 與訓練時驗證、挑選模型所用的設定相同，是目前唯一在 test、valid 都全對的非零設定；**換偵測模型或資料來源時，建議重新檢查這個值**。
- valid 用於挑選模型、test 有 56 張與 valid 相同，所以這個比較偏向 0.05；數字只代表「在這份資料上的敏感度」。

---

## 5. 預訓練資料集

### 5.1 YOLO 起點權重 `license_plate_detector.pt`

| 項目 | 內容（取自權重檔內的訓練紀錄） |
|------|------|
| 訓練日期 / 工具 | 2023-06-04，Ultralytics 8.0.112 |
| 架構 | `yolov8n.yaml` |
| 類別 | 1 類：`license_plate` |
| 訓練設定 | 30 epochs、SGD、lr0 0.01、batch 16、imgsz 640 |
| 資料集 | **權重檔沒有紀錄資料集名稱或來源**（訓練紀錄中的資料設定只有 `config.yaml`）；類別名稱顯示是通用的車牌偵測資料，未確認是否包含台灣車牌 |
| 本專案的用法 | 只作為 fine-tune 的起點，355 / 355 個參數張量全部載入；最終模型以 5.3 的台灣資料集重新訓練並驗證 |

### 5.2 PARSeq 官方預訓練權重 `parseq-bb5792a6.pt`

| 項目 | 內容（取自 `baudm/parseq` 官方程式庫） |
|------|------|
| 來源 | `torch.hub.load("baudm/parseq", "parseq", pretrained=True)`，GitHub release 檔案 `parseq-bb5792a6.pt` |
| 訓練資料 | 官方預設設定 `dataset: real`，使用真實場景文字資料：COCO-Text v2.0、RCTW17、Uber-Text、ArT、LSVT、MLT19、ReCTS、TextOCR、OpenVINO（Open Images 的文字標註） |
| 字元集 | 94 個字元（大小寫英文、數字、標點），區分大小寫 |
| 官方驗證結果 | 6 個標準測試集合計 7,672 張，完全正確率 95.95%、1 − NED 98.78%（小寫英數字比對） |
| 授權 | Apache License 2.0 |
| 特性 | 一般場景文字（招牌、文件、街景），**沒有專門看過台灣車牌**：直接使用時常漏讀 `-`、混淆形近字（見 4.3、6.4） |

### 5.3 本專案的訓練與測試資料集

| 資料集 | 用途 | 來源 / 授權 | 內容 |
|--------|------|-------------|------|
| `taiwan_plate_dataset` | PARSeq 訓練、驗證；整條流程的測試 | Roboflow Universe「High-precision Taiwan license plate number recognition」（jackresearch3，version 5），CC BY 4.0 | train 5,204 張（裁切好的車牌）、valid 76 張、test 84 張（416×416 整張影像）；標註轉為 `image_path,plate_text` CSV |
| `finetune_dataset` | YOLO fine-tune（train 訓練、valid 驗證） | taiwan_plate Computer Vision Model dataset（Roboflow YOLO 格式） | train 1,400 張、valid 400 張、test 200 張，640×640，每張 1 個車牌框 |

車牌格式分佈（`taiwan_plate_dataset`，`A` = 字母、`9` = 數字）：

| 格式 | train | valid | test |
|------|-------|-------|------|
| `AAA-9999` | 4,137 | 62 | 69 |
| `999-AAA` | 575 | 10 | 12 |
| `AA9-999` | 79 | 2 | 1 |
| 其他（`AAA9999`、`9999-AAA`、`A99-999`…） | 413 | 2 | 2 |

所有 split 都沒有字母 `I`、`O` 與數字 `4`（台灣車牌不使用 I、O）。

---

## 6. 訓練細節、測試細節與最終效能數據

### 6.1 YOLO 車牌偵測：訓練細節

| 項目 | 內容 |
|------|------|
| 指令 | `python src/finetune/finetune_yolo.py`（設定：`config.yaml` 的 `finetune_yolo`） |
| 訓練日期 / 工具 | 2026-11-13，Ultralytics 8.4.56，PyTorch 2.11 + CUDA 13.0，RTX 4090 |
| 起點 | `license_plate_detector.pt`（355 / 355 個參數張量全部載入） |
| 資料 | `finetune_dataset` train 1,400 張訓練、valid 400 張驗證（640×640，1 類） |
| 設定 | 50 epochs、batch 16（每 4 個 batch 更新一次，等效 batch 64：Ultralytics `nbs=64`）、imgsz 640、AdamW lr0 5e-4 → 線性降到 5e-6（lrf 0.01）、warmup 3 epochs、weight decay 5e-4、AMP 混合精度、seed 42 |
| Loss | box（CIoU）× 7.5 + cls（BCE）× 0.5 + DFL × 1.5（Ultralytics 預設權重） |
| 模型選擇 | 每個 epoch 在 valid 上算 fitness（0.1 × mAP@0.5 + 0.9 × mAP@0.5:0.95），最高的存成 `best.pt` → **epoch 40** |
| 訓練曲線 | `docs/training_logs/yolo_finetune_trackio.csv`、`models/yolo/finetuned/results.csv` |

驗證集指標隨 epoch 的變化（取自訓練紀錄）：

| epoch | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | fitness |
|-------|-----------|--------|---------|--------------|---------|
| 1 | 0.987 | 0.990 | 0.995 | 0.538 | 0.584 |
| 10 | 0.998 | 1.000 | 0.995 | 0.734 | 0.760 |
| 20 | 0.997 | 1.000 | 0.995 | 0.757 | 0.781 |
| **40（best.pt）** | 1.000 | 0.995 | 0.995 | **0.784** | **0.805** |
| 50（last.pt） | 0.998 | 1.000 | 0.995 | 0.777 | 0.799 |

### 6.2 PARSeq 字元辨識：訓練細節

| 項目 | 內容 |
|------|------|
| 指令 | `python src/train/train.py`（設定：`config.yaml` 的 `train_parseq`） |
| 訓練日期 / 工具 | 2026-11-13，PyTorch 2.11 + CUDA 13.0，RTX 4090 |
| 起點 | torch.hub 官方預訓練權重（`parseq-bb5792a6.pt`），所有層都訓練 |
| 資料 | `taiwan_plate_dataset` train 5,204 張（裁切好的車牌）訓練；valid 76 張驗證 |
| 驗證輸入 | valid 是整張影像：訓練開始時先用推論的 YOLO（`best.pt`、門檻 0.4）偵測、取最高分的框、外擴 5% 裁切 —— 與實際推論完全相同；76 張全部偵測成功 |
| 驗證比對 | 模型輸出經過與推論相同的後處理（轉大寫、過濾字元、`O→0`、`I→L`）後，與答案完全相同才算正確 |
| 設定 | 30 epochs × 82 steps、batch 64、AdamW（lr 5e-5、weight decay 0.01）、cosine 學習率（每 step 更新）、gradient clipping 5.0、seed 42 |
| Loss | Permutation Language Modeling：每個 batch 採樣 6 種字元順序，各算一次 cross-entropy 後依 token 數加權平均（與官方實作相同） |
| 資料增強 | RandomPerspective（0.25）、RandomAffine（±3°、平移、縮放 0.9–1.1，0.5）、GaussianBlur（0.25）、ColorJitter（0.6）、RandomAutocontrast（0.15） |
| 模型選擇 | 驗證準確率較高，或準確率相同但 val loss 較低 → **epoch 28** |
| 訓練曲線 | `docs/training_logs/parseq_train_trackio.csv`（trackio 相容，見 `docs/training_logs/README.md`） |

| epoch | 驗證完全正確率 | val loss | 平均 train loss |
|-------|----------------|----------|-----------------|
| 0（訓練前，官方權重） | 90.79%（69 / 76） | 0.0627 | — |
| 1 | 100.00% | 0.0112 | 0.1558 |
| **28（best.ckpt）** | **100.00%** | **0.0038** | 0.0022 |
| 30（last.ckpt） | 100.00% | 0.0043 | 0.0019 |

### 6.3 測試細節

| 項目 | 內容 |
|------|------|
| 測試方式 | `python main.py --eval data/taiwan_plate_dataset/test/test_gt.csv`：每張影像跑完整流程（YOLO → 裁切 → PARSeq → 後處理），取綜合可信度最高的車牌與答案比對 |
| 測試資料 | `taiwan_plate_dataset` test 84 張（416×416 整張影像；與訓練集沒有重疊），另外也測 valid 76 張 |
| 測試資料中的車牌大小 | 寬 151–416 px（中位數 375）、高 67–297 px |
| 推論設定 | 偵測門檻 0.4、crop_padding 0.05、FP32 |
| 指標 | 完全正確率、字元準確率、未偵測到張數（定義見 2.1） |
| 比較組 | 同一流程替換權重：兩個模型都用起點權重（fine-tune 前）、只換其中一個、兩個都用最終權重 |
| 一致性 | CPU 與 GPU 的辨識結果完全相同；偵測門檻 0.25 與 0.4 的結果相同 |
| 自動化測試 | `python -m pytest`：105 個測試（含真實權重的端到端測試），程式碼覆蓋率 99.8% |

### 6.4 最終效能數據

| 模型 | 資料 | 指標 | 最終數值 |
|------|------|------|----------|
| YOLOv8n `best.pt` | finetune valid 400 張 | Precision / Recall / F1 | 1.000 / 0.997 / 0.998 |
| | | mAP@0.5 / mAP@0.5:0.95 | 0.995 / 0.783 |
| | | 綜合分數（fitness） | **0.804** |
| | finetune test 200 張（門檻 0.4） | 正確框 / 誤判框 | 200 / 1（Recall 100%） |
| PARSeq `best.ckpt` | taiwan valid 76 張（YOLO 裁切） | 完全正確率 / 字元準確率 | 100% / 100% |
| | | val loss | 0.0038 |
| | | 綜合分數 | **100.00** |
| 整體系統 | taiwan test 84 張 | 完全正確率 / 字元準確率 / 未偵測到 | **100% / 100% / 0** |
| | taiwan valid 76 張 | 完全正確率 / 字元準確率 / 未偵測到 | 100% / 100% / 0 |
| | taiwan test | **系統綜合效能分數** | **100.00** |

**Fine-tune 前後的端到端比較**（同一評估流程，只換權重；GPU，門檻 0.4）：

| 組合 | YOLO | PARSeq | test 完全正確率 | test 字元準確率 | valid 完全正確率 | valid 字元準確率 |
|------|------|--------|------|------|------|------|
| fine-tune 前 | 起點 | 官方預訓練 | 94.05% | 98.17% | 96.05% | 98.32% |
| 只 fine-tune YOLO | 最終 | 官方預訓練 | 90.48% | 98.78% | 90.79% | 98.82% |
| 只訓練 PARSeq | 起點 | 最終 | 98.81% | 98.93% | 98.68% | 98.82% |
| **兩者都用最終權重** | **最終** | **最終** | **100%** | **100%** | **100%** | **100%** |

- **PARSeq 的訓練是準確率提升的主要來源**：官方預訓練模型最常漏讀 `-`（例如 `ADY-8639 → ADY8639`），其次是形近字（`W` / `N`、`Q` / `0`）與少讀最後一個字。
- **YOLO fine-tune 修正漏抓**：起點 YOLO 對 `000633`（`258-QEN`）的最高分只有 0.299（低於門檻），fine-tune 後為 0.650。
- 只換 YOLO 時準確率反而略降：框變得更貼合車牌後，官方 PARSeq 漏讀 `-` 的情況變多；PARSeq 以同一個 YOLO 的裁切結果訓練、驗證後，所有錯誤都消失。這也說明兩個模型應該搭配使用（PARSeq 的驗證以實際使用的 YOLO 裁切）。
- 結果可重現：以相同設定重新 fine-tune YOLO，每個 epoch 的指標與原本完全相同（最終 mAP@0.5:0.95 0.783）。

### 6.5 訓練紀錄檔

| 檔案 | 說明 |
|------|------|
| `docs/training_logs/parseq_train_trackio.csv` | **PARSeq 訓練紀錄**（trackio 相容 CSV）：2,461 列，每個 step 的 `train/loss`、`train/lr`，每個 epoch 的 `epoch/loss`、`val/loss`、`val/acc`、`val/best` |
| `docs/training_logs/parseq_train_config.json` | 該次訓練的完整設定 |
| `docs/training_logs/yolo_finetune_trackio.csv` | YOLO fine-tune 紀錄（trackio 相容 CSV）：每個 epoch 的 loss、precision、recall、mAP、學習率 |
| `docs/training_logs/yolo_finetune_config.json` | 該次 fine-tune 的完整設定 |
| `docs/training_logs/experiments/` | 第 4 章 PARSeq 超參數實驗（lr 5e-6、lr 5e-4、freeze_encoder）的訓練紀錄與設定（trackio 相容 CSV） |
| `models/parseq/finetuned/train.log`、`models/yolo/finetuned/finetune_yolo.log` | 訓練程式的文字 log |

匯入 trackio：`trackio.import_csv("docs/training_logs/parseq_train_trackio.csv", project="ALPR-parseq")`，再執行 `trackio show --project ALPR-parseq`。欄位說明見 `docs/training_logs/README.md`。

---

## 7. 訓練與推論所需參考時間

量測環境：i7-14700F（28 執行緒）、32 GB RAM、RTX 4090 24 GB；2026-11-20 實測（訓練時間為整個指令從啟動到結束的牆鐘時間）。

### 7.1 訓練

| 模型 | 設定 | GPU（RTX 4090） | 僅 CPU（i7-14700F） |
|------|------|------------------|---------------------|
| YOLOv8n fine-tune | 1,400 張 × 50 epochs、batch 16 | **約 3–3.3 分鐘**（Ultralytics 計時 172–189 秒，每 epoch 約 3.5–3.8 秒；含載入與最終驗證約 197 秒） | 每 epoch 約 130 秒 → 50 epochs 約 1.8 小時 |
| PARSeq 訓練 | 5,204 張 × 30 epochs、batch 64 | **約 3.1 分鐘**（187–190 秒，每 epoch 約 6 秒） | 每 epoch 約 3.3 分鐘 → 30 epochs 約 1.7 小時 |
| PARSeq（`freeze_encoder: true`） | 同上 | 約 2.4 分鐘（141 秒） | — |

- 兩個模型都很小，GPU 訓練各約 3 分鐘；沒有 GPU 也能訓練，但時間長約 30 倍。
- CPU 的時間是用 1 個 epoch 實測後推算。

### 7.2 推論

每張影像（`taiwan_plate_dataset` test 84 張，416×416，每張 1 個車牌；暖機後；不含存 JSON）：

| 裝置 | YOLO 偵測 | PARSeq 辨識（1 個車牌） | 整條流程：平均 / P50 / P95 | 吞吐量 |
|------|-----------|-------------------------|-----------------------------|--------|
| GPU：RTX 4090 | 2.9 ms | 15.6 ms | **18.5** / 18.6 / 19.2 ms | 約 54 張 / 秒 |
| CPU：i7-14700F（Ultralytics 預設 8 執行緒） | 25.4 ms | 23.3 ms | **48.9** / 46.7 / 64.6 ms | 約 20 張 / 秒 |
| CPU：模擬 4 核心（3 執行緒） | 38.6 ms | 27.5 ms | **66.2** / 64.5 / 70.5 ms | 約 15 張 / 秒 |
| CPU：模擬 2 核心（1 執行緒） | 77.1 ms | 57.6 ms | **134.8** / 134.0 / 139.4 ms | 約 7 張 / 秒 |

- 模擬 N 核心：以 `taskset` 限制只能用 N 個實體核心，執行緒數設成 Ultralytics 在 N 核心機器上的預設值（N − 1）。
- GPU 上 PARSeq 比 YOLO 慢：自迴歸解碼一次產生一個字元，每一步都是很小的 GPU 運算，主要受啟動延遲限制，而不是運算量。

| 項目 | CPU | GPU |
|------|-----|-----|
| 模型載入（啟動到可推論） | 模型載入 1.3 秒；含 Python 啟動與套件匯入約 4–5 秒 | 模型載入 1.4 秒；含 Python 啟動與套件匯入約 4 秒 |
| 評估 test 84 張（`main.py --eval`，含啟動） | 8.5 秒 | 5.3 秒 |

- API Server 啟動時會自動暖機；第一個請求不用再等初始化。
- 經 API 呼叫時，另外加上傳與解碼影像的時間（與影像大小、網路有關）。
- 一張影像有多個車牌時，PARSeq 會把所有車牌組成一個 batch 一起辨識，耗時增加有限。

---

## 8. 最低軟硬體需求

依實測峰值（2026-11-20）加上餘裕給出建議值。

### 8.1 推論

| 項目 | 實測峰值 | **最低需求（CPU）** | **最低需求（GPU）** |
|------|----------|---------------------|---------------------|
| CPU | — | x86_64，**2 核心**（約 135 ms / 張）；建議 4 核心以上（約 66 ms / 張） | x86_64，2 核心 |
| 記憶體（RAM） | CPU 推論 1.0 GB；GPU 推論 1.6 GB | **2 GB** | **4 GB** |
| GPU | 顯示記憶體 0.67 GB | 不需要 | NVIDIA，**Turing（compute capability 7.5，例如 GTX 1650 / T4 / RTX 20 系列）以上**，顯示記憶體 **2 GB** 以上 |
| GPU 驅動 | — | — | 支援 CUDA 13.0 的 NVIDIA driver（**580 以上**）；Docker 需另裝 nvidia-container-toolkit |
| 磁碟 | 映像 8.47 GB（其中權重 203 MB）；權重檔合計 102 MB | **15 GB**（映像 8.5 GB；從壓縮檔載入時另需放壓縮檔的空間） | 同左 |
| 作業系統 | Ubuntu（Linux 6.8） | Linux x86_64 + Docker 20.10 以上；或 conda（Python 3.11） | 同左 |
| 網路 | — | 不需要（映像已內含所有權重） | 同左 |

### 8.2 訓練

| 項目 | YOLO fine-tune（batch 16） | PARSeq 訓練（batch 64） | **最低需求** |
|------|------|------|------|
| GPU 顯示記憶體峰值 | 2.8 GB | 3.75 GB（`freeze_encoder: true` 時 1.22 GB） | **GPU 6 GB 以上**（4 GB 可訓練 YOLO、或 PARSeq 加上 `freeze_encoder` / 調小 batch） |
| 記憶體（RAM）峰值 | 6.9 GB（含 4 個 DataLoader worker） | 2.3 GB | **8 GB**（16 GB 建議）；記憶體不足時調低 `workers` / `num_workers` |
| 共享記憶體（Docker） | DataLoader 多 worker 需要 | 同左 | `--shm-size 8g`（compose 已設定） |
| CPU | 4 個 DataLoader worker | 4 個 DataLoader worker | 4 核心以上 |
| 磁碟 | 資料集 86 MB | 資料集 70 MB；每次訓練輸出約 190 MB（best + last） | 映像 + 資料集 + 輸出約 **15 GB** |
| 沒有 GPU | 可以，50 epochs 約 1.8 小時；RAM 峰值 6.3 GB | 可以，30 epochs 約 1.7 小時；RAM 峰值 6.4 GB | 8 核心以上、RAM 8 GB 以上 |

### 8.3 軟體

| 套件 | 版本 | 說明 |
|------|------|------|
| Python | 3.11 | |
| PyTorch / torchvision | 2.11.0（CUDA 13.0 wheel）/ 0.26.0 | wheel 內含 CUDA runtime，主機只需要 driver |
| Ultralytics | 8.4.56 | YOLO 訓練 / 推論 |
| PARSeq | `baudm/parseq`（torch.hub，main branch） | 映像內已預先下載程式碼與預訓練權重；需要 timm 1.0、pytorch-lightning 2.6、nltk |
| FastAPI / Uvicorn | 0.136 / 0.47 | API Server |
| OpenCV / Pillow | 4.13 / 13.0 | 影像讀取 |
| trackio | 0.45 | 訓練紀錄（選用） |

完整清單見 `environment.yml`；Docker 映像已包含全部套件。

---

## 9. 使用手冊

### 9.1 執行環境

兩種方式擇一：

| 方式 | 內容 | 適合 |
|------|------|------|
| **Docker 映像 `alpr-bundle:1.0.0`（建議）** | 環境（conda ALPR、Python 3.11、PyTorch 2.11 + CUDA 13.0 runtime）、程式碼、推論權重、兩個訓練起點權重全部打包；不需要網路、不需要掛載任何檔案 | 部署、交付、離線環境 |
| conda 環境 | `conda env create -f environment.yml && conda activate ALPR`，權重放在 `models/` | 開發、訓練 |

- 作業系統：Linux x86_64（映像以 Debian 為基礎，linux/amd64）；其他作業系統需透過 Docker Desktop 執行（未測試）。
- GPU：選用。需要 NVIDIA driver（支援 CUDA 13.0，driver ≥ 580）＋ nvidia-container-toolkit。映像預設用 CPU，有 GPU 時以環境變數 `ALPR_DETECTOR__DEVICE=cuda`、`ALPR_RECOGNIZER__DEVICE=cuda` 切換。
- 映像的預設使用者是 UID 1000（非 root）。

**載入映像**（取得 `alpr-bundle_1.0.0.tar.gz` 時）：

```bash
docker load -i alpr-bundle_1.0.0.tar.gz        # 或 gunzip -c ... | docker load
docker image ls alpr-bundle                     # 確認有 alpr-bundle:1.0.0
```

**從原始碼重新建置**：

```bash
docker build -f docker/Dockerfile --target bundle -t alpr-bundle:1.0.0 .
```

### 9.2 使用方式

**A. REST API（Docker）**

```bash
TOKEN=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")   # 至少 16 個字元

# CPU
docker run -d --name alpr -p 8000:8000 -e ALPR_API__TOKENS="{\"web\": \"$TOKEN\"}" alpr-bundle:1.0.0
# GPU
docker run -d --name alpr --gpus all -p 8000:8000 -e ALPR_API__TOKENS="{\"web\": \"$TOKEN\"}" \
    -e ALPR_DETECTOR__DEVICE=cuda -e ALPR_RECOGNIZER__DEVICE=cuda alpr-bundle:1.0.0

docker ps                                    # STATUS 從 health: starting 變成 healthy 即可使用
curl -H "Authorization: Bearer $TOKEN" -F "file=@car.jpg" http://localhost:8000/predict
```

互動式 API 文件：<http://localhost:8000/docs>。辨識結果 JSON 存在容器的 `/app/outputs`，要保留到主機可加 `-v $PWD/outputs:/app/outputs`。

| 端點 | 說明 |
|------|------|
| `GET /health` | 健康檢查（不需 token） |
| `POST /predict` | 上傳一張影像（`multipart/form-data`，欄位 `file`） |
| `POST /predict/batch` | 上傳多張影像（欄位都叫 `files`，最多 20 張） |
| `GET /results/{id}` | 用 `job_id` / `batch_id` 查詢先前結果 |
| `GET /results?limit=50&offset=0` | 列出結果，新的在前 |

**B. 命令列（Docker 或 conda 環境）**

```bash
# 單張影像（結果 JSON 印到 stdout）
docker run --rm -v $PWD/images:/app/data alpr-bundle:1.0.0 python main.py --image data/car.jpg --no-save
python main.py --image car.jpg                      # conda 環境

# 整個資料夾（彙整成一份 JSON）
python main.py --folder images/ --output outputs/batch.json

# 用標準答案評估準確率
python main.py --eval data/taiwan_plate_dataset/test/test_gt.csv
```

**C. Python**

```python
from src.config import load_config
from src.pipeline import LPRPipeline

pipeline = LPRPipeline(load_config("config.yaml"))
result = pipeline.predict("car.jpg")            # 也可以傳 BGR numpy 陣列（cv2.imread 的結果）
best = result.best_plate()
print(best.plate_text if best else "未偵測到車牌", result.model_dump_json(indent=2))
```

**D. 重新訓練（Docker 內可離線執行，起點權重已在映像中）**

```bash
OPTS="--rm --gpus all --shm-size 8g --user $(id -u):$(id -g) -v $PWD/data:/app/data -v $PWD/models:/app/models"
docker run $OPTS alpr-bundle:1.0.0 python src/train/train.py            # PARSeq，設定見 config.yaml 的 train_parseq
docker run $OPTS alpr-bundle:1.0.0 python src/finetune/finetune_yolo.py # YOLO，設定見 config.yaml 的 finetune_yolo
```

- 掛載 `models/` 時，主機的資料夾會取代映像內的權重，訓練完成後新權重寫回主機的 `models/`；`--user` 讓寫出的檔案屬於主機使用者。
- 不掛載 `models/` 也能訓練（結果留在容器內）；起點權重都在映像中，`--network none` 也能執行（已實測）。
- 離線 fine-tune YOLO 時，Ultralytics 會警告無法下載 AMP 檢查用的模型並直接啟用 AMP，不影響訓練。

### 9.3 輸入格式規範

| 項目 | 規範 |
|------|------|
| 影像格式 | JPEG、PNG、BMP、TIFF、WebP（OpenCV 可解碼的格式） |
| 色彩 | 彩色或灰階皆可（內部轉成 3 通道）；Python 直接傳陣列時為 **BGR**、`uint8`、形狀 `(H, W, 3)` |
| 尺寸 | 不限（YOLO 內部縮放到 640）；已驗證的車牌大小為寬 151–416 px、高 67–297 px（416×416 影像），更小的車牌未驗證 |
| API 檔案大小 | 單一檔案 ≤ 10 MB（`api.max_upload_mb`） |
| API 像素上限 | 寬 × 高 ≤ 50,000,000（`api.max_image_pixels`） |
| API 批次張數 | ≤ 20（`api.max_batch_files`） |
| API 驗證 | Header `Authorization: Bearer <token>` |
| 評估用標準答案 | CSV，欄位 `image_path,plate_text`；`plate_text` 為大寫並含 `-`，例如 `ABC-1234` |

### 9.4 輸出格式規範

單張影像（`PredictionOutput`，JSON）：

```json
{
  "job_id": "3f1c2a9e-8d4b-4c41-9a51-6c2f0d7b1e20",
  "timestamp": "2026-11-20T12:00:00.000000+00:00",
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

| 欄位 | 型別 | 說明 |
|------|------|------|
| `job_id` | string（UUID） | 這次辨識的 ID |
| `timestamp` | string（ISO 8601，UTC） | 辨識時間 |
| `source_file` | string | 原始檔名 |
| `image_width` / `image_height` | int | 原圖尺寸（px）；讀圖失敗時為 `null` |
| `plates` | array | 偵測到的所有車牌，依 `confidence` 由高到低；沒有車牌時為空陣列 |
| `plates[].plate_text` | string | 車牌號碼：大寫英數字與 `-`，已做 `O→0`、`I→L` |
| `plates[].detector_score` | float 0–1 | YOLO 偵測信心分數（≥ 0.4） |
| `plates[].recognizer_score` | float 0–1 | PARSeq 各字元機率的平均 |
| `plates[].confidence` | float 0–1 | 綜合可信度 = `detector_score × recognizer_score` |
| `plates[].bbox` | `[x1, y1, x2, y2]` | 車牌在原圖上的位置（px，左上與右下角） |
| `processing_time_ms` | float | 推論耗時（讀圖到辨識完成） |
| `error` | string 或 `null` | 讀圖 / 解碼失敗時的錯誤訊息 |

- 批次（`BatchOutput`）：`{"batch_id", "timestamp", "total_images", "results": [PredictionOutput, ...]}`，`results` 順序與上傳順序相同。
- API 錯誤回應一律為 `{"detail": "<訊息>"}`：400 無法解碼、401 token 錯誤、411 缺 Content-Length、413 超過上傳限制、422 參數錯誤、503 模型未載入。
- 評估報告（`EvalReport`）：`total`、`correct`、`accuracy`、`char_accuracy`、`no_plate` 與每張影像的 `samples`。

### 9.5 主要設定（`config.yaml`，可用環境變數 `ALPR_<區塊>__<欄位>` 覆寫）

| 設定 | 預設 | 說明 |
|------|------|------|
| `detector.weights_path` | `models/yolo/finetuned/best.pt` | YOLO 權重 |
| `detector.confidence_threshold` | 0.4 | 偵測信心門檻 |
| `detector.device` / `recognizer.device` | `cpu` | `cpu`、`cuda` 或 `mps` |
| `recognizer.weights_path` | `models/parseq/finetuned/best.ckpt` | PARSeq 權重 |
| `recognizer.crop_padding` | 0.05 | 裁切外擴比例 |
| `api.tokens` | `{}` | API token（請用環境變數 `ALPR_API__TOKENS` 設定，不要寫進檔案） |

---

## 10. 限制與注意事項

| 項目 | 說明 |
|------|------|
| 測試集小 | test 只有 84 張，1 張約 1.2 個百分點；100% 的結果代表在這份資料上沒有錯誤，不代表所有情境都不會出錯 |
| 資料集 split 有重疊 | `taiwan_plate_dataset` 的 test 84 張中有 56 張與 valid 是同一個檔案（PARSeq 用 valid 選模型）；`finetune_dataset` 的 train / valid 有共用原圖（239 張），YOLO 驗證 mAP 可能偏樂觀。建議日後依原圖重新切分，並保留一份完全沒用過的測試集 |
| 拍攝條件 | 資料集多為白天、正面、清楚的車牌；夜間、逆光、大角度、遮擋、雙行車牌未經驗證 |
| 後處理規則 | `I → L` 是依「台灣車牌不使用 I」設定的規則；`I` 視覺上較像 `1`，規則是否合理需要更多資料評估 |
| 評估方式 | 每張影像只取可信度最高的車牌與答案比對；多車牌影像沒有完整評估 |
| 用途限制 | 辨識結果可能出錯，用於收費、執法、門禁等場合時應保留人工覆核，並遵守個人資料保護相關法規（車牌號碼可連結到個人） |
