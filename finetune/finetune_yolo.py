"""
Fine-tune YOLOv8 (ultralytics) on Taiwan license plate DETECTION data.

跟你原本的 PARSeq fine-tune 腳本對應，但這裡是「車牌偵測」而不是「文字辨識」。

**資料已經是標準 YOLO 格式，不需要任何轉換**：
    每張圖對應一個 .txt，內容像這樣（class x_center y_center width height，
    座標皆已標準化到 0~1）：

        0 0.48515625 0.484375 0.865625 0.65625

    （附註：如果你要 fine-tune 的舊 checkpoint 在「推論」時輸出
    bbox=[x1,y1,x2,y2] 絕對像素，那只是 ultralytics 的
    Results.boxes.xyxy 輸出格式，跟訓練用的 label 檔案無關，不用轉換。）

Expected data layout (相對於 --data-root，跟 PARSeq 腳本的 train/valid 結構一致):
    data/taiwan_plate_det/
        train/
            images/*.jpg
            labels/*.txt      # 檔名要跟 images 對得起來，例如 000001.jpg <-> 000001.txt
        valid/                # 可選，沒有的話用 --val-frac 從 train 隨機切
            images/*.jpg
            labels/*.txt

這支腳本會：
    1. 確認 train / val 的 images+labels 目錄結構正確
    2. 如果沒有給 valid 目錄，就用 --val-frac 從 train 隨機切一份出來
       （用 symlink 放到 <output-dir>/yolo_dataset/ 底下，不動到原始資料）
    3. 產生 data.yaml
    4. 用 ultralytics YOLO API fine-tune（可從官方預訓練權重，或你自己的
       checkpoint 續練）

Usage:
    pip install ultralytics

    # 資料已經有 train/ + valid/ 兩個目錄的情況
    python finetune/finetune_yolov8.py \
        --data-root ../datasets/ \
        --train-dir ../datasets/train \
        --val-dir ../datasets/valid \
        --epochs 50 --batch-size 16 --lr 5e-4 \
        --model yolov8n.pt \
        --output-dir checkpoints_det

    # 只有 train/，自動切 10% 當 val
    python finetune/finetune_yolov8.py \
        --data-root ../datasets/ \
        --train-dir ../datasets/train \
        --val-frac 0.1 \
        --finetune-from checkpoints_det/train/weights/best.pt \
        --output-dir checkpoints_det_v2
"""

import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
CLASS_NAMES = ["plate"]  # 只有一類：車牌


def _lazy_import_ultralytics():
    from ultralytics import YOLO
    return YOLO


# --------------------------------------------------------------------------- #
# 資料檢查 / 切分
# --------------------------------------------------------------------------- #
def list_labeled_images(split_dir: Path):
    """回傳 (image_path, label_path) pair 的 list，只保留兩邊都存在的檔案。"""
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    assert images_dir.is_dir(), f"找不到 {images_dir}"
    assert labels_dir.is_dir(), f"找不到 {labels_dir}"

    pairs = []
    n_missing_label = 0
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            n_missing_label += 1
            continue
        pairs.append((img_path, label_path))

    if n_missing_label:
        print(f"[warn] {split_dir}: {n_missing_label} 張圖片沒有對應的 label，已略過")
    return pairs


def link_pairs(pairs, images_out: Path, labels_out: Path):
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    for img_path, label_path in pairs:
        dst_img = images_out / img_path.name
        dst_label = labels_out / label_path.name
        if not dst_img.exists():
            try:
                dst_img.symlink_to(img_path.resolve())
            except OSError:
                shutil.copy2(img_path, dst_img)
        if not dst_label.exists():
            try:
                dst_label.symlink_to(label_path.resolve())
            except OSError:
                shutil.copy2(label_path, dst_label)


def prepare_dataset(train_dir: Path, val_dir, val_frac: float, output_dir: Path, seed: int):
    """回傳 (train_images_dir, val_images_dir) 給 data.yaml 用。

    - 如果有給 val_dir：直接用兩個原始目錄，不複製任何東西。
    - 如果沒有：從 train_dir 隨機切一份出來，用 symlink 放到
      output_dir/yolo_dataset/ 底下（原始資料不會被動到）。
    """
    train_pairs = list_labeled_images(train_dir)
    print(f"Train dir: {len(train_pairs)} labeled images")

    if val_dir is not None:
        val_pairs = list_labeled_images(val_dir)
        print(f"Val dir: {len(val_pairs)} labeled images")
        return train_dir / "images", val_dir / "images"

    # 沒有獨立的 val 目錄 -> 從 train 隨機切
    random.Random(seed).shuffle(train_pairs)
    n_val = max(1, int(len(train_pairs) * val_frac))
    val_pairs, train_pairs = train_pairs[:n_val], train_pairs[n_val:]
    print(f"Auto split: train={len(train_pairs)}  val={len(val_pairs)} (val_frac={val_frac})")

    dataset_dir = output_dir / "yolo_dataset"
    link_pairs(train_pairs, dataset_dir / "images" / "train", dataset_dir / "labels" / "train")
    link_pairs(val_pairs, dataset_dir / "images" / "val", dataset_dir / "labels" / "val")
    return dataset_dir / "images" / "train", dataset_dir / "images" / "val"


def write_data_yaml(output_dir: Path, train_images_dir: Path, val_images_dir: Path):
    data_yaml = output_dir / "data.yaml"
    content = (
        f"train: {train_images_dir.resolve()}\n"
        f"val: {val_images_dir.resolve()}\n"
        f"names:\n"
        + "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASS_NAMES))
    )
    data_yaml.write_text(content, encoding="utf-8")
    return data_yaml


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data/taiwan_plate_det"),
                     help="只用來當預設 train/val 目錄的基準，實際路徑看 --train-dir/--val-dir")
    ap.add_argument("--train-dir", type=Path, default=None,
                     help="包含 images/ 和 labels/ 的訓練目錄 (預設: <data-root>/train)")
    ap.add_argument("--val-dir", type=Path, default=None,
                     help="包含 images/ 和 labels/ 的驗證目錄；不給的話用 --val-frac 從 train 自動切")
    ap.add_argument("--val-frac", type=float, default=0.1)

    ap.add_argument("--model", type=str, default="yolov8n.pt",
                     help="ultralytics 預訓練權重名稱 (yolov8n/s/m/l/x.pt) 或本地 .pt 路徑")
    ap.add_argument("--finetune-from", type=Path, default=None,
                     help="你自己訓練好的 checkpoint (.pt)，會覆蓋 --model 當作起始權重")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4, help="對應 ultralytics 的 lr0")
    ap.add_argument("--freeze", type=int, default=0,
                     help="凍結前 N 層 backbone（0 = 不凍結，全部 fine-tune）")
    ap.add_argument("--patience", type=int, default=20, help="early stopping patience (epochs)")
    ap.add_argument("--device", type=str, default=None, help='例如 "0" 或 "cpu"，留空自動選擇')
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=Path, default=Path("checkpoints_det"))
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_dir = args.train_dir or (args.data_root / "train")
    val_dir = args.val_dir or (args.data_root / "valid" if (args.data_root / "valid").is_dir() else None)

    # 1) 確認資料結構 + 視需要自動切 val
    train_images_dir, val_images_dir = prepare_dataset(
        train_dir, val_dir, args.val_frac, args.output_dir, args.seed
    )
    data_yaml = write_data_yaml(args.output_dir, train_images_dir, val_images_dir)
    print(f"data.yaml written to: {data_yaml}")

    # 2) 載入模型（官方預訓練權重，或你自己的 checkpoint）
    YOLO = _lazy_import_ultralytics()
    weights = str(args.finetune_from) if args.finetune_from is not None else args.model
    print(f"Loading YOLO weights: {weights}")
    model = YOLO(weights)

    # 3) fine-tune
    # ultralytics 自己處理 optimizer / scheduler / augmentation / checkpoint 儲存，
    # 訓練過程與結果 (results.csv, results.png, weights/best.pt, weights/last.pt)
    # 都會存在 <output-dir>/train/ 底下
    train_kwargs = dict(
        data=str(data_yaml),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch_size,
        lr0=args.lr,
        patience=args.patience,
        workers=args.workers,
        seed=args.seed,
        project=str(args.output_dir),
        name="train",
        exist_ok=True,
        pretrained=args.finetune_from is None,  # 從自己的 checkpoint 續練時不要再套用官方預訓練
    )
    if args.freeze > 0:
        train_kwargs["freeze"] = args.freeze
        print(f"Freezing first {args.freeze} layers; fine-tuning the rest.")
    if args.device is not None:
        train_kwargs["device"] = args.device

    model.train(**train_kwargs)

    # 4) 最終在 val set 上評估（mAP50 / mAP50-95）
    metrics = model.val(data=str(data_yaml), imgsz=args.imgsz)
    print("Final validation metrics:")
    print(f"  mAP50    : {metrics.box.map50:.4f}")
    print(f"  mAP50-95 : {metrics.box.map:.4f}")

    run_dir = args.output_dir / "train"
    print(f"Done. Checkpoints saved to: {run_dir / 'weights'}")
    print(f"  best.pt -> {run_dir / 'weights' / 'best.pt'}")
    print(f"  last.pt -> {run_dir / 'weights' / 'last.pt'}")
    print(f"  loss / metric curves -> {run_dir / 'results.png'}")


if __name__ == "__main__":
    main()