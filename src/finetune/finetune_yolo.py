"""
Fine-tune YOLO 車牌偵測模型（Ultralytics）。

設定讀取 config.yaml 的 finetune_yolo 區塊，CLI 參數可臨時覆寫，例如：
    python src/finetune/finetune_yolo.py
    python src/finetune/finetune_yolo.py --epochs 1 --batch-size 8
"""

import argparse
import random
import shutil
import sys
import tempfile
from pathlib import Path

from ultralytics import YOLO

# 讓 `python src/finetune/finetune_yolo.py` 也能 import 專案的 src 套件
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import FinetuneYoloConfig, add_config_args, apply_overrides, load_config  # noqa: E402
from src.utils import logger, setup_logger  # noqa: E402
from src.utils.files import replace_dir  # noqa: E402
from src.utils.tracking import ExperimentTracker, make_run_name  # noqa: E402

def add_tracking_callbacks(model: YOLO, tracker: ExperimentTracker) -> None:
    """把 Ultralytics 每個 epoch 的 loss / lr / 驗證指標寫進 tracker（step = epoch）。

    結束時 Ultralytics 會再驗證一次 best.pt，並在 step = epochs + 1 觸發
    on_fit_epoch_end，所以最後一個點就是最佳模型的指標。
    """

    def on_train_epoch_end(trainer) -> None:
        step = trainer.epoch + 1
        tracker.log(trainer.label_loss_items(trainer.tloss, prefix="train"), step=step)
        tracker.log(trainer.lr, step=step)

    def on_fit_epoch_end(trainer) -> None:
        tracker.log(trainer.metrics, step=trainer.epoch + 1)

    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
CLASS_NAMES = ["plate"]

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
        logger.warning(f"{split_dir}：{n_missing_label} 張圖片沒有對應的 label，已略過")
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


def prepare_dataset(train_dir: Path, val_dir, val_frac: float, work_dir: Path, seed: int):
    """回傳 (train_images_dir, val_images_dir) 給 data.yaml 用。

    - 如果有給 val_dir：直接用兩個原始目錄，不複製任何東西。
    - 如果沒有：從 train_dir 隨機切一份出來，用 symlink 放到
      work_dir/yolo_dataset/ 底下（原始資料不會被動到）。
    """
    train_pairs = list_labeled_images(train_dir)
    logger.info(f"Train 目錄：{len(train_pairs)} 張有標註的圖片")

    if val_dir is not None:
        val_pairs = list_labeled_images(val_dir)
        logger.info(f"Val 目錄：{len(val_pairs)} 張有標註的圖片")
        return train_dir / "images", val_dir / "images"

    # 沒有獨立的 val 目錄 -> 從 train 隨機切
    random.Random(seed).shuffle(train_pairs)  # NOSONAR S2245：固定 seed 切 train/val，需要可重現，與安全無關
    n_val = max(1, int(len(train_pairs) * val_frac))
    val_pairs, train_pairs = train_pairs[:n_val], train_pairs[n_val:]
    logger.info(f"自動切分 | train={len(train_pairs)} | val={len(val_pairs)} | val_frac={val_frac}")

    dataset_dir = work_dir / "yolo_dataset"
    link_pairs(train_pairs, dataset_dir / "images" / "train", dataset_dir / "labels" / "train")
    link_pairs(val_pairs, dataset_dir / "images" / "val", dataset_dir / "labels" / "val")
    return dataset_dir / "images" / "train", dataset_dir / "images" / "val"


def write_data_yaml(work_dir: Path, train_images_dir: Path, val_images_dir: Path):
    data_yaml = work_dir / "data.yaml"
    content = (
        f"train: {train_images_dir.resolve()}\n"
        f"val: {val_images_dir.resolve()}\n"
        f"names:\n"
        + "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASS_NAMES))
    )
    data_yaml.write_text(content, encoding="utf-8")
    return data_yaml


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap, FinetuneYoloConfig, "finetune_yolo")
    cli = ap.parse_args()
    app_cfg = load_config(cli.config)
    cfg = apply_overrides(app_cfg.finetune_yolo, cli)
    out_dir = cfg.output_dir

    train_dir = cfg.train_dir or (cfg.data_root / "train")
    val_dir = cfg.val_dir or (cfg.data_root / "valid" if (cfg.data_root / "valid").is_dir() else None)

    # 訓練過程全部在暫存資料夾進行（Ultralytics 需要的 data.yaml、自動切分的 symlink、訓練輸出），
    # 成功跑完才整份換進 out_dir；中途失敗或中斷的話，out_dir 裡原本的權重不會被動到。
    with tempfile.TemporaryDirectory(prefix="alpr_yolo_") as tmp:
        work_dir = Path(tmp)
        run_dir = work_dir / "run"
        run_dir.mkdir()
        run_name = make_run_name("yolo")  # trackio 的 run 名稱，用來區分每次 fine-tune
        setup_logger(app_cfg.log_level, log_file=run_dir / "finetune_yolo.log")
        logger.info(f"trackio run：{run_name} | 完成後會取代 {out_dir}/ 的內容")
        logger.info(f"設定：{cfg.model_dump_json()}")

        # 1) 確認資料結構 + 視需要自動切 val
        train_images_dir, val_images_dir = prepare_dataset(train_dir, val_dir, cfg.val_frac, work_dir, cfg.seed)
        data_yaml = write_data_yaml(work_dir, train_images_dir, val_images_dir)

        # 2) 載入權重
        weights = str(cfg.finetune_from) if cfg.finetune_from is not None else cfg.model
        logger.info(f"載入 YOLO 權重：{weights}")
        model = YOLO(weights)

        # 3) Fine-tune
        # ultralytics 會把「相對路徑」的 project 放到 runs/<task>/ 底下，
        # 所以這裡一律傳絕對路徑，輸出才會直接落在 run_dir，不會產生 runs/ 資料夾。
        run_dir = run_dir.resolve()
        train_kwargs = dict(
            data=str(data_yaml),
            imgsz=cfg.imgsz,
            epochs=cfg.epochs,
            batch=cfg.batch_size,
            optimizer=cfg.optimizer,  # 一定要明確指定，預設的 auto 會忽略 lr0
            lr0=cfg.lr,
            patience=cfg.patience,
            workers=cfg.workers,
            seed=cfg.seed,
            project=str(run_dir.parent),
            name=run_dir.name,
            exist_ok=True,  # run_dir 已經建好（放 log），直接寫進去
            # 不傳 pretrained：Ultralytics 預設會沿用上面 YOLO(weights) 載入的權重。
            # 傳 pretrained=False 會丟掉載入的權重、從隨機初始化開始訓練。
            plots=False,
        )
        if cfg.freeze > 0:
            train_kwargs["freeze"] = cfg.freeze
            logger.info(f"凍結前 {cfg.freeze} 層，只訓練其餘部分")
        if cfg.device is not None:
            train_kwargs["device"] = cfg.device

        logger.info(f"開始 fine-tune | epochs={cfg.epochs} | batch={cfg.batch_size} | optimizer={cfg.optimizer} | lr={cfg.lr}")
        with ExperimentTracker(app_cfg.tracking, run_name, cfg.model_dump(mode="json")) as tracker:
            add_tracking_callbacks(model, tracker)
            model.train(**train_kwargs)

        # 4) 最終在 val set 上評估（mAP50 / mAP50-95），指標寫進 log
        # Ultralytics 一定會建輸出資料夾，放在暫存資料夾就不會留在結果裡（也不會跑到 runs/detect/val）
        metrics = model.val(
            data=str(data_yaml),
            imgsz=cfg.imgsz,
            project=str(work_dir),
            name="val",
            exist_ok=True,
            plots=False,
        )
        logger.info(f"驗證集結果 | mAP50={metrics.box.map50:.4f} | mAP50-95={metrics.box.map:.4f}")

        # 5) weights/*.pt 移到最上層（out_dir/best.pt），再整份換進 out_dir；
        #    之後的 log 改寫到 out_dir 裡的 log 檔（接在後面）
        for f in (run_dir / "weights").iterdir():
            f.rename(run_dir / f.name)
        (run_dir / "weights").rmdir()
        replace_dir(run_dir, out_dir)
        setup_logger(app_cfg.log_level, log_file=out_dir / "finetune_yolo.log")

    best = out_dir / "best.pt"
    logger.info(f"Fine-tune 完成 | 已取代 {best}")
    logger.info(f"每個 epoch 的 loss / mAP：{out_dir / 'results.csv'}")
    if Path(app_cfg.detector.weights_path).resolve() == best.resolve():
        logger.info("detector.weights_path 已經指向這個檔案；API Server 重新啟動後才會載入新權重")
    else:
        logger.info(f"要用這個權重推論，請把 config.yaml 的 detector.weights_path 改成 {best}")


if __name__ == "__main__":
    main()