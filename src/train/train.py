"""
訓練 PARSeq 車牌字元辨識模型（以官方預訓練權重為起點，在車牌資料上訓練）。

設定讀取 config.yaml 的 train_parseq 區塊，CLI 參數可臨時覆寫，例如：
    python src/train/train.py
    python src/train/train.py --epochs 1 --output-dir /tmp/debug

CSV 中 image_path 的解析規則見 src/utils/dataset.py。
"""

import argparse
import random
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# 讓 `python src/train/train.py` 也能 import 專案的 src 套件
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.recognizer.parseq_recognizer import PARSEQ_HUB_REPO, normalize_plate_text  # noqa: E402
from src.recognizer.transforms import get_transform  # noqa: E402
from src.config import TrainParseqConfig, add_config_args, apply_overrides, load_config  # noqa: E402
from src.utils import crop_bbox, load_plate_csv, logger, setup_logger  # noqa: E402
from src.utils.files import replace_dir  # noqa: E402
from src.utils.tracking import ExperimentTracker, make_run_name  # noqa: E402

PARSEQ_IMG_SIZE = (32, 128)


class PlateDataset(Dataset):
    """(影像, 車牌文字) 的資料集；影像可以是檔案路徑，或已經裁切好的 PIL 影像。"""

    def __init__(self, samples, img_size, augment: bool = False):
        self.samples = samples
        self.transform = get_transform(img_size, augment=augment)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src, label = self.samples[idx]
        image = src if isinstance(src, Image.Image) else Image.open(src).convert("RGB")
        return self.transform(image), label


def crop_plates(samples, app_cfg, device: torch.device):
    """用 YOLO 偵測車牌並裁切，回傳 [(裁切後的 PIL 影像, 車牌文字), ...]。

    偵測器、信心門檻、crop_padding 都和推論 pipeline 相同，PARSeq 驗證時看到的輸入才會和實際推論一致。
    偵測不到車牌的影像保留整張（仍然計入驗證）。
    """
    from src.detector import YOLODetector

    detector = YOLODetector(app_cfg.detector.model_copy(update={"device": device.type}))
    cropped, n_missed = [], 0
    for img_path, label in samples:
        rgb = np.asarray(Image.open(img_path).convert("RGB"))
        detections = detector.detect(rgb)
        if detections:
            b = detections[0].bbox  # 分數最高的框
            rgb = crop_bbox(rgb, b.x1, b.y1, b.x2, b.y2, padding=app_cfg.recognizer.crop_padding)
        else:
            n_missed += 1
        cropped.append((Image.fromarray(np.ascontiguousarray(rgb)), label))

    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()
    logger.info(f"驗證影像已用 YOLO 裁出車牌：{len(samples) - n_missed}/{len(samples)} 張")
    if n_missed:
        logger.warning(f"{n_missed} 張驗證影像偵測不到車牌，以整張影像驗證")
    return cropped


def collate_fn(batch):
    images, labels = zip(*batch)
    return torch.stack(images, 0), list(labels)


def compute_loss(model, images, labels):
    tgt = model.tokenizer.encode(labels, images.device)

    memory = model.model.encode(images)

    tgt_perms = model.gen_tgt_perms(tgt)
    tgt_in = tgt[:, :-1]
    tgt_out = tgt[:, 1:]
    tgt_padding_mask = (tgt_in == model.pad_id) | (tgt_in == model.eos_id)

    loss = 0.0
    loss_numel = 0
    n = (tgt_out != model.pad_id).sum().item()
    for i, perm in enumerate(tgt_perms):
        tgt_mask, query_mask = model.generate_attn_masks(perm)
        out = model.model.decode(tgt_in, memory, tgt_mask, tgt_padding_mask, tgt_query_mask=query_mask)
        logits = model.model.head(out).flatten(end_dim=1)
        loss += n * F.cross_entropy(logits, tgt_out.flatten(), ignore_index=model.pad_id)
        loss_numel += n
        if i == 1:
            tgt_out = torch.where(tgt_out == model.eos_id, model.pad_id, tgt_out)
            n = (tgt_out != model.pad_id).sum().item()
    return loss / loss_numel


@torch.no_grad()
def evaluate(model, loader, device, charset: str):
    """完全正確率：模型輸出經過和推論相同的後處理（normalize_plate_text）後，與 GT 完全相同才算對。"""
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)  # inference forward (uses model.hparams max_label_length)
        probs = logits.softmax(-1)
        preds, _ = model.tokenizer.decode(probs)
        for p, gt in zip(preds, labels):
            total += 1
            if normalize_plate_text(p, charset) == gt.strip():
                correct += 1
    model.train()
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_loss(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for images, labels in loader:
        images = images.to(device)
        loss = compute_loss(model, images, labels)
        total_loss += loss.item()
        total_batches += 1
    model.train()
    return total_loss / max(total_batches, 1)


def draw_loss_curve(train_losses, val_losses, output_path: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(train_losses) + 1), train_losses, marker="o", label="train loss")
    ax.plot(range(1, len(val_losses) + 1), val_losses, marker="o", label="val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info(f"Loss 曲線已儲存：{output_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap, TrainParseqConfig, "train_parseq")
    cli = ap.parse_args()
    app_cfg = load_config(cli.config)
    cfg = apply_overrides(app_cfg.train_parseq, cli)

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    out_dir = cfg.output_dir

    # 訓練過程都寫在暫存資料夾，成功跑完才整份換進 out_dir；
    # 中途失敗或中斷的話，out_dir 裡原本的權重不會被動到（暫存資料夾在程式結束時自動刪除）
    staging = tempfile.TemporaryDirectory(prefix="alpr_parseq_")
    run_dir = Path(staging.name)
    run_name = make_run_name("parseq")  # trackio 的 run 名稱，用來區分每次訓練
    setup_logger(app_cfg.log_level, log_file=run_dir / "train.log")
    logger.info(f"trackio run：{run_name} | 完成後會取代 {out_dir}/ 的內容")
    logger.info(f"設定：{cfg.model_dump_json()}")

    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"使用裝置：{device}")

    # 1. Load model
    if cfg.finetune_from is not None:
        # Load checkpoint and resume fine-tuning
        logger.info(f"從 checkpoint 載入：{cfg.finetune_from}")
        # weights_only=True：只還原 tensor / 基本型別，不執行 checkpoint 裡的任意 pickle 程式碼
        ckpt = torch.load(cfg.finetune_from, map_location="cpu", weights_only=True)
        model = torch.hub.load(PARSEQ_HUB_REPO, "parseq", pretrained=False, trust_repo=True, verbose=False)
        state_dict = ckpt["model_state_dict"]
        model.load_state_dict(state_dict, strict=True)
        
    else:
        # Load pre-trained model from torch hub
        logger.info("從 torch.hub 載入 PARSeq 預訓練權重")
        model = torch.hub.load(PARSEQ_HUB_REPO, "parseq", pretrained=True, trust_repo=True, verbose=False)

    model = model.to(device)
    model.train()

    if cfg.freeze_encoder:
        for p in model.model.encoder.parameters():
            p.requires_grad = False
        logger.info("已凍結 encoder，只訓練 decoder + head")

    # 2. Datasets
    train_samples = load_plate_csv(cfg.train_csv)
    if cfg.val_csv is not None:
        val_samples = load_plate_csv(cfg.val_csv)
    else:
        # val_csv 為 null：從訓練集隨機切出 val_frac（seed 固定，每次切法相同）
        random.Random(cfg.seed).shuffle(train_samples)  # NOSONAR S2245：固定 seed 切 train/val，需要可重現，與安全無關
        n_val = max(1, int(len(train_samples) * cfg.val_frac))
        val_samples, train_samples = train_samples[:n_val], train_samples[n_val:]
        logger.info(f"val_csv 為 null，從訓練集切出 {cfg.val_frac:.0%} 當驗證集")

    # 訓練集是裁好的車牌，驗證集是整張影像：先用 YOLO 裁出車牌，驗證準確率才反映實際推論
    if cfg.crop_val and cfg.val_csv is not None:
        val_samples = crop_plates(val_samples, app_cfg, device)
    elif cfg.crop_val:
        logger.info("驗證集取自訓練集（已是裁切好的車牌），不再用 YOLO 裁切")

    train_ds = PlateDataset(train_samples, PARSEQ_IMG_SIZE, augment=True)
    val_ds = PlateDataset(val_samples, PARSEQ_IMG_SIZE, augment=False)

    logger.info(f"資料筆數 | train={len(train_ds)} | val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                               num_workers=cfg.num_workers, collate_fn=collate_fn, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, collate_fn=collate_fn)

    # 3. Optimizer / Scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # 4. Training loop（指標紀錄到 trackio）
    with ExperimentTracker(app_cfg.tracking, run_name, cfg.model_dump(mode="json")) as tracker:
        logger.info("評估初始模型（epoch 0）...")
        val_loss_0 = evaluate_loss(model, val_loader, device)
        acc_0 = evaluate(model, val_loader, device, app_cfg.recognizer.charset)
        logger.info(f"Epoch 0/{cfg.epochs} | val_loss={val_loss_0:.4f} | val_acc={acc_0:.4f}")
        best_acc, best_loss = acc_0, val_loss_0
        step = 0
        train_losses = []
        val_losses = []
        tracker.log({"val/loss": val_loss_0, "val/acc": acc_0, "epoch": 0}, step=0)
        for epoch in range(1, cfg.epochs + 1):
            running_loss = 0.0
            for images, labels in train_loader:
                images = images.to(device)
                optimizer.zero_grad()
                loss = compute_loss(model, images, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
                lr = optimizer.param_groups[0]["lr"]  # 這一步實際使用的 learning rate
                optimizer.step()
                scheduler.step()

                running_loss += loss.item()
                step += 1
                tracker.log({"train/loss": loss.item(), "train/lr": lr}, step=step)
                if step % 20 == 0:
                    logger.info(f"Epoch {epoch}/{cfg.epochs} | step {step} | loss={loss.item():.4f}")

            avg_loss = running_loss / len(train_loader)
            val_loss = evaluate_loss(model, val_loader, device)
            acc = evaluate(model, val_loader, device, app_cfg.recognizer.charset)
            train_losses.append(avg_loss)
            val_losses.append(val_loss)
            logger.info(
                f"Epoch {epoch}/{cfg.epochs} | train_loss={avg_loss:.4f} | "
                f"val_loss={val_loss:.4f} | val_acc={acc:.4f}"
            )

            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "hyper_parameters": dict(model.hparams),
                "val_acc": acc,
            }
            torch.save(ckpt, run_dir / "last.ckpt")
            # 準確率較高，或準確率相同但 val_loss 較低，才算更好
            # （驗證集小，準確率很快就持平；只看準確率會一直停在第一個達到該準確率的 epoch）
            if acc > best_acc or (acc == best_acc and val_loss < best_loss):
                best_acc, best_loss = acc, val_loss
                torch.save(ckpt, run_dir / "best.ckpt")
                logger.info(f"新的最佳模型已儲存（val_acc={acc:.4f} | val_loss={val_loss:.4f}）")
            tracker.log(
                {"train/epoch_loss": avg_loss, "val/loss": val_loss, "val/acc": acc,
                 "val/best_acc": best_acc, "epoch": epoch},
                step=step,
            )

        # 5. Plot loss curve
        loss_curve_path = run_dir / "loss_curve.png"
        draw_loss_curve(train_losses, val_losses, loss_curve_path)

        logger.info(f"訓練完成 | 最佳 val_acc={best_acc:.4f} | val_loss={best_loss:.4f}")

    # 6. 換進 out_dir
    # best.ckpt 只在比訓練前（epoch 0）更好時才會存；一次都沒有代表這次訓練沒有變好，
    # 取代的話 out_dir 會沒有 best.ckpt、推論讀不到權重，所以保留原本的內容
    if not (run_dir / "best.ckpt").exists():
        logger.warning(
            f"沒有任何 epoch 比訓練前（val_acc={acc_0:.4f} | val_loss={val_loss_0:.4f}）更好，沒有產生 best.ckpt；"
            f"保留 {out_dir}/ 原本的權重，不取代"
        )
        staging.cleanup()
        return

    replace_dir(run_dir, out_dir)
    setup_logger(app_cfg.log_level, log_file=out_dir / "train.log")  # 之後的 log 接在 out_dir 的 log 檔後面
    staging.cleanup()

    best = out_dir / "best.ckpt"
    logger.info(f"已取代 {best}")
    if Path(app_cfg.recognizer.weights_path).resolve() == best.resolve():
        logger.info("recognizer.weights_path 已經指向這個檔案；API Server 重新啟動後才會載入新權重")
    else:
        logger.info(f"要用這個權重推論，請把 config.yaml 的 recognizer.weights_path 改成 {best}")


if __name__ == "__main__":
    main()