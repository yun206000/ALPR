"""
訓練 PARSeq 車牌字元辨識模型（以官方預訓練權重為起點，在車牌資料上訓練）。

設定讀取 config.yaml 的 train_parseq 區塊，CLI 參數可臨時覆寫，例如：
    python src/train/train.py
    python src/train/train.py --epochs 1 --output-dir /tmp/debug

CSV 中 image_path 的解析規則見 src/utils/dataset.py。
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# 讓 `python src/train/train.py` 也能 import 專案的 src 套件
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.recognizer.transforms import get_transform  # noqa: E402
from src.config import TrainParseqConfig, add_config_args, apply_overrides, load_config  # noqa: E402
from src.utils import load_plate_csv, logger, setup_logger  # noqa: E402
from src.utils.tracking import ExperimentTracker  # noqa: E402

PARSEQ_IMG_SIZE = (32, 128)


class PlateDataset(Dataset):
    """(影像路徑, 車牌文字) 的資料集。"""

    def __init__(self, samples, img_size, augment: bool = False):
        self.samples = samples
        self.transform = get_transform(img_size, augment=augment)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        return self.transform(image), label


def write_split_csv(samples, path: Path) -> None:
    """把切分結果存成 CSV（image_path 為絕對路徑，可直接當 val_csv 再利用）。"""
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "plate_text"])
        writer.writerows((str(img), text) for img, text in samples)


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
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)  # inference forward (uses model.hparams max_label_length)
        probs = logits.softmax(-1)
        preds, _ = model.tokenizer.decode(probs)
        for p, gt in zip(preds, labels):
            total += 1
            if p.strip().upper() == gt.strip().upper():
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
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(app_cfg.log_level, log_file=cfg.output_dir / "train.log")
    logger.info(f"設定：{cfg.model_dump_json()}")

    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"使用裝置：{device}")

    # 1. Load model
    if cfg.finetune_from is not None:
        # Load checkpoint and resume fine-tuning
        logger.info(f"從 checkpoint 載入：{cfg.finetune_from}")
        ckpt = torch.load(cfg.finetune_from, map_location="cpu")
        model = torch.hub.load("baudm/parseq", "parseq", pretrained=False, trust_repo=True, verbose=False)
        state_dict = ckpt["model_state_dict"]
        model.load_state_dict(state_dict, strict=True)
        
    else:
        # Load pre-trained model from torch hub
        logger.info("從 torch.hub 載入 PARSeq 預訓練權重")
        model = torch.hub.load("baudm/parseq", "parseq", pretrained=True, trust_repo=True, verbose=False)

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
        # val_csv 為 null：從訓練集隨機切出 val_frac，切分結果存到 output_dir
        random.Random(cfg.seed).shuffle(train_samples)
        n_val = max(1, int(len(train_samples) * cfg.val_frac))
        val_samples, train_samples = train_samples[:n_val], train_samples[n_val:]
        write_split_csv(train_samples, cfg.output_dir / "_train_split.csv")
        write_split_csv(val_samples, cfg.output_dir / "_val_split.csv")
        logger.info(f"val_csv 為 null，從訓練集切出 {cfg.val_frac:.0%} 當驗證集")

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
    with ExperimentTracker(app_cfg.tracking, "parseq", cfg.model_dump(mode="json")) as tracker:
        logger.info("評估初始模型（epoch 0）...")
        val_loss_0 = evaluate_loss(model, val_loader, device)
        acc_0 = evaluate(model, val_loader, device)
        logger.info(f"Epoch 0/{cfg.epochs} | val_loss={val_loss_0:.4f} | val_acc={acc_0:.4f}")
        best_acc = acc_0
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
            acc = evaluate(model, val_loader, device)
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
            torch.save(ckpt, cfg.output_dir / "last.ckpt")
            if acc > best_acc:
                best_acc = acc
                torch.save(ckpt, cfg.output_dir / "best.ckpt")
                logger.info(f"新的最佳模型已儲存（val_acc={acc:.4f}）")
            tracker.log(
                {"train/epoch_loss": avg_loss, "val/loss": val_loss, "val/acc": acc,
                 "val/best_acc": best_acc, "epoch": epoch},
                step=step,
            )

        # 5. Plot loss curve
        loss_curve_path = cfg.output_dir / "loss_curve.png"
        draw_loss_curve(train_losses, val_losses, loss_curve_path)

        logger.info(f"訓練完成 | 最佳 val_acc={best_acc:.4f}")
        logger.info(f"權重已儲存：{cfg.output_dir}")


if __name__ == "__main__":
    main()