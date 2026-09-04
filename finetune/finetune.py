"""
Fine-tune PARSeq.

Usage:
    python finetune/finetune.py --data-root ../datasets/ --train-csv ../datasets/train/train_gt.csv --val-csv ../datasets/valid/valid_gt.csv --epochs 10 --batch-size 16 --lr 5e-5 --output-dir checkpoints
"""

import argparse
import random
from pathlib import Path
from xml.parsers.expat import model

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


def resolve_image_root(data_root: Path, csv_path: Path) -> Path:
    """Resolve the image base folder from the CSV path."""
    data_root = Path(data_root)
    csv_name = str(csv_path).lower()
    if "valid" in csv_name or "val" in csv_name:
        return data_root / "valid" / "images"
    return data_root / "train" / "images"


class PlateDataset(Dataset):
    """Reads (image_path, plate_text) pairs from a CSV."""

    def __init__(self, data_root: Path, csv_path: Path, img_size, augment: bool = False):
        self.data_root = Path(data_root)
        self.df = pd.read_csv(csv_path)
        assert {"image_path", "plate_text"}.issubset(self.df.columns), \
            f"CSV must have columns image_path, plate_text — got {list(self.df.columns)}"
        self.image_root = resolve_image_root(self.data_root, csv_path)
        self.transform = get_transform(img_size, augment=augment)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_name = Path(row["image_path"])
        if image_name.is_absolute():
            img_path = image_name
        else:
            img_path = self.image_root / image_name
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = str(row["plate_text"]).strip()
        return image, label


def get_transform(img_size, augment: bool = False):
    """Matches strhub.data.module.SceneTextDataModule.get_transform."""
    ops = []

    if augment:
        ops += [
            T.RandomApply(
                [
                    T.RandomPerspective(
                        distortion_scale=0.12,
                        p=1.0,
                        fill=127,
                    )
                ],
                p=0.25,
            ),
            T.RandomApply(
                [
                    T.RandomAffine(
                        degrees=3,
                        translate=(0.025, 0.05),
                        scale=(0.90, 1.10),
                        shear=(-2, 2),
                        fill=127,
                    )
                ],
                p=0.50,
            ),
            T.RandomApply(
                [T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))],
                p=0.25,
            ),
            T.RandomApply(
                [
                    T.ColorJitter(
                        brightness=0.30,
                        contrast=0.35,
                        saturation=0.10,
                        hue=0.02,
                    )
                ],
                p=0.60,
            ),
            T.RandomAutocontrast(p=0.15),
        ]

    ops += [
        T.Resize(img_size, T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(0.5, 0.5),
    ]

    return T.Compose(ops)


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
    print(f"Loss curve saved to: {output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data/taiwan_plate"),
                     help="Root folder that image_path in the CSV is relative to")
    ap.add_argument("--train-csv", type=Path, default=Path("data/taiwan_plate/train/train_gt.csv"),
                     help="The ground truth CSV of training set with columns {image_path,plate_text}")
    ap.add_argument("--val-csv", type=Path, default=None,
                     help="Optional separate validation CSV (same format). "
                          "If omitted, a slice of --train-csv is held out.")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--freeze-encoder", action="store_true",
                     help="Freeze the ViT encoder and only fine-tune the decoder/head")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    ap.add_argument("--finetune-from", type=Path, default=None,
                     help="Optional checkpoint to resume fine-tuning from")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load model
    if args.finetune_from is not None:
        # Load checkpoint and resume fine-tuning
        print(f"Loading checkpoint from {args.finetune_from}")
        ckpt = torch.load(args.finetune_from, map_location="cpu")
        model = torch.hub.load("baudm/parseq", "parseq", pretrained=False, trust_repo=True)
        state_dict = ckpt["model_state_dict"]
        model.load_state_dict(state_dict, strict=True)
        
    else:
        # Load pre-trained model from torch hub
        model = torch.hub.load("baudm/parseq", "parseq", pretrained=True, trust_repo=True)

    model = model.to(device)
    model.train()

    img_size = [32, 128] 

    if args.freeze_encoder:
        for p in model.model.encoder.parameters():
            p.requires_grad = False
        print("Encoder frozen; fine-tuning decoder + head only.")

    # 2. Datasets
    if args.val_csv is not None:
        # train_df = pd.read_csv(args.train_csv)
        # train_df.to_csv(args.output_dir / "_train_split.csv", index=False)
        train_ds = PlateDataset(args.data_root, args.train_csv, img_size, augment=True)
        val_ds = PlateDataset(args.data_root, args.val_csv, img_size, augment=False)
    else:
        # If there's no existing validation CSV, split the training CSV into train/val and save them to `output_dir`.
        full_df = pd.read_csv(args.train_csv).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        n_val = max(1, int(len(full_df) * args.val_frac))
        val_df, train_df = full_df.iloc[:n_val], full_df.iloc[n_val:]
        train_csv_path = args.output_dir / "_train_split.csv"
        val_csv_path = args.output_dir / "_val_split.csv"
        train_df.to_csv(train_csv_path, index=False)
        val_df.to_csv(val_csv_path, index=False)
        train_ds = PlateDataset(args.data_root, train_csv_path, img_size, augment=True)
        val_ds = PlateDataset(args.data_root, val_csv_path, img_size, augment=False)

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, collate_fn=collate_fn, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate_fn)

    # 3. Optimizer / Scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # 4. Training loop
    print("Evaluating initial baseline (Epoch 0)...")
    val_loss_0 = evaluate_loss(model, val_loader, device)
    acc_0 = evaluate(model, val_loader, device)
    print(f"== Epoch 0 (Baseline) val_loss={val_loss_0:.4f} val_exact_match={acc_0:.4f} ==")
    best_acc = acc_0
    step = 0
    train_losses = []
    val_losses = []
    val_accs = []
    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            loss = compute_loss(model, images, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            step += 1
            if step % 20 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)
        val_loss = evaluate_loss(model, val_loader, device)
        acc = evaluate(model, val_loader, device)
        train_losses.append(avg_loss)
        val_losses.append(val_loss)
        val_accs.append(acc)
        print(f"== epoch {epoch}/{args.epochs} train_loss={avg_loss:.4f} val_loss={val_loss:.4f} val_exact_match={acc:.4f} ==")

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
            "val_acc": acc,
        }
        torch.save(ckpt, args.output_dir / "last.ckpt")
        if acc > best_acc:
            best_acc = acc
            torch.save(ckpt, args.output_dir / "best.ckpt")
            print(f"  -> new best model saved (val_exact_match={acc:.4f})")

    # 5. Plot loss curve
    loss_curve_path = args.output_dir / "loss_curve.png"
    draw_loss_curve(train_losses, val_losses, loss_curve_path)
    
    print(f"Done Finetuning. Best val exact-match accuracy: {best_acc:.4f}")
    print(f"Checkpoints saved to: {args.output_dir}")


if __name__ == "__main__":
    main()