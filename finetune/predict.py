"""
Run inference with a fine-tuned PARSeq checkpoint produced by finetune_parseq.py.

Usage:
    python predict.py --ckpt checkpoints/best.ckpt --image path/to/plate.jpg
    python predict.py --ckpt checkpoints/best.ckpt --image-dir some_folder/
    python predict.py --ckpt checkpoints/best.ckpt --image-dir dataset/valid/images --csv dataset/valid/val_gt.csv
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from finetune import get_transform  # reuse the exact same preprocessing


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = torch.hub.load("baudm/parseq", "parseq", pretrained=False, trust_repo=True,
                            **ckpt["hyper_parameters"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


@torch.no_grad()
def predict(model, image_paths, device):
    img_size = model.hparams.img_size
    transform = get_transform(img_size, augment=False)
    results = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        logits = model(x)
        probs = logits.softmax(-1)
        text, confidence = model.tokenizer.decode(probs)
        results.append((p, text[0], confidence[0].mean().item()))
    return results


def evaluate_with_csv(image_dir, csv_path, predictions, device):
    """
    Evaluate predictions against a CSV file.
    CSV should have columns: 'image_path' and 'plate_text'
    """
    df = pd.read_csv(csv_path)
    
    if not {"image_path", "plate_text"}.issubset(df.columns):
        raise ValueError(f"CSV must have columns 'image_path' and 'plate_text', got {list(df.columns)}")
    
    # Build a dict of predictions keyed by filename
    pred_dict = {}
    for path, text, conf in predictions:
        # Extract just the filename for matching
        filename = path.name
        pred_dict[filename] = (text.strip().upper(), conf)
    
    # Evaluate against ground truth
    correct = 0
    total = 0
    mismatches = []
    
    for _, row in df.iterrows():
        img_path = row["image_path"]
        gt_text = str(row["plate_text"]).strip().upper()
        
        # Extract filename from the CSV path (handles various formats)
        if "/" in img_path:
            filename = img_path.split("/")[-1]
        else:
            filename = img_path
        
        if filename in pred_dict:
            pred_text, confidence = pred_dict[filename]
            total += 1
            if pred_text == gt_text:
                correct += 1
            else:
                mismatches.append((filename, pred_text, gt_text, confidence))
    
    # Print results
    accuracy = correct / max(total, 1)
    print(f"\n{'='*70}")
    print(f"Evaluation Results:")
    print(f"  Correct: {correct}/{total}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'='*70}\n")
    
    # Print all mismatches
    if mismatches:
        print(f"All {len(mismatches)} mismatches:")
        print(f"{'Filename':<40} {'Predicted':<15} {'Ground Truth':<15} {'Confidence':<12}")
        print("-" * 82)
        for filename, pred, gt, conf in mismatches:
            print(f"{filename:<40} {pred:<15} {gt:<15} {conf:.4f}")
    
    return accuracy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--image", type=Path, default=None)
    ap.add_argument("--image-dir", type=Path, default=None)
    ap.add_argument("--csv", type=Path, default=None,
                    help="CSV file with columns 'image_path' and 'plate_text' for evaluation")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = load_model(args.ckpt, device)

    if args.image is not None:
        paths = [args.image]
    elif args.image_dir is not None:
        paths = sorted([p for p in args.image_dir.iterdir()
                         if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    else:
        raise SystemExit("Provide --image or --image-dir")

    results = predict(model, paths, device)
    
    # If CSV provided, evaluate accuracy
    if args.csv is not None:
        evaluate_with_csv(args.image_dir, args.csv, results, device)
    else:
        for path, text, conf in results:
            print(f"{path}\t{text}\t(confidence={conf:.3f})")


if __name__ == "__main__":
    main()