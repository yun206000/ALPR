"""
utils/image.py — 影像讀取與預處理共用函式
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def load_image_cv2(path: str | Path) -> np.ndarray:
    """以 BGR ndarray 載入影像，失敗時拋出 ValueError"""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"無法載入影像：{path}")
    return img


def load_image_pil(path: str | Path) -> Image.Image:
    """以 PIL Image (RGB) 載入影像"""
    return Image.open(str(path)).convert("RGB")


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_keep_aspect(
    img: np.ndarray,
    target_size: int,
) -> Tuple[np.ndarray, float]:
    """
    保持長寬比縮放，短邊 pad 至 target_size。
    回傳 (padded_img, scale_factor)
    """
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    canvas[:new_h, :new_w] = resized
    return canvas, scale


def crop_bbox(
    img: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    padding: float = 0.05,
) -> np.ndarray:
    """
    從影像裁切 bbox 區域，並加上 padding 比例的邊界。
    座標為像素絕對值。
    """
    h, w = img.shape[:2]
    pw = (x2 - x1) * padding
    ph = (y2 - y1) * padding

    cx1 = max(0, int(x1 - pw))
    cy1 = max(0, int(y1 - ph))
    cx2 = min(w, int(x2 + pw))
    cy2 = min(h, int(y2 + ph))
    return img[cy1:cy2, cx1:cx2]


def list_images(folder: str | Path) -> list[Path]:
    """列出資料夾內所有支援格式的影像"""
    folder = Path(folder)
    files = [
        p for p in sorted(folder.iterdir())
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return files


def draw_results(img: np.ndarray, plates: list, font_scale: float = 1.0) -> np.ndarray:
    """
    在影像上繪製偵測框與辨識文字（用於 debug 視覺化）。
    plates: List[PlateResult]
    """
    out = img.copy()
    for plate in plates:
        x1, y1, x2, y2 = [int(v) for v in plate.bbox]
        color = (0, 255, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{plate.plate_text} ({plate.confidence:.2f})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            out, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2,
        )
    return out
