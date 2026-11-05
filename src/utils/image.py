"""
utils/image.py — 影像讀取與預處理共用函式
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def load_image_cv2(path: str | Path) -> np.ndarray:
    """以 BGR ndarray 載入影像，失敗時拋出 ValueError"""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"無法載入影像：{path}")
    return img


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


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
