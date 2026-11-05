"""
recognizer/transforms.py — PARSeq 的影像前處理。

訓練（src/train/train.py）與推論（PARSeqRecognizer）共用同一份設定，
確保兩邊的 resize 與正規化完全一致。
"""

from __future__ import annotations

from typing import Sequence

from torchvision import transforms as T


def get_transform(img_size: Sequence[int], augment: bool = False) -> T.Compose:
    """回傳 PIL Image → Tensor 的轉換。

    與 strhub.data.module.SceneTextDataModule.get_transform 相同：
    BICUBIC resize 到 img_size (H, W)，再 Normalize(0.5, 0.5)。

    Args:
        img_size: (H, W)，PARSeq 預設為 (32, 128)。
        augment: 是否加入資料增強（只在訓練時使用）。
    """
    ops = []

    if augment:
        ops += [
            T.RandomApply(
                [T.RandomPerspective(distortion_scale=0.12, p=1.0, fill=127)],
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
                [T.ColorJitter(brightness=0.30, contrast=0.35, saturation=0.10, hue=0.02)],
                p=0.60,
            ),
            T.RandomAutocontrast(p=0.15),
        ]

    ops += [
        T.Resize(tuple(img_size), T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(0.5, 0.5),
    ]

    return T.Compose(ops)
