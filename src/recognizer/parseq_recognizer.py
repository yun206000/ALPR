"""
recognizer/parseq_recognizer.py — PARSeq 車牌字元辨識模組。

PARSeq（Permutation-based Auto-Regressive Sequence）是場景文字辨識（STR）模型，
透過 torch.hub（baudm/parseq）載入。
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
from PIL import Image

from ..config import RecognizerConfig
from ..models import RecognitionResult
from ..utils import logger
from .transforms import get_transform


def postprocess_plate(text: str) -> str:
    """
    台灣車牌後處理：
        O -> 0
        I -> L
    """
    return text.upper().replace("O", "0").replace("I", "L")


class PARSeqRecognizer:
    """
    封裝 PARSeq 模型，提供統一的字元辨識介面。

    使用方式：
        recognizer = PARSeqRecognizer(config)
        results = recognizer.recognize_batch([plate_crop_rgb, ...])
    """

    def __init__(self, config: RecognizerConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self._model = self._load_model()
        self._model.to(self.device).eval()
        # 必須與訓練時的前處理一致（見 transforms.py）
        self._transform = get_transform(self._model.hparams.img_size, augment=False)
        logger.info("PARSeq 模型載入完成")

    def _load_model(self) -> torch.nn.Module:
        weights = self.config.weights_path
        if weights.exists():
            logger.info(f"載入 PARSeq 權重：{weights}")
            ckpt = torch.load(str(weights), map_location=self.device)
            model = torch.hub.load(
                "baudm/parseq", "parseq",
                pretrained=False, trust_repo=True, verbose=False,
                **ckpt["hyper_parameters"],
            )
            model.load_state_dict(ckpt["model_state_dict"])
            return model

        logger.warning(
            f"找不到 PARSeq 權重 {weights}，改用 torch.hub 的預訓練模型"
            f"（{self.config.model_variant}，未針對車牌訓練）"
        )
        return torch.hub.load(
            "baudm/parseq", self.config.model_variant,
            pretrained=True, trust_repo=True, verbose=False,
        )

    @torch.no_grad()
    def recognize_batch(self, plate_crops: List[np.ndarray]) -> List[RecognitionResult]:
        """
        批次辨識多個車牌裁切影像。

        Args:
            plate_crops: RGB ndarray 列表

        Returns:
            與輸入順序相同的 RecognitionResult 列表
        """
        if not plate_crops:
            return []

        tensors = torch.stack([
            self._transform(Image.fromarray(crop)) for crop in plate_crops
        ]).to(self.device)  # [B, 3, H, W]

        probs = self._model(tensors).softmax(-1)
        preds, scores = self._model.tokenizer.decode(probs)

        results = []
        for text, score in zip(preds, scores):
            text = text.strip().upper()
            if self.config.charset:
                text = "".join(c for c in text if c in self.config.charset)
            results.append(
                RecognitionResult(
                    plate_text=postprocess_plate(text),
                    recognizer_score=float(score.mean().item()),
                )
            )
        return results

    def warmup(self) -> None:
        """用空白影像跑一次推論，讓第一個真正的請求不用等初始化。"""
        self.recognize_batch([np.zeros((32, 128, 3), dtype=np.uint8)])
        logger.debug("PARSeq warmup 完成")
