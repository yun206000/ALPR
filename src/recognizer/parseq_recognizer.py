"""
recognizer/parseq_recognizer.py

PARSeq（Permutation-based Auto-Regressive Sequence）文字辨識模組。
PARSeq 是場景文字辨識（STR）的 SOTA 模型，適合車牌字元辨識。

安裝：
    pip install git+https://github.com/baudm/parseq.git
    # 或 torch.hub 方式（自動下載預訓練權重）：
    torch.hub.load('baudm/parseq', 'parseq', pretrained=True)
"""

from __future__ import annotations

from pathlib import Path
from pyexpat import model
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage

from ..config import RecognizerConfig
from ..models import RecognitionResult
from ..utils import logger


# PARSeq 標準輸入尺寸
PARSEQ_IMG_W = 128
PARSEQ_IMG_H = 32


def _preprocess_plate_image(img: np.ndarray) -> PILImage.Image:
    """
    車牌裁切區域前處理：
    1. 轉灰階（車牌辨識不依賴顏色）
    2. 對比增強（CLAHE）
    3. Resize 至 PARSeq 標準尺寸 128×32
    """
    import cv2

    # # BGR → Gray
    # if len(img.shape) == 3:
    #     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # else:
    #     gray = img

    # # CLAHE 對比增強
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    # enhanced = clahe.apply(gray)

    # # 轉回 RGB（PARSeq 需要 3 channel）
    # rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    # pil = PILImage.fromarray(rgb).resize(
    #     (PARSEQ_IMG_W, PARSEQ_IMG_H), PILImage.BICUBIC
    # )
    pil = PILImage.fromarray(img).resize(
        (PARSEQ_IMG_W, PARSEQ_IMG_H), PILImage.BICUBIC
    )
    return PILImage.fromarray(img)

def postprocess_plate(text: str) -> str:
    """
    台灣車牌後處理

    O -> 0
    I -> L
    """

    return (
        text.upper()
            .replace("O", "0")
            .replace("I", "L")
    )


class PARSeqRecognizer:
    """
    封裝 PARSeq 模型，提供統一的字元辨識介面。

    使用方式：
        recognizer = PARSeqRecognizer(config)
        result = recognizer.recognize(plate_crop_rgb)
    """

    def __init__(self, config: RecognizerConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self._model = None
        self._transform = None
        self._load_model()

    def _load_model(self) -> None:
        """載入 PARSeq 模型與對應的影像轉換"""
        logger.info(f"載入 PARSeq 模型：{self.config.model_variant}")

        weights = self.config.weights_path
        try:
            if weights.exists():
                # 從本地 checkpoint 載入
                ckpt = torch.load(str(weights), map_location=self.device)
                # self._model = torch.hub.load(
                #     "baudm/parseq",
                #     self.config.model_variant,
                #     pretrained=False,
                # )
                self._model = torch.hub.load("baudm/parseq", "parseq", pretrained=False, trust_repo=True,
                            **ckpt["hyper_parameters"])
                # state = ckpt.get("state_dict", ckpt)
                # self._model.load_state_dict(state, strict=True)
                self._model.load_state_dict(ckpt["model_state_dict"])
                logger.info(f"使用本地權重：{weights}")
            else:
                # 下載預訓練權重
                logger.warning(f"找不到本地權重 {weights}，從 torch.hub 下載")
                self._model = torch.hub.load(
                    "baudm/parseq",
                    self.config.model_variant,
                    pretrained=True,
                )
        except Exception as e:
            raise RuntimeError(
                f"PARSeq 載入失敗：{e}\n"
                "請確認已安裝：pip install git+https://github.com/baudm/parseq.git"
            ) from e

        self._model.to(self.device)
        self._model.eval()

        # PARSeq 內建 transform
        self._transform = self._model.hparams.get("transform", None)
        if self._transform is None:
            self._build_default_transform()

        logger.info("PARSeq 模型載入完成")

    def _build_default_transform(self) -> None:
        """建立 PARSeq 標準輸入轉換（torchvision）"""
        from torchvision import transforms  # type: ignore

        self._transform = transforms.Compose([
            transforms.Resize((PARSEQ_IMG_H, PARSEQ_IMG_W), transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def recognize(
        self,
        plate_crop: np.ndarray,
        return_raw: bool = False,
    ) -> RecognitionResult:
        """
        辨識單一車牌裁切影像。

        Args:
            plate_crop: RGB ndarray，車牌裁切區域
            return_raw: 是否回傳原始 logits（debug 用）

        Returns:
            RecognitionResult
        """
        pil_img = _preprocess_plate_image(plate_crop)
        tensor = self._transform(pil_img).unsqueeze(0).to(self.device)  # [1, 3, H, W]

        # PARSeq forward
        logits = self._model(tensor)  # [1, T, num_classes]

        # Softmax → 取最佳路徑（greedy decode）
        probs = logits.softmax(-1)
        preds, scores = self._model.tokenizer.decode(probs)

        text = preds[0].strip() if preds else ""
        score = float(scores[0]) if scores else 0.0

        # 過濾非車牌字元
        if self.config.charset:
            text = "".join(c for c in text.upper() if c in self.config.charset)

        text = postprocess_plate(text)

        logger.debug(f"辨識結果：'{text}'  score={score:.3f}")
        return RecognitionResult(plate_text=text, recognizer_score=score)

    @torch.no_grad()
    def recognize_batch(
        self,
        plate_crops: List[np.ndarray],
    ) -> List[RecognitionResult]:
        """
        批次辨識多個車牌裁切影像（提升 GPU 效率）。

        Args:
            plate_crops: List of RGB ndarray

        Returns:
            List[RecognitionResult]
        """
        if not plate_crops:
            return []

        tensors = torch.stack([
            self._transform(_preprocess_plate_image(c))
            for c in plate_crops
        ]).to(self.device)  # [B, 3, H, W]

        logits = self._model(tensors)
        probs = logits.softmax(-1)
        preds, scores = self._model.tokenizer.decode(probs)

        results = []
        for text, score in zip(preds, scores):
            text = text.strip().upper()
            if self.config.charset:
                text = "".join(c for c in text if c in self.config.charset)
            text = postprocess_plate(text)
            results.append(
                RecognitionResult(
                    plate_text=text,
                    recognizer_score=float(score.mean().item()),
                )
            )
        return results

    def warmup(self) -> None:
        dummy = np.zeros((PARSEQ_IMG_H, PARSEQ_IMG_W, 3), dtype=np.uint8)
        self.recognize(dummy)
        logger.debug("PARSeq warm up 完成")
