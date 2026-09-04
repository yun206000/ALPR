"""
config.py — 全域設定，使用 pydantic BaseSettings 管理。
可透過環境變數或 .env 覆寫。
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DetectorConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DETECTOR_")

    model_variant: str = Field("custom_plate", description="YOLO model variant")
    weights_path: Path = Field(Path("models/license_plate_detector.pt"), description="Path to detector weights")
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0)
    nms_iou_threshold: float = Field(0.5, ge=0.0, le=1.0)
    input_size: int = Field(640, description="Resize input to this square size")
    device: str = Field("cpu", description="cuda | mps | cpu")


class RecognizerConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RECOGNIZER_")

    # PARSeq model: parseq | parseq-tiny
    model_variant: str = Field("parseq", description="PARSeq model variant")
    weights_path: Path = Field(Path("checkpoints/best.ckpt"), description="Path to recognizer weights")
    # Expected charset for licence plates (adjust per region)
    charset: str = Field(
        "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ-OI",
        description="Valid characters for licence plates",
    )
    max_label_length: int = Field(25)
    device: str = Field("cpu")
    # Crop padding ratio around detected plate bbox
    crop_padding: float = Field(0.05, ge=0.0, le=0.5)


class PipelineConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Sub-configs
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    recognizer: RecognizerConfig = Field(default_factory=RecognizerConfig)

    # Output
    output_dir: Path = Field(Path("outputs"), description="Directory for JSON results")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1


# Singleton
_config: PipelineConfig | None = None


def get_config() -> PipelineConfig:
    global _config
    if _config is None:
        _config = PipelineConfig()
    return _config
