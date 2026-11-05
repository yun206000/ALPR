"""
config.py — 讀取專案根目錄的 config.yaml（所有設定的唯一來源）。

欄位不設預設值：config.yaml 少了任何欄位或拼錯，載入時就會直接報錯，
不會默默用程式碼裡的預設值。

優先順序：CLI 參數（apply_overrides）> 環境變數 ALPR_<區塊>__<欄位> > config.yaml
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Type, TypeVar

from pydantic import BaseModel, ConfigDict, Field, SecretStr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

CONFIG_ENV_VAR = "ALPR_CONFIG"
DEFAULT_CONFIG_PATH = Path("config.yaml")


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# 推論
# ---------------------------------------------------------------------------

class DetectorConfig(_Section):
    weights_path: Path = Field(description="YOLO 權重")
    confidence_threshold: float = Field(ge=0.0, le=1.0, description="偵測信心門檻")
    device: str = Field(description="cuda | mps | cpu")


class RecognizerConfig(_Section):
    weights_path: Path = Field(description="PARSeq 權重")
    model_variant: str = Field(description="找不到權重時改用的 torch.hub 預訓練模型")
    charset: str = Field(description="辨識結果只保留這些字元")
    device: str = Field(description="cuda | mps | cpu")
    crop_padding: float = Field(ge=0.0, le=0.5, description="裁切車牌時 bbox 外擴的比例")


class ApiConfig(_Section):
    host: str
    port: int
    workers: int
    tokens: Dict[str, SecretStr] = Field(
        description="允許呼叫 API 的 token {用戶端名稱: token}；空的話 API Server 拒絕啟動"
    )


# ---------------------------------------------------------------------------
# 訓練
# ---------------------------------------------------------------------------

class TrackingConfig(_Section):
    enabled: bool = Field(description="是否紀錄到 trackio")
    project: str = Field(description="trackio 專案名稱")
    trackio_dir: Path = Field(description="trackio 資料庫位置")


class TrainParseqConfig(_Section):
    train_csv: Path = Field(description="訓練集 GT CSV（image_path,plate_text）")
    val_csv: Optional[Path] = Field(description="驗證集 GT CSV；null 時從訓練集切出 val_frac")
    val_frac: float = Field(gt=0.0, lt=1.0)
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    freeze_encoder: bool = Field(description="凍結 ViT encoder，只訓練 decoder + head")
    num_workers: int
    seed: int
    device: Optional[str] = Field(description="cuda | cpu；null = 有 GPU 就用 cuda")
    output_dir: Path
    finetune_from: Optional[Path] = Field(description="從既有 checkpoint 繼續訓練")


class FinetuneYoloConfig(_Section):
    data_root: Path = Field(description="底下要有 train/、valid/（各含 images/、labels/）")
    train_dir: Optional[Path] = Field(description="不填則用 <data_root>/train")
    val_dir: Optional[Path] = Field(description="不填則用 <data_root>/valid")
    val_frac: float = Field(gt=0.0, lt=1.0)
    model: str = Field(description="finetune_from 為 null 時使用的預訓練權重")
    finetune_from: Optional[Path] = Field(description="要 fine-tune 的 YOLO 權重")
    imgsz: int
    epochs: int
    batch_size: int
    lr: float
    freeze: int = Field(ge=0, description="凍結前 N 層（0 = 全部都訓練）")
    patience: int = Field(description="early stopping patience（epochs）")
    device: Optional[str] = Field(description='例如 "0" 或 "cpu"；null = 自動選擇')
    workers: int
    seed: int
    output_dir: Path


# ---------------------------------------------------------------------------
# 整份設定
# ---------------------------------------------------------------------------

class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ALPR_", env_nested_delimiter="__", extra="forbid")

    detector: DetectorConfig
    recognizer: RecognizerConfig
    output_dir: Path
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]
    api: ApiConfig
    tracking: TrackingConfig
    train_parseq: TrainParseqConfig
    finetune_yolo: FinetuneYoloConfig

    @property
    def log_file(self) -> Path:
        """推論 / 評估 / API 的 log 檔（每天換檔）。"""
        return self.output_dir / "lpr_{time:YYYY-MM-DD}.log"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        del dotenv_settings, file_secret_settings  # 刻意不讀 .env / secrets，設定只來自 config.yaml
        # 前面的優先：環境變數可以覆寫 config.yaml
        return (init_settings, env_settings, YamlConfigSettingsSource(settings_cls, yaml_file=_config_path()))


def _config_path() -> Path:
    return Path(os.environ.get(CONFIG_ENV_VAR, DEFAULT_CONFIG_PATH))


_config: Optional[AppConfig] = None


def load_config(path: Optional[str | Path] = None) -> AppConfig:
    """讀取設定檔並設為全域設定。

    path 未指定時讀 $ALPR_CONFIG，再沒有就讀 ./config.yaml。
    指定的路徑會寫回 $ALPR_CONFIG，讓 API Server 的 worker 行程也讀同一份。
    """
    global _config
    if path is not None:
        os.environ[CONFIG_ENV_VAR] = str(path)
    config_path = _config_path()
    if not config_path.is_file():
        raise FileNotFoundError(f"找不到設定檔：{config_path.resolve()}（請在專案根目錄執行，或用 --config 指定）")
    _config = AppConfig()
    return _config


def get_config() -> AppConfig:
    """取得全域設定（第一次呼叫時自動讀取設定檔）。"""
    return _config if _config is not None else load_config()


# ---------------------------------------------------------------------------
# 訓練腳本的 CLI 覆寫：每個欄位自動產生一個 --xxx 參數
# ---------------------------------------------------------------------------

_S = TypeVar("_S", bound=BaseModel)


def add_config_args(parser: argparse.ArgumentParser, section_cls: Type[BaseModel], section: str) -> None:
    """依 section_cls 的欄位加入 --config 與覆寫用的 CLI 參數（預設 None = 用設定檔的值）。"""
    parser.add_argument("--config", type=Path, default=None, help=f"設定檔路徑（預設 {DEFAULT_CONFIG_PATH}）")
    group = parser.add_argument_group(f"覆寫 config.yaml 的 {section} 區塊")
    for name, field in section_cls.model_fields.items():
        flag = "--" + name.replace("_", "-")
        help_text = field.description or ""
        if field.annotation is bool:
            group.add_argument(flag, dest=name, action=argparse.BooleanOptionalAction, default=None, help=help_text)
        else:
            nullable = " 可用 null 清空" if "Optional" in str(field.annotation) else ""
            group.add_argument(flag, dest=name, default=None, metavar=name.upper(), help=help_text + nullable)


def apply_overrides(section: _S, args: argparse.Namespace) -> _S:
    """把 CLI 有指定的參數覆寫到設定區塊上（經過 pydantic 驗證與型別轉換）。"""
    updates = {}
    for name in type(section).model_fields:
        value = getattr(args, name, None)
        if value is None:
            continue
        updates[name] = None if isinstance(value, str) and value.lower() in ("null", "none") else value
    return type(section).model_validate({**section.model_dump(), **updates})
