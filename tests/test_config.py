"""src/config：讀取 config.yaml、環境變數覆寫、CLI 覆寫。"""

import argparse

import pytest
from pydantic import ValidationError

from src import config as config_module
from src.config import (
    FinetuneYoloConfig,
    TrainParseqConfig,
    add_config_args,
    apply_overrides,
    get_config,
    load_config,
)

from conftest import REPO_ROOT


def test_load_project_config_and_env_override(monkeypatch):
    monkeypatch.setenv("ALPR_DETECTOR__CONFIDENCE_THRESHOLD", "0.7")
    cfg = load_config(REPO_ROOT / "config.yaml")
    assert cfg.detector.confidence_threshold == 0.7          # 環境變數 > config.yaml
    assert cfg.api.max_batch_files >= 1
    assert str(cfg.log_file).endswith("lpr_{time:YYYY-MM-DD}.log")
    assert get_config() is cfg


def test_get_config_loads_lazily(monkeypatch):
    monkeypatch.setenv("ALPR_CONFIG", str(REPO_ROOT / "config.yaml"))
    assert config_module._config is None
    assert get_config().detector.weights_path.name == "best.pt"


def test_missing_config_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="找不到設定檔"):
        load_config(tmp_path / "nope.yaml")


def test_unknown_or_missing_fields_rejected(tmp_path):
    text = (REPO_ROOT / "config.yaml").read_text(encoding="utf-8")
    typo = tmp_path / "typo.yaml"
    typo.write_text(text.replace("log_level:", "log_levle:"), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(typo)


def test_cli_overrides():
    parser = argparse.ArgumentParser()
    add_config_args(parser, TrainParseqConfig, "train_parseq")
    args = parser.parse_args(["--epochs", "3", "--val-csv", "null", "--no-crop-val", "--config", "x.yaml"])
    base = load_config(REPO_ROOT / "config.yaml").train_parseq
    cfg = apply_overrides(base, args)
    assert cfg.epochs == 3 and cfg.val_csv is None and cfg.crop_val is False
    assert cfg.lr == base.lr  # 沒指定的欄位不變
    assert args.config.name == "x.yaml"


def test_cli_override_validation():
    parser = argparse.ArgumentParser()
    add_config_args(parser, FinetuneYoloConfig, "finetune_yolo")
    args = parser.parse_args(["--optimizer", "auto"])
    with pytest.raises(ValidationError):
        apply_overrides(load_config(REPO_ROOT / "config.yaml").finetune_yolo, args)
