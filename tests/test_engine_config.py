from __future__ import annotations

from pathlib import Path

from agentkv.serving.engine import ModelConfig

MODEL_YAML = Path(__file__).resolve().parents[1] / "configs" / "model.yaml"


def test_model_config_loads_primary_by_default():
    config = ModelConfig.from_yaml(MODEL_YAML)
    assert config.model_name == "Qwen/Qwen3-1.7B"
    assert config.dtype == "bfloat16"
    assert config.enable_prefix_caching is True
    assert config.block_size == 16


def test_model_config_loads_fallback_when_requested():
    config = ModelConfig.from_yaml(MODEL_YAML, use_fallback=True)
    assert config.model_name == "Qwen/Qwen3-0.6B"
