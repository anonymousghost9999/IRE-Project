"""Configuration loader and manager for Paper Feed."""

from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(config_path: str | Path = "configs/default.yaml", overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Load YAML config file and optionally apply dictionary overrides."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")
    
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if overrides:
        _deep_update(cfg, overrides)

    return cfg


def _deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update a nested dictionary."""
    for key, value in update.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base

