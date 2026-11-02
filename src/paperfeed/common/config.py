"""Configuration loader and manager for Paper Feed."""

import os
from pathlib import Path
from typing import Any, Dict
import yaml


def find_project_root() -> Path:
    """Locate the project root directory containing pyproject.toml or configs/."""
    current = Path.cwd()
    if (current / "configs" / "default.yaml").exists():
        return current
    if (current / "IRE-Project" / "configs" / "default.yaml").exists():
        return current / "IRE-Project"
    
    # Check parent directories
    for parent in current.parents:
        if (parent / "configs" / "default.yaml").exists():
            return parent
        if (parent / "IRE-Project" / "configs" / "default.yaml").exists():
            return parent / "IRE-Project"

    # Fallback to package ancestor
    pkg_root = Path(__file__).resolve().parent.parent.parent.parent
    if (pkg_root / "configs" / "default.yaml").exists():
        return pkg_root

    import logging
    logging.getLogger("common.config").warning(
        "Could not locate project root (configs/default.yaml not found). Falling back to cwd=%s", current
    )
    return current


def load_dotenv_file(dotenv_path: str | Path | None = None) -> None:
    """Load key-value environment variables from .env if not already set."""
    if dotenv_path is None:
        root = find_project_root()
        dotenv_path = root / ".env"
    
    path = Path(dotenv_path)
    if not path.exists():
        return

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip("\"'")
            if key not in os.environ:
                os.environ[key] = val


def load_config(config_path: str | Path = "configs/default.yaml", overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Load YAML config file and optionally apply dictionary overrides."""
    load_dotenv_file()

    path = Path(config_path)
    if not path.is_absolute() and not path.exists():
        root = find_project_root()
        candidate = root / path
        if candidate.exists():
            path = candidate

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
