"""Run manifest and metadata tracker for experiment reproducibility (Phase 0.4)."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from typing import Any, Dict


def get_git_commit() -> str:
    """Retrieve current git commit hash."""
    try:
        output = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return output.decode("utf-8").strip()
    except Exception:
        return "unknown"


def compute_config_hash(config: Dict[str, Any]) -> str:
    """Compute deterministic SHA-256 hash of configuration dictionary."""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode("utf-8")).hexdigest()[:16]


def get_hardware_info() -> Dict[str, Any]:
    """Capture runtime machine architecture and GPU info if available."""
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
    }
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["device_count"] = torch.cuda.device_count()
            info["device_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch_version"] = None
        info["cuda_available"] = False
    return info


class RunManifest:
    """Manages creation, execution tracking, and serialization of run manifests."""

    def __init__(self, run_id: str, config: Dict[str, Any], runs_dir: str | Path = "results/runs") -> None:
        self.run_id = run_id
        self.config = config
        self.config_hash = compute_config_hash(config)
        self.git_commit = get_git_commit()
        self.hardware = get_hardware_info()
        self.start_time = datetime.now(timezone.utc).isoformat()
        self.end_time: str | None = None
        self.status = "RUNNING"
        self.extra_metadata: Dict[str, Any] = {}
        
        self.run_dir = Path(runs_dir) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.run_dir / "manifest.json"
        self.save()

    def complete(self, status: str = "COMPLETED", metadata: Dict[str, Any] | None = None) -> None:
        """Mark run as completed with status and end timestamp."""
        self.end_time = datetime.now(timezone.utc).isoformat()
        self.status = status
        if metadata:
            self.extra_metadata.update(metadata)
        self.save()

    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to serializable dictionary."""
        return {
            "run_id": self.run_id,
            "status": self.status,
            "git_commit": self.git_commit,
            "config_hash": self.config_hash,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "hardware": self.hardware,
            "config": self.config,
            "metadata": self.extra_metadata,
        }

    def save(self) -> None:
        """Write manifest JSON to disk."""
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

