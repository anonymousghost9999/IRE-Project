"""Unit tests for Phase 0 common components (config, logger, manifest, cache)."""

import json
from pathlib import Path
from paperfeed.common.config import load_config
from paperfeed.common.logger import get_logger
from paperfeed.common.manifest import RunManifest, compute_config_hash
from paperfeed.common.cache import DiskCache


def test_load_default_config():
    cfg = load_config("configs/default.yaml")
    assert cfg["project"]["cutoff_year"] == 2024
    assert cfg["project"]["target_year"] == 2025
    assert cfg["retrieval"]["candidate_pool_size"] == 100
    assert cfg["retrieval"]["channel_depth"] == 200
    assert cfg["cohort"]["total_authors"] == 400


def test_config_overrides():
    cfg = load_config("configs/default.yaml", overrides={"retrieval": {"candidate_pool_size": 50}})
    assert cfg["retrieval"]["candidate_pool_size"] == 50
    assert cfg["retrieval"]["channel_depth"] == 200  # un-overridden values remain intact


def test_logger():
    logger = get_logger("test_logger")
    assert logger.name == "test_logger"


def test_run_manifest(tmp_path, sample_config):
    runs_dir = tmp_path / "runs"
    manifest = RunManifest(run_id="test_run_01", config=sample_config, runs_dir=runs_dir)
    assert manifest.status == "RUNNING"
    assert (runs_dir / "test_run_01" / "manifest.json").exists()

    manifest.complete(status="SUCCESS", metadata={"evaluated_authors": 10})
    assert manifest.status == "SUCCESS"
    assert manifest.end_time is not None

    with open(manifest.manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["status"] == "SUCCESS"
    assert data["metadata"]["evaluated_authors"] == 10


def test_disk_cache(tmp_path):
    db_path = tmp_path / "cache.sqlite3"
    cache = DiskCache(db_path=db_path)

    item_id = "paper_123"
    model_id = "specter2_v1"
    config_hash = "abc12345"
    payload = {"embedding": [0.1, 0.2, 0.3], "tokens": 42}

    assert not cache.has(item_id, model_id, config_hash)
    assert cache.get(item_id, model_id, config_hash) is None

    cache.set(item_id, model_id, config_hash, payload)
    assert cache.has(item_id, model_id, config_hash)

    cached_val = cache.get(item_id, model_id, config_hash)
    assert cached_val == payload

