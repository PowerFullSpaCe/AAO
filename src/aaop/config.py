'''Skrypt wspomagający'''
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def artifact_workspace() -> Path:
    """Bazowy katalog na `reports/` i `models/`"""
    raw = os.environ.get("AAOP_ARTIFACT_ROOT")
    if raw:
        p = Path(raw).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p
    return project_root()


def figures_dir(cfg: dict[str, Any]) -> Path:
    d = artifact_workspace() / str(cfg.get("paths", {}).get("reports_dir", "reports")) / "figury"
    d.mkdir(parents=True, exist_ok=True)
    return d


def models_output_dir(cfg: dict[str, Any]) -> Path:
    d = artifact_workspace() / str(cfg.get("paths", {}).get("models_dir", "models"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_experiment(path: str | Path | None = None) -> dict[str, Any]:
    root = project_root()
    cfg_path = Path(path) if path else root / "configs" / "experiment.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    env_root = os.environ.get("AAOP_UCF101_ROOT")
    if env_root:
        cfg["dataset"]["root"] = env_root
    return cfg
