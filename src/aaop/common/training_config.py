"""Wspólne parametry Dataset (BNNR / seed) z configs/experiment.yaml"""

from __future__ import annotations

from typing import Any


def dataset_augment_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    art = cfg.get("artifacts") or {}
    return {
        "augment_seed_base": int(cfg.get("seed", 42)),
        "use_bnnr_augment": bool(art.get("use_bnnr_augment", True)),
        "bnnr_preset": str(art.get("bnnr_preset", "light")),
    }
