"""Wspólne ścieżki artefaktów: konsolidacja pod AAOP_RUN_DIR (pipeline z logami)"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aaop.config import figures_dir, load_experiment


def reports_figures_dir(cfg: dict[str, Any] | None = None) -> Path:
    """Katalog figur/metryk (pod AAOP_ARTIFACT_ROOT lub katalog projektu)"""
    cfg = cfg if cfg is not None else load_experiment()
    return figures_dir(cfg)


def optional_run_dir() -> Path | None:
    raw = os.environ.get("AAOP_RUN_DIR")
    if not raw:
        return None
    p = Path(raw).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_run_manifest(cfg: dict[str, Any], lab_tag: str, extra: dict[str, Any] | None = None) -> None:
    rd = optional_run_dir()
    # jeżeli nie ma katalogu na zapis logów, to przerywamy dzialanie
    if rd is None:
        return
    meta = {
        "lab": lab_tag,
        "utc_iso": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "seed": cfg.get("seed"),
        "torch_cuda_available": _cuda_available(),
        "extra": extra or {},
    }
    try:
        meta["pip_freeze_head"] = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], #zrzut zainstalowanych bibliotek
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()[:80]
    except (subprocess.CalledProcessError, FileNotFoundError):
        meta["pip_freeze_head"] = [] # jeśli coś pójdzie nie tak, to zapisujemy pustą listę
    safe_tag = lab_tag.replace("/", "_")
    # zapisujemy do ładnego jsona
    (rd / f"manifest_{safe_tag}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def mirror_to_run(paths: list[Path]) -> None:
    '''Pozwala stworzyć kopię plików z runa w razie gdyby wystąpiła nieznana zmiana w trakcie uruchomienia'''
    rd = optional_run_dir()
    if rd is None:
        return
    sub = rd / "mirrored"
    sub.mkdir(parents=True, exist_ok=True)
    for p in paths:
        if p.is_file():
            shutil.copy2(p, sub / p.name)


def append_pipeline_index(entry: dict[str, Any]) -> None:
    '''Dopisywanie nowej linii do jsonl pipeline'u'''
    rd = optional_run_dir()
    if rd is None:
        return
    idx_path = rd / "pipeline_index.jsonl"
    line = json.dumps(entry, ensure_ascii=False) + "\n"
    with open(idx_path, "a", encoding="utf-8") as f:
        f.write(line)


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False
