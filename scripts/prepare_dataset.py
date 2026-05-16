#!/usr/bin/env python3
"""Skanuje UCF-101 (podklasy z configs) i zapisuje stratyfikowane split CSV"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aaop.config import load_experiment, project_root
from aaop.common.seed import set_global_seed


def find_videos(root: Path, class_names: list[str]) -> list[tuple[Path, str, int]]:
    rows: list[tuple[Path, str, int]] = []
    for i, cname in enumerate(class_names):
        cdir = root / cname
        if not cdir.is_dir():
            continue
        for p in sorted(cdir.glob("*.avi")) + sorted(cdir.glob("*.mp4")):
            rows.append((p, cname, i))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    args = ap.parse_args()
    cfg = load_experiment(args.config)
    set_global_seed(int(cfg.get("seed", 42)))
    pr = project_root()
    ucf = Path(cfg["dataset"]["root"])
    if not ucf.is_absolute():
        ucf = (pr / ucf).resolve()
    class_names: list[str] = cfg["dataset"]["class_names"]
    tr = float(cfg["splits"]["train_ratio"])
    val = float(cfg["splits"]["val_ratio"])
    te = float(cfg["splits"]["test_ratio"])
    assert abs(tr + val + te - 1.0) < 1e-6 # sprawdzamy, czy suma tych trzech wartości jest w przybliżeniu równa 1

    #osobne grupowanie dla każdej klasy
    by_class: dict[str, list[Path]] = {c: [] for c in class_names}
    for path, cname, _ in find_videos(ucf, class_names):
        by_class[cname].append(path)

    train_rows, val_rows, test_rows = [], [], []
    for cname, paths in by_class.items():
        if not paths:
            print(f"UWAGA: brak wideo w klasie {cname} w {ucf / cname}", file=sys.stderr)
            continue
        random.shuffle(paths)
        n = len(paths)
        i1 = int(n * tr) # gdzie się kończy zbiór treningowy
        i2 = int(n * (tr + val)) # gdzie się kończy zbiór walidacyjny
        i1 = min(max(i1, 1), n - 1) if n > 1 else 1 # zabezpieczamy przed pustymi zbiorami
        i2 = min(max(i2, i1 + 1), n) if n > 1 else 1
        sl_train, sl_val, sl_test = paths[:i1], paths[i1:i2], paths[i2:] #właściwy podział

        idx = class_names.index(cname)
        for p in sl_train:
            train_rows.append({"path": str(p), "class_name": cname, "class_idx": idx})
        for p in sl_val:
            val_rows.append({"path": str(p), "class_name": cname, "class_idx": idx})
        for p in sl_test:
            test_rows.append({"path": str(p), "class_name": cname, "class_idx": idx})

    out = pr / "data" / "processed" / "splits"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train_rows).to_csv(out / "train.csv", index=False)
    pd.DataFrame(val_rows).to_csv(out / "val.csv", index=False)
    pd.DataFrame(test_rows).to_csv(out / "test.csv", index=False)

    print(
        f"Zapisano: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)} -> {out}"
    )

    summary = {
        "total_clips": len(train_rows) + len(val_rows) + len(test_rows),
        "train": len(train_rows),
        "val": len(val_rows),
        "test": len(test_rows),
        "meets_l7_min_50": len(train_rows) + len(val_rows) + len(test_rows) >= 50,
        "classes_requested": class_names,
        "classes_with_data": [c for c, ps in by_class.items() if ps],
    }
    (out / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        "Podsumowanie (L7 ≥50 klipów):",
        "OK" if summary["meets_l7_min_50"] else "UWAGA - za mało klipów",
        "→",
        out / "dataset_summary.json",
    )
    if not train_rows:
        print(
            "Brak danych - rozpakuj UCF-101 do data/raw/UCF-101 i upewnij się, "
            "że katalogi klas istnieją.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
