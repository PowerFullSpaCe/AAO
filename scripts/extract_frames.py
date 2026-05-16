#!/usr/bin/env python3
"""Dla każdego wideo w CSV: ekstrakcja równomiernie rozłożonych klatek do JPG"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aaop.config import load_experiment, project_root
from aaop.common.video_io import count_frames, sample_frame_indices


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument(
        "--split", choices=("train", "val", "test", "all"), default="all"
    )
    args = ap.parse_args()
    cfg = load_experiment(args.config)
    pr = project_root()
    jpg_dir = pr / str(cfg["frames"]["jpg_dir"])
    if not jpg_dir.is_absolute():
        jpg_dir = (pr / jpg_dir).resolve()
    nper = int(cfg["frames"]["samples_per_video"])
    max_side = int(cfg["frames"].get("max_side", 224))
    jpg_dir.mkdir(parents=True, exist_ok=True)

    splits = (
        ["train", "val", "test"] if args.split == "all" else [args.split]
    )
    for sp in splits:
        csv_path = pr / "data" / "processed" / "splits" / f"{sp}.csv"
        if not csv_path.is_file():
            print(f"Brak {csv_path} - uruchom prepare_dataset.py", file=sys.stderr)
            continue
        df = pd.read_csv(csv_path)
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Klatki {sp}"):
            vpath = Path(str(row["path"]))
            cname = str(row["class_name"])
            h = hashlib.md5(vpath.name.encode()).hexdigest()[:8] # hashujemy dla unikalności
            out_sub = jpg_dir / sp / cname / f"{vpath.stem}_{h}" # ustawiamy właściwą strukturę folderów dla klas
            out_sub.mkdir(parents=True, exist_ok=True)
            nframes = count_frames(vpath)
            if nframes == 0:
                continue
            idxs = sample_frame_indices(nframes, nper)
            cap = cv2.VideoCapture(str(vpath)) # otwieramy video
            for i, fi in enumerate(idxs):
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(fi)) #ustawiamy klatkę
                ok, frame = cap.read() # odczytujemy obraz
                if not ok or frame is None:
                    continue
                h0, w0 = frame.shape[:2]
                # odpowiednio skalujemy obrazy, jeżeli są zbyt duże
                scale = min(max_side / max(h0, w0), 1.0)
                if scale < 1.0:
                    frame = cv2.resize(
                        frame,
                        (int(w0 * scale), int(h0 * scale)),
                        interpolation=cv2.INTER_AREA,
                    )
                cv2.imwrite(str(out_sub / f"frame_{i:03d}.jpg"), frame) #zapis klatek
            cap.release()
    print("Gotowe:", jpg_dir)


if __name__ == "__main__":
    main()
