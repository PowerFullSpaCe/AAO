"""L1: wczytanie, RGB/HSV/szarość, próbkowanie i kwantyzacja, histogramy"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3] # root projektu
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src")) #umożliwia znalezienie poszczególnych plików w strukturze projektu

from aaop.config import figures_dir, load_experiment, project_root


def discretize_image(
    bgr: np.ndarray, spatial_step: int = 2, levels: int = 8
) -> np.ndarray:
    """Podpróbkowanie przestrzenne + redukcja bitów (na kanale szarości)."""
    # ---dyskretyzacja i kwantyzacja---
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    small = gray[::spatial_step, ::spatial_step] # próbkowanie obrazu: bierzemy co dwa wiersze i co dwie kolumny
    bins = np.linspace(0, 255, levels + 1) # return evenly spaced numbers over a specified interval
    idx = np.digitize(small, bins) - 1 # wrzucamy/przypisujemy poszczególne piksele do odpowiednich binów
    idx = np.clip(idx, 0, levels - 1)
    q = (idx * (255 // max(levels - 1, 1))).astype(np.uint8) #redukcja poziomów jasności - kwantyzacja: każdy pikse dostaje zdykretyzowaną wartość
    return cv2.resize(q, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)


def run_single(image_path: Path, out_dir: Path) -> None:
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(image_path)
    h, w = bgr.shape[:2]
    depth = 8  # uint8

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    disc = discretize_image(bgr, spatial_step=2, levels=8)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title(f"RGB {w}x{h}, {depth}-bit")
    axes[0, 1].imshow(hsv)
    axes[0, 1].set_title("HSV")
    axes[0, 2].imshow(gray, cmap="gray")
    axes[0, 2].set_title("Skala szarości")
    axes[1, 0].imshow(disc, cmap="gray")
    axes[1, 0].set_title("Po dyskretyzacji (prób.+kwant.)")
    for ax in axes[1, 1:]:
        ax.axis("off")
    fig.suptitle("L1: przestrzenie barw i dyskretyzacja")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "l1_przestrzenie.png", dpi=150)
    plt.close(fig)

    # Histogramy intensywności (szarość)
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.hist(gray.ravel(), bins=256, range=(0, 256), alpha=0.7, label="przed") #ravel transformuje multi-dim array na 1D-array
    ax2.hist(disc.ravel(), bins=256, range=(0, 256), alpha=0.7, label="po dyskretyzacji")
    ax2.legend()
    ax2.set_title("Histogram intensywności")
    fig2.savefig(out_dir / "l1_histogram_przed_po.png", dpi=150)
    plt.close(fig2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-jpg", type=str, default=None)
    args = ap.parse_args()
    cfg = load_experiment()
    pr = project_root()
    rep = figures_dir(cfg)
    if args.sample_jpg:
        run_single(Path(args.sample_jpg), rep)
    else:
        jpg_root = pr / "data" / "processed" / "frames" / "train"
        if not jpg_root.is_dir():
            print(
                "Brak wyciągniętych klatek — uruchom extract_frames.py lub --sample-jpg",
                file=sys.stderr,
            )
            # syntetyczny obraz testowy
            from PIL import Image, ImageDraw

            tmp = rep / "_synthetic_l1.png"
            img = Image.new("RGB", (160, 120), (40, 90, 120))
            ImageDraw.Draw(img).ellipse((40, 20, 120, 100), fill=(200, 200, 50))
            img.save(tmp)
            run_single(tmp, rep)
            return
        first = next(jpg_root.rglob("*.jpg"), None)
        if first is None:
            print("Brak JPG", file=sys.stderr)
            sys.exit(1)
        run_single(first, rep)
    print("L1 zapisano do", rep)


if __name__ == "__main__":
    main()
