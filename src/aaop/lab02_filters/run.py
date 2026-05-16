"""L2: Gauss, Sobel, Laplacian/median, obroty/skalowanie/odbicia, Otsu vs Adaptive."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from aaop.config import figures_dir, load_experiment, project_root


def apply_filters(gray: np.ndarray) -> dict[str, np.ndarray]:
    """Wykorzystanie filtrów"""
    g = cv2.GaussianBlur(gray, (5, 5), 0) # im większy kernel, tym większe rozmycie. Piksele dostają wartość ważonej średniej
    sobelx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) # liczymy pochodną po x, a wynik przechowujemy w 32-bitowym floacie
    sobely = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) # liczymy pochodną po y
    sobel = cv2.magnitude(sobelx, sobely) # sqrt(x^2+y^2)
    sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) # normalizujemy pochodne do wartości wygodnych do ponownego wyświetlenia obrazu
    lap = cv2.Laplacian(g, cv2.CV_32F)
    lap = cv2.convertScaleAbs(lap) # 1) wartość bezwględna, 2) skalowanie wartości, 3) zamiana na uint8
    med = cv2.medianBlur(gray, 5)
    return {
        "gauss": g,
        "sobel": sobel,
        "laplacian": lap,
        "median": med,
    }


def geometric(gray: np.ndarray) -> dict[str, np.ndarray]:
    """Obroty, skalowania i odbicia"""
    h, w = gray.shape
    M_rot = cv2.getRotationMatrix2D((w / 2, h / 2), 15, 1.0) # jako środek rotacji wybierany jest środek obrazu, 15 stopni obrót przeciwko ruchowi zegara, brak zmiany skalowania (1.0)
    rot = cv2.warpAffine(gray, M_rot, (w, h)) # stosujemy transformację do każdego piksela
    scaled = cv2.resize(gray, (w // 2, h // 2), interpolation=cv2.INTER_AREA) # zmniejszamy obraz
    scaled = cv2.resize(scaled, (w, h), interpolation=cv2.INTER_LINEAR) # zwiększamy ponownie
    flip = cv2.flip(gray, 1) # odbicie względem osi y (1)
    return {"rotate_15": rot, "scale_down_up": scaled, "flip_h": flip}


def threshold_compare(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Progowanie"""
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adapt = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return otsu, adapt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-jpg", type=str, default=None)
    args = ap.parse_args()
    cfg = load_experiment()
    pr = project_root()
    rep = figures_dir(cfg)

    if args.sample_jpg:
        path = Path(args.sample_jpg)
    else:
        jpg = next((pr / "data/processed/frames/train").rglob("*.jpg"), None)
        if jpg is None:
            print("Brak klatek - użyj --sample-jpg", file=sys.stderr)
            sys.exit(1)
        path = jpg
    bgr = cv2.imread(str(path))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    f = apply_filters(gray)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes[0, 0].imshow(gray, cmap="gray")
    axes[0, 0].set_title("Oryginał")
    axes[0, 1].imshow(f["gauss"], cmap="gray")
    axes[0, 1].set_title("Gauss")
    axes[0, 2].imshow(f["sobel"], cmap="gray")
    axes[0, 2].set_title("Sobel (mag.)")
    axes[1, 0].imshow(f["laplacian"], cmap="gray")
    axes[1, 0].set_title("Laplacian")
    axes[1, 1].imshow(f["median"], cmap="gray")
    axes[1, 1].set_title("Medianowy")
    axes[1, 2].axis("off")
    fig.suptitle("L2: filtry")
    fig.tight_layout()
    fig.savefig(rep / "l2_filtry.png", dpi=150)
    plt.close(fig)

    g = geometric(gray)
    fig2, ax2 = plt.subplots(1, 3, figsize=(12, 4))
    ax2[0].imshow(g["rotate_15"], cmap="gray")
    ax2[0].set_title("Obrót 15°")
    ax2[1].imshow(g["scale_down_up"], cmap="gray")
    ax2[1].set_title("Skalowanie")
    ax2[2].imshow(g["flip_h"], cmap="gray")
    ax2[2].set_title("Odbicie pionowe")
    fig2.suptitle("L2: transformacje geometryczne")
    fig2.tight_layout()
    fig2.savefig(rep / "l2_geometria.png", dpi=150)
    plt.close(fig2)

    otsu, adapt = threshold_compare(gray)
    fig3, ax3 = plt.subplots(1, 2, figsize=(10, 4))
    ax3[0].imshow(otsu, cmap="gray")
    ax3[0].set_title("Otsu (globalne)")
    ax3[1].imshow(adapt, cmap="gray")
    ax3[1].set_title("Adaptacyjne Gauss")
    fig3.suptitle("L2: progowanie")
    fig3.tight_layout()
    fig3.savefig(rep / "l2_progowanie.png", dpi=150)
    plt.close(fig3)

    print("L2 zapisano do", rep)


if __name__ == "__main__":
    main()
