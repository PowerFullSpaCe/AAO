"""L4: Local Binary Pattern + tekstura, HOG, PCA/t-SNE, SVM i k-NN, porównanie"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import feature
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

import json

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from aaop.common.artifacts import append_pipeline_index, mirror_to_run
from aaop.config import figures_dir, load_experiment, project_root
from aaop.common.seed import set_global_seed


def list_frames(root: Path, max_per_class: int = 40) -> tuple[list[Path], list[int], list[str]]:
    # przechodzimy po folderach klas (każdy folder = jedna klasa obiektu)
    paths, y, class_names = [], [], []
    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    for ci, cdir in enumerate(class_dirs):
        class_names.append(cdir.name)
        # ograniczamy liczbę obrazów na klasę (żeby nie przeładować datasetu)
        for j, p in enumerate(sorted(cdir.rglob("*.jpg"))):
            if j >= max_per_class:
                break
            paths.append(p)
            y.append(ci)
    return paths, y, [class_dirs[i].name for i in range(len(class_dirs))]


def hog_feat(gray: np.ndarray) -> np.ndarray:
    # HOG opisuje krawędzie i ich kierunki, czyli strukturę obiektu
    # patrzy gdzie i w jakich kierunkach zmienia się jasność
    g = cv2.resize(gray, (128, 128))
    h = feature.hog(
        g,
        orientations=9, #liczba kierunków gradientóœ (dzieliemy 180 stopni na 9 przedziałów)
        pixels_per_cell=(8, 8), # rozdzielczość dla lokalnej analizy
        cells_per_block=(2, 2), # zakres normalizacji sąsiedztwa
        feature_vector=True, # zwraca jeden wektor cech
        block_norm="L2-Hys", # wariant normalizacji
    )
    return h


def lbp_hist(gray: np.ndarray) -> np.ndarray:
    # LBP opisuje teksturę obrazów, czy piksele są podobne do obrazów
    g = cv2.resize(gray, (128, 128))
    r = 3 # promień sąsiedztwa
    n_points = 8 * r # ostatecznie analizujemy 24 sąsiadów dla każdego piksela
    #zmiana obrazu na mapę tekstury
    lbp = feature.local_binary_pattern(g, n_points, r, method="uniform")
    # histogram: liczymy ile razy wystąpił każdy wzorzec tesktury
    hist, _ = np.histogram(
        lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2), density=True # density = True normalizuje nam histogram
    )
    return hist.astype(np.float32)


def extract(paths: list[Path], y: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # ekstrakcja cech dla całego datasetu
    # dla każdego obrazu liczymy HOG + LBP
    hog, lbp, yy = [], [], []
    for p, label in zip(paths, y, strict=False):
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        hog.append(hog_feat(gray)) #cechy krawędzi
        lbp.append(lbp_hist(gray)) #cechy tekstury
        yy.append(label)
    return (
        np.vstack(hog),
        np.vstack(lbp),
        np.array(yy, dtype=np.int64),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-class", type=int, default=30)
    args = ap.parse_args()
    cfg = load_experiment()
    set_global_seed(int(cfg.get("seed", 42)))
    pr = project_root()
    rep = figures_dir(cfg)
    train_root = pr / "data" / "processed" / "frames" / "train"
    if not train_root.is_dir():
        print("Brak", train_root, file=sys.stderr)
        sys.exit(1)
    paths, y, cnames = list_frames(train_root, max_per_class=args.max_per_class)
    if len(paths) < 20:
        print("Za mało klatek - uruchom extract_frames", file=sys.stderr)
        sys.exit(1)

    # ekstrakcja cech:
    # H = HOG, L = LBP
    H, L, yy = extract(paths, y)

    # łączymy cechy: HOG + LBP w jeden wektor
    X_combo = np.hstack([H, L])
    idx = np.arange(len(yy)) #tworzymy indeksy poszczególnych obrazów, aby później odpowiednio je podzielić
    try:
        tr_ix, te_ix = train_test_split(
            idx, test_size=0.25, random_state=42, stratify=yy
        )
    except ValueError:
        # fallback dla przypadku gdy jakaś klasa np. ma zbyt mało obserwacji do wykonania stratyfikacji
        tr_ix, te_ix = train_test_split(idx, test_size=0.25, random_state=42) # bezstratyfikacji

    X_train_c = X_combo[tr_ix]
    y_train = yy[tr_ix]
    y_test = yy[te_ix]

    pca2 = PCA(n_components=2, random_state=42)
    Z = pca2.fit_transform(StandardScaler().fit_transform(X_train_c))
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(Z[:, 0], Z[:, 1], c=y_train, cmap="tab10", s=8)
    ax.set_title("L4: PCA(2) cech HOG+LBP (zbiór treningowy)")
    fig.tight_layout()
    fig.savefig(rep / "l4_pca2.png", dpi=150)
    plt.close(fig)

    sub = min(500, X_combo.shape[0])
    Xs = StandardScaler().fit_transform(X_combo[:sub])
    # t-SNE jako nieliniowa wizualizacja struktur klas
    ts = TSNE(
        n_components=2, init="pca", learning_rate="auto", random_state=42, perplexity=30
    )
    Zt = ts.fit_transform(Xs)
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    ax2.scatter(Zt[:, 0], Zt[:, 1], c=yy[:sub], cmap="tab10", s=6)
    ax2.set_title("L4: t-SNE (podpróbka, HOG+LBP)")
    fig2.tight_layout()
    fig2.savefig(rep / "l4_tsne2.png", dpi=150)
    plt.close(fig2)

    #testujemy różne warianty cech
    variants: dict[str, np.ndarray] = {
        "hog_only": H,
        "lbp_only": L,
        "hog_lbp": X_combo,
    }
    rows_cmp: list[tuple[str, str, float]] = []
    reports: dict[str, dict[str, dict[str, float | str]]] = {}

    for name, Xm in variants.items():
        X_train = Xm[tr_ix]
        X_test = Xm[te_ix]
        
        # SVM z RBF jądrem, aby wykorzystać nieliniową granicę decyzyjną
        pipe_svm = make_pipeline(
            StandardScaler(), SVC(kernel="rbf", C=1.0, random_state=42)
        )
        pipe_knn = make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=5)
        )
        pipe_svm.fit(X_train, y_train)
        pipe_knn.fit(X_train, y_train)
        p_s = pipe_svm.predict(X_test)
        p_k = pipe_knn.predict(X_test)
        acc_s = accuracy_score(y_test, p_s)
        acc_k = accuracy_score(y_test, p_k)
        rows_cmp.append((name, "SVM_RBF", float(acc_s)))
        rows_cmp.append((name, "kNN_k5", float(acc_k)))
        reports[name] = {
            "svm": classification_report(
                y_test, p_s, target_names=cnames, output_dict=True, zero_division=0
            ),
            "knn": classification_report(
                y_test, p_k, target_names=cnames, output_dict=True, zero_division=0
            ),
        }
        # zapis szczegółowych raportów tylko dla pełnych cech
        if name == "hog_lbp":
            r_s = classification_report(y_test, p_s, target_names=cnames, zero_division=0)
            r_k = classification_report(y_test, p_k, target_names=cnames, zero_division=0)
            (rep / "l4_raport_svm.txt").write_text(
                f"Accuracy SVM (HOG+LBP): {acc_s:.4f}\n\n{r_s}", encoding="utf-8"
            )
            (rep / "l4_raport_knn.txt").write_text(
                f"Accuracy k-NN (HOG+LBP): {acc_k:.4f}\n\n{r_k}", encoding="utf-8"
            )

    # zapis porównania modeli
    pd.DataFrame(rows_cmp, columns=["feature_set", "classifier", "accuracy_test"]).to_csv(
        rep / "l4_porownanie.csv", index=False
    )
    # zapis pełnych raportów
    (rep / "l4_reports.json").write_text(
        json.dumps(reports, indent=2, default=str), encoding="utf-8"
    )

    mirrored = [
        rep / "l4_pca2.png",
        rep / "l4_tsne2.png",
        rep / "l4_porownanie.csv",
        rep / "l4_reports.json",
        rep / "l4_raport_svm.txt",
        rep / "l4_raport_knn.txt",
    ]
    mirror_to_run(mirrored)
    append_pipeline_index({"lab": "l4", "comparison": str(rep / "l4_porownanie.csv")})
    print(pd.DataFrame(rows_cmp, columns=["feature_set", "classifier", "accuracy_test"]))


if __name__ == "__main__":
    main()
