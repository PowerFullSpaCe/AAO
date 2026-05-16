"""L5: trening SmallCNN, augmentacja (BNNR + torchvision fallback), metryki, confusion, krzywe"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from aaop.common.artifacts import append_pipeline_index, mirror_to_run
from aaop.common.bnnr_aug import bnnr_available
from aaop.common.training_config import dataset_augment_kwargs
from aaop.config import figures_dir, load_experiment, models_output_dir, project_root
from aaop.common.seed import set_global_seed
from aaop.lab05_cnn.dataset import ImageFolderList, collect_first_frame_per_video
from aaop.lab05_cnn.model import SmallCNN


def build_loaders(cfg: dict, pr: Path) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    jpg = pr / str(cfg["frames"]["jpg_dir"])
    kw = dataset_augment_kwargs(cfg)
    train_items = collect_first_frame_per_video(jpg / "train")
    val_items = collect_first_frame_per_video(jpg / "val")
    test_items = collect_first_frame_per_video(jpg / "test")
    num_classes = len({y for _, y in train_items})
    batch = int(cfg.get("l5_cnn", {}).get("batch_size", 16))
    tr_ds = ImageFolderList(train_items, train=True, **kw)
    va_ds = ImageFolderList(val_items, train=False, **kw)
    te_ds = ImageFolderList(test_items, train=False, **kw)
    tr = DataLoader(tr_ds, batch_size=batch, shuffle=True, num_workers=0)
    va = DataLoader(va_ds, batch_size=batch, shuffle=False, num_workers=0)
    te = DataLoader(te_ds, batch_size=batch, shuffle=False, num_workers=0)
    return tr, va, te, num_classes


@torch.no_grad()
def predict_loader(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''Zbieramy predykcje i zwracamy dane do ewaluacji'''
    model.eval()
    ys, ps, lls = [], [], [] #true labels, predicted labels, logits
    for x, y in loader:
        x = x.to(device)
        z = model(x)
        lls.append(z.cpu().numpy())
        p = z.argmax(1)
        ys.append(y.numpy())
        ps.append(p.cpu().numpy())
    return (
        np.concatenate(ys), #łączymy wszystko w jedną dużą tablicę
        np.concatenate(ps),
        np.vstack(lls),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    cfg = load_experiment()
    set_global_seed(int(cfg.get("seed", 42)))
    pr = project_root()
    rep = figures_dir(cfg)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va, te, num_classes = build_loaders(cfg, pr)
    if num_classes < 2:
        print("Za mało klas - uzupełnij dane", file=sys.stderr)
        sys.exit(1)
    epochs = args.epochs or int(cfg.get("l5_cnn", {}).get("epochs", 25))
    lr = float(cfg.get("l5_cnn", {}).get("lr", 1e-3))
    model = SmallCNN(num_classes, dropout=0.3).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss() #automatycznie zawiera softmax, dlatego przekazujemy logity
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    for ep in range(epochs):
        model.train()
        tl, n, correct = 0.0, 0, 0 #total loss
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward() # obliczamy gradienty do zmiany wag
            opt.step() # aktualizacja wag modelu
            tl += float(loss) * x.size(0) # bo loss to średnia z batcha
            n += x.size(0)
            correct += (out.argmax(1) == y).sum().item()
        tr_loss = tl / n # wyznaczamy loss dla całego datasetu
        tr_acc = correct / n
        model.eval()
        vl, vn, vcorr = 0.0, 0, 0
        with torch.no_grad():
            for x, y in va:
                x, y = x.to(dev), y.to(dev)
                out = model(x)
                loss = crit(out, y)
                vl += float(loss) * x.size(0)
                vn += x.size(0)
                vcorr += (out.argmax(1) == y).sum().item()
        val_loss = vl / max(vn, 1)
        val_acc = vcorr / max(vn, 1)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(
                f"ep {ep+1}/{epochs} tr_loss {tr_loss:.4f} val_loss {val_loss:.4f} "
                f"tr_acc {tr_acc:.3f} val_acc {val_acc:.3f}"
            )
    mdir = models_output_dir(cfg)
    torch.save(model.state_dict(), mdir / "l5_smallcnn.pt")

    y_true, y_pred, _ = predict_loader(model, te, dev)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("L5: macierz pomyłek (test)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(rep / "l5_confusion.png", dpi=150)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(history["train_loss"], label="train loss")
    ax2.plot(history["val_loss"], label="val loss")
    ax2.legend()
    ax2.set_title("L5: loss")
    fig2.tight_layout()
    fig2.savefig(rep / "l5_curves_loss.png", dpi=150)
    plt.close(fig2)
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.plot(history["train_acc"], label="train acc")
    ax3.plot(history["val_acc"], label="val acc")
    ax3.legend()
    ax3.set_title("L5: accuracy")
    fig3.tight_layout()
    fig3.savefig(rep / "l5_curves_acc.png", dpi=150)
    plt.close(fig3)

    hist_df = pd.DataFrame(
        {
            "epoch": np.arange(1, epochs + 1),
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
            "train_acc": history["train_acc"],
            "val_acc": history["val_acc"],
        }
    )
    hist_df.to_csv(rep / "l5_history.csv", index=False)

    out = {
        "accuracy": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "epochs": epochs,
        "device": str(dev),
        "bnnr_installed": bnnr_available(),
        "bnnr_preset": str((cfg.get("artifacts") or {}).get("bnnr_preset", "")),
    }
    (rep / "l5_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    labels_fit = list(range(num_classes))
    tnames = (
        cfg["dataset"]["class_names"][:num_classes]
        if len(cfg["dataset"]["class_names"]) >= num_classes
        else [f"class_{i}" for i in labels_fit]
    )
    rep_js = classification_report(
        y_true,
        y_pred,
        labels=labels_fit,
        target_names=tnames,
        output_dict=True,
        zero_division=0,
    )
    (rep / "l5_classification_report.json").write_text(
        json.dumps(rep_js, indent=2),
        encoding="utf-8",
    )

    mirrored = [
        rep / "l5_confusion.png",
        rep / "l5_curves_loss.png",
        rep / "l5_curves_acc.png",
        rep / "l5_metrics.json",
        rep / "l5_history.csv",
        rep / "l5_classification_report.json",
        mdir / "l5_smallcnn.pt",
    ]
    mirror_to_run(mirrored)
    append_pipeline_index({"lab": "l5", "metrics": str(rep / "l5_metrics.json")})
    print("L5 test:", out)


if __name__ == "__main__":
    main()
