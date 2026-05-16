"""L6: 2 architektury, regularizacja, LR, transfer (ResNet), Grad-CAM, IoU bbox"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from aaop.common.artifacts import append_pipeline_index, mirror_to_run
from aaop.common.bnnr_aug import bnnr_available
from aaop.common.metrics import iou_xywh
from aaop.common.seed import set_global_seed
from aaop.common.training_config import dataset_augment_kwargs
from aaop.config import figures_dir, load_experiment, models_output_dir, project_root
from aaop.lab05_cnn.dataset import ImageFolderList, collect_first_frame_per_video
from aaop.lab05_cnn.model import SmallCNN
from aaop.lab06_optimization.bbox_data import BBoxImageList
from aaop.lab06_optimization.gradcam import compute_gradcam, overlay_heatmap
from aaop.lab06_optimization.models import ResNetBBox, resnet18_classifier


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def quick_train_cnn(
    tr: DataLoader,
    va: DataLoader,
    model: nn.Module,
    epochs: int,
    lr: float,
    wd: float = 0.0,
) -> dict[str, list[float]]:
    '''Skupiamy się tylko na wartościach funkcji straty'''
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss()
    h = {"tr_loss": [], "va_loss": []}
    for _ in range(epochs):
        model.train()
        tl, n = 0.0, 0
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            tl += float(loss) * x.size(0)
            n += x.size(0)
        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for x, y in va:
                x, y = x.to(device), y.to(device)
                o = model(x)
                loss = crit(o, y)
                vl += float(loss) * x.size(0)
                vn += x.size(0)
        h["tr_loss"].append(tl / max(n, 1))
        h["va_loss"].append(vl / max(vn, 1))
    return h


def main() -> None:
    cfg = load_experiment()
    set_global_seed(int(cfg.get("seed", 42)))
    pr = project_root()
    rep = figures_dir(cfg)
    mdir = models_output_dir(cfg)
    jpg = pr / str(cfg["frames"]["jpg_dir"])
    tr_i = collect_first_frame_per_video(jpg / "train")
    va_i = collect_first_frame_per_video(jpg / "val")
    n_cls = len({y for _, y in tr_i})
    if n_cls < 2:
        print("L6: brak danych (extract_frames).", file=sys.stderr)
        sys.exit(1)
    batch = int(cfg.get("l5_cnn", {}).get("batch_size", 8))
    dkw = dataset_augment_kwargs(cfg)

    tr = DataLoader(
        ImageFolderList(tr_i, train=True, **dkw),
        batch_size=batch,
        shuffle=True,
        num_workers=0,
    )
    va = DataLoader(
        ImageFolderList(va_i, train=False, **dkw),
        batch_size=batch,
        shuffle=False,
        num_workers=0,
    )

    # --- Porównujemy surowy CNN z pretrenowanym ResNetem
    m_small = SmallCNN(n_cls, dropout=0.5)
    h_small = quick_train_cnn(tr, va, m_small, epochs=5, lr=1e-3, wd=0.0) #brak weight decay
    torch.save(m_small.state_dict(), mdir / "l6_smallcnn_highdrop.pt")

    m_r = resnet18_classifier(n_cls)
    '''mniejszy lr, bo ResNet jest większy i pretrenowany; weight decay by penalizować zbyt duże wagi'''
    h_res = quick_train_cnn(tr, va, m_r, epochs=5, lr=1e-4, wd=1e-4)
    torch.save(m_r.state_dict(), mdir / "l6_resnet18.pt")

    lr_sweep: dict[str, float] = {}
    lr_histories: dict[str, dict[str, list[float]]] = {}
    for lr in (1e-2, 1e-3):
        m2 = resnet18_classifier(n_cls)
        h2 = quick_train_cnn(tr, va, m2, epochs=3, lr=lr, wd=0.0)
        key = f"lr_{lr}"
        lr_sweep[key] = float(h2["va_loss"][-1])
        lr_histories[key] = h2
    (rep / "l6_lr_sweep.json").write_text(
        json.dumps(lr_sweep, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "epoch": np.arange(1, len(h_small["va_loss"]) + 1),
            "smallcnn_val_loss": h_small["va_loss"],
            "resnet_val_loss": h_res["va_loss"],
        }
    ).to_csv(rep / "l6_arch_val_loss.csv", index=False)

    fig_a, ax_a = plt.subplots(figsize=(8, 4))
    ax_a.plot(h_small["va_loss"], label="SmallCNN val loss")
    ax_a.plot(h_res["va_loss"], label="ResNet18 val loss")
    ax_a.legend()
    ax_a.set_title("L6: porównanie architektur (loss walidacja)")
    fig_a.tight_layout()
    fig_a.savefig(rep / "l6_arch_val_loss.png", dpi=150)
    plt.close(fig_a)

    fig_lr, ax_lr = plt.subplots(figsize=(6, 4))
    keys = list(lr_sweep.keys())
    ax_lr.bar(keys, [lr_sweep[k] for k in keys])
    ax_lr.set_title("L6: końcowy val loss po krótkim treningu (sweep LR)")
    fig_lr.tight_layout()
    fig_lr.savefig(rep / "l6_lr_sweep.png", dpi=150)
    plt.close(fig_lr)

    lr_low = lr_histories["lr_0.01"]["va_loss"]
    lr_hi = lr_histories["lr_0.001"]["va_loss"]
    pd.DataFrame(
        {
            "epoch": list(range(1, len(lr_low) + 1)),
            "lr_0_01_val": lr_low,
            "lr_0_001_val": lr_hi,
        }
    ).to_csv(rep / "l6_lr_curves.csv", index=False)

    m_r.eval()
    p0, _ = tr_i[0]
    img = cv2.imread(str(p0))
    gradcam_path = rep / "l6_gradcam.png"
    if img is not None:
        from torchvision import transforms
        from PIL import Image as PILI

        t = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )
        im = t(PILI.open(p0).convert("RGB")).unsqueeze(0).to(device)
        im.requires_grad_(True)
        m_r = m_r.to(device)
        with torch.no_grad():
            prd = m_r(im.detach()).argmax(1).item()
        im2 = t(PILI.open(p0).convert("RGB")).unsqueeze(0).to(device)
        im2.requires_grad_(True)
        cam = compute_gradcam(m_r, im2, int(prd), m_r.layer4)
        ovl = overlay_heatmap(img, cam)
        cv2.imwrite(str(gradcam_path), ovl)

    btr = BBoxImageList(tr_i, train=True, **dkw) #przygotowujemy bboxy
    bva = BBoxImageList(va_i, train=False, **dkw)
    bb_tr = DataLoader(btr, batch_size=batch, shuffle=True, num_workers=0)
    bb_va = DataLoader(bva, batch_size=batch, shuffle=False, num_workers=0)
    mb = ResNetBBox(n_cls).to(device)
    opt = torch.optim.Adam(mb.parameters(), lr=float(cfg.get("l6", {}).get("bbox_lr", 5e-4)))
    cls_loss = nn.CrossEntropyLoss()
    reg_loss = nn.L1Loss() #regresja dl bboxów
    epb = int(cfg.get("l6", {}).get("train_bbox_epochs", 10))
    bbox_rows: list[dict[str, float]] = []
    for ep in range(epb):
        mb.train()
        ep_tot, ep_n = 0.0, 0 #total loss and #dataset
        for x, y, box in bb_tr:
            x, y, box = x.to(device), y.to(device), box.to(device)
            opt.zero_grad()
            logits, pred_b = mb(x)
            l = cls_loss(logits, y) + 2.0 * reg_loss(pred_b, box)
            l.backward()
            opt.step()
            ep_tot += float(l) * x.size(0)
            ep_n += x.size(0)
        bbox_rows.append(
            {"epoch": ep + 1, "train_loss_mean_sample": ep_tot / max(ep_n, 1)}
        )
    pd.DataFrame(bbox_rows).to_csv(rep / "l6_bbox_history.csv", index=False)

    mb.eval()
    ious: list[float] = []
    n_batches = 0
    with torch.no_grad():
        for x, y, box in bb_va:
            if n_batches >= 32:
                break
            n_batches += 1
            x, box = x.to(device), box.to(device)
            _, pb = mb(x)
            for i in range(x.size(0)):
                a = box[i].cpu().numpy()
                b = pb[i].cpu().numpy()
                w_img, h_img = 224, 224
                ious.append(
                    iou_xywh(
                        (a[0] * w_img, a[1] * h_img, a[2] * w_img, a[3] * h_img),
                        (b[0] * w_img, b[1] * h_img, b[2] * w_img, b[3] * h_img),
                    )
                )
    mean_iou = float(np.mean(ious)) if ious else 0.0
    (rep / "l6_iou.txt").write_text(
        f"Średni IoU (prox. bbox vs kontur) na partii val: {mean_iou:.4f}\n",
        encoding="utf-8",
    )
    torch.save(mb.state_dict(), mdir / "l6_resnet_bbox.pt")

    summary = {
        "small_cnn": {"final_val_loss": h_small["va_loss"][-1]},
        "resnet": {"final_val_loss": h_res["va_loss"][-1], "gradcam": str(gradcam_path)},
        "mean_iou_val_sample": mean_iou,
        "bnnr_installed": bnnr_available(),
        "lr_sweep_final_val_loss": lr_sweep,
    }
    (rep / "l6_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    mirrored = [
        rep / "l6_summary.json",
        rep / "l6_lr_sweep.json",
        rep / "l6_arch_val_loss.png",
        rep / "l6_arch_val_loss.csv",
        rep / "l6_lr_sweep.png",
        rep / "l6_lr_curves.csv",
        rep / "l6_bbox_history.csv",
        rep / "l6_iou.txt",
        mdir / "l6_resnet18.pt",
        mdir / "l6_smallcnn_highdrop.pt",
        mdir / "l6_resnet_bbox.pt",
    ]
    if gradcam_path.is_file():
        mirrored.append(gradcam_path)
    mirror_to_run(mirrored)
    append_pipeline_index({"lab": "l6", "summary": str(rep / "l6_summary.json")})
    print(summary)


if __name__ == "__main__":
    main()
