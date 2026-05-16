"""L7: sekwencje wideo, CNN+LSTM vs baseline klatkowy, top-1 / top-5, błędy"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from aaop.common.artifacts import append_pipeline_index, mirror_to_run
from aaop.common.metrics import topk_accuracy as batch_topk_acc
from aaop.common.seed import set_global_seed
from aaop.config import figures_dir, load_experiment, project_root
from aaop.lab07_video.dataset import VideoClipDataset, single_frame_from_clip
from aaop.lab07_video.model import CnnLstm, FrameBaseline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_video_rows(
    pr: Path, split: str
) -> list[tuple[Path, int]]:
    '''Wczytanie filmiku i jego klasy'''
    csv_path = pr / "data" / "processed" / "splits" / f"{split}.csv"
    if not csv_path.is_file():
        return []
    df = pd.read_csv(csv_path)
    return [(Path(str(r["path"])), int(r["class_idx"])) for _, r in df.iterrows()]


@torch.no_grad()
def eval_topk(
    model: nn.Module,
    loader: DataLoader,
    clip_paths: list[Path],
) -> tuple[float, float, list[dict[str, object]]]:
    model.eval()
    errors: list[dict[str, object]] = []
    top1, top5, n, idx = 0.0, 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        top1 += float((logits.argmax(1) == y).sum().item())
        top5 += batch_topk_acc(logits, y, k=5) * y.size(0)
        n += y.size(0)
        for i in range(y.size(0)):
            p = int(logits[i].argmax().item())
            #zapisujemy tylko pierwsze 24 błędy
            if p != int(y[i].item()) and len(errors) < 24:
                errors.append(
                    {
                        "clip_index": idx,
                        "path": str(clip_paths[idx]),
                        "y_true": int(y[i].item()),
                        "y_pred": p,
                    }
                )
            idx += 1
    return top1 / max(n, 1), top5 / max(n, 1), errors


def main() -> None:
    cfg = load_experiment()
    set_global_seed(int(cfg.get("seed", 42)))
    pr = project_root()
    rep = figures_dir(cfg)
    rows_tr = load_video_rows(pr, "train")
    rows_te = load_video_rows(pr, "test")
    clip_paths_te = [p for p, _ in rows_te]
    if len(rows_tr) < 4 or not rows_te:
        print("L7: wymagany jest split CSV (prepare_dataset) i wideo", file=sys.stderr)
        sys.exit(1)
    n_cls = max(y for _, y in rows_tr) + 1
    nf = int(cfg.get("l7", {}).get("clip_frames", 16))
    st = int(cfg.get("l7", {}).get("frame_stride", 1))
    ds_tr = VideoClipDataset(rows_tr, nf, st)
    ds_te = VideoClipDataset(rows_te, nf, st)
    bs = int(cfg.get("l7", {}).get("batch_size", 2))
    loader_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, num_workers=0)
    loader_te = DataLoader(ds_te, batch_size=bs, shuffle=False, num_workers=0)

    ep = int(cfg.get("l7", {}).get("epochs", 5))
    hid = int(cfg.get("l7", {}).get("lstm_hidden", 256))
    m_vid = CnnLstm(n_cls, hid).to(device)
    opt = torch.optim.Adam(
        (p for p in m_vid.parameters() if p.requires_grad), lr=1e-4 # Adam modyfikuje tylko odmrożone wagi
    )
    crit = nn.CrossEntropyLoss()
    vid_hist: list[dict[str, float]] = []
    for e in range(ep):
        m_vid.train()
        tot = 0.0
        nb = 0
        for x, y in loader_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            o = m_vid(x)
            loss = crit(o, y)
            loss.backward()
            opt.step()
            tot += float(loss) * y.size(0)
            nb += y.size(0)
        ep_loss = tot / max(nb, 1)
        vid_hist.append({"epoch": e + 1, "train_loss_mean": ep_loss})
        print(f"L7 wideo ep {e+1}/{ep} loss {ep_loss:.4f}")

    t1, t5, err_objs = eval_topk(m_vid, loader_te, clip_paths_te)
    lines = [
        f"top-1 (wideo LSTM): {t1:.4f}",
        f"top-5 (wideo LSTM): {t5:.4f}",
        "Przykładowe pomyłki (ścieżka, y_true, y_pred):",
    ]
    for e in err_objs[:10]:
        lines.append(f"{e['path']} true={e['y_true']} pred={e['y_pred']}")

    class _Wrap(torch.utils.data.Dataset):
        '''Mini-klasa, która zwraca tylko środkowe klatki video'''
        def __init__(self, base: VideoClipDataset) -> None:
            self.b = base

        def __len__(self) -> int:
            return len(self.b)

        def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
            c, y = self.b[i]
            return single_frame_from_clip(c), y

    ds_b = _Wrap(ds_te)
    m_fr = FrameBaseline(n_cls).to(device)
    opt2 = torch.optim.Adam(m_fr.parameters(), lr=1e-4)
    ld_tr2 = DataLoader(
        _Wrap(VideoClipDataset(load_video_rows(pr, "train"), nf, st)),
        batch_size=bs * 2,
        shuffle=True,
        num_workers=0,
    )
    ld_te2 = DataLoader(ds_b, batch_size=bs * 2, shuffle=False, num_workers=0)
    frame_epochs = max(3, ep // 2)
    fr_hist: list[dict[str, float]] = []
    for e in range(frame_epochs):
        m_fr.train()
        tot, nb = 0.0, 0
        for x, y in ld_tr2:
            x, y = x.to(device), y.to(device)
            opt2.zero_grad()
            o = m_fr(x)
            loss = crit(o, y)
            loss.backward()
            opt2.step()
            tot += float(loss) * y.size(0)
            nb += y.size(0)
        fr_hist.append({"epoch": e + 1, "train_loss_mean": tot / max(nb, 1)})
    t1b, t5b, err_fr = eval_topk(m_fr, ld_te2, clip_paths_te)
    lines += ["", f"top-1 (baseline klatka): {t1b:.4f}", f"top-5 (baseline klatka): {t5b:.4f}"]

    (rep / "l7_metrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_js = {
        "video_top1": t1,
        "video_top5": t5,
        "frame_top1": t1b,
        "frame_top5": t5b,
        "num_test_clips": len(ds_te),
        "video_train_history": vid_hist,
        "frame_train_history": fr_hist,
        "sample_errors_video": err_objs[:12],
        "sample_errors_frame": err_fr[:12],
    }
    (rep / "l7_json.json").write_text(json.dumps(out_js, indent=2), encoding="utf-8")

    pd.DataFrame(vid_hist).to_csv(rep / "l7_video_train_history.csv", index=False)
    pd.DataFrame(fr_hist).to_csv(rep / "l7_frame_train_history.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([r["epoch"] for r in vid_hist], [r["train_loss_mean"] for r in vid_hist], label="CNN+LSTM train loss")
    ax.plot(
        [r["epoch"] for r in fr_hist],
        [r["train_loss_mean"] for r in fr_hist],
        label="Baseline klatka train loss",
    )
    ax.legend()
    ax.set_title("L7: krzywe uczenia (średni loss na epokę)")
    fig.tight_layout()
    fig.savefig(rep / "l7_train_curves.png", dpi=150)
    plt.close(fig)

    n_vid = len(rows_te) + len(load_video_rows(pr, "val")) + len(rows_tr)
    with open(rep / "l7_blad_analiza.txt", "w", encoding="utf-8") as f:
        f.write(
            "Błędna klasyfikacja: zestawiono ścieżki klipów z pliku splits oraz indeksy - "
            "porównać z wizualizacją klipu (np. podobne tło między klasami). "
            "Szczegóły strukturalne: `l7_json.json` → sample_errors_*.\n\n"
            f"Łączna liczba klipów w splitach CSV: {n_vid} (wymaganie L7: min. 50).\n"
        )

    mirrored = [
        rep / "l7_metrics.txt",
        rep / "l7_json.json",
        rep / "l7_video_train_history.csv",
        rep / "l7_frame_train_history.csv",
        rep / "l7_train_curves.png",
        rep / "l7_blad_analiza.txt",
    ]
    mirror_to_run(mirrored)
    append_pipeline_index({"lab": "l7", "metrics": str(rep / "l7_json.json")})
    print("L7 zapisano", rep / "l7_json.json")


if __name__ == "__main__":
    main()
