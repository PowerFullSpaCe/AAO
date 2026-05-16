#!/usr/bin/env python3
"""Kolejno uruchamia L1–L7 z logiem konsoli i katalogiem artefaktów uruchomienia"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aaop.config import artifact_workspace, load_experiment

MODS = [
    "aaop.lab01_acquisition.run",
    "aaop.lab02_filters.run",
    "aaop.lab03_segmentation.run",
    "aaop.lab04_features.run",
    "aaop.lab05_cnn.run",
    "aaop.lab06_optimization.run",
    "aaop.lab07_video.run",
]

#piszemy jednocześnie do konsoli i do pliku z logami
class Tee:
    def __init__(self, *streams: object) -> None:
        self.streams = streams

    def write(self, data: str) -> None:
        for s in self.streams:
            s.write(data)

    def flush(self) -> None:
        for s in self.streams:
            s.flush()


def _git_head(pr: Path) -> str | None:
    '''Wyciąga hash ostatniego commita z Gita, żebyśmy wiedzieli, na jakiej wersji kodu odpalono eksperyment'''
    try:
        return subprocess.check_output(
            ["git", "-C", str(pr), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Jeśli to nie repozytorium gita albo nie ma gita w systemie, po prostu zwracamy None i działamy dalej
        return None


def main() -> None:
    pr = ROOT
    cfg = load_experiment() # Wczytujemy konfigurację eksperymentu
    ws = artifact_workspace() # Pobieramy ścieżkę do katalogu roboczego na artefakty (wyniki)
    # Generujemy unikalny timestamp (np. 20260516_142212Z), który posłuży jako nazwa folderu dla tego konkretnego przebiegu
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    # Szukamy w konfigu nazwy folderu na raporty (domyślnie "reports")
    rep_name = str(cfg.get("paths", {}).get("reports_dir", "reports"))
    run_dir = ws / rep_name / "runs" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Przygotowujemy zmienne środowiskowe dla podprocesów, żeby wiedziały gdzie szukać kodu i gdzie zapisywać wyniki
    env = {
        **os.environ,
        "PYTHONPATH": str(pr / "src"),
        "AAOP_RUN_DIR": str(run_dir),
    }

    # Zrzucamy metadane o uruchomieniu do pliku JSON - dla jak najlepszej powtarzalności badań
    meta_path = run_dir / "run_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "started_utc": stamp,
                "python": sys.version.split()[0],
                "repo_root": str(pr),
                "artifact_workspace": str(ws),
                "git_head": _git_head(pr),
                "modules": MODS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Otwieramy plik, do którego poleci kopia wszystkiego, co skrypt wypisze na ekranie
    log_path = run_dir / "pipeline_console.log"
    log_f = open(log_path, "w", encoding="utf-8")

    # Zapamiętujemy oryginalny standardowy strumień wyjścia i błędów
    orig_out, orig_err = sys.stdout, sys.stderr
    # Przekierowujemy sys.stdout i sys.stderr do naszej klasy Tee, żeby logować jednocześnie do terminala i pliku logów
    sys.stdout = Tee(orig_out, log_f)
    sys.stderr = Tee(orig_err, log_f)
    try:
        print(f"AAOP_RUN_DIR={run_dir}")
        print(f"artifact_workspace={ws}")
        for m in MODS:
            print("===", m, "===")
            # Uruchomienie subprocesu (odpowiednik w terminalu: python -m aaop.lab01...)
            r = subprocess.run([sys.executable, "-m", m], cwd=str(pr), env=env)
            line = {"module": m, "returncode": r.returncode}
            with open(run_dir / "pipeline_index.jsonl", "a", encoding="utf-8") as pix:
                pix.write(json.dumps(line, ensure_ascii=False) + "\n")
            if r.returncode != 0:
                print("BŁĄD:", m, file=sys.stderr)
                sys.exit(r.returncode)
        print("Wszystkie etapy zakończone.")
        print("Log konsoli:", log_path)
        print("Metadane:", meta_path)
    finally:
        # Blok sprzątający: musimy przywrócić oryginalne strumienie systemowe i zamknąć plik z logami
        sys.stdout = orig_out
        sys.stderr = orig_err
        log_f.close()


if __name__ == "__main__":
    main()
