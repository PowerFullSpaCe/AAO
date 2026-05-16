# AAO - Automatic Image Analysis

## English

AAOP is a coursework project for the Automatic Image Analysis module. It builds a reproducible image and video analysis pipeline on a selected 12-class subset of the UCF-101 dataset, covering acquisition, filtering, segmentation, feature extraction, CNN classification, model optimization, and video-based recognition.

The project is implemented as Python modules under `src/aaop`, with shared configuration in `configs/experiment.yaml`. Generated outputs are stored in `reports/`, while trained models are stored in `models/`.

### Main Stages

- `lab01_acquisition` - image/video acquisition and basic dataset inspection
- `lab02_filters` - filtering, thresholding, and geometric transforms
- `lab03_segmentation` - segmentation and object measurements
- `lab04_features` - feature extraction and classical ML classifiers
- `lab05_cnn` - CNN training and evaluation
- `lab06_optimization` - optimization experiments, Grad-CAM, and bounding-box model
- `lab07_video` - frame/clip-based video classification

### Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the UCF-101 subset in `data/raw/UCF-101` or set `AAOP_UCF101_ROOT` to the dataset path. Then prepare splits and run the full pipeline:

```bash
python scripts/prepare_dataset.py
python scripts/run_all_labs.py
```

---

## Polski

AAOP to projekt zaliczeniowy z modułu Automatyczna Analiza Obrazu. Projekt buduje powtarzalny pipeline analizy obrazów i wideo dla wybranego, 12-klasowego podzbioru UCF-101, obejmując akwizycję, filtrację, segmentację, ekstrakcję cech, klasyfikację CNN, optymalizację modeli oraz rozpoznawanie akcji w wideo.

Kod znajduje się w modułach Pythona w `src/aaop`, a główna konfiguracja eksperymentu w `configs/experiment.yaml`. Wyniki, wykresy i raporty zapisywane są w `reports/`, natomiast wytrenowane modele w `models/`.

### Główne Etapy

- `lab01_acquisition` - akwizycja obrazu/wideo i wstępna analiza danych
- `lab02_filters` - filtracja, progowanie i transformacje geometryczne
- `lab03_segmentation` - segmentacja i pomiary obiektów
- `lab04_features` - ekstrakcja cech i klasyczne klasyfikatory ML
- `lab05_cnn` - trening i ewaluacja sieci CNN
- `lab06_optimization` - eksperymenty optymalizacyjne, Grad-CAM i model ramek
- `lab07_video` - klasyfikacja wideo na podstawie klatek/sekwencji

### Szybkie Uruchomienie

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Umieść podzbiór UCF-101 w `data/raw/UCF-101` albo ustaw zmienną `AAOP_UCF101_ROOT` na ścieżkę do danych. Następnie przygotuj podziały i uruchom cały potok:

```bash
python scripts/prepare_dataset.py
python scripts/run_all_labs.py
```
