from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True #deterministyczne algorytmy
    # nie dobieramy niedeterministycznych algorytmów w celu optymalizacji prędkości
    torch.backends.cudnn.benchmark = False 
