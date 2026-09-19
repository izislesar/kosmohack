"""Shared helpers for model training scripts (AF + BS).

Generic only: seeding, YAML config loading, split-list loading, GeoTIFF
reading, run logging. No task-specific normalization lives here — that
belongs in af_dataset.py / bs_dataset.py.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import random

import numpy as np
import yaml


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed python/numpy/torch RNGs; optionally force deterministic cuDNN."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # numpy/random seeding still applies (CPU-only callers)


def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def load_split(split_path: str, group: str) -> dict:
    """Load eda/split.json lists verbatim: {'train': [...], 'val': [...]}."""
    with open(split_path) as fh:
        data = json.load(fh)
    grp = data[group]
    return {"train": list(grp["train"]), "val": list(grp["val"]), "seed": data.get("seed")}


def read_tif(path: str) -> np.ndarray:
    """Read GeoTIFF -> float32 array shaped (bands, H, W)."""
    import rasterio

    with rasterio.open(path) as ds:
        arr = ds.read().astype(np.float32)
    return arr


def load_af_fire_lookup(data_root: str) -> dict:
    """chip_id -> n_fire_px from train/af/meta.csv (drives positive oversampling)."""
    lookup = {}
    with open(os.path.join(data_root, "train/af/meta.csv"), newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                n = float(row["n_fire_px"]) if row["n_fire_px"].strip() not in ("", "nan") else 0.0
            except (ValueError, KeyError):
                n = 0.0
            lookup[row["chip_id"]] = n
    return lookup


def expand_ids_for_oversampling(train_ids: list, fire_lookup: dict, factor: int, seed: int) -> list:
    """Repeat fire-positive train chips `factor`x, then seeded shuffle.

    Deterministic given (ids, factor, seed). Negatives appear once.
    """
    pos = [c for c in train_ids if fire_lookup.get(c, 0) > 0]
    neg = [c for c in train_ids if fire_lookup.get(c, 0) <= 0]
    expanded = list(neg)
    for _ in range(max(int(factor), 1)):
        expanded.extend(pos)
    rng = random.Random(seed)
    rng.shuffle(expanded)
    return expanded


def make_logger(log_path: str | None = None) -> logging.Logger:
    logger = logging.getLogger("af_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        fh = logging.FileHandler(log_path, mode="w")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger
