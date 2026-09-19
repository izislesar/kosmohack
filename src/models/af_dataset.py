"""AF dataset: VIIRS 8-band + aux 5-band -> 12-channel model input.

Disk layout per chip (train):
  train/af/viirs/<chip>_VIIRS_I1-I5.tif  (8x float32: I1 I2 I3 I4 I5 solar_zenith sensor_zenith valid)
  train/af/aux/<chip>_AUX.tif            (5x float32: landcover dem t2m rh2m wind_speed)
  train/af/masks/<chip>_MASK.tif         (1x uint8 {0,1}, nodata 255)

Model input (12ch, order fixed):
  [I1, I2, I3, I4n, dI45n, lc, dem, t2m, rh, wind, daynight, valid]
  - I1-I3: reflectance as-is (already ~[0,1]); NaN (night) -> 0.
  - I4n: clip [0,367K], z-norm with facts mean/std.
  - dI45n: I4-I5 z-norm (computed BEFORE NaN fill; NaN diff -> 0).
  - lc: WorldCover code / 100. dem/t2m/rh/wind: robust center/scale from config.
  - daynight: (solar_zenith > 90) float — lets the net learn day/night regimes
    since I1-I3 are NaN (=0-filled) at night by construction.
  - valid: raw valid band as a channel + returned as loss/metric weight.

Night handling: I1-I3 NaN->0 AND daynight=1 AND valid-channel path so the net
can separate "dark because night" from "dark surface".
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .datasets_common import read_tif

N_IN = 12


def build_features(viirs: np.ndarray, aux: np.ndarray, norm: dict) -> tuple:
    """viirs (8,H,W), aux (5,H,W) -> (x (12,H,W) float32, valid (H,W) float32)."""
    I1, I2, I3, I4, I5, solz, _senz, valid = viirs
    lc, dem, t2m, rh, wind = aux

    valid = np.nan_to_num(valid, nan=0.0).astype(np.float32)
    valid_bin = (valid > 0.5).astype(np.float32)

    f1 = np.nan_to_num(I1, nan=0.0)
    f2 = np.nan_to_num(I2, nan=0.0)
    f3 = np.nan_to_num(I3, nan=0.0)

    lo, hi = norm["I4_clip_K"]
    i4c = np.clip(np.nan_to_num(I4, nan=float(norm["I4_mean"])), lo, hi)
    f4 = ((i4c - norm["I4_mean"]) / norm["I4_std"]).astype(np.float32)

    d = I4 - I5
    d = np.nan_to_num(d, nan=float(norm["dI45_mean"]))
    f5 = ((d - norm["dI45_mean"]) / norm["dI45_std"]).astype(np.float32)

    flc = (np.nan_to_num(lc, nan=0.0) / float(norm["landcover_scale"])).astype(np.float32)

    dlo, dhi = norm["dem_clip"]
    demc = np.clip(np.nan_to_num(dem, nan=float(norm["dem_center"])), dlo, dhi)
    fdem = ((demc - norm["dem_center"]) / norm["dem_scale"]).astype(np.float32)

    ft2m = ((np.nan_to_num(t2m, nan=float(norm["t2m_center"])) - norm["t2m_center"]) / norm["t2m_scale"]).astype(np.float32)
    frh = ((np.nan_to_num(rh, nan=float(norm["rh_center"])) - norm["rh_center"]) / norm["rh_scale"]).astype(np.float32)
    fwind = (np.nan_to_num(wind, nan=0.0) / float(norm["wind_scale"])).astype(np.float32)

    # Day-night flag: solar_zenith > 90 -> night. NaN solz (0.65% px) -> day (0).
    solz_f = np.nan_to_num(solz, nan=0.0)
    fday = (solz_f > 90.0).astype(np.float32)

    x = np.stack([f1, f2, f3, f4, f5, flc, fdem, ft2m, frh, fwind, fday, valid_bin], axis=0)
    return x.astype(np.float32), valid_bin


class AFDataset(Dataset):
    def __init__(self, chip_ids: list, data_root: str, norm: dict, train: bool = False, seed: int = 19):
        self.chip_ids = list(chip_ids)
        self.data_root = data_root
        self.norm = norm
        self.train = train
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.chip_ids)

    def _paths(self, chip: str):
        v = os.path.join(self.data_root, "train/af/viirs", f"{chip}_VIIRS_I1-I5.tif")
        a = os.path.join(self.data_root, "train/af/aux", f"{chip}_AUX.tif")
        m = os.path.join(self.data_root, "train/af/masks", f"{chip}_MASK.tif")
        return v, a, m

    def __getitem__(self, idx: int):
        chip = self.chip_ids[idx]
        v_path, a_path, m_path = self._paths(chip)
        viirs = read_tif(v_path)
        aux = read_tif(a_path)
        x, valid = build_features(viirs, aux, self.norm)
        try:
            mask = read_tif(m_path)[0]
            mask = np.where(mask == 255, 0, mask)  # nodata tag present, 0 px observed -> treat as bg
            y = (mask > 0).astype(np.float32)
        except FileNotFoundError:
            y = np.zeros(x.shape[1:], dtype=np.float32)  # test chips have no masks

        if self.train:
            k = self.rng.randint(0, 4)
            if k:
                x = np.rot90(x, k, axes=(1, 2)).copy()
                y = np.rot90(y, k, axes=(0, 1)).copy()
                valid = np.rot90(valid, k, axes=(0, 1)).copy()
            if self.rng.rand() < 0.5:
                x = np.flip(x, axis=2).copy()
                y = np.flip(y, axis=1).copy()
                valid = np.flip(valid, axis=1).copy()

        return (
            torch.from_numpy(x),
            torch.from_numpy(y[None]),
            torch.from_numpy(valid),
            chip,
        )
