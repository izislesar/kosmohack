"""BS dataset — exact input manifest N=27 + facts-recipe indices + SCL/cloud gate.

N = 27 = S2pre9 + S2post9 + S1x4 + AUX3 + IDX2
  ch 0-5 : pre  reflectance B2 B3 B4 B8A B11 B12 (uint16/10000; HLS-mapped, keep pretrained patch weights)
  ch 6-8 : pre  reflectance B5 B6 B7
  ch 9-17: post reflectance B2 B3 B4 B5 B6 B7 B8A B11 B12
  ch 18-21: S1 dB VVpre VHpre VVpost VHpost (int16/100, z-normed w/ facts means/stds)
  ch 22-24: AUX dem/300, slope/45, landcover/100
  ch 25-26: dNBR, RdNBR — NBR=(B8A-B12)/(B8A+B12+1e-6) from uint16, SCL-masked
SCL (file idx 9, both dates) is NOT a feature channel — masking/gating only.
Task shorthand "pre10+post10+S1x4+aux4" counts raw file bands incl. SCL; effective model N=27.

Masks: uint8 {0,1,2,3}, nodata tag 255.
Valid pixel = (SCL_post != 0) AND (mask != 255) AND (SCL_post not in cloud_codes),
  with dry-soil override: landcover==60 (barren) + SCL in {8,9} -> valid
  (SCL is known to mistake dry light soil for cloud; case Fig.6 context).
Cloud/shadow/snow pixels are forced to class 0 in the baseline and masked out of metrics.

File layout (remote /workspace/kosmohack/data/train/bs/):
  sentinel2_{pre,post}/{chip}_Sentinel-2_{pre,post}.tif  (10x uint16)
  sentinel1_{pre,post}/{chip}_Sentinel-1_{pre,post}.tif  (2x int16)
  aux/{chip}_AUX.tif   (3x int16 [dem slope landcover])
  masks/{chip}_MASK.tif (1x uint8)
"""
import json
import os

import numpy as np

# S2 file band indices (facts §10 s2_bands_order)
B2, B3, B4, B5, B6, B7, B8A, B11, B12, SCL = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
# Model channel order: HLS-mapped six first (pretrained patch reuse), then the rest
PRE_HLS = [B2, B3, B4, B8A, B11, B12]
PRE_REST = [B5, B6, B7]
POST_ALL = [B2, B3, B4, B5, B6, B7, B8A, B11, B12]
N_CHANNELS = 27


def build_group_lut(landcover_groups):
    """WorldCover code (0..255) -> group name; default 'other'."""
    lut = np.array(["other"] * 256, dtype=object)
    for g, codes in landcover_groups.items():
        for c in codes:
            lut[int(c)] = g
    return lut


def compute_nbr_dnbr_rdnbr(pre_u16, post_u16, eps=1e-6):
    """Facts recipe: NBR=(B8A-B12)/(B8A+B12+eps) from uint16; dNBR=pre-post; RdNBR=dNBR/(sqrt(|NBR_pre|)+eps)."""
    pre = pre_u16.astype(np.float64)
    post = post_u16.astype(np.float64)
    nbr_pre = (pre[B8A] - pre[B12]) / (pre[B8A] + pre[B12] + eps)
    nbr_post = (post[B8A] - post[B12]) / (post[B8A] + post[B12] + eps)
    dnbr = nbr_pre - nbr_post
    rdnbr = dnbr / (np.sqrt(np.abs(nbr_pre)) + eps)
    return nbr_pre.astype(np.float32), dnbr.astype(np.float32), rdnbr.astype(np.float32)


def cloud_invalid_mask(scl, landcover, cloud_codes, dry_override, dry_lc, dry_relax):
    """True where pixel is cloud/shadow/snow-gated (invalid for train/metrics)."""
    m = np.isin(scl, np.asarray(cloud_codes))
    if dry_override:
        relax = (landcover == dry_lc) & np.isin(scl, np.asarray(dry_relax))
        m = m & (~relax)
    return m


def assemble_features(pre_u16, post_u16, s1pre_i16, s1post_i16, aux_i16, norm, eps=1e-6):
    """Build (27,H,W) float32 model input + (H,W) dNBR/RdNBR/SCL/landcover side arrays."""
    H, W = pre_u16.shape[1], pre_u16.shape[2]
    rscale = float(norm["s2_reflectance_scale"])
    pre = pre_u16.astype(np.float32) / rscale
    post = post_u16.astype(np.float32) / rscale
    vv = [float(v) for v in norm["s1_means_dB"]]
    vs = [float(v) for v in norm["s1_stds_dB"]]
    s1 = np.stack([
        s1pre_i16[0].astype(np.float32) / 100.0,
        s1pre_i16[1].astype(np.float32) / 100.0,
        s1post_i16[0].astype(np.float32) / 100.0,
        s1post_i16[1].astype(np.float32) / 100.0,
    ])  # dB
    s1 = (s1 - np.asarray(vv, dtype=np.float32)[:, None, None]) / np.asarray(vs, dtype=np.float32)[:, None, None]
    dem = aux_i16[0].astype(np.float32) / float(norm["dem_scale"])
    slope = aux_i16[1].astype(np.float32) / float(norm["slope_scale"])
    lc = aux_i16[2].astype(np.float32) / float(norm["landcover_scale"])
    _, dnbr, rdnbr = compute_nbr_dnbr_rdnbr(pre_u16, post_u16, eps)
    dn0, dn1 = norm["dnbr_clip"]
    rd0, rd1 = norm["rdnbr_clip"]
    dnbr = np.clip(dnbr, dn0, dn1)
    rdnbr = np.clip(rdnbr, rd0, rd1)
    x = np.zeros((N_CHANNELS, H, W), dtype=np.float32)
    x[0:6] = pre[PRE_HLS]
    x[6:9] = pre[PRE_REST]
    x[9:18] = post[POST_ALL]
    x[18:22] = s1
    x[22] = dem
    x[23] = slope
    x[24] = lc
    x[25] = dnbr
    x[26] = rdnbr
    scl_post = post_u16[SCL].astype(np.int64) if post_u16.shape[0] > SCL else pre_u16[SCL].astype(np.int64)
    return x, dnbr, rdnbr, scl_post, aux_i16[2].astype(np.int64)


def load_chip_arrays(data_root, chip_id):
    """Lazy rasterio import (local box lacks rasterio); returns (pre,post,s1pre,s1post,aux,mask|None)."""
    import rasterio
    bs = os.path.join(data_root, "train", "bs")
    def rd(*p):
        with rasterio.open(os.path.join(bs, *p)) as ds:
            return ds.read()
    pre = rd("sentinel2_pre", f"{chip_id}_Sentinel-2_pre.tif")
    post = rd("sentinel2_post", f"{chip_id}_Sentinel-2_post.tif")
    s1pre = rd("sentinel1_pre", f"{chip_id}_Sentinel-1_pre.tif")
    s1post = rd("sentinel1_post", f"{chip_id}_Sentinel-1_post.tif")
    aux = rd("aux", f"{chip_id}_AUX.tif")
    try:
        mask = rd("masks", f"{chip_id}_MASK.tif")[0].astype(np.int64)
    except Exception:
        mask = None
    return pre, post, s1pre, s1post, aux, mask


def load_split_ids(split_json, part):
    with open(split_json) as f:
        d = json.load(f)
    return list(d["bs"][part])


class BSChipDataset:
    """Minimal torch-free dataset (torch imported lazily in train_bs.py)."""

    def __init__(self, cfg, part):
        self.cfg = cfg
        root = cfg["data"]["root"]
        sj = cfg["data"]["split_json"]
        if not os.path.isabs(sj):
            # split_json is repo-relative; resolve against CWD (repo root on both boxes)
            sj = os.path.join(os.getcwd(), sj)
        self.ids = load_split_ids(sj, part)
        self.root = root
        self.norm = cfg["norm"]
        self.eps = float(cfg.get("nbr_eps", 1e-6))
        cg = cfg["cloud_gate"]
        self.cloud_codes = list(cg["cloud_codes"])
        self.dry_override = bool(cg.get("dry_soil_override", True))
        self.dry_lc = int(cg.get("dry_soil_landcover", 60))
        self.dry_relax = list(cg.get("dry_soil_relax_codes", [8, 9]))
        self.group_lut = build_group_lut(cfg["landcover_groups"])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        chip = self.ids[i]
        pre, post, s1pre, s1post, aux, mask = load_chip_arrays(self.root, chip)
        x, dnbr, rdnbr, scl, lc = assemble_features(pre, post, s1pre, s1post, aux, self.norm, self.eps)
        gated = cloud_invalid_mask(scl, lc, self.cloud_codes, self.dry_override, self.dry_lc, self.dry_relax)
        nodata = (scl == 0)
        if mask is None:
            valid = (~gated) & (~nodata)
            y = np.zeros(scl.shape, dtype=np.int64)
        else:
            valid = (~gated) & (~nodata) & (mask != 255)
            y = mask.copy()
            y[~valid] = 255
        return {"chip": chip, "x": x, "y": y, "valid": valid,
                "dnbr": dnbr, "scl": scl, "lc": lc}
