#!/usr/bin/env python3
"""Unified inference: AF Unet + BS fallback-Unet -> submission.csv (447 rows).

Contract (docs/case.md section 6+8):
  python inference.py --data-dir /path/to/test --output /path/to/submission.csv
  - reads sample_submission.csv template from data-dir (exact (chip_id,class_id) set)
  - AF chips: 1 row class_id=1; BS chips: 3 rows class_id=1/2/3
  - RLE: 1-indexed row-major, starts ascending, runs merged (no touching),
    empty class -> "" (quoted by csv writer), no NaN/overlap/out-of-range
  - AF decision: weights/af_thresh.json (thr_day/thr_night + wc gate b50_i4_310)
  - BS decision: weights/bs.pt (fallback Unet, 27ch) argmax + cloud->0 + burn-gate snap
  - deterministic: seed 19, eval mode, no aug; logs wall time for <=5min check
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SEED = 19


def rle_encode(mask):
    """Binary (H,W) -> RLE string. 1-indexed row-major, merged runs."""
    flat = np.asarray(mask, dtype=np.uint8).reshape(-1)
    if not flat.any():
        return ""
    # pad to catch edge runs
    p = np.concatenate([[0], flat, [0]])
    d = np.diff(p)
    starts = np.where(d == 1)[0] + 1  # 1-indexed
    ends = np.where(d == -1)[0] + 1
    runs = []
    for s, e in zip(starts.tolist(), ends.tolist()):
        length = int(e - s)
        if runs and s <= runs[-1][0] + runs[-1][1]:  # touching/overlap -> merge
            ps, pl = runs[-1]
            runs[-1] = (ps, int(e - ps))
        else:
            runs.append((int(s), length))
    return " ".join(f"{s} {l}" for s, l in runs)


def load_af(data_dir, chip):
    import rasterio
    with rasterio.open(os.path.join(data_dir, "af", "viirs", f"{chip}_VIIRS_I1-I5.tif")) as ds:
        v = ds.read().astype(np.float32)
    with rasterio.open(os.path.join(data_dir, "af", "aux", f"{chip}_AUX.tif")) as ds:
        a = ds.read().astype(np.float32)
    return v, a


def load_bs(data_dir, chip):
    import rasterio

    def rd(*p):
        with rasterio.open(os.path.join(data_dir, "bs", *p)) as ds:
            return ds.read()
    pre = rd("sentinel2_pre", f"{chip}_Sentinel-2_pre.tif")
    post = rd("sentinel2_post", f"{chip}_Sentinel-2_post.tif")
    s1pre = rd("sentinel1_pre", f"{chip}_Sentinel-1_pre.tif")
    s1post = rd("sentinel1_post", f"{chip}_Sentinel-1_post.tif")
    aux = rd("aux", f"{chip}_AUX.tif")
    return pre, post, s1pre, s1post, aux


def _dn_keys(bs_cfg):
    return ("forest", "steppe", "cropland", "floodplain", "other")


def _apply_remap(pred, dnbr, lc, lut, dn_thr, burn_thr):
    groups = lut[np.clip(lc, 0, 255)]
    for g in ("forest", "steppe", "cropland", "floodplain", "other"):
        m = (groups == g) & (pred > 0)
        if not m.any():
            continue
        c12 = float(dn_thr[g]["cut12"])
        c23 = float(dn_thr[g]["cut23"])
        bt = float(burn_thr[g])
        d = dnbr[m]
        new = np.ones(m.sum(), dtype=np.int64)
        new[d >= bt] = 1
        new[d >= c12] = 2
        new[d >= c23] = 3
        new[d < bt] = 0
        pred[m] = new
    return pred


def main():
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--af-ckpt", default="weights/af.pt")
    ap.add_argument("--af-thresh", default="weights/af_thresh.json")
    ap.add_argument("--bs-ckpt", default="weights/bs.pt")
    ap.add_argument("--bs-config", default="configs/bs-fallback.yaml")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    import torch
    import yaml
    from src.models.af_dataset import build_features
    from src.models.af_unet import build_af_unet
    from src.models.bs_dataset import assemble_features, build_group_lut, cloud_invalid_mask
    from src.models.bs_fallback import build_fallback_bs
    from src.models.bs_postproc import apply_postfilter

    random_seed = SEED
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    device = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"

    # ---- template: exact (chip_id, class_id) set ----
    tpl_path = os.path.join(a.data_dir, "sample_submission.csv")
    if not os.path.exists(tpl_path):
        tpl_path = os.path.join(a.data_dir, "test", "sample_submission.csv")
    rows = []
    with open(tpl_path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append((r["chip_id"].strip(), int(r["class_id"])))
    af_chips = sorted({c for c, k in rows if k == 1 and c.startswith("AF_")})
    # AF rows robust: any AF chip in template regardless of class filter
    af_chips = sorted({c for c, k in rows if c.startswith("AF_")})
    bs_chips = sorted({c for c, k in rows if c.startswith("BS_")})

    # ---- load AF ----
    import yaml as _y
    with open("configs/af.yaml") as fh:
        af_cfg = _y.safe_load(fh)
    with open(a.af_thresh) as fh:
        th = json.load(fh)
    thr_day = float(th["thr_day"])
    thr_night = float(th["thr_night"])
    gate = th.get("wc_gate", {})
    i4_min = gate.get("I4_min_K")

    af_model = build_af_unet(in_channels=12, encoder_weights=None, classes=1).to(device)
    sd = torch.load(a.af_ckpt, map_location=device, weights_only=False)
    af_model.load_state_dict(sd.get("state_dict", sd))
    af_model.eval()

    # ---- load BS ----
    with open(a.bs_config) as fh:
        bs_cfg = yaml.safe_load(fh)
    bs_model, _ = build_fallback_bs(bs_cfg)
    bs_model = bs_model.to(device)
    bsd = torch.load(a.bs_ckpt, map_location=device, weights_only=False)
    bs_model.load_state_dict(bsd.get("model", bsd))
    bs_model.eval()
    lut = build_group_lut(bs_cfg["landcover_groups"])
    cg = bs_cfg["cloud_gate"]
    cc = list(cg["cloud_codes"])
    dso = bool(cg.get("dry_soil_override", True))
    dlc = int(cg.get("dry_soil_landcover", 60))
    drlx = tuple(cg.get("dry_soil_relax_codes", [8, 9]))
    use_amp = (device != "cpu")
    # calib (configs/bs-calib.json): scaled dNBR cuts + remap + morph
    try:
        cal = json.load(open("configs/bs-calib.json"))
        _b = cal["best"]
        _bs, _cs = float(_b["burn_scale"]), float(_b["cut_scale"])
        _snap, _remap = bool(_b["snap"]), bool(_b["remap"])
        _morph = int(cal.get("morph_min_px", 0))
        _dn = {g: {"cut12": bs_cfg["dnbr_thresholds"][g]["cut12"] * _cs,
                   "cut23": bs_cfg["dnbr_thresholds"][g]["cut23"] * _cs} for g in _dn_keys(bs_cfg)}
        _bt = {g: bs_cfg["burn_thresholds"][g] * _bs for g in _dn_keys(bs_cfg)}
    except Exception:
        _dn, _bt, _snap, _remap, _morph = (bs_cfg["dnbr_thresholds"], bs_cfg["burn_thresholds"],
                                           True, False, 0)
    if _morph > 0:
        from scipy.ndimage import binary_fill_holes, label as _label

    out = {}
    # ---- AF loop ----
    with torch.no_grad():
        for chip in af_chips:
            v, ax = load_af(a.data_dir, chip)
            x_np, _ = build_features(v, ax, af_cfg["norm"])
            med_solz = float(np.nanmedian(v[5]))
            if np.isnan(med_solz):
                med_solz = 0.0
            thr = thr_night if med_solz > 90.0 else thr_day
            x = torch.from_numpy(x_np).unsqueeze(0).to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                prob = torch.sigmoid(af_model(x)).cpu().numpy()[0, 0]
            pred = (prob >= thr).astype(np.uint8)
            if i4_min is not None:  # wc gate lc==50 needs I4>=310K
                lc = ax[0]
                i4 = v[3].astype(np.float64)
                with np.errstate(invalid="ignore"):
                    nosup = ~np.isfinite(i4) | (i4 < float(i4_min))
                pred[(lc == 50) & (pred > 0) & nosup] = 0
            out[(chip, 1)] = rle_encode(pred)
    # ---- BS loop ----
    with torch.no_grad():
        for chip in bs_chips:
            pre, post, s1pre, s1post, aux = load_bs(a.data_dir, chip)
            x_np, dnbr, rdnbr, scl, lc = assemble_features(
                pre, post, s1pre, s1post, aux, bs_cfg["norm"], float(bs_cfg.get("nbr_eps", 1e-6)))
            x = torch.from_numpy(x_np).unsqueeze(0).to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = bs_model(x)
                logits = logits.output if hasattr(logits, "output") else logits
            lg = logits.float().cpu().numpy()[0]
            pred = apply_postfilter(lg, dnbr, scl, lc, lut, _dn, _bt,
                                    cc, dso, dlc, drlx, snap_to_gate=_snap)
            if _remap:
                pred = _apply_remap(pred, dnbr, lc, lut, _dn, _bt)
            if _morph > 0:
                filled = binary_fill_holes(pred > 0)
                lab, nc = _label(filled)
                keep = np.zeros_like(filled, dtype=bool)
                if nc:
                    sizes = np.bincount(lab.ravel())
                    for c in range(1, nc + 1):
                        if sizes[c] >= _morph:
                            keep[lab == c] = True
                newp = np.zeros_like(pred)
                newp[keep] = pred[keep]
                pred = newp
            for k in (1, 2, 3):
                out[(chip, k)] = rle_encode((pred == k).astype(np.uint8))

    # ---- write in template order ----
    with open(a.output, "w", newline="") as fh:
        w = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["chip_id", "class_id", "rle"])
        for chip, k in rows:
            w.writerow([chip, k, out.get((chip, k), "")])
    wall = time.time() - t0
    print(f"INFERENCE-DONE chips(AF={len(af_chips)},BS={len(bs_chips)}) rows={len(rows)} "
          f"wall={wall:.1f}s output={a.output}", flush=True)


if __name__ == "__main__":
    main()
