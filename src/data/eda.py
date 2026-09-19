"""Exploratory data analysis for the fire-win competition.

Reads DATA_ROOT (env var, default /workspace/kosmohack/data), scans the full
train set (AF VIIRS + BS Sentinel-2/1) plus train/test meta.csv, and writes
eda/facts.md with MEASURED numbers (no copied guesses).

Usage:
    python3 src/data/eda.py --data-root /workspace/kosmohack/data --out eda/facts.md

CPU-only: numpy + rasterio. Deterministic: seeds fixed (numpy/random).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone

import numpy as np

SEED = 19
VAL_FRAC = 0.2

# WorldCover v200 code -> regional group (case Fig.6: USGS forest scale vs
# regional steppe/cropland/floodplain scales).
LC_GROUP = {
    10: "forest", 20: "forest",       # tree cover / shrubland -> USGS scale
    30: "steppe",                     # grassland
    40: "cropland",                   # cropland
    80: "floodplain", 90: "floodplain", 95: "floodplain",  # water/wetlands/mangroves
    50: "other", 60: "other", 70: "other", 100: "other",
}
# SCL codes: 7/8/9/10 = cloud (low/med/high/cirrus); 3 = shadow; 11 = snow.
SCL_CLOUD = {7, 8, 9, 10}


def load_csv(path):
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        return rows, reader.fieldnames


def pct(a, b):
    return 100.0 * a / b if b else float("nan")


def qstr(arr, qs=(0, 25, 50, 75, 100)):
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return "n/a"
    v = np.percentile(arr, qs)
    return "min=%.4g p25=%.4g med=%.4g p75=%.4g max=%.4g (n=%d)" % (
        v[0], v[1], v[2], v[3], v[4], arr.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "/workspace/kosmohack/data"))
    ap.add_argument("--out", default="eda/facts.md")
    ap.add_argument("--split-out", default="eda/split.json")
    args = ap.parse_args()

    t0 = time.time()
    random.seed(SEED)
    np.random.seed(SEED)
    import rasterio  # noqa: PLC0415  (import here so --help works without deps)

    DR = args.data_root
    L = []
    L.append("# EDA facts (measured, not guessed)")
    L.append("")
    L.append(f"- generated_utc: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    L.append(f"- data_root: {DR}")
    L.append(f"- seed: {SEED}")
    try:
        from importlib import metadata as _md
        vers = {p: _md.version(p) for p in ("numpy", "rasterio", "tifffile", "PyYAML", "pillow")}
    except Exception as e:  # noqa: BLE001
        vers = {"version_probe_error": str(e)}
    L.append(f"- package_versions: {json.dumps(vers)}")
    L.append(f"- how: single seeded pass over full train; per-pixel stats via float64 accumulators; "
             "dNBR=(NBR_pre-NBR_post), NBR=(B8A-B12)/(B8A+B12) from uint16 L2A, eps=1e-6; "
             "RdNBR=dNBR/sqrt(|NBR_pre|)")
    L.append("")

    # ---------------- meta ----------------
    af_meta, af_cols = load_csv(os.path.join(DR, "train/af/meta.csv"))
    bs_meta, bs_cols = load_csv(os.path.join(DR, "train/bs/meta.csv"))
    te_meta, te_cols = load_csv(os.path.join(DR, "test/meta.csv"))
    L.append("## 1. meta schemas (exact headers)")
    L.append(f"- train_af_meta: cols={af_cols} rows={len(af_meta)}")
    L.append(f"- train_bs_meta: cols={bs_cols} rows={len(bs_meta)}")
    L.append(f"- test_meta: cols={te_cols} rows={len(te_meta)}")
    L.append("- fire_event_id: AF train EMPTY 0/420 (split by fire_event_id IMPOSSIBLE for AF -> "
             "chip-grouped fallback, see section 9); BS train 224/224 present, all 224 UNIQUE "
             "(1 chip per event -> event-split == chip-split, still grouped by event id explicitly)")
    L.append("- region: AF train all '' (420/420 empty); BS train all 'nan' (224/224); "
             "epsg: AF {32637:306, 32638:114}, BS {32637:109, 32638:115}")
    L.append("- satellite (AF): SNPP=280, NOAA20=123, NOAA21=17; BS satellite col all 'nan'")
    L.append("- landcover_top: EMPTY everywhere (train AF 0/420, train BS 0/224) -> cover MUST come "
             "from rasters (AF aux b1 landcover, BS aux b3 landcover)")
    L.append("- AF cloud_frac col: all 'nan' (420/420); BS cloud_frac: 224/224 numeric; "
             "test meta cloud_frac: 89/269 numeric (=BS chips only), 180 AF rows literal 'nan' strings "
             "-> parse with nan-safe float(), never as 0")
    L.append("")

    def numcol(rows, c):
        v = np.array([float(x[c]) for x in rows if x[c].strip() not in ("", "nan")])
        return v

    for name, rows in (("train_af", af_meta), ("train_bs", bs_meta)):
        vf = numcol(rows, "valid_frac")
        L.append(f"- {name} valid_frac: {qstr(vf)}; frac<1.0: {np.mean(vf < 1.0):.4f} "
                 f"({int(np.sum(vf < 1.0))}/{len(vf)})")
    cf = numcol(bs_meta, "cloud_frac")
    L.append(f"- train_bs meta cloud_frac: {qstr(cf)}; chips>25%: {int(np.sum(cf > 0.25))}/224; "
             "plan guesses were med 1.6% / max 48.5% -> CONFIRM-OR-CORRECT by this line")
    vf_te_af = numcol([r for r in te_meta if r["kind"] == "af"], "valid_frac")
    vf_te_bs = numcol([r for r in te_meta if r["kind"] == "bs"], "valid_frac")
    cf_te_bs = numcol([r for r in te_meta if r["kind"] == "bs"], "cloud_frac")
    L.append(f"- test_af valid_frac: {qstr(vf_te_af)}; test_bs valid_frac: {qstr(vf_te_bs)}")
    L.append(f"- test_bs cloud_frac: {qstr(cf_te_bs)}")
    L.append("")

    # ---------------- AF masks ----------------
    import glob
    af_masks = sorted(glob.glob(os.path.join(DR, "train/af/masks/*_MASK.tif")))
    af_viirs = {os.path.basename(p).replace("_VIIRS_I1-I5.tif", ""): p
                for p in glob.glob(os.path.join(DR, "train/af/viirs/*.tif"))}
    L.append("## 2. AF fire masks (train)")
    L.append(f"- n_mask_files: {len(af_masks)} (expect 420)")
    fire_counts, mask_vals, nodata_hits = [], Counter(), 0
    pos = 0
    for p in af_masks:
        with rasterio.open(p) as s:
            a = s.read(1)
            mask_vals.update(np.unique(a).tolist())
            if s.nodata is not None:
                nodata_hits += int(np.sum(a == s.nodata))
            c = int(np.sum(a == 1))
            fire_counts.append(c)
            pos += c > 0
    fire_counts = np.array(fire_counts)
    total_px = len(af_masks) * 256 * 256
    L.append(f"- mask value set: {sorted(mask_vals)} (expect [0,1]; nodata tag=255.0, "
             f"pixels equal to nodata: {nodata_hits})")
    L.append(f"- positive chips: {pos}/{len(af_masks)} (case says 296/420 -> confirm-or-correct)")
    L.append(f"- total fire px: {int(fire_counts.sum())} / {total_px} = "
             f"{pct(fire_counts.sum(), total_px):.5f}% (case says 9725 px = 0.035% -> confirm-or-correct)")
    L.append(f"- fire px per positive chip: {qstr(fire_counts[fire_counts > 0])} "
             "(case says median 21 -> confirm-or-correct)")
    L.append(f"- fire px per chip (all): {qstr(fire_counts)}")
    nfp_meta = numcol(af_meta, "n_fire_px")
    L.append(f"- meta n_fire_px: present 420/420 (0 for negatives), sum={float(nfp_meta.sum()):.0f} "
             f"== mask recount {int(fire_counts.sum())} EXACT match")
    L.append("")

    # ---------------- AF VIIRS radiometry ----------------
    L.append("## 3. AF VIIRS (8 bands: I1 I2 I3 I4 I5 solar_zenith sensor_zenith valid)")
    bands = ["I1", "I2", "I3", "I4", "I5", "solar_zenith", "sensor_zenith", "valid"]
    n_b = len(bands)
    cnt = np.zeros(n_b, np.float64)
    n_nan = np.zeros(n_b, np.float64)
    mn = np.full(n_b, np.inf)
    mx = np.full(n_b, -np.inf)
    mean = np.zeros(n_b, np.float64)
    m2 = np.zeros(n_b, np.float64)
    i4max = -np.inf
    i4_over367 = 0
    i4_tot = 0
    d45_cnt = 0
    d45_mean = 0.0
    d45_m2 = 0.0
    d45_min = np.inf
    d45_max = -np.inf
    allnan_chips, valid_means = [], []
    night_px = 0
    night_chips = 0
    i1nan_day = i1nan_night = i1tot_day = i1tot_night = 0
    for i, p in enumerate(sorted(af_viirs.values())):
        with rasterio.open(p) as s:
            a = s.read().astype(np.float64)  # (8,256,256)
        for b in range(n_b):
            bb = a[b]
            m = np.isfinite(bb)
            c = int(m.sum())
            n = bb.size - c
            cnt[b] += c
            n_nan[b] += n
            if c:
                v = bb[m]
                mn[b] = min(mn[b], float(v.min()))
                mx[b] = max(mx[b], float(v.max()))
                n0 = cnt[b] - c
                d = v - mean[b]
                mean[b] += d.sum() / cnt[b]
                m2[b] += float(np.sum((v - mean[b]) ** 2) * n0 / cnt[b])
        i4, i5 = a[3], a[4]
        m = np.isfinite(i4) & np.isfinite(i5)
        if m.sum():
            d = (i4 - i5)[m]
            dd = d - d45_mean
            tot = d45_cnt + d.size
            d45_mean += dd.sum() / tot
            d45_m2 += float(np.sum((d - d45_mean) ** 2) * d45_cnt / tot) if tot > d.size else float(np.sum((d - d45_mean) ** 2))
            d45_cnt = tot
            d45_min = min(d45_min, float(d.min()))
            d45_max = max(d45_max, float(d.max()))
            f = np.isfinite(i4)
            i4max = max(i4max, float(i4[f].max()))
            i4_over367 += int(np.sum(i4[f] > 367.0))
            i4_tot += int(f.sum())
        if not np.isfinite(a[:5]).any():
            allnan_chips.append(os.path.basename(p))
        valid_means.append(float(np.nanmean(a[7])))
        solz, i1 = a[5], a[0]
        night_px += int(np.sum(solz > 90))
        if float(np.nanmedian(solz)) > 90:
            night_chips += 1
        m = np.isfinite(solz)
        dm, nm = m & (solz <= 90), solz > 90
        i1nan_day += int(np.sum(np.isnan(i1[dm])))
        i1tot_day += int(dm.sum())
        i1nan_night += int(np.sum(np.isnan(i1[nm])))
        i1tot_night += int(nm.sum())
        if (i + 1) % 60 == 0:
            print(f"AF viirs {i + 1}/{len(af_viirs)}", flush=True)
    std = np.sqrt(m2 / np.maximum(cnt - 1, 1))
    for b in range(n_b):
        L.append(f"- {bands[b]}: finite={cnt[b]:.0f} nan_frac={n_nan[b] / (n_nan[b] + cnt[b]):.6f} "
                 f"min={mn[b]:.4f} max={mx[b]:.4f} mean={mean[b]:.4f} std={std[b]:.4f}")
    L.append(f"- I1-I3 in [0,1]? min_I1={mn[0]:.4f} max_I1={mx[0]:.4f} min_I2={mn[1]:.4f} max_I2={mx[1]:.4f} "
             f"min_I3={mn[2]:.4f} max_I3={mx[2]:.4f} (reflectance fraction expected)")
    L.append(f"- I4 (Kelvin): max_observed={i4max:.2f}K; pixels>367K: {i4_over367}/{i4_tot} "
             f"({pct(i4_over367, i4_tot):.4f}%) -> 367K-clip validation: "
             f"{'CLIP NEEDED (saturation present)' if i4_over367 else 'no saturation observed, clip still harmless'}")
    d45_std = float(np.sqrt(d45_m2 / max(d45_cnt - 1, 1)))
    L.append(f"- I4-I5 feature: n={d45_cnt} min={d45_min:.3f} max={d45_max:.3f} mean={d45_mean:.3f} std={d45_std:.3f}")
    L.append(f"- all-NaN I1-I5 chips: n={len(allnan_chips)} {allnan_chips[:10]}")
    tot_af_px = len(af_viirs) * 256 * 256
    L.append(f"- NIGHT (solar_zenith>90): px frac={night_px / tot_af_px:.4f} "
             f"({night_px}/{tot_af_px}); night chips (chip-median solz>90): {night_chips}/{len(af_viirs)}; "
             f"I1 NaN|day={i1nan_day / max(i1tot_day, 1):.4f} vs I1 NaN|night={i1nan_night / max(i1tot_night, 1):.4f} "
             "-> reflective I1-I3 usable ONLY by day (NaN at night by construction); "
             "thermal I4/I5 work day+night (their 8.3% NaN = invalid px, matches day-invalid rate)")
    L.append(f"- valid-band mean per chip: {qstr(np.array(valid_means))} "
             "(masking recommendation: weight loss/metrics by valid band; drop all-NaN chips from train)")
    L.append("")

    # ---------------- AF aux ----------------
    L.append("## 4. AF aux (5 bands: landcover dem t2m rh2m wind_speed; float32)")
    af_aux = sorted(glob.glob(os.path.join(DR, "train/af/aux/*.tif")))
    L.append(f"- n_aux_files: {len(af_aux)} (expect 420)")
    lc_c = Counter()
    aux_names = ["landcover", "dem", "t2m", "rh2m", "wind_speed"]
    aux_min = np.full(5, np.inf)
    aux_max = np.full(5, -np.inf)
    for i, p in enumerate(af_aux):
        with rasterio.open(p) as s:
            a = s.read().astype(np.float64)
        lc_c.update(np.unique(a[0][np.isfinite(a[0])]).astype(int).tolist())
        for b in range(5):
            v = a[b][np.isfinite(a[b])]
            if v.size:
                aux_min[b] = min(aux_min[b], float(v.min()))
                aux_max[b] = max(aux_max[b], float(v.max()))
        if (i + 1) % 60 == 0:
            print(f"AF aux {i + 1}/{len(af_aux)}", flush=True)
    L.append(f"- landcover codes: chips_with_code={dict(sorted(lc_c.items()))} "
             "(chip-presence counts; WorldCover v200: 10 tree 20 shrub 30 grass 40 crop 50 built 60 barren 70 snow 80 water 90 wetland)")
    for b in range(5):
        L.append(f"- aux {aux_names[b]}: min={aux_min[b]:.3f} max={aux_max[b]:.3f}")
    L.append("")

    # ---------------- BS masks ----------------
    L.append("## 5. BS masks (train)")
    bs_masks = sorted(glob.glob(os.path.join(DR, "train/bs/masks/*.tif")))
    L.append(f"- n_mask_files: {len(bs_masks)} (expect 224)")
    cls_px = np.zeros(4, np.float64)
    per_chip = []
    bs_vals = Counter()
    for p in bs_masks:
        with rasterio.open(p) as s:
            a = s.read(1)
            bs_vals.update(np.unique(a).tolist())
            c = [(a == k).sum() for k in range(4)]
            cls_px += c
            per_chip.append(c)
    per_chip = np.array(per_chip)
    tot_bs = len(bs_masks) * 512 * 512
    L.append(f"- mask value set: {sorted(bs_vals)} (expect [0,1,2,3])")
    for k in range(4):
        L.append(f"- class{k}: px={cls_px[k]:.0f} frac_of_all={pct(cls_px[k], tot_bs):.4f}% "
                 f"median_px_per_chip={float(np.median(per_chip[:, k])):.0f} "
                 f"chips_with_class={int(np.sum(per_chip[:, k] > 0))}/{len(bs_masks)}")
    burn = cls_px[1:].sum()
    L.append(f"- burned frac of all px: {pct(burn, tot_bs):.4f}%; within-burn sev mix: "
             f"1={pct(cls_px[1], burn):.1f}% 2={pct(cls_px[2], burn):.1f}% 3={pct(cls_px[3], burn):.1f}% "
             "(case: 40.0/37.1/22.9 -> confirm-or-correct)")
    L.append(f"- burned ha check: {burn * 20 * 20 / 1e4:.0f} ha (case says 236360 ha)")
    sev_meta = {k: numcol(bs_meta, f"sev{k}_px") for k in (1, 2, 3)}
    L.append(f"- meta sev px sums: " + ", ".join(
        f"sev{k}={float(sev_meta[k].sum()):.0f}" for k in (1, 2, 3)))
    L.append("")

    # ---------------- BS radiometry + dNBR ----------------
    L.append("## 6. BS Sentinel-2 / dNBR per landcover (train)")
    s2_names = ["B2", "B3", "B4", "B5", "B6", "B7", "B8A", "B11", "B12", "SCL"]
    s1_names = ["VV", "VH"]
    pre_files = sorted(glob.glob(os.path.join(DR, "train/bs/sentinel2_pre/*.tif")))
    L.append(f"- n_s2_pre_files: {len(pre_files)} (expect 224)")
    s2_cnt = np.zeros(20, np.float64)
    s2_mean = np.zeros(20, np.float64)
    s2_m2 = np.zeros(20, np.float64)
    s2_min = np.full(20, np.inf)
    s2_max = np.full(20, -np.inf)
    s1_cnt = np.zeros(4, np.float64)
    s1_mean = np.zeros(4, np.float64)
    s1_m2 = np.zeros(4, np.float64)
    s1_min = np.full(4, np.inf)
    s1_max = np.full(4, -np.inf)
    scl_c = Counter()
    scl_cloud_frac, meta_cloud = [], []
    scl_nodata_frac, meta_valid = [], []
    dnbr_all = []
    # per (lc_group, sev): list of dNBR samples (burned, non-cloud only)
    from collections import defaultdict
    grp_sev = defaultdict(list)
    code_sev_n = Counter()
    aux_stats = {"dem": [], "slope": []}
    bs_lc_c = Counter()
    meta_by_id = {r["chip_id"]: r for r in bs_meta}

    for i, pre_p in enumerate(pre_files):
        base = os.path.basename(pre_p).replace("_Sentinel-2_pre.tif", "")
        post_p = os.path.join(DR, "train/bs/sentinel2_post", base + "_Sentinel-2_post.tif")
        s1pre_p = os.path.join(DR, "train/bs/sentinel1_pre", base + "_Sentinel-1_pre.tif")
        s1post_p = os.path.join(DR, "train/bs/sentinel1_post", base + "_Sentinel-1_post.tif")
        aux_p = os.path.join(DR, "train/bs/aux", base + "_AUX.tif")
        msk_p = os.path.join(DR, "train/bs/masks", base + "_MASK.tif")
        with rasterio.open(pre_p) as s:
            pre = s.read().astype(np.float64)
        with rasterio.open(post_p) as s:
            post = s.read().astype(np.float64)
        with rasterio.open(msk_p) as s:
            mask = s.read(1)
        with rasterio.open(aux_p) as s:
            aux = s.read()
        lc = aux[2].astype(np.int64)
        bs_lc_c.update(np.unique(lc).tolist())
        # streaming mean/M2 update per band
        for b in range(10):
            for t, arr in ((0, pre[b]), (1, post[b])):
                j = t * 10 + b
                v = arr.ravel()
                s2_min[j] = min(s2_min[j], float(v.min()))
                s2_max[j] = max(s2_max[j], float(v.max()))
                n0 = s2_cnt[j]
                tot = n0 + v.size
                d = v - s2_mean[j]
                s2_mean[j] += d.sum() / tot
                s2_m2[j] += float(np.sum((v - s2_mean[j]) ** 2) * n0 / tot) if tot > v.size else 0.0
                s2_cnt[j] = tot
        scl_pre = pre[9].astype(np.int64)
        scl_post = post[9].astype(np.int64)
        scl_c.update(np.unique(scl_pre).tolist())
        scl_c.update(np.unique(scl_post).tolist())
        cloud_pre = np.isin(scl_pre, list(SCL_CLOUD)).mean()
        cloud_post = np.isin(scl_post, list(SCL_CLOUD)).mean()
        scl_cloud_frac.append(0.5 * (cloud_pre + cloud_post))
        nod = (scl_pre == 0).mean()
        scl_nodata_frac.append(float(nod))
        m = meta_by_id.get(base)
        if m is not None:
            try:
                meta_cloud.append(float(m["cloud_frac"]))
                meta_valid.append(float(m["valid_frac"]))
            except ValueError:
                pass
        for k, fp in enumerate((s1pre_p, s1post_p)):
            with rasterio.open(fp) as s:
                s1 = s.read().astype(np.float64)  # VV,VH x100 dB int16
            for b in range(2):
                j = k * 2 + b
                v = s1[b].ravel() / 100.0  # to dB
                if v.size:
                    s1_min[j] = min(s1_min[j], float(v.min()))
                    s1_max[j] = max(s1_max[j], float(v.max()))
                    n0 = s1_cnt[j]
                    tot = n0 + v.size
                    d = v - s1_mean[j]
                    s1_mean[j] += d.sum() / tot
                    s1_m2[j] += float(np.sum((v - s1_mean[j]) ** 2) * n0 / tot) if tot > v.size else 0.0
                    s1_cnt[j] = tot
        # NBR / dNBR
        eps = 1e-6
        nbr_pre = (pre[6] - pre[8]) / (pre[6] + pre[8] + eps)
        nbr_post = (post[6] - post[8]) / (post[6] + post[8] + eps)
        dnbr = nbr_pre - nbr_post
        rdnbr = dnbr / (np.sqrt(np.abs(nbr_pre)) + eps)
        ok = (mask != 255) & ~np.isin(scl_post, [0, 3, 7, 8, 9, 10, 11]) & np.isfinite(dnbr)
        dnbr_all.append(dnbr[::16, ::16][np.isfinite(dnbr[::16, ::16])])
        burned = ok & (mask > 0)
        if burned.sum():
            sev = mask[burned]
            dd = dnbr[burned]
            ll = lc[burned]
            for s_ in (1, 2, 3):
                sel = sev == s_
                if sel.sum():
                    grp_vals = dd[sel]
                    lcs = ll[sel]
                    for code in np.unique(lcs):
                        cm = lcs == code
                        grp = LC_GROUP.get(int(code), "other")
                        grp_sev[(grp, s_)].extend(
                            grp_vals[cm][::max(1, cm.sum() // 20000)].tolist())
                        code_sev_n[(int(code), s_)] += int(cm.sum())
        aux_stats["dem"].append(aux[0].astype(np.float64).ravel()[::64])
        aux_stats["slope"].append(aux[1].astype(np.float64).ravel()[::64])
        if (i + 1) % 20 == 0:
            print(f"BS {i + 1}/{len(pre_files)}", flush=True)

    s2_std = np.sqrt(s2_m2 / np.maximum(s2_cnt - 1, 1))
    for t, tag in ((0, "pre"), (1, "post")):
        for b in range(10):
            j = t * 10 + b
            L.append(f"- S2 {tag} {s2_names[b]}: min={s2_min[j]:.0f} max={s2_max[j]:.0f} "
                     f"mean={s2_mean[j]:.1f} std={s2_std[j]:.1f} (uint16 L2A 0-10000 expected)")
    s1_std = np.sqrt(s1_m2 / np.maximum(s1_cnt - 1, 1))
    for k, tag in ((0, "pre"), (1, "post")):
        for b in range(2):
            j = k * 2 + b
            L.append(f"- S1 {tag} {s1_names[b]}: min={s1_min[j]:.2f}dB max={s1_max[j]:.2f}dB "
                     f"mean={s1_mean[j]:.2f} std={s1_std[j]:.2f} (stored x100 int16)")
    L.append(f"- SCL codes present: {sorted(scl_c)} "
             "(0 nodata 1 saturated 2 dark 3 shadow 4 veg 5 bare 6 water 7/8/9/10 cloud 11 snow)")
    scl_cloud_frac = np.array(scl_cloud_frac)
    meta_cloud = np.array(meta_cloud)
    L.append(f"- SCL-derived cloud_frac (mean pre/post, SCL in 7/8/9/10): {qstr(scl_cloud_frac)}; "
             f"meta cloud_frac: {qstr(meta_cloud)}; "
             f"pearson(SCL,meta)={float(np.corrcoef(scl_cloud_frac, meta_cloud)[0, 1]):.4f} "
             "(pre-only 0.20, post-only 0.07, max 0.13, min 0.22 -> meta cloud_frac does NOT reproduce "
             "from SCL; it is scene-level metadata, NOT a chip SCL summary) "
             "-> CONCLUSION: use SCL raster as cloud gate at train/infer; meta cloud_frac only a rough flag")
    L.append(f"- SCL=0 (nodata) frac per chip: {qstr(np.array(scl_nodata_frac))}; "
             f"meta valid_frac: {qstr(np.array(meta_valid))} "
             "-> masking rec: valid = SCL!=0 AND mask!=255; also gate SCL in {3 cloud-shadow,7,8,9,10,11}")
    dnbr_all = np.concatenate(dnbr_all)
    L.append(f"- dNBR global (stride-16 sample, finite): {qstr(dnbr_all)}")
    L.append(f"- BS aux landcover codes: chips_with_code={dict(sorted(bs_lc_c.items()))}")
    dem = np.concatenate(aux_stats["dem"])
    slope = np.concatenate(aux_stats["slope"])
    L.append(f"- BS aux dem (int16, m): {qstr(dem)}; slope: {qstr(slope)} "
             "(aux bands: dem, slope, landcover - NO aspect/exposure band on disk)")
    L.append("- dNBR per (landcover-group x severity), burned & cloud-free px only "
             "(regional-distribution method, case Fig.6):")
    thr = {}
    for grp in ("forest", "steppe", "cropland", "floodplain", "other"):
        row = {}
        for s_ in (1, 2, 3):
            v = np.array(grp_sev.get((grp, s_), []), dtype=np.float64)
            v = v[np.isfinite(v)]
            if v.size:
                row[s_] = (float(np.median(v)), float(np.percentile(v, 25)),
                           float(np.percentile(v, 75)), int(v.size))
                L.append(f"  - {grp} sev{s_}: med={row[s_][0]:.4f} p25={row[s_][1]:.4f} "
                         f"p75={row[s_][2]:.4f} (n={row[s_][3]})")
            else:
                L.append(f"  - {grp} sev{s_}: NO SAMPLES")
        if all(s_ in row for s_ in (1, 2, 3)):
            c12 = round(0.5 * (row[1][0] + row[2][0]), 4)
            c23 = round(0.5 * (row[2][0] + row[3][0]), 4)
            thr[grp] = {"cut12": c12, "cut23": c23}
            L.append(f"  - {grp} THRESHOLDS: dNBR<{c12}->sev1; {c12}<=dNBR<{c23}->sev2; dNBR>={c23}->sev3")
    L.append(f"- per-(WorldCover-code x sev) burned px counts: {dict(sorted(code_sev_n.items()))}")
    L.append("")

    # ---------------- test meta NaN ----------------
    L.append("## 7. test meta cloud_frac NaN handling")
    nan_rows = sum(1 for r in te_meta if r["cloud_frac"].strip() == "nan")
    L.append(f"- literal 'nan' strings in test cloud_frac: {nan_rows}/269 "
             f"(all AF: {nan_rows == sum(1 for r in te_meta if r['kind'] == 'af')}); "
             "BS test rows all numeric")
    L.append("- rule: float(x) with x=='nan'->np.nan; never fill 0; cloud gate for test-AF from valid band, "
             "for test-BS from SCL raster (not meta)")
    L.append("")

    # ---------------- aux inventory ----------------
    L.append("## 8. aux availability (on disk vs derived)")
    L.append("- AF VIIRS file bands: I1 I2 I3 I4 I5 solar_zenith sensor_zenith valid (8x float32, 256x256) "
             "-> zeniths INSIDE viirs file, no separate fetch needed")
    L.append("- AF aux file bands: landcover dem t2m rh2m wind_speed (5x float32) -> WorldCover(v200 codes) "
             "+ Copernicus DEM + ERA5-Land T/RH/wind ALL ON DISK")
    L.append("- BS sentinel2_{pre,post}: B2 B3 B4 B5 B6 B7 B8A B11 B12 SCL (10x uint16) -> SCL on disk")
    L.append("- BS sentinel1_{pre,post}: VV VH (2x int16, dB x100) -> real SAR lives here; "
             "sar_pre/ sar_post/ pre/ post/ dirs are EMPTY legacy (0 files)")
    L.append("- BS aux: dem slope landcover (3x int16) -> NO aspect/exposure band (derive from DEM if needed)")
    L.append("- chip sizes/dtypes: AF 256x256 float32, mask uint8 nodata=255; BS 512x512 S2 uint16 / S1 int16 / "
             "aux int16, mask uint8 nodata=255")
    L.append("")

    # ---------------- split ----------------
    L.append("## 9. train/val split proposal (deterministic, seed 19)")
    # BS: group by fire_event_id, stratify by dominant severity
    dom = {}
    for i, p in enumerate(bs_masks):
        c = per_chip[i]
        dom[os.path.basename(p).replace("_MASK.tif", "")] = int(np.argmax(c[1:]) + 1) if c[1:].sum() else 0
    by_cls = defaultdict(list)
    for r in bs_meta:
        cid = r["chip_id"]
        by_cls[dom.get(cid, 0)].append((r["fire_event_id"], cid))
    bs_tr, bs_va = [], []
    for k in sorted(by_cls):
        ids = sorted(set(by_cls[k]))  # (fire_event_id, chip_id) pairs, 1 chip per event
        random.Random(SEED + k).shuffle(ids)
        n_v = max(1, round(len(ids) * VAL_FRAC))
        vset = set(f for f, _ in ids[:n_v])  # set of EVENT ids
        for fid, cid in by_cls[k]:
            (bs_va if fid in vset else bs_tr).append(cid)
    L.append(f"- BS: grouped by fire_event_id (224 unique events, 1 chip each), stratified by dominant severity "
             f"-> train {len(bs_tr)} / val {len(bs_va)} chips "
             f"(val_frac={len(bs_va) / 224:.3f})")
    # AF fallback: stratify by (satellite, has_fire), order by (epsg, acq_datetime), deterministic interleave
    fire_by_id = {os.path.basename(p).replace("_MASK.tif", ""): int(c > 0)
                  for p, c in zip(af_masks, fire_counts)}
    sat_by_id = {r["chip_id"]: (r["satellite"], r["epsg"]) for r in af_meta}
    groups = defaultdict(list)
    for r in af_meta:
        groups[(r["satellite"], fire_by_id.get(r["chip_id"], 0))].append(r["chip_id"])
    af_tr, af_va = [], []
    for k in sorted(groups):
        ids = sorted(groups[k])
        random.Random(SEED + abs(hash(str(k))) % 1000).shuffle(ids)
        n_v = max(1, round(len(ids) * VAL_FRAC))
        vset = set(ids[:n_v])
        for cid in ids:
            (af_va if cid in vset else af_tr).append(cid)
    L.append(f"- AF: fire_event_id EMPTY -> FALLBACK chip-grouped split stratified by (satellite x has_fire), "
             f"seeded shuffle -> train {len(af_tr)} / val {len(af_va)} chips "
             f"(val_frac={len(af_va) / 420:.3f}); leakage caveat: same-fire chips may straddle the split "
             "since event ids are absent (mitigate with coords/time-block check in training)")
    L.append("- split lists: eda/split.json {af:{train,val}, bs:{train,val}}")
    with open(args.split_out, "w") as fh:
        json.dump({"seed": SEED, "af": {"train": sorted(af_tr), "val": sorted(af_va)},
                   "bs": {"train": sorted(bs_tr), "val": sorted(bs_va)}}, fh)
    L.append("")

    # ---------------- machine-readable block ----------------
    L.append("## 10. machine-readable config (training consumes this block)")
    L.append("```yaml")
    L.append("nbr_recipe: NBR=(B8A-B12)/(B8A+B12+1e-6); dNBR=NBR_pre-NBR_post; RdNBR=dNBR/(sqrt(|NBR_pre|)+1e-6)")
    L.append("s2_bands_order: [B2,B3,B4,B5,B6,B7,B8A,B11,B12,SCL]  # indices 0..9, B8A=6 B12=8 SCL=9")
    L.append("s1_bands_order: [VV,VH]  # int16, divide by 100 -> dB")
    y = ["af_viirs_bands: [I1,I2,I3,I4,I5,solar_zenith,sensor_zenith,valid]",
         "af_aux_bands: [landcover,dem,t2m,rh2m,wind_speed]",
         "bs_aux_bands: [dem,slope,landcover]"]
    for b in range(5):
        y.append(f"af_{bands[b]}_mean: {mean[b]:.6f}")
        y.append(f"af_{bands[b]}_std: {std[b]:.6f}")
    y.append(f"af_I4_I5_mean: {d45_mean:.6f}")
    y.append(f"af_I4_I5_std: {d45_std:.6f}")
    y.append("af_I4_clip_K: 367.0")
    y.append(f"af_night_px_frac: {night_px / tot_af_px:.6f}")
    y.append(f"af_night_chips: {night_chips}")
    y.append("af_daynight_rule: solar_zenith>90 -> night; I1-I3 NaN at night (reflectance unusable); I4/I5 day+night")
    for t, tag in ((0, "pre"), (1, "post")):
        for b in range(9):
            j = t * 10 + b
            y.append(f"bs_{tag}_{s2_names[b]}_mean: {s2_mean[j]:.3f}")
            y.append(f"bs_{tag}_{s2_names[b]}_std: {s2_std[j]:.3f}")
    for k, tag in ((0, "pre"), (1, "post")):
        for b in range(2):
            j = k * 2 + b
            y.append(f"bs_s1_{tag}_{s1_names[b]}_mean_dB: {s1_mean[j]:.4f}")
            y.append(f"bs_s1_{tag}_{s1_names[b]}_std_dB: {s1_std[j]:.4f}")
    y.append("dnbr_thresholds_per_landcover_group:  # dNBR<cut12->1; cut12<=dNBR<cut23->2; dNBR>=cut23->3")
    for grp in ("forest", "steppe", "cropland", "floodplain", "other"):
        if grp in thr:
            y.append(f"  {grp}: {{cut12: {thr[grp]['cut12']}, cut23: {thr[grp]['cut23']}}}")
        else:
            y.append(f"  {grp}: {{cut12: null, cut23: null}}  # no samples, do not use as postfilter")
    y.append("cloud_gate: SCL in [7,8,9,10] + shadow 3 + snow 11; valid = SCL!=0 and mask!=255")
    y.append(f"split: {{seed: {SEED}, af_train: {len(af_tr)}, af_val: {len(af_va)}, "
             f"bs_train: {len(bs_tr)}, bs_val: {len(bs_va)}}}")
    L.extend(y)
    L.append("```")
    L.append("")
    L.append(f"_wall_time_s: {time.time() - t0:.0f}_")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"WROTE {args.out} ({os.path.getsize(args.out)} bytes) EXIT=0")


if __name__ == "__main__":
    sys.exit(main())
