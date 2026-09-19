"""AF threshold polish (slot2, inference-only — NEVER retrains af.pt).

Loads frozen weights/af.pt, runs val inference (85 chips, seed 19), then:

1. GLOBAL sweep  thr 0.05..0.95 step 0.05  -> micro-F1 table (all val).
2. DAY/NIGHT sweeps (chip-median solar_zenith > 90 -> night, facts §3/§10)
   -> thr_day + thr_night (argmax micro-F1 within each subset).
3. WorldCover gate vs gas flares: suppress positives on built-up pixels
   (WorldCover v200 code 50, aux band b0) unless thermal support holds.
   Gate variants compared by micro-F1 table, best kept.

Metric notes (docs/case.md §7):
  - micro-F1 = pool TP/FP/FN over ALL chips first, then P/R/F1
    (NOT mean-of-per-chip-F1; the mean is logged separately, labelled).
  - zero-denom: (TP+FP)==0 -> P=1.0; (TP+FN)==0 -> R=1.0; P+R==0 -> F1=0.0.

RLE rule for the future submit (fixed here, encoder lives in inference):
  thresholds apply PER-PIXEL on the prob map; touching positive runs are
  MERGED at RLE-encoding time (case §6: series must not touch), i.e. the
  binary mask from thr_day/thr_night (+wc gate) is encoded as-is with
  merge-touching-runs.

VRAM discipline (shared cuda:0 with heavy BS slot): batch <= 8 (default 4),
num_workers 2, no grad, inference only.

Usage (remote, from /workspace/kosmohack):
  python src/models/af_thresholds.py --config configs/af.yaml
Output: weights/af_thresh.json + logs/af_thresholds.log
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.af_dataset import AFDataset
from src.models.af_unet import build_af_unet
from src.models.datasets_common import load_config, load_split, make_logger, read_tif, set_seed
from src.models.metrics import counts_from_masks, micro_f1, prf_from_counts

WC_BUILT = 50  # WorldCover v200 built-up (flare/industrial proxy)

# (gate_name, description, d_min, i4_min, hard_drop)
# d_min/i4_min apply ONLY on lc==WC_BUILT positives; None = no requirement.
# hard_drop=True drops ALL lc==50 positives unconditionally.
GATE_VARIANTS = [
    ("none", "no gate", None, None, False),
    ("hard50", "drop all lc==50 positives", None, None, True),
    ("b50_d5", "lc==50 requires I4-I5>=5K", 5.0, None, False),
    ("b50_d10", "lc==50 requires I4-I5>=10K", 10.0, None, False),
    ("b50_i4_310", "lc==50 requires I4>=310K", None, 310.0, False),
    ("b50_i4_310_d5", "lc==50 requires I4>=310K & I4-I5>=5K", 5.0, 310.0, False),
]


def resolve(p: str, root: str) -> str:
    return p if os.path.isabs(p) else os.path.join(root, p)


def chip_is_night(data_root: str, chip: str) -> bool:
    """Night <=> chip-median solar_zenith > 90 (facts §3: 127/420 night chips)."""
    v = read_tif(os.path.join(data_root, "train/af/viirs", f"{chip}_VIIRS_I1-I5.tif"))
    solz = v[5]
    med = float(np.nanmedian(solz))
    return bool(med > 90.0)


def chip_aux(data_root: str, chip: str) -> tuple:
    """Return (landcover_codes, I4_K, dI45_K) raw rasters for gating."""
    v = read_tif(os.path.join(data_root, "train/af/viirs", f"{chip}_VIIRS_I1-I5.tif"))
    a = read_tif(os.path.join(data_root, "train/af/aux", f"{chip}_AUX.tif"))
    lc = a[0]
    i4 = v[3].astype(np.float64)
    d = (v[3] - v[4]).astype(np.float64)
    return lc, i4, d


def apply_gate(pred: np.ndarray, lc: np.ndarray, i4: np.ndarray, d: np.ndarray,
               d_min, i4_min, hard_drop: bool) -> np.ndarray:
    """Suppress predicted positives on lc==WC_BUILT without thermal support.

    NaN I4/d counts as NO support (pixel suppressed under any gated variant).
    """
    pred = (np.asarray(pred) > 0).astype(np.uint8)
    if hard_drop:
        out = pred.copy()
        out[lc == WC_BUILT] = 0
        return out
    if d_min is None and i4_min is None:
        return pred
    support = np.ones_like(pred, dtype=bool)
    with np.errstate(invalid="ignore"):
        if d_min is not None:
            support &= np.isfinite(d) & (d >= d_min)
        if i4_min is not None:
            support &= np.isfinite(i4) & (i4 >= i4_min)
    out = pred.copy()
    built = (lc == WC_BUILT)
    out[built & (pred > 0) & (~support)] = 0
    return out


def sweep(probs, gts, valids, thresholds) -> list:
    rows = []
    for thr in thresholds:
        preds = [(np.asarray(p) >= thr).astype(np.uint8) for p in probs]
        prec, rec, f1, (tp, fp, fn) = micro_f1(preds, gts, valids)
        rows.append((float(thr), prec, rec, f1, tp, fp, fn))
    return rows


def mean_chip_f1(preds, gts, valids) -> float:
    """MEAN-of-per-chip F1 (diagnostic only — NOT the selection metric)."""
    fs = []
    for p, g, v in zip(preds, gts, valids):
        tp, fp, fn = counts_from_masks(p, g, v)
        _, _, f = prf_from_counts(tp, fp, fn)
        fs.append(f)
    return float(np.mean(fs)) if fs else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()
    assert args.batch_size <= 8, "slot2 VRAM discipline: batch <= 8"

    cfg = load_config(args.config)
    seed = 19
    set_seed(seed, True)
    cwd = os.getcwd()
    data_root = args.data_root or cfg.get("data_root", "/workspace/kosmohack/data")
    ckpt_path = args.checkpoint or resolve(cfg["outputs"]["checkpoint"], cwd)
    out_path = args.out or resolve("weights/af_thresh.json", cwd)
    log_path = resolve("logs/af_thresholds.log", cwd)

    logger = make_logger(log_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"AF-thresholds start: seed={seed} device={device} batch={args.batch_size} ckpt={ckpt_path}")
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        logger.info(f"cuda mem: free={free/2**30:.1f}G total={total/2**30:.1f}G (shared with BS slot)")

    split = load_split(resolve(cfg["split_path"], cwd), cfg.get("split_group", "af"))
    val_ids = split["val"]
    logger.info(f"val chips: {len(val_ids)} (expect 85)")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = cfg["model"]
    model = build_af_unet(
        in_channels=12,
        encoder=model_cfg.get("encoder", "timm-efficientnet-b4"),
        encoder_weights=None,  # type: ignore[arg-type]  # frozen ckpt — no ImageNet download; smp accepts None
        classes=int(model_cfg.get("classes", 1)),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    logger.info(f"ckpt loaded: best_val_f1={ckpt.get('best_val_f1')} @thr={ckpt.get('best_thr')} ep={ckpt.get('best_epoch')}")

    val_ds = AFDataset(val_ids, data_root, cfg["norm"], train=False, seed=seed)
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    probs, gts, valids, chips = [], [], [], []
    with torch.no_grad():
        for x, y, v, c in loader:
            p = torch.sigmoid(model(x.to(device))).cpu().numpy()[:, 0]
            for i in range(p.shape[0]):
                probs.append(p[i])
                gts.append(y.numpy()[i, 0])
                valids.append(v.numpy()[i])
                chips.append(c[i])
    logger.info(f"inference done: {len(chips)} chips")

    # ---- day/night split (chip-median solz>90) ----
    is_night = [chip_is_night(data_root, c) for c in chips]
    day_idx = [i for i, n in enumerate(is_night) if not n]
    night_idx = [i for i, n in enumerate(is_night) if n]
    logger.info(f"day/night split: day={len(day_idx)} night={len(night_idx)} chips "
                f"(facts train-wide: 127/420 night)")

    thresholds = [round(0.05 * k, 2) for k in range(1, 20)]  # 0.05..0.95

    def sub(lst, idx):
        return [lst[i] for i in idx]

    # ---- 1. GLOBAL sweep (micro-F1, all val) ----
    logger.info("SWEEP GLOBAL (val MICRO-F1 pooled over all chips):")
    grows = sweep(probs, gts, valids, thresholds)
    for thr, p, r, f, tp, fp, fn in grows:
        logger.info(f"  thr={thr:.2f} P={p:.4f} R={r:.4f} F1_micro={f:.4f} (TP={tp} FP={fp} FN={fn})")
    gbest = max(grows, key=lambda r: (r[3], -r[0]))

    # ---- 2. DAY sweep ----
    logger.info("SWEEP DAY (val MICRO-F1 pooled over day chips only):")
    drows = sweep(sub(probs, day_idx), sub(gts, day_idx), sub(valids, day_idx), thresholds)
    for thr, p, r, f, tp, fp, fn in drows:
        logger.info(f"  thr={thr:.2f} P={p:.4f} R={r:.4f} F1_micro={f:.4f} (TP={tp} FP={fp} FN={fn})")
    dbest = max(drows, key=lambda r: (r[3], -r[0]))
    thr_day = dbest[0]

    # ---- 3. NIGHT sweep ----
    logger.info("SWEEP NIGHT (val MICRO-F1 pooled over night chips only):")
    nrows = sweep(sub(probs, night_idx), sub(gts, night_idx), sub(valids, night_idx), thresholds)
    for thr, p, r, f, tp, fp, fn in nrows:
        logger.info(f"  thr={thr:.2f} P={p:.4f} R={r:.4f} F1_micro={f:.4f} (TP={tp} FP={fp} FN={fn})")
    nbest = max(nrows, key=lambda r: (r[3], -r[0]))
    thr_night = nbest[0]

    logger.info(f"PICK thr_day={thr_day:.2f} (F1_day_micro={dbest[3]:.4f}) "
                f"thr_night={thr_night:.2f} (F1_night_micro={nbest[3]:.4f}) "
                f"| global best was thr={gbest[0]:.2f} F1={gbest[3]:.4f}")

    # ---- split-threshold combined (no gate yet) ----
    split_preds = []
    for i, p in enumerate(probs):
        thr = thr_night if is_night[i] else thr_day
        split_preds.append((np.asarray(p) >= thr).astype(np.uint8))
    p_all, r_all, f_all, (tp_all, fp_all, fn_all) = micro_f1(split_preds, gts, valids)
    f_mean = mean_chip_f1(split_preds, gts, valids)
    logger.info(f"SPLIT-MIX no-gate: MICRO P={p_all:.4f} R={r_all:.4f} F1={f_all:.4f} "
                f"(TP={tp_all} FP={fp_all} FN={fn_all}) | MEAN-chip-F1={f_mean:.4f} (diagnostic, NOT selection metric)")

    # ---- 4. WorldCover gate variants (on top of thr_day/thr_night) ----
    aux_cache = {c: chip_aux(data_root, c) for c in chips}
    logger.info("WC-GATE variants (gate on lc==50 built-up only, thermal support from raw I4/I5):")
    gate_rows = []
    for name, desc, d_min, i4_min, hard in GATE_VARIANTS:
        gated = []
        for i, pr in enumerate(split_preds):
            lc, i4, d = aux_cache[chips[i]]
            gated.append(apply_gate(pr, lc, i4, d, d_min, i4_min, hard))
        p, r, f, (tp, fp, fn) = micro_f1(gated, gts, valids)
        # day/night sub-F1 under this gate
        _, _, f_d, _ = micro_f1(sub(gated, day_idx), sub(gts, day_idx), sub(valids, day_idx))
        _, _, f_n, _ = micro_f1(sub(gated, night_idx), sub(gts, night_idx), sub(valids, night_idx))
        gate_rows.append((name, desc, p, r, f, f_d, f_n, tp, fp, fn, d_min, i4_min, hard))
        logger.info(f"  gate={name:12s} P={p:.4f} R={r:.4f} F1_micro={f:.4f} "
                    f"(F1_day={f_d:.4f} F1_night={f_n:.4f} TP={tp} FP={fp} FN={fn}) # {desc}")
    best_gate = max(gate_rows, key=lambda r: (r[4], r[0] == "none"))
    # key = (F1_micro, is_none): strict F1 improvement wins; exact tie keeps 'none' (simpler).
    bname, bdesc, bp, br, bf, bf_d, bf_n, btp, bfp, bfn, bd_min, bi4_min, bhard = best_gate
    logger.info(f"GATE-PICK {bname}: F1_micro={bf:.4f} (day={bf_d:.4f} night={bf_n:.4f}) # {bdesc}")

    final_gated = []
    for i, pr in enumerate(split_preds):
        lc, i4, d = aux_cache[chips[i]]
        final_gated.append(apply_gate(pr, lc, i4, d, bd_min, bi4_min, bhard))
    final_mean = mean_chip_f1(final_gated, gts, valids)
    logger.info(f"FINAL: MICRO P={bp:.4f} R={br:.4f} F1={bf:.4f} | MEAN-chip-F1={final_mean:.4f} (diagnostic)")

    out = {
        "ckpt": os.path.basename(ckpt_path),
        "seed": seed,
        "val_chips": len(val_ids),
        "day_chips": len(day_idx),
        "night_chips": len(night_idx),
        "day_rule": "chip-median solar_zenith>90 -> night",
        "thr_day": thr_day,
        "thr_night": thr_night,
        "thr_global_best": {"thr": gbest[0], "F1_micro": round(gbest[3], 4)},
        "wc_gate": {"name": bname, "desc": bdesc, "lc_code": WC_BUILT,
                    "dI45_min_K": bd_min, "I4_min_K": bi4_min, "hard_drop": bhard},
        "F1_day": round(bf_d, 4),
        "F1_night": round(bf_n, 4),
        "F1_all_micro": round(bf, 4),
        "counts": {"TP": btp, "FP": bfp, "FN": bfn},
        "metric": "micro-F1 pooled over val chips (case §7 zero-denom); "
                  "thresholds+gate selected by argmax micro-F1",
        "rle_rule": "thresholds apply PER-PIXEL on prob map; MERGE touching "
                    "positive runs at RLE-encoding time (case §6 series must not touch)",
        "_comment": "frozen ckpt af.pt (val micro-F1=0.8425@thr=0.15); "
                    "this json only adds decision thresholds + WC gate, no weight change",
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    logger.info(f"WROTE {out_path}: {json.dumps({k: out[k] for k in ('thr_day','thr_night','F1_day','F1_night','F1_all_micro')})}")
    logger.info(f"F1_af={bf:.4f} (day={bf_d:.4f}@thr={thr_day:.2f} night={bf_n:.4f}@thr={thr_night:.2f} gate={bname})"
                f" | ckpt af.pt + thr_day/thr_night")


if __name__ == "__main__":
    main()
