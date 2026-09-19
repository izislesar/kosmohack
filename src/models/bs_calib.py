#!/usr/bin/env python3
"""BS post-hoc calibration on val (no retrain, AF frozen).

Stages (run in order, each prints val line):
  1 calib : per-group burn-gate scale x cut scale x snap x sev-remap sweep -> configs/bs-calib.json
  2 tta   : 4-flip logit averaging gain/loss + wall estimate
  3 morph : remove components <N px + fill holes, N in {0,8,25,64,128} -> IoU before/after
  4 scl   : dry-soil override on/off -> FP delta on cropland + Score

Usage (from /workspace/kosmohack):
  PYTHONPATH=src/models python3 src/models/bs_calib.py --stage 1
Cache: /tmp/bs_calib_cache.npz (logits fp16 + side arrays).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

CACHE = "/tmp/bs_calib_cache.npz"
GROUPS = ("forest", "steppe", "cropland", "floodplain", "other")
F1_AF = 0.8434


def score_of(iou_burn, miou):
    return 0.35 * F1_AF + 0.35 * iou_burn + 0.30 * miou


def load_cfg():
    import yaml
    with open("configs/bs-fallback.yaml") as fh:
        return yaml.safe_load(fh)


def build_cache(cfg):
    import torch
    from bs_dataset import BSChipDataset, build_group_lut, assemble_features  # noqa
    from bs_fallback import build_fallback_bs
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = build_fallback_bs(cfg)
    sd = torch.load("weights/bs.pt", map_location=dev, weights_only=False)
    model.load_state_dict(sd["model"])
    model = model.to(dev)
    model.eval()
    ds = BSChipDataset(cfg, "val")
    use_amp = dev == "cuda"
    ids, logits, ys, valids, dnbres, scls, lcs = [], [], [], [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for i in range(len(ds)):
            s = ds[i]
            x = torch.from_numpy(s["x"]).unsqueeze(0).to(dev)
            out = model(x)
            lg = out.float().cpu().numpy()[0].astype(np.float16)
            ids.append(s["chip"])
            logits.append(lg)
            ys.append(s["y"])
            valids.append(s["valid"])
            dnbres.append(s["dnbr"])
            scls.append(s["scl"])
            lcs.append(s["lc"])
    np.savez_compressed(CACHE, ids=np.array(ids),
                        logits=np.stack(logits), ys=np.stack(ys).astype(np.int16),
                        valids=np.stack(valids), dnbres=np.stack(dnbres).astype(np.float32),
                        scls=np.stack(scls).astype(np.int16), lcs=np.stack(lcs).astype(np.int16))
    print(f"CACHE n={len(ds)} wall={time.time()-t0:.0f}s -> {CACHE}", flush=True)


def load_cache():
    z = np.load(CACHE, allow_pickle=False)
    return z


def micro_scores(preds, ys, valids):
    tp = {1: 0, 2: 0, 3: 0}
    fp = {1: 0, 2: 0, 3: 0}
    fn = {1: 0, 2: 0, 3: 0}
    for p, t, v in zip(preds, ys, valids):
        v = v.astype(bool)
        for k in (1, 2, 3):
            pk = (p == k) & v
            tk = (t == k) & v
            tp[k] += int((pk & tk).sum())
            fp[k] += int((pk & ~tk).sum())
            fn[k] += int((~pk & tk).sum())
    def iou(k):
        d = tp[k] + fp[k] + fn[k]
        return 1.0 if d == 0 else tp[k] / d
    i1, i2, i3 = iou(1), iou(2), iou(3)
    tb, fb, fnb = sum(tp.values()), sum(fp.values()), sum(fn.values())
    ib = 1.0 if (tb + fb + fnb) == 0 else tb / (tb + fb + fnb)
    mi = float(np.mean([i1, i2, i3]))
    return ib, mi, (i1, i2, i3), dict(tp=tp, fp=fp, fn=fn)


def decode_with(logits, dnbres, scls, lcs, lut, dn_thr, burn_thr, cloud_codes,
                dry_override, dry_lc, dry_relax, snap, remap):
    """logits (N,4,H,W) f32 -> preds. remap: dNBR-correct argmax sev per group."""
    from bs_dataset import cloud_invalid_mask
    N = logits.shape[0]
    preds = np.argmax(logits, axis=1).astype(np.int64)
    gated = np.zeros_like(preds, dtype=bool)
    for i in range(N):
        g = cloud_invalid_mask(scls[i], lcs[i], cloud_codes, dry_override, dry_lc, list(dry_relax))
        gated[i] = g | (scls[i] == 0)
    preds[gated] = 0
    groups = lut[np.clip(lcs, 0, 255)]  # (N,H,W) object
    for gi, g in enumerate(GROUPS):
        m = (groups == g)
        if not m.any():
            continue
        c12 = float(dn_thr[g]["cut12"])
        c23 = float(dn_thr[g]["cut23"])
        bt = float(burn_thr[g])
        if snap:
            low = m & (preds > 0) & (dnbres < bt)
            preds[low] = 0
        if remap:
            burn = m & (preds > 0)
            d = dnbres[burn]
            new = np.ones_like(d, dtype=np.int64)
            new[d >= bt] = 1
            new[d >= c12] = 2
            new[d >= c23] = 3
            new[d < bt] = 0
            preds[burn] = new
    return preds


def stage1():
    import yaml
    cfg = load_cfg()
    z = load_cache()
    logits = z["logits"].astype(np.float32)
    ys, valids = z["ys"].astype(np.int64), z["valids"]
    dnbres = z["dnbres"].astype(np.float32)
    scls, lcs = z["scls"].astype(np.int64), z["lcs"].astype(np.int64)
    lut = __import__("bs_dataset", fromlist=["build_group_lut"]).build_group_lut(cfg["landcover_groups"])
    cg = cfg["cloud_gate"]
    cc = list(cg["cloud_codes"])
    dso, dlc, drlx = True, 60, (8, 9)
    base_dn, base_bt = cfg["dnbr_thresholds"], cfg["burn_thresholds"]
    print("THR-SWEEP burn_scale x cut_scale x snap x remap (micro-pool val):", flush=True)
    best = None
    for bs in (0.7, 0.85, 1.0, 1.15, 1.3):
        for cs in (0.85, 1.0, 1.15):
            dn = {g: {"cut12": base_dn[g]["cut12"] * cs, "cut23": base_dn[g]["cut23"] * cs} for g in GROUPS}
            bt = {g: base_bt[g] * bs for g in GROUPS}
            for snap, remap in ((True, False), (True, True), (False, False)):
                p = decode_with(logits, dnbres, scls, lcs, lut, dn, bt, cc, dso, dlc, drlx, snap, remap)
                ib, mi, _, _ = micro_scores(p, ys, valids)
                line = (f"burnx{bs:.2f} cutx{cs:.2f} snap={int(snap)} remap={int(remap)} "
                        f"IoU_burn={ib:.4f} mIoU={mi:.4f} Score={score_of(ib, mi):.4f}")
                print(line, flush=True)
                key = (score_of(ib, mi), -abs(bs - 1.0) - abs(cs - 1.0) * 0.5)
                if best is None or key > best[0]:
                    best = (key, dict(burn_scale=bs, cut_scale=cs, snap=snap, remap=remap,
                                      iou_burn=round(ib, 4), miou_sev=round(mi, 4),
                                      score=round(score_of(ib, mi), 4)))
    b = best[1]
    print(f"CALIB-BEST burnx{b['burn_scale']} cutx{b['cut_scale']} snap={b['snap']} remap={b['remap']} "
          f"IoU_burn={b['iou_burn']:.4f} mIoU={b['miou_sev']:.4f} Score={b['score']:.4f}", flush=True)
    out = {"seed": 19, "f1_af": F1_AF, "base": {"burn_scale": 1.0, "cut_scale": 1.0, "snap": True, "remap": False},
           "best": b, "morph_min_px": 0, "tta_flips": False, "dry_soil_override": True,
           "note": "val micro-pool; thresholds scale facts dNBR cuts per group"}
    json.dump(out, open("configs/bs-calib.json", "w"), indent=2)
    print("WROTE configs/bs-calib.json", flush=True)


def current_decode(logits, cfg_path="configs/bs-calib.json"):
    cfg = load_cfg()
    try:
        cal = json.load(open(cfg_path))
        b = cal["best"]
        bs, cs, snap, remap = b["burn_scale"], b["cut_scale"], b["snap"], b["remap"]
    except Exception:
        bs, cs, snap, remap = 1.0, 1.0, True, False
    base_dn, base_bt = cfg["dnbr_thresholds"], cfg["burn_thresholds"]
    dn = {g: {"cut12": base_dn[g]["cut12"] * cs, "cut23": base_dn[g]["cut23"] * cs} for g in GROUPS}
    bt = {g: base_bt[g] * bs for g in GROUPS}
    return cfg, dn, bt, snap, remap


def stage2():
    import torch
    from bs_dataset import BSChipDataset
    from bs_fallback import build_fallback_bs
    cfg = load_cfg()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = build_fallback_bs(cfg)
    sd = torch.load("weights/bs.pt", map_location=dev, weights_only=False)
    model.load_state_dict(sd["model"])
    model = model.to(dev)
    model.eval()
    ds = BSChipDataset(cfg, "val")
    _, dn, bt, snap, remap = current_decode(None)
    from bs_dataset import build_group_lut
    lut = build_group_lut(cfg["landcover_groups"])
    cg = cfg["cloud_gate"]
    cc = list(cg["cloud_codes"])
    use_amp = dev == "cuda"
    t0 = time.time()
    n_flip = 0
    ib0 = mib0 = None
    with torch.no_grad():
        for rep in ("single", "tta4"):
            preds, ys, valids, dnbres, scls, lcs = [], [], [], [], [], []
            for i in range(len(ds)):
                s = ds[i]
                x = torch.from_numpy(s["x"]).unsqueeze(0).to(dev)
                if rep == "single":
                    lg = model(x).float().cpu().numpy()[0]
                else:
                    acc = None
                    for fh, fw in ((False, False), (True, False), (False, True), (True, True)):
                        xx = x.flip(-2) if fh else x
                        xx = xx.flip(-1) if fw else xx
                        o = model(xx).float()
                        if fw:
                            o = o.flip(-1)
                        if fh:
                            o = o.flip(-2)
                        acc = o if acc is None else acc + o
                    lg = (acc / 4).cpu().numpy()[0]
                    n_flip += 1
                preds.append(lg)
                ys.append(s["y"])
                valids.append(s["valid"])
                dnbres.append(s["dnbr"])
                scls.append(s["scl"])
                lcs.append(s["lc"])
            P = decode_with(np.stack(preds), np.stack(dnbres), np.stack(scls), np.stack(lcs),
                            lut, dn, bt, cc, True, 60, (8, 9), snap, remap)
            ib, mi, _, _ = micro_scores(P, ys, valids)
            wall = time.time() - t0
            tag = "TTA4" if rep == "tta4" else "SINGLE"
            print(f"{tag} val IoU_burn={ib:.4f} mIoU={mi:.4f} Score={score_of(ib, mi):.4f} "
                  f"wall_val={wall:.0f}s (~x{4 if rep=='tta4' else 1} infer -> test ~{28*(4 if rep=='tta4' else 1)}s)",
                  flush=True)
            if rep == "single":
                ib0, mib0 = ib, mi
            else:
                print(f"TTA-GAIN dIoU={ib-ib0:+.4f} dmIoU={mi-mib0:+.4f} "
                      f"dScore={score_of(ib, mi)-score_of(ib0, mib0):+.4f}", flush=True)


def stage3():
    from scipy.ndimage import binary_fill_holes, label
    z = load_cache()
    cfg, dn, bt, snap, remap = current_decode(z["logits"].astype(np.float32))
    from bs_dataset import build_group_lut
    lut = build_group_lut(cfg["landcover_groups"])
    cg = cfg["cloud_gate"]
    cc = list(cg["cloud_codes"])
    base = decode_with(z["logits"].astype(np.float32), z["dnbres"].astype(np.float32),
                       z["scls"].astype(np.int64), z["lcs"].astype(np.int64),
                       lut, dn, bt, cc, True, 60, (8, 9), snap, remap)
    ys, valids = z["ys"].astype(np.int64), z["valids"]
    ib0, mi0, _, _ = micro_scores(base, ys, valids)
    print(f"MORPH-BEFORE IoU_burn={ib0:.4f} mIoU={mi0:.4f} Score={score_of(ib0, mi0):.4f}", flush=True)
    for n in (0, 8, 25, 64, 128):
        P = base.copy()
        if n > 0:
            for i in range(P.shape[0]):
                filled = binary_fill_holes(base[i] > 0)
                lab, nc = label(filled)
                keep = np.zeros_like(filled, dtype=bool)
                if nc:
                    sizes = np.bincount(lab.ravel())
                    for c in range(1, nc + 1):
                        if sizes[c] >= n:
                            keep[lab == c] = True
                P[i][:] = 0
                P[i][keep] = base[i][keep]
        ib, mi, _, _ = micro_scores(P, ys, valids)
        print(f"MORPH N={n} IoU_burn={ib:.4f} mIoU={mi:.4f} Score={score_of(ib, mi):.4f} "
              f"d=({ib-ib0:+.4f},{mi-mi0:+.4f})", flush=True)


def stage4():
    z = load_cache()
    cfg, dn, bt, snap, remap = current_decode(z["logits"].astype(np.float32))
    from bs_dataset import build_group_lut
    lut = build_group_lut(cfg["landcover_groups"])
    cg = cfg["cloud_gate"]
    cc = list(cg["cloud_codes"])
    lg = z["logits"].astype(np.float32)
    dnA = z["dnbres"].astype(np.float32)
    sc, lc = z["scls"].astype(np.int64), z["lcs"].astype(np.int64)
    ys, valids = z["ys"].astype(np.int64), z["valids"]
    groups = lut[np.clip(lc, 0, 255)]
    for dso in (True, False):
        P = decode_with(lg, dnA, sc, lc, lut, dn, bt, cc, dso, 60, (8, 9), snap, remap)
        ib, mi, _, c = micro_scores(P, ys, valids)
        crop = (groups == "cropland")
        fp_crop = 0
        for i in range(P.shape[0]):
            v = valids[i].astype(bool) & crop[i]
            for k in (1, 2, 3):
                fp_crop += int((((P[i] == k) & v) & ~(ys[i] == k)).sum())
        print(f"SCL dry_override={int(dso)} IoU_burn={ib:.4f} mIoU={mi:.4f} Score={score_of(ib, mi):.4f} "
              f"FP_cropland={fp_crop}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=(0, 1, 2, 3, 4))
    a = ap.parse_args()
    if a.stage == 0:
        build_cache(load_cfg())
    elif a.stage == 1:
        stage1()
    elif a.stage == 2:
        stage2()
    elif a.stage == 3:
        stage3()
    elif a.stage == 4:
        stage4()


if __name__ == "__main__":
    main()