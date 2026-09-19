"""BS postproc + metrics + dNBR-threshold-only baseline.

Baseline (no learning): per-landcover-group dNBR thresholds from facts §6
  burn  if dNBR >= burn_thresh[group]   (else 0)
  sev   via cut12/cut23  (dNBR<cut12->1; cut12<=dNBR<cut23->2; >=cut23->3)
  cloud/shadow/snow-gated or SCL==0 pixels forced to 0.
Severity argmax non-overlap: model logits argmax over 4 classes gives exactly one
class per pixel by construction; postfilter then re-gates cloud pixels to 0 and
optionally snaps low-confidence burn pixels below the group burn gate back to 0
(keeps submission masks non-overlapping per case §6).

Metrics (case §7, self-contained — AF task owns src/models/metrics.py, do NOT touch):
  micro-pool TP/FP/FN over all val chips on VALID pixels only;
  IoU_burn: burn = {1,2,3} pooled; mIoU_sev = mean(IoU1,IoU2,IoU3);
  zero-denominator (class absent in both target and pred) -> 1.0;
  class in target but missing in pred -> 0.0 (natural outcome).
"""
import numpy as np


def baseline_predict(dnbr, scl, lc, group_lut, dnbr_thr, burn_thr, cloud_codes,
                     dry_override=True, dry_lc=60, dry_relax=(8, 9)):
    from bs_dataset import cloud_invalid_mask
    gated = cloud_invalid_mask(scl, lc, cloud_codes, dry_override, dry_lc, list(dry_relax))
    invalid = gated | (scl == 0)
    groups = group_lut[np.clip(lc, 0, 255)]
    pred = np.zeros_like(scl, dtype=np.int64)
    for g in ("forest", "steppe", "cropland", "floodplain", "other"):
        m = (groups == g) & (~invalid)
        if not m.any():
            continue
        c12 = float(dnbr_thr[g]["cut12"])
        c23 = float(dnbr_thr[g]["cut23"])
        bt = float(burn_thr[g])
        d = dnbr[m]
        p = np.zeros(m.sum(), dtype=np.int64)
        p[d >= bt] = 1
        p[d >= c12] = 2
        p[d >= c23] = 3
        p[d < bt] = 0  # below burn gate -> background even if cuts say otherwise
        pred[m] = p
    return pred


def apply_postfilter(logits, dnbr, scl, lc, group_lut, dnbr_thr, burn_thr, cloud_codes,
                     dry_override=True, dry_lc=60, dry_relax=(8, 9), snap_to_gate=True):
    """logits (4,H,W) -> (H,W) int64 mask: argmax, cloud->0, optional burn-gate snap."""
    from bs_dataset import cloud_invalid_mask
    pred = np.argmax(logits, axis=0).astype(np.int64)
    gated = cloud_invalid_mask(scl, lc, cloud_codes, dry_override, dry_lc, list(dry_relax))
    pred[gated | (scl == 0)] = 0
    if snap_to_gate:
        groups = group_lut[np.clip(lc, 0, 255)]
        for g in ("forest", "steppe", "cropland", "floodplain", "other"):
            m = (groups == g) & (pred > 0)
            if m.any():
                pred[m & (dnbr < float(burn_thr[g]))] = 0
    return pred


class MicroPoolBS:
    """Accumulate micro-pooled TP/FP/FN per severity class over val chips."""

    def __init__(self):
        self.tp = {1: 0, 2: 0, 3: 0}
        self.fp = {1: 0, 2: 0, 3: 0}
        self.fn = {1: 0, 2: 0, 3: 0}

    def add(self, pred, target, valid):
        v = valid.astype(bool)
        for k in (1, 2, 3):
            p = (pred == k) & v
            t = (target == k) & v
            self.tp[k] += int((p & t).sum())
            self.fp[k] += int((p & (~t)).sum())
            self.fn[k] += int(((~p) & t).sum())

    def iou(self, k):
        d = self.tp[k] + self.fp[k] + self.fn[k]
        if d == 0:
            return 1.0  # zero-denom rule per case §7
        return self.tp[k] / d

    def iou_burn(self):
        tp = sum(self.tp.values())
        fp = sum(self.fp.values())
        fn = sum(self.fn.values())
        d = tp + fp + fn
        if d == 0:
            return 1.0
        return tp / d

    def miou_sev(self):
        return float(np.mean([self.iou(k) for k in (1, 2, 3)]))

    def summary(self):
        ious = {k: self.iou(k) for k in (1, 2, 3)}
        ib = self.iou_burn()
        ms = self.miou_sev()
        return {"iou1": ious[1], "iou2": ious[2], "iou3": ious[3],
                "iou_burn": ib, "miou_sev": ms, "monitor": ib + ms}


PALETTE = {0: (0, 0, 0), 1: (255, 235, 59), 2: (255, 152, 0), 3: (183, 28, 28)}


def save_viz_png(path, pred, target, dnbr):
    """PIL-only side-by-side viz (pred | target | dNBR-gray). No matplotlib dependency."""
    from PIL import Image
    H, W = pred.shape
    def colorize(m):
        img = np.zeros((H, W, 3), dtype=np.uint8)
        for k, c in PALETTE.items():
            img[m == k] = c
        return img
    d = np.clip((dnbr + 1.0) / 2.0, 0, 1)
    gray = (d * 255).astype(np.uint8)
    canvas = np.concatenate([colorize(pred), colorize(target),
                             np.stack([gray, gray, gray], -1)], axis=1)
    Image.fromarray(canvas).save(path)
