"""Micro-averaged binary metrics for AF (docs/case.md section 7).

Pixels of all chips are pooled first, then P/R/F1 are computed.
Zero-denominator rule per case: denominator 0 (class absent in both GT and
prediction) -> score 1.0; class present in GT but absent in prediction -> 0.0
(which falls out naturally: recall = 0).
"""

from __future__ import annotations

import numpy as np


def prf_from_counts(tp: int, fp: int, fn: int) -> tuple:
    """Precision/recall/F1 from pooled counts with the case section 7 zero rule."""
    prec = 1.0 if (tp + fp) == 0 else tp / (tp + fp)
    rec = 1.0 if (tp + fn) == 0 else tp / (tp + fn)
    if (prec + rec) == 0:
        f1 = 0.0
    else:
        f1 = 2 * prec * rec / (prec + rec)
    return prec, rec, f1


def counts_from_masks(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray | None = None) -> tuple:
    """TP/FP/FN over valid pixels. pred/gt binary {0,1}; valid None -> all count."""
    pred = (np.asarray(pred) > 0).astype(np.int64)
    gt = (np.asarray(gt) > 0).astype(np.int64)
    if valid is None:
        mask = np.ones_like(gt, dtype=bool)
    else:
        mask = np.asarray(valid) > 0
    tp = int(np.sum(pred[mask] * gt[mask]))
    fp = int(np.sum(pred[mask] * (1 - gt[mask])))
    fn = int(np.sum((1 - pred[mask]) * gt[mask]))
    return tp, fp, fn


def micro_f1(pred_list: list, gt_list: list, valid_list: list | None = None) -> tuple:
    """Pool counts over chips, return (precision, recall, f1, (tp, fp, fn))."""
    tps = fps = fns = 0
    for i, (p, g) in enumerate(zip(pred_list, gt_list)):
        v = None if valid_list is None else valid_list[i]
        tp, fp, fn = counts_from_masks(p, g, v)
        tps += tp
        fps += fp
        fns += fn
    prec, rec, f1 = prf_from_counts(tps, fps, fns)
    return prec, rec, f1, (tps, fps, fns)


def threshold_sweep(
    prob_list: list, gt_list: list, valid_list: list | None, thresholds: list
) -> tuple:
    """Sweep decision thresholds on pooled val pixels.

    Returns (rows, best_thr, best_f1) where rows = [(thr, prec, rec, f1), ...].
    """
    rows = []
    for thr in thresholds:
        preds = [(np.asarray(p) >= thr).astype(np.uint8) for p in prob_list]
        prec, rec, f1, _ = micro_f1(preds, gt_list, valid_list)
        rows.append((float(thr), prec, rec, f1))
    best = max(rows, key=lambda r: (r[3], -r[0]))  # best F1, tie -> lower thr
    return rows, best[0], best[3]
