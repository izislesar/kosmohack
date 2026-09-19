"""Generic config-driven AF trainer.

Usage:
    python src/models/train.py --config configs/af.yaml [--data-root ...] [--smoke]

Flow: seed -> baseline (all-zero + I4>thr on val) -> overfit-single-batch
smoke (with --smoke, or always as a 60-step pre-check logged before epoch 1)
-> train/val loop (AMP, valid-weighted loss) -> per-epoch val threshold sweep
-> best ckpt by val micro-F1 -> final threshold table + 3 val-chip viz.

Determinism: seeds fixed; cudnn deterministic; val sweep/threshold exact.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.af_dataset import AFDataset, N_IN
from src.models.af_unet import build_af_unet
from src.models.datasets_common import (
    expand_ids_for_oversampling,
    load_af_fire_lookup,
    load_config,
    load_split,
    make_logger,
    read_tif,
    set_seed,
)
from src.models.losses import build_loss
from src.models.metrics import micro_f1, threshold_sweep


def resolve(p: str, root: str) -> str:
    return p if os.path.isabs(p) else os.path.join(root, p)


def worker_seed(worker_id: int):
    import random

    seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(seed)
    random.seed(seed)


def evaluate_val(model, loader, device, thresholds, use_amp: bool):
    """Run val: return (probs, gts, valids, chips) as numpy lists + fixed-thr stats."""
    model.eval()
    probs, gts, valids, chips = [], [], [], []
    with torch.no_grad():
        for x, y, v, c in loader:
            x = x.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                logits = model(x)
            p = torch.sigmoid(logits).cpu().numpy()[:, 0]
            for i in range(p.shape[0]):
                probs.append(p[i])
                gts.append(y.numpy()[i, 0])
                valids.append(v.numpy()[i])
                chips.append(c[i])
    rows, best_thr, best_f1 = threshold_sweep(probs, gts, valids, thresholds)
    return probs, gts, valids, chips, rows, best_thr, best_f1


def run_baseline(data_root: str, val_ids: list, i4_thr: float, logger) -> dict:
    """Trivial baselines on val BEFORE any model training (metric-pipeline proof)."""
    from src.models.metrics import counts_from_masks, prf_from_counts

    tp0 = fp0 = fn0 = tpi = fpi = fni = 0
    for chip in val_ids:
        v = read_tif(os.path.join(data_root, "train/af/viirs", f"{chip}_VIIRS_I1-I5.tif"))
        gt = read_tif(os.path.join(data_root, "train/af/masks", f"{chip}_MASK.tif"))[0]
        gt = (gt != 255) & (gt > 0)
        valid = np.nan_to_num(v[7], nan=0.0) > 0.5
        zero = np.zeros_like(gt, dtype=np.uint8)
        a, b, c = counts_from_masks(zero, gt.astype(np.uint8), valid.astype(np.uint8))
        tp0, fp0, fn0 = tp0 + a, fp0 + b, fn0 + c
        i4 = np.nan_to_num(v[3], nan=0.0)
        pred_i4 = (valid & (i4 > i4_thr)).astype(np.uint8)
        a, b, c = counts_from_masks(pred_i4, gt.astype(np.uint8), valid.astype(np.uint8))
        tpi, fpi, fni = tpi + a, fpi + b, fni + c
    pz, rz, fz = prf_from_counts(tp0, fp0, fn0)
    pi, ri, fi = prf_from_counts(tpi, fpi, fni)
    logger.info(f"BASELINE all-zero: P={pz:.4f} R={rz:.4f} F1={fz:.4f} (TP={tp0} FP={fp0} FN={fn0})")
    logger.info(f"BASELINE I4>{i4_thr:.0f}K: P={pi:.4f} R={ri:.4f} F1={fi:.4f} (TP={tpi} FP={fpi} FN={fni})")
    return {"zero_f1": fz, "i4_f1": fi, "i4_thr": i4_thr}


def overfit_single_batch(model, batch, loss_fn, device, logger, steps: int = 60) -> bool:
    """Failing-first proof: one batch must overfit (loss decreases)."""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x, y, v, _ = batch
    x, y, v = x.to(device), y.to(device), v.to(device)
    with torch.no_grad():
        l0 = loss_fn(model(x), y, v).item()
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(model(x), y, v)
        loss.backward()
        opt.step()
    with torch.no_grad():
        l1 = loss_fn(model(x), y, v).item()
    ok = l1 < l0
    logger.info(f"SMOKE overfit-1-batch: loss {l0:.4f} -> {l1:.4f} ({'OK' if ok else 'FAIL'})")
    return ok


def save_viz(viz_dir: str, chips, probs, gts, data_root: str, n: int, logger):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(viz_dir, exist_ok=True)
    # Prefer val chips WITH fire so the viz is informative.
    fire_idx = [i for i, g in enumerate(gts) if g.sum() > 0]
    pick = (fire_idx[:n] + list(range(len(chips))))[:n]
    saved = []
    for k, i in enumerate(pick):
        chip = chips[i]
        v = read_tif(os.path.join(data_root, "train/af/viirs", f"{chip}_VIIRS_I1-I5.tif"))
        i4 = np.nan_to_num(v[3], nan=0.0)
        gt = gts[i]
        pred = (probs[i] >= 0.5).astype(np.uint8)
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow(i4, cmap="hot", vmin=270, vmax=340)
        ax[0].set_title(f"{chip} I4[K]")
        ax[1].imshow(gt, cmap="gray", vmin=0, vmax=1)
        ax[1].set_title(f"GT fire px={int(gt.sum())}")
        ax[2].imshow(pred, cmap="gray", vmin=0, vmax=1)
        ax[2].set_title(f"pred@0.5 px={int(pred.sum())}")
        for a in ax:
            a.axis("off")
        png = os.path.join(viz_dir, f"{chip}_pred.png")
        fig.tight_layout()
        fig.savefig(png, dpi=100)
        plt.close(fig)
        np.save(os.path.join(viz_dir, f"{chip}_pred.npy"), pred)
        np.save(os.path.join(viz_dir, f"{chip}_gt.npy"), gt.astype(np.uint8))
        saved.append(png)
    logger.info(f"VIZ saved {len(saved)} chips: {saved}")
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--smoke", action="store_true", help="only overfit-single-batch smoke test, then exit")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 19))
    set_seed(seed, bool(cfg.get("deterministic", True)))

    data_root = args.data_root or cfg.get("data_root", "/workspace/kosmohack/data")
    split = load_split(resolve(cfg["split_path"], os.getcwd()), cfg.get("split_group", "af"))
    train_ids, val_ids = split["train"], split["val"]

    out_ckpt = resolve(cfg["outputs"]["checkpoint"], os.getcwd())
    out_csv = resolve(cfg["outputs"]["csv_log"], os.getcwd())
    out_log = resolve(cfg["outputs"].get("run_log", "logs/af.log"), os.getcwd())
    viz_dir = resolve(cfg["outputs"]["viz_dir"], os.getcwd())
    thresholds = list(cfg["val"]["thresholds"])
    os.makedirs(os.path.dirname(out_ckpt) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    logger = make_logger(out_log)
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"AF train start: torch={torch.__version__} cuda_build={torch.version.cuda} device={device}")
    try:
        ta = torch.randn(64, 64, device="cuda") if device.type == "cuda" else None
        logger.info(f"cuda tensor-op-ok: {ta.device if ta is not None else 'cpu-only'}")
    except Exception as e:
        logger.error(f"cuda tensor-op FAILED: {e}")
        if device.type == "cuda":
            raise
    try:
        import importlib.metadata as _md

        logger.info(
            "versions smp=%s timm=%s terratorch=%s"
            % (_md.version("segmentation-models-pytorch"), _md.version("timm"), _md.version("terratorch"))
        )
    except Exception as e:
        logger.info(f"version probe: {e}")
    logger.info(f"split: train={len(train_ids)} val={len(val_ids)} split_seed={split.get('seed')} seed={seed}")

    # ---- baseline FIRST (metric pipeline proof on unchanged code path) ----
    base = run_baseline(data_root, val_ids, float(cfg["val"].get("baseline_i4_thr_K", 320.0)), logger)

    norm = cfg["norm"]
    tr_cfg, model_cfg = cfg["train"], cfg["model"]
    assert int(model_cfg.get("in_channels", N_IN)) == N_IN, "config in_channels must match af_dataset.N_IN"

    fire_lookup = load_af_fire_lookup(data_root)
    n_pos = sum(1 for c in train_ids if fire_lookup.get(c, 0) > 0)
    logger.info(f"train positives: {n_pos}/{len(train_ids)}; oversample_pos={tr_cfg.get('oversample_pos', 3)}")

    model = build_af_unet(
        in_channels=N_IN,
        encoder=model_cfg.get("encoder", "timm-efficientnet-b4"),
        encoder_weights=model_cfg.get("encoder_weights", "imagenet"),
        classes=int(model_cfg.get("classes", 1)),
    ).to(device)
    loss_fn = build_loss(cfg["loss"])

    val_ds = AFDataset(val_ids, data_root, norm, train=False, seed=seed)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)

    if args.smoke:
        tr_ds1 = AFDataset(train_ids[:4], data_root, norm, train=False, seed=seed)
        l1 = DataLoader(tr_ds1, batch_size=4, shuffle=False, num_workers=0)
        batch = next(iter(l1))
        ok = overfit_single_batch(model, batch, loss_fn, device, logger)
        sys.exit(0 if ok else 1)

    # Pre-train smoke on 1 batch (failing-first proof, then re-init optimizer below).
    tr_ds0 = AFDataset(train_ids[:4], data_root, norm, train=False, seed=seed)
    b0 = next(iter(DataLoader(tr_ds0, batch_size=4, shuffle=False, num_workers=0)))
    if not overfit_single_batch(model, b0, loss_fn, device, logger):
        logger.error("SMOKE FAILED — aborting full run")
        sys.exit(2)

    expanded = expand_ids_for_oversampling(train_ids, fire_lookup, int(tr_cfg.get("oversample_pos", 3)), seed)
    train_ds = AFDataset(expanded, data_root, norm, train=True, seed=seed)
    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(tr_cfg.get("batch_size", 16)),
        shuffle=True,
        num_workers=int(tr_cfg.get("num_workers", 4)),
        generator=gen,
        worker_init_fn=worker_seed,
        drop_last=True,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=float(tr_cfg.get("lr", 3e-4)), weight_decay=float(tr_cfg.get("weight_decay", 1e-4)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(tr_cfg.get("epochs", 40)))
    use_amp = bool(tr_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    epochs = int(tr_cfg.get("epochs", 40))
    patience = int(tr_cfg.get("early_stopping_patience", 12))

    csv_fh = open(out_csv, "w", newline="")
    csv_w = csv.writer(csv_fh)
    csv_w.writerow(["epoch", "train_loss", "val_f1_best_thr", "val_best_thr", "lr", "elapsed_s"])

    best_f1, best_ep, bad = -1.0, -1, 0
    best_rows: list = []
    best_thr, best_state = 0.5, None
    for ep in range(1, epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for x, y, v, _ in train_loader:
            x, y, v = x.to(device), y.to(device), v.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = loss_fn(model(x), y, v)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item()
            nb += 1
        sched.step()
        tr_loss = tot / max(nb, 1)
        _, _, _, _, rows, bthr, bf1 = evaluate_val(model, val_loader, device, thresholds, use_amp)
        lr_now = opt.param_groups[0]["lr"]
        el = time.time() - t_start
        logger.info(f"epoch {ep}/{epochs} train_loss={tr_loss:.4f} val_F1={bf1:.4f}@thr={bthr:.2f} lr={lr_now:.2e}")
        csv_w.writerow([ep, f"{tr_loss:.5f}", f"{bf1:.5f}", f"{bthr:.3f}", f"{lr_now:.3e}", f"{el:.0f}"])
        csv_fh.flush()
        if bf1 > best_f1 + 1e-6:
            best_f1, best_ep, bad = bf1, ep, 0
            best_rows, best_thr = rows, bthr
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                logger.info(f"early stop at epoch {ep} (patience {patience}); best F1={best_f1:.4f} @ep{best_ep}")
                break
    csv_fh.close()

    torch.save(
        {
            "state_dict": best_state,
            "config": cfg,
            "best_val_f1": best_f1,
            "best_thr": best_thr,
            "best_epoch": best_ep,
            "seed": seed,
            "torch_version": torch.__version__,
        },
        out_ckpt,
    )
    logger.info(f"CHECKPOINT best -> {out_ckpt} (val F1={best_f1:.4f}@thr={best_thr:.2f}, epoch {best_ep})")

    logger.info("THRESHOLD TABLE (val micro-F1, best ckpt):")
    for thr, p, r, f in best_rows:
        logger.info(f"  thr={thr:.2f} P={p:.4f} R={r:.4f} F1={f:.4f}")

    # Reload best ckpt into model (determinism spot-check path) and re-verify val F1.
    assert best_state is not None and best_rows, "no epoch completed — cannot checkpoint"
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    _, _, _, _, _, rthr, rf1 = evaluate_val(model, val_loader, device, thresholds, use_amp)
    logger.info(f"RELOAD-CHECK best ckpt val F1={rf1:.4f}@thr={rthr:.2f} (expect match {best_f1:.4f})")

    # Viz: reload probs from best ckpt for 3 val chips.
    model.eval()
    probs, gts, _, chips, _, _, _ = evaluate_val(model, val_loader, device, [0.5], False)
    save_viz(viz_dir, chips, probs, gts, data_root, int(cfg["outputs"].get("viz_chips", 3)), logger)

    el = time.time() - t_start
    if best_f1 < base["i4_f1"]:
        logger.warning(
            f"MODEL F1 {best_f1:.4f} < I4-BASELINE {base['i4_f1']:.4f} — see flare-limitation note; reporting honestly."
        )
    logger.info(f"AF train DONE in {el:.0f}s. baseline_zero_F1={base['zero_f1']:.4f} baseline_I4_F1={base['i4_f1']:.4f} best_val_F1={best_f1:.4f}@thr={best_thr:.2f}")


if __name__ == "__main__":
    main()
