#!/usr/bin/env python3
"""BS self-contained trainer — main (Prithvi) + fallback (smp Unet) lanes.

Do NOT create/modify src/models/train.py (AF task owns it). QA command:
  python src/models/train_bs.py --config configs/bs.yaml            # main lane
  python src/models/train_bs.py --config configs/bs-fallback.yaml   # fallback lane
Modes:
  baseline : dNBR-threshold-only classifier on val -> IoU_burn / mIoU_sev (BEATS-NOTHING bar)
  smoke    : failing-first single-batch overfit (loss must drop), per lane
  train    : baseline + smoke + full run; best ckpt by (IoU_burn + mIoU_sev); 3 val viz
  eval     : reload ckpt -> reproduce val IoU (stale-state probe)
  decide   : compare lane ckpts, copy winner -> weights/bs.pt
Must run from repo root (split_json is repo-relative).
"""
import argparse
import copy
import json
import os
import random
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import yaml


def set_seed(seed, deterministic=True):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_cfg(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def log(msg, fh=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh:
        fh.write(line + "\n")
        fh.flush()


def run_baseline(cfg, fh):
    from bs_dataset import BSChipDataset, build_group_lut
    from bs_postproc import MicroPoolBS, baseline_predict
    ds = BSChipDataset(cfg, "val")
    lut = build_group_lut(cfg["landcover_groups"])
    cg = cfg["cloud_gate"]
    mp = MicroPoolBS()
    t0 = time.time()
    for i in range(len(ds)):
        s = ds[i]
        p = baseline_predict(s["dnbr"], s["scl"], s["lc"], lut, cfg["dnbr_thresholds"],
                             cfg["burn_thresholds"], cg["cloud_codes"],
                             cg.get("dry_soil_override", True), int(cg.get("dry_soil_landcover", 60)),
                             tuple(cg.get("dry_soil_relax_codes", [8, 9])))
        mp.add(p, s["y"], s["valid"])
    sm = mp.summary()
    log(f"BASELINE val(n={len(ds)}) IoU_burn={sm['iou_burn']:.4f} mIoU_sev={sm['miou_sev']:.4f} "
        f"[IoU1={sm['iou1']:.4f} IoU2={sm['iou2']:.4f} IoU3={sm['iou3']:.4f}] wall={time.time()-t0:.0f}s", fh)
    return sm


def build_model(cfg, fh):
    import torch
    lane = cfg["train"]["lane"]
    if lane == "prithvi":
        from bs_prithvi import build_prithvi_bs
        model, rep = build_prithvi_bs(cfg)
    else:
        from bs_fallback import build_fallback_bs
        model, rep = build_fallback_bs(cfg)
    log(f"MODEL lane={lane} params={sum(p.numel() for p in model.parameters())/1e6:.1f}M rep={rep}", fh)
    return model


def get_batch(cfg, part, idx=0):
    import torch
    from bs_dataset import BSChipDataset
    ds = BSChipDataset(cfg, part)
    s = ds[idx % len(ds)]
    x = torch.from_numpy(s["x"]).unsqueeze(0)
    y = torch.from_numpy(s["y"]).unsqueeze(0)
    return x, y, s["chip"]


def amp_dtype(cfg):
    import torch
    d = str(cfg["train"].get("amp_dtype", "float16")).lower()
    return torch.bfloat16 if d in ("bf16", "bfloat16") else torch.float16


def run_smoke(cfg, fh):
    import torch
    lane = cfg["train"]["lane"]
    set_seed(int(cfg.get("seed", 19)))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg, fh).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]))
    ce = torch.nn.CrossEntropyLoss(ignore_index=int(cfg["train"]["ignore_index"]))
    iters = int(cfg["train"].get("smoke_overfit_iters", 50))
    x, y, chip = get_batch(cfg, "train", 0)
    x, y = x.to(dev), y.to(dev)
    use_amp = bool(cfg["train"].get("amp", True)) and dev == "cuda"
    dt = amp_dtype(cfg)
    scaler = torch.amp.GradScaler("cuda") if (use_amp and dt == torch.float16) else None
    model.train()
    losses = []
    for it in range(iters):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=dt, enabled=use_amp):
            out = model(x)
            out = out.output if hasattr(out, "output") else out
            loss = ce(out, y)
        (scaler.scale(loss).backward() if scaler else loss.backward())
        (scaler.step(opt) if scaler else opt.step())
        if scaler:
            scaler.update()
        losses.append(float(loss.item()))
    log(f"SMOKE lane={lane} chip={chip} loss[0]={losses[0]:.4f} loss[-1]={losses[-1]:.4f} "
        f"{'OVERFIT-OK' if losses[-1] < losses[0] * 0.5 else 'OVERFIT-WEAK'}", fh)
    if not (losses[-1] < losses[0]):
        raise RuntimeError(f"smoke failed: loss did not drop ({losses[0]:.4f}->{losses[-1]:.4f})")
    return losses


def evaluate(model, cfg, part="val", fh=None, postfilter=True):
    import torch
    from bs_dataset import BSChipDataset, build_group_lut
    from bs_postproc import MicroPoolBS, apply_postfilter
    ds = BSChipDataset(cfg, part)
    lut = build_group_lut(cfg["landcover_groups"])
    cg = cfg["cloud_gate"]
    dev = next(model.parameters()).device
    use_amp = bool(cfg["train"].get("amp", True)) and str(dev) != "cpu"
    dt = amp_dtype(cfg)
    was_training = model.training
    model.eval()
    mp = MicroPoolBS()
    viz = []
    with torch.no_grad():
        for i in range(len(ds)):
            s = ds[i]
            x = torch.from_numpy(s["x"]).unsqueeze(0).to(dev)
            with torch.amp.autocast("cuda", dtype=dt, enabled=use_amp):
                out = model(x)
                out = out.output if hasattr(out, "output") else out
            logits = out.float().cpu().numpy()[0]
            if postfilter:
                p = apply_postfilter(logits, s["dnbr"], s["scl"], s["lc"], lut,
                                     cfg["dnbr_thresholds"], cfg["burn_thresholds"], cg["cloud_codes"],
                                     cg.get("dry_soil_override", True), int(cg.get("dry_soil_landcover", 60)),
                                     tuple(cg.get("dry_soil_relax_codes", [8, 9])), snap_to_gate=True)
            else:
                p = logits.argmax(axis=0).astype(np.int64)
            mp.add(p, s["y"], s["valid"])
            if len(viz) < int(cfg["train"].get("viz_chips", 3)):
                viz.append((s["chip"], p, s["y"].copy(), s["dnbr"]))
    if was_training:
        model.train()
    return mp.summary(), viz


def run_train(cfg, cfg_path, fh):
    import torch
    from torch.utils.data import Dataset, DataLoader
    from bs_dataset import BSChipDataset
    from bs_postproc import save_viz_png
    t_start = time.time()
    set_seed(int(cfg.get("seed", 19)))
    tr = cfg["train"]
    lane = tr["lane"]
    # GPU mem in log (hung/long probe)
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
                              "--format=csv"], capture_output=True, text=True, timeout=20).stdout.strip()
        log(f"nvidia-smi: {smi}", fh)
    except Exception as e:
        log(f"nvidia-smi unavailable: {e}", fh)
    try:
        torch.randn(64, 64, device="cuda")
        log("cuda-ok (torch compute works on device)", fh)
    except Exception as e:
        log(f"cuda compute FAILED, CPU fallback: {e}", fh)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"torch={torch.__version__} cuda_build={torch.version.cuda} device={dev}", fh)

    base = run_baseline(cfg, fh)

    class T(Dataset):
        def __init__(self, c, part):
            self.d = BSChipDataset(c, part)
        def __len__(self):
            return len(self.d)
        def __getitem__(self, i):
            s = self.d[i]
            return (torch.from_numpy(s["x"]),
                    torch.from_numpy(s["y"]).long())

    model = build_model(cfg, fh).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(tr["lr"]), weight_decay=float(tr.get("weight_decay", 0.0)))
    cw = torch.tensor(tr.get("class_weights", [1, 1, 1, 1]), dtype=torch.float32).to(dev)
    ce = torch.nn.CrossEntropyLoss(weight=cw, ignore_index=int(tr["ignore_index"]))
    bs = int(tr["batch_size"])
    accum = int(tr.get("grad_accum", 1))
    use_amp = bool(tr.get("amp", True)) and dev == "cuda"
    dt = amp_dtype(cfg)
    scaler = torch.amp.GradScaler("cuda") if (use_amp and dt == torch.float16) else None
    loader = DataLoader(T(cfg, "train"), batch_size=bs, shuffle=True,
                        num_workers=int(tr.get("num_workers", 2)),
                        generator=torch.Generator().manual_seed(int(cfg.get("seed", 19))))
    clip = float(tr.get("grad_clip", 0.0))
    best = {"monitor": -1}
    logdir = tr.get("log_dir", "logs/bs")
    os.makedirs(logdir, exist_ok=True)
    ckpt_path = tr.get("out_ckpt", "weights/bs.pt")
    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
    for ep in range(int(tr["max_epochs"])):
        model.train()
        tot, n = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for bi, (x, y) in enumerate(loader):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            if int((y != int(tr["ignore_index"])).sum()) == 0:
                log(f"EPOCH {ep+1} batch {bi}: EMPTY-VALID chip, batch skipped", fh)
                continue
            with torch.amp.autocast("cuda", dtype=dt, enabled=use_amp):
                out = model(x)
                out = out.output if hasattr(out, "output") else out
                loss = ce(out, y) / accum
            if not torch.isfinite(loss):
                log(f"EPOCH {ep+1} batch {bi}: NON-FINITE loss, step skipped", fh)
                opt.zero_grad(set_to_none=True)
                continue
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            tot += float(loss.item()) * accum
            n += 1
            if (bi + 1) % accum == 0:
                if scaler:
                    scaler.unscale_(opt)
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                if scaler:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
        if n % accum != 0:  # flush remainder
            if scaler:
                scaler.unscale_(opt)
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            if scaler:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)
        if (ep + 1) % int(tr.get("val_every", 1)) == 0:
            sm, viz = evaluate(model, cfg, "val", fh, postfilter=True)
            log(f"EPOCH {ep+1}/{tr['max_epochs']} train_loss={tot/max(n,1):.4f} "
                f"val IoU_burn={sm['iou_burn']:.4f} mIoU_sev={sm['miou_sev']:.4f} "
                f"[IoU1={sm['iou1']:.4f} IoU2={sm['iou2']:.4f} IoU3={sm['iou3']:.4f}] monitor={sm['monitor']:.4f}", fh)
            if sm["monitor"] > best["monitor"]:
                best = {**sm, "epoch": ep + 1}
                torch.save({"lane": lane, "model": copy.deepcopy(model.state_dict()),
                            "cfg_path": cfg_path, "val": sm, "epoch": ep + 1}, ckpt_path)
                log(f"BEST ckpt -> {ckpt_path} monitor={sm['monitor']:.4f}", fh)
                for chip, p, t, d in viz:
                    save_viz_png(os.path.join(logdir, f"viz_{chip}.png"), p, t, d)
                log(f"viz {len(viz)} PNGs -> {logdir}/", fh)
    wall = time.time() - t_start
    log(f"TRAIN-DONE lane={lane} wall={wall/3600:.2f}h baseline(IoU_burn={base['iou_burn']:.4f},mIoU={base['miou_sev']:.4f}) "
        f"best(val IoU_burn={best.get('iou_burn', float('nan')):.4f},mIoU_sev={best.get('miou_sev', float('nan')):.4f},"
        f"epoch={best.get('epoch', '-')}) ckpt={ckpt_path}", fh)
    vs_model = "BEATS-BASELINE" if best.get("monitor", -1) > base["iou_burn"] + base["miou_sev"] else "BELOW-BASELINE (see report)"
    log(f"MODEL-vs-BASELINE: {vs_model}", fh)
    return best


def run_eval(cfg, ckpt, fh):
    import torch
    set_seed(int(cfg.get("seed", 19)))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg, fh).to(dev)
    sd = torch.load(ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(sd["model"])
    sm, _ = evaluate(model, cfg, "val", fh, postfilter=True)
    saved = sd.get("val", {})
    log(f"EVAL-RELOAD ckpt={ckpt} val IoU_burn={sm['iou_burn']:.4f} mIoU_sev={sm['miou_sev']:.4f} monitor={sm['monitor']:.4f} "
        f"(saved monitor={saved.get('monitor', float('nan')):.4f}) "
        f"{'REPRODUCES' if abs(sm['monitor']-saved.get('monitor', -9)) < 1e-6 else 'MISMATCH'}", fh)
    return sm


def run_decide(fh, paths):
    """Compare lane ckpts by saved val monitor; copy winner -> weights/bs.pt."""
    import torch
    best, bpath = None, None
    for p in paths:
        if not os.path.exists(p):
            log(f"DECIDE skip missing {p}", fh)
            continue
        sd = torch.load(p, map_location="cpu", weights_only=False)
        m = sd.get("val", {}).get("monitor", -1)
        log(f"DECIDE {p}: lane={sd.get('lane')} epoch={sd.get('epoch')} "
            f"IoU_burn={sd.get('val', {}).get('iou_burn', float('nan')):.4f} "
            f"mIoU_sev={sd.get('val', {}).get('miou_sev', float('nan')):.4f} monitor={m:.4f}", fh)
        if best is None or m > best:
            best, bpath = m, p
    if bpath:
        import shutil
        os.makedirs("weights", exist_ok=True)
        shutil.copyfile(bpath, "weights/bs.pt")
        log(f"WINNER {bpath} monitor={best:.4f} -> weights/bs.pt", fh)
    else:
        log("DECIDE: no ckpts found", fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/bs.yaml")
    ap.add_argument("--mode", default="train", choices=["baseline", "smoke", "train", "eval", "decide"])
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--decide-ckpts", default="weights/bs.pt,weights/bs-fallback.pt")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    if a.data_root:
        cfg["data"]["root"] = a.data_root
    lane = cfg["train"]["lane"]
    os.makedirs(cfg["train"].get("log_dir", "logs/bs"), exist_ok=True)
    logpath = os.path.join(cfg["train"].get("log_dir", "logs/bs"), f"{a.mode}_{lane}.log")
    fh = open(logpath, "a")
    log(f"=== mode={a.mode} lane={lane} config={a.config} data_root={cfg['data']['root']} seed={cfg.get('seed')} ===", fh)
    try:
        if a.mode == "baseline":
            set_seed(int(cfg.get("seed", 19)))
            run_baseline(cfg, fh)
        elif a.mode == "smoke":
            run_smoke(cfg, fh)
        elif a.mode == "train":
            run_train(cfg, a.config, fh)
        elif a.mode == "eval":
            run_eval(cfg, a.ckpt or cfg["train"]["out_ckpt"], fh)
        elif a.mode == "decide":
            run_decide(fh, [p.strip() for p in a.decide_ckpts.split(",")])
    finally:
        fh.close()
    print(f"log -> {logpath}")


if __name__ == "__main__":
    main()
