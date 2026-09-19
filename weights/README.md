# weights/ — large checkpoints live on the VPS, not in git.

- AF Unet-effb4 best ckpt (remote): `/workspace/kosmohack/weights/af.pt`
  (81.6 MB, epoch 24) — val micro-F1_af **0.8425 @ thr 0.15**
  (baselines: all-zero 0.0000, I4>320K 0.0248).
- Val threshold table + cuda-ok proof: remote `/workspace/kosmohack/logs/af.log`
  (local mirror: `logs/af_remote.log`, per-epoch CSV: `logs/af_train_remote.csv`).
- 3 val-chip viz (png + pred/gt npy): remote `/workspace/kosmohack/weights/af_viz/`
  (AF_tr_000007/000010/000018); sample: `weights/af_viz_sample.png`.
- Train cmd (remote, from `/workspace/kosmohack`):
  `python src/models/train.py --config configs/af.yaml` — exit 0.
