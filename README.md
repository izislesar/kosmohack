# kosmohack — fire monitoring (AF + BS + service)

Pipeline: VIIRS AF Unet → S2 BS Unet → `inference.py` → `submission.csv` → FastAPI service.
Metric: `Score = 0.35*F1_af + 0.35*IoU_burn + 0.30*mIoU_sev` (micro-pool, case §7).
Current: F1_af 0.8434 · IoU_burn 0.5239 · mIoU_sev 0.5430 → Score ≈ 0.64.

## Env / install (оргам: запуск с нуля)
```
# Вариант A — CUDA (рекомендуется, ~30 сек инференс):
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.lock
# Вариант B — CPU (если драйвер старый и torch не видит GPU, упадёт сам на CPU):
pip install torch==2.11.0+cpu torchvision==0.26.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.lock
```
Веса лежат в репо: `weights/af.pt` (82M) + `weights/bs.pt` (82M) + `weights/af_thresh.json` — скачивать ничего не надо. `inference.py` сам падает на CPU (`--device cuda` → авто-fallback), на CPU прогон дольше (~10–20 мин) но результат бит-в-бит тот же.
Seeds: 19 everywhere. Determinism: 2 inference runs → byte-identical (diff 0 ≤ 0.005).

## Inference (contract, case §8)
```
PYTHONPATH=src/models python3 inference.py --data-dir data/test --output submission.csv
```
- Reads `sample_submission.csv` template (447 rows: 180 AF×1 + 89 BS×3).
- RLE: 1-indexed row-major, starts ascending, touching runs merged, empty → `""`.
- AF: `weights/af.pt` + `weights/af_thresh.json` (thr 0.15, WC gate lc50/I4≥310K).
- BS: `weights/bs.pt` (smp-Unet effb4, 27ch) argmax + cloud→0 + burn-gate snap.
- Timed: ~28s on RTX PRO 4500 (log: `INFERENCE-DONE … wall=28.4s`).

## Training
```
python src/models/train.py --config configs/af.yaml          # AF Unet-effb4
python src/models/train_bs.py --config configs/bs-fallback.yaml  # BS smp-Unet (deliverable)
python src/models/train_bs.py --config configs/bs.yaml           # Prithvi lane (dead: IoU 0.16 < baseline 0.33)
python src/models/af_thresholds.py --config configs/af.yaml  # thresholds only, never retrains
```

## Weights
`weights/af.pt` (AF Unet) · `weights/af_thresh.json` (thr+gate) · `weights/bs.pt` (BS fallback winner, ep91).

## Service (Variant A, localhost)
```
SUBMISSION_CSV=submission.csv AOI_GEOJSON=data/fire-aoi/fire_monitoring_aoi.geojson \
  python3 -m uvicorn service.app:app --host 0.0.0.0 --port 8000
# фронт: открыть index.html в браузере; js/config.js -> API_BASE http://localhost:8000, USE_MOCK false
```
- `GET /v1/health` → `{status: ok}`
- `GET /v1/fires?bbox=&date_from=&date_to=&format=` (точки)
- `GET /v1/burned-areas?...&severity=1,2,3&format=` → properties `{id, severity, severity_label, area_ha, date_pre, date_post}`
- `GET /v1/analytics?...` → `{fire_points, total_burned_ha, by_severity_ha{1,2,3}, by_severity_pct, bbox, date_from, date_to, projection_for_area}`
- `POST /v1/{fires,burned-areas,analytics}/query {polygon, date_from, date_to, ...}`
- Formats: geojson (default) | json | shp (zip). CORS: только `http://localhost:3000`, `http://localhost:5173`, `http://localhost:8000`.
- `SUBMISSION_CSV` из env (default `submission.csv`). БД нет — бэк файловый.
- E2E: `bash docs/e2e_curl.sh` (health, точки, полигоны, справка без нулей, severity-фильтр, polygon-POST, CORS allow/deny, config).
- Demo mapping: chips are de-identified (case §5.5); service tiles inference outputs across the AOI bbox in index order with deterministic demo dates — geometry math is real, placement is documented demo, not recovered georeferencing.

## Submission — FINAL
`submission.csv` (4.1M, 447 строк, md5 `97b86c6783c14e168353b93741f1f224`): пары chip/class совпадают с шаблоном, RLE валиден, NaN 0, BS-пересечений 0. Фриз Score 0.654. Переобучение запрещено.

## Layout
`configs/` · `src/models/` (train, datasets, postproc) · `inference.py` · `service/app.py` ·
`weights/` · `logs/` · `eda/facts.md + split.json` · `data/` (train/test, not in git).
