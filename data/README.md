# Data — remote layout on VPS (`/workspace/kosmohack/data/`)

Source (LOCAL, cyrillic path — quote it!): `/home/izislesar/Projects/kosmohack/Мониторинг DATA1/Мониторинг DATA/`
Remote SSH: `root@ssh7.vast.ai -p 11342`. Yandex was NOT used (plan forbids) — only local→VPS copy.

## Archives (local sizes, sha256 in `checksums.sha256`)

| File | Size (bytes) | Content |
|---|---|---|
| `fire-train-renamed.tar` | 2305976320 (2.2G) | `train/` tree: 420 AF + 224 BS chips + `train/{af,bs}/meta.csv` |
| `fire-test-renamed.tar` | 920483840 (878M) | `test/` tree: 180 AF + 89 BS chips + `test/meta.csv` + `test/sample_submission.csv` |
| `fire-aoi/` (dir, 9 files, ~48K) | — | AOI vector: `.geojson/.gpkg/.shp/.dbf/.prj/.shx/.cpg/.zip` + `README.md` |

Integrity: remote `sha256sum` of both tars MATCHES local byte-for-byte (sizes also identical).

## Remote layout (verified `ls -R`)

```
/workspace/kosmohack/data/
├── fire-train-renamed.tar   # kept
├── fire-test-renamed.tar    # kept
├── fire-aoi/                # rsynced (README.md + fire_monitoring_aoi.*)
├── train/                   # from fire-train-renamed.tar, 2.2G unpacked
│   ├── af/{viirs,masks,aux,meta.csv}   # 420× AF_tr_*_VIIRS_I1-I5.tif + *_MASK.tif (images/ EMPTY legacy)
│   └── bs/{sentinel2_pre,sentinel2_post,sentinel1_pre,sentinel1_post,
│           sar_pre,sar_post,aux,masks,meta.csv}  # 224× BS_tr_* (pre/post/ EMPTY legacy, real data in sentinel*)
└── test/                    # from fire-test-renamed.tar, 879M unpacked
    ├── af/{viirs,aux}       # 180× AF_te_* (images/ EMPTY legacy, NO masks in test)
    ├── bs/{sentinel2_pre,sentinel2_post,sentinel1_pre,sentinel1_post,
    │       sar_pre,sar_post,aux}  # 89× BS_te_* (pre/post/ EMPTY legacy)
    ├── meta.csv             # 269 data rows: test chips only (chip_id,kind,width,height,gsd,valid_frac,cloud_frac)
    └── sample_submission.csv  # 448 lines = HEADER + 447 data rows (180 AF×class1 + 89 BS×classes1/2/3)
```

## sample_submission.csv

- Location: `/workspace/kosmohack/data/test/sample_submission.csv` (inside test tar; NOT in train tar).
- `wc -l` = **448** = header `chip_id,class_id,rle` + **447 data rows** (spec: «447 строк без заголовка» ✓).
  - 180× `AF_te_*` class 1; 267× `BS_te_*` (89 chips × classes 1/2/3). All template RLEs EMPTY (test has no masks).
- RLE rules (docs/case.md §6): 1-based row-major, starts ascending, runs non-overlap & non-touching (merge touching),
  empty class = `""`, no NaN, (chip_id,class_id) set == template. Smoke test: `python3` decode of synthetic + empty RLEs — PASS.

## Unpack commands (run on remote, tars contain `train/`+`test/` top-level dirs)

```bash
cd /workspace/kosmohack/data
tar -xf fire-train-renamed.tar   # → train/
tar -xf fire-test-renamed.tar    # → test/
ls -R /workspace/kosmohack/data | head -100
du -sh /workspace/kosmohack/data/*
wc -l $(find /workspace/kosmohack/data -name sample_submission.csv | head -1)  # expect 448 (=447+header)
```

## RLE/metric helpers

NO helper scripts inside the archives (only `*.csv`/`meta.csv`/`.tif`). RLE encoder/decoder + `validate_submission.py`
+ score script (`0.35*F1_af+0.35*IoU_burn+0.30*mIoU_sev`, micro-pool) live in-repo (Tasks 6/F2), semantics per docs/case.md §§6–7.
