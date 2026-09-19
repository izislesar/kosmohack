# EDA facts (measured, not guessed)

- generated_utc: 2026-09-18T22:52:49+00:00
- data_root: /workspace/kosmohack/data
- seed: 19
- package_versions: {"numpy": "1.26.4", "rasterio": "1.3.11", "tifffile": "2024.8.30", "PyYAML": "6.0.2", "pillow": "10.4.0"}
- how: single seeded pass over full train; per-pixel stats via float64 accumulators; dNBR=(NBR_pre-NBR_post), NBR=(B8A-B12)/(B8A+B12) from uint16 L2A, eps=1e-6; RdNBR=dNBR/sqrt(|NBR_pre|)

## 1. meta schemas (exact headers)
- train_af_meta: cols=['chip_id', 'kind', 'fire_event_id', 'region', 'epsg', 'x_min', 'y_min', 'x_max', 'y_max', 'width', 'height', 'gsd', 'acq_datetime', 'satellite', 'date_pre', 'date_post', 's1_date_pre', 's1_date_post', 'valid_frac', 'cloud_frac', 'landcover_top', 'n_fire_px', 'burn_area_ha', 'sev1_px', 'sev2_px', 'sev3_px'] rows=420
- train_bs_meta: cols=['chip_id', 'kind', 'fire_event_id', 'region', 'epsg', 'x_min', 'y_min', 'x_max', 'y_max', 'width', 'height', 'gsd', 'acq_datetime', 'satellite', 'date_pre', 'date_post', 's1_date_pre', 's1_date_post', 'valid_frac', 'cloud_frac', 'landcover_top', 'n_fire_px', 'burn_area_ha', 'sev1_px', 'sev2_px', 'sev3_px'] rows=224
- test_meta: cols=['chip_id', 'kind', 'width', 'height', 'gsd', 'valid_frac', 'cloud_frac'] rows=269
- fire_event_id: AF train EMPTY 0/420 (split by fire_event_id IMPOSSIBLE for AF -> chip-grouped fallback, see section 9); BS train 224/224 present, all 224 UNIQUE (1 chip per event -> event-split == chip-split, still grouped by event id explicitly)
- region: AF train all '' (420/420 empty); BS train all 'nan' (224/224); epsg: AF {32637:306, 32638:114}, BS {32637:109, 32638:115}
- satellite (AF): SNPP=280, NOAA20=123, NOAA21=17; BS satellite col all 'nan'
- landcover_top: EMPTY everywhere (train AF 0/420, train BS 0/224) -> cover MUST come from rasters (AF aux b1 landcover, BS aux b3 landcover)
- AF cloud_frac col: all 'nan' (420/420); BS cloud_frac: 224/224 numeric; test meta cloud_frac: 89/269 numeric (=BS chips only), 180 AF rows literal 'nan' strings -> parse with nan-safe float(), never as 0

- train_af valid_frac: min=0.4597 p25=1 med=1 p75=1 max=1 (n=420); frac<1.0: 0.0333 (14/420)
- train_bs valid_frac: min=0.5152 p25=0.8575 med=0.9865 p75=1 max=1 (n=224); frac<1.0: 0.6473 (145/224)
- train_bs meta cloud_frac: min=0 p25=0 med=0.0135 p75=0.1425 max=0.4848 (n=224); chips>25%: 24/224; plan guesses were med 1.6% / max 48.5% -> CONFIRM-OR-CORRECT by this line
- test_af valid_frac: min=0.5333 p25=1 med=1 p75=1 max=1 (n=180); test_bs valid_frac: min=0.5043 p25=0.7354 med=0.9745 p75=0.9995 max=1 (n=89)
- test_bs cloud_frac: min=0 p25=0.0005 med=0.0255 p75=0.2646 max=0.4957 (n=89)

## 2. AF fire masks (train)
- n_mask_files: 420 (expect 420)
- mask value set: [0, 1] (expect [0,1]; nodata tag=255.0, pixels equal to nodata: 0)
- positive chips: 296/420 (case says 296/420 -> confirm-or-correct)
- total fire px: 9725 / 27525120 = 0.03533% (case says 9725 px = 0.035% -> confirm-or-correct)
- fire px per positive chip: min=1 p25=8 med=21 p75=43.25 max=243 (n=296) (case says median 21 -> confirm-or-correct)
- fire px per chip (all): min=0 p25=0 med=9 p75=36 max=243 (n=420)
- meta n_fire_px: present 420/420 (0 for negatives), sum=9725 == mask recount 9725 EXACT match

## 3. AF VIIRS (8 bands: I1 I2 I3 I4 I5 solar_zenith sensor_zenith valid)
- I1: finite=17521040 nan_frac=0.363453 min=0.0173 max=1.2273 mean=0.1194 std=0.1202
- I2: finite=17521040 nan_frac=0.363453 min=0.0062 max=1.3100 mean=0.1903 std=0.1344
- I3: finite=17521040 nan_frac=0.363453 min=0.0000 max=1.2939 mean=0.1655 std=0.0794
- I4: finite=25239188 nan_frac=0.083049 min=207.9341 max=358.2337 mean=295.0240 std=14.6218
- I5: finite=25239188 nan_frac=0.083049 min=198.8609 max=367.5439 mean=286.4731 std=15.9266
- solar_zenith: finite=27344969 nan_frac=0.006545 min=22.9500 max=143.9300 mean=66.1273 std=34.1889
- sensor_zenith: finite=27344969 nan_frac=0.006545 min=0.0000 max=70.1300 mean=35.7425 std=19.8183
- valid: finite=27525120 nan_frac=0.000000 min=0.0000 max=1.0000 mean=0.9935 std=0.0804
- I1-I3 in [0,1]? min_I1=0.0173 max_I1=1.2273 min_I2=0.0062 max_I2=1.3100 min_I3=0.0000 max_I3=1.2939 (reflectance fraction expected)
- I4 (Kelvin): max_observed=358.23K; pixels>367K: 0/25239188 (0.0000%) -> 367K-clip validation: no saturation observed, clip still harmless
- I4-I5 feature: n=25239188 min=-159.610 max=80.907 mean=8.551 std=9.960
- all-NaN I1-I5 chips: n=0 []
- NIGHT (solar_zenith>90): px frac=0.2997 (8250536/27525120); night chips (chip-median solz>90): 127/420; I1 NaN|day=0.0824 vs I1 NaN|night=1.0000 -> reflective I1-I3 usable ONLY by day (NaN at night by construction); thermal I4/I5 work day+night (their 8.3% NaN = invalid px, matches day-invalid rate)
- valid-band mean per chip: min=0.4597 p25=1 med=1 p75=1 max=1 (n=420) (masking recommendation: weight loss/metrics by valid band; drop all-NaN chips from train)

## 4. AF aux (5 bands: landcover dem t2m rh2m wind_speed; float32)
- n_aux_files: 420 (expect 420)
- landcover codes: chips_with_code={10: 420, 20: 7, 30: 420, 40: 420, 50: 420, 60: 407, 80: 420, 90: 420} (chip-presence counts; WorldCover v200: 10 tree 20 shrub 30 grass 40 crop 50 built 60 barren 70 snow 80 water 90 wetland)
- aux landcover: min=10.000 max=90.000
- aux dem: min=-179.189 max=1246.865
- aux t2m: min=274.523 max=309.263
- aux rh2m: min=12.683 max=99.420
- aux wind_speed: min=0.030 max=12.591

## 5. BS masks (train)
- n_mask_files: 224 (expect 224)
- mask value set: [0, 1, 2, 3] (expect [0,1,2,3])
- class0: px=52811255 frac_of_all=89.9370% median_px_per_chip=242434 chips_with_class=224/224
- class1: px=2362871 frac_of_all=4.0239% median_px_per_chip=7246 chips_with_class=224/224
- class2: px=2192271 frac_of_all=3.7334% median_px_per_chip=5996 chips_with_class=222/224
- class3: px=1353859 frac_of_all=2.3056% median_px_per_chip=2712 chips_with_class=202/224
- burned frac of all px: 10.0630%; within-burn sev mix: 1=40.0% 2=37.1% 3=22.9% (case: 40.0/37.1/22.9 -> confirm-or-correct)
- burned ha check: 236360 ha (case says 236360 ha)
- meta sev px sums: sev1=2362871, sev2=2192271, sev3=1353859

## 6. BS Sentinel-2 / dNBR per landcover (train)
- n_s2_pre_files: 224 (expect 224)
- S2 pre B2: min=0 max=20624 mean=841.8 std=817.7 (uint16 L2A 0-10000 expected)
- S2 pre B3: min=0 max=19120 mean=1083.9 std=789.7 (uint16 L2A 0-10000 expected)
- S2 pre B4: min=0 max=17872 mean=1316.9 std=873.0 (uint16 L2A 0-10000 expected)
- S2 pre B5: min=0 max=17441 mean=1624.5 std=914.8 (uint16 L2A 0-10000 expected)
- S2 pre B6: min=0 max=17182 mean=1980.9 std=932.5 (uint16 L2A 0-10000 expected)
- S2 pre B7: min=0 max=16959 mean=2166.9 std=979.8 (uint16 L2A 0-10000 expected)
- S2 pre B8A: min=0 max=16609 mean=2377.7 std=1018.2 (uint16 L2A 0-10000 expected)
- S2 pre B11: min=0 max=15385 mean=2818.6 std=1014.2 (uint16 L2A 0-10000 expected)
- S2 pre B12: min=0 max=15248 mean=2167.0 std=872.9 (uint16 L2A 0-10000 expected)
- S2 pre SCL: min=0 max=11 mean=5.3 std=1.6 (uint16 L2A 0-10000 expected)
- S2 post B2: min=0 max=20024 mean=823.3 std=762.0 (uint16 L2A 0-10000 expected)
- S2 post B3: min=0 max=18552 mean=1046.9 std=737.1 (uint16 L2A 0-10000 expected)
- S2 post B4: min=0 max=17464 mean=1274.0 std=809.5 (uint16 L2A 0-10000 expected)
- S2 post B5: min=0 max=17067 mean=1548.5 std=852.2 (uint16 L2A 0-10000 expected)
- S2 post B6: min=0 max=16791 mean=1847.5 std=885.9 (uint16 L2A 0-10000 expected)
- S2 post B7: min=0 max=16595 mean=2011.9 std=928.5 (uint16 L2A 0-10000 expected)
- S2 post B8A: min=0 max=16300 mean=2209.6 std=970.7 (uint16 L2A 0-10000 expected)
- S2 post B11: min=0 max=15583 mean=2708.9 std=979.6 (uint16 L2A 0-10000 expected)
- S2 post B12: min=0 max=15368 mean=2160.0 std=841.7 (uint16 L2A 0-10000 expected)
- S2 post SCL: min=0 max=11 mean=5.5 std=1.7 (uint16 L2A 0-10000 expected)
- S1 pre VV: min=-55.80dB max=24.91dB mean=-13.47 std=5.28 (stored x100 int16)
- S1 pre VH: min=-72.59dB max=24.05dB mean=-21.75 std=7.89 (stored x100 int16)
- S1 post VV: min=-52.51dB max=29.72dB mean=-13.43 std=5.44 (stored x100 int16)
- S1 post VH: min=-71.20dB max=24.24dB mean=-22.16 std=8.59 (stored x100 int16)
- SCL codes present: [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] (0 nodata 1 saturated 2 dark 3 shadow 4 veg 5 bare 6 water 7/8/9/10 cloud 11 snow)
- SCL-derived cloud_frac (mean pre/post, SCL in 7/8/9/10): min=0 p25=0.005396 med=0.05019 p75=0.1642 max=0.9276 (n=224); meta cloud_frac: min=0 p25=0 med=0.0135 p75=0.1425 max=0.4848 (n=224); pearson(SCL,meta)=0.1733 (pre-only 0.20, post-only 0.07, max 0.13, min 0.22 -> meta cloud_frac does NOT reproduce from SCL; it is scene-level metadata, NOT a chip SCL summary) -> CONCLUSION: use SCL raster as cloud gate at train/infer; meta cloud_frac only a rough flag
- SCL=0 (nodata) frac per chip: min=0 p25=0 med=0 p75=0 max=0.3564 (n=224); meta valid_frac: min=0.5152 p25=0.8575 med=0.9865 p75=1 max=1 (n=224) -> masking rec: valid = SCL!=0 AND mask!=255; also gate SCL in {3 cloud-shadow,7,8,9,10,11}
- dNBR global (stride-16 sample, finite): min=-1.456 p25=-0.02993 med=0.01581 p75=0.0844 max=1.39 (n=229376)
- BS aux landcover codes: chips_with_code={10: 210, 20: 13, 30: 223, 40: 218, 50: 207, 60: 187, 80: 204, 90: 200}
- BS aux dem (int16, m): min=-23 p25=32 med=83 p75=126 max=331 (n=917504); slope: min=0 p25=0 med=2 p75=4 max=13 (n=917504) (aux bands: dem, slope, landcover - NO aspect/exposure band on disk)
- dNBR per (landcover-group x severity), burned & cloud-free px only (regional-distribution method, case Fig.6):
  - forest sev1: med=0.1654 p25=0.1254 p75=0.2160 (n=67953)
  - forest sev2: med=0.3919 p25=0.3162 p75=0.4755 (n=54540)
  - forest sev3: med=0.6697 p25=0.5763 p75=0.7885 (n=17538)
  - forest THRESHOLDS: dNBR<0.2787->sev1; 0.2787<=dNBR<0.5308->sev2; dNBR>=0.5308->sev3
  - steppe sev1: med=0.1149 p25=0.0838 p75=0.1605 (n=729215)
  - steppe sev2: med=0.2977 p25=0.2518 p75=0.3438 (n=662368)
  - steppe sev3: med=0.4686 p25=0.4201 p75=0.5515 (n=437569)
  - steppe THRESHOLDS: dNBR<0.2063->sev1; 0.2063<=dNBR<0.3831->sev2; dNBR>=0.3831->sev3
  - cropland sev1: med=0.1133 p25=0.0886 p75=0.1443 (n=1118696)
  - cropland sev2: med=0.2686 p25=0.2203 p75=0.3209 (n=1003395)
  - cropland sev3: med=0.4804 p25=0.4253 p75=0.5547 (n=587259)
  - cropland THRESHOLDS: dNBR<0.1909->sev1; 0.1909<=dNBR<0.3745->sev2; dNBR>=0.3745->sev3
  - floodplain sev1: med=0.1629 p25=0.1140 p75=0.2318 (n=62329)
  - floodplain sev2: med=0.4535 p25=0.3879 p75=0.5254 (n=42163)
  - floodplain sev3: med=0.7567 p25=0.6646 p75=0.8445 (n=31791)
  - floodplain THRESHOLDS: dNBR<0.3082->sev1; 0.3082<=dNBR<0.6051->sev2; dNBR>=0.6051->sev3
  - other sev1: med=0.1448 p25=0.1105 p75=0.1884 (n=5809)
  - other sev2: med=0.3299 p25=0.2615 p75=0.4031 (n=692)
  - other sev3: med=0.4807 p25=0.4161 p75=0.5593 (n=63)
  - other THRESHOLDS: dNBR<0.2373->sev1; 0.2373<=dNBR<0.4053->sev2; dNBR>=0.4053->sev3
- per-(WorldCover-code x sev) burned px counts: {(10, 1): 67883, (10, 2): 54528, (10, 3): 17538, (20, 1): 70, (20, 2): 12, (30, 1): 792127, (30, 2): 816643, (30, 3): 462242, (40, 1): 1211331, (40, 2): 1029846, (40, 3): 607490, (50, 1): 3693, (50, 2): 321, (50, 3): 59, (60, 1): 2116, (60, 2): 371, (60, 3): 4, (80, 1): 191, (80, 2): 20, (80, 3): 7, (90, 1): 62138, (90, 2): 42143, (90, 3): 31784}

## 7. test meta cloud_frac NaN handling
- literal 'nan' strings in test cloud_frac: 180/269 (all AF: True); BS test rows all numeric
- rule: float(x) with x=='nan'->np.nan; never fill 0; cloud gate for test-AF from valid band, for test-BS from SCL raster (not meta)

## 8. aux availability (on disk vs derived)
- AF VIIRS file bands: I1 I2 I3 I4 I5 solar_zenith sensor_zenith valid (8x float32, 256x256) -> zeniths INSIDE viirs file, no separate fetch needed
- AF aux file bands: landcover dem t2m rh2m wind_speed (5x float32) -> WorldCover(v200 codes) + Copernicus DEM + ERA5-Land T/RH/wind ALL ON DISK
- BS sentinel2_{pre,post}: B2 B3 B4 B5 B6 B7 B8A B11 B12 SCL (10x uint16) -> SCL on disk
- BS sentinel1_{pre,post}: VV VH (2x int16, dB x100) -> real SAR lives here; sar_pre/ sar_post/ pre/ post/ dirs are EMPTY legacy (0 files)
- BS aux: dem slope landcover (3x int16) -> NO aspect/exposure band (derive from DEM if needed)
- chip sizes/dtypes: AF 256x256 float32, mask uint8 nodata=255; BS 512x512 S2 uint16 / S1 int16 / aux int16, mask uint8 nodata=255

## 9. train/val split proposal (deterministic, seed 19)
- BS: grouped by fire_event_id (224 unique events, 1 chip each), stratified by dominant severity -> train 180 / val 44 chips (val_frac=0.196)
- AF: fire_event_id EMPTY -> FALLBACK chip-grouped split stratified by (satellite x has_fire), seeded shuffle -> train 335 / val 85 chips (val_frac=0.202); leakage caveat: same-fire chips may straddle the split since event ids are absent (mitigate with coords/time-block check in training)
- split lists: eda/split.json {af:{train,val}, bs:{train,val}}

## 10. machine-readable config (training consumes this block)
```yaml
nbr_recipe: NBR=(B8A-B12)/(B8A+B12+1e-6); dNBR=NBR_pre-NBR_post; RdNBR=dNBR/(sqrt(|NBR_pre|)+1e-6)
s2_bands_order: [B2,B3,B4,B5,B6,B7,B8A,B11,B12,SCL]  # indices 0..9, B8A=6 B12=8 SCL=9
s1_bands_order: [VV,VH]  # int16, divide by 100 -> dB
af_viirs_bands: [I1,I2,I3,I4,I5,solar_zenith,sensor_zenith,valid]
af_aux_bands: [landcover,dem,t2m,rh2m,wind_speed]
bs_aux_bands: [dem,slope,landcover]
af_I1_mean: 0.119404
af_I1_std: 0.120158
af_I2_mean: 0.190323
af_I2_std: 0.134409
af_I3_mean: 0.165547
af_I3_std: 0.079410
af_I4_mean: 295.024043
af_I4_std: 14.621845
af_I5_mean: 286.473070
af_I5_std: 15.926562
af_I4_I5_mean: 8.550973
af_I4_I5_std: 9.960024
af_I4_clip_K: 367.0
af_night_px_frac: 0.299746
af_night_chips: 127
af_daynight_rule: solar_zenith>90 -> night; I1-I3 NaN at night (reflectance unusable); I4/I5 day+night
bs_pre_B2_mean: 841.779
bs_pre_B2_std: 817.667
bs_pre_B3_mean: 1083.916
bs_pre_B3_std: 789.707
bs_pre_B4_mean: 1316.925
bs_pre_B4_std: 873.006
bs_pre_B5_mean: 1624.533
bs_pre_B5_std: 914.837
bs_pre_B6_mean: 1980.910
bs_pre_B6_std: 932.526
bs_pre_B7_mean: 2166.944
bs_pre_B7_std: 979.813
bs_pre_B8A_mean: 2377.748
bs_pre_B8A_std: 1018.241
bs_pre_B11_mean: 2818.593
bs_pre_B11_std: 1014.201
bs_pre_B12_mean: 2166.967
bs_pre_B12_std: 872.877
bs_post_B2_mean: 823.318
bs_post_B2_std: 761.974
bs_post_B3_mean: 1046.856
bs_post_B3_std: 737.075
bs_post_B4_mean: 1274.007
bs_post_B4_std: 809.453
bs_post_B5_mean: 1548.542
bs_post_B5_std: 852.226
bs_post_B6_mean: 1847.486
bs_post_B6_std: 885.861
bs_post_B7_mean: 2011.933
bs_post_B7_std: 928.535
bs_post_B8A_mean: 2209.590
bs_post_B8A_std: 970.695
bs_post_B11_mean: 2708.865
bs_post_B11_std: 979.604
bs_post_B12_mean: 2159.985
bs_post_B12_std: 841.698
bs_s1_pre_VV_mean_dB: -13.4705
bs_s1_pre_VV_std_dB: 5.2820
bs_s1_pre_VH_mean_dB: -21.7489
bs_s1_pre_VH_std_dB: 7.8894
bs_s1_post_VV_mean_dB: -13.4316
bs_s1_post_VV_std_dB: 5.4357
bs_s1_post_VH_mean_dB: -22.1552
bs_s1_post_VH_std_dB: 8.5948
dnbr_thresholds_per_landcover_group:  # dNBR<cut12->1; cut12<=dNBR<cut23->2; dNBR>=cut23->3
  forest: {cut12: 0.2787, cut23: 0.5308}
  steppe: {cut12: 0.2063, cut23: 0.3831}
  cropland: {cut12: 0.1909, cut23: 0.3745}
  floodplain: {cut12: 0.3082, cut23: 0.6051}
  other: {cut12: 0.2373, cut23: 0.4053}
cloud_gate: SCL in [7,8,9,10] + shadow 3 + snow 11; valid = SCL!=0 and mask!=255
split: {seed: 19, af_train: 335, af_val: 85, bs_train: 180, bs_val: 44}
```

_wall_time_s: 29_
