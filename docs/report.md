# Отчёт — kosmohack: мониторинг природных пожаров (AF + BS + сервис)

## 1. Задача
Двухэтапный мониторинг по ДЗЗ (кейс §0–§2): модуль детекции активного горения
по ИК-каналам VIIRS (375 м) с отсечением техногенных термоаномалий + модуль
картирования гарей по паре Sentinel-2 «до/после» (20 м) с классификацией
степени поражения 1/2/3 + информационно-аналитический сервис
(термоточки, контуры гарей, справка с площадями) + `submission.csv`.

## 2. Данные
Train/test чипы AF (VIIRS I1–I5 + вспомогательные слои) и BS (S2-пара +
S1-пара, рельеф, тип покрова). Тестовые чипы де-идентифицированы (кейс §5.5:
без гео/дат) — сервис отдаёт предвычисленные выходы, размещённые
детерминированной demo-сеткой по bbox AOI (геометрия честная, привязка —
документированное demo, не восстановленная геопривязка).

## 3. Модели
- **AF:** Unet-effb4, `weights/af.pt` + `weights/af_thresh.json` (thr 0.15,
  WC-гейт lc50/I4≥310K). Val micro-F1 0.8434 (day 0.8519 / night 0.803).
- **BS:** smp-Unet effb4, 27 каналов, `weights/bs.pt` (ep91, fallback-ветка —
  победитель); argmax + cloud→0 + burn-gate snap. Ветка Prithvi мертва
  (IoU 0.16 < бейзлайн 0.33).
- Сиды 19 везде; детерминизм: 2 прогона inference — побайтово идентичны.

## 4. Метрика (кейс §7)
`Score = 0.35·F1_af + 0.35·IoU_burn + 0.30·mIoU_sev` (micro-pool).
Фриз: F1_af 0.8434 · IoU_burn 0.5239 · mIoU_sev 0.5430 → **Score 0.654**.

## 5. Inference (контракт, кейс §8)
`PYTHONPATH=src/models python3 inference.py --data-dir data/test --output submission.csv`
(~28 c на RTX PRO 4500). Шаблон 447 строк (180 AF×1 + 89 BS×3); RLE
1-indexed row-major, старты по возрастанию, касающиеся раны склеены,
пусто → `""`.

## 6. submission.csv — FINAL
4.1M, 447 строк, md5 `97b86c6783c14e168353b93741f1f224`. Валидация
(`validate_submission.py` vs `docs/sample_submission_1.csv`): пары chip/class
совпадают, RLE валиден, NaN 0, BS-пересечений 0, pred-пикселей 2706635.
Переобучение запрещено — файл заморожен.

## 7. Сервис (Variant A, файловый бэк, без БД, только localhost:8000)
- `GET /v1/health` → `{status: ok}`.
- `GET /v1/fires` — точки; `GET /v1/burned-areas` — полигоны, properties
  `{id, severity, severity_label, area_ha, date_pre, date_post}`;
  `severity=1,2,3`-фильтр; форматы geojson/json/shp-zip.
- `GET /v1/analytics` → `{fire_points, total_burned_ha, by_severity_ha{1,2,3},
  by_severity_pct, bbox, date_from, date_to, projection_for_area}` (+ алиасы
  `total_area_ha/weak_ha/moderate_ha/strong_ha/by_severity` для прямого рендера).
- `POST /v1/{fires,burned-areas,analytics}/query` — запрос по полигону.
- CORS только `http://localhost:3000`, `:5173`, `:8000`; `SUBMISSION_CSV` из env.
- Фронт: `index.html` + `js/config.js` (`API_BASE http://localhost:8000`,
  `USE_MOCK false`); `frontend/*` и `js/*` (кроме config.js) не тронуты.

## 8. E2E локально (лог `docs/e2e_curl.sh`)
uvicorn на :8000 + `index.html`: 129 точек, 253 полигона, справка
`fire_points 129, total_burned_ha 1213491.28 (40.25/34.93/24.81%)` — без нулей;
фильтр sev3 → 81 полигон только severity 3; polygon-POST совпадает с bbox;
CORS allow :3000, deny spcase.ru; ID всех stat-элементов и порядок скриптов
в `index.html` проверены.

## 9. Воспроизводимость
`requirements.lock` (torch cu128 + fastapi 0.115.6 + uvicorn 0.34.0 и др.),
сиды 19, веса и пороги в `weights/`, E2E-серия в `docs/e2e_curl.sh`.

## 10. Ограничения и следующие шаги
Demo-привязка чипов (нет гео в тесте); площади — по bbox-компонентам;
следующий шаг — уточнение векторизации (контуры вместо bbox) и выверка
GSD-площадей точек AF. Текущий фриз — финальный сабмит.
