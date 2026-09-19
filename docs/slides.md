# Слайды — kosmohack: мониторинг природных пожаров (10 слайдов)

## 1. Титул
kosmohack — двухэтапный мониторинг природных пожаров по ДЗЗ.
AF-детекция + BS-картирование гарей + аналитический сервис.
Score 0.654 · submission.csv FINAL · сервис на localhost:8000.

## 2. Проблема
Ложные срабатывания FIRMS на техногенных источниках; пропуски малых
пожаров; оценка гарей — выборочно, с задержкой в недели, без единой методики.
Нужно: чистые термоточки + контуры гарей по степеням + запрос без ручных
операций с растрами.

## 3. Схема решения
VIIRS AF Unet → S2 BS Unet → inference.py → submission.csv → FastAPI-сервис.
Два модуля → общий файл ответа → сервис поверх обоих (кейс, рис. 1).

## 4. Модуль AF
Unet-effb4, thr 0.15 + WC-гейт lc50/I4≥310K. Val micro-F1 0.8434
(day 0.8519 / night 0.803). Отделяет природное горение от факелов и
промплощадок.

## 5. Модуль BS
smp-Unet effb4, 27 каналов, argmax + cloud→0 + burn-gate snap.
IoU_burn 0.5239 · mIoU_sev 0.5430. Ветка Prithvi мертва (0.16 < 0.33).

## 6. Метрика и submission FINAL
Score = 0.35·F1_af + 0.35·IoU_burn + 0.30·mIoU_sev = **0.654**.
submission.csv FINAL: 4.1M, 447 строк, RLE валиден, NaN 0, пересечений 0.
Переобучение запрещено.

## 7. Сервис: API
GET /v1/health, /v1/fires (точки), /v1/burned-areas (полигоны
{id, severity, severity_label, area_ha, date_pre, date_post}),
/v1/analytics ({fire_points, total_burned_ha, by_severity_ha/pct, bbox,
dates, projection}), POST /v1/*/query по полигону. Форматы geojson/json/shp.
Файловый бэк, без БД.

## 8. Сервис: E2E
localhost:8000 + index.html: 129 точек, 253 полигона, справка без нулей
(129 / 1.21M га / 40.25/34.93/24.81%). CORS только localhost:3000/5173/8000.
Серия для защиты: docs/e2e_curl.sh.

## 9. Воспроизводимость
Сиды 19, детерминированный inference, requirements.lock, веса+пороги в
weights/, README-прогон, отчёт docs/report.md.

## 10. Ограничения и next steps
Demo-привязка чипов (§5.5); площади по bbox-компонентам. Далее: контурная
векторизация, выверка GSD-площадей AF. Фриз зафиксирован — сабмит final.
