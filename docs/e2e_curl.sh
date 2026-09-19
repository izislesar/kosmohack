#!/usr/bin/env bash
# E2E curl-серия для защиты (localhost:8000). Предполагает запущенный сервис:
#   SUBMISSION_CSV=submission.csv AOI_GEOJSON=data/fire-aoi/fire_monitoring_aoi.geojson \
#     python3 -m uvicorn service.app:app --host 0.0.0.0 --port 8000
# Фронт: открыть index.html в браузере (file:// или http://localhost:3000),
#   js/config.js -> API_BASE http://localhost:8000, USE_MOCK false.
set -u
B="37.0,44.0,44.0,48.0"
D="date_from=2025-04-01&date_to=2025-10-31"
echo "== 1. health =="
curl -s http://localhost:8000/v1/health; echo
echo "== 2. fires (точки) =="
curl -s "http://localhost:8000/v1/fires?bbox=$B&$D" | python3 -c "import json,sys; d=json.load(sys.stdin); print('points:',len(d['features'])); print('sample:',d['features'][0]['properties'])"
echo "== 3. burned-areas (полигоны + severity) =="
curl -s "http://localhost:8000/v1/burned-areas?bbox=$B&$D" | python3 -c "import json,sys; d=json.load(sys.stdin); print('polys:',len(d['features'])); print('sample:',d['features'][0]['properties'])"
echo "== 4. analytics (справка, без нулей) =="
curl -s "http://localhost:8000/v1/analytics?bbox=$B&$D"; echo
echo "== 5. severity filter =="
curl -s "http://localhost:8000/v1/burned-areas?bbox=$B&$D&severity=3" | python3 -c "import json,sys; d=json.load(sys.stdin); s={f['properties']['severity'] for f in d['features']}; print('sev3 n:',len(d['features']),'sevs:',s)"
echo "== 6. POST polygon =="
curl -s -X POST http://localhost:8000/v1/analytics/query -H "Content-Type: application/json" -d '{"polygon":{"type":"Polygon","coordinates":[[[37,44],[44,44],[44,48],[37,48],[37,44]]]},"date_from":"2025-04-01","date_to":"2025-10-31"}'; echo
echo "== 7. CORS allow localhost:3000 =="
curl -s -o /dev/null -D - http://localhost:8000/v1/health -H "Origin: http://localhost:3000" | grep -i access-control-allow-origin
echo "== 8. CORS deny spcase.ru (ожидается пусто) =="
curl -s -o /dev/null -D - http://localhost:8000/v1/health -H "Origin: https://spcase.ru" | grep -i access-control-allow-origin || echo OK-deny
echo "== 9. config.js =="
grep -E "API_BASE|USE_MOCK" js/config.js
