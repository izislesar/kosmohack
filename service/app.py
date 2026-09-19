#!/usr/bin/env python3
"""Service Variant A: fires + burned-areas + analytics over demo chip index.

Demo mapping (documented, honest): test chips are de-identified (no geo/dates
per case section 5.5), so the service serves precomputed inference outputs
(submission.csv) tiled across the fire-aoi bbox in chip index order with
deterministic demo dates. Geometry math (area_ha, GeoJSON, SHP) is real;
placement is a documented demo grid, not recovered georeferencing.

Endpoints:
  GET  /v1/fires?bbox=lon_min,lat_min,lon_max,lat_max&date_from&date_to&format=
  GET  /v1/burned-areas?...&severity=1,2,3&format= -> properties {id, severity, severity_label, area_ha, date_pre, date_post}
  GET  /v1/analytics?...  -> {fire_points, total_burned_ha, by_severity_ha{1,2,3}, by_severity_pct, bbox, date_from, date_to, projection_for_area}
  GET  /v1/health -> {status: ok}
  POST /v1/{fires,burned-areas,analytics}/query {polygon: GeoJSON, date_from, date_to, ...}
Formats: geojson (default) | json (analytics summary / raw) | shp (minimal ESRI polygon zip)
CORS: http://localhost:3000, http://localhost:5173, http://localhost:8000 only
Env: SUBMISSION_CSV (default submission.csv), AOI_GEOJSON
Run: SUBMISSION_CSV=submission.csv uvicorn service.app:app --host 0.0.0.0 --port 8000 (from repo root)
"""
import csv
import io
import json
import os
import struct
import zipfile

import numpy as np

SUBMISSION = os.environ.get("SUBMISSION_CSV", "submission.csv")
AOI_GEOJSON = os.environ.get("AOI_GEOJSON", "data/fire-aoi/fire_monitoring_aoi.geojson")
GSD = {"AF_te": 375.0, "BS_te": 20.0}
SEVERITY_LABELS = {1: "Weak", 2: "Moderate", 3: "Strong"}
PROJECTION_FOR_AREA = "EPSG:6933"

_cache = {}


def rle_decode(s, h, w):
    m = np.zeros(h * w, dtype=np.uint8)
    s = (s or "").strip()
    if not s:
        return m.reshape(h, w)
    nums = list(map(int, s.split()))
    for st, ln in zip(nums[0::2], nums[1::2]):
        m[st - 1:st - 1 + ln] = 1
    return m.reshape(h, w)


def load_index():
    if _cache.get("ready"):
        return _cache
    # AOI bbox
    try:
        with open(AOI_GEOJSON) as fh:
            gj = json.load(fh)
        xs, ys = [], []
        def walk(c):
            if isinstance(c[0], (int, float)):
                xs.append(c[0]); ys.append(c[1])
            else:
                for x in c:
                    walk(x)
        for f in gj.get("features", gj if isinstance(gj, list) else [gj]):
            walk(f["geometry"]["coordinates"] if "geometry" in f else f.get("coordinates", []))
        aoi = (min(xs), min(ys), max(xs), max(ys))
    except Exception:
        aoi = (37.0, 44.0, 44.0, 48.0)  # S/SW Russia fallback box
    rows = list(csv.DictReader(open(SUBMISSION)))
    feats, fires, burns = [], [], []
    n = len({r["chip_id"] for r in rows}) or 1
    # tile AOI in index order
    cols = int(np.ceil(np.sqrt(n)))
    dx = (aoi[2] - aoi[0]) / cols
    dy = (aoi[3] - aoi[1]) / cols
    chips = sorted({r["chip_id"] for r in rows})
    fid = 0
    for i, chip in enumerate(chips):
        cx, cy = aoi[0] + (i % cols) * dx, aoi[3] - (i // cols + 1) * dy
        prefix = chip[:5]
        h, w = (256, 256) if prefix == "AF_te" else (512, 512)
        gsd = GSD[prefix]
        date = f"2025-{(i % 7) + 4:02d}-{(i % 28) + 1:02d}"  # deterministic demo date
        for r in [x for x in rows if x["chip_id"] == chip]:
            k = int(r["class_id"])
            mask = rle_decode(r["rle"], h, w)
            if not mask.any():
                continue
            # vectorize: connected components -> contour bboxes as polygons (chip-px -> demo-deg)
            try:
                import cv2
                ncc, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
                comps = [(stats[j, 0], stats[j, 1], stats[j, 2], stats[j, 3], cent[j])
                         for j in range(1, ncc)][:50]
            except Exception:
                ys_, xs_ = np.where(mask)
                comps = [(xs_.min(), ys_.min(), xs_.max() - xs_.min() + 1,
                          ys_.max() - ys_.min() + 1, (xs_.mean(), ys_.mean()))]
            for (bx, by, bw, bh, (ccx, ccy)) in comps:
                # px->deg scale: chip spans its tile
                x0 = cx + (bx / w) * dx
                x1 = cx + ((bx + bw) / w) * dx
                y0 = cy + (by / h) * dy
                y1 = cy + ((by + bh) / h) * dy
                poly = [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]
                area_ha = float(bw * bh * gsd * gsd / 1e4)
                lon, lat = cx + (ccx / w) * dx, cy + (ccy / h) * dy
                fid += 1
                f = {"fid": f"{chip}:{k}:{fid}", "chip_id": chip, "class_id": k, "date": date,
                     "lon": lon, "lat": lat, "poly": poly, "area_ha": area_ha,
                     "bbox": [x0, y0, x1, y1]}
                feats.append(f)
                (fires if prefix == "AF_te" else burns).append(f)
    _cache.update({"ready": True, "aoi": aoi, "feats": feats, "fires": fires, "burns": burns})
    return _cache


def in_bbox(f, bbox):
    x0, y0, x1, y1 = bbox
    fx, fy = f["lon"], f["lat"]
    return (x0 <= fx <= x1) and (y0 <= fy <= y1)


def in_poly(f, poly_coords):
    # ray-cast on outer ring
    ring = poly_coords[0] if isinstance(poly_coords[0][0], (list, tuple)) else poly_coords
    x, y = f["lon"], f["lat"]
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def parse_bbox(s):
    v = list(map(float, s.split(",")))
    assert len(v) == 4
    return v


def filter_feats(feats, bbox=None, d0=None, d1=None, poly=None, sev=None):
    out = []
    for f in feats:
        if bbox and not in_bbox(f, bbox):
            continue
        if poly and not in_poly(f, poly):
            continue
        if d0 and f["date"] < d0:
            continue
        if d1 and f["date"] > d1:
            continue
        if sev and f["class_id"] not in sev:
            continue
        out.append(f)
    return out


def as_geojson(feats, kind):
    out = []
    for f in feats:
        if kind == "fires":
            props = {"id": f.get("fid", f["chip_id"]),
                     "date": f["date"],
                     "area_ha": round(f["area_ha"], 2)}
            geom = {"type": "Point", "coordinates": [f["lon"], f["lat"]]}
        else:
            sev = int(f["class_id"])
            props = {"id": f.get("fid", f["chip_id"]),
                     "severity": sev,
                     "severity_label": SEVERITY_LABELS.get(sev, f"Level {sev}"),
                     "area_ha": round(f["area_ha"], 2),
                     "date_pre": f["date"],
                     "date_post": f["date"]}
            geom = {"type": "Polygon", "coordinates": f["poly"]}
        out.append({"type": "Feature", "geometry": geom, "properties": props})
    return {"type": "FeatureCollection", "features": out}


def shp_zip(feats, kind):
    """Minimal ESRI Shapefile (Polygon type 5) + SHX + DBF + CPG in a zip. No deps."""
    polys = []
    for f in feats:
        ring = f["poly"][0] if kind != "fires" else [
            [f["lon"] - 1e-4, f["lat"] - 1e-4], [f["lon"] + 1e-4, f["lat"] - 1e-4],
            [f["lon"] + 1e-4, f["lat"] + 1e-4], [f["lon"] - 1e-4, f["lat"] + 1e-4],
            [f["lon"] - 1e-4, f["lat"] - 1e-4]]
        polys.append((ring, f))
    shp, shx = io.BytesIO(), io.BytesIO()
    # headers (100 bytes)
    def header(n, length):
        h = struct.pack(">6i", 9994, 0, 0, 0, 0, 0) + struct.pack(">2i", 1000, length // 2)
        h += struct.pack("<i8d", 5, -180, -90, 180, 90, 0, 0, 0, 0)
        return h
    recs = []
    for ring, f in polys:
        pts = struct.pack("<" + "2d" * len(ring), *[c for p in ring for c in p[:2]])
        content = struct.pack("<ii4dii", 5, len(ring), 0, 0, 0, 0, 1, len(ring)) + \
            struct.pack("<i", 0) + pts
        recs.append(struct.pack(">2i", 0, len(content) // 2) + content)
    allrec = b"".join(recs)
    shp.write(header(0, 50 + len(allrec) // 2) + allrec)
    off = 50
    idx = b""
    for r in recs:
        idx += struct.pack(">2i", off, len(r) // 2 - 4)
        off += len(r) // 2
    shx.write(header(0, 50 + len(idx) // 2) + idx)
    # DBF: CHIP_ID C(16), CLASS_ID N, DATE C(10), AREA_HA N
    dbf = io.BytesIO()
    nf = len(polys)
    dbf.write(struct.pack("<4BIIHH16s4s", 3, 126, 9, 19, nf, 129, 32, 0, b"\0" * 16, b"\0" * 4))
    dbf.write(struct.pack("<11sB4B14s", b"CHIP_ID", 67, 16, 0, 0, 0, b"\0" * 14))
    dbf.write(struct.pack("<11sB4B14s", b"CLASS_ID", 78, 4, 0, 0, 0, b"\0" * 14))
    dbf.write(struct.pack("<11sB4B14s", b"DATE", 67, 10, 0, 0, 0, b"\0" * 14))
    dbf.write(struct.pack("<11sB4B14s", b"AREA_HA", 78, 12, 2, 0, 0, b"\0" * 14))
    dbf.write(b"\x0d")
    for _, f in polys:
        dbf.write(b" " + f["chip_id"][:16].ljust(16).encode()[:16] +
                  str(f["class_id"])[:4].rjust(4).encode() +
                  f["date"][:10].ljust(10).encode()[:10] +
                  f"{f['area_ha']:.2f}"[:12].rjust(12).encode())
    dbf.write(b"\x1a")
    z = io.BytesIO()
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("export.shp", shp.getvalue())
        zf.writestr("export.shx", shx.getvalue())
        zf.writestr("export.dbf", dbf.getvalue())
        zf.writestr("export.cpg", "UTF-8")
    z.seek(0)
    return z.read()


def analytics_summary(burns, fires=None, bbox=None, d0=None, d1=None):
    s = {1: 0.0, 2: 0.0, 3: 0.0}
    for f in burns:
        k = int(f["class_id"])
        if k in s:
            s[k] += float(f["area_ha"])
    total = round(sum(s.values()), 2)
    ha = {1: round(s[1], 2), 2: round(s[2], 2), 3: round(s[3], 2)}
    pct = {1: round(s[1] / total * 100, 2) if total > 0 else 0.0,
           2: round(s[2] / total * 100, 2) if total > 0 else 0.0,
           3: round(s[3] / total * 100, 2) if total > 0 else 0.0}
    return {"fire_points": len(fires or []),
            "total_burned_ha": total,
            "total_area_ha": total,
            "by_severity_ha": ha,
            "by_severity": {str(k): v for k, v in ha.items()},
            "by_severity_pct": pct,
            "weak_ha": ha[1], "moderate_ha": ha[2], "strong_ha": ha[3],
            "bbox": list(bbox) if bbox is not None else None,
            "date_from": d0 or None,
            "date_to": d1 or None,
            "projection_for_area": PROJECTION_FOR_AREA}


# ---- FastAPI wiring (import-time safe without fastapi for precompute tests) ----
try:
    from fastapi import FastAPI, Query, Response
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="kosmohack fire service (Variant A)")
    app.add_middleware(CORSMiddleware, allow_origins=[
        "http://localhost:3000", "http://localhost:5173", "http://localhost:8000"],
        allow_methods=["*"], allow_headers=["*"])

    def _fmt(feats, kind, fmt):
        if fmt == "shp":
            return Response(content=shp_zip(feats, kind), media_type="application/zip",
                            headers={"Content-Disposition": "attachment; filename=export.zip"})
        gj = as_geojson(feats, kind)
        return gj if fmt == "geojson" else {"features": gj["features"], "count": len(feats)}

    @app.get("/v1/fires")
    def get_fires(bbox: str = Query(...), date_from: str = "", date_to: str = "",
                  format: str = "geojson"):
        idx = load_index()
        f = filter_feats(idx["fires"], bbox=parse_bbox(bbox),
                         d0=date_from or None, d1=date_to or None)
        return _fmt(f, "fires", format)

    @app.get("/v1/burned-areas")
    def get_burned(bbox: str = Query(...), date_from: str = "", date_to: str = "",
                   severity: str = "", format: str = "geojson"):
        idx = load_index()
        sev = {int(x) for x in severity.split(",") if x.strip()} or None
        f = filter_feats(idx["burns"], bbox=parse_bbox(bbox),
                         d0=date_from or None, d1=date_to or None, sev=sev)
        return _fmt(f, "burned-areas", format)

    @app.get("/v1/health")
    def get_health():
        return {"status": "ok"}

    @app.get("/v1/analytics")
    def get_analytics(bbox: str = Query(...), date_from: str = "", date_to: str = ""):
        idx = load_index()
        bb = parse_bbox(bbox)
        d0, d1 = date_from or None, date_to or None
        b = filter_feats(idx["burns"], bbox=bb, d0=d0, d1=d1)
        fr = filter_feats(idx["fires"], bbox=bb, d0=d0, d1=d1)
        return analytics_summary(b, fr, bbox=bb, d0=d0, d1=d1)

    @app.post("/v1/{kind}/query")
    def post_query(kind: str, body: dict):
        idx = load_index()
        poly = body.get("polygon", {}).get("coordinates") or body.get("polygon")
        d0, d1 = body.get("date_from"), body.get("date_to")
        if kind == "fires":
            f = filter_feats(idx["fires"], d0=d0, d1=d1, poly=poly)
            return as_geojson(f, "fires")
        if kind == "burned-areas":
            sev = set(body.get("severity", []) or []) or None
            f = filter_feats(idx["burns"], d0=d0, d1=d1, poly=poly, sev=sev)
            return as_geojson(f, "burned-areas")
        if kind == "analytics":
            b = filter_feats(idx["burns"], d0=d0, d1=d1, poly=poly)
            fr = filter_feats(idx["fires"], d0=d0, d1=d1, poly=poly)
            return analytics_summary(b, fr, bbox=None, d0=d0, d1=d1)
        return {"error": "unknown kind"}
except ImportError:
    app = None  # fastapi absent (won't happen on VPS); precompute fns still usable
