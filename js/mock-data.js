const MOCK_FIRES = {
    type: "FeatureCollection",
    features: [
        {
            type: "Feature",
            geometry: { type: "Point", coordinates: [46.034, 51.533] },
            properties: {
                id: "F-001",
                satellite: "Suomi NPP",
                acq_datetime: "2026-09-18T08:32:00",
                brightness_i4_k: 367.2,
                confidence: 0.92
            }
        },
        {
            type: "Feature",
            geometry: { type: "Point", coordinates: [45.970, 51.620] },
            properties: {
                id: "F-002",
                satellite: "NOAA-20",
                acq_datetime: "2026-09-18T09:15:00",
                brightness_i4_k: 358.7,
                confidence: 0.85
            }
        },
        {
            type: "Feature",
            geometry: { type: "Point", coordinates: [46.180, 51.420] },
            properties: {
                id: "F-003",
                satellite: "NOAA-21",
                acq_datetime: "2026-09-18T10:04:00",
                brightness_i4_k: 351.3,
                confidence: 0.78
            }
        },
        {
            type: "Feature",
            geometry: { type: "Point", coordinates: [46.310, 51.710] },
            properties: {
                id: "F-004",
                satellite: "Suomi NPP",
                acq_datetime: "2026-09-18T11:22:00",
                brightness_i4_k: 364.9,
                confidence: 0.88
            }
        },
        {
            type: "Feature",
            geometry: { type: "Point", coordinates: [45.820, 51.480] },
            properties: {
                id: "F-005",
                satellite: "NOAA-20",
                acq_datetime: "2026-09-18T12:11:00",
                brightness_i4_k: 349.1,
                confidence: 0.71
            }
        }
    ]
};


const MOCK_BURNED_AREAS = {
    type: "FeatureCollection",
    features: [
        {
            type: "Feature",
            geometry: {
                type: "Polygon",
                coordinates: [[
                    [46.005, 51.510],
                    [45.995, 51.525],
                    [46.025, 51.545],
                    [46.055, 51.535],
                    [46.045, 51.510],
                    [46.005, 51.510]
                ]]
            },
            properties: {
                id: "B-001",
                severity: 3,
                severity_label: "Strong",
                area_ha: 61.6,
                date_pre: "2026-08-15",
                date_post: "2026-09-18"
            }
        },
        {
            type: "Feature",
            geometry: {
                type: "Polygon",
                coordinates: [[
                    [45.940, 51.595],
                    [45.930, 51.615],
                    [45.960, 51.645],
                    [46.000, 51.640],
                    [45.985, 51.610],
                    [45.940, 51.595]
                ]]
            },
            properties: {
                id: "B-002",
                severity: 2,
                severity_label: "Moderate",
                area_ha: 81.0,
                date_pre: "2026-08-20",
                date_post: "2026-09-18"
            }
        },
        {
            type: "Feature",
            geometry: {
                type: "Polygon",
                coordinates: [[
                    [46.145, 51.395],
                    [46.130, 51.415],
                    [46.155, 51.445],
                    [46.195, 51.450],
                    [46.205, 51.420],
                    [46.145, 51.395]
                ]]
            },
            properties: {
                id: "B-003",
                severity: 1,
                severity_label: "Weak",
                area_ha: 42.0,
                date_pre: "2026-08-25",
                date_post: "2026-09-18"
            }
        }
    ]
};

function filterMockFiresByBbox(fires, bbox) {
    if (!bbox) return fires;

    const [lonMin, latMin, lonMax, latMax] = bbox;

    return {
        type: "FeatureCollection",
        features: fires.features.filter(function (f) {
            const c = f.geometry.coordinates;
            return c[0] >= lonMin && c[0] <= lonMax &&
                   c[1] >= latMin && c[1] <= latMax;
        })
    };
}


function isPointInPolygon(lon, lat, polygon) {    
    let ring = null;

    if (
        polygon &&
        polygon.type === "Polygon" &&
        Array.isArray(polygon.coordinates)
    ) {
        ring = polygon.coordinates[0];
    } else if (
        Array.isArray(polygon) &&
        Array.isArray(polygon[0]) &&
        Array.isArray(polygon[0][0])
    ) {
        ring = polygon[0];
    } else if (
        Array.isArray(polygon) &&
        Array.isArray(polygon[0]) &&
        typeof polygon[0][0] === "number"
    ) {
        ring = polygon;
    }

    if (!ring || ring.length < 3) {
        return false;
    }

    let inside = false;

    for (
        let i = 0, j = ring.length - 1;
        i < ring.length;
        j = i++
    ) {
        const xi = ring[i][0];
        const yi = ring[i][1];
        const xj = ring[j][0];
        const yj = ring[j][1];

        const intersect =
            ((yi > lat) !== (yj > lat)) &&
            (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi);

        if (intersect) {
            inside = !inside;
        }
    }
    return inside;
}

function filterMockFiresByPolygon(fires, polygon) {  
    if (!polygon) return fires;

    return {
        type: "FeatureCollection",
        features: fires.features.filter(function (f) {
            const c = f.geometry.coordinates;
            return isPointInPolygon(c[0], c[1], polygon);
        })
    };
}

function filterMockBurnedByBbox(areas, bbox) {
    if (!bbox) return areas;

    const [lonMin, latMin, lonMax, latMax] = bbox;

    return {
        type: "FeatureCollection",
        features: areas.features.filter(function (f) {
            const c = f.geometry.coordinates[0][0];
            return c[0] >= lonMin && c[0] <= lonMax &&
                   c[1] >= latMin && c[1] <= latMax;
        })
    };
}

function filterMockBurnedByPolygon(areas, polygon) {
    if (!polygon) return areas;

    return {
        type: "FeatureCollection",
        features: areas.features.filter(function (f) {
            const ring = f.geometry.coordinates[0];

            return ring.some(function (c) {
                return isPointInPolygon(c[0], c[1], polygon);
            });
        })
    };
}

function filterMockSeverity(areas, severity) {
    if (!severity || severity.length === 0) {
        return { type: "FeatureCollection", features: [] };
    }

    return {
        type: "FeatureCollection",
        features: areas.features.filter(function (f) {
            return severity.indexOf(Number(f.properties.severity)) !== -1;
        })
    };
}

function calculateMockAnalytics(fires, burnedAreas) {
    const bySeverity = { 1: 0, 2: 0, 3: 0 };
    let total = 0;

    if (burnedAreas && burnedAreas.features) {
        burnedAreas.features.forEach(function (f) {
            const sev = Number(f.properties.severity);
            const area = Number(f.properties.area_ha) || 0;
            if (bySeverity[sev] !== undefined) bySeverity[sev] += area;
            total += area;
        });
    }

    return {
        fire_points: fires && fires.features ? fires.features.length : 0,
        total_burned_ha: total,
        by_severity_ha: bySeverity
    };
}