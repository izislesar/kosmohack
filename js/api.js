async function apiRequest(endpoint, options = {}) {
    const url = apiUrl(endpoint);

    const requestOptions = {
        method: options.method || "GET",

        headers: {
            "Accept": "application/json",

            ...(options.body
                ? {
                    "Content-Type":
                        "application/json"
                }
                : {}),

            ...(options.headers || {})
        }
    };

    if (options.body) {

        requestOptions.body =
            typeof options.body === "string"
                ? options.body
                : JSON.stringify(options.body);
    }

    const response =
        await fetch(
            url,
            requestOptions
        );

    if (!response.ok) {

        let errorMessage =
            `API request failed: ${response.status} ${response.statusText}`;
        try {
            const errorData =
                await response.json();

            if (errorData.detail) {
                errorMessage =
                    String(errorData.detail);
            }

            else if (errorData.message) {
                errorMessage =
                    String(errorData.message);
            }

        } catch (_) {}
        throw new Error(errorMessage);
    }

    if (response.status === 204) {
        return null;
    }

    return response.json();
}

function bboxToString(bbox) {

    if (
        !Array.isArray(bbox) ||
        bbox.length !== 4
    ) {
        throw new Error(
            "Invalid bbox. Expected [west, south, east, north]."
        );
    }

    return bbox
        .map(Number)
        .join(",");
}

function buildQuery(params = {}) {
    const query =
        new URLSearchParams();

    Object.entries(params)
        .forEach(([key, value]) => {

            if (
                value === undefined ||
                value === null ||
                value === ""
            ) {
                return;
            }

            if (Array.isArray(value)) {

                if (value.length === 0) {
                    return;
                }

                query.set(
                    key,
                    value.join(",")
                );

                return;
            }

            query.set(
                key,
                String(value)
            );
        });

    const result =
        query.toString();

    return result
        ? `?${result}`
        : "";
}

async function fetchFiresByBbox(
    bbox,
    dateFrom,
    dateTo
) {

    if (isMockMode()) {

        return normalizeFires(
            filterMockFiresByBbox(
                MOCK_FIRES,
                bbox
            )
        );
    }

    const query =
        buildQuery({
            bbox: bboxToString(bbox),
            date_from: dateFrom,
            date_to: dateTo
        });

    const data =
        await apiRequest(
            `${APP_CONFIG.API.FIRES}${query}`
        );

    return normalizeFires(data);
}

async function fetchFiresByPolygon(
    polygon,
    dateFrom,
    dateTo
) {

    if (isMockMode()) {

        return normalizeFires(
            filterMockFiresByPolygon(
                MOCK_FIRES,
                polygon
            )
        );
    }

    const body = {
        polygon: polygon,
        date_from: dateFrom,
        date_to: dateTo
    };

    const data =
        await apiRequest(
            APP_CONFIG.API.FIRES_QUERY,
            {
                method: "POST",
                body: body
            }
        );

    return normalizeFires(data);
}

async function fetchBurnedAreasByBbox(
    bbox,
    dateFrom,
    dateTo,
    severity
) {

    if (isMockMode()) {

        let data =
            filterMockBurnedByBbox(
                MOCK_BURNED_AREAS,
                bbox
            );

        data =
            filterMockSeverity(
                data,
                severity
            );

        return normalizeBurnedAreas(data);
    }

    const query =
        buildQuery({
            bbox: bboxToString(bbox),
            date_from: dateFrom,
            date_to: dateTo,
            severity: severity
        });

    const data =
        await apiRequest(
            `${APP_CONFIG.API.BURNED_AREAS}${query}`
        );

    return normalizeBurnedAreas(data);
}

async function fetchBurnedAreasByPolygon(
    polygon,
    dateFrom,
    dateTo,
    severity
) {

    if (isMockMode()) {

        let data =
            filterMockBurnedByPolygon(
                MOCK_BURNED_AREAS,
                polygon
            );

        data =
            filterMockSeverity(
                data,
                severity
            );

        return normalizeBurnedAreas(data);
    }

    const body = {
        polygon: polygon,
        date_from: dateFrom,
        date_to: dateTo,
        severity: severity
    };

    const data =
        await apiRequest(
            APP_CONFIG.API.BURNED_AREAS_QUERY,
            {
                method: "POST",
                body: body
            }
        );

    return normalizeBurnedAreas(data);
}

async function fetchAnalyticsByBbox(
    bbox,
    dateFrom,
    dateTo
) {

    if (isMockMode()) {

        const fires =
            filterMockFiresByBbox(
                MOCK_FIRES,
                bbox
            );

        const burnedAreas =
            filterMockBurnedByBbox(
                MOCK_BURNED_AREAS,
                bbox
            );

        return normalizeAnalytics(
            calculateMockAnalytics(
                fires,
                burnedAreas
            )
        );
    }

    const query =
        buildQuery({
            bbox: bboxToString(bbox),
            date_from: dateFrom,
            date_to: dateTo
        });

    const data =
        await apiRequest(
            `${APP_CONFIG.API.ANALYTICS}${query}`
        );

    return normalizeAnalytics(data);
}

async function fetchAnalyticsByPolygon(
    polygon,
    dateFrom,
    dateTo
) {

    if (isMockMode()) {

        const fires =
            filterMockFiresByPolygon(
                MOCK_FIRES,
                polygon
            );

        const burnedAreas =
            filterMockBurnedByPolygon(
                MOCK_BURNED_AREAS,
                polygon
            );

        return normalizeAnalytics(
            calculateMockAnalytics(
                fires,
                burnedAreas
            )
        );
    }

    const body = {
        polygon: polygon,
        date_from: dateFrom,
        date_to: dateTo
    };

    const data =
        await apiRequest(
            APP_CONFIG.API.ANALYTICS_QUERY,
            {
                method: "POST",
                body: body
            }
        );
    return normalizeAnalytics(data);
}

async function checkApiHealth() {

    if (isMockMode()) {

        return {
            status: "ok",
            mode: "mock"
        };
    }
    return apiRequest(
        APP_CONFIG.API.HEALTH
    );
}

async function apiLoadDataByBbox({
    bbox,
    dateFrom,
    dateTo,
    severity
}) {

    if (!bbox) {
        throw new Error(
            "Map bounding box is missing."
        );
    }

    const [
        fires,
        burnedAreas,
        analytics
    ] = await Promise.all([

        fetchFiresByBbox(
            bbox,
            dateFrom,
            dateTo
        ),

        fetchBurnedAreasByBbox(
            bbox,
            dateFrom,
            dateTo,
            severity
        ),

        fetchAnalyticsByBbox(
            bbox,
            dateFrom,
            dateTo
        )
    ]);

    const filteredAnalytics =
        recalculateAnalyticsFromLayers(
            analytics,
            fires,
            burnedAreas
        );
    return {
        fires: fires,
        burnedAreas: burnedAreas,
        analytics: filteredAnalytics
    };
}

async function apiLoadDataByPolygon({
    polygon,
    dateFrom,
    dateTo,
    severity
}) {

    if (!polygon) {
        throw new Error(
            "Polygon geometry is missing."
        );
    }

    const [
        fires,
        burnedAreas,
        analytics
    ] = await Promise.all([

        fetchFiresByPolygon(
            polygon,
            dateFrom,
            dateTo
        ),

        fetchBurnedAreasByPolygon(
            polygon,
            dateFrom,
            dateTo,
            severity
        ),

        fetchAnalyticsByPolygon(
            polygon,
            dateFrom,
            dateTo
        )
    ]);

    const filteredAnalytics =
        recalculateAnalyticsFromLayers(
            analytics,
            fires,
            burnedAreas
        );
    return {
        fires: fires,
        burnedAreas: burnedAreas,
        analytics: filteredAnalytics
    };
}

function normalizeFires(data) {

    if (
        data &&
        data.type === "FeatureCollection" &&
        Array.isArray(data.features)
    ) {
        return {
            type: "FeatureCollection",
            features: data.features
        };
    }

    if (
        data &&
        data.data !== undefined
    ) {
        return normalizeFires(
            data.data
        );
    }

    if (
        data &&
        data.type === "Feature"
    ) {
        return {
            type: "FeatureCollection",
            features: [data]
        };
    }

    if (Array.isArray(data)) {
        return {
            type: "FeatureCollection",

            features:
                data
                    .map(
                        normalizeFireFeature
                    )
                    .filter(Boolean)
        };
    }
    return {
        type: "FeatureCollection",
        features: []
    };
}

function normalizeFireFeature(item) {
    if (!item) {
        return null;
    }

    if (
        item.type === "Feature" &&
        item.geometry
    ) {

        return item;
    }

    let longitude =
        item.longitude ??
        item.lon ??
        item.lng;

    let latitude =
        item.latitude ??
        item.lat;
    if (
        (
            longitude === undefined ||
            latitude === undefined
        ) &&
        Array.isArray(item.coordinates)
    ) {

        longitude =
            item.coordinates[0];

        latitude =
            item.coordinates[1];
    }

    if (
        (
            longitude === undefined ||
            latitude === undefined
        ) &&
        item.location
    ) {

        longitude =
            item.location.longitude ??
            item.location.lon ??
            item.location.lng;

        latitude =
            item.location.latitude ??
            item.location.lat;
    }

    if (
        longitude === undefined ||
        latitude === undefined
    ) {

        console.warn(
            "Could not normalize fire:",
            item
        );

        return null;
    }

    return {
        type: "Feature",

        geometry: {
            type: "Point",

            coordinates: [
                Number(longitude),
                Number(latitude)
            ]
        },

        properties: {

            id:
                item.id ??
                item.fire_id ??
                item.uuid ??
                "unknown",

            acq_datetime:
                item.acq_datetime ??
                item.detected_at ??
                item.datetime ??
                item.timestamp ??
                null,

            satellite:
                item.satellite ??
                item.platform ??
                "Unknown",

            brightness_i4_k:
                item.brightness_i4_k ??
                item.brightness ??
                item.brightness_k ??
                null,

            confidence:
                item.confidence ??
                item.confidence_score ??
                null
        }
    };
}

function normalizeBurnedAreas(data) {
    if (
        data &&
        data.type === "FeatureCollection" &&
        Array.isArray(data.features)
    ) {
        return {
            type: "FeatureCollection",
            features: data.features
        };
    }

    if (
        data &&
        data.data !== undefined
    ) {
        return normalizeBurnedAreas(
            data.data
        );
    }

    if (
        data &&
        data.type === "Feature"
    ) {
        return {
            type: "FeatureCollection",
            features: [data]
        };
    }

    if (Array.isArray(data)) {
        return {
            type: "FeatureCollection",

            features:
                data
                    .map(
                        normalizeBurnedAreaFeature
                    )
                    .filter(Boolean)
        };
    }
    return {
        type: "FeatureCollection",
        features: []
    };
}


function normalizeBurnedAreaFeature(item) {
    if (!item) {
        return null;
    }

    if (
        item.type === "Feature" &&
        item.geometry
    ) {

        return item;
    }

    let geometry =
        item.geometry ??
        null;

    if (!geometry && item.polygon) {

        geometry =
            normalizePolygonGeometry(
                item.polygon
            );
    }

    if (
        !geometry &&
        item.coordinates
    ) {

        geometry =
            normalizePolygonGeometry(
                item.coordinates
            );
    }

    if (!geometry) {

        console.warn(
            "Could not normalize burned area:",
            item
        );

        return null;
    }

    return {
        type: "Feature",
        geometry: geometry,
        properties: {
            id:
                item.id ??
                item.area_id ??
                item.uuid ??
                "unknown",
            severity:
                Number(
                    item.severity ??
                    item.damage_level ??
                    1
                ),
            severity_label:
                item.severity_label ??
                item.damage_label ??
                getSeverityLabel(
                    Number(
                        item.severity ??
                        item.damage_level ??
                        1
                    )
                ),

            area_ha:
                item.area_ha ??
                item.area ??
                item.burned_area_ha ??
                0,

            date_pre:
                item.date_pre ??
                item.before_date ??
                null,

            date_post:
                item.date_post ??
                item.after_date ??
                null
        }
    };
}

function normalizePolygonGeometry(value) {
    if (
        value &&
        value.type === "Polygon" &&
        Array.isArray(value.coordinates)
    ) {
        return value;
    }

    if (
        Array.isArray(value) &&
        value.length > 0
    ) {
        if (
            Array.isArray(value[0]) &&
            typeof value[0][0] === "number"
        ) {
            return {
                type: "Polygon",
                coordinates: [value]
            };
        }
        if (
            Array.isArray(value[0]) &&
            Array.isArray(value[0][0])
        ) {
            return {
                type: "Polygon",
                coordinates: value
            };
        }
    }
    return null;
}

function normalizeAnalytics(data) {
    if (!data) {
        return {
            fire_points: 0,
            total_burned_ha: 0,

            by_severity_ha: {
                1: 0,
                2: 0,
                3: 0
            },

            by_severity_pct: {
                1: 0,
                2: 0,
                3: 0
            }
        };
    }

    if (
        data.data &&
        typeof data.data === "object"
    ) {

        return normalizeAnalytics(
            data.data
        );
    }
    return {
        bbox:
            data.bbox ??
            null,

        date_from:
            data.date_from ??
            null,

        date_to:
            data.date_to ??
            null,

        fire_points:
            Number(
                data.fire_points ??
                data.fire_count ??
                data.active_fires ??
                0
            ),

        total_burned_ha:
            Number(
                data.total_burned_ha ??
                data.total_area_ha ??
                data.burned_area_ha ??
                0
            ),

        by_severity_ha:
            normalizeSeverityObject(
                data.by_severity_ha ??
                data.severity_areas ??
                {}
            ),

        by_severity_pct:
            normalizeSeverityObject(
                data.by_severity_pct ??
                data.severity_percentages ??
                {}
            ),

        projection_for_area:
            data.projection_for_area ??
            "EPSG:6933"
    };
}

function normalizeSeverityObject(value) {
    return {
        1: Number(
            value[1] ??
            value["1"] ??
            0
        ),

        2: Number(
            value[2] ??
            value["2"] ??
            0
        ),

        3: Number(
            value[3] ??
            value["3"] ??
            0
        )
    };
}

function recalculateAnalyticsFromLayers(
    analytics,
    fires,
    burnedAreas
) {

    const result =
        normalizeAnalytics(
            analytics
        )

    result.fire_points =
        fires?.features?.length ?? 0;

    const bySeverity = {
        1: 0,
        2: 0,
        3: 0
    };

    let totalArea = 0;

    if (
        burnedAreas &&
        Array.isArray(
            burnedAreas.features
        )
    ) {

        burnedAreas.features.forEach(
            (feature) => {

                const properties =
                    feature.properties || {};

                const severity =
                    Number(
                        properties.severity
                    );

                const area =
                    Number(
                        properties.area_ha
                    ) || 0;
                if (
                    severity === 1 ||
                    severity === 2 ||
                    severity === 3
                ) {

                    bySeverity[severity] +=
                        area;

                    totalArea +=
                        area;
                }
            }
        );
    }

    result.total_burned_ha =
        totalArea;
    result.by_severity_ha =
        bySeverity;
    result.by_severity_pct = {
        1: 0,
        2: 0,
        3: 0
    };

    if (totalArea > 0) {
        result.by_severity_pct[1] =
            bySeverity[1] /
            totalArea *
            100;

        result.by_severity_pct[2] =
            bySeverity[2] /
            totalArea *
            100;

        result.by_severity_pct[3] =
            bySeverity[3] /
            totalArea *
            100;
    }

    return result;
}

window.apiLoadDataByPolygon =
    apiLoadDataByPolygon;