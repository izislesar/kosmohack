let map = null;

let firesLayer = null;
let burnedAreasLayer = null;
let drawnItems = null;
let drawControl = null;

let currentDrawHandler = null;
let selectedBbox = null;
let selectedPolygon = null;


function initMap() {
    const mapElement = document.getElementById("map");

    if (!mapElement) {
        console.error("Map element #map not found");
        return;
    }

    if (typeof L === "undefined") {
        console.error("Leaflet is not loaded");
        return;
    }

    map = L.map("map", {
        zoomControl: false,
        minZoom: APP_CONFIG.MAP.MIN_ZOOM,
        maxZoom: APP_CONFIG.MAP.MAX_ZOOM,
    }).setView(
        APP_CONFIG.MAP.CENTER,
        APP_CONFIG.MAP.ZOOM
    );

    if (!APP_CONFIG.CARTO_API_KEY) {
        console.error(
            "CARTO API key is missing. Add it to js/config.js"
        );

        showMapError(
            "CARTO API key is missing"
        );

        return;
    }

    const cartoUrl =
        "https://{s}.basemaps.cartocdn.com/dark_all/" +
        "{z}/{x}/{y}{r}.png" +
        "?key=" +
        encodeURIComponent(APP_CONFIG.CARTO_API_KEY);

    L.tileLayer(cartoUrl, {
        maxZoom: APP_CONFIG.MAP.MAX_ZOOM,
        minZoom: APP_CONFIG.MAP.MIN_ZOOM,

        attribution:
            '&copy; OpenStreetMap contributors &copy; CARTO',

        subdomains: "abcd"
    }).addTo(map);

    firesLayer = L.layerGroup().addTo(map);

    burnedAreasLayer = L.layerGroup().addTo(map);

    if (typeof L.FeatureGroup === "function") {
        drawnItems = new L.FeatureGroup();

        map.addLayer(drawnItems);
    }

    if (
        typeof L.Control !== "undefined" &&
        typeof L.Control.Draw !== "undefined"
    ) {
        drawControl = new L.Control.Draw({
            draw: {
                polyline: false,
                circle: false,
                circlemarker: false,
                marker: false,

                rectangle: false,
                polygon: false
            },

            edit: {
                featureGroup: drawnItems,
                edit: false,
                remove: false
            }
        });

        map.addControl(drawControl);
    }

    map.on(
        L.Draw.Event.CREATED,
        handleDrawCreated
    );

    map.on("moveend", function () {
        if (typeof handleMapMove === "function") {
            handleMapMove();
        }
    });

    console.log("Leaflet map initialized");
}

function showMapError(message) {
    const mapElement = document.getElementById("map");

    if (!mapElement) {
        return;
    }

    mapElement.innerHTML = `
        <div class="map-error">
            <div class="map-error-title">
                MAP ERROR
            </div>

            <div class="map-error-message">
                ${escapeHtml(message)}
            </div>

            <div class="map-error-hint">
                Check js/config.js and CARTO API key.
            </div>
        </div>
    `;
}

function initCustomZoom() {
    const zoomIn = document.getElementById("zoomIn");
    const zoomOut = document.getElementById("zoomOut");

    if (!zoomIn || !zoomOut) {
        return;
    }

    zoomIn.addEventListener("click", function () {
        if (!map) return;

        map.zoomIn();
    });

    zoomOut.addEventListener("click", function () {
        if (!map) return;

        map.zoomOut();
    });
}

function mapEnableRectangleDrawing() {
    if (!map) {
        console.error("MAP: map is not initialized");
        return;
    }

    if (
        typeof L.Draw === "undefined" ||
        typeof L.Draw.Rectangle === "undefined"
    ) {
        console.error(
            "MAP: Leaflet.Draw is NOT loaded"
        );
        return;
    }
    mapDisableDrawing();

    currentDrawHandler = new L.Draw.Rectangle(
        map,
        {
            shapeOptions: {
                color: "#ff6b35",
                weight: 2,
                fillOpacity: 0.08
            }
        }
    );
    currentDrawHandler.enable();
}

function mapEnablePolygonDrawing() {

    if (!map) {
        console.error("MAP: map is not initialized");
        return;
    }

    if (
        typeof L.Draw === "undefined" ||
        typeof L.Draw.Polygon === "undefined"
    ) {
        console.error(
            "MAP: Leaflet.Draw is NOT loaded"
        );
        return;
    }

    mapDisableDrawing();

    currentDrawHandler = new L.Draw.Polygon(
        map,
        {
            allowIntersection: false,

            showArea: true,

            shapeOptions: {
                color: "#ff6b35",
                weight: 2,
                fillOpacity: 0.08
            }
        }
    );
    currentDrawHandler.enable();
}

function mapDisableDrawing() {
    if (currentDrawHandler) {
        currentDrawHandler.disable();
        currentDrawHandler = null;
    }
}

function handleDrawCreated(event) {
    if (!drawnItems) {
        return;
    }

    drawnItems.clearLayers();
    const layer = event.layer;
    drawnItems.addLayer(layer);
    mapDisableDrawing();

    if (
        event.layerType === "rectangle"
    ) {
        const bounds = layer.getBounds();

        selectedBbox = [
            bounds.getWest(),
            bounds.getSouth(),
            bounds.getEast(),
            bounds.getNorth()
        ];

        selectedPolygon = null;

        if (
            typeof window.setSelectedBbox === "function"
        ) {
            window.setSelectedBbox(
                selectedBbox
            );
        }
    }

    if (
        event.layerType === "polygon"
    ) {
        const latLngs = layer.getLatLngs();

        const polygon = convertLeafletPolygonToGeoJSON(
            latLngs
        );

        selectedPolygon = polygon;
        selectedBbox = null;

        if (
            typeof window.setSelectedPolygon === "function"
        ) {
            window.setSelectedPolygon(
                selectedPolygon
            );
        }
    }

    if (
        typeof window.handleTerritoryDrawn === "function"
    ) {
        window.handleTerritoryDrawn();
    }
}

function mapClearSelectedTerritory() {
    selectedBbox = null;
    selectedPolygon = null;

    mapDisableDrawing();

    if (drawnItems) {
        drawnItems.clearLayers();
    }
}

function convertLeafletPolygonToGeoJSON(latLngs) {
    if (!Array.isArray(latLngs)) {
        return null;
    }

    let points = latLngs;

    if (Array.isArray(latLngs[0])) {
        points = latLngs[0];
    }

    const coordinates = points.map(
        function (point) {
            return [
                point.lng,
                point.lat
            ];
        }
    );

    if (coordinates.length > 0) {
        const first =
            coordinates[0];

        const last =
            coordinates[
                coordinates.length - 1
            ];

        if (
            first[0] !== last[0] ||
            first[1] !== last[1]
        ) {
            coordinates.push([
                first[0],
                first[1]
            ]);
        }
    }

    return {
        type: "Polygon",

        coordinates: [
            coordinates
        ]
    };
}

function renderFires(data) {
    if (!firesLayer) {
        return;
    }

    firesLayer.clearLayers();

    if (!data) {
        return;
    }

    const features =
        mapNormalizeFeatures(
            data
        );

    features.forEach(
        function (feature) {
            if (
                !feature ||
                !feature.geometry
            ) {
                return;
            }

            const coords =
                getPointCoordinates(
                    feature
                );

            if (!coords) {
                return;
            }

            const longitude =
                coords[0];

            const latitude =
                coords[1];

            const properties =
                feature.properties || {};

            const marker =
                L.marker(
                    [
                        latitude,
                        longitude
                    ],
                    {
                        icon:
                            createFireIcon()
                    }
                );


            marker.bindPopup(
                createFirePopup(
                    properties,
                    latitude,
                    longitude
                ),
                {
                    maxWidth: 340,
                    className:
                        "fire-popup"
                }
            );

            firesLayer.addLayer(
                marker
            );
        }
    );
}

function createFireIcon() {
    return L.divIcon({
        className:
            "fire-marker-wrapper",

        html: `
            <div class="fire-marker">
                <div class="fire-marker-core"></div>
            </div>
        `,

        iconSize: [
            20,
            20
        ],

        iconAnchor: [
            10,
            10
        ],

        popupAnchor: [
            0,
            -10
        ]
    });
}

function createFirePopup(
    properties,
    latitude,
    longitude
) {
    const id =
        properties.id ||
        properties.fire_id ||
        properties.detection_id ||
        "—";

    const satellite =
        properties.satellite ||
        properties.source ||
        properties.platform ||
        "—";

    const temperature =
        properties.temperature ??
        properties.brightness_temperature ??
        properties.temp ??
        null;

    const confidence =
        properties.confidence ??
        properties.confidence_score ??
        null;

    const date =
        properties.date ||
        properties.acquisition_date ||
        properties.detected_at ||
        "—";

    const temperatureText =
        temperature !== null
            ? `${Number(temperature).toFixed(1)} K`
            : "—";

    const confidenceText =
        confidence !== null
            ? `${(
                Number(confidence) * 100
            ).toFixed(0)}%`
            : "—";

    return `
        <div class="popup-content">

            <div class="popup-title">
                ACTIVE FIRE
            </div>

            <div class="popup-grid">

                <div class="popup-row">
                    <span class="popup-label">
                        ID
                    </span>

                    <span class="popup-value">
                        ${escapeHtml(id)}
                    </span>
                </div>

                <div class="popup-row">
                    <span class="popup-label">
                        SATELLITE
                    </span>

                    <span class="popup-value">
                        ${escapeHtml(satellite)}
                    </span>
                </div>

                <div class="popup-row">
                    <span class="popup-label">
                        TEMPERATURE
                    </span>

                    <span class="popup-value">
                        ${temperatureText}
                    </span>
                </div>

                <div class="popup-row">
                    <span class="popup-label">
                        CONFIDENCE
                    </span>

                    <span class="popup-value">
                        ${confidenceText}
                    </span>
                </div>

                <div class="popup-row">
                    <span class="popup-label">
                        DATE
                    </span>

                    <span class="popup-value">
                        ${escapeHtml(date)}
                    </span>
                </div>

                <div class="popup-row">
                    <span class="popup-label">
                        LAT
                    </span>

                    <span class="popup-value">
                        ${Number(latitude).toFixed(5)}
                    </span>
                </div>

                <div class="popup-row">
                    <span class="popup-label">
                        LON
                    </span>

                    <span class="popup-value">
                        ${Number(longitude).toFixed(5)}
                    </span>
                </div>
            </div>
        </div>
    `;
}

function renderBurnedAreas(data) {
    if (!burnedAreasLayer) {
        return;
    }

    burnedAreasLayer.clearLayers();

    if (!data) {
        return;
    }

    const features =
        mapNormalizeFeatures(
            data
        );

    features.forEach(
        function (feature) {
            if (
                !feature ||
                !feature.geometry
            ) {
                return;
            }

            const properties =
                feature.properties || {};

            const severity =
                Number(
                    properties.severity ??
                    properties.level ??
                    properties.damage_level ??
                    1
                );

            const style =
                getSeverityStyle(
                    severity
                );

            const layer =
                L.geoJSON(
                    feature,
                    {
                        style: style,

                        onEachFeature:
                            function (
                                feature,
                                layer
                            ) {
                                layer.bindPopup(
                                    createBurnedPopup(
                                        feature.properties || {}
                                    ),
                                    {
                                        maxWidth: 320,
                                        className:
                                            "burned-popup"
                                    }
                                );
                            }
                    }
                );

            burnedAreasLayer.addLayer(
                layer
            );
        }
    );
}

function getSeverityStyle(severity) {
    switch (Number(severity)) {

        case 3:
            return {
                color: "#ff3b30",
                weight: 2,
                fillColor: "#ff3b30",
                fillOpacity: 0.32
            };

        case 2:
            return {
                color: "#ff9500",
                weight: 2,
                fillColor: "#ff9500",
                fillOpacity: 0.28
            };

        case 1:
        default:
            return {
                color: "#ffd60a",
                weight: 2,
                fillColor: "#ffd60a",
                fillOpacity: 0.22
            };
    }
}

function createBurnedPopup(properties) {
    const id =
        properties.id ||
        properties.area_id ||
        properties.burned_area_id ||
        "—";

    const severity =
        Number(
            properties.severity ??
            properties.level ??
            properties.damage_level ??
            1
        );

    const area =
        properties.area_ha ??
        properties.area ??
        properties.burned_area_ha ??
        null;

    const datePre =
        properties.date_pre ||
        properties.pre_date ||
        "—";

    const datePost =
        properties.date_post ||
        properties.post_date ||
        "—";

    return `
        <div class="popup-content">

            <div class="popup-title">
                BURNED AREA
            </div>

            <div class="popup-grid">

                <div class="popup-row">
                    <span class="popup-label">
                        ID
                    </span>

                    <span class="popup-value">
                        ${escapeHtml(id)}
                    </span>
                </div>

                <div class="popup-row">
                    <span class="popup-label">
                        SEVERITY
                    </span>

                    <span class="popup-value">
                        <span class="
                            severity-badge
                            severity-${severity}
                        ">
                            ${escapeHtml(
                                getSeverityText(
                                    severity
                                )
                            )}
                        </span>
                    </span>
                </div>

                <div class="popup-row">
                    <span class="popup-label">
                        AREA
                    </span>

                    <span class="popup-value">
                        ${
                            area !== null
                                ? `${Number(area).toFixed(1)} ha`
                                : "—"
                        }
                    </span>
                </div>

                <div class="popup-row">
                    <span class="popup-label">
                        PRE
                    </span>

                    <span class="popup-value">
                        ${escapeHtml(datePre)}
                    </span>
                </div>

                <div class="popup-row">
                    <span class="popup-label">
                        POST
                    </span>

                    <span class="popup-value">
                        ${escapeHtml(datePost)}
                    </span>
                </div>
            </div>
        </div>
    `;
}

function renderAnalytics(analytics) {
    if (!analytics) {
        return;
    }

    const firePoints =
        analytics.fire_points ??
        analytics.fires ??
        analytics.total_fires ??
        0;

    const totalArea =
        analytics.total_area_ha ??
        analytics.total_burned_area_ha ??
        analytics.total_area ??
        0;

    const weak =
        analytics.weak_ha ??
        analytics.level_1_ha ??
        analytics.severity_1_ha ??
        analytics.by_severity?.["1"] ??
        0;

    const moderate =
        analytics.moderate_ha ??
        analytics.level_2_ha ??
        analytics.severity_2_ha ??
        analytics.by_severity?.["2"] ??
        0;

    const strong =
        analytics.strong_ha ??
        analytics.level_3_ha ??
        analytics.severity_3_ha ??
        analytics.by_severity?.["3"] ??
        0;

    setText(
        "statFirePoints",
        formatNumber(firePoints, 0)
    );

    setText(
        "statTotalArea",
        `${formatNumber(totalArea, 1)} ha`
    );

    setText(
        "statWeak",
        `${formatNumber(weak, 1)} ha`
    );

    setText(
        "statModerate",
        `${formatNumber(moderate, 1)} ha`
    );

    setText(
        "statStrong",
        `${formatNumber(strong, 1)} ha`
    );
}

function setFiresVisibility(visible) {
    if (!map || !firesLayer) {
        return;
    }

    if (visible) {
        if (!map.hasLayer(firesLayer)) {
            firesLayer.addTo(map);
        }
    } else {
        if (map.hasLayer(firesLayer)) {
            map.removeLayer(firesLayer);
        }
    }
}


function setBurnedAreasVisibility(visible) {
    if (!map || !burnedAreasLayer) {
        return;
    }

    if (visible) {
        if (
            !map.hasLayer(
                burnedAreasLayer
            )
        ) {
            burnedAreasLayer.addTo(map);
        }
    } else {
        if (
            map.hasLayer(
                burnedAreasLayer
            )
        ) {
            map.removeLayer(
                burnedAreasLayer
            );
        }
    }
}

function getCurrentMapBbox() {
    if (!map) {
        return null;
    }

    const bounds =
        map.getBounds();

    return [
        bounds.getWest(),
        bounds.getSouth(),
        bounds.getEast(),
        bounds.getNorth()
    ];
}

function fitMapToData(
    fires,
    burnedAreas
) {
    if (!map) {
        return;
    }

    const bounds =
        L.latLngBounds([]);


    /* fires */

    const fireFeatures =
        mapNormalizeFeatures(
            fires
        );

    fireFeatures.forEach(
        function (feature) {
            const coords =
                getPointCoordinates(
                    feature
                );

            if (coords) {
                bounds.extend([
                    coords[1],
                    coords[0]
                ]);
            }
        }
    );

    const burnedFeatures =
        mapNormalizeFeatures(
            burnedAreas
        );

    burnedFeatures.forEach(
        function (feature) {
            try {
                const layer =
                    L.geoJSON(feature);

                const layerBounds =
                    layer.getBounds();

                if (
                    layerBounds.isValid()
                ) {
                    bounds.extend(
                        layerBounds
                    );
                }

            } catch (error) {
                console.warn(
                    "Unable to calculate burned area bounds",
                    error
                );
            }
        }
    );

    if (bounds.isValid()) {
        map.fitBounds(
            bounds,
            {
                padding: [
                    80,
                    80
                ],
                maxZoom: 13
            }
        );
    }
}

function mapNormalizeFeatures(data) {
    if (!data) {
        return [];
    }

    if (
        data.type === "FeatureCollection" &&
        Array.isArray(data.features)
    ) {
        return data.features;
    }

    if (
        data.type === "Feature"
    ) {
        return [data];
    }

    if (
        Array.isArray(data)
    ) {
        return data.map(
            normalizeSingleFeature
        );
    }

    if (
        Array.isArray(data.features)
    ) {
        return data.features;
    }

    if (
        Array.isArray(data.data)
    ) {
        return data.data.map(
            normalizeSingleFeature
        );
    }
    return [];
}

function normalizeSingleFeature(item) {
    if (!item) {
        return null;
    }

    if (
        item.type === "Feature"
    ) {
        return item;
    }

    if (
        item.geometry
    ) {
        return {
            type: "Feature",
            geometry:
                item.geometry,
            properties:
                item.properties || {}
        };
    }

    const latitude =
        item.latitude ??
        item.lat;

    const longitude =
        item.longitude ??
        item.lon ??
        item.lng;

    if (
        latitude !== undefined &&
        longitude !== undefined
    ) {
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
                ...item
            }
        };
    }
    return null;
}

function getPointCoordinates(feature) {
    if (
        !feature ||
        !feature.geometry
    ) {
        return null;
    }

    if (
        feature.geometry.type !== "Point"
    ) {
        return null;
    }

    const coordinates =
        feature.geometry.coordinates;

    if (
        !Array.isArray(coordinates) ||
        coordinates.length < 2
    ) {
        return null;
    }

    return [
        Number(coordinates[0]),
        Number(coordinates[1])
    ];
}

function getSeverityText(severity) {
    if (
        typeof getSeverityLabel === "function"
    ) {
        return getSeverityLabel(
            Number(severity)
        );
    }

    switch (Number(severity)) {
        case 3:
            return "Strong";

        case 2:
            return "Moderate";

        case 1:
        default:
            return "Weak";
    }
}

function setText(
    id,
    value
) {
    const element =
        document.getElementById(id);

    if (element) {
        element.textContent =
            value;
    }
}

function formatNumber(
    value,
    decimals = 0
) {
    const number =
        Number(value);

    if (!Number.isFinite(number)) {
        return "0";
    }

    return number.toLocaleString(
        "en-US",
        {
            minimumFractionDigits:
                decimals,

            maximumFractionDigits:
                decimals
        }
    );
}

function escapeHtml(value) {
    return String(value)
        .replace(
            /&/g,
            "&amp;"
        )
        .replace(
            /</g,
            "&lt;"
        )
        .replace(
            />/g,
            "&gt;"
        )
        .replace(
            /"/g,
            "&quot;"
        )
        .replace(
            /'/g,
            "&#039;"
        );
}

window.initMap =
    initMap;

window.initCustomZoom =
    initCustomZoom;

window.mapEnableRectangleDrawing =
    mapEnableRectangleDrawing;

window.mapEnablePolygonDrawing =
    mapEnablePolygonDrawing;

window.mapDisableDrawing =
    mapDisableDrawing;

window.mapClearSelectedTerritory =
    mapClearSelectedTerritory;

window.getCurrentMapBbox =
    getCurrentMapBbox;

window.renderFires =
    renderFires;

window.renderBurnedAreas =
    renderBurnedAreas;

window.renderAnalytics =
    renderAnalytics;

window.setFiresVisibility =
    setFiresVisibility;

window.setBurnedAreasVisibility =
    setBurnedAreasVisibility;

window.fitMapToData =
    fitMapToData;