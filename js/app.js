const appState = {
    territoryMode: "full",
    dateFrom: "",
    dateTo: "",
    selectedBbox: null,
    selectedPolygon: null,
    selectedSeverity: [1, 2, 3],
    fires: null,
    burnedAreas: null,
    analytics: null,
    isLoading: false,
    isAutoRefresh: false,
    lastUpdate: null,
    firesVisible: true,
    burnedAreasVisible: true
};

const elements = {};

document.addEventListener(
    "DOMContentLoaded",
    function () {

        if (typeof window.initMap === "function") {
            window.initMap();
        }

        if (typeof window.initCustomZoom === "function") {
            window.initCustomZoom();
        }

        cacheElements();
        initializeFilters();
        initializeTerritoryControls();
        initializeLayerControls();
        initializeSeverityControls();
        initializeButtons();
        initializeNotification();
        initializeStatus();
        exposeMapCallbacks();
        loadCurrentData();
        startAutoRefresh();
    }
);

function cacheElements() {

    elements.dateFrom =
        document.getElementById("dateFrom");

    elements.dateTo =
        document.getElementById("dateTo");

    elements.territoryFull =
        document.getElementById("territoryFull");

    elements.territoryRectangle =
        document.getElementById(
            "territoryRectangle"
        );

    elements.territoryPolygon =
        document.getElementById(
            "territoryPolygon"
        );

    elements.firesToggle =
        document.getElementById(
            "firesToggle"
        );

    elements.burnedToggle =
        document.getElementById(
            "burnedToggle"
        );

    elements.applyFilters =
        document.getElementById(
            "applyFilters"
        );

    elements.apiStatus =
        document.getElementById(
            "apiStatus"
        );

    elements.lastUpdate =
        document.getElementById(
            "lastUpdate"
        );

    elements.systemStatusIndicator =
        document.getElementById(
            "systemStatusIndicator"
        );

    elements.fireCount =
        document.getElementById(
            "fireCount"
        );

    elements.lastScan =
        document.getElementById(
            "lastScan"
        );

    elements.dataSource =
        document.getElementById(
            "dataSource"
        );

    elements.liveBadge =
        document.getElementById(
            "liveBadge"
        );

    elements.loadingOverlay =
        document.getElementById(
            "loadingOverlay"
        );

    elements.notification =
        document.getElementById(
            "notification"
        );

    elements.notificationTitle =
        document.getElementById(
            "notificationTitle"
        );

    elements.notificationMessage =
        document.getElementById(
            "notificationMessage"
        );

    elements.closeNotification =
        document.getElementById(
            "closeNotification"
        );
}

function initializeFilters() {

    if (elements.dateFrom) {
        elements.dateFrom.dataset.value = APP_CONFIG.DEFAULT_DATE_FROM;
        elements.dateFrom.value = APP_CONFIG.DEFAULT_DATE_FROM
            .split("-").reverse().join(".");
    }

    if (elements.dateTo) {
        elements.dateTo.dataset.value = APP_CONFIG.DEFAULT_DATE_TO;
        elements.dateTo.value = APP_CONFIG.DEFAULT_DATE_TO
            .split("-").reverse().join(".");
    }

    appState.dateFrom =
        APP_CONFIG.DEFAULT_DATE_FROM;

    appState.dateTo =
        APP_CONFIG.DEFAULT_DATE_TO;
}

function initializeTerritoryControls() {

    const controls = [
        elements.territoryFull,
        elements.territoryRectangle,
        elements.territoryPolygon
    ];

    controls.forEach(
        function (control) {

            if (!control) {
                return;
            }

            control.addEventListener(
                "change",
                function () {

                    if (!control.checked) {
                        return;
                    }

                    setTerritoryMode(
                        control.value
                    );
                }
            );
        }
    );
}

function setTerritoryMode(mode) {

    appState.territoryMode = mode;

    if (mode === "full") {
        appState.selectedBbox = null;
        appState.selectedPolygon = null;
        if (typeof window.mapClearSelectedTerritory === "function") {
            window.mapClearSelectedTerritory();
        }
        return;
    }

    if (mode === "rectangle") {
        appState.selectedBbox = null;
        appState.selectedPolygon = null; 
        if (typeof window.mapEnableRectangleDrawing === "function") {
            window.mapClearSelectedTerritory();
        }
        if (typeof window.mapEnableRectangleDrawing === "function") {
            window.mapEnableRectangleDrawing();
        }
        return;
    }

    if (mode === "polygon") {
        appState.selectedBbox = null;
        appState.selectedPolygon = null;
        if (typeof window.mapClearSelectedTerritory === "function") {
            window.mapClearSelectedTerritory();
        }
        if (typeof window.mapEnablePolygonDrawing === "function") {
            window.mapEnablePolygonDrawing();
        }
        return;
    }
}

function initializeLayerControls() {

    if (elements.firesToggle) {
        elements.firesToggle.checked =
            true;

        elements.firesToggle.addEventListener(
            "change",
            function () {

                appState.firesVisible =
                    elements.firesToggle.checked;

                if (
                    typeof window.setFiresVisibility ===
                    "function"
                ) {
                    window.setFiresVisibility(
                        appState.firesVisible
                    );
                }
            }
        );
    }

    if (elements.burnedToggle) {
        elements.burnedToggle.checked =
            true;

        elements.burnedToggle.addEventListener(
            "change",
            function () {
                appState.burnedAreasVisible =
                    elements.burnedToggle.checked;

                if (
                    typeof window.setBurnedAreasVisibility ===
                    "function"
                ) {
                    window.setBurnedAreasVisibility(
                        appState.burnedAreasVisible
                    );
                }
            }
        );
    }
}

function initializeSeverityControls() {
    const checkboxes =
        document.querySelectorAll(
            ".severity-checkbox"
        );

    checkboxes.forEach(
        function (checkbox) {

            checkbox.checked =
                true;
            checkbox.addEventListener(
                "change",
                function () {

                    updateSelectedSeverity();

                    applySeverityFilter();
                }
            );
        }
    );

    updateSelectedSeverity();
}

function updateSelectedSeverity() {
    const checkboxes =
        document.querySelectorAll(
            ".severity-checkbox"
        );

    appState.selectedSeverity =
        Array.from(checkboxes)
            .filter(
                function (checkbox) {
                    return checkbox.checked;
                }
            )
            .map(
                function (checkbox) {
                    return Number(
                        checkbox.value
                    );
                }
            );
}

function applySeverityFilter() {
    if (!appState.burnedAreas) {
        return;
    }

    let filtered =
        appState.burnedAreas;
    if (appState.selectedSeverity.length === 0) {
        filtered = {
            type: "FeatureCollection",
            features: []
        };

    } else {

        const features =
            normalizeFeatureCollection(
                appState.burnedAreas
            );
        filtered = {
            type: "FeatureCollection",
            features:
                features.filter(
                    function (feature) {
                        const properties =
                            feature.properties || {};
                        const severity =
                            Number(
                                properties.severity ??
                                properties.level ??
                                properties.damage_level ??
                                1
                            );
                        return appState.selectedSeverity
                            .includes(
                                severity
                            );
                    }
                )
        };
    }

    if (typeof window.renderBurnedAreas === "function") {
        window.renderBurnedAreas(
            filtered
        );
    }

    const recalculated =
        recalculateAnalytics(
            appState.fires,
            filtered
        );

    appState.analytics =
        recalculated;

    if (typeof window.renderAnalytics === "function") {
        window.renderAnalytics(
            recalculated
        );
    }

        updateDashboard(
            appState.fires,
            filtered,
            recalculated
        );
}

function initializeButtons() {
    if (elements.applyFilters) {
        elements.applyFilters.addEventListener(
            "click",
            function () {
                loadCurrentData();
            }
        );
    }
}

function readFiltersFromUI() {

    if (elements.dateFrom) {
        appState.dateFrom =
            elements.dateFrom.dataset.value || elements.dateFrom.value;
    }

    if (elements.dateTo) {
        appState.dateTo =
            elements.dateTo.dataset.value || elements.dateTo.value;
    }

    const territory =
        document.querySelector(
            'input[name="territory"]:checked'
        );

    if (territory) {
        appState.territoryMode =
            territory.value;
    }

    updateSelectedSeverity();
}

function validateFilters() {

    if (
        !appState.dateFrom ||
        !appState.dateTo
    ) {

        showNotification(
            "Invalid filters",
            "Select both start and end dates."
        );

        return false;
    }

    if (
        appState.dateFrom >
        appState.dateTo
    ) {

        showNotification(
            "Invalid date range",
            "Start date must be before end date."
        );

        return false;
    }

    if (
        appState.territoryMode ===
        "rectangle" &&
        !appState.selectedBbox
    ) {

        showNotification(
            "Territory required",
            "Draw a rectangle on the map first."
        );

        return false;
    }

    if (
        appState.territoryMode ===
        "polygon" &&
        !appState.selectedPolygon
    ) {

        showNotification(
            "Territory required",
            "Draw a polygon on the map first."
        );

        return false;
    }

    return true;
}

async function loadCurrentData(isAutoRefresh = false) {
    if (appState.isLoading) {
        return;
    }

    readFiltersFromUI();


    if (!validateFilters()) {
        return;
    }

    appState.isLoading =
        true;
    appState.isAutoRefresh =
        isAutoRefresh;
    setLoading(true);

    setSystemStatus(
        "loading"
    );

    try {
        let result = null;

        if (
            appState.territoryMode ===
            "full"
        ) {

            const bbox =
                typeof window.getCurrentMapBbox ===
                "function"
                    ? window.getCurrentMapBbox()
                    : null;

            if (!bbox) {
                throw new Error(
                    "Unable to determine map bounds."
                );
            }

            result =
                await requestDataByBbox(
                    bbox
                );
        }

        else if (
            appState.territoryMode ===
            "rectangle"
        ) {

            result =
                await requestDataByBbox(
                    appState.selectedBbox
                );
        }

        else if (
            appState.territoryMode ===
            "polygon"
        ) {

            result =
                await requestDataByPolygon(
                    appState.selectedPolygon
                );
        }

        if (!result) {
            throw new Error(
                "No data returned."
            );
        }

        /*DEBUG*/

        console.log("=== DATA LOADED ===");
        console.log("Territory mode:", appState.territoryMode);
        console.log("Selected bbox:", appState.selectedBbox);
        console.log("Selected polygon:", appState.selectedPolygon);
        console.log("Fires count:", result.fires?.features?.length);
        console.log("Burned areas count:", result.burnedAreas?.features?.length);
        console.log("Analytics:", result.analytics);

        appState.fires =
            result.fires || {
                type: "FeatureCollection",
                features: []
            };

        appState.burnedAreas =
            result.burnedAreas || {
                type: "FeatureCollection",
                features: []
            };

        appState.analytics =
            result.analytics ||
            recalculateAnalytics(
                appState.fires,
                appState.burnedAreas
            );

        const filteredBurned =
            filterBurnedBySeverity(
                appState.burnedAreas
            );

        appState.analytics =
            recalculateAnalytics(
                appState.fires,
                filteredBurned
            );

        if (
            typeof window.renderFires ===
            "function"
        ) {

            window.renderFires(
                appState.fires
            );
        }

        if (
            typeof window.renderBurnedAreas ===
            "function"
        ) {

            window.renderBurnedAreas(
                filteredBurned
            );
        }

        if (
            typeof window.renderAnalytics ===
            "function"
        ) {

            window.renderAnalytics(
                appState.analytics
            );
        }

        updateDashboard(
            appState.fires,
            filteredBurned,
            appState.analytics
        );

        appState.lastUpdate =
            new Date();
        setSystemStatus(
            "online"
        );

        updateLastUpdate();
        updateDataSource();

    } catch (error) {
        console.error(
            "Data loading error:",
            error
        );

        setSystemStatus(
            "offline"
        );

        showNotification(
            "Data loading failed",
            error.message ||
            "Unable to load monitoring data."
        );

    } finally {
        appState.isLoading =
            false;
        appState.isAutoRefresh =
            false;
        setLoading(false);
    }
}

async function requestDataByBbox(bbox) {
    return await window.apiLoadDataByBbox({
        bbox: bbox,
        dateFrom: appState.dateFrom,
        dateTo: appState.dateTo,
        severity: appState.selectedSeverity
    });
}

async function requestDataByPolygon(polygon) {
    return await window.apiLoadDataByPolygon({
        polygon: polygon,
        dateFrom: appState.dateFrom,
        dateTo: appState.dateTo,
        severity: appState.selectedSeverity
    });
}

function filterBurnedBySeverity(data) {
    const features =
        normalizeFeatureCollection(
            data
        );

    return {
        type: "FeatureCollection",

        features:
            features.filter(
                function (feature) {

                    const properties =
                        feature.properties || {};

                    const severity =
                        Number(
                            properties.severity ??
                            properties.level ??
                            properties.damage_level ??
                            1
                        );

                    return appState.selectedSeverity
                        .includes(
                            severity
                        );
                }
            )
    };
}

function recalculateAnalytics(
    fires,
    burnedAreas
) {

    const fireFeatures =
        normalizeFeatureCollection(
            fires
        );

    const burnedFeatures =
        normalizeFeatureCollection(
            burnedAreas
        );

    let weak = 0;
    let moderate = 0;
    let strong = 0;

    burnedFeatures.forEach(
        function (feature) {

            const properties =
                feature.properties || {}
            const severity =
                Number(
                    properties.severity ??
                    properties.level ??
                    properties.damage_level ??
                    1
                );

            const area =
                Number(
                    properties.area_ha ??
                    properties.area ??
                    properties.burned_area_ha ??
                    0
                );

            if (severity === 1) {
                weak += area;
            }

            else if (severity === 2) {
                moderate += area;
            }

            else if (severity === 3) {
                strong += area;
            }
        }
    );

    const totalArea =
        weak +
        moderate +
        strong;
    const weakPercent =
        totalArea > 0
            ? weak / totalArea * 100
            : 0;
    const moderatePercent =
        totalArea > 0
            ? moderate / totalArea * 100
            : 0;
    const strongPercent =
        totalArea > 0
            ? strong / totalArea * 100
            : 0;
    return {

        fire_points:
            fireFeatures.length,

        total_area_ha:
            totalArea,

        weak_ha:
            weak,

        moderate_ha:
            moderate,

        strong_ha:
            strong,

        by_severity: {

            "1": weak,

            "2": moderate,

            "3": strong
        },

        percentages: {

            "1":
                weakPercent,

            "2":
                moderatePercent,

            "3":
                strongPercent
        }
    };
}

function updateDashboard(
    fires,
    burnedAreas,
    analytics
) {

    const fireFeatures =
        normalizeFeatureCollection(
            fires
        );


    if (elements.fireCount) {

        elements.fireCount.textContent =
            fireFeatures.length;
    }

    if (elements.lastScan) {

        elements.lastScan.textContent =
            formatTime(
                new Date()
            );
    }

    if (typeof window.renderAnalytics ===
        "function") {

        window.renderAnalytics(
            analytics
        );
    }
}

function updateDataSource() {

    if (!elements.dataSource) {
        return;
    }

    if (
        typeof isMockMode ===
        "function" &&
        isMockMode()
    ) {

        elements.dataSource.textContent =
            "DEMO DATA";

        return;
    }

    elements.dataSource.textContent =
        "SATELLITE API";
}

function initializeStatus() {

    setSystemStatus(
        "loading"
    );
}

function setSystemStatus(
    status
) {

    if (
        elements.apiStatus
    ) {

        if (status === "online") {

            elements.apiStatus.textContent =
                "ONLINE";
        }

        else if (
            status === "offline"
        ) {

            elements.apiStatus.textContent =
                "OFFLINE";
        }

        else {

            elements.apiStatus.textContent =
                "CONNECTING";
        }
    }

    if (
        elements.systemStatusIndicator
    ) {

        elements.systemStatusIndicator.classList.remove(
            "online",
            "offline",
            "loading"
        );

        elements.systemStatusIndicator.classList.add(
            status
        );
    }

    if (
        elements.liveBadge
    ) {

        if (status === "online") {

            elements.liveBadge.textContent =
                "LIVE";

            elements.liveBadge.classList.add(
                "active"
            );

        } else {

            elements.liveBadge.textContent =
                "OFFLINE";

            elements.liveBadge.classList.remove(
                "active"
            );
        }
    }
}

function updateLastUpdate() {
    if (!elements.lastUpdate) {
        return;
    }

    if (!appState.lastUpdate) {

        elements.lastUpdate.textContent =
            "—";

        return;
    }

    elements.lastUpdate.textContent =
        formatTime(
            appState.lastUpdate
        );
}

function formatTime(date) {
    if (!(date instanceof Date)) {
        return "—";
    }

    return date.toLocaleTimeString(
        "en-US",
        {
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit"
        }
    );
}

function setLoading(loading) {
    if (
        !elements.loadingOverlay
    ) {
        return;
    }

    elements.loadingOverlay.classList.toggle(
        "visible",
        loading
    );
}

function initializeNotification() {
    if (
        elements.closeNotification
    ) {

        elements.closeNotification.addEventListener(
            "click",
            function () {

                hideNotification();
            }
        );
    }
}

function showNotification(
    title,
    message
) {

    if (
        !elements.notification
    ) {
        return;
    }

    if (
        elements.notificationTitle
    ) {
        elements.notificationTitle.textContent =
            title;
    }

    if (
        elements.notificationMessage
    ) {
        elements.notificationMessage.textContent =
            message;
    }

    elements.notification.classList.add(
        "visible"
    );

    setTimeout(
        function () {
            hideNotification();
        },
        5000
    );
}

function hideNotification() {
    if (
        elements.notification
    ) {

        elements.notification.classList.remove(
            "visible"
        );
    }
}

function startAutoRefresh() {
    const interval =
        Math.max(
            10,
            Number(
                APP_CONFIG.REFRESH_SECONDS
            ) || 30
        );

    setInterval(
        async function () {

            if (
                appState.isLoading
            ) {
                return;
            }

            await loadCurrentData(
                true
            );

        },
        interval * 1000
    );
}

function exposeMapCallbacks() {
    window.setSelectedBbox =
        function (bbox) {

            appState.selectedBbox =
                bbox;

            appState.selectedPolygon =
                null;
        };

    window.setSelectedPolygon =
        function (polygon) {

            appState.selectedPolygon =
                polygon;

            appState.selectedBbox =
                null;
        };

    window.handleTerritoryDrawn =
        function () {

            if (
                appState.territoryMode ===
                "rectangle"
            ) {

                showNotification(
                    "Territory selected",
                    "Rectangle selected. Press APPLY FILTERS to load data."
                );
            }

            else if (
                appState.territoryMode ===
                "polygon"
            ) {

                showNotification(
                    "Territory selected",
                    "Polygon selected. Press APPLY FILTERS to load data."
                );
            }
        };

    window.handleMapMove =
        function () {
        };
}

function normalizeFeatureCollection(data) {
    if (!data) {
        return [];
    }

    if (
        data.type ===
        "FeatureCollection" &&
        Array.isArray(
            data.features
        )
    ) {

        return data.features;
    }

    if (
        data.type ===
        "Feature"
    ) {

        return [
            data
        ];
    }

    if (
        Array.isArray(data)
    ) {

        return data
            .map(
                function (item) {

                    if (
                        item &&
                        item.type ===
                        "Feature"
                    ) {
                        return item;
                    }

                    if (
                        item &&
                        item.geometry
                    ) {

                        return {
                            type:
                                "Feature",

                            geometry:
                                item.geometry,

                            properties:
                                item.properties ||
                                {}
                        };
                    }

                    const latitude =
                        item?.latitude ??
                        item?.lat;

                    const longitude =
                        item?.longitude ??
                        item?.lon ??
                        item?.lng;

                    if (
                        latitude !==
                        undefined &&
                        longitude !==
                        undefined
                    ) {

                        return {

                            type:
                                "Feature",

                            geometry: {

                                type:
                                    "Point",

                                coordinates: [

                                    Number(
                                        longitude
                                    ),

                                    Number(
                                        latitude
                                    )
                                ]
                            },

                            properties:
                                {
                                    ...item
                                }
                        };
                    }

                    return null;
                }
            )
            .filter(
                Boolean
            );
    }

    if (
        Array.isArray(
            data.features
        )
    ) {

        return data.features;
    }

    if (
        Array.isArray(
            data.data
        )
    ) {

        return normalizeFeatureCollection(
            data.data
        );
    }

    return [];
}

async function checkApiHealth() {
    if (
        typeof isMockMode === "function" &&
        isMockMode()
    ) {
        setSystemStatus("online");
        return;
    }

    if (
        typeof apiRequest !==
        "function"
    ) {
        return;
    }

    try {

        await apiRequest(
            APP_CONFIG.API.HEALTH
        );

        setSystemStatus(
            "online"
        );

    } catch (error) {

        if (
            typeof isMockMode ===
            "function" &&
            isMockMode()
        ) {

            setSystemStatus(
                "online"
            );

        } else {

            setSystemStatus(
                "offline"
            );
        }
    }
}

window.loadCurrentData =
    loadCurrentData;

window.setTerritoryMode =
    setTerritoryMode;

window.checkApiHealth =
    checkApiHealth;