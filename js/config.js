const APP_CONFIG = {

    API_BASE: "http://localhost:8000",

    USE_MOCK: false,

    REFRESH_SECONDS: 30,
    /*Тоже подумай кстати че с этим делать*/
    DEFAULT_DATE_FROM: "2025-07-01",
    DEFAULT_DATE_TO: "2025-07-31",

    MAP: {
        CENTER: [46.0, 40.5],
        ZOOM: 7,

        MIN_ZOOM: 3,
        MAX_ZOOM: 18
    },

    CARTO_API_KEY: "cb1_3q3z_1_bd9aea2de0cd2082d73a68db",
    TILE_PROVIDER: "carto",

    API: {
        FIRES: "/v1/fires",
        FIRES_QUERY: "/v1/fires/query",
        BURNED_AREAS: "/v1/burned-areas",
        BURNED_AREAS_QUERY: "/v1/burned-areas/query",
        ANALYTICS: "/v1/analytics",
        ANALYTICS_QUERY: "/v1/analytics/query",
        HEALTH: "/v1/health"
    },

    SEVERITY: {

        1: {
            label: "Weak",
            shortLabel: "Level 1"
        },

        2: {
            label: "Moderate",
            shortLabel: "Level 2"
        },

        3: {
            label: "Strong",
            shortLabel: "Level 3"
        }
    },

    FIRE_MARKER: {
        radius: 6,
        weight: 2
    },

    REFRESH_ON_MAP_MOVE: true,

    MAX_BBOX_DEGREES: 2
};

/*Build full API URL.*/
function apiUrl(endpoint) {
    const base = APP_CONFIG.API_BASE.replace(/\/+$/, "");
    const path = endpoint.startsWith("/")
        ? endpoint
        : `/${endpoint}`;
    return `${base}${path}`;
}

function isMockMode() {
    return APP_CONFIG.USE_MOCK === true;
}

function getSeverityLabel(severity) {
    return APP_CONFIG.SEVERITY[severity]?.label
        ?? `Level ${severity}`;
}