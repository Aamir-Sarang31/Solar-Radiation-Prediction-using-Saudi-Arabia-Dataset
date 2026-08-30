/**
 * HelioForecast - Modern Solar Radiation Prediction & Analytics Frontend
 * Handles Leaflet.js Mapping, Plotly Visualizations, 1-Click City Presets,
 * Real-time Predictions, and 11-Model Benchmark Leaderboard.
 */

// 1-Click City Meteorological Presets (Extracted from 1,649 KFUPM verified records)
const CITY_PRESETS = {
    "Riyadh": {
        "DNI (Wh/m2)": 5362.90,
        "DHI (Wh/m2)": 2272.65,
        "Standard Deviation DNI (Wh/m2)": 1716.20,
        "Standard Deviation DHI (Wh/m2)": 557.19,
        "Standard Deviation GHI (Wh/m2)": 708.68,
        "Air Temperature (C°)": 27.78,
        "Relative Humidity (%)": 26.30,
        "Barometric Pressure (mB (hPa equiv))": 937.66,
        "Wind Speed at 3m (m/s)": 1.91,
        "Wind Speed at 3m (std dev) (m/s)": 1.11,
        "Wind Direction at 3m (°N)": 173.31,
        "Peak Wind Speed at 3m (m/s)": 12.18,
        "DNI Uncertainty (Wh/m2)": 939.44,
        "DHI Uncertainty (Wh/m2)": 402.19,
        "GHI Uncertainty (Wh/m2)": 853.25,
        "Air Temperature Uncertainty (C°)": 0.50,
        "Relative Humidity Uncertainty (%)": 3.00,
        "Barometric Pressure Uncertainty (mB (hPa equiv))": 4.69,
        "Wind Speed at 3m Uncertainty (m/s)": 0.00,
        "Peak Wind Speed at 3m Uncertainty (m/s)": 0.07,
        "Wind Direction at 3m Uncertainty (°N)": 3.87
    },
    "Jeddah": {
        "DNI (Wh/m2)": 4636.08,
        "DHI (Wh/m2)": 2241.48,
        "Standard Deviation DNI (Wh/m2)": 1500.80,
        "Standard Deviation DHI (Wh/m2)": 522.10,
        "Standard Deviation GHI (Wh/m2)": 571.75,
        "Air Temperature (C°)": 29.68,
        "Relative Humidity (%)": 59.04,
        "Barometric Pressure (mB (hPa equiv))": 1002.62,
        "Wind Speed at 3m (m/s)": 3.41,
        "Wind Speed at 3m (std dev) (m/s)": 2.03,
        "Wind Direction at 3m (°N)": 307.43,
        "Peak Wind Speed at 3m (m/s)": 16.08,
        "DNI Uncertainty (Wh/m2)": 801.79,
        "DHI Uncertainty (Wh/m2)": 361.99,
        "GHI Uncertainty (Wh/m2)": 726.68,
        "Air Temperature Uncertainty (C°)": 0.50,
        "Relative Humidity Uncertainty (%)": 3.01,
        "Barometric Pressure Uncertainty (mB (hPa equiv))": 5.03,
        "Wind Speed at 3m Uncertainty (m/s)": 0.09,
        "Peak Wind Speed at 3m Uncertainty (m/s)": 0.10,
        "Wind Direction at 3m Uncertainty (°N)": 3.78
    },
    "Makkah": {
        "DNI (Wh/m2)": 4928.67,
        "DHI (Wh/m2)": 2355.53,
        "Standard Deviation DNI (Wh/m2)": 1671.23,
        "Standard Deviation DHI (Wh/m2)": 594.65,
        "Standard Deviation GHI (Wh/m2)": 661.70,
        "Air Temperature (C°)": 30.64,
        "Relative Humidity (%)": 34.32,
        "Barometric Pressure (mB (hPa equiv))": 975.37,
        "Wind Speed at 3m (m/s)": 2.15,
        "Wind Speed at 3m (std dev) (m/s)": 1.53,
        "Wind Direction at 3m (°N)": 234.17,
        "Peak Wind Speed at 3m (m/s)": 16.57,
        "DNI Uncertainty (Wh/m2)": 834.20,
        "DHI Uncertainty (Wh/m2)": 364.81,
        "GHI Uncertainty (Wh/m2)": 768.18,
        "Air Temperature Uncertainty (C°)": 0.50,
        "Relative Humidity Uncertainty (%)": 3.01,
        "Barometric Pressure Uncertainty (mB (hPa equiv))": 4.91,
        "Wind Speed at 3m Uncertainty (m/s)": 0.05,
        "Peak Wind Speed at 3m Uncertainty (m/s)": 0.10,
        "Wind Direction at 3m Uncertainty (°N)": 3.96
    },
    "Dhahran": {
        "DNI (Wh/m2)": 4772.51,
        "DHI (Wh/m2)": 2031.85,
        "Standard Deviation DNI (Wh/m2)": 1673.23,
        "Standard Deviation DHI (Wh/m2)": 531.20,
        "Standard Deviation GHI (Wh/m2)": 695.07,
        "Air Temperature (C°)": 27.26,
        "Relative Humidity (%)": 52.84,
        "Barometric Pressure (mB (hPa equiv))": 1003.84,
        "Wind Speed at 3m (m/s)": 1.33,
        "Wind Speed at 3m (std dev) (m/s)": 0.77,
        "Wind Direction at 3m (°N)": 184.38,
        "Peak Wind Speed at 3m (m/s)": 7.91,
        "DNI Uncertainty (Wh/m2)": 774.41,
        "DHI Uncertainty (Wh/m2)": 308.95,
        "GHI Uncertainty (Wh/m2)": 679.45,
        "Air Temperature Uncertainty (C°)": 0.50,
        "Relative Humidity Uncertainty (%)": 3.02,
        "Barometric Pressure Uncertainty (mB (hPa equiv))": 5.06,
        "Wind Speed at 3m Uncertainty (m/s)": 0.00,
        "Peak Wind Speed at 3m Uncertainty (m/s)": 0.05,
        "Wind Direction at 3m Uncertainty (°N)": 2.92
    },
    "Tabuk": {
        "DNI (Wh/m2)": 6970.92,
        "DHI (Wh/m2)": 1613.19,
        "Standard Deviation DNI (Wh/m2)": 1805.66,
        "Standard Deviation DHI (Wh/m2)": 593.60,
        "Standard Deviation GHI (Wh/m2)": 565.76,
        "Air Temperature (C°)": 22.61,
        "Relative Humidity (%)": 27.49,
        "Barometric Pressure (mB (hPa equiv))": 926.14,
        "Wind Speed at 3m (m/s)": 2.33,
        "Wind Speed at 3m (std dev) (m/s)": 1.32,
        "Wind Direction at 3m (°N)": 260.97,
        "Peak Wind Speed at 3m (m/s)": 14.47,
        "DNI Uncertainty (Wh/m2)": 1327.04,
        "DHI Uncertainty (Wh/m2)": 385.72,
        "GHI Uncertainty (Wh/m2)": 981.65,
        "Air Temperature Uncertainty (C°)": 0.50,
        "Relative Humidity Uncertainty (%)": 3.01,
        "Barometric Pressure Uncertainty (mB (hPa equiv))": 4.65,
        "Wind Speed at 3m Uncertainty (m/s)": 0.05,
        "Peak Wind Speed at 3m Uncertainty (m/s)": 0.09,
        "Wind Direction at 3m Uncertainty (°N)": 3.84
    },
    "Al Ahsa": {
        "DNI (Wh/m2)": 4935.08,
        "DHI (Wh/m2)": 2221.19,
        "Standard Deviation DNI (Wh/m2)": 1676.87,
        "Standard Deviation DHI (Wh/m2)": 552.57,
        "Standard Deviation GHI (Wh/m2)": 701.51,
        "Air Temperature (C°)": 27.72,
        "Relative Humidity (%)": 29.85,
        "Barometric Pressure (mB (hPa equiv))": 993.69,
        "Wind Speed at 3m (m/s)": 1.28,
        "Wind Speed at 3m (std dev) (m/s)": 0.81,
        "Wind Direction at 3m (°N)": 182.00,
        "Peak Wind Speed at 3m (m/s)": 9.76,
        "DNI Uncertainty (Wh/m2)": 848.51,
        "DHI Uncertainty (Wh/m2)": 389.50,
        "GHI Uncertainty (Wh/m2)": 774.97,
        "Air Temperature Uncertainty (C°)": 0.51,
        "Relative Humidity Uncertainty (%)": 3.02,
        "Barometric Pressure Uncertainty (mB (hPa equiv))": 5.01,
        "Wind Speed at 3m Uncertainty (m/s)": 0.00,
        "Peak Wind Speed at 3m Uncertainty (m/s)": 0.04,
        "Wind Direction at 3m Uncertainty (°N)": 3.05
    },
    "Madinah": {
        "DNI (Wh/m2)": 5797.52,
        "DHI (Wh/m2)": 2040.83,
        "Standard Deviation DNI (Wh/m2)": 1772.36,
        "Standard Deviation DHI (Wh/m2)": 582.47,
        "Standard Deviation GHI (Wh/m2)": 665.88,
        "Air Temperature (C°)": 28.15,
        "Relative Humidity (%)": 21.62,
        "Barometric Pressure (mB (hPa equiv))": 939.11,
        "Wind Speed at 3m (m/s)": 2.47,
        "Wind Speed at 3m (std dev) (m/s)": 1.31,
        "Wind Direction at 3m (°N)": 239.75,
        "Peak Wind Speed at 3m (m/s)": 15.04,
        "DNI Uncertainty (Wh/m2)": 814.89,
        "DHI Uncertainty (Wh/m2)": 286.93,
        "GHI Uncertainty (Wh/m2)": 693.25,
        "Air Temperature Uncertainty (C°)": 0.50,
        "Relative Humidity Uncertainty (%)": 3.00,
        "Barometric Pressure Uncertainty (mB (hPa equiv))": 4.71,
        "Wind Speed at 3m Uncertainty (m/s)": 0.04,
        "Peak Wind Speed at 3m Uncertainty (m/s)": 0.10,
        "Wind Direction at 3m Uncertainty (°N)": 3.97
    }
};

// Controlled 11-Model Benchmark Data
const BENCHMARK_MODELS = [
    { rank: "🥇 1", name: "FT-Transformer (Ours)", category: "Deep Learning", mae: "94.38 ± 6.27", rmse: 126.21, rmse_str: "126.21 ± 13.17", r2: "0.9896 ± 0.0026", time: "1.94s", champion: true, status: "Production Champion" },
    { rank: "🥈 2", name: "Artificial Neural Net (ANN)", category: "Neural Baseline", mae: "129.13 ± 9.53", rmse: 172.50, rmse_str: "172.50 ± 14.25", r2: "0.9807 ± 0.0037", time: "0.36s", champion: false, status: "Evaluated" },
    { rank: "🥉 3", name: "Histogram Gradient Boosting (HGB)", category: "Ensemble", mae: "129.37 ± 12.36", rmse: 178.39, rmse_str: "178.39 ± 22.46", r2: "0.9792 ± 0.0055", time: "0.16s", champion: false, status: "Evaluated" },
    { rank: "4", name: "Support Vector Regression (SVR)", category: "Classical", mae: "114.11 ± 9.95", rmse: 188.36, rmse_str: "188.36 ± 35.15", r2: "0.9765 ± 0.0080", time: "0.04s", champion: false, status: "Evaluated" },
    { rank: "5", name: "Linear Regression (LR)", category: "Classical", mae: "134.28 ± 9.64", rmse: 194.14, rmse_str: "194.14 ± 42.34", r2: "0.9746 ± 0.0122", time: "0.00s", champion: false, status: "Baseline" },
    { rank: "6", name: "Extreme Gradient Boosting (XGBoost)", category: "Ensemble", mae: "155.65 ± 11.10", rmse: 215.21, rmse_str: "215.21 ± 20.70", r2: "0.9697 ± 0.0067", time: "0.10s", champion: false, status: "Evaluated" },
    { rank: "7", name: "Random Forest (RF)", category: "Ensemble", mae: "163.60 ± 17.87", rmse: 230.28, rmse_str: "230.28 ± 26.87", r2: "0.9653 ± 0.0090", time: "0.22s", champion: false, status: "Evaluated" },
    { rank: "8", name: "Solar 1D CNN (Ours)", category: "Deep Learning", mae: "241.06 ± 15.84", rmse: 318.02, rmse_str: "318.02 ± 24.57", r2: "0.9340 ± 0.0141", time: "1.83s", champion: false, status: "Evaluated" },
    { rank: "9", name: "Solar LSTM (Ours)", category: "Deep Learning", mae: "273.68 ± 40.51", rmse: 362.18, rmse_str: "362.18 ± 55.91", r2: "0.9117 ± 0.0335", time: "1.62s", champion: false, status: "Evaluated" },
    { rank: "10", name: "Decision Tree (DT)", category: "Classical", mae: "280.33 ± 23.14", rmse: 410.28, rmse_str: "410.28 ± 51.76", r2: "0.8895 ± 0.0312", time: "0.01s", champion: false, status: "Baseline" },
    { rank: "11", name: "K-Nearest Neighbors (KNN)", category: "Classical", mae: "337.79 ± 22.74", rmse: 447.20, rmse_str: "447.20 ± 27.45", r2: "0.8699 ± 0.0236", time: "0.00s", champion: false, status: "Baseline" }
];

let map, baseLayerDark, baseLayerLight, markers = {}, gradientCircles = [], isGradientView = false;
let currentTheme = localStorage.getItem('helio_theme') || 'dark';

document.addEventListener('DOMContentLoaded', () => {
    // 1. Initialize Theme
    applyTheme(currentTheme);
    document.getElementById('themeToggle').addEventListener('click', toggleTheme);

    // 2. Initialize Leaflet Map
    initMap();

    // 3. Initialize Select2 Components
    initSelect2();

    // 4. Load Stations and Populate Selectors
    loadInitialData();

    // 5. Populate Default City Preset (Riyadh)
    applyPreset('Riyadh');

    // 6. Populate Leaderboard Table & Chart
    renderLeaderboard('all');

    // 7. Prediction Form Submit Handler
    document.getElementById('predictionForm').addEventListener('submit', handlePredictionSubmit);

    // 8. Station Comparison Run Handler
    document.getElementById('runComparison').addEventListener('click', handleStationComparison);

    // 9. Monthly Resource Event Listeners
    document.getElementById('stationSelect').addEventListener('change', (e) => loadStationDetails(e.target.value));
    $('#parameterSelect').on('change', () => loadStationDetails(document.getElementById('stationSelect').value));
    document.getElementById('chartType').addEventListener('change', () => loadStationDetails(document.getElementById('stationSelect').value));
    document.getElementById('yearSelect').addEventListener('change', () => loadStationDetails(document.getElementById('stationSelect').value));

    // 10. Gradient Map Toggle
    document.getElementById('toggleHeatmap').addEventListener('click', toggleGradientView);
});

/* Theme Handling */
function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    const icon = document.getElementById('themeIcon');
    if (theme === 'dark') {
        icon.className = 'fa-solid fa-sun';
    } else {
        icon.className = 'fa-solid fa-moon';
    }
    localStorage.setItem('helio_theme', theme);
    currentTheme = theme;

    // Switch map base tiles if map is initialized
    if (map) {
        if (theme === 'dark') {
            if (baseLayerLight && map.hasLayer(baseLayerLight)) map.removeLayer(baseLayerLight);
            baseLayerDark.addTo(map);
        } else {
            if (baseLayerDark && map.hasLayer(baseLayerDark)) map.removeLayer(baseLayerDark);
            baseLayerLight.addTo(map);
        }
    }
}

function toggleTheme() {
    applyTheme(currentTheme === 'dark' ? 'light' : 'dark');
}

/* Leaflet Map Setup */
function initMap() {
    map = L.map('map', {
        center: [24.2, 44.5],
        zoom: 5.5,
        zoomControl: true
    });

    baseLayerDark = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '© <a href="https://carto.com/">CARTO</a> OpenStreetMap',
        maxZoom: 19
    });

    baseLayerLight = L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
        attribution: '© <a href="https://carto.com/">CARTO</a> OpenStreetMap',
        maxZoom: 19
    });

    if (currentTheme === 'dark') {
        baseLayerDark.addTo(map);
    } else {
        baseLayerLight.addTo(map);
    }
}

function initSelect2() {
    $('#parameterSelect').select2({ placeholder: "Choose Parameters", width: '100%' });
    $('#compareStations').select2({ placeholder: "Choose 2+ Stations", width: '100%' });
    $('#compareParams').select2({ placeholder: "Choose Parameters", width: '100%' });
}

function loadInitialData() {
    // 1. Fetch Station Names
    fetch('/get-station-names')
        .then(res => res.json())
        .then(data => {
            const select = document.getElementById('stationSelect');
            const compSelect = $('#compareStations');
            select.innerHTML = '';
            compSelect.empty();

            if (data.stations) {
                document.getElementById('kpiStations').textContent = data.stations.length;
                data.stations.forEach(st => {
                    const opt = new Option(st, st);
                    select.add(new Option(st, st));
                    compSelect.append(new Option(st, st));
                });
                if (data.stations.length > 0) {
                    select.value = data.stations[0];
                    loadStationDetails(data.stations[0]);
                }
            }
        })
        .catch(err => console.error("Error fetching station names:", err));

    // 2. Fetch Map Data
    fetch('/map-data')
        .then(res => res.json())
        .then(stations => {
            if (!Array.isArray(stations)) return;
            Object.values(markers).forEach(m => map.removeLayer(m));
            markers = {};

            stations.forEach(st => {
                const ghi = st.avg_ghi || 5500;
                let markerColor = '#10b981'; // Green (high GHI)
                if (ghi < 5000) markerColor = '#f43f5e';
                else if (ghi < 6200) markerColor = '#f59e0b';

                const marker = L.circleMarker([st.latitude, st.longitude], {
                    radius: 7,
                    fillColor: markerColor,
                    color: '#ffffff',
                    weight: 1.5,
                    opacity: 0.9,
                    fillOpacity: 0.85
                }).addTo(map);

                marker.bindPopup(`
                    <div style="font-family: 'Inter', sans-serif; font-size: 0.88rem;">
                        <strong style="color:#10b981;">${st.station_name}</strong><br>
                        <span style="color:#64748b;">Latitude:</span> ${st.latitude.toFixed(2)}°N<br>
                        <span style="color:#64748b;">Longitude:</span> ${st.longitude.toFixed(2)}°E<br>
                        <span style="color:#f59e0b; font-weight:600;">Avg GHI:</span> ${ghi.toFixed(1)} Wh/m²
                    </div>
                `);

                marker.on('click', () => {
                    document.getElementById('stationSelect').value = st.station_name;
                    loadStationDetails(st.station_name);
                });

                markers[st.station_name] = marker;

                // Gradient circle
                const gradCircle = L.circle([st.latitude, st.longitude], {
                    radius: 35000,
                    fillColor: markerColor,
                    fillOpacity: 0.12,
                    stroke: false
                });
                gradientCircles.push(gradCircle);
            });
        })
        .catch(err => console.error("Error loading map data:", err));

    // 3. Populate Available Comparison Parameters
    fetch('/data-analysis')
        .then(res => res.json())
        .then(data => {
            if (data.summary_stats) {
                const compParams = $('#compareParams');
                const paramSelect = $('#parameterSelect');
                const params = new Set();
                Object.keys(data.summary_stats).forEach(k => {
                    const base = k.replace(/_(mean|min|max|std)$/, '');
                    params.add(base);
                });
                params.forEach(p => {
                    compParams.append(new Option(p, p));
                    paramSelect.append(new Option(p, p));
                });
                // Default selections
                paramSelect.val(['DNI (Wh/m2)', 'DHI (Wh/m2)', 'Air Temperature (C°)']).trigger('change');
                compParams.val(['DNI (Wh/m2)', 'Air Temperature (C°)']).trigger('change');
            }
        })
        .catch(err => console.error("Error fetching analysis parameters:", err));
}

function toggleGradientView() {
    isGradientView = !isGradientView;
    const btn = document.getElementById('toggleHeatmap');
    if (isGradientView) {
        gradientCircles.forEach(c => c.addTo(map));
        btn.classList.add('btn-emerald');
        btn.classList.remove('btn-ghost');
    } else {
        gradientCircles.forEach(c => map.removeLayer(c));
        btn.classList.add('btn-ghost');
        btn.classList.remove('btn-emerald');
    }
}

function loadStationDetails(stationName) {
    if (!stationName) return;
    const year = document.getElementById('yearSelect').value;
    const url = `/station-details?station=${encodeURIComponent(stationName)}${year ? '&year=' + year : ''}`;

    fetch(url)
        .then(res => res.json())
        .then(data => {
            if (!data.details) return;
            const detailsDiv = document.getElementById('stationDetails');
            const d = data.details;
            const m = data.mean_values || {};

            detailsDiv.innerHTML = `
                <div class="p-3 mb-2" style="background: rgba(15,23,42,0.5); border-radius: var(--radius-md); border: 1px solid var(--border-color);">
                    <div style="font-weight: 700; color: var(--accent-emerald); font-size: 1.05rem;" class="mb-1">${d.station_name}</div>
                    <div style="font-size: 0.82rem; color: var(--text-secondary);" class="mb-2">
                        <i class="fa-solid fa-location-dot me-1"></i>${d.latitude.toFixed(2)}°N, ${d.longitude.toFixed(2)}°E
                    </div>
                    <div class="row g-2 text-center" style="font-size: 0.8rem;">
                        <div class="col-6">
                            <div class="p-2" style="background: rgba(255,255,255,0.04); border-radius: var(--radius-sm);">
                                <span class="d-block text-muted">Direct (DNI)</span>
                                <strong style="color: var(--accent-solar); font-size: 0.95rem;">${(m['DNI (Wh/m2)'] || 0).toFixed(0)}</strong> Wh/m²
                            </div>
                        </div>
                        <div class="col-6">
                            <div class="p-2" style="background: rgba(255,255,255,0.04); border-radius: var(--radius-sm);">
                                <span class="d-block text-muted">Avg Temperature</span>
                                <strong style="color: var(--accent-emerald); font-size: 0.95rem;">${(m['Air Temperature (C°)'] || 0).toFixed(1)}</strong> °C
                            </div>
                        </div>
                    </div>
                </div>
            `;

            // Render Monthly Chart
            renderMonthlyChart(data.monthly_chart_data);
        })
        .catch(err => console.error("Error loading station details:", err));
}

function renderMonthlyChart(chartData) {
    if (!chartData || !chartData.months || !chartData.data) return;
    const selectedParams = $('#parameterSelect').val() || ['DNI (Wh/m2)', 'Air Temperature (C°)'];
    const chartType = document.getElementById('chartType').value;
    const isDark = currentTheme === 'dark';

    const traces = [];
    const colors = ['#10b981', '#f59e0b', '#06b6d4', '#6366f1', '#f43f5e', '#a855f7'];

    selectedParams.forEach((param, idx) => {
        if (chartData.data[param]) {
            const color = colors[idx % colors.length];
            if (chartType === 'bar') {
                traces.push({
                    x: chartData.months,
                    y: chartData.data[param],
                    name: param,
                    type: 'bar',
                    marker: { color: color }
                });
            } else {
                traces.push({
                    x: chartData.months,
                    y: chartData.data[param],
                    name: param,
                    type: 'scatter',
                    mode: 'lines+markers',
                    line: { shape: 'spline', smoothing: 1.2, width: 2.5, color: color },
                    marker: { size: 6, color: color }
                });
            }
        }
    });

    const layout = {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { family: 'Inter, sans-serif', color: isDark ? '#94a3b8' : '#475569', size: 11 },
        margin: { t: 30, r: 20, l: 50, b: 40 },
        legend: { orientation: 'h', y: 1.15, x: 0 },
        xaxis: { gridcolor: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)' },
        yaxis: { gridcolor: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)' }
    };

    Plotly.newPlot('monthlyChart', traces, layout, { responsive: true, displayModeBar: false });
}

/* Station Comparison */
function handleStationComparison() {
    const stations = $('#compareStations').val();
    const params = $('#compareParams').val();
    if (!stations || stations.length < 2 || !params || params.length === 0) {
        alert("Please select at least 2 stations and 1 parameter to compare.");
        return;
    }

    fetch('/station-comparison', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ stations: stations, params: params })
    })
    .then(res => res.json())
    .then(data => {
        if (!data.values) return;
        const isDark = currentTheme === 'dark';
        const traces = [];
        const colors = ['#10b981', '#f59e0b', '#06b6d4', '#6366f1', '#f43f5e'];

        let cIdx = 0;
        stations.forEach(st => {
            params.forEach(p => {
                if (data.values[st] && data.values[st][p]) {
                    traces.push({
                        x: data.dates[st] || [],
                        y: data.values[st][p] || [],
                        name: `${st} - ${p}`,
                        type: 'scatter',
                        mode: 'lines',
                        line: { width: 1.8, color: colors[cIdx % colors.length] }
                    });
                    cIdx++;
                }
            });
        });

        const layout = {
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: { family: 'Inter, sans-serif', color: isDark ? '#94a3b8' : '#475569', size: 11 },
            margin: { t: 30, r: 20, l: 50, b: 40 },
            legend: { orientation: 'h', y: 1.15, x: 0 },
            xaxis: { gridcolor: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)' },
            yaxis: { gridcolor: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)' }
        };

        Plotly.newPlot('comparisonChart', traces, layout, { responsive: true, displayModeBar: false });

        // Populate Summary Table
        const tbody = document.querySelector('#comparisonTable tbody');
        tbody.innerHTML = '';
        if (data.summary_stats) {
            params.forEach(p => {
                ['mean', 'min', 'max', 'std'].forEach(stat => {
                    const row = document.createElement('tr');
                    const statKey = `${p}_${stat}`;
                    let stationVals = [];
                    stations.forEach(st => {
                        const val = data.summary_stats[statKey] ? data.summary_stats[statKey][st] : null;
                        stationVals.push(`<strong>${st}:</strong> ${val !== null ? val : 'N/A'}`);
                    });
                    row.innerHTML = `
                        <td style="font-weight:600; color:var(--text-primary);">${p}</td>
                        <td><span class="badge badge-model-type">${stat.toUpperCase()}</span></td>
                        <td style="font-size:0.82rem;">${stationVals.join(' | ')}</td>
                    `;
                    tbody.appendChild(row);
                });
            });
        }
    })
    .catch(err => console.error("Error in station comparison:", err));
}

/* 1-Click City Presets */
function applyPreset(cityName) {
    const preset = CITY_PRESETS[cityName];
    if (!preset) return;

    // Populate all input fields matching preset keys
    const form = document.getElementById('predictionForm');
    const inputs = form.querySelectorAll('input');
    inputs.forEach(input => {
        const name = input.name;
        if (preset[name] !== undefined) {
            input.value = preset[name];
        }
    });

    // Auto-calculate prediction upon applying preset
    handlePredictionSubmit(new Event('submit'));
}

function resetFormToAverages() {
    fetch('/get-average-values')
        .then(res => res.json())
        .then(avg => {
            const form = document.getElementById('predictionForm');
            const inputs = form.querySelectorAll('input');
            inputs.forEach(input => {
                if (avg[input.name] !== undefined && avg[input.name] !== null) {
                    input.value = avg[input.name].toFixed(2);
                }
            });
            handlePredictionSubmit(new Event('submit'));
        })
        .catch(err => console.error("Error fetching averages:", err));
}

/* Real-time GHI Prediction */
function handlePredictionSubmit(e) {
    if (e && e.preventDefault) e.preventDefault();

    const form = document.getElementById('predictionForm');
    const formData = new FormData(form);
    const payload = {};
    formData.forEach((value, key) => {
        payload[key] = parseFloat(value) || 0.0;
    });

    const modelType = document.getElementById('predictModelType').value;
    payload['model_type'] = modelType;

    const predValEl = document.getElementById('predValue');
    const tierEl = document.getElementById('predTier');
    const yieldEl = document.getElementById('estPvYield');
    const activeModelTag = document.getElementById('activeModelTag');

    predValEl.textContent = '...';
    activeModelTag.textContent = modelType.toUpperCase();

    fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
    .then(res => res.json())
    .then(data => {
        if (data.prediction !== undefined) {
            const ghi = data.prediction;
            animateValue(predValEl, 0, ghi, 600);

            // Tier Calculation
            tierEl.className = 'tier-indicator';
            if (ghi >= 6800) {
                tierEl.textContent = 'High Solar Irradiance Zone';
                tierEl.classList.add('tier-high');
            } else if (ghi >= 4500) {
                tierEl.textContent = 'Moderate Solar Irradiance Zone';
                tierEl.classList.add('tier-mid');
            } else {
                tierEl.textContent = 'Low Solar Irradiance Zone';
                tierEl.classList.add('tier-low');
            }

            // Estimate PV Yield (Standard 1 kWp system = GHI / 1000 * 0.80 PR)
            const pvYield = ((ghi / 1000) * 0.80).toFixed(2);
            yieldEl.textContent = `${pvYield} kWh / day`;

            // Render Gauge Indicator
            renderGaugePlot(ghi);
        } else if (data.error) {
            predValEl.textContent = 'Error';
            alert(`Prediction Error: ${data.error}`);
        }
    })
    .catch(err => {
        console.error("Prediction error:", err);
        predValEl.textContent = 'Error';
    });
}

function animateValue(obj, start, end, duration) {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        obj.innerHTML = (progress * (end - start) + start).toFixed(1);
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}

function renderGaugePlot(value) {
    const isDark = currentTheme === 'dark';
    const data = [{
        type: "indicator",
        mode: "gauge+number",
        value: value,
        gauge: {
            axis: { range: [2000, 8500], tickwidth: 1, tickcolor: isDark ? "#64748b" : "#94a3b8" },
            bar: { color: "#10b981", thickness: 0.25 },
            bgcolor: "transparent",
            borderwidth: 1,
            bordercolor: isDark ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)",
            steps: [
                { range: [2000, 4500], color: "rgba(244, 63, 94, 0.2)" },
                { range: [4500, 6800], color: "rgba(245, 158, 11, 0.2)" },
                { range: [6800, 8500], color: "rgba(16, 185, 129, 0.2)" }
            ]
        }
    }];

    const layout = {
        width: 280,
        height: 180,
        margin: { t: 15, r: 25, l: 25, b: 10 },
        paper_bgcolor: 'transparent',
        font: { family: 'Outfit, sans-serif', color: isDark ? '#f8fafc' : '#0f172a' }
    };

    Plotly.newPlot('gaugePlot', data, layout, { responsive: true, displayModeBar: false });
}

/* Model Leaderboard Rendering & Filtering */
function renderLeaderboard(filterCategory = 'all') {
    const tbody = document.getElementById('leaderboardBody');
    tbody.innerHTML = '';

    const filtered = BENCHMARK_MODELS.filter(m => {
        if (filterCategory === 'all') return true;
        return m.category.toLowerCase().includes(filterCategory.toLowerCase());
    });

    filtered.forEach(m => {
        const tr = document.createElement('tr');
        if (m.champion) tr.style.background = 'rgba(16, 185, 129, 0.08)';

        tr.innerHTML = `
            <td><strong style="color:${m.champion ? 'var(--accent-solar)' : 'var(--text-primary)'};">${m.rank}</strong></td>
            <td>
                <strong>${m.name}</strong>
                ${m.champion ? '<span class="badge-champion ms-2"><i class="fa-solid fa-crown"></i> Champion</span>' : ''}
            </td>
            <td><span class="badge-model-type">${m.category}</span></td>
            <td>${m.mae}</td>
            <td><strong style="color:${m.champion ? 'var(--accent-emerald)' : 'var(--text-primary)'};">${m.rmse_str}</strong></td>
            <td><strong>${m.r2}</strong></td>
            <td style="color:var(--text-muted); font-size:0.82rem;">${m.time}</td>
            <td><span class="badge" style="background:${m.champion ? 'var(--accent-emerald)' : 'rgba(255,255,255,0.08)'}; color:${m.champion ? '#fff' : 'var(--text-secondary)'}; font-size:0.75rem;">${m.status}</span></td>
        `;
        tbody.appendChild(tr);
    });

    // Render Comparison Bar Chart
    renderLeaderboardChart(filtered);
}

function filterLeaderboard(cat) {
    const buttons = document.querySelectorAll('#tab-leaderboard .btn-group button');
    buttons.forEach(b => b.classList.remove('active'));
    event.target.classList.add('active');
    renderLeaderboard(cat);
}

function renderLeaderboardChart(models) {
    const isDark = currentTheme === 'dark';
    const names = models.map(m => m.name.split('(')[0].trim());
    const rmses = models.map(m => m.rmse);
    const colors = models.map(m => m.champion ? '#10b981' : (m.category === 'Deep Learning' ? '#06b6d4' : (m.category === 'Ensemble' ? '#f59e0b' : '#6366f1')));

    const trace = {
        x: names,
        y: rmses,
        type: 'bar',
        marker: { color: colors },
        text: rmses.map(r => `${r.toFixed(1)}`),
        textposition: 'auto'
    };

    const layout = {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { family: 'Inter, sans-serif', color: isDark ? '#94a3b8' : '#475569', size: 10 },
        margin: { t: 20, r: 15, l: 40, b: 60 },
        xaxis: { tickangle: -25, gridcolor: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)' },
        yaxis: { title: 'RMSE (Wh/m²)', gridcolor: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)' }
    };

    Plotly.newPlot('leaderboardChart', [trace], layout, { responsive: true, displayModeBar: false });
}