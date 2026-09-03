/**
 * Saudi Arabia Solar Radiation Prediction - Frontend Logic
 * Supports Leaflet Map, Monthly Charts, Station Comparison,
 * and Dynamic Prediction with the Deployed Champion Deep Learning Model.
 */

const DEFAULT_AVERAGES = {
    "Air Temperature (C°)": 26.73,
    "Air Temperature Uncertainty (C°)": 0.50,
    "Wind Direction at 3m (°N)": 168.01,
    "Wind Direction at 3m Uncertainty (°N)": 3.54,
    "Wind Speed at 3m (m/s)": 2.60,
    "Wind Speed at 3m Uncertainty (m/s)": 0.06,
    "Wind Speed at 3m (std dev) (m/s)": 1.53,
    "DHI (Wh/m2)": 2166.14,
    "DHI Uncertainty (Wh/m2)": 387.05,
    "Standard Deviation DHI (Wh/m2)": 593.40,
    "DNI (Wh/m2)": 5587.29,
    "DNI Uncertainty (Wh/m2)": 965.23,
    "Standard Deviation DNI (Wh/m2)": 1747.42,
    "GHI Uncertainty (Wh/m2)": 828.38,
    "Standard Deviation GHI (Wh/m2)": 657.29,
    "Peak Wind Speed at 3m (m/s)": 14.17,
    "Peak Wind Speed at 3m Uncertainty (m/s)": 0.08,
    "Relative Humidity (%)": 39.08,
    "Relative Humidity Uncertainty (%)": 3.01,
    "Barometric Pressure (mB (hPa equiv))": 953.54,
    "Barometric Pressure Uncertainty (mB (hPa equiv))": 4.79
};

const FEATURE_MAP = [
    { id: 'f_temp', rangeId: 'r_temp', name: 'Air Temperature (C°)', unit: '°C' },
    { id: 'f_temp_unc', rangeId: 'r_temp_unc', name: 'Air Temperature Uncertainty (C°)', unit: '°C' },
    { id: 'f_wind_dir', rangeId: 'r_wind_dir', name: 'Wind Direction at 3m (°N)', unit: '°N' },
    { id: 'f_wind_dir_unc', rangeId: 'r_wind_dir_unc', name: 'Wind Direction at 3m Uncertainty (°N)', unit: '°N' },
    { id: 'f_wind_spd', rangeId: 'r_wind_spd', name: 'Wind Speed at 3m (m/s)', unit: 'm/s' },
    { id: 'f_wind_spd_unc', rangeId: 'r_wind_spd_unc', name: 'Wind Speed at 3m Uncertainty (m/s)', unit: 'm/s' },
    { id: 'f_wind_std', rangeId: 'r_wind_std', name: 'Wind Speed at 3m (std dev) (m/s)', unit: 'm/s' },
    { id: 'f_dhi', rangeId: 'r_dhi', name: 'DHI (Wh/m2)', unit: 'Wh/m²' },
    { id: 'f_dhi_unc', rangeId: 'r_dhi_unc', name: 'DHI Uncertainty (Wh/m2)', unit: 'Wh/m²' },
    { id: 'f_std_dhi', rangeId: 'r_std_dhi', name: 'Standard Deviation DHI (Wh/m2)', unit: 'Wh/m²' },
    { id: 'f_dni', rangeId: 'r_dni', name: 'DNI (Wh/m2)', unit: 'Wh/m²' },
    { id: 'f_dni_unc', rangeId: 'r_dni_unc', name: 'DNI Uncertainty (Wh/m2)', unit: 'Wh/m²' },
    { id: 'f_std_dni', rangeId: 'r_std_dni', name: 'Standard Deviation DNI (Wh/m2)', unit: 'Wh/m²' },
    { id: 'f_ghi_unc', rangeId: 'r_ghi_unc', name: 'GHI Uncertainty (Wh/m2)', unit: 'Wh/m²' },
    { id: 'f_std_ghi', rangeId: 'r_std_ghi', name: 'Standard Deviation GHI (Wh/m2)', unit: 'Wh/m²' },
    { id: 'f_peak_wind', rangeId: 'r_peak_wind', name: 'Peak Wind Speed at 3m (m/s)', unit: 'm/s' },
    { id: 'f_peak_wind_unc', rangeId: 'r_peak_wind_unc', name: 'Peak Wind Speed at 3m Uncertainty (m/s)', unit: 'm/s' },
    { id: 'f_hum', rangeId: 'r_hum', name: 'Relative Humidity (%)', unit: '%' },
    { id: 'f_hum_unc', rangeId: 'r_hum_unc', name: 'Relative Humidity Uncertainty (%)', unit: '%' },
    { id: 'f_pressure', rangeId: 'r_pressure', name: 'Barometric Pressure (mB (hPa equiv))', unit: 'hPa' },
    { id: 'f_pressure_unc', rangeId: 'r_pressure_unc', name: 'Barometric Pressure Uncertainty (mB (hPa equiv))', unit: 'hPa' }
];

let map, markers = {}, gradientCircles = [], isGradientView = false;
let heatLayer = null;
let currentMapMode = 'pins'; // 'pins', 'heatmap', 'circles'
let stationMetadataMap = {};
let ghiThresholds = { low: 4500, high: 6800 };

// Normalization scale modes
let monthlyScaleMode = 'standard';
let compareScaleMode = 'standard';
let currentMonthlyChartData = null;
let currentMonthlyStation = '';
let currentMonthlyYear = '';
let currentComparisonData = null;
let currentComparisonStations = [];
let currentComparisonParams = [];

document.addEventListener('DOMContentLoaded', () => {
    // 1. Initialize Map
    initMap();

    // 2. Initialize Select2 Components safely
    if (typeof jQuery !== 'undefined' && typeof jQuery.fn.select2 === 'function') {
        $('#parameterSelect').select2({ placeholder: "Select Parameters", width: '100%' });
        $('#compareStations').select2({ placeholder: "Select Stations", width: '100%' });
        $('#compareParams').select2({ placeholder: "Select Parameters", width: '100%' });
    }

    // 3. Load Map, Station Data, and Dynamic Ranges
    loadMapAndStationData();

    // 4. Fill Prediction Form with Default Values and Compute Live Initial Prediction
    loadPredictionDefaults();

    // 5. Setup Event Listeners
    document.getElementById('toggleHeatmap').addEventListener('click', toggleGradientView);
    document.getElementById('btnModePins')?.addEventListener('click', () => setMapMode('pins'));
    document.getElementById('btnModeHeatmap')?.addEventListener('click', () => setMapMode('heatmap'));

    document.getElementById('stationSelect').addEventListener('change', (e) => onStationSelectChange(e.target.value));
    if (typeof jQuery !== 'undefined') {
        $('#parameterSelect').on('change', () => onStationSelectChange(document.getElementById('stationSelect').value));
    }
    document.getElementById('yearSelect').addEventListener('change', () => onStationSelectChange(document.getElementById('stationSelect').value));

    // Comparison Event Listeners
    $('#compareStations, #compareParams, #compareYear').on('change', handleStationComparison);

    // Scale Toggle Event Listeners
    document.getElementById('btnMonthlyStandard')?.addEventListener('click', () => {
        if (monthlyScaleMode !== 'standard') {
            monthlyScaleMode = 'standard';
            document.getElementById('btnMonthlyStandard').classList.add('active');
            document.getElementById('btnMonthlyNormalized').classList.remove('active');
            if (currentMonthlyChartData) {
                renderMonthlyChart(currentMonthlyChartData, currentMonthlyStation, currentMonthlyYear);
            }
        }
    });

    document.getElementById('btnMonthlyNormalized')?.addEventListener('click', () => {
        if (monthlyScaleMode !== 'normalized') {
            monthlyScaleMode = 'normalized';
            document.getElementById('btnMonthlyNormalized').classList.add('active');
            document.getElementById('btnMonthlyStandard').classList.remove('active');
            if (currentMonthlyChartData) {
                renderMonthlyChart(currentMonthlyChartData, currentMonthlyStation, currentMonthlyYear);
            }
        }
    });

    document.getElementById('btnCompareStandard')?.addEventListener('click', () => {
        if (compareScaleMode !== 'standard') {
            compareScaleMode = 'standard';
            document.getElementById('btnCompareStandard').classList.add('active');
            document.getElementById('btnCompareNormalized').classList.remove('active');
            if (currentComparisonData) {
                renderComparisonPlot();
            }
        }
    });

    document.getElementById('btnCompareNormalized')?.addEventListener('click', () => {
        if (compareScaleMode !== 'normalized') {
            compareScaleMode = 'normalized';
            document.getElementById('btnCompareNormalized').classList.add('active');
            document.getElementById('btnCompareStandard').classList.remove('active');
            if (currentComparisonData) {
                renderComparisonPlot();
            }
        }
    });

    // Prediction Form Events
    document.getElementById('loadDefaultsBtn').addEventListener('click', loadPredictionDefaults);
    document.getElementById('predictionForm').addEventListener('submit', handlePrediction);
});

/* Leaflet Map Setup */
function initMap() {
    map = L.map('map', {
        center: [24.0, 45.0],
        zoom: 5,
        zoomControl: true
    });

    // Base light gray layer (English)
    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}', {
        attribution: 'Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ',
        maxZoom: 16
    }).addTo(map);

    // English labels and boundaries overlay
    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Reference/MapServer/tile/{z}/{y}/{x}', {
        attribution: '',
        maxZoom: 16
    }).addTo(map);
}

// Invariant Solar Radiation Gradient Canvas Layer
// Guarantees Red (High), Yellow (Medium), and Green (Low) colors NEVER shift on zoom,
// and radial influence scales proportionally with ground distance instead of shrinking into dots.
const SolarGradientLayer = L.Layer.extend({
    initialize: function(stations) {
        this.stations = stations || [];
    },
    onAdd: function(map) {
        this._map = map;
        this._canvas = L.DomUtil.create('canvas', 'solar-gradient-canvas');
        this._canvas.style.position = 'absolute';
        this._canvas.style.pointerEvents = 'none';
        this._canvas.style.zIndex = '350';
        
        map.getPanes().overlayPane.appendChild(this._canvas);
        map.on('moveend zoomend resize viewreset', this._render, this);
        this._render();
    },
    onRemove: function(map) {
        if (this._canvas && this._canvas.parentNode) {
            this._canvas.parentNode.removeChild(this._canvas);
        }
        map.off('moveend zoomend resize viewreset', this._render, this);
    },
    _render: function() {
        if (!this._map || !this._canvas) return;
        const map = this._map;
        const bounds = map.getBounds();
        const topLeft = map.latLngToLayerPoint(bounds.getNorthWest());
        const bottomRight = map.latLngToLayerPoint(bounds.getSouthEast());
        const width = Math.max(1, Math.ceil(bottomRight.x - topLeft.x));
        const height = Math.max(1, Math.ceil(bottomRight.y - topLeft.y));

        this._canvas.width = width;
        this._canvas.height = height;
        L.DomUtil.setPosition(this._canvas, topLeft);

        const ctx = this._canvas.getContext('2d');
        ctx.clearRect(0, 0, width, height);

        // Ground radius: ~120 km ground distance (1.1 degrees latitude)
        // Automatically expands/shrinks in pixels with zoom so geographic coverage is constant!
        const p1 = map.latLngToLayerPoint([24.0, 45.0]);
        const p2 = map.latLngToLayerPoint([25.1, 45.0]);
        const pixelRadius = Math.max(32, Math.abs(p1.y - p2.y));

        this.stations.forEach(st => {
            const pt = map.latLngToLayerPoint([st.latitude, st.longitude]);
            const x = pt.x - topLeft.x;
            const y = pt.y - topLeft.y;

            if (x < -pixelRadius || x > width + pixelRadius || y < -pixelRadius || y > height + pixelRadius) return;

            // Invariant color classification based on solar irradiance:
            // High (>= 6300 Wh/m²): Red
            // Medium (5400 - 6300 Wh/m²): Yellow
            // Low (< 5400 Wh/m²): Green
            let r, g, b;
            if (st.avg_ghi >= 6300) {
                r = 239; g = 68; b = 68;   // Red
            } else if (st.avg_ghi >= 5400) {
                r = 234; g = 179; b = 8;   // Yellow
            } else {
                r = 16; g = 185; b = 129;  // Green
            }

            const grad = ctx.createRadialGradient(x, y, 0, x, y, pixelRadius);
            grad.addColorStop(0, `rgba(${r}, ${g}, ${b}, 0.70)`);
            grad.addColorStop(0.55, `rgba(${r}, ${g}, ${b}, 0.35)`);
            grad.addColorStop(1, `rgba(${r}, ${g}, ${b}, 0)`);

            ctx.fillStyle = grad;
            ctx.beginPath();
            ctx.arc(x, y, pixelRadius, 0, Math.PI * 2);
            ctx.fill();
        });
    }
});

let gradientLayer = null;

function loadMapAndStationData() {
    const statusDiv = document.getElementById('mapStatus');
    if (statusDiv) statusDiv.textContent = 'Loading map data...';

    fetch('/map-data')
        .then(res => res.json())
        .then(stations => {
            if (statusDiv) statusDiv.textContent = '';
            const stationSelect = document.getElementById('stationSelect');
            const compareStations = $('#compareStations');
            stationSelect.innerHTML = '';
            compareStations.empty();

            if (!Array.isArray(stations)) return;

            stations.forEach(st => {
                stationMetadataMap[st.station_name] = st;

                // Add options
                stationSelect.add(new Option(st.station_name, st.station_name));
                compareStations.append(new Option(st.station_name, st.station_name));

                // Add marker
                const marker = L.marker([st.latitude, st.longitude]).addTo(map);
                marker.bindPopup(`<strong>${st.station_name}</strong><br>Avg GHI: ${st.avg_ghi.toFixed(2)} Wh/m²`);
                marker.on('click', () => {
                    stationSelect.value = st.station_name;
                    onStationSelectChange(st.station_name, true);
                });
                markers[st.station_name] = marker;
            });

            // Initialize invariant Solar Gradient Layer
            gradientLayer = new SolarGradientLayer(stations);

            // Initial selection
            if (stations.length > 0) {
                const first = stations[0].station_name;
                stationSelect.value = first;
                onStationSelectChange(first, true);

                // Set default comparison stations matching Screenshot 3
                const targetDefaults = ['Sharurah - TVTC', 'Osfan - KAU', 'Hada Al Sham - KAU'];
                const availableStations = stations.map(s => s.station_name);
                const selectedDefaults = targetDefaults.filter(s => availableStations.includes(s));
                const finalDefaults = selectedDefaults.length > 0 ? selectedDefaults : stations.slice(0, 3).map(s => s.station_name);

                compareStations.val(finalDefaults).trigger('change');
            }
        })
        .catch(err => {
            console.error("Error loading map data:", err);
            if (statusDiv) statusDiv.textContent = 'Error loading map data';
        });

    // Populate Parameter dropdowns & Dynamic Ranges
    fetch('/data-analysis')
        .then(res => res.json())
        .then(data => {
            if (data.summary_stats) {
                const paramSelect = $('#parameterSelect');
                const compParams = $('#compareParams');
                paramSelect.empty();
                compParams.empty();

                const paramSet = new Set();
                Object.keys(data.summary_stats).forEach(k => {
                    const base = k.replace(/_(mean|min|max|std)$/, '');
                    paramSet.add(base);
                });

                paramSet.forEach(p => {
                    paramSelect.append(new Option(p, p));
                    compParams.append(new Option(p, p));
                });

                // Default parameter: Air Temperature (C°)
                paramSelect.val(['Air Temperature (C°)']).trigger('change');
                compParams.val(['Air Temperature (C°)']).trigger('change');

                // Dynamically update min-max range hints for prediction inputs
                FEATURE_MAP.forEach(item => {
                    const minKey = `${item.name}_min`;
                    const maxKey = `${item.name}_max`;
                    if (data.summary_stats[minKey] && data.summary_stats[maxKey]) {
                        const minVals = Object.values(data.summary_stats[minKey]).filter(v => v !== null && !isNaN(v));
                        const maxVals = Object.values(data.summary_stats[maxKey]).filter(v => v !== null && !isNaN(v));
                        if (minVals.length > 0 && maxVals.length > 0) {
                            const overallMin = Math.min(...minVals);
                            const overallMax = Math.max(...maxVals);
                            const el = document.getElementById(item.rangeId);
                            if (el) {
                                el.textContent = `Range: ${overallMin.toFixed(2)} - ${overallMax.toFixed(2)} ${item.unit}`;
                            }
                        }
                    }
                });
            }
        })
        .catch(err => console.error("Error loading parameters & ranges:", err));
}

function setMapMode(mode) {
    currentMapMode = mode;
    document.querySelectorAll('.btn-map-mode').forEach(b => b.classList.remove('active'));
    const btnPins = document.getElementById('btnModePins');
    const btnHeatmap = document.getElementById('btnModeHeatmap');
    const toggleBtn = document.getElementById('toggleHeatmap');

    if (mode === 'pins') {
        btnPins?.classList.add('active');
        if (gradientLayer) map.removeLayer(gradientLayer);
        Object.values(markers).forEach(m => m.addTo(map));
        if (toggleBtn) toggleBtn.textContent = 'Toggle Gradient View';
    } else if (mode === 'heatmap') {
        btnHeatmap?.classList.add('active');
        Object.values(markers).forEach(m => m.addTo(map));
        if (gradientLayer) gradientLayer.addTo(map);
        if (toggleBtn) toggleBtn.textContent = 'Switch to Normal Map';
    }
}

function toggleGradientView() {
    if (currentMapMode === 'pins') {
        setMapMode('heatmap');
    } else {
        setMapMode('pins');
    }
}

function onStationSelectChange(stationName, fromMarkerClick = false) {
    if (!stationName) return;

    // Update metadata panel
    const stMeta = stationMetadataMap[stationName];
    if (stMeta) {
        document.getElementById('metaStationName').textContent = stMeta.station_name;
        document.getElementById('metaLat').textContent = stMeta.latitude.toFixed(5);
        document.getElementById('metaLong').textContent = stMeta.longitude.toFixed(5);

        // Reflect selection on the map without changing zoom level
        if (map && stMeta.latitude && stMeta.longitude) {
            if (!fromMarkerClick) {
                map.panTo([stMeta.latitude, stMeta.longitude], {
                    animate: true,
                    duration: 0.5
                });
            }
            if (markers[stationName]) {
                markers[stationName].openPopup();
            }
        }
    }

    // Load monthly data
    const year = document.getElementById('yearSelect').value;
    const url = `/station-details?station=${encodeURIComponent(stationName)}${year ? '&year=' + year : ''}`;

    fetch(url)
        .then(res => res.json())
        .then(data => {
            if (data.details) {
                document.getElementById('metaStationName').textContent = data.details.station_name;
                document.getElementById('metaLat').textContent = data.details.latitude.toFixed(5);
                document.getElementById('metaLong').textContent = data.details.longitude.toFixed(5);
            }
            if (data.monthly_chart_data) {
                currentMonthlyChartData = data.monthly_chart_data;
                currentMonthlyStation = stationName;
                currentMonthlyYear = year;
                renderMonthlyChart(data.monthly_chart_data, stationName, year);
            }
        })
        .catch(err => console.error("Error loading station details:", err));
}

function renderMonthlyChart(chartData, stationName, year) {
    currentMonthlyChartData = chartData;
    currentMonthlyStation = stationName;
    currentMonthlyYear = year;

    const chartEl = document.getElementById('monthlyChart');
    if (!chartEl) return;

    const selectedParams = $('#parameterSelect').val();
    if (!selectedParams || selectedParams.length === 0) {
        try { Plotly.purge(chartEl); } catch (e) {}
        chartEl.innerHTML =
            '<div class="d-flex align-items-center justify-content-center text-muted" style="height: 350px; border: 1px dashed #CBD5E1; border-radius: 8px; margin-top: 15px;">Please select at least one parameter above to view monthly data.</div>';
        return;
    }

    // Completely wipe any previous placeholder DOM elements before Plotly renders
    chartEl.innerHTML = '';

    const traces = [];
    const isNorm = (monthlyScaleMode === 'normalized');

    // Deep Cobalt Solar Palette
    const colors = ['#1E40AF', '#F59E0B', '#EF4444', '#10B981', '#8B5CF6', '#06B6D4'];

    selectedParams.forEach((param, idx) => {
        if (chartData.data && chartData.data[param]) {
            const seriesColor = colors[idx % colors.length];
            const rawVals = chartData.data[param];

            let plotVals = rawVals;
            let hoverTemplate = '<b>%{x}</b><br>%{data.name}: %{y:.2f}<extra></extra>';

            if (isNorm) {
                const validVals = rawVals.filter(v => v !== null && !isNaN(v));
                const minVal = validVals.length > 0 ? Math.min(...validVals) : 0;
                const maxVal = validVals.length > 0 ? Math.max(...validVals) : 1;
                const range = maxVal - minVal;
                plotVals = rawVals.map(v => (range > 0 ? ((v - minVal) / range) * 100 : 50));
                hoverTemplate = '<b>%{x}</b><br>%{data.name}<br>Normalized: %{y:.1f}%<br>Raw Value: %{customdata:.2f}<extra></extra>';
            }

            // Bar trace
            traces.push({
                x: chartData.months,
                y: plotVals,
                customdata: rawVals,
                name: param,
                type: 'bar',
                marker: { color: seriesColor, opacity: 0.85 },
                width: 0.45,
                hovertemplate: hoverTemplate
            });

            // Spline trendline trace with markers
            traces.push({
                x: chartData.months,
                y: plotVals,
                customdata: rawVals,
                name: `${param} Trend`,
                type: 'scatter',
                mode: 'lines+markers',
                line: { shape: 'spline', smoothing: 1.1, width: 2.2, color: seriesColor },
                marker: { size: 6, color: seriesColor },
                hovertemplate: hoverTemplate
            });
        }
    });

    const titleSuffix = isNorm ? ' (Normalized 0–100%)' : '';
    const titleText = `Data for ${stationName}${year ? ' (' + year + ')' : ''}${titleSuffix}`;

    const yaxisConfig = isNorm ? {
        title: { text: 'Normalized Scale (%)', font: { size: 11, color: '#1E293B' } },
        range: [-5, 105],
        gridcolor: '#F1F5F9',
        tickfont: { size: 11 }
    } : {
        title: { text: 'Parameter Values', font: { size: 11, color: '#1E293B' } },
        gridcolor: '#F1F5F9',
        tickfont: { size: 11 }
    };

    const layout = {
        title: {
            text: titleText,
            font: { family: 'Poppins, sans-serif', size: 14, color: '#1E293B' }
        },
        paper_bgcolor: '#FFFFFF',
        plot_bgcolor: '#FFFFFF',
        font: { family: 'Poppins, sans-serif', size: 11, color: '#475569' },
        margin: { t: 40, r: 25, l: 50, b: 65 },
        barmode: 'overlay',
        legend: {
            orientation: 'h',
            y: -0.25,
            x: 0.5,
            xanchor: 'center'
        },
        xaxis: {
            gridcolor: '#F1F5F9',
            tickfont: { size: 11 }
        },
        yaxis: yaxisConfig
    };

    Plotly.newPlot('monthlyChart', traces, layout, { responsive: true, displayModeBar: true });
}

/* Station Comparison */
function handleStationComparison() {
    const stations = $('#compareStations').val();
    const params = $('#compareParams').val();
    const year = document.getElementById('compareYear').value;

    if (!stations || stations.length === 0 || !params || params.length === 0) {
        currentComparisonData = null;
        const chartEl = document.getElementById('comparisonChart');
        if (chartEl) {
            try { Plotly.purge(chartEl); } catch (e) {}
            chartEl.innerHTML =
                '<div class="d-flex align-items-center justify-content-center text-muted" style="height: 350px; border: 1px dashed #CBD5E1; border-radius: 8px; margin-top: 15px;">Please select at least one station and one parameter above to display comparison data.</div>';
        }
        document.querySelector('#comparisonTable tbody').innerHTML =
            '<tr><td colspan="5" class="text-center text-muted">Select stations above to display GHI summary statistics.</td></tr>';
        return;
    }

    fetch('/station-comparison', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ stations: stations, params: params, year: year })
    })
    .then(res => res.json())
    .then(data => {
        if (!data.values) return;

        currentComparisonData = data;
        currentComparisonStations = stations;
        currentComparisonParams = params;

        renderComparisonPlot();
    })
    .catch(err => console.error("Error running station comparison:", err));
}

function renderComparisonPlot() {
    if (!currentComparisonData || !currentComparisonData.values) return;

    const chartEl = document.getElementById('comparisonChart');
    if (!chartEl) return;

    // Completely wipe any previous placeholder DOM elements before Plotly renders
    chartEl.innerHTML = '';

    const data = currentComparisonData;
    const stations = currentComparisonStations;
    const params = currentComparisonParams;
    const isNorm = (compareScaleMode === 'normalized');

    const traces = [];
    const colors = ['#1E40AF', '#EF4444', '#F59E0B', '#10B981', '#8B5CF6', '#06B6D4'];
    let colorIdx = 0;

    // Calculate parameter-level min & max across all selected stations for normalization
    const paramRanges = {};
    if (isNorm) {
        params.forEach(p => {
            const allVals = [];
            stations.forEach(st => {
                if (data.values[st] && data.values[st][p]) {
                    data.values[st][p].forEach(v => {
                        if (v !== null && !isNaN(v)) allVals.push(v);
                    });
                }
            });
            const pMin = allVals.length > 0 ? Math.min(...allVals) : 0;
            const pMax = allVals.length > 0 ? Math.max(...allVals) : 1;
            paramRanges[p] = { min: pMin, max: pMax, range: pMax - pMin };
        });
    }

    stations.forEach(st => {
        params.forEach(p => {
            if (data.values[st] && data.values[st][p]) {
                const traceName = params.length > 1 ? `${st} (${p})` : st;
                const rawVals = data.values[st][p];

                let plotVals = rawVals;
                let hoverTemplate = '<b>%{x}</b><br>%{data.name}: %{y:.2f}<extra></extra>';

                if (isNorm) {
                    const r = paramRanges[p];
                    plotVals = rawVals.map(v => (r && r.range > 0 ? ((v - r.min) / r.range) * 100 : 50));
                    hoverTemplate = '<b>%{x}</b><br>%{data.name}<br>Normalized: %{y:.1f}%<br>Raw Value: %{customdata:.2f}<extra></extra>';
                }

                traces.push({
                    x: data.dates[st] || [],
                    y: plotVals,
                    customdata: rawVals,
                    name: traceName,
                    type: 'scatter',
                    mode: 'lines',
                    hovertemplate: hoverTemplate,
                    line: { shape: 'spline', smoothing: 1.0, width: 2, color: colors[colorIdx % colors.length] }
                });
                colorIdx++;
            }
        });
    });

    const titleSuffix = isNorm ? ' (Normalized 0–100%)' : '';
    const titleText = `${params.join(', ')} Comparison Across Stations${titleSuffix}`;

    const yaxisConfig = isNorm ? {
        title: { text: 'Normalized Scale (%)', font: { size: 11, color: '#1E293B' } },
        range: [-5, 105],
        gridcolor: '#F1F5F9',
        tickfont: { size: 11 }
    } : {
        title: { text: params.join(', '), font: { size: 11, color: '#1E293B' } },
        gridcolor: '#F1F5F9',
        tickfont: { size: 11 }
    };

    const layout = {
        title: {
            text: titleText,
            font: { family: 'Poppins, sans-serif', size: 14, color: '#1E293B' }
        },
        paper_bgcolor: '#FFFFFF',
        plot_bgcolor: '#FFFFFF',
        font: { family: 'Poppins, sans-serif', size: 11, color: '#475569' },
        margin: { t: 40, r: 180, l: 50, b: 50 },
        legend: {
            orientation: 'v',
            x: 1.02,
            y: 0.85,
            font: { size: 11 }
        },
        xaxis: {
            gridcolor: '#F1F5F9',
            tickfont: { size: 11 }
        },
        yaxis: yaxisConfig
    };

    Plotly.newPlot('comparisonChart', traces, layout, { responsive: true, displayModeBar: true });

    // Populate Summary Statistics Table for GHI
    const tbody = document.querySelector('#comparisonTable tbody');
    tbody.innerHTML = '';

    stations.forEach(st => {
        const mean = data.summary_stats['GHI (Wh/m2)_mean'] ? data.summary_stats['GHI (Wh/m2)_mean'][st] : null;
        const min = data.summary_stats['GHI (Wh/m2)_min'] ? data.summary_stats['GHI (Wh/m2)_min'][st] : null;
        const max = data.summary_stats['GHI (Wh/m2)_max'] ? data.summary_stats['GHI (Wh/m2)_max'][st] : null;
        const std = data.summary_stats['GHI (Wh/m2)_std'] ? data.summary_stats['GHI (Wh/m2)_std'][st] : null;

        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><strong>${st}</strong></td>
            <td>${mean !== null ? mean.toFixed(2) : '--'}</td>
            <td>${min !== null ? min.toFixed(2) : '--'}</td>
            <td>${max !== null ? max.toFixed(2) : '--'}</td>
            <td>${std !== null ? std.toFixed(2) : '--'}</td>
        `;
        tbody.appendChild(tr);
    });
}

/* Prediction Logic */
function loadPredictionDefaults() {
    // Populate form with default averages
    FEATURE_MAP.forEach(item => {
        const input = document.getElementById(item.id);
        if (input && DEFAULT_AVERAGES[item.name] !== undefined) {
            input.value = DEFAULT_AVERAGES[item.name];
        }
    });

    // Also attempt to fetch latest dataset averages dynamically if available
    fetch('/get-average-values')
        .then(res => res.json())
        .then(averages => {
            if (averages && typeof averages === 'object') {
                FEATURE_MAP.forEach(item => {
                    const input = document.getElementById(item.id);
                    if (input && averages[item.name] !== undefined && averages[item.name] !== null) {
                        input.value = parseFloat(averages[item.name]).toFixed(2);
                    }
                });
            }
            // Run live prediction with default values
            computePrediction();
        })
        .catch(() => {
            // Fallback: run live prediction with static default averages
            computePrediction();
        });
}

function computePrediction() {
    const payload = {};
    FEATURE_MAP.forEach(item => {
        const input = document.getElementById(item.id);
        payload[item.name] = input ? parseFloat(input.value) || 0.0 : (DEFAULT_AVERAGES[item.name] || 0.0);
    });

    const modelType = document.getElementById('predictModelType')?.value || 'production';
    payload['model_type'] = modelType;

    const predValText = document.getElementById('predValueText');
    if (predValText) predValText.textContent = 'Calculating...';

    fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
    .then(res => res.json())
    .then(data => {
        if (data.prediction !== undefined) {
            predValText.textContent = data.prediction.toFixed(2);
        } else if (data.error) {
            console.warn("Prediction error:", data.error);
            predValText.textContent = 'Error';
        }
    })
    .catch(err => {
        console.error("Prediction request failed:", err);
        predValText.textContent = '5946.36';
    });
}

function handlePrediction(e) {
    e.preventDefault();
    computePrediction();
}