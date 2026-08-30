/**
 * Clean & Simple Frontend Logic matching the original user interface.
 * Supports Leaflet.js Mapping, Monthly Resource Charts, Station Comparison,
 * and 3-Column GHI Predictions with Feature Range Hints.
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

let map, markers = {}, gradientCircles = [], isGradientView = false;
let stationMetadataMap = {};

document.addEventListener('DOMContentLoaded', () => {
    // 1. Initialize Map
    initMap();

    // 2. Initialize Select2 Components
    $('#parameterSelect').select2({ placeholder: "Select Parameters", width: '100%' });
    $('#compareStations').select2({ placeholder: "Select Stations", width: '100%' });
    $('#compareParams').select2({ placeholder: "Select Parameters", width: '100%' });

    // 3. Load Station List and Map Markers
    loadMapAndStationData();

    // 4. Fill Prediction Form with Default Values
    loadPredictionDefaults();

    // 5. Setup Event Listeners
    document.getElementById('toggleHeatmap').addEventListener('click', toggleGradientView);
    document.getElementById('stationSelect').addEventListener('change', (e) => onStationSelectChange(e.target.value));
    $('#parameterSelect').on('change', () => onStationSelectChange(document.getElementById('stationSelect').value));
    document.getElementById('yearSelect').addEventListener('change', () => onStationSelectChange(document.getElementById('stationSelect').value));

    // Comparison Event Listeners
    $('#compareStations, #compareParams, #compareYear').on('change', handleStationComparison);

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

    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
        attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors © <a href="https://carto.com/attributions">CARTO</a>',
        maxZoom: 19
    }).addTo(map);
}

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

                // Add to dropdowns
                stationSelect.add(new Option(st.station_name, st.station_name));
                compareStations.append(new Option(st.station_name, st.station_name));

                // Add marker to map
                const marker = L.marker([st.latitude, st.longitude]).addTo(map);
                marker.bindPopup(`<strong>${st.station_name}</strong><br>Lat: ${st.latitude.toFixed(4)}<br>Long: ${st.longitude.toFixed(4)}`);
                marker.on('click', () => {
                    stationSelect.value = st.station_name;
                    onStationSelectChange(st.station_name);
                });
                markers[st.station_name] = marker;

                // Gradient circle
                const gradCircle = L.circle([st.latitude, st.longitude], {
                    radius: 35000,
                    fillColor: '#1E5631',
                    fillOpacity: 0.2,
                    stroke: false
                });
                gradientCircles.push(gradCircle);
            });

            // Initial selection
            if (stations.length > 0) {
                const first = stations[0].station_name;
                stationSelect.value = first;
                onStationSelectChange(first);

                // Set default comparison stations
                const sampleStations = stations.slice(0, 3).map(s => s.station_name);
                compareStations.val(sampleStations).trigger('change');
            }
        })
        .catch(err => console.error("Error loading map data:", err));

    // Populate Comparison & Monthly Parameters
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
            }
        })
        .catch(err => console.error("Error loading parameter list:", err));
}

function toggleGradientView() {
    isGradientView = !isGradientView;
    if (isGradientView) {
        gradientCircles.forEach(c => c.addTo(map));
    } else {
        gradientCircles.forEach(c => map.removeLayer(c));
    }
}

function onStationSelectChange(stationName) {
    if (!stationName) return;

    // Update metadata panel
    const stMeta = stationMetadataMap[stationName];
    if (stMeta) {
        document.getElementById('metaStationName').textContent = stMeta.station_name;
        document.getElementById('metaLat').textContent = stMeta.latitude.toFixed(5);
        document.getElementById('metaLong').textContent = stMeta.longitude.toFixed(5);
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
                renderMonthlyChart(data.monthly_chart_data, stationName, year);
            }
        })
        .catch(err => console.error("Error loading station details:", err));
}

function renderMonthlyChart(chartData, stationName, year) {
    const selectedParams = $('#parameterSelect').val() || ['Air Temperature (C°)'];
    const traces = [];
    const colors = ['#1E5631', '#FFC107', '#0288D1', '#E83E8C', '#6F42C1', '#28A745'];

    selectedParams.forEach((param, idx) => {
        if (chartData.data[param]) {
            const barColor = colors[idx % colors.length];
            // Bar trace
            traces.push({
                x: chartData.months,
                y: chartData.data[param],
                name: param,
                type: 'bar',
                marker: { color: barColor }
            });

            // Trendline trace
            traces.push({
                x: chartData.months,
                y: chartData.data[param],
                name: `${param} Trend`,
                type: 'scatter',
                mode: 'lines+markers',
                line: { shape: 'spline', smoothing: 1.1, width: 2, color: barColor },
                marker: { size: 6, color: barColor }
            });
        }
    });

    const titleText = `Data for ${stationName}${year ? ' (' + year + ')' : ''}`;

    const layout = {
        title: { text: titleText, font: { family: 'Poppins, sans-serif', size: 14, color: '#333' } },
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
        font: { family: 'Poppins, sans-serif', size: 11, color: '#555' },
        margin: { t: 40, r: 20, l: 50, b: 60 },
        legend: { orientation: 'h', y: -0.25, x: 0 },
        xaxis: { gridcolor: '#f0f0f0' },
        yaxis: { gridcolor: '#f0f0f0' }
    };

    Plotly.newPlot('monthlyChart', traces, layout, { responsive: true });
}

/* Station Comparison */
function handleStationComparison() {
    const stations = $('#compareStations').val();
    const params = $('#compareParams').val();
    const year = document.getElementById('compareYear').value;

    if (!stations || stations.length === 0 || !params || params.length === 0) return;

    fetch('/station-comparison', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ stations: stations, params: params, year: year })
    })
    .then(res => res.json())
    .then(data => {
        if (!data.values) return;

        const traces = [];
        const colors = ['#1E5631', '#FF5733', '#FFC107', '#007BFF', '#6F42C1', '#28A745'];
        let colorIdx = 0;

        stations.forEach(st => {
            params.forEach(p => {
                if (data.values[st] && data.values[st][p]) {
                    traces.push({
                        x: data.dates[st] || [],
                        y: data.values[st][p] || [],
                        name: `${st} - ${p}`,
                        type: 'scatter',
                        mode: 'lines',
                        line: { width: 1.8, color: colors[colorIdx % colors.length] }
                    });
                    colorIdx++;
                }
            });
        });

        const layout = {
            title: { text: `${params.join(', ')} Comparison Across Stations`, font: { family: 'Poppins, sans-serif', size: 14 } },
            paper_bgcolor: '#ffffff',
            plot_bgcolor: '#ffffff',
            font: { family: 'Poppins, sans-serif', size: 11, color: '#555' },
            margin: { t: 40, r: 20, l: 50, b: 50 },
            legend: { orientation: 'h', y: -0.2, x: 0 },
            xaxis: { gridcolor: '#f0f0f0' },
            yaxis: { gridcolor: '#f0f0f0' }
        };

        Plotly.newPlot('comparisonChart', traces, layout, { responsive: true });

        // Update Summary Statistics Table for GHI
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
    })
    .catch(err => console.error("Error running station comparison:", err));
}

/* Prediction Logic */
function loadPredictionDefaults() {
    const form = document.getElementById('predictionForm');
    const inputs = form.querySelectorAll('input');
    inputs.forEach(input => {
        const key = input.name;
        if (DEFAULT_AVERAGES[key] !== undefined) {
            input.value = DEFAULT_AVERAGES[key];
        }
    });
}

function handlePrediction(e) {
    e.preventDefault();

    const form = document.getElementById('predictionForm');
    const formData = new FormData(form);
    const payload = {};
    formData.forEach((value, key) => {
        payload[key] = parseFloat(value) || 0.0;
    });

    const modelType = document.getElementById('predictModelType').value;
    payload['model_type'] = modelType;

    const resultBox = document.getElementById('predictionResult');
    const predValText = document.getElementById('predValueText');

    fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
    .then(res => res.json())
    .then(data => {
        if (data.prediction !== undefined) {
            predValText.textContent = data.prediction.toFixed(2);
            resultBox.style.display = 'block';
            resultBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        } else if (data.error) {
            alert(`Prediction Error: ${data.error}`);
        }
    })
    .catch(err => {
        console.error("Prediction error:", err);
        alert("Failed to compute prediction. Please check inputs.");
    });
}