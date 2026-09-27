// Global Variables for Scan Result Images
let rawImageUrl = "";
let heatmapImageUrl = "";
let spectralChartInstance = null;

// DOM Elements
const scanBtn = document.getElementById("scan-btn");
const clearBtn = document.getElementById("clear-btn");
const plantSelect = document.getElementById("plant-select");
const heatmapToggle = document.getElementById("heatmap-toggle");
const cameraFeed = document.getElementById("camera-feed");

// Metrics DOM
const detectedPlantName = document.getElementById("detected-plant-name");
const detectedScientificName = document.getElementById("detected-scientific-name");
const idConfidence = document.getElementById("id-confidence");
const freshnessValue = document.getElementById("freshness-value");
const freshnessBar = document.getElementById("freshness-bar");
const qualityGrade = document.getElementById("quality-grade");
const maturityLevel = document.getElementById("maturity-level");
const shelfLife = document.getElementById("shelf-life");
const qualityScore = document.getElementById("quality-score");

// Diagnostics DOM
const tempVal = document.getElementById("temp-val");
const humidityVal = document.getElementById("humidity-val");
const ndviVal = document.getElementById("ndvi-val");
const wbiVal = document.getElementById("wbi-val");

// XAI and Tabs DOM
const xaiReport = document.getElementById("xai-report");
const infoNutrition = document.getElementById("info-nutrition");
const infoStorage = document.getElementById("info-storage");
const infoSimilar = document.getElementById("info-similar");
const infoEducation = document.getElementById("info-education");

// Compound DOM
const compoundsPlaceholder = document.getElementById("compounds-placeholder");
const compoundsList = document.getElementById("compounds-list");

// Tab Controls
const tabButtons = document.querySelectorAll(".tab-btn");
const tabPanes = document.querySelectorAll(".tab-pane");

// Initialize on load
document.addEventListener("DOMContentLoaded", () => {
    initChart();
    setupEventListeners();
});

// Setup Chart.js
function initChart() {
    const ctx = document.getElementById("spectralChart").getContext("2d");
    
    // Default flat data
    const wavelengths = [410, 435, 460, 485, 510, 535, 560, 585, 610, 645, 680, 705, 730, 760, 810, 860, 900, 940];
    const defaultData = Array(18).fill(0.1);

    spectralChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: wavelengths,
            datasets: [{
                label: 'Reflectance (410nm - 940nm)',
                data: defaultData,
                borderColor: '#00e5ff',
                backgroundColor: 'rgba(0, 229, 255, 0.1)',
                borderWidth: 2,
                pointBackgroundColor: '#bd00ff',
                pointBorderColor: '#fff',
                pointRadius: 3,
                tension: 0.3,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    labels: {
                        color: '#8b9bb4',
                        font: { family: 'Outfit', size: 10 }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Wavelength (nm)',
                        color: '#8b9bb4',
                        font: { family: 'Outfit', size: 9 }
                    },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#8b9bb4', font: { family: 'Outfit', size: 8 } }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Reflectance Ratio',
                        color: '#8b9bb4',
                        font: { family: 'Outfit', size: 9 }
                    },
                    min: 0,
                    max: 1.0,
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#8b9bb4', font: { family: 'Outfit', size: 8 } }
                }
            }
        }
    });
}

// Setup Listeners
function setupEventListeners() {
    // Scan Trigger
    scanBtn.addEventListener("click", triggerScan);
    
    // Clear History
    clearBtn.addEventListener("click", clearLogs);
    
    // Grad-CAM Heatmap Toggle
    heatmapToggle.addEventListener("change", (e) => {
        if (e.target.checked) {
            cameraFeed.src = heatmapImageUrl;
        } else {
            cameraFeed.src = rawImageUrl;
        }
    });

    // Info Tabs
    tabButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            tabButtons.forEach(b => b.classList.remove("active"));
            tabPanes.forEach(p => p.classList.remove("active"));
            
            btn.classList.add("active");
            const tabId = btn.getAttribute("data-tab");
            document.getElementById(tabId).classList.add("active");
        });
    });
}

// Trigger Scan API
function triggerScan() {
    // UI Loading State
    scanBtn.disabled = true;
    scanBtn.textContent = "Scanning...";
    scanBtn.style.opacity = "0.7";
    
    const payload = {};
    if (plantSelect) {
        payload.plant_name = plantSelect.value || null;
    }

    fetch("/scan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    })
    .then(response => response.json())
    .then(result => {
        if (result.status === "success") {
            updateDashboard(result.data);
        } else {
            alert("Scan failed: " + result.message);
        }
    })
    .catch(err => {
        console.error("Scan error:", err);
        alert("Scan request failed. Is the server running?");
    })
    .finally(() => {
        scanBtn.disabled = false;
        scanBtn.textContent = "Start Scan";
        scanBtn.style.opacity = "1.0";
    });
}

// Update Dashboard UI Elements
function updateDashboard(data) {
    // Set Images
    rawImageUrl = data.image_path;
    heatmapImageUrl = data.heatmap_path;
    
    // Default to raw image, uncheck XAI toggle
    cameraFeed.src = rawImageUrl;
    heatmapToggle.disabled = false;
    heatmapToggle.checked = false;

    // Update Core Identity
    detectedPlantName.textContent = data.plant_name;
    detectedScientificName.textContent = data.scientific_name;
    idConfidence.textContent = `${data.confidence}%`;
    
    // Update Freshness Progress Bar
    freshnessValue.textContent = `${data.freshness}%`;
    freshnessBar.style.width = `${data.freshness}%`;
    
    // Color code freshness bar
    if (data.freshness >= 80) {
        freshnessBar.style.background = "linear-gradient(90deg, #00e5ff, #39ff14)";
    } else if (data.freshness >= 55) {
        freshnessBar.style.background = "linear-gradient(90deg, #ff9100, #ffff00)";
    } else {
        freshnessBar.style.background = "linear-gradient(90deg, #ff3333, #ff9100)";
    }

    // Update Grade / Maturity / Shelf-life / Quality Score
    qualityGrade.textContent = data.grade;
    maturityLevel.textContent = data.maturity_label;
    shelfLife.textContent = `${data.shelf_life} Days`;
    qualityScore.textContent = `${data.quality}/100`;

    // Diagnostics
    tempVal.textContent = `${data.temperature} °C`;
    humidityVal.textContent = `${data.humidity} %`;
    ndviVal.textContent = data.indices.NDVI.toFixed(3);
    wbiVal.textContent = data.indices.WBI.toFixed(3);

    // Explainability Text Report
    xaiReport.innerHTML = data.explanation_report;

    // Tabs info lookup
    infoNutrition.innerHTML = data.nutrition;
    infoStorage.innerHTML = data.storage_tips;
    infoSimilar.textContent = data.similar_plants;
    infoEducation.textContent = data.educational_supplement;

    // Render Bioactive Compounds list
    compoundsPlaceholder.classList.add("hidden");
    compoundsList.classList.remove("hidden");
    compoundsList.innerHTML = "";

    // Maximum compound scaling boundaries for visual rendering
    const maxReferenceValues = {
        "Chlorophyll": 80.0,
        "Lycopene": 40.0,
        "Curcumin": 180.0,
        "Anthocyanins": 25.0,
        "Rosmarinic_acid": 35.0,
        "Linalool": 2.5
    };

    for (const [compName, compData] of Object.entries(data.compounds)) {
        const value = compData.value;
        const unit = compData.unit;
        
        // Compute width percentage
        const nameSanitized = compName.replace("-", "_");
        const refVal = maxReferenceValues[nameSanitized] || 100.0;
        const widthPct = Math.min(100, (value / refVal) * 100);

        const compoundItem = document.createElement("div");
        compoundItem.className = `compound-item ${nameSanitized}`;
        compoundItem.innerHTML = `
            <div class="compound-meta">
                <span class="compound-name">${compName}</span>
                <span class="compound-val">${value} ${unit}</span>
            </div>
            <div class="compound-bar-bg">
                <div class="compound-bar" style="width: ${widthPct}%"></div>
            </div>
        `;
        compoundsList.appendChild(compoundItem);
    }

    // Update Chart.js Reflectance Graph
    const spectralDataset = [];
    const wavelengths = [410, 435, 460, 485, 510, 535, 560, 585, 610, 645, 680, 705, 730, 760, 810, 860, 900, 940];
    
    wavelengths.forEach(wl => {
        spectralDataset.push(data.raw_spectrum[wl.toString()] || data.raw_spectrum[wl] || 0.0);
    });

    spectralChartInstance.data.datasets[0].data = spectralDataset;
    spectralChartInstance.data.datasets[0].label = `Reflectance Spectrum: ${data.plant_name}`;
    spectralChartInstance.update();
}

// Clear Logs
function clearLogs() {
    if (!confirm("Are you sure you want to clear scan logs and database entries?")) return;

    fetch("/clear_history", { method: "POST" })
    .then(res => res.json())
    .then(result => {
        alert(result.message);
        resetDashboard();
    })
    .catch(err => console.error("Error clearing logs:", err));
}

// Reset Dashboard to empty state
function resetDashboard() {
    rawImageUrl = "";
    heatmapImageUrl = "";
    cameraFeed.src = "https://placehold.co/400x300/182232/00e5ff?text=No+Active+Scan";
    heatmapToggle.checked = false;
    heatmapToggle.disabled = true;

    detectedPlantName.textContent = "No Scan Yet";
    detectedScientificName.textContent = "";
    idConfidence.textContent = "--";
    freshnessValue.textContent = "--";
    freshnessBar.style.width = "0%";
    qualityGrade.textContent = "--";
    maturityLevel.textContent = "--";
    shelfLife.textContent = "--";
    qualityScore.textContent = "--";

    tempVal.textContent = "--";
    humidityVal.textContent = "--";
    ndviVal.textContent = "--";
    wbiVal.textContent = "--";

    xaiReport.innerHTML = "Waiting for scan to compile analysis report...";
    infoNutrition.innerHTML = "Perform a scan to query agricultural knowledge base.";
    infoStorage.innerHTML = "Perform a scan to query agricultural knowledge base.";
    infoSimilar.textContent = "--";
    infoEducation.textContent = "--";

    compoundsPlaceholder.classList.remove("hidden");
    compoundsList.classList.add("hidden");
    compoundsList.innerHTML = "";

    // Flat chart
    spectralChartInstance.data.datasets[0].data = Array(18).fill(0.1);
    spectralChartInstance.data.datasets[0].label = 'Reflectance (410nm - 940nm)';
    spectralChartInstance.update();
}
