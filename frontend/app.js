/**
 * Smart Campus Vehicle Management System
 * Dashboard Application Logic - HTTP Polling Version
 */

// Configuration
const API_BASE = window.location.origin;
const POLL_INTERVAL = 1000; // Poll stats every 1 second

// State
let isConnected = false;
let pollTimer = null;

// Vehicle Icons
const VEHICLE_ICONS = {
    car: '🚗',
    motorcycle: '🏍️',
    bus: '🚌',
    truck: '🚚'
};

// Chart Colors
const CHART_COLORS = {
    car: { bg: 'rgba(59, 130, 246, 0.8)', border: 'rgb(59, 130, 246)' },
    motorcycle: { bg: 'rgba(16, 185, 129, 0.8)', border: 'rgb(16, 185, 129)' },
    bus: { bg: 'rgba(245, 158, 11, 0.8)', border: 'rgb(245, 158, 11)' },
    truck: { bg: 'rgba(139, 92, 246, 0.8)', border: 'rgb(139, 92, 246)' }
};

// Charts
let pieChart = null;
let barChart = null;

// DOM Elements
const elements = {
    connectionStatus: document.getElementById('connectionStatus'),
    currentTime: document.getElementById('currentTime'),
    videoFeed: document.getElementById('videoFeed'),
    videoOverlay: document.getElementById('videoOverlay'),
    totalIn: document.getElementById('totalIn'),
    totalOut: document.getElementById('totalOut'),
    onCampus: document.getElementById('onCampus'),
    activityLog: document.getElementById('activityLog'),
    resetBtn: document.getElementById('resetBtn'),
    pieChart: document.getElementById('pieChart'),
    barChart: document.getElementById('barChart')
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    initClock();
    initEventListeners();
    startPolling();
});

// Initialize Charts
function initCharts() {
    const commonOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'bottom',
                labels: {
                    padding: 15,
                    usePointStyle: true,
                    font: { size: 11 }
                }
            }
        }
    };

    // Pie Chart
    pieChart = new Chart(elements.pieChart, {
        type: 'doughnut',
        data: {
            labels: ['Cars', 'Motorcycles', 'Buses', 'Trucks'],
            datasets: [{
                data: [0, 0, 0, 0],
                backgroundColor: [
                    CHART_COLORS.car.bg,
                    CHART_COLORS.motorcycle.bg,
                    CHART_COLORS.bus.bg,
                    CHART_COLORS.truck.bg
                ],
                borderColor: [
                    CHART_COLORS.car.border,
                    CHART_COLORS.motorcycle.border,
                    CHART_COLORS.bus.border,
                    CHART_COLORS.truck.border
                ],
                borderWidth: 2
            }]
        },
        options: {
            ...commonOptions,
            cutout: '60%',
            plugins: {
                ...commonOptions.plugins,
                tooltip: {
                    callbacks: {
                        label: (context) => {
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = total > 0 ? ((context.raw / total) * 100).toFixed(1) : 0;
                            return `${context.label}: ${context.raw} (${percentage}%)`;
                        }
                    }
                }
            }
        }
    });

    // Bar Chart
    barChart = new Chart(elements.barChart, {
        type: 'bar',
        data: {
            labels: ['Cars', 'Motorcycles', 'Buses', 'Trucks'],
            datasets: [
                {
                    label: 'Entry',
                    data: [0, 0, 0, 0],
                    backgroundColor: 'rgba(16, 185, 129, 0.8)',
                    borderColor: 'rgb(16, 185, 129)',
                    borderWidth: 1,
                    borderRadius: 4
                },
                {
                    label: 'Exit',
                    data: [0, 0, 0, 0],
                    backgroundColor: 'rgba(239, 68, 68, 0.8)',
                    borderColor: 'rgb(239, 68, 68)',
                    borderWidth: 1,
                    borderRadius: 4
                }
            ]
        },
        options: {
            ...commonOptions,
            scales: {
                x: { grid: { display: false } },
                y: { beginAtZero: true, ticks: { stepSize: 1 } }
            }
        }
    });
}

// Start polling for stats
function startPolling() {
    fetchStats();
    pollTimer = setInterval(fetchStats, POLL_INTERVAL);
}

// Fetch stats from server
async function fetchStats() {
    try {
        const response = await fetch(`${API_BASE}/api/stats`);
        if (response.ok) {
            const stats = await response.json();
            updateStats(stats);
            if (!isConnected) {
                isConnected = true;
                updateConnectionStatus('connected', 'Connected');
            }
        } else {
            throw new Error('Failed to fetch stats');
        }
    } catch (error) {
        console.error('Polling error:', error);
        if (isConnected) {
            isConnected = false;
            updateConnectionStatus('disconnected', 'Disconnected');
        }
    }
}

// Update connection status
function updateConnectionStatus(status, text) {
    elements.connectionStatus.className = 'status-indicator ' + status;
    elements.connectionStatus.querySelector('.status-text').textContent = text;
}

// Update statistics
function updateStats(stats) {
    // Update counters
    animateCounter(elements.totalIn, stats.total_in || 0);
    animateCounter(elements.totalOut, stats.total_out || 0);
    animateCounter(elements.onCampus, stats.on_campus || 0);

    // Update charts
    if (stats.by_type) {
        const byType = stats.by_type;

        // Pie chart - total vehicles by type
        const totals = [
            (byType.car?.in || 0) + (byType.car?.out || 0),
            (byType.motorcycle?.in || 0) + (byType.motorcycle?.out || 0),
            (byType.bus?.in || 0) + (byType.bus?.out || 0),
            (byType.truck?.in || 0) + (byType.truck?.out || 0)
        ];
        pieChart.data.datasets[0].data = totals;
        pieChart.update('none');

        // Bar chart
        barChart.data.datasets[0].data = [
            byType.car?.in || 0,
            byType.motorcycle?.in || 0,
            byType.bus?.in || 0,
            byType.truck?.in || 0
        ];
        barChart.data.datasets[1].data = [
            byType.car?.out || 0,
            byType.motorcycle?.out || 0,
            byType.bus?.out || 0,
            byType.truck?.out || 0
        ];
        barChart.update('none');
    }

    // Update activity log
    if (stats.recent_events && stats.recent_events.length > 0) {
        updateActivityLog(stats.recent_events);
    }
}

// Animate counter value
function animateCounter(element, newValue) {
    const currentValue = parseInt(element.textContent) || 0;
    if (currentValue !== newValue) {
        element.textContent = newValue;
        element.style.transform = 'scale(1.1)';
        setTimeout(() => {
            element.style.transform = 'scale(1)';
        }, 200);
    }
}

// Update activity log
function updateActivityLog(events) {
    if (events.length === 0) {
        const emptyHtml = '<div class="activity-empty">No activity yet</div>';
        if (elements.activityLog.innerHTML !== emptyHtml) {
            elements.activityLog.innerHTML = emptyHtml;
        }
        return;
    }

    const html = events.slice(0, 10).map(event => {
        const icon = VEHICLE_ICONS[event.type] || '🚗';
        const directionClass = event.direction === 'ENTRY' ? 'entry' : 'exit';
        const directionText = event.direction === 'ENTRY' ? '➡️ Entered' : '⬅️ Exited';

        return `
            <div class="activity-item ${directionClass}">
                <span class="activity-icon">${icon}</span>
                <div class="activity-details">
                    <span class="activity-type">${capitalize(event.type)}</span>
                    <span class="activity-direction">${directionText}</span>
                </div>
                <span class="activity-time">${event.time}</span>
            </div>
        `;
    }).join('');

    // Prevent blinking: Only update if content changed
    if (elements.activityLog.innerHTML !== html) {
        elements.activityLog.innerHTML = html;
    }
}

// Initialize clock
function initClock() {
    function updateClock() {
        const now = new Date();
        elements.currentTime.textContent = now.toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            hour12: true
        });
    }
    updateClock();
    setInterval(updateClock, 1000);
}

// Initialize event listeners
function initEventListeners() {
    elements.resetBtn.addEventListener('click', () => {
        if (confirm('Are you sure you want to reset all counts?')) {
            fetch('/api/reset', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    console.log('Reset:', data);
                    elements.totalIn.textContent = '0';
                    elements.totalOut.textContent = '0';
                    elements.onCampus.textContent = '0';
                    pieChart.data.datasets[0].data = [0, 0, 0, 0];
                    pieChart.update();
                    barChart.data.datasets[0].data = [0, 0, 0, 0];
                    barChart.data.datasets[1].data = [0, 0, 0, 0];
                    barChart.update();
                    elements.activityLog.innerHTML = '<div class="activity-empty">No activity yet</div>';
                })
                .catch(error => console.error('Reset error:', error));
        }
    });
}

// Utility: Capitalize first letter
function capitalize(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}
