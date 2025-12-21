/**
 * ChatREL Report Visualization with Chart.js
 * Renders interactive charts from backend report payload
 */

// Parse report data from embedded JSON
const reportData = JSON.parse(document.getElementById('report-data').textContent);

// Make available globally for other scripts (report_chat.js)
window.reportData = reportData;

// Color palettes
const COLORS = {
    primary: ['#667eea', '#764ba2', '#f093fb', '#4facfe'],
    romantic: '#ec4899',
    positive: '#10b981',
    playful: '#f59e0b',
    neutral: '#6b7280',
    negative: '#ef4444',
};

/**
 * Build and render a chart from configuration
 * @param {string} canvasId - Canvas element ID
 * @param {object} config - Chart configuration from backend
 */
function buildChartFromConfig(canvasId, config) {
    if (!config) {
        hideChartContainer(canvasId);
        return;
    }

    const canvas = document.getElementById(canvasId);
    if (!canvas) {
        console.warn(`Canvas not found: ${canvasId}`);
        return;
    }

    const ctx = canvas.getContext('2d');

    // Build Chart.js configuration based on type
    let chartConfig;

    if (config.type === 'line') {
        chartConfig = {
            type: 'line',
            data: {
                labels: config.labels,
                datasets: config.datasets.map((ds, i) => ({
                    label: ds.label,
                    data: ds.data,
                    borderColor: COLORS.primary[i % COLORS.primary.length],
                    backgroundColor: COLORS.primary[i % COLORS.primary.length] + '20',
                    borderWidth: 2,
                    tension: 0.3,
                    fill: true,
                }))
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                aspectRatio: 2,
                plugins: {
                    legend: { display: config.datasets.length > 1 }
                },
                scales: {
                    y: { beginAtZero: true }
                }
            }
        };
    }
    else if (config.type === 'bar') {
        chartConfig = {
            type: 'bar',
            data: {
                labels: config.labels,
                datasets: config.datasets.map((ds, i) => ({
                    label: ds.label,
                    data: ds.data,
                    backgroundColor: COLORS.primary[i % COLORS.primary.length],
                    borderWidth: 0,
                }))
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                aspectRatio: 1.5,
                plugins: {
                    legend: { display: config.datasets.length > 1 }
                },
                scales: {
                    y: { beginAtZero: true }
                }
            }
        };
    }
    else if (config.type === 'stacked_bar') {
        const categoryColors = {
            'Romantic': COLORS.romantic,
            'Positive': COLORS.positive,
            'Playful': COLORS.playful,
            'Neutral': COLORS.neutral,
            'Negative': COLORS.negative,
        };

        chartConfig = {
            type: 'bar',
            data: {
                labels: config.labels,
                datasets: config.datasets.map(ds => ({
                    label: ds.label,
                    data: ds.data,
                    backgroundColor: categoryColors[ds.label] || COLORS.primary[0],
                    borderWidth: 0,
                }))
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                aspectRatio: 1.5,
                plugins: {
                    legend: { position: 'top' }
                },
                scales: {
                    x: { stacked: true },
                    y: { stacked: true, beginAtZero: true }
                }
            }
        };
    }
    else if (config.type === 'pie') {
        chartConfig = {
            type: 'pie',
            data: {
                labels: config.labels,
                datasets: [{
                    data: config.data,
                    backgroundColor: COLORS.primary,
                    borderWidth: 2,
                    borderColor: '#fff',
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                aspectRatio: 1.5,
            }
        };
    }
    else {
        console.warn(`Unknown chart type: ${config.type}`);
        hideChartContainer(canvasId);
        return;
    }

    // Render chart
    new Chart(ctx, chartConfig);

    // Dispatch custom event for chart ready (enables fade-in animation)
    setTimeout(() => {
        const event = new CustomEvent('chartRendered', { detail: { canvasId } });
        document.dispatchEvent(event);
    }, 100);
}

/**
 * Hide chart container if no data available
 */
function hideChartContainer(canvasId) {
    const container = document.getElementById(canvasId + '-container');
    if (container) {
        container.style.display = 'none';
    }
}

/**
 * Initialize all charts from report data
 */
function initializeCharts() {
    const charts = reportData.charts;

    // Render each chart
    buildChartFromConfig('chart-messages-over-time', charts.messages_over_time);
    buildChartFromConfig('chart-messages-by-sender', charts.messages_by_sender);
    buildChartFromConfig('chart-words-by-sender', charts.words_by_sender);
    buildChartFromConfig('chart-response-time', charts.response_time_by_sender);
    buildChartFromConfig('chart-emoji-by-sender', charts.emoji_by_sender);
    buildChartFromConfig('chart-emoji-categories', charts.emoji_categories);

    // NLP charts (only if available)
    if (charts.sentiment_over_time) {
        buildChartFromConfig('chart-sentiment-over-time', charts.sentiment_over_time);
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeCharts);
} else {
    initializeCharts();
}
