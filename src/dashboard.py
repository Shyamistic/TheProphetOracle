"""Dashboard HTML template for the Prophet Forecasting Agent monitoring UI.

Serves a self-contained HTML page with inline CSS and JS (Chart.js from CDN)
that auto-refreshes every 30 seconds by polling /health, /costs, and /logs.
"""


def get_dashboard_html() -> str:
    """Return the full HTML string for the monitoring dashboard.

    The page is self-contained with:
    - Dark theme (#1a1a2e background) with card-based layout
    - Auto-refresh every 30 seconds via fetch()
    - Status indicators (green/red dots)
    - Prediction log table (last 20 entries)
    - Cost-over-time chart via Chart.js
    - Model success rate indicators
    - Responsive design
    """
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prophet Agent — Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
            background: #1a1a2e;
            color: #e0e0e0;
            min-height: 100vh;
            padding: 24px;
        }

        h1 {
            font-size: 1.8rem;
            font-weight: 600;
            color: #ffffff;
            margin-bottom: 8px;
        }

        .subtitle {
            color: #8888aa;
            font-size: 0.9rem;
            margin-bottom: 28px;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 28px;
        }

        .card {
            background: #16213e;
            border-radius: 12px;
            padding: 20px;
            border: 1px solid #0f3460;
            transition: border-color 0.2s;
        }

        .card:hover {
            border-color: #533483;
        }

        .card-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #8888aa;
            margin-bottom: 8px;
        }

        .card-value {
            font-size: 1.6rem;
            font-weight: 700;
            color: #ffffff;
        }

        .card-value.green { color: #00d97e; }
        .card-value.red { color: #e63946; }
        .card-value.amber { color: #f4a261; }

        .status-dot {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
            vertical-align: middle;
        }

        .dot-green { background: #00d97e; box-shadow: 0 0 6px #00d97e88; }
        .dot-red { background: #e63946; box-shadow: 0 0 6px #e6394688; }
        .dot-amber { background: #f4a261; box-shadow: 0 0 6px #f4a26188; }

        .section-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: #ffffff;
            margin-bottom: 14px;
            margin-top: 8px;
        }

        .table-wrap {
            background: #16213e;
            border-radius: 12px;
            border: 1px solid #0f3460;
            overflow-x: auto;
            margin-bottom: 28px;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.82rem;
        }

        th {
            background: #0f3460;
            color: #a0a0c0;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            padding: 12px 14px;
            text-align: left;
            white-space: nowrap;
        }

        td {
            padding: 10px 14px;
            border-top: 1px solid #0f346044;
            color: #c8c8e0;
            white-space: nowrap;
        }

        tr:hover td {
            background: #1a1a3e;
        }

        .chip {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.72rem;
            font-weight: 600;
            text-transform: uppercase;
        }

        .chip-economics { background: #264653; color: #2a9d8f; }
        .chip-sports { background: #3d2645; color: #e76f51; }
        .chip-entertainment { background: #3d3245; color: #f4a261; }
        .chip-geopolitics { background: #1d3557; color: #a8dadc; }
        .chip-technology { background: #2d3a4a; color: #48cae4; }
        .chip-science { background: #2a3a2a; color: #95d5b2; }
        .chip-general { background: #2a2a3a; color: #b8b8d0; }

        .chart-container {
            background: #16213e;
            border-radius: 12px;
            border: 1px solid #0f3460;
            padding: 20px;
            margin-bottom: 28px;
            max-height: 300px;
        }

        .models-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px;
            margin-bottom: 28px;
        }

        .model-card {
            background: #16213e;
            border-radius: 10px;
            padding: 14px 16px;
            border: 1px solid #0f3460;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .model-name {
            font-size: 0.8rem;
            color: #c8c8e0;
            font-weight: 500;
        }

        .model-rate {
            font-size: 0.75rem;
            color: #8888aa;
            margin-top: 2px;
        }

        .refresh-bar {
            position: fixed;
            top: 0;
            left: 0;
            height: 3px;
            background: linear-gradient(90deg, #533483, #00d97e);
            transition: width 0.3s ease;
            z-index: 1000;
        }

        .footer {
            text-align: center;
            color: #555577;
            font-size: 0.75rem;
            padding: 20px 0;
        }

        @media (max-width: 768px) {
            body { padding: 12px; }
            .grid { grid-template-columns: 1fr 1fr; }
            h1 { font-size: 1.4rem; }
        }
    </style>
</head>
<body>
    <div class="refresh-bar" id="refreshBar"></div>

    <h1>&#x1F52E; Prophet Forecasting Agent</h1>
    <p class="subtitle">Real-time monitoring dashboard &mdash; auto-refreshes every 30s</p>

    <!-- Status Cards -->
    <div class="grid" id="statusGrid">
        <div class="card">
            <div class="card-label">Server Status</div>
            <div class="card-value" id="serverStatus">
                <span class="status-dot dot-green"></span>Loading...
            </div>
        </div>
        <div class="card">
            <div class="card-label">Uptime</div>
            <div class="card-value" id="uptime">—</div>
        </div>
        <div class="card">
            <div class="card-label">Total Predictions</div>
            <div class="card-value" id="totalPredictions">0</div>
        </div>
        <div class="card">
            <div class="card-label">Total Spend</div>
            <div class="card-value" id="totalSpend">$0.00</div>
        </div>
        <div class="card">
            <div class="card-label">Budget Remaining</div>
            <div class="card-value green" id="budgetRemaining">$0.00</div>
        </div>
        <div class="card">
            <div class="card-label">Budget Status</div>
            <div class="card-value" id="budgetStatus">
                <span class="status-dot dot-green"></span>OK
            </div>
        </div>
    </div>

    <!-- Model Success Rates -->
    <div class="section-title">Model Success Rates</div>
    <div class="models-grid" id="modelsGrid">
        <div class="model-card">
            <span class="status-dot dot-green"></span>
            <div>
                <div class="model-name">Loading models...</div>
                <div class="model-rate">—</div>
            </div>
        </div>
    </div>

    <!-- Cost Chart -->
    <div class="section-title">Cost Over Time</div>
    <div class="chart-container">
        <canvas id="costChart"></canvas>
    </div>

    <!-- Prediction Log -->
    <div class="section-title">Recent Predictions (Last 20)</div>
    <div class="table-wrap">
        <table>
            <thead>
                <tr>
                    <th>Time</th>
                    <th>Event</th>
                    <th>Category</th>
                    <th>Outcomes</th>
                    <th>Probabilities</th>
                    <th>Duration</th>
                    <th>Disagreement</th>
                </tr>
            </thead>
            <tbody id="logBody">
                <tr><td colspan="7" style="text-align:center;color:#555577;">No predictions yet</td></tr>
            </tbody>
        </table>
    </div>

    <div class="footer">
        Prophet Forecasting Agent &mdash; Dashboard v1.0 &mdash; Last refresh: <span id="lastRefresh">—</span>
    </div>

    <script>
        const startTime = Date.now();
        let costHistory = [];
        let costChart = null;

        // Initialize Chart.js
        function initChart() {
            const ctx = document.getElementById('costChart').getContext('2d');
            costChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Cumulative Spend ($)',
                        data: [],
                        borderColor: '#533483',
                        backgroundColor: 'rgba(83, 52, 131, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.3,
                        pointRadius: 3,
                        pointBackgroundColor: '#533483',
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            labels: { color: '#8888aa', font: { size: 11 } }
                        }
                    },
                    scales: {
                        x: {
                            ticks: { color: '#666688', font: { size: 10 } },
                            grid: { color: '#0f346033' }
                        },
                        y: {
                            ticks: { color: '#666688', font: { size: 10 }, callback: v => '$' + v.toFixed(3) },
                            grid: { color: '#0f346033' }
                        }
                    }
                }
            });
        }

        function formatUptime() {
            const elapsed = Math.floor((Date.now() - startTime) / 1000);
            const h = Math.floor(elapsed / 3600);
            const m = Math.floor((elapsed % 3600) / 60);
            const s = elapsed % 60;
            if (h > 0) return `${h}h ${m}m ${s}s`;
            if (m > 0) return `${m}m ${s}s`;
            return `${s}s`;
        }

        function getCategoryChip(category) {
            const cat = (category || 'general').toLowerCase();
            return `<span class="chip chip-${cat}">${cat}</span>`;
        }

        function formatTime(isoStr) {
            try {
                const d = new Date(isoStr);
                return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
            } catch { return '—'; }
        }

        function formatProbs(probs) {
            if (!probs) return '—';
            return Object.entries(probs)
                .map(([k, v]) => `${k.slice(0, 12)}: ${(v * 100).toFixed(1)}%`)
                .join(', ');
        }

        async function fetchHealth() {
            try {
                const res = await fetch('/health');
                const data = await res.json();

                const statusEl = document.getElementById('serverStatus');
                const checks = data.checks || {};
                const isHealthy = data.status === 'healthy';
                const isDegraded = data.status === 'degraded';

                if (isHealthy) {
                    statusEl.innerHTML = '<span class="status-dot dot-green"></span>Healthy';
                } else if (isDegraded) {
                    statusEl.innerHTML = '<span class="status-dot dot-amber"></span>Degraded';
                } else {
                    statusEl.innerHTML = '<span class="status-dot dot-red"></span>Unhealthy';
                }

                // Budget from health
                if (data.budget) {
                    const budgetStatusEl = document.getElementById('budgetStatus');
                    if (data.budget.is_budget_critical) {
                        budgetStatusEl.innerHTML = '<span class="status-dot dot-red"></span>CRITICAL';
                        budgetStatusEl.className = 'card-value red';
                    } else {
                        budgetStatusEl.innerHTML = '<span class="status-dot dot-green"></span>OK';
                        budgetStatusEl.className = 'card-value green';
                    }
                }

                // Update model cards from health checks
                updateModels(checks);

            } catch (e) {
                document.getElementById('serverStatus').innerHTML =
                    '<span class="status-dot dot-red"></span>Offline';
            }
        }

        function updateModels(checks) {
            const grid = document.getElementById('modelsGrid');
            const models = [
                { name: 'LLM API (OpenRouter)', key: 'llm_api' },
                { name: 'Search API (Tavily)', key: 'search_api' },
            ];

            grid.innerHTML = models.map(m => {
                const status = checks[m.key] || 'unknown';
                const isOk = status === 'ok';
                const dotClass = isOk ? 'dot-green' : (status.includes('degraded') ? 'dot-amber' : 'dot-red');
                return `
                    <div class="model-card">
                        <span class="status-dot ${dotClass}"></span>
                        <div>
                            <div class="model-name">${m.name}</div>
                            <div class="model-rate">${isOk ? 'Operational' : status}</div>
                        </div>
                    </div>
                `;
            }).join('');
        }

        async function fetchCosts() {
            try {
                const res = await fetch('/costs');
                const data = await res.json();

                const spend = data.total_spend_usd || 0;
                const remaining = data.budget_remaining_usd || 0;

                document.getElementById('totalSpend').textContent = '$' + spend.toFixed(4);

                const remEl = document.getElementById('budgetRemaining');
                remEl.textContent = '$' + remaining.toFixed(4);
                if (remaining < 0.5) {
                    remEl.className = 'card-value red';
                } else if (remaining < 2.0) {
                    remEl.className = 'card-value amber';
                } else {
                    remEl.className = 'card-value green';
                }

                // Update chart
                const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                costHistory.push({ time: now, spend: spend });
                if (costHistory.length > 30) costHistory.shift();

                if (costChart) {
                    costChart.data.labels = costHistory.map(p => p.time);
                    costChart.data.datasets[0].data = costHistory.map(p => p.spend);
                    costChart.update('none');
                }

            } catch (e) {
                console.error('Failed to fetch costs:', e);
            }
        }

        async function fetchLogs() {
            try {
                const res = await fetch('/logs');
                const data = await res.json();

                document.getElementById('totalPredictions').textContent = data.total || 0;

                const predictions = data.predictions || [];
                const tbody = document.getElementById('logBody');

                if (predictions.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;color:#555577;">No predictions yet</td></tr>';
                    return;
                }

                // Show last 20, most recent first
                const recent = predictions.slice(-20).reverse();
                tbody.innerHTML = recent.map(p => `
                    <tr>
                        <td>${formatTime(p.timestamp)}</td>
                        <td title="${p.event_ticker || ''}">${(p.title || p.event_ticker || '—').slice(0, 45)}</td>
                        <td>${getCategoryChip(p.category)}</td>
                        <td>${(p.outcomes || []).slice(0, 3).join(', ')}</td>
                        <td>${formatProbs(p.probabilities)}</td>
                        <td>${p.duration ? p.duration + 's' : '—'}</td>
                        <td>${p.had_disagreement ? '<span class="status-dot dot-amber"></span>Yes' : '<span class="status-dot dot-green"></span>No'}</td>
                    </tr>
                `).join('');

            } catch (e) {
                console.error('Failed to fetch logs:', e);
            }
        }

        async function refreshAll() {
            const bar = document.getElementById('refreshBar');
            bar.style.width = '30%';

            await fetchHealth();
            bar.style.width = '60%';

            await fetchCosts();
            bar.style.width = '80%';

            await fetchLogs();
            bar.style.width = '100%';

            document.getElementById('uptime').textContent = formatUptime();
            document.getElementById('lastRefresh').textContent = new Date().toLocaleTimeString();

            setTimeout(() => { bar.style.width = '0%'; }, 500);
        }

        // Init
        initChart();
        refreshAll();
        setInterval(refreshAll, 30000);
    </script>
</body>
</html>"""
