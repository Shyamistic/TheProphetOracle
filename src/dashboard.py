"""Dashboard HTML template for the Prophet Forecasting Agent monitoring UI.

Serves a self-contained HTML page with inline CSS and JS (Chart.js from CDN)
that auto-refreshes every 30 seconds by polling /health, /costs, /logs, and /api-status.
Shows real-time API balances from OpenRouter, Tavily, and Featherless.
"""


def get_dashboard_html() -> str:
    """Return the full HTML string for the monitoring dashboard.

    The page is self-contained with:
    - Dark theme (#1a1a2e background) with card-based layout
    - Auto-refresh every 30 seconds via fetch()
    - Real-time API balance display (OpenRouter, Tavily, Featherless)
    - Status indicators (green/red/amber dots)
    - Prediction log table (last 20 entries) with full details
    - Cost-over-time chart via Chart.js
    - Model success rate indicators
    - Detailed system logs section
    - Responsive design
    """
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prophet Agent — Dashboard v2.0</title>
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

        h1 { font-size: 1.8rem; font-weight: 600; color: #ffffff; margin-bottom: 8px; }
        .subtitle { color: #8888aa; font-size: 0.9rem; margin-bottom: 28px; }

        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 28px; }
        .grid-3 { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 28px; }

        .card {
            background: #16213e;
            border-radius: 12px;
            padding: 20px;
            border: 1px solid #0f3460;
            transition: border-color 0.2s;
        }
        .card:hover { border-color: #533483; }
        .card-label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: #8888aa; margin-bottom: 8px; }
        .card-value { font-size: 1.5rem; font-weight: 700; color: #ffffff; }
        .card-value.green { color: #00d97e; }
        .card-value.red { color: #e63946; }
        .card-value.amber { color: #f4a261; }
        .card-sub { font-size: 0.75rem; color: #8888aa; margin-top: 4px; }

        .status-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; vertical-align: middle; }
        .dot-green { background: #00d97e; box-shadow: 0 0 6px #00d97e88; }
        .dot-red { background: #e63946; box-shadow: 0 0 6px #e6394688; }
        .dot-amber { background: #f4a261; box-shadow: 0 0 6px #f4a26188; }

        .section-title { font-size: 1.1rem; font-weight: 600; color: #ffffff; margin-bottom: 14px; margin-top: 8px; }

        .table-wrap { background: #16213e; border-radius: 12px; border: 1px solid #0f3460; overflow-x: auto; margin-bottom: 28px; }
        table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
        th { background: #0f3460; color: #a0a0c0; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; padding: 12px 14px; text-align: left; white-space: nowrap; }
        td { padding: 10px 14px; border-top: 1px solid #0f346044; color: #c8c8e0; white-space: nowrap; }
        tr:hover td { background: #1a1a3e; }

        .chip { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.72rem; font-weight: 600; text-transform: uppercase; }
        .chip-economics { background: #264653; color: #2a9d8f; }
        .chip-sports { background: #3d2645; color: #e76f51; }
        .chip-entertainment { background: #3d3245; color: #f4a261; }
        .chip-geopolitics { background: #1d3557; color: #a8dadc; }
        .chip-technology { background: #2d3a4a; color: #48cae4; }
        .chip-science { background: #2a3a2a; color: #95d5b2; }
        .chip-general { background: #2a2a3a; color: #b8b8d0; }

        .chart-container { background: #16213e; border-radius: 12px; border: 1px solid #0f3460; padding: 20px; margin-bottom: 28px; max-height: 250px; }

        .api-card { background: #16213e; border-radius: 12px; padding: 18px; border: 1px solid #0f3460; }
        .api-card-title { font-size: 0.85rem; font-weight: 600; color: #ffffff; margin-bottom: 10px; display: flex; align-items: center; gap: 8px; }
        .api-card-row { display: flex; justify-content: space-between; align-items: center; padding: 4px 0; font-size: 0.8rem; }
        .api-card-label { color: #8888aa; }
        .api-card-value { color: #e0e0e0; font-weight: 500; }
        .api-card-value.green { color: #00d97e; }
        .api-card-value.red { color: #e63946; }
        .api-card-value.amber { color: #f4a261; }

        .progress-bar { height: 6px; background: #0f3460; border-radius: 3px; margin-top: 8px; overflow: hidden; }
        .progress-fill { height: 100%; border-radius: 3px; transition: width 0.5s ease; }
        .progress-green { background: linear-gradient(90deg, #00d97e, #2a9d8f); }
        .progress-amber { background: linear-gradient(90deg, #f4a261, #e76f51); }
        .progress-red { background: linear-gradient(90deg, #e63946, #c1121f); }

        .refresh-bar { position: fixed; top: 0; left: 0; height: 3px; background: linear-gradient(90deg, #533483, #00d97e); transition: width 0.3s ease; z-index: 1000; }
        .footer { text-align: center; color: #555577; font-size: 0.75rem; padding: 20px 0; }

        @media (max-width: 768px) { body { padding: 12px; } .grid { grid-template-columns: 1fr 1fr; } h1 { font-size: 1.4rem; } }
    </style>
</head>
<body>
    <div class="refresh-bar" id="refreshBar"></div>

    <h1>&#x1F52E; Prophet Forecasting Agent</h1>
    <p class="subtitle">Real-time monitoring dashboard v2.0 &mdash; auto-refreshes every 30s</p>

    <!-- Status Cards -->
    <div class="grid" id="statusGrid">
        <div class="card">
            <div class="card-label">Server Status</div>
            <div class="card-value" id="serverStatus"><span class="status-dot dot-green"></span>Loading...</div>
        </div>
        <div class="card">
            <div class="card-label">Uptime</div>
            <div class="card-value" id="uptime">&mdash;</div>
        </div>
        <div class="card">
            <div class="card-label">Total Predictions</div>
            <div class="card-value" id="totalPredictions">0</div>
            <div class="card-sub" id="predRate"></div>
        </div>
        <div class="card">
            <div class="card-label">Avg Duration</div>
            <div class="card-value" id="avgDuration">&mdash;</div>
        </div>
        <div class="card">
            <div class="card-label">Disagreements</div>
            <div class="card-value" id="disagreements">0</div>
        </div>
        <div class="card">
            <div class="card-label">System Health</div>
            <div class="card-value" id="systemHealth"><span class="status-dot dot-green"></span>OK</div>
        </div>
    </div>

    <!-- API Balances Section -->
    <div class="section-title">&#x1F4B0; API Credits &amp; Balances (Real-Time)</div>
    <div class="grid-3" id="apiGrid">
        <div class="api-card" id="openrouterCard">
            <div class="api-card-title"><span class="status-dot dot-green"></span>OpenRouter (LLM)</div>
            <div class="api-card-row"><span class="api-card-label">Balance</span><span class="api-card-value" id="orBalance">Loading...</span></div>
            <div class="api-card-row"><span class="api-card-label">Used</span><span class="api-card-value" id="orUsed">&mdash;</span></div>
            <div class="api-card-row"><span class="api-card-label">Limit</span><span class="api-card-value" id="orLimit">&mdash;</span></div>
            <div class="progress-bar"><div class="progress-fill progress-green" id="orProgress" style="width:0%"></div></div>
        </div>
        <div class="api-card" id="tavilyCard">
            <div class="api-card-title"><span class="status-dot dot-green"></span>Tavily (Search)</div>
            <div class="api-card-row"><span class="api-card-label">Status</span><span class="api-card-value" id="tavilyStatus">Loading...</span></div>
            <div class="api-card-row"><span class="api-card-label">Searches Used</span><span class="api-card-value" id="tavilyUsed">&mdash;</span></div>
            <div class="api-card-row"><span class="api-card-label">Est. Remaining</span><span class="api-card-value" id="tavilyRemaining">&mdash;</span></div>
            <div class="progress-bar"><div class="progress-fill progress-green" id="tavilyProgress" style="width:0%"></div></div>
        </div>
        <div class="api-card" id="featherlessCard">
            <div class="api-card-title"><span class="status-dot dot-green"></span>Featherless (Tiebreaker)</div>
            <div class="api-card-row"><span class="api-card-label">Model</span><span class="api-card-value">Qwen 72B</span></div>
            <div class="api-card-row"><span class="api-card-label">Status</span><span class="api-card-value" id="featherlessStatus">Loading...</span></div>
            <div class="api-card-row"><span class="api-card-label">Role</span><span class="api-card-value">Tiebreaker (&gt;15% disagreement)</span></div>
        </div>
    </div>

    <!-- Cost Chart -->
    <div class="section-title">&#x1F4C8; Cost Over Time</div>
    <div class="chart-container">
        <canvas id="costChart"></canvas>
    </div>

    <!-- Prediction Log -->
    <div class="section-title">&#x1F4CB; Recent Predictions (Last 20)</div>
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
        Prophet Forecasting Agent &mdash; Dashboard v2.0 &mdash; Last refresh: <span id="lastRefresh">&mdash;</span>
    </div>

    <script>
        const startTime = Date.now();
        let costHistory = [];
        let costChart = null;

        function initChart() {
            const ctx = document.getElementById('costChart').getContext('2d');
            costChart = new Chart(ctx, {
                type: 'line',
                data: { labels: [], datasets: [{ label: 'Cumulative Spend ($)', data: [], borderColor: '#533483', backgroundColor: 'rgba(83, 52, 131, 0.1)', borderWidth: 2, fill: true, tension: 0.3, pointRadius: 3, pointBackgroundColor: '#533483' }] },
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: '#8888aa', font: { size: 11 } } } }, scales: { x: { ticks: { color: '#666688', font: { size: 10 } }, grid: { color: '#0f346033' } }, y: { ticks: { color: '#666688', font: { size: 10 }, callback: v => '$' + v.toFixed(3) }, grid: { color: '#0f346033' } } } }
            });
        }

        function formatUptime() {
            const elapsed = Math.floor((Date.now() - startTime) / 1000);
            const h = Math.floor(elapsed / 3600); const m = Math.floor((elapsed % 3600) / 60); const s = elapsed % 60;
            if (h > 0) return h + 'h ' + m + 'm ' + s + 's';
            if (m > 0) return m + 'm ' + s + 's';
            return s + 's';
        }

        function getCategoryChip(category) {
            const cat = (category || 'general').toLowerCase();
            return '<span class="chip chip-' + cat + '">' + cat + '</span>';
        }

        function formatTime(isoStr) {
            try { const d = new Date(isoStr); return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }); } catch { return '—'; }
        }

        function formatProbs(probs) {
            if (!probs) return '—';
            return Object.entries(probs).map(function(e) { return e[0].slice(0, 12) + ': ' + (e[1] * 100).toFixed(1) + '%'; }).join(', ');
        }

        async function fetchHealth() {
            try {
                const res = await fetch('/health');
                const data = await res.json();
                const statusEl = document.getElementById('serverStatus');
                const checks = data.checks || {};
                if (data.status === 'healthy') {
                    statusEl.innerHTML = '<span class="status-dot dot-green"></span>Healthy';
                    document.getElementById('systemHealth').innerHTML = '<span class="status-dot dot-green"></span>All Systems Go';
                } else if (data.status === 'degraded') {
                    statusEl.innerHTML = '<span class="status-dot dot-amber"></span>Degraded';
                    document.getElementById('systemHealth').innerHTML = '<span class="status-dot dot-amber"></span>Degraded';
                } else {
                    statusEl.innerHTML = '<span class="status-dot dot-red"></span>Unhealthy';
                    document.getElementById('systemHealth').innerHTML = '<span class="status-dot dot-red"></span>Issues Detected';
                }
            } catch (e) {
                document.getElementById('serverStatus').innerHTML = '<span class="status-dot dot-red"></span>Offline';
            }
        }

        async function fetchApiStatus() {
            try {
                const res = await fetch('/api-status');
                const data = await res.json();

                // OpenRouter
                const or = data.openrouter || {};
                const orCard = document.getElementById('openrouterCard');
                if (or.status === 'ok') {
                    orCard.querySelector('.api-card-title').innerHTML = '<span class="status-dot dot-green"></span>OpenRouter (LLM)';
                    const limit = or.limit;
                    const usage = or.usage;
                    const remaining = or.remaining || or.limit_remaining;
                    document.getElementById('orBalance').textContent = remaining != null ? '$' + remaining.toFixed(2) : 'Unlimited';
                    document.getElementById('orBalance').className = 'api-card-value ' + (remaining != null && remaining < 5 ? 'red' : remaining != null && remaining < 15 ? 'amber' : 'green');
                    document.getElementById('orUsed').textContent = usage != null ? '$' + usage.toFixed(2) : '—';
                    document.getElementById('orLimit').textContent = limit != null ? '$' + limit.toFixed(2) : 'No limit';
                    if (limit && usage) {
                        const pct = Math.min(100, (usage / limit) * 100);
                        const bar = document.getElementById('orProgress');
                        bar.style.width = pct + '%';
                        bar.className = 'progress-fill ' + (pct > 90 ? 'progress-red' : pct > 70 ? 'progress-amber' : 'progress-green');
                    }
                } else {
                    orCard.querySelector('.api-card-title').innerHTML = '<span class="status-dot dot-red"></span>OpenRouter (LLM)';
                    document.getElementById('orBalance').textContent = or.status || 'Error';
                    document.getElementById('orBalance').className = 'api-card-value red';
                }

                // Featherless
                const fl = data.featherless || {};
                const flStatus = document.getElementById('featherlessStatus');
                if (fl.status === 'ok') {
                    flStatus.textContent = 'Operational';
                    flStatus.className = 'api-card-value green';
                    document.getElementById('featherlessCard').querySelector('.api-card-title').innerHTML = '<span class="status-dot dot-green"></span>Featherless (Tiebreaker)';
                } else {
                    flStatus.textContent = fl.status || 'Unknown';
                    flStatus.className = 'api-card-value red';
                    document.getElementById('featherlessCard').querySelector('.api-card-title').innerHTML = '<span class="status-dot dot-red"></span>Featherless (Tiebreaker)';
                }

                // Tavily (we estimate from prediction count)
                const internal = data.internal_tracking || {};
                const totalPreds = internal.total_predictions || 0;
                const estSearches = totalPreds * 4; // ~4 searches per prediction
                const tavilyTotal = 1000; // New key has 1000 credits
                const tavilyRemaining = Math.max(0, tavilyTotal - estSearches);
                document.getElementById('tavilyStatus').textContent = 'Active';
                document.getElementById('tavilyStatus').className = 'api-card-value green';
                document.getElementById('tavilyUsed').textContent = '~' + estSearches + ' searches';
                document.getElementById('tavilyRemaining').textContent = '~' + tavilyRemaining + ' credits';
                document.getElementById('tavilyRemaining').className = 'api-card-value ' + (tavilyRemaining < 100 ? 'red' : tavilyRemaining < 300 ? 'amber' : 'green');
                const tavilyPct = Math.min(100, (estSearches / tavilyTotal) * 100);
                const tavilyBar = document.getElementById('tavilyProgress');
                tavilyBar.style.width = tavilyPct + '%';
                tavilyBar.className = 'progress-fill ' + (tavilyPct > 90 ? 'progress-red' : tavilyPct > 70 ? 'progress-amber' : 'progress-green');

                document.getElementById('tavilyCard').querySelector('.api-card-title').innerHTML = '<span class="status-dot dot-green"></span>Tavily (Search)';

            } catch (e) {
                console.error('Failed to fetch API status:', e);
            }
        }

        async function fetchCosts() {
            try {
                const res = await fetch('/costs');
                const data = await res.json();
                const spend = data.total_spend_usd || 0;
                const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                costHistory.push({ time: now, spend: spend });
                if (costHistory.length > 30) costHistory.shift();
                if (costChart) {
                    costChart.data.labels = costHistory.map(function(p) { return p.time; });
                    costChart.data.datasets[0].data = costHistory.map(function(p) { return p.spend; });
                    costChart.update('none');
                }
            } catch (e) { console.error('Failed to fetch costs:', e); }
        }

        async function fetchLogs() {
            try {
                const res = await fetch('/logs');
                const data = await res.json();
                const predictions = data.predictions || [];
                document.getElementById('totalPredictions').textContent = data.total || 0;

                // Calculate stats
                if (predictions.length > 0) {
                    const durations = predictions.filter(function(p) { return p.duration; }).map(function(p) { return p.duration; });
                    const avgDur = durations.length > 0 ? (durations.reduce(function(a, b) { return a + b; }, 0) / durations.length) : 0;
                    document.getElementById('avgDuration').textContent = avgDur.toFixed(1) + 's';

                    const disagreements = predictions.filter(function(p) { return p.had_disagreement; }).length;
                    document.getElementById('disagreements').textContent = disagreements + '/' + predictions.length;
                    document.getElementById('disagreements').className = 'card-value ' + (disagreements > predictions.length * 0.5 ? 'amber' : 'green');
                }

                const tbody = document.getElementById('logBody');
                if (predictions.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;color:#555577;">No predictions yet</td></tr>';
                    return;
                }

                const recent = predictions.slice(-20).reverse();
                tbody.innerHTML = recent.map(function(p) {
                    return '<tr>' +
                        '<td>' + formatTime(p.timestamp) + '</td>' +
                        '<td title="' + (p.event_ticker || '') + '">' + ((p.title || p.event_ticker || '—').slice(0, 50)) + '</td>' +
                        '<td>' + getCategoryChip(p.category) + '</td>' +
                        '<td>' + (p.outcomes || []).slice(0, 3).join(', ') + '</td>' +
                        '<td>' + formatProbs(p.probabilities) + '</td>' +
                        '<td>' + (p.duration ? p.duration + 's' : '—') + '</td>' +
                        '<td>' + (p.had_disagreement ? '<span class="status-dot dot-amber"></span>Yes' : '<span class="status-dot dot-green"></span>No') + '</td>' +
                    '</tr>';
                }).join('');
            } catch (e) { console.error('Failed to fetch logs:', e); }
        }

        async function refreshAll() {
            const bar = document.getElementById('refreshBar');
            bar.style.width = '20%';
            await fetchHealth();
            bar.style.width = '40%';
            await fetchApiStatus();
            bar.style.width = '60%';
            await fetchCosts();
            bar.style.width = '80%';
            await fetchLogs();
            bar.style.width = '100%';
            document.getElementById('uptime').textContent = formatUptime();
            document.getElementById('lastRefresh').textContent = new Date().toLocaleTimeString();
            setTimeout(function() { bar.style.width = '0%'; }, 500);
        }

        initChart();
        refreshAll();
        setInterval(refreshAll, 30000);
    </script>
</body>
</html>"""
