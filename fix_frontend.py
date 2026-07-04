import re

# 1. Update index.html
with open("index.html", "r") as f:
    html = f.read()

# Remove the "Grade a pending prediction" form entirely
html = re.sub(
    r'<div class="history-controls">.*?</div><!-- end history-controls -->',
    '''<div class="history-controls" style="display: flex; gap: 15px; margin-bottom: 20px;">
        <button onclick="autocheckHistory()" class="primary-btn" style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);">
            <span class="icon">🤖</span> Auto-Grade Pending
        </button>
    </div><!-- end history-controls -->''',
    html,
    flags=re.DOTALL
)

with open("index.html", "w") as f:
    f.write(html)
    
print("Updated index.html")

# 2. Update app.js
with open("app.js", "r") as f:
    js = f.read()

# Fix updateDashboard error: reading 'date' from 'prediction'
update_dashboard_new = """
// ── Update Dashboard ────────────────────────────────────────────────────────
function updateDashboard(data, analysis) {
    if(!data) return;

    // Update Header
    document.getElementById('symbol-header').textContent = data.symbol;
    document.getElementById('current-price').textContent = `Rs. ${data.latest_data.close.toFixed(2)}`;
    document.getElementById('last-updated').textContent = `As of: ${data.latest_data.date}`;
    document.getElementById('model-update-time').textContent = `Last predicted: ${data.last_updated}`;

    // Update Prediction Card (Multi-Timeframe)
    const cardContent = document.querySelector('.prediction-card .card-content');
    let html = `<div style="display: flex; gap: 15px; justify-content: space-between; overflow-x: auto;">`;
    
    ['1D', '5D', '20D'].forEach(tf => {
        const p = data.predictions[tf];
        if(!p) return;
        
        let color = p.predicted_change >= 0 ? '#10b981' : '#ef4444';
        let sigColor = p.signal === 'BUY' ? '#10b981' : (p.signal === 'SELL' ? '#ef4444' : '#f59e0b');
        
        html += `
            <div style="flex: 1; min-width: 150px; background: rgba(30,41,59,0.5); padding: 15px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
                <div style="font-size: 0.9em; color: #94a3b8; margin-bottom: 5px;">${tf} Target</div>
                <div style="font-size: 0.8em; color: #64748b; margin-bottom: 10px;">${p.date}</div>
                <div style="font-size: 1.5em; font-weight: 700; color: ${color};">Rs. ${p.predicted_close.toFixed(2)}</div>
                <div style="font-size: 0.9em; color: ${color}; margin-bottom: 10px;">
                    ${p.predicted_change >= 0 ? '+' : ''}${p.predicted_change.toFixed(2)} 
                    (${p.predicted_change >= 0 ? '+' : ''}${p.predicted_change_percent.toFixed(2)}%)
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.85em; margin-top: 15px; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 10px;">
                    <span>Dir: <span style="color:${color}">${p.direction}</span></span>
                    <span>Conf: ${(p.probability*100).toFixed(1)}%</span>
                </div>
                <div style="margin-top: 10px; font-weight: bold; color: ${sigColor}; text-align: center;">
                    SIGNAL: ${p.signal}
                </div>
            </div>
        `;
    });
    html += `</div>`;
    
    // Add reasons from 1D for context
    if(data.predictions['1D']) {
        html += `<div class="reasons-list" style="margin-top: 20px; font-size: 0.9em; color: #cbd5e1;">`;
        data.predictions['1D'].reasons.forEach(r => {
            html += `<div style="margin-bottom: 5px;">• ${r}</div>`;
        });
        html += `</div>`;
    }
    
    cardContent.innerHTML = html;

    // Metrics
    let m = data.metrics['1D'] || {};
    document.getElementById('mae-val').textContent = m.mae ? m.mae.toFixed(2) : '--';
    document.getElementById('acc-val').textContent = m.accuracy_percent ? `${m.accuracy_percent.toFixed(1)}%` : '--';
    document.getElementById('prec-val').textContent = m.precision_percent ? `${m.precision_percent.toFixed(1)}%` : '--';
"""

js = re.sub(
    r'// ── Update Dashboard ────────────────────────────────────────────────────────.*?document\.getElementById\(\'prec-val\'\)\.textContent = .*?;',
    update_dashboard_new,
    js,
    flags=re.DOTALL
)

# Replace submitResolve with autocheckHistory
autocheck_js = """
async function autocheckHistory() {
    const btn = document.querySelector('.history-controls button');
    btn.innerHTML = '<span class="icon">🤖</span> Grading...';
    try {
        const res = await fetch('/api/history/autocheck', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });
        const data = await res.json();
        if(data.status === 'success') {
            alert(`Auto-grade complete! Resolved ${data.resolved} predictions.`);
            await loadHistory();
            await loadHistoryStats();
        } else {
            alert('Error: ' + data.message);
        }
    } catch(err) {
        alert('Network error during auto-check.');
    }
    btn.innerHTML = '<span class="icon">🤖</span> Auto-Grade Pending';
}
"""

js = re.sub(r'async function submitResolve\(.*?\).*?\}', autocheck_js, js, flags=re.DOTALL)

with open("app.js", "w") as f:
    f.write(js)
print("Updated app.js")
