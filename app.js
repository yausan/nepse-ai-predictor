/* ════════════════════════════════════════════════════════
   NEPSE AI Predictor — app.js v2.0
   Handles: NEPSE Index tab, Stock Predictor tab,
            Prediction History tab, Chart rendering
════════════════════════════════════════════════════════ */

// ── Chart instances ────────────────────────────────────
let nepseChart    = null;
let nepsePerfChart = null;
let stockChart    = null;
let stockPerfChart = null;
let featureChart  = null;
let confRingChart = null;

// ── Global state ───────────────────────────────────────
let historyData = [];
let currentTab  = 'nepse';
let activeTimeframe = '1D';
let currentStockData = null;

// AD to BS Date Converter Helper
function fmtDateBS(adStr) {
    if (!adStr || adStr === '—' || adStr === 'N/A') return '—';
    try {
        const parts = adStr.split(' ')[0].split('-');
        if (parts.length < 3) return adStr;
        const year = parseInt(parts[0]);
        const month = parseInt(parts[1]) - 1; // 0-indexed month
        const day = parseInt(parts[2]);
        const nepali = new NepaliDate(new Date(year, month, day));
        return nepali.format('YYYY-MM-DD') + ' BS';
    } catch (e) {
        return adStr;
    }
}

// Returns short BS date label for display in tables/badges (e.g. "2082-03-18 BS")
function fmtDateBSShort(adStr) {
    if (!adStr || adStr === '—' || adStr === 'N/A') return '—';
    try {
        const parts = adStr.split(' ')[0].split('-');
        if (parts.length < 3) return adStr;
        const year = parseInt(parts[0]);
        const month = parseInt(parts[1]) - 1;
        const day = parseInt(parts[2]);
        const nepali = new NepaliDate(new Date(year, month, day));
        return nepali.format('YYYY-MM-DD') + ' BS';
    } catch (e) {
        return adStr;
    }
}

// Returns timeframe human label
function tfLabel(tf) {
    if (tf === '5D')  return 'Week (5D)';
    if (tf === '20D') return 'Month (20D)';
    return 'Day (1D)';
}

// ─────────────────────────────────────────────────────
//  INIT
// ─────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
    // Dates
    const adDate = new Date();
    let bsDate = '';
    try {
        bsDate = ' (' + new NepaliDate(adDate).format('YYYY-MM-DD') + ' BS)';
    } catch (e) {
        console.warn('NepaliDate library not ready yet:', e);
    }
    const dateStr = adDate.toLocaleDateString('en-US', { weekday:'short', year:'numeric', month:'short', day:'numeric' }) + bsDate;
    setIfExists('current-date-nepse', dateStr);
    setIfExists('current-date-stock', dateStr);

    // Symbol select toggle
    const symSel = document.getElementById('symbol-select');
    const custGrp = document.getElementById('custom-symbol-group');
    const symSelInner = document.getElementById('symbol-select-inner');
    const custGrpInner = document.getElementById('custom-symbol-group-inner');
    
    if (symSel) symSel.addEventListener('change', () => {
        custGrp.style.display = symSel.value === 'custom' ? 'block' : 'none';
        if (symSelInner && symSelInner.value !== symSel.value) {
            symSelInner.value = symSel.value;
            symSelInner.dispatchEvent(new Event('change'));
        }
    });

    if (symSelInner) symSelInner.addEventListener('change', () => {
        if (custGrpInner) custGrpInner.style.display = symSelInner.value === 'custom' ? 'block' : 'none';
        if (symSel && symSel.value !== symSelInner.value) {
            symSel.value = symSelInner.value;
            if (custGrp) custGrp.style.display = symSel.value === 'custom' ? 'block' : 'none';
        }
        // Auto-load predicted stock data immediately on symbol change
        const sym = getSymbol();
        localStorage.setItem('active_symbol', sym);
        tryLoadStockPrediction(sym);
        tryLoadAnalysisPrediction(sym);
        loadAndRenderDeepAnalysis(sym);
        checkPredictLock();
        fetchStatus();
    });

    // Custom symbol text synchronization
    const custSym = document.getElementById('custom-symbol');
    const custSymInner = document.getElementById('custom-symbol-inner');
    if (custSym) custSym.addEventListener('input', () => {
        if (custSymInner) custSymInner.value = custSym.value;
    });
    if (custSymInner) custSymInner.addEventListener('input', () => {
        if (custSym) custSym.value = custSymInner.value;
    });

    // Scrape depth select synchronization
    const nepPagesSel = document.getElementById('nepse-pages-select');
    const nepPagesInner = document.getElementById('nepse-pages-inner');
    if (nepPagesInner && nepPagesSel) {
        nepPagesInner.addEventListener('change', () => {
            nepPagesSel.value = nepPagesInner.value;
        });
        // Sync initial state
        nepPagesInner.value = nepPagesSel.value;
    }

    // Restore last symbol
    const saved = localStorage.getItem('active_symbol') || 'LICN';
    if (symSel) {
        const found = [...symSel.options].some(o => o.value === saved);
        if (found) {
            symSel.value = saved;
            if (symSelInner) symSelInner.value = saved;
        }
        else {
            symSel.value = 'custom';
            if (symSelInner) symSelInner.value = 'custom';
            custGrp.style.display = 'block';
            if (custGrpInner) custGrpInner.style.display = 'block';
            setIfExists('custom-symbol', saved, 'value');
            setIfExists('custom-symbol-inner', saved, 'value');
        }
    }

    // Init ring chart (empty)
    initConfRing(0);

    // Show loading overlay on startup
    showLoader('Initializing Dashboard...', 'Restoring session predictions, status checks, and historical metrics.');

    try {
        await Promise.all([
            fetchStatus(),
            tryLoadNepsePrediction(),
            tryLoadStockPrediction(saved),
            tryLoadAnalysisPrediction(saved),
            loadAndRenderDeepAnalysis(saved),
            loadHistory()
        ]);
    } catch (e) {
        console.warn('Dashboard initialization partial load error:', e);
    } finally {
        hideLoader();
        checkPredictLock();
    }
});

// ─────────────────────────────────────────────────────
//  TAB SWITCHING
// ─────────────────────────────────────────────────────
function switchTab(tab) {
    currentTab = tab;
    ['nepse', 'stocks', 'history', 'analysis'].forEach(t => {
        const panel  = document.getElementById(`tab-${t}`);
        const navBtn = document.getElementById(`nav-${t}`);
        const sPanel = document.getElementById(`panel-${t}`);
        if (panel)  panel.style.display  = t === tab ? '' : 'none';
        if (navBtn) navBtn.classList.toggle('active', t === tab);
        if (sPanel) sPanel.style.display = t === tab ? '' : 'none';
    });
    if (tab === 'history') {
        loadHistory();
        renderHistory();
    }
}

// ─────────────────────────────────────────────────────
//  STATUS
// ─────────────────────────────────────────────────────
async function fetchStatus() {
    const sym = getSymbol();
    try {
        const res  = await fetch(`/api/status?symbol=${sym}`);
        const data = await res.json();

        const nepseOk  = data.nepse_data_exists;
        const stockOk  = data.raw_data_exists;

        const nephEl = document.getElementById('status-nepse-data');
        if (nephEl) {
            nephEl.textContent  = nepseOk ? 'Available' : 'No Data';
            nephEl.className    = `status-badge ${nepseOk ? 'ok' : 'err'}`;
        }
        setIfExists('status-nepse-updated', data.nepse_last_scraped || '—');

        const stkEl = document.getElementById('status-stock-data');
        if (stkEl) {
            stkEl.textContent  = stockOk ? 'Available' : 'No Data';
            stkEl.className    = `status-badge ${stockOk ? 'ok' : 'err'}`;
        }
        setIfExists('status-last-scraped', data.last_scraped || '—');
    } catch (e) {
        console.warn('Status check failed:', e);
    }
}

// ─────────────────────────────────────────────────────
//  NEPSE INDEX — TRIGGER
// ─────────────────────────────────────────────────────
async function triggerNepseUpdate() {
    const pages = document.getElementById('nepse-pages-select')?.value || '3';
    await pollJobProgress({
        type:    'nepse_update',
        symbol:  'NEPSE',
        pages,
        title:   'Analyzing Market Depth…',
        desc:    `Aggregating live exchange data, institutional footprints, and broader macroeconomic sentiment across NEPSE (${pages} cycles).`,
        onDone:  (result) => {
            if (result) updateNepseDashboard(result);
            fetchStatus();
            loadHistory();
        },
    });
}

async function triggerNepsePredict() {
    await pollJobProgress({
        type:    'nepse_predict',
        symbol:  'NEPSE',
        title:   'Synthesizing Forecast…',
        desc:    'Computing historical price action against macro indicators to construct a high-probability directional forecast for the NEPSE index.',
        onDone:  (result) => {
            if (result) updateNepseDashboard(result);
            fetchStatus();
            loadHistory();
        },
    });
}

// Try to load existing NEPSE prediction on startup
async function tryLoadNepsePrediction() {
    try {
        const res = await fetch('/outputs/predictions/nepse_predictions.json');
        if (res.ok) {
            const data = await res.json();
            updateNepseDashboard(data);
        }
    } catch (_) {}
}

// ─────────────────────────────────────────────────────
//  STOCK — TRIGGER
// ─────────────────────────────────────────────────────
async function triggerStockUpdate() {
    const sym = getSymbol();
    await pollJobProgress({
        type:    'update',
        symbol:  sym,
        full:    false,
        title:   `Synchronizing ${sym}…`,
        desc:    `Fetching real-time market data, detecting anomalous volume spikes, and recalculating structural order blocks for ${sym}.`,
        onDone:  (result) => {
            if (result) updateStockDashboard(result);
            fetchStatus();
            localStorage.setItem('active_symbol', sym);
        },
    });
}

async function triggerStockTrain() {
    const sym = getSymbol();
    await pollJobProgress({
        type:    'predict',
        symbol:  sym,
        title:   `Deep Intelligence: ${sym}`,
        desc:    'Executing proprietary multi-layer analysis. Evaluating smart money concepts, liquidity zones, and volatility metrics to generate high-conviction intelligence.',
        onDone:  async (result) => {
            if (result) updateStockDashboard(result);
            fetchStatus();
            await loadAndRenderDeepAnalysis(sym);
            // Unlock the Predict button
            const predictBtn = document.getElementById('btn-update-inner');
            if (predictBtn) {
                predictBtn.disabled = false;
                predictBtn.style.opacity = '1';
                predictBtn.style.cursor  = 'pointer';
                predictBtn.title = '';
            }
            localStorage.setItem(`analysed_${sym}`, '1');
        },
    });
}

// ─────────────────────────────────────────────────────
//  BACKGROUND JOB POLLING
// ─────────────────────────────────────────────────────
/**
 * Kicks off a background job, shows the animated progress overlay,
 * polls every 2 s, and calls onDone(result) when the job completes.
 * @param {object} opts  { type, symbol, pages, full, title, desc, onDone }
 */
async function pollJobProgress(opts) {
    const { type, symbol, pages, full, title, desc, onDone } = opts;

    showLoader(title, desc);
    updateLoaderProgress(5, 'Starting job on server…');
    setButtonsDisabled(true);

    let job_id;
    try {
        const startRes = await fetch('/api/jobs/start', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ type, symbol, pages, full }),
        });
        const startData = await startRes.json();
        if (!startData.job_id) {
            showError(startData.error || 'Failed to start job.');
            setButtonsDisabled(false);
            return;
        }
        job_id = startData.job_id;
    } catch (e) {
        showError(`Could not reach server: ${e.message}`);
        setButtonsDisabled(false);
        return;
    }

    // Poll every 2 seconds
    return new Promise((resolve) => {
        const interval = setInterval(async () => {
            try {
                const res  = await fetch(`/api/jobs/${job_id}`);
                const job  = await res.json();

                updateLoaderProgress(job.progress || 0, job.message || '…');

                if (job.status === 'done') {
                    clearInterval(interval);
                    hideLoader();
                    setButtonsDisabled(false);
                    checkPredictLock();
                    try { await onDone(job.result); } catch (_) {}
                    resolve();
                } else if (job.status === 'error') {
                    clearInterval(interval);
                    hideLoader();
                    setButtonsDisabled(false);
                    showError(job.error || 'The job encountered an error.');
                    resolve();
                }
            } catch (e) {
                // Network hiccup — keep polling silently
                console.warn('[poll] Network hiccup, retrying…', e.message);
            }
        }, 2000);
    });
}

/** Updates the loader progress bar and step label. */
function updateLoaderProgress(pct, message) {
    const bar = document.getElementById('loader-bar');
    if (bar) bar.style.width = `${Math.min(pct, 100)}%`;
    setIfExists('loader-desc', message);
}

function checkPredictLock() {
    const sym = getSymbol();
    const analysed = localStorage.getItem(`analysed_${sym}`);
    const predictBtn = document.getElementById('btn-update-inner');
    if (!predictBtn) return;
    if (analysed) {
        predictBtn.disabled = false;
        predictBtn.style.opacity = '1';
        predictBtn.style.cursor = 'pointer';
        predictBtn.title = '';
    } else {
        predictBtn.disabled = true;
        predictBtn.style.opacity = '0.4';
        predictBtn.style.cursor = 'not-allowed';
        predictBtn.title = 'Run "Retrain & Analyse Market" first to unlock predictions';
    }
}

async function loadAndRenderDeepAnalysis(sym) {
    try {
        const res = await fetch(`/outputs/analysis/${sym.toLowerCase()}_analysis.json?t=${Date.now()}`);
        if (res.ok) {
            const data = await res.json();
            renderDeepAnalysisInPredictor(data);
        }
    } catch (_) {}
}

function renderDeepAnalysisInPredictor(data) {
    if (!data) return;
    const card = document.getElementById('stock-deep-analysis-card');
    if (!card) return;
    card.style.display = 'block';

    const l1 = data.layer1_structure || {};
    const l2 = data.layer2_levels || {};
    const l3 = data.layer3_patterns || {};
    const l4 = data.layer4_volume || {};
    const l5 = data.layer5_trend || {};
    const l6 = data.layer6_momentum || {};
    const l7 = data.layer7_nepse || {};
    const l8 = data.layer8_vwap || {};
    const l9 = data.layer9_volatility || {};
    const cl = data.checklist || {};

    // Header
    setIfExists('sda-symbol', data.symbol);
    setIfExists('sda-updated', data.analyzed_at);

    // Score badge
    const scoreEl = document.getElementById('sda-score');
    if (scoreEl) {
        scoreEl.textContent = `${cl.score || 0}/${cl.total || 9}`;
        const c = cl.rating_color;
        scoreEl.style.background = c === 'green' ? 'rgba(16,185,129,0.2)' : (c === 'gold' ? 'rgba(245,158,11,0.2)' : 'rgba(239,68,68,0.2)');
        scoreEl.style.color = c === 'green' ? 'var(--bullish-green)' : (c === 'gold' ? 'var(--accent-orange)' : 'var(--bearish-red)');
        scoreEl.style.fontWeight = '700';
    }
    const ratingEl = document.getElementById('sda-rating');
    if (ratingEl) {
        ratingEl.textContent = cl.rating || '';
        const c = cl.rating_color;
        ratingEl.style.color = c === 'green' ? 'var(--bullish-green)' : (c === 'gold' ? 'var(--accent-orange)' : 'var(--bearish-red)');
    }

    // Trend
    const trendEl = document.getElementById('sda-trend');
    if (trendEl) {
        trendEl.textContent = l1.trend || '—';
        trendEl.style.color = l1.trend === 'UPTREND' ? 'var(--bullish-green)' : (l1.trend === 'DOWNTREND' ? 'var(--bearish-red)' : 'var(--text-muted)');
    }
    setIfExists('sda-trend-reason', l1.trend_reason || '—');
    const bos = l1.bos;
    if (bos) setIfExists('sda-bos', `${bos.type?.toUpperCase()} BOS @ ${bos.level}`);

    // Levels
    setIfExists('sda-price', (l2.current_price || 0).toFixed(2));
    const sr = l2.nearest_resistance;
    const ss = l2.nearest_support;
    setIfExists('sda-res', sr ? `${sr.price} (${sr.distance_pct?.toFixed(1)}% away, ${sr.strength})` : '—');
    setIfExists('sda-sup', ss ? `${ss.price} (${ss.distance_pct?.toFixed(1)}% away, ${ss.strength})` : '—');

    // Patterns
    const patterns = l3.today_patterns || l3.patterns || [];
    const allPatterns = l3.patterns || [];
    const patEl = document.getElementById('sda-pattern');
    if (patEl) {
        if (patterns.length) {
            patEl.innerHTML = patterns.map(p => `<span style="display:inline-block;padding:3px 10px;border-radius:20px;font-size:0.8rem;margin:2px;background:${p.type==='bullish'?'rgba(16,185,129,0.15)':p.type==='bearish'?'rgba(239,68,68,0.15)':'rgba(255,255,255,0.07)'};color:${p.type==='bullish'?'var(--bullish-green)':p.type==='bearish'?'var(--bearish-red)':'var(--text-muted)'}">${p.name}</span>`).join('');
        } else {
            patEl.textContent = 'No Pattern Today';
        }
    }
    setIfExists('sda-pattern-desc', patterns.length ? patterns.map(p => p.description).join(' | ') : 'No significant price action pattern detected on the latest candle.');

    // Volume
    const volClass = (l4.classification || '').replace(/_/g, ' ');
    const volEl = document.getElementById('sda-volume');
    if (volEl) {
        const isBullVol = volClass.includes('UP VOL UP') || volClass.includes('VOL UP');
        const isBearVol = volClass.includes('DOWN VOL UP');
        volEl.textContent = volClass || '—';
        volEl.style.color = isBearVol ? 'var(--bearish-red)' : (isBullVol ? 'var(--bullish-green)' : 'var(--text-muted)');
    }
    setIfExists('sda-vol-desc', l4.meaning || '—');

    // Indicators
    setIfExists('sda-rsi', `${l6.rsi || '—'} — ${l6.rsi_zone || ''} ${l6.rsi_note || ''}`);
    setIfExists('sda-ema', `EMA9: ${(l5.ema9 || 0).toFixed(2)} | EMA21: ${(l5.ema21 || 0).toFixed(2)} | Signal: ${l5.ema_signal || '—'}`);
    setIfExists('sda-vwap', `VWAP: ${(l8.vwap || 0).toFixed(2)} — ${l8.vwap_note || '—'}`);
    setIfExists('sda-adx', `ADX: ${l5.adx || '—'} — ${l5.adx_note || '—'}`);
    setIfExists('sda-atr', `ATR(14): ${(l9.atr || 0).toFixed(2)} | BB Width: ${((l9.bb_width || 0) * 100).toFixed(2)}%`);
    setIfExists('sda-sector', `${l7.sector || '—'} | Change: ${l7.daily_change_pct?.toFixed(2) || '—'}%`);

    // Checklist
    const clUl = document.getElementById('sda-checklist');
    if (clUl && cl.items) {
        clUl.innerHTML = cl.items.map(item => `
            <li style="display:flex;align-items:flex-start;gap:10px;padding:10px 0;border-bottom:1px solid rgba(255,255,255,0.05);">
                <span style="font-size:1rem;color:${item.pass ? 'var(--bullish-green)' : 'var(--bearish-red)'}">
                    <i class="fa-solid ${item.pass ? 'fa-circle-check' : 'fa-circle-xmark'}"></i>
                </span>
                <div style="flex:1;">
                    <div style="font-size:0.88rem;font-weight:600;">
                        <span style="color:var(--accent-blue);margin-right:5px;">${item.label}</span>
                        <span style="color:var(--text-primary);">${item.question}</span>
                    </div>
                    <div style="font-size:0.78rem;color:var(--text-muted);margin-top:2px;">
                        <code style="background:rgba(255,255,255,0.06);padding:1px 5px;border-radius:3px;margin-right:4px;">${item.value}</code>
                        ${item.detail}
                    </div>
                </div>
            </li>
        `).join('');
    }
}

async function tryLoadStockPrediction(sym) {
    try {
        const res = await fetch(`/outputs/predictions/${sym.toLowerCase()}_predictions.json`);
        if (res.ok) {
            const data = await res.json();
            updateStockDashboard(data);
        }
    } catch (_) {}
}

// ─────────────────────────────────────────────────────
//  NEPSE DASHBOARD UPDATE
// ─────────────────────────────────────────────────────
function updateNepseDashboard(data) {
    if (!data) return;

    // Handle multi-timeframe structure: data.predictions = {1D: {...}, 5D: {...}, 20D: {...}}
    // Fall back to data.prediction (singular) for legacy format
    let pred, metrics;
    if (data.predictions && typeof data.predictions === 'object') {
        // Multi-timeframe: pick 1D as primary display
        pred = data.predictions['1D'] || data.predictions[Object.keys(data.predictions)[0]];
        metrics = data.metrics?.['1D'] || data.metrics?.[Object.keys(data.metrics || {})[0]] || {};
    } else {
        pred = data.prediction;
        metrics = data.metrics || {};
    }
    if (!pred) return;

    const latest  = data.latest_data;
    const ind     = data.indicators || {};

    setIfExists('nepse-subtitle', `Last updated: ${data.last_updated} · NEPSE Index AI Forecast`);

    // Hero
    setIfExists('nepse-today-close', fmt(latest.close));
    setIfExists('nepse-today-date', `as of ${fmtDateBSShort(latest.date)}`);
    setIfExists('nepse-pred-date', `Target: ${fmtDateBSShort(pred.date)}`);

    const predEl = document.getElementById('nepse-pred-close');
    if (predEl) predEl.textContent = fmt(pred.predicted_close);

    const chEl   = document.getElementById('nepse-pred-change');
    const sign   = pred.predicted_change >= 0 ? '+' : '';
    if (chEl) {
        chEl.textContent = `${sign}${pred.predicted_change.toFixed(2)} pts (${sign}${pred.predicted_change_percent.toFixed(2)}%)`;
        chEl.className   = `hero-change ${pred.predicted_change >= 0 ? 'up' : 'down'}`;
    }
    
    const rangeEl = document.getElementById('nepse-pred-range');
    if (rangeEl && pred.predicted_high) {
        rangeEl.textContent = `High: ${fmt(pred.predicted_high)} | Low: ${fmt(pred.predicted_low)}`;
    } else if (rangeEl) {
        rangeEl.textContent = '';
    }

    // Arrow
    const arrow = document.getElementById('nepse-arrow');
    if (arrow) {
        const dir = (pred.direction || '').toUpperCase();
        arrow.style.color = dir === 'UP' ? 'var(--bullish-green)' : (dir === 'DOWN' ? 'var(--bearish-red)' : 'var(--text-secondary)');
    }

    // Direction pill
    const pill    = document.getElementById('nepse-direction-pill');
    const dirIcon = document.getElementById('nepse-dir-icon');
    const dirTxt  = document.getElementById('nepse-dir-text');
    if (pill) {
        const dir = (pred.direction || '').toUpperCase();
        if (dir === 'UP') {
            pill.className = 'direction-pill up';
            if (dirIcon) dirIcon.className = 'fa-solid fa-arrow-trend-up';
            if (dirTxt)  dirTxt.textContent = 'BULLISH';
        } else if (dir === 'DOWN') {
            pill.className = 'direction-pill down';
            if (dirIcon) dirIcon.className = 'fa-solid fa-arrow-trend-down';
            if (dirTxt)  dirTxt.textContent = 'BEARISH';
        } else {
            pill.className = 'direction-pill neutral';
            if (dirIcon) dirIcon.className = 'fa-solid fa-minus';
            if (dirTxt)  dirTxt.textContent = 'NEUTRAL';
        }
    }

    // Confidence ring
    const confPct = Math.round(pred.probability * 100);
    setIfExists('nepse-conf-pct', `${confPct}%`);
    updateConfRing(confPct, pred.direction);

    // KPIs
    setIfExists('nepse-mae', `${(metrics.mae || 0).toFixed(2)}`);
    const skillVal = metrics.skill_score !== undefined ? ` (Skill: ${metrics.skill_score >= 0 ? '+' : ''}${metrics.skill_score.toFixed(2)})` : '';
    setIfExists('nepse-dir-acc', `${(metrics.accuracy_percent || 0).toFixed(1)}%${skillVal}`);
    setIfExists('nepse-mape', `${(metrics.mape_percent || 0).toFixed(2)}%`);

    const rsi      = ind.rsi_14 || ind.rsi || 50;
    const rsiLabel = rsi < 30 ? 'Oversold 📉' : rsi > 70 ? 'Overbought 📈' : 'Neutral';
    setIfExists('nepse-rsi', rsi.toFixed(1));
    setIfExists('nepse-rsi-label', rsiLabel);

    // 7-Layer signals
    const overall = document.getElementById('nepse-overall-signal');
    if (overall && pred.signal) {
        overall.textContent = pred.signal;
        overall.className = 'signal-badge';
        if (pred.signal === 'BUY') overall.classList.add('signal-buy');
        else if (pred.signal === 'SELL') overall.classList.add('signal-sell');
        else overall.classList.add('signal-hold');
    }

    const rList = document.getElementById('nepse-signal-reasons-list');
    if (rList && pred.reasons) {
        rList.innerHTML = pred.reasons.map(r => `<li>${r}</li>`).join('');
    }

    // Charts
    if (data.chart_data) {
        renderNepseChart(data.chart_data, data.predictions || { "1D": pred });
        renderNepsePerfChart(data.chart_data, data.predictions || { "1D": pred });
    }
}

// ─────────────────────────────────────────────────────
//  STOCK DASHBOARD UPDATE
// ─────────────────────────────────────────────────────
function updateStockDashboard(data) {
    if (!data) return;
    currentStockData = data;

    // Handle multi-timeframe structure: data.predictions = {1D: {...}, 5D: {...}, 20D: {...}}
    // Fall back to data.prediction (singular) for legacy format
    let pred, metrics;
    const timeframe = activeTimeframe || '1D';
    
    // Update active tab visual selection just in case
    document.querySelectorAll('.timeframe-tab').forEach(btn => {
        btn.classList.toggle('active', btn.getAttribute('data-timeframe') === timeframe);
    });

    // Update Tomorrow's Prediction card header title
    const labelEl = document.querySelector('.summary-card .card-header h3');
    if (labelEl) {
        if (timeframe === '1D') labelEl.textContent = "Tomorrow's Prediction";
        else if (timeframe === '5D') labelEl.textContent = "Next Week Forecast (5D)";
        else if (timeframe === '20D') labelEl.textContent = "Next Month Forecast (20D)";
    }

    if (data.predictions && typeof data.predictions === 'object') {
        pred = data.predictions[timeframe] || data.predictions['1D'] || data.predictions[Object.keys(data.predictions)[0]];
        metrics = data.metrics?.[timeframe] || data.metrics?.['1D'] || data.metrics?.[Object.keys(data.metrics || {})[0]] || {};
    } else {
        pred = data.prediction;
        metrics = data.metrics || {};
    }
    if (!pred) return;

    const latest = data.latest_data;
    const ind    = data.indicators || {};

    setIfExists('stock-subtitle', `AI analysis for ${data.symbol} · generated ${data.last_updated}`);
    // Show BS date in target badge
    const bsTarget = fmtDateBSShort(pred.date);
    setIfExists('target-date-badge', `🗓 ${bsTarget}`);

    // Price prediction
    setIfExists('pred-close-price', pred.predicted_close.toFixed(2));
    const sign = pred.predicted_change >= 0 ? '+' : '';
    const chEl = document.getElementById('pred-price-change');
    if (chEl) {
        chEl.textContent  = `${sign}${pred.predicted_change.toFixed(2)} (${sign}${pred.predicted_change_percent.toFixed(2)}%)`;
        chEl.className    = `pred-change ${pred.predicted_change >= 0 ? 'up' : 'down'}`;
    }
    
    const rangeEl = document.getElementById('pred-range-text');
    if (rangeEl && pred.predicted_high) {
        rangeEl.textContent = `High: ${pred.predicted_high.toFixed(2)} | Low: ${pred.predicted_low.toFixed(2)}`;
    } else if (rangeEl) {
        rangeEl.textContent = '';
    }

    // Direction pill
    const pill    = document.getElementById('stock-direction-pill');
    const dirIcon = document.getElementById('stock-dir-icon');
    const dirTxt  = document.getElementById('stock-dir-text');
    if (pill) {
        const dir = (pred.direction || '').toUpperCase();
        if (dir === 'UP') {
            pill.className = 'direction-pill up';
            if (dirIcon) dirIcon.className = 'fa-solid fa-arrow-trend-up';
            if (dirTxt)  dirTxt.textContent = 'BULLISH';
        } else if (dir === 'DOWN') {
            pill.className = 'direction-pill down';
            if (dirIcon) dirIcon.className = 'fa-solid fa-arrow-trend-down';
            if (dirTxt)  dirTxt.textContent = 'BEARISH';
        } else {
            pill.className = 'direction-pill neutral';
            if (dirIcon) dirIcon.className = 'fa-solid fa-minus';
            if (dirTxt)  dirTxt.textContent = 'NEUTRAL';
        }
    }

    // Confidence bar
    const conf = Math.round(pred.probability * 100);
    const fillEl = document.getElementById('stock-conf-fill');
    if (fillEl) fillEl.style.width = `${conf}%`;
    setIfExists('stock-conf-pct', `${conf}%`);

    // Signal
    const sig = pred.signal?.toUpperCase() || 'HOLD';
    const sigEl = document.getElementById('signal-badge');
    if (sigEl) {
        sigEl.textContent = sig;
        sigEl.className   = `signal-badge ${sig.toLowerCase()}`;
    }
    const rList = document.getElementById('signal-reasons-list');
    if (rList && pred.reasons) {
        rList.innerHTML = pred.reasons.map(r => `<li>${r}</li>`).join('');
    }

    // Deep Analysis
    const daCard = document.getElementById('deep-analysis-card');
    if (daCard) {
        if (pred.deep_analysis) {
            daCard.style.display = 'block';
            const da = pred.deep_analysis;
            setIfExists('deep-sentiment-badge', `Sentiment: ${da.current_market_sentiment.split('.')[0]}`);
            setIfExists('deep-confidence-badge', `Confidence: ${da.confidence_score}%`);
            setIfExists('deep-exec-summary', da.executive_summary);
            setIfExists('deep-news-impact', da.news_impact_analysis);
            setIfExists('deep-fundamentals', da.fundamental_analysis);
            setIfExists('deep-technicals', da.technical_analysis);
            setIfExists('deep-prob-bull', `${da.bullish_probability}%`);
            setIfExists('deep-prob-bear', `${da.bearish_probability}%`);
            setIfExists('deep-prob-neutral', `${da.neutral_probability}%`);
            const bb = document.getElementById('deep-bar-bull'); if(bb) bb.style.width = `${da.bullish_probability}%`;
            const rb = document.getElementById('deep-bar-bear'); if(rb) rb.style.width = `${da.bearish_probability}%`;
            const nb = document.getElementById('deep-bar-neutral'); if(nb) nb.style.width = `${da.neutral_probability}%`;
            
            const rfList = document.getElementById('deep-risk-factors');
            if (rfList && da.risk_factors) {
                rfList.innerHTML = da.risk_factors.map(r => `<li>${r}</li>`).join('');
            }
            
            const frec = document.getElementById('deep-final-rec');
            if (frec) {
                frec.textContent = da.final_recommendation;
                frec.style.color = da.final_recommendation === 'BUY' ? 'var(--bullish-green)' : (da.final_recommendation === 'SELL' ? 'var(--bearish-red)' : 'var(--accent-gold)');
            }
            setIfExists('deep-rec-justification', da.recommendation_justification);
        } else {
            daCard.style.display = 'none';
        }
    }

    // Indicators
    setIfExists('ind-rsi', ind.rsi_14?.toFixed(2));
    setIfExists('ind-macd', ind.macd?.toFixed(2));
    setIfExists('ind-macd-sig', ind.macd_signal?.toFixed(2));
    setIfExists('ind-adx', ind.adx?.toFixed(2));
    
    const atr = ind.atr_14 || 0;
    const atrPct = latest.close > 0 ? (atr / latest.close * 100).toFixed(2) : '0.00';
    setIfExists('ind-atr', `${atr.toFixed(2)} (${atrPct}%)`);
    
    setIfExists('ind-vwap', ind.vwap?.toFixed(2));
    
    const stDir = ind.supertrend_dir > 0 ? 'Buy' : 'Sell';
    setIfExists('ind-supertrend', `${ind.supertrend?.toFixed(2)} (${stDir})`);
    
    setIfExists('ind-cmf', ind.cmf?.toFixed(3));
    setIfExists('ind-bb-mid', ind.bb_middle?.toFixed(2));
    setIfExists('ind-bb-width', ind.bb_width?.toFixed(3));
    
    // Format OBV cleanly
    const obvStr = ind.obv ? (ind.obv > 1000000 ? (ind.obv/1000000).toFixed(1) + 'M' : ind.obv.toFixed(0)) : '--';
    setIfExists('ind-obv', obvStr);

    setIfExists('ind-ema', `${ind.ema_20?.toFixed(0)} / ${ind.ema_50?.toFixed(0)} / ${ind.ema_200?.toFixed(0)}`);

    // Metrics
    setIfExists('metric-mae',      `${(metrics.mae || 0).toFixed(2)} pts`);
    setIfExists('metric-mape',     `${(metrics.mape_percent || 0).toFixed(2)}%`);
    setIfExists('metric-baseline', `${(metrics.baseline_accuracy_percent || 0).toFixed(1)}%`);
    setIfExists('metric-accuracy', `${(metrics.accuracy_percent || 0).toFixed(1)}%`);
    setIfExists('metric-skill-score', `${metrics.skill_score !== undefined ? (metrics.skill_score >= 0 ? '+' : '') + metrics.skill_score.toFixed(3) : '—'}`);

    // Charts
    if (data.chart_data) {
        renderStockChart(data.chart_data, data.predictions || { "1D": pred });
        renderStockPerfChart(data.chart_data, data.predictions || { "1D": pred });
    }
    if (data.feature_importances) {
        renderFeatureChart(data.feature_importances);
    }
}

// Timeframe selector action
function changeTimeframe(tf) {
    activeTimeframe = tf;
    
    // Update timeframe buttons active state
    document.querySelectorAll('.timeframe-tab').forEach(btn => {
        btn.classList.toggle('active', btn.getAttribute('data-timeframe') === tf);
    });
    
    // Refresh dashboard contents using the newly selected timeframe
    if (currentStockData) {
        updateStockDashboard(currentStockData);
    }
}

// ─────────────────────────────────────────────────────
//  PREDICTION HISTORY
// ─────────────────────────────────────────────────────
async function loadHistory() {
    try {
        const res  = await fetch('/api/history');
        const data = await res.json();
        historyData = data.history || [];

        // Update pending badge
        const pending = historyData.filter(e => e.status === 'pending').length;
        const badge   = document.getElementById('pending-badge');
        if (badge) {
            badge.style.display = pending > 0 ? '' : 'none';
            badge.textContent   = pending;
        }

        await loadHistoryStats();
        renderHistory();
    } catch (e) {
        console.warn('Failed to load history:', e);
    }
}

async function loadHistoryStats() {
    try {
        const res  = await fetch('/api/history/stats');
        const stats = await res.json();

        setIfExists('stat-score',   stats.score !== undefined ? `${stats.score}%` : '—');
        setIfExists('stat-exact',   stats.exact ?? '—');
        setIfExists('stat-close',   stats.close ?? '—');
        setIfExists('stat-wrong',   stats.wrong ?? '—');
        setIfExists('stat-dir-acc', stats.direction_accuracy !== undefined ? `${stats.direction_accuracy}%` : '—');
        setIfExists('stat-avg-err', stats.avg_error_pts !== undefined ? `${stats.avg_error_pts}` : '—');
        setIfExists('stat-pending', stats.pending ?? '—');
    } catch (e) {
        console.warn('Failed to load stats:', e);
    }
}

// ─────────────────────────────────────────────────────
//  HISTORY STATE
// ─────────────────────────────────────────────────────
let historyActiveTF     = 'all';   // 'all' | '1D' | '5D' | '20D'
let historyActiveStatus = 'all';   // 'all' | 'pending' | 'exact' | 'close' | 'wrong'

function switchHistoryTF(tf) {
    historyActiveTF = tf;
    // Update tab buttons in the log card
    document.querySelectorAll('.hist-tf-btn').forEach(b => {
        b.classList.toggle('active', b.dataset.tf === tf || (tf==='all' && b.dataset.tf==='all'));
    });
    // Highlight the overview card
    ['1D','5D','20D'].forEach(t => {
        const card = document.getElementById(`tf-card-${t}`);
        if (card) card.style.borderColor = (tf === t) ? 'var(--accent-blue)' : 'transparent';
    });
    renderHistory();
}

function switchHistoryStatus(status) {
    historyActiveStatus = status;
    document.querySelectorAll('.hist-status-pill').forEach(b => {
        b.classList.toggle('active', b.id === `hsp-${status}`);
    });
    renderHistory();
}

function updateTFCards(data) {
    const tfs = ['1D', '5D', '20D'];
    tfs.forEach(tf => {
        const entries = data.filter(e => (e.timeframe||'1D') === tf);
        const graded  = entries.filter(e => e.status !== 'pending');
        const exact   = graded.filter(e => e.status === 'exact').length;
        const close   = graded.filter(e => e.status === 'close').length;
        const wrong   = graded.filter(e => e.status === 'wrong').length;
        const acc     = graded.length ? Math.round(((exact + close) / graded.length) * 100) : null;

        setIfExists(`tf-${tf}-count`, entries.length);
        setIfExists(`tf-${tf}-exact`, exact);
        setIfExists(`tf-${tf}-close`, close);
        setIfExists(`tf-${tf}-wrong`, wrong);
        setIfExists(`tf-${tf}-acc`,   acc !== null ? `${acc}%` : '—');
        const bar = document.getElementById(`tf-${tf}-bar`);
        if (bar) bar.style.width = acc !== null ? `${acc}%` : '0%';
    });
}

function renderHistory() {
    const tbody  = document.getElementById('history-tbody');
    if (!tbody) return;

    const symFilter  = (document.getElementById('history-symbol-filter')?.value || '').trim().toUpperCase();
    const sortMode   = document.getElementById('history-sort')?.value || 'newest';

    // Update per-TF cards
    updateTFCards(historyData);

    // Filter
    let filtered = historyData.slice();

    // Timeframe filter
    if (historyActiveTF !== 'all') {
        const tfMap = { '1D': '1D', '5D': '5D', '20D': '20D' };
        filtered = filtered.filter(e => (e.timeframe || '1D') === tfMap[historyActiveTF]);
    }

    // Status filter
    if (historyActiveStatus !== 'all') {
        filtered = filtered.filter(e => e.status === historyActiveStatus);
    }

    // Symbol filter
    if (symFilter) {
        filtered = filtered.filter(e => {
            const sym = (e.symbol || e.id?.split('_')[0] || '').toUpperCase();
            return sym.includes(symFilter);
        });
    }

    // Sort
    if (sortMode === 'newest') {
        filtered.sort((a, b) => (b.made_on || '').localeCompare(a.made_on || ''));
    } else if (sortMode === 'oldest') {
        filtered.sort((a, b) => (a.made_on || '').localeCompare(b.made_on || ''));
    } else if (sortMode === 'symbol') {
        filtered.sort((a, b) => {
            const sa = (a.symbol || '').toUpperCase();
            const sb = (b.symbol || '').toUpperCase();
            return sa.localeCompare(sb);
        });
    }

    const count = document.getElementById('history-count');
    if (count) count.textContent = `${filtered.length} records`;

    if (filtered.length === 0) {
        tbody.innerHTML = `<tr><td colspan="12" class="empty-row">No predictions match the current filter.</td></tr>`;
        return;
    }

    // TF colour palette
    const tfColors = { '1D': '#3b82f6', '5D': '#8b5cf6', '20D': '#f59e0b' };
    const tfIcons  = { '1D': 'fa-sun', '5D': 'fa-calendar-week', '20D': 'fa-calendar-days' };

    tbody.innerHTML = filtered.map(entry => {
        const tf = entry.timeframe || '1D';
        const tfColor = tfColors[tf] || 'var(--text-secondary)';
        const tfIcon  = tfIcons[tf]  || 'fa-calendar';
        const tfText  = tfLabel(tf);

        const dir = entry.predicted_direction || '';
        const dirCell = dir === 'UP'
            ? '<span class="val-up"><i class="fa-solid fa-arrow-trend-up"></i> UP</span>'
            : dir === 'DOWN'
            ? '<span class="val-down"><i class="fa-solid fa-arrow-trend-down"></i> DOWN</span>'
            : '<span style="color:var(--text-muted)">FLAT</span>';

        const actualCell = entry.actual_close != null
            ? `<span class="num">${fmt(entry.actual_close)}</span>`
            : `<span style="color:var(--text-muted)">—</span>`;

        const diffPts = entry.diff_points != null
            ? `<span class="${entry.diff_points >= 0 ? 'val-up' : 'val-down'}">${entry.diff_points >= 0 ? '+' : ''}${entry.diff_points.toFixed(2)}</span>`
            : '—';

        const diffPct = entry.diff_percent != null
            ? `<span class="${Math.abs(entry.diff_percent) < 0.5 ? 'val-up' : (Math.abs(entry.diff_percent) < 2 ? '' : 'val-down')}">${entry.diff_percent.toFixed(2)}%</span>`
            : '—';

        const badge = `<span class="result-badge ${entry.status}">${entry.label}</span>`;

        const dirCorr = entry.direction_correct != null
            ? (entry.direction_correct
                ? '<span class="val-up"><i class="fa-solid fa-check"></i></span>'
                : '<span class="val-down"><i class="fa-solid fa-xmark"></i></span>')
            : '—';

        const action = entry.status === 'pending'
            ? `<button class="btn-grade" onclick="openResolveModal('${entry.prediction_date}', '${parseFloat(entry.predicted_close).toFixed(2)}', '${parseFloat(entry.prev_close).toFixed(2)}', '${entry.predicted_direction}')">Grade</button>
               <button class="btn-del" onclick="deleteEntry('${entry.id}')" title="Delete"><i class="fa-solid fa-trash"></i></button>`
            : `<button class="btn-del" onclick="deleteEntry('${entry.id}')" title="Delete"><i class="fa-solid fa-trash"></i></button>`;

        const symbolLabel = entry.type === 'nepse_index'
            ? '<span style="color:var(--accent-blue);font-weight:700">NEPSE</span>'
            : `<span style="color:var(--accent-green);font-weight:700">${entry.symbol || entry.id?.split('_')[0]?.toUpperCase() || '?'}</span>`;

        const madeOnBS  = fmtDateBSShort(entry.made_on);
        const forDateBS = fmtDateBSShort(entry.prediction_date);

        return `<tr>
            <td>${symbolLabel}</td>
            <td>
                <span style="display:inline-flex;align-items:center;gap:5px;font-weight:600;color:${tfColor};background:${tfColor}18;padding:2px 8px;border-radius:20px;font-size:0.8rem;">
                    <i class="fa-solid ${tfIcon}" style="font-size:0.7rem;"></i>${tfText}
                </span>
            </td>
            <td><span title="${entry.made_on}">${madeOnBS}</span></td>
            <td><strong title="${entry.prediction_date}">${forDateBS}</strong></td>
            <td class="num">${fmt(entry.prev_close)}</td>
            <td class="num" style="color:var(--accent-blue)">${fmt(entry.predicted_close)} <small style="color:var(--text-muted);">(${dir})</small></td>
            <td>${actualCell}</td>
            <td>${diffPts}</td>
            <td>${diffPct}</td>
            <td>${dirCorr}</td>
            <td>${badge}</td>
            <td style="display:flex;gap:4px;align-items:center">${action}</td>
        </tr>`;
    }).join('');
}

// ─────────────────────────────────────────────────────
//  ANALYSIS TAB
// ─────────────────────────────────────────────────────
async function triggerAnalysis() {
    const sym = getSymbol();
    await pollJobProgress({
        type:    'analyse',
        symbol:  sym,
        title:   `Analyzing ${sym}…`,
        desc:    `Deploying advanced neural networks to map market structure, liquidity voids, and momentum divergences for ${sym}.`,
        onDone:  (result) => {
            if (result) updateAnalysisDashboard(result);
            fetchStatus();
        },
    });
}

async function tryLoadAnalysisPrediction(sym) {
    try {
        const res = await fetch(`/outputs/analysis/${sym.toLowerCase()}_analysis.json`);
        if (res.ok) {
            const data = await res.json();
            updateAnalysisDashboard(data);
        }
    } catch (_) {}
}

function updateAnalysisDashboard(data) {
    if (!data) return;
    
    setIfExists('current-date-analysis', `Last updated: ${data.analyzed_at}`);
    setIfExists('analysis-subtitle', `Comprehensive 7-layer technical breakdown for ${data.symbol}`);
    
    // Layer 1
    setIfExists('an-trend', data.layer1_structure?.trend);
    setIfExists('an-trend-reason', data.layer1_structure?.trend_reason);
    const anTrendEl = document.getElementById('an-trend');
    if (anTrendEl) {
        const t = data.layer1_structure?.trend;
        anTrendEl.style.color = t === 'UPTREND' ? 'var(--bullish-green)' : (t === 'DOWNTREND' ? 'var(--bearish-red)' : 'var(--text-muted)');
    }

    // Layer 2
    setIfExists('an-price', fmt(data.layer2_levels?.current_price));
    
    const resLvl = data.layer2_levels?.nearest_resistance;
    if (resLvl) {
        setIfExists('an-res', `${fmt(resLvl.price)}`);
    } else {
        setIfExists('an-res', 'None');
    }
    
    const supLvl = data.layer2_levels?.nearest_support;
    if (supLvl) {
        setIfExists('an-sup', `${fmt(supLvl.price)}`);
    } else {
        setIfExists('an-sup', 'None');
    }

    // Layer 3
    const candleOk = data.layer3_patterns?.has_confirmation;
    const todayP = data.layer3_patterns?.today_patterns || [];
    setIfExists('an-candle', todayP.length ? todayP[0].name : 'No Pattern');
    setIfExists('an-candle-desc', todayP.length ? todayP.map(p => p.description).join(', ') : 'No significant pattern on the latest candle.');
    
    // Layer 4
    setIfExists('an-vol', data.layer4_volume?.classification?.replace(/_/g, ' '));
    setIfExists('an-vol-desc', data.layer4_volume?.meaning);

    // Layer 5
    setIfExists('an-adx', `ADX: ${data.layer5_trend?.adx}`);
    setIfExists('an-adx-desc', data.layer5_trend?.adx_note);

    // Layer 6
    setIfExists('an-rsi', `RSI: ${data.layer6_momentum?.rsi} (${data.layer6_momentum?.rsi_zone})`);
    setIfExists('an-rsi-desc', data.layer6_momentum?.rsi_note);

    // Checklist
    const cl = data.checklist;
    if (cl) {
        setIfExists('an-score', `${cl.score}/${cl.total}`);
        setIfExists('an-advice', cl.advice);
        
        const rEl = document.getElementById('an-rating');
        if (rEl) {
            rEl.textContent = cl.rating;
            rEl.style.background = cl.rating_color === 'green' ? 'rgba(16,185,129,0.2)' : 
                                  (cl.rating_color === 'gold' ? 'rgba(245,158,11,0.2)' : 'rgba(239,68,68,0.2)');
            rEl.style.color = cl.rating_color === 'green' ? 'var(--bullish-green)' : 
                             (cl.rating_color === 'gold' ? 'var(--accent-orange)' : 'var(--bearish-red)');
        }
        
        const ul = document.getElementById('checklist-ul');
        if (ul && cl.items) {
            ul.innerHTML = cl.items.map(item => `
                <li style="padding: 12px 15px; border-bottom: 1px solid rgba(255,255,255,0.05); display: flex; align-items: flex-start; gap: 12px;">
                    <div style="color: ${item.pass ? 'var(--bullish-green)' : 'var(--bearish-red)'}; font-size: 1.1rem; margin-top: 2px;">
                        <i class="fa-solid ${item.pass ? 'fa-circle-check' : 'fa-circle-xmark'}"></i>
                    </div>
                    <div>
                        <div style="font-weight: 600; font-size: 0.95rem; margin-bottom: 3px;">
                            <span style="color: var(--accent-blue); margin-right: 6px;">${item.label}</span> 
                            ${item.question}
                        </div>
                        <div style="color: var(--text-muted); font-size: 0.85rem;">
                            <span style="background: rgba(255,255,255,0.05); padding: 2px 6px; border-radius: 4px; margin-right: 6px; font-family: monospace;">${item.value}</span>
                            ${item.detail}
                        </div>
                    </div>
                </li>
            `).join('');
        }
    }
}


// Resolve via sidebar form
async function resolveEntry() {
    const dateEl   = document.getElementById('resolve-date');
    const actualEl = document.getElementById('resolve-actual');
    const date     = dateEl?.value;
    const actual   = parseFloat(actualEl?.value);
    if (!date || isNaN(actual)) {
        showError('Please enter both a date and an actual close value.');
        return;
    }
    await submitResolve(date, actual);
    if (dateEl)   dateEl.value   = '';
    if (actualEl) actualEl.value = '';
}

// Resolve via modal
function openResolveModal(predDate, predClose, prevClose, direction) {
    const modal = document.getElementById('resolve-modal');
    if (!modal) return;

    const info = document.getElementById('modal-pred-info');
    if (info) {
        info.innerHTML = `
            <div><b>Prediction Date:</b> ${predDate}</div>
            <div><b>Previous Close:</b> ${fmt(parseFloat(prevClose))} pts</div>
            <div><b>Predicted Close:</b> ${fmt(parseFloat(predClose))} pts (${direction})</div>
        `;
    }
    setIfExists('modal-pred-date', predDate, 'value');
    document.getElementById('modal-actual-input').value = '';
    modal.style.display = 'flex';
}

function closeModal() {
    const modal = document.getElementById('resolve-modal');
    if (modal) modal.style.display = 'none';
}

async function submitModalResolve() {
    const date   = document.getElementById('modal-pred-date')?.value;
    const actual = parseFloat(document.getElementById('modal-actual-input')?.value);
    if (!date || isNaN(actual)) {
        alert('Please enter the actual close value.');
        return;
    }
    await submitResolve(date, actual);
    closeModal();
}


async function autocheckHistory() {
    const btns = [document.querySelector('#tab-history .btn'), document.getElementById('sidebar-autograde-btn')];
    btns.forEach(b => { if(b) b.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Grading...'; });
    try {
        const res = await fetch('/api/history/autocheck', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });
        const data = await res.json();
        if(data.status === 'success') {
            const msg = data.resolved > 0 ? `✅ Auto-grade complete! Resolved ${data.resolved} predictions.` : '⏳ No new predictions could be graded yet (data may not be available).';
            alert(msg);
            await loadHistory();
            await loadHistoryStats();
        } else {
            alert('Error: ' + data.message);
        }
    } catch(err) {
        alert('Network error during auto-check.');
    }
    btns.forEach(b => { if(b) b.innerHTML = '<i class="fa-solid fa-robot"></i> Auto-Grade Pending'; });
}


async function deleteEntry(id) {
    if (!confirm('Delete this prediction record?')) return;
    try {
        await fetch('/api/history/delete', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ id })
        });
        await loadHistory();
    } catch (e) {
        console.error('Delete failed:', e);
    }
}

// ─────────────────────────────────────────────────────
//  CONFIDENCE RING CHART (Doughnut)
// ─────────────────────────────────────────────────────
function initConfRing(pct) {
    const canvas = document.getElementById('confRing');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (confRingChart) confRingChart.destroy();
    confRingChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            datasets: [{
                data: [pct, 100 - pct],
                backgroundColor: ['rgba(59,130,246,0.8)', 'rgba(255,255,255,0.04)'],
                borderWidth: 0,
            }]
        },
        options: {
            cutout: '78%',
            responsive: true,
            maintainAspectRatio: true,
            plugins: { legend: { display: false }, tooltip: { enabled: false } },
            animation: { animateRotate: true, duration: 800 }
        }
    });
}

function updateConfRing(pct, direction) {
    const dir = (direction || '').toUpperCase();
    const color = dir === 'UP' ? 'rgba(16,185,129,0.85)' : (dir === 'DOWN' ? 'rgba(239,68,68,0.85)' : 'rgba(156,163,175,0.85)');
    if (confRingChart) {
        confRingChart.data.datasets[0].data = [pct, 100 - pct];
        confRingChart.data.datasets[0].backgroundColor = [color, 'rgba(255,255,255,0.04)'];
        confRingChart.update();
    } else {
        initConfRing(pct);
    }
}

// ─────────────────────────────────────────────────────
//  CHARTS — NEPSE
// ─────────────────────────────────────────────────────
function renderNepseChart(chartData, predictionsObj) {
    if (!chartData || !chartData.dates) return;
    const canvas = document.getElementById('nepseChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (nepseChart) nepseChart.destroy();

    const labels = [...chartData.dates];
    const actual = [...chartData.actual];
    const sma20 = [...chartData.sma20];
    const bb_upper = [...chartData.bb_upper];
    const bb_lower = [...chartData.bb_lower];
    const futureData = new Array(labels.length).fill(null);

    if (predictionsObj) {
        if (actual.length > 0) futureData[futureData.length - 1] = actual[actual.length - 1];
        
        const preds = Object.values(predictionsObj).filter(p => p && p.date && p.date > labels[labels.length - 1]);
        preds.sort((a, b) => a.date.localeCompare(b.date));
        
        preds.forEach(pred => {
            labels.push(pred.date);
            actual.push(null);
            sma20.push(null);
            bb_upper.push(null);
            bb_lower.push(null);
            futureData.push(pred.predicted_close);
        });
    }

    const blueGrad = ctx.createLinearGradient(0, 0, 0, 240);
    blueGrad.addColorStop(0, 'rgba(59,130,246,0.22)');
    blueGrad.addColorStop(1, 'rgba(59,130,246,0)');

    nepseChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'NEPSE Index',
                    data: actual,
                    borderColor: '#3b82f6',
                    borderWidth: 2.5,
                    backgroundColor: blueGrad,
                    fill: true,
                    tension: 0.25,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                },
                {
                    label: 'AI Forecast',
                    data: futureData,
                    borderColor: '#8b5cf6',
                    borderWidth: 2.5,
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.25,
                    pointRadius: 3,
                    pointHoverRadius: 5,
                    pointBackgroundColor: '#8b5cf6'
                },
                {
                    label: 'SMA 20',
                    data: sma20,
                    borderColor: '#f59e0b',
                    borderWidth: 1.5,
                    borderDash: [5,5],
                    fill: false,
                    pointRadius: 0,
                    tension: 0.2,
                },
                {
                    label: 'Upper BB',
                    data: bb_upper,
                    borderColor: 'rgba(139,92,246,0.4)',
                    borderWidth: 1,
                    fill: false,
                    pointRadius: 0,
                },
                {
                    label: 'Lower BB',
                    data: bb_lower,
                    borderColor: 'rgba(139,92,246,0.4)',
                    borderWidth: 1,
                    fill: '-1',
                    backgroundColor: 'rgba(139,92,246,0.03)',
                    pointRadius: 0,
                }
            ]
        },
        options: chartOptions()
    });
}

function renderNepsePerfChart(chartData, predictionsObj) {
    if (!chartData || !chartData.dates) return;
    const canvas = document.getElementById('nepsePerfChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (nepsePerfChart) nepsePerfChart.destroy();

    const labels = [...chartData.dates];
    const a30_full = chartData.actual ? [...chartData.actual] : [];
    const p30_full = chartData.predicted ? [...chartData.predicted] : labels.map(date => {
        const match = historyData.find(e => 
            (e.symbol === 'NEPSE' || e.type === 'index' || e.type === 'nepse_index') &&
            e.prediction_date === date
        );
        return match ? match.predicted_close : null;
    });

    if (predictionsObj) {
        const preds = Object.values(predictionsObj).filter(p => p && p.date && p.date > labels[labels.length - 1]);
        preds.sort((a, b) => a.date.localeCompare(b.date));
        
        preds.forEach(pred => {
            labels.push(pred.date);
            a30_full.push(null);
            p30_full.push(pred.predicted_close);
        });
    }

    const d30 = labels.slice(-35);
    const a30 = a30_full.slice(-35);
    const p30 = p30_full.slice(-35);

    nepsePerfChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: d30,
            datasets: [
                { label: 'Actual', data: a30, borderColor: '#3b82f6', borderWidth: 2, pointRadius: 2.5, fill: false, tension: 0.15 },
                { label: 'AI Predicted', data: p30, borderColor: '#8b5cf6', borderWidth: 2, borderDash: [4,4], pointRadius: 2, fill: false, tension: 0.15 }
            ]
        },
        options: perfChartOptions()
    });
}

// ─────────────────────────────────────────────────────
//  CHARTS — STOCK
// ─────────────────────────────────────────────────────
function renderStockChart(chartData, predictionsObj) {
    if (!chartData || !chartData.dates) return;
    const canvas = document.getElementById('priceChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (stockChart) stockChart.destroy();

    const labels = [...chartData.dates];
    const actual = [...chartData.actual];
    const sma20 = [...chartData.sma20];
    const bb_upper = [...chartData.bb_upper];
    const bb_lower = [...chartData.bb_lower];
    const volume = [...(chartData.volume || [])];
    const futureData = new Array(labels.length).fill(null);

    if (predictionsObj) {
        if (actual.length > 0) futureData[futureData.length - 1] = actual[actual.length - 1];
        
        const preds = Object.values(predictionsObj).filter(p => p && p.date && p.date > labels[labels.length - 1]);
        preds.sort((a, b) => a.date.localeCompare(b.date));
        
        preds.forEach(pred => {
            labels.push(pred.date);
            actual.push(null);
            sma20.push(null);
            bb_upper.push(null);
            bb_lower.push(null);
            volume.push(0);
            futureData.push(pred.predicted_close);
        });
    }

    const blueGrad = ctx.createLinearGradient(0, 0, 0, 240);
    blueGrad.addColorStop(0, 'rgba(59,130,246,0.22)');
    blueGrad.addColorStop(1, 'rgba(59,130,246,0)');

    stockChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Actual Close',
                    data: actual,
                    borderColor: '#3b82f6',
                    borderWidth: 2.5,
                    backgroundColor: blueGrad,
                    fill: true,
                    tension: 0.2,
                    pointRadius: 0,
                    yAxisID: 'y'
                },
                {
                    label: 'AI Forecast',
                    data: futureData,
                    borderColor: '#8b5cf6',
                    borderWidth: 2.5,
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.25,
                    pointRadius: 3,
                    pointHoverRadius: 5,
                    pointBackgroundColor: '#8b5cf6',
                    yAxisID: 'y'
                },
                {
                    label: 'SMA 20',
                    data: sma20,
                    borderColor: '#f59e0b',
                    borderWidth: 1.5,
                    borderDash: [5,5],
                    fill: false, pointRadius: 0,
                    tension: 0.2,
                    yAxisID: 'y'
                },
                {
                    label: 'Upper BB',
                    data: bb_upper,
                    borderColor: 'rgba(139,92,246,0.4)',
                    borderWidth: 1, fill: false, pointRadius: 0,
                    yAxisID: 'y'
                },
                {
                    label: 'Lower BB',
                    data: bb_lower,
                    borderColor: 'rgba(139,92,246,0.4)',
                    borderWidth: 1, fill: '-1',
                    backgroundColor: 'rgba(139,92,246,0.03)',
                    pointRadius: 0, yAxisID: 'y'
                },
                {
                    label: 'Volume',
                    type: 'bar',
                    data: volume,
                    backgroundColor: 'rgba(255,255,255,0.06)',
                    barPercentage: 0.6,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            ...chartOptions(),
            scales: {
                x: xScale(),
                y:  { ...yScale(), position: 'left' },
                y1: {
                    position: 'right',
                    grid: { drawOnChartArea: false },
                    ticks: { color: '#4b5563', font: { size: 9 }, maxTicksLimit: 4 }
                }
            }
        }
    });
}

function renderStockPerfChart(chartData, predictionsObj) {
    if (!chartData || !chartData.dates) return;
    const canvas = document.getElementById('perfChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (stockPerfChart) stockPerfChart.destroy();

    const sym = getSymbol();
    const timeframe = activeTimeframe || '1D';

    const labels = [...chartData.dates];
    const a30_full = chartData.actual ? [...chartData.actual] : [];
    const p30_full = chartData.predicted ? [...chartData.predicted] : labels.map(date => {
        const match = historyData.find(e => {
            const isSymMatch = e.symbol === sym || e.id?.split('_')[0]?.toUpperCase() === sym;
            const isDateMatch = e.prediction_date === date;
            const entryTimeframe = e.timeframe || '1D';
            return isSymMatch && isDateMatch && entryTimeframe === timeframe;
        });
        return match ? match.predicted_close : null;
    });

    if (predictionsObj) {
        const preds = Object.values(predictionsObj).filter(p => p && p.date && p.date > labels[labels.length - 1]);
        preds.sort((a, b) => a.date.localeCompare(b.date));
        
        preds.forEach(pred => {
            labels.push(pred.date);
            a30_full.push(null);
            p30_full.push(pred.predicted_close);
        });
    }

    const d30 = labels.slice(-35);
    const a30 = a30_full.slice(-35);
    const p30 = p30_full.slice(-35);

    stockPerfChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: d30,
            datasets: [
                { label: 'Actual', data: a30, borderColor: '#3b82f6', borderWidth: 2, pointRadius: 2.5, fill: false, tension: 0.1 },
                { label: 'Predicted', data: p30, borderColor: '#8b5cf6', borderWidth: 2, borderDash: [4,4], pointRadius: 2, fill: false, tension: 0.1 }
            ]
        },
        options: perfChartOptions()
    });
}

function renderFeatureChart(importances) {
    const canvas = document.getElementById('featureChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (featureChart) featureChart.destroy();

    const labels = Object.keys(importances);
    const values = Object.values(importances);

    const grad = ctx.createLinearGradient(0, 0, 150, 0);
    grad.addColorStop(0, 'rgba(59,130,246,0.8)');
    grad.addColorStop(1, 'rgba(139,92,246,0.8)');

    featureChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{ data: values, backgroundColor: grad, borderRadius: 4, borderWidth: 0 }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false }, tooltip: tooltipConfig() },
            scales: {
                x: { display: false },
                y: {
                    grid: { display: false, borderColor: 'transparent' },
                    ticks: { color: '#94a3b8', font: { size: 9 } }
                }
            }
        }
    });
}

// ─────────────────────────────────────────────────────
//  CHART CONFIGS (shared)
// ─────────────────────────────────────────────────────
function chartOptions() {
    return {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: { legend: { display: false }, tooltip: tooltipConfig() },
        scales: { x: xScale(), y: yScale() }
    };
}
function perfChartOptions() {
    return {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'top',
                labels: { boxWidth: 14, color: '#94a3b8', font: { size: 10 }, padding: 10 }
            },
            tooltip: tooltipConfig()
        },
        scales: { x: xScale(), y: yScale() }
    };
}
function xScale() {
    return {
        grid: { color: 'rgba(255,255,255,0.03)', borderColor: 'transparent' },
        ticks: { color: '#6b7280', font: { size: 9 }, maxTicksLimit: 10 }
    };
}
function yScale() {
    return {
        grid: { color: 'rgba(255,255,255,0.03)', borderColor: 'transparent' },
        ticks: { color: '#9ca3af', font: { size: 10 } }
    };
}
function tooltipConfig() {
    return {
        backgroundColor: 'rgba(13,18,32,0.97)',
        titleFont: { family: 'Outfit', size: 12, weight: 'bold' },
        bodyFont: { family: 'Plus Jakarta Sans', size: 11 },
        borderColor: 'rgba(255,255,255,0.08)',
        borderWidth: 1,
        padding: 10,
        cornerRadius: 8
    };
}

// ─────────────────────────────────────────────────────
//  LOADING OVERLAY
// ─────────────────────────────────────────────────────
function showLoader(title, desc) {
    setIfExists('loader-title', title);
    setIfExists('loader-desc',  desc);
    const el = document.getElementById('loading-overlay');
    if (el) el.style.display = 'flex';
}
function hideLoader() {
    const el = document.getElementById('loading-overlay');
    if (el) el.style.display = 'none';
}
function showError(msg) {
    hideLoader();
    alert(`⚠️ ${msg}`);
}
function setButtonsDisabled(disabled) {
    ['btn-nepse-update', 'btn-nepse-train', 'btn-update', 'btn-train', 'btn-update-inner', 'btn-train-inner', 'btn-nepse-update-inner', 'btn-nepse-train-inner'].forEach(id => {
        const btn = document.getElementById(id);
        if (btn) btn.disabled = disabled;
    });
}

// ─────────────────────────────────────────────────────
//  HELPERS
// ─────────────────────────────────────────────────────
function getSymbol() {
    const sel = document.getElementById('symbol-select');
    if (!sel) return 'LICN';
    if (sel.value === 'custom') {
        return (document.getElementById('custom-symbol')?.value.trim().toUpperCase()) || 'LICN';
    }
    return sel.value;
}

function setIfExists(id, value, prop = 'textContent') {
    const el = document.getElementById(id);
    if (el) el[prop] = value ?? '—';
}

function fmt(n) {
    if (n == null || isNaN(n)) return '—';
    return Number(n).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

// ─────────────────────────────────────────────────────
//  LIVE AI ASSISTANT CHAT
// ─────────────────────────────────────────────────────
function toggleChat() {
    const panel = document.getElementById('chat-panel');
    const fab = document.getElementById('chat-fab');
    if (panel.style.display === 'none' || !panel.style.display) {
        panel.style.display = 'flex';
        fab.style.display = 'none';
        document.getElementById('chat-input').focus();
    } else {
        panel.style.display = 'none';
        fab.style.display = 'flex';
    }
}

function handleChatKey(event) {
    if (event.key === 'Enter') {
        sendChatMessage();
    }
}

async function sendChatMessage() {
    const input = document.getElementById('chat-input');
    const message = input.value.trim();
    if (!message) return;

    // Determine symbol context if user didn't type it
    // Attempt to extract from message, otherwise fallback to active symbol
    const words = message.toUpperCase().split(' ');
    let symbolContext = getSymbol(); // fallback
    if (currentTab === 'nepse') symbolContext = 'NEPSE';
    
    // Add user message
    appendChatMessage('user', message);
    input.value = '';
    
    // Add loading indicator
    const loadingId = 'msg-' + Date.now();
    appendChatMessage('ai', '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing...', loadingId);
    
    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message, symbol: symbolContext })
        });
        const data = await res.json();
        
        // Remove loading
        const loader = document.getElementById(loadingId);
        if (loader) loader.remove();
        
        // Format markdown briefly
        let reply = data.reply || "I didn't get a response.";
        reply = reply.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        reply = reply.replace(/\n/g, '<br>');
        
        appendChatMessage('ai', reply);
    } catch (e) {
        const loader = document.getElementById(loadingId);
        if (loader) loader.remove();
        appendChatMessage('ai', '⚠️ Error communicating with the assistant.');
    }
}

function appendChatMessage(role, htmlContent, id = '') {
    const msgs = document.getElementById('chat-messages');
    const div = document.createElement('div');
    div.className = `chat-msg ${role}`;
    if (id) div.id = id;
    
    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';
    bubble.innerHTML = htmlContent;
    
    div.appendChild(bubble);
    msgs.appendChild(div);
    
    // Scroll to bottom
    msgs.scrollTop = msgs.scrollHeight;
}
