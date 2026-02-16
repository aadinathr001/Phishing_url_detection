/**
 * PhishGuard — Frontend Logic
 * ===========================
 * Handles URL scanning via API, result display, scan history,
 * and background particle animations.
 */

// ─── Constants ──────────────────────────────────
const HISTORY_KEY = 'phishguard_history';
const MAX_HISTORY = 20;

// ─── DOM Elements ───────────────────────────────
const urlInput = document.getElementById('urlInput');
const scanBtn = document.getElementById('scanBtn');
const resultArea = document.getElementById('resultArea');
const resultCard = document.getElementById('resultCard');
const resultIcon = document.getElementById('resultIcon');
const resultTitle = document.getElementById('resultTitle');
const resultUrl = document.getElementById('resultUrl');
const confidenceFill = document.getElementById('confidenceFill');
const confidenceValue = document.getElementById('confidenceValue');
const historySection = document.getElementById('historySection');
const historyList = document.getElementById('historyList');

// ─── Initialize ─────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    createParticles();
    loadHistory();

    // Scan on Enter key
    urlInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') checkURL();
    });
});

// ─── URL Scanning ───────────────────────────────
async function checkURL() {
    const url = urlInput.value.trim();

    if (!url) {
        shakeInput();
        return;
    }

    setLoading(true);
    resultArea.style.display = 'none';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url })
        });

        const data = await response.json();

        if (data.error) {
            showError(data.error);
        } else {
            showResult(data);
            addToHistory(data);
        }
    } catch (err) {
        showError('Could not connect to the server. Is the app running?');
    } finally {
        setLoading(false);
    }
}

// ─── Display Result ─────────────────────────────
function showResult(data) {
    const isPhishing = data.is_phishing;
    const confidence = data.confidence;

    resultCard.className = 'result-card ' + (isPhishing ? 'phishing' : 'safe');
    resultIcon.textContent = isPhishing ? '⚠️' : '✅';
    resultTitle.textContent = isPhishing ? 'Phishing Detected!' : 'URL is Safe';
    resultUrl.textContent = data.url;

    // Animate confidence bar
    confidenceFill.style.width = '0%';
    confidenceValue.textContent = '';

    resultArea.style.display = 'block';

    // Slight delay for animation
    requestAnimationFrame(() => {
        setTimeout(() => {
            confidenceFill.style.width = confidence + '%';
            confidenceValue.textContent = confidence + '%';
        }, 100);
    });
}

function showError(message) {
    resultCard.className = 'result-card phishing';
    resultIcon.textContent = '❌';
    resultTitle.textContent = 'Error';
    resultUrl.textContent = message;
    confidenceFill.style.width = '0%';
    confidenceValue.textContent = '';
    resultArea.style.display = 'block';
}

// ─── Loading State ──────────────────────────────
function setLoading(loading) {
    if (loading) {
        scanBtn.classList.add('loading');
        scanBtn.disabled = true;
    } else {
        scanBtn.classList.remove('loading');
        scanBtn.disabled = false;
    }
}

// ─── Input Shake Animation ──────────────────────
function shakeInput() {
    urlInput.style.animation = 'shake 0.4s ease-out';
    urlInput.style.borderColor = 'var(--danger-red)';
    setTimeout(() => {
        urlInput.style.animation = '';
        urlInput.style.borderColor = '';
    }, 500);
}

// ─── History Management ─────────────────────────
function addToHistory(data) {
    let history = getHistory();
    history.unshift({
        url: data.url,
        is_phishing: data.is_phishing,
        confidence: data.confidence,
        time: new Date().toLocaleTimeString()
    });

    if (history.length > MAX_HISTORY) {
        history = history.slice(0, MAX_HISTORY);
    }

    localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    renderHistory(history);
}

function getHistory() {
    try {
        return JSON.parse(localStorage.getItem(HISTORY_KEY)) || [];
    } catch {
        return [];
    }
}

function loadHistory() {
    const history = getHistory();
    if (history.length > 0) {
        renderHistory(history);
    }
}

function renderHistory(history) {
    if (history.length === 0) {
        historySection.style.display = 'none';
        return;
    }

    historySection.style.display = 'block';
    historyList.innerHTML = history.map(item => `
        <li class="history-item">
            <div class="history-dot ${item.is_phishing ? 'phishing' : 'safe'}"></div>
            <span class="history-url" title="${escapeHtml(item.url)}">${escapeHtml(item.url)}</span>
            <span class="history-badge ${item.is_phishing ? 'phishing' : 'safe'}">
                ${item.is_phishing ? 'Phishing' : 'Safe'} · ${item.confidence}%
            </span>
        </li>
    `).join('');
}

function clearHistory() {
    localStorage.removeItem(HISTORY_KEY);
    historySection.style.display = 'none';
    historyList.innerHTML = '';
}

// ─── Utility ────────────────────────────────────
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ─── Background Particles ───────────────────────
function createParticles() {
    const container = document.getElementById('bgParticles');
    const count = 30;
    const colors = ['#00d2ff', '#7b2ff7', '#00e676', '#ff1744'];

    for (let i = 0; i < count; i++) {
        const particle = document.createElement('div');
        particle.classList.add('particle');
        const size = Math.random() * 6 + 2;
        particle.style.width = size + 'px';
        particle.style.height = size + 'px';
        particle.style.left = Math.random() * 100 + '%';
        particle.style.background = colors[Math.floor(Math.random() * colors.length)];
        particle.style.animationDuration = (Math.random() * 15 + 10) + 's';
        particle.style.animationDelay = (Math.random() * 10) + 's';
        container.appendChild(particle);
    }
}

// ─── Shake Keyframe (injected dynamically) ──────
const style = document.createElement('style');
style.textContent = `
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        20% { transform: translateX(-8px); }
        40% { transform: translateX(8px); }
        60% { transform: translateX(-4px); }
        80% { transform: translateX(4px); }
    }
`;
document.head.appendChild(style);
