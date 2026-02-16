"""
PhishGuard — AI-Powered Phishing URL Detection
================================================
A Streamlit web app that uses a trained CNN+LSTM deep learning model
to predict whether a given URL is phishing or legitimate.

Deploy on Streamlit Cloud or run locally:
    streamlit run app.py
"""

import os
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF info logs

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st

# ─── Configuration ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'phishing_model.keras')
TOKENIZER_PATH = os.path.join(BASE_DIR, 'models', 'tokenizer.pkl')
MAX_LENGTH = 250


# ─── Load Model & Tokenizer (cached) ────────────────────────────
@st.cache_resource
def load_model():
    """Load the trained TensorFlow model."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


@st.cache_resource
def load_tokenizer():
    """Load the saved tokenizer."""
    try:
        with open(TOKENIZER_PATH, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load tokenizer: {e}")
        return None


def predict_url(url, model, tokenizer):
    """Predict whether a URL is phishing or legitimate."""
    seq = tokenizer.texts_to_sequences([url])
    padded = pad_sequences(seq, maxlen=MAX_LENGTH, padding='post')
    raw_prediction = float(model.predict(padded, verbose=0)[0][0])
    is_phishing = raw_prediction < 0.5
    confidence = round((1 - raw_prediction if is_phishing else raw_prediction) * 100, 1)
    return is_phishing, confidence, raw_prediction


# ─── Page Config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="PhishGuard — AI Phishing URL Detector",
    page_icon="🛡️",
    layout="centered",
)

# ─── Custom CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;600;700;800&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #111827 100%);
    }

    .main-header {
        text-align: center;
        padding: 2rem 0 1rem;
    }

    .main-header h1 {
        font-family: 'Outfit', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        color: #f1f5f9;
        margin-bottom: 0;
    }

    .main-header h1 span {
        background: linear-gradient(135deg, #00d2ff 0%, #7b2ff7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .tagline {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    .result-safe {
        background: rgba(0, 230, 118, 0.08);
        border: 1px solid rgba(0, 230, 118, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    .result-safe h3 {
        color: #00e676;
        font-family: 'Outfit', sans-serif;
        font-size: 1.4rem;
        margin: 0;
    }

    .result-phishing {
        background: rgba(255, 23, 68, 0.08);
        border: 1px solid rgba(255, 23, 68, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    .result-phishing h3 {
        color: #ff1744;
        font-family: 'Outfit', sans-serif;
        font-size: 1.4rem;
        margin: 0;
    }

    .confidence-text {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }

    .info-cards {
        display: flex;
        gap: 1rem;
        margin-top: 2rem;
    }

    .info-card {
        flex: 1;
        background: rgba(17, 24, 39, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        backdrop-filter: blur(20px);
    }

    .info-card .icon { font-size: 1.8rem; margin-bottom: 0.5rem; }
    .info-card h4 { color: #f1f5f9; font-family: 'Outfit', sans-serif; margin: 0.3rem 0; }
    .info-card p { color: #64748b; font-size: 0.82rem; line-height: 1.4; }

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.8rem;
        padding: 2rem 0 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── Header ──────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🛡️ Phish<span>Guard</span></h1>
</div>
<p class="tagline">AI-Powered Phishing URL Detection</p>
""", unsafe_allow_html=True)


# ─── Load Resources ──────────────────────────────────────────────
model = load_model()
tokenizer = load_tokenizer()

if model is None or tokenizer is None:
    st.error("⚠️ Model not loaded. Please make sure `models/phishing_model.keras` and `models/tokenizer.pkl` exist.")
    st.info("Run `python train.py` first to train and save the model.")
    st.stop()


# ─── URL Scanner ─────────────────────────────────────────────────
st.markdown("### 🔍 Scan a URL")
st.markdown("Enter any URL below and our deep learning model will analyze it for phishing indicators.")

url_input = st.text_input(
    "URL",
    placeholder="e.g., https://example.com/path",
    label_visibility="collapsed"
)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    scan_clicked = st.button("🔎 Scan URL", use_container_width=True, type="primary")

# ─── Results ─────────────────────────────────────────────────────
if scan_clicked:
    if not url_input.strip():
        st.warning("Please enter a URL to scan.")
    else:
        with st.spinner("Analyzing URL..."):
            is_phishing, confidence, raw_score = predict_url(url_input.strip(), model, tokenizer)

        if is_phishing:
            st.markdown(f"""
            <div class="result-phishing">
                <h3>⚠️ Phishing Detected!</h3>
                <p style="color: #94a3b8; margin: 0.4rem 0 0; font-size: 0.85rem;
                   overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{url_input}</p>
                <p class="confidence-text">Confidence: <strong style="color: #ff1744;">{confidence}%</strong></p>
            </div>
            """, unsafe_allow_html=True)
            st.progress(confidence / 100)
        else:
            st.markdown(f"""
            <div class="result-safe">
                <h3>✅ URL is Safe</h3>
                <p style="color: #94a3b8; margin: 0.4rem 0 0; font-size: 0.85rem;
                   overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{url_input}</p>
                <p class="confidence-text">Confidence: <strong style="color: #00e676;">{confidence}%</strong></p>
            </div>
            """, unsafe_allow_html=True)
            st.progress(confidence / 100)


# ─── Info Section ────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class="info-cards">
    <div class="info-card">
        <div class="icon">🧠</div>
        <h4>Deep Learning</h4>
        <p>CNN+LSTM neural network trained on 480K+ URLs for accurate detection.</p>
    </div>
    <div class="info-card">
        <div class="icon">⚡</div>
        <h4>Instant Results</h4>
        <p>Real-time phishing analysis in under a second with confidence scores.</p>
    </div>
    <div class="info-card">
        <div class="icon">🔒</div>
        <h4>Privacy First</h4>
        <p>All analysis happens locally. Your URLs are never stored or shared.</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    Built with TensorFlow & Streamlit | PhishGuard © 2026
</div>
""", unsafe_allow_html=True)
