"""
Phishing URL Detection — Flask Web Application
=================================================
A Flask-based web app that uses a trained CNN+LSTM deep learning model
to predict whether a given URL is phishing or legitimate.

Usage:
    source venv/bin/activate
    python app.py
"""

import os
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF info logs

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, jsonify, render_template

# ─── Configuration ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'phishing_model.keras')
TOKENIZER_PATH = os.path.join(BASE_DIR, 'models', 'tokenizer.pkl')
MAX_LENGTH = 250

# ─── App Setup ───────────────────────────────────────────────────
app = Flask(__name__)

# ─── Load Model & Tokenizer ─────────────────────────────────────
try:
    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"ERROR: Failed to load model: {e}")
    print("Please run 'python train.py' first to train and save the model.")
    model = None

try:
    print(f"Loading tokenizer from {TOKENIZER_PATH}...")
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"ERROR: Failed to load tokenizer: {e}")
    print("Please run 'python train.py' first to generate the tokenizer.")
    tokenizer = None


# ─── Routes ──────────────────────────────────────────────────────
@app.route('/')
def home():
    """Serve the main UI."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Predict whether a URL is phishing or legitimate."""
    if model is None or tokenizer is None:
        return jsonify({
            'error': 'Model not loaded. Run train.py first.'
        }), 503

    data = request.json
    if not data:
        return jsonify({'error': 'No JSON body provided.'}), 400

    url = data.get('url', '').strip()
    if not url:
        return jsonify({'error': 'No URL provided.'}), 400

    try:
        # Tokenize and pad
        seq = tokenizer.texts_to_sequences([url])
        padded = pad_sequences(seq, maxlen=MAX_LENGTH, padding='post')

        # Predict
        raw_prediction = float(model.predict(padded, verbose=0)[0][0])
        is_phishing = raw_prediction < 0.5
        confidence = round((1 - raw_prediction if is_phishing else raw_prediction) * 100, 1)

        return jsonify({
            'url': url,
            'is_phishing': is_phishing,
            'confidence': confidence,
            'raw_score': round(raw_prediction, 4)
        })

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'tokenizer_loaded': tokenizer is not None,
    })


# ─── Main ────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
