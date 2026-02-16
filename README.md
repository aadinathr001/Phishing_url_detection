# 🛡️ PhishGuard — AI-Powered Phishing URL Detection

A Flask web application that uses a CNN+LSTM deep learning model to detect phishing URLs in real-time. Trained on 480K+ URLs with **97.24% accuracy**.

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/aadinathr001/Phishing_url_detection.git
cd Phishing_url_detection
```

### 2. Create & activate virtual environment
```bash
python3 -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model (first time only)
```bash
python train.py
```
> **Note:** Training takes ~15-20 minutes on CPU. The trained model and tokenizer will be saved to `models/`.

### 5. Run the app
```bash
python app.py
```
Open **http://127.0.0.1:5000** in your browser.

---

## 📂 Project Structure

```
Phishing_url_detection/
├── app.py                    # Flask web server
├── train.py                  # Model training script
├── dataset.csv               # 480K URL dataset (url, label)
├── requirements.txt          # Python dependencies (pinned)
├── models/
│   ├── phishing_model.keras  # Trained CNN+LSTM model
│   └── tokenizer.pkl         # Character-level tokenizer
├── templates/
│   └── index.html            # Main UI template
├── static/
│   ├── css/style.css         # Dark-themed glassmorphism CSS
│   └── js/script.js          # Frontend logic & animations
├── venv/                     # Python virtual environment
└── README.md                 # This file
```

---

## 🧠 Model Architecture

| Layer              | Details                     |
|--------------------|-----------------------------|
| Embedding          | vocab_size × 64, input_length=250 |
| Conv1D + MaxPool   | 64 filters, kernel=3, pool=2 |
| Conv1D + MaxPool   | 128 filters, kernel=3, pool=2 |
| GlobalMaxPooling1D | -                           |
| Reshape            | (1, 128)                    |
| LSTM               | 128 units                   |
| Dense + Dropout    | 128 units, ReLU, 20% dropout |
| Output             | 1 unit, Sigmoid             |

**Optimizer:** Adam (lr=0.00011)  
**Loss:** Binary Crossentropy  
**Early Stopping:** patience=5 on val_loss  

---

## 📊 Performance Metrics

| Metric       | Score   |
|-------------|---------|
| Accuracy    | 97.24%  |
| Precision   | 97.36%  |
| Recall      | 97.12%  |
| F1-Score    | 97.24%  |
| Specificity | 97.36%  |

---

## 🌐 API Reference

### `POST /predict`
```json
// Request
{ "url": "https://example.com/path" }

// Response
{
    "url": "https://example.com/path",
    "is_phishing": false,
    "confidence": 99.8,
    "raw_score": 0.998
}
```

### `GET /health`
Returns model and tokenizer loading status.

---

## 📋 Dataset

- **Source:** `dataset.csv` (480,005 URLs)
- **Labels:** `0` = phishing, `1` = legitimate
- **Split:** 70% train / 15% validation / 15% test

---

## 🛠️ Tech Stack

- **Backend:** Flask 3.1, TensorFlow 2.20
- **ML:** CNN+LSTM with character-level tokenization
- **Frontend:** Vanilla HTML/CSS/JS with glassmorphism design
- **Python:** 3.13+

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.
