# 🛡️ PhishGuard — AI-Powered Phishing URL Detection

A Streamlit web application that uses a CNN+LSTM deep learning model to detect phishing URLs in real-time. Trained on 480K+ URLs with **97.24% accuracy**.

🔗 **Live Demo:** [phishguard-ai-url-detection.streamlit.app](https://phishguard-ai-url-detection.streamlit.app/)

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
streamlit run app.py
```
The app will open at **http://localhost:8501** in your browser.

---

## ☁️ Streamlit Cloud Deployment

This app is designed for one-click deployment to [Streamlit Cloud](https://streamlit.io/cloud):

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select your repo, branch `main`, and main file `app.py`
4. Click **Deploy**

> The `requirements.txt` includes all necessary dependencies. The model and tokenizer in `models/` are committed to the repo, so no extra setup is needed.

---

## 📂 Project Structure

```
Phishing_url_detection/
├── app.py                    # Streamlit web app
├── train.py                  # Model training script
├── dataset.csv               # 480K URL dataset (url, label)
├── requirements.txt          # Python dependencies (pinned)
├── models/
│   ├── phishing_model.keras  # Trained CNN+LSTM model
│   └── tokenizer.pkl         # Character-level tokenizer
├── venv/                     # Python virtual environment (local only)
└── README.md                 # This file
```

---

## 🧠 Model Architecture

| Layer              | Details                     |
|--------------------|-----------------------------|
| Embedding          | vocab_size × 64, input_length=250 |
| Conv1D + MaxPool   | 64 filters, kernel=3, pool=2 |
| Conv1D + MaxPool   | 128 filters, kernel=3, pool=2 |
| GlobalMaxPooling1D | —                           |
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

## 🛠️ Tech Stack

- **Frontend:** Streamlit 1.54 with custom CSS
- **ML:** TensorFlow 2.20, CNN+LSTM with character-level tokenization
- **Training:** scikit-learn, pandas, numpy, matplotlib
- **Deployment:** Streamlit Cloud
- **Python:** 3.13+

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.
