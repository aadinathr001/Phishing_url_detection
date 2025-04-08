from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load the TensorFlow model
model = tf.keras.models.load_model(r'C:\Users\amrut\PycharmProjects\pythonProject2\web app\aa_model_traintestsplit.keras')  # No .pkl, just directory


# Load the tokenizer
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    url = data.get('url')

    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    # Convert URL to tokenized sequence
    features = tokenizer.texts_to_sequences([url])  # Convert to numerical sequence
    features = pad_sequences(features, maxlen=100)  # Ensure correct input shape

    # Make prediction
    prediction = model.predict(features)[0][0]  # Assuming output is sigmoid (probability)

    return jsonify({'url': url, 'is_phishing': bool(prediction > 0.5)})  # Threshold at 0.5

if __name__ == '__main__':
    app.run(debug=True)
