"""
Phishing URL Detection — Model Training Script
=================================================
Trains a CNN+LSTM deep learning model to classify URLs as phishing (0) or
legitimate (1) using character-level tokenization.

Usage:
    # Activate the virtual environment first
    source venv/bin/activate
    python train.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score,
    f1_score, accuracy_score
)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF info logs

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, Conv1D, MaxPooling1D, LSTM, Dense,
    Dropout, GlobalMaxPooling1D, Reshape
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ─── Configuration ───────────────────────────────────────────────
DATASET_PATH = os.path.join(os.path.dirname(__file__), 'dataset.csv')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'phishing_model.keras')
TOKENIZER_PATH = os.path.join(MODEL_DIR, 'tokenizer.pkl')

MAX_LENGTH = 250
EMBEDDING_DIM = 64
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.00011
PATIENCE = 5
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# ─── Ensure output directory exists ─────────────────────────────
os.makedirs(MODEL_DIR, exist_ok=True)


def load_and_split_data():
    """Load dataset.csv and split into train/val/test."""
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    print(f"  Total samples: {len(df)}")
    print(f"  Class distribution:\n{df['label'].value_counts().to_string()}\n")

    X = df['url'].astype(str).values
    y = df['label'].values

    # First split: train+val vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )

    # Second split: train vs val
    val_ratio = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio,
        random_state=42, stratify=y_train_val
    )

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def tokenize_data(X_train, X_val, X_test):
    """Character-level tokenization and padding."""
    print("\nTokenizing (character-level)...")
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(X_train)

    vocab_size = len(tokenizer.word_index) + 1
    print(f"  Vocabulary size: {vocab_size}")

    X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train),
                                maxlen=MAX_LENGTH, padding='post')
    X_val_pad = pad_sequences(tokenizer.texts_to_sequences(X_val),
                              maxlen=MAX_LENGTH, padding='post')
    X_test_pad = pad_sequences(tokenizer.texts_to_sequences(X_test),
                               maxlen=MAX_LENGTH, padding='post')

    return tokenizer, vocab_size, X_train_pad, X_val_pad, X_test_pad


def build_model(vocab_size):
    """Build the CNN+LSTM model."""
    print("\nBuilding CNN+LSTM model...")
    model = Sequential([
        # Embedding
        Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_LENGTH),
        # CNN layers
        Conv1D(64, 3, activation='relu'),
        MaxPooling1D(2),
        Conv1D(128, 3, activation='relu'),
        MaxPooling1D(2),
        # Global pooling → reshape for LSTM
        GlobalMaxPooling1D(),
        Reshape((1, 128)),
        # LSTM
        LSTM(128, return_sequences=False),
        # Dense classifier
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid'),
    ])

    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    model.summary()
    return model


def train_model(model, X_train, y_train, X_val, y_val):
    """Train with early stopping."""
    print("\nTraining...")
    early_stop = EarlyStopping(
        monitor='val_loss', patience=PATIENCE,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
    )
    return history


def evaluate_model(model, X_test, y_test):
    """Evaluate and print metrics."""
    print("\n" + "=" * 50)
    print("EVALUATION ON TEST SET")
    print("=" * 50)

    predictions = (model.predict(X_test) > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)

    print(f"  Accuracy:    {accuracy:.4f}")
    print(f"  Precision:   {precision:.4f}")
    print(f"  Recall:      {recall:.4f}")
    print(f"  F1-Score:    {f1:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print("=" * 50)

    return accuracy


def save_artifacts(model, tokenizer):
    """Save the trained model and tokenizer."""
    print(f"\nSaving model to {MODEL_PATH}...")
    model.save(MODEL_PATH)

    print(f"Saving tokenizer to {TOKENIZER_PATH}...")
    with open(TOKENIZER_PATH, 'wb') as f:
        pickle.dump(tokenizer, f)

    print("Done! All artifacts saved.\n")


def main():
    print("=" * 50)
    print("PHISHING URL DETECTION — MODEL TRAINING")
    print("=" * 50 + "\n")

    # 1. Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data()

    # 2. Tokenize
    tokenizer, vocab_size, X_train_pad, X_val_pad, X_test_pad = \
        tokenize_data(X_train, X_val, X_test)

    # 3. Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    # 4. Build model
    model = build_model(vocab_size)

    # 5. Train
    train_model(model, X_train_pad, y_train_enc, X_val_pad, y_val_enc)

    # 6. Evaluate
    evaluate_model(model, X_test_pad, y_test_enc)

    # 7. Save
    save_artifacts(model, tokenizer)


if __name__ == '__main__':
    main()
