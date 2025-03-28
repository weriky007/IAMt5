from model import build_lstm_model
from data_fetcher import fetch_data
from data_preprocessor import preprocess_data
import numpy as np

# Coleta e pré-processamento
df = fetch_data("EURUSD", mt5.TIMEFRAME_H1, 5000)
X, y, scaler = preprocess_data(df, lookback=60)

# Divisão de dados
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Treinamento
model = build_lstm_model((X_train.shape[1], 1))
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))