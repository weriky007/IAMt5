import MetaTrader5 as mt5
from model import build_lstm_model
from data_preprocessor import preprocess_data
import numpy as np


class TradingBot:
    def __init__(self, symbol, timeframe, lookback=60):
        self.symbol = symbol
        self.timeframe = timeframe
        self.lookback = lookback
        self.model = build_lstm_model((lookback, 1))

    def predict_next_price(self, data):
        last_sequence = data[-self.lookback:]
        last_sequence = np.reshape(last_sequence, (1, self.lookback, 1))
        prediction = self.model.predict(last_sequence)
        return prediction[0][0]

    def execute_trade(self, prediction, current_price):
        if prediction > current_price:
            print("Sinal de COMPRA")
            # Implemente a lógica de compra via MT5
        else:
            print("Sinal de VENDA")
            # Implemente a lógica de venda via MT5


# Exemplo de uso
bot = TradingBot("EURUSD", mt5.TIMEFRAME_H1)
data = fetch_data("EURUSD", mt5.TIMEFRAME_H1, 100)
X, y, scaler = preprocess_data(data)
current_price = data['close'].iloc[-1]
prediction = bot.predict_next_price(X)
bot.execute_trade(prediction, current_price)