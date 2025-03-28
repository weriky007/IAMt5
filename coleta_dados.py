import pandas as pd
import MetaTrader5 as mt5

def fetch_data(symbol, timeframe, n_candles):
    mt5.symbol_select(symbol, True)
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_candles)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

# Exemplo: df = fetch_data("EURUSD", mt5.TIMEFRAME_H1, 1000)