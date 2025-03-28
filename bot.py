import asyncio
import pickle
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import gym
from gym import spaces
from sklearn.linear_model import SGDClassifier  # Suporta partial_fit
from sklearn.utils.validation import check_is_fitted
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback  # Para early stopping
import threading
from datetime import datetime, timedelta
import multiprocessing as mp
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging
from logging.handlers import RotatingFileHandler
from concurrent.futures import ThreadPoolExecutor

# Configuração do logger com rotação
logger = logging.getLogger('trade_monitor')
logger.setLevel(logging.INFO)
handler = RotatingFileHandler('monitoring_log.csv', maxBytes=10*1024*1024, backupCount=5)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Funções de decodificação para cada modelo
def decode_ml_signal(signal):
    if signal == 1 or signal == 1.0:
        return "Compra"
    elif signal == 0 or signal == 0.0:
        return "Venda"
    else:
        return "Indefinido"

def decode_rl_signal(signal):
    if signal == 1:
        return "Compra"
    elif signal == 2:
        return "Venda"
    elif signal == 0:
        return "Hold"
    else:
        return "Indefinido"

def decode_integrated_signal(signal):
    if signal is None:
        return "Esperar"
    elif signal == 1:
        return "Compra"
    elif signal == 0:
        return "Venda"
    else:
        return "Indefinido"

# Variável global para acesso thread-safe ao modelo RL
RL_MODEL_LOCK = threading.Lock()

# =============================
# Conectar ao MetaTrader 5 e coletar o saldo inicial
# =============================
if not mt5.initialize():
    print("❌ Erro ao conectar ao MetaTrader 5!")
    quit()
account_info = mt5.account_info()
if account_info is None:
    print("❌ Erro ao obter informações da conta!")
    mt5.shutdown()
    quit()
saldo_inicial = account_info.balance
print(f"💰 Saldo Inicial: {saldo_inicial:.2f}")

# =============================
# Variáveis globais de configuração
# =============================
TEMPO_DADOS_COLETADOS = 43200    # número de candles de 1 minuto
TIME_TREINO = 1980  # 33 min
INTERVALO_PREDICOES = 15
RL_MODEL = None

symbol = 'AUDCAD'
TIMEFRAME = mt5.TIMEFRAME_M1
risk_per_trade = 0.01  # 1% do capital por trade (para gerenciamento de capital)
VOLUME_DEFAULT = 1.0   # Valor padrão, se não calcular dinamicamente
STOPS = 39
ATR_PERIOD = 30
ATR_SL_MULTIPLIER = 1.0
ATR_TP_MULTIPLIER = 2.0

SPREAD_COST = 0.0002
COMMISSION = 0.0001
SLIPPAGE_FACTOR = 0.0001

# → Função para coletar dados históricos com timestamp (dados recentes)
def get_historical_rates(symbol, timeframe, candles):
    # Cada candle tem duração de 1 minuto
    start_time = datetime.now() - timedelta(minutes=candles)
    end_time = datetime.now()
    # Converte os datetime para timestamps inteiros
    rates = mt5.copy_rates_from(symbol, timeframe, int(start_time.timestamp()), int(end_time.timestamp()))
    # Se não houver dados, tenta usar o método copy_rates_from_pos como fallback
    if rates is None or len(rates) == 0:
        print("⚠️ copy_rates_from() não retornou dados, tentando copy_rates_from_pos()")
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, candles)
    return rates

# =============================
# Função auxiliar para adicionar indicadores técnicos
# =============================
def add_indicators(df):
    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))
    # MACD
    ema_short = df["close"].ewm(span=12, adjust=False).mean()
    ema_long = df["close"].ewm(span=26, adjust=False).mean()
    df["macd_line"] = ema_short - ema_long
    df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd_line"] - df["macd_signal"]
    # Bollinger Bands
    df["bollinger_middle"] = df["close"].rolling(window=20).mean()
    df["bollinger_std"] = df["close"].rolling(window=20).std()
    df["bollinger_upper"] = df["bollinger_middle"] + 2 * df["bollinger_std"]
    df["bollinger_lower"] = df["bollinger_middle"] - 2 * df["bollinger_std"]
    # ADX
    df["prev_close"] = df["close"].shift(1)
    df["tr"] = df.apply(lambda row: max(row["high"] - row["low"],
                                        abs(row["high"] - row["prev_close"]),
                                        abs(row["low"] - row["prev_close"])), axis=1)
    df["up_move"] = df["high"] - df["high"].shift(1)
    df["down_move"] = df["low"].shift(1) - df["low"]
    df["plus_dm"] = np.where((df["up_move"] > df["down_move"]) & (df["up_move"] > 0), df["up_move"], 0)
    df["minus_dm"] = np.where((df["down_move"] > df["up_move"]) & (df["down_move"] > 0), df["down_move"], 0)
    df["tr_sum"] = df["tr"].rolling(window=14).sum()
    df["plus_dm_sum"] = df["plus_dm"].rolling(window=14).sum()
    df["minus_dm_sum"] = df["minus_dm"].rolling(window=14).sum()
    df["plus_di"] = 100 * (df["plus_dm_sum"] / df["tr_sum"])
    df["minus_di"] = 100 * (df["minus_dm_sum"] / df["tr_sum"])
    df["dx"] = 100 * abs(df["plus_di"] - df["minus_di"]) / (df["plus_di"] + df["minus_di"])
    df["adx"] = df["dx"].rolling(window=14).mean()
    # Estocástico
    df["lowest_low"] = df["low"].rolling(window=14).min()
    df["highest_high"] = df["high"].rolling(window=14).max()
    df["stochastic_k"] = 100 * (df["close"] - df["lowest_low"]) / (df["highest_high"] - df["lowest_low"])
    return df

# =============================
# Classe TradingBot (Modelo ML incremental com validação holdout)
# =============================
class TradingBot:
    def __init__(self, symbol):
        self.model_path = "modelo_ml.pkl"
        self.symbol = symbol
        self.ml_model = self.load_or_train_model()

    def load_or_train_model(self):
        try:
            with open(self.model_path, "rb") as f:
                model = pickle.load(f)
            check_is_fitted(model)
            print("✅ Modelo ML carregado com sucesso!")
            return model
        except (FileNotFoundError, pickle.UnpicklingError, AttributeError, ValueError):
            print("⚠️ Modelo ML não encontrado! Criando um novo modelo incremental.")
            model = SGDClassifier(loss='log_loss', learning_rate='optimal', max_iter=1000, tol=1e-3)
            model.partial_fit(np.zeros((1, 11)), [0], classes=np.array([0, 1]))
            return model

    def train_model(self):
        try:
            print(f"📊 Atualizando modelo ML com dados recentes...")
            rates = get_historical_rates(self.symbol, TIMEFRAME, TEMPO_DADOS_COLETADOS)
            if rates is None or len(rates) < 1000:
                raise ValueError("Dados insuficientes para treinamento incremental")
            df = pd.DataFrame(rates)
            df["feature1"] = df["close"] - df["open"]
            df["feature2"] = df["high"] - df["low"]
            df["sma_5"] = df["close"].rolling(window=5).mean()
            df["sma_10"] = df["close"].rolling(window=10).mean()
            df["sma_20"] = df["close"].rolling(window=20).mean()
            df = add_indicators(df).dropna()
            df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
            df = df.dropna()
            features = df[["feature1", "feature2", "sma_5", "sma_10", "sma_20",
                           "rsi", "macd_hist", "bollinger_upper", "bollinger_lower",
                           "adx", "stochastic_k"]]
            target = df["target"]
            X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, shuffle=False)
            self.ml_model.partial_fit(X_train.values, y_train.values)
            y_pred = self.ml_model.predict(X_val.values)
            val_acc = accuracy_score(y_val, y_pred)
            print(f"🔍 Validação Holdout - Accuracy: {val_acc:.2f}")
            if val_acc < 0.5:
                print("⚠️ Acurácia baixa. Reinicializando modelo ML.")
                self.ml_model = SGDClassifier(loss='log_loss', learning_rate='optimal', max_iter=1000, tol=1e-3)
                self.ml_model.partial_fit(np.zeros((1, 11)), [0], classes=np.array([0, 1]))
            with open(self.model_path, "wb") as f:
                pickle.dump(self.ml_model, f)
            print("✅ Modelo ML atualizado incrementalmente, validado e salvo!")
            return self.ml_model
        except Exception as e:
            print(f"❌ Erro no treinamento incremental: {str(e)}")
            return self.ml_model

    def calculate_atr(self, period=ATR_PERIOD):
        rates = mt5.copy_rates_from_pos(self.symbol, TIMEFRAME, 0, period + 1)
        if rates is None or len(rates) < period + 1:
            print("❌ Erro: Dados insuficientes para ATR.")
            return None
        df = pd.DataFrame(rates)
        df["prev_close"] = df["close"].shift(1)
        df["tr1"] = df["high"] - df["low"]
        df["tr2"] = (df["high"] - df["prev_close"]).abs()
        df["tr3"] = (df["low"] - df["prev_close"]).abs()
        df["tr"] = df[["tr1", "tr2", "tr3"]].max(axis=1)
        atr = df["tr"].rolling(window=period).mean().iloc[-1]
        return atr

    def get_live_data(self):
        rates = mt5.copy_rates_from_pos(self.symbol, TIMEFRAME, 0, 30)
        if rates is None or len(rates) < 30:
            print("⚠️ Dados insuficientes para formação de estado")
            return None
        times = pd.to_datetime(np.array(rates['time']), unit='s')
        if (pd.Series(times).diff().dropna() > pd.Timedelta(minutes=2)).any():
            print("⚠️ Gaps detectados nos dados ao vivo")
            return None
        df = pd.DataFrame(rates)
        df["feature1"] = df["close"] - df["open"]
        df["feature2"] = df["high"] - df["low"]
        df["sma_5"] = df["close"].rolling(window=5).mean()
        df["sma_10"] = df["close"].rolling(window=10).mean()
        df["sma_20"] = df["close"].rolling(window=20).mean()
        df = add_indicators(df).dropna()
        features = ["feature1", "feature2", "sma_5", "sma_10", "sma_20",
                    "rsi", "macd_hist", "bollinger_upper", "bollinger_lower",
                    "adx", "stochastic_k"]
        state_df = df[features].iloc[-1:].copy()
        ml_signal = self.predict_signal(state_df)
        state_array = np.append(state_df.values.flatten(), ml_signal)
        return state_array

    def predict_signal(self, df):
        if df is None or self.ml_model is None:
            return None
        return self.ml_model.predict(df.values)[0]

    def get_trade_params(self, signal):
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None or tick.ask == 0 or tick.bid == 0:
            return None, None, None
        price = tick.ask if signal == 1 else tick.bid
        atr = self.calculate_atr()
        if atr is None:
            return None, None, None
        if signal == 1:
            tp = price + (atr * ATR_TP_MULTIPLIER)
            sl = price - (atr * ATR_SL_MULTIPLIER)
            if sl >= price:
                print("⚠️ SL inválido para operação de compra")
                return None, None, None
        else:
            tp = price - (atr * ATR_TP_MULTIPLIER)
            sl = price + (atr * ATR_SL_MULTIPLIER)
            if sl <= price:
                print("⚠️ SL inválido para operação de venda")
                return None, None, None
        return price, tp, sl

    def send_order(self, signal):
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None or tick.ask == 0 or tick.bid == 0:
            print("❌ Tick inválido")
            return
        spread = (tick.ask - tick.bid) * 10000
        if spread > 30:
            print(f"❌ Spread muito alto: {spread} pontos")
            return
        price, tp, sl = self.get_trade_params(signal)
        if price is None:
            print("❌ Não foi possível calcular os parâmetros do trade. Ordem não enviada.")
            return
        order_type = mt5.ORDER_TYPE_BUY if signal == 1 else mt5.ORDER_TYPE_SELL
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            print("❌ Erro ao obter informações do símbolo.")
            return
        stops_level = getattr(symbol_info, "stops_level", None) or STOPS
        min_distance = stops_level * symbol_info.point
        if abs(price - sl) < min_distance:
            sl = price - min_distance if signal == 1 else price + min_distance
        if abs(tp - price) < min_distance:
            tp = price + min_distance if signal == 1 else price - min_distance
        digits = symbol_info.digits
        price, sl, tp = round(price, digits), round(sl, digits), round(tp, digits)
        print(f"DEBUG: price={price}, SL={sl}, TP={tp}")
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": VOLUME_DEFAULT,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 10,
            "magic": 0,
            "comment": "Trade ML ATR",
            "type_filling": mt5.ORDER_FILLING_FOK,
            "type_time": mt5.ORDER_TIME_GTC,
        }
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"✅ Ordem {decode_ml_signal(signal)} executada com sucesso!")
            print(f"""🚀 Ordem enviada:
Ativo: {self.symbol}
Tipo: {decode_ml_signal(signal)}
Preço: {price:.5f}
SL: {sl:.5f}
TP: {tp:.5f}
Volume: {VOLUME_DEFAULT}
""")
        else:
            print(f"❌ Erro ao enviar ordem: {result.comment}")

    async def update_model_periodically(self):
        while True:
            await asyncio.sleep(TIME_TREINO)
            print("🔄 Atualizando o modelo ML de forma incremental...")
            self.train_model()
            # Modelo é atualizado via partial_fit

    async def manage_positions(self, new_signal):
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            print("❌ Erro ao obter posições:", mt5.last_error())
            return
        if positions:
            for pos in positions:
                current_type = pos.type  # 0 = BUY, 1 = SELL
                if (new_signal == 1 and current_type == 1) or (new_signal == 0 and current_type == 0):
                    print("🔄 Sinal oposto detectado. Tentando fechar a posição aberta...")
                    ticket = pos.ticket
                    volume = pos.volume
                    close_tick = mt5.symbol_info_tick(self.symbol)
                    if close_tick is None:
                        print(f"❌ Erro ao obter tick para fechar posição do {self.symbol}")
                        continue
                    close_price = close_tick.bid if current_type == 0 else close_tick.ask
                    close_type = mt5.ORDER_TYPE_BUY if current_type == 0 else mt5.ORDER_TYPE_SELL
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "position": ticket,
                        "symbol": self.symbol,
                        "volume": volume,
                        "type": close_type,
                        "price": close_price,
                        "deviation": 10,
                        "magic": 0,
                        "comment": "Fechamento de posição oposta",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_FOK,
                    }
                    result = mt5.order_send(request)
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        print("✅ Posição fechada com sucesso!")
                        for _ in range(5):
                            if not mt5.positions_get(ticket=ticket):
                                break
                            await asyncio.sleep(1)
                    else:
                        print("❌ Erro ao fechar posição:", result.comment)

    async def run(self):
        while True:
            if not mt5.initialize():
                print("❌ Reconectando ao MT5...")
                await asyncio.sleep(5)
                continue
            state_vector = self.get_live_data()  # vetor de shape (12,)
            if state_vector is None:
                await asyncio.sleep(15)
                continue
            with RL_MODEL_LOCK:
                rl_action, _ = RL_MODEL.predict(state_vector, deterministic=True)
            ml_signal = state_vector[-1]
            if rl_action == 1:
                final_signal = 1 if ml_signal == 1 else None
            elif rl_action == 2:
                final_signal = 0 if ml_signal == 0 else None
            else:
                final_signal = None
            print(f"ML signal: {decode_ml_signal(ml_signal)}, RL action: {decode_rl_signal(rl_action)}, final integrated signal: {decode_integrated_signal(final_signal)}")
            if final_signal is not None:
                await self.manage_positions(final_signal)
                positions = mt5.positions_get(symbol=self.symbol)
                if not positions:
                    self.send_order(final_signal)
            await asyncio.sleep(INTERVALO_PREDICOES)

# =============================
# Classe TradingEnv (Ambiente de RL para treinamento)
# =============================
class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, trading_bot, initial_balance=10000):
        super(TradingEnv, self).__init__()
        self.bot = trading_bot
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.current_step = 0
        self.max_steps = 100
        historical = get_historical_rates(symbol, TIMEFRAME, TEMPO_DADOS_COLETADOS)
        if historical is None or len(historical) == 0:
            raise ValueError("Dados históricos não disponíveis.")
        self.historical_data = pd.DataFrame(historical)
        self.historical_data = add_indicators(self.historical_data)
        self.historical_data["feature1"] = self.historical_data["close"] - self.historical_data["open"]
        self.historical_data["feature2"] = self.historical_data["high"] - self.historical_data["low"]
        self.historical_data["sma_5"] = self.historical_data["close"].rolling(window=5).mean()
        self.historical_data["sma_10"] = self.historical_data["close"].rolling(window=10).mean()
        self.historical_data["sma_20"] = self.historical_data["close"].rolling(window=20).mean()
        self.historical_data.dropna(inplace=True)
        self.historical_data.reset_index(drop=True, inplace=True)
        self.data_index = 0
        self.features = ["feature1", "feature2", "sma_5", "sma_10", "sma_20",
                         "rsi", "macd_hist", "bollinger_upper", "bollinger_lower",
                         "adx", "stochastic_k"]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.features)+1,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

    def reset(self):
        self.balance = self.initial_balance
        self.current_step = 0
        self.data_index = 0
        state = self.historical_data.iloc[self.data_index][self.features].values.astype(np.float32)
        ml_signal = self.bot.predict_signal(pd.DataFrame([state], columns=self.features))
        state = np.append(state, ml_signal)
        return state

    def step(self, action):
        if self.data_index < len(self.historical_data)-1:
            current_row = self.historical_data.iloc[self.data_index]
            next_row = self.historical_data.iloc[self.data_index+1]
            state = current_row[self.features].values.astype(np.float32)
            next_state = next_row[self.features].values.astype(np.float32)
            current_close = current_row["close"]
            next_close = next_row["close"]
        else:
            tick_current = mt5.symbol_info_tick(symbol)
            tick_next = mt5.symbol_info_tick(symbol)
            if tick_current is None or tick_next is None:
                print("❌ Erro ao obter dados de tick em tempo real.")
                state = self.bot.get_live_data()
                next_state = state.copy()
                current_close = 0
                next_close = 0
            else:
                if action == 1:
                    current_price = tick_current.ask
                    next_price = tick_next.bid
                elif action == 2:
                    current_price = tick_current.bid
                    next_price = tick_next.ask
                else:
                    current_price = (tick_current.ask + tick_current.bid)/2
                    next_price = (tick_next.ask + tick_next.bid)/2
                state = self.bot.get_live_data()
                next_state = state.copy()
                current_close = current_price
                next_close = next_price

        if action == 1:
            reward = (next_close - current_close) * VOLUME_DEFAULT
            reward -= (SPREAD_COST + COMMISSION) * VOLUME_DEFAULT
            reward -= abs(next_close - current_close) * SLIPPAGE_FACTOR * VOLUME_DEFAULT
        elif action == 2:
            reward = (current_close - next_close) * VOLUME_DEFAULT
            reward -= (SPREAD_COST + COMMISSION) * VOLUME_DEFAULT
            reward -= abs(next_close - current_close) * SLIPPAGE_FACTOR * VOLUME_DEFAULT
        else:
            reward = 0

        self.balance += reward
        self.data_index += 1
        self.current_step += 1
        done = (self.current_step >= self.max_steps) or (self.data_index >= len(self.historical_data)-1)
        info = {"balance": self.balance}
        ml_signal = self.bot.predict_signal(pd.DataFrame([next_state], columns=self.features))
        next_state = np.append(next_state, ml_signal)
        return np.array(next_state, dtype=np.float32), reward, done, info

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Balance: {self.balance}")

# =============================
# Função de monitoramento da assertividade baseada em TP e SL
# =============================
async def monitor_accuracy_tp_sl(trading_bot):
    header = "timestamp,ml_pred,ml_price,ml_tp,ml_sl,ml_outcome,rl_pred,rl_price,rl_tp,rl_sl,rl_outcome,integrated_pred,integrated_price,integrated_tp,integrated_sl,integrated_outcome,current_close,new_close"
    logger.info(header)
    while True:
        state = trading_bot.get_live_data()
        if state is None:
            await asyncio.sleep(INTERVALO_PREDICOES)
            continue
        ml_pred = state[-1]
        with RL_MODEL_LOCK:
            rl_action, _ = RL_MODEL.predict(state, deterministic=True)
        if rl_action == 0:
            integrated_pred = None
        elif rl_action == 1 and ml_pred == 1:
            integrated_pred = 1
        elif rl_action == 2 and ml_pred == 0:
            integrated_pred = 0
        else:
            integrated_pred = None

        ml_price, ml_tp, ml_sl = trading_bot.get_trade_params(ml_pred) if ml_pred is not None else (None, None, None)
        rl_price, rl_tp, rl_sl = trading_bot.get_trade_params(1 if rl_action == 1 else 0) if rl_action is not None else (None, None, None)
        integrated_price, integrated_tp, integrated_sl = trading_bot.get_trade_params(integrated_pred) if integrated_pred is not None else (None, None, None)

        rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 1, 1)
        if rates is None or len(rates) == 0:
            await asyncio.sleep(INTERVALO_PREDICOES)
            continue
        current_close = rates[0]["close"]
        initial_time = rates[0]["time"]

        new_close = None
        while True:
            await asyncio.sleep(INTERVALO_PREDICOES)
            new_rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 1, 1)
            if new_rates is None or len(new_rates) == 0:
                continue
            if new_rates[0]["time"] != initial_time:
                new_close = new_rates[0]["close"]
                break

        def trade_outcome(signal, price, tp, sl, new_close):
            if price is None or tp is None or sl is None:
                return "N/A"
            if signal == 1:
                if new_close >= tp:
                    return "TP"
                elif new_close <= sl:
                    return "SL"
                else:
                    return "None"
            elif signal == 0:
                if new_close <= tp:
                    return "TP"
                elif new_close >= sl:
                    return "SL"
                else:
                    return "None"
            else:
                return "N/A"

        ml_outcome = trade_outcome(ml_pred, ml_price, ml_tp, ml_sl, new_close)
        rl_outcome = trade_outcome(1 if rl_action == 1 else 0, rl_price, rl_tp, rl_sl, new_close) if rl_action is not None else "N/A"
        integrated_outcome = trade_outcome(integrated_pred, integrated_price, integrated_tp, integrated_sl, new_close) if integrated_pred is not None else "N/A"

        timestamp = datetime.now().isoformat()
        log_line = f"{timestamp},{decode_ml_signal(ml_pred)},{ml_price},{ml_tp},{ml_sl},{ml_outcome},{decode_rl_signal(rl_action)},{rl_price},{rl_tp},{rl_sl},{rl_outcome},{decode_integrated_signal(integrated_pred)},{integrated_price},{integrated_tp},{integrated_sl},{integrated_outcome},{current_close},{new_close}"
        logger.info(log_line)
        print(f"Monitoramento: {log_line}")
        await asyncio.sleep(INTERVALO_PREDICOES)

# =============================
# Função de treinamento do agente RL usando multiprocessing via executor
# =============================
def train_rl_agent_mp():
    bot = TradingBot(symbol)
    env = TradingEnv(trading_bot=bot, initial_balance=saldo_inicial)
    eval_callback = EvalCallback(env, best_model_save_path='./logs/', log_path='./logs/', eval_freq=1000, deterministic=True)
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000, callback=eval_callback)
    model.save("dqn_trading_agent")
    print("✅ Agente de RL treinado e salvo!")
    return model

async def update_rl_model_periodically():
    global RL_MODEL
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as executor:
        while True:
            await asyncio.sleep(TIME_TREINO)
            print("🔄 Atualizando o agente RL...")
            new_model = await loop.run_in_executor(executor, train_rl_agent_mp)
            with RL_MODEL_LOCK:
                RL_MODEL = new_model
            print("✅ Agente RL atualizado com sucesso!")

# =============================
# Função principal
# =============================
async def main():
    rl_model = train_rl_agent_mp()
    global RL_MODEL
    RL_MODEL = rl_model
    bot = TradingBot(symbol)
    bot_task = bot.run()
    ml_update_task = bot.update_model_periodically()
    rl_update_task = update_rl_model_periodically()
    monitor_task = monitor_accuracy_tp_sl(bot)
    await asyncio.gather(
        bot_task,
        ml_update_task,
        rl_update_task,
        monitor_task
    )

if __name__ == "__main__":
    asyncio.run(main())
