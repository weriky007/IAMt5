import asyncio
import pickle
import glob
import os
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import gymnasium as gym
from gymnasium import spaces
import threading
from datetime import datetime, timedelta

import logging
from logging.handlers import RotatingFileHandler

from concurrent.futures import ThreadPoolExecutor

import time
import asyncio

from sklearn.linear_model import SGDClassifier  # Suporta partial_fit
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback  # Para early stopping
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.buffers import ReplayBuffer, ReplayBufferSamples
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.dqn.policies import DQNPolicy
#from stable_baselines3.common.policies import register_policy

import torch as th
import requests
import math

from typing import Union
from typing import Union, Optional, Any  # Adicione conforme necessário

from colorama import init, Fore, Back

# ==================================================================================================================
# -----------------------------
# Variáveis globais
# -----------------------------
RL_MODEL_LOCK = threading.Lock()
TEMPO_DADOS_COLETADOS = 96300  # número de candles de 1 minuto
TIME_TREINO = 10800  # 33 min para treinamento incremental do ML e atualização do RL
INTERVALO_PREDICOES = 60 #60
TIME_STEPS = 900000
RL_MODEL = None

#symbol = 'EURUSD'
symbol = 'AUDCAD'
TIMEFRAME = mt5.TIMEFRAME_M1
risk_per_trade = 0.01  # 1% do capital por trade
STOPS = 54
ATR_PERIOD = 30
ATR_SL_MULTIPLIER = 1.5
ATR_TP_MULTIPLIER = 2.5
SPREAD_COST = 0.0002
COMMISSION = 0.0001
SLIPPAGE_FACTOR = 0.0001

TAMANHO_JANELA = 30

#CONFIGURACOES TREINO RL
SALDO_INICIAL_TREINO = 10000
N_INTERACOES = 10000  #MIN 100000
QUANTIDADE_ACERTOS_TREINO = 5 #MIN 5
MAX_CICLES = 50 #MIN 50
PASSOS_ATUALIZAR_REDE = 4 #MIN 4
PASSOS_BACKPROP = 2 #MIN 2

TRAILING_STEP = 0.0002  # 2 pips para pares de 5 decimais
TRAILING_MAX_DISTANCE = 0.0020  # 20 pips

TEMPO_MENSAGEM = 3600 # 1 hora
# ==================================================================================================================
# -----------------------------
# Cores print
# -----------------------------
init()
# ==================================================================================================================
# -----------------------------
# Dicionário de códigos de erro do MT5
# -----------------------------
MT5_ERROR_CODES = {
    10004: 'TRADE_RETCODE_REQUOTE',
    10006: 'TRADE_RETCODE_REJECT',
    10007: 'TRADE_RETCODE_CANCEL',
    10008: 'TRADE_RETCODE_PLACED',
    10009: 'TRADE_RETCODE_DONE',
    10010: 'TRADE_RETCODE_DONE_PARTIAL',
    10011: 'TRADE_RETCODE_ERROR',
    10012: 'TRADE_RETCODE_TIMEOUT',
    10013: 'TRADE_RETCODE_INVALID',
    10014: 'TRADE_RETCODE_TRADE_MODIFY_DENIED',
    10015: 'TRADE_RETCODE_TRADE_CONTEXT_BUSY',
    10016: 'TRADE_RETCODE_INVALID_PARAMS',
    10017: 'TRADE_RETCODE_INVALID_STOPS',
    10018: 'TRADE_RETCODE_TRADE_DISABLED'
}
# ==================================================================================================================
# -----------------------------
# Função para obter mensagem de erro
# -----------------------------
def get_error_message(error_code):
    return MT5_ERROR_CODES.get(error_code, 'Código de erro desconhecido')
# ==================================================================================================================
# -----------------------------
# Função para escolha do modo de operação
# -----------------------------
def choose_mode():
    #print(Fore.YELLOW + "Dentro do choose_mode")

    print(Fore.LIGHTWHITE_EX + "Escolha o modo de operação:")
    print(Fore.BLUE + "1. Apenas ML")
    print(Fore.BLUE + "2. Apenas RL")
    print(Fore.BLUE + "3. Integrado (Ambos)")
    choice = input(Fore.LIGHTWHITE_EX + "Digite a opção (1, 2 ou 3): ")
    if choice == "1":
        return "ml"
    elif choice == "2":
        return "rl"
    elif choice == "3":
        return "both"
    else:
        print(Fore.RED + "Opção inválida. Utilizando modo integrado por padrão.")
        return "both"
# ==================================================================================================================
# -----------------------------
# Função para escolher entre treinamento ou trade
# -----------------------------
def choose_training_mode():
    print(Fore.LIGHTWHITE_EX + "Escolha a opção de inicialização:")
    print(Fore.BLUE + "1. Entrar no ambiente de treinamento do RL")
    print(Fore.BLUE + "2. Iniciar diretamente o trading")
    choice = input(Fore.LIGHTWHITE_EX + "Digite a opção (1 ou 2): ")
    return choice == "1"
# ==================================================================================================================
# Variável global com o modo escolhido
OPERATION_MODE = choose_mode()

# Verifica se o arquivo de ordens existe e adiciona cabeçalho
if not os.path.exists('ordens.txt'):
    with open('ordens.txt', 'w', encoding='utf-8') as f:
        f.write("REGISTRO DE ORDENS\n")
        f.write("Formato: [DATA HORA] Balance: SALDO | Action: TIPO | Details: DETALHES | Outcome: RESULTADO | PnL: VALOR\n\n")

# ==================================================================================================================
# -----------------------------
# Classe para rastreamento de desempenho
# -----------------------------
# Adicione esta classe se não existir
class PerformanceTracker:
    def __init__(self):
        self.trades = []

    def add_trade(self, entry, exit, pnl):
        self.trades.append({'entry': entry, 'exit': exit, 'pnl': pnl})

    def summary(self):
        if not self.trades:
            return {}
        wins = [t for t in self.trades if t['pnl'] > 0]
        return {
            'total_trades': len(self.trades),
            'win_rate': len(wins) / len(self.trades),
            'avg_win': np.mean([t['pnl'] for t in wins]) if wins else 0,
            'avg_loss': np.mean([t['pnl'] for t in self.trades if t['pnl'] <= 0]) if (
                        len(self.trades) > len(wins)) else 0
        }

tracker = PerformanceTracker()

# ==================================================================================================================
# -----------------------------
# Variaveis bot Telegram
# -----------------------------
TELEGRAM_BOT_TOKEN = "7886922588:AAFRaCUWB82PhSs7Qk50jGA-1X44fgJ-5_M"
TELEGRAM_CHAT_ID = "-1002436254099"

# =============================
# Telegram mensagem
# =============================
async def send_initial_balance():
    #print(Fore.YELLOW + "Enviando saldo inicial...")
    try:
        account_info = mt5.account_info()
        if account_info:
            balance = account_info.balance
            message = f"✅ *Bot Iniciado!* \n💰 Saldo Atual: ${balance:.2f}"
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            params = {
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "Markdown"
            }
            response = requests.post(url, params=params)
            print(Fore.BLUE + f"✅ Mensagem inicial enviada: {response.status_code}")
        else:
            print(Fore.RED + "❌ Não foi possível obter o saldo inicial")
    except Exception as e:
        print(Fore.RED + f"❌ Erro ao enviar mensagem inicial: {str(e)}")

async def send_telegram_message():
    print(Fore.YELLOW + "Iniciando envio periódico de mensagens...")
    while True:
        await asyncio.sleep(TEMPO_MENSAGEM)
        await send_initial_balance()  # Reutiliza a mesma função de envio
# ==================================================================================================================
# -----------------------------
# Função para versionamento de modelos
# -----------------------------
def save_model(model, prefix="dqn_trading_agent"):
    model.save(f"{prefix}_best")  # Forçar salvamento da política
    #print(Fore.YELLOW + "Dentro do dqn_trading_agent")

    # model.save(f"{prefix}_best", policy=CustomDQNPolicy)  # Forçar salvamento da política
    # Salve o modelo sem especificar 'policy'

    # Salva o modelo com nome fixo para fácil acesso (SB3 adiciona .zip automaticamente)
    best_model_path = f"{prefix}_best"
    model.save(best_model_path)

    # Versionamento adicional (opcional)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_filename = f"{prefix}_{timestamp}.zip"
    model.save(versioned_filename)

    # Mantém apenas os 5 últimos modelos salvos
    model_files = sorted(glob.glob(f"{prefix}_*.zip"))
    for old_file in model_files[:-5]:
        try:
            os.remove(old_file)
        except Exception as e:
            print(Fore.BLUE + f"Erro ao remover o arquivo {old_file}: {e}")

# ==================================================================================================================
# Configuração do logger com rotação
logger = logging.getLogger('trade_monitor')
logger.setLevel(logging.INFO)
handler = RotatingFileHandler('monitoring_log.csv', maxBytes=10 * 1024 * 1024, backupCount=5)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# ==================================================================================================================
# -----------------------------
# Funções de decodificação
# -----------------------------
def decode_ml_signal(signal):
    #print(Fore.YELLOW + "Dentro do decode_ml_signal")

    if signal == 1 or signal == 1.0:
        return Fore.LIGHTGREEN_EX + "Compra"
    elif signal == 0 or signal == 0.0:
        return Fore.LIGHTGREEN_EX + "Venda"
    else:
        return Fore.LIGHTBLUE_EX + "Indefinido"

def decode_rl_signal(signal):
    #print(Fore.YELLOW + "Dentro do decode_rl_signal")

    if signal in [1, 2, 3]:  # Compra conservadora, moderada, agressiva
        return Fore.LIGHTGREEN_EX + "Compra"
    elif signal in [4, 5, 6]:  # Venda conservadora, moderada, agressiva
        return Fore.LIGHTGREEN_EX + "Venda"
    elif signal == 7:
        return Fore.BLUE + "Trail Stop"
    elif signal == 0:
        return Fore.LIGHTYELLOW_EX + "Esperar"
    else:
        return Fore.LIGHTBLUE_EX + "Indefinido"

def decode_integrated_signal(signal):
    #print(Fore.YELLOW + "Dentro do decode_integrated_signal")

    if signal is None:
        return Fore.LIGHTYELLOW_EX + "Esperar"
    elif signal == 1:
        return Fore.LIGHTGREEN_EX + "Compra"
    elif signal == 0:
        return Fore.LIGHTGREEN_EX + "Venda"
    else:
        return Fore.LIGHTBLUE_EX + "Indefinido"

# ==================================================================================================================
# -----------------------------
# Conexão com MT5
# -----------------------------
print(Fore.BLUE + "Conexão MT5")
if not mt5.initialize():
    print("❌ Erro ao conectar ao MetaTrader 5!")
    quit()
account_info = mt5.account_info()
if account_info is None:
    print("❌ Erro ao obter informações da conta!")
    mt5.shutdown()
    quit()
saldo_inicial = account_info.balance
print(Fore.BLUE + f"💰 Saldo Inicial: "+Fore.LIGHTGREEN_EX + f"${saldo_inicial:.2f}")

# ==================================================================================================================
# -----------------------------
# Função para coletar dados históricos com validação
# -----------------------------
def get_historical_rates(symbol, timeframe, candles, max_retries=3):
    """Obtém dados históricos com múltiplos fallbacks e retentativas"""
    attempts = 0
    rates = None

    while attempts < max_retries:
        try:
            if not mt5.initialize():
                print(Fore.RED + "⚠️ Falha na conexão MT5. Tentando reconectar...")
                mt5.shutdown()
                time.sleep(5)
                continue

            # Tentativa 1: Obter por período de tempo
            utc_from = datetime.now() - timedelta(minutes=candles * 2)
            rates = mt5.copy_rates_from(symbol, timeframe, utc_from, candles)

            if rates is not None and len(rates) >= candles:
                return rates

            # Tentativa 2: Obter a partir da posição zero
            print(Fore.YELLOW + "⚠️ Método 1 falhou. Tentando copy_rates_from_pos...")
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, candles)

            if rates is not None and len(rates) > 0:
                print(Fore.YELLOW + f"⚠️ Obteve apenas {len(rates)} candles via posição")
                return rates

            # Tentativa 3: Obter dados sintéticos se tudo falhar
            print(Fore.RED + "⚠️ Todos os métodos falharam. Gerando dados sintéticos...")
            return _generate_synthetic_data(candles)

        except Exception as e:
            print(Fore.RED + f"❌ Erro na tentativa {attempts + 1}: {str(e)}")
            attempts += 1
            time.sleep(5)

        finally:
            mt5.shutdown()

    raise ValueError(Fore.RED + "Falha crítica: Não foi possível obter dados históricos após múltiplas tentativas")

# ==================================================================================================================
def _generate_synthetic_data(candles):
    """Gera dados sintéticos de emergência"""
    base_price = 1.0000
    synthetic = []
    for i in range(candles):
        synthetic.append({
            'time': int((datetime.now() - timedelta(minutes=i)).timestamp()),
            'open': base_price + i * 0.0001,
            'high': base_price + i * 0.0001 + 0.0002,
            'low': base_price + i * 0.0001 - 0.0002,
            'close': base_price + i * 0.0001,
            'tick_volume': 1000
        })
    return synthetic

# ==================================================================================================================
class TemporalAttentionExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_features = observation_space.shape[0] // TAMANHO_JANELA
        self.lstm = th.nn.LSTM(input_size=n_features, hidden_size=128, batch_first=True)
        self.attention = th.nn.MultiheadAttention(embed_dim=128, num_heads=4)
        self.fc = th.nn.Linear(128, features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        batch_size = observations.size(0)
        seq = observations.view(batch_size, TAMANHO_JANELA, -1)
        lstm_out, _ = self.lstm(seq)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        return self.fc(attn_out[:, -1, :])

# ==================================================================================================================
class CustomDQNPolicy(DQNPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, net_arch=None, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=[256, 128],
            #features_extractor_class=TemporalAttentionExtractor,
            #features_extractor_kwargs={},
            **kwargs
        )

# ==================================================================================================================
class MetaDQN(DQN):
    #print(Fore.YELLOW + "Dentro do MetaDQN")

    def __init__(self, policy, env, *args, **kwargs):
        # Garanta que a política é passada para a classe pai
        super().__init__(policy, env, *args, **kwargs)
        #self.meta_optimizer = th.optim.Adam(self.policy.parameters(), lr=1e-3)
        #print(Fore.YELLOW + "Dentro do __init__ MetaDQN")

    def _setup_model(self):
        super()._setup_model()  # Chama o _setup_model() da classe DQN para criar a política
        self.meta_optimizer = th.optim.Adam(self.policy.parameters(), lr=1e-3)  # Agora self.pol

    def adapt(self, live_data=None, batch_size=32):
        #print(Fore.YELLOW + "Dentro do adapt MetaDQN")

        if live_data is not None:
            live_data = np.array(live_data)
            if live_data.ndim == 1:
                live_data = live_data.reshape(1, -1)

            obs_tensor = th.tensor(live_data, dtype=th.float32).to(self.device)

            with th.no_grad():
                actions, _ = self.predict(live_data)
                next_obs = obs_tensor.roll(-1, dims=0)
                rewards = th.randn(obs_tensor.size(0))

            for i in range(obs_tensor.size(0)):
                self.replay_buffer.add(
                    obs=obs_tensor[i].cpu().numpy().reshape(1, -1),
                    next_obs=next_obs[i].cpu().numpy().reshape(1, -1),
                    action=np.array([actions[i]], dtype=np.int64),
                    reward=np.array([rewards[i].item()], dtype=np.float32),
                    done=np.array([False], dtype=np.bool_),
                    infos=[{}]
                )

            if len(self.replay_buffer) > batch_size:
                losses = []
                for _ in range(3):
                    samples = self.replay_buffer.sample(batch_size)
                    loss = self._calc_loss(samples)
                    losses.append(loss.item())
                    self.meta_optimizer.zero_grad()
                    loss.backward()
                    self.meta_optimizer.step()
                return np.mean(losses)

    def _calc_loss(self, samples: ReplayBufferSamples) -> th.Tensor:
        #print(Fore.YELLOW + "Dentro do _calc_loss MetaDQN")

        with th.no_grad():
            # Acesso correto à rede target via política
            next_q_values = self.policy.q_net_target(samples.next_observations)
            target_q = next_q_values.max(dim=1)[0].view(-1, 1)
            target_q = samples.rewards + (1 - samples.dones) * self.gamma * target_q

        # Rede principal
        current_q = self.policy.q_net(samples.observations)
        current_q = current_q.gather(1, samples.actions.long())

        return th.nn.functional.mse_loss(current_q, target_q)

    def clone_for_scenario(self, env):
        #print(Fore.YELLOW + "Dentro do clone_for_scenario MetaDQN")

        cloned_model = MetaDQN(
            policy=self.policy_class,
            env=env,
            learning_rate=self.learning_rate,
            buffer_size=self.buffer_size,
            learning_starts=self.learning_starts,
            batch_size=self.batch_size,
            tau=self.tau,
            gamma=self.gamma,
            train_freq=self.train_freq,
            gradient_steps=self.gradient_steps,
            replay_buffer_class=self.replay_buffer_class,
            replay_buffer_kwargs=self.replay_buffer_kwargs,
            policy_kwargs=self.policy_kwargs,
            device=self.device
        )
        cloned_model.set_parameters(self.get_parameters())
        return cloned_model

# ==================================================================================================================
class TrendPrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size: int, observation_space: gym.Space, action_space: gym.Space,
                 device: Union[str, th.device] = "auto"):
        super().__init__(buffer_size, observation_space, action_space, device)
        self.priorities = np.zeros(buffer_size, dtype=np.float32)
        self.max_priority = 1.0
        self.size = 0  # Adicionar contador explícito

    def add(self, obs, next_obs, action, reward, done, infos):
        super().add(obs, next_obs, action, reward, done, infos)
        # Atualizar contador de tamanho
        self.size = min(self.size + 1, self.buffer_size)

    def __len__(self):
        return self.size  # Usar contador explícito

# ==================================================================================================================
class DynamicEarlyStopping(BaseCallback):
    #print(Fore.YELLOW + "Dentro do DynamicEarlyStopping")

    def __init__(self, patience=5):
        #print(Fore.YELLOW + "Dentro do __init__ DynamicEarlyStopping")

        super().__init__()
        self.patience = patience
        self.best_sharpe = -np.inf

    def _on_step(self):
        #print(Fore.YELLOW + "Dentro do _on_step DynamicEarlyStopping")

        returns = np.diff(self.model.balance_history)
        sharpe = np.mean(returns) / np.std(returns) if len(returns) > 0 else 0

        if sharpe > self.best_sharpe:
            self.best_sharpe = sharpe
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                return False
        return True

# ==================================================================================================================
# -----------------------------
# Nova função para obter dados em tempo real
# -----------------------------
def get_realtime_data(window_size=TAMANHO_JANELA):
    #print(Fore.YELLOW + "Dentro do get_realtime_data")

    bot = TradingBot(symbol)
    return bot.get_live_data(window_size)

# ==================================================================================================================
# -----------------------------
# Função de validação no mercado real
# -----------------------------
def validate_on_live_market(model, bot):
    #print(Fore.YELLOW + "Dentro do validate_on_live_market")

    print(Fore.BLUE + "\nIniciando validação em tempo real...")
    env = TradingEnv(trading_bot=bot, initial_balance=saldo_inicial)

    mean_reward, _ = evaluate_policy(
        model,
        env,
        n_eval_episodes=1, #3
        deterministic=True,
        return_episode_rewards=True
    )

    print(Fore.BLUE + f"Recompensa média na validação ao vivo: {np.mean(mean_reward):.2f}")

    # Reset se performance cair abaixo do threshold
    if np.mean(mean_reward) < -500:
        print(Fore.BLUE + "⚠️ Performance crítica detectada! Reiniciando modelo...")
        model = train_rl_agent_mp()

# ==================================================================================================================
async def train_hybrid():
    #print(Fore.YELLOW + "Dentro do train_hybrid")

    model = train_rl_agent_mp()
    bot = TradingBot(symbol)
    loop = asyncio.get_running_loop()

    try:
        while True:
            live_data = bot.get_live_data()

            if live_data is not None:
                model.adapt(live_data=live_data)

                if model.num_timesteps % 1000 == 0:
                    # Executar validação em segundo plano
                    await loop.run_in_executor(
                        None,
                        lambda: validate_on_live_market(model, bot)
                    )

            await asyncio.sleep(TIME_TREINO)

    except KeyboardInterrupt:
        print(Fore.BLUE + "\nInterrupção manual")

    return model

# ==================================================================================================================
class PerformanceDashboard:
    #print(Fore.YELLOW + "Dentro do PerformanceDashboard")

    def __init__(self):
        #print(Fore.YELLOW + "Dentro do __init__ PerformanceDashboard")

        self.metrics = {Fore.BLUE +
            'win_rate': [],
            'sharpe_ratio': [],
            'max_drawdown': [],
            'trend_capture_ratio': []
        }

    def update(self, trades):
        #print(Fore.YELLOW + "Dentro do update PerformanceDashboard")

        # Cálculo do TCR (Trend Capture Ratio)
        upside = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        downside = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
        self.metrics['trend_capture_ratio'].append(upside / downside if downside else 1)

# ==================================================================================================================
class ScenarioWrapper(gym.Wrapper):
    #print(Fore.YELLOW + "Dentro do ScenarioWrapper")

    def __init__(self, env, scenario_config):
        #print(Fore.YELLOW + "Dentro do __init__ ScenarioWrapper")

        super().__init__(env)  # Apenas 1 argumento: env
        self.scenario_type = scenario_config['type']
        self.severity = scenario_config.get('severity', 0.2)
        self.duration = scenario_config.get('duration', 100)
        self.current_step = 0

    def reset(self, **kwargs):
        #print(Fore.YELLOW + "Dentro do reset ScenarioWrapper")

        self.current_step = 0
        return super().reset(**kwargs)

    def step(self, action):
        #print(Fore.YELLOW + "Dentro do step ScenarioWrapper")

        self.current_step += 1

        if self.scenario_type == 'flash_crash':
            return self._apply_flash_crash(action)
        elif self.scenario_type == 'fomo_rally':
            return self._apply_fomo_rally(action)
        elif self.scenario_type == 'sideways':
            return self._apply_sideways(action)
        else:
            return super().step(action)

    def _apply_flash_crash(self, action):
        #print(Fore.YELLOW + "Dentro do _apply_flash_crash ScenarioWrapper")

        obs, reward, done, truncated, info = super().step(action)

        if self.current_step % 10 == 0:
            obs[0] *= (1 - self.severity)  # Modifica o preço (feature 0)

        return obs, reward, done, truncated, info

    def _apply_fomo_rally(self, action):
        #print(Fore.YELLOW + "Dentro do _apply_fomo_rally ScenarioWrapper")

        obs, reward, done, truncated, info = super().step(action)

        if self.current_step % 5 == 0:
            obs[0] *= (1 + self.severity)

        return obs, reward, done, truncated, info

    def _apply_sideways(self, action):
        #print(Fore.YELLOW + "Dentro do _apply_sideways ScenarioWrapper")

        obs, reward, done, truncated, info = super().step(action)
        return obs, reward, done, truncated, info

    def get_max_drawdown(self):
        #print(Fore.YELLOW + "Dentro do get_max_drawdown ScenarioWrapper")

        """Acessa o método do ambiente original"""
        return self.env.get_max_drawdown()

# ==================================================================================================================
def stress_test(model):
    #print(Fore.YELLOW + "Dentro do stress_test")

    if not mt5.initialize():
        print(Fore.RED + "❌ Falha na conexão ao MT5 durante o stress test!")
        return

    scenarios = [
        {'type': 'flash_crash', 'severity': 0.3},
        {'type': 'fomo_rally', 'severity': 0.4},
        {'type': 'sideways', 'duration': 500}
    ]

    bot = TradingBot(symbol)
    original_env = TradingEnv(trading_bot=bot, initial_balance=saldo_inicial)

    for scenario in scenarios:
        print(Fore.BLUE + f"\n🔥 Iniciando teste de estresse: {scenario['type']}")

        # Criar ambiente modificado
        test_env = ScenarioWrapper(original_env, scenario)

        # Avaliar o modelo sem treinar
        mean_reward, std_reward = evaluate_policy(
            model,
            test_env,
            n_eval_episodes=3,
            deterministic=True,
            warn=False
        )
        print(Fore.BLUE + f"Desempenho no cenário {scenario['type']}:")
        print(Fore.BLUE + f"Recompensa média: {mean_reward:.2f} ± {std_reward:.2f}")
        print(Fore.BLUE + f"Drawdown máximo: {test_env.get_max_drawdown():.2f}%")

# ==================================================================================================================
# -----------------------------
# Função auxiliar para adicionar indicadores técnicos
# -----------------------------
def add_indicators(df):
    #print(Fore.YELLOW + "Dentro do add_indicators")

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))
    ema_short = df["close"].ewm(span=12, adjust=False).mean()
    ema_long = df["close"].ewm(span=26, adjust=False).mean()
    df["macd_line"] = ema_short - ema_long
    df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd_line"] - df["macd_signal"]

    # df["bollinger_middle"] = df["close"].rolling(window=20).mean()
    # df["bollinger_std"] = df["close"].rolling(window=20).std()
    # df["bollinger_upper"] = df["bollinger_middle"] + 2 * df["bollinger_std"]
    # df["bollinger_lower"] = df["bollinger_middle"] - 2 * df["bollinger_std"]

    df["bollinger_middle"] = df["close"].rolling(window=20).mean()
    df["bollinger_std"] = df["close"].rolling(window=20).std()
    df["bollinger_upper"] = df["bollinger_middle"] + 2 * df["bollinger_std"]
    df["bollinger_lower"] = df["bollinger_middle"] - 2 * df["bollinger_std"]

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
    df["lowest_low"] = df["low"].rolling(window=14).min()
    df["highest_high"] = df["high"].rolling(window=14).max()
    df["stochastic_k"] = 100 * (df["close"] - df["lowest_low"]) / (df["highest_high"] - df["lowest_low"])
    return df

# ==================================================================================================================
# -----------------------------
# Função unificada para cálculo das features
# -----------------------------
def calculate_features(df):
    #print(Fore.YELLOW + "Dentro do calculate_features")

    df = df.copy()

    # Features básicas de preço
    df["feature1"] = df["close"] - df["open"]
    df["feature2"] = df["high"] - df["low"]

    # Médias móveis simples (SMA)
    for window in [5, 10, 20, 50, 200]:
        df[f"sma_{window}"] = df["close"].rolling(window=window).mean()

    # Indicadores técnicos tradicionais (supondo que add_indicators já os calcule, como RSI, MACD, Bollinger Bands, ADX, etc.)
    df = add_indicators(df)

    # Features de momentum e volatilidade já existentes
    df['momentum_5'] = df['close'].pct_change(5)
    df['momentum_15'] = df['close'].pct_change(15)
    df['volatility_30'] = df['close'].rolling(30).std()

    # ----------------------------------------
    # Novas features de contexto de mercado
    # ----------------------------------------

    # Indicadores de tendência: EMAs
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()

    # Indicadores de momentum: Rate of Change (ROC)
    df['roc_5'] = df['close'].pct_change(5)   # Taxa de mudança em 5 períodos
    df['roc_10'] = df['close'].pct_change(10)   # Taxa de mudança em 10 períodos

    # Volatilidade: Average True Range (ATR)
    # Primeiro, calcular a True Range (TR)
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = df.apply(lambda row: max(
        row['high'] - row['low'],
        abs(row['high'] - row['prev_close']),
        abs(row['low'] - row['prev_close'])
    ), axis=1)
    df['atr'] = df['tr'].rolling(14).mean()

    # Volume: Média móvel do tick_volume (ou volume, se tick_volume não estiver disponível)
    if 'tick_volume' in df.columns:
        df['volume_ma'] = df['tick_volume'].rolling(20).mean()
    elif 'volume' in df.columns:
        df['volume_ma'] = df['volume'].rolling(20).mean()
    else:
        df['volume_ma'] = 0

    # Sentimento do mercado: Distância do preço às bandas de Bollinger
    # (supondo que as colunas 'bollinger_upper' e 'bollinger_lower' já foram calculadas em add_indicators)
    df['price_distance_to_upper_band'] = df['close'] / df['bollinger_upper']
    df['price_distance_to_lower_band'] = df['close'] / df['bollinger_lower']

    # ----------------------------------------
    # Preenchimento de valores nulos
    df.ffill(inplace=True)
    df.fillna(0, inplace=True)

    # Padronização adaptativa das novas features de contexto e das existentes de momentum/volatilidade
    scaler = StandardScaler()
    features_to_scale = [
        'momentum_5', 'momentum_15', 'volatility_30',
        'roc_5', 'roc_10', 'atr', 'volume_ma',
        'price_distance_to_upper_band', 'price_distance_to_lower_band'
    ]
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    # Seleção final das features (incluindo todas as novas features de contexto)
    final_features = [
        "close", "feature1", "feature2",
        "sma_5", "sma_10", "sma_20", "sma_50", "sma_200",
        "rsi", "macd_hist", "bollinger_upper", "bollinger_lower",
        "adx", "stochastic_k",
        "momentum_5", "momentum_15", "volatility_30",
        "ema_21", "ema_50",
        "roc_5", "roc_10",
        "atr", "volume_ma",
        "price_distance_to_upper_band", "price_distance_to_lower_band"  # 25 features
    ]
    return df[final_features]

# ==================================================================================================================
# -----------------------------
# Mecanismo de reconexão robusto
# -----------------------------
async def check_connection():
    #print(Fore.YELLOW + "Dentro do check_connection")

    while True:
        if not mt5.initialize():
            print(Fore.RED + "❌ Conexão perdida. Tentando reconectar em 10s...")
            mt5.shutdown()
            await asyncio.sleep(10)
        else:
            await asyncio.sleep(60)

# ==================================================================================================================
# -----------------------------
# validação com dados ao vivo durante o treino
# -----------------------------
class CustomCNN(BaseFeaturesExtractor):
    #print(Fore.YELLOW + "Dentro do CustomCNN")

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        #print(Fore.YELLOW + "Dentro do __init__ CustomCNN")

        super().__init__(observation_space, features_dim)
        self.net = th.nn.Sequential(
            th.nn.Linear(observation_space.shape[0], 256),
            th.nn.ReLU(),
            th.nn.Linear(256, 256),
            th.nn.ReLU(),
            th.nn.Linear(256, features_dim),
            th.nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        #print(Fore.YELLOW + "Dentro do forward CustomCNN")

        return self.net(observations)

# ==================================================================================================================
# -----------------------------
# validação com dados ao vivo durante o treino
# -----------------------------
class LiveValidationCallback(BaseCallback):
    #print(Fore.YELLOW + "Dentro do LiveValidationCallback")

    def __init__(self, eval_env, freq=10000, verbose=0):
        #print(Fore.YELLOW + "Dentro do __init__ LiveValidationCallback")

        super().__init__(verbose)
        self.eval_env = eval_env
        self.freq = freq
        self.best_mean_reward = -np.inf
        self.balance_history = []

    def _on_step(self) -> bool:
        #print(Fore.YELLOW + "Dentro do _on_step LiveValidationCallback")

        if self.n_calls % self.freq == 0:
            print(Fore.BLUE + "\nIniciando validação em tempo real...")
            rewards = []
            balances = []

            for episode in range(3):
                obs, _ = self.eval_env.reset()
                done = False
                episode_reward = 0

                while not done:
                    action, _ = self.model.predict(obs, deterministic=False)
                    action = int(action)  # Converter a ação para inteiro
                    obs, reward, done, _, _ = self.eval_env.step(action)
                    episode_reward += reward

                rewards.append(episode_reward)
                balances.append(self.eval_env.get_current_balance())
                print(Fore.BLUE + f"Episódio {episode + 1}: Recompensa: {episode_reward:.2f}, Saldo: ${balances[-1]:.2f}")

            mean_reward = np.mean(rewards)
            mean_balance = np.mean(balances)
            self.balance_history.append(mean_balance)

            print(Fore.BLUE + f"Recompensa média: {mean_reward:.2f}")
            print(Fore.BLUE + f"Saldo médio: ${mean_balance:.2f}")
            print(Fore.BLUE + f"Histórico de saldos: {self.balance_history[-5:]}")

            # Registrar no TensorBoard
            self.logger.record("validation/mean_reward", mean_reward)
            self.logger.record("validation/mean_balance", mean_balance)

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save("dqn_trading_agent_live_val_best")
                print(Fore.LIGHTGREEN_EX + "✅ Novo melhor modelo salvo com base na validação!")

        return True

# ==================================================================================================================
class BalanceCallback(BaseCallback):
    #print(Fore.YELLOW + "Dentro do BalanceCallback")

    def __init__(self, verbose=0):
        #print(Fore.YELLOW + "Dentro do __init__ BalanceCallback")

        super(BalanceCallback, self).__init__(verbose)
        self.balances = []

    def _on_step(self) -> bool:
        #print(Fore.YELLOW + "Dentro do _on_step BalanceCallback")

        # Se o ambiente de treinamento estiver disponível, obtenha e registre o saldo atual
        if self.training_env:
            balance = self.training_env.get_attr('balance')[0]
            self.balances.append(balance)
            self.logger.record("train/balance", balance)

            # Cálculo da diversidade de ações, se o modelo armazenar as ações do episódio
            if hasattr(self.model, 'episode_actions') and len(self.model.episode_actions) > 0:
                action_counts = np.bincount(self.model.episode_actions)
                diversity_bonus = 1.0 - (np.max(action_counts) / np.sum(action_counts))
                self.logger.record("train/diversity_bonus", diversity_bonus)
                # Ajusta a recompensa (se disponível) com um bônus de diversidade
                if 'reward' in self.locals:
                    self.locals['reward'] += diversity_bonus * 0.3

            # Registre métricas adicionais a cada 100 passos
            if len(self.balances) % 100 == 0:
                returns = pd.Series(self.balances).pct_change().dropna()
                sharpe = math.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
                max_drawdown = (pd.Series(self.balances).cummax() - pd.Series(self.balances)).max()
                self.logger.record("train/sharpe_ratio", sharpe)
                self.logger.record("train/max_drawdown", max_drawdown)
                print(Fore.BLUE + f"Saldo: "+Fore.LIGHTGREEN_EX +f"${balance:.2f}"+Fore.BLUE + " | "+ Fore.BLUE + f"Sharpe: {sharpe:.2f}"+Fore.BLUE + " | "+Fore.RED + f"Máximo Drawdown: {max_drawdown:.2f}")

        # Atualiza o replay buffer a cada 1000 timesteps, se o buffer suportar essa operação
        if self.num_timesteps % 1000 == 0:
            if hasattr(self.model, "replay_buffer") and hasattr(self.model.replay_buffer, "update_priorities"):
                # Exemplo: Atualiza prioridades com base em algum critério dos episódios armazenados
                # self.model.replay_buffer.update_priorities(self.model.episode_transitions)
                pass

        return True

# ==================================================================================================================
# -----------------------------
# Classe TradingBot (Modelo ML incremental com validação holdout)
# -----------------------------
class TradingBot:
    #print(Fore.YELLOW + "Dentro do TradingBot")

    def __init__(self, symbol):
        #print(Fore.YELLOW + "Dentro do __init__ TradingBot")

        self.symbol = symbol
        self.tracker = PerformanceTracker()  # ✅ Inicialização do tracker
        self.model_path = "modelo_ml.pkl"
        self.ml_model = self.load_or_train_model()
        self.historical_features = pd.DataFrame()  # Inicializa vazio
        self._load_initial_data()  # ← Novo método para carregar dados iniciais
        # Supondo que self.model seja o modelo RL (caso exista)
        self.model = None
        self.rl_model = None
        # Integração do dashboard de performance
        self.dashboard = PerformanceDashboard()
        params = self.get_symbol_params()
        required_keys = ['point', 'stops_level', 'volume_step']
        for key in required_keys:
            if key not in params:
                raise ValueError(f"Parâmetro crítico ausente: {key}")

    def load_or_train_model(self):
        #print(Fore.YELLOW + "Dentro do load_or_train_model TradingBot")

        try:
            with open(self.model_path, "rb") as f:
                model = pickle.load(f)
            check_is_fitted(model)
            print(Fore.GREEN + "✅ Modelo ML carregado com sucesso!")
            return model
        except (FileNotFoundError, pickle.UnpicklingError, AttributeError, ValueError):
            print(Fore.RED + "⚠️ Modelo ML não encontrado!"+Fore.LIGHTWHITE_EX+" Criando um novo modelo incremental.")
            model = SGDClassifier(
                loss='log_loss',
                penalty='l2',  # Adiciona regularização L2
                alpha=0.0001,  # Força da regularização
                learning_rate='optimal',
                max_iter=1000,
                tol=1e-3
            )
            model.partial_fit(np.zeros((1, 25)), [0], classes=np.array([0, 1]))
            return model

    def _integrate_signals(self, ml_signal, rl_action):
        """
        Integra os sinais do ML e RL para decisão final.

        Parâmetros:
            ml_signal (int): Sinal do modelo de machine learning (0 ou 1)
            rl_action (int): Ação do reinforcement learning (0-7)

        Retorna:
            tuple: (sinal_final, multiplicador_tamanho)
        """
        # Mapeamento das ações do RL
        if rl_action == 0:  # Esperar
            return None, 0.0
        elif 1 <= rl_action <= 3:  # Comprar (conservador, moderado, agressivo)
            if ml_signal == 1:
                # Confirmação do ML, usar tamanho completo
                return 1, 1.0
            else:
                # ML não confirma, reduzir tamanho
                return 1, 0.5 if rl_action == 1 else 0.7 if rl_action == 2 else 1.0
        elif 4 <= rl_action <= 6:  # Vender (conservador, moderado, agressivo)
            if ml_signal == 0:
                # Confirmação do ML, usar tamanho completo
                return 0, 1.0
            else:
                # ML não confirma, reduzir tamanho
                return 0, 0.5 if rl_action == 4 else 0.7 if rl_action == 5 else 1.0
        elif rl_action == 7:  # Trailing stop
            return None, 0.0  # Ação de gerenciamento, não abre nova posição
        else:
            return None, 0.0

    def train_model(self):
        #print(Fore.YELLOW + "Dentro do train_model TradingBot")

        try:
            print(Fore.BLUE + "📊 Atualizando modelo ML com dados recentes...")
            rates = get_historical_rates(self.symbol, TIMEFRAME, TEMPO_DADOS_COLETADOS)
            if rates is None or len(rates) < 1000:
                raise ValueError(Fore.RED + "Dados insuficientes para treinamento incremental")
            raw_df = pd.DataFrame(rates)
            df = calculate_features(raw_df).dropna()
            df["target"] = (raw_df["close"].shift(-1) > raw_df["close"]).astype(int)
            df = df.dropna()

            # Atualiza historical_features com as features mais recentes (INCLUINDO 'close')
            self.historical_features = df[[
                "close",  # ✅ Adicionado aqui
                "feature1", "feature2",
                "sma_5", "sma_10", "sma_20", "sma_50", "sma_200",
                "rsi", "macd_hist", "bollinger_upper", "bollinger_lower",
                "adx", "stochastic_k",
                "momentum_5", "momentum_15", "volatility_30",
                "ema_21", "ema_50",
                "roc_5", "roc_10",
                "atr", "volume_ma",
                "price_distance_to_upper_band", "price_distance_to_lower_band"
            ]].copy()

            features = df[[
                "feature1", "feature2",
                "sma_5", "sma_10", "sma_20", "sma_50", "sma_200",
                "rsi", "macd_hist", "bollinger_upper", "bollinger_lower",
                "adx", "stochastic_k",
                "momentum_5", "momentum_15", "volatility_30",
                "ema_21", "ema_50", "roc_5", "roc_10",
                "atr", "volume_ma",
                "price_distance_to_upper_band", "price_distance_to_lower_band"
            ]]
            target = df["target"]

            X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, shuffle=False)
            self.ml_model.partial_fit(X_train.values, y_train.values)
            y_pred = self.ml_model.predict(X_val.values)
            val_acc = accuracy_score(y_val, y_pred)
            print(Fore.BLUE + f"🔍 Validação Holdout - Accuracy: {val_acc:.2f}")

            if val_acc < 0.5:
                print(Fore.RED + "⚠️ Acurácia baixa. Reinicializando modelo ML.")
                self.ml_model = SGDClassifier(loss='log_loss', learning_rate='optimal', max_iter=1000, tol=1e-3)
                self.ml_model.partial_fit(np.zeros((1, 11)), [0], classes=np.array([0, 1]))

            with open(self.model_path, "wb") as f:
                pickle.dump(self.ml_model, f)
            print(Fore.LIGHTGREEN_EX + "✅ Modelo ML atualizado incrementalmente, validado e salvo!")
            return self.ml_model
        except Exception as e:
            print(Fore.RED + f"❌ Erro no treinamento incremental: {str(e)}")
            return self.ml_model

    def calculate_atr(self, period=ATR_PERIOD):
        #print(Fore.YELLOW + "Dentro do calculate_atr TradingBot")

        rates = mt5.copy_rates_from_pos(self.symbol, TIMEFRAME, 0, period + 1)
        if rates is None or len(rates) < period + 1:
            print(Fore.RED + "❌ Erro: Dados insuficientes para ATR.")
            return None
        df = pd.DataFrame(rates)
        df["prev_close"] = df["close"].shift(1)
        df["tr1"] = df["high"] - df["low"]
        df["tr2"] = (df["high"] - df["prev_close"]).abs()
        df["tr3"] = (df["low"] - df["prev_close"]).abs()
        df["tr"] = df[["tr1", "tr2", "tr3"]].max(axis=1)
        atr = df["tr"].rolling(window=period).mean().iloc[-1]
        return atr

    def get_live_data(self, window_size=TAMANHO_JANELA):
        #print(Fore.YELLOW + "Dentro do get_live_data TradingBot")

        rates = mt5.copy_rates_from_pos(self.symbol, TIMEFRAME, 0, window_size)
        if rates is None or len(rates) < window_size:
            print(Fore.RED + "⚠️ Dados insuficientes para formação de estado")
            # return np.zeros(25 * window_size + 1)  # Retorna zeros se falhar
            # Correção: Remover o +1 para manter 25*21=525 features
            return np.zeros(25 * window_size)  # Retorna zeros se falhar

        df = pd.DataFrame(rates)
        df = calculate_features(df).dropna()

        # As features devem corresponder exatamente às do TradingEnv (25)
        features = [
            "close",
            "feature1", "feature2",
            "sma_5", "sma_10", "sma_20", "sma_50", "sma_200",
            "rsi", "macd_hist", "bollinger_upper", "bollinger_lower",
            "adx", "stochastic_k",
            "momentum_5", "momentum_15", "volatility_30",
            "ema_21", "ema_50",
            "roc_5", "roc_10",
            "atr", "volume_ma",
            "price_distance_to_upper_band", "price_distance_to_lower_band"
        ]

        if len(features) != 25:
            raise ValueError("Número incorreto de features no bot.")

        df_window = df.iloc[-window_size:]
        if len(df_window) < window_size:
            padding = np.zeros((window_size - len(df_window), len(features)))
            df_values = np.vstack([padding, df_window[features].values])
        else:
            df_values = df_window[features].values

        state = df_values.flatten()
        ml_signal = self.predict_signal(df_window.iloc[-1:])
        print(f"Live Data Stats - Min: {np.min(state)} | Max: {np.max(state)} | Mean: {np.mean(state)}")
        return state

    def get_symbol_params(self):
        params = {
            'point': 0.00001,
            'stops_level': 54,
            'digits': 5,
            'volume_min': 0.01,
            'volume_max': 500.0,
            'volume_step': 0.01  # Valor padrão
        }

        try:
            for _ in range(3):
                symbol_info = mt5.symbol_info(self.symbol)
                if symbol_info:
                    params.update({
                        'point': symbol_info.point,
                        'stops_level': symbol_info.trade_stops_level,
                        'digits': symbol_info.digits,
                        'volume_min': symbol_info.volume_min,
                        'volume_max': symbol_info.volume_max,
                        'volume_step': getattr(symbol_info, 'volume_step', 0.01)
                    })
                    break
                time.sleep(1)
        except Exception as e:
            print(f"Erro ao obter parâmetros: {str(e)}")

        # Garantir valores mínimos críticos
        params['volume_step'] = max(params['volume_step'], 0.001)
        params['stops_level'] = max(params['stops_level'], 54)

        return params

    def predict_signal(self, df):
        #print(Fore.YELLOW + "Dentro do predict_signal TradingBot")

        if df is None or self.ml_model is None or df.empty:
            print(Fore.RED + "⚠️ DataFrame vazio ou modelo não carregado.")
            return 0  # Retorna valor padrão
        try:
            return self.ml_model.predict(df.values)[0]
        except Exception as e:
            print(Fore.RED + f"❌ Erro na predição: {str(e)}")
            return 0

    def get_trade_params(self, signal):
        """Calcula parâmetros de trade com fallbacks robustos"""
        try:
            # 1. Obter parâmetros básicos
            symbol_params = self.get_symbol_params()
            tick = mt5.symbol_info_tick(self.symbol)

            if not tick or tick.ask <= 0 or tick.bid <= 0:
                raise ValueError("Cotações inválidas")

            # 2. Calcular preço base
            price = tick.ask if signal == 1 else tick.bid
            price = round(price, symbol_params['digits'])

            # 3. Calcular ATR com fallback
            atr = self.calculate_atr() or (20 * symbol_params['point'])
            min_distance = symbol_params['stops_level'] * symbol_params['point'] * 1.5

            # 4. Cálculo inicial de SL/TP
            if signal == 1:  # Compra
                sl = price - (atr * ATR_SL_MULTIPLIER)
                tp = price + (atr * ATR_TP_MULTIPLIER)  # ← Aqui estava o erro!
            else:  # Venda
                sl = price + (atr * ATR_SL_MULTIPLIER)
                tp = price - (atr * ATR_TP_MULTIPLIER)  # ← E aqui!

            # 5. Ajustar distâncias
            if abs(price - sl) < min_distance:
                sl = price - min_distance if signal == 1 else price + min_distance

            if abs(price - tp) < min_distance:
                tp = price + (min_distance * 2) if signal == 1 else price - (min_distance * 2)

            # 6. Arredondar valores finais
            sl = round(sl, symbol_params['digits'])
            tp = round(tp, symbol_params['digits'])  # ← Garantir definição

            # 7. Calcular volume
            volume = self.calculate_volume(atr)

            return (price, tp, sl, volume)

        except Exception as e:
            print(f"Erro crítico no get_trade_params: {str(e)}")
            return (None, None, None, None)

    def log_order(self, action_type, details, outcome=None, pnl=None):
        #print(Fore.YELLOW + "Dentro do log_order TradingBot")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        balance = mt5.account_info().balance if mt5.account_info() else 0

        log_entry = {
            'timestamp': timestamp,
            'balance': f"{balance:.2f}",
            'action': action_type,
            'details': details,
            'outcome': outcome,
            'pnl': f"{pnl:.2f}" if pnl is not None else None
        }

        line = (
            f"[{log_entry['timestamp']}] "
            f"Balance: {log_entry['balance']} | "
            f"Action: {log_entry['action']} | "
            f"Details: {log_entry['details']}"
        )

        if outcome and pnl is not None:
            line += f" | Outcome: {outcome} | PnL: {log_entry['pnl']}"

        with open('ordens.txt', 'a', encoding='utf-8') as f:
            f.write(line + '\n')

    def calculate_min_distance(self, symbol_info):
        if not symbol_info:
            return 0.00015  # Fallback se não houver symbol_info

        point = symbol_info.point
        stops_level = getattr(symbol_info, 'trade_stops_level', 54)  # Fallback para 54
        if stops_level <= 0:
            stops_level = 54  # Garantir valor mínimo

        min_dist = stops_level * point
        return round(min_dist * 1.5, symbol_info.digits)

    def send_order(self, signal, size_multiplier=1.0):
        print(f"\n=== DEBUG ENVIO DE ORDEM ===")
        print(f"Symbol: {self.symbol}")
        print(f"Signal: {signal}")
        print(f"Ticks disponíveis: {mt5.symbol_info_tick(self.symbol)}")
        print(f"Volume step: {self.get_symbol_params()['volume_step']}")
        """
        Envia ordens com validação robusta de distância de stops e tratamento de valores zero
        """
        max_attempts = 5
        attempt = 0
        success = False
        volume = 0.0
        price = tp = sl = 0.0
        last_error = ""

        # Parâmetros do símbolo com fallback seguro
        symbol_params = self.get_symbol_params()  # ✅ Sem argumentos!
        min_distance = symbol_params['stops_level'] * symbol_params['point']
        point = symbol_params['point']
        digits = symbol_params['digits']

        print(f"\n{'=' * 30}\n📤 Iniciando envio de ordem\n{'=' * 30}")
        print(f"▪ Sinal: {signal} ({'Compra' if signal == 1 else 'Venda'})")
        print(f"▪ Multiplicador Tamanho: {size_multiplier}")

        while attempt < max_attempts and not success:
            try:
                if not mt5.initialize():
                    raise ConnectionError("Falha na conexão MT5")

                price, tp, sl, volume = self.get_trade_params(signal)
                if None in (price, tp, sl, volume):
                    raise ValueError("Parâmetros inválidos do get_trade_params")

                volume *= size_multiplier
                volume = max(symbol_params['volume_min'], min(volume, symbol_params['volume_max']))

                price = round(price, digits)
                tp = round(tp, digits)
                sl = round(sl, digits)

                print(f"\n🔧 Parâmetros da Ordem (Tentativa {attempt + 1}):")
                print(f"▪ Preço: {price:.{digits}f}")
                print(f"▪ TP: {tp:.{digits}f} | SL: {sl:.{digits}f}")
                print(f"▪ Volume: {volume:.2f} lotes")
                print(f"▪ Distância Mínima: {min_distance:.{digits}f}")

                # Validação de distância mínima dos stops
                if signal == 1:  # Compra
                    if (price - sl) < min_distance or (tp - price) < min_distance:
                        print("🚫 Stop inválido mesmo após ajustes! Abortando ordem")
                        return False
                else:  # Venda
                    if (sl - price) < min_distance or (price - tp) < min_distance:
                        print("🚫 Stop inválido mesmo após ajustes! Abortando ordem")
                        return False

                # Monta e envia a ordem
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": self.symbol,
                    "volume": volume,
                    "type": mt5.ORDER_TYPE_BUY if signal == 1 else mt5.ORDER_TYPE_SELL,
                    "price": price,
                    "sl": sl,
                    "tp": tp,
                    "deviation": 30,
                    "magic": 0,
                    "comment": f"AutoTrade {datetime.now().strftime('%Y%m%d%H%M')}",
                    "type_filling": mt5.ORDER_FILLING_FOK,
                    "type_time": mt5.ORDER_TIME_GTC,
                }

                result = mt5.order_send(request)
                print(f"\n📩 Resposta do Broker:")
                print(f"▪ Retcode: {result.retcode} ({get_error_message(result.retcode)})")
                print(f"▪ Comentário: {result.comment}")

                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"\n✅ Ordem executada com sucesso! Ticket: {result.order}")
                    self.log_order(
                        "EXECUTADA",
                        f"{'COMPRA' if signal == 1 else 'VENDA'} {result.volume:.2f} lotes | "
                        f"Preço: {result.price:.{digits}f} | TP: {tp:.{digits}f} | SL: {sl:.{digits}f}",
                        "SUCESSO",
                        pnl=0
                    )
                    success = True
                else:
                    last_error = get_error_message(result.retcode)
                    print(f"\n⚠️ Falha na ordem: {last_error}")
                    if result.retcode in [10004, 10006, 10016, 10017]:  # Problemas de preço
                        print("🔄 Atualizando preços...")
                        new_tick = mt5.symbol_info_tick(self.symbol)
                        new_price = new_tick.ask if signal == 1 else new_tick.bid
                        if abs(new_price - price) > min_distance:
                            price = new_price
                            print(f"▪ Novo Preço: {price:.{digits}f}")
                            tp, sl = self._recalculate_sl_tp(price, signal, min_distance)
                            # Aplicação do arredondamento conforme os dígitos do símbolo
                            tp = round(tp, digits)
                            sl = round(sl, digits)
                    elif result.retcode == 10015:
                        print("⏳ Aguardando liberação do contexto...")
                        time.sleep(5)

                attempt += 1

            except Exception as e:
                last_error = str(e)
                print(f"\n❌ Erro crítico: {last_error}")
                import traceback
                traceback.print_exc()
                attempt += 1
                time.sleep(3)

            finally:
                mt5.shutdown()
                time.sleep(1)

        if not success:
            print(f"\n❌ Falha após {max_attempts} tentativas")
            print(f"▪ Último erro: {last_error}")
            self.log_order(
                "FALHA",
                f"{'COMPRA' if signal == 1 else 'VENDA'} {volume:.2f} lotes | "
                f"Preço: {price:.{digits}f} | Erro: {last_error}",
                "FALHA"
            )

        print(f"{'=' * 30}\n🏁 Fim do processo\n{'=' * 30}")
        return success

    def _recalculate_sl_tp(self, price, signal, min_distance):
        """Recalcula SL/TP com base na distância mínima exigida pelo broker"""
        if signal == 1:  # Compra
            sl = price - min_distance
            tp = price + (min_distance * 2)  # TP como 2x a distância mínima
        else:  # Venda
            sl = price + min_distance
            tp = price - (min_distance * 2)
        return tp, sl

    def dynamic_stop_adjustment(current_price, entry_price, position_type, symbol_params):
        min_distance = symbol_params['stops_level'] * symbol_params['point']
        volatility = abs(current_price - entry_price)

        if position_type == 'buy':
            new_sl = current_price - max(min_distance, volatility * 0.7)
            new_sl = max(new_sl, entry_price - (min_distance * 3))  # Limite máximo de retração
        else:
            new_sl = current_price + max(min_distance, volatility * 0.7)
            new_sl = min(new_sl, entry_price + (min_distance * 3))

        return round(new_sl, symbol_params['digits'])

    async def update_model_periodically(self):
        while True:
            if OPERATION_MODE == "rl":  # Pular treino ML no modo RL
                await asyncio.sleep(TIME_TREINO)
                continue
            # Restante do código de treinamento ML
            print("Atualizando modelo ML...")
            self.train_model()

    async def manage_positions(self, new_signal):
        #print(Fore.YELLOW + "Dentro do manage_positions TradingBot")

        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            return

        current_price = mt5.symbol_info_tick(self.symbol).ask

        for pos in positions:
            entry_price = pos.price_open
            tp = pos.tp
            sl = pos.sl
            position_type = pos.type  # 0 = Compra, 1 = Venda

            new_signal_direction = 1 if new_signal == 1 else 0
            same_direction = (position_type == 0 and new_signal_direction == 1) or (position_type == 1 and new_signal_direction == 0)

            if position_type == 0:
                if tp > entry_price:
                    tp_distance = tp - entry_price
                    tp_target = entry_price + 0.7 * tp_distance
                    is_near_tp = current_price >= tp_target
                else:
                    is_near_tp = False
            else:
                if tp < entry_price:
                    tp_distance = entry_price - tp
                    tp_target = entry_price - 0.7 * tp_distance
                    is_near_tp = current_price <= tp_target
                else:
                    is_near_tp = False

            if position_type == 0:
                if sl < entry_price:
                    sl_distance = entry_price - sl
                    sl_target = entry_price - 0.8 * sl_distance
                    is_sl_near = current_price <= sl_target
                else:
                    is_sl_near = False
            else:
                if sl > entry_price:
                    sl_distance = sl - entry_price
                    sl_target = entry_price + 0.8 * sl_distance
                    is_sl_near = current_price >= sl_target
                else:
                    is_sl_near = False

            is_rsi_reversal = ((position_type == 0 and self.get_current_rsi() > 70) or (position_type == 1 and self.get_current_rsi() < 30))
            macd_line, signal_line, histogram = self.get_macd()
            is_macd_reversal = ((position_type == 0 and histogram < 0) or (position_type == 1 and histogram > 0))
            upper_band, middle_band, lower_band = self.get_bollinger_bands()
            is_bollinger_reversal = ((position_type == 0 and current_price >= upper_band) or (position_type == 1 and current_price <= lower_band))

            is_reversal_indicator = is_rsi_reversal or is_macd_reversal or is_bollinger_reversal
            is_reversal = is_reversal_indicator and is_sl_near

            close_condition = is_near_tp if same_direction else (is_near_tp or is_reversal)

            if close_condition:
                motivos = []
                if is_near_tp:
                    motivos.append("TP Próximo (70%)")
                if is_reversal:
                    motivos.append("Reversão (80% SL)")
                motivo_str = ", ".join(motivos)
                print(Fore.BLUE + f"🔵 Fechando posição #{pos.ticket} - Motivo: {motivo_str}")
                await self.close_position(pos)

    async def close_position(self, position):
        #print(Fore.YELLOW + "Dentro do close_position TradingBot")

        close_type = mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(self.symbol).bid if position.type == 0 else mt5.symbol_info_tick(self.symbol).ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": position.ticket,
            "symbol": self.symbol,
            "volume": position.volume,
            "type": close_type,
            "price": price,
            "deviation": 20,
            "comment": "Fechamento Automático",
            "type_time": mt5.ORDER_TIME_GTC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(Fore.RED + f"Erro ao fechar posição {position.ticket}: {result.comment}")

    def get_current_rsi(self, period=14):
        #print(Fore.YELLOW + "Dentro do get_current_rsi TradingBot")

        rates = mt5.copy_rates_from_pos(self.symbol, TIMEFRAME, 0, period+1)
        if rates is None:
            return 50
        closes = pd.DataFrame(rates)['close']
        delta = closes.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean().iloc[-1]
        avg_loss = loss.rolling(period).mean().iloc[-1]
        rs = avg_gain / avg_loss if avg_loss != 0 else 1
        return 100 - (100 / (1 + rs))

    def get_macd(self, fast_period=12, slow_period=26, signal_period=9):
        #print(Fore.YELLOW + "Dentro do get_macd TradingBot")

        rates = mt5.copy_rates_from_pos(self.symbol, TIMEFRAME, 0, slow_period + 50)
        if rates is None or len(rates) < slow_period:
            return None, None, None
        df = pd.DataFrame(rates)
        close = df['close']
        ema_fast = close.ewm(span=fast_period, adjust=False).mean()
        ema_slow = close.ewm(span=slow_period, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]

    def get_bollinger_bands(self, period=20, std_factor=2):
        #print(Fore.YELLOW + "Dentro do get_bollinger_bands TradingBot")

        rates = mt5.copy_rates_from_pos(self.symbol, TIMEFRAME, 0, period)
        if rates is None or len(rates) < period:
            return None, None, None
        df = pd.DataFrame(rates)
        close = df['close']
        middle = close.rolling(window=period).mean().iloc[-1]
        std = close.rolling(window=period).std().iloc[-1]
        upper = middle + std_factor * std
        lower = middle - std_factor * std
        return upper, middle, lower

    def calculate_volume(self, atr):
        """Calcula o volume de forma segura com fallbacks robustos"""
        try:
            # 1. Obter parâmetros do símbolo
            symbol_params = self.get_symbol_params()  # <-- Adicionado

            # 2. Obter informações da conta com fallback
            account_info = mt5.account_info()
            balance = account_info.balance if account_info else SALDO_INICIAL_TREINO

            # 3. Validar parâmetros críticos
            if not all(key in symbol_params for key in ['volume_step', 'volume_min', 'volume_max', 'point']):
                raise ValueError("Parâmetros do símbolo incompletos")

            # 4. Cálculo seguro do volume
            risk_amount = min(balance * risk_per_trade, balance * 0.05)  # Máx 5%
            pip_value = 10  # Valor fixo para AUDCAD
            stop_loss_pips = (atr * ATR_SL_MULTIPLIER) / symbol_params['point'] if atr else 30
            stop_loss_pips = max(stop_loss_pips, 54)  # Garantir mínimo

            if stop_loss_pips <= 0:
                return 0.0

            volume = risk_amount / (stop_loss_pips * pip_value)
            volume_step = symbol_params['volume_step']

            # 5. Ajustar volume aos limites
            return max(
                symbol_params['volume_min'],
                min(
                    round(volume / volume_step) * volume_step,
                    symbol_params['volume_max']
                )
            )

        except Exception as e:
            print(f"Erro crítico no cálculo de volume: {str(e)}")
            return symbol_params['volume_min'] if 'volume_min' in symbol_params else 0.01

    # def calculate_volume(self, atr, symbol_params):
    #     try:
    #         account_info = mt5.account_info()
    #         balance = account_info.balance if account_info else SALDO_INICIAL_TREINO
    #
    #         risk_amount = balance * risk_per_trade
    #         pip_value = 10  # Valor por pip para AUDCAD
    #
    #         # Cálculo de stop_loss_pips com fallback
    #         stop_loss_pips = (atr * ATR_SL_MULTIPLIER) / symbol_params['point'] if atr else 30
    #
    #         if stop_loss_pips <= 0:
    #             return symbol_params['volume_min']
    #
    #         volume = risk_amount / (stop_loss_pips * pip_value)
    #
    #         # Ajustes finais com limites
    #         return max(
    #             symbol_params['volume_min'],
    #             min(
    #                 round(volume / symbol_params['volume_step']) * symbol_params['volume_step'],
    #                 symbol_params['volume_max']
    #             )
    #         )
    #     except Exception as e:
    #         print(f"Erro no cálculo de volume: {str(e)}")
    #         return symbol_params['volume_min']

    async def _monitor_performance(self, last_balance):
        current_balance = mt5.account_info().balance if mt5.account_info() else last_balance
        drawdown = (last_balance - current_balance) / last_balance if last_balance > 0 else 0

        if drawdown > 0.05:  # Alerta para drawdown de 5%
            print(Fore.RED + f"⚠️ Drawdown significativo detectado: {drawdown:.2%}")

        # Registrar métricas no dashboard
        self.dashboard.update(self.tracker.trades)
        self.log_metrics()

    def check_market_trend(self, window=50):
        #print(Fore.YELLOW + "Dentro do check_market_trend TradingBot")

        rates = mt5.copy_rates_from_pos(self.symbol, TIMEFRAME, 0, window)
        if rates is None or len(rates) < 34:
            return {'direction': "NEUTRAL", 'strength': 0}

        df = pd.DataFrame(rates)
        ema_fast = df['close'].ewm(span=12, adjust=False).mean().iloc[-1]
        ema_slow = df['close'].ewm(span=26, adjust=False).mean().iloc[-1]

        df['high_low'] = df['high'] - df['low']
        df['high_prev_close'] = abs(df['high'] - df['close'].shift(1))
        df['low_prev_close'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['high_low', 'high_prev_close', 'low_prev_close']].max(axis=1)
        df['atr'] = df['tr'].rolling(14).mean()

        plus_di = (df['high'].diff().clip(lower=0).rolling(14).mean() / df['atr']) * 100
        minus_di = (abs(df['low'].diff()).clip(lower=0).rolling(14).mean() / df['atr']) * 100
        dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        adx = dx.rolling(14).mean().iloc[-1]

        last_3_closes = df['close'].iloc[-3:].values
        ascending = all(last_3_closes[i] > last_3_closes[i - 1] for i in range(1, 3))
        descending = all(last_3_closes[i] < last_3_closes[i - 1] for i in range(1, 3))

        direction = "NEUTRAL"
        if ema_fast > ema_slow and adx > 25 and ascending:
            direction = "UP_STRONG"
        elif ema_fast < ema_slow and adx > 25 and descending:
            direction = "DOWN_STRONG"
        elif ema_fast > ema_slow:
            direction = "UP"
        elif ema_fast < ema_slow:
            direction = "DOWN"

        return {'direction': direction, 'strength': adx}

    def _calculate_reward(self, action_type, size_multiplier, current_price, next_price, volume):
        #print(Fore.YELLOW + "Dentro do _calculate_reward TradingBot")

        # Novos componentes para trailing
        if action == 7:
            # Penaliza inatividade no trailing
            reward += 0.1 * (self.current_sl - previous_sl) / atr_value

            # Bonificação por proteção de ganhos
            if (position_type == 'buy' and current_price > entry_price):
                reward += 0.5 * (current_price - entry_price) / entry_price

            # Penalização por trailing prematuro
            if (current_price - entry_price) < (atr_value * 0.5):
                reward -= 0.2

        # Cálculo do PnL (Profit and Loss)
        if action_type == 'buy':
            pnl = (next_price - current_price) * volume * 10000
        else:
            pnl = (current_price - next_price) * volume * 10000
        base_reward = (pnl / mt5.account_info().balance) * 100
        reward = base_reward

        # Bônus de entrada precoce, utilizando indicadores históricos (ADX e ROC)
        if (hasattr(self, 'historical_features') and
                not self.historical_features.empty and
                'adx' in self.historical_features.columns and
                'roc_5' in self.historical_features.columns):
            adx = self.historical_features['adx'].iloc[-1]
            roc_5 = self.historical_features['roc_5'].iloc[-1]
        else:
            adx, roc_5 = 0, 0

        trend_strength = adx / 100
        early_entry_bonus = trend_strength * (roc_5 * 2)
        reward += early_entry_bonus * 100

        # Penalizações baseadas na tendência de mercado
        trend = self.check_market_trend()
        if (trend['direction'] == "UP_STRONG" and action_type == "sell") or \
                (trend['direction'] == "DOWN_STRONG" and action_type == "buy"):
            reward -= 50
        elif (trend['direction'] == "UP" and action_type == "sell") or \
                (trend['direction'] == "DOWN" and action_type == "buy"):
            reward -= 20

        return reward

    def log_metrics(self):
        #print(Fore.YELLOW + "Dentro do log_metrics TradingBot")

        metrics = self.dashboard.metrics
        print(Fore.BLUE + f"Trend Capture Ratio: {metrics['trend_capture_ratio'][-1]:.2f}")

    def _load_initial_data(self):
        #print(Fore.YELLOW + "Dentro do _load_initial_data TradingBot")

        try:
            rates = get_historical_rates(self.symbol, TIMEFRAME, TEMPO_DADOS_COLETADOS)
            if rates is not None and len(rates) > 0:
                df = pd.DataFrame(rates)
                self.historical_features = calculate_features(df)
                print(Fore.LIGHTGREEN_EX + "✅ Dados históricos carregados com sucesso!")
            else:
                print(Fore.RED + "⚠️ Não foi possível carregar dados iniciais. Usando dataframe vazio.")
                self.historical_features = pd.DataFrame()
        except Exception as e:
            print(Fore.RED + f"❌ Erro crítico ao carregar dados iniciais: {str(e)}")
            self.historical_features = pd.DataFrame()

    def adjust_strategy(self, regime):
        #print(Fore.YELLOW + "Dentro do adjust_strategy TradingBot")

        if regime == 'trending':
            self.model.set_parameters(exploration_rate=0.1)
        elif regime == 'ranging':
            self.model.set_parameters(exploration_rate=0.3)

    def execute_trail_stop(self):
        positions = mt5.positions_get(symbol=self.symbol)
        if not positions:
            return

        current_price = mt5.symbol_info_tick(self.symbol).ask
        atr = self.calculate_atr(period=ATR_PERIOD)

        # Ordena posições por lucro (maiores primeiro)
        positions_sorted = sorted(
            positions,
            key=lambda p: (p.price_current - p.price_open) / p.price_open,
            reverse=True
        )

        for pos in positions_sorted:
            try:
                # Cálculo dinâmico do trailing step
                price_action = abs(pos.price_current - pos.price_open)
                trailing_step = max(
                    atr * 0.5,
                    price_action * 0.3
                )

                new_sl = None
                if pos.type == 0:  # Compra
                    new_sl = pos.price_current - trailing_step
                    new_sl = max(new_sl, pos.sl + (atr * 0.2))  # Garante movimento ascendente
                else:  # Venda
                    new_sl = pos.price_current + trailing_step
                    new_sl = min(new_sl, pos.sl - (atr * 0.2))

                if self._is_sl_valid(pos, new_sl) and self._is_better_sl(pos, new_sl):
                    self._modify_position_with_retry(pos, new_sl)

            except Exception as e:
                print(f"Erro em trailing stop: {str(e)}")
                continue

            tracker.log_metrics({
                'trail_adjustments': len(positions),
                'avg_trail_distance': np.mean([abs(p.sl - p.price_current) for p in positions])
            })

    def _modify_position_with_retry(self, position, new_sl, max_attempts=3):
        for attempt in range(max_attempts):
            result = mt5.order_send({
                "action": mt5.TRADE_ACTION_SLTP,
                "position": position.ticket,
                "symbol": self.symbol,
                "sl": new_sl,
                "deviation": 20 * (attempt + 1),  # Aumenta tolerância progressivamente
            })

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"✅ SL ajustado para {new_sl} na tentativa {attempt + 1}")
                return True

            print(f"⚠️ Tentativa {attempt + 1} falhou. Erro: {result.comment}")
            time.sleep(1)

        print(f"❌ Falha após {max_attempts} tentativas")
        return False

    def _is_sl_valid(self, position, new_sl):
        #print(Fore.YELLOW + "Dentro do _is_sl_valid TradingBot")

        """Verifica se o novo SL está dentro dos limites do broker"""
        symbol_info = mt5.symbol_info(self.symbol)
        if not symbol_info:
            return False

        price = position.price_open
        point = symbol_info.point
        min_dist = symbol_info.stops_level * point

        if position.type == 0:  # Compra
            return (price - new_sl) >= min_dist
        else:  # Venda
            return (new_sl - price) >= min_dist

    def _is_better_sl(self, position, new_sl):
        #print(Fore.YELLOW + "Dentro do _is_better_sl TradingBot")

        """Determina se o novo SL melhora a posição"""
        if position.type == 0:  # Compra
            return new_sl > position.sl
        else:  # Venda
            return new_sl < position.sl

    def _modify_position(self, position, new_sl):
        #print(Fore.YELLOW + "Dentro do _modify_position TradingBot")

        """Modifica o stop loss da posição"""
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": position.ticket,
            "symbol": self.symbol,
            "sl": new_sl,
            "tp": position.tp,
            "deviation": 20,
        }

        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(Fore.LIGHTGREEN_EX + f"✅ Trailing stop atualizado para {new_sl:.5f} na posição #{position.ticket}")
        else:
            print(Fore.RED + f"❌ Falha ao atualizar trailing stop: {result.comment}")

    def _validate_trend(self, signal):
        """Valida o sinal contra a tendência atual do mercado"""
        trend = self.check_market_trend()

        if trend['direction'] == "UP_STRONG" and signal == 0:
            return False
        if trend['direction'] == "DOWN_STRONG" and signal == 1:
            return False

        if trend['strength'] > 40:  # Tendência forte
            if trend['direction'] == "UP" and signal == 0:
                return False
            if trend['direction'] == "DOWN" and signal == 1:
                return False

        return True

    async def adaptive_risk_adjustment(self):
        """Ajusta o risco dinamicamente com limites rigorosos"""
        global risk_per_trade
        current_balance = mt5.account_info().balance if mt5.account_info() else saldo_inicial

        # Limitar máximo de ajustes mesmo sem trades
        MAX_RISK = 0.02  # 2% máximo
        MIN_RISK = 0.005  # 0.5% mínimo

        # Lógica revisada
        if current_balance < saldo_inicial:
            new_risk = risk_per_trade * 0.8  # Reduz 20%
        else:
            new_risk = risk_per_trade * 1.1  # Aumenta 10%

        # Aplicar limites
        new_risk = max(MIN_RISK, min(MAX_RISK, new_risk))

        risk_per_trade = new_risk
        print(f"🛑 Risco ajustado para: {risk_per_trade:.3f} (Limites: {MIN_RISK}-{MAX_RISK})")

    async def run(self):
        """
        Loop principal de operação do bot
        """
        global RL_MODEL
        last_balance = saldo_inicial  # Usar saldo inicial como fallback
        consecutive_losses = 0

        last_balance = saldo_inicial  # Usar saldo inicial como fallback

        while True:
            try:
                # Verificar conexão de forma robusta
                if not mt5.initialize():
                    print(Fore.RED + "❌ Falha na conexão MT5! Tentando reconectar...")
                    await asyncio.sleep(5)
                    continue

                # Obter informações da conta com verificação
                account_info = mt5.account_info()
                if not account_info:
                    print(Fore.RED + "⚠️ Falha ao obter informações da conta. Usando saldo inicial.")
                    current_balance = saldo_inicial
                else:
                    current_balance = account_info.balance
                    last_balance = current_balance

                # Atualização de dados
                state = self.get_live_data()
                if state is None:
                    await asyncio.sleep(INTERVALO_PREDICOES)
                    continue

                # Detecção de regime
                regime = detect_market_regime(self)
                self.adjust_strategy(regime)

                # Predições
                ml_signal = self.predict_signal(state[-25:]) if OPERATION_MODE != "rl" else None
                rl_action = RL_MODEL.predict(state, deterministic=True)[0] if RL_MODEL else 0

                # Lógica de integração
                final_signal, size_multiplier = self._integrate_signals(ml_signal, rl_action)

                # Validação de tendência
                if not self._validate_trend(final_signal):
                    final_signal = None

                # Execução
                if final_signal is not None:
                    await self.manage_positions(final_signal)
                    if not mt5.positions_get(symbol=self.symbol):
                        self.send_order(final_signal, size_multiplier)

                # Monitoramento de performance
                await self._monitor_performance(last_balance)
                last_balance = mt5.account_info().balance

                # Atualização dinâmica de parâmetros
                # await self._dynamic_adjustments()
                await self.adaptive_risk_adjustment()

                await asyncio.sleep(INTERVALO_PREDICOES)

                rl_action = RL_MODEL.predict(state, deterministic=True)[0]
                print(Fore.LIGHTWHITE_EX +f"RL Action: {rl_action} | Decoded: {decode_rl_signal(rl_action)}")

            except Exception as e:
                print(f"Erro no loop principal: {str(e)}")
                mt5.shutdown()
                await asyncio.sleep(10)

# ==================================================================================================================
def detect_market_regime(bot: TradingBot, window: int = 50) -> str:
    try:
        if bot.historical_features.empty or 'bollinger_middle' not in bot.historical_features.columns:
            return 'neutral'

        df = bot.historical_features

        # Cálculo de métricas-chave
        metrics = {
            'trend': bot.check_market_trend(window),
            'volatility': df['close'].pct_change().rolling(window).std().iloc[-1] * 100,
            # Correção: usar 'volume_ma' em vez de 'tick_volume'
            'volume_ratio': df['volume_ma'].iloc[-1] / df['volume_ma'].rolling(window).mean().iloc[-1],
            'adx': df['adx'].iloc[-1] if 'adx' in df.columns else 0,
            'bollinger_width': (df['bollinger_upper'].iloc[-1] - df['bollinger_lower'].iloc[-1]) /
                               df['bollinger_middle'].iloc[-1]
        }

        # Lógica de decisão hierárquica
        if metrics['volatility'] > 1.5 and metrics['volume_ratio'] > 2.0:
            return 'trending'
        elif metrics['bollinger_width'] < 0.1 and metrics['volatility'] < 0.8:
            return 'ranging'
        elif metrics['adx'] > 25 and 'STRONG' in metrics['trend']['direction']:
            return 'trending'
        return 'transition'

    # except Exception as e:
    #     print(f"Erro na detecção de regime: {str(e)}")
    #     return 'neutral'
    except Exception as e:
        print(f"Erro na detecção de regime: {str(e)}")
        return 'neutral'

# ==================================================================================================================
# -----------------------------
# Classe TradingEnv com Monitor (Gymnasium)
# -----------------------------
class TradingEnv(gym.Env):
    #print(Fore.YELLOW + "Dentro do TradingEnv")

    metadata = {'render.modes': ['human']}

    # def __init__(self, trading_bot, initial_balance=SALDO_INICIAL_TREINO):
    #     #print(Fore.YELLOW + "Dentro do __init__ TradingEnv")
    #
    #     super(TradingEnv, self).__init__()
    #     self.bot = trading_bot
    #     self.initial_balance = initial_balance
    #     self.balance = initial_balance
    #     self.current_step = 0
    #     self.max_steps = 100
    #     self.episode_balances = []
    #     self.action_history = []  # Novo histórico de ações
    #     self.data_index = 0
    #     # Novo: Rastreamento de posições simuladas
    #     self.open_positions = []  # Lista de dicionários com detalhes das posições
    #     self.current_sl = None  # Stop loss atual
    #     self.current_tp = None  # Take profit atual
    #
    #     self.training_scalers = {}  # Para armazenar parâmetros de normalização
    #     self._init_normalization()  # Novo método de inicialização
    #
    #     # Mapeamento de ações atualizado para 8 ações
    #     self.ACTIONS_MAP = {
    #         0: ('wait', 0),
    #         1: ('buy', 0.3),  # Compra conservadora
    #         2: ('buy', 0.7),  # Compra moderada
    #         3: ('buy', 1.0),  # Compra agressiva
    #         4: ('sell', 0.3),  # Venda conservadora
    #         5: ('sell', 0.7),  # Venda moderada
    #         6: ('sell', 1.0),  # Venda agressiva
    #         7: ('trail_stop', 0.5)  # Stop móvel
    #     }
    #
    #     # Configurações de stops
    #     self.stops_level = STOPS  # Ex: 54 pontos
    #     self.point = 0.00001      # Para AUDCAD (ajuste conforme o símbolo)
    #     self.min_distance = self.stops_level * self.point
    #
    #     # Carregar dados históricos
    #     historical = get_historical_rates(symbol, TIMEFRAME, TEMPO_DADOS_COLETADOS)
    #     if historical is None or len(historical) == 0:
    #         raise ValueError("Dados históricos não disponíveis.")
    #
    #     df = calculate_features(pd.DataFrame(historical)).dropna()
    #     df.reset_index(drop=True, inplace=True)
    #
    #     # Configuração de features
    #     self.features = [
    #         "close",
    #         "feature1", "feature2",
    #         "sma_5", "sma_10", "sma_20", "sma_50", "sma_200",
    #         "rsi", "macd_hist", "bollinger_upper", "bollinger_lower",
    #         "adx", "stochastic_k",
    #         "momentum_5", "momentum_15", "volatility_30",
    #         "ema_21", "ema_50", "roc_5", "roc_10",
    #         "atr", "volume_ma",
    #         "price_distance_to_upper_band", "price_distance_to_lower_band"  # 25 features
    #     ]
    #
    #     self.price_col = "close"
    #     self.historical_data = self._normalize_data(df)
    #
    #     # Novo cálculo de dimensão do observation_space:
    #     # n_features = 25 (número de features por candle)
    #     # window_size = TAMANHO_JANELA (número de candles históricos)
    #     # +1 para incluir o sinal de ML
    #     n_features = 25  # Número de features por candle
    #     window_size = TAMANHO_JANELA
    #     self.observation_space = spaces.Box(
    #         low=-np.inf,
    #         high=np.inf,
    #         shape=(n_features * window_size,),
    #         dtype=np.float32
    #     )
    #
    #     # Atualização do espaço de ações para 8 ações
    #     self.action_space = spaces.Discrete(8)
    #
    #     # Configuração do logger
    #     self.logger = logging.getLogger('TradingEnv')
    #     self.logger.setLevel(logging.INFO)
    #     handler = logging.StreamHandler()
    #     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    #     handler.setFormatter(formatter)
    #     self.logger.addHandler(handler)

    def __init__(self, trading_bot, initial_balance=SALDO_INICIAL_TREINO):
        super(TradingEnv, self).__init__()

        # ====================================================================
        # 1. Carregar e processar dados históricos PRIMEIRO
        # ====================================================================
        historical = get_historical_rates(symbol, TIMEFRAME, TEMPO_DADOS_COLETADOS)
        if historical is None or len(historical) == 0:
            raise ValueError("Dados históricos não disponíveis.")

        df = calculate_features(pd.DataFrame(historical)).dropna()
        df.reset_index(drop=True, inplace=True)

        # ====================================================================
        # 2. Definir lista de features APÓS processamento dos dados
        # ====================================================================
        self.features = [
            "close",
            "feature1", "feature2",
            "sma_5", "sma_10", "sma_20", "sma_50", "sma_200",
            "rsi", "macd_hist", "bollinger_upper", "bollinger_lower",
            "adx", "stochastic_k",
            "momentum_5", "momentum_15", "volatility_30",
            "ema_21", "ema_50", "roc_5", "roc_10",
            "atr", "volume_ma",
            "price_distance_to_upper_band", "price_distance_to_lower_band"
        ]

        # ====================================================================
        # 3. Inicialização de componentes que dependem das features
        # ====================================================================
        self.price_col = "close"
        self.historical_data = self._normalize_data(df)
        self.training_scalers = {}
        self._init_normalization()  # ← Agora self.features já está definido

        # ====================================================================
        # 4. Configurações gerais do ambiente
        # ====================================================================
        self.bot = trading_bot
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.current_step = 0
        self.max_steps = 100
        self.episode_balances = []
        self.action_history = []
        self.data_index = 0
        self.open_positions = []
        self.current_sl = None
        self.current_tp = None

        # ====================================================================
        # 5. Configurações de ações e stops
        # ====================================================================
        self.ACTIONS_MAP = {
            0: ('wait', 0),
            1: ('buy', 0.3),
            2: ('buy', 0.7),
            3: ('buy', 1.0),
            4: ('sell', 0.3),
            5: ('sell', 0.7),
            6: ('sell', 1.0),
            7: ('trail_stop', 0.5)
        }

        self.stops_level = STOPS
        self.point = 0.00001
        self.min_distance = self.stops_level * self.point

        # ====================================================================
        # 6. Espaços de observação e ação
        # ====================================================================
        n_features = 25
        window_size = TAMANHO_JANELA
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features * window_size,),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(8)

        # ====================================================================
        # 7. Configuração do sistema de logging
        # ====================================================================
        self.logger = logging.getLogger('TradingEnv')
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def _init_normalization(self):
        """Inicializa os scalers de normalização para cada feature"""
        try:
            for col in self.features:
                # Verificação crítica de existência da coluna
                if col not in self.historical_data.columns:
                    raise KeyError(f"Feature '{col}' não encontrada nos dados históricos!")

                self.training_scalers[col] = {
                    'mean': self.historical_data[col].mean(),
                    'std': self.historical_data[col].std()
                }
            print(Fore.GREEN + "✅ Normalização inicializada com sucesso!")
        except Exception as e:
            print(Fore.RED + f"❌ Falha crítica na normalização: {str(e)}")
            raise

    def _simulate_trailing_stop(self):
        """Versão corrigida do método de trailing stop"""
        try:
            tick = mt5.symbol_info_tick(self.bot.symbol)
            current_price = self._get_current_price(tick)
            atr = self._calculate_simulated_atr()

            for pos in self.open_positions:
                # Cálculo dinâmico do trailing step
                price_action = abs(pos['price'] - pos['sl'])
                trailing_step = max(
                    atr * 0.5,
                    price_action * 0.3
                )

                new_sl = None
                if pos['type'] == 'buy':
                    new_sl = current_price - trailing_step
                    new_sl = max(new_sl, pos['sl'] + (atr * 0.2))
                else:
                    new_sl = current_price + trailing_step
                    new_sl = min(new_sl, pos['sl'] - (atr * 0.2))

                # Aplicar ajuste apenas se válido
                if self._is_sl_valid(pos, new_sl):
                    pos['sl'] = new_sl
                    self.current_sl = new_sl

        except Exception as e:
            print(Fore.RED + f"Erro no trailing stop: {str(e)}")

    def _normalize_data(self, df):
        #print(Fore.YELLOW + "Dentro do _normalize_data TradingEnv")

        scaled = df.copy()
        for col in self.features:
            # Normalização adaptativa usando estatísticas da janela
            window = 200
            rolling_mean = scaled[col].rolling(window).mean().shift(1)  # Evitar look-ahead
            rolling_std = scaled[col].rolling(window).std().shift(1)
            scaled[col] = (scaled[col] - rolling_mean) / rolling_std
            scaled[col] = scaled[col].clip(lower=-3, upper=3)
        return scaled.dropna()

    # def reset(self, seed=None, options=None):
    #     print("Dentro do reset TradingEnv")
    #     super().reset(seed=seed)
    #     self.balance = self.initial_balance
    #     self.current_step = 0
    #     window_size = TAMANHO_JANELA
    #
    #     # Verificar se há dados suficientes
    #     if len(self.historical_data) < window_size:
    #         raise ValueError("Dados históricos insuficientes para formar a janela.")
    #
    #     # Coletar janela inicial
    #     df_window = self.historical_data.iloc[:window_size]
    #     state = df_window[self.features].values.flatten().astype(np.float32)
    #     ml_signal = self.bot.predict_signal(df_window.iloc[-1:])
    #     self.data_index = window_size  # Atualizar data_index
    #     return state, {}
    def reset(self, seed=None, options=None):
        #print(Fore.YELLOW + "Dentro do reset TradingEnv")

        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.current_step = 0
        window_size = TAMANHO_JANELA

        # Garantir inicialização idêntica à operação real
        self.current_sl = None
        self.current_tp = None
        self.open_positions = []

        # Verificação robusta de dados históricos
        if len(self.historical_data) < window_size:
            self._generate_sideways_market_data()  # Nova função necessária
            print(Fore.RED + "⚠️ Dados insuficientes. Gerando dados sintéticos...")
            synthetic_data = np.random.randn(window_size, len(self.features))
            state = synthetic_data.flatten().astype(np.float32)
        else:
            df_window = self.historical_data.iloc[:window_size]
            state = df_window[self.features].values.flatten().astype(np.float32)

        ml_signal = self.bot.predict_signal(df_window.iloc[-1:]) if not df_window.empty else 0
        self.data_index = window_size
        return state, {}

    def step(self, action):
        # print(Fore.YELLOW + "Dentro do step TradingEnv")

        try:
            if not mt5.initialize():
                print(Fore.RED + "❌ Falha na conexão com o MT5 durante o step!")
                return self._get_next_state(), 0, True, False, {}

            # 1. Obter dados do mercado
            tick = mt5.symbol_info_tick(self.bot.symbol)
            spread = tick.ask - tick.bid if tick else 0.0002
            slippage = np.random.normal(0, SLIPPAGE_FACTOR)

            # 2. Obter preços atual e seguinte
            current_price = self._get_current_price(tick)
            next_price = self._get_next_price() + slippage

            # 3. Processar ação
            action_type, size_multiplier = self.ACTIONS_MAP.get(action, ('wait', 0))

            if action == 7:  # Ação de trailing stop
                self._simulate_trailing_stop()

            # 4. Calcular parâmetros do trade
            atr = self.bot.calculate_atr()
            stop_loss_pips = (atr * ATR_SL_MULTIPLIER) / 0.0001 if atr else 0
            volume = self.bot.calculate_volume(stop_loss_pips) * size_multiplier if atr else 0

            # 5. Calcular recompensa com todos os parâmetros
            reward = self._calculate_reward(
                action_type=action_type,
                size_multiplier=size_multiplier,
                current_price=current_price,
                next_price=next_price,
                volume=volume,
                spread=spread,
                slippage=slippage
            )

            # 6. Atualizar estado do ambiente
            self._update_state(action, current_price, next_price, volume, reward)
            self.data_index = min(self.data_index + 1, len(self.historical_data) - 1)

            # 7. Verificar condições de término
            done = (
                    self.current_step >= self.max_steps or
                    self.data_index >= len(self.historical_data) - 1 or
                    self.balance <= self.initial_balance * 0.7  # Stop loss de 30%
            )

            # 8. Preparar próximo estado e informações
            next_state = self._get_next_state()
            info = {
                "balance": self.balance,
                "spread": spread,
                "slippage": slippage,
                "action": action_type,
                "position_size": volume
            }

            return next_state, reward, done, False, info

        except Exception as e:
            print(Fore.RED + f"❌ Erro crítico no step: {str(e)}")
            return self._get_next_state(), 0, True, False, {}
        finally:
            mt5.shutdown()
            self.current_step += 1

    def _get_prices(self, action, data_index):
        #print(Fore.YELLOW + "Dentro do _get_prices TradingEnv")

        if data_index < len(self.historical_data) - 1:
            current_row = self.historical_data.iloc[data_index]
            next_row = self.historical_data.iloc[data_index + 1]
            return current_row[self.price_col], next_row[self.price_col]

        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return 0, 0

        action_type, _ = self.ACTIONS_MAP[action]
        if action_type == 'buy':
            return tick.ask, tick.bid
        elif action_type == 'sell':
            return tick.bid, tick.ask
        return (tick.ask + tick.bid) / 2, (tick.ask + tick.bid) / 2

    def _get_current_price(self, tick=None):
        """Obtém o preço atual considerando dados históricos e ticks ao vivo"""
        try:
            # Prioridade para dados em tempo real
            if tick is None:
                tick = mt5.symbol_info_tick(self.bot.symbol)

            if tick and tick.ask > 0 and tick.bid > 0:
                return (tick.ask + tick.bid) / 2  # Preço médio

            # Fallback para dados históricos
            if self.data_index < len(self.historical_data):
                return self.historical_data[self.price_col].iloc[self.data_index]

            return self.historical_data[self.price_col].iloc[-1]  # Último preço disponível

        except Exception as e:
            print(Fore.RED + f"Erro no _get_current_price: {str(e)}")
            return 0.0  # Valor padrão seguro

    def _get_next_price(self):
        # print(Fore.YELLOW + "Dentro do _get_next_price TradingEnv")
        try:
            if self.data_index + 1 < len(self.historical_data):
                return self.historical_data[self.price_col].iloc[self.data_index + 1]

            # Fallback para dados ao vivo
            tick = mt5.symbol_info_tick(self.bot.symbol)
            return (tick.ask + tick.bid) / 2 if tick else self.historical_data[self.price_col].iloc[-1]

        except Exception as e:
            print(Fore.RED + f"Erro no _get_next_price: {str(e)}")
            return self.historical_data[self.price_col].iloc[-1] if len(self.historical_data) > 0 else 0

    def _calculate_simulated_atr(self, period=ATR_PERIOD):
        """Calcula o ATR simulado para operações históricas"""
        try:
            if len(self.historical_data) < period + 1:
                return 0.0

            df = self.historical_data.iloc[-period - 1:]
            return df['atr'].iloc[-1] if 'atr' in df.columns else 0.0

        except Exception as e:
            print(Fore.RED + f"Erro no cálculo do ATR: {str(e)}")
            return 0.0

    def _calculate_reward(self, action_type, size_multiplier, current_price, next_price, volume, spread, slippage):
        # Cálculo base do PnL
        if action_type == 'buy':
            pnl = (next_price - current_price) * volume * 10000
        elif action_type == 'sell':
            pnl = (current_price - next_price) * volume * 10000
        else:  # Ação de wait ou trailing
            return 0  # Sem recompensa imediata

        # Custos de transação
        commission = volume * COMMISSION * 2  # Entrada e saída
        spread_cost = volume * spread
        slippage_cost = volume * abs(slippage)

        # Aplicar custos ao PnL
        pnl -= (commission + spread_cost + slippage_cost)

        # Impacto no saldo
        balance_impact = pnl / self.balance if self.balance > 0 else 0
        base_reward = balance_impact * 100

        # Fatores de mercado
        adx = self.historical_data['adx'].iloc[-1] if 'adx' in self.historical_data else 0
        trend_strength = 1 + (adx / 100)  # Multiplicador de tendência

        # Bônus por momentum
        roc_5 = self.historical_data['roc_5'].iloc[-1] if 'roc_5' in self.historical_data else 0
        momentum_bonus = abs(roc_5) * 50  # Bônus de até 50 pontos

        # Recompensa final
        reward = (base_reward * trend_strength) + momentum_bonus

        # Penalidades por contra-tendência
        trend = self.bot.check_market_trend()
        if (trend['direction'] == "UP_STRONG" and action_type == "sell") or \
                (trend['direction'] == "DOWN_STRONG" and action_type == "buy"):
            reward -= 100  # Penalidade alta por operar contra tendência forte
        elif (trend['direction'] == "UP" and action_type == "sell") or \
                (trend['direction'] == "DOWN" and action_type == "buy"):
            reward -= 50  # Penalidade média por operar contra tendência

        return reward

    def _calculate_inactivity_penalty(self, current_price):
        #print(Fore.YELLOW + "Dentro do _calculate_inactivity_penalty TradingEnv")

        recent_actions = self.action_history[-10:]
        inaction_count = sum(1 for a in recent_actions if self.ACTIONS_MAP[a][0] == 'wait')
        volatility = abs(self.calculate_market_volatility())
        opportunity_cost = abs(self.calculate_opportunity_cost(current_price))
        penalty = -2 * (1 + (self.current_step / self.max_steps))
        penalty -= inaction_count * 0.5
        penalty -= volatility * 5
        penalty -= opportunity_cost * 3
        return penalty

    def _calculate_trading_costs(self, volume, price):
        #print(Fore.YELLOW + "Dentro do _calculate_trading_costs TradingEnv")

        spread_cost = SPREAD_COST * volume * price
        commission = COMMISSION * volume
        slippage = SLIPPAGE_FACTOR * volume
        return spread_cost + commission + slippage

    def _update_state(self, action, current_price, next_price, volume, reward):
        #print(Fore.YELLOW + "Dentro do _update_state TradingEnv")

        self.action_history.append(action)
        if len(self.action_history) > 10:
            self.action_history.pop(0)
        self.balance += reward
        self.data_index += 1
        self.current_step += 1
        self.logger.info(
            Fore.BLUE + f"Step: {self.current_step} | Balance: {self.balance:.2f} | Action: {self.ACTIONS_MAP[action]} | Volume: {volume:.2f}")

    # def _get_next_state(self):
    #     #print(Fore.YELLOW + "Dentro do _get_next_state TradingEnv")
    #
    #     window_size = TAMANHO_JANELA
    #     if self.data_index < len(self.historical_data):
    #         start_idx = max(0, self.data_index - window_size + 1)
    #         end_idx = self.data_index + 1
    #         df_window = self.historical_data.iloc[start_idx:end_idx]
    #         if len(df_window) < window_size:
    #             padding = np.zeros((window_size - len(df_window), len(self.features)))
    #             padded_values = np.vstack([padding, df_window[self.features].values])
    #             state = padded_values.flatten().astype(np.float32)
    #         else:
    #             state = df_window[self.features].values.flatten().astype(np.float32)
    #     else:
    #         state = self.bot.get_live_data()
    #         if state is None or len(state) < (25 * window_size):
    #             state = np.zeros(25 * window_size, dtype=np.float32)
    #     ml_signal = self.bot.predict_signal(
    #         pd.DataFrame([state[-25:]], columns=self.features))
    #     return state

    def _get_next_state(self):
        window_size = TAMANHO_JANELA
        try:
            if self.data_index < len(self.historical_data):
                start_idx = max(0, self.data_index - window_size + 1)
                end_idx = self.data_index + 1
                df_window = self.historical_data.iloc[start_idx:end_idx]

                # Garantir o preenchimento correto se faltarem dados
                if len(df_window) < window_size:
                    padding = np.zeros((window_size - len(df_window), len(self.features)))
                    df_values = np.vstack([padding, df_window[self.features].values])
                else:
                    df_values = df_window[self.features].values

                return df_values.flatten().astype(np.float32)
        except Exception as e:
            print(f"Erro na formação do estado: {str(e)}")
            return np.zeros(25 * TAMANHO_JANELA, dtype=np.float32)

    def update_data(self, new_df):
        #print(Fore.YELLOW + "Dentro do update_data TradingEnv")

        full_df = pd.concat([self.historical_data, new_df], axis=0)
        self.historical_data = full_df[~full_df.index.duplicated(keep='last')].iloc[-5000:]
        self.historical_data = self._normalize_data(self.historical_data)

    def calculate_market_volatility(self):
        #print(Fore.YELLOW + "Dentro do calculate_market_volatility TradingEnv")

        window = 50
        closes_std = self.historical_data[self.price_col].rolling(window).std().iloc[-1]
        return abs(closes_std / self.historical_data[self.price_col].iloc[-1])

    def calculate_opportunity_cost(self, current_price):
        #print(Fore.YELLOW + "Dentro do calculate_opportunity_cost TradingEnv")

        lookback = 20
        future_prices = self.historical_data[self.price_col].shift(-lookback)
        valid_prices = future_prices.iloc[:len(future_prices) - lookback]
        max_gain = np.max(future_prices - current_price) if len(future_prices) > 0 else 0
        return abs(max_gain * 0.1)

    def render(self, mode='human'):
        #print(Fore.YELLOW + "Dentro do render TradingEnv")

        print(Fore.BLUE + f"Step: {self.current_step} | Balance: ${self.balance:.2f}")

    def get_final_balance(self):
        #print(Fore.YELLOW + "Dentro do get_final_balance TradingEnv")

        return self.episode_balances[-1] if self.episode_balances else self.initial_balance

    def get_current_balance(self):
        #print(Fore.YELLOW + "Dentro do get_current_balance TradingEnv")

        return self.balance

    def get_max_drawdown(self):
        #print(Fore.YELLOW + "Dentro do get_max_drawdown TradingEnv")

        """Calcula o drawdown máximo com base no histórico de saldos."""
        if not self.episode_balances:
            return 0.0

        peak = self.episode_balances[0]
        max_drawdown = 0.0

        for balance in self.episode_balances:
            if balance > peak:
                peak = balance
            drawdown = (peak - balance) / peak * 100  # Em porcentagem
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return max_drawdown

    def _normalize_live_data(self, df, window_size):
        # Replicar exatamente o pré-processamento do ambiente de treino
        for col in self.features:
            df[col] = (df[col] - self.training_scalers[col]['mean']) / self.training_scalers[col]['std']

        return df.dropna()

    def update_live_data(self):
        #print(Fore.YELLOW + "Dentro do update_live_data TradingEnv")

        """Atualiza os dados do ambiente com informações em tempo real"""
        # Garantir mesma normalização usada no treino
        raw_rates = mt5.copy_rates_from_pos(...)
        df = calculate_features(pd.DataFrame(raw_rates))

        # Aplicar EXATAMENTE a mesma normalização do ambiente de treino
        df = self._normalize_live_data(df, window_size)  # Nova função necessária

        try:
            rates = mt5.copy_rates_from_pos(self.symbol, TIMEFRAME, 0, TEMPO_DADOS_COLETADOS)
            df = pd.DataFrame(rates)
            df = calculate_features(df).dropna()
            self.historical_data = self._normalize_data(df)
            print(Fore.LIGHTGREEN_EX + "🔄 Dados do ambiente RL atualizados com sucesso!")
        except Exception as e:
            print(Fore.RED + f"Erro ao atualizar dados: {str(e)}")

# ==================================================================================================================
# -----------------------------
# Classe para ajuste do schedule de exploração
# -----------------------------
class ConstantSchedule:
    #print(Fore.YELLOW + "Dentro do ConstantSchedule")

    def __init__(self, value):
        #print(Fore.YELLOW + "Dentro do __init__ ConstantSchedule")
        self._value = value

    def get_value(self):
        #print(Fore.YELLOW + "Dentro do get_value ConstantSchedule")
        return self._value

    def __call__(self, _):
        #print(Fore.YELLOW + "Dentro do __call__ ConstantSchedule")
        return self._value

# ==================================================================================================================
# -----------------------------
# Função de monitoramento da assertividade com PerformanceTracker
# -----------------------------
async def monitor_accuracy_tp_sl(trading_bot):
    #print(Fore.YELLOW + "Dentro do monitor_accuracy_tp_sl")

    current_balance = mt5.account_info().balance if mt5.account_info() else saldo_inicial

    # Header dinâmico baseado no modo de operação
    if OPERATION_MODE == "rl":
        header = "timestamp,rl_pred,rl_price,rl_tp,rl_sl,rl_outcome,current_close,new_close"
    else:
        header = "timestamp,ml_pred,ml_price,ml_tp,ml_sl,ml_outcome,rl_pred,rl_price,rl_tp,rl_sl,rl_outcome,integrated_pred,integrated_price,integrated_tp,integrated_sl,integrated_outcome,current_close,new_close"

    logger.info(header)

    while True:
        state = trading_bot.get_live_data()
        if state is None:
            await asyncio.sleep(INTERVALO_PREDICOES)
            continue

        # Modificação para modo RL
        if OPERATION_MODE == "rl":
            ml_pred = None
            with RL_MODEL_LOCK:
                rl_action, _ = RL_MODEL.predict(state, deterministic=True)
            integrated_pred = rl_action  # Usa ação RL diretamente
        else:
            ml_pred = state[-1] if OPERATION_MODE != "rl" else None
            with RL_MODEL_LOCK:
                rl_action, _ = RL_MODEL.predict(state, deterministic=True)
            # Lógica original para integrated_pred
            if rl_action == 0:
                integrated_pred = None
            elif rl_action == 1 and ml_pred == 1:
                integrated_pred = 1
            elif rl_action == 2 and ml_pred == 0:
                integrated_pred = 0
            else:
                integrated_pred = None

        # Cálculo de parâmetros condicionais
        ml_price, ml_tp, ml_sl, _ = (None, None, None, None)
        if OPERATION_MODE != "rl":
            ml_price, ml_tp, ml_sl, _ = trading_bot.get_trade_params(ml_pred) if ml_pred is not None else (
            None, None, None, None)

        rl_price, rl_tp, rl_sl, _ = trading_bot.get_trade_params(
            1 if rl_action in [1, 2, 3] else 0 if rl_action in [4, 5, 6] else None
        ) if rl_action is not None else (None, None, None, None)

        integrated_price, integrated_tp, integrated_sl, _ = (None, None, None, None)
        if OPERATION_MODE == "both":
            integrated_price, integrated_tp, integrated_sl, _ = trading_bot.get_trade_params(
                integrated_pred) if integrated_pred is not None else (None, None, None, None)

        # Coleta de preços
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
            if new_rates and new_rates[0]["time"] != initial_time:
                new_close = new_rates[0]["close"]
                break

        # Função de resultado genérica
        def trade_outcome(signal, price, tp, sl, new_close):
            if price is None or tp is None or sl is None:
                return "N/A"

            # Mapeamento de ações para sinais
            if OPERATION_MODE == "rl":
                if signal in [1, 2, 3]:  # Compras
                    return "TP" if new_close >= tp else "SL" if new_close <= sl else "None"
                elif signal in [4, 5, 6]:  # Vendas
                    return "TP" if new_close <= tp else "SL" if new_close >= sl else "None"
                else:  # Esperar ou trail
                    return "N/A"
            else:
                if signal == 1:
                    return "TP" if new_close >= tp else "SL" if new_close <= sl else "None"
                elif signal == 0:
                    return "TP" if new_close <= tp else "SL" if new_close >= sl else "None"
                else:
                    return "N/A"

        # Cálculo de resultados
        ml_outcome = "N/A"
        if OPERATION_MODE != "rl":
            ml_outcome = trade_outcome(ml_pred, ml_price, ml_tp, ml_sl, new_close)

        rl_outcome = trade_outcome(rl_action, rl_price, rl_tp, rl_sl, new_close)
        integrated_outcome = "N/A" if OPERATION_MODE == "rl" else trade_outcome(integrated_pred, integrated_price,
                                                                                integrated_tp, integrated_sl, new_close)

        # Construção da linha de log
        if OPERATION_MODE == "rl":
            log_line = (
                f"{datetime.now().isoformat()},"
                f"{decode_rl_signal(rl_action)},"
                f"{rl_price or 'N/A'},{rl_tp or 'N/A'},{rl_sl or 'N/A'},"
                f"{rl_outcome},"
                f"{current_close},{new_close}"
            )
        else:
            log_line = (
                f"{datetime.now().isoformat()},"
                f"{decode_ml_signal(ml_pred)},{ml_price or 'N/A'},{ml_tp or 'N/A'},{ml_sl or 'N/A'},{ml_outcome},"
                f"{decode_rl_signal(rl_action)},{rl_price or 'N/A'},{rl_tp or 'N/A'},{rl_sl or 'N/A'},{rl_outcome},"
                f"{decode_integrated_signal(integrated_pred)},{integrated_price or 'N/A'},{integrated_tp or 'N/A'},"
                f"{integrated_sl or 'N/A'},{integrated_outcome},"
                f"{current_close},{new_close}"
            )

        logger.info(log_line)
        print(Fore.BLUE + f"Monitoramento: {log_line}")

        # Registro de trades
        if OPERATION_MODE == "rl" and rl_outcome in ["TP", "SL"] and rl_price is not None:
            entry = rl_price
            exit_price = new_close
            pnl = (exit_price - entry) if rl_action in [1, 2, 3] else (entry - exit_price)

            order_details = (
                f"Ativo: {symbol} | Tipo: {decode_rl_signal(rl_action)} | "
                f"Entrada: {entry:.5f} | Saída: {exit_price:.5f} | "
                f"TP: {rl_tp or 'N/A':.5f} | SL: {rl_sl or 'N/A':.5f}"
            )
            trading_bot.log_order("RESULTADO_ORDEM", order_details, outcome=rl_outcome, pnl=pnl)
            tracker.add_trade(entry, exit_price, pnl)

        elif OPERATION_MODE != "rl" and integrated_outcome in ["TP", "SL"] and integrated_price is not None:
            entry = integrated_price
            exit_price = new_close
            pnl = (exit_price - entry) if integrated_pred == 1 else (entry - exit_price)

            order_details = (
                f"Ativo: {symbol} | Tipo: {decode_integrated_signal(integrated_pred)} | "
                f"Entrada: {entry:.5f} | Saída: {exit_price:.5f} | "
                f"TP: {integrated_tp or 'N/A':.5f} | SL: {integrated_sl or 'N/A':.5f}"
            )
            trading_bot.log_order("RESULTADO_ORDEM", order_details, outcome=integrated_outcome, pnl=pnl)
            tracker.add_trade(entry, exit_price, pnl)

        if tracker.trades:
            summary = tracker.summary()
            print(Fore.CYAN + "\nResumo de Performance:")
            print(Fore.CYAN + f"• Trades Totais: {summary['total_trades']}")
            print(Fore.CYAN + f"• Taxa de Acerto: {summary['win_rate']:.2%}")
            print(Fore.CYAN + f"• Ganho Médio: ${summary['avg_win']:.2f}")
            print(Fore.CYAN + f"• Perda Média: ${summary['avg_loss']:.2f}")

        await asyncio.sleep(INTERVALO_PREDICOES)

# ==================================================================================================================
# -----------------------------
# Função de treinamento do agente RL usando execução em thread dedicada
# -----------------------------
def train_rl_agent_mp():
    global RL_MODEL
    bot = TradingBot(symbol)

    # 1. Configurar política e callbacks
    policy_kwargs = dict(
        features_extractor_class=TemporalAttentionExtractor,
        features_extractor_kwargs={},
        net_arch=[256, 128]
    )

    # 2. Criar ambiente inicial
    def create_env():
        new_data = get_historical_rates(symbol, TIMEFRAME, TEMPO_DADOS_COLETADOS)
        df = calculate_features(pd.DataFrame(new_data)).dropna()
        base_env = TradingEnv(trading_bot=bot, initial_balance=SALDO_INICIAL_TREINO)
        base_env.update_data(df)
        return Monitor(base_env)

    env = create_env()

    # 3. Tentar carregar modelo existente
    model_path = "dqn_trading_agent_best.zip"
    if os.path.exists(model_path):
        try:
            RL_MODEL = DQN.load(
                model_path,
                env=env,
                custom_objects={
                    "policy_class": CustomDQNPolicy,
                    "policy_kwargs": policy_kwargs
                },
                device='auto'
            )
            print(Fore.GREEN + "✅ Modelo RL carregado com sucesso!")
            return RL_MODEL
        except Exception as e:
            print(Fore.RED + f"Erro ao carregar modelo: {str(e)}")

    # 4. Criar novo modelo se não existir
    print(Fore.YELLOW + "⚠️ Criando novo modelo RL...")

    # Parâmetros de exploração
    exploration_params = {
        'exploration_fraction': 0.3,
        'exploration_initial_eps': 1.0,
        'exploration_final_eps': 0.05,
        'learning_starts': 10000
    }

    RL_MODEL = MetaDQN(
        CustomDQNPolicy,
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-4,
        buffer_size=100000,
        batch_size=256,
        gamma=0.99,
        **exploration_params,
        target_update_interval=2000,
        train_freq=PASSOS_ATUALIZAR_REDE,
        gradient_steps=PASSOS_BACKPROP,
        tensorboard_log="./dqn_tensorboard/",
        device='auto',
        verbose=1
    )

    # 5. Configurar callbacks
    eval_callback = EvalCallback(
        env,
        best_model_save_path='./',
        log_path='./logs/',
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
        callback_after_eval=StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=5,
            min_evals=10
        )
    )

    live_val_env = TradingEnv(trading_bot=bot, initial_balance=SALDO_INICIAL_TREINO)
    live_val_callback = LiveValidationCallback(live_val_env, freq=10000)
    balance_callback = BalanceCallback()

    # 6. Treinamento principal
    try:
        RL_MODEL.learn(
            total_timesteps=N_INTERACOES,
            callback=[eval_callback, live_val_callback, balance_callback],
            tb_log_name="dqn_log",
            progress_bar=True
        )
    except KeyboardInterrupt:
        print(Fore.YELLOW + "\nTreinamento interrompido pelo usuário")

    # 7. Salvamento final
    save_model(RL_MODEL, prefix="dqn_trading_agent")
    return RL_MODEL

# ==================================================================================================================
# -----------------------------
# Atualização assíncrona do modelo RL em thread dedicada
# -----------------------------
async def update_rl_model_periodically():
    global RL_MODEL
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=1) as executor:
        while True:
            await asyncio.sleep(TIME_TREINO)

            # Verificar desempenho recente
            if len(tracker.trades) == 0:
                print(Fore.BLUE + "Nenhum trade realizado ainda. Pulando verificação...")
                continue

            # Calcular métricas de desempenho
            recent_trades = tracker.trades[-10:]  # Últimos 10 trades
            total_pnl = sum(trade['pnl'] for trade in recent_trades)
            win_rate = len([t for t in recent_trades if t['pnl'] > 0]) / len(recent_trades) if recent_trades else 0

            # Condições para retreinamento
            losing_money = total_pnl < 0
            bad_win_rate = win_rate < 0.4
            drawdown = tracker.summary().get('max_drawdown', 0) > 0.1 * saldo_inicial

            if losing_money or bad_win_rate or drawdown:
                print(Fore.BLUE + "🔄 Desempenho ruim detectado. Atualizando o agente RL...")
                print(Fore.BLUE + f"• PnL Recente: ${total_pnl:.2f}")
                print(Fore.BLUE + f"• Win Rate: {win_rate:.2%}")
                print(Fore.BLUE + f"• Drawdown: {tracker.summary().get('max_drawdown', 0):.2f}")

                new_model = await loop.run_in_executor(executor, train_rl_agent_mp)
                with RL_MODEL_LOCK:
                    RL_MODEL = new_model
                print(Fore.LIGHTGREEN_EX + "✅ Agente RL atualizado com sucesso!")
            else:
                print(Fore.BLUE + "✅ Desempenho dentro dos parâmetros aceitáveis. Não é necessário retreinar.")

# ==================================================================================================================
# -----------------------------
# Função principal
# -----------------------------
async def main():
    global RL_MODEL

    # Conexão MT5 robusta
    for _ in range(5):
        if mt5.initialize():
            break
        print(Fore.RED + "❌ Falha na conexão. Tentando novamente em 5s...")
        await asyncio.sleep(5)
    else:
        print(Fore.RED + "❌ Falha crítica na conexão ao MT5!")
        return

    # Verificação final
    if not mt5.initialize() or not mt5.account_info():
        print(Fore.RED + "❌ Conexão MT5 inválida!")
        return

    # Inicialização do bot
    await send_initial_balance()
    bot = TradingBot(symbol)

    # Gerenciamento do ambiente de treino
    if choose_training_mode():
        print(Fore.CYAN + "\n=== MODO DE TREINAMENTO RL ===")
        RL_MODEL = train_rl_agent_mp()
    else:
        print(Fore.CYAN + "\n=== MODO DE TRADING ===")
        model_path = "dqn_trading_agent_best.zip"
        if os.path.exists(model_path):
            env = TradingEnv(trading_bot=bot, initial_balance=saldo_inicial)
            try:
                RL_MODEL = DQN.load(
                    model_path,
                    env=env,
                    custom_objects={
                        "policy_class": CustomDQNPolicy,
                        "policy_kwargs": dict(
                            features_extractor_class=TemporalAttentionExtractor,
                            net_arch=[256, 128]
                        )
                    },
                    device='auto'
                )
                print(Fore.GREEN + "✅ Modelo RL carregado com sucesso!")
            except Exception as e:
                print(Fore.RED + f"Erro ao carregar modelo: {str(e)}")
                RL_MODEL = train_rl_agent_mp()
        else:
            print(Fore.YELLOW + "⚠️ Modelo não encontrado. Iniciando treinamento...")
            RL_MODEL = train_rl_agent_mp()

    # Configuração de tarefas
    tasks = [
        bot.run(),
        check_connection(),
        send_telegram_message(),
        update_rl_model_periodically()
    ]

    if OPERATION_MODE in ["ml", "both"]:
        tasks.append(bot.update_model_periodically())

    await asyncio.gather(*tasks)

if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)

    # Forçar modo de diagnóstico do MetaTrader
    mt5.shutdown()
    if not mt5.initialize(log_level=logging.DEBUG, log_file="mt5_debug.log"):
        print("Falha crítica na inicialização do MT5")
        exit()

    # Verificação completa da conexão
    print("Versão MT5:", mt5.__version__)
    print("Terminal info:", mt5.terminal_info())
    print("Símbolo info:", mt5.symbol_info(symbol))

    asyncio.run(main())
