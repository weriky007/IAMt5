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

# =============================
# Variáveis globais
# =============================
RL_MODEL_LOCK = threading.Lock()
TEMPO_DADOS_COLETADOS = 96300  # número de candles de 1 minuto
TIME_TREINO = 10800  # 33 min para treinamento incremental do ML e atualização do RL
INTERVALO_PREDICOES = 15 #60
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

TAMANHO_JANELA = 21

#CONFIGURACOES TREINO RL
SALDO_INICIAL_TREINO = 10000
N_INTERACOES = 10000  #MIN 100000
QUANTIDADE_ACERTOS_TREINO = 5 #MIN 5
MAX_CICLES = 25 #MIN 50
PASSOS_ATUALIZAR_REDE = 4 #MIN 4
PASSOS_BACKPROP = 2 #MIN 2

TRAILING_STEP = 0.0002  # 2 pips para pares de 5 decimais
TRAILING_MAX_DISTANCE = 0.0020  # 20 pips

TEMPO_MENSAGEM = 3600 # 1 hora

# =============================
#register_policy("CustomDQNPolicy", CustomDQNPolicy)

# =============================
# Cores print
# =============================
init()

# =============================
# Função para escolha do modo de operação
# =============================
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

# =============================
# Função para escolher entre treinamento ou trade
# =============================
def choose_training_mode():
    print(Fore.LIGHTWHITE_EX + "Escolha a opção de inicialização:")
    print(Fore.BLUE + "1. Entrar no ambiente de treinamento do RL")
    print(Fore.BLUE + "2. Iniciar diretamente o trading")
    choice = input(Fore.LIGHTWHITE_EX + "Digite a opção (1 ou 2): ")
    return choice == "1"

# Variável global com o modo escolhido
OPERATION_MODE = choose_mode()

# Verifica se o arquivo de ordens existe e adiciona cabeçalho
if not os.path.exists('ordens.txt'):
    with open('ordens.txt', 'w', encoding='utf-8') as f:
        f.write("REGISTRO DE ORDENS\n")
        f.write("Formato: [DATA HORA] Balance: SALDO | Action: TIPO | Details: DETALHES | Outcome: RESULTADO | PnL: VALOR\n\n")

# =============================
# Classe para rastreamento de desempenho
# =============================
class PerformanceTracker:
    #print(Fore.YELLOW + "Dentro do PerformanceTracker")

    def __init__(self):
        #print(Fore.YELLOW + "Dentro do __init__ PerformanceTracker")

        self.trades = []

    def add_trade(self, entry, exit, pnl):
        #print(Fore.YELLOW + "Dentro do add_trade PerformanceTracker")
        self.trades.append({
            'entry': entry,
            'exit': exit,
            'pnl': pnl
        })

    def summary(self):
        #print(Fore.YELLOW + "Dentro do summary PerformanceTracker")
        if len(self.trades) == 0:
            return {}
        wins = [t for t in self.trades if t['pnl'] > 0]
        return {
            'total_trades': len(self.trades),
            'win_rate': len(wins) / len(self.trades),
            'avg_win': np.mean([t['pnl'] for t in wins]) if wins else 0,
            'avg_loss': np.mean([t['pnl'] for t in self.trades if t['pnl'] <= 0]) if len(self.trades) > len(wins) else 0
        }

# Cria uma instância global do tracker
tracker = PerformanceTracker()

# =============================
# Variaveis bot Telegram
# =============================
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

# =============================
# Função para versionamento de modelos
# =============================
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

# Configuração do logger com rotação
logger = logging.getLogger('trade_monitor')
logger.setLevel(logging.INFO)
handler = RotatingFileHandler('monitoring_log.csv', maxBytes=10 * 1024 * 1024, backupCount=5)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# =============================
# Funções de decodificação
# =============================
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

# =============================
# Conexão com MT5
# =============================
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

# =============================
# Função para calcular o volume dinamicamente com risco ajustado
# =============================
# def calculate_volume(self, atr):
#     balance = mt5.account_info().balance
#     risk_per_trade = 0.01  # 1% do capital
#     risk_amount = balance * risk_per_trade
#     pip_value = 10  # Valor por pip para AUDCAD
#     volume = risk_amount / (atr * pip_value)
#     return round(volume, 2)
# =============================
# Função para coletar dados históricos com validação
# =============================
def get_historical_rates(symbol, timeframe, candles):
    #print(Fore.YELLOW + "Dentro do get_historical_rates")

    start_time = datetime.now() - timedelta(minutes=candles)
    end_time = datetime.now()
    rates = mt5.copy_rates_from(symbol, timeframe, int(start_time.timestamp()), int(end_time.timestamp()))
    if rates is None or len(rates) == 0:
        print(Fore.RED + "⚠️ Aviso:"+Fore.LIGHTWHITE_EX+" Não foi possível obter dados históricos via copy_rates_from; tentando copy_rates_from_pos()...")
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, candles)
        if rates is None or len(rates) == 0:
            raise ValueError(Fore.RED + "Falha crítica: Não foi possível obter dados históricos")
    timestamps = pd.to_datetime(rates['time'], unit='s')
    time_diffs = np.diff(timestamps)
    if any(diff > pd.Timedelta('2min') for diff in time_diffs):
        print(Fore.RED + "⚠️ Aviso:"+Fore.LIGHTWHITE_EX+" Grandes gaps temporais nos dados históricos!")
    return rates

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

# class MetaDQN(DQN):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         # Otimizador específico para adaptação contínua
#         self.meta_optimizer = th.optim.Adam(self.policy.parameters(), lr=1e-3)
#
#     # def adapt(self, live_data=None, fast_batch=None):
#     #     """
#     #     Mecanismo de adaptação contínua.
#     #
#     #     Parâmetros:
#     #       - live_data: lista de novas observações.
#     #       - fast_batch: lote rápido de transições para atualização imediata.
#     #
#     #     Se 'fast_batch' for fornecido, realiza uma atualização rápida usando o meta_optimizer.
#     #     Caso contrário, se 'live_data' for fornecido e seu tamanho for múltiplo de 100,
#     #     chama o método learn para uma adaptação contínua.
#     #     """
#     #     if fast_batch is not None:
#     #         losses = []
#     #         for transition in fast_batch:
#     #             loss = self._calc_loss(transition)
#     #             losses.append(loss)
#     #             self.meta_optimizer.zero_grad()
#     #             loss.backward()
#     #             self.meta_optimizer.step()
#     #         return np.mean(losses)
#     #     elif live_data is not None:
#     #         if len(live_data) % 100 == 0:
#     #             self.learn(1000, reset_num_timesteps=False)
#
#     def adapt(self, live_data=None, batch_size=32):
#         """
#         Adaptação online com dados em tempo real
#         """
#         if live_data is not None:
#             # Garantir formato 2D mesmo para single instance
#             live_data = np.array(live_data)
#             if live_data.ndim == 1:
#                 live_data = live_data.reshape(1, -1)
#
#             # Converter para tensor
#             obs_tensor = th.tensor(live_data, dtype=th.float32).to(self.device)
#
#             # Gerar experiência sintética
#             with th.no_grad():
#                 actions, _ = self.predict(live_data)
#                 next_obs = obs_tensor.roll(-1, dims=0)
#                 rewards = th.randn(obs_tensor.size(0))
#
#             # Adicionar cada experiência com formatação correta
#             for i in range(obs_tensor.size(0)):
#                 self.replay_buffer.add(
#                     obs=obs_tensor[i].cpu().numpy().reshape(1, -1),  # (1, n_features)
#                     next_obs=next_obs[i].cpu().numpy().reshape(1, -1),
#                     action=np.array([actions[i]], dtype=np.int64),  # Array numpy
#                     reward=np.array([rewards[i].item()], dtype=np.float32),
#                     done=np.array([False], dtype=np.bool_),
#                     infos=[{}]
#                 )
#
#             # Treino rápido (mesmo código anterior)
#             if len(self.replay_buffer) > batch_size:
#                 losses = []
#                 for _ in range(3):
#                     samples = self.replay_buffer.sample(batch_size)
#                     loss = self._calc_loss(samples)
#                     losses.append(loss.item())
#
#                     self.meta_optimizer.zero_grad()
#                     loss.backward()
#                     self.meta_optimizer.step()
#
#                 return np.mean(losses)
#
#     def clone_for_scenario(self, env):
#         """Clona o modelo mantendo a arquitetura mas isolando os parâmetros"""
#         cloned_model = MetaDQN(
#             policy=self.policy_class,
#             env=env,
#             learning_rate=self.learning_rate,
#             buffer_size=self.buffer_size,
#             learning_starts=self.learning_starts,
#             batch_size=self.batch_size,
#             tau=self.tau,
#             gamma=self.gamma,
#             train_freq=self.train_freq,
#             gradient_steps=self.gradient_steps,
#             replay_buffer_class=self.replay_buffer_class,
#             replay_buffer_kwargs=self.replay_buffer_kwargs,
#             policy_kwargs=self.policy_kwargs,
#             device=self.device
#         )
#         # Correção aqui: usar set_parameters ao invés de load_parameters
#         cloned_model.set_parameters(self.get_parameters())
#         return cloned_model
#
#     def _calc_loss(self, samples: ReplayBufferSamples) -> th.Tensor:
#         """
#         Calcula a loss de forma compatível com a arquitetura customizada.
#         """
#         with th.no_grad():
#             target_q = self.target_q_net(samples.next_observations)
#             target_q = target_q.max(dim=1)[0].reshape(-1, 1)
#             target_q = samples.rewards + (1 - samples.dones) * self.gamma * target_q
#
#         current_q = self.q_net(samples.observations)
#         current_q = current_q.gather(1, samples.actions)
#
#         return th.nn.functional.mse_loss(current_q, target_q)

# class MetaDQN(DQN):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.meta_optimizer = th.optim.Adam(self.policy.parameters(), lr=1e-3)
#
#     def adapt(self, live_data=None, batch_size=32):
#         if live_data is not None:
#             live_data = np.array(live_data)
#             if live_data.ndim == 1:
#                 live_data = live_data.reshape(1, -1)
#
#             obs_tensor = th.tensor(live_data, dtype=th.float32).to(self.device)
#
#             with th.no_grad():
#                 actions, _ = self.predict(live_data)
#                 next_obs = obs_tensor.roll(-1, dims=0)
#                 rewards = th.randn(obs_tensor.size(0))
#
#             for i in range(obs_tensor.size(0)):
#                 self.replay_buffer.add(
#                     obs=obs_tensor[i].cpu().numpy().reshape(1, -1),
#                     next_obs=next_obs[i].cpu().numpy().reshape(1, -1),
#                     action=np.array([actions[i]], dtype=np.int64),
#                     reward=np.array([rewards[i].item()], dtype=np.float32),
#                     done=np.array([False], dtype=np.bool_),
#                     infos=[{}]
#                 )
#
#             if len(self.replay_buffer) > batch_size:
#                 losses = []
#                 for _ in range(3):
#                     samples = self.replay_buffer.sample(batch_size)
#                     loss = self._calc_loss(samples)
#                     losses.append(loss.item())
#                     self.meta_optimizer.zero_grad()
#                     loss.backward()
#                     self.meta_optimizer.step()
#                 return np.mean(losses)
#
#     def _calc_loss(self, samples: ReplayBufferSamples) -> th.Tensor:
#         with th.no_grad():
#             # Usar a rede target para obter os Q-values do próximo estado
#             next_q_values = self.target_q_net(samples.next_observations)
#             target_q = next_q_values.max(dim=1)[0].view(-1, 1)
#             target_q = samples.rewards + (1 - samples.dones) * self.gamma * target_q
#
#         current_q = self.q_net(samples.observations)
#         current_q = current_q.gather(1, samples.actions.long())
#
#         return th.nn.functional.mse_loss(current_q, target_q)
#
#     def clone_for_scenario(self, env):
#         cloned_model = MetaDQN(
#             policy=self.policy_class,
#             env=env,
#             learning_rate=self.learning_rate,
#             buffer_size=self.buffer_size,
#             learning_starts=self.learning_starts,
#             batch_size=self.batch_size,
#             tau=self.tau,
#             gamma=self.gamma,
#             train_freq=self.train_freq,
#             gradient_steps=self.gradient_steps,
#             replay_buffer_class=self.replay_buffer_class,
#             replay_buffer_kwargs=self.replay_buffer_kwargs,
#             policy_kwargs=self.policy_kwargs,
#             device=self.device
#         )
#         cloned_model.set_parameters(self.get_parameters())
#         return cloned_model

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

class MetaDQN(DQN):
    #print(Fore.YELLOW + "Dentro do MetaDQN")

    def __init__(self, policy, env, *args, **kwargs):
        # Garanta que a política é passada para a classe pai
        super().__init__(policy, env, *args, **kwargs)
        self.meta_optimizer = th.optim.Adam(self.policy.parameters(), lr=1e-3)

        #print(Fore.YELLOW + "Dentro do __init__ MetaDQN")


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

class TrendPrioritizedReplayBuffer(ReplayBuffer):
    #print(Fore.YELLOW + "Dentro do TrendPrioritizedReplayBuffer")

    def __init__(self, buffer_size: int, observation_space: gym.Space, action_space: gym.Space,
                 device: Union[str, th.device] = "auto"):
        #print(Fore.YELLOW + "Dentro do __init__ TrendPrioritizedReplayBuffer")

        super().__init__(buffer_size, observation_space, action_space, device)
        self.priorities = np.ones(buffer_size, dtype=np.float32)
        self.max_priority = 1.0

    def add(self, obs: np.ndarray, next_obs: np.ndarray, action: np.ndarray, reward: np.ndarray, done: np.ndarray,
            infos: list):
        #print(Fore.YELLOW + "Dentro do add TrendPrioritizedReplayBuffer")

        super().add(obs, next_obs, action, reward, done, infos)
        new_indices = np.arange(self.pos - len(reward), self.pos) % self.buffer_size
        for idx in new_indices:
            info = infos[idx % len(infos)] if infos else {}
            trend_strength = info.get('trend_strength', 0)
            current_reward = reward[idx % len(reward)]
            priority = 2.0 if trend_strength > 30 else 1.5 if current_reward > 0 else 1.0
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def sample(self, batch_size: int, env: Optional[gym.Env] = None, beta: float = 0.4) -> ReplayBufferSamples:
        #print(Fore.YELLOW + "Dentro do sample TrendPrioritizedReplayBuffer")

        # Calcular o tamanho atual usando pos e full
        current_size = self.pos if not self.full else self.buffer_size
        current_size = int(current_size)  # Garantir que é inteiro

        if current_size == 0:
            raise ValueError("Buffer vazio!")

        valid_priorities = self.priorities[:current_size]
        sum_priorities = valid_priorities.sum()

        if sum_priorities <= 0:
            probs = np.ones(current_size) / current_size  # Distribuição uniforme
        else:
            probs = valid_priorities / sum_priorities

        indices = np.random.choice(current_size, size=batch_size, p=probs)
        return super()._get_samples(indices)

    def __len__(self):
        #print(Fore.YELLOW + "Dentro do __len__ TrendPrioritizedReplayBuffer")

        """Retorna o tamanho atual do buffer"""
        if self.full:
            return self.buffer_size
        return self.pos

# def __len__(self):
#     return self.size

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

# =============================
# Nova função para obter dados em tempo real
# =============================
def get_realtime_data(window_size=TAMANHO_JANELA):
    #print(Fore.YELLOW + "Dentro do get_realtime_data")

    bot = TradingBot(symbol)
    return bot.get_live_data(window_size)

# =============================
# Função de validação no mercado real
# =============================
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

# def train_hybrid():
#     # Fase 1: Treinamento inicial com dados históricos
#     model = train_rl_agent_mp()
#
#     # Fase 2: Fine-tuning online
#     bot = TradingBot(symbol)  # Instância do bot para operações em tempo real
#
#     try:
#         while True:
#             # Coletar novos dados usando o método do bot
#             live_data = bot.get_live_data()  # ✅ Correção aqui
#
#             if live_data is not None:
#                 # Adaptação do modelo com dados recentes
#                 model.adapt(live_data=live_data)
#
#                 # Validação periódica
#                 if model.num_timesteps % 1000 == 0:
#                     validate_on_live_market(model, bot)
#
#             # Intervalo entre atualizações
#             time.sleep(TIME_TREINO)
#
#     except KeyboardInterrupt:
#         print("\nTreinamento híbrido interrompido pelo usuário")
#
#     return model
# async def train_hybrid():
#     # Fase 1: Treinamento inicial com dados históricos
#     model = train_rl_agent_mp()  # ← Adicione esta linha no início
#
#     # Fase 2: Fine-tuning online
#     bot = TradingBot(symbol)
#
#     try:
#         while True:
#             live_data = bot.get_live_data()
#
#             if live_data is not None:
#                 # Converter para tensor antes de adaptar
#                 # processed_data = preprocess_data(live_data)  # ← Adicione pré-processamento
#                 if live_data is not None:
#                     model.adapt(live_data=live_data)  # Já está pré-processado
#                 model.adapt(live_data=processed_data)
#
#                 if model.num_timesteps % 1000 == 0:
#                     validate_on_live_market(model, bot)
#
#             # Espera assíncrona CORRETA
#             await asyncio.sleep(TIME_TREINO)  # ✅ Correção final
#
#     except KeyboardInterrupt:
#         print("\nInterrupção manual")
#
#     return model

# async def train_hybrid():
#     print("Dentro do train_hybrid")
#     # Fase 1: Treinamento inicial com dados históricos
#     model = train_rl_agent_mp()
#
#     # Fase 2: Fine-tuning online
#     bot = TradingBot(symbol)
#
#     try:
#         while True:
#             live_data = bot.get_live_data()
#
#             if live_data is not None:
#                 # Já está pré-processado pelo bot.get_live_data()
#                 model.adapt(live_data=live_data)  # ✅ Correção definitiva
#
#                 if model.num_timesteps % 1000 == 0:
#                     validate_on_live_market(model, bot)
#
#             await asyncio.sleep(TIME_TREINO)
#
#     except KeyboardInterrupt:
#         print("\nInterrupção manual")
#
#     return model
# async def train_hybrid():
#     print("Dentro do train_hybrid")
#     # Fase 1: Treinamento inicial com dados históricos
#     model = train_rl_agent_mp()
#
#     # Fase 2: Fine-tuning online
#     bot = TradingBot(symbol)
#     loop = asyncio.get_running_loop()
#
#     try:
#         while True:
#             live_data = bot.get_live_data()
#
#             if live_data is not None:
#                 # Já está pré-processado pelo bot.get_live_data()
#                 model.adapt(live_data=live_data)  # ✅ Correção definitiva
#
#                 if model.num_timesteps % 1000 == 0:
#                     # Executar validação em thread separada para não bloquear
#                     await loop.run_in_executor(None, validate_on_live_market, model, bot)
#
#             await asyncio.sleep(TIME_TREINO)
#
#     except KeyboardInterrupt:
#         print("\nInterrupção manual")
#
#     return model

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

# =============================
# Função auxiliar para adicionar indicadores técnicos
# =============================
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

# =============================
# Função unificada para cálculo das features
# =============================
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

# =============================
# Mecanismo de reconexão robusto
# =============================
async def check_connection():
    #print(Fore.YELLOW + "Dentro do check_connection")

    while True:
        if not mt5.initialize():
            print(Fore.RED + "❌ Conexão perdida. Tentando reconectar em 10s...")
            mt5.shutdown()
            await asyncio.sleep(10)
        else:
            await asyncio.sleep(60)

# =============================
# validação com dados ao vivo durante o treino
# =============================
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

# =============================
# validação com dados ao vivo durante o treino
# =============================
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

# =============================
# Classe TradingBot (Modelo ML incremental com validação holdout)
# =============================
class TradingBot:
    #print(Fore.YELLOW + "Dentro do TradingBot")

    def __init__(self, symbol):
        #print(Fore.YELLOW + "Dentro do __init__ TradingBot")

        self.model_path = "modelo_ml.pkl"
        self.symbol = symbol
        self.ml_model = self.load_or_train_model()
        self.historical_features = pd.DataFrame()  # Inicializa vazio
        self._load_initial_data()  # ← Novo método para carregar dados iniciais
        # Supondo que self.model seja o modelo RL (caso exista)
        self.model = None
        self.rl_model = None
        # Integração do dashboard de performance
        self.dashboard = PerformanceDashboard()

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
            return np.zeros(25 * window_size + 1)  # Retorna zeros se falhar

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
        return state

    # def predict_signal(self, df):
    #     print("Dentro do predict_signal TradingBot")
    #     if df is None or self.ml_model is None:
    #         return None
    #     return self.ml_model.predict(df.values)[0]
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
        #print(Fore.YELLOW + "Dentro do get_trade_params TradingBot")

        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None or tick.ask == 0 or tick.bid == 0:
            return None, None, None, None

        price = tick.ask if signal == 1 else tick.bid
        atr = self.calculate_atr(period=14)  # ATR de 14 períodos
        if atr is None:
            return None, None, None, None

        if signal == 1:
            tp = price + (atr * ATR_TP_MULTIPLIER)
            sl = price - (atr * ATR_SL_MULTIPLIER)
            if sl >= price:
                print(Fore.RED + "⚠️ SL inválido para operação de compra")
                return None, None, None, None
        else:
            tp = price - (atr * ATR_TP_MULTIPLIER)
            sl = price + (atr * ATR_SL_MULTIPLIER)
            if sl <= price:
                print(Fore.RED + "⚠️ SL inválido para operação de venda")
                return None, None, None, None

        min_sl_distance = atr * 1.5

        if signal == 1:
            sl = min(sl, price - min_sl_distance)
        else:
            sl = max(sl, price + min_sl_distance)

        stop_loss_pips = abs(price - sl) / 0.0001  # pip = 0.0001
        volume = self.calculate_volume(stop_loss_pips)
        return price, tp, sl, volume

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

    def send_order(self, signal, size_multiplier=1.0):
        #print(Fore.YELLOW + "Dentro do send_order TradingBot")

        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None or tick.ask == 0 or tick.bid == 0:
            print(Fore.RED + "❌ Tick inválido")
            return
        spread = (tick.ask - tick.bid) * 10000
        if spread > 30:
            print(Fore.RED + f"❌ Spread muito alto: {spread} pontos")
            return
        params = self.get_trade_params(signal)
        if params is None or None in params:
            print(Fore.RED + "❌ Não foi possível calcular os parâmetros do trade. Ordem não enviada.")
            return
        price, tp, sl, volume = params

        stop_loss_pips = abs(price - sl) / 0.0001
        volume = self.calculate_volume(stop_loss_pips) * size_multiplier

        order_type = mt5.ORDER_TYPE_BUY if signal == 1 else mt5.ORDER_TYPE_SELL
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            print(Fore.RED + "❌ Erro ao obter informações do símbolo.")
            return
        stops_level = getattr(symbol_info, "stops_level", None) or STOPS
        min_distance = stops_level * symbol_info.point
        if abs(price - sl) < min_distance:
            sl = price - min_distance if signal == 1 else price + min_distance
        if abs(tp - price) < min_distance:
            tp = price + min_distance if signal == 1 else price - min_distance
        digits = symbol_info.digits
        price, sl, tp = round(price, digits), round(sl, digits), round(tp, digits)
        print(Fore.BLUE + f"DEBUG: price={price}, SL={sl}, TP={tp}, volume={volume}")
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": volume,
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
            print(Fore.LIGHTGREEN_EX + f"✅ Ordem {decode_ml_signal(signal)} executada com sucesso!")
            order_details = (
                f"Ativo: {self.symbol} | Tipo: {decode_ml_signal(signal)} | "
                f"Preço: {price:.5f} | SL: {sl:.5f} | TP: {tp:.5f} | "
                f"Volume: {volume}"
            )
            self.log_order("ENVIO_ORDEM", order_details)
            print(Fore.LIGHTGREEN_EX + f"""🚀 Ordem enviada:
            Ativo: {self.symbol}
            Tipo: {decode_ml_signal(signal)}
            Preço: {price:.5f}
            SL: {sl:.5f}
            TP: {tp:.5f}
            Volume: {volume}
            """)
        else:
            print(Fore.RED + f"❌ Erro ao enviar ordem: {result.comment}")

        return result

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

    def calculate_volume(self, stop_loss_pips, final_signal=None):
        #print(Fore.YELLOW + "Dentro do calculate_volume TradingBot")

        balance = mt5.account_info().balance
        win_rate = tracker.summary().get('win_rate', 0.5)
        risk_multiplier = 1.0 - (0.5 * (1 - win_rate))
        adjusted_risk = risk_per_trade * risk_multiplier
        risk_amount = balance * adjusted_risk
        pip_value = 10  # Valor de pip para o par, ex.: AUDCAD

        if stop_loss_pips == 0:
            return 0.0

        volume = risk_amount / (stop_loss_pips * pip_value)

        # =============================================
        # NOVO: Ajuste de Volume por Tendência
        # =============================================
        trend = self.check_market_trend()
        if "STRONG" in trend['direction']:
            volume *= 2.0  # Aumento de 100% em vez de 30%
        if final_signal is not None:
            if trend['direction'] == "UP" and final_signal == 0:
                volume *= 0.5
            elif trend['direction'] == "DOWN" and final_signal == 1:
                volume *= 0.5

        return round(volume, 2)

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

    # def _load_initial_data(self):
    #     print("Dentro do _load_initial_data TradingBot")
    #     """Carrega dados iniciais para evitar historical_features vazio"""
    #     try:
    #         rates = get_historical_rates(self.symbol, TIMEFRAME, TEMPO_DADOS_COLETADOS)
    #         if rates is not None:
    #             df = pd.DataFrame(rates)
    #             self.historical_features = calculate_features(df)[[
    #                 "close", "feature1", "feature2",  # Garante que 'close' está incluso
    #                 # ... (lista completa de features)
    #             ]].copy()
    #     except Exception as e:
    #         print(f"Erro ao carregar dados iniciais: {e}")
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
        #print(Fore.YELLOW + "Dentro do execute_trail_stop TradingBot")

        """Executa trailing stop em todas as posições abertas"""
        try:
            positions = mt5.positions_get(symbol=self.symbol)
            if not positions:
                return

            atr = self.calculate_atr()
            if atr is None:
                return

            for position in positions:
                current_price = mt5.symbol_info_tick(self.symbol).ask if position.type == 0 else mt5.symbol_info_tick(
                    self.symbol).bid
                current_sl = position.sl
                new_sl = None

                # Calcular novo stop loss baseado no ATR
                if position.type == 0:  # Posição de compra
                    new_sl = current_price - (atr * ATR_SL_MULTIPLIER * 0.8)  # Mais agressivo
                    min_distance = current_price - (atr * 1.2)  # Distância mínima
                    new_sl = max(new_sl, min_distance)
                else:  # Posição de venda
                    new_sl = current_price + (atr * ATR_SL_MULTIPLIER * 0.8)
                    max_distance = current_price + (atr * 1.2)
                    new_sl = min(new_sl, max_distance)

                # Verificar se o novo SL é válido e melhor que o atual
                if self._is_sl_valid(position, new_sl) and self._is_better_sl(position, new_sl):
                    self._modify_position(position, new_sl)

        except Exception as e:
            print(Fore.RED + f"Erro ao executar trailing stop: {str(e)}")

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

    async def run(self):
        #print(Fore.YELLOW + "Dentro do run TradingBot")
        global RL_MODEL

        # Verificar se o modelo RL está carregado se necessário
        if OPERATION_MODE in ["rl", "both"] and RL_MODEL is None:
            print(Fore.RED + "? Modelo RL não carregado! Iniciando treinamento...")
            RL_MODEL = train_rl_agent_mp()

        # Verificar conexão MT5
        if not mt5.initialize():
            print(Fore.RED + "❌ Falha na conexão com o MT5!")
            return

        # Garantir que temos dados iniciais
        while self.historical_features.empty:
            print(Fore.BLUE + "🔄 Aguardando dados históricos...")
            await asyncio.sleep(1)
            self._load_initial_data()

        dynamic_threshold = 0.6

        # Verificação inicial para garantir que historical_features não está vazio
        if self.historical_features.empty:
            self._load_initial_data()

        def calculate_dynamic_threshold():
            #print(Fore.YELLOW + "Dentro do calculate_dynamic_threshold run TradingBot")

            win_rate = tracker.summary().get('win_rate', 0.5)
            if not self.historical_features.empty and 'volatility_30' in self.historical_features.columns:
                volatility = np.mean(self.historical_features['volatility_30'][-100:].values)
            else:
                volatility = 0.0
            return 0.4 + (win_rate * 0.3) - (volatility * 0.2)

        while True:
            # Garantir que o modelo RL está carregado
            if RL_MODEL is not None and self.rl_model is None:
                self.rl_model = RL_MODEL
                print(Fore.LIGHTGREEN_EX + "✅ Modelo RL carregado no TradingBot!")

            self.dashboard.update(tracker.trades)
            self.log_metrics()

            current_regime = detect_market_regime(self, window=50)
            self.adjust_strategy(current_regime)

            if not mt5.initialize():
                print(Fore.RED + "❌ Reconectando ao MT5...")
                await asyncio.sleep(5)
                continue

            rates = mt5.copy_rates_from_pos(self.symbol, TIMEFRAME, 0, TAMANHO_JANELA)
            if rates is not None and len(rates) >= TAMANHO_JANELA:
                df = pd.DataFrame(rates)
                df = calculate_features(df).dropna()
                # Calcular sinal ML apenas se necessário
                ml_signal = self.predict_signal(df.iloc[-1:]) if OPERATION_MODE != "rl" else None
            else:
                ml_signal = None

            state_vector = self.get_live_data()
            if state_vector is None:
                await asyncio.sleep(15)
                continue

            new_threshold = calculate_dynamic_threshold()
            dynamic_threshold = 0.9 * dynamic_threshold + 0.1 * new_threshold

            q_values = [0, 0, 0]
            with RL_MODEL_LOCK:
                if RL_MODEL is not None:
                    try:
                        obs_tensor = th.tensor(state_vector, dtype=th.float32).unsqueeze(0).to(RL_MODEL.device)
                        with th.no_grad():
                            q_values = RL_MODEL.policy.q_net(obs_tensor).cpu().numpy().flatten()
                        rl_action, _ = RL_MODEL.predict(state_vector, deterministic=True)
                        rl_action = int(rl_action)
                        if rl_action not in [0, 1, 2, 3, 4, 5, 6, 7]:
                            print(Fore.RED + f"Ação RL inválida recebida: {rl_action}. Definindo como 0 (wait)")
                            rl_action = 0
                    except Exception as e:
                        print(Fore.RED + f"Erro ao obter Q-values: {str(e)}")
                        rl_action = 0
                else:
                    rl_action = 0

            final_signal = None
            size_multiplier = 1.0
            if OPERATION_MODE == "ml":
                final_signal = ml_signal

            elif OPERATION_MODE == "rl":
                # Mapeamento completo das 8 ações
                if rl_action == 0:
                    final_signal = None  # Esperar
                elif rl_action in (1, 2, 3):
                    final_signal = 1  # Compra (conservadora, moderada, agressiva)
                    size_multiplier = [0.3, 0.7, 1.0][rl_action - 1]
                elif rl_action in (4, 5, 6):
                    final_signal = 0  # Venda (conservadora, moderada, agressiva)
                    size_multiplier = [0.3, 0.7, 1.0][rl_action - 4]
                elif rl_action == 7:
                    final_signal = None  # Trail stop (ação específica)
                    self.execute_trail_stop()  # Função a ser implementada

            if OPERATION_MODE == "both":

                # Lógica mais robusta para integração ML+RL

                if rl_action in [1, 2, 3] and ml_signal == 1:

                    final_signal = 1

                    size_multiplier = [0.3, 0.7, 1.0][rl_action - 1]

                elif rl_action in [4, 5, 6] and ml_signal == 0:

                    final_signal = 0

                    size_multiplier = [0.3, 0.7, 1.0][rl_action - 4]

                else:

                    final_signal = None

            if final_signal is None and q_values is not None:
                if q_values[1] > dynamic_threshold or q_values[2] > dynamic_threshold:
                    final_signal = 1 if q_values[1] > q_values[2] else 0

            trend = self.check_market_trend()
            if "UP" in trend:
                if final_signal == 0:
                    print(Fore.RED + f"🚫 Venda bloqueada (Tendência: {trend})")
                    final_signal = None
                elif final_signal == 1:
                    size_multiplier = min(size_multiplier * 1.5, 2.0)
            elif "DOWN" in trend:
                if final_signal == 1:
                    print(Fore.RED + f"🚫 Compra bloqueada (Tendência: {trend})")
                    final_signal = None
                elif final_signal == 0:
                    size_multiplier = min(size_multiplier * 1.5, 2.0)

            print(Fore.LIGHTWHITE_EX + f"ML: {decode_ml_signal(ml_signal)}"
                  +Fore.LIGHTWHITE_EX + f" | RL: {decode_rl_signal(rl_action)}"
                  +Fore.LIGHTWHITE_EX + " | Q-values: "+Fore.BLUE + f"{np.round(q_values, 2)}"
                  +Fore.LIGHTWHITE_EX + " | Thresh: "+Fore.BLUE + f"{dynamic_threshold:.2f}"
                  +Fore.LIGHTWHITE_EX + " | Final: "+Fore.BLUE + f"{decode_integrated_signal(final_signal)}"
                  +Fore.LIGHTWHITE_EX + " | Size Multiplier: "+Fore.BLUE + f"{size_multiplier}")

            if final_signal is not None:
                await self.manage_positions(final_signal)
                positions = mt5.positions_get(symbol=self.symbol)
                if not positions:
                    self.send_order(final_signal, size_multiplier)

            consecutive_losses = sum(1 for t in tracker.trades[-3:] if t['pnl'] <= 0)
            if consecutive_losses >= 3:
                print(Fore.RED + "⚠️ 3 perdas consecutivas - Reavaliando modelo RL...")
                with RL_MODEL_LOCK:
                    RL_MODEL = train_rl_agent_mp()

            await asyncio.sleep(INTERVALO_PREDICOES)

# def detect_market_regime(window=50):
#     metrics = {
#         'trend': bot.check_market_trend(window),
#         'volatility': df['close'].pct_change().std() * 100,
#         'volume_ratio': df['volume'].iloc[-1] / df['volume'].mean()
#     }
#
#     if metrics['volatility'] > 1.5 and metrics['volume_ratio'] > 2.0:
#         return 'trending'
#     elif metrics['volatility'] < 0.5:
#         return 'ranging'
#     return 'transition'


# class TrendPrioritizedReplayBuffer(ReplayBuffer):
#     def __init__(self, buffer_size, observation_space, action_space, device='auto'):
#         super().__init__(buffer_size, observation_space, action_space, device)
#         self.priorities = np.zeros((buffer_size,), dtype=np.float32)
#         self.max_priority = 1.0
#
#     def add(self, obs, next_obs, action, reward, done, infos):
#         start_idx = self.pos
#         super().add(obs, next_obs, action, reward, done, infos)
#         num_added = len(reward) if isinstance(reward, (list, np.ndarray)) else 1
#
#         for i in range(num_added):
#             idx = (start_idx + i) % self.buffer_size
#             # Correção: Acessar cada info individualmente
#             info = infos[i] if (infos is not None and i < len(infos)) else {}
#             trend_strength = info.get('trend_strength', 0)
#             current_reward = reward[i] if isinstance(reward, (list, np.ndarray)) else reward
#             priority = 2.0 if trend_strength > 30 else 1.5 if current_reward > 0 else 1.0
#             self.priorities[idx] = priority
#             self.max_priority = max(self.max_priority, priority)
#
#     def sample(self, batch_size: int, env=None, beta: float = 0.4):
#         if self.size == 0:
#             raise ValueError("Buffer vazio!")
#
#         # Calcula o tamanho usando self.pos e self.full diretamente
#         size = self.pos if not self.full else self.buffer_size
#         size = int(size)
#
#         probs = self.priorities[:size] / self.priorities[:size].sum()
#         indices = np.random.choice(size, batch_size, p=probs)
#         weights = (size * probs[indices]) ** (-beta)
#         weights /= weights.max()
#
#         return self._get_samples(indices), indices, weights
#
#     def __len__(self):
#         return self.size

# class TrendPrioritizedReplayBuffer(ReplayBuffer):
#     """
#     Buffer de Replay com Priorização baseada em Tendência.
#     Modificado para manter compatibilidade com a interface do Stable Baselines3.
#     """
#
#     # def __init__(self, buffer_size: int, observation_space: spaces.Space, action_space: spaces.Space,
#     #              device: Union[th.device, str] = "auto"):  # Agora Union está definido
#     #     super().__init__(buffer_size, observation_space, action_space, device)
#     #     self.priorities = np.ones((buffer_size,), dtype=np.float32)
#     #     self.max_priority = 1.0
#
#     # def __init__(self, buffer_size: int, observation_space: spaces.Space, action_space: spaces.Space,
#     #              device: Union[str, th.device] = "auto"):
#     #     super().__init__(buffer_size, observation_space, action_space, device)
#     #     self.priorities = np.ones(buffer_size, dtype=np.float32)
#     #     self.max_priority = 1.0
#
#     # def __init__(self, buffer_size: int, observation_space: spaces.Space, action_space: spaces.Space,
#     #              device: Union[str, th.device] = "auto"):
#     #     super().__init__(buffer_size, observation_space, action_space, device)
#     #     self.priorities = np.ones(buffer_size, dtype=np.float32)
#     #     self.max_priority = 1.0
#
#     # def __init__(self, buffer_size: int, observation_space: spaces.Space, action_space: spaces.Space,
#     #              device: Union[str, th.device] = "auto"):
#     #     super().__init__(buffer_size, observation_space, action_space, device)
#     #     self.priorities = np.ones(buffer_size, dtype=np.float32)
#     #     self.max_priority = 1.0
#     #     print(f"Tipo do size: {type(self.size)}, Valor: {self.size}")
#
#     # def __init__(self, buffer_size: int, observation_space: gym.Space, action_space: gym.Space,
#     #              device: Union[str, th.device] = "auto"):
#     #     super().__init__(buffer_size, observation_space, action_space, device)
#     #     self.priorities = np.ones(buffer_size, dtype=np.float32)
#     #     print(f"[DEBUG] Tipo de self.size: {type(self.size)}")
#
#     # def __init__(self, buffer_size: int, observation_space: gym.Space, action_space: gym.Space,
#     #              device: Union[str, th.device] = "auto"):
#     #     super().__init__(buffer_size, observation_space, action_space, device)
#     #     self.priorities = np.ones(buffer_size, dtype=np.float32)
#     #     self.max_priority = 1.0
#
#     # def add(self, obs: np.ndarray, next_obs: np.ndarray, action: np.ndarray, reward: np.ndarray, done: np.ndarray,
#     #         infos: list) -> None:
#     #     """
#     #     Adiciona uma transição ao buffer, calculando prioridades com base na força da tendência.
#     #     """
#     #     super().add(obs, next_obs, action, reward, done, infos)
#     #
#     #     # Atualiza prioridades para as novas entradas
#     #     new_indices = np.arange(self.pos - len(reward), self.pos) % self.buffer_size
#     #     for idx in new_indices:
#     #         info = infos[idx % len(infos)] if infos else {}
#     #         trend_strength = info.get('trend_strength', 0)
#     #         current_reward = reward[idx % len(reward)]
#     #         priority = 2.0 if trend_strength > 30 else 1.5 if current_reward > 0 else 1.0
#     #         self.priorities[idx] = priority
#     #         self.max_priority = max(self.max_priority, priority)
#
#     # def sample(self, batch_size: int, env: Optional[gym.Env] = None, beta: float = 0.4) -> ReplayBufferSamples:
#     #     """
#     #     Amostra do buffer com prioridades, retornando ReplayBufferSamples.
#     #     """
#     #     if self.size == 0:
#     #         raise ValueError("Buffer vazio!")
#     #
#     #     # Converter size para inteiro explicitamente
#     #     current_size = int(self.size)
#     #
#     #     # Cálculo das probabilidades
#     #     probs = self.priorities[:current_size] / self.priorities[:current_size].sum()
#     #     indices = np.random.choice(current_size, batch_size, p=probs)
#     #
#     #     # Obtém as amostras (ReplayBufferSamples)
#     #     samples = super()._get_samples(indices)
#     #
#     #     return samples
#
#     # def sample(self, batch_size: int, env: Optional[gym.Env] = None, beta: float = 0.4) -> ReplayBufferSamples:
#     #     if self.size == 0:  # Acesso CORRETO à propriedade size
#     #         raise ValueError("Buffer vazio!")
#     #
#     #     current_size = int(self.size)  # self.size já é inteiro
#     #     probs = self.priorities[:current_size] / self.priorities[:current_size].sum()
#     #     indices = np.random.choice(current_size, batch_size, p=probs)
#     #     return super()._get_samples(indices)
#
#     # def sample(self, batch_size: int, env: Optional[gym.Env] = None, beta: float = 0.4) -> ReplayBufferSamples:
#     #     # Acesso CORRETO à property size (não use parênteses!)
#     #     if self.size == 0:
#     #         raise ValueError("Buffer vazio!")
#     #
#     #     current_size = self.size  # Já é inteiro, não precisa de conversão
#     #     probs = self.priorities[:current_size] / self.priorities[:current_size].sum()
#     #     indices = np.random.choice(current_size, batch_size, p=probs)
#     #     return super()._get_samples(indices)
#     #
#     # def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
#     #     """
#     #     Atualiza prioridades após o treino (se usando PER clássico).
#     #     """
#     #     self.priorities[indices] = priorities
#     #     self.max_priority = max(self.max_priority, np.max(priorities))
#
#     # def sample(self, batch_size: int, env: Optional[gym.Env] = None, beta: float = 0.4) -> ReplayBufferSamples:
#     #     # Verificação robusta do tamanho
#     #     current_size = int(self.size)  # Converter EXPLICITAMENTE para int
#     #     if current_size == 0:
#     #         raise ValueError("Buffer vazio!")
#     #
#     #     # Cálculo seguro das probabilidades
#     #     valid_priorities = self.priorities[:current_size]
#     #     probs = valid_priorities / valid_priorities.sum()
#     #
#     #     # Seleção de índices com checagem de inteiro
#     #     indices = np.random.choice(current_size, size=batch_size, p=probs)
#     #     return super()._get_samples(indices)
#
#     def __init__(self, buffer_size: int, observation_space: gym.Space, action_space: gym.Space,
#                  device: Union[str, th.device] = "auto"):
#         super().__init__(buffer_size, observation_space, action_space, device)
#         self.priorities = np.ones(buffer_size, dtype=np.float32)
#         self.max_priority = 1.0
#
#     def add(self, obs: np.ndarray, next_obs: np.ndarray, action: np.ndarray, reward: np.ndarray, done: np.ndarray,
#             infos: list):
#         super().add(obs, next_obs, action, reward, done, infos)
#
#         # Atualização correta das prioridades
#         new_indices = np.arange(self.pos - len(reward), self.pos) % self.buffer_size
#         for idx in new_indices:
#             info = infos[idx % len(infos)] if infos else {}
#             trend_strength = info.get('trend_strength', 0)
#             current_reward = reward[idx % len(reward)]
#             priority = 2.0 if trend_strength > 30 else 1.5 if current_reward > 0 else 1.0
#             self.priorities[idx] = priority
#             self.max_priority = max(self.max_priority, priority)  # Atualiza máximo
#
#     # def sample(self, batch_size: int, env=None) -> ReplayBufferSamples:
#     #     # Acesso DIRETO à propriedade size (sem conversão!)
#     #     if self.size == 0:
#     #         raise ValueError("Buffer vazio!")
#     #
#     #     # Cálculo seguro com tamanho inteiro
#     #     current_size = self.size
#     #     probs = self.priorities[:current_size] / self.priorities[:current_size].sum()
#     #     indices = np.random.choice(current_size, batch_size, p=probs)
#     #     return super()._get_samples(indices)
#
#     def sample(self, batch_size: int, env: Optional[gym.Env] = None, beta: float = 0.4) -> ReplayBufferSamples:
#         # Garantir que current_size seja inteiro
#         current_size = int(self.size)  # Conversão EXPLÍCITA para int
#         if current_size == 0:
#             raise ValueError("Buffer vazio!")
#
#         # Verificação adicional de tipos (para debug)
#         print(f"[DEBUG] Tipo de current_size: {type(current_size)}, Valor: {current_size}")
#
#         # Cálculo seguro das probabilidades
#         valid_priorities = self.priorities[:current_size]
#         probs = valid_priorities / valid_priorities.sum()
#
#         # Seleção de índices com checagem explícita
#         indices = np.random.choice(current_size, size=batch_size, p=probs)
#         return super()._get_samples(indices)
# def detect_market_regime(bot: TradingBot, window=50):
#     print("Dentro do detect_market_regime")
#     metrics = {
#         'trend': bot.check_market_trend(window),
#         'volatility': bot.historical_features['close'].pct_change().std() * 100,
#         'volume_ratio': bot.historical_features['volume'].iloc[-1] / bot.historical_features['volume'].mean()
#     }
#
#     if metrics['volatility'] > 1.5 and metrics['volume_ratio'] > 2.0:
#         return 'trending'
#     elif metrics['volatility'] < 0.5:
#         return 'ranging'
#     return 'transition'
def detect_market_regime(bot: TradingBot, window: int = 50) -> str:
    """
    Detecta o regime de mercado com base nas métricas calculadas a partir dos dados históricos.

    Parâmetros:
        bot (TradingBot): Instância que contém os dados históricos e métodos de análise.
        window (int): Período de janela para análise da tendência do mercado.

    Retorna:
        str: Regime de mercado ('trending', 'ranging', 'transition' ou 'neutral' em caso de dados insuficientes ou erro).
    """
    #print(Fore.YELLOW + "Dentro do detect_market_regime")

    try:
        # Verifica se há dados históricos disponíveis
        if bot.historical_features.empty:
            print(Fore.RED + "Dados históricos vazios, retornando 'neutral'")
            return 'neutral'

        # Calcula as métricas necessárias
        trend = bot.check_market_trend(window)
        volatility = bot.historical_features['close'].pct_change().std() * 100
        volume_ratio = bot.historical_features['volume'].iloc[-1] / bot.historical_features['volume'].mean()

        # Exibe as métricas para depuração
        print(Fore.BLUE + f"Métricas calculadas: trend={trend}, volatility={volatility:.2f}, volume_ratio={volume_ratio:.2f}")

        # Define o regime de mercado com base nas métricas
        if volatility > 1.5 and volume_ratio > 2.0:
            regime = 'trending'
        elif volatility < 0.5:
            regime = 'ranging'
        else:
            regime = 'transition'

        print(Fore.BLUE + f"Regime detectado: {regime}")
        return regime
    except Exception as e:
        print(Fore.RED + f"Erro na detecção de regime: {e}")
        return 'neutral'

# =============================
# Classe TradingEnv com Monitor (Gymnasium)
# =============================
class TradingEnv(gym.Env):
    #print(Fore.YELLOW + "Dentro do TradingEnv")

    metadata = {'render.modes': ['human']}

    def __init__(self, trading_bot, initial_balance=SALDO_INICIAL_TREINO):
        #print(Fore.YELLOW + "Dentro do __init__ TradingEnv")

        super(TradingEnv, self).__init__()
        self.bot = trading_bot
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.current_step = 0
        self.max_steps = 100
        self.episode_balances = []
        self.action_history = []  # Novo histórico de ações
        self.data_index = 0
        # Novo: Rastreamento de posições simuladas
        self.open_positions = []  # Lista de dicionários com detalhes das posições
        self.current_sl = None  # Stop loss atual
        self.current_tp = None  # Take profit atual

        # Mapeamento de ações atualizado para 8 ações
        self.ACTIONS_MAP = {
            0: ('wait', 0),
            1: ('buy', 0.3),  # Compra conservadora
            2: ('buy', 0.7),  # Compra moderada
            3: ('buy', 1.0),  # Compra agressiva
            4: ('sell', 0.3),  # Venda conservadora
            5: ('sell', 0.7),  # Venda moderada
            6: ('sell', 1.0),  # Venda agressiva
            7: ('trail_stop', 0.5)  # Stop móvel
        }

        # Configurações de stops
        self.stops_level = STOPS  # Ex: 54 pontos
        self.point = 0.00001      # Para AUDCAD (ajuste conforme o símbolo)
        self.min_distance = self.stops_level * self.point

        # Carregar dados históricos
        historical = get_historical_rates(symbol, TIMEFRAME, TEMPO_DADOS_COLETADOS)
        if historical is None or len(historical) == 0:
            raise ValueError("Dados históricos não disponíveis.")

        df = calculate_features(pd.DataFrame(historical)).dropna()
        df.reset_index(drop=True, inplace=True)

        # Configuração de features
        self.features = [
            "close",
            "feature1", "feature2",
            "sma_5", "sma_10", "sma_20", "sma_50", "sma_200",
            "rsi", "macd_hist", "bollinger_upper", "bollinger_lower",
            "adx", "stochastic_k",
            "momentum_5", "momentum_15", "volatility_30",
            "ema_21", "ema_50", "roc_5", "roc_10",
            "atr", "volume_ma",
            "price_distance_to_upper_band", "price_distance_to_lower_band"  # 25 features
        ]

        self.price_col = "close"
        self.historical_data = self._normalize_data(df)

        # Novo cálculo de dimensão do observation_space:
        # n_features = 25 (número de features por candle)
        # window_size = TAMANHO_JANELA (número de candles históricos)
        # +1 para incluir o sinal de ML
        n_features = 25  # Número de features por candle
        window_size = TAMANHO_JANELA
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features * window_size,),
            dtype=np.float32
        )

        # Atualização do espaço de ações para 8 ações
        self.action_space = spaces.Discrete(8)

        # Configuração do logger
        self.logger = logging.getLogger('TradingEnv')
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def _simulate_trailing_stop(self):
        #print(Fore.YELLOW + "Dentro do _simulate_trailing_stop TradingEnv")

        current_price = self._get_current_price()

        for pos in self.open_positions:
            atr = self._calculate_simulated_atr()  # ATR baseado em dados históricos

            # Lógica adaptada para dados históricos
            if pos['type'] == 'buy':
                new_sl = current_price - (atr * ATR_SL_MULTIPLIER * 0.8)
                new_sl = max(new_sl, pos['sl'] + (atr * 0.2))  # Garante movimento uni-direcional
            else:
                new_sl = current_price + (atr * ATR_SL_MULTIPLIER * 0.8)
                new_sl = min(new_sl, pos['sl'] - (atr * 0.2))

            # Atualiza SL apenas se melhora a posição
            if (pos['type'] == 'buy' and new_sl > pos['sl']) or \
                    (pos['type'] == 'sell' and new_sl < pos['sl']):
                pos['sl'] = new_sl
                self.current_sl = new_sl  # Atualiza estado

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

        # Verificação robusta de dados históricos
        if len(self.historical_data) < window_size:
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
        #print(Fore.YELLOW + "Dentro do step TradingEnv")

        # Converter ação para inteiro se for um array do NumPy
        if isinstance(action, np.ndarray):
            action = action.item()

        # Modificação para simular trailing
        if action == 7 and self.open_positions:
            self._simulate_trailing_stop()

        # Obter preços e estado
        current_close, next_close = self._get_prices(action, self.data_index)
        action_type, size_multiplier = self.ACTIONS_MAP[action]

        # Calcular parâmetros do trade
        atr = self.bot.calculate_atr()
        stop_loss_pips = (atr * ATR_SL_MULTIPLIER) / 0.0001 if atr else 0
        volume = self.bot.calculate_volume(stop_loss_pips) * size_multiplier if atr else 0

        # Calcular recompensa base
        reward = self._calculate_reward(
            action_type,
            size_multiplier,
            current_close,
            next_close,
            volume
        )

        # Atualizar estado
        self._update_state(action, current_close, next_close, volume, reward)

        # Verificar término do episódio
        done = (self.current_step >= self.max_steps) or (self.data_index >= len(self.historical_data) - 1)

        # Obter o próximo estado com sinal ML
        next_state = self._get_next_state()

        # Incluindo informações adicionais, como trend_strength e balance
        info = {
            "trend_strength": self.bot.check_market_trend(window=50)['strength'],
            "balance": self.balance
        }

        # Registrar o saldo atual a cada passo
        self.episode_balances.append(self.balance)

        return next_state, reward, done, False, info

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

    def _calculate_reward(self, action_type, size_multiplier, current_price, next_price, volume):
        #print(Fore.YELLOW + "Dentro do _calculate_reward TradingEnv")

        if action_type == 'wait':
            return self._calculate_inactivity_penalty(current_price)

        atr = self.bot.calculate_atr()
        if atr is None:
            return 0

        if action_type == 'buy':
            initial_sl = current_price - (atr * ATR_SL_MULTIPLIER)
            initial_tp = current_price + (atr * ATR_TP_MULTIPLIER)
        else:
            initial_sl = current_price + (atr * ATR_SL_MULTIPLIER)
            initial_tp = current_price - (atr * ATR_TP_MULTIPLIER)

        if action_type == 'buy':
            sl = max(initial_sl, current_price - self.min_distance)
            tp = max(initial_tp, current_price + self.min_distance)
        else:
            sl = min(initial_sl, current_price + self.min_distance)
            tp = min(initial_tp, current_price - self.min_distance)

        stop_loss_pips = abs(current_price - sl) / 0.0001
        volume = self.bot.calculate_volume(stop_loss_pips) * size_multiplier

        if action_type == 'buy':
            pnl = (next_price - current_price) * volume * 10000
        else:
            pnl = (current_price - next_price) * volume * 10000

        balance_impact = pnl / self.balance
        base_reward = balance_impact * 100

        adx = self.historical_data['adx'].iloc[-1] if 'adx' in self.historical_data else 0
        trend_multiplier = 1 + (adx / 100)

        roc_5 = self.historical_data['roc_5'].iloc[-1] if 'roc_5' in self.historical_data else 0
        momentum_bonus = abs(roc_5) * 100

        reward = (base_reward * trend_multiplier) + momentum_bonus
        reward -= self._calculate_trading_costs(volume, current_price)

        trend = self.bot.check_market_trend()
        if (trend == "UP_STRONG" and action_type == "sell") or \
           (trend == "DOWN_STRONG" and action_type == "buy"):
            reward -= 50
        elif (trend == "UP" and action_type == "sell") or \
             (trend == "DOWN" and action_type == "buy"):
            reward -= 20

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

    def _get_next_state(self):
        #print(Fore.YELLOW + "Dentro do _get_next_state TradingEnv")

        window_size = TAMANHO_JANELA
        if self.data_index < len(self.historical_data):
            start_idx = max(0, self.data_index - window_size + 1)
            end_idx = self.data_index + 1
            df_window = self.historical_data.iloc[start_idx:end_idx]
            if len(df_window) < window_size:
                padding = np.zeros((window_size - len(df_window), len(self.features)))
                padded_values = np.vstack([padding, df_window[self.features].values])
                state = padded_values.flatten().astype(np.float32)
            else:
                state = df_window[self.features].values.flatten().astype(np.float32)
        else:
            state = self.bot.get_live_data()
            if state is None or len(state) < (25 * window_size):
                state = np.zeros(25 * window_size, dtype=np.float32)
        ml_signal = self.bot.predict_signal(
            pd.DataFrame([state[-25:]], columns=self.features))
        return state

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

    def update_live_data(self):
        #print(Fore.YELLOW + "Dentro do update_live_data TradingEnv")

        """Atualiza os dados do ambiente com informações em tempo real"""
        try:
            rates = mt5.copy_rates_from_pos(self.symbol, TIMEFRAME, 0, TEMPO_DADOS_COLETADOS)
            df = pd.DataFrame(rates)
            df = calculate_features(df).dropna()
            self.historical_data = self._normalize_data(df)
            print(Fore.LIGHTGREEN_EX + "🔄 Dados do ambiente RL atualizados com sucesso!")
        except Exception as e:
            print(Fore.RED + f"Erro ao atualizar dados: {str(e)}")

# =============================
# Classe para ajuste do schedule de exploração
# =============================
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

# =============================
# Função de monitoramento da assertividade com PerformanceTracker
# =============================
async def monitor_accuracy_tp_sl(trading_bot):
    #print(Fore.YELLOW + "Dentro do monitor_accuracy_tp_sl")

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

# =============================
# Função de treinamento do agente RL usando execução em thread dedicada
# =============================
def train_rl_agent_mp():
    #print(Fore.YELLOW + "Dentro do train_rl_agent_mp")

    global RL_MODEL
    bot = TradingBot(symbol)

    balance_callback = BalanceCallback()

    # 1. Configurar política e callbacks com os novos parâmetros
    policy_kwargs = dict(
        features_extractor_class=TemporalAttentionExtractor,  # Definir apenas aqui
        features_extractor_kwargs={},
        net_arch=[256, 128]
    )

    # Parâmetros de exploração revisados
    exploration_params = {
        'exploration_fraction': 0.3,      # Período mais longo de exploração
        'exploration_initial_eps': 1.0,     # Exploração máxima inicial
        'exploration_final_eps': 0.05,      # Exploração mínima final
        'learning_starts': 10000          # Atraso no início do aprendizado
    }

    # 2. Criar ambiente com dados atualizados
    def create_env():
        #print(Fore.YELLOW + "Dentro do create_env train_rl_agent_mp")

        new_data = get_historical_rates(symbol, TIMEFRAME, TEMPO_DADOS_COLETADOS)
        df = calculate_features(pd.DataFrame(new_data)).dropna()
        base_env = TradingEnv(trading_bot=bot, initial_balance=saldo_inicial)
        base_env.update_data(df)
        return Monitor(base_env)

    env = create_env()

    # 3. Configurar callbacks
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

    # 4. Ambiente de validação
    live_val_env = TradingEnv(trading_bot=bot, initial_balance=saldo_inicial)
    live_val_callback = LiveValidationCallback(live_val_env, freq=10000)

    # 5. Inicialização do modelo com os novos parâmetros
    model_path = "dqn_trading_agent_best.zip"
    if os.path.exists(model_path):
        try:
            model = MetaDQN.load(
                model_path,
                env=env,
                device='auto',
                tensorboard_log="./dqn_tensorboard/"
            )
            print(Fore.LIGHTGREEN_EX + "✅ Modelo carregado com sucesso!")
        except Exception as e:
            print(Fore.RED + f"❌ Erro ao carregar modelo: {str(e)}")
            print("Criando novo modelo...")
            model = None
    else:
        model = None

    if model is None:
        print(Fore.LIGHTWHITE_EX + "⚠️ Criando novo modelo...")
        model = MetaDQN(
            CustomDQNPolicy,  # Use a política personalizada aqui
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

    # Adicionar buffer de replay baseado em tendência
    model.replay_buffer = TrendPrioritizedReplayBuffer(
        buffer_size=model.buffer_size,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=model.device
    )

    # 6. Loop de treinamento
    max_cycles = MAX_CICLES #MAXIMO CICLOS PARA ENCERRAR
    required_consecutive_success = QUANTIDADE_ACERTOS_TREINO
    success_count = 0

    for cycle in range(max_cycles):
        print(Fore.LIGHTWHITE_EX + "\n=== Ciclo de Treinamento"+Fore.LIGHTBLUE_EX+f" {cycle + 1}/{max_cycles}"+Fore.LIGHTWHITE_EX+" ===")

        # 7. Treinar com dados atualizados
        model.learn(
            total_timesteps=N_INTERACOES, #número de interações (passos) que o agente de reinforcement learning (RL) realizará com o ambiente durante o treinamento
            callback=[eval_callback, live_val_callback, balance_callback],
            reset_num_timesteps=False,
            tb_log_name="dqn_log",
            progress_bar=True
        )

        # 8. Atualizar dados e validar
        new_data = get_historical_rates(symbol, TIMEFRAME, TEMPO_DADOS_COLETADOS)
        df = calculate_features(pd.DataFrame(new_data)).dropna()

        # Atualizar ambiente de treino
        env.env.update_data(df[-2000:])  # Acesso direto ao ambiente base

        # Atualizar ambiente de validação
        live_val_env.update_data(df[-2000:])

        # 9. Verificação rigorosa de desempenho
        current_balance = live_val_env.get_current_balance()
        print(Fore.YELLOW + f"\n💰 Saldo Atual: ${current_balance:.2f} (Inicial: ${saldo_inicial:.2f})")

        # 10. Condição de parada estrita
        if current_balance > saldo_inicial:
            success_count += 1
            print(Fore.LIGHTYELLOW_EX + f"✅ Sucesso consecutivo {success_count}/{required_consecutive_success}")
            if success_count >= required_consecutive_success:
                print(Fore.YELLOW + "🎯 Condição de lucro consistente atingida!")
                break
        else:
            success_count = 0
            print(Fore.BLUE + "⚠️ Reiniciando contagem de sucessos")

    # Após o treinamento inicial, executar os testes de estresse
    stress_test(model)

    # 11. Salvamento final e relatório
    save_model(model, prefix="dqn_trading_agent")
    RL_MODEL = model

    final_balance = live_val_env.get_current_balance()
    print(Fore.BLUE + f"\n=== TREINAMENTO CONCLUÍDO ===")
    print(Fore.GREEN + f"Saldo Final: ${final_balance:.2f}")
    print(Fore.LIGHTGREEN_EX + f"Retorno: {((final_balance / saldo_inicial) - 1) * 100:.2f}%")

    return model

# =============================
# Atualização assíncrona do modelo RL em thread dedicada
# =============================
async def update_rl_model_periodically():
    #print(Fore.YELLOW + "Dentro do update_rl_model_periodically")

    global RL_MODEL
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=1) as executor:
        while True:
            await asyncio.sleep(TIME_TREINO)
            print(Fore.BLUE + "🔄 Atualizando o agente RL...")
            new_model = await loop.run_in_executor(executor, train_rl_agent_mp)
            with RL_MODEL_LOCK:
                RL_MODEL = new_model
            print(Fore.LIGHTGREEN_EX + "✅ Agente RL atualizado com sucesso!")

# =============================
# Função principal
# =============================
async def main():
    global RL_MODEL, env
    #print(Fore.YELLOW + "Dentro do main")

    # Estabelecer conexão MT5 primeiro
    print(Fore.BLUE + "Conectando ao MT5...")
    if not mt5.initialize():
        print(Fore.RED + "❌ Falha na conexão inicial ao MT5!")
        return
    print(Fore.LIGHTGREEN_EX + "✅ Conexão MT5 estabelecida com sucesso!")

    # Enviar saldo inicial
    await send_initial_balance()

    # Criar instância do bot
    bot = TradingBot(symbol)
    env = TradingEnv(trading_bot=bot, initial_balance=saldo_inicial)  # <--- DEFINA O AMBIENTE

    # Escolher modo de inicialização
    if choose_training_mode():
        print(Fore.LIGHTGREEN_EX + "\n=== INICIANDO TREINAMENTO RL ===")
        RL_MODEL = train_rl_agent_mp()
    else:
        print(Fore.LIGHTGREEN_EX + "\n=== CARREGANDO MODELO PRÉ-TREINADO ===")

        model_path = "dqn_trading_agent_best.zip"
        if os.path.exists(model_path):
            RL_MODEL = MetaDQN.load(
                "dqn_trading_agent_best.zip",
                env=env,
                custom_objects={"policy": CustomDQNPolicy},  # Especificar política personalizada
                device='auto'
            )
        else:
            RL_MODEL = train_rl_agent_mp()
        # if not os.path.exists(model_path):
        #     print(Fore.RED + "❌ Modelo pré-treinado não encontrado!")
        #     choice = input(Fore.LIGHTWHITE_EX + "Deseja iniciar o treinamento agora? (s/n): ").strip().lower()
        #     if choice == 's':
        #         RL_MODEL = train_rl_agent_mp()
        #     else:
        #         print(Fore.RED + "Encerrando o programa...")
        #         mt5.shutdown()
        #         return
        # else:
        #     try:
        #         env = TradingEnv(trading_bot=bot, initial_balance=saldo_inicial)
        #         RL_MODEL = MetaDQN.load(
        #             "dqn_trading_agent_best",
        #             env=env,
        #             custom_policy=CustomDQNPolicy,
        #             policy_kwargs=policy_kwargs,
        #             device='auto',
        #             tensorboard_log="./dqn_tensorboard/"
        #         )
        #         print(Fore.LIGHTGREEN_EX + "✅ Modelo RL carregado com sucesso!")
        #     except Exception as e:
        #         print(Fore.RED + f"❌ Erro ao carregar modelo: {str(e)}")
        #         print(Fore.LIGHTWHITE_EX + "⚠️ Iniciando treinamento emergencial...")
        #         RL_MODEL = train_rl_agent_mp()

    # Criar tarefas
    tasks = [
        bot.run(),
        check_connection(),
        send_telegram_message()
    ]

    # Adicionar tarefas condicionalmente com base no modo
    if OPERATION_MODE in ["ml", "both"]:
        tasks.append(bot.update_model_periodically())

    if OPERATION_MODE in ["rl", "both"]:
        tasks.append(update_rl_model_periodically())
        tasks.append(monitor_accuracy_tp_sl(bot))

    await asyncio.gather(*tasks)
# async def main():
#     print(Fore.YELLOW + "Dentro do main")
#     bot = TradingBot(symbol)  # Instância do bot deve ser criada antes do treino
#
#     # Treinar o modelo RL primeiro
#     global RL_MODEL
#     RL_MODEL = train_rl_agent_mp()
#
#     while True:
#         if not mt5.initialize():
#             print(Fore.RED + "❌ Reconectando ao MT5...")
#             mt5.shutdown()
#             await asyncio.sleep(5)
#             continue
#
#         # Executar o bot apenas se conectado
#         bot_task = asyncio.create_task(bot.run())
#         await asyncio.gather(bot_task)
#
#         await asyncio.sleep(1)
#
#     while True:
#         print(Fore.BLUE + f"Tarefas ativas: {len(asyncio.all_tasks())}")
#         await asyncio.sleep(3)
#         break
#
#     loop = asyncio.get_running_loop()
#     with ThreadPoolExecutor(max_workers=1) as executor:
#         # Iniciar notificação via Telegram
#         telegram_task = asyncio.create_task(send_telegram_message())
#
#         # Treinar o modelo híbrido em uma thread separada
#         # model = await loop.run_in_executor(executor, train_hybrid)
#         #model =  train_rl_agent_mp()  # ← Adicionar await
#         model = await train_hybrid()  # ← Adicionar await
#         #global RL_MODEL
#         RL_MODEL = model
#
#         # Iniciar instância do bot e tarefas
#         bot = TradingBot(symbol)
#         bot_task = asyncio.create_task(bot.run())
#         ml_update_task = asyncio.create_task(bot.update_model_periodically())
#         rl_update_task = asyncio.create_task(update_rl_model_periodically())
#         monitor_task = asyncio.create_task(monitor_accuracy_tp_sl(bot))
#         connection_task = asyncio.create_task(check_connection())
#
#         # Adicionar task de atualização de dados
#         async def live_data_updater():
#             print(Fore.YELLOW + "Dentro do live_data_updater main")
#             while True:
#                 env.update_live_data()
#                 live_val_env.update_live_data()
#                 await asyncio.sleep(300)  # Atualizar a cada 5 minutos
#
#         update_task = asyncio.create_task(live_data_updater())
#
#         await asyncio.gather(
#             bot_task,
#             ml_update_task,
#             rl_update_task,
#             monitor_task,
#             connection_task,
#             telegram_task,
#             update_task  # Nova task adicionada
#         )

if __name__ == "__main__":
    asyncio.run(main())
