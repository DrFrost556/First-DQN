# /NewDQN.py

from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from collections import deque
import random
import os
import logging
import json


# ---- Helper Functions ----

def calculate_macd(close, short_window=12, long_window=26, signal_window=9):
    close_series = pd.Series(close)
    ema_short = close_series.ewm(span=short_window, adjust=False).mean()
    ema_long = close_series.ewm(span=long_window, adjust=False).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    return (macd_line - signal_line).values


def calculate_rsi(close, window=14):
    delta = np.diff(close)
    gain, loss = np.maximum(0, delta), -np.minimum(0, delta)
    avg_gain, avg_loss = np.mean(gain[:window]), np.mean(loss[:window])
    rsi_values = np.zeros_like(close)

    if avg_loss != 0:
        rs = avg_gain / avg_loss
        rsi_values[window] = 100 - (100 / (1 + rs))

    for i in range(window + 1, len(close)):
        avg_gain = ((window - 1) * avg_gain + gain[i - 1]) / window
        avg_loss = ((window - 1) * avg_loss + loss[i - 1]) / window
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi_values[i] = 100 - (100 / (1 + rs)) if avg_loss != 0 else 100

    return rsi_values


class TransformerStockModel(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, num_encoder_layers, num_stocks, num_actions=2, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = self._generate_positional_encoding(d_model, 120)
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc = nn.Linear(d_model, num_actions)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x = self.input_projection(x)
        x += self.positional_encoding[:x.size(1), :].unsqueeze(0).to(x.device)
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        return self.fc(x[:, -1, :])

    def _generate_positional_encoding(self, d_model, max_len=120):
        positional_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        return positional_encoding


# ---- Data Preprocessing ----

def load_stock_data(folder, max_files):
    stock_data = []
    stock_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')]
    stock_files = stock_files[:max_files] if max_files else stock_files

    for file in stock_files:
        df = pd.read_csv(file, index_col=0).dropna()
        if len(df) < 120:
            continue
        df['returns'] = df['adjclose'].pct_change()
        df['rsi'] = calculate_rsi(df['adjclose'])
        df['macd'] = calculate_macd(df['adjclose'])
        df['scaled_volume'] = df['volume'] / df['volume'].rolling(window=30).mean()
        df.dropna(inplace=True)
        stock_data.append(df)

    return stock_data


# ---- Environment Class ----

class StockMarketEnv:
    def __init__(self, stock_data, lookback_window=120, reload_interval=120):
        self.stock_data = stock_data
        self.lookback_window = lookback_window
        self.reload_interval = reload_interval
        self.current_step = 0
        self.stocks = random.sample(stock_data, 100)
        self.data = self.filter_data(self.stocks)

    def reset(self):
        self.current_step = 0
        return self.get_state()

    def filter_data(self, stock_data):
        start_date = pd.to_datetime('2023-01-01')
        filtered_data, indexes = [], []
        for i, df in enumerate(stock_data):
            self.current_index = \
            pd.to_datetime(df.index).get_indexer([start_date.strftime('%Y-%m-%d')], method='nearest')[
                0] + self.current_step
            if len(df) >= (self.current_index + 120):
                filtered_data.append(df)
                indexes.append(i)
        return filtered_data

    def get_state(self):
        return np.stack([df[['returns', 'scaled_volume', 'rsi', 'macd']].iloc[
                         self.current_step:self.current_step + self.lookback_window].values for df in self.data],
                        axis=1)

    def step(self, action):
        self.current_step += 1
        stock_returns = np.array([df['returns'].iloc[self.current_step] for df in self.data])
        selected_returns = stock_returns[action]
        market_returns = np.delete(stock_returns, action)
        reward = self.calculate_relative_return(selected_returns, market_returns)
        done = self.current_step >= len(self.data[0]) - self.lookback_window
        return self.get_state(), reward, done

    def calculate_relative_return(self, selected_returns, market_returns, risk_free_rate=0):
        mean_selected = np.mean(selected_returns)
        mean_market = np.mean(market_returns)
        volatility = np.std(selected_returns)
        sharpe_ratio = (mean_selected - risk_free_rate) / volatility if volatility > 0 else 0
        return (mean_selected - mean_market) + sharpe_ratio


class DQNAgent:
    def __init__(self, model, target_model, device, lr=1e-4, gamma=0.95, epsilon=1, epsilon_min=0.1,
                 epsilon_decay=0.9995, memory_size=100000, batch_size=64, target_update_freq=10,
                 checkpoint_interval=200):
        self.model, self.target_model, self.device = model, target_model, device
        self.optimizer, self.loss_fn = optim.Adam(model.parameters(), lr=lr), nn.MSELoss()
        self.gamma, self.epsilon, self.epsilon_min, self.epsilon_decay = gamma, epsilon, epsilon_min, epsilon_decay
        self.memory, self.batch_size = deque(maxlen=memory_size), batch_size
        self.target_update_freq, self.checkpoint_interval, self.steps = target_update_freq, checkpoint_interval, 0

    def select_action(self, state, top_n=5):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(state.shape[1], top_n, replace=False)
        q_values = self.model(torch.tensor(state, dtype=torch.float32).to(self.device))
        return q_values[:, 1].topk(top_n, dim=0)[1].squeeze().cpu().numpy()

    def remember(self, state, action, reward, next_state, done):
        # Store the experience in memory
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            state, next_state, reward, done = map(lambda x: torch.tensor(x, dtype=torch.float32).to(self.device),
                                                  (state, next_state, reward, done))
            q_value = self.model(state)[action, 1]
            next_q_value = self.target_model(next_state)[action].max(dim=1)[0]
            target_q_value = reward + (1 - done) * self.gamma * next_q_value
            loss = self.loss_fn(q_value, target_q_value)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if self.steps % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        self.steps += 1

    def save_checkpoint(self, model_path="model_checkpoint.pt", variables_path="variables.json"):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, model_path)
        with open(variables_path, 'w') as f:
            json.dump({'epsilon': self.epsilon}, f)

    def load_checkpoint(self, model_path="model_checkpoint.pt"):
        """
        Load model, target model, optimizer state, and important variables from checkpoint files.
        """
        if os.path.isfile(model_path):
            # Load the model and optimizer states from the .pt file
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Loaded model, target model, and optimizer states from {model_path}")
        else:
            print(f"No checkpoint found at {model_path}")


def train_dqn(env, agent, num_episodes=1000, top_n_stocks=5):
    for episode in range(num_episodes):
        state, done, total_reward = env.reset(), False, 0
        while not done:
            action = agent.select_action(state, top_n=top_n_stocks)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            total_reward += reward
        print(f"Episode {episode} - Total Reward: {total_reward}")


# ---- Usage ----

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

stock_data = load_stock_data(r'C:\Users\Oscar\PycharmProjects\pythonProject\stock_data', max_files=200)
env = StockMarketEnv(stock_data)

input_dim, d_model, n_heads, num_encoder_layers = 4, 256, 4, 3
dqn_model = TransformerStockModel(input_dim=input_dim, d_model=d_model, n_heads=n_heads,
                                  num_encoder_layers=num_encoder_layers, num_stocks=len(env.get_state())).to(device)
target_model = TransformerStockModel(input_dim=input_dim, d_model=d_model, n_heads=n_heads,
                                     num_encoder_layers=num_encoder_layers, num_stocks=len(env.get_state())).to(device)
target_model.load_state_dict(dqn_model.state_dict())

agent = DQNAgent(dqn_model, target_model, device)
agent.load_checkpoint()
train_dqn(env, agent)
