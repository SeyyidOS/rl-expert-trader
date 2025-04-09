import random
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback


class PriceActionEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, dfs: [pd.DataFrame], window_size=24, initial_balance=10000.0, fee=0.001, leverage=5,
                 render_mode=None):
        self.dfs = dfs
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.fee = fee
        self.leverage = leverage
        self.render_mode = render_mode

        self.balance = self.initial_balance
        self.total_pnl = 0.0
        self.trade_log = []

        # [direction, position_size (10% increments), stop_loss_level (0.5% increments)]
        self.action_space = spaces.MultiDiscrete([3, 10, 5])
        self.observation_space = spaces.Box(low=0, high=1, shape=(window_size + 4,), dtype=np.float32)
        self._reset_session()

    def _reset_session(self):
        self.df = random.choice(self.dfs).reset_index(drop=True)
        max_start = len(self.df) - self.window_size - 500  # leave room
        self.current_step = random.randint(self.window_size, max_start)
        # self.balance = self.initial_balance
        # self.total_pnl = 0.0
        self.position = None
        self.is_in_position = False
        # self.trade_log = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_session()
        return self._get_observation(), {}

    def _get_observation(self):
        prices = self.df['close'].iloc[self.current_step - self.window_size:self.current_step].values
        norm_prices = (prices - prices.min()) / (prices.max() - prices.min() + 1e-8)

        obs = np.concatenate([
            norm_prices,
            np.array([
                self.balance / self.initial_balance,
                float(self.position['direction']) if self.position else 0.0,
                self.position['entry_price'] / prices[-1] if self.position else 0.0,
                self.position['capital'] / self.balance if self.position else 0.0
            ])
        ])
        return obs.astype(np.float32)

    def _open_position(self, direction, price, size_percent, sl_pct):
        self.is_in_position = True

        capital = self.balance * size_percent
        leverage_capital = (capital * self.leverage)
        self.position = {
            "capital": capital,
            'direction': direction,
            'entry_price': price,
            'leverage': self.leverage,
            'stop_loss_pct': sl_pct,
            'quantity': leverage_capital / price
        }
        fee_cost = leverage_capital * self.fee
        self.balance -= fee_cost
        self.total_pnl -= fee_cost

        self.trade_log.append({'step': self.current_step, 'type': 'OPEN', 'dir': direction,
                               'price': price, "capital": capital, "leverage": self.leverage})

    def _close_position(self, price):
        self.is_in_position = False

        p = self.position
        pnl = (price - p['entry_price']) * p['quantity'] * p['direction']
        capital = abs(p['quantity']) * p['entry_price'] / self.leverage
        fee_cost = abs(p['quantity']) * price * self.fee
        self.balance += pnl - fee_cost
        self.total_pnl += pnl - fee_cost
        self.trade_log.append({'step': self.current_step, 'type': 'CLOSE', 'exit_price': price, 'pnl': pnl})
        self.position = None
        return pnl

    def _check_stop_loss(self, price):
        if not self.position:
            return False
        p = self.position
        delta = (price - p['entry_price']) / p['entry_price']
        if p['direction'] == -1:
            delta *= -1
        return delta <= -p['stop_loss_pct']

    def _is_liquidated(self):
        return self.balance <= 10

    def _is_terminal(self):
        def _is_end():
            return self.current_step >= len(self.df) - 1

        if self._is_liquidated() or _is_end() or not self.is_in_position:
            return True
        return False

    def _terminal_reward(self):
        if self._is_liquidated():
            return -self.balance
        return self.total_pnl

    def step(self, action):
        direction_code, size_code, sl_code = action
        direction_map = {0: 0, 1: 1, 2: -1}
        direction = direction_map[direction_code]
        size_percent = (size_code + 1) / 10
        sl_pct = 0.005 * (sl_code + 1)

        price = self.df.loc[self.current_step, 'close']
        reward = 0.0

        if self.position and self._check_stop_loss(price):
            reward = self._close_position(price)

        elif direction != 0 and not self.position:
            self._open_position(direction, price, size_percent, sl_pct)

        elif direction == 0 and self.position:
            reward = self._close_position(price)

        self.current_step += 1

        done = self._is_terminal()
        if done:
            reward += self._terminal_reward()

        truncated = False
        obs = self._get_observation()

        info = {
            "balance": self.balance,
            "total_pnl": self.total_pnl,
            "position": self.position
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, done, truncated, info

    def render(self):
        if self.is_in_position:
            price = self.df.loc[self.current_step, 'close']
            print("=" * 60)
            print(f"Step        : {self.current_step}")
            print(f"Price       : ${price:.2f}")
            print(f"Balance     : ${self.balance:.2f}")
            if self.position:
                d = self.position
                side = "LONG" if d['direction'] == 1 else "SHORT"
                print(f"Position    : {side} {d['capital']:.4f} @ ${d['entry_price']:.2f}")
                print(f"Stop Loss   : {d['stop_loss_pct'] * 100:.2f}%")
            else:
                print("Position    : None")
            print(f"Total PnL   : ${self.total_pnl:.2f}")
            print("=" * 60)

    def save_trade_log(self, path="trade_log.csv"):
        if not self.trade_log:
            print("⚠️ No trades to save.")
            return
        df = pd.DataFrame(self.trade_log)
        df.to_csv(path, index=False)
        print(f"✅ Trade log saved to {path}")


if __name__ == '__main__':
    btc_df = pd.read_csv("../../data/btc_usdt/btc_usdt_1h.csv")
    sol_df = pd.read_csv("../../data/sol_usdt/sol_usdt_1h.csv")
    xrp_df = pd.read_csv("../../data/xrp_usdt/xrp_usdt_1h.csv")
    shib_df = pd.read_csv("../../data/shib_usdt/shib_usdt_1h.csv")
    doge_df = pd.read_csv("../../data/doge_usdt/doge_usdt_1h.csv")
    bnb_df = pd.read_csv("../../data/bnb_usdt/bnb_usdt_1h.csv")
    eth_df = pd.read_csv("../../data/eth_usdt/eth_usdt_1h.csv")


    def make_train_env():
        raw_env = PriceActionEnv([btc_df, sol_df, xrp_df, shib_df, doge_df, bnb_df], window_size=24)
        return Monitor(raw_env)


    def make_eval_env():
        raw_env = PriceActionEnv([eth_df[-2400:]], window_size=24)
        return Monitor(raw_env)


    # Wrap with DummyVecEnv for SB3 compatibility
    env = DummyVecEnv([make_train_env])
    eval_env = DummyVecEnv([make_eval_env])

    # Evaluation logging
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="../../logs/checkpoints/",
        log_path="../../logs/evaluation_logs/",
        eval_freq=25000,
        deterministic=True,
        render=False,
    )

    # Define model
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=0,
        n_steps=512,
        batch_size=64,
        learning_rate=3e-4,
        tensorboard_log="../../logs/tensorboard/",
    )

    # Train the model
    model.learn(total_timesteps=200_000, callback=eval_callback, progress_bar=True)

    # Save model
    model.save("../../logs/last_model")
