from typing import List, Optional, Tuple, Dict, Any
from src.utils.logger import TradeLogger
from src.utils.helper import load_config
from gymnasium import spaces

import gymnasium as gym
import pandas as pd
import numpy as np
import logging
import random

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class SpotTradingEnv(gym.Env):
    """
    A simplified spot trading environment that supports only long positions.

    Actions:
      0 - Hold: do nothing.
      1 - Buy: open a long position (only if no position is open).
      2 - Sell: close a long position (only if a position is open) and realize profit.

    Reward:
      When closing a position, the reward is computed as (sell_price - buy_price).
      Otherwise, the reward is 0.

    Observation:
      The observation is a vector that includes:
        - A window of normalized closing prices.
        - A binary flag indicating whether a position is open (1.0 if open, 0.0 otherwise).
        - The current account balance normalized by the initial balance.
    """

    def __init__(self,
                 dfs: List[pd.DataFrame],
                 mode: str = "train",  # "train" or "eval"
                 render_mode: Optional[str] = None
                 ):
        super().__init__()

        self.config = load_config('./config/env.yaml')["SpotTradingEnv"]

        # Constructor parameters
        self.dfs = dfs
        self.mode = mode.lower()
        self.render_mode = render_mode

        # Config
        self.wrong_action_punishment = self.config["wrong_action_punishment"]
        self.eval_episode_length = self.config["eval_episode_length"]
        self.initial_balance = self.config["initial_balance"]
        self.episode_length = self.config["episode_length"]
        self.window_size = self.config["window_size"]
        self.balance = self.config["initial_balance"]
        self.size_pct = self.config["size_pct"]
        self.sl_pct = self.config["sl_pct"]
        self.fee = self.config["fee_rate"]

        # Custom parameters
        self.current_step = self.config["window_size"]
        self.trade_logger = TradeLogger()
        self.episode_count = 0
        self.position = None

        # Action and state space definitions
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell.

        obs_len = self.config["window_size"] + 3
        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_len,), dtype=np.float32)

        # Constructor methods
        self._reset_session()

    def _get_observation(self) -> np.ndarray:
        """
        Constructs the observation consisting of:
          - A window of the last window_size normalized closing prices.
          - A binary flag (1.0 if a position is open; otherwise 0.0).
          - The current balance normalized by the initial balance.
        """
        window_prices = self.df['close'].iloc[self.current_step - self.window_size:self.current_step].values
        # Min-max normalization of the price window:
        norm_prices = (window_prices - window_prices.min()) / (window_prices.max() - window_prices.min() + 1e-8)

        # Extra features: position flag and balance ratio.
        position_flag = np.array([1.0]) if self.position is not None else np.array([0.0])
        balance_feature = np.array([self.balance / self.initial_balance])

        if self.position:
            entry_price_norm = (self.position["entry_price"] - window_prices.min()) / (
                    window_prices.max() - window_prices.min() + 1e-8)
            entry_price_norm = np.array([entry_price_norm])
        else:
            entry_price_norm = np.array([-1.0])

        return np.concatenate([norm_prices, entry_price_norm, position_flag, balance_feature]).astype(np.float32)

    def _reset_session(self) -> None:
        """
        Resets the session by selecting a dataset and a starting index.
        """
        if self.mode == "eval":
            self.df = self.dfs[0].reset_index(drop=True)
            self.current_step = self.window_size
            self.episode_length = len(self.df) if self.eval_episode_length == -1 else self.eval_episode_length
        else:
            self.df = random.choice(self.dfs).reset_index(drop=True)
            max_start = len(self.df) - self.window_size - self.episode_length
            self.current_step = random.randint(self.window_size, max_start)
        self.start_step = self.current_step

        self.position = None
        self.balance = self.initial_balance
        self.total_pnl = 0.0
        self.trade_logger.clear()

    def _open_position(self, direction: int, price: float, size_pct: float, sl_pct: float) -> None:
        """
        Opens a new trading position.
        """
        capital = self.balance * size_pct
        self.position = {
            "entry_step": self.current_step,
            "capital": capital,
            "direction": direction,
            "entry_price": price,
            "stop_loss_pct": sl_pct,
            "quantity": capital / price
        }

        fee_cost = capital * self.fee
        self.balance -= fee_cost
        self.total_pnl -= fee_cost

        # trade_event = {
        #     "step": self.current_step,
        #     "type": "OPEN",
        #     "balance": self.balance,
        #     "dir": direction,
        #     "price": price,
        #     "capital": capital,
        #     "fee_cost": fee_cost
        # }
        # self.trade_logger.log_trade(trade_event)

    def _close_position(self, price: float) -> Tuple[float, Dict[str, float]]:
        """
        Closes the current position and returns a tuple:
          (sparse_reward, breakdown)
        The breakdown contains:
          - realized_component: K_SCALE * (net return)
          - cumulative_time_penalty: the accumulated penalty while holding
          - sparse_reward: (realized_component - cumulative_time_penalty)
        The account balance is updated accordingly.
        """
        p = self.position  # type: ignore
        entry_price = p["entry_price"]

        pnl = (price - entry_price) * p["quantity"]
        fee_cost = abs(p["quantity"]) * price * self.fee
        net_pnl = pnl - fee_cost
        normalized_net_pnl = (pnl - fee_cost) / p["capital"]

        self.balance += pnl - fee_cost
        self.total_pnl += pnl - fee_cost

        trade_event = {
            "type": "CLOSE",
            "entry_step": p["entry_step"],
            "close_step": self.current_step,
            "duration": self.current_step - p["entry_step"],
            "balance": self.balance,
            "capital": p["capital"],
            "fee_cost": fee_cost,
            "entry_price": p["entry_price"],
            "close_price": price,
            "pnl": pnl,
            "net_pnl": net_pnl,
            "normalized_net_pnl": normalized_net_pnl
        }

        reward_breakdown = {
            "normalized_net_pnl": normalized_net_pnl,
        }

        self.trade_logger.log_trade(trade_event)

        self.position = None

        return normalized_net_pnl, reward_breakdown

    def _check_stop_loss(self, price: float) -> bool:
        """
        Checks if the current price triggers the stop loss condition.
        """
        if not self.position:
            return False
        p = self.position
        delta = (price - p["entry_price"]) / p["entry_price"]
        return delta <= -p["stop_loss_pct"]

    def _is_terminal(self) -> bool:
        """
        Checks if the episode should terminate.
        """
        episode_ended = self.current_step >= self.start_step + self.episode_length
        end_of_data = self.current_step >= len(self.df) - 1
        is_liquidate = self.balance <= self.initial_balance * 0.05
        return is_liquidate or episode_ended or end_of_data

    def _terminal_reward(self) -> float:
        if self.balance < self.initial_balance:
            return -5.0
        return 0.0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """
        Resets the environment for a new episode.
        """
        if self.mode == "train" and self.episode_count > 0 and self.trade_logger.trades:
            filename = f"./logs/trade_logs/trade_log_episode_{self.episode_count}.csv"
            self.trade_logger.save(filename)
            # logging.info("Saved trade log for episode %d to %s", self.episode_count, filename)
        self.episode_count += 1

        super().reset(seed=seed)
        self._reset_session()
        return self._get_observation(), {}

    def step(self, action: int) -> tuple:
        """
        Executes the given action, returning the next observation, reward, done flag, truncated flag, and info.

        Action meanings:
          - Buy (action 1): Opens a long position if none is open.
          - Sell (action 2): If in a position, closes it, computes profit as (current_price - entry_price),
            updates the balance, and sets the reward equal to the profit.
          - Hold (action 0) or an invalid action for the current state leaves the environment unchanged.
        """
        episode_ended = self.current_step >= self.start_step + self.episode_length
        current_price = self.df.loc[self.current_step, 'close']
        reward_components: Dict[str, Any] = {}
        reward = 0.0

        if self.position:
            if action == 2 or episode_ended:
                reward, reward_components = self._close_position(current_price)
                if action != 2:
                    logging.info(f"SL - Reward: {reward}\tBalance: {self.balance}")
            elif action == 1:
                reward -= self.wrong_action_punishment
                reward_components['wrong_action_punishment'] = self.wrong_action_punishment

        else:
            if action == 1:
                self._open_position(1, current_price, self.size_pct, self.sl_pct)
            elif action == 2:
                reward -= self.wrong_action_punishment
                reward_components['wrong_action_punishment'] = self.wrong_action_punishment

        self.current_step += 1
        done = self._is_terminal()

        if done:
            terminal_reward = self._terminal_reward()
            reward_components['terminal_reward'] = terminal_reward
            reward += terminal_reward

        info = {
            "balance": self.balance,
            "total_pnl": self.total_pnl,
            "position": self.position,
            "reward_components": reward_components,
            "raw_reward": reward  # final reward at this step (dense or sparse event)
        }

        return self._get_observation(), reward, done, False, info

    def render(self):
        """
        Renders the current state of the environment.
        """
        current_price = self.df.loc[self.current_step, 'close']
        pos_str = f"Entry Price: {self.position:.2f}" if self.position is not None else "None"
        print(
            f"Step: {self.current_step}, Price: {current_price:.2f}, Balance: {self.balance:.2f}, Position: {pos_str}")

    def close(self):
        """
        Performs any necessary cleanup.
        """
        pass

    def save_trade_log(self, path: str = "trade_log.csv") -> None:
        """Saves the trade log to a CSV file."""
        self.trade_logger.save(path)