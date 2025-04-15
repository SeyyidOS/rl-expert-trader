from typing import List, Optional, Tuple, Dict, Any
from src.utils.logger import TradeLogger
from src.utils.helper import load_config
from gymnasium import spaces

import gymnasium as gym
import pandas as pd
import numpy as np
import logging
import random
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# bu yaziyi goruyorsan isini gucunu birakip cicine sarilmaya gel
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
      Additionally, if no position is open, the agent is penalized for missed opportunities
      (i.e. opportunity cost) when the market goes up.

    Observation:
      The observation is a dictionary that includes:
        - A window of normalized closing prices.
        - A binary flag indicating whether a position is open (1.0 if open, 0.0 otherwise).
        - The current account balance normalized by the initial balance.
    """

    def __init__(self,
                 data: Dict[str, pd.DataFrame],
                 mode: str = "train",  # "train" or "eval"
                 log_path: str = "./logs",
                 render_mode: Optional[str] = None
                 ):
        super().__init__()

        self.config = load_config('./config/env.yaml')["SpotTradingEnv"]

        # Constructor parameters
        self.data = data
        self.mode = mode.lower()
        self.log_path = log_path
        self.render_mode = render_mode

        # Config parameters
        self.wrong_action_punishment = self.config["wrong_action_punishment"]
        self.eval_episode_length = self.config["eval_episode_length"]
        self.initial_balance = self.config["initial_balance"]
        self.episode_length = self.config["episode_length"]
        self.window_size = self.config["window_size"]
        self.balance = self.config["initial_balance"]
        self.size_pct = self.config["size_pct"]
        self.sl_pct = self.config["sl_pct"]
        self.fee = self.config["fee_rate"]

        # New config parameter: opportunity cost rate.
        # This rate controls how harsh the penalty is when idle during a rising market.
        self.opp_cost_rate = self.config.get("opp_cost_rate", 1.0)

        # Custom parameters
        self.current_step = self.config["window_size"]
        self.trade_logger = TradeLogger()
        self.episode_count = 0
        self.process_id=None
        self.position = None
        self.coin = None
        # Action and state space definitions
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell.

        obs_len = self.config["window_size"] + 3
        self.observation_space = spaces.Dict({
            "window": spaces.Box(low=0.0, high=1.0, shape=(self.window_size, 1), dtype=np.float32),
            "other": spaces.Box(
                low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 10.0], dtype=np.float32),
                dtype=np.float32)
        })

        # Constructor methods
        self._reset_session()

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Constructs the observation dictionary with:
          - "window": A window of normalized closing prices reshaped as [window_size, 1].
          - "other": A vector of 3 features:
              [normalized entry price (or -1.0 if no position), position flag, normalized balance].
        """
        window_prices = self.df['close'].iloc[self.current_step - self.window_size:self.current_step].values.astype(
            np.float32)
        min_price = window_prices.min()
        max_price = window_prices.max()
        norm_prices = (window_prices - min_price) / (max_price - min_price + 1e-8)
        window_data = norm_prices.reshape(-1, 1)

        if self.position:
            entry_price = self.position["entry_price"]
            entry_price_norm = (entry_price - min_price) / (max_price - min_price + 1e-8)
        else:
            entry_price_norm = -1.0  # Indicator for no open position.
        position_flag = 1.0 if self.position is not None else 0.0
        balance_feature = self.balance / self.initial_balance

        other = np.array([entry_price_norm, position_flag, balance_feature], dtype=np.float32)
        return {"window": window_data, "other": other}

    def _reset_session(self) -> None:
        """Resets the session by selecting a new dataset and starting index."""
        if self.mode == "eval":
            self.df = self.data["eth"].reset_index(drop=True)
            self.current_step = self.window_size
            self.episode_length = len(self.df) if self.eval_episode_length == -1 else self.eval_episode_length
        else:
            coins = list(self.data.keys())
            coins.remove("eth")
            self.coin = random.choice(coins)
            self.df = self.data[self.coin].reset_index(drop=True)
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
            "entry_timestamp": self.df.iloc[self.current_step]["timestamp"],
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

    def _close_position(self, price: float) -> Tuple[float, Dict[str, float]]:
        """
        Closes the current position and returns a tuple:
          (sparse_reward, breakdown)
        The breakdown contains:
          - normalized_net_pnl: profit/loss as a fraction of the position's capital.
        The account balance is updated accordingly.
        """
        p = self.position  # type: ignore

        fee_cost = abs(p["quantity"]) * price * self.fee
        pnl = (price - p["entry_price"]) * p["quantity"]
        net_pnl = pnl - fee_cost
        normalized_net_pnl = (pnl - fee_cost) / p["capital"]

        self.balance += pnl - fee_cost
        self.total_pnl += pnl - fee_cost

        trade_event = {
            "type": "CLOSE",
            "pair": f"{self.coin.upper()}/USDT",
            "entry_timestamp": p["entry_timestamp"],
            "close_timestamp": self.df.iloc[self.current_step]["timestamp"],
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
        if self.balance <= self.initial_balance * 1.01:
            return -5.0
        return 0.0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[dict[str, np.array], dict]:
        """
        Resets the environment for a new episode.
        """
        if self.mode == "train" and self.episode_count > 0 and self.trade_logger.trades:
            filename = os.path.join(self.log_path, f"trade_logs/{self.process_id}_trade_log_episode_{self.episode_count}.csv")
            self.trade_logger.save(filename)
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

        Opportunity cost penalty:
          At every step in which no position is open, if the market goes up from the previous timestep,
          a penalty proportional to the missed gain is subtracted from the reward.
        """
        current_price = self.df.loc[self.current_step, 'close']
        reward_components: Dict[str, Any] = {}
        reward = 0.0

        # Process action based on whether a position is open.
        if self.position:
            if action == 2 or (self.current_step >= self.start_step + self.episode_length):
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

        # Store the price at the decision point.
        old_price = current_price

        # Advance time.
        self.current_step += 1

        # Opportunity cost: if the agent is not invested, it misses out on market gains.
        # Compute return over the interval from the old price to the new price.
        if self.position is None and self.current_step < len(self.df):
            new_price = self.df.loc[self.current_step, 'close']
            price_return = (new_price / old_price) - 1
            if price_return > 0:
                opp_penalty = self.opp_cost_rate * price_return
                reward -= opp_penalty
                reward_components['opp_cost_penalty'] = opp_penalty

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
            "raw_reward": reward  # final reward at this step
        }

        return self._get_observation(), reward, done, False, info

    def render(self):
        """
        Renders the current state of the environment.
        """
        current_price = self.df.loc[self.current_step, 'close']
        pos_str = f"Entry Price: {self.position['entry_price']:.2f}" if self.position is not None else "None"
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

    def set_process_id(self, data):
        self.process_id = data