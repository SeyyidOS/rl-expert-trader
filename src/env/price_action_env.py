from typing import List, Optional, Dict, Any, Tuple
from gymnasium import spaces

import gymnasium as gym
import pandas as pd
import numpy as np
import random
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
class TradingConfig:
    """
    Holds all configuration parameters and hyperparameters for the environment and training.
    """
    INITIAL_BALANCE = 10000.0
    FEE_RATE = 0.0005
    LEVERAGE = 5
    WINDOW_SIZE = 24
    EPISODE_LENGTH = 1000  # Number of steps per episode
    TRAIN_EVAL_FREQUENCY = 25000
    TOTAL_TIMESTEPS = 200_000

    # New reward hyperparameters for normalized, unified rewards:
    K_SCALE = 1.0  # Scaling factor to express returns as percentages
    DENSE_SCALE = 0.01  # Scaling factor to express returns as percentages
    TIME_PENALTY = 0.001  # Per-step penalty for holding a position
    LIQUIDATION_PENALTY = 1.0  # Severe penalty (normalized) on liquidation

    # (Other parameters may remain if needed)
    DRAWNDOWN_PENALTY_MULTIPLIER = 0.0005
    VOLATILITY_PENALTY_MULTIPLIER = 0.2
    VOL_WINDOW = 50


# ------------------------------------------------------------------------------
# Dedicated Trade Logger Class
# ------------------------------------------------------------------------------
class TradeLogger:
    """
    A dedicated class for logging trade events.

    This class collects trade events and can save the log to a CSV file for later analysis.
    """

    def __init__(self) -> None:
        self.trades: List[Dict[str, Any]] = []

    def log_trade(self, trade: Dict[str, Any]) -> None:
        """Appends a trade dictionary to the log."""
        self.trades.append(trade)

    def save(self, path: str) -> None:
        """Saves the trade log to a CSV file at the given path."""
        if not self.trades:
            logging.warning("No trades to save.")
            return
        df = pd.DataFrame(self.trades)
        df.to_csv(path, index=False)
        logging.info("Trade log saved to %s", path)

    def clear(self) -> None:
        """Clears the logged trades."""
        self.trades.clear()


# ------------------------------------------------------------------------------
# Environment Implementation
# ------------------------------------------------------------------------------
class PriceActionEnv(gym.Env):
    """
    Custom Trading Environment for Price Action Strategies with a unified reward structure.

    Reward Structure:
      * Dense (per-step) rewards:
          - If in position:
              - Dense Reward = K_SCALE * ((P_t / P_entry) - 1) [for long]
                                or = K_SCALE * ((P_entry / P_t) - 1) [for short]
                                minus a TIME_PENALTY.
              - Reward components: "unrealized_component" and "time_penalty_component".
          - If out of position:
              - Dense Reward = -K_SCALE * ((P_t / P_{t-1}) - 1)
                (reflects opportunity cost).
      * Sparse rewards (event-driven):
          - On trade closure:
              - Sparse Reward = K_SCALE * (net return) - (cumulative time penalty)
                with components "realized_component" and "cumulative_time_penalty".
          - On liquidation:
              - Sparse Reward = -LIQUIDATION_PENALTY.
    """
    metadata = {"render_modes": ["human"], "render_fps": 4}
    # Action mapping: 0 => hold, 1 => long, 2 => short (here, 2 corresponds to -1)
    DIRECTION_MAP = {0: 0, 1: 1, 2: -1}

    def __init__(self,
                 dfs: List[pd.DataFrame],
                 config: TradingConfig,
                 mode: str = "train",  # "train" or "eval"
                 render_mode: Optional[str] = None) -> None:
        super().__init__()
        self.dfs = dfs
        self.config = config
        self.mode = mode.lower()
        self.render_mode = render_mode

        self.window_size = self.config.WINDOW_SIZE
        self.initial_balance = self.config.INITIAL_BALANCE
        self.fee = self.config.FEE_RATE
        self.leverage = self.config.LEVERAGE

        self.balance = self.initial_balance
        self.total_pnl = 0.0
        self.trade_logger = TradeLogger()
        self.max_balance = self.balance

        # Reset cumulative holding penalty for each trade.
        self.cum_time_penalty = 0.0

        # Define action and observation spaces.
        self.action_space = spaces.MultiDiscrete([3, 10, 5])
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.window_size + 4,), dtype=np.float32)

        self.position: Optional[Dict[str, Any]] = None
        self.current_step: int = 0

        self.episode_count = 0

        self._reset_session()

    def _reset_session(self) -> None:
        """
        Resets the session by selecting a dataset and a starting index.
        """
        if self.mode == "eval":
            self.df = self.dfs[0].reset_index(drop=True)
            self.current_step = self.window_size
        else:
            self.df = random.choice(self.dfs).reset_index(drop=True)
            max_start = len(self.df) - self.window_size - self.config.EPISODE_LENGTH
            self.current_step = random.randint(self.window_size, max_start)
        self.start_step = self.current_step

        self.position = None
        self.balance = self.initial_balance
        self.total_pnl = 0.0
        self.trade_logger.clear()
        self.max_balance = self.balance
        self.cum_time_penalty = 0.0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """
        Resets the environment for a new episode.
        """
        if self.mode == "train" and self.episode_count > 0 and self.trade_logger.trades:
            filename = f"./logs/trade_logs/trade_log_episode_{self.episode_count}.csv"
            self.trade_logger.save(filename)
            logging.info("Saved trade log for episode %d to %s", self.episode_count, filename)
        self.episode_count += 1

        super().reset(seed=seed)
        self._reset_session()
        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """
        Constructs the current observation consisting of normalized recent price data and account features.
        """
        prices = self.df['close'].iloc[self.current_step - self.window_size:self.current_step].values
        norm_prices = (prices - prices.min()) / (prices.max() - prices.min() + 1e-8)
        extra_features = np.array([
            self.balance / self.initial_balance,
            float(self.position['direction']) if self.position else 0.0,
            self.position['entry_price'] / prices[-1] if self.position else 0.0,
            0.0  # Placeholder for additional features
        ])
        return np.concatenate([norm_prices, extra_features]).astype(np.float32)

    def _open_position(self, direction: int, price: float, size_percent: float, sl_pct: float) -> None:
        """
        Opens a new trading position.
        """
        capital = self.balance * size_percent
        self.position = {
            "capital": capital,
            "direction": direction,
            "entry_price": price,
            "leverage": self.leverage,
            "stop_loss_pct": sl_pct,
            "quantity": capital * self.leverage / price
        }
        fee_cost = capital * self.leverage * self.fee
        self.balance -= fee_cost
        self.total_pnl -= fee_cost

        # Reset cumulative time penalty for this new trade.
        self.cum_time_penalty = 0.0

        trade_event = {
            "step": self.current_step,
            "type": "OPEN",
            "balance": self.balance,
            "dir": direction,
            "price": price,
            "capital": capital,
            "leverage": self.leverage,
            "fee_cost": fee_cost
        }
        self.trade_logger.log_trade(trade_event)

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
        direction = p["direction"]

        if direction == 1:
            net_return = (price / entry_price) - 1.0
        else:
            net_return = (entry_price / price) - 1.0

        realized_component = self.config.K_SCALE * net_return
        time_penalty_component = self.cum_time_penalty
        sparse_reward = realized_component - time_penalty_component

        reward_breakdown = {
            "realized_component": realized_component,
            # "cumulative_time_penalty": time_penalty_component,
            "sparse_reward": sparse_reward
        }

        pnl = p["capital"] * self.leverage * net_return
        fee_cost = abs(p["quantity"]) * price * self.fee
        self.balance += pnl - fee_cost
        self.total_pnl += pnl - fee_cost

        trade_event = {
            "step": self.current_step,
            "type": "CLOSE",
            "balance": self.balance,
            "price": price,
            "capital": p["capital"],
            "leverage": self.leverage,
            "fee_cost": fee_cost,
            "net_return": net_return,
            "pnl": pnl,
        }
        self.trade_logger.log_trade(trade_event)

        self.position = None
        self.cum_time_penalty = 0.0

        return sparse_reward, reward_breakdown

    def _check_stop_loss(self, price: float) -> bool:
        """
        Checks if the current price triggers the stop loss condition.
        """
        if not self.position:
            return False
        p = self.position
        delta = (price - p["entry_price"]) / p["entry_price"]
        if p["direction"] == -1:
            delta *= -1
        return delta <= -p["stop_loss_pct"]

    def _is_liquidated(self) -> bool:
        """
        Determines if the account is liquidated.
        """
        return self.balance <= 10

    def _is_terminal(self) -> bool:
        """
        Checks if the episode should terminate.
        """
        episode_ended = self.current_step >= self.start_step + self.config.EPISODE_LENGTH
        end_of_data = self.current_step >= len(self.df) - 1
        return self._is_liquidated() or episode_ended or end_of_data

    def _terminal_reward(self) -> Tuple[float, Optional[Dict[str, float]]]:
        """
        Computes the terminal reward.
          - If liquidated: returns (-LIQUIDATION_PENALTY, breakdown)
          - If a position is open: force-close the position.
        """
        if self._is_liquidated():
            breakdown = {
                "liquidation_penalty": self.config.LIQUIDATION_PENALTY,
                "sparse_reward": -self.config.LIQUIDATION_PENALTY
            }
            return -self.config.LIQUIDATION_PENALTY, breakdown
        if self.position:
            forced_close_price = self.df.loc[self.current_step, "close"]
            logging.info("Forced closing position at terminal step %d, price: %.2f", self.current_step,
                         forced_close_price)
            return self._close_position(forced_close_price)
        return 0.0, None

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Executes the given action and computes rewards using the unified reward structure.
        In addition, a dictionary 'reward_components' is built containing the contribution of each component.
        """
        direction_code, size_code, sl_code = action
        action_direction = self.DIRECTION_MAP[direction_code]
        size_percent = (size_code + 1) / 10  # fraction of balance to allocate
        sl_pct = 0.005 * (sl_code + 1)  # stop loss percentage

        price = self.df.loc[self.current_step, "close"]
        reward = 0.0
        reward_components: Dict[str, Any] = {}

        if self.position:
            # If the agent signals to close (action_direction == 0) or if stop loss triggers:
            if action_direction == 0 or self._check_stop_loss(price):
                reward, breakdown = self._close_position(price)
                reward_components.update(breakdown)
            else:
                # Continue holding: accumulate a time penalty.
                self.cum_time_penalty += self.config.TIME_PENALTY
                if self.position["direction"] == 1:
                    unrealized_component = self.config.DENSE_SCALE * ((price / self.position["entry_price"]) - 1)
                else:
                    unrealized_component = self.config.DENSE_SCALE * ((self.position["entry_price"] / price) - 1)
                time_penalty_component = self.config.TIME_PENALTY
                dense_reward = unrealized_component - time_penalty_component
                reward = dense_reward
                reward_components = {
                    "unrealized_component": unrealized_component,
                    "time_penalty_component": time_penalty_component,
                    "dense_reward": dense_reward
                }
        else:
            # When not in a position:
            if action_direction != 0:
                # Open a new position; reward is zero at the moment of opening.
                self._open_position(action_direction, price, size_percent, sl_pct)
            else:
                # Compute opportunity cost from market movement.
                if self.current_step > self.window_size:
                    prev_price = self.df.loc[self.current_step - 1, "close"]
                    opportunity_cost = -self.config.DENSE_SCALE * ((price / prev_price) - 1)
                    reward = opportunity_cost
                    reward_components = {
                        "opportunity_cost": opportunity_cost,
                        "dense_reward": opportunity_cost
                    }
                else:
                    reward = 0.0
                    reward_components = {"dense_reward": 0.0}

        self.current_step += 1
        if self.balance > self.max_balance:
            self.max_balance = self.balance

        done = self._is_terminal()
        if done:
            terminal_reward, terminal_breakdown = self._terminal_reward()
            reward += terminal_reward
            if terminal_breakdown:
                reward_components["terminal"] = terminal_breakdown

        info = {
            "balance": self.balance,
            "total_pnl": self.total_pnl,
            "position": self.position,
            "reward_components": reward_components,
            "raw_reward": reward  # final reward at this step (dense or sparse event)
        }

        if self.render_mode == "human":
            self.render()

        return self._get_observation(), reward, done, False, info

    def render(self) -> None:
        """Logs the current state of the environment."""
        price = self.df.loc[self.current_step, "close"]
        logging.info("=" * 60)
        logging.info("Step        : %d", self.current_step)
        logging.info("Price       : $%.2f", price)
        logging.info("Balance     : $%.2f", self.balance)
        if self.position:
            side = "LONG" if self.position["direction"] == 1 else "SHORT"
            logging.info("Position    : %s, Capital: %.4f @ $%.2f",
                         side, self.position["capital"], self.position["entry_price"])
            logging.info("Stop Loss   : %.2f%%", self.position["stop_loss_pct"] * 100)
        else:
            logging.info("Position    : None")
        logging.info("Total PnL   : $%.2f", self.total_pnl)
        logging.info("=" * 60)

    def save_trade_log(self, path: str = "trade_log.csv") -> None:
        """Saves the trade log to a CSV file."""
        self.trade_logger.save(path)
