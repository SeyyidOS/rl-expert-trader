import random
import logging
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


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

    # Risk penalty hyperparameters (existing)
    DRAWNDOWN_PENALTY_MULTIPLIER = 0.0005
    VOLATILITY_PENALTY_MULTIPLIER = 0.2

    # New hyperparameters for decomposed reward components:
    ALPHA_OPP_COST = 2.0  # Scale for opportunity cost penalty
    GAMMA_UNREALIZED = 1.0  # Scale for unrealized profit (or loss)
    DELTA_TIME = 0.001  # Time penalty per step when holding a position

    # Volatility window (for risk penalty calculation)
    VOL_WINDOW = 50


# ------------------------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


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
    Custom Trading Environment for Price Action Strategies.

    The environment simulates trading using historical data with fees, leverage,
    and risk management. Each episode lasts for a fixed number of steps, regardless
    of open or closed positions. At the end of an episode, any open position is
    force-closed. Additionally, the environment distinguishes between training and
    evaluation mode:
      - In "train" mode, data is sampled randomly and the episode resets are random.
      - In "eval" mode, a fixed (deterministic) start index is used.

    When running in training mode, the environment saves a trade log at the end
    of each episode for later analysis.
    """
    metadata = {"render_modes": ["human"], "render_fps": 4}
    # Define the direction mapping as a class-level constant.
    DIRECTION_MAP = {0: 0, 1: 1, 2: -1}

    def __init__(self,
                 dfs: List[pd.DataFrame],
                 config: TradingConfig,
                 mode: str = "train",  # "train" or "eval"
                 render_mode: Optional[str] = None) -> None:
        """
        Args:
            dfs (List[pd.DataFrame]): List of historical market data DataFrames.
            config (TradingConfig): Configuration parameters.
            mode (str): "train" for training (random start) or "eval" for evaluation (deterministic start).
            render_mode (Optional[str]): If "human", rendering is enabled.
        """
        super().__init__()
        self.dfs = dfs
        self.config = config
        self.mode = mode.lower()
        self.render_mode = render_mode

        # Environment parameters.
        self.window_size = self.config.WINDOW_SIZE
        self.initial_balance = self.config.INITIAL_BALANCE
        self.fee = self.config.FEE_RATE
        self.leverage = self.config.LEVERAGE

        # Account and risk management state.
        self.balance = self.initial_balance
        self.total_pnl = 0.0
        self.trade_logger = TradeLogger()  # Use the dedicated logger class.
        self.max_balance = self.balance
        self.reward_buffer: List[float] = []

        # Cumulative reward components for the episode.
        self.cum_trade_reward = 0.0
        self.cum_opp_cost = 0.0
        self.cum_unrealized_reward = 0.0
        self.cum_time_penalty = 0.0
        self.cum_risk_penalty = 0.0

        # Define action and observation spaces.
        self.action_space = spaces.MultiDiscrete([3, 10, 5])
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.window_size + 4,), dtype=np.float32)

        self.position: Optional[Dict[str, Any]] = None
        self.current_step: int = 0
        self.is_in_position: bool = False

        # For episode management.
        self.episode_count = 0

        self._reset_session()

    def _reset_session(self) -> None:
        """
        Resets the session by selecting a dataset and a starting index.
        In eval mode, the start index is deterministic.
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
        self.is_in_position = False
        self.balance = self.initial_balance
        self.total_pnl = 0.0
        self.trade_logger.clear()
        self.max_balance = self.balance
        self.reward_buffer = []

        # Reset cumulative reward component trackers.
        self.cum_trade_reward = 0.0
        self.cum_opp_cost = 0.0
        self.cum_unrealized_reward = 0.0
        self.cum_time_penalty = 0.0
        self.cum_risk_penalty = 0.0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """
        Resets the environment for a new episode.
        For training mode, saves the previous episode's trade log if available.

        Returns:
            Tuple[np.ndarray, dict]: The initial observation and an empty info dictionary.
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

        Returns:
            np.ndarray: The observation vector.
        """
        prices = self.df['close'].iloc[self.current_step - self.window_size:self.current_step].values
        norm_prices = (prices - prices.min()) / (prices.max() - prices.min() + 1e-8)
        extra_features = np.array([
            self.balance / self.initial_balance,
            float(self.position['direction']) if self.position else 0.0,
            self.position['entry_price'] / prices[-1] if self.position else 0.0,
            self.position['capital'] / self.balance if self.position else 0.0
        ])
        return np.concatenate([norm_prices, extra_features]).astype(np.float32)

    def _open_position(self, direction: int, price: float, size_percent: float, sl_pct: float) -> None:
        """
        Opens a new trading position.

        Args:
            direction (int): +1 for long, -1 for short.
            price (float): Current market price.
            size_percent (float): Fraction of the balance to allocate.
            sl_pct (float): Stop loss percentage.
        """
        self.is_in_position = True
        # Allocate the capital at risk.
        capital = self.balance * size_percent
        leverage_capital = capital * self.leverage
        self.position = {
            "capital": capital,
            "direction": direction,
            "entry_price": price,
            "leverage": self.leverage,
            "stop_loss_pct": sl_pct,
            "quantity": leverage_capital / price
        }
        fee_cost = leverage_capital * self.fee
        self.balance -= fee_cost
        self.total_pnl -= fee_cost

        trade_event = {
            "step": self.current_step,
            "type": "OPEN",
            "dir": direction,
            "price": price,
            "capital": capital,
            "leverage": self.leverage,
            "fee_cost": fee_cost
        }
        self.trade_logger.log_trade(trade_event)

    def _close_position(self, price: float) -> float:
        """
        Closes the current position and returns a trade reward computed as the percentage return
        on the allocated capital.

        Args:
            price (float): Price at which the position is closed.

        Returns:
            float: Trade reward (percentage return relative to allocated capital).
        """
        self.is_in_position = False
        p = self.position  # type: ignore
        pnl = (price - p["entry_price"]) * p["quantity"] * p["direction"]
        fee_cost = abs(p["quantity"]) * price * self.fee
        self.balance += pnl - fee_cost
        self.total_pnl += pnl - fee_cost

        trade_event = {
            "step": self.current_step,
            "type": "CLOSE",
            "exit_price": price,
            "pnl": pnl,
            "fee_cost": fee_cost
        }
        self.trade_logger.log_trade(trade_event)

        self.position = None
        # Compute trade reward as percentage return on allocated capital.
        trade_reward = (pnl - fee_cost) / (p["capital"] + 1e-8)
        return trade_reward

    def _check_stop_loss(self, price: float) -> bool:
        """
        Checks if the current price triggers the stop loss condition.

        Args:
            price (float): Current price.

        Returns:
            bool: True if stop loss is hit.
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

        Returns:
            bool: True if the balance is too low.
        """
        return self.balance <= 10

    def _is_terminal(self) -> bool:
        """
        Checks if the episode should terminate.

        Returns:
            bool: True if the episode is over.
        """
        episode_ended = self.current_step >= self.start_step + self.config.EPISODE_LENGTH
        end_of_data = self.current_step >= len(self.df) - 1
        return self._is_liquidated() or episode_ended or end_of_data

    def _terminal_reward(self) -> float:
        """
        Computes the terminal reward. If a position is open at the end of the episode,
        a forced close is executed.

        Returns:
            float: The final reward after forced closure.
        """
        if self._is_liquidated():
            return -self.balance
        if self.position:
            forced_close_price = self.df.loc[self.current_step, "close"]
            logging.info("Forced closing position at terminal step %d, price: %.2f", self.current_step,
                         forced_close_price)
            self._close_position(forced_close_price)
        return self.total_pnl

    def _update_max_balance(self) -> None:
        """Updates the maximum balance reached."""
        if self.balance > self.max_balance:
            self.max_balance = self.balance

    def _compute_drawdown_penalty(self) -> float:
        """
        Computes the normalized drawdown penalty as the percentage drawdown multiplied
        by the configured multiplier.
        """
        if self.balance < self.max_balance:
            # Compute the drawdown fraction relative to the maximum balance.
            drawdown_fraction = (self.max_balance - self.balance) / self.max_balance
            return drawdown_fraction * self.config.DRAWNDOWN_PENALTY_MULTIPLIER
        return 0.0

    def _compute_volatility_penalty(self) -> float:
        """
        Computes the volatility penalty based on the standard deviation of recent reward values.
        Optionally, we can normalize the volatility by the mean absolute reward over the window.
        """
        if len(self.reward_buffer) < 2:
            return 0.0

        # Compute the raw volatility (standard deviation).
        vol = np.std(self.reward_buffer[-self.config.VOL_WINDOW:])

        # Optional: Normalize volatility by the mean absolute reward to get a relative measure.
        # Uncomment the following lines if you wish to use normalization.
        # mean_abs = np.mean(np.abs(self.reward_buffer[-self.config.VOL_WINDOW:]))
        # if mean_abs > 1e-8:
        #     vol = vol / mean_abs

        return vol * self.config.VOLATILITY_PENALTY_MULTIPLIER

    def _compute_risk_penalty(self) -> float:
        """Aggregates drawdown and volatility penalties."""
        return self._compute_drawdown_penalty() + self._compute_volatility_penalty()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Executes the given action, computes reward components, and returns the new state.

        Reward components:
          - Trade Reward: Percentage return on allocated capital if a trade is closed.
          - Opportunity Cost: Applied when idle during upward market trends.
          - Unrealized Reward: Percentage return on an open position.
          - Time Penalty: A small constant penalty per step when holding a position.
          - Risk Penalty: Computed from drawdown and volatility.

        Both instantaneous (per step) and cumulative reward component sums are added to the info dictionary.

        Returns:
            Tuple (observation, adjusted reward, done flag, truncation flag, info dict).
        """
        direction_code, size_code, sl_code = action
        direction = self.DIRECTION_MAP[direction_code]
        size_percent = (size_code + 1) / 10
        sl_pct = 0.005 * (sl_code + 1)

        price = self.df.loc[self.current_step, "close"]

        # Initialize instantaneous reward components.
        trade_reward = 0.0
        opp_cost = 0.0
        unrealized_reward = 0.0
        time_penalty = 0.0

        # --- Position Management ---
        if self.position:
            if self._check_stop_loss(price) or (direction == 0):
                trade_reward = self._close_position(price)
            # Otherwise, continue holding the position.
        elif direction != 0:
            self._open_position(direction, price, size_percent, sl_pct)

        # --- Opportunity Cost (when idle) ---
        if not self.is_in_position and self.current_step > 0:
            prev_price = self.df.loc[self.current_step - 1, "close"]
            market_return = (price - prev_price) / (prev_price + 1e-8)
            if market_return > 0:
                opp_cost = -abs(market_return * self.config.ALPHA_OPP_COST)

        # --- Unrealized Reward & Time Penalty (for open position) ---
        if self.is_in_position and self.position:
            p = self.position
            # Compute percentage return based on current price.
            if p["direction"] == 1:
                unrealized = (price - p["entry_price"]) / p["entry_price"]
            else:
                unrealized = (p["entry_price"] - price) / p["entry_price"]
            unrealized_reward = unrealized * self.config.GAMMA_UNREALIZED
            time_penalty = self.config.DELTA_TIME

        # --- Combine Instantaneous Reward Components ---
        if trade_reward != 0.0:
            reward = trade_reward
        elif self.is_in_position:
            reward = unrealized_reward - time_penalty
        else:
            reward = opp_cost

        # --- Update Cumulative Reward Component Sums ---
        self.cum_trade_reward += trade_reward
        self.cum_opp_cost += opp_cost
        self.cum_unrealized_reward += unrealized_reward
        self.cum_time_penalty += time_penalty
        # Risk penalty is computed each step, so accumulate it as well.
        risk_penalty = self._compute_risk_penalty()
        self.cum_risk_penalty += risk_penalty

        # --- Final Reward ---
        adjusted_reward = reward - risk_penalty

        self.current_step += 1
        self._update_max_balance()
        self.reward_buffer.append(reward)

        done = self._is_terminal()
        if done:
            adjusted_reward += self._terminal_reward()

        # Prepare the info dictionary with both instantaneous and cumulative reward components.
        info = {
            "balance": self.balance,
            "total_pnl": self.total_pnl,
            "position": self.position,
            "instant_reward_components": {
                "trade_reward": trade_reward,
                "opp_cost": opp_cost,
                "unrealized_reward": unrealized_reward,
                "time_penalty": time_penalty,
                "risk_penalty": risk_penalty
            },
            "cumulative_reward_components": {
                "trade_reward": self.cum_trade_reward,
                "opp_cost": self.cum_opp_cost,
                "unrealized_reward": self.cum_unrealized_reward,
                "time_penalty": self.cum_time_penalty,
                "risk_penalty": self.cum_risk_penalty
            }
        }

        if self.render_mode == "human":
            self.render()

        return self._get_observation(), adjusted_reward, done, False, info

    def render(self) -> None:
        """Logs the current state of the environment."""
        price = self.df.loc[self.current_step, "close"]
        logging.info("=" * 60)
        logging.info("Step        : %d", self.current_step)
        logging.info("Price       : $%.2f", price)
        logging.info("Balance     : $%.2f", self.balance)
        if self.position:
            side = "LONG" if self.position["direction"] == 1 else "SHORT"
            logging.info("Position    : %s, Capital: %.4f @ $%.2f", side, self.position["capital"],
                         self.position["entry_price"])
            logging.info("Stop Loss   : %.2f%%", self.position["stop_loss_pct"] * 100)
        else:
            logging.info("Position    : None")
        logging.info("Total PnL   : $%.2f", self.total_pnl)
        logging.info("=" * 60)

    def save_trade_log(self, path: str = "trade_log.csv") -> None:
        """Saves the trade log to a CSV file."""
        self.trade_logger.save(path)
