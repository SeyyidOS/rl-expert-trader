# src/train/env_factory.py
from typing import Dict
import pandas as pd
from stable_baselines3.common.monitor import Monitor
from src.env.spot_trading_env import SpotTradingEnv


def make_train_env(data: Dict[str, pd.DataFrame]) -> Monitor:
    """
    Creates a monitored training environment using all assets except ETH.
    Args:
        data (Dict[str, pd.DataFrame]): Market data.
    Returns:
        Monitor: A monitored training environment.
    """
    # Exclude ETH to diversify the training dataset.
    dfs = [df for key, df in data.items() if key != "eth"]
    env_instance = SpotTradingEnv(dfs, mode="train")
    return Monitor(env_instance)


def make_eval_env(data: Dict[str, pd.DataFrame]) -> Monitor:
    """
    Creates a monitored evaluation environment using a subset of ETH data.
    Args:
        data (Dict[str, pd.DataFrame]): Market data.
    Returns:
        Monitor: A monitored evaluation environment.
    """
    eth_df = data["eth"]
    # Use a subset (e.g., last 2400 data points) for evaluation.
    env_instance = SpotTradingEnv([eth_df[-2400:]], mode="eval")
    return Monitor(env_instance)
