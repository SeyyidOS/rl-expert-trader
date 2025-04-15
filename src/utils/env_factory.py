# src/train/env_factory.py
from typing import Dict
import pandas as pd
from stable_baselines3.common.monitor import Monitor
from src.env.spot_trading_env import SpotTradingEnv
from gymnasium.wrappers import FlattenObservation


def make_train_env(data: Dict[str, pd.DataFrame], run_dir: str = None,
                   monitor: bool = True, flatten: bool = True) -> Monitor:
    """
    Creates a monitored training environment using all assets except ETH.
    Args:
        :param run_dir:
        :param data:
        :param flatten:
        :param monitor:
    Returns:
        Monitor: A monitored training environment.

    """
    env_instance = SpotTradingEnv(data, mode="train", log_path=run_dir)

    if monitor:
        env_instance = Monitor(env_instance)

    if flatten:
        env_instance = FlattenObservation(env_instance)

    return env_instance


def make_eval_env(data: Dict[str, pd.DataFrame], run_dir: str = None,
                   monitor: bool = True, flatten: bool = True) -> Monitor:
    """
    Creates a monitored evaluation environment using a subset of ETH data.
    Args:
        :param run_dir:
        :param data:
        :param flatten:
        :param monitor:
    Returns:
        Monitor: A monitored evaluation environment.

    """
    env_instance = SpotTradingEnv(data, mode="eval", log_path=run_dir)

    if monitor:
        env_instance = Monitor(env_instance)

    if flatten:
        env_instance = FlattenObservation(env_instance)

    return env_instance
