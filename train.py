import logging
from typing import Dict

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

# Import your custom trading environment and its config.
from src.env.price_action_env import PriceActionEnv, TradingConfig


# ------------------------------------------------------------------------------
# Data Loading and Environment Factory Functions
# ------------------------------------------------------------------------------
def load_market_data() -> Dict[str, pd.DataFrame]:
    """
    Loads market data from CSV files.
    Update file paths as necessary.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary with asset keys.
    """
    base_path = "./data"
    data_files = {
        "btc": f"{base_path}/btc_usdt/btc_usdt_1h.csv",
        "sol": f"{base_path}/sol_usdt/sol_usdt_1h.csv",
        "xrp": f"{base_path}/xrp_usdt/xrp_usdt_1h.csv",
        "shib": f"{base_path}/shib_usdt/shib_usdt_1h.csv",
        "doge": f"{base_path}/doge_usdt/doge_usdt_1h.csv",
        "bnb": f"{base_path}/bnb_usdt/bnb_usdt_1h.csv",
        "eth": f"{base_path}/eth_usdt/eth_usdt_1h.csv"
    }
    return {key: pd.read_csv(path) for key, path in data_files.items()}


def make_train_env(data: Dict[str, pd.DataFrame], config: TradingConfig) -> Monitor:
    """
    Creates a monitored training environment using all assets except ETH.

    Args:
        data (Dict[str, pd.DataFrame]): Market data.
        config (TradingConfig): Configuration parameters.

    Returns:
        Monitor: A monitored training environment.
    """
    # Exclude ETH from training, so that training uses diverse datasets.
    dfs = [df for key, df in data.items() if key != "eth"]
    env_instance = PriceActionEnv(dfs, config, mode="train")
    return Monitor(env_instance)


def make_eval_env(data: Dict[str, pd.DataFrame], config: TradingConfig) -> Monitor:
    """
    Creates a monitored evaluation environment using a subset of ETH data.

    Args:
        data (Dict[str, pd.DataFrame]): Market data.
        config (TradingConfig): Configuration parameters.

    Returns:
        Monitor: A monitored evaluation environment.
    """
    # Use a subset of ETH data (e.g., last 2400 data points) for evaluation.
    eth_df = data["eth"]
    env_instance = PriceActionEnv([eth_df[-2400:]], config, mode="eval")
    return Monitor(env_instance)


# ------------------------------------------------------------------------------
# Training Pipeline
# ------------------------------------------------------------------------------
def train_model():
    """
    Sets up and trains the PPO model using the custom trading environment.
    This function loads market data, creates training and evaluation environments,
    and trains the PPO model with periodic evaluation callbacks.
    """
    config = TradingConfig()
    data = load_market_data()

    # Wrap environments with DummyVecEnv as required by Stable Baselines3.
    train_env = DummyVecEnv([lambda: make_train_env(data, config)])
    eval_env = DummyVecEnv([lambda: make_eval_env(data, config)])

    # Set up the evaluation callback for periodic model evaluation and checkpointing.
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/checkpoints/",
        log_path="./logs/evaluation_logs/",
        eval_freq=config.TRAIN_EVAL_FREQUENCY,
        deterministic=True,
        render=False,
    )

    # Define and initialize the PPO model.
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        verbose=0,
        n_steps=512,
        batch_size=64,
        learning_rate=3e-4,
        tensorboard_log="./logs/tensorboard/"
    )

    logging.info("Starting model training...")
    model.learn(total_timesteps=config.TOTAL_TIMESTEPS, callback=eval_callback, progress_bar=True)
    logging.info("Training completed. Saving model...")
    model.save("./logs/last_model")
    logging.info("Model saved successfully.")


# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    train_model()
