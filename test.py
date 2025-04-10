import logging

import gymnasium
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Any

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from src.env.price_action_env import PriceActionEnv, TradingConfig


# -------------------------------------------------------------------
# Data Loading Function
# -------------------------------------------------------------------
def load_market_data() -> Dict[str, pd.DataFrame]:
    """
    Loads market data from CSV files.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary with asset keys.
    """
    base_path = "./data"
    data_files = {
        "eth": f"{base_path}/eth_usdt/eth_usdt_1h.csv"
        # Add other assets if needed.
    }
    data = {key: pd.read_csv(path) for key, path in data_files.items()}
    return data


# -------------------------------------------------------------------
# Environment Creation Function
# -------------------------------------------------------------------
def create_eval_env(data: Dict[str, pd.DataFrame], config: TradingConfig, render_mode: str = None) -> PriceActionEnv:
    """
    Creates a monitored evaluation environment using ETH data.

    Args:
        data (Dict[str, pd.DataFrame]): Market data.
        config (TradingConfig): Configuration parameters.
        render_mode (str, optional): Rendering mode.

    Returns:
        Monitor: A monitored evaluation environment.
    """
    eth_df = data["eth"]
    env_instance = PriceActionEnv([eth_df], config, mode="eval", render_mode=render_mode)
    return env_instance


# -------------------------------------------------------------------
# Evaluation Runner Function
# -------------------------------------------------------------------
def run_evaluation(eval_env: PriceActionEnv) -> Dict[str, Any]:
    """
    Runs one evaluation episode using the given environment.
    It collects and returns information on the reward components,
    account balance, total PnL, and step indices.

    Returns:
        Dict[str, Any]: A dictionary with keys:
            - "info_history": list of info dicts per step.
            - "balance_history": account balance per step.
            - "pnl_history": total PnL per step.
            - "steps": list of step indices.
            - "total_reward": overall episode reward.
    """
    info_history: List[Dict[str, Any]] = []
    balance_history: List[float] = []
    pnl_history: List[float] = []
    steps: List[int] = []

    model = PPO.load("./logs/checkpoints/best_model.zip")
    obs, _ = eval_env.reset()
    done = False
    truncated = False
    total_reward = 0.0
    step = 0

    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        # action = eval_env.action_space.sample()
        obs, reward, done, truncated, info = eval_env.step(action)
        total_reward += reward

        info_history.append(info)
        balance_history.append(info["balance"])
        pnl_history.append(info["total_pnl"])
        steps.append(step)
        step += 1

    # Save the trade log.
    eval_env.save_trade_log("logs/eval_trades.csv")
    logging.info("Total reward: {:.4f}".format(total_reward))

    return {
        "info_history": info_history,
        "balance_history": balance_history,
        "pnl_history": pnl_history,
        "steps": steps,
        "total_reward": total_reward
    }


# -------------------------------------------------------------------
# Plotting Functions
# -------------------------------------------------------------------
def plot_cumulative_rewards(info_history: List[Dict[str, Any]], steps: List[int]) -> None:
    """
    Plots the cumulative reward components over steps.
    """
    cum_trade = [info["cumulative_reward_components"]["trade_reward"] for info in info_history]
    cum_opp = [info["cumulative_reward_components"]["opp_cost"] for info in info_history]
    cum_unrealized = [info["cumulative_reward_components"]["unrealized_reward"] for info in info_history]
    cum_time = [info["cumulative_reward_components"]["time_penalty"] for info in info_history]
    cum_risk = [info["cumulative_reward_components"]["risk_penalty"] for info in info_history]

    plt.figure(figsize=(12, 8))
    plt.plot(steps, cum_trade, label="Cumulative Trade Reward")
    plt.plot(steps, cum_opp, label="Cumulative Opportunity Cost")
    plt.plot(steps, cum_unrealized, label="Cumulative Unrealized Reward")
    plt.plot(steps, cum_time, label="Cumulative Time Penalty")
    plt.plot(steps, cum_risk, label="Cumulative Risk Penalty")
    plt.title("Cumulative Reward Components Over Steps")
    plt.xlabel("Step")
    plt.ylabel("Reward Component (Percentage)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_account_metrics(steps: List[int], balance_history: List[float], pnl_history: List[float]) -> None:
    """
    Plots account balance and total PnL over the course of the episode.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(steps, balance_history, label="Account Balance")
    plt.plot(steps, pnl_history, label="Total PnL")
    plt.title("Account Balance and Total PnL Over Steps")
    plt.xlabel("Step")
    plt.ylabel("Value ($)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_instantaneous_rewards(info_history: List[Dict[str, Any]], steps: List[int]) -> None:
    """
    Plots the instantaneous reward components over steps.
    """
    inst_trade = [info["instant_reward_components"]["trade_reward"] for info in info_history]
    inst_opp = [info["instant_reward_components"]["opp_cost"] for info in info_history]
    inst_unrealized = [info["instant_reward_components"]["unrealized_reward"] for info in info_history]
    inst_time = [info["instant_reward_components"]["time_penalty"] for info in info_history]
    inst_risk = [info["instant_reward_components"]["risk_penalty"] for info in info_history]

    plt.figure(figsize=(12, 8))
    plt.plot(steps, inst_trade, label="Instant Trade Reward")
    plt.plot(steps, inst_opp, label="Instant Opportunity Cost")
    plt.plot(steps, inst_unrealized, label="Instant Unrealized Reward")
    plt.plot(steps, inst_time, label="Instant Time Penalty")
    plt.plot(steps, inst_risk, label="Instant Risk Penalty")
    plt.title("Instantaneous Reward Components Over Steps")
    plt.xlabel("Step")
    plt.ylabel("Reward Component (Percentage)")
    plt.legend()
    plt.grid(True)
    plt.show()


# -------------------------------------------------------------------
# Main Function
# -------------------------------------------------------------------
def main():
    logging.info("Starting evaluation...")
    data = load_market_data()
    config = TradingConfig()
    eval_env = create_eval_env(data, config, render_mode=None)
    eval_results = run_evaluation(eval_env)

    plot_cumulative_rewards(eval_results["info_history"], eval_results["steps"])
    plot_account_metrics(eval_results["steps"], eval_results["balance_history"], eval_results["pnl_history"])
    plot_instantaneous_rewards(eval_results["info_history"], eval_results["steps"])


if __name__ == '__main__':
    main()
