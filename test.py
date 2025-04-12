import logging
import gymnasium
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Any

from stable_baselines3 import PPO
from src.env.price_action_env import PriceActionEnv, TradingConfig
from src.env.spot_trading_env import SpotTradingEnv


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
def create_eval_env(data: Dict[str, pd.DataFrame],
                    config: TradingConfig,
                    render_mode: str = None) -> PriceActionEnv:
    """
    Creates a monitored evaluation environment using ETH data.

    Args:
        data (Dict[str, pd.DataFrame]): Market data.
        config (TradingConfig): Configuration parameters.
        render_mode (str, optional): Rendering mode.

    Returns:
        PriceActionEnv: An evaluation environment instance.
    """
    eth_df = data["eth"]
    env_instance = SpotTradingEnv([eth_df], mode="eval", render_mode=render_mode)
    return env_instance


# -------------------------------------------------------------------
# Evaluation Runner Function
# -------------------------------------------------------------------
def run_evaluation(eval_env: PriceActionEnv) -> Dict[str, Any]:
    """
    Runs one evaluation episode using the given environment.
    It collects and returns the history of reward info,
    account balance, total PnL, and steps.

    Returns:
        Dict[str, Any]: Dictionary with keys:
            - "info_history": list of info dictionaries per step.
            - "balance_history": account balance per step.
            - "pnl_history": total PnL per step.
            - "steps": list of step indices.
            - "total_reward": overall episode reward.
    """
    info_history: List[Dict[str, Any]] = []
    balance_history: List[float] = []
    pnl_history: List[float] = []
    steps: List[int] = []

    # Load the trained PPO model.
    model = PPO.load("./logs/checkpoints/best_model.zip")
    obs, _ = eval_env.reset()
    done = False
    truncated = False
    total_reward = 0.0
    step = 0

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
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
def plot_reward_components(info_history: List[Dict[str, Any]], steps: List[int]) -> None:
    """
    Extracts and plots each reward component on its own subplot.

    It builds time series for each key in the "reward_components" dictionary (found in the
    info dict returned by the environment), using a default value 0 when missing.
    """
    # Determine the union of reward component keys over all steps.
    component_keys = set()
    for info in info_history:
        rc = info.get("reward_components", {})
        component_keys.update(rc.keys())

    # Build time series for each component key.
    components_data = {k: [] for k in component_keys}
    for info in info_history:
        rc = info.get("reward_components", {})
        for k in component_keys:
            value = rc.get(k, 0)
            # In case the value is a nested dict (e.g., terminal breakdown),
            # try to get "sparse_reward" if available; otherwise sum the values.
            if isinstance(value, dict):
                value = value.get("sparse_reward", sum(value.values()))
            components_data[k].append(value)

    num_components = len(component_keys)
    fig, axs = plt.subplots(num_components, 1, figsize=(12, 3 * num_components), sharex=True)

    if num_components == 1:
        axs = [axs]

    for ax, key in zip(axs, sorted(component_keys)):
        ax.plot(steps, components_data[key], label=key)
        ax.set_title(f"{key} Over Steps")
        ax.set_ylabel("Reward Value (Normalized)")
        ax.legend()
        ax.grid(True)

    axs[-1].set_xlabel("Step")
    plt.tight_layout()
    plt.show()


def plot_raw_rewards(info_history: List[Dict[str, Any]], steps: List[int]) -> None:
    """
    Plots the overall (raw) reward received at each step.
    """
    raw_rewards = [info.get("raw_reward", 0) for info in info_history]
    plt.figure(figsize=(12, 6))
    plt.plot(steps, raw_rewards, label="Raw Reward", color="purple")
    plt.title("Overall (Raw) Reward Over Steps")
    plt.xlabel("Step")
    plt.ylabel("Reward")
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


# -------------------------------------------------------------------
# Main Function
# -------------------------------------------------------------------
def main():
    logging.info("Starting evaluation...")
    data = load_market_data()
    config = TradingConfig()
    eval_env = create_eval_env(data, config, render_mode=None)
    eval_results = run_evaluation(eval_env)

    plot_reward_components(eval_results["info_history"], eval_results["steps"])
    plot_raw_rewards(eval_results["info_history"], eval_results["steps"])
    plot_account_metrics(eval_results["steps"],
                         eval_results["balance_history"],
                         eval_results["pnl_history"])


if __name__ == '__main__':
    main()
