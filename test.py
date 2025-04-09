from stable_baselines3 import PPO
from src.env.price_action_env import PriceActionEnv
from stable_baselines3.common.monitor import Monitor

import pandas as pd

def test():
    eth_df = pd.read_csv("./data/eth_usdt/eth_usdt_1h.csv")

    def make_eval_env():
        raw_env = PriceActionEnv([eth_df], window_size=24, render_mode="human")
        return Monitor(raw_env)

    best_model = PPO.load("./logs/checkpoints/best_model", env=make_eval_env())

    obs = best_model.env.reset()
    done = False

    while not done:
        action, _ = best_model.predict(obs, deterministic=True)
        obs, reward, done, info = best_model.env.step(action)
        best_model.env.render()

    best_model.env.envs[0].env.save_trade_log("logs/eval_trades.csv")


if __name__ == '__main__':
    test()
