# src/train/training.py
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

from src.utils.data_loader import load_market_data
from src.utils.env_factory import make_train_env, make_eval_env
from src.utils.helper import load_config

# Optionally import custom policies if needed.
from src.policies.custom_policies import CustomLSTMPolicy, CustomCNNPolicy, CustomAttentionPolicy


def train_model():
    """
    Sets up and trains the PPO model using a custom trading environment.
    Uses dynamic policy selection (e.g., MLP, Custom LSTM, CNN, or Attention).
    """
    config = load_config("./config/rl/ppo.yaml")["PPO"]
    data = load_market_data()

    # Wrap environments with DummyVecEnv as required by Stable Baselines3.
    train_env = DummyVecEnv([lambda: make_train_env(data)])
    eval_env = DummyVecEnv([lambda: make_eval_env(data)])

    # Evaluation callback for periodic evaluation and checkpointing.
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/checkpoints/",
        log_path="./logs/evaluation_logs/",
        eval_freq=config["eval_freq"],
        deterministic=True,
        render=False,
    )

    # Select policy based on configuration
    policy_type = config.get("policy_type", "MlpPolicy")
    if policy_type == "CustomLSTMPolicy":
        policy = CustomLSTMPolicy
    elif policy_type == "CustomCNNPolicy":
        policy = CustomCNNPolicy
    elif policy_type == "CustomAttentionPolicy":
        policy = CustomAttentionPolicy
    else:
        # Default to the built-in MLP policy.
        policy = "MlpPolicy"

    model = PPO(
        policy=policy,
        env=train_env,
        verbose=0,
        n_steps=512,
        batch_size=64,
        learning_rate=3e-4,
        tensorboard_log="./logs/tensorboard/SpotTradingEnv_PNL",
    )

    logging.info("Starting model training...")
    model.learn(total_timesteps=config["total_ts"], callback=eval_callback, progress_bar=True)
    logging.info("Training completed. Saving model...")
    model.save("./logs/last_model")
    logging.info("Model saved successfully.")


if __name__ == "__main__":
    train_model()