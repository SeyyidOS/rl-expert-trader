import os
import datetime
import logging
from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback

from src.utils.env_factory import make_train_env, make_eval_env
from src.utils.data_loader import load_market_data
from src.env.custom_vec_env import CustomVecEnv
from src.utils.helper import load_config

# Optionally import custom policies if needed.
from src.policies.custom_policies import CustomLSTMPolicy, CustomCNNPolicy, CustomAttentionPolicy

import torch as th


def train_model():
    """
    Sets up and trains the PPO model using a custom trading environment.
    Uses dynamic policy selection (e.g., MLP, Custom LSTM, CNN, or Attention) and saves
    the model and logs under a unique run directory.

    Modified to use SubprocVecEnv with 4 parallel training environments.
    """
    algo = "PPO"
    cfg = "SpotTrading-CustomLSTM-PNL"

    config = load_config("./config/rl/ppo.yaml")[algo]
    data = load_market_data()

    # Create a unique run directory using a timestamp.
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(f"./logs/tensorboard/{cfg}", f"{algo}_run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "trade_logs"), exist_ok=True)

    # Create a subdirectory for saving the best model related to this run.
    model_dir = os.path.join(run_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    # Optionally, create a separate folder for evaluation logs.
    eval_logs_dir = os.path.join(run_dir, "evaluation_logs")
    os.makedirs(eval_logs_dir, exist_ok=True)

    # Use SubprocVecEnv for training with 4 parallel environments.
    num_train_envs = 8
    train_env = CustomVecEnv(
        [lambda: make_train_env(data, run_dir, True, False) for _ in range(num_train_envs)]
    )

    num_eval_envs = 2
    # You may use a single environment for evaluation.
    eval_env = CustomVecEnv([lambda: make_eval_env(data, run_dir, True, False)])

    # Evaluation callback for periodic evaluation and checkpointing.
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=eval_logs_dir,
        eval_freq=config["eval_freq"],
        deterministic=True,
        render=False,
    )

    # Select policy based on configuration.
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

    device = 'cuda' if th.cuda.is_available() else 'cpu'
    model = PPO(
        policy=policy,
        env=train_env,
        verbose=0,
        n_steps=512,
        batch_size=64,
        learning_rate=3e-4,
        tensorboard_log=run_dir,  # Log everything under the run-specific directory.
        device=device
    )

    try:
        logging.info("Starting model training...")
        logging.info(f"Using device: {next(model.policy.parameters()).device}")
        model.learn(total_timesteps=config["total_ts"], callback=eval_callback, progress_bar=True)
        logging.info("Training completed. Saving model...")

        # Save the latest model under the run-specific folder.
        model.save(os.path.join(model_dir, "last_model"))
        logging.info("Model saved successfully.")

    except KeyboardInterrupt as e:
        logging.info(f"Exception occurred during model training. {e}")
        model.save(os.path.join(model_dir, "last_model"))
        logging.info("Model saved successfully.")


if __name__ == "__main__":
    train_model()
