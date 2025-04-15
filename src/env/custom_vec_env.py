from stable_baselines3.common.vec_env import SubprocVecEnv
from typing import Optional, Callable

import gymnasium as gym


class CustomVecEnv(SubprocVecEnv):
    def __init__(self, env_fns: list[Callable[[], gym.Env]], start_method: Optional[str] = None):
        super().__init__(env_fns, start_method)
        for i in range(len(self.remotes)):
            self.env_method("set_process_id", data=i, indices=i)