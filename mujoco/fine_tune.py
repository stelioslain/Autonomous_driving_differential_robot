from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
#from maze_env_fine_tune import MazeEnv
from maze_env_rand_start_only import MazeEnv
import os

os.makedirs("./logs_finetune", exist_ok=True)

# Training env (4 parallel)
env = DummyVecEnv([lambda: MazeEnv(render=False) for _ in range(4)])

# Evaluation env (single, monitored)
eval_env = DummyVecEnv([lambda: Monitor(MazeEnv(render=False))])

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs_finetune/best_model",
    log_path="./logs_finetune/eval",
    eval_freq=10000,
    n_eval_episodes=20,
    deterministic=True,
    render=False
)

model = PPO.load(
    "./logs/best_model/best_model.zip",
    env=env
)

# Reduce entropy
model.ent_coef = 0.0005

model.learn(
    total_timesteps=1_500_000,
    reset_num_timesteps=False,
    callback=eval_callback
)

model.save("ppo_mujoco_fine_tuned.zip")