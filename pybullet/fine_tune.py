from stable_baselines3 import PPO
from maze_env import MazeEnv

env = MazeEnv(render=True)

model = PPO.load(
    "ppo_maze_visual.zip",   # your trained model
    env=env
)

model.learn(
    total_timesteps=200_000,   # start small
    reset_num_timesteps=False  # IMPORTANT
)

model.save("ppo_maze_fine_tuned.zip")