# train.py - Single environment with visualization
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
#from maze_env_fine_tune import MazeEnv
from maze_env_rand_start_only import MazeEnv

def main():
    # Create logs directory
    os.makedirs("./logs", exist_ok=True)
    
    # Create SINGLE environment WITH visualization
    print("Creating environment with visualization...")
    env = DummyVecEnv([lambda: MazeEnv(render=False) for _ in range(4)])
    
    # Evaluation environment
    eval_env = DummyVecEnv([lambda: Monitor(MazeEnv(render=False))])
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model",
        log_path="./logs/eval",
        eval_freq=10000, # Evaluate every 10k steps
        n_eval_episodes=20, # 20 episodes per evaluation
        deterministic=True,
        render=False
    )
    
    # Create PPO model
    print("Initializing PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./logs",
        verbose=1,
        device='auto'
    )
    
    # Train with visualization
    print("Starting training with visualization...")
    print("A MuJoCo window should open showing the robot.")
    print("Training will be SLOWER but you can watch learning.")
    
    model.learn(total_timesteps=3_000_000, callback=eval_callback)
    
    # Save model
    model.save("ppo_mujoco")
    env.close()
    print("Training complete!")

if __name__ == "__main__":
    main()