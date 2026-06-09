# train.py - Single environment with visualization
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from maze_env import MazeEnv

def main():
    # Create logs directory
    os.makedirs("./logs", exist_ok=True)
    
    # Create SINGLE environment WITH visualization
    print("Creating environment with visualization...")
    env = MazeEnv(render=True)  # Single env with GUI
    
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
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./logs",
        verbose=1,
        device='auto'
    )
    
    # Train with visualization
    print("Starting training with visualization...")
    print("A PyBullet window should open showing the robot.")
    print("Training will be SLOWER but you can watch learning.")
    
    model.learn(total_timesteps=600_000)  # Start with 100k steps
    
    # Save model
    model.save("ppo_maze_visual")
    env.close()
    print("Training complete!")

if __name__ == "__main__":
    main()