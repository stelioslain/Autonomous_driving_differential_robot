import numpy as np
import torch
from stable_baselines3 import PPO
from maze_env_rand_start_only import MazeEnv

####### TO TEST THE TRAINED MODEL CHANGE THE DIRECTORY BELOW TO "../models/ppo_mujoco.zip", ELSE IF YOU WANT TO TEST THE FINE-TUNED MODEL LEAVE IT AS IS #######
def test_model(model_path="../models/ppo_mujoco_fine_tuned.zip", num_episodes=100):
    # Load environment
    env = MazeEnv(render=False)
    # env.start = np.array([1.0, 1.0])
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path, device="cpu")
    
    policy = model.policy
    policy.eval()
    
    # Quantize ONLY the policy (actor) network
    actor = policy.mlp_extractor.policy_net
    
    quantized_actor = torch.quantization.quantize_dynamic(
        actor,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    
    policy.mlp_extractor.policy_net = quantized_actor
    
    success_count = 0
    total_steps = 0
    total_reward = 0.0
    min_reward = 70.0
    max_reward = 50.0
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        
        print(f"\n=== Episode {episode + 1}/{num_episodes} ===")
        
        while not done:
            # Get action from policy
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, _ = env.step(action)
            
            done = terminated or truncated
            episode_reward += reward
            episode_steps += 1
            
        
        # Check if goal was reached
        if terminated and env._distance_to_goal() < 1.0:
            success_count += 1
            print(f"  SUCCESS! Reward: {episode_reward:.2f}, Steps: {episode_steps}")
            if (episode_reward < min_reward):
                min_reward = episode_reward
            if (episode_reward > max_reward):
                max_reward = episode_reward
        else:
            print(f"  Failed. Reward: {episode_reward:.2f}, Steps: {episode_steps}")
        
        total_steps += episode_steps
        total_reward += episode_reward
    
    # Print statistics
    print("\n" + "="*50)
    print(f"Testing Results ({num_episodes} episodes):")
    print(f"  Success Rate: {success_count}/{num_episodes} ({100*success_count/num_episodes:.1f}%)")
    print(f"  Average Steps: {total_steps/num_episodes:.1f}")
    print(f"  Average Reward: {total_reward/num_episodes:.2f}")
    print(f"  Minimum Reward: {min_reward:.2f}")
    print(f"  Maximum Reward: {max_reward:.2f}")
    print("="*50)
    
    env.close()

if __name__ == "__main__":
    # Run the test
    test_model()