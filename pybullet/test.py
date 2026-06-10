import numpy as np
from stable_baselines3 import PPO
from maze_env import MazeEnv

def test_model(model_path="ppo_maze_visual", num_episodes=10):
    # Load environment
    env = MazeEnv(render=True)
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    
    success_count = 0
    total_steps = 0
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        
        print(f"\n=== Episode {episode + 1}/{num_episodes} ===")
        
        while not done:
            # Get action from policy
            action, _states = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, _ = env.step(action)
            
            done = terminated or truncated
            episode_reward += reward
            episode_steps += 1
            
            # Optional: Slow down for visualization
            # import time
            # time.sleep(0.01)
        
        # Check if goal was reached
        if terminated and env._distance_to_goal() < 0.3:
            success_count += 1
            print(f"  SUCCESS! Reward: {episode_reward:.2f}, Steps: {episode_steps}")
        else:
            print(f"  Failed. Reward: {episode_reward:.2f}, Steps: {episode_steps}")
        
        total_steps += episode_steps
    
    # Print statistics
    print("\n" + "="*50)
    print(f"Testing Results ({num_episodes} episodes):")
    print(f"  Success Rate: {success_count}/{num_episodes} ({100*success_count/num_episodes:.1f}%)")
    print(f"  Average Steps: {total_steps/num_episodes:.1f}")
    print("="*50)
    
    env.close()

def interactive_test(model_path="ppo_maze_visual"):
    """Run a single episode with manual control option"""
    import time
    
    env = MazeEnv(render=True)
    model = PPO.load(model_path)
    
    obs, _ = env.reset()
    done = False
    episode_reward = 0
    
    print("\n=== Interactive Test ===")
    print("Press 'm' for model control, 'q' to quit")
    
    while not done:
        # Get action from model
        action, _states = model.predict(obs, deterministic=True)
        
        # Step environment
        obs, reward, terminated, truncated, _ = env.step(action)
        
        done = terminated or truncated
        episode_reward += reward
        
        # Slow down for visualization
        time.sleep(0.02)
    
    # Show result
    if terminated and env._distance_to_goal() < 0.3:
        print(f"\nSUCCESS! Total reward: {episode_reward:.2f}")
    else:
        print(f"\nFailed. Total reward: {episode_reward:.2f}")
    
    env.close()

if __name__ == "__main__":
    # Choose which test to run
    test_option = input("Choose test mode:\n1. Batch test (10 episodes)\n2. Interactive test\nEnter 1 or 2: ")
    
    if test_option == "1":
        test_model()
    else:
        interactive_test()
