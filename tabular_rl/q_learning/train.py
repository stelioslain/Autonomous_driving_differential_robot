# train.py
import numpy as np
from environment import PyBulletMazeEnvironment
from q_learning import (
    get_state, select_action, compute_reward, 
    q_learning_update, decay_epsilon, get_epsilon,
    get_q_table, set_epsilon
)
from utils import save_q_table, visualize_maze_progress
from config import N_EPISODES, MAX_STEPS_PER_EPISODE, ACTIONS

def train_maze_navigation(gui=True):
    """Main training function for maze navigation with discrete actions"""
    print("=" * 60)
    print("PYBULLET MAZE NAVIGATION TRAINING (DISCRETE)")
    print("=" * 60)
    
    env = PyBulletMazeEnvironment(gui=gui)
    
    print("\nMaze Configuration:")
    print(f"Size: {env.maze_width} x {env.maze_height} meters")
    print(f"Start: ({env.robot_x:.1f}, {env.robot_y:.1f})")
    print(f"Goal: ({env.goal_pos[0]:.1f}, {env.goal_pos[1]:.1f})")
    print(f"Cell size: {env.cell_size} meters")
    
    episode_rewards = []
    success_history = []
    steps_history = []
    bfs_distances = []  # Track BFS distances
    
    set_epsilon(0.5)
    
    print("\nStarting training...")
    for episode in range(N_EPISODES):
        lidar, goal_angle = env.reset()
        grid_x, grid_y, direction = env.get_robot_grid_position()
        state = get_state(lidar, goal_angle, grid_x, grid_y, direction)
        episode_reward = 0
        done = False
        success = False
        
        # Get initial BFS distance
        old_bfs_distance = env.get_bfs_distance_to_goal()
        no_progress_steps = 0  # Count steps without progress
        last_best_distance = old_bfs_distance
        
        for step in range(MAX_STEPS_PER_EPISODE):
            # Select action
            action = select_action(state, lidar[0])
            
            # Execute action and get new state
            lidar, goal_angle, action_success = env.step(action)
            grid_x, grid_y, direction = env.get_robot_grid_position()
            next_state = get_state(lidar, goal_angle, grid_x, grid_y, direction)
            
            # Check terminal conditions
            reached_goal = env.detect_goal()
            collision = env.collision
            
            # Get new BFS distance
            new_bfs_distance = env.get_bfs_distance_to_goal()
            
            # Check if stuck (no progress for too long)
            stuck = False
            if new_bfs_distance < last_best_distance:
                last_best_distance = new_bfs_distance
                no_progress_steps = 0
            else:
                no_progress_steps += 1
                if no_progress_steps > 10:  # Stuck if no progress for 10 steps
                    stuck = True
            
            # Compute reward with BFS distance
            reward, done = compute_reward(
                collision=collision,
                reached_goal=reached_goal,
                step_count=step,
                old_distance=old_bfs_distance,
                new_distance=new_bfs_distance,
                action=action,
                stuck=stuck
            )
            
            episode_reward += reward
            
            # Q-learning update
            q_learning_update(state, action, reward, next_state, done)
            state = next_state
            old_bfs_distance = new_bfs_distance
            
            # Visualization
            if gui and episode % 10 == 0 and step % 5 == 0:
                visualize_maze_progress(env, episode, success, step, get_epsilon())
            
            # Check if episode should end
            if done or reached_goal or collision:
                success = reached_goal
                if reached_goal:
                    print(f"Episode {episode+1}: Reached goal in {step+1} steps!")
                elif collision:
                    print(f"Episode {episode+1}: Collision after {step+1} steps")
                break
        
        # Handle timeout (max steps reached)
        if not done and step == MAX_STEPS_PER_EPISODE - 1:
            success = False
            print(f"Episode {episode+1}: Timeout after {MAX_STEPS_PER_EPISODE} steps")
        
        # Decay epsilon
        decay_epsilon()
        
        # Record episode stats
        episode_rewards.append(episode_reward)
        success_history.append(1 if success else 0)
        steps_history.append(step + 1)
        bfs_distances.append(old_bfs_distance)  # Final distance
        
        # Print progress
        if (episode + 1) % 10 == 0:
            recent_success = np.mean(success_history[-10:]) * 100 if len(success_history) >= 10 else 0
            recent_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else 0
            recent_steps = np.mean(steps_history[-10:]) if len(steps_history) >= 10 else 0
            q_table = get_q_table()
            
            print(f"Episode {episode + 1:4d} | "
                  f"Success: {recent_success:5.1f}% | "
                  f"Avg Reward: {recent_reward:6.2f} | "
                  f"Avg Steps: {recent_steps:5.1f} | "
                  f"Epsilon: {get_epsilon():.3f} | "
                  f"States: {len(q_table):5d}")
    
    print("\nTraining completed!")
    
    # Print summary statistics
    success_rate = np.mean(success_history) * 100
    avg_steps = np.mean(steps_history)
    avg_bfs_distance = np.mean([d for d in bfs_distances if d < float('inf')])
    
    print(f"\nSummary:")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Average Steps: {avg_steps:.1f}")
    print(f"Average BFS Distance: {avg_bfs_distance:.1f}")
    print(f"Final Q-table size: {len(get_q_table())} states")
    
    env.close()
    return episode_rewards, success_history, steps_history, bfs_distances

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Q-learning agent for maze navigation")
    parser.add_argument("--gui", action="store_true", help="Show PyBullet GUI during training")
    parser.add_argument("--headless", action="store_true", help="Run without GUI (faster)")
    args = parser.parse_args()
    
    gui = args.gui and not args.headless
    rewards, successes, steps, bfs_distances = train_maze_navigation(gui=gui)
    
    save_q_table("discrete_maze_q_table.pkl")
    print(f"\nTraining complete! Model saved to 'discrete_maze_q_table.pkl'")