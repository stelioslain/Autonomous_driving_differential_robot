# utils.py
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p

def save_q_table(filename="discrete_maze_q_table.pkl"):
    """Save Q-table to file"""
    from sarsa import get_q_table
    Q = get_q_table()
    # Convert defaultdict to regular dict for saving
    q_dict = dict(Q)
    with open(filename, "wb") as f:
        pickle.dump(q_dict, f)
    print(f"Q-table saved to {filename} ({len(Q)} states)")

def load_q_table(filename="discrete_maze_q_table.pkl"):
    """Load Q-table from file"""
    from sarsa import set_q_table
    from collections import defaultdict
    import numpy as np
    
    try:
        with open(filename, "rb") as f:
            q_dict = pickle.load(f)
            # Create defaultdict with zeros initialization
            from config import N_ACTIONS
            Q = defaultdict(lambda: np.zeros(N_ACTIONS), q_dict)
            set_q_table(Q)
        print(f"Q-table loaded from {filename} ({len(Q)} states)")
        return True
    except FileNotFoundError:
        print(f"No saved Q-table found at {filename}")
        return False
    except Exception as e:
        print(f"Error loading Q-table: {e}")
        return False

def visualize_maze_progress(env, episode, success, steps, epsilon):
    """Visualize the maze and robot's path for discrete movement"""
    # Draw the robot's path
    if len(env.path) > 1:
        for i in range(len(env.path) - 1):
            p.addUserDebugLine(
                [env.path[i][0], env.path[i][1], 0.05],
                [env.path[i+1][0], env.path[i+1][1], 0.05],
                [1, 0.5, 0],  # Orange color for path
                lineWidth=3,
                lifeTime=5.0
            )
    
    # Draw goal cell
    goal_x, goal_y = env.goal_pos[0], env.goal_pos[1]
    p.addUserDebugLine(
        [goal_x - 0.4, goal_y - 0.4, 0.05],
        [goal_x + 0.4, goal_y - 0.4, 0.05],
        [0, 1, 0],  # Green
        lineWidth=3,
        lifeTime=5.0
    )
    p.addUserDebugLine(
        [goal_x + 0.4, goal_y - 0.4, 0.05],
        [goal_x + 0.4, goal_y + 0.4, 0.05],
        [0, 1, 0],
        lineWidth=3,
        lifeTime=5.0
    )
    p.addUserDebugLine(
        [goal_x + 0.4, goal_y + 0.4, 0.05],
        [goal_x - 0.4, goal_y + 0.4, 0.05],
        [0, 1, 0],
        lineWidth=3,
        lifeTime=5.0
    )
    p.addUserDebugLine(
        [goal_x - 0.4, goal_y + 0.4, 0.05],
        [goal_x - 0.4, goal_y - 0.4, 0.05],
        [0, 1, 0],
        lineWidth=3,
        lifeTime=5.0
    )
    
    # Display episode info
    success_text = "SUCCESS" if success else "RUNNING"
    status_color = [0, 1, 0] if success else [1, 1, 0]
    
    p.addUserDebugText(
        f"Episode: {episode} | Status: {success_text} | Steps: {steps}",
        [1, 18, 2],
        textColorRGB=[1, 1, 1],
        textSize=1.2,
        lifeTime=5.0
    )
    
    p.addUserDebugText(
        f"Epsilon: {epsilon:.3f} | Position: ({int(env.robot_x)}x{int(env.robot_y)})",
        [1, 16, 2],
        textColorRGB=[1, 1, 1],
        textSize=1.2,
        lifeTime=5.0
    )

def plot_training_results(rewards, successes, steps, bfs_distances=None, filename="discrete_training_results.png"):
    """Plot training statistics"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    window = 20
    
    # Plot 1: Success Rate
    if len(successes) > window:
        success_rate_smooth = np.convolve(successes, np.ones(window)/window, mode='valid') * 100
        axes[0, 0].plot(range(window-1, len(successes)), success_rate_smooth, 'g-', linewidth=2)
    else:
        axes[0, 0].plot(range(len(successes)), np.array(successes)*100, 'g-', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Success Rate (%)')
    axes[0, 0].set_title('Maze Navigation Success Rate')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 100)
    
    # Plot 2: Rewards
    if len(rewards) > window:
        reward_smooth = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[0, 1].plot(range(window-1, len(rewards)), reward_smooth, 'b-', linewidth=2)
    else:
        axes[0, 1].plot(range(len(rewards)), rewards, 'b-', linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Average Reward')
    axes[0, 1].set_title('Learning Progress')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Steps per episode
    if len(steps) > window:
        steps_smooth = np.convolve(steps, np.ones(window)/window, mode='valid')
        axes[1, 0].plot(range(window-1, len(steps)), steps_smooth, 'r-', linewidth=2)
    else:
        axes[1, 0].plot(range(len(steps)), steps, 'r-', linewidth=2)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Steps to Completion')
    axes[1, 0].set_title('Efficiency Improvement')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: BFS distances if provided
    if bfs_distances is not None and len(bfs_distances) > 0:
        # Convert inf to max finite value for plotting
        finite_distances = [d if d < float('inf') else 100 for d in bfs_distances]
        if len(finite_distances) > window:
            bfs_smooth = np.convolve(finite_distances, np.ones(window)/window, mode='valid')
            axes[1, 1].plot(range(window-1, len(finite_distances)), bfs_smooth, 'purple', linewidth=2)
        else:
            axes[1, 1].plot(range(len(finite_distances)), finite_distances, 'purple', linewidth=2)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('BFS Distance to Goal')
        axes[1, 1].set_title('Path Optimality')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        # Q-table size estimation
        axes[1, 1].plot(range(len(rewards)), np.arange(len(rewards)) * 0.5, 'purple', linewidth=2)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Estimated States')
        axes[1, 1].set_title('Q-table State Space Growth')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Training plot saved to {filename}")
    plt.show()

def analyze_q_table():
    """Analyze and print Q-table statistics"""
    from sarsa import get_q_table
    Q = get_q_table()
    
    print("\n" + "="*60)
    print("Q-TABLE ANALYSIS")
    print("="*60)
    print(f"Total states: {len(Q)}")
    
    if len(Q) > 0:
        # Calculate average Q-values
        all_q_values = []
        for state in Q:
            all_q_values.extend(Q[state])
        
        print(f"Average Q-value: {np.mean(all_q_values):.4f}")
        print(f"Std Q-value: {np.std(all_q_values):.4f}")
        print(f"Min Q-value: {np.min(all_q_values):.4f}")
        print(f"Max Q-value: {np.max(all_q_values):.4f}")
        
        # Count optimal actions
        optimal_count = 0
        for state in Q:
            if np.max(Q[state]) > 0:
                optimal_count += 1
        
        print(f"States with optimal actions: {optimal_count}/{len(Q)} ({optimal_count/len(Q)*100:.1f}%)")
        
        # Sample some states
        print("\nSample states and their Q-values:")
        sample_states = list(Q.keys())[:5]
        for i, state in enumerate(sample_states):
            print(f"State {i+1}: {state}")
            print(f"  Q-values: {Q[state]}")
            print(f"  Best action: {np.argmax(Q[state])} (value: {np.max(Q[state]):.3f})")

def visualize_best_path(env, Q):
    """Visualize the best path found by the agent"""
    env.reset()
    lidar, goal_angle = env.get_lidar(), env.get_goal_angle()
    grid_x, grid_y, direction = env.get_robot_grid_position()
    state = get_state(lidar, goal_angle, grid_x, grid_y, direction)
    
    done = False
    max_steps = 100
    path_positions = []
    
    while not done and max_steps > 0:
        # Get best action
        action = np.argmax(Q[state])
        
        # Execute action
        env.step(action)
        
        # Record position
        path_positions.append((env.robot_x, env.robot_y))
        
        # Check for completion
        if env.detect_goal():
            done = True
            print("Goal reached!")
            break
        
        # Get new state
        lidar, goal_angle = env.get_lidar(), env.get_goal_angle()
        grid_x, grid_y, direction = env.get_robot_grid_position()
        state = get_state(lidar, goal_angle, grid_x, grid_y, direction)
        
        max_steps -= 1
    
    # Draw the path
    for i in range(len(path_positions) - 1):
        p.addUserDebugLine(
            [path_positions[i][0], path_positions[i][1], 0.1],
            [path_positions[i+1][0], path_positions[i+1][1], 0.1],
            [0, 1, 1],  # Cyan for best path
            lineWidth=4,
            lifeTime=10.0
        )
    
    return path_positions