# evaluate.py
import numpy as np
from environment import PyBulletMazeEnvironment
from q_learning import get_state, get_q_table, set_epsilon
from utils import visualize_maze_progress
from config import MAX_STEPS_PER_EPISODE, ACTIONS

def evaluate_trained_agent(gui=True, num_tests=10):
    """Test the trained agent with discrete actions"""
    print("\n" + "="*60)
    print("EVALUATING TRAINED AGENT (DISCRETE)")
    print("="*60)
    
    env = PyBulletMazeEnvironment(gui=gui)
    Q = get_q_table()
    
    set_epsilon(0.0)  # No exploration
    
    successes = 0
    total_steps = 0
    
    for test in range(num_tests):
        lidar, goal_angle = env.reset()
        grid_x, grid_y, direction = env.get_robot_grid_position()
        state = get_state(lidar, goal_angle, grid_x, grid_y, direction)
        done = False
        test_steps = 0
        
        while not done and test_steps < MAX_STEPS_PER_EPISODE:
            # Choose best action
            q_values = Q[state]
            action = np.argmax(q_values)
            
            # Execute action
            lidar, goal_angle, _ = env.step(action)
            
            # Get new state
            grid_x, grid_y, direction = env.get_robot_grid_position()
            state = get_state(lidar, goal_angle, grid_x, grid_y, direction)
            
            test_steps += 1
            
            if gui:
                visualize_maze_progress(env, test+1, False, test_steps, 0.0)
            
            # Check for completion
            if env.detect_goal():
                successes += 1
                done = True
                print(f"Test {test+1}: Success! Reached exit in {test_steps} steps")
            elif env.collision:
                done = True
                print(f"Test {test+1}: Failed - Collision after {test_steps} steps")
        
        total_steps += test_steps
    
    env.close()
    
    success_rate = (successes / num_tests) * 100
    avg_steps = total_steps / num_tests
    
    print(f"\nFinal Evaluation Results:")
    print(f"Success Rate: {success_rate:.1f}% ({successes}/{num_tests})")
    print(f"Average Steps: {avg_steps:.1f}")
    print(f"Q-table Size: {len(Q)} states")
    
    return success_rate, avg_steps