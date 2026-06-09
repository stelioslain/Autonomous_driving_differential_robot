# q_learning.py
import numpy as np
import math
from collections import defaultdict
from config import N_ACTIONS, ALPHA, GAMMA, EPSILON, EPSILON_MIN, EPSILON_DECAY, MAX_BFS_DISTANCE

# Global Q-table
Q = defaultdict(lambda: np.zeros(N_ACTIONS))

# Global epsilon
epsilon = EPSILON

def discretize_dist(d):
    """Discretize distance measurements for grid-based navigation"""
    if d < 0.5:
        return 0   # Very close to wall (danger)
    elif d < 1.0:
        return 1   # Near wall
    elif d < 1.5:
        return 2   # Medium distance
    else:
        return 3   # Far from wall

def discretize_goal_angle(angle):
    """Discretize goal angle for grid navigation"""
    angle = math.atan2(math.sin(angle), math.cos(angle))
    
    # 5 discrete directions
    if angle < -1.5:
        return 0   # goal far left/behind
    elif angle < -0.3:
        return 1   # goal left
    elif angle < 0.3:
        return 2   # goal front
    elif angle < 1.5:
        return 3   # goal right
    else:
        return 0   # goal far right/behind

def get_state(lidar_distances, goal_angle, robot_x=None, robot_y=None, direction=None):
    """Convert sensor readings to discrete state"""
    # 4-direction lidar: front, left, right, back
    front = discretize_dist(lidar_distances[0])
    left = discretize_dist(lidar_distances[1])
    right = discretize_dist(lidar_distances[2])
    back = discretize_dist(lidar_distances[3])
    goal = discretize_goal_angle(goal_angle)
    
    # Add grid position if provided
    if robot_x is not None and robot_y is not None and direction is not None:
        return (int(robot_x), int(robot_y), direction, front, left, right, back, goal)
    else:
        return (front, left, right, back, goal)

def select_action(state, front_distance=None):
    """Select action with epsilon-greedy policy and safety check"""
    global epsilon
    
    # Safety first: if front is very close to wall, avoid moving forward
    if front_distance is not None and front_distance < 0.3:
        # Choose only turning actions (1: left, 2: right)
        if np.random.rand() < epsilon:
            return np.random.choice([1, 2])
        else:
            # Choose best among turning actions
            q_values = Q[state]
            turning_q = [q_values[1], q_values[2]]
            best_action = 1 if turning_q[0] >= turning_q[1] else 2
            return best_action
    
    # Normal epsilon-greedy
    if np.random.rand() < epsilon:
        return np.random.randint(N_ACTIONS)
    
    # Exploitation: choose best action
    q_values = Q[state]
    max_q = np.max(q_values)
    best_actions = np.where(q_values == max_q)[0]
    return np.random.choice(best_actions)

def compute_reward(collision, reached_goal, step_count, 
                   old_distance, new_distance, action, stuck=False):
    """
    Improved reward function with BFS distance
    
    Args:
        collision: True if robot collided
        reached_goal: True if robot reached goal
        step_count: Current step number
        old_distance: BFS distance to goal before action
        new_distance: BFS distance to goal after action
        action: Action taken (0: forward, 1: left, 2: right, 3: backward)
        stuck: True if robot hasn't made progress in a while
    """
    # Terminal rewards
    if reached_goal:
        return 100.0, True  # Large positive reward, episode done
    
    if collision:
        return -20.0, True  # Negative reward for collision, episode done
    
    # Initialize reward
    reward = 0.0
    
    # Progress-based reward (most important!)
    if old_distance < np.inf and new_distance < np.inf:
        distance_change = old_distance - new_distance  # Positive if closer
        reward += distance_change * 2.0  # Scale the progress reward
        
        # Extra bonus for making significant progress
        if distance_change > 0:
            reward += 1.0
        # Penalty for moving away
        elif distance_change < 0:
            reward -= 1.0
    
    # Action efficiency rewards/penalties
    if action == 0:  # Forward - encouraged
        reward += 0.1
    elif action == 3:  # Backward - discouraged (usually inefficient)
        reward -= 0.2
    
    # Time penalty (encourage efficiency)
    reward -= 0.05
    
    # Stuck penalty
    if stuck:
        reward -= 1.0
    
    # Small penalty for too many steps
    if step_count > 50:
        reward -= 0.01 * (step_count - 50)
    
    return reward, False

def q_learning_update(state, action, reward, next_state, done):
    """Q-learning update rule"""
    current_q = Q[state][action]
    
    if done:
        target = reward
    else:
        next_max_q = np.max(Q[next_state])
        target = reward + GAMMA * next_max_q
    
    # Update Q-value
    Q[state][action] = current_q + ALPHA * (target - current_q)

def decay_epsilon():
    """Decay exploration rate"""
    global epsilon
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

def get_epsilon():
    return epsilon

def set_epsilon(value):
    global epsilon
    epsilon = value

def get_q_table():
    return Q

def set_q_table(new_q):
    global Q
    Q = new_q

def reset_q_table():
    global Q
    Q = defaultdict(lambda: np.zeros(N_ACTIONS))