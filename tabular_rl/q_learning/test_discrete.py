# test_discrete.py
import numpy as np
from environment import PyBulletMazeEnvironment
from q_learning import reset_q_table, get_q_table

def test_discrete_movement():
    """Test the discrete movement system"""
    print("Testing discrete movement...")
    env = PyBulletMazeEnvironment(gui=True)
    
    # Reset Q-table
    reset_q_table()
    
    print("\nTesting basic movements:")
    print("Initial position:", env.robot_x, env.robot_y)
    print("Initial direction:", env.direction)
    
    # Test forward movement
    print("\n1. Moving forward...")
    lidar, goal_angle, success = env.step(0)  # Move forward
    print(f"Success: {success}, New position: {env.robot_x}, {env.robot_y}")
    
    # Test left turn
    print("\n2. Turning left...")
    lidar, goal_angle, success = env.step(1)  # Turn left
    print(f"Success: {success}, New direction: {env.direction}")
    
    # Test right turn
    print("\n3. Turning right...")
    lidar, goal_angle, success = env.step(2)  # Turn right
    print(f"Success: {success}, New direction: {env.direction}")
    
    # Test backward movement
    print("\n4. Moving backward...")
    lidar, goal_angle, success = env.step(3)  # Move backward
    print(f"Success: {success}, New position: {env.robot_x}, {env.robot_y}")
    
    # Test lidar
    print("\n5. Testing lidar...")
    lidar = env.get_lidar()
    print(f"Lidar distances (front, left, right, back): {lidar}")
    
    # Test goal detection
    print("\n6. Testing goal detection...")
    at_goal = env.detect_goal()
    print(f"At goal: {at_goal}")
    
    env.close()
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_discrete_movement()