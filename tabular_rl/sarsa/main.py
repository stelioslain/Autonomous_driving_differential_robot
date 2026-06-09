# main.py
import numpy as np
import random
from train import train_maze_navigation
from evaluate import evaluate_trained_agent
from utils import load_q_table, save_q_table, plot_training_results, analyze_q_table

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    print("=" * 60)
    print("PYBULLET DISCRETE MAZE NAVIGATION WITH Q-LEARNING")
    print("=" * 60)
    
    print("\nSelect mode:")
    print("1. Train new model (with GUI)")
    print("2. Train new model (headless - faster)")
    print("3. Load and evaluate existing model")
    print("4. Train and evaluate")
    print("5. Analyze existing Q-table")
    print("6. Quick test (10 episodes with GUI)")
    
    choice = input("Enter choice (1-6): ").strip()
    
    if choice == "1":
        print("\nTraining with GUI (slower but visual)...")
        rewards, successes, steps, bfs = train_maze_navigation(gui=True)
        save_q_table("discrete_maze_q_table.pkl")
        plot_training_results(rewards, successes, steps, bfs)
        success_rate, avg_steps = evaluate_trained_agent(gui=True, num_tests=5)
        analyze_q_table()
        
    elif choice == "2":
        print("\nTraining headless (faster)...")
        rewards, successes, steps, bfs = train_maze_navigation(gui=False)
        save_q_table("discrete_maze_q_table.pkl")
        plot_training_results(rewards, successes, steps, bfs)
        success_rate, avg_steps = evaluate_trained_agent(gui=True, num_tests=5)
        analyze_q_table()
        
    elif choice == "3":
        print("\nLoading existing model...")
        if load_q_table("discrete_maze_q_table.pkl"):
            success_rate, avg_steps = evaluate_trained_agent(gui=True, num_tests=10)
            analyze_q_table()
        else:
            print("No saved model found. Training new model...")
            rewards, successes, steps, bfs = train_maze_navigation(gui=False)
            save_q_table("discrete_maze_q_table.pkl")
            plot_training_results(rewards, successes, steps, bfs)
            success_rate, avg_steps = evaluate_trained_agent(gui=True, num_tests=5)
            analyze_q_table()
            
    elif choice == "4":
        print("\nTraining and evaluating...")
        rewards, successes, steps, bfs = train_maze_navigation(gui=False)
        save_q_table("discrete_maze_q_table.pkl")
        plot_training_results(rewards, successes, steps, bfs)
        success_rate, avg_steps = evaluate_trained_agent(gui=True, num_tests=10)
        analyze_q_table()
        
    elif choice == "5":
        print("\nAnalyzing Q-table...")
        if load_q_table("discrete_maze_q_table.pkl"):
            analyze_q_table()
            success_rate, avg_steps = evaluate_trained_agent(gui=True, num_tests=3)
        else:
            print("No saved model found.")
            choice = input("Train new model? (y/n): ").strip().lower()
            if choice == 'y':
                rewards, successes, steps, bfs = train_maze_navigation(gui=False)
                save_q_table("discrete_maze_q_table.pkl")
                plot_training_results(rewards, successes, steps, bfs)
                analyze_q_table()
                
    elif choice == "6":
        print("\nQuick test...")
        if load_q_table("discrete_maze_q_table.pkl"):
            success_rate, avg_steps = evaluate_trained_agent(gui=True, num_tests=1)
        else:
            print("No saved model found. Training briefly...")
            # Quick training with reduced episodes
            from config import N_EPISODES
            original_episodes = N_EPISODES
            import config
            config.N_EPISODES = 100  # Quick training
            
            rewards, successes, steps, bfs = train_maze_navigation(gui=True)
            save_q_table("discrete_maze_q_table_quick.pkl")
            config.N_EPISODES = original_episodes
            
            success_rate, avg_steps = evaluate_trained_agent(gui=True, num_tests=1)
            
    else:
        print("\nInvalid choice. Running default training...")
        rewards, successes, steps, bfs = train_maze_navigation(gui=True)
        save_q_table("discrete_maze_q_table.pkl")
        plot_training_results(rewards, successes, steps, bfs)
        success_rate, avg_steps = evaluate_trained_agent(gui=True, num_tests=5)
        analyze_q_table()
    
    print("\n" + "=" * 60)
    print("PROGRAM COMPLETE")
    print("=" * 60)
    print(f"Final success rate: {success_rate:.1f}%")
    print(f"Average steps to goal: {avg_steps:.1f}")
    print("\nFiles saved:")
    print("1. 'discrete_maze_q_table.pkl' - Trained Q-table")
    print("2. 'discrete_training_results.png' - Learning curves")
    
    # Cleanup
    try:
        import pybullet as p
        p.disconnect()
    except:
        pass

if __name__ == "__main__":
    main()