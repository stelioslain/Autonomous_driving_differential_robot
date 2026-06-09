# Real-World Evaluation Results

The real-world experiments were performed using the physical Raspberry Pi--Arduino differential mobile robot inside the cardboard maze.

Two navigation methods were tested:

1. Left-hand-rule algorithm
2. Fine-tuned PPO policy

## Left-Hand-Rule Algorithm

Approximate number of trials: 3  
Successful assisted runs: 1  
Fully autonomous repeatable completion: No  

Main limitations:
- Wheel friction
- Turning overshoot
- Occasional contact with cardboard walls
- Sensitivity to stop-and-turn maneuvers

## PPO Policy

Approximate number of trials: 3  
Successful assisted runs: 1  
Fully autonomous repeatable completion: No  

Main limitations:
- Wheel friction
- Actuator dead-zone
- Imperfect low-speed motion execution
- Occasional motion errors caused by real-world dynamics

## Observations

The PPO policy produced smoother motion than the left-hand-rule algorithm because it generated continuous control actions. The left-hand-rule algorithm required more stop-and-turn maneuvers, which made it more sensitive to turning overshoot.

The RPLidar C1 sensor was stable during testing, and the RF2O laser odometry yaw estimation was reliable. The main limitations were not related to perception, but to physical actuation and motion execution.