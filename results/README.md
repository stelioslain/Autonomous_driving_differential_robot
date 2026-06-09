# Results

This folder contains the main experimental results reported in the diploma thesis.

The most important quantitative result is the evaluation of the fine-tuned PPO policy in the MuJoCo simulation environment. The policy was tested for 100 episodes with random starting positions and random initial orientations.

The real-world results are also summarized qualitatively. In the real robot experiments, both the left-hand-rule algorithm and the PPO policy successfully completed the cardboard maze in assisted trials. The main limitations were related to physical motion execution, including wheel friction, actuator dead-zone, turning overshoot, and occasional contact with the cardboard walls.