# Trained Models

This folder contains the trained PPO models used in the diploma thesis.

## Included Models

```text
ppo_mujoco.zip
ppo_mujoco_fine_tuned.zip
```

## Model Description

The first model, `ppo_mujoco.zip`, was obtained after the initial PPO training phase in the MuJoCo maze environment.

The second model, `ppo_mujoco_fine_tuned.zip`, was obtained after fine-tuning the best model from the initial training phase.

## Training Information

Initial training:

- Algorithm: PPO
- Policy: MLP Policy
- Environment: MuJoCo maze environment
- Training timesteps: 3,000,000
- Parallel environments: 4

Fine-tuning:

- Starting model: Best model from the initial training phase
- Additional timesteps: 1,500,000
- Evaluation frequency: Every 10,000 timesteps
- Evaluation episodes: 20

## Final Evaluation Result

The fine-tuned PPO policy was evaluated for 100 simulation episodes with random starting positions and random initial orientations.

Results:

- Success rate: 96 / 100 episodes
- Success rate percentage: 96.0%
- Average steps: 170.2
- Average reward: 53.88

## Testing the Models

The MuJoCo testing script is located in:

```text
mujoco/test.py
```

If the script is executed from inside the `mujoco/` folder, the fine-tuned model can be tested using:

```bash
python test.py --model_path ../models/ppo_mujoco_fine_tuned.zip --episodes 100
```

The initially trained model can be tested using:

```bash
python test.py --model_path ../models/ppo_mujoco.zip --episodes 100
```

The default model path inside `test.py` is:

```text
../models/ppo_mujoco_fine_tuned.zip
```

If the models are stored in a different location, the `--model_path` argument must be changed accordingly.

## Loading a Model Manually

The models can also be loaded manually using Stable-Baselines3:

```python
from stable_baselines3 import PPO

model = PPO.load("models/ppo_mujoco_fine_tuned.zip")
```

The model must be used with an environment compatible with the observation and action spaces defined in the MuJoCo environment code.