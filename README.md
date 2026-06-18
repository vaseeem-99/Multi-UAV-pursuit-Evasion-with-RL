# Multi-UAV Pursuit-Evasion with Deep Reinforcement Learning

## Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

### clone pre modeled drones
https://github.com/utiasDSL/gym-pybullet-drones.git

# Clone the project
git clone <repository-url>
cd multi-uav-pursuit-evasion
```

## Testing the Environment

To Verify the environment is learnable (should achieve ~2% capture rate)

```bash
python compare_environments.py
```

**Expected Output**:
```
Testing IMPROVED Environment
Using RANDOM actions to test if evader is catchable...

[50/50] Capture: 2.0% | Avg Min Dist: 6.46 | Avg Reward: -780

✓ SUCCESS! Environment is catchable (2.0% with random actions)
  With trained MAPPO, expect 40-80% capture rate!
```

**Output Files**:
- `environment_test.png` - Visualization of random policy performance

## Training the Policy

### Train MAPPO with Adaptive Environment Generator

```bash
python PPO_AEG.py
```

**Training Configuration**:
- Episodes: 3000
- Episodes per update: 10
- PPO epochs: 10
- Minibatch size: 1024
- Evaluation interval: 500 episodes

**Expected Headless Training Time**:
- GPU (RTX 3090): ~2 hours 

**Training expected Output**:
```
[EP   20] Reward:-3255.7 | Capture: 75.0% | Archive:  5 | MinDist:2.82
[EP   40] Reward:-1499.3 | Capture: 80.0% | Archive:  5 | MinDist:2.69
...
[EP 3000] Reward:  493.3 | Capture:100.0% | Archive:  5 | MinDist:2.20
```

**Output Files**:
- `checkpoints_mappo/mappo_aeg_ep500.pt` (every 500 episodes)
- `checkpoints_mappo/training_curves.png` (training metrics)
- `checkpoints_mappo/evaluation_results.png` (evaluation plots)
- `checkpoints_mappo/training_metrics.json` (detailed logs)

**Monitoring Training**:
- Watch capture rate should increase overtime
- Check collision statistics: should decrease over time
- Monitor minimum distance: should approach 2.5 m capture threshold

## Test Trained Policy (Demo)
Watch the trained MAPPO policy in action

```bash
python demo_final.py
```
**Expected Behavior**:
- change the mode from easy to hard and also, test custom mode to visualize actual drones pursuing the evader 


## Human vs AI Interactive Demo
Play against the trained AI (you control the evader)

```bash
python Demo_with_controller.py
```

**Controls**:
- **Arrow Keys**: Move evader (Up/Down/Left/Right)
- **Q or ESC**: Quit game

## References

- Chen, J., et al. (2025). "Multi-UAV Pursuit-Evasion with Online Planning in Unknown Environments using Deep Reinforcement Learning." *IEEE Robotics and Automation Letters*.

## Contact

**Author**: Hothur Vaseem Ahmed  
**Course**: CS 5392 - Reinforcement Learning  
**Institution**: Texas State University  
**Date**: December 2025