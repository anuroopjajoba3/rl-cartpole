# RL CartPole Environment

A reinforcement learning project using OpenAI Gymnasium's CartPole-v1 environment to understand how agents learn from feedback loops.

## What it does

Implements two approaches to the classic pole-balancing problem:

1. **Random Policy Baseline** — agent takes random actions, average ~20 steps survived
2. **Q-Learning Agent** — agent learns from rewards over 300 episodes using epsilon-greedy exploration and Q-table updates

## Results

| Policy | Avg Reward |
|--------|-----------|
| Random baseline | ~20 steps |
| Q-Learning (300 episodes) | ~27 steps |

The Q-learning agent consistently outperforms the random baseline by learning which actions keep the pole upright longer.

## Core RL Concepts Demonstrated

- **Environment**: CartPole-v1 (continuous state space, 2 discrete actions)
- **Agent**: Q-learning with discretized state space
- **Reward**: +1 for every step the pole stays upright
- **Exploration vs Exploitation**: epsilon-greedy strategy with decay
- **Q-table update**: `Q(s,a) += alpha * (reward + gamma * max(Q(s')) - Q(s,a))`

## Stack

- Python 3.11
- Gymnasium 1.2.3
- NumPy

## Run it

```bash
pip install gymnasium numpy
python cartpole.py
```

## Next Steps

- Implement Deep Q-Network (DQN) with PyTorch to handle continuous state space
- Add reward plotting with matplotlib
- Experiment with different hyperparameters (alpha, gamma, epsilon decay)