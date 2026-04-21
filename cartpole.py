"""
CartPole-v1 RL Environment
Implements random policy baseline and simple Q-learning agent
using OpenAI Gymnasium to understand feedback-driven learning loops.
"""

import gymnasium as gym
import numpy as np
import random
from collections import defaultdict


def random_policy_baseline(episodes=10):
    """Baseline: random actions, measures average survival steps."""
    env = gym.make("CartPole-v1")
    rewards = []

    for episode in range(episodes):
        observation, info = env.reset()
        total_reward = 0

        for step in range(500):
            action = env.action_space.sample()  # random left or right
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        rewards.append(total_reward)
        print(f"Episode {episode + 1:>2}: Steps survived = {int(total_reward)}")

    env.close()
    print(f"\nRandom Policy — Avg reward: {np.mean(rewards):.1f} over {episodes} episodes")
    return rewards


def discretize(observation, bins=10):
    """Discretize continuous state space for Q-table lookup."""
    cart_pos = int(np.clip(observation[0] / 0.5, -1, 1) * bins // 2 + bins // 2)
    cart_vel = int(np.clip(observation[1] / 2.0, -1, 1) * bins // 2 + bins // 2)
    pole_ang = int(np.clip(observation[2] / 0.2, -1, 1) * bins // 2 + bins // 2)
    pole_vel = int(np.clip(observation[3] / 2.0, -1, 1) * bins // 2 + bins // 2)
    return (cart_pos, cart_vel, pole_ang, pole_vel)


def q_learning_agent(episodes=300, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995):
    """
    Q-learning agent that learns to balance the pole.
    - alpha: learning rate
    - gamma: discount factor (how much future rewards matter)
    - epsilon: exploration rate (starts high, decays over time)
    """
    env = gym.make("CartPole-v1")
    q_table = defaultdict(lambda: [0.0, 0.0])  # state -> [left_value, right_value]
    rewards = []

    for episode in range(episodes):
        observation, info = env.reset()
        state = discretize(observation)
        total_reward = 0

        for step in range(500):
            # Epsilon-greedy: explore randomly or exploit learned Q-values
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q_table[state]))

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = discretize(next_obs)
            total_reward += reward

            # Q-learning update: learn from reward + estimated future value
            best_next = max(q_table[next_state])
            q_table[state][action] += alpha * (
                reward + gamma * best_next - q_table[state][action]
            )

            state = next_state
            if terminated or truncated:
                break

        epsilon = max(0.01, epsilon * epsilon_decay)
        rewards.append(total_reward)

        if (episode + 1) % 50 == 0:
            avg = np.mean(rewards[-50:])
            print(f"Episode {episode + 1:>3}: Avg reward (last 50) = {avg:.1f}  |  epsilon = {epsilon:.3f}")

    env.close()
    print(f"\nQ-Learning Agent — Final avg reward: {np.mean(rewards[-50:]):.1f}")
    return rewards


if __name__ == "__main__":
    print("=" * 50)
    print("RANDOM POLICY BASELINE")
    print("=" * 50)
    random_policy_baseline(episodes=10)

    print("\n" + "=" * 50)
    print("Q-LEARNING AGENT (300 episodes)")
    print("=" * 50)
    q_learning_agent(episodes=300)