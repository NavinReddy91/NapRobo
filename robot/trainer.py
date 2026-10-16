"""
Training Engine
================
Orchestrates the learning loop:
1. Robot interacts with environment
2. Experiences stored in memory
3. Brain trains on sampled experiences
4. Metrics tracked and exported
"""

import json
import os
import time
import numpy as np

from robot.brain import RobotBrain
from robot.body import Robot
from robot.environment import GridWorld
from robot.memory import ReplayBuffer


class TrainingEngine:
    """
    Main training loop for the self-learning robot.

    Manages the cycle of:
    - Environment interaction
    - Experience collection
    - Neural network training
    - Metrics tracking
    """

    def __init__(
        self,
        grid_size: int = 12,
        max_episodes: int = 500,
        max_steps: int = 200,
        batch_size: int = 64,
        memory_size: int = 10000,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_decay: float = 0.995,
        train_frequency: int = 2,
        save_dir: str = "checkpoints",
    ):
        # Environment
        self.env = GridWorld(
            width=grid_size,
            height=grid_size,
            max_steps=max_steps,
        )

        # Robot
        self.robot = Robot(name="Atlas-1", state_size=10, action_size=5)
        self.robot.attach_environment(self.env)

        # Brain
        self.brain = RobotBrain(
            state_size=10,
            action_size=5,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon_decay=epsilon_decay,
            hidden_layers=[64, 64],
        )

        # Memory
        self.memory = ReplayBuffer(capacity=memory_size)

        # Training parameters
        self.max_episodes = max_episodes
        self.batch_size = batch_size
        self.train_frequency = train_frequency
        self.save_dir = save_dir

        # Metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.best_reward = -float("inf")

    def train_episode(self) -> dict:
        """Run a single training episode."""
        state = self.robot.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done:
            # Robot perceives and acts
            action = self.brain.select_action(state, explore=True)
            next_state, reward, done, info = self.robot.act(action)

            # Store experience
            self.memory.push(state, action, reward, next_state, done)

            # Train brain periodically
            if (
                self.memory.is_ready(self.batch_size)
                and steps % self.train_frequency == 0
            ):
                batch = self.memory.sample(self.batch_size)
                loss = self.brain.train(batch)
                self.losses.append(loss)

            state = next_state
            total_reward += reward
            steps += 1

        # Decay exploration rate
        self.brain.decay_epsilon()

        # Record metrics
        self.brain.record_episode(total_reward, steps)
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(steps)

        if total_reward > self.best_reward:
            self.best_reward = total_reward

        # Track goals
        if info.get("event") == "goal":
            self.robot.record_goal(total_reward)

        return {
            "episode": self.brain.episode_count,
            "reward": round(total_reward, 1),
            "steps": steps,
            "epsilon": round(self.brain.epsilon, 4),
            "event": info.get("event"),
            "memory_size": len(self.memory),
        }

    def train(self, verbose: bool = True, callback=None) -> dict:
        """
        Run full training loop.

        Args:
            verbose: Print progress every N episodes
            callback: Optional function called each episode with metrics

        Returns:
            Training summary dictionary
        """
        start_time = time.time()

        for ep in range(1, self.max_episodes + 1):
            result = self.train_episode()

            if callback:
                callback(result)

            if verbose and (ep % 50 == 0 or ep == 1):
                avg_reward = np.mean(self.episode_rewards[-50:])
                avg_length = np.mean(self.episode_lengths[-50:])
                success = sum(1 for r in self.episode_rewards[-50:] if r > 50)
                print(
                    f"Episode {ep:4d} | "
                    f"Avg Reward: {avg_reward:7.1f} | "
                    f"Avg Steps: {avg_length:5.1f} | "
                    f"Epsilon: {self.brain.epsilon:.3f} | "
                    f"Success: {success}/50 | "
                    f"Memory: {len(self.memory)}"
                )

        elapsed = time.time() - start_time

        # Save final model
        os.makedirs(self.save_dir, exist_ok=True)
        self.brain.save(os.path.join(self.save_dir, "brain_final.json"))
        self._save_metrics()

        summary = {
            "total_episodes": self.max_episodes,
            "total_time": round(elapsed, 1),
            "best_reward": round(self.best_reward, 1),
            "final_avg_reward": round(np.mean(self.episode_rewards[-50:]), 1),
            "final_avg_steps": round(np.mean(self.episode_lengths[-50:]), 1),
            "goals_reached": self.robot.goals_reached,
            "success_rate": round(
                self.robot.goals_reached / self.max_episodes * 100, 1
            ),
            "brain_stats": self.brain.get_stats(),
            "robot_stats": self.robot.get_stats(),
            "memory_stats": self.memory.get_stats(),
        }

        return summary

    def evaluate(self, num_episodes: int = 20) -> dict:
        """Evaluate the trained brain without exploration."""
        rewards = []
        steps_list = []
        goals = 0

        for _ in range(num_episodes):
            state = self.robot.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.brain.select_action(state, explore=False)
                state, reward, done, info = self.robot.act(action)
                total_reward += reward

            rewards.append(total_reward)
            steps_list.append(self.env.step_count)
            if info.get("event") == "goal":
                goals += 1

        return {
            "episodes": num_episodes,
            "avg_reward": round(np.mean(rewards), 1),
            "std_reward": round(np.std(rewards), 1),
            "avg_steps": round(np.mean(steps_list), 1),
            "goals": goals,
            "success_rate": round(goals / num_episodes * 100, 1),
        }

    def _save_metrics(self):
        """Save training metrics to JSON."""
        metrics = {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "losses": self.losses[-1000:],  # Keep last 1000
            "training_history": self.brain.training_history,
        }
        with open(os.path.join(self.save_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f)

    def get_live_stats(self) -> dict:
        """Get current training stats for live dashboard."""
        recent = self.episode_rewards[-50:] if self.episode_rewards else [0]
        return {
            "brain": self.brain.get_stats(),
            "robot": self.robot.get_stats(),
            "memory": self.memory.get_stats(),
            "environment": self.env.get_stats(),
            "training": {
                "avg_reward_50": round(np.mean(recent), 1),
                "max_reward": round(max(recent), 1) if recent else 0,
                "best_ever": round(self.best_reward, 1),
                "total_episodes": len(self.episode_rewards),
            },
        }
