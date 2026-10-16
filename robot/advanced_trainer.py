"""
Advanced Training Engine with Knowledge & Online Learning
==========================================================
Enhanced training that integrates:
- Knowledge base for pattern storage
- Online learning capabilities
- Curriculum learning
- Advanced metrics and analytics
"""

import json
import os
import time
import numpy as np

from robot.brain import RobotBrain
from robot.body import Robot
from robot.environment import GridWorld
from robot.memory import ReplayBuffer
from robot.knowledge import KnowledgeBase
from robot.learning import OnlineLearner


class AdvancedTrainer:
    """
    Advanced trainer with knowledge integration and online learning.
    """

    def __init__(
        self,
        grid_size: int = 10,
        max_episodes: int = 1000,
        max_steps: int = 150,
        batch_size: int = 64,
        memory_size: int = 20000,
        learning_rate: float = 0.003,
        gamma: float = 0.97,
        epsilon_decay: float = 0.997,
        train_frequency: int = 2,
        save_dir: str = "checkpoints",
        enable_knowledge: bool = True,
        enable_online_learning: bool = True,
    ):
        # Environment
        self.env = GridWorld(
            width=grid_size,
            height=grid_size,
            max_steps=max_steps,
        )

        # Robot
        self.robot = Robot(name="NapRobo-Advanced", state_size=10, action_size=5)
        self.robot.attach_environment(self.env)

        # Brain
        self.brain = RobotBrain(
            state_size=10,
            action_size=5,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon_decay=epsilon_decay,
            hidden_layers=[128, 64, 32],
        )

        # Memory
        self.memory = ReplayBuffer(capacity=memory_size)

        # Knowledge Base
        self.enable_knowledge = enable_knowledge
        if enable_knowledge:
            self.knowledge = KnowledgeBase(
                filepath=os.path.join(save_dir, "knowledge_base.json")
            )
        else:
            self.knowledge = None

        # Online Learner
        self.enable_online_learning = enable_online_learning
        if enable_online_learning:
            self.online_learner = OnlineLearner(
                cache_file=os.path.join(save_dir, "online_cache.json")
            )
            # Pre-load some knowledge
            self._preload_knowledge()
        else:
            self.online_learner = None

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
        self.goals_per_phase = []

    def _preload_knowledge(self):
        """Pre-load knowledge from online sources."""
        print("Loading initial knowledge from online sources...")

        # Search for relevant topics
        topics = [
            "reinforcement learning",
            "robot navigation",
            "neural networks",
        ]

        for topic in topics:
            results = self.online_learner.search(topic, max_results=2)
            print(f"  - Found {len(results)} resources on '{topic}'")

        # Store some initial strategies
        self.knowledge.learn_strategy(
            name="Goal Seeking",
            description="Move towards goal using direction vector",
            effectiveness=0.8,
            context={"situation": "goal_visible"},
        )

        self.knowledge.learn_strategy(
            name="Obstacle Avoidance",
            description="Use proximity sensors to avoid walls",
            effectiveness=0.9,
            context={"situation": "obstacle_near"},
        )

        self.knowledge.store_heuristic(
            name="Exploration Bonus",
            rule="Prefer unexplored areas when stuck",
            confidence=0.7,
        )

        print(
            f"  Knowledge base initialized with {self.knowledge.get_knowledge_summary()['patterns_count']} patterns"
        )

    def train_episode(self) -> dict:
        """Run a single training episode with knowledge integration."""
        state = self.robot.reset()
        total_reward = 0
        steps = 0
        done = False
        visited_states = []

        while not done:
            # Use knowledge to inform action selection (if available)
            action = self.brain.select_action(state, explore=True)

            # Optionally boost action with knowledge
            if self.knowledge and np.random.random() < 0.1:  # 10% chance
                similar = self.knowledge.search_patterns(state, top_k=3)
                if similar and similar[0]["success"]:
                    # Bias towards successful past actions
                    if np.random.random() < 0.5:
                        action = similar[0]["action"]

            next_state, reward, done, info = self.robot.act(action)

            # Store experience
            self.memory.push(state, action, reward, next_state, done)
            visited_states.append(state.copy())

            # Learn pattern if successful
            if self.knowledge and reward > 0:
                self.knowledge.learn_pattern(state, action, reward, success=True)

            # Train brain
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

        # Decay exploration
        self.brain.decay_epsilon()

        # Record metrics
        self.brain.record_episode(total_reward, steps)
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(steps)

        if total_reward > self.best_reward:
            self.best_reward = total_reward

        # Track goals
        goal_reached = info.get("event") == "goal"
        if goal_reached:
            self.robot.record_goal(total_reward)

            # Learn successful strategy
            if self.knowledge:
                self.knowledge.learn_strategy(
                    name=f"Success_Ep{self.brain.episode_count}",
                    description=f"Reached goal in {steps} steps with reward {total_reward:.1f}",
                    effectiveness=min(1.0, total_reward / 50.0),
                    context={"steps": steps, "reward": total_reward},
                )

        return {
            "episode": self.brain.episode_count,
            "reward": round(total_reward, 1),
            "steps": steps,
            "epsilon": round(self.brain.epsilon, 4),
            "event": info.get("event"),
            "memory_size": len(self.memory),
            "goal_reached": goal_reached,
        }

    def train(self, verbose: bool = True, callback=None) -> dict:
        """Run full training loop with advanced features."""
        start_time = time.time()

        print(f"\n{'=' * 60}")
        print(f"  ADVANCED TRAINING - NapRobo")
        print(f"{'=' * 60}")
        print(f"  Episodes: {self.max_episodes}")
        print(f"  Knowledge Base: {'Enabled' if self.enable_knowledge else 'Disabled'}")
        print(
            f"  Online Learning: {'Enabled' if self.enable_online_learning else 'Disabled'}"
        )
        print(f"{'=' * 60}\n")

        phase_goals = 0
        phase_episodes = 0

        for ep in range(1, self.max_episodes + 1):
            result = self.train_episode()
            phase_episodes += 1

            if result["goal_reached"]:
                phase_goals += 1

            if callback:
                callback(result)

            # Phase reporting
            if verbose and ep % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_length = np.mean(self.episode_lengths[-100:])
                phase_success = (phase_goals / phase_episodes) * 100

                print(
                    f"Episode {ep:4d} | "
                    f"Avg Reward: {avg_reward:7.1f} | "
                    f"Avg Steps: {avg_length:5.1f} | "
                    f"Epsilon: {self.brain.epsilon:.3f} | "
                    f"Phase Success: {phase_success:5.1f}% | "
                    f"Memory: {len(self.memory)}"
                )

                self.goals_per_phase.append(phase_goals)
                phase_goals = 0
                phase_episodes = 0

        elapsed = time.time() - start_time

        # Save everything
        os.makedirs(self.save_dir, exist_ok=True)
        self.brain.save(os.path.join(self.save_dir, "brain_advanced.json"))
        self._save_metrics()

        # Generate comprehensive summary
        summary = {
            "total_episodes": self.max_episodes,
            "total_time": round(elapsed, 1),
            "best_reward": round(self.best_reward, 1),
            "final_avg_reward": round(np.mean(self.episode_rewards[-100:]), 1),
            "final_avg_steps": round(np.mean(self.episode_lengths[-100:]), 1),
            "goals_reached": self.robot.goals_reached,
            "success_rate": round(
                self.robot.goals_reached / self.max_episodes * 100, 1
            ),
            "brain_stats": self.brain.get_stats(),
            "robot_stats": self.robot.get_stats(),
            "memory_stats": self.memory.get_stats(),
            "knowledge_stats": self.knowledge.get_knowledge_summary()
            if self.knowledge
            else None,
            "online_stats": self.online_learner.get_learning_summary()
            if self.online_learner
            else None,
        }

        # Save summary
        with open(os.path.join(self.save_dir, "training_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        return summary

    def evaluate(self, num_episodes: int = 50) -> dict:
        """Evaluate the trained brain."""
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
        """Save training metrics."""
        metrics = {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "losses": self.losses[-2000:],
            "training_history": self.brain.training_history,
            "goals_per_phase": self.goals_per_phase,
        }
        with open(os.path.join(self.save_dir, "advanced_metrics.json"), "w") as f:
            json.dump(metrics, f)

    def export_knowledge_for_ai(self) -> dict:
        """Export all knowledge in AI-readable format."""
        export = {
            "brain_stats": self.brain.get_stats(),
            "robot_stats": self.robot.get_stats(),
            "training_summary": {
                "episodes": self.max_episodes,
                "success_rate": self.robot.goals_reached / self.max_episodes * 100,
                "avg_reward": np.mean(self.episode_rewards[-100:]),
            },
        }

        if self.knowledge:
            export["knowledge"] = self.knowledge.export_for_ai()

        if self.online_learner:
            export["online_learning"] = self.online_learner.get_learning_summary()

        return export
