"""
Robot Body
===========
Abstraction layer for the robot's physical interface.
In simulation, this wraps the environment. For real hardware,
this would interface with ROS 2 and physical sensors/actuators.
"""

import numpy as np


class Robot:
    """
    Robot body abstraction.

    Provides a unified interface for:
    - Reading sensors (get_state)
    - Executing actions (take_action)
    - Getting body status

    Currently backed by simulation, but designed for hardware swap.
    """

    def __init__(
        self, name: str = "Atlas-1", state_size: int = 10, action_size: int = 5
    ):
        self.name = name
        self.state_size = state_size
        self.action_size = action_size
        self.environment = None
        self.total_episodes = 0
        self.total_steps = 0
        self.goals_reached = 0
        self.total_reward_all_time = 0

    def attach_environment(self, env):
        """Connect robot to an environment (simulation or real)."""
        self.environment = env

    def perceive(self) -> np.ndarray:
        """Read sensors and return current state."""
        if self.environment is None:
            raise RuntimeError("Robot not attached to any environment")
        return self.environment.get_state()

    def act(self, action: int) -> tuple:
        """Execute an action and return (state, reward, done, info)."""
        if self.environment is None:
            raise RuntimeError("Robot not attached to any environment")
        self.total_steps += 1
        return self.environment.step(action)

    def reset(self) -> np.ndarray:
        """Reset for a new episode."""
        if self.environment is None:
            raise RuntimeError("Robot not attached to any environment")
        self.total_episodes += 1
        return self.environment.reset()

    def record_goal(self, reward: float):
        """Record a goal reached."""
        self.goals_reached += 1
        self.total_reward_all_time += reward

    def get_stats(self) -> dict:
        """Get robot statistics."""
        success_rate = (self.goals_reached / max(self.total_episodes, 1)) * 100
        return {
            "name": self.name,
            "total_episodes": self.total_episodes,
            "total_steps": self.total_steps,
            "goals_reached": self.goals_reached,
            "success_rate": round(success_rate, 1),
            "avg_reward": round(
                self.total_reward_all_time / max(self.total_episodes, 1), 1
            ),
        }
