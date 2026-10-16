"""
Replay Buffer — Experience Memory
===================================
Stores past experiences and samples random batches for training.
This breaks correlation between consecutive experiences and allows
the robot to learn from the same experience multiple times.
"""

import random
import numpy as np
from collections import namedtuple

Experience = namedtuple(
    "Experience", ["state", "action", "reward", "next_state", "done"]
)


class ReplayBuffer:
    """
    Fixed-size experience replay buffer.

    Stores (state, action, reward, next_state, done) tuples and
    samples random mini-batches for training. When full, oldest
    experiences are overwritten (FIFO).
    """

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.total_added = 0

    def push(self, state, action, reward, next_state, done):
        """Store a new experience."""
        experience = Experience(
            state=np.array(state, dtype=np.float32),
            action=action,
            reward=reward,
            next_state=np.array(next_state, dtype=np.float32),
            done=float(done),
        )

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience

        self.position = (self.position + 1) % self.capacity
        self.total_added += 1

    def sample(self, batch_size: int) -> list:
        """Sample a random batch of experiences."""
        batch_size = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough experiences for training."""
        return len(self.buffer) >= batch_size

    def get_stats(self) -> dict:
        """Get buffer statistics."""
        if len(self.buffer) == 0:
            return {
                "size": 0,
                "capacity": self.capacity,
                "fill_pct": 0,
                "avg_reward": 0,
            }

        rewards = [exp.reward for exp in self.buffer]
        return {
            "size": len(self.buffer),
            "capacity": self.capacity,
            "fill_pct": round(len(self.buffer) / self.capacity * 100, 1),
            "avg_reward": round(np.mean(rewards), 3),
            "total_added": self.total_added,
        }
