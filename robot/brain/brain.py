"""
Robot Brain — Deep Q-Network (DQN) — Pure NumPy Implementation
================================================================
Neural network that learns to map sensor inputs to optimal actions
through reinforcement learning. Uses a multi-layer perceptron with
experience replay and target network for stable training.

No PyTorch dependency — runs anywhere with NumPy.
"""

import numpy as np
import json
import os
import logging

from robot.utils.security import (
    validate_path,
    safe_json_load,
    safe_json_save,
    log_safe_error,
)

logger = logging.getLogger(__name__)


class Layer:
    """A single fully-connected layer with ReLU activation."""

    def __init__(self, input_size: int, output_size: int, dropout: float = 0.0):
        # He initialization
        scale = np.sqrt(2.0 / input_size)
        self.W = np.random.randn(input_size, output_size).astype(np.float32) * scale
        self.b = np.zeros(output_size, dtype=np.float32)
        self.dropout = dropout

        # Cache for backprop
        self.input = None
        self.pre_activation = None
        self.mask = None

        # Adam optimizer state
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self.input = x
        self.pre_activation = x @ self.W + self.b
        out = np.maximum(0, self.pre_activation)  # ReLU

        if training and self.dropout > 0:
            self.mask = (np.random.rand(*out.shape) > self.dropout).astype(np.float32)
            out *= self.mask

        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # ReLU gradient
        grad = grad_output.copy()
        grad[self.pre_activation <= 0] = 0

        if self.mask is not None:
            grad *= self.mask

        # Gradients for weights and biases
        batch_size = self.input.shape[0]
        self.dW = (self.input.T @ grad) / batch_size
        self.db = grad.mean(axis=0)

        # Gradient for next layer
        return grad @ self.W.T

    def step(
        self,
        lr: float,
        t: int,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        """Adam optimizer step."""
        # Update weights
        self.mW = beta1 * self.mW + (1 - beta1) * self.dW
        self.vW = beta2 * self.vW + (1 - beta2) * (self.dW**2)
        mW_hat = self.mW / (1 - beta1**t)
        vW_hat = self.vW / (1 - beta2**t)
        self.W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)

        # Update biases
        self.mb = beta1 * self.mb + (1 - beta1) * self.db
        self.vb = beta2 * self.vb + (1 - beta2) * (self.db**2)
        mb_hat = self.mb / (1 - beta1**t)
        vb_hat = self.vb / (1 - beta2**t)
        self.b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)

    def get_params(self) -> dict:
        return {"W": self.W.tolist(), "b": self.b.tolist()}

    def set_params(self, params: dict):
        self.W = np.array(params["W"], dtype=np.float32)
        self.b = np.array(params["b"], dtype=np.float32)


class QNetwork:
    """Deep Q-Network: maps states to action values."""

    def __init__(self, state_size: int, action_size: int, hidden_layers: list = None):
        if hidden_layers is None:
            hidden_layers = [64, 64]

        self.layers = []
        prev_size = state_size
        for i, h in enumerate(hidden_layers):
            dropout = 0.1 if i < len(hidden_layers) - 1 else 0.0
            self.layers.append(Layer(prev_size, h, dropout=dropout))
            prev_size = h

        # Output layer (no activation)
        self.output_layer = Layer(prev_size, action_size)
        self.state_size = state_size
        self.action_size = action_size

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass: state -> Q-values for each action."""
        for layer in self.layers:
            x = layer.forward(x, training=training)
        return self.output_layer.forward(x, training=False)

    def backward(self, grad: np.ndarray):
        """Backward pass: propagate gradient through network."""
        grad = self.output_layer.backward(grad)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def step(self, lr: float, t: int):
        """Update all parameters."""
        for layer in self.layers:
            layer.step(lr, t)
        self.output_layer.step(lr, t)

    def copy_from(self, other: "QNetwork"):
        """Copy parameters from another network."""
        for l1, l2 in zip(self.layers, other.layers):
            l1.W = l2.W.copy()
            l1.b = l2.b.copy()
        self.output_layer.W = other.output_layer.W.copy()
        self.output_layer.b = other.output_layer.b.copy()

    def save(self) -> dict:
        return {
            "layers": [l.get_params() for l in self.layers],
            "output": self.output_layer.get_params(),
        }

    def load(self, data: dict):
        for layer, params in zip(self.layers, data["layers"]):
            layer.set_params(params)
        self.output_layer.set_params(data["output"])


class RobotBrain:
    """
    The robot's learning brain using Double DQN.

    Maintains two networks:
    - policy_net: Used to select actions (the "learning" network)
    - target_net: Used to compute target Q-values (the "stable" network)
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        target_update_freq: int = 10,
        hidden_layers: list = None,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq

        # Create networks
        self.policy_net = QNetwork(state_size, action_size, hidden_layers)
        self.target_net = QNetwork(state_size, action_size, hidden_layers)
        self.target_net.copy_from(self.policy_net)

        # Training tracking
        self.train_step = 0
        self.episode_count = 0
        self.training_history = []

    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_size)

        state_batch = state.reshape(1, -1).astype(np.float32)
        q_values = self.policy_net.forward(state_batch, training=False)
        return int(np.argmax(q_values[0]))

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions."""
        state_batch = state.reshape(1, -1).astype(np.float32)
        q_values = self.policy_net.forward(state_batch, training=False)
        return q_values[0]

    def train(self, batch: list) -> float:
        """
        Train on a batch of experiences using Double DQN.
        Each experience: (state, action, reward, next_state, done)
        """
        if len(batch) == 0:
            return 0.0

        states = np.array([exp[0] for exp in batch], dtype=np.float32)
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch], dtype=np.float32)
        next_states = np.array([exp[3] for exp in batch], dtype=np.float32)
        dones = np.array([exp[4] for exp in batch], dtype=np.float32)

        batch_size = len(batch)

        # Current Q-values
        current_q = self.policy_net.forward(states, training=True)

        # Double DQN targets
        next_q_policy = self.policy_net.forward(next_states, training=False)
        next_q_target = self.target_net.forward(next_states, training=False)
        best_actions = np.argmax(next_q_policy, axis=1)
        best_next_q = next_q_target[np.arange(batch_size), best_actions]
        target_q = rewards + self.gamma * best_next_q * (1 - dones)

        # Compute gradient (MSE loss derivative)
        grad = current_q.copy()
        grad[np.arange(batch_size), actions] -= target_q
        grad *= 2.0 / batch_size

        # Clip gradients
        grad = np.clip(grad, -1.0, 1.0)

        # Backward pass and update
        self.policy_net.backward(grad)
        self.train_step += 1
        self.policy_net.step(self.learning_rate, self.train_step)

        # Update target network periodically
        if self.train_step % self.target_update_freq == 0:
            self.target_net.copy_from(self.policy_net)

        # Return loss for monitoring
        loss = np.mean((current_q[np.arange(batch_size), actions] - target_q) ** 2)
        return float(loss)

    def decay_epsilon(self):
        """Reduce exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def record_episode(self, total_reward: float, steps: int):
        """Record training metrics."""
        self.episode_count += 1
        self.training_history.append(
            {
                "episode": self.episode_count,
                "total_reward": float(total_reward),
                "steps": steps,
                "epsilon": float(self.epsilon),
                "train_step": self.train_step,
            }
        )

    def save(self, path: str):
        """Save brain state to file."""
        data = {
            "policy_net": self.policy_net.save(),
            "target_net": self.target_net.save(),
            "epsilon": self.epsilon,
            "episode_count": self.episode_count,
            "train_step": self.train_step,
            "training_history": self.training_history,
            "config": {
                "state_size": self.state_size,
                "action_size": self.action_size,
                "gamma": self.gamma,
                "epsilon_end": self.epsilon_end,
                "epsilon_decay": self.epsilon_decay,
            },
        }
        safe_json_save(path, data)

    def load(self, path: str):
        """Load brain state from file."""
        data = safe_json_load(
            path,
            expected_keys=[
                "policy_net",
                "target_net",
                "epsilon",
                "episode_count",
                "train_step",
            ],
        )
        self.policy_net.load(data["policy_net"])
        self.target_net.load(data["target_net"])
        self.epsilon = data["epsilon"]
        self.episode_count = data["episode_count"]
        self.train_step = data["train_step"]
        self.training_history = data.get("training_history", [])

    def get_stats(self) -> dict:
        """Get current brain statistics."""
        recent = [h["total_reward"] for h in self.training_history[-50:]]
        return {
            "episode": self.episode_count,
            "train_step": self.train_step,
            "epsilon": round(self.epsilon, 4),
            "avg_reward_last_50": round(float(np.mean(recent)), 2) if recent else 0,
            "max_reward": round(float(max(recent)), 2) if recent else 0,
        }
