"""
Grid World Environment
=======================
A 2D grid environment where the robot learns to navigate from
a start position to a goal while avoiding obstacles.

The environment provides:
- State observations (sensor readings)
- Rewards for good/bad behavior
- Episode reset for repeated training
"""

import numpy as np
import random


class GridWorld:
    """
    2D grid environment for robot navigation.

    Grid cells:
      0 = empty
      1 = wall/obstacle
      2 = goal
      3 = hazard (extra penalty)
      4 = unexplored (for bonus reward)

    The robot receives a state vector containing:
    - Its normalized position (x, y)
    - Distance readings in 4 directions (up, right, down, left)
    - Direction to goal (dx, dy normalized)
    """

    # Actions
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    SCAN = 4  # Reveal nearby unexplored cells

    ACTION_NAMES = ["UP", "RIGHT", "DOWN", "LEFT", "SCAN"]
    ACTION_DELTAS = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]

    # Rewards
    REWARD_GOAL = 50.0
    REWARD_STEP = -0.5
    REWARD_WALL = -5.0
    REWARD_HAZARD = -10.0
    REWARD_EXPLORE = 2.0

    def __init__(
        self,
        width: int = 12,
        height: int = 12,
        obstacle_density: float = 0.2,
        hazard_count: int = 3,
        max_steps: int = 200,
        seed: int = None,
    ):
        self.width = width
        self.height = height
        self.obstacle_density = obstacle_density
        self.hazard_count = hazard_count
        self.max_steps = max_steps

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.grid = None
        self.robot_pos = None
        self.goal_pos = None
        self.explored = None
        self.step_count = 0
        self.episode_reward = 0

        # Generate initial environment
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset environment for a new episode. Returns initial state."""
        # Create empty grid
        self.grid = np.zeros((self.height, self.width), dtype=np.int32)
        self.explored = np.zeros((self.height, self.width), dtype=bool)

        # Place obstacles
        for y in range(self.height):
            for x in range(self.width):
                if random.random() < self.obstacle_density:
                    self.grid[y, x] = 1

        # Place hazards
        hazard_placed = 0
        attempts = 0
        while hazard_placed < self.hazard_count and attempts < 100:
            hx, hy = (
                random.randint(0, self.width - 1),
                random.randint(0, self.height - 1),
            )
            if self.grid[hy, hx] == 0:
                self.grid[hy, hx] = 3
                hazard_placed += 1
            attempts += 1

        # Place robot and goal at opposite corners (with some randomness)
        self.robot_pos = (
            random.randint(0, self.width // 3),
            random.randint(0, self.height // 3),
        )
        self.goal_pos = (
            random.randint(self.width * 2 // 3, self.width - 1),
            random.randint(self.height * 2 // 3, self.height - 1),
        )

        # Ensure start and goal are clear
        self.grid[self.robot_pos[1], self.robot_pos[0]] = 0
        self.grid[self.goal_pos[1], self.goal_pos[0]] = 2

        # Mark starting area as explored
        self._update_explored()

        self.step_count = 0
        self.episode_reward = 0

        return self.get_state()

    def _update_explored(self):
        """Mark cells near robot as explored."""
        rx, ry = self.robot_pos
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                nx, ny = rx + dx, ry + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    self.explored[ny, nx] = True

    def _get_distance(self, dx: int, dy: int) -> float:
        """
        Get distance to nearest obstacle/wall in a direction.
        Returns normalized distance (0 = adjacent wall, 1 = no wall seen).
        """
        rx, ry = self.robot_pos
        max_dist = max(self.width, self.height)
        dist = 0

        for i in range(1, max_dist + 1):
            nx, ny = rx + dx * i, ry + dy * i
            if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                dist = i - 1
                break
            if self.grid[ny, nx] == 1:
                dist = i - 1
                break
        else:
            dist = max_dist

        return dist / max_dist  # Normalize to [0, 1]

    def get_state(self) -> np.ndarray:
        """
        Get current state as a feature vector.

        State vector (10 values):
        [0] robot_x / width         (normalized position)
        [1] robot_y / height
        [2] distance UP             (proximity sensor)
        [3] distance RIGHT
        [4] distance DOWN
        [5] distance LEFT
        [6] (goal_x - robot_x) / width   (direction to goal)
        [7] (goal_y - robot_y) / height
        [8] explored_pct            (how much of the map is explored)
        [9] steps_remaining / max_steps
        """
        rx, ry = self.robot_pos
        gx, gy = self.goal_pos

        explored_pct = np.sum(self.explored) / (self.width * self.height)

        state = np.array(
            [
                rx / max(self.width - 1, 1),
                ry / max(self.height - 1, 1),
                self._get_distance(0, -1),  # UP
                self._get_distance(1, 0),  # RIGHT
                self._get_distance(0, 1),  # DOWN
                self._get_distance(-1, 0),  # LEFT
                (gx - rx) / max(self.width - 1, 1),
                (gy - ry) / max(self.height - 1, 1),
                explored_pct,
                1.0 - (self.step_count / self.max_steps),
            ],
            dtype=np.float32,
        )

        return state

    def step(self, action: int) -> tuple:
        """
        Execute an action and return (state, reward, done, info).
        """
        self.step_count += 1
        rx, ry = self.robot_pos
        dx, dy = self.ACTION_DELTAS[action]
        nx, ny = rx + dx, ry + dy

        reward = self.REWARD_STEP  # Small penalty per step
        done = False
        info = {"action": self.ACTION_NAMES[action], "event": None}

        if action == self.SCAN:
            # Reveal a larger area around robot
            for sdy in range(-3, 4):
                for sdx in range(-3, 4):
                    sx, sy = rx + sdx, ry + sdy
                    if 0 <= sx < self.width and 0 <= sy < self.height:
                        was_explored = self.explored[sy, sx]
                        self.explored[sy, sx] = True
                        if not was_explored and self.grid[sy, sx] == 4:
                            reward += self.REWARD_EXPLORE
            info["event"] = "scan"

        elif nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
            # Hit boundary
            reward = self.REWARD_WALL
            info["event"] = "wall"

        elif self.grid[ny, nx] == 1:
            # Hit obstacle
            reward = self.REWARD_WALL
            info["event"] = "obstacle"

        elif self.grid[ny, nx] == 3:
            # Hit hazard
            reward = self.REWARD_HAZARD
            info["event"] = "hazard"
            self.robot_pos = (nx, ny)

        elif self.grid[ny, nx] == 2:
            # Reached goal!
            reward = self.REWARD_GOAL
            done = True
            self.robot_pos = (nx, ny)
            info["event"] = "goal"

        else:
            # Valid move to empty cell
            self.robot_pos = (nx, ny)

            # Bonus for exploring new territory
            self._update_explored()

        # Time limit
        if self.step_count >= self.max_steps:
            done = True
            info["event"] = "timeout"

        self.episode_reward += reward
        state = self.get_state()

        return state, reward, done, info

    def get_grid_display(self) -> list:
        """Get grid as a 2D list for visualization."""
        display = self.grid.copy()
        display[self.robot_pos[1], self.robot_pos[0]] = 5  # Robot marker
        return display.tolist()

    def get_explored_display(self) -> list:
        """Get explored map as a 2D list for visualization."""
        return (self.explored.astype(int)).tolist()

    def get_stats(self) -> dict:
        """Get environment statistics."""
        explored_pct = np.sum(self.explored) / (self.width * self.height) * 100
        dist_to_goal = abs(self.robot_pos[0] - self.goal_pos[0]) + abs(
            self.robot_pos[1] - self.goal_pos[1]
        )
        return {
            "robot_pos": self.robot_pos,
            "goal_pos": self.goal_pos,
            "step": self.step_count,
            "max_steps": self.max_steps,
            "explored_pct": round(explored_pct, 1),
            "manhattan_to_goal": dist_to_goal,
            "episode_reward": round(self.episode_reward, 1),
        }
