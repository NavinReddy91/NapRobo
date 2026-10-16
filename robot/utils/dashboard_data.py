"""
Dashboard Data Generator
=========================
Generates training data and exports it for the web dashboard.
Run this to create a visualization of the robot's learning process.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot.trainer import TrainingEngine


def generate_dashboard_data(episodes: int = 300):
    """Run training and generate data for the web dashboard."""
    print(f"Training robot for {episodes} episodes...")

    engine = TrainingEngine(
        grid_size=12,
        max_episodes=episodes,
        batch_size=64,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_decay=0.995,
        save_dir="checkpoints",
    )

    training_log = []
    grid_snapshots = []

    def on_episode(result):
        training_log.append(result)
        # Save grid snapshots at key episodes
        if result["episode"] % 50 == 0 or result["episode"] <= 5:
            grid_snapshots.append(
                {
                    "episode": result["episode"],
                    "grid": engine.env.get_grid_display(),
                    "explored": engine.env.get_explored_display(),
                    "robot_pos": engine.env.robot_pos,
                    "goal_pos": engine.env.goal_pos,
                }
            )

    summary = engine.train(verbose=True, callback=on_episode)
    eval_results = engine.evaluate(num_episodes=20)

    # Build complete dashboard dataset
    data = {
        "training_log": training_log,
        "grid_snapshots": grid_snapshots,
        "summary": summary,
        "evaluation": eval_results,
        "config": {
            "grid_size": 12,
            "episodes": episodes,
            "actions": ["UP", "RIGHT", "DOWN", "LEFT", "SCAN"],
            "rewards": {
                "goal": 100,
                "step": -1,
                "wall": -10,
                "hazard": -25,
                "explore": 5,
            },
        },
    }

    os.makedirs("checkpoints", exist_ok=True)
    with open("checkpoints/dashboard_data.json", "w") as f:
        json.dump(data, f)

    print(f"\nDashboard data saved to checkpoints/dashboard_data.json")
    print(f"Open dashboard/index.html to visualize the results.")

    return data


if __name__ == "__main__":
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    generate_dashboard_data(episodes)
