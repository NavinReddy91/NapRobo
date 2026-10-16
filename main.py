#!/usr/bin/env python3
"""
Self-Learning AI Robot — Main Entry Point
==========================================
Train a robot to navigate a grid world using deep reinforcement learning.

Usage:
    python main.py                    # Train with defaults
    python main.py --episodes 1000    # Train for 1000 episodes
    python main.py --evaluate         # Evaluate saved model
    python main.py --dashboard        # Launch web dashboard during training
"""

import argparse
import json
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from robot.trainer import TrainingEngine


def main():
    parser = argparse.ArgumentParser(description="Self-Learning AI Robot")
    parser.add_argument("--episodes", type=int, default=500, help="Training episodes")
    parser.add_argument("--grid-size", type=int, default=12, help="Grid world size")
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument(
        "--epsilon-decay", type=float, default=0.995, help="Epsilon decay"
    )
    parser.add_argument(
        "--save-dir", type=str, default="checkpoints", help="Save directory"
    )
    parser.add_argument("--evaluate", action="store_true", help="Evaluate saved model")
    parser.add_argument(
        "--dashboard", action="store_true", help="Export dashboard data"
    )
    parser.add_argument("--quiet", action="store_true", help="Minimal output")

    args = parser.parse_args()

    print("=" * 60)
    print("  SELF-LEARNING AI ROBOT")
    print("  Deep Reinforcement Learning System")
    print("=" * 60)
    print()

    # Create training engine
    engine = TrainingEngine(
        grid_size=args.grid_size,
        max_episodes=args.episodes,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon_decay=args.epsilon_decay,
        save_dir=args.save_dir,
    )

    if args.evaluate:
        # Evaluate mode
        print("Loading trained model...")
        model_path = os.path.join(args.save_dir, "brain_final.json")
        if not os.path.exists(model_path):
            print(f"No saved model found at {model_path}")
            print("Train first with: python main.py --episodes 500")
            return

        engine.brain.load(model_path)
        print(f"Model loaded. Running evaluation...")
        results = engine.evaluate(num_episodes=20)

        print(f"\n{'=' * 40}")
        print(f"  EVALUATION RESULTS")
        print(f"{'=' * 40}")
        print(f"  Episodes:     {results['episodes']}")
        print(f"  Avg Reward:   {results['avg_reward']}")
        print(f"  Success Rate: {results['success_rate']}%")
        print(f"  Avg Steps:    {results['avg_steps']}")
        print(f"  Goals:        {results['goals']}/{results['episodes']}")
        return

    # Training mode
    print(f"Configuration:")
    print(f"  Grid Size:     {args.grid_size}x{args.grid_size}")
    print(f"  Episodes:      {args.episodes}")
    print(f"  Batch Size:    {args.batch_size}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Gamma:         {args.gamma}")
    print(f"  Hidden Layers:   [128, 128, 64]")
    print()
    print("Starting training...")
    print("-" * 60)

    # Optional: export data for dashboard
    dashboard_data = []

    def track_progress(result):
        dashboard_data.append(result)

    callback = track_progress if args.dashboard else None

    # Train
    summary = engine.train(verbose=not args.quiet, callback=callback)

    # Print summary
    print()
    print("=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Total Episodes:   {summary['total_episodes']}")
    print(f"  Training Time:    {summary['total_time']}s")
    print(f"  Best Reward:      {summary['best_reward']}")
    print(f"  Final Avg Reward: {summary['final_avg_reward']}")
    print(f"  Success Rate:     {summary['success_rate']}%")
    print(f"  Goals Reached:    {summary['goals_reached']}")
    print(f"  Model Saved:      {args.save_dir}/brain_final.json")
    print()

    # Evaluate
    print("Running evaluation (no exploration)...")
    eval_results = engine.evaluate(num_episodes=20)
    print(f"  Eval Avg Reward:  {eval_results['avg_reward']}")
    print(f"  Eval Success:     {eval_results['success_rate']}%")
    print()

    # Save dashboard data
    if args.dashboard:
        dashboard_file = os.path.join(args.save_dir, "dashboard_data.json")
        with open(dashboard_file, "w") as f:
            json.dump(
                {
                    "training_log": dashboard_data,
                    "summary": summary,
                    "evaluation": eval_results,
                    "live_stats": engine.get_live_stats(),
                },
                f,
                indent=2,
            )
        print(f"Dashboard data saved: {dashboard_file}")

    print("Done! Use --evaluate to test the trained model.")


if __name__ == "__main__":
    main()
