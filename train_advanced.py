#!/usr/bin/env python3
"""
NapRobo — Advanced Self-Learning AI Robot
==========================================
Train an advanced robot with:
- Deep Q-Network learning
- Knowledge base integration
- Online learning capabilities
- Pattern recognition and storage
- Strategy learning

Usage:
    python train_advanced.py                    # Train with defaults
    python train_advanced.py --episodes 2000    # Train for 2000 episodes
    python train_advanced.py --no-knowledge     # Disable knowledge base
    python train_advanced.py --export           # Export knowledge for AI
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from robot.advanced_trainer import AdvancedTrainer


def main():
    parser = argparse.ArgumentParser(
        description="NapRobo - Advanced Self-Learning AI Robot"
    )
    parser.add_argument("--episodes", type=int, default=1000, help="Training episodes")
    parser.add_argument("--grid-size", type=int, default=10, help="Grid world size")
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.003, help="Learning rate"
    )
    parser.add_argument("--gamma", type=float, default=0.97, help="Discount factor")
    parser.add_argument(
        "--epsilon-decay", type=float, default=0.997, help="Epsilon decay"
    )
    parser.add_argument(
        "--save-dir", type=str, default="checkpoints", help="Save directory"
    )
    parser.add_argument(
        "--no-knowledge", action="store_true", help="Disable knowledge base"
    )
    parser.add_argument(
        "--no-online", action="store_true", help="Disable online learning"
    )
    parser.add_argument("--evaluate", action="store_true", help="Evaluate saved model")
    parser.add_argument("--export", action="store_true", help="Export knowledge for AI")

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  NapRobo — Advanced Self-Learning AI Robot")
    print("  With Knowledge Base & Online Learning")
    print("=" * 60 + "\n")

    # Create advanced trainer
    trainer = AdvancedTrainer(
        grid_size=args.grid_size,
        max_episodes=args.episodes,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon_decay=args.epsilon_decay,
        save_dir=args.save_dir,
        enable_knowledge=not args.no_knowledge,
        enable_online_learning=not args.no_online,
    )

    if args.evaluate:
        # Evaluate mode
        print("Loading trained model...")
        model_path = os.path.join(args.save_dir, "brain_advanced.json")
        if not os.path.exists(model_path):
            print(f"No saved model found at {model_path}")
            print("Train first with: python train_advanced.py --episodes 1000")
            return

        trainer.brain.load(model_path)
        print(f"Model loaded. Running evaluation...")
        results = trainer.evaluate(num_episodes=50)

        print(f"\n{'=' * 50}")
        print(f"  EVALUATION RESULTS")
        print(f"{'=' * 50}")
        print(f"  Episodes:     {results['episodes']}")
        print(f"  Avg Reward:   {results['avg_reward']} ± {results['std_reward']}")
        print(f"  Success Rate: {results['success_rate']}%")
        print(f"  Avg Steps:    {results['avg_steps']}")
        print(f"  Goals:        {results['goals']}/{results['episodes']}")
        return

    if args.export:
        # Export knowledge
        print("Exporting knowledge for AI...")
        model_path = os.path.join(args.save_dir, "brain_advanced.json")
        if os.path.exists(model_path):
            trainer.brain.load(model_path)

        export = trainer.export_knowledge_for_ai()
        export_path = os.path.join(args.save_dir, "ai_knowledge_export.json")
        with open(export_path, "w") as f:
            json.dump(export, f, indent=2)

        print(f"Knowledge exported to: {export_path}")
        print(f"\nKnowledge Summary:")
        print(f"  - Brain trained for {export['brain_stats']['episode']} episodes")
        print(f"  - Robot success rate: {export['robot_stats']['success_rate']}%")
        if "knowledge" in export:
            k = export["knowledge"]
            print(
                f"  - Learned patterns: {k.get('patterns_count', len(k.get('learned_patterns', [])))}"
            )
            print(
                f"  - Strategies: {k.get('strategies_count', len(k.get('top_strategies', [])))}"
            )
        return

    # Training mode
    print(f"Configuration:")
    print(f"  Grid Size:       {args.grid_size}x{args.grid_size}")
    print(f"  Episodes:        {args.episodes}")
    print(f"  Batch Size:      {args.batch_size}")
    print(f"  Learning Rate:   {args.learning_rate}")
    print(f"  Gamma:           {args.gamma}")
    print(f"  Knowledge Base:  {'Enabled' if not args.no_knowledge else 'Disabled'}")
    print(f"  Online Learning: {'Enabled' if not args.no_online else 'Disabled'}")
    print()

    # Train
    summary = trainer.train(verbose=True)

    # Print summary
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Total Episodes:   {summary['total_episodes']}")
    print(f"  Training Time:    {summary['total_time']}s")
    print(f"  Best Reward:      {summary['best_reward']}")
    print(f"  Final Avg Reward: {summary['final_avg_reward']}")
    print(f"  Success Rate:     {summary['success_rate']}%")
    print(f"  Goals Reached:    {summary['goals_reached']}")

    if summary.get("knowledge_stats"):
        ks = summary["knowledge_stats"]
        print(f"\n  Knowledge Base:")
        print(f"    - Patterns:   {ks['patterns_count']}")
        print(f"    - Strategies: {ks['strategies_count']}")
        print(f"    - Heuristics: {ks['heuristics_count']}")

    if summary.get("online_stats"):
        ols = summary["online_stats"]
        print(f"\n  Online Learning:")
        print(f"    - Searches:   {ols['total_searches']}")
        print(f"    - Learned:    {ols['learned_items']}")

    print(f"\n  Model Saved:      {args.save_dir}/brain_advanced.json")
    print()

    # Evaluate
    print("Running evaluation (no exploration)...")
    eval_results = trainer.evaluate(num_episodes=50)
    print(f"  Eval Avg Reward:  {eval_results['avg_reward']}")
    print(f"  Eval Success:     {eval_results['success_rate']}%")
    print()

    print("Done! Use --evaluate to test the trained model.")
    print("Use --export to export knowledge for AI integration.")


if __name__ == "__main__":
    main()
