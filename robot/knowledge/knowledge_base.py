"""
Knowledge Base — AI-Usable Knowledge System
=============================================
Stores and retrieves learned knowledge, patterns, and strategies.
The robot can search, learn from experiences, and apply knowledge
to new situations.
"""

import json
import os
import numpy as np
from typing import Dict, List, Any
from datetime import datetime


class KnowledgeBase:
    """
    Persistent knowledge storage for the robot.

    Stores:
    - Learned patterns (state-action-success mappings)
    - Successful strategies
    - Environmental knowledge
    - Meta-learned heuristics
    """

    def __init__(self, filepath: str = "knowledge_base.json"):
        self.filepath = filepath
        self.knowledge = {
            "patterns": [],
            "strategies": [],
            "environmental": {},
            "heuristics": {},
            "metadata": {
                "created": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "total_entries": 0,
            },
        }
        self._load()

    def _load(self):
        """Load knowledge from file if exists."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r") as f:
                    self.knowledge = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load knowledge base: {e}")

    def _save(self):
        """Save knowledge to file."""
        self.knowledge["metadata"]["last_updated"] = datetime.now().isoformat()
        self.knowledge["metadata"]["total_entries"] = len(
            self.knowledge["patterns"]
        ) + len(self.knowledge["strategies"])
        os.makedirs(
            os.path.dirname(self.filepath) if os.path.dirname(self.filepath) else ".",
            exist_ok=True,
        )
        with open(self.filepath, "w") as f:
            json.dump(self.knowledge, f, indent=2)

    def learn_pattern(
        self, state: np.ndarray, action: int, reward: float, success: bool
    ):
        """Learn a pattern from an experience."""
        pattern = {
            "state": state.tolist(),
            "action": action,
            "reward": float(reward),
            "success": success,
            "timestamp": datetime.now().isoformat(),
        }
        self.knowledge["patterns"].append(pattern)

        # Keep only last 1000 patterns to avoid memory bloat
        if len(self.knowledge["patterns"]) > 1000:
            self.knowledge["patterns"] = self.knowledge["patterns"][-1000:]

        self._save()

    def learn_strategy(
        self, name: str, description: str, effectiveness: float, context: Dict
    ):
        """Store a learned strategy."""
        strategy = {
            "name": name,
            "description": description,
            "effectiveness": effectiveness,
            "context": context,
            "timestamp": datetime.now().isoformat(),
        }
        self.knowledge["strategies"].append(strategy)
        self._save()

    def store_environmental_knowledge(self, key: str, value: Any):
        """Store knowledge about the environment."""
        self.knowledge["environmental"][key] = value
        self._save()

    def store_heuristic(self, name: str, rule: str, confidence: float):
        """Store a learned heuristic/rule."""
        self.knowledge["heuristics"][name] = {
            "rule": rule,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
        }
        self._save()

    def search_patterns(self, state: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Search for similar past patterns."""
        if not self.knowledge["patterns"]:
            return []

        # Calculate similarity (simple Euclidean distance)
        similarities = []
        for pattern in self.knowledge["patterns"]:
            pattern_state = np.array(pattern["state"])
            distance = np.linalg.norm(state - pattern_state)
            similarities.append((distance, pattern))

        # Sort by similarity (lower distance = more similar)
        similarities.sort(key=lambda x: x[0])
        return [p for _, p in similarities[:top_k]]

    def get_best_strategies(self, top_k: int = 3) -> List[Dict]:
        """Get most effective strategies."""
        strategies = sorted(
            self.knowledge["strategies"], key=lambda x: x["effectiveness"], reverse=True
        )
        return strategies[:top_k]

    def get_knowledge_summary(self) -> Dict:
        """Get summary of all knowledge."""
        return {
            "patterns_count": len(self.knowledge["patterns"]),
            "strategies_count": len(self.knowledge["strategies"]),
            "environmental_keys": list(self.knowledge["environmental"].keys()),
            "heuristics_count": len(self.knowledge["heuristics"]),
            "metadata": self.knowledge["metadata"],
        }

    def export_for_ai(self) -> Dict:
        """Export knowledge in AI-readable format."""
        return {
            "learned_patterns": self.knowledge["patterns"][-100:],  # Last 100
            "top_strategies": self.get_best_strategies(10),
            "environmental_knowledge": self.knowledge["environmental"],
            "heuristics": self.knowledge["heuristics"],
        }
