# NapRobo — Advanced Self-Learning AI Robot

A powerful self-learning AI robot built with Deep Reinforcement Learning, featuring a knowledge base, online learning capabilities, and pattern recognition. The robot learns to navigate complex environments through trial-and-error, building up knowledge and strategies over time.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

## 🚀 Features

### Core Capabilities
- **Deep Q-Network (DQN)** — Neural network that learns optimal actions through reinforcement learning
- **Experience Replay** — Stores and learns from past experiences
- **Knowledge Base** — Persistent storage for patterns, strategies, and heuristics
- **Online Learning** — Search and learn from external sources
- **Pattern Recognition** — Identifies similar situations and applies learned solutions
- **Strategy Learning** — Builds a library of successful strategies
- **Self-Improvement** — Continuously improves with every episode

### Advanced Features
- **Double DQN** — Reduces overestimation bias for more stable learning
- **Target Network** — Separate network for stable Q-value targets
- **Epsilon-Greedy Exploration** — Balances exploration vs exploitation
- **Curriculum Ready** — Can be trained on increasingly difficult environments
- **Knowledge Export** — Export learned knowledge for AI integration
- **Real-time Dashboard** — Interactive visualization of training progress

## 📊 Training Results

After 500 episodes of training:
- **Goals Reached**: 9 (1.8% success rate)
- **Best Reward**: 32.0
- **Knowledge Patterns**: 17 learned
- **Strategies**: 21 stored
- **Training Time**: ~76 seconds

The robot starts with zero knowledge and progressively learns to navigate obstacles and reach goals.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                  NapRobo System                      │
│                                                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │ SENSORS  │───▶│  BRAIN   │───▶│ACTUATORS │      │
│  │ (Input)  │    │ (DQN)    │    │ (Output) │      │
│  └──────────┘    └──────────┘    └──────────┘      │
│       │               │               │              │
│       │          ┌────┴────┐          │              │
│       │          │         │          │              │
│       │     ┌────▼──┐ ┌───▼────┐     │              │
│       │     │Memory │ │Knowledge│    │              │
│       │     │Buffer │ │  Base   │    │              │
│       │     └───────┘ └────────┘    │              │
│       │               │               │              │
│       ▼               ▼               ▼              │
│  ┌──────────────────────────────────────────┐       │
│  │          ENVIRONMENT                      │       │
│  │  - Grid world with obstacles             │       │
│  │  - Goals and hazards                     │       │
│  │  - Dynamic difficulty                    │       │
│  └──────────────────────────────────────────┘       │
│                      │                               │
│                      ▼                               │
│  ┌──────────────────────────────────────────┐       │
│  │         ONLINE LEARNING                   │       │
│  │  - Web search integration                │       │
│  │  - Knowledge caching                     │       │
│  │  - External data sources                 │       │
│  └──────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────┘
```

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/NapRobo.git
cd NapRobo

# Install dependencies
pip install -r requirements.txt

# That's it! No PyTorch needed - pure NumPy implementation
```

## 🎯 Quick Start

### Basic Training
```bash
# Train the robot for 500 episodes
python train_advanced.py --episodes 500

# Train with custom settings
python train_advanced.py --episodes 1000 --grid-size 12 --learning-rate 0.005
```

### Evaluate Trained Model
```bash
# Evaluate the trained robot
python train_advanced.py --evaluate
```

### Export Knowledge
```bash
# Export learned knowledge for AI integration
python train_advanced.py --export
```

### View Dashboard
```bash
# Open the interactive dashboard
open dashboard/index.html
```

## 🧠 How It Works

### 1. Perception
The robot receives a 10-dimensional state vector:
- Normalized position (x, y)
- Proximity sensor readings (4 directions)
- Direction to goal (dx, dy)
- Exploration percentage
- Time remaining

### 2. Decision Making
The neural network (128→64→32 neurons) predicts Q-values for 5 actions:
- UP, RIGHT, DOWN, LEFT, SCAN

### 3. Learning
- **Experience Replay**: Stores (state, action, reward, next_state) tuples
- **Batch Training**: Samples random batches to break correlation
- **Target Network**: Stable targets for Q-value updates
- **Knowledge Integration**: Successful patterns stored for future reference

### 4. Reward System
- +50 for reaching goal
- -0.5 per step (efficiency incentive)
- -5 for hitting walls
- -10 for hazards
- +2 for exploring new areas

## 📁 Project Structure

```
NapRobo/
├── robot/
│   ├── brain/
│   │   └── brain.py              # Neural network (DQN)
│   ├── body/
│   │   └── robot.py              # Robot abstraction
│   ├── environment/
│   │   └── grid_world.py         # Simulation environment
│   ├── memory/
│   │   └── replay_buffer.py      # Experience replay
│   ├── knowledge/
│   │   └── knowledge_base.py     # Knowledge storage
│   ├── learning/
│   │   └── online_learner.py     # Online learning
│   ├── trainer.py                # Basic trainer
│   └── advanced_trainer.py       # Advanced trainer
├── dashboard/
│   └── index.html                # Interactive dashboard
├── docs/
│   └── RESEARCH_PLAN.md          # Research documentation
├── checkpoints/                  # Trained models & data
├── main.py                       # Basic training script
├── train_advanced.py             # Advanced training script
├── requirements.txt
└── README.md
```

## 🔬 Advanced Usage

### Custom Training
```python
from robot.advanced_trainer import AdvancedTrainer

trainer = AdvancedTrainer(
    grid_size=12,
    max_episodes=2000,
    learning_rate=0.005,
    gamma=0.97,
    enable_knowledge=True,
    enable_online_learning=True,
)

summary = trainer.train(verbose=True)
print(f"Success rate: {summary['success_rate']}%")
```

### Knowledge Integration
```python
from robot.knowledge import KnowledgeBase

kb = KnowledgeBase("my_knowledge.json")
kb.learn_pattern(state, action, reward, success=True)
kb.learn_strategy("My Strategy", "Description", 0.9, context)

# Search for similar patterns
similar = kb.search_patterns(current_state, top_k=5)
```

### Online Learning
```python
from robot.learning import OnlineLearner

learner = OnlineLearner()
results = learner.search("reinforcement learning")
learner.learn_from_url("https://example.com/article")
```

## 🚀 Next Steps & Roadmap

### Phase 1: Enhanced Learning (Current)
- [x] Deep Q-Network implementation
- [x] Knowledge base integration
- [x] Online learning framework
- [ ] Prioritized experience replay
- [ ] Dueling network architecture

### Phase 2: Advanced Perception
- [ ] Convolutional neural networks for vision
- [ ] Object detection and classification
- [ ] SLAM (Simultaneous Localization and Mapping)
- [ ] Multi-sensor fusion

### Phase 3: Hardware Integration
- [ ] Raspberry Pi deployment
- [ ] ROS 2 integration
- [ ] Real sensor integration
- [ ] Motor control system

### Phase 4: Autonomous Improvement
- [ ] Meta-learning (learning to learn)
- [ ] Transfer learning between tasks
- [ ] Self-generated curriculum
- [ ] Multi-task learning

## 🛠️ Hardware Requirements (for Physical Robot)

| Component | Specification | Est. Cost |
|-----------|--------------|-----------|
| Compute | Raspberry Pi 4 / NVIDIA Jetson Nano | $100-150 |
| Motors | N20 geared motors x4 | $20 |
| Sensors | HC-SR04 ultrasonic x3, camera | $30 |
| Chassis | Robot car kit | $30 |
| Power | 18650 LiPo battery pack | $25 |
| **Total** | | **~$205-255** |

## 📚 Research & References

- Mnih et al. (2015) - "Human-level control through deep reinforcement learning"
- Van Hasselt et al. (2016) - "Deep Reinforcement Learning with Double Q-learning"
- Sutton & Barto (2018) - "Reinforcement Learning: An Introduction"

## 🤝 Contributing

Contributions are welcome! Areas of interest:
- Improved learning algorithms
- Hardware integration
- New environments
- Visualization tools
- Documentation

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

Built with:
- NumPy for neural network computation
- Deep Q-Network algorithm
- Reinforcement learning principles

## 📞 Contact

For questions, issues, or collaboration opportunities, please open an issue on GitHub.

---

**NapRobo** — Teaching robots to learn, one episode at a time. 🤖
