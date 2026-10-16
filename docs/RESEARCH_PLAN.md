# Self-Learning AI Robot — Research Plan & Architecture

## 1. Project Vision

Build an autonomous AI robot that learns tasks through interaction with its environment
using deep reinforcement learning. The robot starts with zero knowledge and progressively
improves through trial-and-error, experience replay, and self-evaluation.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                  ROBOT SYSTEM                        │
│                                                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │ SENSORS  │───▶│  BRAIN   │───▶│ACTUATORS │      │
│  │ (Input)  │    │ (Neural  │    │ (Output) │      │
│  └──────────┘    │ Network) │    └──────────┘      │
│       │          │          │         │              │
│       │          │ ┌──────┐ │         │              │
│       │          │ │Memory│ │         │              │
│       │          │ │Buffer│ │         │              │
│       │          │ └──────┘ │         │              │
│       │          └──────────┘         │              │
│       │               │               │              │
│       ▼               ▼               ▼              │
│  ┌──────────────────────────────────────────┐       │
│  │          ENVIRONMENT (Simulation)         │       │
│  │  - Grid world / Maze / Object navigation │       │
│  │  - Rewards & penalties                   │       │
│  │  - Obstacles & dynamic elements          │       │
│  └──────────────────────────────────────────┘       │
│                      │                               │
│                      ▼                               │
│  ┌──────────────────────────────────────────┐       │
│  │         LEARNING ENGINE                   │       │
│  │  - Deep Q-Network (DQN)                  │       │
│  │  - Experience Replay                     │       │
│  │  - Target Network                        │       │
│  │  - Epsilon-greedy Exploration            │       │
│  │  - Adaptive Learning Rate                │       │
│  └──────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────┘
```

---

## 3. Core Components

### 3.1 Brain (Neural Network)
- **Architecture**: Multi-layer perceptron (MLP)
- **Input layer**: Sensor readings (position, distances, object detection)
- **Hidden layers**: 2-3 fully connected layers with ReLU activation
- **Output layer**: Q-values for each possible action
- **Optimizer**: Adam with adaptive learning rate

### 3.2 Learning Engine
- **Algorithm**: Deep Q-Network (DQN) with extensions:
  - **Double DQN**: Reduces overestimation bias
  - **Experience Replay**: Stores past experiences, samples randomly for training
  - **Target Network**: Separate network for stable Q-value targets
  - **Epsilon-greedy**: Balances exploration vs exploitation
  - **Reward Clipping**: Stabilizes training across different reward scales

### 3.3 Memory System
- **Experience Buffer**: Stores (state, action, reward, next_state, done) tuples
- **Prioritized Replay**: Important experiences are sampled more often
- **Capacity**: Configurable (default 10,000 experiences)

### 3.4 Environment
- **Grid World**: 2D environment with obstacles, goals, and hazards
- **State Space**: Robot position, sensor readings, object locations
- **Action Space**: Move (up/down/left/right), rotate, scan
- **Reward Structure**:
  - +100 for reaching goal
  - -1 for each step (encourages efficiency)
  - -10 for hitting obstacles
  - +10 for discovering new areas

### 3.5 Robot Body (Abstraction Layer)
- **Sensors**: Proximity, vision (simulated), position
- **Actuators**: Motors (movement), gripper, scanner
- **Interface**: Hardware-agnostic API for sim-to-real transfer

---

## 4. Learning Process

```
Episode 1: Random exploration → Poor performance
Episode 10: Basic patterns emerge → Avoids obvious obstacles
Episode 50: Path optimization → Finds shorter routes
Episode 100: Advanced behavior → Efficient goal-seeking
Episode 500+: Near-optimal → Mastered the environment
```

### Training Loop
1. Robot observes current state from sensors
2. Brain (neural network) predicts Q-values for each action
3. Epsilon-greedy selects action (explore or exploit)
4. Environment returns reward + new state
5. Experience stored in memory buffer
6. Every N steps: sample batch from memory, train network
7. Every M episodes: copy weights to target network
8. Repeat until convergence

---

## 5. Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Language | Python 3.10+ | Primary development |
| ML Framework | PyTorch | Neural network + training |
| Simulation | Custom 2D grid | Fast training environment |
| Visualization | HTML/JS Dashboard | Real-time monitoring |
| Data | JSON | Config + experience logging |

---

## 6. Research Phases

### Phase 1: Foundation (Current)
- [x] Architecture design
- [x] Neural network brain
- [x] DQN learning algorithm
- [x] Simulation environment
- [x] Experience replay memory
- [x] Training loop
- [x] Visualization dashboard

### Phase 2: Advanced Learning
- [ ] Prioritized experience replay
- [ ] Dueling network architecture
- [ ] Multi-step returns (n-step DQN)
- [ ] Curriculum learning (increasing environment difficulty)

### Phase 3: Perception
- [ ] Convolutional neural network for visual input
- [ ] Object detection and classification
- [ ] Spatial mapping (SLAM basics)

### Phase 4: Sim-to-Real Transfer
- [ ] Domain randomization
- [ ] System identification
- [ ] Hardware integration (ROS 2)
- [ ] Sensor fusion

### Phase 5: Autonomous Improvement
- [ ] Meta-learning (learning to learn)
- [ ] Transfer learning between tasks
- [ ] Self-generated curriculum
- [ ] Multi-task learning

---

## 7. Key Research Questions

1. **How fast can the robot learn a new environment from scratch?**
2. **Can learned behaviors transfer to modified environments?**
3. **What is the optimal exploration-exploitation balance?**
4. **How does memory size affect learning quality?**
5. **Can the robot learn multiple tasks simultaneously?**

---

## 8. Success Metrics

- **Learning Speed**: Episodes to reach 80% optimal performance
- **Final Performance**: Average reward per episode after convergence
- **Generalization**: Performance on unseen environment variations
- **Stability**: Variance in performance across training runs

---

## 9. Hardware Requirements (for physical robot)

| Component | Specification | Est. Cost |
|-----------|--------------|-----------|
| Compute | NVIDIA Jetson Nano / Raspberry Pi 4 | $100-150 |
| Motors | N20 geared motors x4 | $20 |
| Sensors | HC-SR04 ultrasonic x3, camera module | $30 |
| Chassis | 2WD/4WD robot car kit | $30 |
| Power | 18650 LiPo battery pack | $25 |
| Servo | SG90 micro servo (scanner) | $5 |
| **Total** | | **~$210-260** |

---

## 10. References & Inspiration

- Mnih et al. (2015) — "Human-level control through deep reinforcement learning"
- Van Hasselt et al. (2016) — "Deep Reinforcement Learning with Double Q-learning"
- Schaul et al. (2016) — "Prioritized Experience Replay"
- Sutton & Barto (2018) — "Reinforcement Learning: An Introduction"
