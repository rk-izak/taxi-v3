# Taxi-DQN

Deep Q-Network agent that learns to solve the classic **Gymnasium Taxi-v3** environment.

---

## Test Results

```bash
--- TEST COMPLETE — AVG. REWARD OVER 100 EP.: 8.27 ---
```

---

## Requirements

| Package     | Tested Version | Notes                                            |
|-------------|---------------|--------------------------------------------------|
| Python      | >= 3.10  | Tested on Python 3.10.16 but any `Python 3.10.*` should be OK  |
| torch       | >= 2.1.0       | CPU preferred                                       |
| gymnasium   | 0.29.1        |           N/A                    |
| numpy       | 1.24.*        | **Pinned < 2.0** to match pre-built wheels       |
| pygame      | >= 2.5.2       |             N/A                                    |
| matplotlib  | >= 3.8.0       |              N/A            |
| PyYAML      | >= 6.0         | YAML config parsing                              |

All dependencies are used by **`pyproject.toml`** and stored in **`requirements.txt`** for quick installs.

> **_NOTE:_**  If running on MacOS, beware that PyGame is **EXTREMELY** unreliable and you might run into issues regarding env. rendering.
---

## Repository Structure

```
.
├── src/
│   └── taxi_dqn/
│       ├── main.py            # train / test CLI
│       └── utils/
│           ├── agent.py
│           ├── environment.py
│           ├── replay_buffer.py
│           └── graphs.py
├── configs/
│   ├── agent.yaml
│   ├── train.yaml
│   └── test.yaml
├── checkpoints/               # created at runtime
└── results/                   # created at runtime
```

Model checkpoints are stored every `checkpoint_interval` based on your `train.yaml` file. Example checkpoints have been preloaded in the `best_checkpoints/` folder.

Example results files regarding testing/training can be found in the `best_results/` folder.

---


## Installation (Editable Mode)

```bash
git clone git@github.com:rk-izak/taxi-v3.git
cd taxi-v3/

# create and activate a virtualenv
python3.10 -m venv venv

# Windows: .\venv\Scripts\activate
source venv/bin/activate

# install project and deps
pip install --upgrade pip
# can skip -e editable flag but recommended
pip install -e .

```

---

## Usage

### 1. Training

```bash
cd taxi-v3/
python -m taxi_dqn.main --mode train --mode-cfg configs/train.yaml
```
OR
```bash
taxi-main --mode train --mode-cfg configs/train.yaml
```

Outputs:
* `results/training_rewards.csv` -> per-episode reward log  
* `results/training_rewards.png`  -> 4-panel figure (reward curve, epsilon-decay, average loss, success rate)  
* `checkpoints/dqn_taxi_final.pt` -> final model (and periodic checkpoints every N episodes)

### 2. Testing

```bash
cd taxi-v3/
python -m taxi_dqn.main --mode test --mode-cfg configs/test.yaml
```
OR
```bash
taxi-main --mode test --mode-cfg configs/test.yaml
```

Outputs:

* `results/testing_results.csv` -> per-episode rewards  
* `results/testing_rewards.png` -> histogram of test reward distribution  
*  Average reward printed to console.

---

## YAML Configuration Files

| File | Description |
|------|-------------|
| **`configs/agent.yaml`** | Neural-network & replay-buffer settings: embedding size, hidden layers, batch size, buffer capacity, etc. |
| **`configs/train.yaml`** | Training loop parameters: number of episodes, epsilon-greedy schedule, target-network update interval, checkpointing, logging paths, random seed, render flag, etc. |
| **`configs/test.yaml`** | Testing parameters: number of episodes, model checkpoint path, render flag, random seed, etc. |

---
